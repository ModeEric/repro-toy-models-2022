"""Activation streaming.

`ActivationStream` is a generic iterator that yields batches of activation
vectors of shape `[batch, d_in]`. Three implementations:

  * `LMActivationStream` — runs a TransformerLens HookedTransformer on text
    from a HuggingFace dataset, captures activations from a chosen layer/site,
    and shuffles them in a small replay buffer so consecutive minibatches
    aren't all from the same document. Use for ad-hoc / single runs.

  * `CachedActivationStream` — reads pre-dumped activations from a sharded
    fp16 binary cache on disk (built by `scripts/cache_activations.py`). Use
    for sweeps: amortize the LM forward across all runs and keep the GPU
    100 % bound on SAE training.

  * `SyntheticActivationStream` — sparse-coded synthetic data with a fixed
    ground-truth dictionary. Used for unit tests, smoke runs, and offline
    development when network access isn't available.

All three share the same `iter(...) -> Iterator[Tensor]` interface so the
trainer doesn't care which is in use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Synthetic stream — for tests and offline smoke tests.
# -----------------------------------------------------------------------------


class SyntheticActivationStream:
    """Activations from x = W * z + noise, where z is k-sparse over n_features.

    The true dictionary `W` (n_features x d_in) has random unit-norm rows,
    z is sampled with exactly `k` nonzero entries per token (uniform in [0, 1]).
    A small Gaussian noise is added. This gives the SAE something to recover.

    With `dim_hard=True`, a fixed subset of dims gets extra heteroscedastic noise
    so focal reweighting has a target structure to find — useful for sanity-
    checking that focal weights actually concentrate where they should.
    """

    def __init__(
        self,
        d_in: int = 64,
        n_features: int = 256,
        k: int = 8,
        noise: float = 0.05,
        batch_size: int = 256,
        device: str = "cpu",
        seed: int = 0,
        dim_hard: bool = False,
        hard_frac: float = 0.1,
        hard_noise_mult: float = 5.0,
    ):
        self.d_in = d_in
        self.n_features = n_features
        self.k = k
        self.noise = noise
        self.batch_size = batch_size
        self.device = device
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        W = torch.randn(n_features, d_in, generator=self.gen)
        W /= W.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.W = W.to(device)
        if dim_hard:
            n_hard = max(1, int(d_in * hard_frac))
            mask = torch.zeros(d_in, device=device)
            hard_idx = torch.randperm(d_in, generator=self.gen)[:n_hard]
            mask[hard_idx] = hard_noise_mult - 1.0  # additive multiplier
            self.hard_mask = mask
        else:
            self.hard_mask = None

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def __next__(self) -> torch.Tensor:
        b, n, k = self.batch_size, self.n_features, self.k
        # Sample k indices per row (sampling without replacement via topk on noise)
        scores = torch.rand(b, n, generator=self.gen)
        topk = scores.topk(k, dim=-1).indices
        z = torch.zeros(b, n)
        vals = torch.rand(b, k, generator=self.gen)
        z.scatter_(1, topk, vals)
        z = z.to(self.device)
        x = z @ self.W
        noise_scale = self.noise
        if self.hard_mask is not None:
            scale = 1.0 + self.hard_mask
            x = x + torch.randn(b, self.d_in, device=self.device) * noise_scale * scale
        else:
            x = x + torch.randn(b, self.d_in, device=self.device) * noise_scale
        return x


# -----------------------------------------------------------------------------
# Real activations from a language model.
# -----------------------------------------------------------------------------


@dataclass
class LMStreamConfig:
    model_name: str = "EleutherAI/pythia-70m-deduped"
    layer: int = 3
    site: str = "resid_post"
    dataset_name: str = "NeelNanda/pile-10k"
    dataset_split: str = "train"
    text_field: str = "text"
    seq_len: int = 128
    buffer_size: int = 16384       # how many activation vectors to hold for shuffling
    refill_threshold: float = 0.5  # when buffer drops below this fraction, refill
    batch_size: int = 4096
    n_seqs_per_refill: int = 256   # sized for A100 (~64k tokens / refill in fp32)
    device: str = "cpu"
    dtype: str = "auto"            # "auto" → bf16 on cuda, fp32 elsewhere
    seed: int = 0


class LMActivationStream:
    """Stream activations from a language model on text data.

    Uses TransformerLens. Builds a small replay buffer to decorrelate token
    activations within a batch — important because consecutive tokens in a doc
    are highly correlated.

    Iteration yields `Tensor[batch_size, d_in]` until the underlying dataset
    is exhausted. After exhaustion StopIteration is raised. Wrap in `cycle(...)`
    if you want infinite iteration.
    """

    def __init__(self, cfg: LMStreamConfig):
        # Local imports — heavy and only needed for this path.
        from transformer_lens import HookedTransformer  # noqa: WPS433
        from datasets import load_dataset  # noqa: WPS433
        from saetst.utils import default_lm_dtype  # noqa: WPS433

        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.model = HookedTransformer.from_pretrained(cfg.model_name, device=cfg.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # bf16 on cuda saves ~2× LM forward time; activations get cast back to fp32
        # before the SAE consumes them (SAE is small enough that fp32 is fine).
        if cfg.dtype == "auto":
            self.lm_dtype = default_lm_dtype(cfg.device)
        else:
            self.lm_dtype = getattr(torch, cfg.dtype)
        self.model = self.model.to(self.lm_dtype)
        self.d_in = self.model.cfg.d_model
        self.hook_name = f"blocks.{cfg.layer}.hook_{cfg.site}"

        ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split, streaming=True)
        self._iter = iter(ds)
        self._buffer: Optional[torch.Tensor] = None
        self._exhausted = False

    @torch.no_grad()
    def _refill(self) -> None:
        cfg = self.cfg
        seqs = []
        for _ in range(cfg.n_seqs_per_refill):
            try:
                row = next(self._iter)
            except StopIteration:
                self._exhausted = True
                break
            text = row[cfg.text_field]
            tok = self.model.to_tokens(text, prepend_bos=True)
            if tok.shape[-1] < 8:  # skip very short docs
                continue
            tok = tok[:, : cfg.seq_len]
            seqs.append(tok)

        if not seqs:
            return

        # Pad to a common length so we can stack — but only up to seq_len.
        max_len = max(s.shape[-1] for s in seqs)
        pad_id = self.model.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.model.tokenizer.eos_token_id
        batched = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=cfg.device)
        attention_mask = torch.zeros(len(seqs), max_len, dtype=torch.bool, device=cfg.device)
        for i, s in enumerate(seqs):
            batched[i, : s.shape[-1]] = s[0]
            attention_mask[i, : s.shape[-1]] = True

        # Run with a hook capturing the chosen site.
        captured = {}

        def hook(act, hook):  # noqa: WPS430, A002
            captured["x"] = act.detach()

        with self.model.hooks(fwd_hooks=[(self.hook_name, hook)]):
            self.model(batched, return_type=None)
        acts = captured["x"]            # [n_seqs, seq_len, d_in]

        # Drop padded positions and the BOS token (first position carries no useful signal here).
        valid = attention_mask.clone()
        valid[:, 0] = False
        flat = acts[valid].float()       # [n_valid_tokens, d_in], cast to fp32 for SAE
        if flat.numel() == 0:
            return

        if self._buffer is None or self._buffer.numel() == 0:
            self._buffer = flat
        else:
            self._buffer = torch.cat([self._buffer, flat], dim=0)

        # Shuffle the buffer once after refill.
        perm = torch.randperm(self._buffer.shape[0], device=self._buffer.device)
        self._buffer = self._buffer[perm]

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def __next__(self) -> torch.Tensor:
        cfg = self.cfg
        bs = cfg.batch_size

        # Refill if buffer empty or low.
        need_refill = (
            self._buffer is None
            or self._buffer.shape[0] < int(cfg.buffer_size * cfg.refill_threshold)
        )
        while need_refill and not self._exhausted:
            self._refill()
            if self._buffer is not None and self._buffer.shape[0] >= bs:
                break
            need_refill = (
                self._buffer is None
                or self._buffer.shape[0] < int(cfg.buffer_size * cfg.refill_threshold)
            )

        if self._buffer is None or self._buffer.shape[0] < bs:
            raise StopIteration

        batch = self._buffer[:bs]
        self._buffer = self._buffer[bs:]
        return batch


# -----------------------------------------------------------------------------
# Cached activations on disk — for sweeps. Built by scripts/cache_activations.py.
# -----------------------------------------------------------------------------


CACHE_META = "meta.json"


def _shard_path(cache_dir: Path, idx: int) -> Path:
    return cache_dir / f"shard_{idx:04d}.bin"


class CachedActivationStream:
    """Iterate batches over a sharded fp16 activation cache on disk.

    Cache format (written by `scripts/cache_activations.py`):
        cache_dir/meta.json     — {d_in, n_per_shard, n_shards, dtype, ...}
        cache_dir/shard_NNNN.bin — raw bytes, [n_per_shard, d_in], dtype as in meta

    Each call to `__next__` returns a `[batch_size, d_in]` fp32 tensor on CPU
    (move to device in the trainer). When `repeat=True`, the stream cycles
    forever, reshuffling the shard order each epoch.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        batch_size: int,
        shuffle: bool = True,
        repeat: bool = True,
        seed: int = 0,
    ):
        self.cache_dir = Path(cache_dir)
        meta_path = self.cache_dir / CACHE_META
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no cache at {self.cache_dir} (missing {CACHE_META}). "
                f"build one with `python scripts/cache_activations.py ...`"
            )
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.d_in = int(self.meta["d_in"])
        self.n_per_shard = int(self.meta["n_per_shard"])
        self.n_shards = int(self.meta["n_shards"])
        self.dtype = np.dtype(self.meta["dtype"])
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.repeat = bool(repeat)
        self._rng = np.random.default_rng(seed)
        self._reset_epoch()

    @property
    def n_total(self) -> int:
        return self.n_shards * self.n_per_shard

    def _reset_epoch(self) -> None:
        self._shard_order = list(range(self.n_shards))
        if self.shuffle:
            self._rng.shuffle(self._shard_order)
        self._next_shard = 0
        self._buffer: Optional[np.ndarray] = None
        self._cursor = 0

    def _load_next_shard(self) -> bool:
        """Load the next shard into RAM (shuffled). Returns True iff one was loaded."""
        if self._next_shard >= len(self._shard_order):
            return False
        shard_idx = self._shard_order[self._next_shard]
        path = _shard_path(self.cache_dir, shard_idx)
        # Memmap → np.array forces a single sequential read off disk; faster than
        # random-indexing a memmap when we then shuffle the whole shard.
        mm = np.memmap(path, dtype=self.dtype, mode="r")
        arr = np.array(mm.reshape(-1, self.d_in))
        del mm
        if self.shuffle:
            idx = self._rng.permutation(len(arr))
            arr = arr[idx]
        self._buffer = arr
        self._cursor = 0
        self._next_shard += 1
        return True

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def __next__(self) -> torch.Tensor:
        bs = self.batch_size
        while True:
            # Need a new shard?
            if self._buffer is None or self._cursor + bs > len(self._buffer):
                # Drop the partial tail of the current shard (max bs-1 tokens, negligible).
                loaded = self._load_next_shard()
                if not loaded:
                    if self.repeat:
                        self._reset_epoch()
                        continue
                    raise StopIteration
                continue
            batch = self._buffer[self._cursor : self._cursor + bs]
            self._cursor += bs
            # Convert fp16 → fp32 for the SAE; keep on CPU (trainer moves to device).
            return torch.from_numpy(batch.astype(np.float32, copy=False))
