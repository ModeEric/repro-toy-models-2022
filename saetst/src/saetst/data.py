"""Activation streaming.

`ActivationStream` is a generic iterator that yields batches of activation
vectors of shape `[batch, d_in]`. Two implementations:

  * `LMActivationStream` — runs a TransformerLens HookedTransformer on text
    from a HuggingFace dataset, captures activations from a chosen layer/site,
    and shuffles them in a small replay buffer so consecutive minibatches
    aren't all from the same document.

  * `SyntheticActivationStream` — sparse-coded synthetic data with a fixed
    ground-truth dictionary. Used for unit tests, smoke runs, and offline
    development when network access isn't available.

The two share the same `iter(...) -> Iterator[Tensor]` interface so the trainer
doesn't care which is in use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

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
    n_seqs_per_refill: int = 64
    device: str = "cpu"
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

        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.model = HookedTransformer.from_pretrained(cfg.model_name, device=cfg.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
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
        flat = acts[valid]               # [n_valid_tokens, d_in]
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
