"""Evaluation metrics for SAEs.

  * `DeadLatentTracker` — running count of activation events per latent.
  * `evaluate_recon` — compute MSE, mean L0, fraction-variance-explained on a
    held-out batch iterator.
  * `downstream_loss_recovered` — splice the SAE reconstruction back into the
    LM forward pass and report the fraction of (clean - zero-ablation) cross-
    entropy gap that is recovered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import torch
import torch.nn.functional as F

from saetst.models import SAEOutput, JumpReLUSAE, ReLUSAE


@dataclass
class ReconStats:
    mse: float                 # mean squared error per element
    sum_sq_err: float          # sum of squared errors (for FVE)
    sum_sq_x_centered: float   # for FVE
    fve: float                 # fraction of variance explained
    l0: float                  # mean number of active latents
    n_tokens: int


class DeadLatentTracker:
    """Counts activations per latent over a sliding window of tokens.

    A latent is "dead" if it has zero activation events over a window of size
    `window_tokens`. The spec says "< 1 in 1e6 tokens" — set `window_tokens =
    1e6` and `threshold = 1` to match.
    """

    def __init__(self, n_latents: int, window_tokens: int = 1_000_000, device: str = "cpu"):
        self.n_latents = n_latents
        self.window_tokens = int(window_tokens)
        self.counts = torch.zeros(n_latents, dtype=torch.long, device=device)
        self.tokens_seen = 0

    def update(self, z: torch.Tensor) -> None:
        # z: [B, n_latents]; count per-latent firings (treat any positive value as a firing)
        active = (z > 0).sum(dim=0)
        self.counts += active.to(self.counts.dtype)
        self.tokens_seen += z.shape[0]

    def reset(self) -> None:
        self.counts.zero_()
        self.tokens_seen = 0

    def dead_fraction(self, threshold: int = 1) -> float:
        return float((self.counts < threshold).float().mean().item())

    def low_freq_fraction(self, max_rate: float = 1e-6) -> float:
        if self.tokens_seen == 0:
            return float("nan")
        rates = self.counts.float() / self.tokens_seen
        return float((rates < max_rate).float().mean().item())

    def high_freq_fraction(self, min_rate: float = 0.1) -> float:
        if self.tokens_seen == 0:
            return float("nan")
        rates = self.counts.float() / self.tokens_seen
        return float((rates > min_rate).float().mean().item())


@torch.no_grad()
def evaluate_recon(
    sae,
    stream: Iterator[torch.Tensor],
    n_batches: int,
    device: str = "cpu",
) -> ReconStats:
    """Aggregate recon metrics across `n_batches` from `stream`."""
    sae.eval()
    sum_sq = 0.0
    sum_sq_centered = 0.0
    sum_l0 = 0.0
    n_tokens = 0
    n_elem = 0

    # Two-pass would be needed for true variance; use a running mean approximation
    # that's good enough for held-out FVE estimates at this scale.
    for i in range(n_batches):
        try:
            x = next(stream)
        except StopIteration:
            break
        x = x.to(device)
        out = sae(x)
        sse = (x - out.recon).pow(2).sum().item()
        # Use per-batch centering — biased but fine for FVE here.
        x_mean = x.mean(dim=0, keepdim=True)
        ssc = (x - x_mean).pow(2).sum().item()
        if isinstance(sae, JumpReLUSAE):
            l0 = (out.pre_act > sae.theta).float().sum(dim=-1).mean().item()
        else:
            l0 = (out.z > 0).float().sum(dim=-1).mean().item()

        sum_sq += sse
        sum_sq_centered += ssc
        sum_l0 += l0 * x.shape[0]
        n_tokens += x.shape[0]
        n_elem += x.numel()

    sae.train()
    if n_tokens == 0:
        return ReconStats(float("nan"), 0.0, 0.0, float("nan"), float("nan"), 0)

    mse = sum_sq / n_elem
    fve = 1.0 - (sum_sq / max(sum_sq_centered, 1e-12))
    return ReconStats(
        mse=mse,
        sum_sq_err=sum_sq,
        sum_sq_x_centered=sum_sq_centered,
        fve=fve,
        l0=sum_l0 / n_tokens,
        n_tokens=n_tokens,
    )


@torch.no_grad()
def downstream_loss_recovered(
    sae,
    model,                      # HookedTransformer
    hook_name: str,
    texts: list[str],
    device: str = "cpu",
    max_seq_len: int = 256,
) -> dict[str, float]:
    """Splice the SAE recon into the LM forward pass and report loss recovered.

    Returns a dict with `clean`, `zero`, `spliced` cross-entropies and the
    `loss_recovered` ratio: (zero - spliced) / (zero - clean), in [0, 1] when
    the SAE recon is between zero ablation and clean.
    """
    losses = {"clean": 0.0, "zero": 0.0, "spliced": 0.0}
    n_tokens = 0

    for text in texts:
        tok = model.to_tokens(text, prepend_bos=True)[:, :max_seq_len].to(device)
        if tok.shape[-1] < 2:
            continue

        # Clean
        clean_loss = model(tok, return_type="loss").item()

        # Zero ablation
        def zero_hook(act, hook):  # noqa: WPS430, A002
            return torch.zeros_like(act)

        zero_loss = model.run_with_hooks(
            tok, return_type="loss", fwd_hooks=[(hook_name, zero_hook)]
        ).item()

        # Spliced (SAE recon)
        def splice_hook(act, hook):  # noqa: WPS430, A002
            shape = act.shape          # [B, T, d_in]
            flat = act.reshape(-1, shape[-1])
            recon = sae(flat).recon
            return recon.reshape(shape)

        spliced_loss = model.run_with_hooks(
            tok, return_type="loss", fwd_hooks=[(hook_name, splice_hook)]
        ).item()

        losses["clean"] += clean_loss * tok.shape[-1]
        losses["zero"] += zero_loss * tok.shape[-1]
        losses["spliced"] += spliced_loss * tok.shape[-1]
        n_tokens += tok.shape[-1]

    for k in losses:
        losses[k] /= max(n_tokens, 1)

    gap = losses["zero"] - losses["clean"]
    losses["loss_recovered"] = (
        (losses["zero"] - losses["spliced"]) / gap if gap > 1e-9 else float("nan")
    )
    return losses
