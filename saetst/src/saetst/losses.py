"""Focal-reweighted reconstruction loss + standard SAE losses.

The reweighter holds two EMAs:

  * `error_ema[i]` — running mean of (x_i - x̂_i)^2 across batches
  * `var_ema[i]`   — running mean of (x_i - mean_x_i)^2 across batches
                    (i.e. per-dim variance of activations)

After each batch we update with detached tensors and recompute weights `w[i]`
using one of two schemes from the spec. Weights are detached from the graph —
they shape the loss but receive no gradient themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from saetst.models import SAEOutput, JumpReLUSAE, ReLUSAE, heaviside_ste

FocalScheme = Literal["none", "relative", "fit"]


class FocalReweighter(nn.Module):
    """Maintains per-dimension EMAs and produces detached weight vectors.

    Args:
        d_in: activation dim.
        scheme: "none" (uniform, baseline), "relative" (Scheme A) or "fit" (Scheme B).
        gamma: focusing parameter; gamma=0 → uniform regardless of scheme.
        ema_decay: alpha, weight on past EMA. New value gets (1 - alpha).
        clip: (lo, hi) clipping range for produced weights.
        warmup_steps: number of update calls before focal weights kick in;
            during warmup `weights()` returns ones.
        eps: numerical floor for division.
    """

    def __init__(
        self,
        d_in: int,
        *,
        scheme: FocalScheme = "fit",
        gamma: float = 1.0,
        ema_decay: float = 0.99,
        clip: tuple[float, float] = (0.1, 10.0),
        warmup_steps: int = 1000,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.d_in = d_in
        self.scheme = scheme
        self.gamma = float(gamma)
        self.ema_decay = float(ema_decay)
        self.clip_lo, self.clip_hi = clip
        self.warmup_steps = int(warmup_steps)
        self.eps = float(eps)
        # Buffers persist with state_dict but receive no gradient.
        self.register_buffer("error_ema", torch.zeros(d_in))
        self.register_buffer("var_ema", torch.ones(d_in))
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("clip_hits", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, x: torch.Tensor, recon: torch.Tensor) -> None:
        """Update EMAs from a batch. Should be called once per training step."""
        a = self.ema_decay
        # Per-dim squared error, averaged over batch.
        sq_err = (x - recon).pow(2).mean(dim=0)
        # Per-dim variance: use the batch's biased variance estimate. For very
        # small batches this is noisy, but the EMA smooths it.
        x_mean = x.mean(dim=0)
        var = (x - x_mean).pow(2).mean(dim=0)
        if self.step.item() == 0:
            # First step: seed the EMAs directly so we don't bias toward zeros.
            self.error_ema.copy_(sq_err)
            self.var_ema.copy_(var)
        else:
            self.error_ema.mul_(a).add_(sq_err, alpha=1 - a)
            self.var_ema.mul_(a).add_(var, alpha=1 - a)
        self.step += 1

    @torch.no_grad()
    def weights(self) -> torch.Tensor:
        """Return per-dim weights for the focal MSE. Detached, shape [d_in]."""
        if self.scheme == "none" or self.gamma == 0.0 or self.step.item() < self.warmup_steps:
            return torch.ones_like(self.error_ema)

        if self.scheme == "relative":
            # Scheme A: w_i = (e_i / mean(e))^gamma
            mean_e = self.error_ema.mean().clamp_min(self.eps)
            ratio = self.error_ema / mean_e
            w = ratio.pow(self.gamma)
        elif self.scheme == "fit":
            # Scheme B: fit_i = 1 - e_i / Var(x_i); w_i = (1 - clamp(fit_i, 0, 1))^gamma
            #                                          = clamp(e_i / Var(x_i), 0, 1)^gamma
            shortfall = (self.error_ema / self.var_ema.clamp_min(self.eps)).clamp(0.0, 1.0)
            w = shortfall.pow(self.gamma)
        else:
            raise ValueError(f"unknown scheme: {self.scheme}")

        # Clip and track whether clipping triggered.
        w_clipped = w.clamp(self.clip_lo, self.clip_hi)
        if not torch.equal(w, w_clipped):
            self.clip_hits += 1
        # Renormalize so mean(w) = 1 — keeps the loss scale comparable to baseline
        # and decoupled from the choice of gamma. The relative weighting across
        # dimensions is what matters.
        w_norm = w_clipped / w_clipped.mean().clamp_min(self.eps)
        return w_norm


@dataclass
class LossOutput:
    total: torch.Tensor
    recon: torch.Tensor       # weighted recon term, scalar
    sparsity: torch.Tensor    # sparsity term (already scaled by lambda)
    l0: torch.Tensor          # mean L0 across batch (no grad)
    weights: torch.Tensor     # per-dim weights used (no grad)


def sae_loss(
    x: torch.Tensor,
    out: SAEOutput,
    sae: nn.Module,
    *,
    reweighter: Optional[FocalReweighter] = None,
    sparsity_coef: float = 1e-3,
) -> LossOutput:
    """Compute the SAE training loss.

    Reconstruction term:  mean over batch of  Σ_i w_i * (x_i - x̂_i)^2
    Sparsity term:
      ReLU SAE  → λ * mean_i sum_j |z_j| * ||W_dec_j||  (norm-weighted L1, Bricken)
      JumpReLU  → λ * mean L0 surrogate via the same rectangle STE used in encode
    """
    if reweighter is not None:
        w = reweighter.weights().detach()
    else:
        w = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)

    # Per-sample weighted squared error, summed over dims, mean over batch.
    sq = (x - out.recon).pow(2)
    recon = (sq * w).sum(dim=-1).mean()

    if isinstance(sae, ReLUSAE):
        # Norm-weighted L1: penalize z * ||W_dec_row||. Bricken et al. note this
        # makes L1 invariant to rescaling decoder rows (we already renormalize, but
        # this also makes the penalty proportional to "actual contribution to recon").
        dec_norms = sae.W_dec.norm(dim=-1)
        sparsity = sparsity_coef * (out.z.abs() * dec_norms).sum(dim=-1).mean()
        with torch.no_grad():
            l0 = (out.z > 0).float().sum(dim=-1).mean()
    elif isinstance(sae, JumpReLUSAE):
        # L0 surrogate: heaviside_ste(pre, theta). Same gradient route as encode.
        gate = heaviside_ste(out.pre_act, sae.theta, sae.bandwidth)
        sparsity = sparsity_coef * gate.sum(dim=-1).mean()
        with torch.no_grad():
            l0 = (out.pre_act > sae.theta).float().sum(dim=-1).mean()
    else:
        raise TypeError(f"unsupported SAE type: {type(sae)}")

    return LossOutput(
        total=recon + sparsity,
        recon=recon.detach(),
        sparsity=sparsity.detach(),
        l0=l0,
        weights=w,
    )
