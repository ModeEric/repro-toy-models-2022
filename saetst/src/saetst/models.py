"""Sparse autoencoder architectures: ReLU+L1 and JumpReLU.

The decoder is parameterized as `W_dec ∈ R^{n×d}` whose rows are kept at
unit L2 norm (Bricken et al. 2023). Encoder is `W_enc ∈ R^{d×n}` with bias.
A pre-encoder bias `b_pre` is subtracted before encoding and added back
after decoding (the "centered" SAE form used in most modern codebases).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEOutput:
    z: torch.Tensor          # latent activations [B, n]
    recon: torch.Tensor      # reconstruction [B, d]
    pre_act: torch.Tensor    # pre-activation values (for JumpReLU L0 surrogate)


def _init_decoder_weights(n_latents: int, d_in: int, device, dtype) -> torch.Tensor:
    W = torch.randn(n_latents, d_in, device=device, dtype=dtype)
    W /= W.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return W


class ReLUSAE(nn.Module):
    """Vanilla ReLU SAE. Sparsity comes from L1 on z (handled in loss)."""

    def __init__(self, d_in: int, n_latents: int, *, device=None, dtype=torch.float32):
        super().__init__()
        self.d_in = d_in
        self.n_latents = n_latents
        W_dec = _init_decoder_weights(n_latents, d_in, device, dtype)
        self.W_dec = nn.Parameter(W_dec)
        # Tied init: encoder = decoder^T (Bricken et al.). Free parameter from there.
        self.W_enc = nn.Parameter(W_dec.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(n_latents, device=device, dtype=dtype))
        self.b_pre = nn.Parameter(torch.zeros(d_in, device=device, dtype=dtype))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x - self.b_pre
        pre = h @ self.W_enc + self.b_enc
        z = F.relu(pre)
        return z, pre

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_pre

    def forward(self, x: torch.Tensor) -> SAEOutput:
        z, pre = self.encode(x)
        recon = self.decode(z)
        return SAEOutput(z=z, recon=recon, pre_act=pre)

    @torch.no_grad()
    def renorm_decoder(self) -> None:
        norms = self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(norms)


class _RectangleSTE(torch.autograd.Function):
    """Rectangle-window straight-through estimator for the JumpReLU step.

    Forward returns the indicator 1[pre > theta]; the backward pass uses a
    rectangular kernel of half-width `bandwidth` centered at theta, divided by
    `2 * bandwidth`, so the gradient w.r.t. theta is a finite-difference
    approximation of d/dtheta E[1[pre > theta]] over the local distribution.
    Follows Rajamanoharan et al. (2024).
    """

    @staticmethod
    def forward(ctx, pre: torch.Tensor, theta: torch.Tensor, bandwidth: float):
        ctx.save_for_backward(pre, theta)
        ctx.bandwidth = bandwidth
        return (pre > theta).to(pre.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        pre, theta = ctx.saved_tensors
        bw = ctx.bandwidth
        # d/dtheta of the smoothed step: -1/(2*bw) inside the window, 0 outside.
        in_window = ((pre - theta).abs() < bw).to(grad_out.dtype)
        grad_theta = -grad_out * in_window / (2 * bw)
        # Sum across batch dim — theta is per-latent, broadcast across batch.
        if grad_theta.dim() > theta.dim():
            grad_theta = grad_theta.sum(dim=tuple(range(grad_theta.dim() - theta.dim())))
        return None, grad_theta, None


def heaviside_ste(pre: torch.Tensor, theta: torch.Tensor, bandwidth: float) -> torch.Tensor:
    return _RectangleSTE.apply(pre, theta, bandwidth)


class JumpReLUSAE(nn.Module):
    """JumpReLU SAE (Rajamanoharan et al. 2024).

    z_i = pre_i * 1[pre_i > theta_i], with theta_i learned via a rectangle STE.
    Sparsity comes from an L0 penalty surrogate: sum_i 1[pre_i > theta_i],
    differentiable via the same STE — handled in `losses.sae_loss`.
    """

    def __init__(
        self,
        d_in: int,
        n_latents: int,
        *,
        bandwidth: float = 1e-3,
        log_theta_init: float = -3.0,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.d_in = d_in
        self.n_latents = n_latents
        self.bandwidth = bandwidth
        W_dec = _init_decoder_weights(n_latents, d_in, device, dtype)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(n_latents, device=device, dtype=dtype))
        self.b_pre = nn.Parameter(torch.zeros(d_in, device=device, dtype=dtype))
        # Parameterize theta = exp(log_theta) so theta stays positive and the
        # optimizer sees roughly log-scale steps.
        self.log_theta = nn.Parameter(
            torch.full((n_latents,), log_theta_init, device=device, dtype=dtype)
        )

    @property
    def theta(self) -> torch.Tensor:
        return self.log_theta.exp()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x - self.b_pre
        pre = h @ self.W_enc + self.b_enc
        gate = heaviside_ste(pre, self.theta, self.bandwidth)
        z = pre * gate
        return z, pre

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_pre

    def forward(self, x: torch.Tensor) -> SAEOutput:
        z, pre = self.encode(x)
        recon = self.decode(z)
        return SAEOutput(z=z, recon=recon, pre_act=pre)

    @torch.no_grad()
    def renorm_decoder(self) -> None:
        norms = self.W_dec.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(norms)


SAEArch = Literal["relu", "jumprelu"]


def make_sae(arch: SAEArch, d_in: int, n_latents: int, **kwargs) -> nn.Module:
    if arch == "relu":
        return ReLUSAE(d_in, n_latents, **kwargs)
    if arch == "jumprelu":
        return JumpReLUSAE(d_in, n_latents, **kwargs)
    raise ValueError(f"unknown arch: {arch}")
