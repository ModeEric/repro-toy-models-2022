"""Smoke tests for the SAE architectures."""

from __future__ import annotations

import torch

from saetst.losses import sae_loss, FocalReweighter
from saetst.models import JumpReLUSAE, ReLUSAE, heaviside_ste


def test_relu_sae_shapes_and_decoder_unit_norm_init():
    sae = ReLUSAE(d_in=16, n_latents=64)
    x = torch.randn(8, 16)
    out = sae(x)
    assert out.z.shape == (8, 64)
    assert out.recon.shape == (8, 16)
    norms = sae.W_dec.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_relu_decoder_renorm_keeps_unit_rows_after_perturbation():
    sae = ReLUSAE(d_in=8, n_latents=16)
    with torch.no_grad():
        sae.W_dec.mul_(torch.linspace(0.5, 2.0, 16)[:, None])
    sae.renorm_decoder()
    norms = sae.W_dec.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_jumprelu_forward_uses_threshold():
    sae = JumpReLUSAE(d_in=16, n_latents=64, log_theta_init=0.0)  # theta=1
    x = torch.randn(8, 16)
    out = sae(x)
    pre = out.pre_act
    z = out.z
    # All positive activations should have pre > theta (=1) — others gated to 0.
    active = pre > 1.0
    assert torch.allclose(z[active], pre[active])
    assert torch.all(z[~active] == 0)


def test_heaviside_ste_gradient_in_window():
    # When pre is exactly at theta, gradient mass should be 1/(2*bw) * grad_out.
    pre = torch.tensor([1.0, 1.0, 1.0])
    theta = torch.tensor([1.0, 1.0, 1.0], requires_grad=True)
    bw = 0.5
    g = heaviside_ste(pre, theta, bw)
    g.sum().backward()
    expected = -1.0 / (2 * bw)
    assert torch.allclose(theta.grad, torch.full((3,), expected))


def test_heaviside_ste_no_gradient_outside_window():
    pre = torch.tensor([5.0, 5.0])
    theta = torch.tensor([0.0, 0.0], requires_grad=True)
    bw = 0.5
    g = heaviside_ste(pre, theta, bw)
    g.sum().backward()
    assert torch.allclose(theta.grad, torch.zeros(2))


def test_loss_relu_runs_and_backprops():
    sae = ReLUSAE(d_in=16, n_latents=32)
    x = torch.randn(8, 16)
    out = sae(x)
    loss = sae_loss(x, out, sae, sparsity_coef=1e-3)
    loss.total.backward()
    assert sae.W_enc.grad is not None
    assert sae.W_dec.grad is not None


def test_loss_jumprelu_runs_and_backprops():
    sae = JumpReLUSAE(d_in=16, n_latents=32, log_theta_init=-2.0)
    x = torch.randn(8, 16)
    out = sae(x)
    loss = sae_loss(x, out, sae, sparsity_coef=1e-3)
    loss.total.backward()
    # log_theta should receive gradient via the STE (active dims fall in the rectangle window
    # depending on init; with random init at least one element is typically in-window).
    assert sae.W_enc.grad is not None


def test_loss_with_focal_reweighter_changes_recon_term_relative_to_baseline():
    torch.manual_seed(0)
    sae = ReLUSAE(d_in=16, n_latents=32)
    x = torch.randn(64, 16)
    out = sae(x)
    loss_baseline = sae_loss(x, out, sae, sparsity_coef=0.0)

    rew = FocalReweighter(d_in=16, scheme="fit", gamma=2.0, warmup_steps=0)
    rew.update(x, out.recon.detach())
    loss_focal = sae_loss(x, out, sae, reweighter=rew, sparsity_coef=0.0)

    # Focal recon term will differ unless weights are all 1; with gamma=2 they won't be.
    assert not torch.allclose(loss_baseline.recon, loss_focal.recon)
