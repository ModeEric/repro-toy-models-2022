"""Unit tests for the focal reweighter — the novel piece of the spec.

These cover:
  * gamma=0 → uniform weights for both schemes
  * Scheme A (relative): worse dims get larger weights, ranking preserved
  * Scheme B (fit): unexplained-variance dims get larger weights, capped at 1
  * EMA actually decays (ema_decay=0 should track only the latest batch)
  * Warmup gates focal weights off
  * Clipping triggers when one dim is wildly worse
  * Mean of returned weights is ~1 (renormalization invariant)
"""

from __future__ import annotations

import math

import torch

from saetst.losses import FocalReweighter


def _ones(d):
    return torch.ones(d)


def test_gamma_zero_returns_uniform_for_both_schemes():
    for scheme in ("relative", "fit"):
        rew = FocalReweighter(d_in=8, scheme=scheme, gamma=0.0, warmup_steps=0)
        # Update with biased error so EMAs aren't symmetric.
        x = torch.randn(32, 8)
        recon = x + torch.randn(32, 8) * torch.linspace(0.1, 1.0, 8)
        rew.update(x, recon)
        w = rew.weights()
        assert torch.allclose(w, _ones(8))


def test_warmup_gates_weights_off():
    rew = FocalReweighter(d_in=4, scheme="relative", gamma=2.0, warmup_steps=5)
    # Heterogeneous per-dim error: dim 0 has the worst error.
    x = torch.zeros(16, 4)
    recon = torch.zeros(16, 4)
    recon[:, 0] = 4.0
    recon[:, 1] = 1.0
    recon[:, 2] = 0.5
    recon[:, 3] = 0.25
    for _ in range(3):
        rew.update(x, recon)
    w = rew.weights()
    assert torch.allclose(w, _ones(4)), "weights should be uniform during warmup"
    for _ in range(10):
        rew.update(x, recon)
    w = rew.weights()
    assert not torch.allclose(w, _ones(4))


def test_relative_scheme_amplifies_worse_dims():
    rew = FocalReweighter(d_in=4, scheme="relative", gamma=2.0, warmup_steps=0)
    # Dim 3 has 4x the squared error of dim 0; dim 2 has 3x; dim 1 has 2x.
    x = torch.zeros(64, 4)
    recon = torch.zeros(64, 4)
    recon[:, 0] = 1.0
    recon[:, 1] = math.sqrt(2.0)
    recon[:, 2] = math.sqrt(3.0)
    recon[:, 3] = 2.0
    rew.update(x, recon)
    w = rew.weights()
    # Ranking preserved: w[0] < w[1] < w[2] < w[3]
    assert w[0] < w[1] < w[2] < w[3]
    # Mean is approximately 1.
    assert abs(w.mean().item() - 1.0) < 1e-5


def test_fit_scheme_caps_shortfall_at_one():
    # When recon is wildly worse than per-dim variance, fit < 0 → shortfall clamped to 1.
    rew = FocalReweighter(d_in=4, scheme="fit", gamma=1.0, warmup_steps=0)
    x = torch.randn(64, 4) * 0.1            # small variance
    recon = x + torch.randn(64, 4) * 5.0    # huge error >> Var(x)
    rew.update(x, recon)
    w = rew.weights()
    # All dims have shortfall ≈ 1, so all weights ≈ 1 after renorm. Allow some noise.
    assert torch.allclose(w, _ones(4), atol=1e-3)


def test_fit_scheme_focuses_on_underexplained_dims():
    # Two dims have variance 1, two dims have variance 1. Recon explains the
    # first two perfectly but is off by ~0.5 on the last two. The latter should
    # get higher weight.
    torch.manual_seed(0)
    rew = FocalReweighter(d_in=4, scheme="fit", gamma=1.0, warmup_steps=0, ema_decay=0.0)
    x = torch.randn(2048, 4)
    recon = x.clone()
    recon[:, 2] = 0.0    # unexplained
    recon[:, 3] = 0.0    # unexplained
    rew.update(x, recon)
    w = rew.weights()
    # First two should be lower than last two.
    assert w[0] < w[2] and w[0] < w[3]
    assert w[1] < w[2] and w[1] < w[3]


def test_ema_decay_zero_uses_only_latest_batch():
    rew = FocalReweighter(d_in=4, scheme="relative", gamma=1.0, warmup_steps=0, ema_decay=0.0)
    # First batch: dim 0 has the worst error.
    x = torch.zeros(8, 4)
    bad0 = torch.zeros(8, 4); bad0[:, 0] = 1.0
    rew.update(x, bad0)
    # Second batch: dim 3 has the worst error. With ema_decay=0, EMA fully replaced.
    bad3 = torch.zeros(8, 4); bad3[:, 3] = 1.0
    rew.update(x, bad3)
    w = rew.weights()
    assert w[3] > w[0]


def test_clipping_triggers_and_limits_weights():
    rew = FocalReweighter(
        d_in=4, scheme="relative", gamma=4.0, warmup_steps=0,
        clip=(0.5, 2.0),
    )
    # Dim 3 absurdly worse than the rest. Without clipping, gamma=4 would push
    # dim 3's pre-renorm weight to 256 — clipping caps at 2.0.
    x = torch.zeros(8, 4)
    recon = torch.zeros(8, 4)
    recon[:, 3] = 100.0
    rew.update(x, recon)
    w = rew.weights()
    # Bound after renormalization: max(w) ≤ clip_hi / clip_lo (since renorm divides by
    # mean ≥ clip_lo). For clip=(0.5, 2.0) that's 4.0.
    assert w.max().item() <= 2.0 / 0.5 + 1e-5
    assert int(rew.clip_hits.item()) >= 1
    # And the clipping-triggered dim is still the largest.
    assert w.argmax().item() == 3


def test_weights_detached_from_graph():
    rew = FocalReweighter(d_in=4, scheme="fit", gamma=1.0, warmup_steps=0)
    x = torch.randn(8, 4, requires_grad=True)
    recon = torch.zeros_like(x)
    rew.update(x, recon)
    w = rew.weights()
    assert not w.requires_grad
