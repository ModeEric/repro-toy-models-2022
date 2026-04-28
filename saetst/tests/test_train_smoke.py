"""End-to-end smoke test: train a tiny SAE on synthetic data for a few steps.

This is fast (CPU-only, < 30s) and catches integration bugs across models +
losses + reweighter + training loop without needing network or a GPU.
"""

from __future__ import annotations

import torch

from saetst.config import TrainConfig
from saetst.data import SyntheticActivationStream
from saetst.train import train_sae


def _stream(d_in=32, n_features=128, k=5, batch=128, seed=0, dim_hard=False):
    return SyntheticActivationStream(
        d_in=d_in, n_features=n_features, k=k, batch_size=batch, seed=seed,
        device="cpu", dim_hard=dim_hard,
    )


def test_relu_sae_smoke_trains_and_lowers_loss():
    d_in = 32
    cfg = TrainConfig(
        arch="relu",
        expansion=4,
        n_steps=200,
        batch_size=128,
        sparsity_coef=1e-3,
        warmup_lr_steps=20,
        log_every=50,
        eval_every=10000,
        device="cpu",
        out_dir="/tmp/saetst_smoke",
        run_name="smoke_relu",
        dead_token_window=10_000,
    )
    result = train_sae(cfg, _stream(d_in=d_in), d_in=d_in)
    history = [h for h in result["history"] if "loss" in h]
    assert len(history) >= 2
    assert history[-1]["loss"] < history[0]["loss"]
    # Some latents should be alive at end.
    assert result["dead_tracker"].dead_fraction() < 1.0


def test_jumprelu_sae_smoke_trains_and_lowers_loss():
    d_in = 32
    cfg = TrainConfig(
        arch="jumprelu",
        expansion=4,
        n_steps=200,
        batch_size=128,
        sparsity_coef=1e-3,
        warmup_lr_steps=20,
        log_every=50,
        eval_every=10000,
        device="cpu",
        out_dir="/tmp/saetst_smoke",
        run_name="smoke_jumprelu",
        dead_token_window=10_000,
    )
    result = train_sae(cfg, _stream(d_in=d_in), d_in=d_in)
    history = [h for h in result["history"] if "loss" in h]
    assert history[-1]["loss"] < history[0]["loss"]


def test_focal_smoke_concentrates_weights_on_hard_dims():
    """With dim_hard=True, certain dims have 5x more noise. Focal scheme A
    (relative-error) should give those dims systematically higher weight after
    a few hundred steps."""
    torch.manual_seed(0)
    d_in = 32
    stream = _stream(d_in=d_in, dim_hard=True)
    cfg = TrainConfig(
        arch="relu",
        expansion=4,
        n_steps=300,
        batch_size=128,
        sparsity_coef=1e-3,
        warmup_lr_steps=20,
        log_every=50,
        eval_every=10000,
        device="cpu",
        out_dir="/tmp/saetst_smoke",
        run_name="smoke_focal",
        dead_token_window=10_000,
        focal_scheme="relative",
        focal_gamma=2.0,
        focal_warmup_steps=50,
        focal_ema_decay=0.95,
    )
    result = train_sae(cfg, stream, d_in=d_in)
    rew = result["reweighter"]
    assert rew is not None
    w = rew.weights().cpu()
    # Hard mask is stored on the stream; the dims with the multiplier should rank above mean.
    hard_mask = stream.hard_mask.cpu() > 0  # bool mask, [d_in]
    if hard_mask.any():
        # Mean weight on hard dims should be strictly greater than mean weight on easy dims.
        assert w[hard_mask].mean().item() > w[~hard_mask].mean().item()
