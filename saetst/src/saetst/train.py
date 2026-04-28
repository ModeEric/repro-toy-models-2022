"""SAE training loop.

Designed to be small and inspectable — a single function that takes a
TrainConfig and an activation iterator, runs the loop, and returns a final
state dict with metrics. CLI wraps this in `scripts/train_sae.py`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from saetst.config import TrainConfig
from saetst.losses import FocalReweighter, sae_loss
from saetst.metrics import DeadLatentTracker
from saetst.models import make_sae


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    """Linear warmup, then linear decay to 0.1 * base_lr by end of training."""
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return base_lr * (1.0 - 0.9 * progress)


def train_sae(
    cfg: TrainConfig,
    stream: Iterator[torch.Tensor],
    *,
    d_in: int,
    eval_stream_factory=None,   # callable returning a fresh iterator for eval
    log_callback=None,          # optional fn(step, dict) for custom logging
) -> dict:
    """Train a SAE end-to-end.

    Args:
        cfg: training config.
        stream: iterator yielding [batch_size, d_in] activation tensors.
        d_in: activation dimension (passed explicitly so this fn doesn't need
              to peek at the stream).
        eval_stream_factory: optional callable() -> Iterator producing eval
              batches; if provided, run a small eval every cfg.eval_every steps.
        log_callback: optional callable(step, log_dict) for wandb/tb hooks.

    Returns:
        dict with `sae` (state_dict), `reweighter` (state_dict or None), `cfg`,
        `metrics` (training history list), and `final` (last log dict).
    """
    torch.manual_seed(cfg.seed)
    device = _resolve_device(cfg.device)

    sae = make_sae(cfg.arch, d_in=d_in, n_latents=d_in * cfg.expansion, device=device)
    sae.train()

    reweighter: Optional[FocalReweighter] = None
    if cfg.focal_scheme != "none":
        reweighter = FocalReweighter(
            d_in=d_in,
            scheme=cfg.focal_scheme,
            gamma=cfg.focal_gamma,
            ema_decay=cfg.focal_ema_decay,
            clip=(cfg.focal_clip_lo, cfg.focal_clip_hi),
            warmup_steps=cfg.focal_warmup_steps,
        ).to(device)

    optim = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    dead_tracker = DeadLatentTracker(
        n_latents=sae.n_latents,
        window_tokens=cfg.dead_token_window,
        device=str(device),
    )

    history: list[dict] = []
    pbar = tqdm(range(cfg.n_steps), desc=f"train {cfg.run_name or cfg.arch}")
    t0 = time.time()
    last_log = {}

    for step in pbar:
        try:
            x = next(stream)
        except StopIteration:
            print(f"[train] activation stream exhausted at step {step}")
            break
        x = x.to(device)

        # LR schedule
        for g in optim.param_groups:
            g["lr"] = _lr_at(step, cfg.lr, cfg.warmup_lr_steps, cfg.n_steps)

        out = sae(x)
        loss_out = sae_loss(x, out, sae, reweighter=reweighter, sparsity_coef=cfg.sparsity_coef)

        optim.zero_grad(set_to_none=True)
        loss_out.total.backward()
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.grad_clip)
        optim.step()
        sae.renorm_decoder()

        if reweighter is not None:
            reweighter.update(x.detach(), out.recon.detach())

        with torch.no_grad():
            dead_tracker.update(out.z.detach())

        if (step + 1) % cfg.log_every == 0 or step == 0:
            log = {
                "step": step,
                "loss": float(loss_out.total.item()),
                "recon": float(loss_out.recon.item()),
                "sparsity": float(loss_out.sparsity.item()),
                "l0": float(loss_out.l0.item()),
                "dead_frac": dead_tracker.dead_fraction(),
                "elapsed_s": time.time() - t0,
            }
            if reweighter is not None:
                w = loss_out.weights
                log["w_min"] = float(w.min().item())
                log["w_max"] = float(w.max().item())
                log["w_std"] = float(w.std().item())
                log["clip_hits"] = int(reweighter.clip_hits.item())
            history.append(log)
            last_log = log
            pbar.set_postfix(
                loss=f"{log['loss']:.4f}",
                l0=f"{log['l0']:.1f}",
                dead=f"{log['dead_frac']:.3f}",
            )
            if log_callback is not None:
                log_callback(step, log)

        if eval_stream_factory is not None and (step + 1) % cfg.eval_every == 0:
            from saetst.metrics import evaluate_recon
            eval_stream = eval_stream_factory()
            stats = evaluate_recon(sae, eval_stream, n_batches=10, device=str(device))
            eval_log = {
                "step": step,
                "eval_mse": stats.mse,
                "eval_fve": stats.fve,
                "eval_l0": stats.l0,
            }
            history.append({"eval": eval_log})
            if log_callback is not None:
                log_callback(step, eval_log)

    out_dir = Path(cfg.out_dir)
    if cfg.run_name:
        out_dir = out_dir / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sae_state = {k: v.cpu() for k, v in sae.state_dict().items()}
    rew_state = (
        {k: v.cpu() for k, v in reweighter.state_dict().items()} if reweighter else None
    )
    torch.save(
        {
            "sae": sae_state,
            "reweighter": rew_state,
            "cfg": asdict(cfg),
            "d_in": d_in,
        },
        out_dir / "checkpoint.pt",
    )
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    return {
        "sae": sae,
        "reweighter": reweighter,
        "cfg": cfg,
        "history": history,
        "final": last_log,
        "out_dir": str(out_dir),
        "dead_tracker": dead_tracker,
    }
