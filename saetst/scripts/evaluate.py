"""Evaluate trained SAEs and produce metrics tables for the writeup.

For each checkpoint under `runs/<sweep>/`, computes:
  * eval MSE, FVE, mean L0 on a held-out activation slice
  * dead-latent fraction over a long-window pass (default 200k tokens)
  * downstream loss recovered (small held-out text set)

Writes per-checkpoint json next to each checkpoint, and a single
`runs/<sweep>/metrics.json` aggregate suitable for plotting.

Usage:
    python scripts/evaluate.py --sweep relu_focal_b
    python scripts/evaluate.py --sweep all --eval-batches 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from saetst.config import TrainConfig
from saetst.data import LMActivationStream, LMStreamConfig
from saetst.metrics import DeadLatentTracker, evaluate_recon, downstream_loss_recovered
from saetst.models import make_sae


def load_sae(ckpt_path: Path):
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**blob["cfg"])
    d_in = blob["d_in"]
    sae = make_sae(cfg.arch, d_in=d_in, n_latents=d_in * cfg.expansion)
    sae.load_state_dict(blob["sae"])
    sae.eval()
    return sae, cfg, d_in


def find_checkpoints(root: Path) -> list[Path]:
    return sorted(root.rglob("checkpoint.pt"))


def stream_factory(cfg: TrainConfig, seed: int):
    def make():
        sc = LMStreamConfig(
            model_name=cfg.model_name, layer=cfg.layer, site=cfg.site,
            dataset_name=cfg.dataset_name, dataset_split=cfg.dataset_split,
            text_field=cfg.text_field, seq_len=cfg.seq_len,
            buffer_size=cfg.activations_per_shard, batch_size=cfg.batch_size,
            device="cpu", seed=seed,
        )
        return LMActivationStream(sc)
    return make


def evaluate_checkpoint(
    ckpt: Path,
    eval_batches: int,
    dead_batches: int,
    downstream_texts: list[str] | None,
) -> dict:
    sae, cfg, d_in = load_sae(ckpt)
    factory = stream_factory(cfg, seed=cfg.seed + 9999)

    # Recon stats
    eval_stream = factory()
    recon = evaluate_recon(sae, eval_stream, n_batches=eval_batches, device="cpu")

    # Dead latents over a long window
    dead_stream = factory()
    tracker = DeadLatentTracker(
        n_latents=sae.n_latents,
        window_tokens=cfg.dead_token_window,
        device="cpu",
    )
    seen = 0
    for _ in range(dead_batches):
        try:
            x = next(dead_stream)
        except StopIteration:
            break
        with torch.no_grad():
            out = sae(x)
        tracker.update(out.z)
        seen += x.shape[0]

    metrics = {
        "ckpt": str(ckpt),
        "arch": cfg.arch,
        "focal_scheme": cfg.focal_scheme,
        "focal_gamma": cfg.focal_gamma,
        "sparsity_coef": cfg.sparsity_coef,
        "seed": cfg.seed,
        "eval_mse": recon.mse,
        "eval_fve": recon.fve,
        "eval_l0": recon.l0,
        "n_eval_tokens": recon.n_tokens,
        "dead_frac": tracker.dead_fraction(),
        "low_freq_frac_1e-6": tracker.low_freq_fraction(1e-6),
        "high_freq_frac_0.1": tracker.high_freq_fraction(0.1),
        "n_dead_tokens": seen,
    }

    if downstream_texts:
        from transformer_lens import HookedTransformer  # noqa: WPS433
        model = HookedTransformer.from_pretrained(cfg.model_name, device="cpu")
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        hook_name = f"blocks.{cfg.layer}.hook_{cfg.site}"
        ds_metrics = downstream_loss_recovered(
            sae, model, hook_name, downstream_texts, device="cpu",
        )
        metrics["downstream"] = ds_metrics

    out_path = ckpt.parent / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", required=True, help="sweep dir under runs/, or 'all'")
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--dead-batches", type=int, default=20)
    p.add_argument("--no-downstream", action="store_true",
                   help="skip the downstream loss-recovered metric (slow without GPU)")
    p.add_argument("--downstream-n", type=int, default=8,
                   help="number of held-out texts for downstream eval")
    args = p.parse_args()

    if args.sweep == "all":
        roots = [Path("runs") / d.name for d in Path("runs").iterdir() if d.is_dir()]
    else:
        roots = [Path("runs") / args.sweep]

    downstream_texts = None
    if not args.no_downstream:
        from datasets import load_dataset  # noqa: WPS433
        # Use the same dataset as training; sample `downstream_n` distinct texts.
        ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
        downstream_texts = []
        for row in ds:
            t = row["text"]
            if len(t) > 200:
                downstream_texts.append(t)
            if len(downstream_texts) >= args.downstream_n:
                break

    all_metrics = []
    for root in roots:
        ckpts = find_checkpoints(root)
        print(f"[eval] {root}: {len(ckpts)} checkpoints")
        for c in ckpts:
            print(f"  evaluating {c}")
            m = evaluate_checkpoint(c, args.eval_batches, args.dead_batches, downstream_texts)
            all_metrics.append(m)

        out = root / "metrics.json"
        with open(out, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
