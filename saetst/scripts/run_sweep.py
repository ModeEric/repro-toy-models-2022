"""Run the focal-loss sweep specified in spec.md.

For each (architecture, focal_scheme, gamma, sparsity_coef, seed) cell, train an
SAE and write the checkpoint + history to `runs/<sweep_name>/<run_name>/`.

Sparsity coefficients are swept to produce a Pareto curve; the spec calls for
hitting a range of L0 values. We sweep `sparsity_coef` over a log-spaced grid
and let the resulting L0 fall where it may.

Usage:
    python scripts/run_sweep.py --sweep relu_focal --steps 10000 --seeds 1
    python scripts/run_sweep.py --sweep all --steps 30000 --seeds 3
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from saetst.config import TrainConfig
from saetst.data import LMActivationStream, LMStreamConfig
from saetst.train import train_sae


SWEEPS: dict[str, dict] = {
    "relu_baseline": {
        "arch": "relu",
        "focal": [("none", 0.0)],
        "sparsity_grid": np.logspace(-4, -2, 5).tolist(),
    },
    "relu_focal_a": {
        "arch": "relu",
        "focal": [("relative", g) for g in (0.5, 1.0, 2.0)],
        "sparsity_grid": np.logspace(-4, -2, 5).tolist(),
    },
    "relu_focal_b": {
        "arch": "relu",
        "focal": [("fit", g) for g in (0.5, 1.0, 2.0)],
        "sparsity_grid": np.logspace(-4, -2, 5).tolist(),
    },
    "jumprelu_baseline": {
        "arch": "jumprelu",
        "focal": [("none", 0.0)],
        "sparsity_grid": np.logspace(-3, -1, 5).tolist(),
    },
    "jumprelu_focal_a": {
        "arch": "jumprelu",
        "focal": [("relative", g) for g in (0.5, 1.0, 2.0)],
        "sparsity_grid": np.logspace(-3, -1, 5).tolist(),
    },
    "jumprelu_focal_b": {
        "arch": "jumprelu",
        "focal": [("fit", g) for g in (0.5, 1.0, 2.0)],
        "sparsity_grid": np.logspace(-3, -1, 5).tolist(),
    },
}


def build_runs(sweep_name: str, n_seeds: int, steps: int) -> list[TrainConfig]:
    if sweep_name == "all":
        names = list(SWEEPS.keys())
    else:
        names = [sweep_name]

    out: list[TrainConfig] = []
    for sn in names:
        spec = SWEEPS[sn]
        for (scheme, gamma), sp, seed in itertools.product(
            spec["focal"], spec["sparsity_grid"], range(n_seeds)
        ):
            run_name = f"{sn}/g{gamma}_sp{sp:.4g}_seed{seed}"
            cfg = TrainConfig(
                arch=spec["arch"],
                focal_scheme=scheme,
                focal_gamma=gamma,
                sparsity_coef=sp,
                seed=seed,
                n_steps=steps,
                run_name=run_name,
                out_dir=f"runs/{sn}",
            )
            out.append(cfg)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", required=True,
                   help=f"one of: all, {', '.join(SWEEPS)}")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--dry-run", action="store_true",
                   help="just print the run plan and exit")
    args = p.parse_args()

    runs = build_runs(args.sweep, args.seeds, args.steps)
    print(f"[sweep] {len(runs)} runs planned")
    if args.dry_run:
        for cfg in runs:
            print(f"  {cfg.run_name}: arch={cfg.arch} scheme={cfg.focal_scheme} "
                  f"gamma={cfg.focal_gamma} sparsity={cfg.sparsity_coef:.4g} seed={cfg.seed}")
        return

    # One LM activation stream per arch/seed combo would be ideal, but the
    # streaming buffer is shared per run anyway. For simplicity we build a
    # fresh stream per run; this re-tokenizes from the start of the dataset.
    summary = []
    for i, cfg in enumerate(runs):
        print(f"\n[sweep] run {i+1}/{len(runs)}: {cfg.run_name}")
        stream_cfg = LMStreamConfig(
            model_name=cfg.model_name, layer=cfg.layer, site=cfg.site,
            dataset_name=cfg.dataset_name, dataset_split=cfg.dataset_split,
            text_field=cfg.text_field, seq_len=cfg.seq_len,
            buffer_size=cfg.activations_per_shard, batch_size=cfg.batch_size,
            device="cpu" if cfg.device == "auto" else cfg.device, seed=cfg.seed,
        )
        stream = LMActivationStream(stream_cfg)
        result = train_sae(cfg, stream, d_in=stream.d_in)
        summary.append({
            "run_name": cfg.run_name,
            "out_dir": result["out_dir"],
            "final": result["final"],
        })

    sweep_dir = Path("runs") / (args.sweep if args.sweep != "all" else "all")
    sweep_dir.mkdir(parents=True, exist_ok=True)
    with open(sweep_dir / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[sweep] summary written to {sweep_dir}/sweep_summary.json")


if __name__ == "__main__":
    main()
