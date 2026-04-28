"""Pareto plot: reconstruction MSE (or 1-FVE) vs. mean L0, per condition.

Reads `runs/<sweep>/metrics.json` aggregates produced by `evaluate.py` and
emits a per-architecture PNG showing every (focal_scheme, gamma) condition
as a connected line across sparsity_coef points, with one marker per seed
and a frontier line through condition means.

Usage:
    python scripts/plot_pareto.py --metrics runs/relu_focal_b/metrics.json \\
        --out runs/relu_focal_b/pareto.png
    python scripts/plot_pareto.py --metrics runs --out runs/all_pareto.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(path: Path) -> list[dict]:
    if path.is_dir():
        out = []
        for p in path.rglob("metrics.json"):
            with open(p) as f:
                blob = json.load(f)
            if isinstance(blob, list):
                out.extend(blob)
            else:
                out.append(blob)
        return out
    with open(path) as f:
        blob = json.load(f)
    return blob if isinstance(blob, list) else [blob]


def condition_key(m: dict) -> tuple:
    return (m["arch"], m["focal_scheme"], m["focal_gamma"])


def condition_label(arch: str, scheme: str, gamma: float) -> str:
    if scheme == "none" or gamma == 0:
        return f"{arch} baseline"
    return f"{arch} {scheme} γ={gamma}"


def plot_pareto(metrics: list[dict], out_path: Path, y_metric: str = "eval_mse") -> None:
    by_arch: dict[str, list[dict]] = defaultdict(list)
    for m in metrics:
        by_arch[m["arch"]].append(m)

    n = len(by_arch)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for ax, (arch, ms) in zip(axes[0], by_arch.items()):
        by_cond = defaultdict(list)
        for m in ms:
            by_cond[(m["focal_scheme"], m["focal_gamma"])].append(m)

        for (scheme, gamma), entries in sorted(by_cond.items()):
            # Group by sparsity_coef and average over seeds.
            by_sp = defaultdict(list)
            for e in entries:
                by_sp[e["sparsity_coef"]].append(e)
            xs, ys, yerr = [], [], []
            for sp in sorted(by_sp):
                ls = [e["eval_l0"] for e in by_sp[sp]]
                vs = [e[y_metric] for e in by_sp[sp]]
                xs.append(np.mean(ls))
                ys.append(np.mean(vs))
                yerr.append(np.std(vs) / max(len(vs) ** 0.5, 1))
            label = condition_label(arch, scheme, gamma)
            ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=label)

        ax.set_xlabel("Mean L0 (active latents per token)")
        ax.set_ylabel(y_metric)
        ax.set_title(f"{arch} — Pareto frontier")
        ax.set_yscale("log") if y_metric == "eval_mse" else None
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"[plot_pareto] wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True, help="metrics.json file or runs/ root")
    p.add_argument("--out", required=True)
    p.add_argument("--y", default="eval_mse",
                   choices=["eval_mse", "eval_fve", "dead_frac"])
    args = p.parse_args()

    metrics = load_metrics(Path(args.metrics))
    if not metrics:
        raise SystemExit(f"no metrics found at {args.metrics}")
    plot_pareto(metrics, Path(args.out), y_metric=args.y)


if __name__ == "__main__":
    main()
