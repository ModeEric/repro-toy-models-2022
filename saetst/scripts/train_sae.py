"""Train a single SAE from a YAML config.

Usage:
    python scripts/train_sae.py configs/relu_baseline.yaml
    python scripts/train_sae.py configs/relu_focal_b.yaml --override n_steps=500
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import yaml

# Make `import saetst` work whether or not the package is pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from saetst.config import TrainConfig
from saetst.data import LMActivationStream, LMStreamConfig, SyntheticActivationStream
from saetst.train import train_sae


def parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"override must be key=value, got: {item}")
        k, v = item.split("=", 1)
        # Try to coerce int/float/bool, else keep string.
        for cast in (int, float):
            try:
                out[k] = cast(v); break
            except ValueError:
                continue
        else:
            if v.lower() in ("true", "false"):
                out[k] = v.lower() == "true"
            else:
                out[k] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str, help="path to YAML config")
    p.add_argument("--override", "-o", action="append", default=[],
                   help="override config field, e.g. -o n_steps=500")
    p.add_argument("--synthetic", action="store_true",
                   help="use synthetic activations instead of LM (for testing)")
    p.add_argument("--synthetic-d-in", type=int, default=64)
    args = p.parse_args()

    with open(args.config) as f:
        raw = yaml.safe_load(f) or {}
    raw.update(parse_overrides(args.override))

    valid = {f.name for f in fields(TrainConfig)}
    bad = set(raw) - valid
    if bad:
        raise SystemExit(f"unknown config fields: {bad}")

    cfg = TrainConfig(**raw)
    if cfg.run_name is None:
        cfg.run_name = Path(args.config).stem

    print(f"[train_sae] config: {cfg}")

    if args.synthetic:
        d_in = args.synthetic_d_in
        stream = SyntheticActivationStream(
            d_in=d_in,
            n_features=cfg.expansion * d_in,
            k=8,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
            device=cfg.device if cfg.device != "auto" else "cpu",
        )
    else:
        stream_cfg = LMStreamConfig(
            model_name=cfg.model_name,
            layer=cfg.layer,
            site=cfg.site,
            dataset_name=cfg.dataset_name,
            dataset_split=cfg.dataset_split,
            text_field=cfg.text_field,
            seq_len=cfg.seq_len,
            buffer_size=cfg.activations_per_shard,
            batch_size=cfg.batch_size,
            device="cpu" if cfg.device == "auto" else cfg.device,
            seed=cfg.seed,
        )
        stream = LMActivationStream(stream_cfg)
        d_in = stream.d_in

    result = train_sae(cfg, stream, d_in=d_in)
    print(f"[train_sae] done. final: {result['final']}")
    print(f"[train_sae] checkpoint: {result['out_dir']}/checkpoint.pt")


if __name__ == "__main__":
    main()
