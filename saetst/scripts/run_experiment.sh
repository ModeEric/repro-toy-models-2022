#!/usr/bin/env bash
# Build cache → sweep → eval → plots, in one paste-proof shell script.
#
# Usage (from repo root saetst/):
#   bash scripts/run_experiment.sh                  # quick first sweep (1 seed, 10k steps)
#   bash scripts/run_experiment.sh --full           # spec sweep (3 seeds, 30k steps)
#   bash scripts/run_experiment.sh --skip-cache     # reuse existing cache
#
# Env knobs:
#   CACHE_DIR=caches/pythia70m_l3   # where to put / read the activation cache
#   N_TARGET=10_000_000             # acts to cache (10M ≈ 10 GB, in spec range)
#   DATASET=Skylion007/openwebtext  # source corpus
#   PYTHON=.venv/bin/python         # interpreter

set -euo pipefail

CACHE_DIR="${CACHE_DIR:-caches/pythia70m_l3}"
N_TARGET="${N_TARGET:-10000000}"
DATASET="${DATASET:-Skylion007/openwebtext}"
PYTHON="${PYTHON:-.venv/bin/python}"

STEPS=10000
SEEDS=1
SKIP_CACHE=false

for arg in "$@"; do
    case "$arg" in
        --full)       STEPS=30000; SEEDS=3 ;;
        --skip-cache) SKIP_CACHE=true ;;
        --steps=*)    STEPS="${arg#*=}" ;;
        --seeds=*)    SEEDS="${arg#*=}" ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

echo "[run] config: cache=$CACHE_DIR n_target=$N_TARGET dataset=$DATASET steps=$STEPS seeds=$SEEDS"

# 1) Build activation cache (skip if --skip-cache or cache already exists)
if $SKIP_CACHE && [ -f "$CACHE_DIR/meta.json" ]; then
    echo "[run] skipping cache build; reusing $CACHE_DIR"
elif [ -f "$CACHE_DIR/meta.json" ]; then
    echo "[run] cache already exists at $CACHE_DIR; skipping build"
    echo "      (delete $CACHE_DIR or pass --skip-cache=false to force rebuild)"
else
    echo "[run] building activation cache → $CACHE_DIR"
    "$PYTHON" scripts/cache_activations.py \
        --out "$CACHE_DIR" \
        --dataset "$DATASET" \
        --layer 3 --site resid_post \
        --seq-len 256 --n-target "$N_TARGET" \
        --batch-seqs 512 --device cuda
fi

# 2) Sweep — every run reads from the cache, no LM in the training loop
echo "[run] sweep: $STEPS steps × $SEEDS seeds, all conditions"
"$PYTHON" scripts/run_sweep.py \
    --sweep all \
    --steps "$STEPS" \
    --seeds "$SEEDS" \
    --cache-dir "$CACHE_DIR" 2>&1 | tee runs/sweep.log

# 3) Eval (recon, dead latents, downstream loss recovered)
echo "[run] evaluating all checkpoints"
"$PYTHON" scripts/evaluate.py \
    --sweep all \
    --cache-dir "$CACHE_DIR" \
    --downstream-n 16 \
    --device auto

# 4) Plots
echo "[run] generating Pareto + dead-frac plots"
"$PYTHON" scripts/plot_pareto.py --metrics runs --out runs/pareto_mse.png   --y eval_mse
"$PYTHON" scripts/plot_pareto.py --metrics runs --out runs/pareto_dead.png  --y dead_frac

echo "[run] done."
echo "      sweep summary: runs/all/sweep_summary.json"
echo "      plots:         runs/pareto_mse.png  runs/pareto_dead.png"
