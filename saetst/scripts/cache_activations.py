"""Dump LM activations to a sharded fp16 binary cache once, so every SAE in a
sweep can read from disk instead of re-running the LM forward.

Output layout:
    out_dir/
      meta.json
      shard_0000.bin
      shard_0001.bin
      ...

Run once before the sweep:

    python scripts/cache_activations.py \\
        --out caches/pythia70m_layer3 \\
        --model EleutherAI/pythia-70m-deduped --layer 3 --site resid_post \\
        --dataset NeelNanda/pile-10k --seq-len 256 \\
        --n-target 50_000_000 --n-per-shard 1_000_000 \\
        --batch-seqs 256 --device cuda

50M activations × 512 dims × fp16 ≈ 50 GB. Each shard is then ~1 GB.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from saetst.data import CACHE_META, _shard_path  # noqa: E402
from saetst.utils import default_lm_dtype, resolve_device  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="cache output directory")
    p.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--site", default="resid_post")
    p.add_argument("--dataset", default="NeelNanda/pile-10k")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--n-target", type=int, default=50_000_000,
                   help="target number of activation vectors to write")
    p.add_argument("--n-per-shard", type=int, default=1_000_000)
    p.add_argument("--batch-seqs", type=int, default=256,
                   help="sequences per LM forward pass")
    p.add_argument("--device", default="auto")
    p.add_argument("--lm-dtype", default="auto",
                   help="auto | bfloat16 | float16 | float32")
    p.add_argument("--store-dtype", default="float16",
                   choices=["float16", "float32", "bfloat16"],
                   help="on-disk dtype; float16 halves disk vs fp32 for free at this scale")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = resolve_device(args.device)
    if args.lm_dtype == "auto":
        lm_dtype = default_lm_dtype(device)
    else:
        lm_dtype = getattr(torch, args.lm_dtype)
    store_dtype_torch = getattr(torch, args.store_dtype)
    # numpy doesn't have a native bfloat16 dtype — fall back to fp16 storage.
    if args.store_dtype == "bfloat16":
        np_store = np.float16
        store_name = "float16"
        print("[cache] note: numpy lacks bfloat16; storing as float16")
    else:
        np_store = np.dtype(args.store_dtype).type
        store_name = args.store_dtype

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[cache] device={device} lm_dtype={lm_dtype} store={store_name}")
    print(f"[cache] target={args.n_target:,} acts × seq_len={args.seq_len} → ~{args.n_target/1e6:.0f}M tokens")

    from transformer_lens import HookedTransformer
    from datasets import load_dataset

    torch.manual_seed(args.seed)
    model = HookedTransformer.from_pretrained(args.model, device=str(device))
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    model = model.to(lm_dtype)
    d_in = model.cfg.d_model
    hook_name = f"blocks.{args.layer}.hook_{args.site}"
    pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id

    ds = load_dataset(args.dataset, split=args.dataset_split, streaming=True)
    ds_iter = iter(ds)

    # Pre-allocate one shard buffer; flush whenever it fills.
    shard_buf = np.empty((args.n_per_shard, d_in), dtype=np_store)
    shard_fill = 0
    shard_idx = 0
    n_written = 0

    pbar = tqdm(total=args.n_target, unit="acts", smoothing=0.05)
    t0 = time.time()
    docs_seen = 0
    docs_skipped = 0

    while n_written < args.n_target:
        # Collect a batch of token sequences.
        seqs = []
        while len(seqs) < args.batch_seqs:
            try:
                row = next(ds_iter)
            except StopIteration:
                print(f"[cache] dataset exhausted after {docs_seen} docs; "
                      f"wrote {n_written:,} / {args.n_target:,} activations")
                break
            docs_seen += 1
            text = row[args.text_field]
            tok = model.to_tokens(text, prepend_bos=True)
            if tok.shape[-1] < 8:
                docs_skipped += 1
                continue
            tok = tok[:, : args.seq_len]
            seqs.append(tok)
        if not seqs:
            break

        max_len = max(s.shape[-1] for s in seqs)
        batched = torch.full(
            (len(seqs), max_len), pad_id, dtype=torch.long, device=device,
        )
        attention = torch.zeros(len(seqs), max_len, dtype=torch.bool, device=device)
        for i, s in enumerate(seqs):
            batched[i, : s.shape[-1]] = s[0]
            attention[i, : s.shape[-1]] = True

        captured: dict = {}
        def hook(act, hook):  # noqa: A002
            captured["x"] = act.detach()
        with model.hooks(fwd_hooks=[(hook_name, hook)]):
            model(batched, return_type=None)
        acts = captured["x"]

        # Drop BOS + padding, cast to store dtype on the device, then to numpy.
        attention[:, 0] = False
        flat = acts[attention].to(store_dtype_torch).cpu().numpy()
        if flat.dtype != np_store:
            flat = flat.astype(np_store)
        # Append into shard buffer; flush when full.
        i = 0
        while i < len(flat) and n_written < args.n_target:
            take = min(args.n_per_shard - shard_fill, len(flat) - i, args.n_target - n_written)
            shard_buf[shard_fill : shard_fill + take] = flat[i : i + take]
            shard_fill += take
            i += take
            if shard_fill == args.n_per_shard:
                _shard_path(out, shard_idx).write_bytes(shard_buf.tobytes())
                shard_idx += 1
                n_written += args.n_per_shard
                shard_fill = 0
                pbar.update(args.n_per_shard)

    # Flush a final partial shard if we have anything in it AND we hit the target;
    # otherwise drop it so all shards are uniform-size (simpler for the stream).
    pbar.close()

    meta = {
        "d_in": int(d_in),
        "n_per_shard": int(args.n_per_shard),
        "n_shards": int(shard_idx),
        "n_total": int(shard_idx * args.n_per_shard),
        "dtype": store_name,
        "model_name": args.model,
        "layer": args.layer,
        "site": args.site,
        "dataset_name": args.dataset,
        "dataset_split": args.dataset_split,
        "seq_len": args.seq_len,
        "elapsed_s": time.time() - t0,
        "docs_seen": docs_seen,
        "docs_skipped": docs_skipped,
    }
    with open(out / CACHE_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[cache] wrote {shard_idx} shards × {args.n_per_shard:,} = "
          f"{shard_idx * args.n_per_shard:,} activations to {out}")
    print(f"[cache] meta: {out / CACHE_META}")


if __name__ == "__main__":
    main()
