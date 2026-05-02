"""Round-trip the CachedActivationStream through a fake on-disk cache.

We bypass the LM here — write deterministic fp16 shards directly so the test
runs in milliseconds and doesn't need a model download.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from saetst.data import CACHE_META, CachedActivationStream, _shard_path


def _write_fake_cache(tmpdir: Path, n_shards=4, n_per_shard=100, d_in=8, dtype=np.float16):
    tmpdir.mkdir(parents=True, exist_ok=True)
    for s in range(n_shards):
        # Each token is the integer (shard*n_per_shard + i), broadcast across d_in.
        # That makes shard membership trivially auditable from the value.
        idx = np.arange(s * n_per_shard, (s + 1) * n_per_shard, dtype=np.int64)
        arr = np.broadcast_to(idx[:, None], (n_per_shard, d_in)).astype(dtype)
        _shard_path(tmpdir, s).write_bytes(np.ascontiguousarray(arr).tobytes())
    meta = {
        "d_in": d_in, "n_per_shard": n_per_shard, "n_shards": n_shards,
        "n_total": n_shards * n_per_shard, "dtype": np.dtype(dtype).name,
    }
    with open(tmpdir / CACHE_META, "w") as f:
        json.dump(meta, f)


def test_cache_round_trip_yields_expected_shape_and_dtype(tmp_path):
    _write_fake_cache(tmp_path)
    stream = CachedActivationStream(tmp_path, batch_size=32, repeat=False, shuffle=False)
    assert stream.d_in == 8
    batch = next(stream)
    assert batch.shape == (32, 8)
    assert batch.dtype == torch.float32   # always fp32 out, regardless of disk dtype


def test_cache_no_shuffle_preserves_intra_shard_order(tmp_path):
    _write_fake_cache(tmp_path, n_shards=2, n_per_shard=64, d_in=4)
    stream = CachedActivationStream(tmp_path, batch_size=8, shuffle=False, repeat=False)
    batches = []
    for b in stream:
        batches.append(b)
    full = torch.cat(batches)
    # With shuffle=False, values should be a permutation of [0..127] — the shard
    # order is also unshuffled, so values are 0,0,0,0,1,1,1,1,...,127,127,127,127.
    expected = torch.arange(128, dtype=torch.float32).repeat_interleave(4).reshape(128, 4)
    assert torch.equal(full, expected)


def test_cache_repeat_true_yields_infinite_batches(tmp_path):
    _write_fake_cache(tmp_path, n_shards=2, n_per_shard=50, d_in=4)
    # Total 100 tokens; 5 batches of 20 = one epoch. Pull 12 batches.
    stream = CachedActivationStream(tmp_path, batch_size=20, shuffle=True, repeat=True, seed=0)
    for _ in range(12):
        b = next(stream)
        assert b.shape == (20, 4)


def test_cache_shuffle_changes_order_with_different_seeds(tmp_path):
    _write_fake_cache(tmp_path, n_shards=4, n_per_shard=32, d_in=2)
    s0 = CachedActivationStream(tmp_path, batch_size=16, shuffle=True, repeat=False, seed=0)
    s1 = CachedActivationStream(tmp_path, batch_size=16, shuffle=True, repeat=False, seed=1)
    b0 = next(s0); b1 = next(s1)
    # Different seeds → different first batch with high probability.
    assert not torch.equal(b0, b1)


def test_cache_missing_meta_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        CachedActivationStream(tmp_path, batch_size=4)
