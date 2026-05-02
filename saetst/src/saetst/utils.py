"""Small shared utilities."""

from __future__ import annotations

import torch


def resolve_device(name: str) -> torch.device:
    """Resolve "auto" → cuda > mps > cpu, else honor the explicit string."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def default_lm_dtype(device: torch.device | str) -> torch.dtype:
    """bf16 on cuda, fp32 elsewhere. MPS bf16 support is uneven across torch
    versions and CPU bf16 is often slower than fp32 in practice."""
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type == "cuda":
        return torch.bfloat16
    return torch.float32
