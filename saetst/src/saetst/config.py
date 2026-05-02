"""Training configuration objects (kept dataclass-simple for serialization)."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional


@dataclass
class TrainConfig:
    # Subject model + activations
    model_name: str = "EleutherAI/pythia-70m-deduped"
    layer: int = 3
    site: str = "resid_post"          # TransformerLens hook name suffix
    dataset_name: str = "NeelNanda/pile-10k"
    dataset_split: str = "train"
    text_field: str = "text"
    seq_len: int = 128
    activations_per_shard: int = 8192   # how many activation vectors to buffer

    # SAE
    arch: Literal["relu", "jumprelu"] = "relu"
    expansion: int = 8
    bandwidth: float = 1e-3
    log_theta_init: float = -3.0

    # Optimizer
    lr: float = 1e-3
    batch_size: int = 4096
    n_steps: int = 10_000
    warmup_lr_steps: int = 200
    grad_clip: float = 1.0

    # Sparsity
    sparsity_coef: float = 1e-3       # L1 (ReLU) or L0 (JumpReLU) coefficient

    # Focal reweighting
    focal_scheme: Literal["none", "relative", "fit"] = "none"
    focal_gamma: float = 0.0
    focal_ema_decay: float = 0.99
    focal_warmup_steps: int = 1000
    focal_clip_lo: float = 0.1
    focal_clip_hi: float = 10.0

    # Dead-latent detection
    dead_token_window: int = 200_000   # number of tokens for dead-latent stat

    # Activation source. If `cache_dir` is set, training reads from a
    # pre-built sharded cache (fast — no LM forward in the loop).
    cache_dir: Optional[str] = None

    # Bookkeeping
    seed: int = 0
    device: str = "auto"               # "auto" → cuda > mps > cpu (resolved at runtime)
    log_every: int = 100
    eval_every: int = 1000
    out_dir: str = "runs"
    run_name: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
