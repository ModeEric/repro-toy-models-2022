# Focal-Reweighted Reconstruction Loss for Sparse Autoencoders

## Hypothesis

Standard SAE training uses uniform MSE on the reconstruction term, which allocates gradient mass proportional to current reconstruction error magnitude. This implicitly favors capacity toward dimensions/features that already fit reasonably well (because they dominate the squared error sum) and underweights dimensions that are systematically poorly reconstructed.

A focal-loss-style reweighting — borrowed from object detection (Lin et al. 2017) — should redirect gradient toward poorly-reconstructed dimensions, with two predicted effects:

1. **Reduced dead-latent count.** Dead latents persist partly because the reconstruction objective doesn't strongly penalize their absence; if other features cover the relevant variance "well enough," gradient toward reviving the dead latent is weak. Focal reweighting amplifies gradient on whatever dimensions remain under-reconstructed, plausibly recruiting dead latents to cover them.
2. **Improved sparsity/fidelity Pareto frontier**, particularly in regimes where the standard SAE saturates (e.g., the high-`k` regime where OpenAI's scaling laws bend, around `k ≥ 256` for GPT-2 small).

The hypothesis is **falsifiable**: if focal reweighting improves Pareto curves and dead-latent counts, it's a useful drop-in modification. If it doesn't, the failure mode itself is informative — most plausibly that residual reconstruction error is concentrated in *intrinsically hard* dimensions (irreducible noise, dictionary-too-small) rather than in *underused-feature-shaped* dimensions, which would be a meaningful negative result about where SAE error actually lives.

## Background

- Toy Models of Superposition (Elhage et al. 2022) establishes the superposition phenomenon and motivates SAEs as a recovery method.
- Sharkey et al. 2022, Bricken et al. 2023, Cunningham et al. 2023 establish ReLU + L1 SAEs.
- Gao et al. 2024 (TopK), Rajamanoharan et al. 2024 (Gated, JumpReLU) push the Pareto frontier via thresholding/gating activations. AuxK loss (Gao et al.) directly targets dead latents by forcing dead latents to reconstruct residual error.
- Focal loss (Lin et al. 2017) reweights cross-entropy by `(1 − p)^γ` to focus on hard examples in detection.

The gap: focal-style reweighting on the *reconstruction* term of an SAE, targeting the dead-latent and feature-imbalance problems via gradient shaping rather than structural intervention, has not been published to my knowledge.

## Method

### Standard SAE baseline

For activation vector `x ∈ ℝ^d`, encoder `E`, decoder `D` with dictionary size `n`:

```
z = σ(E(x))         # encoder activations, sparse
x̂ = D(z)            # reconstruction
L_baseline = ‖x − x̂‖²₂ + λ · S(z)
```

where `σ` is the activation function (ReLU, JumpReLU, or TopK depending on architecture) and `S` is the sparsity penalty (L1 for ReLU, L0 for JumpReLU, none for TopK).

### Focal-reweighted SAE

Replace the reconstruction term with a per-dimension focal-weighted MSE:

```
L_recon_focal = Σ_i w_i · (x_i − x̂_i)²
```

where `w_i ∈ ℝ^d` is a per-dimension weight derived from a running estimate of fit quality on dimension `i`. Two candidate weighting schemes:

**Scheme A — relative-error focal:**
```
w_i = (e_i / ē)^γ
```
where `e_i = EMA of (x_i − x̂_i)²` over recent training batches, `ē` is the mean across dimensions, and `γ ≥ 0` controls focus strength. `γ = 0` recovers baseline.

**Scheme B — fit-quality focal:**
```
fit_i = 1 − (e_i / Var(x_i))   # explained variance per dimension
w_i = (1 − clamp(fit_i, 0, 1))^γ
```
This is closer to the original focal-loss formulation, normalizing by per-dimension variance so we're tracking explained-variance shortfall rather than raw error magnitude.

Scheme B is the primary, Scheme A is a robustness check.

EMA of per-dimension error tracked with decay `α = 0.99` (≈100-batch effective window). Weights detached from the gradient — they shape the loss but do not themselves receive gradient.

### Hyperparameters to sweep

- `γ ∈ {0, 0.5, 1.0, 2.0}` (γ=0 is baseline)
- EMA decay `α ∈ {0.9, 0.99, 0.999}` (one secondary sweep at best γ)
- Tested across two architectures: vanilla ReLU+L1 and JumpReLU

## Experimental Setup

### Subject model

**Pythia-70M** (residual stream, layer 3 or 4 — pick one and hold constant). Justification: small enough for fast iteration, large enough to have nontrivial superposition, well-studied in the SAE literature so baselines are well-characterized.

Backup if Pythia-70M doesn't show clean baselines: GPT-2 small (where OpenAI's published scaling laws give a strong reference point at `k=256`).

### Activation dataset

10–50M activation vectors from the Pile (Gao et al. 2020) or OpenWebText. Streamed via TransformerLens hook on the chosen layer.

### SAE configurations

- Dictionary size `n = 8 · d` (8x expansion factor — standard).
- For ReLU+L1: sweep `λ` to hit a range of L0 values, plot Pareto.
- For JumpReLU: sweep target L0 directly.
- Train for 1–3 epochs over the activation dataset.
- Optimizer: Adam, lr=1e-3, decoder columns renormalized to unit norm each step (per Bricken et al.).

### Conditions

For each architecture, train:

1. **Baseline** (`γ = 0`)
2. **Focal-A** with best `γ` from `{0.5, 1, 2}` (Scheme A)
3. **Focal-B** with best `γ` from `{0.5, 1, 2}` (Scheme B)

Three seeds per (architecture × condition × γ) cell. Total runs: 2 × (1 + 3 + 3) × 3 = 42 SAE trainings, each ~1 hour on a single A100. Realistic compute budget: ~2 GPU-days.

## Metrics

### Primary

1. **Pareto frontier**: reconstruction MSE vs. L0 (averaged over batches). For each architecture, plot all conditions on the same axes; "wins" means dominating frontier, not just lower MSE at one point.
2. **Dead latent fraction**: fraction of dictionary elements active on < 1 in 10⁶ tokens after training. Reported per condition.
3. **Downstream loss recovered**: replace the residual stream activation with the SAE reconstruction during a forward pass on held-out text; measure increase in cross-entropy loss vs. clean run. Report as fraction of clean-vs-zero-ablation gap recovered.

### Secondary

4. **Per-dimension error distribution**: histogram of EMA `e_i` at end of training. Focal conditions should show flatter distributions than baseline.
5. **Feature interpretability**: spot-check top-activating examples for 20 randomly sampled features per condition. This is qualitative and a sanity check, not a primary signal — the literature is clear that automated interp scores are noisy at this scale.
6. **High-frequency feature count**: features active on > 10% of tokens. Gao et al. note these correlate with reduced interpretability.

## Predictions

In rough order of confidence:

- **Strong**: Focal-B with `γ = 1` reduces dead-latent count vs. baseline at matched L0. (Mechanistic: the reweighting directly increases gradient on dimensions current features fail to cover.)
- **Medium**: Focal-B Pareto-dominates baseline in the high-`k` / low-sparsity regime where the OpenAI scaling laws bend.
- **Medium**: Focal-A is noisier and less consistent than Focal-B because raw error magnitude conflates "hard dimension" with "high-variance dimension."
- **Weak**: Improvements are smaller for JumpReLU than for ReLU+L1, because JumpReLU already partially addresses the underlying issue via its threshold mechanism.
- **Possible negative result**: residual error is dominated by intrinsically-hard dimensions (irreducible noise, narrow dictionary) and focal reweighting just shifts MSE around without improving meaningful metrics. This would be informative — it would suggest dictionary capacity, not gradient allocation, is the binding constraint.

## Falsification criteria

The hypothesis is rejected if, across all `γ > 0` conditions on Scheme B:

- No Pareto improvement at any sparsity level (both architectures), AND
- No statistically significant reduction in dead-latent fraction (one-sided t-test across seeds, p > 0.1).

Mixed results (e.g., dead latents reduced but Pareto unchanged) are reported as such and discussed — they're not failure, they're a refinement of the hypothesis.

## Risks and Mitigations

- **EMA staleness**: per-dimension error estimates lag actual error during fast learning. Mitigation: warmup period (first 1k steps with `γ = 0`) before turning on focal reweighting; sweep over `α`.
- **Gradient explosion from extreme weights**: a single dimension being very poorly reconstructed could dominate the loss with high `γ`. Mitigation: clip `w_i` to `[0.1, 10]`; report whether clipping triggers.
- **Confound with learning rate**: focal reweighting effectively scales gradient magnitudes; could be confused with an LR change. Mitigation: tune LR per condition or report results across an LR sweep at the best `γ`.
- **Implementation bugs invisible from final metrics**: a broken EMA could silently fall back to uniform weights and produce baseline-like results. Mitigation: log per-dimension weights periodically and verify they vary.

## Deliverables

1. **Code**: clean PyTorch implementation extending an existing SAE training library (likely `ai-safety-foundation/sparse_autoencoder` or `EleutherAI/sae`). One file diff for the loss function, plus weight-tracking instrumentation.
2. **Pareto curves and dead-latent tables** across all conditions, with confidence intervals over seeds.
3. **Writeup**: 3–5 page report with method, results, and one of three conclusions: (a) focal reweighting is a useful drop-in improvement, with recommended `γ`; (b) focal reweighting helps in specific regimes (specify which); (c) it doesn't help, and here's what we learned about SAE reconstruction error structure.
4. **Open-source release** of code and trained SAE weights for at least the baseline and best focal condition.

## Timeline

Designed to fit in a 2-week sprint:

- **Days 1–2**: Implement focal loss + weight-tracking on top of an existing SAE codebase. Reproduce baseline numbers from Cunningham et al. or Bricken et al. on Pythia-70M as a sanity check.
- **Days 3–5**: Run γ sweep on ReLU+L1 architecture (single seed per condition for speed). Identify best γ per scheme.
- **Days 6–8**: Run JumpReLU sweep. Re-run best conditions with full 3 seeds.
- **Days 9–10**: Compute downstream-loss-recovered, dead-latent fractions, per-dimension error histograms.
- **Days 11–13**: Spot-check interpretability, write up results.
- **Day 14**: Buffer / cleanup / open-source release.

## Out of scope

- Larger models (Gemma 2 9B, etc.) — the result, if positive, transfers as a hypothesis but verifying transfer is a separate project.
- Comparison to AuxK loss specifically. AuxK and focal reweighting are structurally different (revival vs. gradient shaping) and could be combined — that's a follow-up.
- Theoretical analysis of why/whether focal reweighting should help. This experiment is empirical-first; if results are positive and clean, theory is a follow-up.
- Using focal reweighting on activations rather than reconstruction. That's the JumpReLU/TopK direction and is well-explored.

## Why this is worth doing

The experiment is small (≤ 2 GPU-days), the implementation is ≈ 50 lines of code on top of existing SAE libraries, and the result is informative either way. A positive result is a small but real Pareto improvement and a new tool for the dead-latent problem. A negative result is a meaningful claim about where SAE reconstruction error comes from, with implications for whether dictionary scaling or training dynamics is the right next lever.