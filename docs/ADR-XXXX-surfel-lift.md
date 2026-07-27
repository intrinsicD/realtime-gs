# ADR-XXXX — Surfel Lift: Closed-Form Construction of Covariance, Opacity, and Color from 2D Gaussian Captures

**Status:** proposed
**Date:** 2026-07-27
**Implements:** PREREG `20260725_init_value_program` experiments E7 (covariance) and E8 (opacity/color)
**Supersedes:** the CI-covariance lift path and the 2026-07-23 track-LSQ rejection (re-adjudicated under §2.4)
**Renumber:** replace XXXX with the next free ADR number on merge.

---

## Context

The tomographic beam fusion produces accurate means (to be confirmed by E1c) whose rendered state
is nonetheless empty: initial fitted-view alpha IoU 0.0107. The information destroyed between
Stage 1 and Stage 2 lives in the non-positional parameters. This ADR specifies their construction.

**Governing constraints:**

1. **No optimization.** Every step is closed-form or a small linear solve. Fitting init parameters
   by SGD against renders would merely relocate training, forfeiting the warm start this exists to
   create.
2. **Image-free.** Inputs are exclusively: lifted means, per-view 2D Gaussian parameters
   (mean, covariance, color, alpha — including the exact bit-packed alphas), camera intrinsics and
   extrinsics, and beam-fusion lineage (which 2D observations contributed to which primitive).
   Source RGB is never read.
3. **SPD by construction.** All covariances are stored and manipulated as `(q, log s)` with
   `Σ = R(q) · diag(exp(2s)) · R(q)ᵀ`. Raw-matrix fitting is prohibited anywhere in the lift; the
   635/800 non-SPD failure mode is thereby structurally impossible.
4. **Fixed order: rotation → scale → opacity → color.** Each step consumes the previous one.
   Solving opacity around a wrong covariance converges to the opacity that best *compensates* the
   covariance error, baking the defect in and hiding it from every later diagnostic (PREREG E8
   ordering rule).

**Notation.** Primitive `i` has contributing observations `O_i = {(v, g)}`: view `v` with 2D
Gaussian `g` = (μ₂D, Σ₂D, rgb, α₂D). `f_v` focal length in pixels, `z_v` primitive depth in view
`v`'s camera frame, `r_v` unit ray from camera `v` to the mean, `θ_v` incidence angle between `r_v`
and the surface normal.

---

## Decision

### §1 Rotation — normal from local PCA over lifted means

For each primitive: k-NN over the lifted means (k = 12 default, k ∈ [8, 16] acceptable), covariance
`C` of the neighborhood, eigenvalues λ₁ ≥ λ₂ ≥ λ₃ with eigenvectors e₁, e₂, e₃.

- **Normal** `n = e₃`, sign-disambiguated toward the mean contributing-camera direction:
  `n ← −n` if `n · mean_v(−r_v) < 0`.
- **Tangent frame** `t₁ = e₁`, `t₂ = e₂` (re-orthonormalized against n).
- **Planarity / coherence**
  `c = (λ₂ − λ₃) / (λ₁ + λ₂ + λ₃)  ∈ [0, ⅓]`,
  normalized to `ĉ = 3c ∈ [0, 1]`. High ĉ → locally planar, trust the normal. Low ĉ → edge,
  corner, or noise → fall back isotropic (§2.3).

**What is explicitly rejected:** estimating the normal from the mean contributing-ray direction.
That estimates the *viewing* direction, not the surface normal; the error is maximal at grazing
incidence, exactly where covariance matters most.

**Secondary normal (cross-check, not fusion).** The major axis of each contributing 2D Gaussian is
the image-space projection of a surface tangent direction. Back-project the axis direction of each
observation into world space; with K ≥ 2 views the normal is the null space (smallest right
singular vector) of the stacked tangent directions. Compute `Δ = angle(n_PCA, n_axis)` per
primitive and **log it**. `Δ > 30°` flags the primitive (`flag_normal_disagree`); do not average
the two normals — disagreement is a data-quality signal, and averaging two differently-wrong
normals produces a confidently wrong one.

### §2 Scale — tangential back-projected from the 2D footprints; along-ray as prior

#### §2.1 Per-view tangential candidate

The Stage-1 fit *measured* the world-space tangential footprint, in pixels. Recover it. Under the
local perspective Jacobian `J ≈ (f_v / z_v) · I` (valid for footprints small relative to depth):

```
Σ_tan^(v) = (z_v / f_v)² · Σ₂D^(v)                    [pixels² → world units²]
```

Foreshortening correction: the footprint along the projected-normal image direction is compressed
by `cos θ_v`. Decompose `Σ_tan^(v)` in the image-plane basis aligned with the projected normal and
divide that axis's standard deviation by `cos θ_v`. **Reject the observation entirely if
θ_v > 70°** — the correction diverges and a single grazing view otherwise dominates the fusion.

Express each surviving candidate in the shared tangent frame `(t₁, t₂)` from §1 (rotate, take the
2×2 tangential block).

#### §2.2 Fusion across views

Fuse per-axis in **log-eigenvalue space** using the median over views (geometric mean acceptable
at K ≤ 3). Arithmetic averaging of variances is prohibited: it is dominated by the largest
candidate, i.e. precisely the least-reliable near-grazing observation.

Consistency check: if `max_v σ_t^(v) / min_v σ_t^(v) > 3` on either axis, set
`flag_scale_disagree` — this indicates a wrong beam-fusion correspondence and must be known before
training, not discovered by it. Fused values are still produced (median is robust to one outlier).

#### §2.3 Anisotropy modulation and along-ray prior

```
σ_t1, σ_t2  = fused tangential sigmas, then:
  aspect capped at 3:1 initially   (rotation gradients are degenerate at λ₁≈λ₂ and stiff at
                                    extreme aspect; both freeze wrong orientations — the cap is
                                    relaxed per level by the coarse-to-fine schedule, not here)
  blended toward isotropic by ĉ:   σ_tk ← ĉ·σ_tk + (1−ĉ)·√(σ_t1 σ_t2)

σ_n = ρ · √(σ_t1 σ_t2),   ρ ∈ {0.05, 0.1, 0.25}   (E7 sweep; ρ is a PRIOR, not an estimate —
                                                    two views cannot resolve this axis in
                                                    principle; that is the CI lesson)
```

Assemble `R = [t₁ t₂ n]` → quaternion `q`; `s = (log σ_t1, log σ_t2, log σ_n)`. Store `(q, s)`.

#### §2.4 Track-LSQ re-adjudication

The 2026-07-23 track-LSQ covariance (alpha IoU 0.011 → 0.551, PSNR-AUC +9.11%) is re-run fitting
in `(q, s)` under the same residual. It enters E7 as an arm alongside the §1–§2.3 construction.
Selection between them is by the E7 downstream gate only (ΔLE **and** T@τ), per PREREG §0.

### §3 Opacity — log-transmittance linearization → sparse NNLS

**Objective:** the lifted state rendered into each fitted view must reproduce that view's Stage-1
alpha composite `A₂D^(v)(x)`. The compositing equation is nonlinear in α; the transmittance
linearizes it:

```
log T(x) = Σ_i log(1 − α_i G_i(x)) ≈ −Σ_i α_i G_i(x)        (accurate for α_i G_i ≲ 0.7)
target:   Σ_i α_i G_i^(v)(x) = −log(1 − Â₂D^(v)(x)),   Â = min(A₂D, 0.995)   (clip: log diverges)
```

Linear in α with `G_i` known from §1–§2. Assembly and solve:

- Stack equations over all fitted views and a pixel subsample (every 4th pixel in x and y;
  the system is massively overdetermined and the subsample is an implementation default, not a
  tunable).
- Solve **NNLS with box constraint** `α_i ∈ (0, 0.99]`, per tile: the system is sparse and
  tile-local, so it decomposes into independent small solves (projected gradient or
  Lawson–Hanson per tile; embarrassingly parallel).
- Compositing-order note: the linearized form is order-free, which is exactly why it is usable
  here; the order-dependent error is second-order in α G and is what the ≲0.7 validity bound
  controls.

**Interpretation of solutions:**
- Single-primitive-dominant pixels collapse to `α₃D ≈ α₂D`; the NNLS is the *correction for
  overlap*, not the typical path.
- `α_i → 0` from the solve means the primitive's footprint is fully explained by its neighbors →
  mark `merge_candidate`, consolidate **before** training rather than training dead weight.
- Only Stage-1 alphas are read. The image-free property is preserved exactly.

### §4 Color / SH

- **SH-0 (DC):** confidence-weighted mean of contributing 2D colors, weight
  `w_v = quality_v · cos θ_v` where `quality_v` is the Stage-1 per-Gaussian fit quality if
  captured, else 1.
- **SH ≥ 1: initialized to zero.** With K = 2–5 directional samples, a band-1 fit is
  underdetermined and fits noise; cross-view color variation confounds specularity with misfit and
  the two are not separable at this K. Zeroing is consistent with the coarse-to-fine SH schedule
  (bands unlock per level during training). If a variant with band-1 init is ever tested, it must
  use Tikhonov shrinkage toward DC and enter as a separate E7 arm — never as a silent default.

### §5 Degenerate cases (all mandatory, all logged)

| Case | Detection | Handling |
| --- | --- | --- |
| Single-view primitive | `K = 1` | no depth confidence: ρ ← 2× default; α ← α₂D directly (no solve); `flag_single_view` |
| Line-like 2D fits | median 2D aspect over O_i > 8:1 | represents a curve, not a surface; keep isotropic-tangential (ĉ-blend forces this if PCA agrees); count reported per scene |
| Normal disagreement | `Δ > 30°` (§1) | keep PCA normal; `flag_normal_disagree` |
| Scale disagreement | ratio > 3 (§2.2) | median fusion; `flag_scale_disagree`; report flagged fraction — it estimates the beam-fusion false-correspondence rate |
| Isolated primitive | k-NN radius > τ_iso · median spacing | PCA meaningless; isotropic, ρ = 1 (ball), `flag_isolated` |
| Saturated alpha | A₂D ≥ 0.995 | clipped (§3); saturated-pixel fraction logged per view |

Flag fractions are part of the E1 audit output (`rtgs audit-init`), so they are visible **before**
any training run consumes the init.

---

## Acceptance criteria (gates for merging the implementation, prior to E7/E8 runs)

**A1 — Synthetic ground truth.** On the synthetic scene, with ground-truth surfels available:
median angular normal error < 15° on primitives with ĉ > 0.5; median tangential-scale log-ratio
error < 0.35 (≈ factor 1.4); solved α within 0.1 of GT composite alpha on ≥ 80% of sampled pixels.

**A2 — Round-trip property test.** Rendering the lifted init into each fitted view must approach
that view's Stage-1 render as ρ → small and footprints match: fitted-view ΔLE ≤ 3 dB and init
alpha IoU ≥ 0.8 (the E8 gate, evaluated pre-training). If A2 fails, run PREREG E1b block
substitution to localize which of §1–§4 is at fault before touching anything.

**A3 — Determinism.** Identical inputs → bit-identical `(q, s, α, SH)` output (feeds PREREG M0).

**A4 — No hidden optimization.** Code review confirms: no autograd graph, no render-loss loop
anywhere in the lift path.

## Consequences

- The lift becomes ~5 small deterministic kernels (PCA, back-projection, log-eig fusion, tiled
  NNLS, weighted color mean) — testable in isolation, no training in the loop.
- E7's sweep surface is exactly {ρ} × {§2 construction, §2.4 track-LSQ-SPD}; E8 is §3 on/off.
  Nothing else in this ADR is a tunable, which keeps the PREREG's factor spaces honest.
- The flags become the first quantitative estimate of beam-fusion correspondence quality,
  feeding E1c interpretation.
- Risk accepted: the perspective-Jacobian and foreshortening approximations degrade for footprints
  large relative to depth and near the 70° cutoff; A1/A2 bound the aggregate effect, and per-flag
  breakdowns localize it if the bound is missed.
