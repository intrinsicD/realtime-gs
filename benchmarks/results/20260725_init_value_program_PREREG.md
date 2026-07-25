# PREREG — Initialization Value Program

**Date frozen:** 2026-07-25
**Status:** pre-registration. No results may be entered into this file. Results go to
`20260725_init_value_program_RESULT.md`, audit to `..._AUDIT.md`.
**Supersedes as the governing protocol for init claims:** `20260721_all_initializers_frame00008_PREREG.md`

---

## 0. Why this program exists

The 2026-07-21 suite produced a null result: seven initializers spanning 7 to 5,000 initial
primitives all converged to 36.96–38.25 dB fitted-view foreground PSNR after growing to
35.6k–49.2k Gaussians. Random finished fourth.

The standing interpretation was "no initializer is materially superior." That interpretation is
**not supported by the design**, because the design cannot distinguish between:

- (a) initialization genuinely does not matter,
- (b) initialization matters but the protocol erased it (unbounded growth, all-fitted-view
  evaluation, 70k iterations, downscale-16 images, single scene, single seed), and
- (c) initialization matters but *our* initialization does not reach the optimizer, because the
  lift destroys the render state even when the means are correct.

Evidence for (c) already exists in `20260723_beam_covariance_refit_RESULT.md`: initial fitted-view
alpha IoU is **0.0107**. A reconstruction whose initial render overlaps the target silhouette by
1% is, from the optimizer's point of view, indistinguishable from noise placed inside the correct
bounding volume.

This program separates (a), (b), and (c) in that order, and only then tests fixes.

**Governing principle:** an initializer is an *optimizer preconditioner*, not a geometry estimate.
Its only legitimate gates are downstream: convergence speed, quality under a constrained budget,
and held-out generalization. Photogrammetric validity is a means, never a terminal criterion. The
2026-07-23 rejection of track-LSQ covariance (alpha IoU 0.011 → 0.551, AUC +9.11%, rejected for
non-SPD raw matrices and whitened residual 13.45) applied a terminal criterion that this program
explicitly retires. Non-SPD is a parameterization defect to be fixed, not a reason to discard a
render-space gain.

---

## 1. Standing rules (apply to every experiment below)

### 1.1 Evaluation contract

| Item | Frozen value |
| --- | --- |
| Primary evaluation | **held-out views only** |
| Held-out split | deterministic, every 8th camera, fixed before any run |
| Fitted-view metrics | recorded, reported, **never** used for a claim |
| Foreground masking | as in the 2026-07-21 protocol, unchanged |
| Seeds | ≥ 3 per cell; ≥ 5 for any cell that will carry a headline claim |
| Scenes | ≥ 3 distinct scenes for any claim; single-scene = diagnostic only |
| Reporting | median across seeds + full min/max, never mean-only |

### 1.2 The noise floor gate (E0) blocks everything

No effect may be claimed until the seed-to-seed spread of the identical configuration is measured.
**Any observed difference smaller than 2× the seed spread of the same cell is reported as "within
noise" and carries no claim.** The 2026-07-21 suite's 1.3 dB range has never been compared against
a noise floor and may be entirely seed variance.

### 1.3 Results that will not be accepted as evidence

Declared now so they cannot be rationalized later:

1. Fitted-view PSNR improvements of any size.
2. alpha IoU or init-render improvements without a downstream convergence gain.
3. Single-scene wins.
4. Single-seed wins.
5. Final-plateau PSNR at unbounded budget — this program predicts all arms tie there, and a tie
   is the expected outcome, not a finding.
6. Wall-clock timings on non-identical hardware/driver state.
7. Any improvement obtained by also changing the density schedule in the same run.

### 1.4 Frozen definitions

**Stage-1 reference PSNR** `P_2D(v)` — PSNR of the Stage-1 2D Gaussian fit of view `v` against the
original RGB of `v`, computed once per capture and checked in. Required because it is the ceiling
of the entire pipeline.

**Init render PSNR** `P_init(v)` — PSNR of the lifted 3D state rendered into view `v` at iteration
0, same rasterizer, same SH degree, same background as training.

**Lift efficiency** `ΔLE(v) = P_2D(v) − P_init(v)`. Reported as median over fitted views.
This is the primary Stage-2 metric for the rest of the project.

**Init alpha IoU** — IoU of (rendered alpha > 0.5) against the foreground mask, held-out and
fitted reported separately.

**Overparameterization ratio** `Ω = (N_gaussians × params_per_gaussian) / (N_train_views × pixels_per_view × 3)`.
Recorded for every run. Hypothesis H-B2 below is stated in terms of Ω.

**Iterations-to-target** `T@τ` — first iteration at which held-out foreground PSNR reaches τ and
stays within 0.1 dB of it for 200 further iterations (prevents crediting a transient spike).
`τ` is set per scene from a reference run (see E4) and frozen before the sweep.

**Normalized displacement** `d_i = ‖μ_i^final − μ_i^init‖ / σ_i^tangential,init`, where
`σ^tangential` is the geometric mean of the two largest init scale axes.

**Alpha-mass attribution** `A` — fraction of accumulated rendered alpha over held-out views
contributed by primitives whose lineage root is an init primitive.

---

## 2. Part 0 — Instrumentation (must land before any experiment)

None of the experiments are interpretable without these. Build and unit-test them first.

### M0 — Seed and determinism harness
Every run records: seed, git SHA, gsplat version, CUDA/driver, full resolved config, and a hash of
the initial state. Two runs with identical inputs must produce bit-identical initial states.

### M1 — Stage-1 reference cache
`P_2D(v)` for every view of every capture, checked into the dataset directory. One-time cost.
Without this the lift cannot be scored, because we do not know what the lift was given.

### M2 — Init render audit
A command (`rtgs audit-init`) that, from a `gaussians_init.ply` plus scene, emits:
`P_init` per view, `ΔLE`, alpha IoU, and histograms of opacity, of the three scale axes, and of the
ratio `σ_along-ray / σ_tangential` (using the source camera ray of each primitive's lineage).
Must run without training.

### M3 — Lineage tracking
The load-bearing piece. Every primitive carries an immutable `root_id` and a `provenance` record.
Required to survive:

- **clone** — child inherits `root_id`, provenance appends `(clone, iter)`
- **split** — both children inherit `root_id`, provenance records the sampled offset **and its
  decomposition into along-ray and tangential components** relative to the root's source ray
- **prune** — death iteration and reason recorded, not just deletion
- **MCMC relocation** — relocated primitive's `root_id` is set to `NULL_ROOT` (relocation is
  destruction plus creation, not inheritance). This must be explicit; treating relocation as
  survival would fabricate the result E2 is meant to test.

Emit per-run: survival curve, displacement distribution, alpha-mass attribution `A`, and the
clone/split census.

### M4 — Budget-honest density control
A mode where `max_3d_gaussians` is a hard cap **and** the growth criterion is not silently
schedule-saturating. Verify: with cap = init count and growth enabled, the run must terminate with
exactly the init count and no primitive churn beyond pruning.

**Gate:** M0–M4 pass their unit tests, and `rtgs audit-init` reproduces the known 0.0107 alpha IoU
on the Beam CI initialization. If it does not reproduce, stop — the instrumentation is wrong.

---

## 3. Part 1 — Diagnostics: is there an initialization at all?

Cheap. Run these before spending GPU-days on sweeps. They answer the question "are my assumptions
about growth, cloning, and inheritance correct."

### E0 — Noise floor

**Question:** how large is seed variance in the quantities we intend to compare?

**Setup:** one initializer (`beam-fusion`), one scene, 5 seeds, otherwise the exact 2026-07-21
configuration. Then repeat with the budget-constrained configuration of E4 (cap = 1× init).

**Records:** spread of held-out FG PSNR, of `T@τ`, of final primitive count, of the training
objective.

**Decision rule:** the resulting spreads become the significance thresholds for every later
experiment. Write them into the RESULT file as a frozen table before E4 begins.

**Watch for:** spread being much larger in the constrained regime than the unconstrained one. If
so, constrained comparisons need more seeds, not fewer.

---

### E1 — Init render audit: does the lift preserve Stage-1 information?

**Question:** does our initialization exist as a render state, or only as a point set?

**Hypothesis H-C1:** the lift is information-destroying. Predicted `ΔLE > 10 dB` and init alpha
IoU `< 0.05` for the CI covariance path.

**Setup:** `rtgs audit-init` over all seven checked-in initializers × ≥3 captures. No training.

**Primary metric:** median `ΔLE`, init alpha IoU.

**Pre-declared interpretation:**

| `ΔLE` | Reading |
| --- | --- |
| < 3 dB | lift is information-preserving; hypothesis (c) refuted; the problem is elsewhere |
| 3–10 dB | partial loss; localize it via the ablation in E1b |
| > 10 dB | **the optimizer receives no warm start.** All prior init comparisons were comparing support regions, not initializations. |

**E1b — loss localization.** Starting from the true 3D state where available (synthetic scene with
ground-truth geometry), substitute one component at a time from the lift: means only, means+scales,
means+scales+rotation, +opacity, +SH. Report `ΔLE` after each substitution. This produces an
attribution of the lift loss across the parameter blocks and tells us exactly which one to fix.
Prediction: the drop is dominated by the scale/rotation block, secondarily opacity.

**Confound to avoid:** measuring `ΔLE` with a different background, SH degree, or antialiasing
setting than training uses. The audit must read `gaussians.config.json`.

**What would falsify H-C1:** `ΔLE < 3 dB` with alpha IoU > 0.6. Then the lift is fine and the null
result of 2026-07-21 is about the protocol (Part 2), not the pipeline.

---

### E1c — Geometric accuracy of the init means (reference-anchored)

**Question:** are the tomographically lifted positions actually accurate, measured against
reference geometry rather than assessed by eye in the orbit viewer?

**Why this must exist and must run before E7/E8.** The entire repair path assumes the means are
good and only the covariance/opacity packaging is broken. That assumption is currently supported by
visual inspection of an 800-primitive orbit view without a reference. Dense point sets look
plausible whenever the coarse shape is right, and the viewer offers no calibrated snapshot for the
compact-only bundles. If the means are in fact mediocre, E7 and E8 will produce a
well-packaged wrong geometry and the program will spend its budget in the wrong place. This is a
one-off, training-free measurement and it is the cheapest way to protect the rest of the plan.

**Hypothesis H-C4:** beam fusion's means are materially more accurate than the non-tomographic
compact-only initializers at comparable primitive counts.

**Setup:** training-free, from saved `gaussians_init.ply`. Reference geometry per scene:

| Scene type | Reference |
| --- | --- |
| Synthetic | ground-truth mesh / point set |
| Janelle captures | COLMAP dense / MVS reconstruction on the original RGB, built once and checked in as a dataset sidecar |
| COLMAP scenes | dense MVS from the same reconstruction |

Building the MVS reference uses source RGB. That is legitimate: it is **evaluation infrastructure,
not pipeline input**, and it is constructed offline once per capture. State this explicitly in the
RESULT file so the image-free property of the method is not confused with an image-free property of
the evaluation.

**Primary metrics:**
- bidirectional Chamfer distance, init means vs. reference, scale-normalized by the scene extent
- accuracy/completeness at fixed thresholds (fraction of init means within ε of the reference, and
  fraction of reference within ε of an init mean), swept over ε — the two directions must be
  reported separately, since a compact init is expected to have good accuracy and poor completeness
  and averaging the two hides exactly that
- per-camera depth error of each lifted primitive against the reference depth map, reported as
  median and 90th percentile
- **along-ray error component vs. tangential error component**, decomposed relative to the source
  camera ray, which is the axis the tomographic fusion is supposed to be resolving

**Arms:** beam-fusion, dense-merge, splat-sfm, top-K, field, random. Random is the control and
establishes the error a method must beat to have found anything at all.

**Pre-declared interpretation:**

| Observation | Reading |
| --- | --- |
| beam-fusion Chamfer materially below all non-tomographic arms, and along-ray error ≈ tangential error | **the tomographic fusion works.** The means are an asset, the repair path (E7/E8) is correctly targeted, and this is a standalone contribution independent of any convergence result. |
| beam-fusion Chamfer comparable to random | the means carry no information; E7/E8 are pointless and the program reduces to Part 2 + E11 |
| accuracy good, completeness poor | expected and acceptable for a compact init; report as a coverage statement, not a defect, and check it against the support argument in E2 |
| along-ray error ≫ tangential error | fusion is not resolving depth; the lift is effectively a reprojection of 2D fits and the beam-fusion framing must be corrected in the paper |

**Threshold for the standalone claim:** beam-fusion's median depth error at least 2× lower than the
best non-tomographic compact arm, on ≥ 3 scenes, exceeding the E0 spread of the init construction
itself (build the init 3× with different seeds where the method is stochastic).

**Watch for:**
- Chamfer being dominated by outliers/floaters. Report trimmed variants (95th percentile clipped)
  alongside the raw number, and report the outlier fraction separately rather than clipping it away
  silently.
- Reference bias. COLMAP dense fails on the same textureless regions that make lifting hard, so
  agreement may be measuring shared failure. Where possible, restrict the comparison to a
  reference-confidence mask and report masked and unmasked numbers.
- Scale/frame mismatch between the init and the reference. Verify the alignment is exact (same
  calibration, no Umeyama fit); if any alignment is required, that is itself a finding and must be
  logged, not absorbed.

**Deliverable:** one table (arms × scenes × metrics) plus the accuracy/completeness curves over ε.
This table is the evidence for "beam fusion finds the geometry," a statement currently made on
visual grounds only.

---

### E2 — Survival, displacement, attribution: is the init refined or rejected?

**Question:** do init primitives persist and carry the reconstruction, or are they discarded and
replaced?

**Hypothesis H-C2:** the init is rejected. Predicted survival at iteration 2,000 below 30%, median
normalized displacement above 3, alpha-mass attribution `A < 0.25` at convergence.

**Setup:** M3 lineage tracking on, 3 initializers (`beam-fusion`, `dense-merge`, `random`), 3
scenes, 3 seeds, standard unbounded schedule (deliberately the same regime as 2026-07-21 so this is
a direct reinterpretation of that null result).

**Primary metrics:**
- survival curve `S(t)`, reported at t = 500, 2,000, 7,000, final
- median and IQR of `d_i` over survivors
- alpha-mass attribution `A(t)`

**Pre-declared interpretation:**

| Pattern | Reading |
| --- | --- |
| `S(2000) < 0.3` | init is being **rejected**. Fix the lift (E7/E8), not the optimizer. |
| `S` high, `d` median > 3 | init survives but position information is discarded; means are not the bottleneck. |
| `S` high, `d` median < 1, `A` low | init is correct and stationary but contributes little alpha — an opacity/scale problem, i.e. E8. |
| `S` high, `d` low, `A` high | **the init does carry the reconstruction** and the 2026-07-21 null result is purely a protocol artifact. Skip to Part 2. |

**Critical implementation note:** the `random` arm is the control. If random primitives show the
same survival and attribution as beam-fusion primitives, then "survival" is measuring the
optimizer's willingness to keep whatever it is given, not init quality. Report the contrast, not
the absolute numbers.

---

### E3 — Clone/split census: is the inheritance mechanism even running?

**Question:** the design assumption is that init primitives act as clone sources so that new
primitives inherit good 3D positions. Does the clone path actually fire?

**Hypothesis H-C3:** it does not. Because CI covariance produces large along-ray extents, init
primitives exceed the scale threshold and take the **split** path, whose children are sampled from
the parent's covariance and are therefore scattered along exactly the unresolved ray axis. Under
this hypothesis, densification *destroys* the tomographic position information rather than
propagating it.

**Setup:** same runs as E2, additional instrumentation.

**Primary metrics:**
- fraction of densification events that are clone vs. split, over time, split by lineage
- distribution of child offset magnitude, **decomposed into along-ray and tangential components
  relative to the root primitive's source camera ray**
- generation depth distribution at convergence

**Pre-declared interpretation:**

| Observation | Reading |
| --- | --- |
| split fraction > 0.7 in the first 2,000 iterations | the assumed inheritance mechanism is not exercised; H-C3 supported |
| child offsets anisotropic, along-ray ≫ tangential | densification is actively scattering along the unresolved axis; motivates E9 |
| mean generation depth ≥ 3 | ancestor position cannot be expected to survive regardless; inheritance is a one-generation effect at best |
| clone fraction > 0.7 and offsets isotropic | H-C3 refuted; inheritance works, look elsewhere |

**Watch for:** the AbsGS gradient criterion interacting with primitive size. Large primitives
accumulate large absolute gradients almost by construction, so a high split rate is partly
mechanical. Report split rate conditioned on init scale decile to separate "large primitives split"
from "our primitives are large."

---

**Part 1 exit condition.** Write the four diagnostic answers (E1, E1c, E2, E3) into the RESULT file
as a single paragraph each, with the pre-declared reading applied verbatim. Part 1 either localizes
the failure to the lift (→ Part 3) or exonerates the lift (→ Part 2 alone). Do not start Part 2
before these paragraphs exist.

The decisive combination is **E1c good means × E1 large `ΔLE`**: accurate geometry that does not
reach the renderer. That is the case in which the repair path has the most to give, and it is the
outcome the current evidence points to. The combination **E1c poor means × E1 large `ΔLE`** ends
the repair path and reduces the program to Part 2 plus E11. Record which of the four quadrants the
data lands in, explicitly, before proceeding.

---

## 4. Part 2 — Regime: under what conditions can initialization matter?

This part is worth running **even if the current lift is broken**, because it establishes the
window in which any initializer could pay off. It is also, independently, the more publishable of
the two parts: a characterization of where init value lives is a positive contribution, whereas
"our init wins on one scene" is not.

### E4 — Primitive budget sweep

**Question:** does init quality matter under a constrained primitive budget?

**Hypothesis H-B1:** init value is monotonically decreasing in budget. Separation between
initializers is large at cap = 1× init count and vanishes by 8×.

**Setup:**

| Factor | Levels |
| --- | --- |
| Initializer | beam-fusion, dense-merge, splat-sfm, random, (top-K) |
| Cap `max_3d_gaussians` | 1×, 2×, 4×, 8× the arm's native init count, and unbounded |
| Scenes | ≥ 3 |
| Seeds | ≥ 3 (5 for the 1× and 2× cells) |

Native init counts differ across arms (5,000 / 2,088 / 943 / …). Run **both** normalizations and
report both: cap relative to each arm's own init count, and cap at absolute matched counts
(2,000 / 5,000 / 20,000 / 40,000). The relative version tests "does a good init let you stay
small"; the absolute version tests "at equal model size, which init wins." They answer different
questions and conflating them is the most likely way to generate a spurious result here.

**Reference run for `τ`:** random init, unbounded, 70k iterations, held-out FG PSNR plateau.
`τ = plateau − 0.5 dB`, frozen per scene before the sweep.

**Primary metrics:** `T@τ` (iterations-to-target), and held-out FG PSNR at fixed iteration budgets
{2k, 7k, 30k, 70k}.

**Pre-declared decision rule for the project's core claim:**

> The "skip the cold start" claim is **supported** iff, at cap ≤ 2× init count, the best
> structured initializer achieves `T@τ` at most half that of random, with the effect exceeding
> 2× the E0 noise floor, replicated on ≥ 3 scenes and ≥ 3 seeds.
>
> It is **refuted** if the ratio is above 0.8 under the same conditions.
>
> Between 0.5 and 0.8: the effect is real but modest; the project must be reframed as a
> memory-efficiency contribution (E11) rather than a speed contribution.

**Watch for:**
- The cap being reached early and the run then being a fixed-topology optimization for 60k
  iterations. Report the iteration at which each cap saturates; if all arms saturate by 5k, the
  comparison is about fixed-topology refinement, which is a different (still interesting) claim.
- Pruning interacting with a hard cap to produce oscillation. Log primitive count trajectories, not
  just endpoints.
- The unbounded cell reproducing the 2026-07-21 tie. **It should.** If it does not, something in
  the harness changed and the whole sweep is suspect.

---

### E5 — View sparsity sweep

**Question:** does init value grow as photometric constraint is removed?

**Hypothesis H-B2:** separation between initializers scales with the overparameterization ratio Ω;
at 26 views on one object at downscale 16 the problem is in the memorization regime and no init can
show an effect.

**Setup:** N_train ∈ {3, 6, 12, 26}, cameras chosen by a frozen maximally-distributed rule (not
random, not first-N — camera baseline distribution is a confound and must be controlled). Held-out
set is **identical across all N**, drawn from cameras excluded at every level. Budget: the two best
cells from E4 plus unbounded.

**Primary metric:** held-out FG PSNR gap between the best structured init and random, as a function
of N. Secondary: LPIPS and a depth/geometry error where ground truth exists — PSNR is a poor
detector of floaters, which is the failure mode sparse views produce.

**Pre-declared prediction:** gap ≥ 1.5 dB at N = 3, ≥ 0.5 dB at N = 6, within noise at N = 26.

**What would falsify H-B2:** no gap even at N = 3. That is a strong result and would mean the
lift's information content is genuinely zero (consistent with a large `ΔLE` in E1), or that
3DGS's inductive bias dominates any init.

**Watch for:** at N = 3, all methods may fail catastrophically and tie at the bottom. Include a
"can any method reconstruct this at all" sanity arm (SfM init with dense COLMAP points, or GT
geometry init on the synthetic scene) to confirm the regime is solvable before concluding a tie.

---

### E6 — Resolution / Ω control

**Question:** is the null result an artifact of downscale 16?

**Setup:** downscale ∈ {16, 4, 2} on ≥ 2 scenes, best two budget cells, ≥ 3 seeds. Report Ω for
every cell.

**Primary metric:** init-vs-random gap plotted against Ω, pooling E4/E5/E6 cells.

**Deliverable:** a single figure — gap vs. Ω across all three sweeps. If the points collapse onto
one curve, that curve is the paper's central claim: *initialization value is a function of the
constraint-to-capacity ratio, and prior 3DGS init comparisons were conducted at Ω where it cannot
be observed.* That is a contribution independent of whether our specific lift wins.

---

## 5. Part 3 — Fixes (gated on Part 1)

Run only the fixes that Part 1's attribution justifies. Each has an independent gate; do not bundle.

### E7 — Surfel initialization (SPD-parameterized, normal-oriented)

**Precondition:** E1c must have shown that the means are an asset. Fixing the covariance around
inaccurate means produces a well-packaged wrong geometry, which is harder to detect than an
obviously broken one.

**Motivation:** two-view covariance intersection cannot resolve along-ray variance in principle.
Replace estimation with a prior.

**Method:**
- Normal from multi-view fusion / structure tensor (StructSplat path), not from CI.
- Along-ray extent set to a fixed fraction `ρ` of the tangential geometric mean; `ρ ∈ {0.05, 0.1, 0.25}` swept.
- Anisotropy modulated by structure-tensor coherence `(λ₁−λ₂)/(λ₁+λ₂)`; low coherence → isotropic.
- **Parameterization: `Σ = R(q)·diag(exp(s))·R(q)ᵀ`, fitted in `(q, s)`.** SPD by construction.
  This retires the 635/800 non-SPD failure as a parameterization bug rather than a finding.

**Gate:** `ΔLE` improvement **and** `T@τ` improvement at cap ≤ 2×, held-out, ≥3 scenes, exceeding
noise floor. An alpha IoU gain alone does not pass (rule 1.3.2).

**Also re-run:** the 2026-07-23 track-LSQ configuration under the SPD parameterization. Its
render-space gain (IoU 0.011 → 0.551, AUC +9.11%) was discarded on a criterion this program has
retired, and it must be re-adjudicated under the downstream gate. Report explicitly whether the
SPD-constrained fit retains the gain; if it does, the 2026-07-23 conclusion is formally superseded
and the AUDIT must say so.

---

### E8 — Appearance-preserving lift

**Motivation:** Stage 1 fits each image to a known PSNR. A correct lift should reproduce that
composite from the fitted views at iteration 0. `ΔLE` measures the shortfall directly.

**Method:** solve per-primitive opacity (and, if needed, an SH-0 correction) so that the α-composite
of the lifted 3D state matches the Stage-1 2D composite in the fitted views. The checked-in exact
bit-packed alpha is a constraint, not just payload. Closed-form per-pixel where the ordering is
fixed; otherwise a short non-photometric least-squares against the Stage-1 render — **no source RGB
is required**, preserving the image-free property.

**Gate:** `ΔLE ≤ 3 dB` and init alpha IoU ≥ 0.8, **plus** a `T@τ` gain at cap ≤ 2×.

**Watch for:** an appearance-matched init that is geometrically worse (opacity compensating for bad
depth). Report the depth error of the init before and after; if E8 improves appearance while
degrading geometry, it is masking the E7 problem and the two must not be stacked.

---

### E9 — Split-inheritance repair

**Motivation:** conditional on E3 showing split-dominated growth with along-ray-scattered children.

**Variants (each independently ablatable):**
1. Reduced split sampling radius during the first `k` iterations.
2. Split offsets constrained to the two largest (tangential) axes for lineage-rooted primitives.
3. Appearance-preserving split: solve child opacities so the α-composite is invariant across the
   split, rather than copying parent opacity and dividing scale by 1.6.
4. MCMC relocation as an alternative — noting that MCMC's init-robustness cuts both ways: it
   repairs bad init *and* erases good init, so it is the wrong strategy for demonstrating init
   value, and must be reported as a separate column rather than folded into the default.

**Gate:** improvement in alpha-mass attribution `A` **and** `T@τ`, at fixed init.

---

### E10 — Ablation matrix

Once E7–E9 have individually passed or failed, run the factorial over the passing components on
≥3 scenes, ≥3 seeds, at the two budget cells identified in E4. Report the full matrix including the
cells that lose. Reviewers will ask which single component carries the effect; the matrix must
answer that without a follow-up run.

---

## 6. Part 4 — The systems claim

### E11 — Memory and throughput

**This is the defensible contribution and it does not depend on any of the above.** The pipeline is
image-free after Stage 1; that is a measurable systems property that no quality tie can undermine.

**Metrics:**
- peak host and device memory, measured with a fixed sampler, for the compact-only path vs. the
  RGB-backed path at identical primitive counts and identical view counts
- maximum number of simultaneously-held views at fixed VRAM (the headline number: *N views on a
  4090 vs. N/k for RGB-backed*)
- bytes-on-disk per view (168,000-byte cap per `.rtgsv` vs. source RGB)
- iterations/second at matched primitive count, same machine, same driver, interleaved A/B ordering
  to cancel thermal drift

**Rules:** same machine, same session, alternating order, ≥5 repetitions, report median and spread.
No cross-machine comparisons. State the measurement method inline; "nonportable" is not a
disclaimer that excuses an unstated method.

**Pre-declared claim form:** "at fixed VRAM budget V and fixed primitive count P, the compact path
holds k× more views than the RGB path, measured as follows." Nothing about quality.

---

## 7. Execution order and dependency graph

```
M0–M4  (instrumentation gate)
  └─ E0  noise floor  ──────────────► thresholds for everything below
       ├─ E1  init audit    ─┐
       ├─ E1c geometry acc.  ├─ Part 1 exit paragraphs + quadrant call
       ├─ E2  lineage        │
       └─ E3  census       ──┘
             │
             ├─ if means good AND lift broken ──► E7, E8 (then re-enter Part 2)
             ├─ if means poor ──────────────────► skip Part 3, go to Part 2 + E11
             └─ Part 2: E4 ──► E5 ──► E6 ──► gap-vs-Ω figure
                          │
                          └─ E9 (if E3 justified) ──► E10 ablation
E11 runs in parallel, independent of all of the above.
```

Estimated GPU cost is dominated by E4 (5 initializers × 5 caps × 3 scenes × 3 seeds = 225 runs).
Run E4 at reduced iteration count (30k) first as a screen, then re-run only the cells that matter
at full length. Declare the screen as a screen in the RESULT file.

---

## 8. What "we have shown X" means in this program

A claim is established only when all five hold:

1. It was **pre-declared** in this file with a direction and a threshold.
2. It is measured on **held-out** views.
3. The effect exceeds **2× the E0 noise floor** for that cell.
4. It replicates on **≥ 3 scenes** and **≥ 3 seeds**.
5. An **independent audit pass** confirms the numbers against the raw logs and confirms that the
   decision rule was applied as written, not adjusted after seeing results.

Anything meeting 1–4 but not 5 is "pending audit." Anything failing 1 is an **observation**, may be
reported as such, and requires a new pre-registration before it can become a claim.

Negative results are first-class here. "Initialization does not matter above Ω = X" is a result. So
is "our lift loses 14 dB of Stage-1 quality." The failure mode this program is built to prevent is
not a negative result — it is a positive-looking result produced by a protocol that could not have
produced a negative one.

One outcome is named here in advance so that it does not have to be argued for later: the program
may conclude that the memory property (E11) is the contribution and that the initialization confers
no convergence advantage at any budget or view count. That outcome is accepted as a complete result
of this program. Image-free reconstruction at k× the view count for a given VRAM budget stands on
its own and is not weakened by a quality tie. It is written down now, before the data exists, so
that a tie does not later get relitigated into a win.

---

## 9. Known limitations of this program, stated up front

- Single dataset family (Janelle object captures + synthetic + COLMAP). Object-centric captures may
  not generalize to unbounded scenes; the Ω framing predicts they should, and that prediction is
  untested here.
- `T@τ` depends on the reference plateau, which depends on the reference run's own seed. E0 must
  include the reference run's variance, and `τ` must be set from the median plateau across seeds.
- Foreground masking makes PSNR optimistic and is inherited from the prior protocol for
  comparability. Report unmasked numbers alongside.
- E1c's real-scene reference is an MVS reconstruction, not ground truth. It shares failure modes
  with the lift on textureless and occluded regions, so agreement there is weak evidence. The
  synthetic scene is the only cell with a genuinely independent reference and should be weighted
  accordingly when the two disagree.
- E8's appearance-preserving solve is only exact for the additive/peak-mixture regime already
  documented for the field lift; under normalized finite-support StructSplat semantics it is an
  approximation and must be validated by bounded sampled comparison, as in the existing field-lift
  validation.
