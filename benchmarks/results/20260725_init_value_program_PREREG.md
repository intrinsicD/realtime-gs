# PREREG — Initialization Value Program

**Date frozen:** 2026-07-25
**Amended:** 2026-07-25, before any experiment in this program was executed and while zero of its
data existed. The amendment adds the headroom measurement (§0b, E0b), makes E1b's substitution
ladder downstream, promotes COLMAP SfM points to a first-class arm, states the positive hypothesis
that was previously only implied, and ungates the training-free diagnostics from M3/M4. Nothing
here was adjusted in response to a result, because there were none.
**Status:** pre-registration. No results may be entered into this file. Results go to
`20260725_init_value_program_RESULT.md`, audit to `..._AUDIT.md`.
**Supersedes as the governing protocol for init claims:** `20260721_all_initializers_frame00008_PREREG.md`
**Ledger action on supersession:** every `ara/logic/claims.md` row whose `Proof` rests on the
superseded protocol must be re-read against this document and moved to `superseded` where its
licence came from the retired design. That sweep ships with the first RESULT commit of this
program, not as later cleanup.
**Bundle rule:** every results-bearing run here is a Hard Rule 7 run — `--out` artifacts, previews,
a summary-bound relative-link `index.html`, a viewer receipt, and a passing
`python scripts/check_results_bundle.py <run_dir>`. Training-free audits (E1, E1c) are exempt from
the viewer receipt but not from the artifact and index requirements.

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

## 0b. The positive hypothesis (what a win would be, mechanistically)

The program above is written defensively — it is built to stop a false positive. Stated alone it
would be satisfied by a well-audited null, and a null is not the reason this pipeline exists. So the
positive thesis is written down here, in falsifiable form, before any of it is tested.

**What Stage 1 uniquely produces.** A fitted 2D Gaussian set is a *per-view reconstruction of the
image*, not a repeatable keypoint detection. Every pixel region that carries radiance gets
primitives, including flat, textureless, low-gradient regions. Classical SfM — the standard 3DGS
initializer — is the opposite: it is detector-driven and texture-biased, and returns nothing at all
where there is no corner to match. This is a structural difference in *what gets represented*, not a
quality difference in how well it is represented.

**H-P1 (the asset is coverage, not accuracy).** The distinctive property of the multi-view + mask →
2D Gaussian → tomographic lift path is **completeness at low primitive count**: coverage of surface
that SfM leaves empty, at a primitive count far below a dense reconstruction. Accuracy of the means
is a *precondition* for that coverage to be useful, not the contribution itself. This is why "the
means look excellent and it did not help" is not a paradox — accurate means with no coverage
advantage, or with coverage the optimizer can trivially manufacture, buy nothing.

**H-P2 (the payoff regime follows from H-P1).** A coverage advantage can only pay where the
optimizer cannot manufacture coverage on its own. Densification manufactures coverage from
photometric residual; that requires photometric constraint and iterations to spend. Therefore init
value should rise as constraint per unit capacity falls — few views, bounded primitive budget, few
iterations — and vanish in the memorization regime. This is the same prediction as H-B2, arrived at
from the mechanism rather than from the Ω arithmetic, and E5 is its sharpest test.

**Pre-declared consequences, so this cannot be retrofitted:**

| If | Then |
| --- | --- |
| beam-fusion beats `colmap-sfm` on E1c *completeness* while tying or losing on *accuracy* | H-P1's premise holds; the asset is real and correctly named |
| beam-fusion beats `colmap-sfm` on accuracy but not completeness | the method is a better triangulator, not a better initializer; the thesis must be rewritten as a geometry contribution and defended on E1c alone |
| the completeness advantage exists but produces no `T@τ` or fixed-budget gain at *any* E5 view count | H-P2 is refuted. Coverage is not the lever, and no repair in Part 3 targets the right quantity. |
| `colmap-sfm` is unavailable or degenerate on a scene (too few points to run 3DGS at all) | that is itself the coverage result, and is reported as such rather than as a missing arm |

**What this hypothesis forbids.** It forbids claiming a win from accuracy of the means alone (E1c is
necessary, never sufficient), and it forbids claiming a win in the unbounded/many-view cell, where
H-P2 predicts a tie and §1.3.5 already declares ties uninformative.

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

**Headroom** `H(c) = metric(oracle, c) − metric(random, c)` for a cell `c` (scene × budget × view
count × resolution), on held-out FG PSNR and on `T@τ`. The maximum advantage any initialization can
confer in that cell. Measured in E0b against the oracle of M5; a lower bound on true headroom, since
the real-scene oracle is a fixed point of the same optimizer.

**Fraction of headroom captured** `F(a, c) = (metric(a, c) − metric(random, c)) / H(c)`. The
normalized quantity reported for every arm in Part 2. Undefined — and reported as undefined, never
as zero — where `H(c)` does not exceed 2× the E0 spread. `F > 1` is possible and is not an error.

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

### M5 — Reference arms: oracle and COLMAP SfM

Two arms that E0b, E1b, E1c and Part 2 all depend on, and neither of which exists today.

**Oracle init builder.** Given a scene and a budget `N`, produce the E0b ceiling state: on synthetic
scenes by sampling GT geometry to `N`; on real captures by training a long unbounded run on **train
views only**, pruning the converged state to `N` by alpha-mass contribution, and refitting at fixed
topology to convergence at `N`. Must refuse to run if the source run's view split does not match the
evaluation split — an oracle contaminated by held-out views silently invalidates every `F` in the
program, so this is a hard failure, not a warning.

**`colmap-sfm` initializer.** COLMAP SfM points as a registered `rtgs.lift.base.Lifter`, so the
standard 3DGS initialization is a first-class arm rather than evaluation infrastructure.
`src/rtgs/data/colmap.py` already loads the reconstructions; what is missing is the points→3D
Gaussian arm and its registration in `rtgs.lift.get_lifter`. Per the repository's working rules this
also needs a pipeline test, a `benchmarks/run.py` entry, and a row in `docs/ARCHITECTURE.md`. Record
the point count per scene — where it is too low to initialize, that is an E1c/H-P1 result.

**Parameter-substitution harness.** The E1b ladder needs component-wise substitution between two
Gaussian states (means / scales / rotation / opacity / SH) with an assertion that untouched blocks
are bit-identical to the donor. Shared with E10.

**Gate:** M0–M5 pass their unit tests, and `rtgs audit-init` reproduces the known 0.0107 alpha IoU
on the Beam CI initialization. If it does not reproduce, stop — the instrumentation is wrong.
Additionally, the oracle builder must beat random by more than the E0 spread on at least one cell of
one scene before E0b's grid is trusted — an oracle that never wins anywhere is far more likely to be
a broken builder than a true universal null, and must be debugged before it is believed.

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

### E0b — Headroom: how much could *any* initializer buy here?

**Question:** in a given cell (scene × budget × view count × resolution), what is the maximum
downstream advantage an initialization can confer at all?

**Why this must exist and must run before Part 2 and Part 3.** Every other experiment compares
initializers to each other. None of them establishes whether the cell has anything to win. E1, E1c,
E2 and E3 can all return clean, encouraging answers in a cell where the ceiling over random is 0.2
dB, and Part 3 would then spend its budget repairing an initialization whose best possible version
changes nothing. The 2026-07-21 null result is consistent with a zero ceiling and nobody has
measured one. This is the cheapest experiment in the program — one arm — and it is the one that
decides whether the rest of it is winnable.

**Oracle construction.** The oracle is a *reference ceiling*, never a competing method, and is
labelled as such everywhere it appears.

| Scene type | Oracle init at budget `N` |
| --- | --- |
| Synthetic | ground-truth geometry, sampled to `N` primitives, GT colour |
| Real capture | converged state of a long unbounded run **trained on train views only**, pruned to `N` by alpha-mass contribution, then refit at fixed topology to convergence at `N` |

The real-scene oracle is the optimizer's own fixed point handed back to it at iteration 0. The
fixed-topology refit at `N` matters: a pruned converged state is not the best `N`-primitive state,
and without the refit the ceiling is understated. Training the oracle source on train views only is
mandatory — an oracle that has seen held-out views measures nothing.

**Setup:** oracle and `random` only, over the full Part 2 regime grid (the E4 caps × the E5 view
counts × the E6 downscales), ≥ 3 scenes, ≥ 3 seeds. No other arm runs at this stage.

**Primary metrics.** For each cell `c`, on held-out FG PSNR and on `T@τ`:

- **headroom** `H(c) = metric(oracle, c) − metric(random, c)`
- and for every later arm `a`, the normalized quantity reported throughout Part 2:
  **fraction of headroom captured** `F(a, c) = (metric(a, c) − metric(random, c)) / H(c)`

`F` is the quantity that separates "no initializer can help in this cell" from "our initializer does
not help in this cell" — the exact distinction the 2026-07-21 design could not make. Where `H(c)`
does not exceed 2× the E0 spread, `F` is undefined and must be reported as undefined, never as zero
and never imputed.

**Pre-declared interpretation:**

| Observation | Reading |
| --- | --- |
| `H` within E0 noise in every cell | **no initialization can matter on these scenes at any budget or view count.** Part 3 is unwinnable and is not run. The program reduces to E11 plus the negative result, which is a complete outcome under §8. |
| `H` within noise at high Ω, large at low Ω | expected under H-P2. The low-Ω cells are the entire experiment; Part 2's full arm set runs **only** there, and the high-Ω cells are reported as measured ties. |
| `H` large everywhere, including the unbounded 2026-07-21 cell | the 07-21 protocol did not erase the effect after all, and something in the current harness differs from it. Stop and reconcile before proceeding — this contradicts the tie that suite observed. |
| `H` large but no real arm captures more than 0.1 of it anywhere | initialization matters and none of our initializers is an initialization. That is a publishable negative and redirects the work to Part 3 with a known target. |

**Consequence for Part 2 — oracle-first grid.** E4/E5/E6 are restructured: run this one-arm sweep
over the grid first, then run the full arm set **only in the cells where `H` exceeds 2× the E0
spread**. Cells with no headroom are reported with their `H` and closed, not populated with five
arms that are guaranteed to tie. The 225-run E4 estimate in §7 is an upper bound that this
restructuring is expected to cut substantially; the reduction is recorded in the RESULT file.

**Watch for:**
- Treating the oracle as an arm. It is not a method, it cannot be proposed, and no claim of the form
  "our init approaches the oracle" is admissible without the E0b noise check on `H` itself.
- The synthetic oracle being *too* good — GT geometry at GT colour can exceed anything reachable
  from images, inflating `H` and deflating every `F`. Report the synthetic and real ceilings
  separately and never pool them.
- A negative `H` (random beating the oracle) in some cell. That is a real and interesting signal —
  it means the schedule is tuned to a cold start — and is reported, not clipped to zero.

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

**E1b — loss localization, measured downstream.** Starting from the true 3D state where available
(synthetic scene with ground-truth geometry), substitute one component at a time from the lift:
means only, means+scales, means+scales+rotation, +opacity, +SH. Report `ΔLE` after each
substitution, **and train each substituted state to report `T@τ` and held-out FG PSNR at the E4
constrained budget cell.**

The downstream half is not optional and is the point of the experiment. `ΔLE` attributes loss of
the *initial render*; `T@τ` attributes loss of *convergence*, and the two can invert. A parameter
block can destroy iteration 0 and be repaired by the optimizer within a few hundred iterations
(irrelevant, however bad it looks), while another can render acceptably and still sit in a basin the
optimizer cannot leave (decisive, and invisible to `ΔLE`). Spending Part 3 on the block that
dominates `ΔLE` rather than the block that dominates `T@τ` is the most expensive mistake available
in this program.

**The two crossover arms carry the result** and must be reported as a pair:

| Arm | Composition | Reading if it converges like the full lift |
| --- | --- | --- |
| `GT-means` | GT means + lift's scale, rotation, opacity, SH | the means were never the bottleneck; tomographic accuracy is not the lever and E7/E8 are correctly targeted at packaging |
| `GT-packaging` | lift's means + GT scale, rotation, opacity, SH | the packaging was never the bottleneck; the means are the lever and E7/E8 are aimed at the wrong block |

**Pre-declared interpretation:**

| Observation | Reading |
| --- | --- |
| `GT-packaging` ≈ full GT, `GT-means` ≈ full lift | packaging is the whole loss. E7/E8 are justified and bounded by this gap. |
| `GT-means` ≈ full GT, `GT-packaging` ≈ full lift | the means are the whole loss. E7/E8 are pointless as specified and the lift itself must be revisited. |
| both crossovers sit between the two ends | the loss is distributed; the split between them is the budget allocation between E7/E8 and lift work |
| neither crossover, nor full GT, beats `random` at this cell | there is no headroom here — cross-check against E0b, and if E0b agrees, stop |

**This experiment is the direct test of the standing puzzle** that beam fusion produced visually
excellent means with no downstream payoff. `GT-means` converging no better than the full lift
explains that observation mechanically and retires it as a mystery.

Prediction: the `ΔLE` drop is dominated by the scale/rotation block, secondarily opacity — and the
`T@τ` attribution is explicitly *not* predicted to follow the same ordering. Recording that
divergence, in either direction, is a result.

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

**Arms:** beam-fusion, dense-merge, splat-sfm, top-K, field, random, **colmap-sfm**. Random is the
control and establishes the error a method must beat to have found anything at all. `colmap-sfm` —
classical COLMAP SfM points, the standard 3DGS initializer — is the external baseline and the arm
that H-P1 is stated against; note that `splat-sfm` is *structure-from-splats*, an internal RGB-free
method, and is not a substitute for it. The accuracy/completeness split is the decisive comparison:
H-P1 predicts beam-fusion wins **completeness** against `colmap-sfm` at equal or lower primitive
count, and does not require winning accuracy. Report the two directions separately for this pair
even if they are pooled elsewhere.

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

**E0b and E1b override the quadrant call.** The quadrant describes where the lift's loss is; it does
not establish that recovering that loss is worth anything. Two results outrank it:

- if E0b finds no headroom in any cell, Part 3 is not run regardless of which quadrant the data
  lands in — a perfectly repaired initialization would still change nothing;
- if E1b's `GT-means` arm converges like the full lift, then accurate means do not help even when
  handed to the optimizer for free, and no improvement to the tomography is worth funding.

Record the headroom verdict and the E1b crossover verdict in the exit paragraphs **before** the
quadrant call, and treat the quadrant as conditional on both.

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
| Initializer | beam-fusion, dense-merge, splat-sfm, **colmap-sfm**, random, (top-K) |
| Cap `max_3d_gaussians` | 1×, 2×, 4×, 8× the arm's native init count, and unbounded |
| Scenes | ≥ 3 |
| Seeds | ≥ 3 (5 for the 1× and 2× cells) |

`colmap-sfm` is the field's standard 3DGS initialization and is a required arm, not a courtesy
baseline. Without it the best available outcome of this sweep is "beam fusion beats our other
lifts," which does not support any claim of the form "better than anything else." Where a scene's
COLMAP reconstruction yields too few points to initialize at all, that is recorded as the coverage
result predicted by H-P1 and the arm is reported as degenerate rather than dropped.

Per E0b, this sweep runs **only in cells whose headroom `H` exceeds 2× the E0 spread.** Cells closed
by E0b are reported with their `H` and not populated.

Native init counts differ across arms (5,000 / 2,088 / 943 / …). Run **both** normalizations and
report both: cap relative to each arm's own init count, and cap at absolute matched counts
(2,000 / 5,000 / 20,000 / 40,000). The relative version tests "does a good init let you stay
small"; the absolute version tests "at equal model size, which init wins." They answer different
questions and conflating them is the most likely way to generate a spurious result here.

**Reference run for `τ`:** random init, unbounded, 70k iterations, held-out FG PSNR plateau.
`τ = plateau − 0.5 dB`, frozen per scene before the sweep.

**Primary metrics — two co-primaries, not one:**

1. `T@τ` (iterations-to-target) — the *speed* axis.
2. **held-out FG PSNR at cap = 1× and 2× init count** — the *compactness* axis: quality reachable at
   a fixed primitive budget, irrespective of how long it takes.

Held-out FG PSNR at fixed iteration budgets {2k, 7k, 30k, 70k} is recorded as secondary.

`T@τ` alone presumes the win is speed. This pipeline produces a compact representation by
construction, and §1.3.5 already predicts every arm ties at the unbounded plateau; if a real effect
exists it is at least as likely to appear as *more quality per primitive* as *fewer iterations*.
Pre-declaring only the speed metric would discard that outcome. Both co-primaries are reported for
every cell, and `F(a, c)` from E0b is reported alongside each.

**Pre-declared decision rule for the project's core claim:**

> The "skip the cold start" claim is **supported** iff, at cap ≤ 2× init count, the best
> structured initializer achieves `T@τ` at most half that of random, with the effect exceeding
> 2× the E0 noise floor, replicated on ≥ 3 scenes and ≥ 3 seeds.
>
> It is **refuted** if the ratio is above 0.8 under the same conditions.
>
> Between 0.5 and 0.8: the effect is real but modest; the project must be reframed as a
> memory-efficiency contribution (E11) rather than a speed contribution.

**Pre-declared decision rule for the compactness claim (independent of the speed claim):**

> The "more quality per primitive" claim is **supported** iff, at cap = 1× init count, the best
> structured initializer exceeds random's held-out FG PSNR by more than 2× the E0 spread of that
> cell, on ≥ 3 scenes and ≥ 3 seeds, **and** captures `F ≥ 0.5` of the E0b headroom there.
>
> It is **refuted** if the margin is within the E0 spread, or if `F < 0.1`, under the same
> conditions.
>
> The two decision rules are independent. Either may be supported without the other, and a
> compactness win with a speed tie is a complete positive result for this program — not a
> consolation reframing.

**Against `colmap-sfm` specifically:** any claim of the form "better than the standard
initialization" requires beating the `colmap-sfm` arm under whichever of the two rules is being
invoked. Beating only internal arms supports a claim about our own variants and nothing more, and
must be worded that way in the RESULT file.

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
of N, **and the same gap against `colmap-sfm`, reported separately** — the gap against random says
the init carries information, the gap against `colmap-sfm` says it beats the standard practice, and
only the second supports the project's claim. `F(a, c)` from E0b is reported for every N. Secondary:
LPIPS and a depth/geometry error where ground truth exists — PSNR is a poor detector of floaters,
which is the failure mode sparse views produce.

This is the sharpest test of H-P2, and the cell where the coverage mechanism of H-P1 should be most
visible: at N = 3 the optimizer has the least photometric residual with which to manufacture its own
coverage, and `colmap-sfm` has the fewest points.

**Pre-declared prediction:** gap ≥ 1.5 dB at N = 3, ≥ 0.5 dB at N = 6, within noise at N = 26.

**What would falsify H-B2:** no gap even at N = 3. That is a strong result and would mean the
lift's information content is genuinely zero (consistent with a large `ΔLE` in E1), or that
3DGS's inductive bias dominates any init.

**Watch for:** at N = 3, all methods may fail catastrophically and tie at the bottom. The E0b oracle
arm at each N is the "can any method reconstruct this at all" control and must be run at every view
count for exactly this reason — a tie at the bottom with `H` within noise is a dead cell, whereas a
tie at the bottom with large `H` is the most interesting cell in the program.

---

### E6 — Resolution / Ω control

**Question:** is the null result an artifact of downscale 16?

**Setup:** downscale ∈ {16, 4, 2} on ≥ 2 scenes, best two budget cells, ≥ 3 seeds. Report Ω for
every cell.

**Primary metric:** init-vs-random gap plotted against Ω, pooling E4/E5/E6 cells. Plotted **twice**:
once as the E0b headroom `H` vs. Ω (what is available), once as `F` vs. Ω (what our arms capture).
Separating these is what makes the figure interpretable — a low gap at high Ω means something
entirely different when `H` is also low than when `H` is large.

**Deliverable:** a single figure — gap vs. Ω across all three sweeps. If the points collapse onto
one curve, that curve is the paper's central claim: *initialization value is a function of the
constraint-to-capacity ratio, and prior 3DGS init comparisons were conducted at Ω where it cannot
be observed.* That is a contribution independent of whether our specific lift wins.

**Pre-declared branch if they do not collapse.** Collapse is a hypothesis, not a given, and
non-collapse is at least as likely: the three sweeps change physically different things — capacity
(budget), angular baseline and coverage (view count), and supervision frequency content
(resolution) — and Ω compresses all three into one scalar that has no reason to be sufficient.

| Observation | Reading |
| --- | --- |
| points collapse onto one curve in Ω | Ω is the governing variable; the claim above stands as written |
| view-count cells lie on a separate, steeper curve | init value is governed by **angular coverage**, not capacity ratio. This is a *better* outcome for a multi-view tomographic method than the collapse, and the paper's variable becomes coverage rather than Ω. It must be reported as the primary finding, not as a failed collapse. |
| resolution cells separate | supervision frequency content governs; the 07-21 downscale-16 confound is confirmed as a distinct mechanism and must be reported separately from the budget effect |
| no ordering in any variable | there is no regime structure to find at this resolution of sweep; report the null and do not fit a curve to it |

Fitting a single curve through non-collapsing points, or dropping the sweep that fails to collapse,
is prohibited. The figure reports whatever structure exists, including none.

---

## 5. Part 3 — Fixes (gated on Part 1)

Run only the fixes that Part 1's attribution justifies. Each has an independent gate; do not bundle.

### E7 — Surfel initialization (SPD-parameterized, normal-oriented)

**Preconditions — all three, not just the first:**

1. E1c must have shown that the means are an asset. Fixing the covariance around inaccurate means
   produces a well-packaged wrong geometry, which is harder to detect than an obviously broken one.
2. E0b must have found a cell with headroom. Repairing an initialization in a cell where the oracle
   ties random is unfalsifiable work.
3. E1b's crossover must have implicated the packaging block rather than the means. If `GT-packaging`
   converges like the full lift, the covariance was never the loss and this experiment is aimed at
   the wrong parameter.

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

The instrumentation gate is **per-experiment, not monolithic**. M3 (lineage) and M4 (budget-honest
density) are the expensive builds, and only E2/E3 and the budgeted runs need them; the training-free
diagnostics need only M0–M2 and must not wait.

```
M0 seed/determinism ─┬─ M1 stage-1 cache ─┬─ M2 audit-init ──► E1, E1c   (training-free)
                     │                                          │
                     ├─ M4 budget-honest density ──► E0 noise floor
                     │                                  └─────► E0b headroom  ◄── gates Parts 2 & 3
                     └─ M3 lineage ────────────────► E2, E3
                                                       │
   E1 + E1c + E1b + E2 + E3 ──► Part 1 exit paragraphs
                                 │  (headroom verdict and E1b crossover verdict first,
                                 │   then the quadrant call, conditional on both)
                                 │
   ├─ if E0b headroom ≈ 0 everywhere ────────► stop. Report the negative + E11. Part 3 not run.
   ├─ if E1b GT-means ≈ full lift ───────────► means are not the lever; do not fund tomography work
   ├─ if means good AND lift broken ─────────► E7, E8 (then re-enter Part 2)
   ├─ if means poor ─────────────────────────► skip Part 3, go to Part 2 + E11
   └─ Part 2, oracle-first: E0b grid ──► full arms in headroom cells only
                                          E4 ──► E5 ──► E6 ──► H-vs-Ω and F-vs-Ω figures
                                           │
                                           └─ E9 (if E3 justified) ──► E10 ablation
E11 runs in parallel, independent of all of the above.
```

Estimated GPU cost was dominated by E4 (5 initializers × 5 caps × 3 scenes × 3 seeds = 225 runs,
now 6 arms with `colmap-sfm`). **The oracle-first restructuring is the cost control:** E0b sweeps the
grid with one arm, and the full arm set runs only where headroom exists. If headroom is confined to
the low-Ω corner as H-P2 predicts, this removes most of the grid. Report the realized run count
against the 225-run upper bound in the RESULT file, with the cells closed by E0b listed explicitly
so the reduction is auditable rather than silent.

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
- The real-scene E0b oracle is a fixed point of the *same* optimizer, so it measures reachable
  quality, not a true supremum. A better `N`-primitive state may exist that this optimizer cannot
  find from any start. `H` is therefore a **lower bound** on true headroom, `F > 1` is possible and
  is not an error, and "no headroom" means "none this optimizer can exploit" — which is the
  decision-relevant quantity here, but is not the same statement.
- `colmap-sfm` is only as strong as its reconstruction. A weak COLMAP makes an easy baseline and a
  worthless comparison. Use the same reconstruction that produced the calibration, state the
  settings and the resulting point count per scene, and do not tune our arms against a baseline that
  was not tuned at all.
- H-P1's coverage argument is measured on masked, object-centric foregrounds, which is close to the
  regime where classical SfM is weakest. It is the favourable case for the hypothesis, and a
  completeness win there does not establish one on textured, unbounded scenes where SfM is strong.
  Say so wherever the coverage claim is made.
