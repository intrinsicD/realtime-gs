# Preregistration: stage-1 weight/color gauge contract audit

## Chronology, question, and scope

Frozen at `2026-07-16T00:38:30+02:00`, before any implementation, pilot, diagnostic,
transformed render, coverage map, retained-set comparison, lift, or outcome for this audit.

Pre-implementation clarification at `2026-07-16T00:59:00+02:00`, after an independent API and
executability review but still before any implementation, diagnostic, transform, render, lift, or
outcome. The initial text had SHA-256
`2317a03254a4b6bd4802198ffd3cc83f5a69ae3a90d20109e5137e14ad66d362`. The clarification
corrects the held-out isolation wording to acknowledge full synthetic-scene construction, binds
the points/bounds that Carve consumes, freezes empty-lift handling, and makes the diagnostic
Carve reconstruction executable. It changes no gauge, tolerance, scene, seed, backend, metric,
materiality gate, or interpretation.

Pre-implementation artifact-routing clarification at `2026-07-16T01:38:57+02:00`, while the
harness was still under implementation and before any official seal, attempt marker, fit,
diagnostic, transform, render, lift, or outcome. The preceding text had SHA-256
`6084c4542141c829126809c22fb756f595340e130cc79ba7ecbc33587053feba`. The sole command's
`--output` is now uniquely defined as the prospective valid audit path; the invalid sibling is
derived and both are preflighted before the marker. This closes an append-only routing ambiguity
without changing any scientific choice or permitting a retry.

Pre-implementation seal-command clarification at `2026-07-16T01:42:26+02:00`, before any
official seal, attempt marker, fit, diagnostic, transform, render, lift, or outcome. The preceding
document had SHA-256
`e7672199f3e97a230adbf6bcc9a13405b642f2912237b3b27f800fb71edcc225`. It supplies the missing
outcome-free command that creates the already-required implementation seal and distinguishes that
preparatory operation from the sole scientific command. It changes no scientific choice.

Pre-implementation verification-binding clarification at `2026-07-16T01:45:29+02:00`, while the
harness and toy tests were still under implementation and before any seal, official attempt,
official-seed preparation, gauge transform, diagnostic, lift, or outcome. The preceding document
had SHA-256 `ddd26effde4b4ec7f89dc383f4174a23bb36fef6ee406b8c6a581f295c03270a`. The seal is now
explicitly permitted and required to run a fixed verification subprocess list containing only
lint/format, non-slow toy tests, docs sync, and diff validation. This resolves how the already-
required verification evidence enters the seal; it does not authorize a pilot or scientific
preparation and changes no experiment choice.

The native stage-1 renderer represents component `i` by

`w_i * c_i * exp(-q_i/2) * 1[q_i < 12]`,

where scalar `w_i` and RGB `c_i` are optimized independently. Consequently, multiple bounded
`(w_i,c_i)` factorizations can encode the same additive RGB component. The downstream boundary is
not factorization-neutral: Carve uses `w_i` as soft coverage and a retention threshold and uses
`c_i` in tunnel color matching; Depth uses `w_i` as a retention/merge signal; both initialize
3D opacity independently and copy `c_i` into degree-0 SH. This audit asks whether exact
product-preserving gauge changes that leave every source-view stage-1 RGB reconstruction
numerically equivalent nevertheless cause material coverage, retention, or unoptimized 3D-lift
changes.

This is a deterministic CPU synthetic **representation-contract validity audit**, not an
optimization or quality experiment. It has no candidate training arm, held-out metric, density
control, parameter sweep, production default, real-data, CUDA/gsplat, speed, or memory claim. A
positive result would establish only that the current native `Gaussians2D` boundary is materially
factorization-dependent in the tested Carve and/or Depth path; it would not identify the correct
physical interpretation of weight, color, opacity, confidence, or coverage, and would not
authorize a canonicalization or default change. A negative result would be restricted to the two
frozen gauges, scenes, and thresholds below and would not prove general gauge invariance.

The already-frozen Carve merge experiment at
`benchmarks/results/20260715_carve_merge_controls_PREREG.md` asks whether moment merging beats two
exact-count allocation controls from one fixed raw Carve tensor. This audit neither amends nor
duplicates that protocol: it performs no merging, exact-count construction, pruning, refinement,
or checkpoint evaluation; it creates fresh fits and reruns an unmerged lift only to test the input
representation contract. No artifact or outcome from either experiment may enter, gate, repair,
or reinterpret the other.

## Literature and method boundary

The 2026-07-12 through 2026-07-16 Scholar Inbox digest motivates auditing the boundary but does
not predict an outcome. [AsySplat](https://arxiv.org/abs/2607.10995) separates coarse geometry from
fine appearance processing. [SalientGS](https://arxiv.org/abs/2607.11285) derives allocation from
multi-view residual underfit/redundancy, and
[SpeedyGS](https://arxiv.org/abs/2607.12656) treats pruning and structural formation under a
rate-distortion objective. These are reasons not to silently equate one fitted scalar amplitude
factor with physical occupancy or value. [MAC-Splat](https://arxiv.org/abs/2607.10792) and
[Incremental Gaussian Triangulation](https://arxiv.org/abs/2607.10690) motivate explicit
world-frame attribute and planar-geometry semantics, respectively, but this audit implements none
of their matching, losses, surfel constraints, or training machinery.

The source representation and controls remain the repository's GaussianImage-style accumulated
mixture, the existing Seitz-Dyer/Kutulakos-Seitz-inspired Carve path, and the existing metric-depth
surface lift. Runnalls/Hierarchical-3DGS moment reduction and classic 3DGS density allocation are
deliberately absent because merge and budget utility are separate questions.

## Frozen execution environment, data, and fit

- CPU only: `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, Torch intra-op
  threads `4`, deterministic algorithms enabled, and no optional StructSplat or gsplat import.
- Seeds are exactly `0,1,2`.
- For each seed, call
  `make_synthetic_scene(n_gaussians=40, n_cameras=12, image_size=48, seed=seed)` once. This helper
  necessarily constructs/renders all twelve synthetic views internally. Freeze training indices
  `[0,1,2,4,5,6,8,9,10]`, held-out indices `[3,7,11]`, and the local-to-original training-view
  map, then physically subset the scene to the nine training views. After that prescribed full
  scene construction, held-out per-view RGB, depth, masks, cameras, or Gaussians are never
  subsequently accessed, scored, hashed, fit, transformed, lifted, rendered, or used in a
  decision. Before either lifter, pass a field-minimal training-scene copy with `gt_gaussians=None`;
  retain only the nine aligned per-view fields and the explicitly bound world-space priors below.
- Fit the nine training images exactly once per seed with native stage 1 and share those immutable
  tensors across every gauge and both lifters. Use
  `FitConfig(n_gaussians=150, max_gaussians=5000, iterations=120, backend="native",
  adaptive_density=True, growth_waves=5, relocate_fraction=0.0,
  structsplat_renderer="auto", lr=0.01, grad_init_mix=0.7, row_chunk=64, log_every=50,
  convergence_patience=0, convergence_tol=0.05, convergence_check_every=25)` and
  `fit_views(train_scene.images, config, seed=seed, masks=train_scene.masks)`.
- Require exactly nine fitted sets and exactly 150 finite components per set. Require every
  `xy/chol/color/weight` tensor to satisfy its public shape/range contract and both Cholesky
  diagonals to be positive. Hash every training image/camera/depth, fitted field, fit history,
  per-view source ordering, local-to-original view map, and aggregate before constructing a gauge.
  Separately hash the retained `scene.points`, point-visibility tensors, `bounds_hint` (including an
  explicit null value), and the exact center and full-diameter extent returned by
  `train_scene.center_and_extent()`. Serialize the center/extent and require every gauge and both
  lifters to use that identical bound input; these values determine Carve's volume and normalized
  displacement metrics.
- No refit, additional fitting iteration, mask change, component reorder, clamp, threshold change,
  or outcome-dependent transform is permitted.

## Frozen exact product-preserving gauges

For every fitted component, compute the native-float32 additive amplitude vector once:

`a_i = w_i * c_i`.

Copy `xy` and `chol` bit-for-bit in every arm. Construct exactly these three representations in
the original order:

1. `identity`: unchanged `(w_i,c_i)`.
2. `unit_weight`: `w_i' = 1`, `c_i' = a_i`.
3. `peak_color`: let `p_i=max_c(a_i,c)`. If `p_i>0`, set `w_i'=p_i` and
   `c_i'=a_i/p_i`; if `p_i=0`, set both fields to exact zero.

Both transformed gauges preserve `w_i' c_i'=a_i` mathematically while selecting opposite valid
factorizations: unit scalar weight versus one saturated peak color channel. They are not proposed
physical models or candidate defaults. Construction must not use the source target, coverage,
depth, Carve volume, a render, or another component.

For each transformed component require finite fields, `0<=w_i'<=1`, `0<=c_i'<=1`, bit-exact
`xy/chol`, and amplitude agreement with `a_i` at maximum absolute error `<=1e-7` and maximum
relative error `<=1e-6`, with relative error computed only where `|a_i|>1e-8`. Serialize raw
maximum errors and hashes. Any violation invalidates the entire official attempt before a source
render.

Report, but do not gate downstream choices on, fixed bins for original/transformed weight
`[0,.01),[.01,.02),[.02,.05),[.05,.10),[.10,.25),[.25,.50),[.50,.75),[.75,1]`, component
peak color using the same bins, and fractions whose weight, any color channel, or both change by
more than `1e-7`.

## Mandatory source-reconstruction equivalence prerequisite

This stage must complete for all 54 transformed source views (3 seeds x 9 views x 2 transforms)
before the harness may call `render_gaussian_coverage_2d`, form a retention mask, instantiate a
lifter, or compute/print/serialize any downstream comparison. Render `identity`, `unit_weight`,
and `peak_color` with `render_gaussians_2d(height=48,width=48,row_chunk=64)`, black additive
background, no clamp, and no gradient.

For each transform/view, require:

- all three raw renders finite;
- maximum absolute transformed-minus-identity RGB error `<=5e-6`;
- float64 `sum(abs(delta))/sum(abs(identity)) <=1e-6`, with a strictly positive finite
  denominator;
- PSNR between raw transformed and identity renders `>=100 dB`, using an MSE floor of `1e-12`
  only for this equivalence diagnostic.

Hash and serialize every render and raw numerator/denominator only after **all** equivalence checks
pass. If any check fails, write an append-only invalid result containing only preparation,
transform, and equivalence evidence; print no coverage, retention, lift, or materiality statistic;
do not repair a transform or relax a tolerance. The official attempt is consumed.

## Downstream diagnostic 1: coverage and retention

Only after the global source-equivalence prerequisite passes, compute the following on all nine
training source views for every gauge.

### Coverage

Use the unchanged `render_gaussian_coverage_2d(g,48,48,row_chunk=64)` definition
`C=1-exp(-sum_i w_i G_i)`. Relative to identity, aggregate float64 raw sums per seed and across
seed-tagged pools:

- `coverage_delta_l1=sum|C_g-C_0|`;
- `coverage_reference_l1=sum|C_0|`;
- `coverage_delta_over_reference=coverage_delta_l1/coverage_reference_l1`;
- count and fraction of pixels for which `(C_g>0.40) xor (C_0>0.40)`;
- maximum absolute delta and per-view raw values.

All maps and denominators must be finite and `coverage_reference_l1>0`.

### Retention

Apply only the production strict mask `weight>0.05`. Identify every component by
`(seed, local_train_view_index, original_component_index)`. Report per-view, per-seed, and pooled
identity/transformed counts, intersection, union, symmetric difference, Jaccard, and exact keys.
No color, coverage, target, or depth may repair or replace this mask.

For a given transform/seed, `input_consumption_material` is true only if at least 10% of
components jointly changed weight and color and either:

1. retention symmetric difference is at least 10 keys and at least `1%` of the union; or
2. `coverage_delta_over_reference>=0.01` and at least `0.1%` of source pixels cross the frozen
   `0.40` coverage threshold.

Pool counts and sums before forming percentages. A transform has a material consumed-input effect
only if the **same named transform** passes in at least two of three seeds and independently in the
raw-sum pool. The two transforms may never be selected separately per seed or averaged together
to manufacture a pass.

## Downstream diagnostic 2: unmerged Depth lift

For every gauge, run exactly one fresh unmerged metric-depth lift from the same fitted tensors and
training scene:

`DepthLifter(backend=GroundTruthDepth(train_scene.gt_depths), sh_degree=0,
min_weight=0.05, init_opacity=0.1, normal_thickness=0.15, covariance_mode="surface",
isotropic_sigma=None, robust_depth_gradients=True, merge=False, merge_voxel_frac=0.01)`.

The harness must independently reproduce the production finite-depth, `z>0.05`, strict-weight,
optional-mask, and confidence validity masks, bind every emitted primitive to its immutable source
key, and assert production output order and count. For source keys shared with identity, require
means, covariance, and opacity agreement within `atol=2e-6, rtol=2e-5`; this is a validity control
showing that unchanged pixel geometry/depth did not silently change. SH/color is intentionally not
required to agree.

Report exact output key-set comparisons; field hashes; shared-key mean/covariance/opacity/SH
deltas; and raw train-view color/alpha/accumulated-depth render deltas under
`TorchRasterizer(sh_color_activation="hard",kernel_support_mode="hard",
visibility_margin_sigma=3.0)`, degree 0, black background, and no clamp.

## Downstream diagnostic 3: unmerged Carve lift

For every gauge, run exactly one fresh unmerged lift:

`CarveLifter(grid_res=48, bounds_scale=0.5, min_views=2, hull_fraction=0.85,
color_std_sigma=0.20, color_match_sigma=0.35, coverage_thresh=0.40,
samples_per_ray=64, min_score=0.05, min_weight=0.05, merge=False,
merge_voxel_scale=1.0, init_opacity=0.1, sh_degree=0)`.

For each gauge, call ordinary `CarveLifter.lift` exactly once. A separate harness-only diagnostic
reconstruction duplicates the current Carve arithmetic but is not a second lifter invocation: it
must construct bounds, source coverage maps, `n_seen`, `n_covered`, color moments, hull,
consistency, ray tunnels, scores, and placements without changing any core expression. For every
source key, serialize `keep` and nullable `valid_ray`, `best_score`, `best_idx`, selected depth,
raw score-weighted depth variance, clamped ray sigma, and `placed`; undefined JSON fields are
`null`, never NaN. Hash the coverage maps and all named volume tensors. Form emitted keys as the
view-ordered, ascending-original-index `keep_indices[placed]` sequence. Before any comparison,
require the reconstruction for **every** gauge to match its one ordinary core output in ordered
keys, means, covariance, opacity, and SH at `atol=2e-6,rtol=2e-5`. Quaternion signs are not a
parity field because `q` and `-q` encode the same covariance. Any parity failure invalidates the
audit.

Report coverage-volume hashes, hull/consistency summaries, per-source keep/valid-ray/placed counts,
output key-set comparisons, and, on shared keys, center displacement normalized by train-scene
extent, relative covariance Frobenius delta, opacity delta, SH delta, tunnel-score delta, and
selected-depth delta. Render every unmerged output in all nine training views with the same frozen
Torch renderer used for Depth and report raw color/alpha/accumulated-depth deltas.

No Carve raw tensor, grouping, group count, merge arm, representative, global prune, or metric from
`20260715_carve_merge_controls` may be loaded or compared. Conversely, this audit cannot amend
that protocol's raw tensor or interpretation.

## Frozen lift materiality formulas and decisions

For backend `b`, seed `s`, and named transform `g`, let `S_0,S_g` be seed-tagged emitted source-key
sets and let `R_0,R_g` be all nine raw training color renders. Compute float64:

- `set_disagreement=|S_0 symmetric_difference S_g|/|S_0 union S_g|`;
- `render_delta_l1=sum|R_g-R_0|`;
- `render_delta_over_signal=render_delta_l1/sum|R_0|`;
- `render_delta_over_residual=render_delta_l1/sum|R_0-target|`.

Every denominator must be finite and strictly positive. Also serialize raw alpha/depth differences,
but they are diagnostic and cannot rescue a failed color/set gate.

`lift_material` is true for one backend/seed/transform only when at least 10% of source components
jointly changed weight and color and either:

1. at least 10 emitted source keys differ and `set_disagreement>=0.01`; or
2. `render_delta_over_signal>=0.001` **and** `render_delta_over_residual>=0.01`.

Pool source-key counts and render numerators/denominators over seed-tagged data before recomputing
the same formula. A backend is declared materially gauge-dependent only if the **same transform**
passes `lift_material` in at least two of three seeds and independently in the pooled data. Depth
and Carve decisions are separate; one cannot rescue the other. Input-consumption materiality is
reported as mechanism evidence but is neither necessary nor sufficient to override the backend
lift decision.

Interpretation is frozen:

- Source-equivalence prerequisite failure: invalid representation audit; no downstream evidence.
- Source equivalence passes, neither backend passes: the two tested gauges did not produce a
  material unoptimized-lift difference under this setup; make no universal invariance claim and
  do not tune transforms or gates.
- Depth only passes: the retained/color boundary is materially gauge-dependent here; no claim
  about Carve geometry.
- Carve only passes: the coverage/color-match placement boundary is materially gauge-dependent
  here; no claim about Depth.
- Both pass: both tested boundaries are materially gauge-dependent in this narrow setup, still
  without identifying a correct replacement.

The audit ends after this decision. There is no Phase B, candidate canonicalization, optimizer,
held-out evaluation, merge, density control, or follow-up sweep in this protocol.

If any gauge produces an empty global Depth or Carve output, production concatenation and the
frozen render/set denominators are not defined. Treat that event as a fail-closed invalid official
attempt, not as material gauge dependence: serialize only the evidence validly reached before the
empty-output exception, make no backend decision, and do not repair the lifter or add an empty-set
rule. The once-only attempt remains consumed.

## Invariants, provenance, and fail-closed artifacts

The future harness and focused CPU tests must be complete and independently reviewed before an
official seal. Tests must cover transform bounds/product identity including zero amplitude,
source-render fail-closed ordering, exact source keys, raw-sum pooling, fixed thresholds,
Depth shared-geometry control, Carve sidecar parity, transform-specific 2/3 decisions, and refusal
to overwrite artifacts. The complete repository CPU verification gate must pass before sealing.

The seal must bind this preregistration, harness, tests, every repository-owned loaded source,
revision and dirty diff, environment, command, fixed configs, and source aggregate. The runtime
must re-hash all bound sources before consuming one atomic attempt marker. Serialize complete raw
per-view numerators/counts/keys and effective configs; summaries alone are insufficient. No
official artifact may be overwritten or reconstructed from terminal output.

Fresh append-only namespace:

- harness: `benchmarks/stage1_weight_gauge_audit.py`;
- seal: `benchmarks/results/20260716_stage1_weight_gauge_SEAL.json`;
- once-only marker: `benchmarks/results/20260716_stage1_weight_gauge_AUDIT_ATTEMPT.json`;
- prospective valid output: `<UTC>_cpu_stage1_weight_gauge_audit.json`;
- derived invalid output: replace the prospective valid path's terminal `_audit.json` exactly
  once with `_invalid.json`;
- companion note: matching `_RESULT.md`;
- independent scientist review: matching `_AUDIT.md` (and machine-readable review if produced).

The sole outcome-free implementation-seal command is:

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python benchmarks/stage1_weight_gauge_audit.py seal --output benchmarks/results/20260716_stage1_weight_gauge_SEAL.json`.

Seal creation must run, in order, exactly `.venv/bin/python -m ruff check .`,
`.venv/bin/python -m ruff format --check .`,
`.venv/bin/python -m pytest -q -m "not slow"`, `.venv/bin/python scripts/docs_sync.py`, and
`git diff --check`. It records each literal command, exit status, complete stdout/stderr, and
SHA-256 of that output in the seal and refuses to write the seal on any nonzero exit. These
verification subprocesses may execute repository unit tests that construct/render toy fixtures;
the focused gauge tests remain forbidden from fitting, transforming, lifting, or otherwise
preparing official seeds/configurations. The seal command itself and its verification children may
not invoke the harness's `audit` action, claim the attempt marker, run the official fit/lifters,
probe an official seed, or expose a scientific metric. Apart from the bounded toy-test exception,
seal creation may only hash/read source and environment. The sole future official scientific
command is:

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python benchmarks/stage1_weight_gauge_audit.py audit --seal benchmarks/results/20260716_stage1_weight_gauge_SEAL.json --output <fresh-UTC>_cpu_stage1_weight_gauge_audit.json`.

The supplied path must end in `_audit.json`. Before creating the attempt marker, the harness must
derive the `_invalid.json` sibling, require that neither path nor either possible companion note
exists, and bind both paths into the marker. A valid completion writes only the supplied audit path
and its note. A fail-closed completion writes only the derived invalid path and its note. The
harness must never write both scientific JSON outcomes, redirect after observing an outcome, or
overwrite either possibility.

An invalid or interrupted official attempt remains consumed. Any repair requires an append-only
retry preregistration, new seal, new marker, and new output namespace while retaining the failed
artifacts. No result from this audit may retroactively alter
`20260715_carve_merge_controls_PREREG.md`, change a default, open an optimization utility arm, or
support a real/CUDA/performance claim without a separately preregistered experiment.
