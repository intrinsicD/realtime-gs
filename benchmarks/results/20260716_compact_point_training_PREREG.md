# Compact RGB-free point-training experiment preregistration

Status: outcome-blind preregistration amended before sealing; no official outcome access

Date: 2026-07-16 (Europe/Berlin)

## Question and claim boundary

Can a fixed-topology 3D Gaussian model be refined directly against lossless fitted 2D Gaussian
fields, without loading raw source RGB after the Stage-1 boundary, by evaluating both teacher and
student only at sampled image-plane points? The teachers still encode color supervision; the
boundary is a process/authority and representation boundary, not a claim that color information
has disappeared. Within that mechanism, does an
importance-corrected Gaussian proposal improve convergence per attempted query over a
domain-matched uniform proposal?

This experiment may establish only a CPU synthetic mechanism result and a bounded calibrated
integration result. It cannot establish novel-view quality, superiority to conventional RGB
3DGS, GPU speed, end-to-end memory savings, density-control quality, or a production default.
Topology remains exactly `N_init,3D = N_opt,3D` throughout. Split, clone, prune, merge, opacity
reset, SSIM, masks, RGB supervision, depth supervision, and regularizers are excluded.

The original proposed three-arm comparison is strengthened to four arms because finite-pixel and
continuous-area estimators target different risks. The paired contrasts are:

1. `pixel_uniform` versus `pixel_gaussian`: uniform fitted-window pixel centers versus the
   established importance-corrected `GaussianPixelProposal` mixture.
2. `area_uniform` versus `area_gaussian`: uniform continuous fitted-window coordinates versus the
   established importance-corrected `GaussianPointProposal` mixture.

The two pairs must not be compared as if they optimize the same objective. A common exhaustive
pixel-center score across all four arms is descriptive only.

## State-of-the-art grounding frozen before implementation

The design is grounded in the previously reviewed Gaussian Point Splatting / W-VEG sampling idea:
sample compact Gaussian fields directly rather than reconstructing dense source images. A fresh
authenticated Scholar Inbox digest search on 2026-07-16 added current context from DiffGI
(continuous subpixel differentiability), SalientGS (importance-guided allocation), Incremental
Online Scene Reconstruction (memory-aware freezing), Glob3R (global refinement), CASA-SDF
(curvature-guided density), and Bake It Till You Make It (sparsity and appearance decoupling).
None supplies a reason to mix topology changes into this estimator test. Density and residual
allocation are deferred hypotheses.

## Frozen mathematical objective

For view `i`, frozen teacher `T_i`, student rendering `C_theta,i`, and RGB point loss

```text
l_theta,i(x) = ||C_theta,i(x) - T_i(x)||_2^2 / 3,
```

views are sampled uniformly, independent of fitted-window area, teacher mass, or `N_opt,2D`.
The discrete pair estimates

```text
J_pixel(theta) = (1/V) sum_i (1/P_i) sum_{p in fitted pixels i} l_theta,i(p),
```

and the continuous pair estimates

```text
J_area(theta) = (1/V) sum_i (1/A_i) integral_{fitted window i} l_theta,i(x) dx.
```

For `S` attempted draws in one step, including rejection nulls, the only training estimator is

```text
L_hat = (1/S) sum_s importance_s * l_s.
```

`fixed_attempt_mean` performs this reduction. It must use marginal `proposal_density`, never
`joint_density`; never resample nulls; and never divide by active count, weight sum, chunk count,
or teacher mass. Inactive attempts contribute differentiable zero. Black is the only student
background. Inside CompactTrainer, supervision is teacher color only; teacher `weight_sum`,
proposal component id, and initialization lineage cannot select/filter student Gaussians or become
alpha supervision. This does not prohibit the separately frozen `CompactCarveInitializer` from
using `weight_sum` for its documented coverage score before training.

The mixture fraction is exactly `eta=0.25`, bounding active importance by `1/eta=4`. Uniform arms
use the same proposal class as their paired mixture with `uniform_fraction=1.0`. Gaussian arms use
`uniform_fraction=0.25`. The existing rejection sampler already makes every outside-fit Gaussian
draw inactive because paired component weight includes `valid_domain`; an explicit regression
must prove `active => inside_fit_window` and zero joint/target/proposal/importance on invalid rows.

## CompactTrainer contract frozen before implementation

Add a separate `rtgs.optim.compact_trainer` module. It consumes only
`ReconstructionInputs`, `Gaussians3D`, pluggable `ObservationQueryBackend` instances, and a
pluggable `PointRasterizer`. It must not accept `SceneData`, image tensors, masks, image paths, or
RGB loaders. The CPU default is one `GaussianObservationIndex` per teacher and
`TorchPointRasterizer`; faster backends can implement the same interfaces later.

The fixed topology optimizer mirrors the established 3DGS parameterization and Adam settings:
raw means; raw quaternions normalized only by covariance evaluation; log-scales; opacity logits
mapped through sigmoid; degree-zero SH and remaining SH in separate groups; Adam `eps=1e-15`;
group order means, quats, scales, opacities, sh0, shN; learning rates `1.6e-4 * extent`, `1e-3`,
`5e-3`, `5e-2`, `2.5e-3`, and `2.5e-3/20`; and means LR decay `0.01**(1/iterations)` after each
update. Use six separate Adam instances in that order with `betas=(0.9,0.999)`, `eps=1e-15`,
`weight_decay=0`, `amsgrad=False`, `maximize=False`, `foreach=False`, `fused=False`, no gradient
clipping, and `zero_grad(set_to_none=True)` before the zero anchor and microbatch backwards. Extent
comes from an explicit config value, then `bounds_hint`, then the detached initial mean cloud.
Official runs use explicit extent `1.0` and SH degree zero throughout.

The official point renderer is `TorchPointRasterizer(point_chunk=32, gaussian_chunk=64,
sh_color_activation='hard', sh_smu1_mu=2/255, kernel_support_mode='hard',
visibility_margin_sigma=3.0)` with black background and no diagnostic/surrogate modes. Training
outer microbatch size is 32 and teacher component-query chunk is 64. The five nonempty official
parameter families are means, quaternions, log-scales, opacity logits, and degree-zero SH; the
empty `shN` group still receives the zero anchor and an aligned Adam clock.

Each arm receives an exact detached clone of one initialization and fresh Adam state. A view
schedule is generated before optimization from an isolated generator. Each step resets a second
generator from the frozen seed/step mapping, so differing proposal RNG consumption cannot shift a
later view or later step. Within each domain pair, both arms use identical seed mapping. Every
step performs exactly 128 attempts and one optimizer step, including all-null/no-visible steps;
such steps attach `0 * sum(all parameters)` so all group clocks remain aligned. All attempted
coordinates, including zero-importance nulls, pass through teacher and student point queries, so
attempt counts and query-call counts are matched. Point--Gaussian work is not claimed matched:
student visibility and teacher tile candidates can differ, so time, pairs, and memory are
descriptive. The trainer backpropagates fixed outer query microbatches, scaling
each microbatch sum by the original 128-attempt denominator, and takes one optimizer step only
after all microbatches. Internal `TorchPointRasterizer` chunks alone are not an autograd-memory
bound because their graphs survive until concatenated backward; the outer microbatch is the bound.
No early stopping is allowed. History records view, attempts, active/null/invalid counts, visible
count, sampled loss, importance max, importance ESS, rendered point--Gaussian pairs, per-group LR,
cardinality, wall time, and peak resident memory.

Before allocating default tile indices, the trainer preflights fitted pixels, component count,
estimated component--tile overlap entries, and archive bytes when a bundle path is known;
configured caps fail closed. Constructed indices report non-empty tiles, total entries, and maximum
candidates per tile. Evaluation streams coordinates in bounded chunks and never materializes an
image. Synthetic official runs evaluate every frozen checkpoint. A narrow
`evaluate_checkpoint_risks=False` mode may be used only by the calibrated integration run: it
still constructs, detaches, hashes, and exposes every requested checkpoint snapshot to the
checkpoint callback, but does not invoke the exhaustive evaluator or place risk values in the
checkpoint record. This exception prevents full-resolution checkpoint diagnostics from dominating
the compact training workload; it does not change sampling, gradients, optimizer updates, or the
official synthetic experiment. Standalone evaluation remains available and reports:

- exact equal-view `J_pixel` over every fitted-window pixel center;
- deterministic equal-view area quadrature using offsets `(0.25,0.25)`, `(0.75,0.25)`,
  `(0.25,0.75)`, `(0.75,0.75)` in every fitted pixel cell;
- per-view SSE/count and elapsed time.

All reductions cast squared errors to float64 and pool within a view before averaging view means.
Teacher and prediction fractions below zero and above one are reported without clamping. Teacher
epsilon remains black pseudo-mass, never alpha; `weight_sum` remains diagnostic only.

The literal default safety caps are 64 views, 50,000,000 fitted pixels per view, 2,000,000
components per view, 16,000,000 component--tile entries per view, and 200,000 candidates in one
tile. The no-allocation entry estimate uses the index's exact rounded support center, integer
radius, fit-window clipping, floor-divided tile bounds, and inclusive tile-area sum for every
positive-amplitude component. It runs before any tile list/tensor allocation; the constructor
checks the same cap while building and the per-tile cap afterward.

With no explicit extent and no `bounds_hint`, extent is computed from detached initial means:
use per-axis 0.01/0.99 quantiles when `N>=20` (otherwise min/max), their midpoint as center, the
0.99 quantile of radial distances when `N>=20` (otherwise max), then
`max(2.2*radius, 1e-3)`. This value is frozen before optimizer construction.

## Outcome-free prerequisite gates

Before sealing, development-only seeds distinct from official seeds must prove:

1. the active/null fixed-attempt formula, including all-null behavior;
2. finite, materially nonzero off-grid coordinate, mean, and log-scale gradients for a safely
   interior anisotropic Gaussian, with float64 central-difference agreement;
3. global-compositor loss changes when an unrelated visible 3D Gaussian changes, regardless of
   proposal component metadata;
4. same pre-sampled loss and all five effective parameter-family gradients across small and large
   point/Gaussian chunks;
5. same deterministic view schedule across all modes and stable per-step sample streams;
6. fixed `N`, unchanged teacher tensor/digest values, finite parameters, and aligned Adam clocks;
7. training a saved/reloaded `ReconstructionInputs` bundle in a fresh process while PIL,
   `SceneData`, calibrated RGB loading, and source-image access are patched to fail, with strict
   identifier/path grammar, exact manifest keys, symlink/resolved-escape rejection, and an archive
   byte cap;
8. translated fit-window coordinates, field versus indexed query parity, and chunked evaluation
   parity with a materialized tiny reference;
9. four-arm official fixture construction is unreachable on module import and only callable
   after a once-only marker.

The numeric development gates are frozen as follows. In gate 2 the maximum absolute coordinate,
mean, and log-scale gradients must each exceed `1e-8`; symmetric float64 central differences use
step `1e-4` and must match autograd at `atol=2e-6, rtol=2e-3`. Gate 3 requires a maximum RGB loss
change above `1e-5`. Gate 4 compares scalar loss and the gradients of means, quaternions,
log-scales, opacity logits, and degree-zero SH between one batch and outer microbatches at
`atol=5e-6, rtol=5e-5`; every family must be finite and have maximum absolute reference gradient
above `1e-10`.

Strict bundle preflight occurs before any `np.load`: manifest size at most 8,388,608 bytes; at
most 64 teacher archives; each compressed archive at most 268,435,456 bytes; all referenced
compressed archives together at most 2,147,483,648 bytes; at most 64 ZIP members per archive;
each uncompressed member at most 268,435,456 bytes; and aggregate uncompressed members per archive
at most 1,073,741,824 bytes. Exact manifest/metadata key sets, identifier regex
`[A-Za-z0-9][A-Za-z0-9_.-]{0,127}`, ordinary-file/no-symlink requirements, and resolved containment
under the bundle root are mandatory.

The focused tests, then `ruff check .`, `ruff format --check .`, and full `pytest -q`, must pass.
An outcome-blind implementation review must bind the final preregistration and source hashes and
return `Verdict: PASS` with no unresolved findings before the seal can be written.

## Frozen official synthetic fixture

Official perturbation/sampling seeds are exactly `(74101, 74102, 74103)`. They are forbidden in
development tests, pilots, notebooks, interactive calls, or pre-seal construction.

One float32 target contains four degree-zero colored anisotropic 3D Gaussians inside a unit-scale
scene and three fixed 32x32 inward-looking cameras. The harness projects the target means and
covariances using the established point-rasterizer equations, including 0.3 pixel-squared
dilation, and constructs normalized compact teachers with epsilon `1e-8`,
`sigma_cutoff=sqrt(12)`, hard rectangular support, black outside support, constant color, and
amplitude equal to target opacity. This matches the real StructSplat blend family but remains a
synthetic, imperfectly reachable teacher rather than RGB or target-geometry truth. Components
are co-located and amplitude-split without changing the field: view cardinalities are exactly
`N_opt,2D=(4,5,6)`, with recorded `N_init,2D=(8,8,8)`. This exercises unequal per-view fitted
budgets without allowing cardinality to change view weighting. Provider metadata must truthfully
identify a synthetic fixture rather than StructSplat. Schema v1 is extended only to accept the
literal providers `structsplat` and `synthetic_fixture`; the official fields use
`provider='synthetic_fixture'` and round-trip that value. Every fit window is the full 32x32 canvas.
Exact camera-depth ties and initialization values within `1e-5` of near-plane, support,
opacity-clamp, visibility, or SH-activation boundaries are hard nonvacuity failures.

For each official seed, initialization is the same four target identities with frozen seeded
asymmetric perturbations to means, quaternions, log-scales, opacity logits, and degree-zero color.
Thus `N_init,3D=N_opt,3D=4`. All arms run 120 updates, 128 attempts per update, `eta=0.25`, black
background, extent 1.0, SH degree zero, checkpoints `(0,30,60,120)`, point/outer chunk 32,
Gaussian chunk 64, teacher query chunk 64, teacher tile size 8, and evaluation chunk 256. Each
seed--arm runs in a fresh `multiprocessing` spawn worker which revalidates the marker and seal
before construction; only that worker's `ru_maxrss` is reported as its comparable peak RSS.

The literal target arrays, cameras, split map, perturbation formulas, seed mapping, and canonical
configuration are implementation-review bindings. They may be written after this preregistration
but may not be executed with an official seed until the sealed run.

## Frozen statistics and gates

For each seed/arm/checkpoint, report both risks. For arm `a`, seed `s`, and its domain-matched risk,

```text
A[a,s] = trapz_{t/120} log((R[a,s,t] + 1e-12) / (R[a,s,0] + 1e-12)).
```

No seed is excluded. The common pixel score across risk families remains descriptive.

Primary estimand:

```text
G_AUC_pixel = exp(mean_seed(A[pixel_gaussian,s] - A[pixel_uniform,s])).
```

Confirmatory secondary estimand:

```text
G_AUC_area = exp(mean_seed(A[area_gaussian,s] - A[area_uniform,s])).
```

Here `R_area` is exactly the four-offset-per-cell quadrature defined above. For domain pair `d`,
uniform arm `u`, mixture arm `m`, final step `T=120`, and `e=1e-12`, compute

```text
q_init[d,s]  = (R[u,s,T] + e) / (R[u,s,0] + e)
G_init[d]    = exp(mean_s log(q_init[d,s]))
q_final[d,s] = (R[m,s,T] + e) / (R[u,s,T] + e)
G_final[d]   = exp(mean_s log(q_final[d,s]))
delta_A[d,s] = A[m,s] - A[u,s]
G_AUC[d]     = exp(mean_s delta_A[d,s]).
```

Label precedence is exact. A hard invariant failure makes the whole result `MECHANISM_FAIL`.
Otherwise a pair is `INCONCLUSIVE_TRAINER` if `G_init>0.90` or fewer than two seeds have
`q_init<1`. Otherwise it is `MATERIAL_SAMPLING_WIN` if `G_AUC<=0.95`, `G_final<=1.05`, at least
two seeds have `delta_A<0`, and `max_s q_final<=1.20`. Otherwise it is `NONINFERIOR` if
`G_AUC<=1.05`, `G_final<=1.05`, and `max_s q_final<=1.20`. Otherwise it is
`NEUTRAL_OR_NEGATIVE`. Report per-seed directions, active fractions, ESS per
attempt, wall time, peak RSS, all parameter-family motions, and the common exact pixel score for
all arms. A global `SAMPLING_WIN` requires material wins in both pairs plus every mechanism and
safety gate. These labels cannot change a default; they only prioritize a later density-enabled
experiment.

Any NaN/Inf, cardinality change, RGB/source-image access, missing/null resampling, importance over
`4.00001`, mismatched view schedule, official-seed access before the marker, source/prereg drift,
teacher mutation, or optimizer-step mismatch is a hard mechanism failure. No rerun, repaired
official seed, altered threshold, or replacement result is allowed.

## One-shot lifecycle and artifacts

The implementation must provide exactly these lifecycle operations with fixed artifact paths:

```text
.venv/bin/python benchmarks/compact_point_training.py seal
.venv/bin/python benchmarks/compact_point_training.py run
.venv/bin/python benchmarks/compact_point_training.py calibrated
```

The exact repository artifacts are:

```text
benchmarks/results/20260716_compact_point_training_PREREG_REVIEW.md
benchmarks/results/20260716_compact_point_training_IMPLEMENTATION_REVIEW.md
benchmarks/results/20260716_compact_point_training_SEAL.json
benchmarks/results/20260716_compact_point_training_ATTEMPT.json
benchmarks/results/20260716_compact_point_training_RAW.json
benchmarks/results/20260716_compact_point_training_RESULT.json
benchmarks/results/20260716_compact_point_training_AUDIT.md
```

`seal` is outcome-free and writes the seal exclusively after validating the final preregistration
hash, both reviews, source manifest, focused tests, and full verification. `run` revalidates the
seal and exact `.venv`/CPU environment, then creates and fsyncs the attempt marker exclusively
before importing/calling any official constructor or RNG. Each fresh worker revalidates marker,
seal, source, git diff, and environment before fixture construction. The parent writes RAW
exclusively, strict-reloads it, recomputes RESULT only from that reload, and binds the RAW and
marker SHA-256s. `status=PASS` means a complete protocol-valid result; the scientific label is a
separate `decision`, so a neutral/negative outcome remains valid. Any existing downstream path,
glob collision, or source drift causes refusal. Strict standard JSON forbids NaN/Infinity.
Exclusive creation uses `open('x')`, flush, and fsync. A caught failure after the marker but before
RAW commit writes best-effort RAW and RESULT artifacts with `status=FAIL`. If RAW with
`status=PASS` has already been exclusively committed, it is immutable evidence and must not be
rewritten or contradicted; a later caught failure writes a `status=FAIL` RESULT that binds the
committed RAW SHA-256, records `failure_phase='post_raw_commit'`, and contains no scientific
decision. Fault-injection tests cover both sides of this commit boundary. Interruption or failure
still consumes the experiment.

The official result must then receive an independent scientist audit under the repository's
results-audit skill. The audit binds preregistration, seal, marker, RAW, RESULT, source, and every
gate. Calibrated execution requires a PASS/QUALIFIED verdict containing the exact current hashes
and the literal line `CALIBRATED_INTEGRATION_AUTHORIZED: YES`; Markdown verdict text alone without
all bindings cannot authorize it.

`calibrated` is not decision-bearing and cannot alter the official result. Before the first RGB
decode it exclusively writes both
`runs/compact_point_training_20260716/CALIBRATED_PLAN.json` and
`runs/compact_point_training_20260716/CALIBRATED_ATTEMPT.json`, binding the audit authorization,
the following literal configuration, input hashes, source manifest, and all output paths. Existing
plan/attempt/downstream files consume or refuse the integration; there is no fallback.

The exact calibrated split reproduces the prior eight-view selection order
`(C0001,C0008,C0014,C0021,C0026,C0031,C0039,C1004)` with `test_every=8`: the first seven are the
only acquisition/initialization/training views and C1004 is held out until final PLY hashes are
frozen. Scene is `dataset/2025_03_07_stage_with_fabric/frame_00008`; source and camera resolution
are 5328x4608; downscale is one; calibration is `calibration_dome.json`; undistortion is enabled;
and masks are not loaded or used so every teacher fit window is the full canvas. An acquisition
spawn worker streams exactly one selected RGB at a time. It uses StructSplat on CUDA with
`FitConfig(n_gaussians=640,max_gaussians=640,iterations=100,backend='structsplat',
adaptive_density=False,growth_waves=1,relocate_fraction=0.0,structsplat_renderer='cuda_tiled',
lr=1e-2,grad_init_mix=0.7,row_chunk=64,log_every=20,convergence_patience=0,
convergence_tol=0.05,convergence_check_every=25,appearance_parameterization='weight_color_9p',
freeze_geometry=False)`, view seeds 0 through 6 in the named order, and
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`. Exact external StructSplat source/version
hashes are recorded. It saves only exact `.teacher.npz` fields plus camera records into a strict
`ReconstructionInputs` bundle with `points=None`, `point_visibility=None`, and `bounds_hint=None`.

The dataset path is not passed to the fresh trainer spawn worker. That worker denies PIL,
`SceneData`, calibrated/image loaders, and any open below `dataset/`; strict-loads the bundle; and
runs the same archive/index caps above. It initializes with the complete literal
`CompactCarveConfig(n_init_3d=835,candidate_multiplier=4,samples_per_ray=48,
query_batch_size=4096,query_component_chunk=256,max_query_pairs=1048576,tile_size=16,seed=75200,
bounds_scale=0.5,near=0.05,min_views=2,hull_fraction=0.85,coverage_scale=1.0,
coverage_threshold=0.40,color_std_sigma=0.20,min_score=0.05,peak_radius_steps=3.0,
init_opacity=0.1,sh_degree=0,max_anchor_rounds=8)`. It then runs only `pixel_gaussian`, seed
75201, 40 updates, 128 attempts, `eta=0.25`, checkpoints `(0,10,20,40)`,
`evaluate_checkpoint_risks=False`, extent from the frozen
initial-cloud fallback, tile size 16, outer/point chunk 32, Gaussian/query chunk 64, and all other
official optimizer/renderer controls above. Converted legacy `fits/*.npz` and the existing
RGB-backed 835-row PLY are forbidden as supervision or initialization.

The calibrated trainer must prove that no exhaustive checkpoint evaluator was called and record
the four detached checkpoint hashes/callback events. Before held-out decode, the parent
freezes/hashes initial/final PLYs and compact histories. A fresh evaluation child uses seed 75202
to draw 4,096 uniform pixel centers with replacement per training teacher for initial/final
self-distillation metrics; these bounded samples are the only calibrated training-risk diagnostic.
Only then may another child decode
and undistort C1004 RGB/mask; seed 75203 selects 4,096 uniform pixel centers with replacement and
reports unclamped all-sample and foreground RGB MSE/PSNR. These are integration diagnostics, not
official decisions. The viewer smoke command is exactly `.venv/bin/rtgs view --gaussians
runs/compact_point_training_20260716/gaussians.ply --initial
runs/compact_point_training_20260716/gaussians_init.ply --scene
dataset/2025_03_07_stage_with_fabric/frame_00008 --downscale 1 --max-images 8 --rasterizer gsplat
--device cuda --snapshot-dir runs/compact_point_training_20260716/viewer_snapshots --host 127.0.0.1
--port 8876 --no-open`. It is launched only after hashes/metrics freeze, polled, recorded, and may
then be relaunched unchanged for the user's live handoff.

The exact calibrated output namespace is:

```text
runs/compact_point_training_20260716/reconstruction_inputs/
runs/compact_point_training_20260716/teacher_acquisition.json
runs/compact_point_training_20260716/compact_training_raw.json
runs/compact_point_training_20260716/gaussians_init.ply
runs/compact_point_training_20260716/gaussians.ply
runs/compact_point_training_20260716/heldout_evaluation.json
runs/compact_point_training_20260716/viewer_smoke.json
runs/compact_point_training_20260716/viewer_snapshots/
runs/compact_point_training_20260716/calibrated_result.json
```

If full-resolution acquisition, compact initialization/training, held-out evaluation, gsplat, or
viewer integration is infeasible, calibrated result records a bounded FAIL. It may not downscale,
decode held-out early, use crop-only teachers, alter the split/config, or relabel a legacy fit.

## Frozen pre-implementation anchors

```text
c380f6ab921ca18b7947d7764bb49bc13bf80ec9091804475ca3c7c3d3dc2441  src/rtgs/core/observation2d.py
2f93b571760c61d8fce6ecc5bfcfe103ecbce2049d4c15c3c43c33132577376b  src/rtgs/data/reconstruction_inputs.py
252e66eda091a7b9a769155889e11a2ed3f905a5bdf984164e842820c11203f7  src/rtgs/render/point_base.py
f0648a20e357f28414337f55fe387d8f9a6b785a8eb53ac9600848790067645b  src/rtgs/render/torch_points.py
3bb73a2071ff3525c07c0d1a57387ecccb5b5f16a3cc18398091b2606752053f  src/rtgs/optim/trainer.py
d417a4a103ae7ea1e3f4a7799c2b709597014b8966acb0e72b2bd447a0ad0ba5  src/rtgs/core/gaussians3d.py
afc9d036ad1c037a5cb3eab7fd5b19f97d37d920f520cb5c51bf37f41f989916  benchmarks/results/20260716_point_rasterizer_parity_PREREG.md
1abbdec0fd0fb71a3aa746430ca7f84b08999476951eb5386852d804cbfd4d85  benchmarks/results/20260716_point_rasterizer_parity_RESULT.json
```

These are chronology anchors, not immutable implementation hashes. The seal binds the complete
post-implementation manifest. Any implementation change after sealing invalidates the run.
