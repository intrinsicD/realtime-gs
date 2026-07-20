# Preregistration: multiscale fixed-topology refinement

## Chronology, question, and scope

Frozen at `2026-07-16T01:30:51+02:00`, after a read-only source/literature design pass and
before any implementation, diagnostic render, schedule probe, fitted seed, initialization,
training step, timing sample, metric, or outcome for this experiment.

Pre-implementation executability amendment at `2026-07-16T01:37:36+02:00`, still before any
implementation, test of the new seam, official-seed preparation, render, schedule probe, training,
timing sample, metric, or outcome. The preceding text had SHA-256
`b2033bb9108c57fc8ff156c26791b9b3cfad0fb26c21975ffb6800be0e56bad2`. Independent review found
two outcome-neutral specification gaps. This amendment freezes exact held-out per-view arithmetic
and raw evidence, including the no-clamp convention, and requires a focused bit-exact
`step_controls=None` versus all-`(1,1)` equivalence test. It changes no seed, scene, fit, Carve
initialization, arm, schedule, renderer, optimizer, loss, checkpoint, metric family, threshold,
gate, interpretation, timing rule, or stopping decision.

Pre-implementation verification-binding clarification at `2026-07-16T01:46:09+02:00`, during
harness/toy-test implementation but before any seal, official-seed preparation, schedule probe,
arm, timing sample, metric, or outcome. The preceding document had SHA-256
`b188d8f84efd5a6aebdc6c5fcb349582eccd4b15cd58bcf7fb6d9f8eb82b2701`. It explicitly permits
the seal's already-required full verification to execute nonofficial toy render fixtures while
continuing to forbid the scientific preparation and arms. No scientific choice changes.

The question is whether a conservative half-to-full image-resolution continuation improves the
quality or deterministic rendering exposure of short refinement from the repository's current
production Carve initialization. Four arms separate three effects:

1. ordinary full-resolution optimization;
2. a blocked half-resolution camera/rendering stage followed by full resolution;
3. the same blocked low-frequency loss while retaining a full-resolution renderer; and
4. an interleaved half/full camera schedule with the same number of half/full optimization
   renders as the blocked camera arm.

This is a CPU synthetic, fixed-topology, degree-zero refinement experiment. It can establish only
quality and exact rasterized/loss pixel exposure under the frozen setup. Wall-clock measurements
are descriptive and **cannot** enter any success gate because the shared workstation may execute
other research processes and this protocol does not reserve or isolate CPU cores. In particular,
an exact reduction in rasterized pixels is not a measured speedup. The experiment cannot establish
real-scene transfer, CUDA/gsplat speed, density-control interaction, full-SH behavior, memory or
energy reduction, a production-default change, or an optimal resolution schedule.

There is no resolution, duration, filter, loss, seed, split, initialization, learning-rate, or
threshold sweep. A 12-pixel stage is deliberately excluded: the existing zero-padded 11x11 SSIM
window would make nearly every 12x12 pixel boundary-affected and would conflate scale continuation
with a severe change in SSIM boundary statistics. The only resolutions are 24x24 and 48x48.

## Literature and adaptation boundary

[DashGaussian](https://arxiv.org/abs/2503.18402) directly studies progressive rendering
resolution during optimization and couples it to primitive growth. It supports asking whether
resolution scheduling can remove redundant early computation, but it does not predict an outcome
for this repository's Carve initialization, 120-step CPU reference renderer, fixed primitive set,
or two-resolution schedule. This experiment adapts only the general resolution-continuation
question and implements none of DashGaussian's frequency-energy scheduler, primitive scheduler,
budget estimator, CUDA system, or claimed performance result.

[AsySplat](https://arxiv.org/abs/2607.10995), surfaced by the 2026-07-12 through 2026-07-16
Scholar Inbox digest, uses coarse geometry tokens and fine appearance tokens in a feed-forward
asymmetric architecture. That is only an analogy for a possible later parameter-specific loss
test. This protocol does **not** freeze parameter blocks, route different losses to geometry and
appearance, add branches, or claim to reproduce AsySplat. Geometry/appearance asymmetry requires
a separate preregistration after the present scale mechanism is resolved.

## Frozen environment and append-only chronology

- Execute with `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`, and `MKL_NUM_THREADS=4`.
- Use the repository `.venv`, Torch reference renderer, Torch intra-op threads `4`, and
  deterministic algorithms enabled. Record Python, Torch, NumPy, Pillow, platform, processor,
  logical CPU count, Torch thread counts, environment variables, git revision, and dirty diff.
- No optional StructSplat, gsplat, CUDA, depth-model, network, or external dataset import is
  permitted.
- Record `/proc/loadavg` when available, process wall time, process CPU time, Trainer native
  elapsed checkpoints, and arm order. These fields are descriptive only. There is no CPU-idleness
  preflight and execution must not start, stop, retry, select, or reinterpret an arm based on
  system load or observed timing.
- The official attempt marker is created atomically with exclusive creation before the first
  stage-1 fit, Carve lift, truth render, schedule probe, or arm execution. Once created, it is
  consumed even if the process is interrupted or an invariant fails.
- Never overwrite a seal, marker, result, review, or note. Any retry after a consumed marker needs
  a new outcome-free preregistration and namespace; it may repair only a documented failure.

## Frozen data, split, and production initialization

Seeds are exactly `3,4,5`, chosen before implementation because seeds `0,1,2` have been reused in
the preceding synthetic experiment series.

For each seed, call exactly:

`make_synthetic_scene(n_gaussians=40, n_cameras=12, image_size=48, seed=seed)`.

Freeze original training indices `[0,1,2,4,5,6,8,9,10]` and held-out indices `[3,7,11]`.
Physically subset the full scene to the nine training views before fitting or lifting. Stage-1
fitting, Carve, image pyramids, optimization losses, native Trainer evaluation, and sampled-view
schedules receive only this training subset. Held-out RGB/cameras and synthetic truth are used
only after the common initialization and every arm schedule have been frozen, solely by the
read-only checkpoint evaluator described below. No held-out value may enter initialization,
downsampling, training, stopping, selection, or a threshold.

Fit the nine training images exactly once per seed and share detached immutable fits across the
single Carve lift using the current native production configuration:

```text
FitConfig(
  n_gaussians=150, max_gaussians=5000, iterations=120,
  backend="native", adaptive_density=True, growth_waves=5,
  relocate_fraction=0.0, structsplat_renderer="auto", lr=0.01,
  grad_init_mix=0.7, row_chunk=64, log_every=50,
  convergence_patience=0, convergence_tol=0.05,
  convergence_check_every=25,
)
```

Call `fit_views(train_scene.images, config, seed=seed, masks=train_scene.masks)` exactly once.
The synthetic scene has no masks; require `train_scene.masks is None`. Hash every training image,
camera, fitted tensor, fit history, local-to-original view map, sparse points, bounds, and their
aggregate before lifting.

Create exactly one current production Carve initialization per seed:

```text
CarveLifter(
  grid_res=48, bounds_scale=0.5, min_views=2, hull_fraction=0.85,
  color_std_sigma=0.20, color_match_sigma=0.35, coverage_thresh=0.40,
  samples_per_ray=64, min_score=0.05, min_weight=0.05,
  merge=True, merge_voxel_scale=1.0, init_opacity=0.1, sh_degree=0,
)
```

Call `lift(fitted, train_scene)` exactly once. No unmerged raw tensor, exact-count arm, pruning,
second lift, relift, post-lift filtering, or density operation is permitted. Require a nonempty
finite degree-zero result and hash every initialization field. Every refinement arm starts from a
fresh detached clone with an exactly matching field hash and primitive count. The result of the
separate Carve merge-controls experiment cannot select, replace, or modify this initialization.

## Frozen image pyramid and camera convention

Let `D_1(x)=x`. For any even `(H,W,...)` float32 tensor, define the only nontrivial downsample as
non-overlapping 2x2 area averaging:

`D_2(x)[j,k] = (1/4) * sum_{a=0..1,b=0..1} x[2j+a,2k+b]`.

The implementation uses `torch.nn.functional.avg_pool2d` with `kernel_size=2`, `stride=2`, no
padding, and the necessary channel-first/channel-last permutation. A separate float64 direct 2x2
sum must agree within `atol=1e-7, rtol=1e-6`. No bilinear, bicubic, Lanczos, Gaussian, Fourier,
adaptive, antialiased-resize, threshold, clamp, or post-pooling renormalization is allowed.

For a repository `Camera(fx,fy,cx,cy,W,H,R,t)`, define:

`C_2 = Camera(fx/2, fy/2, cx/2, cy/2, W/2, H/2, R, t)`.

Widths and heights must be even. There is no added or subtracted half-pixel offset. Repository
coordinates place the top-left pixel center at `(0.5,0.5)`, so low pixel center `u'=j+0.5` maps
to full coordinate `2u'=2j+1`, exactly the mean center of full pixels `2j+0.5` and `2j+1.5`.
Before any training, deterministic tests and runtime assertions must establish for finite test
points/rays:

- `project(C_2, X) == project(C_1, X)/2` within `atol=1e-6, rtol=1e-6`;
- `pixel_rays(C_2, uv) == pixel_rays(C_1, 2*uv)` within the same tolerance;
- `C_2` is exactly 24x24 and `D_2(image)` matches it for all nine training views; and
- extrinsics, camera positions, points, bounds, and view order are unchanged.

The two low-frequency arms are intentionally not forward-equivalent. With the reference
renderer's fixed screen dilation `delta=0.3`, a half camera gives

`Sigma_half = (1/4) * J_full Sigma J_full^T + 0.3 I`.

Mapped back to full-pixel units, the dilation is `1.2 I`, whereas rendering full resolution and
pooling retains the renderer's `0.3 I` before the box filter. Compact support, coarse visibility,
and front-to-back alpha compositing are also nonlinear under downsampling. These are mechanisms to
be distinguished by the frozen arms, not implementation differences to repair after an outcome.

## Frozen Trainer extension and persistent-state requirement

The research seam is an immutable keyword-only sequence supplied to `Trainer.train`:

```text
TrainStepControl(render_downscale: int = 1, loss_downscale: int = 1)
step_controls: Sequence[TrainStepControl] | None = None
```

The entire length-120 sequence must be materialized, validated, and hashed before optimization.
`render_downscale` and `loss_downscale` can only be `1` or `2` here; require
`loss_downscale % render_downscale == 0`, exact divisibility of every image dimension, and no
density control. Controls cannot read a render, loss, gradient, parameter, metric, elapsed time,
or system load.

`step_controls=None` is the exact established Trainer path and is used by the full-resolution
baseline. It must not construct a pyramid or change the established render/loss/optimizer/RNG
sequence. The candidate arms run through exactly one `Trainer.train` call each, with one persistent
set of parameter tensors and Adam optimizers. Resolution transitions must not recreate Trainer,
parameters, Adam moments, RNG generators, means-learning-rate decay, or any other optimization
state. Chaining multiple Trainer calls is forbidden.

For a step with render downscale `d_r` and loss downscale `d_l`:

1. render the current Gaussian set with `C_d_r`;
2. if `d_l/d_r=2`, apply `D_2` to the raw rendered RGB; otherwise keep it unchanged;
3. obtain the target directly as `D_d_l(original_training_image)`; and
4. compute the ordinary unmasked loss on those matching tensors:
   `0.8 * mean(abs(pred-target)) + 0.2 * (1-ssim(pred,target,window_size=11))`.

No alpha loss, mask crop, random background, scale/opacity regularizer, alternate SSIM window,
per-scale loss weight, gradient rescaling, learning-rate rescaling, parameter freezing, or
geometry/appearance routing is permitted. Every parameter group updates on every step.

Pyramid construction happens before the Trainer native timer and its duration is recorded
separately. Native checkpoint evaluation always renders the unscaled 48x48 training scene and is
identical across arms. The held-out callback runs after native elapsed time is recorded and is
therefore excluded from Trainer native elapsed, as in the existing callback contract.

## Frozen arms and counterbalancing

All arms use exactly 120 optimization updates:

1. `full`: pass `step_controls=None`; every step has `(d_r,d_l)=(1,1)`.
2. `camera_blocked`: steps 1-60 use `(2,2)`; steps 61-120 use `(1,1)`.
3. `pyramid_blocked`: steps 1-60 use `(1,2)`; steps 61-120 use `(1,1)`.
4. `camera_interleaved`: odd steps use `(2,2)` and even steps use `(1,1)`. It therefore has
   exactly 60 half and 60 full renders and ends on a full-resolution step.

Freeze arm execution order to reduce monotone process-order imbalance while making no timing
claim:

- seed 3: `full`, `camera_blocked`, `pyramid_blocked`, `camera_interleaved`;
- seed 4: `camera_blocked`, `pyramid_blocked`, `camera_interleaved`, `full`;
- seed 5: `pyramid_blocked`, `camera_interleaved`, `full`, `camera_blocked`.

Each arm receives `TrainConfig(seed=seed)` and a fresh initialization clone. The Trainer's local
generator must produce the same 120 local training-view indices in every arm of a seed. Before the
first arm, an isolated generator probe performs exactly 120
`torch.randint(0,9,(1,),generator=Generator().manual_seed(seed))` draws without fitting,
rendering, or training. Hash this schedule and require every official history to reproduce it
exactly. Step controls must consume no RNG. Report per-view counts assigned to half and full steps;
these are diagnostics and cannot invalidate or select a seed after the frozen sequence is known.

## Frozen refinement configuration

Every arm uses exactly:

```text
TrainConfig(
  iterations=120,
  lr_means=1.6e-4, lr_quats=1e-3, lr_scales=5e-3,
  lr_opacity=5e-2, lr_sh=2.5e-3, lr_sh_rest=1.25e-4,
  ssim_lambda=0.2,
  rasterizer="torch", device="cpu",
  densify=False, density_strategy="classic",
  eval_every=30,
  target_sh_degree=0, sh_degree_interval=120,
  use_masks=False, outside_alpha_lambda=0.01,
  mask_alpha_lambda=0.05, random_background=False,
  opacity_reg=None, scale_reg=None,
  packed=False, antialiased=False,
  sh_color_activation="hard",
  collect_sh_color_diagnostics=False,
  kernel_support_mode="hard",
  collect_kernel_support_diagnostics=False,
  visibility_margin_sigma=3.0,
  validate_render_finite=True,
  seed=seed,
)
```

The unused mask/regularizer fields are serialized to bind the complete established loss
configuration. Require active SH degree zero at every checkpoint and exactly constant primitive
count throughout.

## Frozen deterministic pixel-exposure accounting

Optimization render exposure counts spatial pixels passed to the renderer, not RGB scalars,
visible primitive-pixel pairs, evaluation renders, or setup. With one sampled training view per
step:

- `full`: `120 * 48 * 48 = 276480` render pixels and `276480` loss pixels;
- `camera_blocked`: `60 * 24 * 24 + 60 * 48 * 48 = 172800` render and loss pixels;
- `pyramid_blocked`: `276480` render pixels and `172800` loss pixels;
- `camera_interleaved`: `172800` render and loss pixels.

Thus each camera schedule has exact optimization-render ratio `172800/276480 = 0.625`, a 37.5%
reduction. `pyramid_blocked` has no render-exposure reduction. Native Trainer evaluation adds four
common checkpoints times nine training views times `48*48 = 82944` render pixels to every arm;
the held-out observer and manual step-zero evaluation are reported separately and cannot enter the
optimization-exposure ratio. Runtime must independently recompute and exactly match all counts.

This accounting can authorize an **exposure-efficiency** classification only when paired with the
quality noninferiority gate below. It can never be called wall-clock, FLOP, memory, energy, or
hardware speedup evidence.

## Frozen held-out checkpoint evaluation

Construct immutable held-out truth records only after fitting, Carve initialization, arm schedules,
and source hashes are frozen. Render the full scene's degree-zero GT Gaussians with the common
48x48 hard Torch renderer and black background. For each held-out view, define truth support as
`truth.alpha > 0.05` and expected truth depth as
`truth.depth / clamp_min(truth.alpha,1e-6)`. Hash RGB, camera, truth color/alpha/depth/support, and
their aggregate. No truth field enters training.

Evaluate the common initialization manually at step 0. At completed steps 30, 60, 90, and 120,
the existing read-only `checkpoint_callback` receives an isolated detached clone after the native
Trainer checkpoint and elapsed-time record. It must evaluate all three held-out views at the
original 48x48 resolution with `TorchRasterizer`, hard SH color, hard support, visibility margin
3.0, degree zero, and black background. Callback work is excluded from native elapsed timing and
cannot mutate training.

### Exact per-view formulas and raw evidence

For one held-out view, let raw float32 target RGB be `I` and the raw renderer fields be `P` (RGB),
`A` (accumulated alpha), and `Z` (alpha-weighted accumulated depth). Require shapes `(H,W,3)`,
`(H,W)`, and `(H,W)` with `(H,W)=(48,48)`, exactly three color channels, and finite values. Do
**not** clamp, clip, tone-map, normalize, or otherwise alter `P`, `I`, `A`, or `Z` for any metric.
The only numerical floor in a color metric is the explicit PSNR MSE floor below.

Let raw GT-render fields be `A_gt,Z_gt`; require both to be finite float32 tensors of shape
`(H,W)`. Define:

- truth support `S = (A_gt > 0.05)`;
- predicted support `Q = (A > 0.05)`;
- intersection `J = S & Q`; and
- union `U = S | Q`.

Require `sum(S)>0`, `sum(J)>0`, `sum(U)>0`, and finite positive scene `extent`. Boolean masks are
cast to float64 only when multiplying a numeric residual. All squared-error subtractions below
cast each operand to float64 **before** subtraction.

Foreground RGB evidence and PSNR are:

```text
foreground_rgb_squared_error_sum =
    sum_yxc S[y,x] * (double(P[y,x,c]) - double(I[y,x,c]))^2
foreground_rgb_value_count = 3 * sum_yx S[y,x]
foreground_mse = foreground_rgb_squared_error_sum / foreground_rgb_value_count
psnr_fg = -10 * log10(max(foreground_mse, 1e-12))
```

Full-canvas RGB evidence and PSNR are:

```text
full_rgb_squared_error_sum = sum_yxc (double(P[y,x,c]) - double(I[y,x,c]))^2
full_rgb_value_count = H * W * 3
full_mse = full_rgb_squared_error_sum / full_rgb_value_count
psnr_full = -10 * log10(max(full_mse, 1e-12))
```

For crop metrics, freeze the repository `masked_crop` spatial convention without its implicit
choice of image. Find the inclusive extrema of `where(S)`, set
`margin=max(1,round(max(H,W)*0.05))` using Python's `round`, and clamp the half-open bounds to the
canvas exactly as `masked_crop` does. Form raw float32 masked crops using the same bounds:

```text
P_crop = (P * float32(S)[...,None])[y0:y1,x0:x1]
I_crop = (I * float32(S)[...,None])[y0:y1,x0:x1]
crop_rgb_squared_error_sum = sum_yxc (double(P_crop)-double(I_crop))^2
crop_rgb_value_count = (y1-y0) * (x1-x0) * 3
crop_mse = crop_rgb_squared_error_sum / crop_rgb_value_count
psnr_crop = -10 * log10(max(crop_mse, 1e-12))
ssim_crop = ssim(P_crop,I_crop,window_size=11)
```

Require nonempty crop dimensions. `ssim_crop` is the existing float32 repository SSIM with its
11x11 sigma-1.5 Gaussian window, zero padding, `c1=0.01^2`, and `c2=0.03^2`; no output clamp or
extra crop normalization is allowed.

Expected-depth evidence and normalized RMSE are:

```text
D = Z / clamp_min(A,1e-6)
D_gt = Z_gt / clamp_min(A_gt,1e-6)
depth_squared_error_sum = sum_yx J[y,x] * (double(D[y,x])-double(D_gt[y,x]))^2
depth_intersection_pixel_count = sum_yx J[y,x]
depth_rmse_over_extent =
    sqrt(depth_squared_error_sum / depth_intersection_pixel_count) / extent
```

Alpha intersection-over-union and truth-foreground coverage are:

```text
alpha_intersection_pixel_count = sum_yx J[y,x]
alpha_union_pixel_count = sum_yx U[y,x]
truth_support_pixel_count = sum_yx S[y,x]
alpha_iou = alpha_intersection_pixel_count / alpha_union_pixel_count
foreground_coverage = alpha_intersection_pixel_count / truth_support_pixel_count
```

Serialize the exact named raw sums/counts above, crop bounds, support/intersection/union hashes,
per-view metric values, renderer-field hashes, truth hash, and primitive count. Recompute every
metric from those raw fields before accepting the record. A zero/nonfinite denominator, nonfinite
metric, shape/channel mismatch, or mismatch between raw evidence and a reported metric invalidates
the official result rather than receiving a favorable sentinel.

Within a seed, every decision metric is the arithmetic mean of the three named per-view values;
across seeds, every reported mean or mean delta is the arithmetic mean of the three paired seed
values. Raw sums are retained for audit but are not pooled across views or seeds to form a
different decision metric. For metric `m` at checkpoint steps `T=(0,30,60,90,120)`, define the
step-normalized trapezoidal AUC exactly as:

`AUC(m) = (1/120) * sum_j ((T[j+1]-T[j]) * (m[j]+m[j+1]) / 2)`.

The primary is AUC of mean held-out foreground PSNR in dB. Do not choose checkpoints, interpolate
extra checkpoints, smooth curves, or select a best intermediate model.

## Frozen decisions and safety gates

All comparisons are paired by seed. For arm `a`, define deltas as `a-full`; higher is better for
PSNR, SSIM, IoU, and coverage. For depth, define relative regression from the across-seed means as
`(mean(depth_a)-mean(depth_full))/mean(depth_full)`, requiring a finite positive baseline.

### Common quality noninferiority

An arm is quality-noninferior to `full` only if all are true:

1. mean final foreground-PSNR delta is at least `-0.05 dB`;
2. every seed's final foreground-PSNR delta is at least `-0.15 dB`;
3. mean and minimum per-seed final crop-SSIM deltas are at least `-0.002` and `-0.005`;
4. relative regression of the across-seed mean final depth RMSE is at most `0.02`;
5. across-seed mean final alpha-IoU delta is at least `-0.02`; and
6. across-seed mean final truth-foreground-coverage delta is at least `-0.02`.

Guardrails cannot rescue a failed primary quality-improvement gate; they only prevent a primary
or exposure classification from hiding geometry/coverage damage.

### Quality improvement

An arm has a held-out quality improvement over `full` only if it is quality-noninferior and:

1. mean foreground-PSNR AUC delta is at least `+0.10 dB`;
2. at least two of three seeds have strictly positive AUC delta;
3. mean final foreground-PSNR delta is nonnegative; and
4. at least two of three seeds have nonnegative final foreground-PSNR delta.

Report all signed deltas and wins even when the gate fails. The primary named candidate is
`camera_blocked`; `pyramid_blocked` and `camera_interleaved` are mechanism controls with their own
identically computed gates.

### Exposure efficiency

`camera_blocked` or `camera_interleaved` has an exposure-efficiency result only if its schedule and
raw exposure counts exactly reproduce the frozen `0.625` ratio and it is quality-noninferior to
`full`. This classification means only 37.5% fewer optimization pixels were rasterized without a
material frozen-metric regression. It is not a runtime-speed result.

### Blocked-order attribution

The blocked ordering is material relative to the equal-exposure interleaved control only if:

1. mean `camera_blocked-camera_interleaved` foreground-PSNR AUC is at least `+0.05 dB`;
2. `camera_blocked` has strictly greater AUC in at least two of three seeds;
3. its mean final foreground-PSNR delta versus interleaved is at least `-0.05 dB` and every seed
   is at least `-0.15 dB`; and
4. both camera arms pass common quality noninferiority versus `full`.

If this fails, no coarse-to-fine or continuation-order claim is allowed. An exposure-efficient
camera result may be described only as mixed-resolution exposure reduction.

### Frozen mechanism classification

Use the quality-improvement booleans, never the larger observed number or timing:

- both `camera_blocked` and `pyramid_blocked` pass: a low-frequency curriculum is a plausible
  shared mechanism, while renderer-specific effects remain unresolved;
- only `camera_blocked` passes: attribute the tested benefit to renderer-scale/EWA/visibility or
  another forward-render difference, not to low-pass supervision alone;
- only `pyramid_blocked` passes: low-frequency supervision helps under this setup but the tested
  low-resolution renderer does not;
- neither passes: there is no quality-improvement result for this 24-to-48 schedule.

This classification is an inference from controls, not a proof of a unique causal mechanism.
Regardless of quality, wall time remains descriptive.

## Validity, implementation review, seal, and official run

Before sealing, focused tests must cover:

- omitted versus explicit `step_controls=None` bit-exact final fields and histories excluding
  elapsed time;
- established `step_controls=None` versus a length-`iterations` immutable sequence containing
  only `TrainStepControl(1,1)` bit-exact final fields and all established history fields excluding
  elapsed time and newly added control-metadata fields; this comparison must include loss,
  loss-term, sampled-view, PSNR, primitive-count, active-SH, and final Gaussian tensors, so the
  nondefault seam is proven equivalent when it requests established full-resolution behavior;
- area-pool arithmetic, shape, dtype/device, and direct float64 parity;
- camera projection, ray, extrinsic, dimension, and pixel-center invariants;
- exact schedules, 60/60 counts, last-step convention, pixel exposure, and loss exposure;
- unchanged per-seed sampled-view schedule across all four arms;
- one persistent optimizer/parameter identity across resolution boundaries;
- full-resolution native and held-out evaluation despite low-resolution training steps;
- constant degree-zero SH and primitive count;
- callback detachment/isolation and exclusion from native elapsed time;
- rejection of wrong sequence length, unsupported scale, nondivisible dimensions,
  `loss_downscale < render_downscale`, and non-unit schedules with density enabled; and
- tampering with schedules, source hashes, result numerators, AUC, decisions, seal, or marker.

Focused tests may use analytic tensors, toy cameras, and tiny nonofficial synthetic seeds, but may
not fit, lift, or refine seeds `3,4,5` with the official configuration. Test execution is not a
pilot and must expose no official arm metric before the attempt marker.

Run the complete repository verification command before seal creation. A reviewer who did not
author the implementation must compare the harness and tests line-by-line with this protocol and
write `benchmarks/results/20260716_multiscale_refinement_IMPLEMENTATION_REVIEW.md` with an explicit
`pass` or `fail` verdict and unresolved findings. Only `pass` permits sealing.

The immutable seal is
`benchmarks/results/20260716_multiscale_refinement_SEAL.json`, artifact type
`multiscale_refinement_implementation_seal`. It binds this preregistration, implementation review,
harness, focused tests, all repository-owned loaded source and tests, `pyproject.toml`, environment,
git revision, dirty diff/patch, verification commands/results, and a canonical source aggregate.
Seal creation must not fit, lift, render a scene, probe the official view schedule, execute an arm,
or expose a metric.

For clarity, seal creation itself must run, in order, `.venv/bin/python -m ruff check .`,
`.venv/bin/python -m ruff format --check .`, `.venv/bin/python -m pytest -q -m "not slow"`,
`.venv/bin/python scripts/docs_sync.py`, and `git diff --check`, refuse on any nonzero exit, and
record each literal command, exit status, complete stdout/stderr, and output SHA-256. Those
verification children may construct/render toy fixtures allowed by the focused-test rules. They
may not invoke the harness's `run` action, prepare seeds 3/4/5 under the official configuration,
claim the attempt marker, execute an official arm, or expose an official metric. The preceding
seal-creation prohibition applies to official scientific preparation, not bounded toy unit tests.

The atomic once-only marker is
`benchmarks/results/20260716_multiscale_refinement_ATTEMPT.json`. The result is
`benchmarks/results/<UTC>_cpu_multiscale_refinement.json`, artifact type
`multiscale_refinement_ablation`, with a companion `_RESULT.md` generated only from that JSON.

Expected commands are:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python benchmarks/multiscale_refinement_ablation.py seal \
  --output benchmarks/results/20260716_multiscale_refinement_SEAL.json

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python benchmarks/multiscale_refinement_ablation.py run \
  --seal benchmarks/results/20260716_multiscale_refinement_SEAL.json \
  --output benchmarks/results/<UTC>_cpu_multiscale_refinement.json
```

At runtime, verify the exact seal/source aggregate and absence of the marker/output before atomic
marker creation. After the marker, any nonfinite field, source drift, invariant failure, exception,
or interrupted run consumes the attempt. If possible, write a fail-closed invalid artifact with
only preparation/validity evidence and no success classification. Never resume from a partial fit,
initialization, arm, cache, checkpoint, or in-memory result.

## Independent results audit and stopping rules

Before any number or capability statement enters `README.md`, `docs/`, ARA crystallized layers, a
configuration default, or a follow-up selection, a scientist who did not implement or execute the
official run must apply the `realtime-gs-results-audit` skill. The audit must independently verify:

- chronology, marker, seal, source aggregate, environment, and exact result binding;
- raw per-view checkpoint numerators and arithmetic per-view/seed mean recomputation;
- AUC, signed deltas, seed wins, exposure counts, and every decision boolean;
- arm order, initialization equality, sampled-view equality, fixed topology, SH degree, and
  held-out isolation; and
- that descriptive timing did not enter a success gate or become a speed claim.

The audit is saved adjacent to the result as `_AUDIT.md` with its SHA-256 recorded in any later
documentation. A failed or qualified audit forbids a quantitative claim until repaired by a new
append-only protocol where necessary.

After an audited result:

- do not tune the 24x24 scale, 60/60 boundary, pool filter, SSIM window, loss weight, learning
  rate, seed, or gate on these outcomes;
- do not automatically try 12x12, a third scale, a smoother schedule, or a chosen per-seed arm;
- if neither blocked arm has a quality improvement and neither camera arm has an
  exposure-efficiency result, close this exact fixed-topology 24-to-48 branch on the tested
  synthetic setup;
- an exposure-efficiency result without a quality improvement can motivate only a separately
  preregistered, idle-machine or GPU runtime confirmation; it does not change defaults;
- a quality result requires fresh real-scene and density/full-SH confirmation before any production
  recommendation; and
- geometry-coarse/appearance-fine gradient routing remains a distinct, separately preregistered
  experiment and cannot be inferred or launched conditionally inside this attempt.
