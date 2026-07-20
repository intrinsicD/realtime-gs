# Preregistration: Stage-1 fit-time appearance parameterization

## Chronology, question, and scope

Frozen at `2026-07-16T03:56:08+02:00`, before implementation, a pilot, construction or fitting
of an official seed, creation of an official arm, an attempt marker, or any outcome for this
experiment.

The native Stage-1 fitter currently gives each 2D Gaussian five learned geometry values and four
learned appearance values. For raw scalar `s` and raw RGB vector `u`, the renderer observes only

`a = sigmoid(s) * sigmoid(u)`.

Thus the current nine-parameter component maps four appearance coordinates to a three-channel
additive amplitude. This experiment asks whether replacing that appearance map by the identifiable
three-coordinate map

`a = sigmoid(r), weight = 1`

improves fit-time conditioning and preserves or improves fixed-budget Stage-1 source-image
reconstruction. It contains two orthogonal blocks in one predetermined run:

1. an **appearance-only mechanism block**, in which the common initialized geometry is frozen and
   only appearance is optimized; and
2. a **joint Stage-1 utility block**, in which geometry and appearance are optimized together
   through the ordinary native fitting path.

Both blocks compare exactly two arms at a fixed primitive count and from a common step-zero
amplitude and geometry. No result from the mechanism block selects, tunes, stops, or changes the
joint block. This is a CPU synthetic Stage-1 experiment only. It performs no lift, merge,
retention, coverage interpretation, Stage-3 refinement, held-out novel-view evaluation, CUDA/
gsplat work, runtime comparison, memory comparison, compression evaluation, or default change.

The source facts used to freeze this protocol were bound at the chronology checkpoint by:

- `src/rtgs/image2gs/fit.py`:
  `2a9b76d41e83cc444fa98b3a0f3aa45eb8b6032806fa3d899377acfd98257e18`;
- `src/rtgs/image2gs/renderer2d.py`:
  `d0bd6b90b8a690a2ebb36cbc55c8cceb56c3fc33c04fd3895a123e0abb660144`;
- `src/rtgs/core/gaussians2d.py`:
  `390c6940bea8f4f1c80df19396a38ee29585dfd3127c8a3823654ffe09098351`;
- `src/rtgs/data/synthetic.py`:
  `b2b16f02a92c89003439062085e39d1f5ced2cc9ebaf5b8874cf80c0fd4d70b2`;
- `src/rtgs/core/metrics.py`:
  `d489c07c65ac4c74f0f927d41c62b887724cf3216f2ef28a116ff169d08272d4`;
- `src/rtgs/render/torch_ref.py`:
  `61716787329e85a186982f81c2a89cb270255473ca26688c409191a1b53bd86e`;
- `src/rtgs/core/camera.py`:
  `1e6a42c7cd9fa14b2ffff19808e6e88c106df4562d30fc18b0ca107c00072ac2`;
- `src/rtgs/core/gaussians3d.py`:
  `d417a4a103ae7ea1e3f4a7799c2b709597014b8966acb0e72b2bd447a0ad0ba5`;
- `src/rtgs/core/sh.py`:
  `554f3a25e25c7312248a98c15685e9bf805c85a81a96f56e13e1481619eb4687`;
- `src/rtgs/data/scene.py`:
  `3fa557f03bab5eb7666476968e0a70ff3e5639d6e24251807905691df36004c3`;
- `src/rtgs/render/base.py`:
  `1175cf359e2800ff3a518849b43c4d9a6fd6dccc3dfb7c24459f13e9f81ca0b9`.

Later implementation must preserve this initial source snapshot in the implementation review and
prove exact current-path parity. Only the additive `fit.py` implementation change named below may
alter one of these frozen files; any other drift requires a fresh preregistration. The future seal
binds the complete executed source.

### Pre-seal implementation-feasibility amendment

Amended at `2026-07-16T10:38:57+02:00`, before an implementation PASS, seal, attempt marker,
official scene construction, official initialization, official fit, or official result. No
official seed (`7727`, `8837`, `9941`, `10007`, `11003`, or `12007`) was constructed, rendered,
fit, or inspected. An independent adversarial implementation check used only view zero from six
disjoint development seeds (`424242`, `314159`, `271828`, `161803`, `8675309`, and `123456`) to
test whether the callback identities were numerically feasible. It did not inspect PSNR, a
scientific decision, or any official artifact.

That check found that replaying the production raw-coordinate-to-render-to-MSE graph reproduces
the production appearance gradients bit-for-bit, while deriving the same chain rule through the
independent direct-amplitude probe introduces sub-nanounit float32 differences because the two
renders associate multiplications differently. All six development seeds exceeded the original
relative-only chain-rule limit of `2e-5` (observed range `3.04e-5` to `6.03e-5`) while remaining
far below its unchanged absolute limit (observed maximum absolute-error range `2.18e-10` to
`8.15e-10`). This is an implementation-feasibility calibration, not a scientific outcome.

Accordingly, at this first amendment only the appearance chain-rule maximum-relative-error limit
below was amended from `2e-5` to `1e-4`. Its `2e-6` absolute limit and `1e-8` derived-gradient
magnitude condition were otherwise unchanged. The direct-amplitude render/loss gates, Adam
reconstruction gates, null-orthogonality
gates, seeds, checkpoints, pooling, utility thresholds, direction rule, and all scientific
decisions remain frozen exactly as originally written. The preregistration hash bound by the
future implementation seal and attempt is the hash of this explicitly amended document.

A second implementation-feasibility amendment was made at `2026-07-16T10:45:22+02:00`, under the
same pre-seal and no-official-seed conditions. A first bounded rerun with the `1e-4` limit failed
at mechanism row 42 with maximum absolute residual `8.1490725e-10` but maximum relative residual
`1.4473908e-4`. A complete 120-update callback calibration on view zero of the same six
development seeds, with only the first-amended relative gate relaxed in that process, then
completed every other recorder check. Across the six runs, the maximum absolute
chain-rule residual was `1.1641532e-9` to `2.3283064e-9`. The maximum relative residual was
`3.1231859e-4` to `1.8310522e-3` only because the original relative population included derived
gradients immediately above `1e-8`; the global worst had derived magnitude `1.1121074e-8` and
absolute residual `2.0363267e-11`. When relative error was evaluated only where derived magnitude
exceeded `1e-5`, the six maxima were `2.3847385e-6` to `7.3362103e-6`.

The appearance chain-rule relative-population threshold below is therefore amended from
`|derived|>1e-8` to `|derived|>1e-5`. The maximum absolute error remains required over every
coordinate, including smaller derived gradients, at the unchanged `2e-6` limit; the relative
limit remains `1e-4`. This avoids making relative roundoff at near-zero coordinates a protocol
gate while retaining both an all-coordinate absolute check and a wide-margin proportional check.
No PSNR, scientific decision, official seed, or official artifact was inspected. Every other
frozen item listed in the first amendment remains unchanged.

## Prior evidence and literature boundary

The completed Stage-1 weight/color gauge audit and its independent `QUALIFIED` review are
background evidence only:

- `benchmarks/results/20260716T003140Z_cpu_stage1_weight_gauge_audit.json`, SHA-256
  `e001d6efdfcf0beea30ae578069d6057350e47b3f3516ad95f216ae495793791`;
- `benchmarks/results/20260716T003140Z_cpu_stage1_weight_gauge_audit_AUDIT.md`, SHA-256
  `871c3235954f1025b05641385d70cd33c6160d200f74a26fb322dc20e390dfd6`.

That audit changed already-fitted `(weight,color)` representatives while keeping `weight*color`
fixed. It established a narrow downstream representation-contract problem, but it did not compare
fit-time parameterizations, optimization curves, or reconstruction utility. No numeric result or
threshold distance from that audit determines a seed, gate, or checkpoint here.

The frozen semantic-factorial protocol at
`benchmarks/results/20260716_stage1_semantic_factorial_PREREG.md` (SHA-256
`f53146f12894d5e804baf699b0ba0df51d5768ef708884f5a0343c523d96e1ce`) is also separate. It keeps
the current fit fixed and asks how scalar and color fields should be routed into Stage 2. This
experiment instead changes the fit-time coordinates, measures only Stage-1 pixels, and never
constructs `m=max(weight*color)`, sampled source-observation color, a factorial arm, or a lifter.
Neither protocol can gate, repair, or reinterpret the other.

The closest primary source is
[GaussianImage](https://arxiv.org/abs/2403.08551) and its
[official Cholesky implementation pinned at commit
`d53393b`](https://github.com/Xinjie-Q/GaussianImage/blob/d53393bee7c9fbb24e3510614e3ff2c85b8fbbc1/gaussianimage_cholesky.py).
The paper specifies eight parameters per Gaussian. The official code registers opacity as an
all-ones buffer and learns a three-vector feature, rather than this repository's extra learned
scalar. The same code exposes that feature directly and clamps the final image, so the primary arm
below is an **upstream-inspired control**, not a reproduction of upstream training.

The official raw/unbounded feature version is deliberately excluded. Adding it would jointly
change scalar redundancy, the feature activation, feasible per-component amplitudes, and clamp
interaction. It therefore cannot identify the effect asked here. A raw/unbounded arm would require
a separate preregistration with an explicit common-forward construction and out-of-range/clamp
guard; it may not be added after seeing this result.

[Adam](https://arxiv.org/abs/1412.6980) motivates recording the coordinatewise first and second
moments and the actual update, but it does not predict which arm wins. The 2026-07-12 through
2026-07-16 Scholar Inbox digest offers only distant analogies: AsySplat separates geometry and
appearance computation, SalientGS treats allocation separately from appearance error, and
Grassmannian Splatting uses a constrained representation to encode an intended geometric family.
None studies this repository's 2D product gauge, bounded amplitude arm, optimizer, loss, or gates.
No external loss, architecture, or result is imported.

This experiment is also unrelated to the Smooth Maximum Unit. There is no `max` in either fit-time
appearance map: both use sigmoid, the renderer keeps its established hard spatial support, and the
loss is unchanged. The semantic-factorial protocol's detached `max(weight*color)` and the earlier
SMU color-floor audit cannot support or refute this intervention.

## Frozen arms and local differential hypothesis

The two arm names and their complete appearance definitions are:

| arm | learned appearance raw values per component | built `weight` | built `color` | learned total |
|---|---:|---|---|---:|
| `weight_color_9p` | `s in R`, `u in R^3` | `sigmoid(s)` | `sigmoid(u)` | 9/component |
| `unit_weight_bounded_8p` | `r in R^3` | exact one, no gradient/state | `sigmoid(r)` | 8/component |

With finite raw coordinates, both arms cover the same open bounded amplitude family `(0,1)^3`.
The intervention changes its coordinates and removes one redundant degree of freedom; it does not
give the candidate a larger per-component amplitude range.

The five learned geometry values are identical in definition: two raw center coordinates, two raw
positive Cholesky diagonals, and one unconstrained Cholesky off-diagonal. The candidate is native-
backend only and opt-in. The default and explicit current setting remain `weight_color_9p`;
StructSplat must reject the candidate before importing or constructing that backend.

For current built values `w=sigmoid(s)` and `c=sigmoid(u)`, the float64 local Jacobian of amplitude
with respect to raw appearance coordinates, ordered `(u_0,u_1,u_2,s)`, is

`J_current = [diag(w*c*(1-c)) | w*(1-w)*c]`, with shape `3 x 4`.

At an interior point it has a one-dimensional right nullspace. One analytic null vector is
proportional to

`(-(1-w)/(1-c_0), -(1-w)/(1-c_1), -(1-w)/(1-c_2), 1)`.

The implementation must form the Jacobian directly and also obtain the unit right-null vector from
a float64 SVD with `full_matrices=True`; sign is immaterial. It must report
`||J_current*n||_2`. The analytic vector and its absolute alignment with the SVD vector are reported
only when all three denominators `1-c_k` are strictly positive and the SVD numerical rank is three;
otherwise their numeric arrays carry `defined=false` and the SVD diagnostic remains authoritative.
Rows with saturated derivatives are retained and reported rather than removed; their numerical
rank may be below three.

For the candidate,

`J_candidate = diag(a*(1-a)), a=sigmoid(r)`, with shape `3 x 3`.

It has no structural fourth coordinate, although sigmoid saturation can still make it poorly
conditioned. Therefore the causal hypothesis is modest: removing the redundant coordinate may
improve the optimization curve, but it is not assumed to eliminate ordinary saturation,
component overlap, or geometry/appearance co-adaptation.

## Additive implementation and exact production-current parity

Future implementation may add only:

- one native appearance-parameterization control whose default is `weight_color_9p`;
- a disabled, benchmark-only appearance-freeze control and read-only diagnostic callbacks; and
- the benchmark harness and focused CPU tests.

No CLI default, `Gaussians2D` public field, renderer expression, initializer, loss, optimizer,
scheduler, mask behavior, StructSplat behavior, or downstream consumer may otherwise change.
Refactoring the native fitter into initialization and fit-from-initialization helpers is allowed
only if the default path remains bit-exact.

Before an implementation review can say `PASS`, focused nonofficial tests must compare the
post-change default and explicit `weight_color_9p` paths to a test-local legacy reference bound to
the preimplementation `fit.py` hash above. On deterministic tiny unmasked and masked inputs,
require bit-exact final `xy`, `chol`, `color`, `weight`, stopped iteration, checkpoint iterations,
PSNR values, and all other non-time history. Require the same initialization tensors, optimizer
parameter order, per-step learning rates, raw losses, and sampled RNG state. Tests must also prove:

- the candidate has exactly three learned appearance channels and no weight optimizer state;
- its built weight is bit-exact one at entry, every checkpoint, and exit;
- current and candidate update equations and gradient-chain identities;
- appearance-only mode leaves every geometry raw/built tensor bit-exact;
- callbacks receive detached clones and cannot mutate parameters, gradients, optimizer state, RNG,
  target, or schedule;
- invalid/non-finite raw rows fail closed; and
- candidate plus StructSplat is rejected before optional-backend import.

The official joint current arm must use the same code path exercised by production `fit_image`, not
a benchmark reimplementation. Timing is excluded from every parity assertion and scientific gate.

## Common CPU environment, data, and fixed fit

- CPU only: `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, Torch intra-op
  threads `4`, deterministic algorithms enabled, and no optional StructSplat or gsplat import.
- Appearance-only mechanism seeds are exactly `[7727,8837,9941]`.
- Joint utility seeds are exactly `[10007,11003,12007]`. Both sets are disjoint from each other,
  the prior gauge audit's `[0,1,2]`, and the semantic-factorial protocol's
  `[1103,2203,3301,4409,5519,6637]`.
- For each seed, call
  `make_synthetic_scene(n_gaussians=40,n_cameras=12,image_size=48,seed=seed)` exactly once.
  Physically subset original views `[0,1,2,4,5,6,8,9,10]` in that order. Original views
  `[3,7,11]`, depths, cameras, points, and `gt_gaussians` are discarded immediately and never
  subsequently read, hashed, scored, or passed to the fitter. Stage 1 has no cross-view model, so
  this protocol makes no held-out-view claim.
- Every selected image is unmasked and has shape `(48,48,3)`. Require values finite and in
  `[0,1]`. Hash and serialize every target and the local-to-original view map.
- Use exactly
  `FitConfig(n_gaussians=150,max_gaussians=5000,iterations=120,backend="native",
  adaptive_density=True,growth_waves=5,relocate_fraction=0.0,
  structsplat_renderer="auto",lr=0.01,grad_init_mix=0.7,row_chunk=64,log_every=50,
  convergence_patience=0,convergence_tol=0.05,convergence_check_every=25)` plus only the named
  appearance control. Native count remains exactly 150; adaptive-density fields are inert and may
  not create, remove, relocate, or reorder a component.
- Require exact `torch.__version__="2.9.0+cu128"`. Adam is `torch.optim.Adam` with
  `betas=(0.9,0.999),eps=1e-8,weight_decay=0,amsgrad=False,foreach=None,maximize=False,
  capturable=False,differentiable=False,fused=None,decoupled_weight_decay=False`; the scheduler is
  `CosineAnnealingLR(T_max=120,eta_min=0.001,last_epoch=-1)`. All parameters in an arm share the same
  scalar LR. Serialize the effective optimizer param groups and defaults; seal and runtime must
  refuse a Torch-version or flag drift.
- Training loss is the established mean squared error on the raw, unclamped additive render.
  Background is black, `row_chunk=64`, kernel support remains `q<12`, and there is no early stop.
- Arm execution order is `weight_color_9p,unit_weight_bounded_8p` for the first and third seed of
  each block and the reverse for the middle seed. Runtime is descriptive only.
- Execute the complete appearance-only block first and the complete joint block second, with seeds
  and local views in their listed order. Do not compute, print, or branch on a block-level decision
  between them; only a validity failure may terminate early into the invalid artifact path.

All model, render, loss, and optimizer arithmetic is repository-standard float32. Repository PSNR
and SSIM are likewise evaluated on float32 images; only operations explicitly named float64 below
use float64. Aggregate stored float32 per-view metric scalars in float64.

The seed is the replicate unit. Each seed aggregates its nine source views by an unweighted
arithmetic mean; views and pixels are not treated as independent replicates.

## Common step-zero construction and mandatory equivalence gate

For local selected-view index `j in [0,...,8]`, construct a new
`torch.Generator(device="cpu").manual_seed(seed+j)` and call
`init_gaussians_2d(image,n=150,grad_mix=0.7,generator=<that generator>)` exactly once per
`(block,seed,j)`. This is the ordinary `fit_views` local-index schedule applied after physical
subsetting. Serialize the seed, pre-call state, and post-call state. Clone that immutable `g0` into
both arms without reordering. Require its field shapes, finite values, center bounds, positive
Cholesky diagonals, color range, exact `weight=0.5`, field hashes, and generator state.

Construct the established raw geometry and appearance exactly as production current:

- `xy_raw=logit(clamp(g0.xy/[W,H],1e-4,1-1e-4))`;
- `diag_raw=softplus_inverse(g0.chol[:,[0,2]]-0.3)` and `off_raw=g0.chol[:,1]`;
- `u0=logit(clamp(g0.color,1e-3,1-1e-3))`;
- `s0=logit(clamp(g0.weight,1e-3,1-1e-3))`.

Build current step-zero `w0=sigmoid(s0)`, `c0=sigmoid(u0)`, and compute the float32 common
amplitude once as `a0=w0*c0`. Initialize candidate raw appearance only as `r0=logit(a0)`. No
additional clamp is allowed because this construction is strictly inside `(0,1)`.

Before either optimizer is constructed, require for every image and both blocks:

- raw and built geometry bit-exact across arms;
- component count/order and target bit-exact;
- candidate built weight bit-exact one;
- candidate built color versus `a0` maximum absolute error `<=1e-7` and maximum relative error
  `<=1e-6` where `|a0|>1e-8`;
- raw step-zero render maximum absolute arm difference `<=5e-6`;
- float64 `sum(abs(candidate-current))/sum(abs(current)) <=1e-6`, with a finite positive
  denominator;
- step-zero render-to-render PSNR `>=100 dB` using an MSE floor of `1e-12` only for this
  equivalence diagnostic; and
- raw MSE-loss absolute difference `<=1e-7` and relative difference `<=1e-6`, using the current
  loss as a finite strictly positive denominator.

Serialize all step-zero fields and renders before continuing. This is one global prerequisite: a
single failure writes an invalid artifact and neither optimization block may expose a curve or
decision. The marker remains consumed; tolerances may not be repaired.

## Fixed optimization blocks and checkpoints

Checkpoints are exactly `[0,1,5,10,20,40,80,120]`. Step zero is the pre-optimizer model. A step
`t>0` checkpoint is freshly built and freshly rendered after exactly `t` completed Adam updates and
the corresponding scheduler call; it may not reuse the stale render that produced the last
gradient. Checkpoints are read-only and do not affect the next step.

### Appearance-only mechanism block

For seeds `[7727,8837,9941]`, freeze `xy_raw`, `diag_raw`, and `off_raw` at the common step-zero
values. They are not optimizer parameters, receive no gradient or Adam state, and must remain
bit-exact at all 121 states. Optimize only `(u,s)` for current and only `r` for the candidate for
120 steps. Renderer, target, raw MSE objective, LR, scheduler, and checkpoint work are otherwise
identical.

This block isolates the appearance map under a fixed basis of Gaussian kernels. Its PSNR curve is
the primary optimization-curve evidence relevant to conditioning. It cannot establish joint-fit
utility because production also moves centers and covariances.

### Joint Stage-1 utility block

For seeds `[10007,11003,12007]`, optimize all five common geometry values plus the arm's appearance
values for exactly 120 steps through the ordinary production native fitter. No field is frozen.
The only allowed difference is the appearance map and consequent absence of candidate weight
state. This block measures source-image reconstruction at a common primitive and update budget.

The two blocks use disjoint fresh scenes. No mechanism tensor, metric, rank, optimizer state,
parameter value, or result is loaded into the joint block.

## Reconstruction metrics, pooling, and AUC

At every checkpoint serialize the raw predicted render and target. Retain the float32 objective
loss and independently recompute in float64 the raw RGB squared-error sum, channel count, and raw
MSE. Separately clamp only the float32 reporting prediction to `[0,1]` and compute:

- repository full-canvas PSNR with MSE floor `1e-12`; and
- repository SSIM using the 11x11 separable Gaussian window, sigma `1.5`, constants
  `0.01^2,0.03^2`, and a mean over pixels/channels.

The target is already in `[0,1]` and is not altered. Also report the count/fraction of raw rendered
channels below zero and above one; clamp incidence is diagnostic and cannot rescue quality.

For seed `s`, arm `x`, and checkpoint `t`, let `P_s,x,t` be the unweighted mean of the nine
per-view PSNR values. Define normalized trapezoidal PSNR AUC in dB over the frozen checkpoints:

`AUC_s,x = (1/120) * sum_j (t[j+1]-t[j])*(P[j]+P[j+1])/2`.

SSIM is averaged per view and then per seed. Do not concatenate images, pool pixels before PSNR,
select a checkpoint, interpolate another checkpoint, drop a view/seed, or form a significance
test. Raw pooled SSE/count is additionally reported but is diagnostic because the frozen primary
replicate is the seed-level mean view metric.

## Saturation, Jacobian, null-direction, and Adam diagnostics

These diagnostics explain a curve; they do not replace it. All reductions are float64 and retain
the underlying raw arrays.

At every checkpoint for both blocks and arms, serialize raw appearance values, built amplitude,
weight/color fields, analytic Jacobians, singular values, and:

- fractions of raw logits with `abs(raw)>=8`;
- output fractions in `[0,1e-3]` and `[1-1e-3,1]`;
- sigmoid-derivative fractions `<=1e-4`;
- current weight/color/amplitude and candidate amplitude histograms using fixed edges
  `[0,.001,.01,.05,.10,.25,.50,.75,.90,.95,.99,.999,1]`;
- smallest positive Jacobian singular value, largest singular value, identifiable-subspace
  condition number, and numerical rank at threshold
  `max(shape)*eps(float64)*largest_singular_value`.

Histogram bins are left-closed/right-open except that the last bin includes `1`; every field value
must enter exactly one bin. Define `tau=max(shape)*eps(float64)*largest_singular_value` and numerical
rank exactly as `count(singular_value>tau)`. The smallest positive singular value is undefined at
rank zero, and the three-direction condition number is defined as `largest/smallest` only at rank
three; otherwise its numeric value is exact zero with `defined=false`. Define a component as
`weakly_responsive` when its numerical rank is below three or, when rank is three, its smallest of
the three singular values is below `1e-4`. Report exact counts by view, seed, step, arm, and pool.

For every appearance-only update, retain raw arrays at the state before backward, gradient before
Adam, Adam `exp_avg`/`exp_avg_sq` before and after, optimizer step counter before and after, LR used,
raw parameter immediately after Adam, and actual displacement. Define `grad_a` independently at
that pre-update
state: detach and clone the built geometry and amplitude, make only the cloned amplitude require a
gradient, render it as `color=a,weight=exact_one`, apply the same raw MSE target, and call
`torch.autograd.grad` only on that probe. The probe may not touch production parameters, `.grad`,
optimizer/scheduler state, RNG, or the optimization render. For both arms require probe/optimization
render maximum absolute difference `<=2e-5`, relative L1 difference `<=2e-6` using the optimization
render as a positive finite denominator, and loss absolute/relative differences `<=2e-6/2e-5`
using the optimization loss as a positive finite denominator; otherwise fail closed. Check the
float32 chain rule against:

- current `grad_u = grad_a * w*c*(1-c)` and
  `grad_s = sum_k grad_a_k*w*(1-w)*c_k`;
- candidate `grad_r = grad_a*a*(1-a)`.

Require maximum absolute error `<=2e-6` and maximum relative error `<=1e-4` where the derived
gradient magnitude exceeds `1e-5`. Independently reconstruct every Adam moment and displacement
from the recorded preceding state and effective PyTorch configuration; require maximum absolute
displacement error `<=2e-7` and maximum relative error `<=2e-5` where the derived displacement
magnitude exceeds `1e-8`. Any mismatch invalidates the experiment rather than becoming a result.

At every pre-update current appearance-only state, form the per-component float64 Jacobian and
`full_matrices=True` SVD used by the following null calculation; checkpoint Jacobian diagnostics
are an exact indexed subset of these pre-update states except for terminal checkpoint 120, which
has no following update and receives its own diagnostic.

For each current component-step whose SVD rank is three and whose update norm exceeds `1e-12`, let
`n` be the SVD unit right-null vector and `dtheta` the actual four-vector Adam displacement. Define

- `null_fraction = abs(dot(n,dtheta))/||dtheta||_2`;
- `null_energy_ratio = sum(dot(n,dtheta)^2)/sum(||dtheta||_2^2)` over a named seed/pool; and
- `null_large_fraction = fraction(null_fraction>=0.10)`.

The sign of `n` cannot affect these values. Serialize every eligibility flag, dot product, norm,
and numerator/denominator. Also verify the raw gradient is orthogonal to `n` within absolute dot
error `<=2e-6` and relative cosine `<=2e-5` when both norms exceed `1e-12`. Adam's coordinatewise
preconditioning is permitted to produce a null component in the displacement; that is the
mechanism being measured.

The global null pool contains exactly all eligible current rows for updates `1..120` across the
three mechanism seeds, nine views, and 150 components; a per-seed pool uses that seed's nine views.
For each pool, `null_energy_ratio` is the sum of recorded squared projections divided by the sum of
recorded squared update norms, not a mean of component ratios, and `null_large_fraction` is the
eligible-row count at or above `0.10` divided by the eligible-row count. If there are zero eligible
rows, store both reductions with `defined=false` and force `null_update_material=false`; a zero
eligible count does not itself invalidate otherwise complete evidence. Any non-finite numerator or
denominator is instead a global validity failure. For the candidate, do not invent a fourth
coordinate or a null statistic. Report its three singular directions and saturation only. In the
joint block, checkpoint
`t>0` stores the pre-update gradient, LR, and pre-update moments used for update `t`, plus the
post-update parameters, moments, and displacement; the corresponding step-zero fields have
`defined=false`. The full per-step null analysis remains restricted to the appearance-only block
so the raw archive stays executable in size.

## Frozen validity and decisions

### Global validity

The official artifact is valid only if all six seeds, nine views, two arms, 150 components, 120
updates, and eight checkpoints are present in exact order and all of the following pass:

- source/isolation, common-initialization, step-zero render, field-shape/range, finite-value, fixed-
  count, checkpoint, optimizer-step, LR-schedule, and arm-identity checks;
- candidate unit weight and absence of weight optimizer state;
- appearance-only geometry bit identity;
- current production parity and default preservation;
- raw per-view metric reductions, AUC recomputation, chain-rule identities, SVD/null identities,
  probe equivalence, Adam equations, array manifest, and hash bindings; and
- no overwrite, missing sidecar, non-finite denominator, stale checkpoint render, hidden early
  stop, or optional backend import.

One failure invalidates the entire attempt and authorizes no conditioning or utility statement.
No seed majority can rescue a validity failure.

### Appearance-only optimization-curve gate

For each mechanism seed define `d_auc=AUC_candidate-AUC_current` and final
`d_final=P_candidate,120-P_current,120`. The bounded unit-weight arm has
`appearance_curve_improved=true` only if all hold:

1. mean `d_auc >= +0.10 dB`;
2. at least two of three seeds have `d_auc >= +0.05 dB`;
3. worst-seed `d_auc >= -0.10 dB`;
4. mean `d_final >= -0.05 dB` and worst-seed `d_final >= -0.15 dB`;
5. mean final SSIM delta `>=-0.002` and worst-seed delta `>=-0.010`; and
6. every global validity gate passes.

The current null diagnostic is `null_update_material=true` only if the global raw pool has
`null_energy_ratio>=0.01` and `null_large_fraction>=0.10`, and at least two of three seeds have
`null_energy_ratio>=0.005`.

The candidate saturation guard passes only if its pooled weakly-responsive fraction minus the
current fraction is `<=0.05` and the per-seed difference never exceeds `0.10`, computed over the
same appearance-only population. The global weak pool contains exactly all `3*9*8*150`
component-checkpoint rows per arm; each per-seed pool contains exactly `9*8*150`, using all eight
checkpoints without eligibility filtering.

`fit_time_redundant_coordinate_interference_consistent=true` only when
`appearance_curve_improved`, `null_update_material`, and the saturation guard all pass. This is
evidence consistent with interference from the redundant coordinate, not proof that a finite
nonlinear projected update was globally wasted: the local null direction changes along the path,
and the candidate also changes the metric on the three identifiable coordinates. If the curve
improves but the null gate does not, report an optimization-curve effect without a null-motion
explanation. If the null gate passes but the curve does not, report local projected motion with no
material curve benefit under this budget.

### Joint Stage-1 utility gate

On each joint seed define candidate-minus-current final PSNR, final SSIM, and PSNR AUC deltas. The
candidate is `joint_stage1_noninferior=true` only if all hold:

1. mean final PSNR delta `>=-0.10 dB`;
2. at least two of three final PSNR deltas are `>=-0.10 dB`;
3. worst final PSNR delta `>=-0.30 dB`;
4. mean final SSIM delta `>=-0.002` and worst delta `>=-0.010`;
5. mean PSNR-AUC delta `>=-0.10 dB` and worst delta `>=-0.30 dB`; and
6. every global validity gate passes.

It has `joint_stage1_material_improvement=true` only if it is non-inferior and additionally:

1. mean final PSNR delta `>=+0.10 dB`;
2. at least two of three final PSNR deltas are `>=+0.05 dB`;
3. worst final PSNR delta `>=-0.10 dB`;
4. mean PSNR-AUC delta `>=+0.10 dB`; and
5. mean final SSIM delta is nonnegative.

Report the exact one-learned-parameter-per-component reduction, but do not call it a memory,
runtime, bitrate, or compression improvement.

Interpretation is frozen:

- invalid evidence: no scientific decision; preserve the consumed attempt and repair only in a
  fresh namespace;
- redundant-coordinate interference consistency gate fails and joint non-inferior fails: retain
  current and close this exact bounded candidate on the CPU synthetic fixed-count setup without
  tuning;
- appearance-only curve improves but joint non-inferior fails: an isolated fixed-basis effect that
  does not survive geometry co-adaptation;
- joint non-inferior without material improvement: bounded unit weight preserves Stage-1 source
  quality at one fewer learned coordinate, but supplies no quality-improvement or default claim;
- joint material improvement: a Stage-1 synthetic research candidate only; it still cannot enter
  Stage 2 or become a default without a separate downstream semantic protocol, real-image
  evidence, and independent result audits;
- any combination not named above is reported literally, without selecting a new activation, LR,
  initialization, seed, iteration budget, or arm.

## Raw sidecar and independent recomputation contract

Every valid or invalid scientific JSON is accompanied by exactly one uncompressed
`numpy.savez` sidecar readable with `allow_pickle=False`. Object arrays and pickle are forbidden.
For a valid artifact, stable slash-separated arrays must include, at minimum:

- every selected target image and common initializer field;
- every initializer seed and pre/post generator state;
- all raw/built geometry and appearance fields at step zero and every checkpoint;
- all checkpoint renders, losses, SSE/counts, PSNR/SSIM inputs, and clamp masks;
- every appearance-only per-step raw parameter, gradient, direct amplitude-probe gradient,
  probe-equivalence numerator/count/loss, Adam moment/state/LR, displacement, SVD/null vector,
  eligibility mask, and reduction input;
- joint checkpoint parameter/gradient/state and geometry-displacement arrays; and
- seed/view/arm/checkpoint identities and exact ordering arrays.

An invalid artifact instead contains every array that was successfully materialized before the
failure, plus explicit scalar phase/completion arrays and the same manifest/hash treatment. It may
not fabricate arrays for phases that were never reached. If a non-finite floating array triggered
the failure, preserve that offending array and a uint8 classification mask (`0=finite,1=NaN,
2=+Inf,3=-Inf`) in the invalid sidecar. Invalid artifacts carry no scientific decision, and the
independent review verifies the failure boundary as well as every available array.

All floating arrays in a valid sidecar must be finite; non-finite values are allowed in an invalid
sidecar only under the explicit failure-evidence rule above. Nullable values use a numeric value
array plus a boolean defined mask; NaN is never a null encoding. The JSON stores raw decisive
numerators/counts and a
sorted manifest entry for every array: logical name, dtype, shape, byte length, and content
SHA-256. Content hashes use the semantic-factorial protocol's name-independent little-endian
contract. For array `x`, set
`dtype_token=np.dtype(x.dtype).newbyteorder("<").str.encode("ascii")`,
`shape_bytes=np.asarray(x.shape,dtype="<i8").tobytes(order="C")`, and `data_bytes` to the
C-contiguous bytes after converting multibyte data to the corresponding little-endian dtype. Then:

`SHA256(dtype_token || b"\0" || int64_shape_bytes || b"\0" || C_contiguous_data_bytes)`.

Here `int64_shape_bytes=shape_bytes` and `C_contiguous_data_bytes=data_bytes`. The collection digest
is SHA-256 of UTF-8
`json.dumps(sorted([[name,content_digest],...]),separators=(",",":"),ensure_ascii=True)`.

The JSON and result note bind the completed sidecar file SHA-256 and collection digest. The
once-only marker binds only the prospective path. The independent reviewer must reconstruct every
decisive per-view value, seed aggregate, AUC, Jacobian/null statistic, Adam equation, saturation
guard, and final decision from the raw arrays rather than trusting stored flags. Raw target/render
arrays are mandatory specifically to avoid the tensor-level qualification of the earlier gauge
audit.

## Append-only implementation, seal, attempt, and review

Fresh namespace:

- future harness: `benchmarks/stage1_fit_parameterization.py`;
- focused tests: `tests/test_stage1_fit_parameterization.py`;
- outcome-free implementation review:
  `benchmarks/results/20260716_stage1_fit_parameterization_IMPLEMENTATION_REVIEW.md`;
- implementation seal:
  `benchmarks/results/20260716_stage1_fit_parameterization_SEAL.json`;
- once-only marker:
  `benchmarks/results/20260716_stage1_fit_parameterization_ATTEMPT.json`;
- valid output triple: `<UTC>_cpu_stage1_fit_parameterization.json`, matching
  `_RAW.npz`, and matching `_RESULT.md`;
- invalid output triple: replace terminal `.json`, `_RAW.npz`, and `_RESULT.md` with
  `_invalid.json`, `_invalid_RAW.npz`, and `_invalid_RESULT.md` respectively; and
- independent review: matching `_AUDIT.md` and `_SCIENTIST_REVIEW.json`.

The implementation review must be outcome-free, preserve the preregistration chronology, bind the
preimplementation sources named above plus the complete candidate implementation/tests, and say
exact `Verdict: PASS` before seal creation. It may run only tiny nonofficial deterministic tests;
it may not initialize, fit, or render an official seed.

The sole outcome-free seal command is:

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python benchmarks/stage1_fit_parameterization.py seal --output benchmarks/results/20260716_stage1_fit_parameterization_SEAL.json`.

Seal creation runs, in order, exactly `.venv/bin/python -m ruff check .`,
`.venv/bin/python -m ruff format --check .`, `.venv/bin/python -m pytest -q -m "not slow"`,
`.venv/bin/python scripts/docs_sync.py`, and `git diff --check`. It records literal commands,
return codes, complete stdout/stderr and hashes, and refuses a failure or source drift. The seal
binds this preregistration, implementation review, harness, focused tests, every repository Python
source/test, complete revision and dirty diff, effective environment/configs, raw schema, and
command templates. Full-sealed and runtime-loaded source aggregates are distinct domains and must
each be recomputed against their own explicit path map.

The sole scientific command is:

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python benchmarks/stage1_fit_parameterization.py run --seal benchmarks/results/20260716_stage1_fit_parameterization_SEAL.json --output benchmarks/results/<fresh-UTC>_cpu_stage1_fit_parameterization.json`.

Before constructing any official scene or initializer, the harness must validate the sole seal
path, preregistration/review/source/environment bindings, derive all valid and invalid JSON/raw/note
paths, require every path absent, and exclusively create the marker binding all possibilities.
Valid execution writes only the valid triple; fail-closed execution writes only the invalid
triple. An interruption consumes the marker. The harness may not overwrite, resume, redirect after
an outcome, retry a seed, replace a failed view, or write both result forms.

After completion, an independent reviewer using the repository results-audit protocol must verify
chronology, isolation, source and marker bindings, production-current parity, raw archive hashes,
all equations/reductions/gates, and artifact routing. The machine review binds the preregistration,
seal, marker, implementation review, harness, tests, JSON, raw archive, result note, and human audit
SHA-256; records reviewer identity and UTC time; and gives an exact verdict. Its top-level schema is
exactly these keys and no others: `artifact_type`, `verdict`, `reviewer_identity`,
`reviewed_at_utc`, `preregistration_sha256`, `seal_sha256`, `attempt_sha256`,
`implementation_review_sha256`, `harness_sha256`, `tests_sha256`, `result_sha256`,
`raw_sha256`, `result_note_sha256`, `human_audit_sha256`, `runtime_source_aggregate_sha256`,
`decisions_recomputed`, and `default_change_authorized`. A claim-admitting review requires
`artifact_type="stage1_fit_parameterization_scientist_review"`, exact `verdict="pass"`,
`decisions_recomputed=true`, and `default_change_authorized=false`; all hash strings must be
lowercase 64-hex and all other fields must match the reviewed artifacts literally. No quantitative
claim may enter `README.md`, `docs/`, or `ara/`, and no follow-up/default proposal may open, before
that independent review.

This preregistration authorizes only future implementation, outcome-free review/sealing, and the
single once-only scientific command. It authorizes no pilot, official execution, source/default
change, documentation claim, or ARA claim now.
