# Preregistration: quaternion radial-gauge optimization audit

## Chronology, question, and scope

Frozen at `2026-07-16T01:20:42+02:00`, after a read-only mathematical and source-code review but
before any harness implementation, quaternion diagnostic, pilot, training run, render comparison,
or outcome for this experiment.

Pre-implementation clarification at `2026-07-16T01:35:17+02:00`, after an independent read-only
preregistration review and still before any harness implementation, quaternion diagnostic, pilot,
training run, render comparison, or outcome. The prior frozen document had SHA-256
`e513b6a7b4fc7410516e67878b7c8eaa6e9da6b11151b006c3dea02d2256a77b`. This amendment makes
already-prespecified reductions, validity invariants, held-out metrics, Trainer entry ordering,
and append-only artifact bindings uniquely executable. It removes one non-decisional,
non-identifiable parameter-to-GT diagnostic. It changes no scene, seed, fit, initialization,
diagnostic row, perturbation, arm, optimizer, schedule, checkpoint, threshold, decision gate, or
interpretation boundary.

Pre-implementation seal-digest clarification at `2026-07-16T01:41:39+02:00`, during harness
implementation but before any seal, official diagnostic, pilot, attempt marker, render comparison,
or outcome. The preceding document had SHA-256
`6a1235ac798adabaf8e33cfa1c82b7f2e03684e4a9b69f82c179c4cc6054c5ca`. It uniquely defines the
already-required non-self-referential `seal["sha256"]` payload digest. It changes no scientific
input, arm, calculation, threshold, gate, or interpretation.

Pre-implementation verification-binding clarification at `2026-07-16T01:46:09+02:00`, during
harness/toy-test implementation but before any seal, official diagnostic, attempt marker, pilot,
training run, render comparison, or outcome. The preceding document had SHA-256
`532a1beaa1e76cd27f51f2ca7881786d6a4bf392d61dcb99f510915e68ef9448`. It explicitly permits
the seal's already-required full verification to execute nonofficial toy render fixtures while
continuing to forbid Phase-A/Phase-B scientific preparation. No scientific choice changes.

The repository forms a 3D covariance as

`Sigma(q,s) = R(normalize(q)) diag(exp(2s)) R(normalize(q))^T`.

For every nonzero scalar `c`, `q` and `c*q` therefore encode the same covariance and render. The
current refinement path nevertheless gives the raw four quaternion coordinates to coordinatewise
Adam and does not retract the raw parameter to the unit sphere. This experiment asks two ordered
questions:

1. **Phase A, mechanism:** does current ambient Adam turn this exact radial gauge into materially
   different physical optimization trajectories, and does a unit canonicalization/retraction
   remove that dependence for a controlled anisotropic Depth initialization?
2. **Phase B, utility:** only if the mechanism is material and independently audited, does unit
   retraction or projection of the actual Adam displacement followed by retraction improve
   held-out time-to-quality during ordinary joint 3DGS refinement from the full initialization?

This is a CPU synthetic optimizer audit. It is not a claim that current 3DGS is broken: optimizing
raw quaternion coordinates while normalizing in the forward pass follows common 3DGS practice.
It cannot establish real-scene utility, CUDA/gsplat parity, density-control interaction, export
compatibility, speed, memory, or a production-default change. Phase A's self-reconstruction target
is a deliberately controlled mechanism probe, not novel-view evidence. Phase B is the only utility
phase, and even a positive Phase B remains synthetic fixed-topology evidence requiring separate
real/CUDA confirmation.

No PLY or NPZ file is read or written by either phase. In particular, this protocol does not decide
whether a physical scene export should contain normalized quaternions or raw optimizer coordinates.
`Gaussians3D`'s documented unit-quaternion contract, the original 3DGS raw-rotation PLY convention,
and resumable optimizer state are separate interface questions. No serializer, loader, container
validation, or export default may change from this experiment.

## Literature and method boundary

The 2026-07-12 through 2026-07-16 Scholar Inbox digest provides representation analogies, not
evidence for an expected outcome. [Grassmannian Splatting I](https://arxiv.org/abs/2607.10489)
explicitly represents a unit normal in `S^3/{+/-}` and uses a projector to construct a
rank-constrained spacetime covariance. That motivates making a constrained geometric
representation explicit, but the paper does not test quaternion retraction or the optimizer arms
below. [Incremental Gaussian Triangulation](https://arxiv.org/abs/2607.10690) uses planar
elliptical surfels, plane pulling, and shortest-axis alignment when oriented surface evidence is
available. It motivates using genuinely anisotropic covariance rows in this audit; it does not
provide the quaternion update, validate this repository's RGB-only surface targets, or authorize a
planarity constraint here.

No external method or code is reproduced. The experiment retains the repository's native stage 1,
metric synthetic Depth lift, standard covariance factorization, Torch reference renderer, Adam
hyperparameters, and joint refinement loss. Rank-constrained covariance, plane/normal losses,
multiscale training, and coarse-to-fine schedules are absent and require separate preregistrations.

## Frozen environment, data, stage 1, and Depth initialization

- CPU only: `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, Torch intra-op
  threads `4`, deterministic algorithms enabled, and no optional StructSplat or gsplat import.
- Seeds are exactly `0,1,2`.
- For each seed, call
  `make_synthetic_scene(n_gaussians=40,n_cameras=12,image_size=48,seed=seed)` once per phase.
  This helper necessarily constructs all twelve synthetic views internally. Freeze training
  indices `[0,1,2,4,5,6,8,9,10]`, held-out indices `[3,7,11]`, and their original identities.
  Physically subset to the nine training views before fitting, lifting, target construction, or
  optimization. Phase A may not read, render, hash, score, or serialize a held-out image, camera,
  depth, alpha, or metric after the prescribed full-scene construction. Phase B receives a fresh
  held-out evaluation capability only after its authorization checks pass. Held-out data never
  enters fitting, lifting, arm construction, training loss, Phase-A validity or materiality gates,
  schedule generation, stopping, checkpoint selection, or any adaptive choice. After Phase-B
  authorization it enters only the frozen Phase-B utility and safety decisions below. Immediately
  after Phase A forms the physical nine-view subset, it drops the twelve-view scene handle and
  removes `gt_gaussians` from the training-only scene capability; only training images, cameras,
  depths, masks, points/bounds, and the frozen original-index mapping remain accessible. In Phase B,
  every authorization and input-binding check occurs before fresh scene construction, and only the
  physical nine-view subset is passed to fitting, lifting, or Trainer; held-out and GT capabilities
  remain confined to the read-only evaluator.
- Fit the nine training images exactly once per seed and share immutable clones across every
  subsequent construction in that phase. Use native stage 1 with
  `FitConfig(n_gaussians=150,max_gaussians=5000,iterations=120,backend="native",
  adaptive_density=True,growth_waves=5,relocate_fraction=0.0,
  structsplat_renderer="auto",lr=0.01,grad_init_mix=0.7,row_chunk=64,log_every=50,
  convergence_patience=0,convergence_tol=0.05,convergence_check_every=25)` and
  `fit_views(train_scene.images,config,seed=seed,masks=train_scene.masks)`. Require exactly nine
  finite fitted sets of exactly 150 components, valid public field shapes/ranges, and positive
  Cholesky diagonals. No refit, component reorder, extra iteration, or arm-dependent RNG is allowed.
- Construct exactly one shared full initialization per seed with
  `DepthLifter(backend=GroundTruthDepth(train_scene.gt_depths),sh_degree=0,min_weight=0.05,
  init_opacity=0.1,normal_thickness=0.15,covariance_mode="surface",isotropic_sigma=None,
  robust_depth_gradients=True,merge=True,merge_voxel_frac=0.01)`. The explicit backend is created
  fresh at cursor zero and the lifter is called exactly once. It may consume training-view metric
  depths only. No other lifter, covariance mode, normal thickness, merge setting, voxel size, or
  initialization is compared.
- Require the full initialization to contain at least 256 primitives, have finite fields, opacity
  in `[0,1]`, positive covariance eigenvalues, and quaternion norms finite and greater than
  `1e-8`. Hash every training image/camera/depth, fitted field/history/order, scene points/bounds,
  full Depth field, covariance, and aggregate before selecting diagnostic rows. Phase B must
  recreate and match these training-side hashes; it cannot load Phase A optimizer outcomes.

All initialization and target construction use native float32. Auditing reductions, ratios, AUCs,
and decision formulas use float64. JSON must contain the effective configurations and raw values,
not only rounded summaries.

The following reduction definitions apply everywhere below unless a formula explicitly says
otherwise. For covariance tensors `A,B` with the same ordered rows, define

`rel_cov(A,B) = sum_i ||A_i-B_i||_F / max(sum_i ||B_i||_F,1e-18)`.

Every sum and norm in this definition is evaluated after promotion to float64. An absolute
covariance error is the maximum absolute tensor element. Quantiles flatten exactly the included
finite float64 row values and use `torch.quantile(...,interpolation="linear")`; p90, median, and
p99 mean quantiles `0.90`, `0.50`, and `0.99`. A cross-seed mean is the unweighted arithmetic mean
of the three paired seed values. A normalized trapezoidal AUC for checkpoints `(t_j,v_j)` is

`sum_j (t_(j+1)-t_j)*(v_j+v_(j+1))/2 / (t_last-t_first)`.

No mean of already-rounded summaries, per-view PSNR average, alternate quantile convention, or
unstated epsilon may replace these definitions.

## Frozen anisotropic diagnostic subset and perturbation

Phase A uses a fixed 128-row diagnostic subset so the five-by-three optimizer mechanism factorial
is CPU-bounded. It does not prune or alter the full initialization used in Phase B.

For each full-initialization covariance, compute float64 eigenvalues
`0 < lambda_1 <= lambda_2 <= lambda_3` and
`anisotropy_i=lambda_3/max(lambda_1,1e-18)`. A row is diagnostic-eligible only when
`anisotropy_i>=2.0`. Require at least 128 eligible rows. Select exactly the 128 greatest
anisotropy values, breaking exact ties by lower original full-initialization index, and emit them
in increasing original-index order. The selection may inspect only the sealed initialization
covariance and index. Serialize all eigenvalues, ratios, eligible indices, selected indices, and
hashes. Near-isotropic rows remain in the full Phase-B initialization but are excluded from
Phase-A rotation diagnostics because their orientation is weakly or non-identifiable; they cannot
be reclassified from an outcome.

Let `q_i=(qw,qx,qy,qz)` be each selected quaternion normalized once for perturbation construction.
With a separate CPU generator seeded `50000+seed`, draw one float32 three-vector
`a_i ~ N(0,I_3)` per row and set `axis_i=a_i/norm(a_i)`. Require every pre-normalization axis norm
to be finite and at least `1e-6`; do not redraw a failed row. Freeze physical perturbation angle
`theta=20 degrees`, form the wxyz delta quaternion

`d_i = (cos(theta/2), sin(theta/2)*axis_i)`,

and construct `q_perturbed_i = normalize(d_i tensor_product q_i)` with Hamilton product

`(dw*qw-dv dot qv, dw*qv+qw*dv+dv cross qv)`.

Left multiplication makes the construction antipodally equivariant: replacing `q_i` by `-q_i`
replaces the perturbed parameter by its negative without changing the physical rotation. Copy
selected means, log-scales, opacity, and SH bit-for-bit. Render the unperturbed
selected set once in each of the nine training cameras under no-grad with
`TorchRasterizer(sh_color_activation="hard",kernel_support_mode="hard",
visibility_margin_sigma=3.0)`, degree 0, black background, and no output clamp. These nine renders
are the immutable Phase-A self-targets. No source RGB or held-out value enters the self-target.

The diagnostic subset and perturbation are mechanism instruments only. Their metrics may not be
reported as reconstruction quality, compared with another lifter, or used as the Phase-B start.

## Mandatory algebra and representation-contract prerequisites

Before any optimizer is constructed, form radial scales `c in {0.25,1.0,4.0}` in that order from
the same `q_perturbed`, plus the reporting-only antipodal tensor `-q_perturbed`. No arm may
recompute a perturbation.

### Covariance and render equivalence

For every seed and radial representation:

- evaluate `quat_to_rotmat` and covariance from parameters promoted to float64 before any
  rotation/covariance arithmetic;
- relative to `c=1`, require maximum rotation and covariance absolute errors `<=2e-12` and
  `rel_cov` errors `<=2e-12` for `c=0.25`, `c=4`, and the antipodal tensor. The rotation relative
  Frobenius error uses the same ordered-row sum-of-Frobenius-norms formula as `rel_cov`, with
  rotation matrices in place of covariances, and must also be `<=2e-12`;
- in native float32, render all nine training diagnostic views with the same frozen renderer and
  require every output finite, maximum color/alpha/accumulated-depth absolute error `<=5e-6`, and
  float64 raw `sum(abs(color_delta))/sum(abs(reference_color))<=1e-6`, with a finite strictly
  positive denominator.

Serialize raw numerators, denominators, per-view maxima, and tensor hashes. These tolerances audit
the mathematical gauge through the repository's real code paths; they do not require bit identity.

### Step-zero gradient identities

The Phase-A loss for a scheduled diagnostic view is exactly

`0.8 * mean(abs(pred_color-target_color)) + 0.2 * (1-ssim(pred_color,target_color))`.

Use the first position in the frozen Phase-A schedule and independently backpropagate this loss at
each current-policy radial scale before creating Adam state. Require finite positive loss, finite
gradients, and at least 32 rows at each scale with gradient norm greater than `1e-12`. For active
rows at that scale define

`tangent_residual_i = abs(dot(q_i,g_i))/(norm(q_i)*norm(g_i))`.

Require the maximum over that scale's active rows to be `<=1e-5`. With `g_c` the gradient at raw
parameter `c*q`, define for every row `i`

`scaled_gradient_difference_i = norm(c*g_c_i-g_1_i)/max(norm(g_1_i),1e-12)`.

For `c=0.25,4.0`, require the maximum over all 128 row values to be `<=5e-4`. Also explicitly
project `g_1` onto the tangent plane. For every row define the removed-gradient numerator as
`abs(dot(normalize(q_1_i),g_1_i))`, denominator as `max(norm(g_1_i),1e-12)`, and their ratio;
require the maximum over all 128 ratios to be `<=1e-5`. Record every row's dot product, norm,
residual (zero plus an inactive flag when its gradient norm is at most `1e-12`), removed-gradient
numerator/denominator/ratio, and scaled-gradient numerator/denominator/difference. Any algebra,
render, active-gradient, or gradient-identity failure makes Phase A invalid; no optimizer
trajectory or materiality statistic may be computed or exposed, and the official attempt remains
consumed.

## Phase A: optimizer-only mechanism factorial

Only after every seed passes the global prerequisites, run exactly 40 quaternion-only steps for
the Cartesian product of five policies and three radial scales, for 15 arms per seed. Means,
log-scales, opacity, and SH remain immutable, topology is fixed, and the only optimized tensor is
the 128-by-4 raw quaternion parameter.

Generate one schedule per seed with a fresh CPU
`torch.Generator().manual_seed(seed)` and exactly 40 calls equivalent to
`torch.randint(0,9,(1,),generator=generator)`. Serialize the integer schedule and hash before the
first arm. Every arm must visit that exact schedule. At each step render only the scheduled
training diagnostic target and use the frozen Phase-A loss.

Use one fresh optimizer per arm:
`torch.optim.Adam([q],lr=1e-3,betas=(0.9,0.999),eps=1e-15,weight_decay=0,
amsgrad=False,foreach=False,fused=False,capturable=False,differentiable=False)`.
There is no LR schedule, clipping, regularizer, weight decay, optimizer-state sharing, or moment
transport.

The policies are:

1. `current`: initialize the raw parameter as `c*q_perturbed`; use ordinary backward and ambient
   Adam; never normalize the stored parameter after a step.
2. `entry_canonical`: initialize as `normalize(c*q_perturbed)` before Adam state exists; thereafter
   use the current ambient update without retraction.
3. `unit_retraction`: apply the same entry canonicalization; immediately after each Adam step set
   `q <- normalize(q)` under no-grad. This changes no rendered orientation at that instant but
   fixes the next step's raw norm.
4. `tangent_displacement_retraction`: apply entry canonicalization. Before Adam, clone `q_old`.
   After Adam gives `q_star`, compute `delta=q_star-q_old`,
   `delta_T=delta-q_old*dot(q_old,delta)` row-wise, then set
   `q <- normalize(q_old+delta_T)` under no-grad. Adam's first and coordinatewise second moments
   remain untouched; this is an attribution candidate, not a fully intrinsic/Riemannian Adam.
5. `gradient_projection_current`: initialize as `c*q_perturbed`. After backward and before ambient
   Adam, replace `g` with `g-normalize(q)*dot(normalize(q),g)` row-wise. Do not canonicalize or
   retract. This arm tests whether explicit tangent-gradient projection adds anything beyond the
   forward normalization; it is Phase-A attribution only and can never enter Phase B.

For every policy and radial scale, require every initialization, pre-backward/`q_old`, `q_star`,
and post-policy row norm to be finite and greater than `1e-8`. Policies 3 and 4 therefore also fail
on any invalid pre-retraction norm. No epsilon clamp or identity fallback is permitted. The
post-policy stored quaternion is the one used by the next forward pass and checkpoint. Policy
application occurs after the quaternion Adam step and before any evaluation. No policy projects
or resets Adam state.

### Checkpoints and raw diagnostics

Checkpoints are `[0,10,20,30,40]`. At every optimization step, serialize per-row raw arrays and
tensor hashes for:

- raw quaternion norm before backward, gradient norm, `dot(q,g)`, and tangent residual;
- raw quaternion before Adam, `q_star` immediately after Adam, and raw Adam displacement;
- `abs(dot(normalize(q_old),delta))/norm(delta)` for rows with `norm(delta)>1e-12`;
- pre- and post-policy norms, physical sign-invariant angular step
  `2*acos(clamp(abs(dot(normalize(q_old),normalize(q_new))),0,1))`;
- for the gradient-projection arm, removed-gradient numerator and denominator;
- scheduled-view loss and finite-status flags.

At every checkpoint, evaluate all nine self-target views under no-grad and serialize pooled and
per-view color MSE/PSNR, loss, normalized covariance error to the unperturbed diagnostic target,
sign-invariant orientation error to its quaternion, raw norms, and all field/covariance/render
hashes. Define pooled self-target PSNR from one raw pixel/channel SSE and count with an MSE floor
of `1e-12`; a per-view loss uses the frozen Phase-A loss and checkpoint loss is the arithmetic
mean of the nine per-view losses. The normalized covariance error is `rel_cov(current,target)`.
Per-row orientation error is
`2*acos(clamp(abs(dot(normalize(q_current),normalize(q_target))),0,1))`; serialize all rows plus
their arithmetic mean, linear-interpolation median, p90, and maximum. Define normalized
trapezoidal self-target AUC in dB over `[0,10,20,30,40]` with the common AUC definition above,
whose denominator is 40. This AUC is a mechanism statistic, not a quality claim.

Rows with gradient or displacement norm at most `1e-12` are retained in fields and renders but
excluded from the corresponding ratio distribution; their counts and raw zero values are
serialized. Each step must have at least 32 active-gradient rows and 32 active-displacement rows.
All reductions pool raw numerators/denominators or raw row values before computing ratios and
quantiles. A summary without the underlying arrays is invalid.

### Construction and attribution invariants

- Every arm has bit-identical non-quaternion fields, selected row order, target renders, view
  schedule, optimizer hyperparameters, and topology.
- Step-zero normalized covariances and renders pass the same equivalence tolerances as the algebra
  prerequisite across all 15 arms.
- For `entry_canonical`, `unit_retraction`, and `tangent_displacement_retraction`, the three radial
  replicas must have maximum checkpoint pairwise covariance difference `<=1e-6`. For every pair
  `(a,b)`, including `(0.25,4.0)`, compute the numerator as
  `sum_i ||Sigma_a,i-Sigma_b,i||_F` and the denominator as
  `max(sum_i ||Sigma_c=1,i||_F,1e-18)` at that same checkpoint; divide once. The maximum self-target
  AUC spread must be `<=0.001 dB`. Take the maximum across all three unordered radial pairs and all
  five checkpoints, and require both conditions separately for every seed and policy. This proves
  that entry canonicalization actually collapses the tested positive radial gauges. Failure is an
  invalid implementation, not evidence against the candidate.
- Across all gradient-projection row-steps, pooled and per-seed p99 removed-gradient fraction must
  be `<=1e-5`. Compute each fraction from the original finite gradient immediately before explicit
  projection, and include all three radial replicas in each seed's flattened distribution. The
  arm's longer trajectory is reporting-only: numerical differences may compound, but they cannot
  rescue or veto materiality after the per-step tangent identity passes.
- At each checkpoint, re-render a normalized copy of every current-policy raw tensor and require
  covariance and render agreement with its unmodified raw tensor under the frozen algebra
  tolerances, with the unmodified raw tensor as the denominator reference for `rel_cov` and render
  relative error. This confirms that raw norm itself, rather than an unintended forward change,
  is the redundant coordinate.

### Frozen Phase-A materiality decision

For seed `s`, compute only from the three `current` radial arms:

- `gauge_auc_spread_s = max_c(AUC_s,c)-min_c(AUC_s,c)`;
- `gauge_cov_spread_s`: at step 40, the maximum pairwise float64 sum of per-row covariance
  Frobenius differences divided by the sum of `c=1` covariance Frobenius norms;
- `unit_effective_lr_p90_s`: p90 over all post-Adam steps and rows in the `current,c=1` arm of
  `abs(1/norm(q_star)-1)`;
- `unit_radial_fraction_median_s`: median of the raw Adam displacement radial fractions in the
  `current,c=1` arm.

A seed is `ambient_gauge_material` only if all four hold:

1. `gauge_auc_spread_s >= 0.05 dB`;
2. `gauge_cov_spread_s >= 0.001`;
3. `unit_effective_lr_p90_s >= 0.01`;
4. `unit_radial_fraction_median_s >= 0.10`.

The pooled four tests are computed uniquely as follows. For every radial scale and checkpoint,
sum raw checkpoint color SSE and channel count across the three seeds, derive one pooled PSNR,
compute its normalized checkpoint AUC, then take the maximum-minus-minimum radial AUC. For every
unordered radial pair at step 40, sum the three seeds' per-row covariance Frobenius-difference
numerators and sum the three `c=1` covariance Frobenius-norm denominators, divide once, then take
the maximum pair. Concatenate the seed-tagged `current,c=1` effective-LR row-step values before the
linear p90 and concatenate the corresponding active-displacement radial fractions before the
linear median. Phase A authorizes Phase B only when at least two of three seeds are material, the
pooled decision is material, every validity and attribution invariant passes, and all three
canonicalizing policies pass radial-replica collapse for every seed.
The scales, perturbation, subset size, steps, optimizer, thresholds, or four-way conjunction may
not be tuned after observing a failure. A failed materiality gate is a valid negative mechanism
result and permanently stops this branch before Phase B. A non-finite/undefined denominator or
invariant failure is an invalid consumed attempt and also stops.

The entry-only and gradient-projection arms are controls. Their loss or AUC can neither authorize
Phase B nor be promoted to a candidate after outcomes are visible.

## Phase B: fresh joint-refinement time-to-quality test

Phase B is forbidden until a strict independent scientist review recomputes Phase A from raw
evidence, verifies seal/source/attempt binding, and emits machine-readable execution clearance.
After authorization, regenerate each seed's synthetic scene, native fits, and full Depth
initialization from scratch under the frozen preparation. Require exact training-side fitted and
initialization hashes from Phase A. Do not load the diagnostic subset, perturbation, self-targets,
or any Phase-A optimizer parameter/state.

Clone the complete, unperturbed Depth initialization into exactly three arms:

1. `current`: established raw ambient quaternion Adam behavior;
2. `unit_retraction`: entry canonicalization and post-step unit retraction as defined in Phase A;
3. `tangent_displacement_retraction`: entry canonicalization and projection of the actual Adam
   displacement followed by retraction as defined in Phase A.

Implement the policies as an opt-in `TrainConfig`/Trainer research control named
`quaternion_update_policy`, with allowed values exactly `current`, `unit_retraction`, and
`tangent_displacement_retraction`, and default `current`. Each Phase-B arm sets this field to its
arm name; all other frozen `TrainConfig` values remain common. The default path must remain
bit-exact in fields, non-time history, sampled views, and RNG versus an explicit `current` setting.
A non-current mode must reject `densify=True` before parameter or optimizer construction. For each
candidate, entry canonicalization occurs exactly once, after device transfer/SH padding and before
the quaternion `Parameter` and every optimizer are constructed. The effective step-zero
quaternion is therefore the actual stored post-entry tensor: raw initialization for `current`, and
`normalize(raw initialization)` for each candidate. The harness must capture and audit that actual
tensor before optimizer construction, serialize both common-input and effective-parameter hashes,
and may not substitute a separately normalized surrogate or normalize the candidate a second time.

For candidate modes, snapshot `q_old` immediately before the quaternion Adam `step()`. Preserve the
existing optimizer iteration order; after every optimizer's `step()` has returned, clone the
actual `q_star`, apply the frozen policy, and do so before any history append, LR decay, density
hook, callback, or evaluation. No Adam moment is projected, transported, reset, or rescaled.
For every Phase-B arm, every effective step-zero, pre-backward/`q_old`, `q_star`, and post-policy
row norm must be finite and greater than `1e-8`; candidate post-policy norms additionally satisfy
the unit tolerance below. No epsilon clamp or identity fallback is permitted.
Focused CPU tests must prove the update equations, single entry behavior and actual step-zero
capture, norm/finiteness failure, rejection with density enabled, fixed default, callback
isolation, exact optimizer-state non-mutation, and exact non-quaternion preservation.

Train every arm with
`TrainConfig(iterations=120,lr_means=1.6e-4,lr_quats=1e-3,lr_scales=5e-3,
lr_opacity=5e-2,lr_sh=2.5e-3,lr_sh_rest=1.25e-4,ssim_lambda=0.2,
rasterizer="torch",device="cpu",densify=False,eval_every=30,target_sh_degree=3,
sh_degree_interval=30,use_masks=False,outside_alpha_lambda=0.01,mask_alpha_lambda=0.05,
random_background=False,opacity_reg=None,scale_reg=None,packed=False,antialiased=False,
sh_color_activation="hard",kernel_support_mode="hard",visibility_margin_sigma=3.0,
validate_render_finite=True,seed=seed,quaternion_update_policy=arm_name)`.

All parameters are jointly optimized; only the quaternion update policy differs. Topology remains
fixed. Training order is cyclically counterbalanced:

- seed 0: `current,unit_retraction,tangent_displacement_retraction`;
- seed 1: `unit_retraction,tangent_displacement_retraction,current`;
- seed 2: `tangent_displacement_retraction,current,unit_retraction`.

Before training, independently instantiate the exact local Trainer generator and draw 120 local
training-view positions. Require all arm schedule probes and official sampled-view histories to
match exactly. Require step-zero physical covariance/render equality within `atol=5e-6,
rtol=2e-5` on the captured effective parameters: apply `torch.allclose` separately to covariance
and to every view's color, alpha, and accumulated-depth tensors, always using `current` as the
reference. Require equal counts/parameter shapes and bit-identical non-quaternion fields. Record
both local subset positions and their original global view identities. Candidate post-policy norms
must remain within maximum `abs(norm-1)<=2e-5` after every step; current norms are finite reporting
data and must remain greater than `1e-8`.

### Held-out evaluation and primary AUC

Held-out views `[3,7,11]` are opened only after Phase-B authorization and never enter training.
Render the frozen synthetic GT Gaussians with
`TorchRasterizer(sh_color_activation="hard",kernel_support_mode="hard",
visibility_margin_sigma=3.0)`, degree 0, black background, and no output clamp. The resulting raw,
unclamped GT render color is the held-out color target; define truth support as `GT alpha > 0.05`
and GT expected depth as
`GT accumulated depth / clamp_min(GT alpha,1e-6)`. Hash and serialize these truth fields. At steps
`[0,30,60,90,120]`, a detached read-only Trainer checkpoint callback renders each arm with degree
3, hard color/support, 3-sigma visibility, black background, and no clamp. Step 0 uses the captured
effective parameter before any optimizer state or training draw. Callback work is outside native
training timing and cannot access or mutate optimizer state, gradients, training data, or the
schedule generator.

At each checkpoint, pool raw squared RGB error and channel count over all truth-support pixels in
all three held-out views, then define

`heldout_fg_psnr = -10*log10(max(SSE/count,1e-12))`.

Define normalized trapezoidal checkpoint AUC in dB over `[0,30,60,90,120]` by dividing the
weighted integral by 120. This held-out foreground AUC is the sole primary utility metric.

For candidate `x`, define paired per-seed `delta_auc_x=AUC_x-AUC_current`. Candidate utility
passes only when:

- mean paired `delta_auc_x >= +0.05 dB`;
- `delta_auc_x > 0` in at least two of three seeds;
- no seed has `delta_auc_x < -0.15 dB`.

The two candidates are judged separately against the same current baseline. One cannot rescue the
other. If both pass utility and safety, tangent displacement is preferred over the simpler unit
retraction only if its mean paired AUC advantage over unit retraction is at least `0.03 dB` and it
wins strictly in at least two of three seeds; otherwise unit retraction is the parsimonious
follow-up. This rule selects at most a confirmatory candidate, never a production default.

### Secondary metrics and safety

At every checkpoint serialize each held-out view's raw full-image RGB SSE/channel count, truth-
support RGB SSE/channel count, PSNRs derived from those values, repository `ssim` on the raw
unclamped predicted and GT colors, predicted color/alpha/accumulated-depth/expected-depth arrays,
truth arrays, field/covariance/render hashes, raw quaternion norms, and wall time. Pooled full-image
and foreground PSNR each sum their three views' raw SSE and counts before the logarithm. Pooled
and per-view PSNR use `-10*log10(max(SSE/count,1e-12))`. Pooled
SSIM is the unweighted arithmetic mean of the three per-view SSIM values; SSIM is never pooled by
pixels or recomputed on concatenated images.

At step 120 additionally report these pooled raw-count metrics:

- predicted expected depth is
  `predicted accumulated depth / clamp_min(predicted alpha,1e-6)`; normalized depth RMSE is the
  square root of the sum of squared predicted-versus-GT expected-depth error over every truth-
  support pixel in all three views divided by the total truth-support pixel count, then divided by
  the frozen scene extent;
- alpha IoU sums intersection counts for `predicted alpha>0.05` and truth support across all three
  views and divides once by the summed union count;
- truth-support coverage sums pixels satisfying both truth support and `predicted alpha>0.05`
  across all three views and divides once by the summed truth-support pixel count.

There is no parameter-to-GT quaternion, covariance, means, scale, opacity, or SH comparison: the
lifted and synthetic GT sets have no sealed row correspondence, and those non-decisional values
cannot enter interpretation.

For each candidate versus current, safety passes only if all hold:

- mean final foreground-PSNR delta `>=-0.05 dB` and no seed `<-0.15 dB`;
- mean final SSIM delta `>=-0.002` and no seed `<-0.005`;
- mean relative normalized-depth-RMSE regression `<=2%` and no seed `>5%`;
- mean alpha-IoU and coverage regressions each `<=0.01`, with no seed regression `>0.03`;
- finite parameters, losses, renders, metrics, complete checkpoints, exact schedules, fixed
  topology, and the candidate norm invariant all pass.

Every PSNR and pooled-SSIM delta above is `candidate-current` for the same seed. Relative
normalized-depth-RMSE regression is `(candidate-current)/current`. Alpha-IoU and coverage
regressions are `current-candidate`, so a positive value is worse. The mean is the frozen
arithmetic cross-seed mean; each no-seed condition applies to the corresponding paired seed value.

A zero current depth-RMSE denominator, empty truth support, missing checkpoint, non-finite value,
or source/schedule mismatch invalidates Phase B. Safety cannot rescue failed AUC utility, and AUC
cannot rescue failed safety. Runtime is descriptive only; there is no speed claim.

## Interpretation and stopping rules

- Algebra/contract failure: the harness did not establish the assumed gauge through the tested
  code path; stop as invalid without optimizer interpretation.
- Valid Phase A but failed materiality: ambient Adam's radial gauge was not materially implicated
  under the frozen conjunction; do not run Phase B or tune the screen.
- Phase A passes but neither Phase-B candidate passes utility and safety: retain current behavior;
  close quaternion retraction tuning for this CPU synthetic branch.
- Unit retraction alone passes: record it as the simpler confirmatory candidate. Do not infer a
  fully intrinsic optimizer or export policy.
- Tangent displacement passes the frozen preference rule: record it as the confirmatory candidate,
  while explicitly noting that its Adam moments remain ambient.
- Any positive result is restricted to this Depth-surface, fixed-topology, synthetic setup. A
  separate preregistered real/calibrated test and CUDA/gsplat parity audit are required before any
  default change. Density interaction requires its own experiment because clone/split/prune edits
  row identity and optimizer state.

No result here may justify a rank/planarity constraint, multiscale schedule, Lie-algebra optimizer,
Riemannian moment transport, quaternion LR sweep, alternate radial scales, export normalization,
or serializer change. Those are distinct branches.

## Source sealing, independent review, and append-only artifacts

Before an official seal, the complete harness, all focused tests, decision code, and Trainer
research seam must pass the repository's full CPU verification. An independent implementation
review must check the equations, operation ordering, held-out capability boundary, raw-evidence
completeness, and default-path exactness before any official attempt.

Seal creation must run, in order, `.venv/bin/python -m ruff check .`,
`.venv/bin/python -m ruff format --check .`, `.venv/bin/python -m pytest -q -m "not slow"`,
`.venv/bin/python scripts/docs_sync.py`, and `git diff --check`; it refuses on any nonzero exit and
records each literal command, exit status, complete stdout/stderr, and output SHA-256. Those
verification children may construct/render toy fixtures allowed by the focused-test rules. They
may not invoke the harness's `audit` or `run` actions, prepare official seeds/configurations, claim
an attempt marker, execute an official arm, or expose an official metric. This bounded toy-test
permission does not broaden either phase's scientific capability boundary.

The seal must bind this preregistration, harness, tests, every repository-owned loaded source,
revision and dirty diff, environment and dependency versions, effective commands/configurations,
and a source aggregate. Runtime must re-hash every bound source before atomically and exclusively
creating an attempt marker. An interrupted, invalid, or failed attempt remains consumed. Artifacts
are append-only and may never be overwritten, deleted, reconstructed from terminal output, or
silently repaired.

The seal's required `sha256` field is its canonical payload digest, not the byte hash of the
pretty-printed file. Construct the complete seal object without a `sha256` key, encode it as UTF-8
with `json.dumps(payload,sort_keys=True,separators=(",",":"),allow_nan=False)`, hash those exact
bytes with SHA-256, then insert the hexadecimal digest as `seal["sha256"]` before exclusive
serialization. Every seal load removes that one field, repeats the same canonical encoding, and
requires exact equality. The ordinary byte-level file SHA-256 is reported separately where useful
and must never be substituted for `seal["sha256"]` in the machine-review schema.

Fresh namespace:

- harness: `benchmarks/quaternion_gauge_ablation.py`;
- seal: `benchmarks/results/20260716_quaternion_gauge_SEAL.json`;
- seal artifact type: `quaternion_gauge_implementation_seal`;
- Phase-A marker: `benchmarks/results/20260716_quaternion_gauge_PHASE_A_ATTEMPT.json`;
- Phase-B marker: `benchmarks/results/20260716_quaternion_gauge_PHASE_B_ATTEMPT.json`;
- valid Phase-A artifact type: `quaternion_gauge_phase_a_audit`;
- invalid Phase-A artifact type: `quaternion_gauge_phase_a_invalid`;
- Phase-B artifact type: `quaternion_gauge_phase_b_ablation`;
- Phase-A scientist-review artifact type: `quaternion_gauge_phase_a_scientist_review`;
- valid Phase-A output: `<UTC>_cpu_quaternion_gauge_audit.json`;
- invalid Phase-A output: `<UTC>_cpu_quaternion_gauge_invalid.json`;
- authorized Phase-B output: `<UTC>_cpu_quaternion_gauge_ablation.json`;
- matching human notes: `_RESULT.md`;
- independent scientist reviews: matching `_AUDIT.md` plus machine-readable review JSON.

Every `<UTC>` token above is exactly a UTC timestamp formatted `YYYYMMDDTHHMMSSZ`. The Phase-A
output-prefix basename must therefore match `<UTC>_cpu_quaternion_gauge` exactly.

The sole official seal command is

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python benchmarks/quaternion_gauge_ablation.py seal --output benchmarks/results/20260716_quaternion_gauge_SEAL.json`.

The sole official Phase-A command is

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python benchmarks/quaternion_gauge_ablation.py audit --seal benchmarks/results/20260716_quaternion_gauge_SEAL.json --output-prefix benchmarks/results/<UTC>_cpu_quaternion_gauge`.

Before claiming the Phase-A marker, the harness derives and exclusively preflights all four paths
`<prefix>_audit.json`, `<prefix>_audit_RESULT.md`, `<prefix>_invalid.json`, and
`<prefix>_invalid_RESULT.md`; it refuses the attempt if any exists and records the prefix plus all
derived paths in the marker. A valid materiality pass or valid materiality failure writes only the
audit JSON/note pair. A failed algebra, representation, denominator, or validity prerequisite
writes only the invalid JSON/note pair and must not contain optimizer trajectories or materiality
statistics. An unexpected interruption may leave only the consumed marker. Every JSON and note is
written by exclusive temporary-file creation plus atomic rename and may not overwrite a path.

For a valid Phase-A output `<phase-a>`, the required human audit path is derived exactly as
`<phase-a-stem>_AUDIT.md`, and the required machine review path is
`<phase-a-stem>_AUDIT.json`. The machine review is strict JSON with exactly these keys:

```json
{
  "artifact_type": "quaternion_gauge_phase_a_scientist_review",
  "verdict": "pass",
  "phase_b_execution_clearance": true,
  "phase_a_sha256": "<sha256 of phase-a JSON>",
  "human_audit_sha256": "<sha256 of derived human audit markdown>",
  "seal_sha256": "<exact seal['sha256'] value from the verified seal>",
  "phase_a_attempt_sha256": "<sha256 of the Phase-A attempt marker>",
  "source_aggregate": "<verified seal source aggregate>"
}
```

No missing or additional key, alternate artifact type, non-`pass` verdict, false clearance, or
digest mismatch is accepted. The machine-review file's own SHA-256 is computed externally and
bound into the Phase-B marker; it is not self-referential JSON content.

Phase B additionally requires explicit `--phase-a <audited-output> --review <machine-review>`
arguments. Before consuming its marker, the harness must verify the preregistration, seal,
Phase-A artifact, derived human-audit, attempt-marker, and machine-review hashes; independently
recompute all Phase-A validity and materiality decisions from raw evidence; require the strict
review schema above; require the Phase-A decision to authorize Phase B; preflight both the exact
`_ablation.json` output and its `_RESULT.md` note; and refuse if either exists. The marker binds the
hash of every input plus both output paths. The harness re-hashes every input again before writing
the result. The supplied machine-review path must equal the derived `<phase-a-stem>_AUDIT.json`,
and the supplied output must end exactly in `_cpu_quaternion_gauge_ablation.json`. The sole Phase-B
form is

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python benchmarks/quaternion_gauge_ablation.py run --seal benchmarks/results/20260716_quaternion_gauge_SEAL.json --phase-a <audited-output> --review <machine-review> --output <fresh-output>`.

Every official artifact receives an independent `realtime-gs-results-audit` before a claim enters
`docs/EXPERIMENTS.md`, `docs/RESEARCH.md`, or any README/default. Phase A must be audited before
Phase B; Phase B must be audited before interpretation. A repair requires an append-only retry
preregistration, fresh seal, fresh marker names, and fresh output namespace while retaining all
failed artifacts. No post-outcome scene, seed, perturbation, subset, arm, threshold, LR, schedule,
metric, or gate change is permitted in this namespace.
