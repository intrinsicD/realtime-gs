# Preregistration: Carve moment merge versus exact-count controls

## Chronology, question, and literature boundary

Frozen at `2026-07-15T23:49:00+02:00`, before any implementation, diagnostic, pilot, or
outcome for this experiment. The question is whether Carve's current per-voxel moment merge
preserves useful information and improves fixed-budget refinement, or whether its benefit is
explained by reducing the primitive count. The comparisons are two deterministic, exact-count
controls constructed from the same unmerged Carve tensor: one representative per occupied voxel
and a global importance prune.

Pre-implementation clarification at `2026-07-15T23:52:00+02:00`, still before any code or outcome:
independent executability review noted that Trainer's native checkpoints evaluate the physically
subset training scene, not held-out cameras. Phase B below therefore freezes a detached read-only
checkpoint callback and an explicit GT-alpha foreground mask for the held-out AUC; no scene, arm,
threshold, or success gate changed.

A second pre-implementation clarification at `2026-07-15T23:55:00+02:00` replaces an impossible
zero-iteration schedule check with an isolated 120-draw generator probe and explicitly exempts the
one diagnostic `merge=True` parity lift from the no-recarving rule. It also requires render
materiality against both exact-count controls. No implementation or outcome had been accessed.

The 2026-07-12 through 2026-07-15 Scholar Inbox digest motivated the control structure.
[SalientGS](https://arxiv.org/abs/2607.11285) reallocates Gaussian capacity using multi-view
underfit/redundancy signals, while [SpeedyGS](https://arxiv.org/abs/2607.12656) jointly considers
pruning and rate-distortion-aware structural formation. They are analogies for separating
allocation from representation, not evidence for Carve merging, the frozen importance score, or
any expected outcome here. This experiment implements neither paper and makes no SOTA claim.

This is a CPU synthetic, fixed-topology mechanism/utility test. It cannot establish real-scene
utility, exposure/occlusion robustness, same-view versus cross-view fusion effects (the current
Carve tensor has no source-view IDs), density-control interaction, CUDA/gsplat performance, or a
production-default change.

## Frozen data and common preparation

- CPU-only Torch reference renderer, exactly four Torch/OMP/MKL threads, deterministic algorithms,
  `CUDA_VISIBLE_DEVICES=""`, and seeds `0,1,2`.
- Per seed, `make_synthetic_scene(n_gaussians=40, n_cameras=12, image_size=48, seed=seed)`.
  Training views are `[0,1,2,4,5,6,8,9,10]`; held-out views are `[3,7,11]`. The scene is physically
  subset to the nine training views before stage-1 fitting and lifting. Held-out images and metrics
  cannot enter any arm construction, threshold, or optimization loss.
- One shared native stage-1 fit per seed: `n_gaussians=150`, `max_gaussians=5000`,
  `iterations=120`, `adaptive_density=True`, `growth_waves=5`, `relocate_fraction=0`, `lr=1e-2`,
  `grad_init_mix=0.7`, `row_chunk=64`, `log_every=50`, `convergence_patience=0`,
  `convergence_tol=0.05`, and `convergence_check_every=25`.
- Exactly one raw Carve lift per seed, using `CarveLifter(grid_res=48, bounds_scale=0.5,
  min_views=2, hull_fraction=0.85, color_std_sigma=0.20, color_match_sigma=0.35,
  coverage_thresh=0.40, samples_per_ray=64, min_score=0.05, min_weight=0.05, merge=False,
  merge_voxel_scale=1.0, init_opacity=0.1, sh_degree=0)`. No arm may rerun fitting or carving.
- Exactly one additional `merge=True` Carve lift from the same frozen fitted tensors is permitted
  only for the Phase-A parity identity below. Its output cannot supply, replace, filter, reorder,
  or otherwise influence `raw` or any arm.
- Let `center, extent = train_scene.center_and_extent()`, `half=extent*0.5`, and
  `voxel_size=(2*half/48)*1.0`, matching `CarveLifter`. Serialize and assert shared scene, camera,
  image, fitted-set, fit-history, raw-tensor, and raw-order hashes before constructing any arm.

The current implementation hashes merge cells against the world origin, not the carving lower
bound: `key_i=floor(raw.means_i/voxel_size)`. The official harness must use this literal key and
must prove that its group IDs and moment result reproduce
`merge_by_voxel(raw, voxel_size, opacity_mode="mean")`. Changing the hash origin, using a second
lift, or inspecting held-out outcomes invalidates the experiment.

## Frozen equal-count arms

For raw primitive `i`, define the current merge weight in float64 for auditing and in native dtype
for construction as
`w_i = clamp_min(raw.opacity_i * product(exp(raw.log_scales_i)), 1e-12)`.
Let `K` be the number of unique world-origin voxel keys. All trained arms contain exactly `K`
primitives.

1. `moment` (current Carve behavior): call
   `merge_by_voxel(raw, voxel_size, opacity_mode="mean")`. For every group, this uses `w_i` to
   compute the weighted mean, normalized mixture covariance including the between-mean term, and
   weighted SH; opacity is the `w_i`-weighted mean, clamped to `[0.01,0.995]`.
2. `voxel_representative`: in each identical voxel group retain the raw primitive with greatest
   `w_i`; break exact ties by lowest original raw index. Emit representatives in the sorted unique
   key/group order returned by `torch.unique(..., dim=0, return_inverse=True)`. All retained fields
   are copied bitwise from `raw`.
3. `global_budget_prune`: retain the globally greatest `K` values of `w_i`, breaking ties by lowest
   original raw index; after selection, emit retained primitives in increasing original-index
   order. All retained fields are copied bitwise from `raw`.

`raw_keep_all` is evaluated at initialization as a transparent count-confounded reference but is
not trained and cannot enter a success gate. There is no weight, voxel, count, opacity, pruning,
or tie-break sweep. The two controls deliberately isolate complementary alternatives: preserving
cell coverage without moments, and preserving the globally largest components without cell
coverage. They do not isolate source provenance because Carve currently discards it.

## Phase A: materiality and construction-preservation audit

Phase A constructs all arms, renders initialization diagnostics under no-grad, and performs no
refinement. All gates below must pass independently in every seed before Phase B is authorized.

### Structural validity and exact-count invariants

- Raw fields are finite; `raw.n >= 500`; `K >= 100`; all three arms have exactly `K` finite
  primitives; `1 <= K < raw.n`; group IDs cover `[0,K)` without gaps.
- Recomputing current Carve `merge=True` from the already frozen fitted tensors yields a tensor
  hash exactly equal to `moment`; this parity lift is diagnostic only and may not replace `raw` as
  the control source.
- Representative indices are unique, belong to their stated groups, attain the group maximum
  weight, obey the lowest-index tie rule, and reproduce their raw fields bitwise. Global-prune
  indices are unique, reproduce raw fields bitwise, and equal the frozen lexicographic top-`K` set.
- Arm construction reads only raw fields, voxel size, and original indices. Hash arm indices,
  group keys/IDs/counts, weights, and every output field.

### Moment-preservation identities

Recompute each group independently in float64 from the raw tensor. The serialized maximum absolute
and relative errors must be finite and pass `atol=2e-6, rtol=2e-5` against the native result for:

- weighted mean `sum(w_i*mu_i)/sum(w_i)`;
- normalized second central moment
  `sum(w_i*(Sigma_i+(mu_i-mu)(mu_i-mu)^T))/sum(w_i)` versus `moment.covariance()`;
- weighted SH and weighted-mean opacity after the documented clamp.

Also require every reconstructed covariance to be symmetric within `2e-6`, positive definite,
and every moment mean to lie inside its group's axis-aligned raw-mean bounds within `2e-6`.
These are normalized moment identities, not conservation of unnormalized opacity-volume mass.

### Materiality gates

Report raw/group count distributions, singleton/multi-member groups, raw primitives in
multi-member groups, compression, within-group center/color/covariance dispersion, selected-index
overlap, and train/held-out initialization metrics for all four reporting arms. Phase B requires:

- `1 - K/raw.n >= 0.10`;
- at least `50` multi-member groups and at least `15%` of raw primitives in multi-member groups;
- `voxel_representative` and `global_budget_prune` selected-index Jaccard `< 0.95`, proving the two
  controls are not effectively identical;
- against **each** exact-count control on the nine training views, raw-summed
  `sum(abs(render(moment).color-render(control).color)) /
  sum(abs(render(raw_keep_all).color-target)) >= 0.005`.

The render ratio establishes that moment construction is not numerically inert; it is not a
quality claim. A zero/non-finite denominator or failed identity invalidates Phase A. If any
materiality gate fails, stop without altering grid resolution, scene framing, seeds, or thresholds.

## Phase B: paired fixed-topology time-to-quality test

Only after Phase A passes and an independent strict scientist audit binds its raw evidence, train
the three exact-count arms for 120 steps. Use `TrainConfig(iterations=120, lr_means=1.6e-4,
lr_quats=1e-3, lr_scales=5e-3, lr_opacity=5e-2, lr_sh=2.5e-3,
lr_sh_rest=1.25e-4, ssim_lambda=0.2, rasterizer="torch", device="cpu", densify=False,
eval_every=30, target_sh_degree=3, sh_degree_interval=30, use_masks=False,
outside_alpha_lambda=0.01, mask_alpha_lambda=0.05, random_background=False, opacity_reg=None,
scale_reg=None, packed=False, antialiased=False, sh_color_activation="hard",
kernel_support_mode="hard", visibility_margin_sigma=3.0, validate_render_finite=True, seed=seed)`.

Before training, render the frozen GT Gaussians with the default hard renderer and define held-out
foreground support as `truth.alpha > 0.05`. Evaluate masked held-out foreground PSNR at step 0 and
at steps `[30,60,90,120]`; full-image PSNR is secondary. Because `Trainer`'s native checkpoint
metric sees only the physically subset training scene, implement a zero-default, read-only
checkpoint callback on `Trainer.train` for this experiment. After the optimizer step at each frozen
checkpoint, it receives only a detached cloned Gaussian snapshot and the integer step; it may render
held-out views under no-grad but cannot access/mutate optimizer state, gradients, the training scene,
or schedule generator. Default/no-callback behavior must remain bit-exact and CPU-tested. The
native Trainer checkpoint PSNR remains train-only/reporting and must not be substituted for the
held-out primary. Define the per-seed normalized trapezoidal checkpoint AUC in dB:

`AUC = sum_j 0.5*(PSNR_j+PSNR_(j+1))*(step_(j+1)-step_j)/120`,

over fixed steps `[0,30,60,90,120]`. This checkpoint AUC, not wall-clock timing or final PSNR, is
the primary. The moment-merge hypothesis succeeds only if, against **each** exact-count control,
its mean paired AUC gain is at least `0.10 dB` and it wins strictly in at least two of three seeds.
Both pairwise requirements must pass; one control cannot be ignored after outcomes are known.

Final held-out PSNR/SSIM, normalized expected-depth RMSE against the frozen GT renderer, alpha IoU,
foreground coverage, training loss, initialization metrics, per-checkpoint values, and wall time
are secondary. Safety requires moment versus each control: mean final PSNR at least `-0.10 dB`
and no seed below `-0.25 dB`; mean SSIM at least `-0.002` and no seed below `-0.005`; mean depth
RMSE regression at most `2%`; mean alpha-IoU and coverage regressions at most `0.02`. Safety cannot
rescue failed AUC utility.

All arms must begin from their sealed Phase-A hashes; have equal primitive and parameter counts;
visit the exact same 120-view schedule; have identical active-SH/checkpoint schedules; retain fixed
topology; and produce finite parameters, renders, losses, and metrics. The harness must verify a
pure isolated-generator schedule probe: for each arm/hash, instantiate the exact frozen local
Torch generator state used by Trainer, draw 120 local train-view positions without training, and
require all three integer schedules and hashes to be identical. The official training histories
must equal that frozen schedule. Serialize complete raw numerators and per-view metrics rather than
only rounded summaries. Training order is cyclically counterbalanced by seed:
`0: moment,voxel_representative,global_budget_prune`; `1: voxel_representative,
global_budget_prune,moment`; `2: global_budget_prune,moment,voxel_representative`.

## Interpretation, stopping rule, and append-only artifacts

- Phase-A failure: current merging is not sufficiently material or the implementation does not
  reproduce its stated identities; do not run utility arms or tune this protocol.
- Phase-A pass and Phase-B moment success: moment reduction is promising relative to both tested
  count controls in this narrow synthetic setup; real/calibrated, provenance-aware, and
  density-enabled confirmation is still required before a default change.
- One or both controls match/beat moment: the current merge has no demonstrated distinct
  time-to-quality benefit here. If a control exceeds moment by the same `0.10 dB`, 2/3-seed rule
  while passing safety, record it only as the next confirmatory candidate; do not replace the
  default from this experiment.
- AUC gain without safety is an early-optimization trade-off, not a success. Timing differences are
  descriptive because exact counts equalize the dominant renderer load but CPU wall time remains
  noisy.

Before Phase A, the complete harness, tests, protocol, decision code, and all repository-owned
loaded source must pass full verification and be bound into a source/environment seal. Use fixed,
atomically exclusive attempt markers for Phase A and Phase B; refuse overwrite; bind Phase B to
the preregistration, seal, Phase-A output, and independent audit hashes; recompute every gate from
raw evidence before authorization. Official namespace:

- harness `benchmarks/carve_merge_controls_ablation.py`;
- seal `benchmarks/results/20260715_carve_merge_controls_SEAL.json`;
- markers `20260715_carve_merge_controls_PHASE_A_ATTEMPT.json` and
  `20260715_carve_merge_controls_PHASE_B_ATTEMPT.json`;
- outputs `<UTC>_cpu_carve_merge_controls_audit.json` and, only after authorization,
  `<UTC>_cpu_carve_merge_controls_ablation.json`.

Every official result receives an independent `realtime-gs-results-audit` review and a dated
experiment-log entry. Any failed official attempt requires an append-only retry preregistration,
fresh seal, fresh marker names, and fresh output namespace. No output permits a post-outcome
weight, grid, count, schedule, gate, scene, or seed sweep.
