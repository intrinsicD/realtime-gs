# Preregistration: residual-responsibility density allocation

## Chronology, question, and claim boundary

Frozen at `2026-07-16T02:35:49+02:00`, after a read-only literature/source-design pass and
before any implementation, diagnostic, pilot, fitted seed, render, score, selection, training
arm, metric, timing sample, seal, or outcome for this experiment.

Pre-implementation executability amendment at `2026-07-16T02:57:53+02:00`, still before any
implementation, diagnostic, pilot, official-seed preparation, schedule or generator probe, fit,
lift, render, score, selection, training step, timing sample, seal, marker, or outcome. The
preceding preregistration had SHA-256
`626a35cf935fba198833567af10fa485c680837a48e67101ed206132647ec60f`; its independent
outcome-free executability review had SHA-256
`666e61b4dc8a41aabd68fa96710f30832236513e15b3eaa55d348d95ed04812d` and verdict `FAIL`.
This amendment resolves only the review's seven execution ambiguities: density scheduling and
the 160-step horizon, hook timing, score arithmetic, diagnostic schema, persistent identity and
noise ordering, held-out metric arithmetic/routing, and machine-enforced phase authorization. It
changes no question, literature adaptation, scene, seed, split, fit, lift, arm, score definition,
residual map, quota, stratum, wave, optimizer, learning rate, loss, checkpoint, metric family,
threshold, decision gate, interpretation, claim boundary, or stopping rule.

The causal question is deliberately narrower than “which densifier is best?” At identical CPU
initialization, optimization, primitive-count trajectory, clone/split quota, and sampled-view
schedule, does ranking candidate parents by native-compositing-weighted image residual improve
short Gaussian fitting relative to:

1. the repository's current screen-space position-gradient score; and
2. the same residual scores reassigned within predeclared scale/support strata?

The three arms are named `gradient_topB`, `error_topB`, and `stratum_shuffled`. Here `topB` means
the union of four fixed-quota stratum-wise top sets, not an unconstrained global top-k. Every arm
adds exactly `B=32` primitives at each of three waves and performs exactly 16 clone and 16 split
operations per wave. Parent identity is the only density-policy variable.

This is a CPU synthetic mechanism and time-to-quality experiment. It cannot establish real-scene
transfer, CUDA/gsplat behavior, perceptual quality, optimal budgets, an unrestricted-density
ranking, a production default, or a state-of-the-art claim. It does not test pruning,
relocation/MCMC, opacity reset, AbsGS, a smooth sampling distribution, redundancy recycling,
rate-distortion optimization, or Taylor saliency. Failure cannot be repaired by tuning a score,
quota, error map, stratum, seed, scene, or threshold inside this namespace.

## Literature grounding and adaptation boundary

The authenticated Scholar Inbox digest for `2026-07-12` through `2026-07-16` surfaced
[SalientGS](https://arxiv.org/abs/2607.11285) at relevance 91 and
[SpeedyGS](https://arxiv.org/abs/2607.12656) at relevance 88. Primary arXiv sources, rather than
digest summaries, govern the method statements below.

[Revising Densification](https://arxiv.org/abs/2404.06109) identifies a failure mode in which
absolute image error remains high while an infinitesimal center displacement changes the loss
little. It assigns a pixel error to Gaussian `i` in view `v` as

`E_vi = sum_p error_v(p) * w_vpi`

and uses `max_v E_vi` between density events. Its `w_vpi` is the Gaussian's alpha-compositing
contribution, not raw opacity or an unoccluded 2D kernel. It also studies growth caps and opacity
handling. This experiment adapts only that error-attribution and max-over-observed-views idea,
using the repository's literal renderer weights and a fixed-count causal control.

SalientGS independently supports treating multi-view underfit as an allocation signal, but its
importance/redundancy construction, robust quantile normalization, footprint approximation,
smooth importance-weighted MCMC sampling, relocation, pose optimization, and large fixed budget
are not implemented here. SpeedyGS separates content-aware structural formation from statistical
coding and couples learned pruning/quantization to a rate proxy. It motivates keeping allocation,
removal, and representation cost as separate questions; it does not establish this residual score
and is not evidence for Taylor pruning.

A first-order or second-order Taylor removal score is reserved for a later, separately
preregistered fixed-budget pruning experiment, and only after the present birth-allocation
mechanism is resolved. A future factorial may combine a successful residual birth policy with a
Taylor-ranked removal policy, but no Taylor score, prune, relocation, or capacity recycling is
permitted here.

## Frozen renderer arithmetic and diagnostic seam

For a pixel `p`, visible Gaussians are the Torch reference renderer's coarse-cull rows sorted
front-to-back by camera-space center depth. For sorted visible column `i`, the established renderer
computes

```text
q_pi     = (pixel_p - mean2d_i)^T covariance2d_i^-1 (pixel_p - mean2d_i)
g_pi     = kernel_support_weight(q_pi, mode="hard")
alpha_pi = clamp(opacity_i * g_pi, 0, 0.999)
T_p0     = 1
T_pi     = product_{j<i} (1 - alpha_pj + 1e-10)
w_pi     = alpha_pi * T_pi
```

and then

```text
color_p = sum_i w_pi * color_i + T_p,last * (1-alpha_p,last) * black
alpha_p = sum_i w_pi
depth_p = sum_i w_pi * z_i
```

The black background makes the last color term numerically zero, but its scalar weight must still
be audited. Depth is the native unnormalized accumulated depth. Global Gaussian identity is the
`visible` index corresponding to each sorted column. Residual responsibility uses exactly `w_pi`;
it must not use `alpha_pi`, `opacity_i*g_pi`, normalized `w_pi/sum_j w_pj`, a binary footprint,
projected radius, or a re-rendered approximation.

The only permitted renderer research seam is an additive, default-off
`RenderOutput.compositing_diagnostics: CompositingDiagnostics | None = None`, analogous to the
existing kernel diagnostics. `None` is the exact default; the established output values, fields,
graph topology, and arithmetic remain unchanged. On the CPU renderer, diagnostics-on exposes:

```text
w_chunks:                    list[detached Tensor(P,V)]
background_weight_chunks:    list[detached Tensor(P)]
gaussian_indices:             detached LongTensor(V), in sorted column order
visible_colors:               detached Tensor(V,3), exactly consumed by compositing
visible_depths:               detached Tensor(V), camera-space z consumed by compositing
pixel_intervals:              list[(start,end)], half-open flattened row-major bounds
```

For renderer row chunk `[r0,r1)`, `P=(r1-r0)*W`, its interval is `[r0*W,r1*W)`, and its rows are
flattened y-major then x-major. For `V>0`, background weight is literally
`T_exclusive[:,-1] * (1-alpha[:,-1])`; it is not `1-sum_i(w_i)` because the established epsilon
appears in preceding transmittance factors but not in the last factor. For `V=0`, diagnostics-on
still returns the normal row chunks with `w.shape=(P,0)`, background weights exactly one, and empty
index/color/depth tensors, while the normal render remains the established background image with
zero alpha/depth and `means2d=None`.

Every diagnostic tensor is detached when captured and is never used to form output or loss. The
harness concatenates chunks in increasing `pixel_intervals`, contracts the diagnostic immediately,
clears both chunk lists, and sets the output field back to `None`. Targets and error maps remain in
the harness; they never enter the rasterizer interface. The seam must not expose a mutable graph
tensor.

Before any official score is accepted, two independent paths must agree:

- reconstruct native color, alpha, and depth from diagnostic `w`, visible colors/depths, and
  background at `atol=2e-6, rtol=2e-5` for every audited render;
- recompute `q`, hard support, clamped alpha, exclusive transmittance, `w`, and background weight
  in a slow float64 explicit pixel/Gaussian loop and match native-dtype diagnostics at
  `atol=2e-6, rtol=2e-5` after casting; and
- on deterministic toy and official common-prefix renders, require diagnostics-off versus
  diagnostics-on equality of color/alpha/depth, scalar loss, every parameter gradient, and
  `means2d.grad` at maximum absolute error `0` on the established CPU path. Diagnostic tensors
  must be detached and finite.

Any mismatch invalidates Phase A. Reimplementing compositing only in the benchmark, silently
normalizing contributions, or changing renderer support/visibility/order is forbidden.

## Frozen data, split, initialization, and environment

- Execute with `CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`, and `MKL_NUM_THREADS=4`; set Torch
  intra-op threads to four and deterministic algorithms on. Use the repository `.venv` and the
  pure-Torch renderer only.
- Seeds are exactly `6,7,8`, chosen before implementation because `0..5` have already been used in
  earlier synthetic experiment series. There is no replacement seed.
- For each seed call
  `make_synthetic_scene(n_gaussians=40,n_cameras=12,image_size=48,seed=seed)` exactly once per
  preparation. Training indices are `[0,1,2,4,5,6,8,9,10]`; held-out indices are `[3,7,11]`.
  Physically subset to the nine training views before fitting, lifting, score construction, or
  optimization. Synthetic held-out images, cameras, and truth cannot enter any score, stratum,
  schedule, surgery, loss, gate, or stopping decision.
- Require `scene.masks is None`. All renders use black background, no random background, and the
  established hard SH-color floor, hard kernel support, and default visibility margin.
- Fit the nine training images once per preparation with the exact current native configuration:

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

- Call `fit_views(train_scene.images,config,seed=seed,masks=None)`. No optional StructSplat import
  is allowed. Hash images, cameras, fits, fit histories, sparse points, bounds, split maps, and RNG
  metadata.
- Produce exactly one current production Carve initialization per preparation:

```text
CarveLifter(
  grid_res=48, bounds_scale=0.5, min_views=2, hull_fraction=0.85,
  color_std_sigma=0.20, color_match_sigma=0.35, coverage_thresh=0.40,
  samples_per_ray=64, min_score=0.05, min_weight=0.05,
  merge=True, merge_voxel_scale=1.0, init_opacity=0.1, sh_degree=0,
)
```

  Call `lift(fitted,train_scene)` once. There is no alternate lift, raw unmerged tensor,
  relifting, post-lift filtering, initial cull, or held-out selection. Require nonempty finite
  fields and serialize their hashes and initial count `N0`.
- Phase B may deterministically repeat preparation only to reconstruct the sealed inputs. Every
  repeated fit/lift/history/tensor hash must equal Phase A exactly before an arm is created.

Record Python, Torch, NumPy, Pillow, OS/platform, CPU, logical CPU count, Torch thread counts,
environment variables, git revision, dirty diff, loaded-source hashes, and `/proc/loadavg` when
available. Wall and process CPU time are descriptive and cannot authorize, stop, reorder, or
select a run.

## Exact training objective and residual map

All arms execute 160 Adam updates with a precomputed training-view schedule. There are no masks or
regularizers. At step `t` with prediction `P_t` and training target `I_t`, the optimized loss is

```text
L1_t   = mean_{p,c} abs(P_t[p,c] - I_t[p,c])
SSIM_t = rtgs.core.metrics.ssim(P_t, I_t, window_size=11)
loss_t = 0.8 * L1_t + 0.2 * (1 - SSIM_t)
```

`ssim` is the repository's separable 11x11 Gaussian window with sigma 1.5, zero padding, and
constants `0.01^2` and `0.03^2`. Predictions are not clamped before training loss. The density
error map is intentionally the executable per-pixel L1 map, not a decomposition of scalar SSIM:

`e_t[p] = mean_{c in RGB} abs(stopgrad(P_t[p,c] - I_t[p,c]))`.

It is computed from the identical forward render used by `loss_t`, before Adam, with no clamp,
crop, quantile transform, threshold, smoothing, normalization, or gradient. This experiment does
not ask which residual map is optimal.

All arms use:

```text
TrainConfig(
  iterations=160, lr_means=1.6e-4, lr_quats=1e-3,
  lr_scales=5e-3, lr_opacity=5e-2, lr_sh=2.5e-3,
  lr_sh_rest=1.25e-4, ssim_lambda=0.2,
  rasterizer="torch", device="cpu", densify=True,
  density_strategy="classic",
  density=DensityConfig(
    start_iter=40, stop_iter=120, every=40,
    grad_threshold=2e-4, absgrad=False,
    split_scale_frac=0.01, split_factor=1.6,
    prune_opacity=-1.0, prune_scale_frac=float("inf"),
    max_gaussians=N0+96, opacity_reset_every=0,
    opacity_reset_value=0.011, revised_opacity=True,
    mcmc_noise_lr=500000.0,
  ),
  eval_every=20,
  target_sh_degree=3, sh_degree_interval=40,
  use_masks=False, outside_alpha_lambda=0.01,
  mask_alpha_lambda=0.05, random_background=False,
  opacity_reg=None, scale_reg=None, packed=False,
  antialiased=False, sh_color_activation="hard",
  kernel_support_mode="hard", visibility_margin_sigma=3.0,
  validate_render_finite=True, seed=seed,
)
```

The unused mask coefficients remain serialized but contribute exactly zero because masks are
disabled. The means learning-rate decay, Adam eps/order, active-SH schedule, optimizer state, and
all non-density Trainer behavior remain current. `iterations` remains 160 in Phase A as well as
Phase B; no 40-iteration configuration is permitted. Runtime `prune_scale_frac` is positive
infinity, but standard-JSON artifacts serialize that field as the exact string
`"positive_infinity"` in a normalized configuration record and reject non-standard `Infinity`.

The opt-in experiment density policy supplies the exact selected small/large parent rows and
bypasses gradient-threshold selection, significance-budget top-k, and pruning. It does not bypass
the controller's 40/80/120 scheduling or any established clone/split field arithmetic. There is no
density action at step 160. The default/`None` policy calls the existing threshold controller path
without constructing persistent IDs or changing RNG consumption and must remain bit-exact.

## Frozen schedules, streams, identities, and score windows

The implementation must add opt-in fixed-training-view, fixed-density-policy, and
`stop_before_density_after_step` Trainer seams. Every omitted/`None` path must remain bit-exact,
including final fields, all established histories, RNG consumption, and classic-controller
behavior. Before fitting or optimization, generate exactly 160 integer positions in `[0,9)` using
a CPU `torch.Generator` seeded with `900000+seed`, one `torch.randint` call per position. Materialize
and hash the complete list. A non-`None` fixed-view sequence must have exactly `iterations` integer
positions in bounds and replaces only the per-step `torch.randint`; it must not advance any Trainer,
surgery, or shuffle generator. Every arm consumes that list and no arm may draw a view from its
surgery generator. There is no random background.

Split standard-normal draws use a separate CPU generator reset immediately before each wave to
`930000 + 1000*seed + 100*wave`, where waves are `1,2,3`. The identical stream is used in all
arms. Because every arm has exactly 16 split parents per wave and the existing implementation
draws one `(16,3)` tensor for each of two child blocks, all arms consume exactly 96 standard-normal
values per wave in the same order. Selected split parents are first ordered by ascending persistent
ID. Draw the child-ordinal-0 `(16,3)` block first and the child-ordinal-1 block second; row `k` of
each block belongs to the `k`th ordered split parent. Values are scaled/rotated by that parent's
current scale/quaternion. No density randomness may advance the view or shuffle stream.

Assign persistent audit IDs `0..N0-1` in initial row order. Survivors retain current physical row
order and IDs. Within each operation, materialize selected parents by ascending persistent ID.
Append and assign monotonically increasing birth IDs in exact physical block order: clone children
(`block_code=0`), split children with child ordinal 0 (`block_code=1`), then split children with
child ordinal 1 (`block_code=2`). Within a block the ordering key is ascending parent persistent
ID, so the complete birth ordering key is `(wave,block_code,parent_persistent_id)`. A clone parent
survives and gets one new child ID; a split parent is removed and gets one child in each split
block. Record row-to-ID maps before and after every surgery. All score tie breaks are descending
numeric score, then ascending persistent ID; `torch.topk` tie order is not authoritative.

The three score windows are steps `1..40`, `41..80`, and `81..120`. Scores use each step's
pre-Adam render/backward and the persistent row existing at that step. Statistics reset to exact
zero after each surgery; newborns therefore receive no retroactive evidence. For Gaussian `i`:

```text
g_ti = ||d loss_t / d mean2d_ti||_2 * (max(width,height)/2)
G_i  = sum_{visible t} g_ti / max(number_of_coarse-visible t, 1)

r_ti = sum_p e_t[p] * w_tpi
R_i  = max_{t in window} r_ti

s_ti = sum_p w_tpi
S_i  = max_{t in window} s_ti
```

`G_i` is the repository controller's literal ordinary-gradient score on Torch; `absgrad=False`.
It is authoritative for `gradient_topB` in native float32: compute native float32 `g_ti`, use the
current float32 `index_add_` accumulator and count, and divide in float32 exactly as
`DensityController` does. Coarse visibility increments its denominator even if compact support is
zero. Serialize a separately accumulated float64 `G_i` only as audit evidence; it cannot rank,
select, or break a tie. The distinct-positive-`G` gate uses the authoritative native float32 values.

`R_i` and `S_i` use native float32 contributions but their authoritative decision reductions are
explicit float64. Concatenate detached diagnostic chunks by increasing half-open pixel interval,
let `e64=e.detach().reshape(-1).to(torch.float64)` and
`w64=cat(w_chunks,dim=0).to(torch.float64)`, then compute literally:

```text
r_t = (e64[:,None] * w64).sum(dim=0, dtype=torch.float64)
s_t = w64.sum(dim=0, dtype=torch.float64)
```

No native-float32 product, per-chunk regrouping, normalized contribution, or rounded value is
authoritative. The pooled assigned-residual numerator and denominator use the same float64 operands
and flattened order. `R_i` and `S_i` are zero when a row contributes nowhere. Exact max ties retain
the earliest step and then the lower physical training-view index for serialized argmax fields;
selection ties still use ascending persistent ID. Serialize native and float64 values, per-step
observations, maxima, argmax step/view, support, visibility count, and hashes.

Every training step has one exact order: zero gradients; consume the fixed view; render once; form
detached residual/support evidence from that render; form scalar loss; backward; accumulate
`G,R,S`, visibility, and argmax evidence against the pre-Adam row-to-ID map; run the six Adam
optimizers in current insertion order; append established history; decay the means learning rate;
then, only after completed steps 40/80/120, read post-Adam scale/opacity, construct strata and
selections, run the Phase-B-only common detached pre-surgery observer exactly where frozen below,
and perform surgery. Phase A instead stops at that boundary. Native evaluation and detached
checkpoint callbacks remain after surgery. The score state resets to exact zeros only after
successful surgery.

At a wave, scale and opacity are read after that step's Adam update, matching current surgery
timing. Let `scale_max_i=max(exp(log_scales_i))` and `extent` be the frozen training-scene extent.
An eligible parent has `visibility_count_i>0`, finite `G_i,R_i,S_i`, and `S_i>0`. It is `small`
when `scale_max_i <= 0.01*extent` and `large` otherwise, matching the controller's strict
split boundary. Within each operator class, sort eligible rows by ascending `(S_i,persistent_id)`;
the first `floor(n_class/2)` are `low_support` and the remainder are `high_support`. This yields
four exhaustive strata:

```text
small_low, small_high, large_low, large_high
```

Each arm selects exactly eight parents from each stratum. A stratum with fewer than eight eligible
rows invalidates the official attempt; there is no borrowing, backfill, boundary change, or rerun.

## Frozen arms and surgery semantics

At waves after completed steps `40,80,120`:

1. `gradient_topB`: choose the eight highest `G_i` in each stratum.
2. `error_topB`: choose the eight highest `R_i` in each stratum.
3. `stratum_shuffled`: within each stratum, take members in ascending persistent-ID order, create
   one `torch.randperm(n)` from a fresh CPU generator seeded with
   `910000 + 1000*seed + 100*wave + stratum_code`, where codes in the displayed order are
   `0,1,2,3`, and assign
   `R_shuffled[member_j] = R[member_perm[j]]`. Choose the eight highest reassigned scores.

There is no redraw or derangement repair even if a permutation contains fixed points. Hash every
member list, permutation, original/reassigned score vector, and selected set before surgery.

Use the current classic surgery field arithmetic unchanged, except that the frozen selected masks
replace threshold selection and pruning is disabled:

- small selections are cloned: parent survives and one bitwise parameter copy is appended;
- large selections are split: parent is removed; two parameter copies receive independent native
  standard-normal local offsets scaled by current scales and rotated by the current quaternion;
  every child scale is divided by `1.6`;
- preserve the repository's literal `revised_opacity=True` behavior: current clone rows retain the
  copied opacity, while current split children receive
  `1-sqrt(1-clamp(opacity,max=1-1e-6))` before logit conversion;
- for every parameter tensor, survivor `exp_avg`/`exp_avg_sq` rows are bitwise preserved, every
  appended moment row is exact zero, and the optimizer scalar step plus param-group LR, name, eps,
  and order remain current; and
- append survivors in current row order, then the ascending-parent-ID clone block, split child
  ordinal-0 block, and split child ordinal-1 block, with IDs and normal rows aligned exactly as
  frozen above.

Set `prune_opacity=-1`, `prune_scale_frac=inf`, `opacity_reset_every=0`, and
`max_gaussians=N0+96` in the experiment policy. No threshold eligibility, significance cull,
opacity reset, relocation, noise outside split children, or prune is allowed. Runtime assertions
must establish exactly this topology/count trajectory in every arm:

```text
before step-40 surgery: N0
after step-40 surgery:  N0 + 32  (16 clones, 16 splits, 0 prunes)
after step-80 surgery:  N0 + 64  (cumulative 32 clones, 32 splits)
after step-120 surgery: N0 + 96  (cumulative 48 clones, 48 splits)
through step 160:       N0 + 96
```

Thus the operation topology and count are fixed while parent/lineage identities remain the causal
treatment. Any count, operation-quota, schedule, RNG-consumption, active-SH, optimizer-step, or
checkpoint mismatch invalidates all arms for that seed.

## Phase A: pre-utility mechanism and executability gate

Phase A runs one common, no-surgery 40-step prefix per seed from the frozen initialization,
collects both scores, constructs all three counterfactual selections on the identical step-40
snapshot, and renders no trained candidate arm. It retains `TrainConfig(iterations=160,...)` and
therefore the 160-step means-LR horizon and 40-step SH interval; configuring 40 iterations is
forbidden. The opt-in `stop_before_density_after_step=40` seam stops after step-40 Adam, history,
means-LR decay, post-Adam scale/opacity read, strata construction, and all three selection hashes,
but before surgery, native evaluation, or checkpoint callback. The stop returns a detached model,
complete history/score evidence, and a pre-surgery state binding; it is not resumable.

The state binding hashes parameter order/values, row-to-ID map, every Adam scalar step and moment
tensor, param-group LR/name/eps/order, score accumulators/counts/maxima/argmax fields, sampled views,
active-SH history, loss/loss-term history, all selections/permutations, and the complete fixed-view
schedule. Phase A serializes complete raw evidence. Phase B is forbidden unless every
construction/parity invariant above and every gate below passes.

For every seed require:

- all four strata contain at least eight eligible rows, all decision values are finite, and both
  `G` and `R` have at least 16 distinct positive values over eligible rows;
- pooled assigned residual fraction
  `sum_{t,p} e_t[p]*sum_i w_tpi / sum_{t,p} e_t[p] >= 0.10`, with a finite positive denominator;
- the 32-parent `gradient_topB` versus `error_topB` Jaccard is `<0.80`;
- the 32-parent `error_topB` versus `stratum_shuffled` Jaccard is `<0.80`;
- at least 75% of eligible score assignments move to a different persistent ID under the four
  fixed permutations, pooled by count; and
- the sum of original `R_i` over `error_topB` parents is at least `1.01` times the corresponding
  sum for each of `gradient_topB` and `stratum_shuffled`, with finite positive denominators.

The final condition is a materiality check on a top-ranked signal, not a quality result. The
Jaccard conditions prove the causal arms differ; they do not imply one is better. Report Pearson
and stable-rank Spearman correlations, per-stratum overlaps, residual/support concentration,
score distributions, scale/opacity/visibility summaries, and selection lineage fields, but none
is an additional gate.

Phase A must also replay the exact 160-view schedule without training, prove its hash and bounds,
probe each split/shuffle stream for the frozen draw count without advancing official generators,
and verify deterministic toy fixtures remain bit-exact between omitted and explicit `None` fixed
view, density-policy, stop, and compositing-diagnostic seams. The classic-controller fixture must
compare final fields, established histories, density stats, optimizer surgery, and RNG-derived
outputs. A failed gate stops permanently under this preregistration. Do not adjust `B`, strata,
support handling, thresholds, error map, seed, or scene to make Phase B executable.

## Phase B: paired fixed-count utility test

Only after an independent strict results audit recomputes Phase A from raw evidence may Phase B
run all three arms. Each arm deterministically replays steps `1..40`; immediately before its first
surgery, require exact parameter, Adam-state, score-state, sampled-view, active-SH, loss-history,
row-to-ID, fixed-schedule, and selection hashes equal the audited Phase-A common prefix. Each arm
uses one persistent 160-update Trainer call and may not restart or restore around surgery. The arm
then applies its first surgery and continues through step 160. Execute arms in cyclic order:

```text
seed 6: gradient_topB, error_topB, stratum_shuffled
seed 7: error_topB, stratum_shuffled, gradient_topB
seed 8: stratum_shuffled, gradient_topB, error_topB
```

Render held-out metrics only through a detached read-only evaluator after Phase-B authorization
and at post-surgery/checkpoint steps `[40,60,80,100,120,140,160]`. It receives a detached Gaussian
snapshot and cannot access or mutate training tensors, loss, optimizer, controller, schedule,
generators, strata, selections, or decisions. Before training, render the frozen GT Gaussians on
held-out cameras and define fixed foreground support `M_v = truth_alpha_v > 0.05`.

The common pre-surgery step-40 metric is routed only in Phase B. At the first arm's step-40
pre-surgery hook, first freeze and hash the parameter/optimizer/score state, strata, and all three
counterfactual selections; then pass only a detached Gaussian snapshot to the held-out evaluator.
The evaluator returns nothing to the density policy. Bind this metric to the pre-surgery parameter
hash. Later arms must match that hash and reuse the bound common metric without another
pre-surgery held-out render. Each arm's distinct step-40 post-surgery metric is rendered only after
its surgery. Phase A never constructs held-out truth or renders a held-out metric.

At checkpoint `k`, detach the native float32 prediction and target, clamp each to `[0,1]` for
metrics only, cast each to float64 before subtraction, and pool raw foreground squared RGB error
over the three held-out views:

```text
SSE_k   = sum_{v,p,c} M_v[p] * (P_kv[p,c] - I_v[p,c])^2
COUNT   = 3 * sum_{v,p} M_v[p]
MSE_k   = SSE_k / COUNT
PSNR_k  = -10 * log10(max(MSE_k, 1e-12))
AUC     = sum_j 0.5*(PSNR_j+PSNR_{j+1})*(step_{j+1}-step_j) / 120
```

The AUC uses the seven post-surgery/checkpoint values from step 40 through 160. Also serialize the
common pre-surgery step-40 metric, but it cannot replace the post-surgery point. Raw SSE/count and
every per-view numerator are authoritative; rounded dB values are not.

`error_topB` succeeds only if, separately against both `gradient_topB` and
`stratum_shuffled`:

- mean paired AUC gain over seeds is at least `0.10 dB`; and
- the AUC gain is strictly positive in at least two of three seeds.

For each comparator construct the literal seed-ordered vector
`[AUC_error(seed)-AUC_comparator(seed) for seed in (6,7,8)]`; the reported mean is
`statistics.fmean` of that vector and wins use strict `>0.0` on its unrounded elements.
Both comparisons must pass. Beating gradient but not the shuffled mapping does not establish that
residual-to-parent correspondence caused the gain. Safety cannot rescue a failed primary gate.

Secondary metrics are final and checkpoint full-canvas PSNR, foreground-crop PSNR/SSIM, training
loss/PSNR, alpha coverage/IoU, normalized expected-depth RMSE, score and selection diagnostics,
gradient/error correlations, and process/wall time. Expected depth is
`accumulated_depth/clamp_min(alpha,1e-8)` and is evaluated only where both predicted and truth
alpha exceed `0.05`; RMSE is divided by scene extent. Alpha IoU thresholds both predicted and
truth alpha at `0.05`.

Every held-out view serializes its raw float64 RGB SSE/count, depth SSE/intersection count, alpha
intersection/union counts, truth-support count, crop bounds, and crop SSIM. Crop bounds and masking
are exactly `rtgs.core.metrics.masked_crop(value,M_v,margin_fraction=0.05)` using the same fixed
binary truth support for clamped prediction and target; at 48x48 its margin expression remains the
repository's `max(1,round(max(H,W)*0.05))`. Crop SSIM is exactly
`rtgs.core.metrics.ssim(pred_crop,target_crop,window_size=11)`, and a seed's crop SSIM is the
arithmetic mean of its three per-view values. Predicted and truth expected depths are each their
native accumulated depth divided by `clamp_min(alpha,1e-8)` before the supported difference is
formed in float64. A seed's normalized depth RMSE, alpha IoU, and foreground coverage are the
pooled raw-count values:

```text
depth_seed = sqrt(sum_v depth_SSE_v / sum_v depth_intersection_count_v) / extent
iou_seed   = sum_v alpha_intersection_v / sum_v alpha_union_v
coverage   = sum_v alpha_intersection_v / sum_v truth_support_count_v
```

The depth denominator, alpha union, and truth-support denominator must be finite and positive. No
mean of per-view depth RMSE, IoU, or coverage is authoritative.

For `error_topB` relative to each comparator, safety additionally requires mean final foreground
PSNR difference at least `-0.10 dB` and no seed below `-0.25 dB`; mean crop-SSIM difference at
least `-0.002` and no seed below `-0.005`; mean normalized depth-RMSE relative regression at most
`2%`; and mean alpha-IoU regression at most `0.02`. Empty depth intersections, non-finite values,
or violated structural invariants invalidate the affected official attempt rather than count as
a scientific loss.

These safety quantities are computed separately for each comparator with no rounded inputs. In
seed order `(6,7,8)`, form paired vectors
`fg_delta=error_final_fg_psnr-comparator_final_fg_psnr`,
`ssim_delta=error_final_crop_ssim-comparator_final_crop_ssim`, and
`iou_regression=comparator_final_iou-error_final_iou`. Each “mean” is literally
`statistics.fmean` of its paired vector; the per-seed floors apply to the corresponding delta
vector. Depth relative regression is exactly
`(fmean(error_final_depth)-fmean(comparator_final_depth)) /
fmean(comparator_final_depth)` and requires a finite positive comparator mean. Alpha-IoU
regression is the absolute paired drop above, not a relative percentage. Every inclusive safety
threshold must pass against both comparators.

## Interpretation and stopping rules

- Phase-A failure means the frozen CPU setup does not provide a sufficiently material,
  attributable, and distinct residual-allocation intervention. Stop; do not tune it.
- Phase-A pass plus both Phase-B primary comparisons and all safety guards means native residual
  responsibility is a promising parent-allocation signal for this narrow CPU Carve initialization.
  It still requires preregistered real calibrated scenes, GPU parity, perceptual metrics, and a
  broader budget test before any default discussion.
- If `error_topB` beats `gradient_topB` but not `stratum_shuffled`, the result does not isolate the
  residual mapping; record the negative control failure.
- If it beats the shuffled arm but not gradient, residual mapping is active but not better than
  the current score here.
- If early AUC passes but safety fails, record an early-speed/late-quality trade-off, not success.
- If a comparator wins, it may motivate a new outcome-free confirmation but cannot change a
  default from this experiment. Timing is always descriptive.

No result authorizes an error-map, normalization, max-versus-mean, quota, wave, budget, shuffle,
or seed sweep. Taylor pruning remains a separate future combination even if this experiment
succeeds.

## Tamper-resistant seal, markers, and audit

No seal or official command is run while drafting this preregistration. Before sealing, a reviewer
who did not author the implementation must inspect the complete amended protocol, harness, seams,
and focused tests and write
`benchmarks/results/20260716_residual_responsibility_density_IMPLEMENTATION_REVIEW.md`. Only an
exact `Verdict: PASS` with `Unresolved findings: none` permits sealing; the review is in the sealed
manifest.

Seal creation hashes the complete sealed path list, per-file digests, canonical source aggregate,
amended preregistration, preregistration review, implementation review, and default-seam evidence
before verification. It then runs these literal commands in order, recording complete stdout,
stderr, exit status, duration, and output digests:

```text
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m pytest -q -m "not slow"
.venv/bin/python scripts/docs_sync.py
git diff --check
```

It rehashes the same complete snapshot after verification and refuses unless paths, every digest,
aggregate, all three document bindings, and default-seam evidence are unchanged. The seal binds the
verified pre-snapshot, repository-owned loaded source, environment, git revision and dirty diff,
the exact verification records, and declared absence of prior official outputs. It refuses
overwrite, a nonfixed output path, any nonzero command, source drift, and any unsealed loaded
repository source. Seal creation may execute bounded nonofficial toy tests but may not prepare an
official seed, probe an official schedule/generator, fit, lift, render an official scene, collect a
score, create a phase marker, or execute an arm.

Official namespace:

```text
harness: benchmarks/residual_responsibility_density_ablation.py
prereg:  benchmarks/results/20260716_residual_responsibility_density_PREREG.md
prereg review:
         benchmarks/results/20260716_residual_responsibility_density_PREREG_REVIEW.md
implementation review:
         benchmarks/results/20260716_residual_responsibility_density_IMPLEMENTATION_REVIEW.md
seal:    benchmarks/results/20260716_residual_responsibility_density_SEAL.json
phase A marker:
         benchmarks/results/20260716_residual_responsibility_density_PHASE_A_ATTEMPT.json
phase B marker:
         benchmarks/results/20260716_residual_responsibility_density_PHASE_B_ATTEMPT.json
phase A result:
         benchmarks/results/<UTC>_cpu_residual_responsibility_density_audit.json
phase A independent audit:
         benchmarks/results/<phase-A-result-stem>_AUDIT.json
phase B result:
         benchmarks/results/<UTC>_cpu_residual_responsibility_density_ablation.json
```

The only official harness commands are:

```text
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python benchmarks/residual_responsibility_density_ablation.py seal \
  --output benchmarks/results/20260716_residual_responsibility_density_SEAL.json

env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python benchmarks/residual_responsibility_density_ablation.py phase-a \
  --seal benchmarks/results/20260716_residual_responsibility_density_SEAL.json \
  --output benchmarks/results/<UTC>_cpu_residual_responsibility_density_audit.json

env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python benchmarks/residual_responsibility_density_ablation.py phase-b \
  --seal benchmarks/results/20260716_residual_responsibility_density_SEAL.json \
  --phase-a-result benchmarks/results/<exact-phase-A-result>.json \
  --phase-a-audit benchmarks/results/<exact-phase-A-result-stem>_AUDIT.json \
  --output benchmarks/results/<UTC>_cpu_residual_responsibility_density_ablation.json
```

`<UTC>` must match `YYYYMMDDTHHMMSSZ`; each output parent and complete basename are validated before
marker creation. Every action requires the fixed resolved seal path; an equivalent copied seal,
implicit latest/glob discovery, alternate mode, or alternate output namespace is rejected.

Each marker is created atomically with exclusive creation before the first fit, lift, diagnostic,
schedule or generator probe, or arm belonging to that phase. Its payload binds phase, exact output
path, seal digest/source aggregate, command, and environment. The harness immediately re-reads and
binds the exact marker payload, raw-file digest, and canonical-payload digest, then rehashes them
before decision acceptance and artifact serialization. A marker is consumed by interruption,
post-creation drift, or failure.

Phase A receives an independent `realtime-gs-results-audit` review that recomputes all identities,
native/float64 scores, residual evidence, permutations, strata, selected sets, stream counts, and
gates from raw serialized evidence. It writes standard JSON at the exact derived Phase-A audit path
with this machine schema:

```text
artifact_type: "residual_responsibility_density_phase_a_results_audit"
verdict:       "PASS" | "FAIL"
unresolved_findings: list
auditor:       independent identity/provenance record
bindings:
  preregistration_sha256
  seal_sha256
  phase_a_marker_sha256
  phase_a_marker_payload_sha256
  phase_a_result_sha256
  phase_a_result_payload_sha256
recomputed:
  per-seed construction/parity invariants, raw-evidence hashes, and every frozen gate/decision
```

Phase B accepts only exact `PASS`, an empty unresolved list, complete finite recomputation, and
literal equality between the result's, audit's, and its own independently recomputed evidence and
decisions. Before creating its marker, Phase B obtains the sole authorized Phase-A result path from
the fixed Phase-A marker payload; requires `--phase-a-result` to resolve to that path and
`--phase-a-audit` to resolve to its exact derived audit path; rehashes the amended preregistration,
fixed seal, Phase-A marker/payload, Phase-A result/payload, and audit/payload; verifies every schema
binding; and recomputes every Phase-A authorization gate from raw result evidence. Any mismatch
refuses before Phase-B marker creation and cannot be bypassed. The Phase-B marker records all of
those exact file and canonical-payload digests. Phase B rehashes every authorization input and its
own marker again before decisions and serialization. Phase B receives a second independent results
audit before any claim enters documentation.

Never overwrite or delete a seal, marker, result, or review. Any implementation change after seal,
failed official attempt, insufficient raw evidence, or audit rejection requires a fresh
append-only preregistration, seal, marker names, and output namespace. No known or partial output
may select a repair threshold or scientific choice.
