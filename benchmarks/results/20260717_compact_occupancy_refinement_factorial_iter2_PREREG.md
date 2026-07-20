# Compact occupancy-point refinement factorial iter2 — preregistration — 2026-07-17

## Status, repair scope, and question

This is a fresh, once-only repair experiment for the consumed protocol failure documented in
`20260717_compact_occupancy_refinement_factorial_FAILURE_AUDIT.md` (SHA-256
`67bf419e696273a7b47d729b7e0c07f5afb468e297568bfc694e6ddec5c0ccc7`). The first attempt
stopped while constructing evaluation seed 76503/view C0039: the largest float32 deviate below
one rounded through `u * 3599 + 1282` to the fitted window's exclusive upper x coordinate 4881.
No worker or optimizer arm ran. Its `NO_REFINEMENT_TARGET_PROMOTION` label is retired as a
scientific decision and means only `promotion_authorized=false` for that failed attempt.

Iter2 changes only the continuous uniform-coordinate primitive, failure receipt, RNG namespaces,
and append-only lifecycle. It asks the original two questions without changing their arms,
budgets, metrics, minimum effects, or gates:

1. Does a deterministic balanced-cycle view schedule improve equal-view compact-teacher fitting
   relative to the current IID schedule?
2. At identical sampled coordinates, does optimizing the active continuous Gaussian-point
   proposal submeasure improve occupancy-region fitting relative to importance-correcting those
   samples back to uniform continuous image area?

No source RGB, dense target image, or mask may be opened by bank construction, optimization, or
metric evaluation. Exact color targets come only from the frozen compact 2D Gaussian teachers.
The experiment keeps
$N_{\mathrm{init}}^{3D}=N_{\mathrm{opt}}^{3D}=835$; split, clone, merge, prune, relocation, and
all other density control remain deferred. A positive result authorizes only a later fresh
variable-$N_{\mathrm{opt}}^{3D}$ experiment, never a default.

The original exploratory seed-76201 observations remain disclosed but non-official: D/B was
`0.802971` for final $J_Q$, `0.893980` for AUC-derived $J_Q$, and `0.949092` for final $J_U$;
balanced scheduling alone had B/A final $J_U=1.023196$. No iter2 threshold, budget, arm, or seed
was chosen after seeing any iter2 outcome. No iter2 official bank or worker may be constructed
before an independent implementation review passes and an immutable seal is published.

## Endpoint-safe repair and test boundary

The continuous uniform branch retains the exact number and order of RNG draws. Given a generated
`unit_xy` in `[0,1)` and fitted window `(x,y,w,h)`, it still computes the affine coordinate in the
field dtype, then clamps each result to the closed representable interval

$$[(x,y),\;\operatorname{nextafter}((x+w,y+h),(x,y))].$$

This is the dtype predecessor of the mathematical exclusive upper endpoint. The repair never
resamples, discards, duplicates, or reweights an attempt. Uniform target density remains
`1/(w*h)`, null semantics are unchanged, and mixed-proposal importance remains bounded by
`1/eta=4` for `eta=0.25`.

Focused tests must force the largest representable float32 and float64 values below one on both
axes, with zero-origin and translated native-scale windows. They require finite/direct/active
uniform draws strictly inside the half-open window, positive finite proposal density, and finite
mixed-proposal importance at most four. Tests use only the isolated training seeds
`991601,991602,991603` and evaluation seeds `991701,991702`; they may not invoke any official
iter2 seed. Test banks carry `seed_domain=focused_test`; official workers accept only
`seed_domain=official_iter2`.

If any bank invariant fails, the finite terminal failure must set
`scientific_decision=UNAVAILABLE` and `promotion_authorized=false` and retain a structured
diagnostic: evaluation seed/domain, view, kind, derived generator seed, first failing index and
coordinate, fitted window, predicate counts, and tensor hashes. Weakening the fully active/direct
uniform-bank guard is forbidden.

## Frozen inputs and proposal

- Exact teachers/cameras:
  `runs/compact_masked_bundle_640_20260717/reconstruction_inputs`.
- Center-occupancy proxy:
  `runs/compact_occupancy_scalar_ablation_20260717/proxy_bundles/center`.
- Common initialization:
  `runs/compact_occupancy_scalar_ablation_20260717/stage_b/center/gaussians.ply`, SHA-256
  `0cf0340117739bb4b0491ff9c90d8d4b622b57a57f6bf8e6a3cfc9984b5c416e`.
- Ordered views: `C0001,C0008,C0014,C0021,C0026,C0031,C0039`.
- Current per-view counts are $m_{\mathrm{opt},i}^{2D}=640$, but every interface and receipt must
  retain the variable-length list and $\sum_i m_{\mathrm{opt},i}^{2D}$ without an equality
  assumption.

For teacher component $j$ in view $i$, the proposal amplitude is the exact float32 product
$b_{ij}=a_{ij}o_{ij}$ of teacher amplitude and aligned center-occupancy scalar. Proposal colors
are unused constants. Before multiplication, raw teacher/proxy view order, IDs, camera, dtype,
canvas, fit window, blend/coordinate/support semantics, $m_{\mathrm{init},i}^{2D}$,
$m_{\mathrm{opt},i}^{2D}$, means, log-scales, rotations, and optional filter variance must match
exactly; each scalar must be finite in `[0,1]`.

Let $u_i(x)=1/A_i$, $D_i^o(x)=\sum_jb_{ij}K_{ij}(x)$, and
$M_i^o=\sum_j b_{ij}2\pi\sqrt{\det\Sigma_{ij}}$. The rejection-thinned active density is
$g_i(x)=D_i^o(x)/M_i^o$ and the mixed active proposal subdensity is
$q_i(x)=0.25u_i(x)+0.75g_i(x)$. Rejected Gaussian attempts remain explicit nulls.

- `uniform` target: active importance $u_i/q_i$, null weight zero.
- `proposal_attempt` target: active importance one, null weight zero. This estimates the
  unnormalized attempt risk, not a normalized occupancy probability.

No active-count normalization, estimated acceptance normalization, or null resampling is allowed.

## Frozen factorial, RNG, runtime, and budget

| Arm | View schedule | Target measure |
| --- | --- | --- |
| A | `iid` | `uniform` |
| B | `balanced_cycle` | `uniform` |
| C | `iid` | `proposal_attempt` |
| D | `balanced_cycle` | `proposal_attempt` |

Official training seeds are `76601,76602,76603`; paired evaluation seeds are
`76701,76702,76703`. These sets are fresh and disjoint from the consumed first attempt, excluded
dry seed, and focused-test domains. Every arm runs 140 updates with 128 attempts/update,
`eta=0.25`, checkpoints `0,35,70,140`, and exactly 20 complete seven-view cycles in balanced
arms. Each official seed/view has a newly generated 4096-attempt uniform bank and 4096-attempt
proposal bank. Neither partial first-attempt archive may be copied or queried by iter2.

Workers use PyTorch float32 on `cuda:0`, bound to NVIDIA GeForce RTX 3050 capability 8.6, with
`TorchPointRasterizer`, point/gaussian chunks 256, outer microbatch 128, teacher query chunk 640,
tile size 16, degree-zero SH, hard SH activation, hard EWA support, visibility margin 3.0, and
black background. Built-in dense checkpoint evaluation is disabled.

The explicit extent is `1.5469313859939577`. Learning rates are
`means=1.6e-4*extent`, `quaternions=1e-3`, `log_scales=5e-3`,
`opacity_logits=5e-2`, `SH-DC=2.5e-3`, and empty higher SH `=1.25e-4`. Six Adam optimizers use
betas `(0.9,0.999)`, epsilon `1e-15`, no weight decay, and AMSGrad/foreach/fused/maximize false;
mean LR decays by `0.01**(1/140)` after every update.

Paired A/C and B/D histories must agree at all 140 steps on view, sample seed, coordinates,
active/inside flags, component IDs, proposal density, and joint density. Target-density and
importance hashes must differ on all 140 steps. All arms must have exactly equal step-zero
semantic snapshots and fixed-bank metrics within a seed. Callback snapshots at `0,35,70,140`
must hash-match trainer history, whose built-in evaluations remain `null`.

## Frozen evaluation and decision

Bank generator seeds use the first eight little-endian bytes of SHA-256 over
`rtgs.compact-occupancy-factorial.eval.v1\0{seed}\0{view_id}\0{kind}`, masked to 63 bits.
Uniform banks must be fully active/direct and strictly inside. Proposal banks retain every null.
Coordinates, flags, component IDs, proposal/joint densities, and teacher colors are immutable and
shared by all arms/checkpoints for the paired seed.

For RGB-channel MSE $\ell$, report per-view and equal-view means

$$J_U=\frac1{4096}\sum_k\ell_k,\qquad
J_Q=\frac1{4096}\sum_k\mathbf1[\mathrm{active}_k]\ell_k.$$

$J_Q$ never divides by active count. Active-mass guards use the 21 unique proposal banks: every
fraction must be at least 0.95 and global max/min at most 1.03. Log-AUC integrates
`log(max(risk,1e-12))` with trapezoids at normalized abscissae `(0,0.25,0.5,1)`.

The sole authorizing contrast is D/B on $J_Q$. It passes only if all gates hold:

- geometric-mean final D/B $J_Q\le0.95$;
- geometric-mean AUC-derived D/B $J_Q\le0.97$;
- strict D final-$J_Q$ wins in at least two of three seeds;
- geometric-mean final D/B $J_U\le1.05$ and every seed ratio $\le1.10$;
- both active-mass guards pass.

Only then is `scientific_decision=AUTHORIZE_DENSITY_FOLLOWUP` and
`promotion_authorized=true`. A completed negative result uses
`NO_REFINEMENT_TARGET_PROMOTION`; any protocol/runtime/serialization failure uses
`scientific_decision=UNAVAILABLE`. Secondary C/A, B/A, D/C, interaction, per-view, and worst-view
diagnostics are non-authorizing.

## Once-only lifecycle and claim limits

The new namespace is `*_factorial_iter2_*` with run directory
`runs/compact_occupancy_refinement_factorial_iter2_20260717`. The seal binds this preregistration,
the failure audit and original consumed preregistration, current source/tests, both complete
compact bundles, initialization PLY, failed-attempt provenance, exact runtime/module origins,
effective config, and a fresh independent implementation review. Source/input/runtime/config
bindings are sampled before and after focused verification and immediately before exclusive seal
publication. The attempt token is exclusive and precedes every bank. All three bank archives and
their manifest precede twelve fresh workers, each bounded to 180 seconds. No retry or overwrite is
allowed.

After an immutable `status=PASS` result, all four seed-76601 final PLYs must be rendered on the
same seven native 5328x4608 compact-bundle cameras using repository gsplat with `packed=false` and
`antialiased=false`, assembled into a labelled contact sheet, and exposed in a smoke-tested live
viewer. Source RGB/reference loading is allowed only after the result for viewer diagnosis and
cannot affect optimization, selection, metrics, or the decision.

Even a positive result remains single-scene, single-producer, fixed-topology, finite-budget
development evidence. It does not establish source-RGB equivalence, novel-view accuracy,
multi-scene transfer, fitting scalability, speed/memory superiority, ordinary-3DGS superiority,
GaussianImage_plus integration, or a production default.
