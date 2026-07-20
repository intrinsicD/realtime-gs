# Compact occupancy-point refinement factorial — preregistration — 2026-07-17

## Status and question

This document freezes a single-scene, fixed-topology development experiment before any official
factorial arm is executed. Exploratory mechanism probes used seed 76201 only; the official training
and evaluation seeds below are fresh. The experiment asks two separable questions:

1. Does a deterministic balanced-cycle view schedule improve equal-view compact-teacher fitting
   relative to the current IID schedule?
2. At identical sampled coordinates, does optimizing the active continuous Gaussian-point proposal
   submeasure improve occupancy-region fitting relative to importance-correcting those samples back
   to uniform continuous image area?

No source RGB, dense target image, or mask may be opened by the official run. Exact color targets
come only from frozen compact 2D Gaussian teachers. This experiment keeps
$N_{\mathrm{init}}^{3D}=N_{\mathrm{opt}}^{3D}=835$; split, clone, merge, prune, relocation, and any
other density control are deferred. A positive result therefore authorizes only a later
$N_{\mathrm{opt}}^{3D}$ experiment, not a production default.

An official attempt may begin only after an independent implementation review passes. Sealing
binds this document, the harness and focused tests, compact trainer, observation sampler/query
equations, point rasterizer, 3D Gaussian IO, both complete compact bundles, and the initialization
PLY. The official namespace must be empty before sealing. The run claims one exclusive attempt
before loading or constructing official banks, executes every arm in a fresh bounded worker, and
writes one finite terminal result even on failure; no retry or overwrite is allowed. A protocol
`PASS` means the frozen Cartesian product and decision arithmetic completed, irrespective of
whether the scientific decision is positive.

After the thresholds and official seeds above were written, but before sealing, excluded seed
`76201` was used for an end-to-end dry mechanism check with the now-frozen configuration. It was
finite, preserved 835 Gaussians, and passed the bank guards (`min=0.991943`,
`max/min=1.005661`). Its D/B ratios were `0.802971` for final $J_Q$, `0.893980` for the
AUC-derived $J_Q$ risk, and `0.949092` for final $J_U$; balanced scheduling alone had B/A final
$J_U=1.023196`. These observations are disclosed because the implementation author saw them.
They do not enter any official aggregate, and no threshold, seed, bank size, or decision rule was
changed after seeing them.

## Frozen inputs and upstream qualifications

- Exact color teachers and calibrated cameras:
  `runs/compact_masked_bundle_640_20260717/reconstruction_inputs`.
- Center-occupancy proxy geometry/scalars:
  `runs/compact_occupancy_scalar_ablation_20260717/proxy_bundles/center`.
- Common 3D initialization:
  `runs/compact_occupancy_scalar_ablation_20260717/stage_b/center/gaussians.ply`, expected SHA-256
  `0cf0340117739bb4b0491ff9c90d8d4b622b57a57f6bf8e6a3cfc9984b5c416e`.
- Ordered views: `C0001,C0008,C0014,C0021,C0026,C0031,C0039`.
- Per-view optimized 2D counts are $m_{\mathrm{opt},i}^{2D}=640$ in this bundle, but code and
  preflight must retain a list of per-view counts rather than assume equality.

The Stage-1 acquisition is content-valid but lifecycle-qualified and has six incorrect descriptive
seed fields. The center occupancy screen is same-training-mask mechanism evidence only. The
independently replayed smooth-scalar ablation rejected mean/LSE replacement, so the proposal uses
the center scalar and makes no smooth-maximum claim.

## Compact occupancy proposal

For teacher component $j$ in view $i$, construct an immutable proposal field with the teacher's
exact canvas, fitted window, mean, covariance/filter, finite-support, and fade semantics, but with

$$b_{ij}=a_{ij}\,o_{ij},$$

where $a_{ij}$ is the exact color teacher amplitude and $o_{ij}$ is the aligned frozen center-mask
occupancy scalar. Proposal colors are unused constants. The trainer must query exact color only
through the original teacher backend. Exact co-located splitting with amplitudes summing to the
original and duplicated occupancy/geometry leaves the proposal density unchanged; using the scalar
alone would not have that property as $m_{\mathrm{opt},i}^{2D}$ changes.

Let $u_i(x)=1/A_i$ on the fitted window, let
$D_i^o(x)=\sum_j b_{ij}K_{ij}(x)$, and let
$M_i^o=\sum_j b_{ij}2\pi\sqrt{\det\Sigma_{ij}}$. The rejection-thinned Gaussian active subdensity is
$g_i(x)=D_i^o(x)/M_i^o$. With uniform fraction $\eta=0.25$, the active proposal subdensity is

$$q_i(x)=\eta u_i(x)+(1-\eta)g_i(x).$$

Rejected Gaussian draws remain explicit null attempts and are never resampled. If
$Z_i=\int g_i(x)dx$, the active attempt mass is
$r_i=\eta+(1-\eta)Z_i$.

- `uniform` target: active importance $u_i(x)/q_i(x)$ and null weight zero. Fixed-attempt means
  estimate $J_{U,i}=\int u_i(x)\ell_i(x)dx$ and weights are bounded by $1/\eta=4$.
- `proposal_attempt` target: active importance exactly one and null weight zero. Fixed-attempt means
  estimate the unnormalized attempt risk
  $J_{Q,i}=\int q_i(x)\ell_i(x)dx$, not a normalized occupancy probability risk.

No active-count normalization, estimated $1/r_i$ factor, or null resampling is allowed. The
proposal-attempt interpretation is considered stable only when every unique frozen evaluation
bank/view active fraction is at least 0.95 and their global largest-to-smallest ratio is at most
1.03; these are interpretation guards, not tunable thresholds.

## Frozen factorial and execution budget

All arms use `proposal_mode=area_gaussian`, the same product-amplitude proposal fields, exact color
teachers, optimizer hyperparameters, $\eta=0.25$, 128 fixed attempts per step, degree-zero SH, and
the common 835-Gaussian initialization.

| Arm | Schedule | Target |
| --- | --- | --- |
| A | `iid` | `uniform` |
| B | `balanced_cycle` | `uniform` |
| C | `iid` | `proposal_attempt` |
| D | `balanced_cycle` | `proposal_attempt` |

Official training seeds are `76401,76402,76403`. Each arm runs 140 updates, exactly 20 complete
seven-view cycles for balanced arms, with checkpoints `0,35,70,140`. Evaluation-bank seeds are
`76501,76502,76503`, respectively, and are isolated from training RNG. Each seed/view has one
frozen 4096-attempt uniform-area bank and one frozen 4096-attempt occupancy-proposal bank. Banks and
exact teacher colors are shared by every arm for that seed. At every training step, paired A/C and
B/D histories must agree exactly on scheduled view, sample seed, coordinates, active/inside flags,
component ids, proposal density, and joint density. Importance and target-density hashes are
expected to differ because target mode changes only those tensors. All four step-zero semantic
snapshots and both step-zero bank risks must agree exactly.

The official worker must strict-load both bundles and the common PLY, validate all geometry/support
alignment before proposal-index allocation, deny dataset/source-image/PIL/calibrated-scene access,
and record zero source-RGB attempts. Complexity and retained state are reported in terms of
$\sum_i m_{\mathrm{opt},i}^{2D}$, tile overlaps, sampled attempts, and
$N_{\mathrm{init}}^{3D}$; no runtime or memory scaling claim is permitted from one equal-count
bundle.

Before copying any scalar positionally, the harness must compare each raw center-proxy field with
its exact color teacher: ordered view/name/id, camera, canvas, fitted window, component count,
means, log-scales, rotations, optional filter variance, cutoff/fade/AA support, and coordinate
semantics must agree exactly. Every proxy scalar must be finite and in `[0,1]`. The constructed
field must then satisfy `product_amplitude == teacher_amplitude * proxy_amplitude` bit-exactly.
Copying teacher geometry first and validating only the copied proposal is insufficient.

## Frozen implementation configuration

The official workers use PyTorch float32 on `cuda:0`, bound at seal/run time to the current NVIDIA
GeForce RTX 3050 (compute capability 8.6), with `TorchPointRasterizer`. The harness must fail
terminally if this device or the sealed software/runtime binding changes. Trainer values are:

- explicit extent `1.5469313859939577`; learning rates `means=1.6e-4*extent`,
  `quaternions=1e-3`, `log_scales=5e-3`, `opacity_logits=5e-2`, `SH-DC=2.5e-3`, and
  empty higher-order SH `=1.25e-4`;
- six separate Adam optimizers with betas `(0.9,0.999)`, epsilon `1e-15`, no weight decay,
  and AMSGrad, fused, foreach, maximize all false; mean LR decays by `0.01**(1/140)` per update;
- `point_chunk=256`, `gaussian_chunk=256`, `outer_microbatch=128`,
  `query_component_chunk=640`, `teacher_tile_size=16`, hard SH-color activation, hard EWA support,
  visibility margin `3.0`, black point-render background, and `sh_degree=0`;
- built-in exhaustive checkpoint-risk evaluation disabled. The callback captures exactly steps
  `0,35,70,140`; only the frozen banks below are used for decision metrics.

These chunk values were chosen in an excluded seed-76201 feasibility probe before sealing. On one
4096-point view, `(256,256)` differed from the CPU-first reference chunk configuration `(32,64)` by
at most `8.95e-8` in rendered color and changed no experimental threshold. This is engineering
qualification only, not outcome evidence.

## Frozen evaluation banks

For each `(evaluation seed, view id, bank kind)`, derive an isolated CPU generator seed as the
little-endian integer in the first eight SHA-256 bytes of
`rtgs.compact-occupancy-factorial.eval.v1\0{seed}\0{view_id}\0{kind}`, masked to 63 bits. Thus a
different $m_{\mathrm{opt},i}^{2D}$ or view order cannot perturb any other bank. The `uniform` bank
contains 4096 direct fitted-window continuous draws and is fully active. The `proposal` bank
contains exactly 4096 attempts from `GaussianPointProposal(product_field)` at $\eta=0.25$,
including every explicit null without resampling. For each bank, precompute and hash coordinates,
active/inside flags, component ids, proposal and joint densities, and exact colors queried only
from the original color teacher. These immutable bank tensors are shared by all four arms and all
four checkpoints for that seed.

For a per-point RGB-channel MSE $\ell$, define `J_U = mean(loss)` over all 4096 uniform draws and
`J_Q = sum(active * loss) / 4096` over all proposal attempts. `J_Q` must never divide by the active
count. The decision's active-mass guards are computed over the 21 unique frozen proposal banks
`(evaluation seed, view)`, not over training draws or duplicated paired arms: every bank's active
fraction must be at least `0.95`, and the global maximum divided by the global minimum must be at
most `1.03`. Training active rates remain diagnostics only.

## Metrics and decision rule

For each checkpoint, arm, seed, and view, report fixed-bank $J_U$ and $J_Q$, equal-view means,
per-view values, worst-view values, active/null counts, importance effective sample size, gradient
maxima, wall time, peak RSS, parameter motion, and exact artifact hashes. Log-AUC uses trapezoidal
integration of `log(max(risk,1e-12))` over normalized checkpoint step. Lower is better. $J_U$ and
$J_Q$ are different measures and must never be numerically ranked against each other.

Concretely, for checkpoint abscissae $x=(0,0.25,0.5,1)$, let
$L_X=\operatorname{trapz}(\log(\max(J_X,10^{-12})),x)$. The per-seed D/B AUC-derived risk ratio is
$\exp(L_D-L_B)$ and its three-seed aggregate is
$\exp(\operatorname{mean}_s(L_{D,s}-L_{B,s}))$; signed log-AUC values are never divided. Final-risk
ratios likewise use the three-seed geometric mean of per-seed ratios from equal-view means. A win
requires strict `D < B`; ties are not wins.

The primary contrast is D versus B on the shared $J_Q$ bank. It passes only if:

- the three-seed geometric-mean D/B final-risk ratio is at most 0.95;
- the geometric-mean D/B log-AUC-derived risk ratio is at most 0.97;
- D beats B on final $J_Q$ in at least two of three seeds;
- D's uniform $J_U$ final-risk geometric-mean ratio versus B is at most 1.05 and no seed exceeds
  1.10; and
- both active-mass interpretation guards pass.

If all conditions pass, the decision is `AUTHORIZE_DENSITY_FOLLOWUP`; otherwise it is
`NO_REFINEMENT_TARGET_PROMOTION`. Secondary, non-authorizing contrasts are C/A (target under IID),
B/A (schedule under uniform target), D/C (schedule under proposal target), the factorial
interaction, and per-view/worst-view diagnostics. No threshold may change after sealing.

## Visualization and claim limits

After immutable metric/artifact hashes are written, all four final PLYs from the first official
seed must be rendered from the same full-resolution calibrated camera set with the repository's
gsplat backend and assembled into a labelled contact sheet. A live `rtgs view` instance must expose
the selected comparison. Viewer/reference-image loading is a post-result visual diagnostic and may
not feed any optimizer, selector, metric, or decision.

A passing result is limited to this scene, producer, compact bundles, fixed topology, and finite
budget. It does not establish novel-view accuracy, source-RGB equivalence, multiple-scene transfer,
full-resolution fitting scalability, speed, memory reduction, superiority over ordinary 3DGS,
GaussianImage++ integration, or a production/default change. Scholar Inbox papers on continuous
subpixel fields, saliency-weighted allocation, and memory-aware Gaussian updates motivate the
factorization but provide no evidence for its outcome.
