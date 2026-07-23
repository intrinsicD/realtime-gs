# State-of-the-art survey and reuse decisions

Compiled 2026-07-07 from three research sweeps (3DGS fitting/rendering; 2D gaussian image
representations; depth/lifting/carving), then updated through the 2026-07-21 experiment record.
Each section ends with what this repo reuses. License claims were verified against the repositories
on the original compile date — re-verify before anything license-sensitive.

## 1. 3DGS fitting & rendering backends

- **gsplat** (Ye et al., JMLR 2025, [arXiv 2409.06765](https://arxiv.org/abs/2409.06765),
  [code](https://github.com/nerfstudio-project/gsplat), **Apache-2.0**, v1.5.x on PyPI).
  Independent CUDA reimplementation of 3DGS: `rasterization()` takes means/quats/scales/
  opacities/SH + viewmats/Ks and returns colors/alphas/meta (with `means2d`, `radii`,
  `depths` for densification). Flags we care about: `packed=True` (memory),
  `rasterize_mode="antialiased"` (Mip-Splatting AA), `absgrad=True` (AbsGS gradients),
  `render_mode="RGB+D"` (built-in depth). Ships densification as library code:
  `DefaultStrategy` (classic clone/split/prune/reset) and `MCMCStrategy` (3DGS-MCMC).
  Installs without a GPU; only kernel calls need CUDA. Native 2DGS entry point exists
  (`rasterization_2dgs`). ~10-15% faster and up to 4x less memory than the INRIA code at
  equal quality.
- **INRIA 3DGS** (Kerbl et al., SIGGRAPH 2023, [arXiv 2308.04079](https://arxiv.org/abs/2308.04079),
  [code](https://github.com/graphdeco-inria/gaussian-splatting)) — **non-commercial
  license that virally taints forks** (3DGS-MCMC, AbsGS, Speedy-Splat, RAIN-GS repos are
  all INRIA-derived). Use only as a numbers/behavior reference; never vendor code.
- **Densification SOTA**: 3DGS-MCMC (Kheradmand et al., NeurIPS 2024,
  [2404.09591](https://arxiv.org/abs/2404.09591)) replaces heuristics with relocation +
  noise under a fixed budget and is notably **less init-sensitive** — relevant since our
  init is not SfM; reusable via gsplat's Apache `MCMCStrategy`. AbsGS
  ([2404.10484](https://arxiv.org/abs/2404.10484)): absolute-value gradient accumulation
  fixes gradient collision (gsplat `absgrad=True`). Revising Densification (Bulò et al.,
  ECCV 2024, [2404.06109](https://arxiv.org/abs/2404.06109)): per-pixel-error criterion,
  corrected clone opacity (gsplat `revised_opacity`). Taming 3DGS
  ([2406.15643](https://arxiv.org/abs/2406.15643)): budgeted score-based densification +
  fused SSIM + per-splat backward (perf parts MIT, upstreamed into gsplat).
- **Training-speed SOTA**: the recurring ingredients are fused SSIM
  ([rahul-goel/fused-ssim](https://github.com/rahul-goel/fused-ssim), MIT), per-gaussian
  sparse Adam, tighter tile culling (Speedy-Splat's SnugBox/AccuTile — adopted by gsplat),
  resolution/frequency schedules (DashGaussian, [2503.18402](https://arxiv.org/abs/2503.18402),
  ~200 s/scene; FastGS [2511.04283](https://arxiv.org/abs/2511.04283), ~100 s), and strict
  primitive budgets. Faster-GS (CVPR 2026, [2602.09999](https://arxiv.org/abs/2602.09999))
  consolidates best practices. **The lever no one fully exploits is initialization — our
  target.**

### 1.1 Smooth SH color-floor update (2026-07-15)

Biswas et al., **Smooth Maximum Unit** (CVPR 2022,
[arXiv 2111.04682](https://arxiv.org/abs/2111.04682)) distinguish an erf-based SMU approximation
from below from the square-root approximation from above that they call SMU-1. For the renderer's
`max(x,0)` floor, the repository's frozen specialization was SMU-1 with `alpha=0`, `mu=2/255`:
`0.5 * (x + sqrt(x^2 + mu^2))`. Its maximum fixed-parameter forward bias is `1/255`. The exact SMU
was not an equivalent candidate because its `alpha=0` form can emit negative values. The Scholar
Inbox digest through 2026-07-15 contained no paper directly evaluating either function as the
post-SH nonnegative color floor in 3D Gaussian Splatting; continuity mechanisms in other Gaussian
geometry terms are analogies, not evidence for this intervention.

The preregistered hard-only incidence audit did not authorize a candidate test. Across three
view-dependent CPU synthetic seeds, pooled negative-channel incidence was 0.336527% versus a 1%
gate, recoverable blocked-gradient mass was 0.090828% versus 5%, and the fixed SMU-1 derivative
would recover 0.025266% versus 0.5%. All seeds failed all three gates; view coverage and observation
counts passed. Phase B therefore remained unrun under the permanent stop rule.

**Reuse decision:** retain the hard preactivation/activation split, opt-in SMU-1 and hard-forward
gradient-control primitives, and Torch-only diagnostic plumbing as research infrastructure. Keep
the standard hard floor as the default. Do not infer SMU-1 quality, CUDA parity, density-control
compatibility, real-scene behavior, or a default-change case from an audit in which neither
candidate trained. Close parameter/schedule tuning for this color-floor branch; a hard raster-
support cutoff is a separate mechanism and needs its own preregistered incidence audit.

### 1.2 Smooth spatial-support update (2026-07-15)

The next distinct hard operation in the reference rasterizer is its compact EWA kernel
`exp(-q/2) * 1[q < 12]`, with squared Mahalanobis distance `q`. The Scholar Inbox digest through
2026-07-15 contained no paper directly testing a smooth spatial-support replacement in a 3DGS
rasterizer. Grassmannian Splatting I
([arXiv 2607.10489](https://arxiv.org/abs/2607.10489)) smooths an unrelated Schur denominator while
retaining a standard 3DGS renderer, and SplatCtrl
([arXiv 2607.08948](https://arxiv.org/abs/2607.08948)) uses continuous Gaussian distance fields
for control and collision queries. They motivate asking about continuity but do not support a
raster-tail benefit claim.

The frozen repository candidate left the kernel bit-identical for `q < 12`, multiplied it by the
cubic C1 step `1 - 3t^2 + 2t^3` for `12 <= q < 16`, and returned to zero at `q >= 16`; its maximum
added weight was `exp(-6)`. A hard-forward control exposed exactly the taper derivative while
preserving the established forward value. In the independently audited Phase-A replay, the
diffuse pool contained 48,290,887 eligible observations: the annulus carried 40.7745% of local
upstream gradient mass, 24.6717% of that mass was loss-recoverable, and the frozen derivative
recovered 0.252269% of active and 5.43819% of boundary hard q-gradient mass. All three seeds and
the pool passed the prespecified mechanism gate.

That positive local screen did not translate to utility. Under common-hard final evaluation, the
C1 arm changed diffuse held-out foreground PSNR by -0.018741/-0.013265/-0.011443 dB (mean
-0.014483 dB; 0/3 wins), while the hard-forward control changed it by
-0.028500/-0.013335/-0.013576 dB (mean -0.018470 dB; 0/3 wins). All seven SSIM, depth, alpha,
coverage, and view-dependent guardrails passed, but neither the +0.10 dB/two-win primary gate nor
the attribution gates did. Exact replay and outcome evidence are
`benchmarks/results/20260715T202218Z_cpu_kernel_support_taper_iter2_audit.json` and
`benchmarks/results/20260715T202917Z_cpu_kernel_support_taper_iter2_ablation.json`. Their exact
independent reviews,
`benchmarks/results/20260715T202218Z_cpu_kernel_support_taper_iter2_audit_AUDIT.md` and
`benchmarks/results/20260715T202917Z_cpu_kernel_support_taper_iter2_ablation_AUDIT.md`, recompute
both decisions and document the consumed first Phase-B attempt and representation-only retry.

**Reuse decision:** retain the hard kernel default and keep the C1/straight-through modes and
Torch-only diagnostic collector as opt-in research infrastructure. Reject width, shape, cutoff,
schedule, seed, and visibility-margin tuning for this branch. This CPU synthetic,
depth-initialized, fixed-topology result establishes neither real-scene transfer, density-control
interaction, gsplat/CUDA parity or speed, nor a default-change case. A next experiment, if any,
had to separately preregister a hard-only incidence audit of the detached 3-sigma visibility cull
against a support-safe `sqrt(12)` image-intersection envelope. Section 1.3 records that audit.

### 1.3 Coarse visibility-support update (2026-07-15)

The reference renderer's default coarse cull accepts projected centers within
`3*sqrt(lambda_max(Sigma_2D))` of the image. Because its established compact kernel accepts
`q < 12`, `sqrt(12)*sqrt(lambda_max)` is an exact conservative image-intersection envelope; only
the narrow `9 <= q < 12` shell can differ. This is a set-boundary question distinct from smoothing
the kernel. The Scholar Inbox digest through 2026-07-15 contained no paper directly testing this
3DGS raster-culling mismatch. Grassmannian Splatting I and SplatCtrl remain mechanism analogies,
not evidence for changing the cull.

The first sealed audit attempt failed closed before producing an artifact or exposing a candidate
outcome. Expanding the set by one primitive changed the unspecified `torch.argsort` order of two
current primitives with exactly equal float32 depth. Before replay, the retry preregistration froze
a baseline-preserving representation repair: preserve the established current order, order new
primitives separately, and stable-sort their concatenation. The default 3-sigma path stayed
bit-exact. The retry then replayed all six current-margin runs from scratch; view-dependent results
were reporting-only.

All diffuse validity prerequisites passed, including exact target parity for both margins, all nine
training views, set/support/order invariants, and per-seed support counts. Nevertheless, 3-sigma
culling missed only 4 of 2,480,463 pooled final `q < 12` pairs, a fraction of 1.612602e-6, spanning
two Gaussian/view exposures. Their effective-mass fraction was 1.646359e-8 and the support-safe
versus current render delta was 3.986964e-8 of the current target residual. All three seed material
decisions and the pooled decision failed the frozen gates, so Phase B was forbidden and no
support-safe training arm ran. Replay evidence is
`benchmarks/results/20260715T213132Z_cpu_visibility_margin_iter2_audit.json`; its independent
`_AUDIT.md` has SHA-256
`21c262aad36f02cf9a6520d50c2d2a867a22758e0486daa35094cdd78b9eb928`. Chronology and exact
source are bound by `benchmarks/results/20260715_visibility_margin_PREREG.md`,
`benchmarks/results/20260715_visibility_margin_iter2_PREREG.md`, and their matching seals/attempt
markers.

**Reuse decision:** retain the 3-sigma default, keep the finite positive Torch-only margin option
as research infrastructure, and do not tune the margin on this consumed setup. The
baseline-preserving expanded-order convention remains necessary to isolate set extensions, but is
not evidence of a quality improvement. This CPU synthetic, depth-initialized, fixed-topology
audit establishes no real-scene, density-enabled, near-plane, CUDA/gsplat, performance, or default
claim. With the smooth floor, smooth tail, and coarse-cull branches closed here, the subsequent
Carve equal-count audit also stopped at its materiality gate: production grouping merged only
2.34%-2.68% of primitives, so merge-versus-prune utility was not tested. Prioritize earlier
representation and allocation interfaces rather than another smooth-gate or grid-scale variant.

### 1.4 Fixed-topology multiscale update (2026-07-16)

DashGaussian and related coarse-to-fine training work motivated asking whether cheaper early
camera exposure or coarse loss supervision could accelerate this repository's Stage-3 refinement.
The analogy did not determine the schedule or predict an outcome. A separate protocol froze one
exact 24-to-48, 60/60 schedule, a full-resolution control, an exposure-matched interleaved camera
control, fixed topology, degree-zero SH, 120 updates, and held-out checkpoints for CPU synthetic
seeds 3/4/5.

The camera-blocked, pyramid-blocked, and camera-interleaved arms changed foreground-PSNR AUC by
`-0.338645`, `-0.088758`, and `-0.345927 dB` on average relative to full resolution, with zero
seed wins for every arm. Both camera schedules used exactly 62.5% of the full optimization raster
pixels, but neither was quality-noninferior; blocked ordering beat interleaving by only
`0.007282 dB` mean AUC and failed its attribution gate. The official artifact is
`benchmarks/results/20260716T003735Z_cpu_multiscale_refinement.json`; its independent scientist
pass is the adjacent `_AUDIT.md`.

**Reuse decision:** close only this exact fixed-topology 24-to-48 branch. Do not report the
37.5% exposure reduction as a speedup and do not combine the failed schedule with another
intervention. Parameter-specific geometry/appearance routing, adaptive density, full SH, real
scenes, and CUDA remain different questions that require new preregistrations.

### 1.5 Quaternion radial-gauge validity update (2026-07-16)

Normalized quaternion rotation has an exact positive radial gauge in real arithmetic, making
unit-manifold or Riemannian updates a natural optimization question. The repository froze a
mechanism-first comparison of ambient Adam, entry canonicalization, unit retraction, tangent-
displacement retraction, and a projected-gradient control. Neither consumed Phase-A attempt
produced an optimizer outcome. The first exposed only a mixed-precision diagnostic-order mismatch;
its append-only repair made the removed-gradient producer and validator share one promote-first
float64 calculation while preserving the native float32 optimizer path.

Retry-2 then exposed a separate precondition error. Applying `F.normalize` in native float32 and
normalizing the stored result again after promotion to float64 is not direction-idempotent. From
the retained step-zero fields, the three seed direction differences were `1.08e-8`, `2.06e-8`,
and `1.51e-8`; corresponding covariance maximum errors were `6.13e-10`, `2.05e-9`, and
`9.53e-10`. Those values necessarily fail the inherited `2e-12` covariance-equivalence gate even
though the underlying algebraic gauge is exact. The invalid artifact deliberately strips every
arm trajectory, checkpoint, AUC, and materiality decision. Its independent audit is
`benchmarks/results/20260716T030759Z_cpu_quaternion_gauge_iter2_invalid_AUDIT.md`.

**Reuse decision:** keep the shared promote-first diagnostic helper, adversarial arithmetic-order
tests, append-only seal/review binding, and native float32 replay tests as research infrastructure.
Do not infer a quaternion-policy ranking or change the `current` default. Any renewed experiment
must prospectively define a precision-aware construction contract and prove it before optimizer
arms run; it must not tune the consumed threshold from hidden checkpoint behavior. The more direct
Stage-1 representation question has since completed in the deterministic CPU-synthetic,
150-component, 120-update scope: the independently audited nine-parameter `weight*color` versus
bounded unit-weight eight-parameter comparison found material local null-direction Adam motion,
but the candidate lost every appearance-only and joint seed and failed joint non-inferiority.
Retain the current default and close only this exact bounded candidate without tuning.

## 2. Images as 2D gaussians (stage 1 foundations)

- **GaussianImage** (Zhang et al., ECCV 2024, [2403.08551](https://arxiv.org/abs/2403.08551),
  [code](https://github.com/Xinjie-Q/GaussianImage), **MIT**): 8 params/gaussian —
  position, **Cholesky factor (l11, l21, l22)** of the 2D covariance, weighted color.
  The official Cholesky implementation fixes its raster opacity tensor to one and optimizes one
  three-vector color/feature; it does not add a separately trainable scalar weight. This
  repository's bounded `weight*color` factorization is therefore a ninth-parameter extension,
  not an upstream GaussianImage contract.
  **Accumulated summation** blending (no sorting, no alpha compositing) is
  order-independent and beats alpha blending by **+0.8 dB** while being faster. Kodak
  768x512: 70k gaussians → 44.1 dB in ~107 s (V100, 50k Adan steps); 30k → 38.6 dB.
  Loss ablation: **plain L2 beats L1/L1+SSIM/L2+SSIM for PSNR**. Rendering ~2000 FPS.
- **Image-GS** (Y. Zhang et al., SIGGRAPH 2025, [2407.01866](https://arxiv.org/abs/2407.01866),
  [code](https://github.com/NYU-ICL/image-gs), **MIT**): content-adaptive allocation —
  positions sampled from **gradient magnitude mixed ~70/30 with uniform**, then
  **error-guided progressive addition** (start N/2, add N/8 every 500 steps at max-error
  pixels); reaches ~95% of final quality within ~400 steps. Loss L1 + 0.1·SSIM.
- **Feed-forward fitting**: Instant-GI (ICCV 2025, [2506.23479](https://arxiv.org/abs/2506.23479),
  MIT) predicts a full 2D gaussian set in one pass (~10x less wall-clock than optimizing);
  Fast 2DGS ([2512.12774](https://arxiv.org/abs/2512.12774)) similar. Future stage-1
  speedup path.
- **Local StructSplat implementation** (`~/Documents/structsplat`, **MIT**) is a 2D image
  representation despite the name collision with an unrelated paper. It combines a structure
  tensor, feature-aware anisotropic weighted-sample-elimination placement, normalized weighted
  splatting, and residual-driven density growth. The low-budget control consumed here uses
  `aniso_onedge`; StructSplat's broader evidence currently prefers `quadtree_wse` at 5k and
  `aniso_onedge` around 2k, so the policy remains configurable rather than universal.
- **Gaussian Point Splatting** (Rijsdijk et al., SIGGRAPH/TOG 2026,
  [project](https://jorisar.nl/gaussian_point_splatting/),
  [DOI](https://doi.org/10.1145/3811272)) samples pixel-sized opaque points from in-view 3D
  Gaussians and uses stochastic transparency to render enormous scenes without sorting or a
  tile hierarchy. We reuse the broader systems idea—sample work from Gaussian mass and preserve
  the sampling probability—for compact-teacher supervision. The repository's continuous
  2D-field proposal, uniform coverage floor, importance correction, and null thinning are a new
  fitting experiment; they are not that paper's renderer and inherit none of its performance or
  reconstruction claims.
- **Do not vendor**: LIG (GPL-3.0), MiraGe (INRIA non-commercial license via GaMeS).
  MiraGe ([2410.01521](https://arxiv.org/abs/2410.01521)) is conceptually interesting:
  flat gaussians in 3D rendered by the 3DGS renderer for a single image.
- **Budget is task- and resolution-dependent, not a fixed cap.** The local max-side-160,
  640-budget StructSplat control reached 27.70 dB, while adaptive growth to 952 reached 28.67 dB.
  On Janelle at 333×288, a fixed 640 start was already enough for the strongest measured 3D
  initialization; 320 converged to essentially the same final held-out result after 3D density
  control. Keep the initial count and maximum independent; large 15–30k/image compression
  budgets are not justified for this initialization objective.

### 2.1 Weight/color gauge contract update (2026-07-16)

The native Stage-1 renderer observes each component only through the RGB product
`a_i = w_i c_i`. This creates a scale ambiguity: bounded pairs `(w_i,c_i)` can be changed while
leaving the additive RGB component fixed. The repository nevertheless uses `w_i` separately for
coverage, retention, and merge weighting, and uses `c_i` separately for Depth/Carve appearance and
Carve tunnel matching. Upstream GaussianImage instead fixes opacity to one and directly optimizes
the three-vector accumulated color, so the extra scalar gauge is local to this repository.
GaussianImage motivates the accumulated RGB representation, while standard
factorization non-uniqueness motivates treating this downstream interpretation as an empirical
contract rather than a mathematical identity. The 2026-07 Scholar papers on geometry/appearance
decoupling and multi-attribute consistency are research analogies, not evidence for a particular
repair.

The preregistered contract audit constructed two exact product-preserving representatives from
each fitted component: unit weight with color `a`, and peak amplitude
`m=max_c(a_c)` with color `a/m`. Across seeds 0/1/2, all 54 transformed source renders passed the
frozen equivalence checks, with maximum absolute RGB error below `1.79e-7`. Despite that source
equivalence, both named transforms passed the materiality gates independently in 3/3 seeds and
the raw pool for both unmerged Depth and unmerged Carve. Pooled Carve output-key disagreement was
9.21% for unit weight and 64.16% for peak color; Depth and Carve render changes also exceeded the
frozen materiality margins by orders of magnitude. Exact evidence is
`benchmarks/results/20260716T003140Z_cpu_stage1_weight_gauge_audit.json`; the adjacent independent
audit has verdict `QUALIFIED` because raw tensors were hash-bound rather than serialized, while
all decision reductions and set arithmetic were independently reconstructed.

The follow-up semantic factorial has now supplied that held-out test in a three-seed deterministic
CPU-synthetic scope. It crossed fitted weight versus `m=max(w*c)` with fitted color versus sampled
source RGB, after passing the frozen tolerance-bound Phase-A invariance gates and separate exact-
product controls, and at matched per-view capacity. The full invariant-scalar/observed-color
candidate gained **+3.127230 dB** mean final PSNR for Depth but lost **-2.205314 dB** for Carve, so
the frozen cross-backend repair gate failed. Factorial attribution isolated a positive
observed-color signal in both backends (**+3.127289 dB** Depth, **+2.326600 dB** Carve), while `m`
was neutral for Depth and strongly harmful for Carve (**-4.531913 dB**), with a further negative
interaction. Raw evidence and the unqualified independent audit are
`benchmarks/results/20260716T063637Z_cpu_stage1_semantic_factorial_utility.json` and its adjacent
`_AUDIT.md`.

**Reuse decision:** retain the current `w_fit__c_fit` boundary and do not select the favorable
color-only arm post hoc. The Carve scalar intervention changes coverage, retention, available
source keys, tunnel placement, and refinement; even after exact count matching, the Carve
selected-set Jaccard was only about 47%-66%. Its negative effect is therefore not a scalar-at-
fixed-correspondence test and does not establish physical opacity semantics. The detached hard `max`
in this protocol never participates in Stage-1 learning, so this result is not evidence that an
SMU would improve its optimization. A differentiable smooth scalar inside fitting remains a
separate, presently unsupported question.

With geometry frozen, the additive renderer is linear in per-component RGB amplitude:
`prediction = K(xy,chol) a`. A known alternative to changing Adam coordinates is therefore
**variable projection / bounded linear least squares**: solve each channel's convex conditional
problem `min_{0<=a<=1} ||Ka-y||^2`, then update geometry outside that solve. This would remove the
weight/color gauge and distinguish coordinate-conditioning effects from merely under-solving
appearance. It is a separate future question, not an arm that may be inserted into the frozen N78
protocol.

**Experiment order:** the gauge-invariant semantic factorial and the bounded 8p source-fit
comparison are both complete and reject only their exact tested replacements in their frozen
CPU-synthetic scopes. N78's source freeze is now consumed, so the next implementation may proceed
to the already-frozen residual-responsibility density protocol N79. Variable projection remains a
separate future question and may not be inferred from the failed bounded-Adam arm.

### 2.2 Lossless compact observation boundary (2026-07-16)

Converted `Gaussians2D` files are useful initialization inputs but are not exact StructSplat
teachers: conversion clamps colors, materializes covariance filtering, and drops normalized-blend
epsilon, fitted-crop clipping, support fade, and optional affine color. `GaussianObservationField`
therefore freezes the live terminal RS field before conversion, including activated amplitude,
unbounded color, filter/AA semantics, full canvas plus fitted viewport, and independent
`N_init,2D`/`N_opt,2D`. Its equation matches the optional StructSplat CPU renderer on complete CPU
pixel grids for constant/affine colors, opacity, filtering, AA, fade, normalized/additive blend,
off-canvas support, and translated crops. This test does not establish CUDA or arbitrary
continuous-coordinate parity; arbitrary continuous coordinates are the repository's selected
extension of the captured equation. The seven previously saved native-resolution fits under
`runs/dataset_viewer_fullres_20260716/fits` remain initialization-only; exact compact refinement
requires re-exporting or refitting their live fields.

The first proposal has O(`N_opt,2D`) base component state. Its all-component reference query is
O(samples x components); the sparse index stores O(component x overlapped tiles) entries, which is
data-dependent rather than universally O(`N_opt,2D`). It draws a component by analytic Gaussian
mass, samples a continuous point, thins against the exact compact support/fade, and retains every
rejection as a zero-loss null attempt. A uniform branch covers background and bounds importance by
the inverse uniform fraction. This is exactly an estimator of **continuous fitted-window area**,
not the legacy uniform discrete-pixel objective. An exact co-located amplitude split preserves the
teacher, ideal proposal marginal, and ideal estimator distribution; finite seeded sequences may
differ only through categorical representation and floating-point reduction. Non-coincident
fragmentation can change the proposal and variance and requires a separate study. Proposal-origin
component ids are not supervision labels; the
exact queried field color is the first target. Any later direct component-color arm must handle
normalized epsilon explicitly, for example with a black pseudo-component.

`ReconstructionInputs` declares only these ordered fields, calibrated cameras, optional sparse
points/visibility, bounds, and restricted identifiers. Its schema has no RGB, mask, or source-path
field. The strict archive loader now checks the exact key set, identifier grammar, archive/member
byte ceilings, ordinary contained files, and symlink rejection before `np.load`; the compact
teachers can then be queried with image decoding disabled. A generic caller may still retain a
separate `SceneData`, so the type alone is not authority enforcement. The calibrated experiment
therefore loaded the bundle and trained inside fresh workers whose RGB/image APIs were denied. This
is a bounded process boundary for the tested lifecycle, not proof of arbitrary-scale memory,
runtime, reconstruction quality, or production isolation.

### 2.3 Compact-field Carve initialization (2026-07-16)

`CompactCarveInitializer` is a separate CPU correctness path over `ReconstructionInputs`; it does
not adapt the RGB-backed legacy `CarveLifter`. For view $i$, an anchor component $j$ is proposed in
proportion to its analytic mass
$m_{ij}=a_{ij}2\pi\sqrt{\det\Sigma_{ij}}$, then rejection-thinned against its captured finite
support and fitted window. A fixed candidate pool is distributed across views. Each accepted anchor
defines a camera ray, but every sampled depth is scored with every compact teacher.

For a projected point in view $i$, let $D_i$ be the exact field denominator, $C_i$ its queried
color, $A_i$ the fitted-window area, and $M_i=\sum_j m_{ij}$. The current coverage statistic is

$$r_i=A_iD_i/M_i,\qquad h_i=1-\exp(-r_i/\tau_\rho).$$

A visual-hull gate counts views whose $h_i$ exceeds a threshold. Among visible views, the candidate
color and channel-mean variance are weighted by $h_i$; the score multiplies mean soft coverage by
the resulting color-consistency term. The best supported depth on each tunnel is retained, and the
top view-balanced candidates produce exactly $N_{\mathrm{init}}^{3D}$ outputs on success or fail
closed when too few pass. Initial RGB is the clamped all-view consensus. The source component still
determines the proposed ray and lifted lateral covariance; its lineage is therefore hard proposal
metadata, although it never selects a teacher target or rendering subset.

An exact co-located split into identical components whose amplitudes sum to the original preserves
the tested scores and initialization geometry within tolerance. That does not extend to shifted or
otherwise non-identical fragmentation. Tunnel samples are camera-depth values; their local spread
is multiplied by the pixel-ray norm before becoming a Euclidean ray-axis covariance. Query point
batches and reference-backend point–component pairs are capped explicitly. The static tile index
still costs O(component–tile overlaps), and custom backends must honor the chunk contract. A later
full 26-view top-K run measured 99.7 s placement, 11.8629 dB initial fitted-view foreground PSNR,
and 37.2992 dB after adaptive density, but it was an unrepeated contended diagnostic without the
task's grouped-reference parity/RSS gates. Portable runtime, arbitrary-scale memory, CUDA query
behavior, and held-out reconstruction quality therefore remain unestablished. Current `from_scene`
geometry/bounds also lack proof of train-only provenance, so this mechanism does not establish
strict held-out isolation.

### 2.4 Sparse 3D point rendering and discrete-pixel risk (2026-07-16)

The compact refinement path now has a CPU correctness anchor for its second half. The separate
`PointRasterizer` protocol renders a flat list of image coordinates, and
`TorchPointRasterizer` deliberately uses one camera-wide visible set and one front-to-back depth
order. It does not accept proposal or lineage ids: every visible 3D Gaussian remains eligible at
every queried point. Point and Gaussian chunks bound the largest local point-component temporary,
while transmittance is carried across Gaussian chunks so chunk boundaries do not reset the global
compositor.

The sealed synthetic experiment matched dense `TorchRasterizer` color, alpha, depth, all five 3D
parameter-gradient families, and retained screen-space `means2d` gradients at pixel centers across
three seeds and all nine chunk pairs. Worst absolute forward and gradient discrepancies were
`2.3841858e-07` and `1.8626451e-09`. A near Gaussian outside the proposal lineage changed the
answer materially, confirming that lineage cannot silently become a rendering subset. Those
original off-grid fixtures happened to produce finite zero coordinate gradients. A later compact-
training prerequisite used a different interior anisotropic fixture and established materially
nonzero point-coordinate, projected-mean, and projected-log-scale gradients with central-difference
agreement. Dense off-grid renderer parity and CUDA-gradient parity remain unproven.

`GaussianPixelProposal` keeps only one clipped integer support rectangle per 2D component. It
selects component $i$ in proportion to amplitude times rectangle pixel count, draws a pixel center
uniformly inside that rectangle, and accepts with probability equal to exact component weight over
amplitude. Rejections remain zero-loss attempts. Therefore the active Gaussian-branch marginal at
pixel $p$ is $D(p)/M$, where $D$ is the exact field weight sum and $M$ is total rectangle-envelope
mass. With a uniform fraction $\eta>0$,

$$q(p)=\eta/P+(1-\eta)D(p)/M,\qquad
  w(p)=\frac{1/P}{q(p)},$$

and the mean over the original fixed number of attempts, including rejected nulls, is unbiased for
uniform risk over the $P$ fitted-window pixels. This is deliberately distinct from the existing
continuous-area proposal. The exact float64 fixture reproduced risk $55/96$ analytically and by
enumeration; its Monte Carlo and microchunk gates passed. This proves the estimator identity, not
which proposal has lower optimization variance on real compact teachers.

The authorized real-input interaction read the existing 835-Gaussian PLY and C0001 calibration
directly, without decoding source RGB or masks, and sparse/dense color, alpha, and depth agreed
within `4.7683716e-07` on 4,096 frozen replacement draws from the 333x288 downscale-16 domain. The
viewer then loaded its normal RGB references and saved scene camera 0 (`C0000`) as an exact
Torch/CPU snapshot through its normal UI; calibrated parity used `C0001`, so the viewer receipt is
a separate integration smoke rather than another parity sample. This is renderer
integration only: it does not refine the PLY, evaluate appearance against RGB, demonstrate full-
resolution training, or measure speed or memory. The next bounded experiment was the fixed-topology
bundle-only optimizer described below.

### 2.5 RGB-free fixed-topology point refinement (2026-07-16)

`CompactTrainer` consumes only a `ReconstructionInputs` bundle and an initialized 3D Gaussian set.
It samples a fixed number of pixel attempts, queries each compact 2D teacher independently, renders
every eligible 3D Gaussian at those coordinates, and optimizes means, quaternions, log-scales,
opacity logits, and SH coefficients with six Adam parameter groups. Sampling uses isolated RNG
streams; rejected proposal attempts remain zero-loss observations; topology is fixed, so
$N_{\mathrm{opt}}^{3D}=N_{\mathrm{init}}^{3D}$ in this experiment. The calibrated lifecycle ran
acquisition separately and performed bundle loading plus all optimizer steps in fresh workers with
RGB/image access denied.

The sealed synthetic comparison found **no global sampling win**. Against uniform pixels, the
discrete-pixel Gaussian mixture had geometric-mean final-loss ratio `1.0681355694` and AUC ratio
`1.0245665262`, losing in all three seeds. The continuous-area mixture had final ratio
`0.9873547158` and AUC ratio `0.9910818462`; all three AUC directions favored it, but it missed the
preregistered `0.95` materiality threshold. Uniform sampling remains the conservative comparison
baseline; neither compact proposal became a validated default. The experimental
`CompactTrainConfig` still has its pre-result `pixel_gaussian` convenience default, which is not
validated by this result and is not wired into the CLI/pipeline; proposal selection must become
explicit before production integration.

The bounded full-resolution interaction fitted seven 5328x4608 StructSplat teachers with 640 2D
Gaussians per view and serialized 4,480 components. The seven compressed teacher archives totaled
140,945 bytes, plus a separate 4,146-byte manifest. It initialized 835 3D Gaussians from a 3,340-
candidate pool (1,433 eligible). Forty RGB-denied refinement steps moved all five effective degree-
zero parameter families; the empty higher-order SH Adam group clock advanced without motion.
Equal-view compact-teacher MSE fell from `0.2846208576` to `0.2267813044` (-20.32%). Post-training
RGB evaluation on held-out C1004 improved 4,096-sample MSE
from `0.3904081687` to `0.3402060777` (-12.86%, +0.598 dB); its 256 foreground samples improved
8.40% (+0.381 dB). These are phase-local diagnostics, not a successful calibrated result: the
frozen run failed during gsplat import in its first exact-render operation because changing
`LD_PRELOAD` inside the already-started Miniconda process did not replace the old `libstdc++`. No
authorized snapshot was saved, and the HTTP viewer smoke was never reached. Separately labelled
post-failure gsplat snapshots are visibly coarse and blurry.

The harness has since routed exact-snapshot attempts into a fresh spawned process after freezing the
`LD_PRELOAD` path. New plans bind the requested path, resolved symlink target, SHA-256, and
`__cxa_call_terminate@CXXABI_1.3.15`; the worker verifies default-namespace resolution with
`dlvsym`, `dladdr`, and `/proc/self/maps`, then rechecks it after rendering. Result-schema,
dimension, count, backend, extension, artifact-hash, and input-hash checks remain. A separate fresh-
process ABI diagnostic (`postfailure_abi_diagnostic.json`) and process-owned live-viewer probe
(`postfailure_viewer_diagnostic.json`) pass, but this outcome-free repair has not run a real spawned
gsplat/CUDA render inside a new lifecycle and cannot retroactively repair the consumed run. A new
preregistered namespace is required to validate the complete lifecycle.

The scientific next step is density control with an explicit variable
$N_{\mathrm{opt}}^{3D}$, comparing residual/responsibility-driven allocation to a uniform-control
baseline at matched birth/death budgets. Compact training now preflights the original inputs before
transferring the teacher/camera working set or initialization to the configured device and constructs
a teacher/camera-only device-tensor working set, so optional global points, visibility, and bounds
tensors are not transferred; non-CPU tile-overlap counts use bounded device-native chunks. Scaling
work still needs aggregate device-byte/index budgets, CSR or lazy
indexes, indexed CUDA teacher queries, and a bounded-backward strategy. All teachers remain resident,
the reference CUDA query fallback is not indexed, and the autograd graph retains saved state on the
order of one outer microbatch times the visible Gaussian count. These mechanisms do not establish
production-scale memory.

### 2.6 Occupancy summaries and proposal-target separation (2026-07-17)

The independently replayed footprint-scalar screen rejected the tested smooth-maximum idea for
compact-Carve initialization. A 32-sample Gaussian-footprint mean and normalized LSE beta 2 both
increased eligible component-center rays, but neither improved the selected 835-Gaussian set. On
selector-isolated report views, beta 2 gained only 0.144 percentage points of recall while losing
11.287 points of precision and 6.962 points of IoU relative to center sampling. Its Stage-B
foreground compact-teacher MSE was 0.518% higher and its foreground-in-at-least-six-view fraction
fell from 0.899401 to 0.871856. The sealed run and all three PLYs replay exactly; the audit verdict
is a qualified protocol pass and a negative scientific result. This closes only the tested
mean/LSE/hard-max footprint summaries on one scene, not occupancy estimation generally.

The next refinement question is different from replacing that scalar. The existing Gaussian
proposal importance-corrects its samples back to uniform image-area risk, so it changes estimator
variance but intentionally does not make foreground regions more important. The new research seam
therefore separates three objects:

1. the exact compact color teacher queried at every sampled coordinate;
2. a density-only proposal field with amplitude equal to exact teacher amplitude times the audited
   center occupancy scalar; and
3. the target measure, either uniform continuous area or the active proposal-attempt submeasure.

The amplitude product makes exact co-located component splits invariant when a Stage-1 producer
changes $m_{\mathrm{opt},i}^{2D}$: duplicated geometry/occupancy with child amplitudes summing to
the parent leaves both analytic mass and point density unchanged. Scalar-only amplitudes would
overweight a region merely because it was split into more components. Proposal colors are unused
and may never become supervision. Rejected Gaussian draws remain explicit null attempts; unit
active importance optimizes an unnormalized proposal-attempt risk rather than a normalized
occupancy probability. A balanced-cycle camera schedule is tested independently because the old
40-step IID calibrated interaction assigned between 2 and 10 updates per view.

The sealed iter3 fixed-topology factorial passed its independent audit. At
$N_{\mathrm{init}}^{3D}=N_{\mathrm{opt}}^{3D}=835$, the balanced-schedule proposal-attempt arm
reduced final $J_Q$ relative to the balanced uniform-area arm in all three seeds; the geometric
final ratio was `0.7773749` and the recovery-log-AUC-derived ratio was `0.8812884`. Its final
uniform-area-risk ratio was `0.9480007`, within every frozen aggregate/per-seed guard. The audit
therefore confirms only `AUTHORIZE_DENSITY_FOLLOWUP`. One per-view $J_U$ ratio still worsened by
about 8.7%, so later density experiments require prospective per-seed safety and explicit per-view
reporting.

The factorial does not establish balanced scheduling: the B/A and D/C AUC ratios were
`0.9993201` and `1.0031376`, with no schedule-specific promotion gate. Retain `balanced_cycle`
only to preserve complete equal-view exposure in the next experiment. The supported intervention
is the proposal-attempt target, not the schedule itself.

The immediate variable-count study isolates parent allocation at one matched birth wave. Every
causal arm receives the same 16-clone/16-split budget and the same 835-to-867 trajectory; only the
mapping from a score to parent identity changes. This avoids confusing better parent selection
with extra capacity. A broader fixed/birth/death/birth+death comparison and convergence-selected
$N_{\mathrm{opt}}^{3D}$ remain later questions. Proposal/index complexity continues to be stated
against the ordered per-view $m_{\mathrm{opt},i}^{2D}$ list, but neither the fixed-topology result
nor the one-wave design establishes production-scale memory.

The Scholar Inbox refresh through 2026-07-17 adds four useful perspectives. Incremental 3D
Gaussian Triangulation
([2607.10690](https://arxiv.org/abs/2607.10690)) couples local plane pulling with freezing already
optimized regions; G2SR ([2607.14470](https://arxiv.org/abs/2607.14470)) analytically triangulates
3D splats from cross-view 2D-splat correspondences; Bake It Till You Make It
([2607.13808](https://arxiv.org/abs/2607.13808)) uses sparsity penalties to prune insignificant
surfels; and SpeedyGS
([2607.12656](https://arxiv.org/abs/2607.12656)) treats pruning/precision as a rate-distortion
allocation problem. For this repository they motivate separating residual/responsibility-based
birth, significance-based death, geometry stabilization, and correspondence-aware initialization,
with matched-count causal controls. They do not justify importing thresholds or claiming an
outcome under compact-teacher supervision.

### 2.7 GaussianImage_plus provider boundary (2026-07-17)

Because StructSplat is under active development, the first alternate-producer gate targeted the
clean local GaussianImage_plus checkout. Its native renderer is not the normalized StructSplat
teacher equation: it projects direct packed 2D covariance, accumulates additive color, clamps only
after summation, evaluates an integer lattice, and has a hard 256-candidate tile limit. A sealed
adapter/reference harness matched the exact frozen CUDA binary on seven rendered semantic fixtures,
one over-cap sentinel, the raw 639-component checkpoint, and a deterministic 626-component
finite-SPD subset. The independent audit recomputed every gate with both the sealed Torch reference
and a separate NumPy implementation and issued a qualified pass.

The qualification is deliberately narrow. The 13 rejected non-SPD components are not ignorable:
removing them changes 570/19,200 pixels by more than `1e-6`, with maximum clamped-channel change
`0.3718417883`. The experiment had no source RGB and therefore could not decide whether rejection
preserves image-fit quality. It also tested only one 160x120 checkpoint and one exact `csrc.so` on
an RTX 3050, without a reproducible extension build receipt.

**Reuse decision:** retain the direct-covariance adapter/reference and fail-closed tile cap as a
research mechanism. Do not yet add GaussianImage_plus to the lossless production observation
schema or claim provider quality, full-resolution scaling, memory/speed benefit, StructSplat
superiority, or downstream 3D improvement. The next provider experiment must fit selected
full-resolution calibrated views, preserve variable $m_{\mathrm{opt},i}^{2D}$, quantify the
quality cost of finite-SPD filtering before RGB is closed, and then exercise lift/refinement and
the exact gsplat viewer.

### 2.8 Exact-fiber correspondence boundary (2026-07-18)

`rtgs.lift.inverse_projection_fiber` implements the exact inverse image of one fitted 2D Gaussian:
camera-z depth and one Cholesky completion row provide the four source-null-space geometric
coordinates while the measured source center and tangent covariance remain exact.
`rtgs.lift.fiber_correspondence` adds full-covariance Bhattacharyya costs, row+dust and augmented
unbalanced-transport plans, explicit capacity accounting, and fail-closed projection/state checks.
`rtgs.lift.source_anchored_sh` similarly constrains only source-direction SH preactivation. These
are CPU-tested research modules, not registered production lifters or CLI paths.

The three-iteration synthetic line is closed. Hard row minima failed correspondence in Iteration 1;
post-hoc residual contraction could not recover a missing mode in Iteration 2. The once-only
Iteration 3 execution then completed one of three roots before a supported projection left the
valid camera domain during a frozen-plan M-step. Its raw NPZ independently rejects both real-release
arms on the completed root: UOT-area purity/completeness was `0.5468/0.25`, with track/observation
outlier recall `0.2730/0.0560`; UOT-uniform was `0.5259/0.25` and `0.2068/0.0349`. Every accepted
arm had to meet `0.90/0.90/0.80/0.80` in every root, so neither later roots nor secondary metrics
could authorize the calibrated interaction. Root-local UOT-area purity improved over hardmin by
`0.1220`, but three-root effects and area-capacity attribution remain unknowable.

The failure narrows the ray-constraint thesis. Exact source equality held to numerical precision,
yet the oracle-label arm retained center p90 `1.0593` and only `0.6125` held-out parent assignment.
The unequal-decomposition generator gives a structural explanation: 83.75% of its inlier child
centers differ from their parent moment center (p90 `0.843 px`). Moment-split fragments are image
decomposition elements, not guaranteed physical projections of the latent parent. Preserve the
exact fiber for a stable track or dynamically moment-merged source aggregate; do not equate every
independently fitted component with one physical 3D Gaussian.

**Reuse decision:** retain the exact-fiber, transport, and source-anchored-SH modules as disabled
research infrastructure. Do not run the withheld real fit or promote a backend/default. A future,
separately authorized study must first pass an oracle-topology feasibility ceiling, make M-steps
transactional with projection-valid rollback/backtracking, checkpoint failure context per arm, and
calibrate an outlier likelihood with sparse epipolar/visibility candidates. G2SR's tracked 2D splats
remain a closer premise than independently refitted mixtures; its supplied/stabilized tracks do not
establish this repository's latent many-to-many solution.

### 2.9 Field-level lift implementation boundary (2026-07-18)

The new `FieldLifter` reuses the exact inverse-projection fiber without reviving the rejected
component-correspondence premise. `SceneFits` is its typed image-free input: ordered immutable
teachers and calibrated cameras, optional lossless `PackedAlpha`, optional depth/sparse geometry,
and a required complete disjoint train/held-out partition. `run_field_pipeline` and
`rtgs lift-field --dataset ... --heldout-stride ... --field-args ... --out ...` expose the native
path; the legacy registry adapter is named `field`. In addition to the standard PLY/NPZ,
the CLI persists field masses, render opacity, source/fiber state, visibility, gains, split indices,
and correspondences in `Path(--out).with_suffix(".field.npz")`; its strict JSON sidecar preserves
the separate per-view and train/held-out semantic validation.
Unverified pre-split points/bounds never enter a held-out field fit: callers must set the explicit
`geometry_is_train_only` provenance bit, otherwise training-camera frustum consensus is used.
Trusted visible SfM points then provide footprint-gated, confidence-weighted source-ray depth
seeds; explicit per-component depth/confidence priors remain the more direct anchor.

Its inner analytic objective has a deliberately narrow contract. Whole-plane Gaussian product
kernels give exact L2 terms for additive peak-mixture density $D$ and RGB numerator $N$. A
normalized StructSplat teacher with epsilon normalization, finite support/fade, or optional affine
color does **not** render that objective. The implementation therefore keeps the analytic proxy for
refit/topology and separately performs bounded deterministic sampled queries against each frozen
teacher's actual semantics, aggregating training and held-out views independently.

The topology interface provides prune/merge/split/birth transactions. The integrated proposals
use lowest field mass for prune, bounded multi-view projected Runnalls-KL field-bound ranking for
merge, a co-located mass split, and unexplained-field residual scoring over unused source
components for birth. A directed residual split exists behind `TopologyOps` and in tests but is
not the default. Transactions are accepted only when the additive density proxy plus parsimony
decreases with visibility/gains fixed; that is a deterministic mechanism contract, not evidence
that topology improves reconstruction.

**Reuse decision:** keep this as a CPU-tested research path and reuse its decomposition-free field
loss, observability reports, exact source fiber, and immutable-teacher validation boundary. Do not
claim normalized-renderer objective equivalence, calibrated held-out quality, speed/memory
benefit, topology utility, or a production default until a frozen train/held-out experiment and
independent audit establish them. The fixed-topology sampled `CompactTrainer` and its outstanding
proposal/density questions remain separate.

### 2.10 Compact-initializer convergence update (2026-07-21)

The full `frame_00008` compact-only suite compared every applicable repository family under one
ordinary adaptive-density/convergence schedule: balanced top-K, dense+merge, easy-only,
structure-from-splats, complete field lift, random, and a disclosed historical beam-fusion anchor.
Native initial counts ranged from 7 to 5,000 and were not trimmed. All 26 views were fit, all arms
reached the joint plateau at the 70k assessment, and final topologies ranged from 35,644 to 49,177.

Dense+merge was the clear initialization-quality leader (2,088 Gaussians, 20.7546 dB fitted-view
foreground PSNR) and the final foreground-PSNR leader (38.2480 dB). Beam fusion instead had the
best selected equal-view objective, 0.002447 versus dense's 0.002555. Dense's +0.3607 dB PSNR lead
came with a 4.4003% worse objective, so no arm improved both metrics by the frozen material margins.
Random finished fourth at 37.4257 dB after growing to 39,513, illustrating how strongly ordinary
density control can recover fitted targets and confound causal attribution to initialization.

**Reuse decision:** retain all compact-native initializers as research arms and balanced top-K as
the conservative default. Do not select dense or beam post hoc from this metric split. A default
question now requires fresh multi-scene/multi-seed evidence with genuinely held-out cameras and
explicit capacity/budget control. RGB/depth/classic-SfM lifters require a separate cohort rather
than being labeled losers on data they cannot consume. The complete result and scientist pass are
`benchmarks/results/20260721_all_initializers_frame00008_{RESULT,AUDIT}.md`.

### 2.11 Beam lineage constrains covariance, not coverage (2026-07-23)

Beam Fusion's retained contributor CSR is a real Gaussian-to-Gaussian partial correspondence
structure: the reduced Janelle screen produced 800 3D hypotheses with 6,029 view links. It is not a
dense correspondence field. Only 4,704 unique fitted 2D Gaussians—11.76% of the selected input
pool—occurred in those links, and a source Gaussian may contribute to more than one retained 3D
hypothesis.

Applying Splat-SfM's linear covariance equations to those Beam tracks exposed an identifiability
failure rather than a covariance solution. The median linear residual was 0.737 and 635/800 raw
3D matrices were non-SPD. Eigenvalue clamping created extremely anisotropic splats: median maximum
sigma rose 4.59× over CI, median minimum sigma hit the `1e-4` floor, and median condition reached
178,541. The bounded result's median whitened 2D covariance residual was 13.4478, far worse than
CI's 0.6888. It therefore cannot be interpreted as better physical 3D covariance.

The same invalid-SPD repair was useful as a coverage intervention. With count, means, opacity, and
SH/color bit-identical, fitted-view initial alpha IoU rose 0.01073→0.55056, the 1,000-step
fixed-topology foreground-PSNR AUC rose 9.108%, and final foreground PSNR rose 0.5569 dB. A robust
whitened Cholesky fit reduced median residual only 7.80% versus CI and collapsed the wide scales;
its coverage and optimization returned to the CI curve. Thus local covariance fidelity and sparse
surface coverage are distinct objectives. The 2D covariance of a matched fitted component does
not imply enough footprint for a sparse 3D representation to cover the mask.

**Reuse decision:** keep CI as the Beam covariance and do not integrate either tested treatment.
Reuse the CSR only as lineage. If coverage is revisited, test a clearly named bounded scale prior
or CI/LSQ spectral blend with outside-mask leakage guards; do not present it as covariance
recovery. A physical estimator needs a PSD-constrained reprojection gate before downstream
optimization. The exact single-scene, all-fitted-view CPU result and scientist pass are
`benchmarks/results/20260723_beam_covariance_refit_{RESULT,AUDIT}.md`; no held-out, production
gsplat/density, multi-scene, or default claim follows.

### 2.12 Masked density partitions improve conditioning, not initial coverage (2026-07-23)

The sparse-lineage limitation can be addressed without inventing 3D-to-2D matches. Beam Fusion now
retains each contributor's implied depth, and `rtgs.lift.beam_partition` deduplicates its exact
native source component ids per view. Those native 2D means remain fixed anchors. Frozen order-5
Gaussian quadrature samples the full 5,000-component source mixture, removes samples outside the
packed foreground mask, and assigns every retained sample to the nearest anchor. Fixed-anchor
second moments therefore include both within-Gaussian covariance and the spatial spread of nearby
source density. Shared contributors are partitioned once even when multiple 3D tracks reuse them.

On the same reduced Janelle screen, 6,029 links became 4,704 unique view/component anchors. Every
partition had support, partition-of-unity mass error was at most `2.70e-16`, and replaying native
covariance through the newly retained exact depths reproduced CI within `1.23e-6` relative error.
The partition geometry was surprising: per-view median determinant-matching scalar covariance
multipliers were only 0.333–0.821× the native anchor covariance, while a rare tail reached
21,290×. Thus “give each survivor its surrounding density” is not equivalent to heuristic global
scale inflation.

The full partition moment produced better-behaved 3D shapes than either CI or determinant-only
scaling: median condition number fell from 15.04 to 3.89 and median whitened residual against the
new partition targets fell from 0.6888 to 0.5523. It simultaneously became worse against the
original local contributor covariances (1.0778), so neither target is physical ground truth.
Most importantly, the preregistered coverage hypothesis failed. Initial alpha IoU fell from
0.01073 (CI) to 0.00625 (`pou-area`) and 0.00886 (`pou-full`); full alpha-inside rose only 8.77%,
short of the required 25%.

There is nevertheless a reproducible optimization signal. With 800 fixed Gaussians and identical
means/colors/opacity, foreground-PSNR AUC improved 4.86% for area-only and 6.82% for full moments.
`pou-full` reached CI's final fitted-view PSNR 50 steps earlier and ended +0.118 dB with unchanged
final alpha IoU. Full shape added 1.87% AUC beyond determinant-only scaling, but the overall frozen
shape decision remained negative because it required the primary coverage gate.

**Reuse decision:** retain the native-anchor partition as an opt-in research mechanism, not a
Beam default. Its supported interpretation is better fixed-topology optimization conditioning on
one all-fitted-view CPU scene, not increased coverage or recovered physical covariance. Do not
replace it with blind covariance enlargement; persist raw partition tensors and replicate
`pou-full` against CI across scenes/seeds and untouched held-out cameras first. A later production
gsplat split/merge test must be separate because topology can turn the extreme covariance tail
into a different causal mechanism. Exact result and 70/70 scientist pass:
`benchmarks/results/20260723_beam_partition_covariance_{RESULT,AUDIT}.md`.

## 3. Depth estimation & feed-forward geometry (variant B backends)

| Model | Output | License | Integration |
| --- | --- | --- | --- |
| **Depth Anything V2 Small** ([2406.09414](https://arxiv.org/abs/2406.09414)) | relative inverse depth | **Apache-2.0** (Small only! B/L are CC-BY-NC) | HF `transformers` pipeline — our default real backend |
| **MoGe-2** (Microsoft, [2507.02546](https://arxiv.org/abs/2507.02546)) | **metric point maps** + normals | **MIT** | pip from GitHub; ~60 ms/img; ships ROE scale/shift solvers — best upgrade target |
| **Metric3D v2** ([2404.15506](https://arxiv.org/abs/2404.15506)) | metric depth + normals | **BSD-2** | one-line torch.hub |
| Depth Pro (Apple, [2410.02073](https://arxiv.org/abs/2410.02073)) | metric + focal | research-only weights | reference/eval only |
| UniDepth v2 ([2403.18913](https://arxiv.org/abs/2403.18913)) | metric + intrinsics + uncertainty | CC-BY-NC | blocked for reuse |
| **Depth Anything 3 Small/Base** ([2511.10647](https://arxiv.org/abs/2511.10647)) | multi-view depth+ray | **Apache-2.0** (S/B) | multi-view alternative |
| **MapAnything** (Meta, [2509.13414](https://arxiv.org/abs/2509.13414)) | metric multi-view + cameras | code Apache; one **Apache model** | best permissive multi-view backend |
| VGGT (CVPR 2025 best paper, [2503.11651](https://arxiv.org/abs/2503.11651)) | cameras+depth+pointmaps, <1 s many views | commercial checkpoint gated | candidate fourth lifter |
| DUSt3R/MASt3R/Fast3R/CUT3R | pointmaps | all non-commercial | avoid in license-sensitive work |

**Depth-to-covariance in feed-forward gaussian methods**: pixelSplat (MIT,
[2312.12337](https://arxiv.org/abs/2312.12337)) sets `scale = bounded_factor * depth *
||K^-1 pixel||` — the **z/f pixel-footprint prior** (verified in its gaussian_adapter);
SplaTAM ([2312.02126](https://arxiv.org/abs/2312.02126)) uses isotropic radius z/f;
MVSplat (MIT), DepthSplat (**Apache-2.0**, [2410.13862](https://arxiv.org/abs/2410.13862),
DA-V2 + cost volume) and Flash3D predict scales by network around depth-placed centers.
No published closed form for the **along-ray** dimension — ours (below) fills that gap.

**Scale alignment of relative depth**: closed-form least-squares (s, b) against projected
SfM points (Chung et al. [2311.13398](https://arxiv.org/abs/2311.13398); the official 3DGS
`make_depth_scale.py` does exactly this; MoGe's ROE solver is a permissive
implementation). Correlation losses (FSGS/SparseGS) sidestep alignment but only as a
training loss. Implemented in `rtgs/depth/align.py`.

## 4. Initialization literature (closest related work)

- **EDGS** (CompVis, CVPR 2026, [2504.13204](https://arxiv.org/abs/2504.13204),
  [code](https://github.com/CompVis/EDGS)) — triangulates dense 2D correspondences (RoMa)
  into a one-shot dense init and **disables densification entirely**; reaches 3DGS LPIPS
  in 25% of training time. **Closest published work to our idea** — but its per-image
  unit is a point match; ours is a fitted 2D gaussian carrying covariance + color.
- **InstantSplat** (NVIDIA, [2403.20309](https://arxiv.org/abs/2403.20309)) — MASt3R
  pointmaps + short joint pose/gaussian optimization, no densification (non-commercial
  stack). The "foundation-model init + short joint refine" pattern is worth copying on a
  permissive stack.
- **Init sensitivity**: "Does 3DGS need SfM init?" ([2404.12547](https://arxiv.org/abs/2404.12547))
  — good random init in the right bounding volume closes much of the gap; RAIN-GS
  ([2403.09413](https://arxiv.org/abs/2403.09413)) — splats overfit in place instead of
  relocating (a failure our dense near-surface init sidesteps). Desiatov & Sattler
  ([2603.20714](https://arxiv.org/abs/2603.20714)): **current densifiers cannot fully
  exploit dense inits** — both a warning and our opportunity; their benchmark is reusable.
- **Feed-forward per-pixel gaussians** (Splatter Image, pixelSplat, MVSplat, GS-LRM,
  AnySplat, Flash3D, DepthSplat): networks predicting dense pixel-aligned gaussians.
  Conceptually "per-view gaussians → 3D", but none uses a *fitted, sparse* 2D
  representation.
- **Novelty check (2026-07-07)**: no published work fits 2D gaussian splats per image and
  lifts those primitives into a 3DGS initialization. Nearest neighbors: EDGS, MiraGe,
  Splatter Image. Re-verify at publication time.

### 4.1 Scholar Inbox update (2026-07-14)

- **DP-GS** ([2607.03765](https://arxiv.org/abs/2607.03765), ECCV 2026) identifies reliable
  rendered depth through multi-view geometric and photometric consistency, propagates it under a
  normal prior, and regularizes depth edges that are unsupported by normal edges. **Repo
  inference:** confidence-aware derivatives or anchor losses are a better next test than globally
  smoothing every monocular-depth discontinuity.
- **Incremental Gaussian Triangulation**
  ([2607.10690](https://arxiv.org/abs/2607.10690)) represents geometric Gaussians as planar
  elliptical surfels, pulls centers toward local oriented-point planes, aligns the shortest axis
  with the local normal, and freezes optimized historical regions. **Repo inference:** the current
  surface-covariance arm is worth retaining, while a local-plane constraint and active-window
  optimization are bounded follow-ups for lifting and streaming respectively.
- **AnchorSplat** ([2607.01290](https://arxiv.org/abs/2607.01290), ECCV 2026) constrains generated
  primitives to local anchor domains and replaces iterative density growth with a learned 1-to-K
  mapping for 3DGS asset enhancement. **Repo inference:** bounded local offsets support the existing
  bounded-ray design and motivate testing anchor-local density growth; its trained enhancement
  network is not a drop-in initializer.
- **NoDrift3R** ([2607.07168](https://arxiv.org/abs/2607.07168), ECCV 2026) derives Gaussian centers
  from predicted ray origins, directions, and depths, then couples RGB, ray-map, and camera
  supervision. **Repo inference:** geometry and appearance should remain coupled during short ray
  refinement, but posed-camera experiments here do not need its pose-estimation machinery.

**Reuse decision:** these papers guide hypotheses only; no implementation is copied. The nearest
CPU-testable follow-ups are confidence-weighted bounded-ray anchoring, leave-one-source-view-out
photometric supervision, and local plane/normal consistency. The completed covariance ablation in
`docs/RESEARCH_LOOP.md` retains pluggable surface/footprint/isotropic controls because none of the
papers establishes a universal covariance construction for fitted 2D Gaussians.

### 4.2 Scholar Inbox update (2026-07-15)

- **ExtraGS** ([2607.12785](https://arxiv.org/abs/2607.12785)) applies confidence-weighted
  fine-tuning when adding uncertainty-guided pseudo-observations, so reliable regions are protected
  from generated content. **Repo inference:** confidence remains plausible when its uncertainty is
  tied to an actual observation-generation process, but this does not justify more synthetic
  bounded-ray weight sweeps after the exact attribution test failed its materiality criteria.
- **MAC-Splat** ([2607.10792](https://arxiv.org/abs/2607.10792)) argues that photometric-only sparse
  supervision leaves depth and correspondence ambiguous, and instead enforces robust world-frame
  consistency of matched Gaussian position, shape, and appearance. **Repo inference:** after the
  leave-one-source-view-out control, direct cross-view geometry/correspondence is the highest-value
  mechanistic pivot if photometric exclusion still lacks a depth signal.
- **FlowPainter** ([2607.10140](https://arxiv.org/abs/2607.10140)) uses confidence to gate a local
  prior during iterative completion rather than weighting an undifferentiated global objective.
  **Repo inference:** any later confidence mechanism should be derived from task-specific evidence
  and localized; this paper does not supply a confidence estimator or threshold for this pipeline.

**Reuse decision:** these mechanisms ground the stopping/pivot decision; no external method or code
is copied. The exact sampled-confidence and LOSO experiments closed their branches. The subsequent
fixed-pair position-only oracle test strongly localized its represented nodes but covered only about
8.4% of retained primitives and missed global materiality gates. That result motivates exactly one
denser train-only matcher experiment, closest in spirit to EDGS's dense-correspondence geometry,
while retaining this repository's bounded-ray optimization and pluggable matcher boundary.

### 4.3 Oriented-point experiment update (2026-07-15)

The next experiment implemented Incremental Gaussian Triangulation's point-to-plane and
shortest-axis-to-normal losses as an explicitly scoped repository adaptation. IGT receives
oriented RGB-D surface points and jointly treats its geometry as planar elliptical surfels. Here,
the losses were added as CPU-testable, retained-indexed Hybrid controls with zero default
coefficients, while the candidate oriented points were built from four nearby corrupted-depth
points in other training views. Separate plane and alignment normals allowed an exact within-
source shuffled-normal attribution arm.

The frozen constructor looked structurally healthy on all three seeds: 318-339 targets covered
24.59%-26.02% of retained nodes, all locality/incidence/planarity/reachability floors passed, and
the control normal separation was strong. The post-freeze clean audit nevertheless rejected every
seed before optimization. All-target plane p90 was 0.160-0.175 of scene extent and corrupted-
target p90 was 0.239-0.271, both above the 0.10 ceiling; clean-normal agreement also failed some
seed/stratum checks. Thus geometric compactness and small PCA eigenvalues did not establish that
corrupted cross-view neighbors described the correct surface plane.

**Reuse decision:** retain the generic point-to-plane/selected-axis API as disabled research
infrastructure, but reject this four-neighbor target constructor and do not run or infer utility
from its withheld arms. This is not negative evidence about IGT with valid oriented RGB-D input.
Any later plane/normal experiment must start from an independently justified pluggable
metric-depth/RGB-D oriented-point backend on calibrated data and pass a pre-optimization target
audit. Do not tune this synthetic constructor or infer an RGB-only Gradient target.

### 4.4 Real RGB-D oriented-point transfer update (2026-07-15)

The follow-up added the missing public boundary rather than embedding one data source into a
lifter. `rtgs.lift.surface` now defines immutable oriented-point prediction/provenance records, a
view-keyed backend protocol, validation/canonicalization into detached world-space maps, and a
deterministic registered-depth five-point normal estimator. The experiment-specific TUM backend
remained in `benchmarks/tum_rgbd_oriented_validity.py`; production lifting and all plane/normal
coefficients were unchanged.

The sealed target-only audit used official TUM RGB-D data and withheld appearance and utility:
`fr1/xyz` mechanically calibrated nine thresholds, while a durable atomic seal allowed exactly
one `fr1/desk` confirmatory run. Each sequence split 64 pose-only keyframes into 48 T views for
target construction, eight V views for independent audit, and eight H views whose PNG payloads
were never decoded. Construction and validation received disjoint backend capabilities. Target
identity bound the full grid eligibility mask, source pixels/poses, calibration, configuration,
and float64 point/normal tensors.

The source contract itself worked: desk eligibility was 68.44%, two-V oriented support was 80.36%,
median normal cosine was 0.887, and the 6.25% free-space contradiction rate passed its transferred
limit. Cross-view tails did not transfer. Desk symmetric surface p90 was 202.11 mm versus a
42.45 mm limit, relative-depth p90 was 25.19% versus 5%, and p10 normal cosine was 0.503 versus
0.522. Median desk errors were only 15.42 mm and 2.14%, so the failure is a heavy-tail validity
problem across supported targets, not simply an empty audit. The artifact cannot attribute that
tail post hoc among occlusion, motion, approximate calibration, and sparse construction-only
visibility because signed residuals and semantic motion labels were intentionally not exposed.

**Reuse decision:** retain the generic CPU-tested backend/canonicalization and zero-default loss
APIs, but reject this exact TUM registered-depth target/visibility protocol as a transferable
precondition and withhold Phase B. The same-pixel plane identity also means a future plane term is
incidence-weighted sensor-depth regularization along the bounded ray, not unrestricted 3D plane
pulling; any utility experiment must include an ordinary extra-depth anchor. Do not tune the
consumed desk case. A revisit requires new sequences and a separately preregistered
occlusion/rigidity attribution audit using construction-only controls and signed discrepancies.

### 4.5 Signed occlusion/rigidity attribution update (2026-07-15)

The follow-up grounded its factorization in three current mechanisms: IGT's explicit oriented
surface support ([2607.10690](https://arxiv.org/abs/2607.10690)), Grassmannian Splatting I's moving
spacetime surfels ([2607.10489](https://arxiv.org/abs/2607.10489)), and Hallo4D's motion-aware
keyframes/visibility pruning ([2607.12752](https://arxiv.org/abs/2607.12752)). The repository
adaptation did not copy those methods. It instead froze signed camera-z discrepancies and two
nested construction-only visibility arms so an occlusion-like positive tail could be separated
from observed-free-space contradictions before any loss or optimization.

On official TUM `fr3/sitting_xyz`, a stride-8 T-only cloud retained 87.83% of sparse-visible
depth-valid evidence and 27,135 supported targets. The removed population was sharply asymmetric:
30.11% behind-observed versus 2.90% in-front-of-observed, with 32.48% positive-tail recall but only
5.52% negative-tail recall. Target-paired positive enrichment and its bootstrap interval were
clearly nonzero, and relative-depth p90 improved about 8%. The primary effect size was still too
small: target-balanced positive rate decreased from 11.674% to 10.250%, short of the frozen 15%
relative-reduction floor, while target-balanced removed/retained risk ratio was 1.720 with its
entire interval below 2x. The development gate therefore failed and `fr3/walking_xyz` remained
unopened.

Dense-visible contradictions also increased 11.19 percentage points from near to far time strata;
a four-cell pose-conditioned sensitivity remained +10.01 points. This does not prove dynamic
object motion, but it shows that density alone does not absorb capture-state dependence even in
TUM's slowly dynamic sequence.

**Reuse decision:** retain the signed/nested diagnostic harness as a benchmark-only research
tool, but do not authorize plane/normal utility or reinterpret pair-weighted enrichment as a pass.
Close the pure-density branch. A new attribution study should preregister time-local or
source-conditioned T-only visibility versus the pooled construction cloud on new captures, with
pose-matched strata and an ordinary depth utility control still withheld until target validity is
established.

### 4.6 Scholar Inbox update (2026-07-16)

- **Bake It Till You Make It** ([2607.13808](https://arxiv.org/abs/2607.13808)) separates
  low-frequency surfel geometry/view-dependent appearance from high-frequency texture stored in a
  baked atlas, and couples that representation with semi-transparency/falloff sparsity penalties.
  **Repo inference:** this independently strengthens the case for an explicit geometry/appearance
  boundary, but it does not choose the Stage-1 weight/color gauge or validate the bounded 8p arm.
- **CASA-SDF** ([2607.13492](https://arxiv.org/abs/2607.13492)) combines uncertainty-aware
  curriculum supervision with curvature-dependent local representation capacity. **Repo
  inference:** curvature is a plausible prospective stratum or second-stage allocation signal, but
  it must not be added retrospectively to the already-frozen residual-responsibility protocol.
- **Residual-Christoffel Sampling**
  ([2607.13382](https://arxiv.org/abs/2607.13382)) samples operator residual equations using a
  residual-Christoffel density, inverse-density weights, and coefficient whitening to improve
  conditioning and rank. **Repo inference:** after the current residual-responsibility mechanism
  gate, a renderer-Jacobian leverage score with inverse-propensity correction is a mathematically
  grounded combination of residual allocation and preconditioning; it is distinct from simply
  ranking raw image gradients.

**Reuse decision:** none of these papers changes a consumed protocol or production default. The
new concrete follow-up is to test whether residual responsibility remains useful after controlling
for renderer-Jacobian leverage and curvature strata. That question requires a fresh prospective
preregistration after N79 Phase A, not an outcome-dependent arm added to N79.

## 5. Visual hull & space carving (variant C foundations)

- Classics: Laurentini's visual hull (TPAMI 1994); **Seitz-Dyer voxel coloring**
  (CVPR 1997) — keep voxels whose projected colors are consistent across views;
  **Kutulakos-Seitz space carving** (IJCV 2000) — the photo hull as the maximal
  photo-consistent shape.
- **torchhull** ([code](https://github.com/vc-bonn/torchhull), **MIT**, pip) — CUDA visual
  hull from masks via sparse voxel octrees; drop-in when masks exist.
- **GaussianObject** (SIGGRAPH Asia 2024, [2402.10259](https://arxiv.org/abs/2402.10259))
  — initializes 3DGS from a visual hull of ~4 masked views; proof hull-init beats sparse
  SfM in the few-view regime. Nobody has shipped **photo-consistency carving as a 3DGS
  initializer for unmasked scenes** — open space our `carve` variant occupies (using 2D
  gaussian coverage as a learned silhouette substitute).

## 6. Gaussian merging (mixture reduction)

Moment-preserving merge of components (w_i, mu_i, Sigma_i):
`w = sum w_i; mu = sum w_i mu_i / w; Sigma = sum w_i (Sigma_i + (mu_i-mu)(mu_i-mu)^T) / w`.
**Runnalls 2007** gives a cheap KL-bound merge cost for greedy pairwise reduction.
**Hierarchical 3DGS** (Kerbl et al., SIGGRAPH 2024) is the production precedent for
merging splats: weights ∝ opacity × splat size, opacity renormalized after merging.
LightGaussian ([2311.17245](https://arxiv.org/abs/2311.17245)) significance scores and
Reduced-3DGS redundancy tests are useful merge-candidate selectors. Implemented in
`rtgs/lift/merge.py` (voxel-hash grouping + moment matching, opacity 1-prod(1-a)).

The sealed equal-count audit did not reach its refinement comparison. Across seeds 0/1/2,
production-scale voxel grouping reduced 1156/1160/1155 raw Carve primitives to 1125/1129/1128:
only 2.34%-2.68% compression, 27-31 multi-member cells, and 4.68%-5.34% raw multi-member exposure.
All native moment identities and production-parity checks passed, but every seed failed the frozen
10% compression, 50-cell, and 15%-exposure floors; therefore Phase B was forbidden. This closes
the consumed grid-scale mechanism test, not the utility question and not moment matching itself.
Do not tune its voxel scale after outcome access. A future equal-count study needs an independently
motivated allocator or scene regime with material collision incidence. Official evidence is
`benchmarks/results/20260715T232244Z_cpu_carve_merge_controls_iter2_audit.json`; the independent
review has SHA-256 `190a43465ac1108a7f4964766ac32e7b7cb890ff5df15486cac937cf66fd2d74`.

## 7. Evaluation protocol (for GPU-scale runs, roadmap M2)

- Datasets: Mip-NeRF 360 (7 public scenes, every-8th-image test split, outdoor at 1/4
  resolution, indoor 1/2), Tanks&Temples (truck, train), Deep Blending (drjohnson,
  playroom), NeRF-Synthetic for sanity.
- Metrics: PSNR, SSIM, **LPIPS-VGG** (state the variant!), plus wall-clock, peak VRAM,
  final gaussian count, FPS. Checkpoints at 7k/30k iterations; for a speed paper the
  headline is **time-to-quality curves** (e.g., time to reach 3DGS-30k LPIPS).
- Reference wall-clocks (consumer GPU, per 360 scene): INRIA 3DGS ~25-40 min to 30k;
  gsplat ~19 min; accelerated ~10-15 min; DashGaussian ~200 s; FastGS ~100 s; EDGS ~25%
  of baseline. Consider [nerfbaselines](https://github.com/nerfbaselines/nerfbaselines)
  for reproducible comparisons.
- Baselines to beat: SfM-init 3DGS (gsplat Default/MCMC), EDGS, InstantSplat,
  DashGaussian/FastGS.

## 8. Decisions adopted in this repo

1. **Rasterizer**: gsplat (Apache-2.0) as the GPU backend behind `rtgs.render.base`;
   pure-PyTorch reference renderer defines semantics and keeps CPU CI honest. No INRIA
   code anywhere.
2. **Stage 1**: GaussianImage Cholesky parametrization + accumulated summation, with the
   amplitude factored as `weight * color`. Accumulated amplitude is **not identifiable as
   alpha opacity**, so lifting uses an independent low opacity prior and fuses repeated
   observations without union-inflating it. L2 loss; gradient-magnitude init at 70/30 mix
   (Image-GS). The optional StructSplat backend adds feature-aware initialization and
   residual-driven progressive growth from a configurable initial count to a separate maximum.
3. **Missing-dimension covariance** (variant B): `DepthLifter` keeps three controlled modes.
   `surface` (the default) lifts the full local depth-surface Jacobian and adds a small normal
   thickness; `footprint` uses projection-correct lateral covariance plus
   `sigma_ray^2 = grad(D)^T Sigma_2D grad(D) + (z/f)^2 s_min^2`, clamped to the lateral
   extent; `isotropic` accepts one global world-space ray sigma. Depth derivatives are
   validity-aware so zero/NaN background is not mistaken for a steep surface. A three-seed
   synthetic ablation on 2026-07-14 found no universal mode winner: surface led robust clean/noisy
   initialization slightly, while train-tuned isotropic led after merge+density refinement.
   Keep the mode pluggable until real monocular-depth evidence exists. Implemented in
   `rtgs/lift/base.py` and `rtgs/lift/depth.py`.
4. **Depth backends**: GT/mock for tests; Depth Anything V2 **Small** (the Apache one)
   as the first real backend, always through scale/shift alignment; MoGe-2 (MIT, metric
   point maps) is the planned upgrade; Metric3D v2 optional.
5. **Variant C**: coverage-based hull test (2D gaussian weight maps as soft silhouettes)
   + Seitz-Dyer color-consistency scoring on a dense grid; ray-tunnel argmax placement;
   Runnalls/Hierarchical-3DGS moment-matched merging.
6. **Refinement**: classic 3DGS screen-gradient density control remains the CPU/reference
   stack. GPU runs can use gsplat Default (including AbsGS and revised opacity) or MCMC
   (relocation/teleportation plus position noise), with canonical per-field optimizers and a
   hard primitive budget. Given EDGS and Desiatov-Sattler, evaluate with densification
   *disabled or shortened* — that is where the speed win lives (roadmap M3).
7. **Evaluation**: stage 1 and lifting consume training views only. Held-out images are used for
   explicit full-canvas, foreground, foreground-crop PSNR, and crop SSIM; synthetic
   ground-truthed scenes remain the CI-scale check, and §7 remains the publication protocol.
8. **Bounded-ray anchors**: `GradientLifter` and `HybridLifter` expose `legacy`, `normalized`,
   `valid_uniform`, `confidence`, `confidence_shuffled`, and `thresholded` modes. Normalized modes
   regularize the unjittered bounded ray fraction with stiffness matched to the historical raw-logit
   L2; confidence modes sanitize and gate optional per-pixel backend confidence. In the repaired
   2026-07-15 preregistered test, exact sampled confidence improved held-out depth RMSE by 1.15%
   across 3/3 seeds but worsened corrupted-source p90 by 0.77%, failing both materiality gates.
   `legacy` therefore remains default, and the exact uniform/shuffled modes remain research controls.
   Stop anchor-loss/lambda/threshold/weighting sweeps on this setup and pivot to cross-view
   identifiability. Do not treat controlled synthetic confidence as a deployable estimator.
9. **Source-aware photometric supervision**: both lifters expose opt-in `all`,
   `leave_one_source_out`, and `matched_nonself_dropout` training-render modes. The matched control
   uses one globally balanced non-self assignment, so every target loses its own retained count and
   every primitive is excluded exactly once. In the preregistered 2026-07-15 test, LOSO improved
   held-out depth RMSE by only 0.15% for Gradient while worsening its source p90 by 0.52%; corrupted
   Hybrid worsened held-out RMSE by 0.01%, all-source p90 by 2.31%, and corrupted-source p90 by
   1.36%. Inclusive `all` remains default. The small cross-only training-loss improvement shows the
   intervention changed the objective and motivated the direct world-frame position-consistency
   identifiability test recorded next, because photometric exclusion did not produce robust
   geometry.
10. **Fixed-pair position consistency**: `GradientLifter` and `HybridLifter` accept explicit,
    detached cross-source pair tensors through `lift_with_position_pairs`. The opt-in loss applies
    Huber after extent-normalized world-coordinate L1 discrepancy; its default coefficient is zero,
    so production behavior is unchanged. In the preregistered 2026-07-15 oracle experiment, the
    frozen 0.25-coefficient position term reduced correct-edge p90 by 86%-91% and represented-node
    assigned-GT p90 by 82%-90%, all in 3/3 seeds. Whole-scene held-out RMSE improved only 0.90%-
    1.00%, all-source p90 5.86%-6.50%, and corrupted-source p90 5.69%, below every materiality
    threshold. The graph touched only 7.73%-9.43% of retained primitives. Keep this API as a
    research mechanism, preserve inclusive/default behavior, and stop loss hyperparameter sweeps.
    The one allowed follow-up is a denser train-only matcher with the same frozen loss; if coverage
    does not propagate the local effect, pivot to local plane/normal consistency. The result does
    not reproduce full MAC-Splat or establish deployable matching.
11. **Train-only correspondence boundary**: `rtgs.lift.matching` defines a CPU-first
    `PositionMatcher` protocol plus a deterministic `PatchEpipolarMatcher`. The reference backend
    uses only train RGB, calibrated cameras, and detached retained fitted centers; it combines raw
    5x5 RGB patches, reciprocal ratio matching, epipolar restriction, and triangulation/
    reprojection filters, then emits both a positive graph and an exact-degree per-camera-pair
    cyclic control. It is a tested research backend, not a production default or a reproduction of
    learned RoMa/MAC-Splat features. In the frozen 2026-07-15 audit it broadened node coverage to
    17.99%-19.10% (1.91x-2.47x the preceding sparse oracle), but strict compositor-identity
    precision was only 9.04%-11.76% because 77.73%-82.85% of represented nodes had insufficient
    meaningful GT contribution. The preregistered gate therefore stopped before any position-loss
    arm. Do not tune this matcher or the position loss on the result. The next mechanism is local
    plane pulling and shortest-axis normal alignment from detached train-depth oriented points,
    scoped first to Hybrid; the IGT formulation assumes RGB-D geometry and cannot be claimed for
    RGB-only Gradient without an independent depth/normal backend.
12. **Oriented-point surface controls**: `rtgs.lift.surface` validates detached, retained-indexed
    world points with separate plane and alignment normals. `GradientLifter` and `HybridLifter`
    expose extent-normalized absolute point-to-plane loss and sign-invariant selected-axis normal
    loss; coefficients default to zero, and the selected scale axis is frozen from step zero. The
    preregistered Hybrid experiment never evaluated these losses: although the four-neighbor
    corrupted-depth PCA builder passed every structural floor at 24.59%-26.02% coverage, its clean
    plane p90 was 0.160-0.175 overall and 0.239-0.271 on corrupted targets versus a 0.10 ceiling.
    Reject that constructor, preserve the disabled API, and do not tune or execute the withheld
    arms. The result neither reproduces nor refutes IGT and does not authorize an RGB-only Gradient
    claim. The independent real-RGB-D backend audit is recorded next.
13. **Pluggable oriented-point input boundary**: `rtgs.lift.surface` now accepts view-keyed
    geometry/normals with explicit frames, validity, confidence, and immutable provenance, then
    canonicalizes them into detached world-space maps. A registered-depth estimator supplies
    deterministic five-point normals without adding a heavyweight dependency. The sealed TUM
    reference audit constructed plentiful targets on both `fr1/xyz` and `fr1/desk`, but desk failed
    transferred surface p90 (202.11 mm versus 42.45 mm), relative-depth p90 (25.19% versus 5%),
    and p10 normal cosine (0.503 versus 0.522). Keep the API, not the harness-local TUM backend as a
    production default; withhold all utility arms and do not tune the consumed desk case.
14. **RGB-free compact observation schema**: freeze live StructSplat fields without clamping and
    bind them to cameras in `ReconstructionInputs`; do not treat clamped converted NPZs as
    teachers. Captured equations currently have CPU-reference fixture pixel-grid parity only. Use
    exact queried field color first. The continuous null-thinned point
    proposal is an explicit research objective, not a silent replacement for discrete-pixel loss,
    and proposal component ids are diagnostic only. The old RGB-backed Carve/refinement path and
    the 835-Gaussian full-resolution artifact remain integration baselines until the new bundle is
    consumed end to end and independently audited on held-out calibrated views.
