# Inverse-projection fiber fitting, iteration 1b — reviewed preregistration

Frozen at: 2026-07-17 (Europe/Berlin)

Status: **CLOSED BEFORE ANY OFFICIAL ROOT OR SCIENTIFIC OUTCOME**

Closure note (2026-07-17): the independent preregistration review passed, but the later
implementation review failed before official execution. The free arm's Cholesky round-trip made
the realized initial covariance numerically, but not bytewise, equal to the fiber covariance;
the ambient default dtype was not fail-closed; unrelated-seed development receipts used this
official namespace under `/tmp`; and official failures lacked a durable INVALID receipt. No
iter1b scene root, depth root, rank result, fit, metric, gate, or official artifact was used.
This namespace and all six roots are retired. The replacement is
`20260717_inverse_projection_fiber_iter1c_PREREG.md`.

## Chronology

The first `iter1` preregistration was independently reviewed before any root or outcome was used.
The review failed it because several outcome-determining reductions, transforms, evaluation
rules, and ratio gates were ambiguous. That namespace and its roots are closed. Two development
modules had been added before the review returned:

```text
src/rtgs/render/projection.py
src/rtgs/lift/inverse_projection_fiber.py
```

They had not been imported, compiled, tested, or executed. They contain no experiment roots and
confer no knowledge of an outcome. They are unreviewed development code, not frozen semantics;
they must be conformed to this file and independently implementation-reviewed before an official
root is used.

The replacement namespace and fresh roots are:

```text
rtgs.inverse-projection-fiber.iter1b.v1

scene roots:          17684011, 17684012, 17684013
initial-depth roots:  17684111, 17684112, 17684113
```

An exact repository search found no occurrence of these literals before this file. The dirty
research tree is preserved, and this experiment must not use an existing `compact_*`, `anchor`,
or density-control artifact or random stream.

## Question and claim boundary

Does hard parameterization on the exact inverse-projection fiber of a spawning 2D Gaussian make
idealized multi-view 3D center, covariance, and hypothesis-wise component association
recoverable without sacrificing fit relative to a free 3D SPD control?

This iteration uses exact synthetic component observations. It excludes RGB, opacity, spherical
harmonics, Stage-1 error, visibility/occlusion, global track partitioning, merge, split, prune,
and teleport. “Correspondence” below means a per-hypothesis component association in each view.
It does not mean a globally one-to-one or physically proven track partition.

## Frozen scene and split

For each scene root, build eight anisotropic degree-zero Gaussians with
`make_gt_gaussians(n=8, seed=root)` and six cameras with
`make_ring_cameras(n_cameras=6, image_size=64)`. Project every primitive into every camera with
the production perspective-EWA equations and `0.3 I` pixel-squared dilation. Do not apply
visibility filtering; all eight finite, positive-depth component projections remain eligible.

Cameras `0..3` are optimization views; cameras `4..5` are held out. Every
`(optimization view, primitive)` pair spawns one hypothesis, ordered first by view then by
primitive, for 32 hypotheses. GT IDs are unavailable to latent arms and are used only by the
oracle/shuffled objectives and post-fit evaluation.

The paired initial-depth generator draws 32 float64 values uniformly on `[1.2,3.6]`, using the
matching initial-depth root. The interval endpoints themselves have probability zero; an exact
endpoint is a protocol error. Every arm within a replicate receives byte-identical initial
means and covariances.

## Source-fiber parameterization

The source center uses a camera-depth ray, not a Euclidean unit ray:

```text
d_cam = [(u-cx)/fx, (v-cy)/fy, 1]
mu(t) = R^-1 (t d_cam - camera_t),       t in [1.2,3.6].
```

The implementation solves with the stored, promoted-float64 `R`, rather than assuming its
float32 values remain exactly orthogonal. Depth is

```text
t = 1.2 + sigmoid(depth_logit) * 2.4,
depth_logit_init = logit((t_init-1.2)/2.4).
```

Let `n=normalize(d_cam)` and choose a deterministic orthonormal basis
`B=[T_1,T_2,n]` by projecting the coordinate axis least aligned with `n`. Recompute the
perspective Jacobian and the following tangent block at every forward pass:

```text
A = J [T_1,T_2]
Q = A^-1 (S_eff - 0.3 I) A^-T

Sigma_B(b,r) = [[Q,             Q b],
                [b^T Q, b^T Q b + exp(2r)]]

Sigma_world = R^-1 B Sigma_B B^T R^-T.
```

`b` starts at zero. `r` starts at one quarter of `log(det(Q))`, so the initial ray standard
deviation equals the geometric-mean tangent standard deviation. A non-SPD
`S_eff - 0.3 I` fails closed. The learnable coordinates are the 32 depth logits, 64 cross
coordinates, and 32 ray log-scales.

## Free control parameterization

The free control receives the materialized initial fiber means and covariances. It directly
optimizes 32 world means and a lower-triangular Cholesky factor per covariance. Cholesky diagonal
entries are `exp(log_diagonal)` and the three strict-lower entries are unconstrained. This
guarantees SPD. It has no position box; finite values and positive camera depths are mandatory.

## Exact costs

For predicted Gaussian `(p,P)` and target `(q,Q)`, define:

```text
delta = p-q
M = stop_gradient((P+Q)/2)
C_center = delta^T M^-1 delta

W = P^-1/2 Q P^-1/2
C_conic = sum_j log(eigenvalue_j(W))^2

C_geom = C_center + 0.25 C_conic.
```

`P^-1/2` is computed by symmetric float64 eigendecomposition. Any non-positive input or relative
eigenvalue is a protocol error; there is no scientific-path clamp. `stop_gradient` means the
center term trains the projected center but does not use footprint inflation to reduce center
loss.

For each hypothesis, compute one cost in each of its three non-source optimization views and
average those three scalars. Average the 32 hypothesis losses to obtain `L_non_source`.

- Latent arms take the hard minimum over the eight target components in each non-source view.
  `torch.min` first-index behavior is normative, so an exact tie chooses the lowest component
  index.
- Oracle directly selects the target with the hypothesis's GT ID.
- Shuffled directly selects `(GT_ID + 1 + target_view_index) mod 8`, a non-identity cyclic
  mapping for optimization target views `0..3`.

The free control additionally computes `C_geom` against each hypothesis's fixed spawning
observation, averages over 32 hypotheses, and optimizes:

```text
L_free = L_non_source + 25 * mean_source_C_geom.
```

The fiber loss is `L_non_source`; it has no redundant source penalty.

## Arms and optimizer

| Arm | Geometry | Non-source objective |
| --- | --- | --- |
| `free_center` | free | latent `C_center` |
| `free_conic` | free | latent `C_geom` |
| `fiber_center` | exact fiber | latent `C_center` |
| `fiber_conic` | exact fiber | latent `C_geom` |
| `oracle` | exact fiber | GT-selected `C_geom` |
| `shuffled` | exact fiber | cyclic-selected `C_geom` |

All arms use deterministic full-batch float64 Adam with learning rate `0.025`, default betas and
epsilon, no weight decay, no schedule, and exactly 400 optimizer updates. Record checkpoints
before update 1 (step 0), after every 20 updates, and after update 400. No clipping, early
stopping, retry, checkpoint selection, or outcome-conditioned rerun is allowed.

The same learning rate deliberately removes optimizer-rate confounding. Parameter scale and
gradient norms are reported; poor conditioning remains a possible explanation rather than a
hidden retuning opportunity.

## Validity sentinels

Scientific results are `INVALID` unless all sentinels pass:

1. For every official root and each of its eight GT means, use the promoted-float64 production
   cameras and form the linear projection design from world covariance `vech` to projected
   covariance `vech`. Every pair among optimization cameras `0..3` must have SVD rank exactly
   five; every triple must have rank six. Rank tolerance is
   `sigma_max * 1e-10`. Report the minimum `sigma_5/sigma_1` over pairs and
   `sigma_6/sigma_1` over triples; each must be at least `1e-8`.
2. In an exactly orthonormal, off-axis float64 construction test, maximum source-center error is
   `<=1e-8 px` and relative source-covariance Frobenius error is `<=1e-8`, both before and after
   deterministic parameter perturbation.
3. Central finite difference with epsilon `1e-6` versus autograd has relative error
   `|g_fd-g_ad| / max(1e-8,|g_fd|,|g_ad|) <=2e-4` for one depth logit, both cross coordinates,
   and one ray log-scale in an off-axis anisotropic two-camera loss.
4. Appending an exact co-located duplicate of any non-source target leaves its hard-min loss
   bit-exact.
5. At every checkpoint: loss, parameters, means, and covariances are finite; covariance
   eigenvalues are positive; every fiber depth is in `[1.2,3.6]`; and every projected depth used
   by a loss is positive.

Focused tests use unrelated development seeds and may not construct an official scene or depth
generator.

## Evaluation and denominators

After fitting, every arm—including oracle and shuffled—is evaluated with ordinary nearest
`C_geom`, never its forced training association.

- Train association accuracy excludes each hypothesis's spawning view and has denominator
  `32*3=96`.
- Held-out association accuracy uses cameras `4..5` and denominator `32*2=64`.
- Correct-track fraction has denominator 32 and requires all five evaluated non-source
  associations (three train plus two held-out) to equal the spawning primitive's GT ID.
- Consistent-track fraction has denominator 32 and requires those five IDs to be identical,
  whether correct or not.

Exact ties choose the lowest component index. All eight targets remain eligible.

Projected center and conic evaluation costs use the target with the GT ID, not a nearest target,
so a wrong association cannot make geometric error look small. Source residuals compare to the
fixed spawning component.

Report per arm and replicate:

- source center maximum and source covariance relative-Frobenius maximum;
- GT 3D center median/p90 and GT 3D covariance median affine-invariant error;
- train and held-out association accuracy;
- correct-track and consistent-track fraction;
- train and held-out GT-ID projected center/conic costs;
- fiber depth bound margin and fraction within `1e-4` of a bound;
- covariance condition number p50/p95/max;
- parameter and gradient norms at every checkpoint;
- full loss trajectory, wall time, and peak process RSS.

Use arithmetic means for accuracies/fractions. For strictly positive error summaries across
replicates, report the raw values, arithmetic mean, and a labeled geometric mean computed as
`exp(mean(log(max(value,1e-12))))`. No scientific gate silently replaces a raw value by that
floor.

## Primary gates

Iteration 1b passes only if:

1. every sentinel passes;
2. in every replicate, `fiber_conic` source center max is `<=1e-6 px`, source covariance
   relative max is `<=1e-5`, train and held-out association accuracies are each `>=0.95`,
   correct-track fraction is `>=0.90`, and GT-center p90 is `<=0.05` world units;
3. for each replicate,
   `fiber_conic_center_p90 <= max(0.01, 1.25*oracle_center_p90)`;
4. using replicate arithmetic means, fiber versus shuffled relative center improvement is
   `(shuffled_p90-fiber_p90)/max(shuffled_p90,0.01) >=0.50`, and held-out accuracy improves by
   at least `0.50`;
5. for each replicate, fiber is non-inferior to free:
   `fiber_p90 <= 1.05*free_p90 + 0.002` and
   `fiber_heldout_cost <= 1.05*free_heldout_cost + 0.01`; and
6. the free comparison is attribution-valid only when `free_conic` source center max is
   `<=0.05 px` and source covariance relative max is `<=0.01`. If not, gate 5 is
   `UNINTERPRETABLE` and the overall result cannot pass; it is reported as soft-source failure,
   not evidence that the fiber parameterization is superior.

`fiber_center` versus `fiber_conic` is exploratory. No minimum conic benefit is required on this
clean dataset.

## Three-iteration boundary

This remains the first of exactly three evidence-driven iterations.

- Iteration 2 is specified only after these mechanisms are observed. It may add noisy,
  missing, duplicated, and ambiguous observations; source appearance; ragged observation
  lineage; and one delayed validation-gated merge with a receipt.
- Iteration 3 is specified only after iteration 2 and is the calibrated-data fit.
- Split, prune, and teleport are unauthorized until their correspondence-ownership semantics
  are defined and evidence demonstrates a need.

No public capability claim, default change, global “true correspondence” claim, or real-data
held-out access is authorized by this file.
