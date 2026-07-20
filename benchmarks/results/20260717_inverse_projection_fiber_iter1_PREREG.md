# Inverse-projection fiber fitting, iteration 1 — preregistration

Frozen at: 2026-07-17 (Europe/Berlin)

Status: **CLOSED BEFORE ANY OFFICIAL ROOT OR SCIENTIFIC OUTCOME**

Closure note (2026-07-17): an independent pre-run review found that the loss reduction,
evaluation association, transform, tie, and ratio-gate definitions below were not sufficiently
executable. Development implementation had begun, but no focused or official test, random root,
optimizer, metric, or scientific outcome had run. This namespace and all six roots below are
permanently retired. The outcome-neutral replacement is
`20260717_inverse_projection_fiber_iter1b_PREREG.md`.

## Question and claim boundary

The user proposes lifting every observed 2D Gaussian into 3D, retaining its known source
projection exactly, and using the other calibrated views to infer depth, 3D covariance, and
cross-view correspondence. This first iteration asks only whether that geometric mechanism is
identifiable and optimizable under exact synthetic observations.

This iteration does **not** test RGB, opacity, spherical harmonics, Stage-1 approximation error,
occlusion, split, merge, prune, or teleport. It cannot establish that the full reconstruction
method works. It can falsify the geometric core before those mechanisms are added.

The fresh experiment namespace is:

```text
rtgs.inverse-projection-fiber.iter1.v1
```

It is independent of every existing `compact_*`, `anchor`, and density-control experiment.
Implementation and artifacts must use only `inverse_projection_fiber` names. Existing dirty-tree
changes are user-owned and must not be rewritten or pooled with this result.

## Hypothesis

For a 2D Gaussian spawned in source camera `s`, parameterizing its 3D mean and covariance on the
exact inverse-projection fiber will:

1. preserve its source center and effective EWA covariance to numerical tolerance throughout
   optimization;
2. recover the correct physical 3D primitive and its cross-view component correspondences from
   three additional calibrated views; and
3. perform no worse than a free 3D mean/SPD-covariance parameterization supervised by a soft
   source reprojection penalty.

Adding projected-covariance mismatch to non-source matching is exploratory in iteration 1. Its
causal utility will be reported, but no minimum benefit over center-only matching is required on
clean data.

## Geometric model

For source center `u_s`, camera center `C_s`, and its camera-depth ray `d_s`,

```text
mu(t) = C_s + t d_s,            t in [1.2, 3.6].
```

At `mu(t)`, let `J_s` be the renderer's perspective Jacobian and let
`B=[T_1,T_2,n]` be an orthonormal camera-space basis whose third column is the source ray.
With `A=J_s[T_1,T_2]`, target effective covariance `S_eff`, and renderer dilation
`D=0.3 I`,

```text
Q = A^-1 (S_eff - D) A^-T

Sigma_B(b,r) = [[Q,             Q b],
                [b^T Q, b^T Q b + exp(2r)]],

Sigma_world = R_s^T B Sigma_B B^T R_s.
```

The learnable covariance null coordinates are `b in R^2` and `r in R`. They span the three
degrees of freedom left by one projected 2D covariance. Any source covariance for which
`S_eff - D` is not SPD is rejected rather than clamped silently.

The free control optimizes an unconstrained world mean and a six-parameter Cholesky SPD
covariance from the identical initial 3D state. It receives the fixed source observation as a
soft identity loss. The fiber treatment receives no redundant source penalty.

## Synthetic observations and frozen roots

Each replicate uses eight anisotropic degree-zero 3D Gaussians from the repository synthetic
generator, six established ring cameras at `64 x 64`, and exact per-component projections from
the production EWA equations, including `+0.3 I` dilation.

```text
scene roots:          17683011, 17683012, 17683013
initial-depth roots:  17683111, 17683112, 17683113
```

These roots and the domain literal had no exact repository occurrence when frozen. Focused unit
tests must use unrelated seeds and may not invoke an official root.

Cameras `0..3` are optimization cameras. Every `(optimization view, physical primitive)` pair
spawns one independently optimized 3D hypothesis, for `4 * 8 = 32` hypotheses. Cameras `4..5`
are held out from fitting. Ground-truth primitive IDs are retained only for evaluation and the
explicit oracle/shuffled controls.

Initial source depths are independent uniform draws on `[1.2, 3.6]`. Fiber cross terms start at
zero, and ray standard deviation starts at the geometric mean source-tangent standard deviation.
The free arm starts from the exact same means and covariances.

## Loss and matching

For predicted and observed 2D Gaussians, the dimensionless center cost is symmetric Mahalanobis
distance under their mean covariance. The conic cost is squared affine-invariant SPD distance.
The geometric cost is:

```text
C_geom = C_center + 0.25 C_conic.
```

For each predicted hypothesis and non-source view, latent correspondence is the hard minimum
over that view's observed components. Hard minimum is deliberate: it is invariant when a
non-source observation is replaced by co-located identical children, while a normalized soft
minimum is not. No component amplitude, opacity, color, or confidence enters correspondence.

The free control's fixed source identity term is:

```text
25 * C_geom(predicted source projection, spawning observation).
```

All arms use deterministic full-batch float64 Adam for 400 updates. Fiber learning rate is
`0.03`; free learning rate is `0.02`. No checkpoint, threshold, or hyperparameter may be chosen
from official outcomes.

## Frozen arms and sentinels

The four factorial arms are:

| Arm | 3D parameterization | Non-source latent cost |
| --- | --- | --- |
| `free_center` | free mean + free SPD covariance | center |
| `free_conic` | free mean + free SPD covariance | `C_geom` |
| `fiber_center` | exact source fiber | center |
| `fiber_conic` | exact source fiber | `C_geom` |

Two diagnostic controls use the `fiber_conic` parameterization:

- `oracle`: the evaluator-provided correct component association replaces latent matching;
- `shuffled`: a frozen per-view cyclic derangement replaces latent matching.

Before scientific utility is evaluated, the implementation must pass:

1. projection-design rank `5` for two generic views and `6` for three generic views;
2. maximum source-center error `<= 1e-8 px` and relative source-covariance error
   `<= 1e-8` in float64 construction tests;
3. central finite-difference versus autograd relative error `<= 2e-4` for depth, both covariance
   cross coordinates, and ray log-scale at an off-axis anisotropic case;
4. exact non-source co-located split invariance to `1e-12`; and
5. finite values, positive covariance eigenvalues, and bounded fiber depth at every checkpoint.

Failure of a sentinel makes the scientific result `INVALID`, not negative.

## Metrics

Reported per arm and replicate:

- source center maximum error and source covariance relative Frobenius maximum error;
- GT 3D center median and p90 Euclidean error over all 32 hypotheses;
- GT 3D covariance median affine-invariant error;
- train and held-out correspondence accuracy, using IDs only after fitting;
- train and held-out projected center/conic costs;
- depth-bound margin and fraction within `1e-4` of either bound;
- covariance condition-number p50/p95/max;
- loss trajectory, runtime, and peak resident memory.

All aggregate comparisons use the geometric mean across the three independently rooted
replicates where defined. Raw replicate results remain visible.

## Primary gates and falsification

The geometric mechanism passes iteration 1 only if all of the following hold:

1. every validity sentinel passes;
2. `fiber_conic` source-center maximum is `<= 1e-6 px` and source-covariance relative maximum is
   `<= 1e-5` in every official replicate;
3. `fiber_conic` train and held-out correspondence accuracy are each `>= 0.95` in every
   replicate;
4. `fiber_conic` GT-center p90 is `<= 0.05` world units in every replicate;
5. `fiber_conic` geometric-mean GT-center p90 is no more than `1.25x` the oracle value;
6. relative to `shuffled`, `fiber_conic` improves geometric-mean GT-center p90 by at least `50%`
   and correspondence accuracy by at least `0.50`; and
7. relative to `free_conic`, `fiber_conic` is non-inferior: GT-center p90 and held-out projected
   cost may each be at most `5%` worse.

Any failed gate falsifies the corresponding idealized claim. Results may motivate iteration 2,
but thresholds and failed arms may not be repaired or rerun in this namespace.

## Iteration discipline and next-step boundary

This is iteration 1 of exactly three evidence-driven iterations required by
`docs/RESEARCH_LOOP.md`.

- Iteration 2 may be designed only after iteration-1 mechanisms are observed. It is expected to
  add noisy/missing/duplicated observations, source appearance, explicit observation lineage,
  and a delayed merge with a receipt.
- Iteration 3 may be designed only after iteration 2. It is the sole calibrated-data fitting
  iteration and must freeze development/validation/held-out access before execution.
- Split, prune, and teleport remain unauthorized until a lineage-preserving definition and a
  demonstrated need exist.

No README capability claim, default change, or statement of “true correspondence” is authorized
by this preregistration alone.
