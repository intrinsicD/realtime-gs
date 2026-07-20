# Iteration 3 preregistration addendum 3 — capacity-normalized evaluation and sealed execution

Date frozen: 2026-07-18 (Europe/Berlin), before any official Iteration 3 synthetic root,
optimizer outcome, or calibrated-bundle fit.

This prospective addendum follows the base preregistration (SHA-256
`59f0de21da20bb5785e2c5f14c89fc82114fed2d5945c704115d64b9fb3c27c8`), valid-ray
Addendum 1 (SHA-256
`f4ef57320edf1e099c24033753bf3e939d2c87fcf6b927b65bd5d6af213c91fc`), and
consistent-dust-cost Addendum 2 (SHA-256
`2fbb29d2bdea86018009d1b3913820edda38de9f3881ae503eca9041c2c2eddc`). It
freezes evaluation denominators, transport-validity checks, information barriers, and one-shot
execution semantics exposed by the final pre-outcome implementation review. It changes no root,
camera, split pattern, optimizer schedule, arm, cost, scientific comparison threshold, real camera
role, anchor budget, or appearance schedule.

The official synthetic ATTEMPT, RESULT, and artifact paths and the official real ATTEMPT, RESULT,
and output paths were confirmed absent immediately before this freeze. No official synthetic root
has been constructed. The official real runner has not decoded the bundle. The development-only
valid-ray preflight already disclosed in Addendum 1 remains the only Iteration 3 bundle access;
validation fields and C1004 RGB/mask data remain sealed.

## Frozen fitting semantics

1. For a target view, tracks sourced by that same view are removed from the active row set before
   row-softmax or augmented transport is solved. They are scattered back afterward as structural
   zero rows. Their excluded capacity cannot change the active tracks' augmented marginals,
   dustbin mass, normalization, or M-step. The exact source observation remains a constraint, not
   matching evidence.
2. All arms use one renderer-compatible projection-validity mask: finite center, finite SPD
   covariance, finite depth, depth beyond the EWA near plane, and footprint inside the camera.
   Invalid hardmin rows are structural zero; invalid soft-assignment rows are routed to dust.
   Invalid values are replaced only by harmless finite placeholders while costs are formed. They
   never receive real supervision.
3. Plans are detached before geometry updates. Each M-step objective is the equal mean over active
   camera views of that view's real-mass-normalized expected Bhattacharyya cost. Root metrics are
   likewise equal camera-view means. Thus a view cannot dominate merely because its decomposition
   has more observations or its unbalanced plan realizes more total mass.
4. Every plan and raw fiber state is validated fail-closed for declared shape, finiteness,
   nonnegative mass, strict interior depth, positive finite ray-variance innovation, finite world
   geometry, and positive-definite 3D covariance. A missing or invalid plan, a source-invariant
   violation, or insufficient total real mass terminates the attempt; it is not converted to a
   favorable metric.

## Frozen capacity-normalized metrics

Unbalanced transport does not promise exact realized marginals. Using realized row or column mass
as the denominator could therefore improve a support or dust score by destroying mass. For every
view, evaluation instead uses the marginal capacities declared to the solver.

- Parent completeness is the fraction of the eight hidden parents for which, in every fitting
  view, correct-parent real mass divided by the active source rows' declared track capacity is at
  least `0.20`. The support must be real and label-correct; dust or wrong-parent mass does not
  count.
- Track-side outlier dust recall and inlier dust false-positive rate divide track-to-dust mass by
  the corresponding outlier/inlier declared track capacity.
- Observation-side outlier dust recall and inlier dust false-positive rate divide dust-to-
  observation mass by the corresponding outlier/inlier declared observation capacity.
- Realized-mass conditional fractions are still saved as diagnostics, but never decide a gate.
  The absolute D gate and transport-arm acceptance require both dust routes separately: each
  outlier recall is at least `0.80` and each inlier false-positive rate is at most `0.20`.

For C, D, and S, every view must additionally pass these deliberately broad numerical-validity
checks:

- realized-to-declared real-track-row, real-observation-column, and total-augmented mass ratios are
  each within `[0.20, 5.0]`;
- maximum componentwise relative error against a positive augmented target marginal is at most
  `4.0`;
- the final generalized log-Sinkhorn fixed-point residual is at most `0.05`.

These bounds detect collapsed, exploded, or unconverged transport; they are not a claim that an
unbalanced plan should reproduce balanced marginals. They are validity gates and cannot rescue a
failed purity, completeness, dust, attribution, or negative-control gate.

## Frozen release and information barriers

All non-oracle state and plan tensors are frozen and hashed before the oracle or evaluator labels
are invoked. Held-out cameras, projections, and labels are materialized only after every fitted
state, final plan, and support decision is frozen. Evaluator labels cannot influence a non-oracle
gradient, plan, arm choice, or checkpoint. Invalid held-out projections are excluded from the
held-out denominator and their per-view incidence is saved.

Real interaction is permitted only if the complete three-root, six-arm official synthetic result
passes all validity checks and at least one transport arm satisfies, in every root: purity and
capacity-normalized completeness at least `0.90`, both outlier dust recalls at least `0.80`, both
inlier dust false-positive rates at most `0.20`, and its transport-mass diagnostics. D (`uot_area`)
is the primary whenever it is accepted; C (`uot_uniform`) is the fallback only when D is not
accepted and C is. Both remain paired real diagnostics. The real runner independently recomputes
the eight frozen synthetic validity booleans, both arm acceptances, and this area-first disposition
from cross-hashed root results rather than trusting mutable top-level summaries.

The official synthetic runner reserves and fsyncs an exclusive ATTEMPT receipt before constructing
an official root and refuses occupied ATTEMPT, RESULT, or artifact paths. The official real runner
requires the exact synthetic result, official paths, explicit confirmation, bundle manifest, and
protocol/source receipts; it reserves and fsyncs its own exclusive ATTEMPT before creating output
or loading the bundle. A crash consumes that namespace. Both results bind executed-source hashes,
protocol hashes, inputs, and artifact descriptors, and both runners refuse source changes during
execution. Wall time remains descriptive because other host workloads are not controlled.

## Appearance and implementation receipt

The degree-one source-anchored SH equality applies to SH **preactivation in the implemented basis**
at the source direction. It is not an equality to the isolated post-activation or alpha-composited
physical color of a scene point. Rendering opacity remains the independent fixed value `0.1`.

Before this addendum was bound into the runners, the focused CPU suite for the fiber, transport,
synthetic harness, real harness, and source-anchored SH passed 83 tests. The pre-outcome core hashes
were:

- `src/rtgs/lift/inverse_projection_fiber.py`:
  `4e9bd0c62954b2361d2cb79491d97cdfbad29c7c2dd23be0e5f5bff7e75cdd8b`;
- `src/rtgs/lift/fiber_correspondence.py`:
  `0e0605e1ecea00fd3ccfe6c9585e23594f1b114f9d4135ce1a9bae21de744453`;
- `src/rtgs/lift/source_anchored_sh.py`:
  `c38424bcb46fd0cbd883a1fe179e359c0388bec8ceb3380ea9d9dff35f06c6d3`.

The post-binding runner and test hashes are intentionally not embedded here, avoiding a circular
receipt; each ATTEMPT records the final executed sources, and the prospective implementation review
records the final harness/test hashes before execution.

## Claim effect

This addendum prevents mass deletion, source-capacity leakage, invalid projections, evaluator
leakage, summary editing, or reruns from manufacturing a positive result. It establishes no
association, geometry, calibrated-data, appearance, topology, performance, or novelty outcome.
Real tracks remain inferred associations, not true correspondences, and a failed official run
remains a failure under the frozen claim boundary.
