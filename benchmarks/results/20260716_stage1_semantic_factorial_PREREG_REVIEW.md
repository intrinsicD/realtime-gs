# Independent preregistration review: Stage-1 scalar/color semantics factorial

Verdict: PASS

Unresolved blocking or major findings: none

## Scope and chronology

I independently reviewed the complete frozen preregistration against the current repository before
implementation. Its SHA-256 matched the requested frozen value:
`f53146f12894d5e804baf699b0ba0df51d5768ef708884f5a0343c523d96e1ce`.

This was an outcome-blind document and executability review. I did not implement or modify a
harness, source seam, test, preregistration, documentation, or ARA entry; construct an official
seed; run a toy or official pilot; fit, transform, lift, render, refine, or score an experiment
scene; create a seal or attempt marker; inspect an outcome; or authorize either scientific phase.
At review time the future harness, focused tests, seal, markers, and matching official result
artifacts named by the preregistration did not exist. This review file is the only file written by
the review.

## Mathematical and representation checks

- The native Stage-1 renderer consumes the component RGB amplitude `a=w*c`, so both frozen
  transforms preserve its represented additive field. `unit_weight=(1,a)` is exact up to the
  stated float32 checks. For `peak_color`, `p=max(a)` and `(p,a/p)` preserve every nonzero
  component, with an explicit zero branch. Copying `xy/chol` preserves support and ordering.
- `m=max(a)` is invariant to those product-preserving gauges. With `h=a/m` for `m>0`, `m*h=a`;
  the zero branch is defined and checked. Bilinear `o` depends only on the immutable source image
  and bit-identical `xy`, so it is also gauge-invariant. The frozen ranges follow from native
  sigmoid-bounded `w,c`, bounded source RGB, and positive Cholesky diagonals.
- The sampling rule exactly matches `rtgs.lift.base.bilinear_sample`: subtract `0.5`, clamp to the
  valid interior, and interpolate the four neighboring pixel centers. The coverage equation and
  strict `>0.05` retention rule match the current Stage-1 coverage, Depth, and Carve paths.
- For `L=[[L11,0],[L21,L22]]`, `sqrt(det(L L^T))=L11*L22`. Therefore the frozen rank score
  `rho=m*L11*L22` is proportional to the continuous integral of `m*exp(-q/2)`. The stated
  untruncated constant `2*pi` and common `q<12` constant `2*pi*(1-exp(-6))` are correct. Omitting
  either constant cannot alter a rank.
- Every `max`, division, and sampling operation is expressly detached after fitting. The protocol
  therefore correctly excludes an SMU or another smooth maximum from this experiment and avoids
  implying a fit-time gradient claim.

## Causal identifiability and capacity fairness

- The four utility arms are a complete `2 x 2` intervention: `00` is current/current, `10` changes
  only the routed scalar, `01` changes only the routed RGB, and `11` combines those exact fields.
  Mechanical bit-identity checks, per-seed minimum treatment separation, immutable fits, and one
  fit shared across all arms prevent representation drift or refitting from entering the
  contrasts.
- In Depth, the scalar can affect strict retention while RGB affects SH; in Carve, the scalar can
  additionally affect coverage/volume and RGB can affect tunnel color matching. The factorial
  effects are therefore correctly interpreted as total downstream effects of routing each
  boundary attribute, including legitimate geometry/selection mediation, rather than as direct
  parameter effects or proof of physical opacity/albedo.
- The quota `K=min_a |E_a|` is determined symmetrically before refinement or held-out access, is
  identical across arms within each seed/backend/view, and cannot require synthesizing a missing
  production primitive. The common gauge-invariant `rho` score, frozen tie-break, exact per-view
  floors, canonical output ordering, fixed topology, and equal training schedules remove count,
  per-view allocation, and optimization-exposure advantages.
- Selected source-key identities may still differ because availability and Carve placement are
  treatment consequences. The preregistration explicitly records this, reports set overlaps, and
  limits the estimand to matched-capacity boundary utility. Natural unpruned metrics are separately
  labeled count-confounded diagnostics and cannot enter a gate. Thus the design does not overstate
  this as a natural-count end-to-end or direct-effect comparison.
- Main effects, interaction, and the primary `Y11-Y00` comparison are fully specified on both PSNR
  and SSIM with seeds—not cameras or pixels—as replicates. Non-inferiority, material-improvement,
  cross-backend, validity, and attribution rules cover every pass/fail combination without a
  backend rescue, post-hoc arm selection, significance test, or threshold repair.

## Executability against the reviewed repository

- The complete `FitConfig` is accepted by the current native fitter. Native fitting keeps the
  requested fixed count, `fit_views` deterministically uses `seed+view`, and physical `SceneData`
  subsetting produces local training indices `0..8` with no testing indices.
- The frozen `DepthLifter`, `CarveLifter`, `TorchRasterizer`, and `TrainConfig` arguments match
  their current signatures and semantics. A fresh `GroundTruthDepth` instance per production
  Depth call prevents cursor carryover. `merge=False`, fixed degree zero, and fixed opacity make
  source-key reconstruction from the production concatenation order executable.
- The existing Stage-1 gauge harness demonstrates the required independent Depth-mask and Carve
  diagnostic reconstruction pattern without a second production call. Comparing covariances
  rather than quaternion rows correctly avoids the harmless quaternion sign ambiguity while raw
  output fields remain archived.
- Pruning each ordinary unmerged production output by its reconstructed source keys is executable
  with `Gaussians3D.subset`. Canonical view/component ordering remains available after ranking, and
  `densify=False` guarantees unchanged topology during the 120-step refinement.
- The current Trainer seeds one device-local generator and, with random background and density
  disabled, consumes it once per step for the training-view `torch.randint`. The separately
  reconstructed CPU schedule is therefore the correct expected stream. `eval_every=30` provides
  the frozen post-update checkpoints; step zero can be rendered before training without selecting
  a checkpoint.
- Held-out final metrics match `image_metrics`: full-canvas clamped MSE/PSNR and the repository's
  single-scale 11x11, sigma-1.5 SSIM. Accumulated depth divided by clamped alpha matches the
  renderer's depth convention and remains diagnostic only.

## Isolation, parity, and fail-closed routing

- Phase-A and Phase-B seeds are disjoint from one another and from the prior gauge audit. Only the
  nine physically subset training views enter fitting, candidate construction, lifting, capacity
  matching, or Trainer. The minimal lift scene drops `gt_gaussians` and held-out fields while
  retaining only explicit training data and world priors.
- Phase B guards held-out RGB, cameras, and depths until every natural lift, matched initialization,
  final model, count, schedule, hash, and pre-unlock gate is frozen. One global unlock permits
  reporting only; there is no post-unlock model update, selection, retry, or repair. Phase A never
  reads a held-out field.
- Source-render equivalence is a global prerequisite. Candidate fields, coverage, retention,
  exact-product controls, ordinary-production/independent parity, source keys, output fields, and
  nine-view renders must all pass per seed, gauge, representation, and backend. No averaging or
  successful backend can rescue one failure.
- Phase B is inaccessible without a valid Phase-A artifact and an independently recomputed,
  hash-bound machine review whose verdict is `PASS` and whose authorization bit is true. Both
  phases preflight and exclusively bind every valid and invalid path before an official seed is
  constructed; interruption consumes the named attempt.

## Raw-evidence and provenance checks

- The uncompressed `numpy.savez` sidecar forbids pickle/object arrays and supplies every decisive
  fitted, transformed, routed, source-key, coverage/volume, lift, model, render, target, metric,
  schedule, capacity, and rank tensor needed for an independent recomputation. Numeric null values
  have explicit defined masks, and all floating arrays must be finite.
- The per-array digest has a single canonical little-endian dtype/shape/data domain independent of
  logical name. The sorted collection digest and common coverage-array reference eliminate the
  label-dependent hash ambiguity that qualified the prior audit. JSON and result notes bind the
  completed raw archive, while the pre-run marker truthfully binds only prospective paths.
- The seal freezes the preregistration, reviewed implementation, tests, loaded repository source,
  full revision/dirty state, environment, configurations, artifact rules, and both command
  templates after the complete CPU verification sequence. Scientific commands rehash those inputs
  before claiming an attempt.
- The claim boundary is complete: a Phase-A pass establishes only operational gauge invariance;
  Phase-B conclusions are limited to fixed-count, fixed-budget CPU synthetic evidence. No real,
  CUDA/gsplat, performance, physical-opacity/albedo, natural-count, upstream-training, or default
  claim is authorized.

## Reviewed source binding and checks performed

The relevant source snapshot inspected for executability had these SHA-256 values:

```text
390c6940bea8f4f1c80df19396a38ee29585dfd3127c8a3823654ffe09098351  src/rtgs/core/gaussians2d.py
2a9b76d41e83cc444fa98b3a0f3aa45eb8b6032806fa3d899377acfd98257e18  src/rtgs/image2gs/fit.py
d0bd6b90b8a690a2ebb36cbc55c8cceb56c3fc33c04fd3895a123e0abb660144  src/rtgs/image2gs/renderer2d.py
b19fa04733c42c1bb5c210e1ac2fced73b7fcbfe7b0d7521ec62fd1d68ba503d  src/rtgs/lift/base.py
19e2e59d8c8a32d1dcc1b86b79364c09f845c493956cb12ea94597fadd874021  src/rtgs/lift/depth.py
35135d6c93de3a836c9f9843fdfc63e3c08d973a080a9743372d3ee057a829eb  src/rtgs/lift/carve.py
3bb73a2071ff3525c07c0d1a57387ecccb5b5f16a3cc18398091b2606752053f  src/rtgs/optim/trainer.py
d489c07c65ac4c74f0f927d41c62b887724cf3216f2ef28a116ff169d08272d4  src/rtgs/core/metrics.py
3fa557f03bab5eb7666476968e0a70ff3e5639d6e24251807905691df36004c3  src/rtgs/data/scene.py
b2b16f02a92c89003439062085e39d1f5ced2cc9ebaf5b8874cf80c0fd4d70b2  src/rtgs/data/synthetic.py
61716787329e85a186982f81c2a89cb270255473ca26688c409191a1b53bd86e  src/rtgs/render/torch_ref.py
```

I read `CLAUDE.md`, the repository experiment and results-audit procedures, the full frozen
preregistration, the current source paths above, the prior Stage-1 gauge reconstruction seams, and
the existing adversarial preregistration/implementation-review conventions. I verified the frozen
preregistration hash, inspected current call signatures without constructing a scene, and checked
that the review path was absent before writing it. No scientific, seal, marker, fit, lift, render,
training, metric, or official-seed command was executed.

This PASS clears the frozen preregistration for an independent implementation and implementation
review only. It does not certify a future harness, authorize sealing, authorize Phase A or B,
validate a result, or support a scientific/default claim.
