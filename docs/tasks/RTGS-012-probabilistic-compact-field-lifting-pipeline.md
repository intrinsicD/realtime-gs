# Current Task

## Title

Probabilistic compact-field lifting pipeline and protected mechanism experiment

## Task ID

RTGS-012

## Role Assignment

- Driver: Codex-probabilistic-field-driver
- Reviewer: Codex-probabilistic-field-protocol-reviewer
- Turn: none

## Mode

Implement

## Risk

Protected

## Maturity

- Target: Pipeline-integrated
- Reached: Pipeline-integrated

## Goal

Turn the literature synthesis into a clearly documented, CPU-first, opt-in compact-field
pipeline that combines explicit probabilistic support, shared-latent soft transport, analytic
source-projection fibers, observability-aware covariance fitting, projection-nonlinearity topology
proposals, reversible progressive-view refinement, and independent-half stability diagnostics;
then create a frozen experiment that can kill or support each added mechanism separately.

## Motivation

RTGS-011 identified a potentially useful combination but deliberately produced no implementation
or repository evidence. The repository already contains most required low-level mechanisms, plus a
failed earlier component-correspondence study. The next step is therefore a small, opt-in
integration with explicit rollback and ablations—not a new learned converter or a default change.

## Success Criteria

- A dedicated Markdown design lays out every stage, inputs/outputs, expected benefit, likely
  failure, fallback, masked/unmasked behavior, complexity, and evidence needed for promotion.
- The implementation reuses `SceneFits`, `FieldLifter`, exact inverse-projection fibers, analytic
  field refit, visibility, topology, and validation; disabled controls retain the incumbent
  algorithmic path and its existing deterministic regression contracts.
- Soft transport is projection-gated, capacity-weighted, source-view-excluding, CPU-first, and
  transactional; failure is raised or explicitly rolled back according to a frozen policy.
- Floating masks act only as geometric support probabilities and never silently become render
  opacity or association confidence.
- Topology can select a ray-depth split candidate from a sampled perspective-nonlinearity score,
  while Runnalls-screened representative-fiber merge and exact field-objective acceptance remain
  explicit limitations.
- Progressive-view refinement ends with a frozen full-view cleanup interval and reports its
  active-view schedule.
- An independent-half wrapper fits disjoint training-camera halves and reports world-frame
  stability without calling it accuracy or resolution.
- Deterministic CPU tests cover configuration, invariants, rollback, masked/unmasked behavior,
  topology selection, scheduling, half isolation, and the public pipeline entry.
- One draft task-first experiment freezes synthetic exact/refitted-field arms, mask conditions,
  native baselines, metrics, stopping rules, a calibrated micro-capture follow-up, and an exact
  command; it validates but is not initialized without a distinct prospective reviewer.
- A scratch calibrated-data interaction exercises the implementation without retaining or citing
  its outcome, and full repository verification passes.

## Constraints

- Opt-in research path only; no default, production backend, paper claim, benchmark-table, or
  maturity promotion beyond available evidence.
- Preserve the earlier failed correspondence evidence: the new path must add transactionality,
  a finite projection-compatibility gate, explicit capacities, and a hard/native control rather than relabeling
  the old result.
- Keep imports CPU-first and optional mechanisms behind existing public seams.
- No held-out field, RGB, mask, or geometry may influence fitting, scheduling, topology, stopping,
  or hyperparameter selection.
- Do not initialize or execute the official experiment until a distinct reviewer approves the
  exact protocol digest with no outcome access.
- Preserve all unrelated `.idea/` changes and all append-only experiment evidence.

## Non-Goals

- A true globally coupled multi-marginal OT solver; the first implementation is a shared-latent,
  per-view unbalanced transport approximation and must be named accordingly.
- Learned optical flow, learned depth, source-image loading, or a GPS-Gaussian reproduction.
- Claiming that independent-half agreement measures reconstruction accuracy or spatial resolution.
- Proving GPU speed, real-time behavior, cross-scene generalization, or superiority to standard
  RGB-backed 3DGS.
- External publication, production-default promotion, or treating this development run as
  confirmatory evidence.

## Selected Skills

- rtgs-core
- rtgs-task-workflow
- rtgs-research-ideation
- rtgs-experiment
- rtgs-review
- realtime-gs-results-audit
- rtgs-docs-sync
- rtgs-verify
- research-manager

## Experiment Contract

experiments/tasks/20260805_probabilistic_field_pipeline_association_rollback_mixed.json

## Current Evidence

- Accepted documentation input: `docs/tasks/RTGS-011-2d-to-3d-gaussian-literature-review.md`.
- Existing implementation anchors: `src/rtgs/lift/field_lifter.py`,
  `field_refit.py`, `field_topology.py`, `field_observability.py`,
  `inverse_projection_fiber.py`, `fiber_correspondence.py`, and
  `src/rtgs/data/field_inputs.py`.
- Earlier exact-fiber correspondence evidence is negative: Iteration 3 failed its projection
  domain during a frozen-plan M-step and the completed root failed every preregistered release
  gate. This task may reuse the infrastructure only with the stated repairs and controls.
- The incumbent field path already separates density mass from render opacity, uses visibility and
  gain, screens merges by a projected Runnalls bound, transactionally accepts topology under an
  analytic field objective, and evaluates held-out teachers only after fitting.
- Starting commit: `36630c7fef14c0907134d2f3c532be3da4a0c43e`; the worktree also contains
  the accepted but uncommitted RTGS-011 documentation/ARA changes and unrelated owner IDE changes.
- Pipeline contract: `docs/DESIGN_probabilistic_field_pipeline.md`; architecture, README, research
  index, and repository map link the opt-in path without promoting it.
- Implementation: optional probability support, explicitly finite-gated shared-latent per-view
  transport with transactional rollback, nonlinearity-selected ray-depth split proposals,
  progressive active-view/gain/refit scheduling with an all-view cleanup, elapsed-time reporting,
  and an independent-half wrapper that drops geometry without half-local provenance.
- Deterministic tests cover masked/unmasked mass-opacity separation, explicit gating and rollback,
  split conservation/selection, active-view gain isolation and cleanup, half-view/geometry
  isolation, public entry parity, CLI nesting/serialization, and the protected plan compiler.
- The earlier exact draft task and compact-byte seal validated with an outcome-free 489-cell plan.
  The owner has now widened the requested execution surface to every Gaussian2D field set under
  `dataset/`; that earlier digest is superseded and must be resealed/re-reviewed before outcome
  access.
- One disposable calibrated `frame_00008` compact interaction completed through probability
  support, field-mass transport, progressive refit, nonlinearity topology, semantic validation,
  and serialization. Its temporary output was moved to trash and no number was retained or cited.
- `./scripts/verify.sh` and the complete CPU-only pytest suite, including slow tests, pass after the
  final source changes. Both emit only the two pre-existing PyTorch warnings in
  `test_init_preserving_density` and `surfel_lift`; no CUDA/GPU test or benchmark ran.
- The prospectively approved association-rollback successor completed its immutable producer run
  under `runs/20260805_probabilistic_field_pipeline_association_rollback_mixed`: all 483 synthetic
  cells, one discarded warmup, and all 66 calibrated terminals completed. The independent audit
  accepts 59 successes and seven explicit candidate failures with no imputation or native
  substitution; native is `33/33`, candidate is `26/33`, and two successful cells use the exact
  whole-cell unmasked fallback.
- The frozen mechanism decisions are rank-aware shape `3/3` pass, probability support `3/3` pass,
  and association/topology/scheduling `0/3` fail. The latter three variants are retired for this
  protocol. Combined calibrated metrics are descriptive only, and host timing/RSS supports no
  performance claim. Canonical audit:
  `benchmarks/results/20260805_probabilistic_field_pipeline_association_rollback_mixed_AUDIT.md`.
- The root and eleven per-dataset HTML reports are rendered. A final Chrome/WebGL2 smoke checked
  `586/586` local report targets and all `11/11` orbit viewers for ready diagnostics,
  non-background pixels, and camera change. Strict and rich receipts are retained in the run root;
  seven canonical repository evidence artifacts are mirrored byte-exactly under the run root only
  so source-bound repository-relative links resolve from the frozen run-root HTTP server.
- The stale lifecycle test is repaired downstream without modifying the frozen producer. The
  focused six-file pipeline/protocol/contract suite and final `./scripts/verify.sh` pass; the latter
  covers Ruff, formatting, all non-slow CPU tests, docs sync, 40 ARA claims, script layout, agent
  workflow, and experiment contracts with only the two established PyTorch warnings.

## Minimal Plan

1. Completed: archive RTGS-011 under the owner's documentation acceptance and freeze this
   implementation task.
2. Completed: write the pipeline design and expectation/failure table before behavior changes.
3. Completed: implement the smallest opt-in integration and deterministic CPU tests.
4. Completed: widen the protected task to all 11 Gaussian2D field sets, implement deterministic
   bounded workers and canonical per-dataset comparison pages, then freeze and obtain a distinct
   prospective protocol review.
5. Completed: preserve the immutable support-fallback failure, review and execute the
   forward-AABB-eligible successor, obtain an independent results audit, render/smoke every report
   and orbit viewer, log the evidence, and verify.
6. In progress: obtain a distinct final closeout review, archive the accepted task record, rerun
   the workflow gate, and hand off the live reports/viewers without committing or publishing.

## Status

Accepted

## Human Decisions

### Proceed from literature synthesis to implementation

#### Question

Should the staged pipeline be documented, implemented, and given a protected experiment?

#### Options

- Implement the bounded opt-in pipeline and create the experiment.
- Keep the work as literature-only hypotheses.

#### Recommendation

Implement only the smallest testable combination, preserve native controls, and stop before
official outcome access until prospective review.

#### Decision

The owner explicitly requested the Markdown pipeline, implementation, and experiment on
2026-08-05.

#### Date

2026-08-05

### Execute the protected matrix over every local Gaussian2D field set

#### Question

Should the earlier draft-only protocol be widened into a bounded development execution over every
`gaussians2d*` manifest in `dataset/`, with a canonical comparison page and orbit viewer per set?

#### Options

- Freeze, review, execute, audit, and present all discovered field sets with explicit scalability
  approximations.
- Retain the two-set draft and leave outcome production blocked.

#### Recommendation

Run every discovered field set, but preregister a deterministic component cap for the 100k/view
teachers, retain original/used counts in every receipt, and label all results development-only.

#### Decision

The owner explicitly requested execution on every Gaussian2D field in `dataset/`, one `index.html`
per set with comparison curves, and an opened orbit viewer for every completed result on 2026-08-05.

#### Date

2026-08-05

## Handoff Log

Append Driver handoffs, Reviewer verdicts, and session-completion entries in chronological order.
Use `###` for entries and `####` for their fields so entries remain nested below this section.
Never delete earlier entries. On terminal closeout, archive the complete record as
`docs/tasks/<task-id>-<slug>.md`, change the archived `Turn` to `none`, and reset this file to the
unchanged template.

### Handoff

#### Objective

Review the opt-in probabilistic compact-field pipeline, its masked/unmasked semantics, failure and
leakage boundaries, deterministic tests, design documentation, and draft protected experiment.

#### Reviewed state

Base commit `36630c7fef14c0907134d2f3c532be3da4a0c43e`; implementation and protocol work is uncommitted
alongside the already accepted RTGS-011 documentation/ARA changes and unrelated owner `.idea/`
changes. The official experiment has no run root and no outcome access.

#### Changes

- Added explicit support policies without coupling geometric mass to render opacity.
- Composed finite-gated, capacity-aware per-view transport around a cloned exact fiber with
  fail-or-rollback semantics and source-view exclusion.
- Added perspective-cubature split selection, progressive view/gain/refit scheduling with full-view
  cleanup, elapsed timing, and leak-resistant independent-half stability orchestration.
- Added public/CLI serialization seams, deterministic CPU tests, the detailed pipeline contract,
  architecture/index updates, and a draft task/data seal/489-cell plan compiler.

#### Evidence

- Focused field/pipeline/protocol suite: 69 tests passed before the final scheduling audit; the
  final focused scheduling/lifter suite passed 22 tests.
- `./scripts/verify.sh`: passed Ruff, format, all non-slow CPU tests, docs sync, ARA, script layout,
  agent workflow, and experiment-contract validation.
- `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q`: complete suite passed, including slow
  tests; only two pre-existing warnings remained.
- Experiment task/data validation passed and the outcome-free plan compiler reports 489 unique
  cells. A calibrated scratch invocation completed and was discarded without retaining metrics.

#### Assumptions

- A finite dense candidate mask is sufficient for this first transport integration; it is not
  described as a sparse-compute OT implementation or true multi-marginal solver.
- Floating alpha is a geometric source-support probability. It is not render opacity,
  correspondence confidence, or an independently calibrated silhouette model.
- Alternating original-order camera halves are a stability perturbation, not an accuracy target.

#### Uncertainties

- None of the added mechanisms has isolated outcome evidence for better quality, convergence, or
  runtime. Native 640-component teachers make exact mixture/topology work visibly expensive, but
  the scratch interaction retained no timing and supports no performance statement.
- The protected driver compiles and audits the exact plan but intentionally has no result workers
  or canonical report-v2 aggregator yet. The task stays draft and cannot be initialized.
- No CUDA/GPU behavior, cross-scene generalization, globally coupled transport, or RGB-backed
  comparison was tested.

#### Review Focus

Challenge mask/opacity/capacity separation, source and split invariants, inactive-view target
access, half-fit geometry leakage, default-path preservation, experiment chronology, and any prose
that could be mistaken for an outcome claim.

#### Protected actions not taken

No official experiment initialization or run, result bundle, viewer smoke, benchmark-table update,
ARA claim, default change, maturity claim beyond code integration, commit, push, or external
publication was performed. Unrelated owner IDE changes were preserved.

#### Recommended Next Action

Have a distinct outcome-unseen reviewer audit the protocol and exact plan. After acceptance,
implement and review the task-bound result workers and report-v2 aggregation before changing the
task to ready or calling `init-run`; do not enable mechanisms together until each isolated gate
survives.

### Review

#### Verdict

Accepted with follow-up

#### Self-reviewed

Yes

#### Correctness

The new behavior is opt-in, CPU-first, source-projection preserving, and bounded by explicit mask,
transport, topology, scheduling, and half-fit contracts. The review found and repaired inactive
training-view gain queries in the progressive path and removed unproven depth priors from half
fits. Incumbent controls retain their prior algorithmic route and existing regression tests pass.

#### Evidence Quality

Unit/integration tests and one discarded calibrated interaction establish shape, rollback,
isolation, serialization, and operability only. They do not establish a quality, speed,
convergence, robustness, novelty, or production claim. The task/data contract is valid, but no
protected outcome has been opened and no result has entered `docs/EXPERIMENTS.md` or ARA.

#### Simplicity

The implementation reuses `FieldLifter`, exact fibers, analytic field loss, visibility,
observability, topology transactions, and `SceneFits`; the only new module orchestrates optional
half fits. All mechanisms have native off-switches and explicit diagnostics instead of a second
renderer or optimizer stack.

#### Missing Cases

The outcome workers/report bundle, re-componentized identity labels, calibrated mask perturbations,
multi-seed measured repeats, browser/results-page smoke, GPU performance, and cross-scene evidence
remain absent. The dense projection gate bounds support but does not reduce pairwise cost storage or
compute like a sparse OT implementation.

#### Required Changes

None for the requested pipeline implementation and draft experiment creation. Before any official
run, complete the explicit producer blocker and obtain a distinct prospective review of the exact
digest without outcome access.

#### Optional Improvements

After isolated gates pass, consider a genuinely sparse candidate representation and cached/vectorized
mixture kernels; benchmark them separately rather than inferring speed from active-view counts.

### Handoff — all-dataset protocol freeze

#### Objective

Prospectively review the exact, outcome-blind protocol and implementation binding for the widened
11-dataset development experiment before any run initialization or calibrated outcome access.

#### Reviewed state

The source, task, exact dataset selections, data-byte seal, splits, three seeds, 549-cell expanded
plan, 512-component calibrated proxy, native/all-candidate arms, resource protocol, metrics,
report-v2 child-page schema, and one canonical run command are frozen. No canonical run root or
result artifact exists.

#### Changes

- Widened the calibrated matrix from two compact sets to all eleven `gaussians2d*` manifests found
  under `dataset/`, including the two unmasked karate fields.
- Added a deterministic 8x8-stratified then mass-area component selector, inactive by default, and
  froze a 512-component-per-view cap with original/used counts and selection digests.
- Added fresh-process guarded workers for 324 shape, 60 association, 81 mask, 6 topology, 6
  scheduling, 6 half-stability, and 66 calibrated cells, plus atomic aggregation.
- Extended report schema v2 to emit one generated `datasets/<id>/index.html` per set containing
  every final metric across seeds, optimizer/stage curves, artifacts, and its orbit-viewer command.

#### Evidence

- Outcome-free plan inspection reports 549 unique cells and exactly six calibrated cells for each
  of eleven datasets.
- One non-frozen seed probe from every synthetic mechanism family completed; no official seed or
  calibrated result was accessed.
- Focused experiment/contract/lifter/refit tests pass (46 tests).
- The compact-byte seal was deliberately refreshed and `validate-data` passes.
- `./scripts/verify.sh` passes all lint, format, non-slow CPU, docs, ARA, layout, workflow, and
  experiment-contract gates; the complete CPU suite including slow tests also passes with only
  the two documented pre-existing PyTorch warnings.

#### Assumptions

- The 512-component selector is a bounded proxy for CPU feasibility, not an estimate of
  complete-field fidelity; it may omit low-mass detail.
- Both calibrated arms execute descriptively even when an isolated synthetic mechanism fails;
  failed mechanisms remain ineligible for interpretation.
- Embedded packed alpha is available support evidence only where present; source RGB and external
  mask files remain forbidden.

#### Uncertainties

- The widened production matrix has not been executed, so runtime, quality, convergence, viewer
  behavior, and mechanism decisions remain unknown.
- The dense 512-component association path may still be expensive or fail on a specific field;
  the run is fail-closed and any protocol-bearing repair requires a new review/run root lifecycle.

#### Review Focus

Challenge exact dataset completeness, byte sealing, cap determinism and diagnostics, split and
input-boundary enforcement, arm isolation, decision gates, metric aggregation, worker freshness,
atomic publication, report schema, claim boundary, and whether the resource budget is executable.

#### Protected actions not taken

No canonical run initialization/execution, official calibrated fit, result/audit artifact, report
render, viewer launch, quantitative interpretation, claim, default change, commit, or publication
was performed.

#### Recommended Next Action

Recompute the prospective digest independently. Approve only if the exact frozen protocol is fit
to execute with Outcome Access `none`; otherwise reject with concrete required changes.

### Review

#### Verdict

Revision required

#### Self-reviewed

No

#### Correctness

The outcome-blind reviewer confirmed the eleven-set inventory, exact splits, data-byte seal,
deterministic component cap, no-image boundary, 549-cell plan, child-report design, and bounded CPU
surface. The reviewed producer is not yet fit to execute because several frozen decision and
failure semantics are not implemented exactly.

#### Evidence Quality

The digest, data seal, task validation, plan counts, dataset coverage, and 44 focused tests were
independently reproduced without outcome access. The independent full verification run exposed a
task status/turn mismatch. No calibrated worker or canonical run was accessed.

#### Simplicity

The existing pipeline remains a viable base, but the producer needs fewer implicit promises:
source bytes must be explicitly bound, treatment configurations must come from the task, and
aggregation must stage before publication.

#### Missing Cases

Fresh-process schedule timing, matched-coverage association decisions, actual shuffled candidate
corruption, dustbin/convergence residuals, a field-level mask Pareto test, separate covariance and
split invariants, structured failure receipts, exact resource accounting, and aggregate staging
were absent or mismatched.

#### Required Changes

Resolve all nine findings in
`experiments/reviews/20260805_probabilistic_field_pipeline_mixed_PROTOCOL_REVIEW_V1_REJECTED.md`,
rerun outcome-free verification, compute a new digest, and obtain a second prospective review.

#### Optional Improvements

None before the bounded correctness repairs. Do not initialize the run or simplify away the hard
invariants merely to make the matrix executable.

### Handoff

#### Objective

Independently re-review the repaired v2 protocol/source binding with Outcome Access `none` and
decide whether the exact task is fit to initialize and execute.

#### Reviewed state

The rejected v1 artifact is preserved at
`experiments/reviews/20260805_probabilistic_field_pipeline_mixed_PROTOCOL_REVIEW_V1_REJECTED.md`.
The repaired driver, experiment contract, all `src/rtgs/**/*.py` bytes, task, 309-file data seal,
splits, seeds, 549-cell plan, metrics, decisions, failure semantics, resource scope, and report
schema are frozen under a new digest. The canonical run root remains absent.

#### Changes

- Bound 102 result-affecting Python files by a task-carried length-prefixed SHA-256 and changed
  development locking to include untracked-file content hashes.
- Made the five cleanup iterations task-driven and moved each of six timed schedule cells into a
  separate guarded subprocess.
- Implemented common-minimum-coverage association scoring, actual candidate/gate corruption for
  the shuffled negative, fixed-point/dustbin/candidate residuals, and fail-closed transport gates.
- Replaced the support-vector mask toy with the production support/mass/opacity/refit path over
  synthetic Gaussian fields containing train-only nuisance components and a clean held-out field;
  the decision now checks all four registered Pareto coordinates against hard and none.
- Added held-out projected covariance error, separate source mean/covariance gates, production
  split mass/optical-thickness probes, held-out/final-view invariants, and calibrated hard-gate
  enforcement before any result is serialized.
- Froze `failure_policy="raise"` for the protected candidate arm, added JSON failures at cell,
  synthetic, orchestration, and aggregation levels, staged all aggregate output before commit,
  and made the completed run receipt the final publication marker.
- Extended calibrated resource receipts through directory publication with input/output bytes,
  stage timings, CPU/CUDA fields, and per-dataset/arm repeat min/median/max summaries.

#### Evidence

- All 48 focused experiment-contract/pipeline/refit tests pass.
- A complete 483-cell rehearsal under three non-registered seeds finished in 136.56 seconds and
  passed every hard invariant; no official seed or calibrated outcome was accessed or retained.
- The exact task reports 549 unique cells, six calibrated cells for each of eleven datasets, and
  five cleanup iterations for all schedule cells; task/data validation passes.
- `./scripts/verify.sh` passes Ruff, format, all non-slow CPU tests, docs sync, ARA, layout,
  workflow, and experiment contracts. The complete CPU suite including slow tests passes with
  only the two documented pre-existing PyTorch warnings.

#### Assumptions

- A staged multi-file commit with a final commit receipt/run-receipt marker is the repository's
  practical aggregate transaction boundary; OS/power failure during individual renames remains a
  detectable failed run rather than a resumable run.
- Fixed-point residual plus finite, non-negative real/dustbin masses and exact candidate-zero
  accounting is the applicable invariant for finite-penalty unbalanced transport; exact balanced
  target marginals or hard capacity bounds are not claimed.
- The field-level mask mechanism test is synthetic and known-truth; calibrated embedded alpha is
  not perturbed or treated as ground-truth mask calibration.

#### Uncertainties

- Calibrated runtime and operability remain unknown until approved execution. Dense association
  is still quadratic inside the 512-component cap.
- Utility rules may fail; that is an allowed result. Only hard invariants stop calibrated
  execution, and combined-arm measurements remain descriptive.

#### Review Focus

Recheck all nine v1 findings against the exact source, especially source-byte circularity,
fresh-process schedule proof, matched-coverage semantics, actual field mask construction, hard
gate coverage, calibrated raise behavior, failure publication, aggregate transaction ordering,
and resource boundary arithmetic.

#### Protected actions not taken

No canonical initialization/run, official-seed synthetic matrix, calibrated fit, result artifact,
report render, viewer, audit, claim, default change, commit, push, or publication was performed.

#### Recommended Next Action

Approve only if every v1 blocker is resolved without weakening the frozen question. Otherwise
reject the v2 digest with exact remaining blockers; do not access protected outcomes.

### Review

#### Verdict

Rejected

#### Self-reviewed

No

#### Correctness

The outcome-blind v2 reviewer independently matched the protocol digest, source-tree binding,
data seal, 549-cell plan, eleven datasets, and all prior repair claims, but rejected execution for
three remaining hard-gate defects.

#### Evidence Quality

All 48 focused tests and the workflow check passed. The reviewer did not initialize or inspect the
canonical run and recorded Outcome Access `none`.

#### Simplicity

The remaining repairs are local: enforce association gates inside each cell, derive held-out
access from realized indices, and remove the unsupported balanced-capacity claim from finite-
penalty UOT.

#### Missing Cases

Association failures were deferred until aggregate decision time; synthetic held-out access was
literal or defaulted; and the dustbin balance metric compared two summation orders of the same
matrix, making it tautological.

#### Required Changes

Resolve the three findings preserved in
`experiments/reviews/20260805_probabilistic_field_pipeline_mixed_PROTOCOL_REVIEW_V2_REJECTED.md`,
refresh the exact source/protocol digests, rerun outcome-free verification, and obtain a new
prospective review.

#### Optional Improvements

None before protocol approval. Do not initialize or execute the official run.

### Handoff

#### Objective

Independently review the repaired v3 protocol/source binding with Outcome Access `none` and decide
whether the exact 549-cell, eleven-dataset task is fit to initialize and execute.

#### Reviewed state

The v1 and v2 rejections are preserved as append-only artifacts. The canonical review path is
vacant, the canonical run root remains absent, and the v3 task digest is
`5ea3680db4b33c36d2fbdf334a0d114f0ca5230d52c6321f5b450d8790f3d0e1`; its exact source-tree
binding is `0d574a1989cd5fa428d74a18b7f55491b0605fb084524b9d6e91c9fe91092e6e`.

#### Changes

- Association cells now receive the frozen task and enforce finite/non-negative real and dustbin
  fields, minimum real mass, fixed-point convergence, and exact candidate gating after each view,
  before another view or cell can execute.
- Synthetic mask, topology, schedule, independent-half, and calibrated records now derive
  `heldout_fit_access_count` from realized optimized indices; they also require optimized indices
  to remain within train and the result's reporting split to equal the frozen held-out split.
- Removed the tautological dustbin balance residual and its unsupported hard-capacity wording.
  Finite-penalty UOT is now described and gated only as capacity-weighted transport with finite,
  non-negative dustbin fields, fixed-point convergence, positive real mass, and candidate zeros.
- Added focused regression checks for measured held-out isolation, fail-fast association source,
  and absence of the unsupported balance gate.

#### Evidence

- 50 focused protocol/pipeline/contract/refit tests pass on CPU.
- `./scripts/verify.sh` passes Ruff, formatting, the complete non-slow CPU suite, docs sync, ARA,
  layout, workflow, and experiment contracts; only the two documented pre-existing PyTorch
  warnings remain.
- Task and data validation pass; the source binding recomputes exactly; plan inspection returns
  549 unique cells with the frozen 324/60/81/6/6/6/66 stage counts.
- No official seed, calibrated worker, run root, result, report, or viewer was accessed.

#### Assumptions

- For finite-penalty UOT, exact balanced marginals and hard capacity bounds are not valid claims.
  The supported validity surface is finite/non-negative realized real and dustbin fields, fixed-
  point convergence, positive real mass, and exact candidate exclusion.
- A staged multi-file aggregate with final commit and run receipts remains the practical,
  detectable transaction boundary already reviewed in v2.

#### Uncertainties

- Calibrated operability, runtime, and utility remain unknown until an approved execution.
- Dense association remains quadratic inside the deterministic 512-component-per-view proxy cap;
  this run cannot establish complete-field, GPU, or real-time performance.

#### Review Focus

Recheck the three v2 blockers specifically: enforcement ordering within association cells, actual
held-out index measurement without a zero default, and removal—not relabeling—of the
mathematically unsupported capacity-balance gate. Also confirm the exact source/task/data binding
and that no outcome access occurred.

#### Protected actions not taken

No canonical initialization or execution, official-seed synthetic cell, calibrated fit, result
artifact, audit, report render, viewer launch, claim, default change, commit, push, or publication
was performed.

#### Recommended Next Action

Approve only if the repaired source implements every frozen hard gate fail-closed and the renamed
finite-penalty UOT boundary is honest. Otherwise reject with a concrete blocker and preserve
Outcome Access `none`.

### Review

#### Verdict

Rejected

#### Self-reviewed

No

#### Correctness

The outcome-blind v3 reviewer confirmed the exact source/task/data bindings and all three v2
repairs, but rejected execution because two additional realized-fit paths did not yet carry the
same hard evidence as their synthetic counterparts.

#### Evidence Quality

The reviewer independently reproduced task/data validation, the 549-cell plan, source digest, and
all 50 focused tests. Outcome Access remained `none`; no run root, worker, result, report, or
viewer was created.

#### Simplicity

Both findings have production-seam repairs: validate every independent-half result against its
own partition, and retain the exact candidate mask with each final correspondence plan so
calibrated violations can be recomputed.

#### Missing Cases

The two half reconstructions were discarded by the cell-level access receipt, and calibrated
association plans did not retain a candidate mask or emit/enforce disallowed real mass.

#### Required Changes

Resolve both findings preserved in
`experiments/reviews/20260805_probabilistic_field_pipeline_mixed_PROTOCOL_REVIEW_V3_REJECTED.md`,
refresh the exact source/protocol digests, rerun outcome-free verification, and obtain another
prospective review.

#### Optional Improvements

None before approval. Do not initialize or execute the official run.

### Handoff — all-failure aggregation repair and third refreshed freeze

#### Objective

Independently review the completion protocol after repairing every zero-success reporting path,
with Outcome Access still `none`.

#### Reviewed state

The receipt contract survived the prior reviewer's valid fixture plus 73 one-at-a-time adversarial
mutations. That reviewer then found all-failure aggregation blockers and stopped the superseded
`696b9ce987a03e96d399f5cf962f8a50048e0b7ba4fb4f70716203cd2bddd550` /
`137f99f870a3e79c4d9984c0f6912b0037fded92b61ae630d10780b14ba52097` review before
approval, artifact creation, initialization, or outcome access. The review path and run root
remain absent. The third refreshed protocol digest is
`0dbadcc19678c2310fc7889d8973b7af0acc56c8e16a11c9f904b9d3dd826980`; its source binding is
`8023a293045cfde0ee02dcc06216339820fdfbea976534637c3385c0c8bea391`.

#### Changes

- Added a frozen failure-reporting contract (SHA-256
  `9bdc0de5ec3b8f52f9df2c4c37b8292c6e436f1e98463d8f5829cfe0a6aaa989`) and a frozen
  conditional-metric definition (SHA-256
  `296a2b248175612c26044f1f3d63155666d8445e881914054594ddf9055e2a23`) to each calibrated
  plan cell.
- Conditional calibrated medians now use a total helper: a non-empty successful set yields its
  median and denominator, while an empty set yields JSON null and denominator zero in
  `conditional_metric_results`; no empty `_median` call remains.
- Conditional metrics with zero denominators are omitted from the canonical finite metric table;
  their explicit successful-cell counts remain numeric. Always-realized synthetic metrics and the
  calibrated success fraction remain frozen primary metrics.
- A dataset with six hard-gate failures still emits both arm/seed status curves, exact success and
  attempt counts, schema-valid required cards labeled unavailable and plotting only zero
  successful-cell counts, failure links, and rejected orbit models. It emits no quality/resource/
  runtime surrogate.
- If all 66 calibrated attempts fail, root cards use the same labeled successful-count fallback,
  while training history contains only calibrated-cell-success zeros with complete stage-marker
  structure. No failed-cell time or quality value enters history.
- Embedded-alpha availability on a failure-only dataset is read from its sealed compact manifest
  rather than inferred from nonexistent successful metrics.
- The shared experiment-contract source was deliberately left unchanged; all three immutable
  predecessor task/source bindings still recompute exactly.

#### Evidence

- A new all-failure regression builds six failed cells for one dataset and proves all required
  charts are non-empty schema-valid availability fallbacks, status curves are all zero, explicit
  counts are 0/3 per arm, and completed history contains only zero success indicators.
- Empty/non-empty conditional aggregation fixtures return `{value: null, successful_cell_count:
  0}` and `{value: 2.0, successful_cell_count: 2}` respectively.
- The 78-test focused protocol/correspondence/pipeline/contract/refit suite passes.
- The 549-cell plan has unchanged stage counts and canonical SHA-256
  `acc3ed8d2de7a271b2f95f7039d08b5b5d931d71ecf4d75a2e08bd29bae62a6b`; the 483 synthetic
  cells, datasets, seeds, splits, pipeline, invariant gates, and decisions remain unchanged.
- Task/data validation and full `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh` pass with only the
  two documented pre-existing PyTorch warnings.

#### Assumptions

- Plotting a clearly labeled successful-cell count in a required card when its scientific metric
  is unavailable is availability evidence, not metric imputation; the card title and labels say
  the metric is unavailable.
- Nullable conditional results belong in the producer/audit machine record, while canonical report
  metrics remain finite under the shared immutable report contract.

#### Uncertainties

- Completion outcomes, the actual failure distribution, and rendered/browser behavior remain
  unknown.
- A dataset still needs at least one accepted or preserved rejected model to satisfy the requested
  orbit viewer; otherwise aggregation aborts.

#### Review Focus

Re-run the 73 receipt mutations; simulate six failures for one dataset and all 66 globally; verify
no empty median, empty chart, empty history, null canonical metric, or fabricated failure metric;
then recheck immutable predecessor bindings and unchanged scientific cells/gates.

#### Protected actions not taken

No completion root, initialization, official worker, outcome, aggregate, audit, report, viewer,
claim, default change, commit, push, or publication was created or accessed.

#### Recommended Next Action

Approve only if valid zero-success terminals produce a complete, honest, schema-valid bundle and
every ineligible failure still aborts before aggregation.

### Handoff — completion protocol correction and refreshed freeze

#### Objective

Independently review the refreshed failure-tolerant completion protocol with Outcome Access
`none`, after narrowing the only continuable worker failure to the exact scientific hard-gate
class already observed in the immutable predecessor.

#### Reviewed state

The earlier completion bindings `b906d84c0e0f1a2d7ebc3ccb04e36525ae0a57b3c7c1c19c8b50cecc9c698e47`
and `705b1e805b180b2c3bf183396771437748a425829ebbc4ce4d2b4b526656df40` are
superseded before review and before initialization. The completion run root and canonical review
artifact remain absent. The refreshed protocol digest is
`0e6550b0b0d5b156b2c18bdf661980c23174e5b996be14c4e50b6f036fbb3d95`; the exact bound-source
digest is `647b95aaaafe60cf1c56b1c2982c2511362de5c50c1abe2dc6e1100cf4d676eb`.

#### Changes

- Replaced the overbroad draft `continue_structured_field_fit_failure` rule with
  `continue_structured_hard_invariant_failure`.
- A coordinator may continue only after a worker reports phase exactly `field_fit`, exception type
  exactly `RuntimeError`, and a message beginning exactly `hard invariant violation:`, with exact
  task/context, a passing live input guard, and complete failed boundary/resource receipts.
- Every other exception class or message—including `ValueError` from configuration, loading, or
  input-contract defects—aborts the root.
- Retained the unchanged hard gates, including `minimum_transport_real_mass=1e-10` and
  association `failure_policy=raise`; no threshold, seed, arm, split, input cap, component cap,
  mechanism, metric, or decision rule changed.
- Failed eligible cells still contribute only `cell_success=0`; no quality/runtime value is
  imputed, and a preserved rejected model remains presentation-only.

#### Evidence

- Task and sealed-data validation pass; the unchanged plan contains 549 unique cells: 483
  synthetic cells and 66 calibrated attempts. Its canonical cell SHA-256 is
  `99e578d799298945b5015f4d57807a3ef200c7dbaa3cc64223e4f340d0134e96`.
- The 77-test focused protocol/correspondence/pipeline/contract/refit suite passes, including a
  negative regression proving that an otherwise well-formed `ValueError` cannot continue.
- Full `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh` passes lint, formatting, all non-slow CPU
  tests, docs sync, ARA, layout, workflow, and experiment contracts, with only the two documented
  pre-existing PyTorch warnings.
- The prior prospective reviewer stopped before producing an artifact or accessing outcomes when
  the overbroad draft was superseded.

#### Assumptions

- The exact `hard invariant violation:` RuntimeError prefix is private to pre-serialization
  scientific gate enforcement in this frozen driver; configuration and input-contract exceptions
  use different exception classes/messages and remain fatal.
- Independent cells are scientifically separable attempts, while within-cell gate rejection is
  terminal and cannot supply successful metrics.

#### Uncertainties

- All completion-run outcomes, further hard-gate failures, aggregate metrics, report rendering,
  and viewer behavior remain unknown.
- A dataset with neither an accepted nor a preserved rejected model will fail aggregation instead
  of receiving a fabricated viewer.

#### Review Focus

Reproduce the exact protocol/source/data/plan bindings; compare the 483 synthetic cells and all
scientific gates against the approved predecessor; inject wrong-phase, wrong-context, dirty-guard,
missing-receipt, wrong-exception, and wrong-message failures; and trace eligible failures through
status curves, no-imputation medians, report links, viewer rejection labels, and root accounting.

#### Protected actions not taken

No completion root, initialization, official worker, outcome, aggregate, audit, report, viewer,
claim, default change, commit, push, or publication was created or accessed.

#### Recommended Next Action

Approve only if the refreshed seam cannot hide an implementation/input failure or turn a rejected
cell into metric evidence; otherwise reject before initialization.

### Handoff — exact receipt-contract repair and second refreshed freeze

#### Objective

Independently review the completion protocol after closing the prospective reviewer's adversarial
receipt-validation blocker, still with Outcome Access `none`.

#### Reviewed state

The reviewer stopped the superseded
`0e6550b0b0d5b156b2c18bdf661980c23174e5b996be14c4e50b6f036fbb3d95` /
`647b95aaaafe60cf1c56b1c2982c2511362de5c50c1abe2dc6e1100cf4d676eb` review before
approval, artifact creation, initialization, or outcome access. The completion review path and run
root remain absent. The second refreshed protocol digest is
`696b9ce987a03e96d399f5cf962f8a50048e0b7ba4fb4f70716203cd2bddd550`; its exact source binding
is `137f99f870a3e79c4d9984c0f6912b0037fded92b61ae630d10780b14ba52097`.

#### Changes

- Added a machine-bound continued-failure receipt contract with digest
  `65a687550a6ef11029185e5f55d0fd52420255f32c6b70ff97dbc75e22d7b283` to every calibrated
  plan cell.
- Continuation now requires exact JSON key sets and exact task/dataset/seed/arm/warmup context in
  the failure, input-boundary, and resource receipts.
- The guard must equal the frozen clean record: zero real path/import denials, exactly three
  negative-control denials, and no forbidden module. A contradictory extra `violations` field is
  rejected even when `passed=true`.
- The coordinator recomputes and exactly matches path, byte count, and SHA-256 for the manifest and
  every compact view; external-mask and held-out-training access must both be false.
- Both rejected PLYs must serialize without preservation error and match their recorded bytes and
  SHA-256 before a hard-gate failure can continue.
- Failed resource receipts now carry matching context, exact CPU threading, CUDA inventory and
  non-use, input/output bytes, finite non-negative load/fit/scope/process timings, and positive peak
  RSS; missing, extra, inconsistent, or invalid fields abort the root.

#### Evidence

- The reviewer's original forged combination—false failure-context guard, wrong boundary context,
  dirty guard details, mask/held-out access, and an incomplete wrong-context resource receipt—is
  now rejected by a regression while a complete exact fixture is accepted.
- The `ValueError` negative regression remains fail-closed. The 77-test focused suite passes.
- A direct predecessor comparison proves all 483 synthetic cell dictionaries are byte-canonical
  equal (SHA-256 `c04d4c22db682807968e879dafd87a36db21f10493ae86c0963dcf27c225e83c`), and the
  datasets, seeds, splits, pipeline configuration, invariant gates, and decision rules are exact.
- Task/data validation pass. The plan remains 549 unique attempts with unchanged stage counts; its
  new canonical cell SHA-256 is
  `2f609017ec8a97bbd478a68989dede84213a8491e40b6c1cedb47769f4a02dd6`.
- Full `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh` passes with only the two documented pre-existing
  PyTorch warnings.

#### Assumptions

- An exact hard-invariant failure is reached only after a reconstruction exists, so failure to
  preserve either rejected model is a serialization failure and must remain root-fatal.
- Rehashing sealed compact inputs in the coordinator is acceptable overhead for the small number
  of rejected cells and materially strengthens failure provenance.

#### Uncertainties

- Completion outcomes and viewer/report behavior remain entirely unknown.
- The distribution of legitimate hard-gate failures remains unknown.

#### Review Focus

Repeat the forged-receipt attack and independently vary each context, guard, input hash, rejected
artifact, resource field, exception class, phase, and message. Confirm the successful-cell metric
path cannot consume failed artifacts and the 483 synthetic cells/scientific gates remain unchanged.

#### Protected actions not taken

No completion root, initialization, official worker, outcome, aggregate, audit, report, viewer,
claim, default change, commit, push, or publication was created or accessed.

#### Recommended Next Action

Approve only if the exact receipt contract is both emitted and independently fail-closed under the
adversarial cases; otherwise reject before initialization.

### Review Accepted — retry v2 amendment

#### Verdict

Accepted

#### Self-reviewed

No

#### Correctness

The distinct reviewer approved the task-only amendment after independently reconstructing the
superseded retry-v1 digest and confirming that removing `depends_on` is the only protocol-bearing
change. The source binding, sealed inputs, 549-cell plan, treatments, controls, gates, resources,
and exact command are unchanged.

#### Evidence Quality

The reviewer matched protocol digest
`03722e78b6b303bd87b60da1d8ac61b210a5cb4e4ecf52a49431139f64a78418`, source digest
`e1c442a54585aa67fbfa1e57c6bbb73c77eb5a61c95889a38a33f8941383651a`, the 309-file data seal,
and 549-cell plan; 71 focused tests and full verification passed. Outcome Access remained `none`,
and the retry root was absent throughout review.

#### Simplicity

The amendment removes an invalid executable dependency on the intentionally incomplete zero-cell
predecessor while retaining that failure as explicit provenance. It changes no scientific
behavior or bound source byte.

#### Missing Cases

Full-run operability, scientific outcomes, report rendering, and viewer behavior remain unknown
until execution and independent results audit.

#### Required Changes

None before initializing the vacant retry root and executing the exact reviewed command.

#### Optional Improvements

Preserve both the original failed run and superseded retry-v1 review as append-only evidence.

### Handoff — calibrated warmup input-contract failure

#### Objective

Preserve the immutable retry run after its first discarded calibrated warmup failed, identify the
cause without changing or interpreting scientific thresholds, and prepare a new task identity.

#### Reviewed state

`runs/20260805_probabilistic_field_pipeline_retry_mixed` is immutable and failed during the
discarded `stage_00008_default` warmup. Its frozen synthetic matrix completed, but the run receipt
records `measured_cell_count=0`; no calibrated measured cell, aggregate result, report, or viewer
was produced.

#### Changes

- Made no change to the failed run, its locked task, driver, synthetic payload, or receipts.
- Traced the warmup exception to a strict mismatch between every sealed field's
  `aa_dilation=0.0` and the calibrated lifter's default nonzero projection dilation.
- Outcome-blind inspection of all sealed compact metadata also found two declared loader
  contracts: 168000 bytes for bounded compact sets and 8388608 bytes for full-resolution sets;
  the failed worker had used only the smaller default.

#### Evidence

- Root `failure.json` and `run_receipt.json` record a warmup subprocess failure with zero measured
  calibrated cells and preserve the exact `observation and fiber AA dilation must agree` stack.
- All eleven datasets were loaded read-only with their declared cap; all 309 sealed inputs remain
  unchanged, and every one of the 296 observations declares zero AA dilation.

#### Assumptions

- A discarded warmup failure is infrastructure evidence; its presence does not authorize reuse
  or overwrite of the initialized run root.
- The completed synthetic payload remains part of the failed attempt and is not used to tune the
  next retry's hypotheses, gates, metrics, seeds, arms, or thresholds.

#### Uncertainties

- End-to-end calibrated execution remained unknown at this point.

#### Recommended Next Action

Use a new task/run identity that freezes the sealed bundle caps and exact zero-dilation convention,
then obtain a fresh outcome-blind prospective review before execution.

### Handoff — input-contract retry protocol freeze

#### Objective

Independently review the new input-contract retry with Outcome Access `none`, focusing on the two
bounded changes required to make every sealed Gaussian2D set executable.

#### Reviewed state

The new task is
`experiments/tasks/20260805_probabilistic_field_pipeline_input_retry_mixed.json`; its run root and
canonical review artifact are absent. The prior two run roots remain immutable failure evidence.
The exact protocol digest is
`20a71565f43d9b6879ec8382ef00c0cbe27846669814a0f1509ccb13a3ba89d1`; its exact bound-source
digest is `2d997a21d4c923ede25555994bafa9cec01bb7c741edd0e6da93fadf37bcbe71`.

#### Changes

- Added a task-local `compact_view_byte_caps` map that exactly covers all eleven dataset IDs with
  the 168000- or 8388608-byte cap declared by every sealed view in that dataset.
- Froze calibrated `projection_dilation=0.0`, matching every sealed observation's declared AA
  convention, and made the worker fail closed if loaded observations differ.
- Passed the frozen cap to `CompactDataset.load`; recorded cap, observed AA values, and configured
  dilation in each input receipt and calibrated cell plan.
- Kept all 549 cells, eleven datasets, three official seeds, two calibrated arms, 512-component
  cap, metrics, gates, decision thresholds, train/held-out splits, and image-denial guards.
- Added protocol regressions and repaired the draft-refusal test so it constructs an actual draft
  envelope rather than depending on a task whose immutable lifecycle has advanced to ready.

#### Evidence

- The plan has 549 unique cells with unchanged stage counts; its canonical cell SHA-256 is
  `8a3749c00b73f8a791d2cae6e91e37c81c2b75b87b6245cbca82acf339000a00`.
- Task/data validation passes against the unchanged 309-file seal.
- 72 focused CPU tests pass, including exact cap/dilation plan checks.
- One disposable non-measured `stage_00008_default` candidate warmup passed the formerly failing
  seam. Its temporary output was deleted without reading or retaining metrics.
- Full `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh` passes with only the two documented
  pre-existing PyTorch warnings.

#### Assumptions

- Matching the Gaussian field's declared AA convention is a required input contract, not a tuned
  quality parameter; the new fail-closed check prevents silent cross-convention comparison.
- The embedded byte cap is a sealed loader safety contract, not a capacity or performance
  treatment.

#### Uncertainties

- Official synthetic decisions, calibrated measurements, aggregate outcomes, report rendering,
  and browser/viewer behavior for this new retry remain unknown.
- The disposable operability smoke establishes only that the repaired first warmup can complete;
  it supplies no retained numerical evidence.

#### Review Focus

Confirm from sealed metadata that the per-dataset caps and zero AA dilation are exact; diff the
new driver/task against the failed retry; verify no hypothesis, comparator, metric, gate, seed,
split, or component cap drift; reproduce source/protocol/data/plan digests and the 72 focused/full
verification results; keep Outcome Access `none`.

#### Protected actions not taken

No input-retry initialization, official cell, result, report, viewer, claim, default change,
commit, push, or publication was created or accessed.

#### Recommended Next Action

Approve only if both additions are exact sealed-input contracts and the new identity preserves
the scientific matrix; otherwise reject before initialization.

### Review Accepted — input-contract retry

#### Verdict

Accepted

#### Self-reviewed

No

#### Correctness

The distinct reviewer approved the exact input-retry task/source binding after confirming that
all 296 sealed observations declare AA dilation zero, each dataset uniformly declares its frozen
loader cap, both arms receive the same convention, and every input mismatch fails before fitting.

#### Evidence Quality

The reviewer matched protocol digest
`20a71565f43d9b6879ec8382ef00c0cbe27846669814a0f1509ccb13a3ba89d1`, source digest
`2d997a21d4c923ede25555994bafa9cec01bb7c741edd0e6da93fadf37bcbe71`, the unchanged data seal,
and 549-cell hash `8a3749c00b73f8a791d2cae6e91e37c81c2b75b87b6245cbca82acf339000a00`.
All 72 focused tests and full verification passed. Outcome Access remained `none`.

#### Simplicity

Only exact sealed-input cap/dilation contracts and their receipts were added to calibrated cells;
all 483 synthetic cells and all scientific hypotheses, controls, gates, thresholds, seeds, and
splits remain unchanged.

#### Missing Cases

Official execution, calibrated outcomes, report rendering, and viewer/browser behavior remain
unknown until the protected run and independent results audit complete.

#### Required Changes

None before initialization and exact-command execution.

#### Optional Improvements

Keep the two prior failed roots immutable and distinguish the actual AA exception from the latent
full-resolution byte-cap blocker in all later records.

### Handoff — calibrated scientific hard-gate stop

#### Objective

Preserve the input-contract retry after a preregistered calibrated hard gate fired, without
weakening the gate or overwriting any result, and identify the minimum protocol needed to finish
the owner's all-dataset inspection request.

#### Reviewed state

`runs/20260805_probabilistic_field_pipeline_input_retry_mixed` is immutable and failed at the
`stage_00008_native_fullres`, seed `80501`, `all_candidate_mechanisms` worker. Seven measured cells
completed successfully; the eighth attempt published a structured `field_fit` failure. The root
receipt records `measured_cell_count=8` and preserves the exact stop chronology.

#### Changes

- Made no change to the failed run, its locked task/driver, successful cells, failed cell, or
  synthetic payload.
- Read only the terminal root and failed-cell receipts needed to classify the stop. The failed
  cell reports `hard invariant violation: transport real mass`.
- Did not lower `minimum_transport_real_mass=1e-10`, disable independent-half checking, change a
  seed, or reinterpret the failed candidate as successful.

#### Evidence

- The run executed from `2026-08-05T11:45:46.943359+00:00` to
  `2026-08-05T13:59:00.309577+00:00` and stopped with exit code 1.
- Seven cell directories contain successful atomic terminals; the eighth contains a structured
  field-fit failure and no successful summary.

#### Assumptions

- The hard-gate stop is a negative scientific/operability outcome, not an infrastructure defect.
- Finishing all independent cells requires a new task identity and explicit failure accounting;
  it does not justify changing the gate.

#### Uncertainties

- The other 58 calibrated attempts remain unknown.
- The number and distribution of further hard-gate failures are unknown.

#### Recommended Next Action

Freeze a new development-only completion task that attempts all cells, continues only after a
complete guarded `field_fit` failure, plots failures without imputation, and aborts on every other
failure class.

### Handoff — failure-tolerant completion protocol freeze

#### Objective

Independently review the new completion task with no access to its outcomes before initialization.

#### Reviewed state

The task is `experiments/tasks/20260805_probabilistic_field_pipeline_completion_mixed.json`; its
run root and canonical review path are absent. Protocol digest is
`b906d84c0e0f1a2d7ebc3ccb04e36525ae0a57b3c7c1c19c8b50cecc9c698e47`; bound-source digest is
`705b1e805b180b2c3bf183396771437748a425829ebbc4ce4d2b4b526656df40`.

#### Changes

- Kept the exact 549-cell scientific matrix and every existing hard/utility gate, including
  `minimum_transport_real_mass=1e-10` and association `failure_policy=raise`.
- Added `continue_structured_field_fit_failure`: a failed worker may not release the coordinator
  unless task/context match, phase is exactly `field_fit`, the live image/input guard passed, and
  both boundary and resource receipts exist. Every other failure still aborts the root.
- Each failed calibrated seed contributes `cell_success=0` and no imputed quality/runtime value;
  successful medians use only finite successful cells and report their denominators.
- When a completed reconstruction is rejected by a later hard gate, preserve its model only for a
  viewer labeled `rejected by hard gate`; it is never a successful metric cell.
- Made per-dataset reports/viewers dynamic: available accepted or rejected arms are presented,
  every failure receipt is linked, status is plotted across all three seeds, and quality,
  convergence, resource, topology, input, and guardrail curves retain only realized values.
- Fixed presentation loading to reuse each dataset's frozen compact-view byte cap.

#### Evidence

- Task/data validation passes against the unchanged 309-file seal.
- The plan has 549 unique cells and unchanged stage counts; its canonical cell SHA-256 is
  `25a22678a9235dd927fdbb0d239d634521bb8c8048a1637ef5d8a60fa65a9aa3`.
- 77 focused tests pass. New tests cover strict failure eligibility, coordinator continuation,
  missing-metric non-imputation, report-schema-valid success curves, and rejected-viewer fallback.
- Full `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh` passes with only the two documented
  pre-existing PyTorch warnings.

#### Assumptions

- Cross-cell continuation changes evidence collection, not within-cell acceptance: every hard
  gate still terminates its worker and a failed cell never supplies a scientific metric.
- A rejected model is useful visual failure evidence only when unmistakably labeled and excluded
  from success counts and medians.

#### Uncertainties

- Completion-run outcomes, failure distribution, aggregate metrics, presentation rendering, and
  browser/viewer behavior remain unknown.
- At least one accepted or preserved rejected model per dataset is required to launch that
  dataset's requested viewer; otherwise aggregation fails rather than fabricate a model.

#### Review Focus

Diff the completion task/driver against the approved input retry; confirm all scientific gates and
483 synthetic cells are unchanged; prove only safely structured field-fit failures continue;
trace failed-cell receipts through success curves, medians, per-dataset pages, rejected viewer
labels, root counts, and audit inputs; reproduce task/source/data/plan digests and verification.

#### Protected actions not taken

No completion-run initialization, official cell, outcome, aggregate, report, viewer, claim,
default change, commit, push, or publication was created or accessed.

#### Recommended Next Action

Approve only if failure continuation cannot convert a failed cell into a metric or silently mask
an unsafe failure; otherwise reject before initialization.

### Handoff

#### Objective

Independently review the repaired v5 source with Outcome Access `none`, specifically proving that
all hard invariant gates cover every association-bearing primary and independent-half fit.

#### Reviewed state

The four rejected review artifacts are preserved, the canonical review path is vacant, and no run
root exists. The exact v5 protocol digest is
`ae3b9cbb11157480f33c1520244582409c0f09cd7cad7f6637cde161e8811044`; its exact source binding is
`0684bc11ac3d4b5057bf18f81db842440f790ffdc4ce6395c1a44338fa1840b9`.

#### Changes

- Added one fail-closed pipeline-result invariant seam that calls the complete existing invariant
  enforcement separately for the primary result and both half reconstructions.
- Aggregation now reports worst source/covariance/split/fixed-point/candidate values, minimum real
  mass, total plan count, joint finiteness, and `hard_invariant_checked_fit_count` over every fit.
- Calibrated and synthetic independent-half cells use this all-fit seam; ordinary cells record one
  checked fit. Enforcement happens before serialization.
- Extended the injected candidate-violation regression so a bad second-half plan raises, while
  three valid plans report checked-fit count three and zero off-gate mass.

#### Evidence

- The 69-test focused correspondence/protocol/pipeline/contract/refit suite passes on CPU,
  including the second-half candidate injection.
- `./scripts/verify.sh` passes the full non-slow CPU suite and all repository gates with only the
  two documented pre-existing PyTorch warnings.
- Task/data validation, the 309-file seal, exact source/protocol digests, and the 549-cell plan all
  recompute exactly. No run or outcome has been accessed.

#### Assumptions

- Worst-case aggregation is appropriate for hard invariants; no half-fit quality value is used to
  select, tune, or rescue the primary reconstruction.
- The prior bounded CPU proxy, finite-penalty UOT, and development-only claim boundaries remain
  unchanged.

#### Uncertainties

- Official outcomes, runtime, mechanism decisions, calibrated quality, and operability remain
  unknown until approval and execution.
- Browser/viewer behavior remains untested until audited results are published.

#### Review Focus

Inject invalid candidate, real-mass, fixed-point, or finite evidence into either returned half and
confirm the calibrated all-candidate path raises before serialization. Confirm all three checked
fit metrics aggregate correctly, then recheck task/source/data binding and zero outcome access.

#### Protected actions not taken

No canonical initialization/run, official seed, calibrated worker, result/audit, report render,
viewer launch, claim, default change, commit, push, or publication was performed.

#### Recommended Next Action

Approve only if all primary and half-fit hard invariants are now enforced at the same boundary;
otherwise reject with the exact remaining bypass and keep Outcome Access `none`.

### Review

#### Verdict

Accepted

#### Self-reviewed

No

#### Correctness

The distinct v5 reviewer approved the exact protocol and source binding after proving that the
primary and both association-bearing half fits pass the complete hard-invariant set before any
serialization. No concrete protocol blocker remains.

#### Evidence Quality

The reviewer independently matched the 102-file source digest, task digest, 309-file data seal,
and 549-cell plan; injected candidate, minimum-mass, fixed-point, and non-finite failures into
either half; confirmed valid worst-case aggregation and checked-fit count three; ran 69 focused
tests and full verification; and recorded Outcome Access `none`.

#### Simplicity

The final invariant surface uses one shared all-fit enforcement helper and the existing per-result
gates. No parallel exception path or weakened threshold was introduced.

#### Missing Cases

No execution blocker remains. Official runtime, utility decisions, calibrated metrics, rendering,
browser behavior, and report integrity are intentionally still unknown until the approved run and
independent results audit.

#### Required Changes

None before guarded initialization and execution of the exact approved task.

#### Optional Improvements

Any future protocol or bound-source change requires a new task review; do not modify the approved
bytes during this run.

### Handoff

#### Objective

Prospectively review the new task-ID infrastructure retry with Outcome Access `none`, then approve
only if it preserves the v5 scientific protocol and safely fixes the zero-cell import failure.

#### Reviewed state

The approved predecessor initialized once and failed before its first synthetic cell. Its
top-level, orchestration, and synthetic failure receipts are preserved under
`runs/20260805_probabilistic_field_pipeline_mixed/`; they record `measured_cell_count=0` and
`completed_cells=0`. The retry run root and canonical retry review path are absent. The retry
protocol digest is `91e2c662d079742e99b7336d2f5a5de7ed4a55e733653e5537a6090570b9ce0d`; its source binding is
`e1c442a54585aa67fbfa1e57c6bbb73c77eb5a61c95889a38a33f8941383651a`.

#### Changes

- Created new task `20260805_probabilistic_field_pipeline_retry_mixed`, depending on the failed
  predecessor, with the same 309-file seal, datasets, splits, seeds, arms, gates, resources,
  report schema, and 549-cell plan.
- Copied the previously approved task-specific driver under the new task ID. Its only behavioral
  delta is iterating `(fromlist or ())`, matching Python's valid `__import__` call contract; the
  other textual delta is the task ID constant.
- Added a clean-subprocess regression that enters the live image/import guard and imports `torch`
  through `__import__(..., fromlist=None)`, reproducing the exact predecessor failure surface.
- Bound the new task/driver/run paths and exact retry source digest. The failed predecessor task,
  review, source driver, run receipts, and data seal remain untouched.

#### Evidence

- 71 focused protocol/correspondence/pipeline/contract/refit tests pass, including the exact live
  guard subprocess and the unchanged 549-cell retry plan.
- `./scripts/verify.sh` passes Ruff, format, all non-slow CPU tests, docs sync, ARA, script layout,
  workflow, and experiment contracts with only two documented pre-existing warnings.
- Retry task/data validation passes and reuses the byte-identical 309-file, 204,306,829-byte seal.
- The predecessor failure receipts hash to `933f07c5…`, `ace96bd5…`, and `05e409f5…`; no scientific
  or calibrated outcome was produced or accessed.

#### Assumptions

- The predecessor's zero-cell import failure is infrastructure provenance, not scientific outcome
  access; the retry's scientific question and outcome-blind prospective boundary remain intact.
- A new task ID and run root are required because the predecessor was already initialized before
  its task-local source repair.

#### Uncertainties

- The retry has not been initialized, so later runtime, utility, metrics, and viewer behavior are
  unknown.
- The live guard regression proves the exact failing import form, while the complete run remains
  the only operability check for all worker phases.

#### Review Focus

Diff the two task-specific drivers and confirm the only behavioral change is `fromlist=None`
normalization; verify the retry task otherwise preserves the approved v5 protocol, exact data
seal, and source bindings; independently run the live guard regression; inspect predecessor
receipts only for zero-cell failure provenance and do not access any retry outcome.

#### Protected actions not taken

No retry initialization, synthetic/calibrated worker, retry result, report render, viewer launch,
claim, default change, commit, push, or publication was performed.

#### Recommended Next Action

Approve only if the retry is a minimal infrastructure-only repair with no scientific protocol
drift and the guarded `torch` import succeeds; otherwise reject before retry initialization.

### Review

#### Verdict

Accepted

#### Self-reviewed

No

#### Correctness

The distinct reviewer approved the retry after confirming the complete scientific task body and
all 549 cells are identical to the approved predecessor, while the driver delta is limited to the
new task ID and valid `fromlist=None` normalization.

#### Evidence Quality

The reviewer matched exact retry protocol/source/data digests, reconstructed the predecessor's
zero-cell failure only from its four infrastructure receipts, passed the clean-process live guard,
71 focused tests, and full verification, and recorded Retry Outcome Access `none`.

#### Simplicity

The retry uses a new immutable task/run identity and a one-expression task-local fix. It neither
weakens the import guard nor changes any scientific cell, gate, metric, resource rule, or output.

#### Missing Cases

No retry initialization blocker remains. Full-run operability and every scientific outcome remain
unknown until guarded execution.

#### Required Changes

None before recording approval, transitioning the retry task to `ready`, and executing its exact
reviewed command.

#### Optional Improvements

Preserve the predecessor failure receipts and do not reuse or overwrite its run root.

### Handoff

#### Objective

Review the retry's task-only v2 amendment with Outcome Access `none`: remove the invalid completed-
run dependency while retaining explicit zero-cell predecessor provenance.

#### Reviewed state

The first retry approval is preserved as
`experiments/reviews/20260805_probabilistic_field_pipeline_retry_mixed_PROTOCOL_REVIEW_V1_SUPERSEDED.md`.
`init-run` refused before creating a retry root because `depends_on` accepts only complete canonical
runs and the predecessor is correctly failed. The retry source is byte-identical to the approved
binding `e1c442a54585aa67fbfa1e57c6bbb73c77eb5a61c95889a38a33f8941383651a`; the amended task digest
is `03722e78b6b303bd87b60da1d8ac61b210a5cb4e4ecf52a49431139f64a78418`.

#### Changes

- Removed the predecessor from `depends_on`; this is required because that field is an execution
  prerequisite for a completed result bundle, not a provenance link to a failed attempt.
- Kept the predecessor ID, zero-cell chronology, and unchanged-protocol statement in the retry
  claim boundary and durable handoffs. No dataset, split, seed, arm, gate, resource, command,
  source, or 549-cell plan field changed.
- Reset the retry lifecycle to draft/pending and moved the superseded approval out of the
  canonical review path. The retry run root remains absent.

#### Evidence

- Retry task/data validation passes; the 71 focused tests and full `./scripts/verify.sh` pass.
- The exact retry source digest remains unchanged and the data seal remains the same 309 files.
- The refused initialization created no directory, lock, worker, cell, metric, or outcome.

#### Assumptions

- Failed-predecessor provenance belongs in the claim boundary/review record, while `depends_on`
  is reserved for complete result prerequisites under the contract's fail-closed semantics.
- Because no retry root was created, this task-only amendment is permitted under the same retry
  task ID after a refreshed prospective review.

#### Uncertainties

- Retry execution and all outcomes remain unknown.
- No generic machine-readable failed-run provenance field exists in the task v2 schema beyond
  bounded prose and durable review/handoff records.

#### Review Focus

Confirm `init-run` reached dependency validation before root creation, the retry task/source/data
and 549-cell scientific body are otherwise unchanged from the first approval, and removing the
failed predecessor from `depends_on` is the contract-correct minimal amendment.

#### Protected actions not taken

No retry root, lock, coordinator, cell, calibrated worker, result, render, viewer, claim, commit,
push, or publication was created or accessed.

#### Recommended Next Action

Approve the amended digest only if it preserves the scientific protocol and uses `depends_on`
according to its completed-run semantics; then initialize the still-vacant retry root.

### Handoff

#### Objective

Independently review the repaired v4 protocol/source binding with Outcome Access `none`, focusing
on every realized independent-half fit and calibrated candidate-gate enforcement.

#### Reviewed state

All three earlier rejection artifacts are preserved, the canonical review path is vacant, and the
canonical run root remains absent. The exact v4 protocol digest is
`d8b4fc34e42059bcd57c5642ba72b6f2d50862a5cd8a5e76035e5db48d3f7421`; its exact source-tree
binding is `b0208dc7b4364f3f64b0b4ce0bfde64ea8703522f760f8ce08a466ffd878be53`.

#### Changes

- The public probabilistic pipeline now validates the primary reconstruction and each half
  reconstruction against the exact `SceneFits` partition that produced it before stability is
  computed or returned.
- The experiment driver retains both half reconstructions, validates each against the complement
  of its exact half, checks the two halves are disjoint and cover the original train set, sums
  realized held-out access across all executed fits, and emits the number of checked fits.
- `CorrespondencePlan` now retains and validates the exact candidate mask. Row-softmax and UOT
  constructors populate it; detach, scatter, and empty-view paths preserve a full-shape mask.
- Calibrated association invariants recompute maximum real mass outside every retained mask and
  fail before serialization when a mask is missing or the frozen tolerance is exceeded.
- Added regressions that inject leakage into the second half fit, prove all three fit receipts are
  counted, retain masks across solvers, and inject calibrated off-gate mass that must raise.

#### Evidence

- 69 focused correspondence/protocol/pipeline/contract/refit tests pass on CPU.
- `./scripts/verify.sh` passes Ruff, format, the full non-slow CPU suite, docs sync, ARA, layout,
  workflow, and experiment contracts with only two documented pre-existing PyTorch warnings.
- Task/data validation and the 309-file seal pass; plan inspection still returns the exact 549
  cells and frozen stage counts; source and protocol digests recompute exactly.
- No official seed, calibrated worker, run root, result, report, viewer, or outcome was accessed.

#### Assumptions

- A candidate mask is part of final correspondence evidence, not a rendering or confidence field;
  exact zero outside it is checked from realized plan mass.
- Half-fit reporting views intentionally include the other training half plus the original held-
  out views, because each half's complement must remain reporting-only for that fit.

#### Uncertainties

- Official operability, runtime, mechanism decisions, and field metrics remain unknown until an
  approved run.
- The bounded CPU proxy and claim limitations stated in prior handoffs are unchanged.

#### Review Focus

Trace both half-fit partitions from construction through core validation and cell receipts,
including an injected second-half leak. Trace the production candidate mask through make,
scatter, detach, final result, calibrated metric, and pre-serialization gate. Recheck exact
source/task/data binding and zero outcome access.

#### Protected actions not taken

No canonical initialization/execution, official synthetic cell, calibrated fit, result/audit,
report render, viewer launch, claim, default change, commit, push, or publication was performed.

#### Recommended Next Action

Approve only if every realized fit and calibrated candidate plan is now auditable and fail-closed;
otherwise reject with a concrete remaining path while preserving Outcome Access `none`.

### Review

#### Verdict

Rejected

#### Self-reviewed

No

#### Correctness

The outcome-blind v4 reviewer confirmed both v3 repairs for camera isolation and primary calibrated
candidate evidence, but found that association-enabled half fits at the calibrated independent-
half seed bypassed transport/candidate hard gates applied only to the primary reconstruction.

#### Evidence Quality

The reviewer matched exact task/source/data digests, the 549-cell plan, 69 focused tests, and full
repository verification. An injected second-half off-gate mass passed camera receipts but was not
rejected by primary-only invariant enforcement. Outcome Access remained `none`.

#### Simplicity

The repair belongs in one aggregation seam: enforce the same complete invariant set on the primary
and each returned half reconstruction, then aggregate their extrema and checked-fit count.

#### Missing Cases

Calibrated seed `80501` executes two additional association fits; their candidate mask, minimum
real mass, fixed-point, finiteness, source projection, and split invariants were not evaluated by
the cell before stability serialization.

#### Required Changes

Resolve the finding preserved in
`experiments/reviews/20260805_probabilistic_field_pipeline_mixed_PROTOCOL_REVIEW_V4_REJECTED.md`,
refresh exact source/protocol digests, rerun outcome-free verification, and obtain a new
prospective review.

#### Optional Improvements

None before approval. Do not initialize or execute the official run.

### Handoff — active completion freeze pointer

#### Objective

Review the currently active all-failure-safe completion freeze; the full repair record appears in
`Handoff — all-failure aggregation repair and third refreshed freeze` above.

#### Reviewed state

Outcome Access is `none`; the completion review artifact and run root are absent. Active protocol
SHA-256 is `0dbadcc19678c2310fc7889d8973b7af0acc56c8e16a11c9f904b9d3dd826980`, source SHA-256 is
`8023a293045cfde0ee02dcc06216339820fdfbea976534637c3385c0c8bea391`, and 549-cell plan SHA-256
is `acc3ed8d2de7a271b2f95f7039d08b5b5d931d71ecf4d75a2e08bd29bae62a6b`.

#### Evidence

All 78 focused tests and full verification pass; immutable predecessor bindings still match.

#### Protected actions not taken

No completion initialization, worker, outcome, report, audit, or viewer was created or accessed.

#### Recommended Next Action

Perform the distinct prospective review of these exact active bindings before initialization.

### Handoff — canonical report-command correction

#### Objective

Review the all-failure-safe completion protocol after correcting the only argv mismatch found by
the independent 66-failure simulation.

#### Reviewed state

The reviewer proved the superseded all-failure accounting itself valid, then stopped before
approval, artifact creation, initialization, or outcome access when the generated report command
used an absolute Python path. The active protocol SHA-256 is
`87f2c81702a712d29fb19a9032839df5ddff0218c53aad843dcb17e1f4e48444`, source SHA-256 is
`e8cd3f26f82f2bbebba17454717f34e72a1aa2cadfd489510dccc7d35c5d64c8`, and unchanged 549-cell
plan SHA-256 is `acc3ed8d2de7a271b2f95f7039d08b5b5d931d71ecf4d75a2e08bd29bae62a6b`.

#### Changes

- The producer now emits the schema-exact report command beginning with literal
  `.venv/bin/python`; fresh worker subprocesses continue using the actual `sys.executable` and are
  scientifically unchanged.
- Added a regression that passes the exact completed command object through
  `experiment_contract._v2_commands_errors` and requires no errors.

#### Evidence

- The reviewer independently simulated all 66 cells as eligible failures before this argv fix:
  0/66 accounting, all five null conditional results with zero denominators, finite canonical
  metrics, 22-point unavailable cards, and 66 zero-only history records with 924 markers passed;
  the sole check-run error was serve-report argv index zero.
- All 79 focused tests, task/data validation, and full verification pass after the exact fix.

#### Protected actions not taken

No completion review artifact, initialization, worker, outcome, report, audit, or viewer was
created or accessed.

#### Recommended Next Action

Re-run the exact all-66 simulation/check-run and approve only if it is now clean.

### Handoff — exact terminal enumeration and phase provenance

#### Objective

Prospectively review the completion protocol after repairing the root-failure provenance defect
found during the independent all-66-failure simulation.

#### Reviewed state

Outcome Access remains `none`; the completion review artifact and run root are absent. The active
protocol SHA-256 is `1b0382109dfaaba43807daf59afac6e646c5cfb6f062e428be50ae3544b257dd`,
the exact source-tree SHA-256 is
`b6982acb92923ed38f41d0fe05f414cf3bb1d2da816592a8b0f340a647ff26b0`, and the unchanged
549-cell plan SHA-256 is
`acc3ed8d2de7a271b2f95f7039d08b5b5d931d71ecf4d75a2e08bd29bae62a6b`.

#### Changes

- Root failure accounting now enumerates only the 66 exact frozen dataset/seed/arm cell paths and
  counts a cell only when exactly one atomic terminal, `summary.json` or `failure.json`, exists.
  Hidden worker staging directories and invalid dual-terminal cells cannot inflate the count.
- The coordinator carries an explicit `aggregation_started` state set immediately before entering
  aggregate publication. A late orchestration failure is therefore labeled `orchestration` even
  when all 66 cell terminals exist; only a failure after actual aggregate entry is labeled
  `aggregation`.
- No dataset, seed, arm, threshold, mechanism, metric, stopping rule, input boundary, or aggregate
  policy changed.

#### Evidence

- A regression constructs 65 exact terminals plus a hidden worker failure and proves the measured
  count is 65, then 66 after the exact final terminal, and 65 again for an invalid dual terminal.
- Parametrized root-failure regressions construct all 66 exact terminals and prove that both the
  top-level failure and run receipt use the explicit pre/post-aggregation phase while retaining
  measured-cell count 66.
- All 82 focused protocol/correspondence/pipeline/contract/refit tests collect; the directly
  affected protocol and pipeline suites pass. Task/data validation and full
  `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh` pass with only the two documented pre-existing
  PyTorch warnings.

#### Assumptions

- An atomic measured-cell terminal is exactly one of success or eligible structured failure at an
  exact frozen cell path; staging directories and dual terminals are invalid provenance rather
  than completed attempts.

#### Uncertainties

- Completion outcomes, the realized eligible-failure distribution, aggregate rendering, and
  browser behavior remain unknown and unobserved.

#### Review Focus

Independently rerun the all-66 eligible-failure simulation and canonical check-run, including the
new exact-count and phase-provenance counterexamples. Approve only if the active bindings and all
failure-only output remain clean.

#### Protected actions not taken

No completion review artifact, initialization, worker, outcome, report, audit, or viewer was
created or accessed.

#### Recommended Next Action

Record the prospective verdict against these exact bindings; initialize only after independent
approval.

### Review — completion protocol accepted

#### Verdict

Accepted

#### Self-reviewed

No

#### Correctness

The distinct outcome-blind reviewer approved the exact failure-tolerant completion task and source
binding. Exact terminal enumeration, explicit aggregate-entry provenance, strict structured
failure receipts, no-imputation conditional aggregation, rejected-model labeling, and canonical
report commands all passed their direct counterexamples without changing a scientific treatment,
gate, threshold, seed, split, or input boundary.

#### Evidence Quality

Protocol digest `1b0382109dfaaba43807daf59afac6e646c5cfb6f062e428be50ae3544b257dd`,
102-file source digest `b6982acb92923ed38f41d0fe05f414cf3bb1d2da816592a8b0f340a647ff26b0`,
data-seal digest `20e719d89628375c515db94102abf6e5018dbd6d686d0633235407fad5c7deb6`,
and 549-cell plan digest
`acc3ed8d2de7a271b2f95f7039d08b5b5d931d71ecf4d75a2e08bd29bae62a6b` matched. One exact
receipt passed and all 73 one-field tamper cases failed. A 66/66 eligible-failure fixture rendered
and validated with five null/zero-denominator conditional metrics, no imputation, 66 zero-only
history records, and unavailable-count fallbacks. Both actual-orchestrator phase branches passed;
all 82 focused tests and full verification passed. Outcome Access remained `none`.

#### Simplicity

The completion task preserves all 483 synthetic cells and the old calibrated scientific factors;
its 66 calibrated plan changes are limited to five explicit failure-accounting disclosures. The
implementation confines continuation, terminal counting, aggregation fallback, and presentation
behavior to the orchestration/reporting boundary.

#### Missing Cases

Official outcomes, the realized successful/failed-cell distribution, performance and quality
curves, browser-specific rendering, and orbit-viewer behavior remain unknown until the protected
execution and independent results audit complete.

#### Required Changes

None before lifecycle transition, canonical initialization, and exact-command execution. Any
digest-bearing task or bound-source edit requires a new prospective review.

#### Optional Improvements

Keep every predecessor root and superseded review immutable, preserve the four repaired
prospective defects in the durable chronology, and do not interpret or render official outcomes
until the distinct results audit authorizes it.

### Review Accepted — completion execution authorized

The Driver verified the canonical review artifact SHA-256
`bb35f0bd30ca0c9d8bbb6881365d05f4162ad4d5f41c3973b5cefa13fa287f50`, exact approved
protocol digest `1b0382109dfaaba43807daf59afac6e646c5cfb6f062e428be50ae3544b257dd`,
and exact approved source digest
`b6982acb92923ed38f41d0fe05f414cf3bb1d2da816592a8b0f340a647ff26b0`. The task records
reviewer `Codex-probabilistic-field-protocol-reviewer`, verdict `approved`, and status `ready`.
Turn returns to the Driver for canonical initialization and exact-command execution; outcome
interpretation and report rendering remain gated on the independent results audit.

### Handoff — exact empty-support fallback retry freeze

#### Objective

Prospectively review the new task-ID retry that converts only the exact all-placement-sources-
rejected condition into one visibly unmasked whole-cell retry, then approve execution only if it
can complete an orbitable result without hiding mask failure or weakening any scientific gate.

#### Reviewed state

The approved completion run is immutable and failed after 11 measured terminals. Its root
`failure.json` SHA-256 is
`06e07156e4c2a937fa5d1dbd402c1d9b04b625877d1ce23f9188328ebf474104`, root run-receipt
SHA-256 is `506db34780753d91145550713273d3402be023ab02e83f2d04b5bcb4104395b0`, and the terminal
empty-support failure SHA-256 is
`0718cf26f83df954302063afb597f978bdf4606bd50a3a5531853fa9222106ae`. The new task is
`experiments/tasks/20260805_probabilistic_field_pipeline_support_fallback_mixed.json`; its
canonical review artifact and run root are absent. Its active protocol SHA-256 is
`65a773c15ced9c688be41505ca856f5c101110d1e84956f2061e393a0567d25f`, source-tree
SHA-256 is `6a033a294a5c92a4cda656c94bcf68bc0195a68f91b5d8c950c275e9d575861f`, and
549-cell-list SHA-256 is
`4b84ce525d19054ba17422919c1ece652232d69c072428ee40348ed15a7c5724`.

#### Changes

- Preserved the failed completion root and created a new draft task/driver with the same sealed
  eleven datasets, 483 synthetic cells, 66 calibrated cells, seeds, arms, splits, input caps,
  component cap, optimizer, hard invariants, decision rules, and structured hard-gate continuation
  policy.
- Only exact base `ValueError` with message `support-mask policy rejected every field-placement
  source` is caught. Torch is reset to the frozen cell seed and the entire cell is retried once
  with mask mode `none`; a second occurrence, subclass, different message/type, or any retry
  failure remains fatal.
- At the independent-half seed, primary and both half reconstructions are rerun under one effective
  mode. Requested/effective mask modes, RNG reset seed, retry count, checked-fit count, unmasked-fit
  count, trigger, and operability-only interpretation are copied into every fit and serialized
  identically across summary, boundary, resource, and requested/effective configuration records.
- Per-dataset pages add fallback-use/retry/fit-count curves. Root metrics add the descriptive
  successful-fallback-cell fraction. The failed attempt remains included in process and field-fit
  wall time; fallback cells are explicitly ineligible as hard-versus-probability mask evidence.
- The design document now explains the trigger, expected recovery/cost, quality risk, reporting,
  and fail-closed boundary.

#### Evidence

- The predecessor completed the unchanged synthetic matrix, then produced nine success terminals,
  one validated transport-real-mass terminal, and the exact empty-support failure at measured cell
  11/66; the latter had clean sealed-input/guard receipts and no 3D model to preserve.
- The new plan has the unchanged 324/60/81/6/6/6/66 stage counts. Every synthetic cell is
  byte-equivalent at the plan level; calibrated cells differ only by the six explicit fallback
  disclosures.
- Tests prove whole-pipeline retry, hard/probability-to-none transition, deterministic seed reset,
  all-three-fit annotation, exact exception filtering, serialized cross-record consistency,
  tamper rejection, plan preservation, and the prior completion lifecycle transition. All 87
  focused protocol/correspondence/pipeline/contract/refit tests pass.
- Task/data validation and full `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh` pass with only the two
  documented pre-existing PyTorch warnings.

#### Assumptions

- Retrying the whole cell with `mask_mode="none"` is the smallest honest way to obtain a 3D result
  from an internally valid field set whose supplied support rejects every placement source; it is
  an explicit unmasked control path, not recovery of masked semantics.
- Resetting Torch to the frozen cell seed and reusing immutable `SceneFits` makes the retry
  deterministic; measured process/fit wall time still includes the rejected first attempt.

#### Uncertainties

- Whether the unmasked retry succeeds on the failed dataset, how many other cells need it, its
  quality/runtime cost, all remaining calibrated outcomes, aggregate rendering, and browser/viewer
  behavior remain unknown. No new-task outcome has been accessed.

#### Review Focus

Attack exception exactness, retry cardinality, whole-pipeline/half consistency, RNG reset,
requested/effective-mode provenance, serialized tamper resistance, timing inclusion, metric/report
labeling, plan equivalence, and the rule excluding fallback cells from mask-mode interpretation.

#### Protected actions not taken

No canonical support-fallback review, initialization, worker, result, audit, report, or viewer was
created or accessed. No fallback outcome was probed in scratch space. The failed completion run was
not modified.

#### Recommended Next Action

Perform a distinct prospective review of these exact bindings; initialize the replacement run only
after approval.

### Handoff — refreshed fallback freeze after rejected-model provenance repair

This handoff supersedes the immediately preceding fallback freeze and its obsolete protocol,
source, and cell-list digests. No outcome was accessed while repairing the protocol.

#### Objective

Prospectively review the task-scoped exact-empty-support retry, including the strengthened failure
path, and approve execution only if neither accepted nor rejected unmasked output can be mistaken
for masked evidence.

#### Exact bindings

- Task: `experiments/tasks/20260805_probabilistic_field_pipeline_support_fallback_mixed.json`
- Driver: `scripts/experiments/20260805_probabilistic_field_pipeline_support_fallback_mixed.py`
- Protocol SHA-256: `1ed2e94b9544943c03da736becea8b1750eee4a55bfda5208b8f211b68e2244c`
- Source-tree SHA-256: `4cf70d7d053cada2d64b6c17474765811287fbf533417f95b361f1b5f77bae18`
- Canonical 549-cell-list SHA-256:
  `1af09dabc5de411ff09acdef30efa5da17e010f6fcad2b450b0dc08d31f005fc`
- Canonical plan-payload SHA-256:
  `9d8aebe18312fecb9f4141b6ecbcebf4abb1a13b38b86595dad3299f04e8ad78`
- Stage counts: 324 exact shape, 60 re-componentized association, 81 support-mask, 6
  topology, 6 schedule, 6 independent-half, and 66 calibrated operability cells.

#### Repair since the superseded freeze

- A fallback model that later raises an eligible hard-invariant `RuntimeError` now writes the same
  exact requested/effective-mode, retry, RNG-reset, trigger, checked-fit, fallback-fit, and
  interpretation record into `failure.json`, `input_boundary_receipt.json`,
  `resource_receipt.json`, and `gaussians.config.json`.
- Continuation recomputes the only valid record from task/seed/arm, checks exact JSON key sets,
  requires all four records to agree, and independently reconstructs the complete requested and
  effective `FieldLiftConfig` payload. Missing or mutated provenance is fatal.
- Accepted and preserved rejected viewer models now disclose effective unmasked fallback status in
  the comparison-manifest label, rendered panels/orbits, presentation receipt, and root
  representative config. Rejected fallback PLYs remain presentation-only failure evidence.
- The task contract and design document explicitly state these semantics. No library default,
  scientific treatment, seed, split, optimizer, threshold, invariant, or decision rule changed.

#### Verification

- A worker-level regression simulates an unmasked fallback followed by a hard-gate rejection,
  validates the emitted terminal, and proves that tampering with the failure, boundary, resource,
  fallback-config, or effective-config surface prevents continuation. It also proves that a
  consistently disclosed non-fallback hard-gate rejection remains eligible.
- All 88 focused protocol, probabilistic-pipeline, correspondence, experiment-contract, and refit
  tests pass.
- `experiment_contract.py validate`, sealed-data validation, and the full CPU-only
  `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh` pass. The full suite reports only the two known
  PyTorch warnings.
- The canonical review artifact and run directory remain absent. The predecessor completion run
  and all of its immutable terminals remain untouched.

#### Review focus

Attack exact-exception matching, one-retry cardinality, whole-cell/half-mode consistency, timing
inclusion, exact four-surface fallback provenance on both accepted and rejected outputs,
viewer/report labeling, conditional no-imputation behavior, and preservation of all scientific
cells and gates.

#### Recommended next action

Create the canonical prospective review for these exact bindings with `Outcome Access: none`.
Initialize and run only after an approved verdict and verified review-artifact digest.

### Handoff — refreshed fallback freeze after dependency-lifecycle repair

This handoff supersedes both earlier fallback freezes. The reviewer correctly found that the
failed completion run cannot appear in the executable `depends_on` list because initialization
requires each listed root to be canonically complete. The task now uses `depends_on: []`, matching
the repository's prior failed-run retry pattern, while retaining the predecessor's exact failure
hashes and chronology in the claim boundary and handoffs. No run was initialized and no outcome
was accessed.

#### Exact bindings

- Protocol SHA-256: `fb07bda9117ed836d42536e4f08470e95d2b3121a0fc15a071bdfa85b9f6681c`
- Source-tree SHA-256: `4cf70d7d053cada2d64b6c17474765811287fbf533417f95b361f1b5f77bae18`
- Canonical 549-cell-list SHA-256:
  `1af09dabc5de411ff09acdef30efa5da17e010f6fcad2b450b0dc08d31f005fc`
- Canonical plan-payload SHA-256:
  `32dcf34dbeb1dca41e670b2f3663eb76be89bc55a7a530c9b23894e22e1f6835`

#### Delta and evidence

- The sole task delta from the preceding freeze is removal of the invalid failed-run dependency.
  The driver/source digest and all 549 cell bytes remain unchanged.
- The plan-preservation regression now explicitly requires `depends_on == []`.
- All six support-fallback tests, contract validation, sealed-data validation, and the full
  CPU-only `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh` pass; only the two known PyTorch warnings
  remain.
- Canonical review and run roots remain absent. Outcome Access remains `none`.

#### Recommended next action

Review the refreshed protocol digest and confirm initialization is now reachable without weakening
the immutable failed-predecessor chronology. Create the canonical prospective artifact only on an
approved verdict.

### Handoff — refreshed fallback freeze after success-diagnostic tamper repair

This handoff supersedes every earlier fallback freeze. The prospective reviewer demonstrated that
three fit-level fallback diagnostics in an otherwise valid successful summary could be mutated
without rejection. The aggregate validator now checks all serialized fit diagnostics exactly
against the canonical fallback record: requested mode, effective mode, effective `mask_mode`, used
flag, retry count, trigger message, and operability-only interpretation.

#### Exact bindings

- Protocol SHA-256: `e67abc02e7aebb7e2b39aebb65e8ce6db16e65e4f33bfcdaa286c4c125eb961f`
- Source-tree SHA-256: `dc275444b7198201abd525f441c5e572cdbc98feacb80333c4e0de8ffc7c037e`
- Canonical 549-cell-list SHA-256:
  `1af09dabc5de411ff09acdef30efa5da17e010f6fcad2b450b0dc08d31f005fc`
- Canonical plan-payload SHA-256:
  `4614f5ae82579c1557f14c440de962283582e2ea323a57efd25d4db640521229`

#### Delta and evidence

- The driver adds three exact comparisons in `_validate_serialized_support_fallback`; no fit,
  treatment, gate, data, schedule, metric, seed, split, or report computation changed.
- The successful-cell provenance regression independently mutates retry count, trigger, and
  interpretation and requires each mutation to fail aggregation. Existing cross-record and config
  tamper cases remain passing.
- The invalid failed-run dependency remains removed (`depends_on: []`), with predecessor chronology
  preserved narratively and no change to the 549-cell list.
- All six support-fallback tests, all 88 focused tests, contract/data validation, and full CPU-only
  verification pass with only the two known PyTorch warnings.
- Canonical review and run roots remain absent; Outcome Access remains `none`.

#### Recommended next action

Resume the full prospective review on these exact bytes, rerun the independent successful-cell
tamper attacks, and create the canonical review artifact only if no blocker remains.

### Handoff — refreshed fallback freeze after mixed-outcome presentation repair

This handoff supersedes every earlier fallback freeze. The prospective reviewer demonstrated that
an accepted representative seed could hide a preserved rejected fallback from the orbit manifest
and leave its report line without explicit effective-mode provenance.

#### Exact bindings

- Protocol SHA-256: `746520b2e2e07e1622305bcc759ee754ee43c5b5ff8ff6df8b55d7afcc8833fd`
- Source-tree SHA-256: `0e33d90cb14259bbdd448dd4428b41424a660d62b28f9183e6e7a3366e9247db`
- Canonical 549-cell-list SHA-256:
  `1af09dabc5de411ff09acdef30efa5da17e010f6fcad2b450b0dc08d31f005fc`
- Canonical plan-payload SHA-256:
  `7ac763d34bf1508573cc8a8bab051d160f1651160d7fac8f7817ef8216b74d1b`

#### Delta and evidence

- Each arm still has one preferred representative for root compatibility. In addition, every
  preserved rejected cell is copied under a unique arm/seed-qualified presentation ID and added to
  the orbit comparison manifest, contact sheet, reconstruction animation, novel-orbit/elevation
  animations, presentation receipt, and report artifact list.
- Rejected viewer labels say `presentation-only`, `rejected by hard gate`, the frozen seed, and
  either `effective unmasked fallback` or `no unmasked fallback`. Accepted labels separately state
  whether the requested mask was retained or the effective unmasked fallback was used.
- Every failed-cell report note serializes requested/effective modes, used flag, retry count,
  checked/fallback fit counts, RNG reset seed, trigger, interpretation, hard-gate message, and
  presentation-only status. Dataset prose separately counts successful and rejected fallback
  cells.
- A mixed-outcome regression uses one accepted representative, one rejected unmasked-fallback
  seed, and one rejected mask-retained seed in the same arm. It requires all three viewer entries,
  exact fallback distinctions, both presentation-receipt rejection records, explicit report notes,
  and both seed-qualified PLY artifact links.
- The design document states that an accepted seed cannot hide a rejected seed. No scientific cell,
  fit, metric, threshold, gate, seed, split, input, or decision rule changed.
- All seven support-fallback tests, all 89 focused tests, contract/data validation, and full
  CPU-only verification pass with only the two known PyTorch warnings. `depends_on` remains empty.
- Canonical review and run roots remain absent; Outcome Access remains `none`.

#### Recommended next action

Resume the prospective review on these exact bytes and attack mixed-arm viewer membership,
seed-qualified report/viewer provenance, accepted/rejected selection, and fallback/nonfallback
distinctions before issuing an approved canonical artifact.

### Handoff — refreshed fallback freeze after JSON type-strictness repair

This handoff supersedes every earlier fallback freeze. The reviewer demonstrated Python's
`True == 1` behavior against consistently mutated serialized fallback records. The repair makes
the provenance contract JSON-type-strict instead of relying on Python value equality.

#### Exact bindings

- Protocol SHA-256: `f4e5b6bf7cd75d55d14d8cbdc019f2c996e7f6ccad7960f1143d1885b00d16e3`
- Source-tree SHA-256: `13db64f01e211791ff67fcccfcd3f7e923761a5c8cc707763dcf56259a36b663`
- Canonical 549-cell-list SHA-256:
  `1af09dabc5de411ff09acdef30efa5da17e010f6fcad2b450b0dc08d31f005fc`
- Canonical plan-payload SHA-256:
  `5596baf2728b73bf283c4185a3cc00ec4ec2a036dcea37bc05851d149a807d91`

#### Delta and evidence

- One centralized validator requires exact record keys and types: `used` is a JSON boolean;
  retry/checked-fit/fallback-fit counts are JSON integers and never booleans; fallback RNG seed is
  an integer only when used; requested/effective modes and used-trigger fields are strings only in
  their eligible state; all unused RNG/trigger/interpretation fields are exactly null.
- Canonical JSON hashes compare each record to the task-derived expected record, distinguishing
  booleans, integers, floats, strings, and nulls. The requested/effective config receipt is compared
  the same way and independently validates its nested fallback record.
- Successful aggregation applies the validator separately to summary, boundary, resource, and
  config surfaces. It also requires fallback metrics to be JSON floats and fit diagnostics to have
  the exact expected type/value. Hard-gate continuation applies it independently to failure,
  boundary, resource, and config surfaces.
- Regressions coherently mutate all successful surfaces with boolean and floating retry/count/seed
  values, attack float-metric-to-int/bool substitutions, attack bool/float aliases for one-fit
  records, attack null/value transitions for unused fields, and inject a boolean retry count into
  each individual rejected-cell surface. Every attack fails.
- Mixed-outcome report/viewer coverage from the preceding repair remains intact. No scientific
  cell, fit, metric, gate, threshold, seed, split, input, or decision rule changed.
- All seven support-fallback tests, all 89 focused tests, contract/data validation, and full
  CPU-only verification pass with only the two known PyTorch warnings. `depends_on` remains empty.
- Canonical review and run roots remain absent; Outcome Access remains `none`.

#### Recommended next action

Resume prospective review on these exact bytes and independently attack bool/int, int/float,
null/value, coherent cross-surface, and individual failure-surface mutations before completing the
remaining review.

### Review Accepted — support-fallback execution authorized

The Driver read the canonical review in full and verified artifact SHA-256
`e3bb91eb6a257bf4d3f97dac385c672eaa765f206682d75c35305a0c66120d99`, exact approved
protocol digest `f4e5b6bf7cd75d55d14d8cbdc019f2c996e7f6ccad7960f1143d1885b00d16e3`,
and exact approved source digest
`13db64f01e211791ff67fcccfcd3f7e923761a5c8cc707763dcf56259a36b663`. The task records
reviewer `Codex-probabilistic-field-protocol-reviewer`, verdict `approved`, and status `ready`;
the experiment index is synchronized. Turn returns to the Driver for canonical development
initialization and exact-command producer execution. Outcome interpretation and report rendering
remain gated on the independent results audit.

### Review — support-fallback protocol accepted

#### Verdict

Accepted

#### Self-reviewed

No

#### Correctness

The distinct outcome-blind reviewer approved the exact support-fallback task and source binding.
Only the exact base empty-placement-source `ValueError` can trigger one whole-cell unmasked retry;
Torch is reset to the frozen cell seed; primary and independent-half fits share one effective
mode; every scientific hard gate remains active; and any different, repeated, or failed retry
aborts. Successful and rejected terminals carry the same type-strict requested/effective support
record across their required provenance surfaces.

#### Evidence Quality

Protocol digest `f4e5b6bf7cd75d55d14d8cbdc019f2c996e7f6ccad7960f1143d1885b00d16e3`,
102-file source digest `13db64f01e211791ff67fcccfcd3f7e923761a5c8cc707763dcf56259a36b663`,
data-seal digest `20e719d89628375c515db94102abf6e5018dbd6d686d0633235407fad5c7deb6`,
549-cell digest `1af09dabc5de411ff09acdef30efa5da17e010f6fcad2b450b0dc08d31f005fc`,
and plan-payload digest `5596baf2728b73bf283c4185a3cc00ec4ec2a036dcea37bc05851d149a807d91`
matched independently. Exact exception/RNG/whole-half counterexamples passed; bool/int,
int/float, null/value, coherent cross-surface, and individual failure-surface mutations failed;
mixed accepted/rejected viewer and report provenance remained explicit; an all-failure dataset
retained zero-only success history without imputation; all seven fallback tests, all 89 focused
tests, contract/data validation, and full verification passed. Outcome Access remained `none`.

#### Simplicity

All 483 synthetic cells are byte-equivalent to the completion plan. The 66 calibrated cells add
only seven explicit fallback disclosures. Implementation changes are confined to the task-scoped
retry, provenance/receipt validation, operability aggregation, and report/viewer labeling; no
scientific treatment, comparator, gate, threshold, seed, split, input, or decision rule changed.

#### Missing Cases

Official fallback frequency, accepted/rejected distribution, performance and quality curves,
browser-specific report rendering, and live orbit-viewer behavior remain unknown until protected
execution and the independent results audit complete.

#### Required Changes

None before lifecycle transition, canonical initialization, and exact-command execution. Any
digest-bearing task or bound-source edit requires a new prospective review.

#### Optional Improvements

Preserve the failed predecessor and all superseded fallback freezes as immutable chronology. Keep
fallback cells excluded from hard-versus-probability mask interpretation, and do not render or
interpret official outcomes before the distinct results audit authorizes them.

### Handoff — support-fallback run failed closed on a new geometry domain

#### Objective

Preserve the approved support-fallback execution exactly and classify its terminal before making
any successor change.

#### Reviewed state

Canonical run root `runs/20260805_probabilistic_field_pipeline_support_fallback_mixed` was bound to
approved protocol `f4e5b6bf7cd75d55d14d8cbdc019f2c996e7f6ccad7960f1143d1885b00d16e3`
and source digest `13db64f01e211791ff67fcccfcd3f7e923761a5c8cc707763dcf56259a36b663`.

#### Changes

None to that task, driver, run root, or its terminals. The run remains immutable.

#### Evidence

- The driver completed all 483 synthetic cells and the discarded calibrated warmup.
- It preserved sixteen successful calibrated terminals and two eligible structured
  `hard invariant violation: transport real mass` terminals through measured cell 18/66.
- Measured cell 19, `karate_00005_default`, seed 80501, native controls, raised exact
  `ValueError: fixed-anchor field sweep found a source ray with no forward AABB intersection`.
  This was correctly outside the one exact empty-support retry and structured hard-invariant
  continuation policies, so the root aborted.
- Root failure SHA-256:
  `aaebf3f69e5ff43ba1bf48b8e5f9088c26324af96337fef0b6c5d749ccdb7dd6`.
- Run-receipt SHA-256:
  `388913389fb878283786568082b12698d83f7d3535d2fda589d87926ce72c7cd`.
- Terminal failure SHA-256:
  `f3e9d1352a920c8d2994f19fddd63b1012a054d89723626a5f1bdb90cdc73763`.

#### Assumptions

Terminal counts and exact exception provenance are execution-state facts, not outcome-quality or
performance interpretation. No report was rendered and no metric was promoted.

#### Uncertainties

The failed run cannot answer the all-dataset comparison because 47 measured cells were never
attempted and the results audit gate was never reached.

#### Review Focus

Retain the failed root as negative infrastructure evidence. Do not repair or resume it in place.

#### Protected actions not taken

No failed-run artifact was edited, no report or viewer was launched, no result was interpreted,
and no retry sibling was initialized without a new task and review.

#### Recommended Next Action

Review a new task that corrects only the forward-AABB anchor-selection domain and reruns the whole
matrix from a fresh canonical root.

### Handoff — forward-AABB-eligible successor frozen for prospective review

#### Objective

Repair only the fixed-anchor geometry domain while preserving every scientific cell, support
fallback, hard gate, failure-accounting rule, input boundary, metric, seed, arm, split, and report
contract from the approved predecessor.

#### Reviewed state

Successor task
`experiments/tasks/20260805_probabilistic_field_pipeline_aabb_eligible_mixed.json` is `draft` with
no run root and a pending distinct prospective review. The failed predecessor and all earlier run
roots remain untouched. Prior outcome exposure is limited to the immutable terminal chronology
above; no successor outcome exists.

#### Changes

- `FieldSweepInitializer` now computes its train-only search AABB before anchor selection, tests
  every capped component-center ray for a positive-depth intersection, capacity-balances the
  unchanged requested count over eligible views, then performs the existing seeded top-mass-pool
  draw. It fails closed if the eligible total is below `n_init_3d`.
- Successful fits record exact policy, total/per-view candidate, eligible, and rejected counts.
  The successor validates those diagnostics with JSON-type-strict count consistency before every
  primary or independent-half hard-invariant evaluation.
- A small successor driver pins the complete approved support-fallback base driver by exact byte
  hash, changes only task/run identity, and requires the new eligibility contract. Spawned workers
  resolve to the successor file and canonical root.
- The design document now places forward-AABB eligibility before support-aware placement and
  distinguishes it from the task-scoped unmasked-support retry.
- Historical protocol tests replay immutable plans under their stored source bindings; runtime
  validation remains strict, and the successor test requires its current live source digest.

#### Evidence

- Exact protocol SHA-256:
  `b9abe3314b7fe7794362be3e8eb8c98ffa9606b6b1ec1ad194aa5f7f934c150d`.
- Exact 102-file source SHA-256:
  `5d9f9df94e2c3ba659932f8d7b18fd11af102a8e162292ee324f36b318bc94d4`.
- Pinned support-fallback base-driver SHA-256:
  `9d453b967b09005b63d3bef6aac48b817ac841ea2b7faa3593d4f980e4310169`.
- Canonical 549-cell-list SHA-256 remains
  `1af09dabc5de411ff09acdef30efa5da17e010f6fcad2b450b0dc08d31f005fc`.
- Canonical plan-payload SHA-256:
  `8cda2c39a1938da5388dbdfff2e9507548b91beb56692d9a9b61578a6962c321`.
- Data-seal SHA-256 remains
  `20e719d89628375c515db94102abf6e5018dbd6d686d0633235407fad5c7deb6`.
- A placement-only diagnostic on the failing sealed input retained all 64 requested tracks from
  4,074 eligible capped rays and rejected 22 non-intersecting rays, all from one of the eight
  selected cameras; it evaluated no held-out metric or quality/runtime claim.
- Eight focused geometry/successor tests passed, the broader field/protocol/contract tests passed,
  contract/data validation passed, and complete CPU-only `./scripts/verify.sh` passed. Only the
  two known PyTorch warnings remained.

#### Assumptions

A source ray with no positive-depth intersection with the train-only search volume cannot host a
valid source-exact anchor under the frozen placement domain. Filtering it before randomized
selection is a deterministic domain restriction, not an outcome-contingent fallback.

#### Uncertainties

The eligibility distribution over unattempted seeds/datasets and every successor quality,
convergence, stability, failure, and runtime outcome remain unknown. A prospective review must not
read a successor run because none is authorized yet.

#### Review Focus

Independently verify the bounds-before-draw ordering, held-out/mask/arm independence, unchanged
track count, capacity-aware balance, deterministic seeding, fail-closed insufficient-capacity
case, exact diagnostics, pinned base-driver import, live source binding, unchanged 549 cell bytes,
and immutable failed-run chronology. Attack bool/int/float count aliases and inconsistent totals.

#### Protected actions not taken

No successor run was initialized, no successor cell was executed, no protocol review artifact was
authored by the Driver, and no result report/viewer was rendered or opened.

#### Recommended Next Action

The distinct reviewer should perform an outcome-blind prospective protocol review and author
`experiments/reviews/20260805_probabilistic_field_pipeline_aabb_eligible_mixed_PROTOCOL_REVIEW.md`.
Only an exact approved digest may clear the blocker and authorize initialization.

### Review — forward-AABB-eligible successor v1

#### Verdict

Rejected

#### Self-reviewed

No

#### Correctness

The outcome-blind reviewer independently reproduced the proposed protocol, source, pinned-base,
cell-list, plan-payload, and data-seal hashes and confirmed that the geometry implementation does
compute train-only bounds before eligibility, filters before quota allocation and the seeded draw,
retains the requested pre-mask anchor budget when sufficient capacity exists, and fails before a
draw when global eligible capacity is insufficient. Spawned synthetic and calibrated commands
resolve to the successor wrapper, and the placement mode, cap, seed, and geometry preprocessing
are identical across native and candidate arms.

Execution is rejected for two prospective defects:

1. The sole pending-review blocker is part of protocol SHA-256
   `b9abe3314b7fe7794362be3e8eb8c98ffa9606b6b1ec1ad194aa5f7f934c150d` because both protocol
   digest implementations exclude only `status` and `protocol_review`. Clearing that blocker, as
   required before a `ready` task can validate, changes the protocol digest to
   `623cfb67496a9eb8f283b406e662de900ebb10f659569be3d826aa78d6e776c9`. Retaining it makes
   `ready` invalid. Therefore this exact reviewed digest has no valid transition to executable
   state.
2. `_validate_anchor_eligibility_diagnostics` does not bind each per-view eligible/rejected pair
   to that optimized view's actual capped candidate count. A reviewer-authored receipt with
   `target_component_counts_used=[2,2,2]`, eligible counts `[1,3,1]`, rejected counts `[1,0,0]`,
   and otherwise consistent totals is accepted even though one view reports three eligible rays
   from two candidates. The validator also accepts a negative integer `n_init_3d`. Thus aggregate
   arithmetic and JSON types are strict, but the promised exact per-view capacity receipt and
   positive requested budget are not fail-closed.

#### Evidence Quality

Outcome Access remained `none`. No successor run root or canonical review artifact exists. The
reviewer did not inspect predecessor quality, runtime, successful summaries, models, or reports;
only the three permitted failure-chronology hashes were recomputed, and all matched the handoff.
The seven focused fixed-sweep/successor tests passed. Direct task-policy, wrapper identity,
arm-config, lifecycle, negative-budget, internally consistent false-count, and impossible
per-view-capacity counterexamples were evaluated without running an official cell.

#### Simplicity

Both repairs are local. Remove the administrative prospective-review blocker before freezing the
next digest (the pending review record and draft status already prevent execution), and strengthen
the receipt validator to require positive `n_init_3d` and exact per-view
`eligible + rejected == target_component_counts_used[optimized_view]`, with type/range checks for
the referenced candidate-count vector.

#### Missing Cases

Add regressions for the blocker-to-ready digest transition, negative/zero `n_init_3d`, a
self-consistent but false candidate total, per-view capacity overflow or redistribution with
unchanged aggregate totals, wrong candidate-vector types/lengths, and all existing bool/int/float
aliases. Recompute the protocol and plan-payload hashes after the repair.

#### Required Changes

Repair both findings, leave the successor run absent, refresh the exact task/source/plan bindings,
rerun outcome-free validation, and return a new handoff for distinct prospective review. Do not
author the canonical `..._PROTOCOL_REVIEW.md`, clear the task for execution, or initialize the run
under the rejected digest.

#### Optional Improvements

None before approval.

### Handoff — forward-AABB successor v2 after prospective rejection

#### Objective

Repair both v1 review blockers without changing the experiment cells, scientific treatments, or
geometry-selection algorithm, then return a genuinely executable digest for distinct review.

#### Reviewed state

The v1 digest was rejected with Outcome Access `none`; no canonical review artifact or run root
was created. The exact rejection remains immediately above. The v2 task remains `draft`, review
fields remain pending, `blockers` is empty, and status/review fields remain outside the protocol
hash by construction.

#### Changes

- Removed the administrative prospective-review sentence from `blockers` before freezing. Draft
  status plus the pending review record already fail closed. A regression converts an in-memory
  copy to `ready` with an approval record and proves the protocol digest is unchanged.
- The eligibility validator now requires `n_init_3d > 0`, validates the complete
  `target_component_counts_used` vector and optimized indices with exact integer types and ranges,
  rejects duplicate/out-of-range optimized views, derives actual per-view capped capacities, and
  requires `eligible + rejected == actual capacity` for each optimized view plus exact global
  totals.
- Added the reviewer's internally consistent false-count and unchanged-total redistribution
  attacks, zero/negative budgets, false global candidate totals, wrong vector lengths,
  float/bool vector entries, and duplicate optimized indices.

#### Evidence

- Exact v2 protocol SHA-256:
  `7237f73f45b3515b8496b6a629a1e728414f634bee61b7b74391dbbfb189db06`.
- Exact v2 102-file source SHA-256:
  `0cca172519fe6840f2051b3683dac8336c01caecbb0d3dbd24e2e04d0b1b4c47`.
- Pinned base-driver SHA-256 remains
  `9d453b967b09005b63d3bef6aac48b817ac841ea2b7faa3593d4f980e4310169`.
- Canonical 549-cell-list SHA-256 remains
  `1af09dabc5de411ff09acdef30efa5da17e010f6fcad2b450b0dc08d31f005fc`.
- Canonical v2 plan-payload SHA-256:
  `205eb30228c3c2ac4d976fac9f7d91263703e35b379cface83555a59a5b3a75b`.
- Data-seal SHA-256 remains
  `20e719d89628375c515db94102abf6e5018dbd6d686d0633235407fad5c7deb6`.
- Direct draft-to-ready simulation returns the identical protocol digest and an empty blocker
  list.
- All five successor tests, all five fixed-sweep tests, the complete broader
  field/protocol/contract set, contract/data validation, and full CPU-only `./scripts/verify.sh`
  pass. Only the two known PyTorch warnings remain.

#### Assumptions

The actual capacity authority is the successful fit's capped component vector indexed by its exact
optimized global view indices; placement per-view diagnostics must reconcile to those bytes.

#### Uncertainties

All successor outcomes remain unknown. No official cell has run.

#### Review Focus

Re-run both original counterexamples first. Then mutate zero/negative/bool/float budgets, coherent
global totals with impossible per-view redistribution, target-vector length/type/range, duplicate
and out-of-range view indices, and the draft-to-ready lifecycle. Rebind every exact digest and
confirm the 549 cell bytes are unchanged.

#### Protected actions not taken

No canonical review artifact was authored by the Driver, no task was made ready, no run was
initialized, and no successor outcome, report, or viewer was accessed.

#### Recommended Next Action

Perform the second distinct prospective review on these exact v2 bytes. Approve only if the task
can transition to ready without digest drift and all capacity-receipt counterexamples fail.

### Review — forward-AABB successor v2 protocol accepted

#### Verdict

Accepted

#### Self-reviewed

No

#### Correctness

The distinct outcome-blind reviewer approved exact protocol
`7237f73f45b3515b8496b6a629a1e728414f634bee61b7b74391dbbfb189db06`, live 102-file source
binding `0cca172519fe6840f2051b3683dac8336c01caecbb0d3dbd24e2e04d0b1b4c47`, and pinned base
driver `9d453b967b09005b63d3bef6aac48b817ac841ea2b7faa3593d4f980e4310169`. Bounds precede
eligibility and the seeded draw; capacity-aware selection preserves the requested pre-mask count
or fails before drawing; native and candidate arms share identical geometry preprocessing; and
every primary/half result must carry a type- and capacity-consistent eligibility receipt.

Both v1 blockers are resolved. An in-memory pending-draft to approved-ready transition preserves
the exact protocol digest because `blockers` is already empty. The original impossible per-view
redistribution is now rejected against `target_component_counts_used`; zero/negative/bool/float
budgets, coherent false totals, wrong capacity vector types/lengths/ranges, duplicate/out-of-range
optimized views, aggregate aliases, and task/source/base/command mutations all fail closed.

#### Evidence Quality

The 549-cell digest
`1af09dabc5de411ff09acdef30efa5da17e010f6fcad2b450b0dc08d31f005fc`, plan-payload digest
`205eb30228c3c2ac4d976fac9f7d91263703e35b379cface83555a59a5b3a75b`, and data-seal digest
`20e719d89628375c515db94102abf6e5018dbd6d686d0633235407fad5c7deb6` matched independently;
all 549 cells are exactly equal to the approved base plan. All nine focused successor/fixed-sweep
tests, the broader six-file suite, the complete CPU suite including slow tests, and
`./scripts/verify.sh` passed with only the two documented PyTorch warnings. The canonical review
is `experiments/reviews/20260805_probabilistic_field_pipeline_aabb_eligible_mixed_PROTOCOL_REVIEW.md`
with SHA-256 `b55662d7de5d1c4476b67a060f2335348d1d69a29e9ae92f871e2962e65f6e17`.

Outcome Access remained `none`. No successor run was initialized or inspected, and predecessor
access was restricted to the three permitted immutable failure-chronology hashes.

#### Simplicity

The v2 repair is confined to removing the digest-bearing administrative blocker and strengthening
the successor receipt validator/tests. The geometry algorithm, pinned base producer, scientific
cells, treatments, thresholds, seeds, splits, metrics, failure rules, and report contract are
unchanged.

#### Missing Cases

Official successor operability, failure distribution, quality, convergence, resource curves,
report rendering, and orbit-viewer behavior remain unknown until exact-command execution and the
independent results audit. Approval is not outcome evidence.

#### Required Changes

None before the Driver records this exact approval, transitions only administrative task/review
fields to `ready`, initializes the single canonical run root, and executes the exact command. Any
digest-bearing task edit, bound-source change, or pinned-base byte change requires a new
prospective review.

#### Optional Improvements

Preserve the failed predecessor and v1 rejection as immutable chronology. Retain the task's
development-only claim boundary and do not interpret or render official outcomes before the
independent results audit.

### Review Accepted — forward-AABB successor execution authorized

The Driver read the canonical v2 review in full and verified artifact SHA-256
`b55662d7de5d1c4476b67a060f2335348d1d69a29e9ae92f871e2962e65f6e17`, exact approved
protocol `7237f73f45b3515b8496b6a629a1e728414f634bee61b7b74391dbbfb189db06`, exact live source
binding `0cca172519fe6840f2051b3683dac8336c01caecbb0d3dbd24e2e04d0b1b4c47`, and pinned base
driver `9d453b967b09005b63d3bef6aac48b817ac841ea2b7faa3593d4f980e4310169`. The task now records
reviewer `Codex-probabilistic-field-protocol-reviewer`, verdict `approved`, empty blockers, and
status `ready`; the experiment index is synchronized. The protocol digest remains identical after
the administrative transition. Turn remains with the Driver for canonical initialization and
exact-command execution. Outcome interpretation and report rendering remain gated on the
independent results audit.

### Run Failed Closed — forward-AABB successor

The single canonical development run
`runs/20260805_probabilistic_field_pipeline_aabb_eligible_mixed` was initialized against the
approved task lock and executed with the exact reviewed command. It completed the unchanged
483-cell synthetic matrix, the discarded warmup, and twenty-one calibrated attempts before
failing closed at measured cell 22/66 (`karate_00060_default`, seed `80501`,
`all_candidate_mechanisms`). Nineteen of the preceding measured cells were successful terminals;
two candidate cells were retained structured failures under the declared
`hard invariant violation: transport real mass` gate.

The failing cell raised exact `RuntimeError: a supported projection left the valid camera domain
during M-step` from the transactional association clone. Because the exception occurred before
the whole-fit support-fallback wrapper could return its explicit no-fallback provenance, the
failure receipt carried `support_fallback: null`. The approved continuable-failure validator
therefore rejected it and the root aborted rather than skipping, imputing, or weakening a gate.

Immutable failure chronology:

- Root failure SHA-256:
  `63206958f3fd963e277d8487000f9b76383ae5a8aa9120b4235a65cbd32216d4`.
- Run-receipt SHA-256:
  `0d1abb1b84bba0d3d72edede63cf3582e31e22a665961ef9ab25448fc97758fa`.
- Terminal cell failure SHA-256:
  `07a2ff8e0fbe8cfcf2926a56b77ea8ac40704f509b38a53fca35b4324671396b`.
- Failure timestamp: `2026-08-06T05:04:59.015687+00:00`.

No metric payload, aggregate, report, or viewer was opened. This run remains immutable and cannot
be resumed or republished as a completed matrix.

### Driver Handoff — transactional association rollback successor

The narrow completion repair is to use the already implemented `FieldAssociationConfig`
transaction boundary with `failure_policy="rollback"` for the candidate association stage. An
association exception then returns the untouched placement plus an exact rollback diagnostic;
the existing candidate hard-invariant gate still rejects the cell for missing transport plans,
after the whole-fit wrapper has attached an explicit requested/effective support-fallback record.
This changes run continuity only: it cannot turn a failed association into a successful candidate,
cannot impute quality or runtime, and cannot make a rolled-back model eligible for mechanism
claims. A new task ID, exact task/source/driver bindings, focused adversarial tests, and a distinct
prospective review are required before any successor outcome access.

### Driver Handoff — association-rollback successor v1 protocol

#### Summary

Created draft task
`experiments/tasks/20260805_probabilistic_field_pipeline_association_rollback_mixed.json` and
driver `scripts/experiments/20260805_probabilistic_field_pipeline_association_rollback_mixed.py`.
The wrapper reuses the immutable support-fallback producer, retains the exact forward-AABB
eligibility contract and receipt validator, and changes only candidate association failure policy
from `raise` to the existing transactional `rollback` mode.

For every candidate primary/half fit, the wrapper accepts only one of two exact states:

- association object present, `association_status=committed`, no failure field; or
- association absent, `association_status=rolled_back`, and one identical non-empty
  `RuntimeError: ...` or `ValueError: ...` string on the result and placement diagnostics.

A rolled-back result then enters the unchanged base invariant checker and must fail the required
transport plan/real-mass/fixed-point/candidate gates. It cannot become cell success or contribute a
quality/runtime value. Native arms remain association-disabled. The task claim boundary discloses
the prior measured-cell failure and all three immutable hashes; no prior metric payload was read.

#### Exact review bindings

- Protocol SHA-256:
  `e57d58112fd6f95467e8ddacdb4daad7fc9d83ed48b8b9f336a32b1966a92e87`.
- Live source-binding SHA-256:
  `b17dec48edf2f07469dc9f6b197d062e4c0ee59698a05772c3674ad1fdf9b2eb`.
- Successor driver SHA-256:
  `9a53815e2e0f17c2b40c9c67295c319eec4fff163541012804438717d5801bff`.
- Pinned base-driver SHA-256:
  `9d453b967b09005b63d3bef6aac48b817ac841ea2b7faa3593d4f980e4310169`.
- Canonical 549-cell-list SHA-256, unchanged:
  `1af09dabc5de411ff09acdef30efa5da17e010f6fcad2b450b0dc08d31f005fc`.
- Canonical plan-payload SHA-256:
  `dc1f350715518c2d78a6e066c7173a12e9b038ba290903f09800d88322cd34fa`.
- Data-seal SHA-256, unchanged:
  `20e719d89628375c515db94102abf6e5018dbd6d686d0633235407fad5c7deb6`.

#### Verification

- Ten focused AABB/association-successor tests pass, including task/contract tampering,
  draft-to-ready digest invariance, exact committed/rolled-back provenance, empty or wrong
  exception types, mismatched diagnostics, and preservation of the existing hard gate.
- The broader protocol, pipeline, field-lifter, and fiber-correspondence suite passes.
- `experiment_contract.py validate` and `validate-data` pass.
- Full CPU-only `./scripts/verify.sh` passes: lint, formatting, non-slow tests, docs sync, ARA,
  script layout, agent workflow, and experiment contracts. Only the two known PyTorch warnings
  remain.

#### Protected actions not taken

No successor run root exists. No official successor cell, metric, aggregate, report, or viewer has
been accessed. The canonical review artifact was not authored by the Driver, and the task remains
`draft` with pending review.

#### Review focus

Independently verify that rollback is transactional, cannot produce success, and reaches the
unchanged hard gate; mutate status/failure values, exception types, task policy/contract, source and
base bindings, command identity, and lifecycle fields. Confirm exact cell equality with the
approved AABB plan and that prior outcome exposure is limited to immutable failure chronology.

### Review — association-rollback successor v1 protocol accepted

#### Verdict

Accepted

#### Self-reviewed

No

#### Correctness

The distinct outcome-blind reviewer approved exact protocol
`e57d58112fd6f95467e8ddacdb4daad7fc9d83ed48b8b9f336a32b1966a92e87`, live 102-file source
binding `b17dec48edf2f07469dc9f6b197d062e4c0ee59698a05772c3674ad1fdf9b2eb`, successor driver
`9a53815e2e0f17c2b40c9c67295c319eec4fff163541012804438717d5801bff`, and pinned base driver
`9d453b967b09005b63d3bef6aac48b817ac841ea2b7faa3593d4f980e4310169`.

The bound association stage mutates a copied inverse-projection fiber and commits it only after
success. An exact simulated partial-M-step failure left every original placement tensor bitwise
unchanged. Exact committed/rolled-back states are required on candidate primary and realized
independent-half fits; malformed provenance aborts outside structured continuation. A valid
rollback reaches the unchanged hard failure `transport plan missing, transport real mass,
transport fixed point, candidate gate`, so it cannot become success, impute a metric, substitute
the native arm, or make a rejected model claim-eligible. Native configuration is exactly unchanged.

#### Evidence Quality

The unchanged 549-cell digest
`1af09dabc5de411ff09acdef30efa5da17e010f6fcad2b450b0dc08d31f005fc`, successor plan digest
`dc1f350715518c2d78a6e066c7173a12e9b038ba290903f09800d88322cd34fa`, and data-seal digest
`20e719d89628375c515db94102abf6e5018dbd6d686d0633235407fad5c7deb6` matched independently;
all 549 cells are exactly equal to the approved forward-AABB plan. Candidate realized config
differs only at `association.failure_policy`; native realized config has no difference.

All thirteen malformed association-receipt attacks and twelve task/source/base/command attacks
failed closed. All eleven focused tests, the broader six-file suite, task/data validation, and
`./scripts/verify.sh` passed with only the two documented PyTorch warnings. The canonical review is
`experiments/reviews/20260805_probabilistic_field_pipeline_association_rollback_mixed_PROTOCOL_REVIEW.md`
with SHA-256 `7c0b99c864ff7fecb0e3f5e3180c615f4e157ea70d0d010c78ce29e3238d65f4`.

Outcome Access remained `none`. Predecessor access was restricted to the three permitted immutable
failure files, whose hashes matched the task chronology. No successor run was initialized or
inspected, and no predecessor metric, summary, aggregate, model, report, preview, or viewer was
opened.

#### Simplicity

The successor wraps the immutable support-fallback producer and changes one candidate association
configuration field. The scientific cells, native path, forward-AABB selection, support fallback,
thresholds, metrics, aggregation, report behavior, and failure continuation remain pinned.

#### Missing Cases

Official successor operability, failure distribution, quality, convergence, resources, report
rendering, and orbit-viewer behavior remain unknown until exact-command execution and the
independent results audit. Approval is not outcome evidence.

#### Required Changes

None before the Driver records this exact approval, transitions only administrative task/review
fields to `ready`, initializes the single canonical run root, and executes the exact command. Any
digest-bearing task edit or bound source/successor/base byte change requires a new prospective
review.

#### Optional Improvements

The bound fitter already clones field-mass capacities immediately on entry. A future revision may
also clone them at the outer association boundary for defense in depth, but that is not required
for these exact reviewed bytes and would require a new source binding and review.

### Review Accepted — association-rollback successor execution authorized

The Driver read the canonical review in full and verified artifact SHA-256
`7c0b99c864ff7fecb0e3f5e3180c615f4e157ea70d0d010c78ce29e3238d65f4`, exact approved
protocol `e57d58112fd6f95467e8ddacdb4daad7fc9d83ed48b8b9f336a32b1966a92e87`, live source
binding `b17dec48edf2f07469dc9f6b197d062e4c0ee59698a05772c3674ad1fdf9b2eb`, successor driver
`9a53815e2e0f17c2b40c9c67295c319eec4fff163541012804438717d5801bff`, and pinned base
driver `9d453b967b09005b63d3bef6aac48b817ac841ea2b7faa3593d4f980e4310169`.
The task now records the exact reviewer, approved verdict, protocol digest, canonical artifact,
empty blockers, and `ready` status; the experiment index is synchronized. The administrative
transition leaves the protocol digest unchanged. Turn is back with the Driver for canonical run
initialization and exact-command execution. Outcome interpretation and report rendering remain
gated on the independent results audit.

### Handoff — association-rollback successor results audit

#### Objective

Independently audit the completed protected development run before any Driver outcome
interpretation, report rendering, experiment-ledger entry, claim disposition, or viewer launch.

#### Reviewed state

The exact reviewed command completed under
`runs/20260805_probabilistic_field_pipeline_association_rollback_mixed` with the task lock created
at `2026-08-06T05:42:15.590692+00:00`. The prospective protocol, source, successor driver, pinned
base driver, data seal, plan, and review bindings remain those recorded in the immediately
preceding accepted review. The Driver observed producer progress and terminal stdout only and did
not inspect `metrics.json`, `cell_results.json`, per-dataset `result.json`, model outputs, curves,
previews, or quantitative summaries.

#### Changes

The producer finished the frozen 483-cell synthetic matrix, discarded warmup, all 66 measured
dataset/seed/arm attempts, and aggregate artifact production. Terminal stdout reported structured
candidate hard-invariant rejections while continuing according to the approved failure contract;
it ended with `Producer execution complete. Independent results audit is required before render.`
No protocol, task, source, driver, threshold, seed, split, metric, or continuation rule was edited
during execution.

#### Evidence

- Canonical run root:
  `runs/20260805_probabilistic_field_pipeline_association_rollback_mixed`.
- Root producer records now exist, including `task.lock.json`, `run_receipt.json`,
  `aggregate_commit_receipt.json`, `cell_results.json`, `metrics.json`,
  `training_history.json`, `environment.json`, `input_boundary_receipt.json`, and
  `resource_receipt.json`.
- Producer stdout reached `[69/69] aggregate models, curves, previews, and producer records` and
  exited with status zero.
- Terminal stdout exposed seven structured candidate failures: six named `transport real mass`
  and one named `transport plan missing, transport real mass, transport fixed point, candidate
  gate`. These counts and their rollback/support-fallback classification are untrusted producer
  orientation and must be recomputed from immutable receipts by the Reviewer.

#### Assumptions

The completed producer state is append-only. Audit records may be added only at the canonical
`benchmarks/results/20260805_probabilistic_field_pipeline_association_rollback_mixed_AUDIT.{md,json}`
paths; generated reports and browser receipts remain gated until audit acceptance.

#### Uncertainties

The Driver has not established terminal-cell accounting, task-lock/source integrity, AABB
eligibility consistency, transactional rollback counts, support-fallback counts, absence of
imputation/substitution, metric recomputation, claim gates, comparative quality, convergence,
resource behavior, or viewer operability. All outcome statements remain development-only and
unavailable pending audit.

#### Review Focus

Use `realtime-gs-results-audit` with Outcome Access allowed. Independently verify the exact sealed
bindings and all 66 terminal cells; classify success, hard-gate rejection, association rollback,
and support fallback; verify every forward-AABB eligibility receipt and failure continuation;
prove failed cells contribute no imputed, substituted, or claim-eligible metrics; recompute all
frozen gates and aggregate directions from raw records; and dispose every proposed claim within
the development-only boundary.

#### Protected actions not taken

The Driver did not render `index.html`/`README.md`/`manifest.json`, open any result JSON or metric
curve for interpretation, write `docs/EXPERIMENTS.md` or ARA claims, run a results viewer, change a
default, commit, push, or publish.

#### Recommended Next Action

Have the distinct Reviewer write the canonical Markdown and JSON results-audit records and append
an independent Review verdict. Only after acceptance should the Driver render the root and eleven
per-dataset pages, launch and browser-smoke all eleven orbit viewers, rerender the final manifest,
run the bundle/contract gates, log the bounded result, review the diff, and verify the repository.

### Review — association-rollback successor results accepted with claim narrowing

#### Verdict

Accepted

#### Self-reviewed

No

#### Correctness

The immutable producer result is complete and internally consistent for bounded development
interpretation. Independent raw-record recomputation exactly reproduced the 483-cell synthetic
factorial, all hard-invariant measurements, every per-seed mechanism predicate, all 66 calibrated
terminals, all root medians and successful-cell denominators, and the producer result payload.

The frozen isolated decisions are now authoritative for this attempt: rank-aware covariance and
probability support pass at `3/3` seeds; field-mass association, projection-nonlinearity topology,
and progressive scheduling fail at `0/3` seeds and are retired for this protocol. The all-candidate
calibrated arm remains descriptive and cannot rescue those failures.

Calibrated accounting is exact: native controls succeed `33/33`; all-candidate succeeds `26/33`;
seven candidate terminals are structured failures. Six fail `transport real mass`; the
`karate_00060_default` seed-80501 candidate fails the exact four missing-transport gates expected
after rollback. Failed terminals have no summary or quality/runtime value, appear in history only
as a zero success indicator, and retain rejected presentation-only models. There is no imputation,
native substitution, or failed-model claim eligibility.

Exactly two successful cells—the native and candidate arms of
`stage_00008_structsplat_no_boundary_fullres`, seed 80501—use the frozen whole-cell unmasked retry.
Their identical cross-surface provenance is valid and their interpretation is restricted to
unmasked operability. The calibrated success fraction is `59/66 = 0.8939393939393939`; the
successful fallback fraction is `2/66 = 0.030303030303030304`.

#### Evidence Quality

Canonical independent records are:

- `benchmarks/results/20260805_probabilistic_field_pipeline_association_rollback_mixed_AUDIT.md`,
  SHA-256 `3622adcf90d01e9e23d0390f8b4a75d40d17d2ae65097682e5f482403e8e0026`;
- `benchmarks/results/20260805_probabilistic_field_pipeline_association_rollback_mixed_AUDIT.json`,
  SHA-256 `e18b302126b4a51b0b9829a54342602f1ecccfcff517bb20cb8fb90e1c762143`.

The audit independently matched task `42ec54…ff43`, protocol `e57d58…6a92`, prospective review
`7c0b99…65f4`, data seal `20e719…deb6`, 102-file result-producing source `b17dec…2eb`, successor
driver `9a5381…1bff`, pinned base driver `9d453b…0169`, task lock `d73b1d…ae7e`, and the exact command.
All 1,842 repeated compact input records match current bytes and the 309-file seal. Every worker
guard, held-out boundary, terminal receipt hash, successful-only resource summary, dataset curve,
presentation-copy hash, and PLY structure checked cleanly. The pre-audit run inventory contains
565 files / 9,786,401 bytes with digest
`0e12cb1e4b4435d0344e64e1fe288c285e35daf887c16a7fff9ceabc59a46bad`.

The source lock is deliberately dirty and development-only. Exact result-producing Python bytes
are separately and completely bound, but the wider dirty-state digest is not itself a stored
source snapshot. This evidence must not be described as a clean, replay-complete repository
checkpoint.

All 59 successful primary summaries directly serialize consistent forward-AABB totals and
per-view counters. The exact bound wrapper validates AABB diagnostics before successful
publication and before every published primary hard gate. Independent-half and failed-primary
counters are not serialized, however, so their exact values cannot be recounted from raw artifact
bytes. Likewise, the rollback-consistent four-gate terminal proves the bound rollback path passed
in-memory provenance validation, but the original caught exception and `association_failure`
string were not retained. These are mandatory claim narrowings and future evidence-hardening
targets, not grounds to impute or reject the current zero-success terminal.

The environment records Linux/Python/NumPy/Torch versions, one CPU/Torch thread, no CUDA use,
finite timing/RSS, and `x86_64`. It does not record a named host, useful CPU model, host load, or
idleness. All timing, convergence, and RSS values are therefore host-local diagnostics; no speed,
memory, GPU, or real-time claim is accepted.

#### Simplicity

No scientific source, task, protocol, threshold, producer record, or model was repaired after
outcome access. The audit adds only the canonical append-only Markdown/JSON dispositions and this
durable review. It uses the frozen rules directly and rejects secondary or combined metrics as a
rescue for failed primary gates.

#### Missing Cases

- The original association exception text and per-fit rollback receipt are not serialized.
- Failed-primary and independent-half AABB counter arrays are not serialized.
- A named idle-host/load receipt is absent; performance remains unclaimable.
- Rendered root/per-dataset pages, browser behavior, and all eleven orbit-viewer launches remain
  untested by design; visual PNG/GIF bytes were hashed but not opened.
- This is capped development evidence over eleven sealed Gaussian2D field sets, not complete-field,
  source-RGB, physical-geometry, spatial-resolution, GPS-Gaussian reproduction, globally coupled
  OT, cross-scene, GPU, production-default, or independent-half-accuracy evidence.

#### Required Changes

The Driver must preserve every audit claim disposition while rendering and logging results. Before
task closeout, update the stale lifecycle assertion in
`test_association_rollback_successor_preserves_cells_and_binds_failure_completion`: it still
expects the pre-review task status `draft`, while the prospectively approved executed task is
correctly `ready`. Then run the complete verification gate. Do not mutate or rerun the producer
attempt to address that downstream test.

The Driver may now render the root and eleven per-dataset reports, launch and browser-smoke the
eleven orbit viewers, attach truthful smoke receipts, log the bounded experiment disposition, and
perform final review/verification. Turn returns to the Driver with task status `In progress`.

#### Optional Improvements

For a future task/source binding, serialize one per-fit receipt for primary and both independent
halves, including forward-AABB totals/per-view arrays and committed/rolled-back association status,
typed failure text, and a digest of the untouched rollback placement. Record a stable host ID,
full CPU model, and load/idleness evidence before any performance experiment. These changes are
not authorized retroactive edits to this immutable run.

### Handoff — final pipeline-integrated closeout review

#### Objective

Independently review the complete RTGS-012 implementation and downstream closeout state after the
already accepted protected results audit, including the lifecycle repair, generated-report/browser
receipts, bounded experiment wording, ARA dispositions, and final verification evidence.

#### Reviewed state

The canonical producer remains immutable at
`runs/20260805_probabilistic_field_pipeline_association_rollback_mixed`. Its independently accepted
audit is unchanged at
`benchmarks/results/20260805_probabilistic_field_pipeline_association_rollback_mixed_AUDIT.{md,json}`.
The working tree contains the owner-authorized RTGS-011 literature artifacts, RTGS-012 opt-in
implementation/protocol/result artifacts, downstream report receipts and closeout documentation,
plus unrelated owner `.idea/` changes that remain untouched. No commit or push is requested.

#### Changes

- Added the CPU-first, opt-in probabilistic field orchestration, explicit mask/mass/opacity
  semantics, finite-gated transactional association, nonlinearity topology selection, progressive
  scheduling with mandatory cleanup, independent-half stability, public/CLI seams, and deterministic
  tests.
- Added the all-dataset protected protocol and immutable chronology of failed predecessors, then
  executed only the prospectively accepted association-rollback successor.
- Preserved the independent scientific dispositions: shape and probability support pass `3/3`;
  association, topology, and scheduling fail `0/3` and are retired; calibrated native/candidate
  completion is descriptive `33/33` versus `26/33`, with seven explicit failures, two successful
  unmasked fallbacks, and no imputation or substitution.
- Updated the stale lifecycle assertion to the approved `ready` state without touching frozen
  producer source, task, thresholds, or records.
- Rendered one root and eleven child reports, retained strict/rich browser-smoke receipts, and made
  seven source-bound repository evidence links HTTP-reachable through run-local byte-exact mirrors.
  No producer artifact or result-producing source was modified by that presentation-only step.
- Logged the bounded result in `docs/EXPERIMENTS.md`, reconciled the design/architecture boundary,
  and crystallized O159–O164 into C35–C40, including three explicit refutations.

#### Evidence

- Canonical audit SHA-256 values remain
  `3622adcf90d01e9e23d0390f8b4a75d40d17d2ae65097682e5f482403e8e0026` (Markdown) and
  `e18b302126b4a51b0b9829a54342602f1ecccfcff517bb20cb8fb90e1c762143` (JSON).
- `experiment_contract.py check-run` and `check_results_bundle.py` pass on the canonical run.
- Final Chrome 149/WebGL2 smoke: 12/12 report pages, 586/586 local targets, and 11/11 viewers pass
  ready/WebGL2/Gaussian-renderer/non-background/orbit-change checks with no fatal or unclassified
  client error. The known Viser first-frame bounding-sphere and THREE.Clock warnings are classified
  and bounded by explicit renderer-ready diagnostics.
- Focused pipeline/protocol/contract suite: 116 tests passed.
- Final `./scripts/verify.sh`: Ruff, formatting, non-slow CPU tests, docs sync, ARA (40 claims),
  script layout, agent workflow, and experiment contracts all pass; only the two established
  PyTorch warnings remain.

#### Assumptions

The run-local evidence mirrors are presentation-bundle artifacts only: each copy is byte-identical
to its canonical repository source and exists because the frozen renderer emits repository-relative
links while the requested HTTP server is rooted at the run directory. The live local servers and
Chrome tabs are intentionally left running for owner inspection and are not scientific evidence.

#### Uncertainties

The original caught association exception and failed/half per-fit AABB arrays are not serialized.
The run is a dirty-source-bound development attempt over capped 512-component proxies; host timing
and RSS lack a named idle-host receipt. Complete-field quality, visual usability, physical geometry,
independent-half accuracy, GPU/real-time performance, memory superiority, cross-scene generality,
and production-default utility remain unsupported.

#### Review Focus

Challenge the implementation invariants and disabled-control parity; failure/rollback/fallback and
no-imputation boundaries; the exact three negative mechanism dispositions; lifecycle-only test
repair; report-smoke receipt truthfulness; run-local mirror isolation; ARA/doc consistency; and
whether the Pipeline-integrated task can close without changing any default or overstating quality,
convergence, or performance.

#### Protected actions not taken

No producer rerun or repair, source/task/protocol/threshold mutation after outcome access, failed
metric imputation, native substitution, default promotion, CUDA benchmark, external publication,
commit, push, or change to unrelated owner `.idea/` files was performed.

#### Recommended Next Action

If accepted with no required changes, append a distinct final Review verdict, return Turn to the
Driver, archive this complete record as
`docs/tasks/RTGS-012-probabilistic-compact-field-lifting-pipeline.md`, reset the active task file,
rerun workflow/docs/ARA checks, and hand off the still-live report/viewer URLs.

### Review (independent final pipeline-integrated closeout)

#### Verdict

Revision required

#### Self-reviewed

No

#### Correctness

The implementation, protected result boundary, and downstream presentation state are otherwise
fit for Pipeline-integrated closeout. The added mechanisms remain opt-in: association is `None`,
the target-component cap is `None`, the incumbent hard-mask and largest-density-mass split paths
remain selected, and refit uses all views unless the new controls are requested. Transactional
association runs on a cloned fiber and either commits the complete stage or restores the original
placement; the independent-half wrapper removes shared and view-indexed geometry without proven
half-local provenance and verifies the realized camera partitions. The lifecycle-only assertion
now expects the prospectively approved task state `ready` and does not change producer source,
task thresholds, or records.

The final experiment disposition preserves the accepted audit exactly: rank-aware shape and
probability support pass only their frozen synthetic rules at `3/3`; association, topology, and
progressive scheduling fail at `0/3` and remain retired for this protocol. The calibrated matrix
remains descriptive at native `33/33` and candidate `26/33`, with seven explicit failed candidate
terminals, two successful whole-cell unmasked fallbacks, no failed quality/runtime imputation, and
no native substitution. No default, quality, convergence, memory, GPU, real-time, or visual
utility conclusion follows from those records.

One documentation statement is too absolute for the now outcome-bearing design. Lines 3–4 of
`docs/DESIGN_probabilistic_field_pipeline.md` say that nothing in the document is a speed or
convergence claim, while section 11.1 records the bounded finding that progressive scheduling
reduced fresh-process refit time by `15.8–17.4%` but failed the joint 1% endpoint guard. A bounded,
negative, host-local finding is still a speed/convergence statement, even though it is not a
general or positive performance claim. The disclaimer must be narrowed before acceptance.

#### Evidence Quality

The canonical audit remains byte-identical at SHA-256
`3622adcf90d01e9e23d0390f8b4a75d40d17d2ae65097682e5f482403e8e0026` (Markdown) and
`e18b302126b4a51b0b9829a54342602f1ecccfcff517bb20cb8fb90e1c762143` (JSON). Independent
rechecks reproduced the result-producing source binding `b17dec48edf2f07469dc9f6b197d062e4c0ee59698a05772c3674ad1fdf9b2eb`
and pinned base-driver binding `9d453b967b09005b63d3bef6aac48b817ac841ea2b7faa3593d4f980e4310169`.
`experiment_contract.py check-run` and `check_results_bundle.py` both pass.

The run contains one root and eleven per-dataset HTML reports. Its summaries expose every finite
successful arm/seed metric as a curve, preserve failed cells only through the explicit zero-success
curve and notes, and do not invent failed quality or runtime points. All seven HTTP presentation
mirrors compare byte-for-byte with their canonical repository sources. The final Chrome
149/WebGL2 receipt covers 12/12 pages, 586/586 local targets, and 11/11 viewers with ready,
Gaussian-renderer-ready, non-background, WebGL2, and orbit-change checks; all viewer processes and
the report server responded during this review. The two uniformly classified Viser/THREE warnings
are non-fatal under the recorded renderer diagnostics. This is technical operability evidence,
not visual-quality evidence.

The focused six-file pipeline/protocol/contract suite passes all 116 tests. The full CPU pytest
suite passes with only the two established PyTorch warnings. `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh`
passes Ruff, formatting, the complete non-slow CPU suite, docs sync, all 40 ARA
claims, script layout, workflow validation, and experiment contracts. C35–C40 and A20 retain the
two supported bounded findings, three explicit refutations, descriptive terminal accounting, and
all accepted audit limitations.

#### Simplicity

The implementation extends existing field placement, inverse-projection fiber, transport, refit,
topology, validation, report, and viewer seams rather than introducing a parallel production
pipeline. The remaining required change is one disclaimer narrowing; it requires no producer
rerun, scientific artifact edit, source change, report regeneration, or audit mutation.

#### Missing Cases

- The original caught association exception and failed/independent-half per-fit AABB arrays are
  not serialized.
- The dirty development source lock is not a clean replay-complete repository checkpoint.
- Host identity/load/idleness is insufficient for a performance claim; timing and RSS remain
  host-local diagnostics.
- The calibrated fields are deterministic 512-component proxies, not complete fields.
- Browser smoke does not establish visual quality, physical geometry, resolution,
  independent-half accuracy, GPS-Gaussian reproduction, globally coupled OT, cross-scene
  generality, GPU/real-time performance, memory superiority, or production-default utility.

#### Required Changes

Narrow the design header's absolute disclaimer so it excludes **general or positive** quality,
speed, and convergence claims (or equivalent wording) while permitting section 11.1's bounded
negative timing/convergence disposition. Do not change the quantified outcome, scientific source,
task, protocol, thresholds, producer records, canonical audit, or unrelated `.idea/` files.

After that one-line documentation repair, return the unchanged closeout state for a bounded
independent re-review before archiving RTGS-012.

#### Optional Improvements

In a future protected attempt, serialize typed association failures and complete primary/half-fit
AABB receipts, record stable idle-host/load metadata, and retain browser-smoke tooling as a durable
script if this presentation workflow will recur. These are evidence-hardening opportunities, not
requirements for this Pipeline-integrated task.

### Handoff — bounded disclaimer repair

#### Objective

Re-review only the final closeout review's sole required change: narrow the design header so the
document can report bounded negative timing/convergence evidence without implying a general or
positive performance claim.

#### Reviewed state

The preceding independent review accepted every implementation, protocol, result, report/viewer,
claim-ledger, and verification surface except the absolute wording at lines 3–4 of
`docs/DESIGN_probabilistic_field_pipeline.md`. The canonical producer, audit, source/task bindings,
reports, browser receipts, and unrelated owner `.idea/` changes remain untouched.

#### Changes

Changed only the header sentence from “Nothing in this document is a quality, speed, convergence,
novelty, or production-default claim” to “Nothing in this document is a general or positive
quality, speed, convergence, novelty, or production-default claim.” Section 11.1's exact
`15.8–17.4%` timing observation, failed 1% endpoint guard, three retired mechanisms, and all audit
boundaries are unchanged.

#### Evidence

After the one-line repair, `docs_sync.py`, `check_ara.py` (40 claims),
`check_agent_workflow.py`, and `git diff --check` all pass. The preceding review independently
passed the focused 116 tests, full CPU pytest, canonical `verify.sh`, run/bundle gates, hashes,
mirrors, report counts, viewer receipts, and live endpoints; no code or executable input changed.

#### Assumptions

“General or positive” narrows the disclaimer exactly as requested while preserving permission to
record bounded negative host-local observations and prohibiting their promotion into an unqualified
speed, convergence, quality, novelty, or default claim.

#### Uncertainties

None beyond the evidence limitations already preserved in the preceding review and canonical audit.

#### Review Focus

Confirm the one-line wording repair satisfies the required change without weakening section 11.1,
changing any scientific outcome, or expanding the accepted evidence boundary.

#### Protected actions not taken

No source, test, task, protocol, threshold, producer record, audit, run artifact, report, viewer,
default, claim number, unrelated `.idea/` file, commit, or remote was changed.

#### Recommended Next Action

If the sole required change is accepted, append the bounded independent acceptance, set Status to
`Accepted` and Turn to `driver`, then let the Driver archive RTGS-012, restore the active-task
template, rerun final workflow gates, and hand off the live report/viewer URLs.

### Review (independent acceptance after disclaimer repair)

#### Verdict

Accepted

#### Self-reviewed

No

#### Correctness

The sole required repair is exact. The design header now excludes **general or positive** quality,
speed, convergence, novelty, and production-default claims, while section 11.1 still records the
bounded negative result: the host-local refit-time reduction did not pass the frozen joint
time-to-quality rule. This removes the absolute contradiction without changing or weakening the
quantified outcome, its three retired mechanism decisions, or any accepted evidence limitation.

No implementation, test, task, protocol, threshold, producer record, run artifact, report/viewer
receipt, claim number, or default was changed for this repair. The canonical audit hashes remain
`3622adcf90d01e9e23d0390f8b4a75d40d17d2ae65097682e5f482403e8e0026` and
`e18b302126b4a51b0b9829a54342602f1ecccfcff517bb20cb8fb90e1c762143`.

#### Evidence Quality

The repaired sentence matches the preceding review's requested wording and the handoff identifies
no broader change. Independent bounded checks pass: `docs_sync.py`, `check_ara.py` with all 40
claims, `check_agent_workflow.py`, and `git diff --check`. The preceding final review's focused 116
tests, full CPU pytest, canonical verification, source/audit hashes, contract/bundle checks,
byte-exact mirrors, report inventory, browser receipts, and live endpoints remain applicable
because no executable or evidence input changed.

#### Simplicity

The repair is the minimum documentation-only scope correction. It requires no producer rerun,
report regeneration, claim rewrite, or compensating code path.

#### Missing Cases

All limitations listed in the preceding review and canonical audit remain unchanged, including
missing typed rollback/AABB receipts, dirty development-source scope, capped-field proxies,
host-local timing, and the absence of visual-quality, physical-geometry, independent-half-accuracy,
complete-field, GPU/real-time, memory, cross-scene, GPS-Gaussian, globally coupled OT, or default
evidence.

#### Required Changes

None.

#### Optional Improvements

The future evidence-hardening suggestions in the preceding review remain optional and are outside
this bounded documentation repair.
