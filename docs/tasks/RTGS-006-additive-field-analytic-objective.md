# Current Task

## Title

Additive-field analytic 2D-to-3D refinement objective: registration, additive teachers, and
matched comparison against point-sampled supervision

## Task ID

RTGS-006

## Role Assignment

- Driver: Claude-additive-objective-driver
- Reviewer: Claude-additive-objective-driver
- Turn: none

## Mode

Implement

## Risk

Protected

## Maturity

- Target: Calibrated
- Reached: Scaffolded

## Goal

Test whether an exact analytic objective on native additive 2D Gaussian fields — closed-form
density/RGB-numerator matching of projected 3D Gaussians under surfel-constrained covariances
with detached transmittance-times-incidence per-view weighting — improves image-free 3D
refinement over the incumbent point-sampled compositing objective, and whether an
analytic-warmup-then-sampled-finish hybrid beats both, at matched initialization, topology,
and budget. The protocol authority is
`experiments/tasks/20260730_additive_analytic_objective_stage_frames00008_00009.json`; this
record coordinates the work and does not duplicate its arms, seeds, gates, or command.

## Motivation

Owner decisions (2026-07-30, in chat, recorded below): build on additive native fields only,
because for `blend_mode="additive"` with zero support fade the analytic density/RGB loss is
exact rather than the documented normalized-blend proxy (`src/rtgs/lift/field_loss.py`
docstring; the `analytic_semantics` branch in `src/rtgs/lift/field_lifter.py`), and the
priority is a working 2D-to-3D pipeline. The occlusion mismatch of additive matching is
addressed with detached center-transmittance weights (`src/rtgs/lift/field_visibility.py`)
combined with surfel incidence weighting in the spirit of ADR-XXXX (`surfel_lift.py`), and
diagnosed with a per-primitive conflict-score table.

## Success Criteria

- The experiment task JSON is registered as `draft`, passes
  `scripts/experiment_contract.py validate`, and names every remaining open blocker honestly
  (objective implementation/config freeze, task driver, and prospective review).
- Additive native 2D bundles for both protocol frames exist under `dataset/`, produced by seeded,
  mask-aware, source/binary-bound stage-1 native fitting with one fresh process per view and
  recorded effective config and seed. The exact realized bytes are sealed; because the native
  CUDA renderer uses atomic accumulation, cross-run bit identity is explicitly not claimed. The
  additive data seal is built with `seal-data` and swapped into the task before review.
- The surfel-constrained visibility-weighted analytic objective, the hybrid schedule, and the
  matched point-sampled budget rule are implemented behind existing seams with CPU tests
  covering analytic-equals-rendered-image exactness on additive fields, occlusion-weighting
  behavior, and determinism.
- A distinct prospective reviewer approves the protocol digest without outcome access; the
  run executes under the one canonical run root; the bundle passes `check-run` and
  `scripts/check_results_bundle.py`; the outcome is logged in `docs/EXPERIMENTS.md` and
  audited before any claim or default motion.
- `./scripts/verify.sh` passes at every commit.

## Constraints

- Hard Rules 7-9: task-first registration before result-bearing driver code; no run before an
  approved prospective review; reconstruction and evaluation stay image-free per the
  `direct_compact` arm; decision metrics come only from the exact selected rasterizer
  semantics on the frozen split.
- The incumbent point-sampled compact trainer remains the untouched baseline; no behavior
  changes to StructSplat or normalized-blend paths.
- No edits to other registered experiment tasks, their reviews, or their run roots.

## Non-Goals

- StructSplat or normalized-blend upgrades, and any exact-quotient (cross-multiplied)
  normalized loss — deferred by owner decision until the additive path is shown to work.
- Beam-fusion or carrier-path changes, GPU work, performance claims, cross-dataset
  generalization, topology/densification changes, or production-default changes.
- Promoting any quantitative statement without the results audit and ARA gates.

## Selected Skills

- rtgs-core
- rtgs-task-workflow
- rtgs-experiment
- rtgs-review
- realtime-gs-results-audit

## Experiment Contract

experiments/tasks/20260730_additive_analytic_objective_stage_frames00008_00009.json

## Current Evidence

- No protected 3D refinement run and no objective-comparison result numbers. Design basis
  verified in source: exact-vs-proxy semantics (`field_loss.py`, `field_lifter.py`
  `analytic_semantics`), detached transmittance visibility (`field_visibility.py`), incidence
  gating and thin-axis surfels (`surfel_lift.py`, rho 0.1, 70-degree gate), the additive legacy
  conversion seam (`data/field_inputs.py`), and the point-sampled baseline loss
  (`optim/compact_trainer.py:1672`).
- Empirical cautions from `docs/EXPERIMENTS.md` motivating the protocol shape: 2026-07-17
  (latent hard-min correspondence settles in stable wrong basins) and 2026-07-20
  (a correspondence-free consensus objective improved itself while distance-to-truth
  worsened). Decisions therefore bind to held-out compact metrics, never to the training
  objective, and the occlusion story gets an explicit unweighted-analytic control arm.
- Blockers 1+2 are closed: 52/52 full-resolution native-additive compact views strictly reload
  across frames 00008/00009 (640 components each; 33,280 total), with per-view source/config/seed,
  fresh-process, CUDA-binary, output, and runtime receipts copied into the two sealed
  `production_manifest.json` sidecars. The frame manifests are
  `fa1240c8...ae2c6` / `2c9c0108...6882`; the production manifests are
  `ee8c9e5b...d1046` / `bf1f0d0f...70c7`.
- `experiments/data/stage_frames00008_00009_additive.json` binds 57 files and 2,044,282
  selected bytes; `experiment_contract.py validate-data` and the full contract validator pass,
  and the seal contains no direct RGB or mask file. Two fresh C0001 fit replays with identical
  inputs/config/seed produced distinct tensor hashes (`cd8439...aea1` / `c1aae5...b8ec`), so
  only the sealed realized fields—not bit-exact CUDA regeneration—are evidence.

## Minimal Plan

1. Register the draft experiment task JSON with frozen splits, seeds, stages, comparators,
   metrics, resource protocol, and honest blockers. (this change)
2. Produce additive native bundles for both frames with seeded, per-view-isolated stage-1
   tooling; bind the CUDA atomic replay boundary, then build and swap the additive data seal.
   (Complete: 52/52 views and additive seal pass strict verification.)
3. Implement the surfel/visibility-weighted analytic objective and hybrid schedule with CPU
   tests behind existing seams; freeze the provisional refinement configuration.
4. Write and freeze the task driver and `run_command`; obtain the distinct prospective
   review; set the task `ready`.
5. `init-run`, execute, render and gate the bundle, log the outcome, and run the results
   audit before any claim motion.

## Status

Superseded

## Human Decisions

### Question

Which 2D-field semantics does the analytic-objective experiment build on, given the
documented normalized-blend proxy gap?

### Options

Additive native fields only; keep StructSplat normalized fields under the density/numerator
proxy; implement the exact cross-multiplied quotient loss for normalized fields now.

### Recommendation

Additive-only: the analytic loss becomes exact by construction, the experiment isolates the
2D-to-3D objective question, and the normalized handling stays available later because
`blend_mode` is stored per field artifact.

### Decision

(Owner, in chat.) Use additive native fields only, not StructSplat. Everything potentially
given up by that choice (StructSplat stage-1 quality, exact handling of the normalized mode)
is addressed later, after the approach is shown to work. The priority is that the 2D-to-3D
pipeline works.

### Date

2026-07-30

## Handoff Log

Append Driver handoffs, Reviewer verdicts, and session-completion entries in chronological order.
Use `###` for entries and `####` for their fields so entries remain nested below this section.
Never delete earlier entries. On terminal closeout, archive the complete record as
`docs/tasks/<task-id>-<slug>.md`, change the archived `Turn` to `none`, and reset this file to the
unchanged template.

### Handoff — 2026-07-30 additive bundle production and seal

#### Objective

Close experiment blockers 1+2 only: produce native-additive compact inputs for every frozen
frame-00008/00009 train and held-out view, bind their acquisition provenance, build the additive
data seal, and leave the protected refinement run unopened.

#### Reviewed state

Base commit `44be09f5e9fdfa50e376c383fb7895cbfbea18d7`. The implementation, protocol, docs,
tests, seal, and both additive data trees—excluding this self-referential coordination
record—have aggregate content digest
`be845dbce9fce8aa91adde79da016fc494e56e202a3a5f928091fc15c173f407`.
Production workers bind source aggregate
`2a37b336e9a6325ce02affe646557cc75ae00ae315ea9b47762d56b2663ae2b5`.
The unrelated user-owned `.idea/rtgs.iml` modification was excluded and untouched.

#### Changes

- Added the native `Gaussians2D` → additive `GaussianObservationField` adapter with exact
  crop-local float32 mean recovery, explicit `provider="native"`, source/config provenance, and
  CPU round-trip/covariance tests.
- Added a task-bound, non-overwriting producer that loads one named calibrated RGB/mask pair per
  fresh worker, executes the frozen full-resolution 640×100 native-CUDA fit, saves and strictly
  reloads one capped `.rtgsv`, and publishes frame/production manifests only after all 26 views
  pass.
- Produced 52/52 compact views and both provenance sidecars, extended `seal-data` to bind an
  optional adjacent `production_manifest`, created the additive seal, repointed the draft task,
  and removed only the first two blockers. The linked 20260728 VRAM task was not edited.
- Documented the provider/producer/seal seam and the CUDA atomic replay boundary.

#### Evidence

- Bundle verifier: PASS for 26 views / 16,640 Gaussians / 819,438 bytes on frame 00008 and
  26 / 16,640 / 829,385 on frame 00009; frame manifest SHA-256s
  `fa1240c8...ae2c6`, `2c9c0108...6882`.
- `experiment_contract.py validate-data` and `validate`: PASS. The additive seal binds 57 files
  and 2,044,282 selected bytes, includes both production manifests, and directly binds no
  `/rgb/` or `/mask/` file.
- Focused native-observation/compact/field/contract tests: PASS. Native CUDA renderer parity:
  8 passed. Full CPU suite including slow tests: PASS with exactly six historical ABI nodes
  deselected.
- Ruff check/format, docs-sync, ARA, script-layout, agent-workflow, experiment-contract,
  additive-data validation, bundle verification, and `git diff --check`: PASS.
- Canonical `./scripts/verify.sh` reached pytest but is not green on this host: the six
  historical benchmark tests hard-bind missing
  `/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33`; the host now has only `6.0.35`. An untouched
  `git archive HEAD` reproduces the identical six failures, so no historical protocol or system
  library was mutated to hide the environmental drift.

#### Assumptions

The user's 20260728 link was treated as the existing frame/split/data context. The matching active
RTGS-006 protocol remains the authority for additive bundles, consistent with its explicit
constraint not to edit other registered experiment tasks.

#### Uncertainties

The native CUDA renderer uses atomic accumulation. Inputs, order, config, seeds, source, compiled
extension, and exact output bytes are bound, but bit-exact regeneration is not: two fresh C0001
replays produced different tensor hashes. The 100-step Stage-1 setting is an input-production
choice and carries no RGB-quality/default claim. The canonical full gate remains environmentally
red until the historical `6.0.33` ABI is available or a separately scoped owner decision repairs
those old host-bound tests without rewriting their evidence.

#### Review Focus

Check crop-local/native-coordinate recovery, provider semantics, per-view source isolation,
receipt-to-manifest consistency, optional production-sidecar sealing, and that only blockers 1+2
were removed.

#### Protected actions not taken

No `review-digest`, prospective approval, `init-run`, objective comparison, held-out result
inspection, claim/default motion, commit, branch operation, or edit to the 20260728 VRAM task was
performed.

#### Recommended Next Action

Implement and CPU-test the surfel-constrained visibility-weighted analytic objective, hybrid
schedule, and matched point-sampled budget rule (remaining blocker 3), then freeze the provisional
refinement configuration before writing the official driver or seeking prospective review.

### Closeout — 2026-07-30 owner supersession

#### Objective

Preserve the completed native-additive adapter, production tooling, 640-by-100 smoke bundles, and
data seal while stopping the analytic-objective branch before it displaced the three paper paths
the owner actually asked to inspect.

#### Reviewed state

The owner clarified in chat that the immediate target is a full-resolution GaussianImage-style
2D capture followed by three compact-field-supervised 3DGS fits from bounded-random, SfM, and Beam
Fusion initialization, all with full clone/split/prune densification. The active RTGS-006 task
instead prohibited topology changes and proposed a 128-track, 40-step, fixed-topology objective
comparison.

#### Changes

RTGS-006 is closed as Superseded without implementing its analytic objective, writing a protected
driver, obtaining a prospective review, or starting an official run. Its 52 additive bundles and
seal remain preserved as explicitly low-capacity mechanism-smoke artifacts. Replacement work is
owned by RTGS-007.

#### Evidence

Direct source inspection confirmed that the 640-by-100 artifacts contain only per-view 2D fields,
that no RTGS-006 3D run command exists, and that the requested compact clone/split/prune controller
is not yet a production component. The owner rejected the visible quality of the smoke artifacts
and explicitly supplied the replacement pipeline.

#### Assumptions

None about result quality or task completion. Supersession is an owner decision, not a scientific
verdict on the unimplemented analytic objective.

#### Uncertainties

The preserved native CUDA fields are not cross-run bit-exact because of atomic accumulation. They
must not be reused as the paper-quality 2D captures merely because their canvas metadata is native
resolution.

#### Review Focus

No review requested. Verify only that the replacement task does not relabel the old bundles as
paper-quality evidence and that it keeps the no-StructSplat and post-fit no-image boundaries.

#### Protected actions not taken

No objective implementation, protected comparison, result interpretation, claim/default motion,
commit, branch operation, or deletion of prior artifacts.

#### Recommended Next Action

Execute RTGS-007: qualify a high-capacity native-additive Stage 1, implement production compact
density control, and run the three matched initialization paths.
