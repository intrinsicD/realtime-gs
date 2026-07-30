# Current Task

## Title

Additive-field analytic 2D-to-3D refinement objective: registration, additive teachers, and
matched comparison against point-sampled supervision

## Task ID

RTGS-006

## Role Assignment

- Driver: Codex
- Reviewer: Codex
- Turn: driver

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

Before result-bearing implementation continues, the owner also adopted A17, Experiment Bundle
Contract v2, and explicitly transferred the driver turn to Codex. This task is its first consumer:
the shared reporting prerequisite is implemented without changing the frozen scientific question,
arms, data boundary, metrics, or evidence phase.

## Success Criteria

- The experiment task JSON is registered as `draft`, passes
  `scripts/experiment_contract.py validate`, and names every open blocker honestly
  (additive bundles and seal, implementation, config freeze, prospective review).
- Additive native 2D bundles for both protocol frames exist under `dataset/`, produced by
  deterministic mask-aware stage-1 native fitting with recorded config and seed; the additive
  data seal is built with `build-seal` and swapped into the task before review.
- The surfel-constrained visibility-weighted analytic objective, the hybrid schedule, and the
  matched point-sampled budget rule are implemented behind existing seams with CPU tests
  covering analytic-equals-rendered-image exactness on additive fields, occlusion-weighting
  behavior, and determinism.
- A distinct prospective reviewer approves the protocol digest without outcome access; the
  run executes under the one canonical run root; the bundle passes `check-run` and
  `scripts/check_results_bundle.py`; the outcome is logged in `docs/EXPERIMENTS.md` and
  audited before any claim or default motion.
- Before that run is initialized, report template v2 is CPU-contracted and documented: it accepts
  dimensioned fitting histories, generates `index.html`, a run-local `README.md`, and a
  checksummed artifact manifest from shared machine records, presents exact reproduction,
  report-serving, and orbit-viewer commands, and fail-closes missing or inconsistent links while
  grandfathering template-v1 runs.
- `./scripts/verify.sh` passes at every commit.

## Constraints

- Hard Rules 7-9: task-first registration before result-bearing driver code; no run before an
  approved prospective review; reconstruction and evaluation stay image-free per the
  `direct_compact` arm; decision metrics come only from the exact selected rasterizer
  semantics on the frozen split.
- The incumbent point-sampled compact trainer remains the untouched baseline; no behavior
  changes to StructSplat or normalized-blend paths.
- No edits to other registered experiment tasks, their reviews, or their run roots.
- Report-template work remains in the shared `scripts/experiment_contract.py` and bundle checker;
  no second renderer or task-specific report implementation is permitted.

## Non-Goals

- StructSplat or normalized-blend upgrades, and any exact-quotient (cross-multiplied)
  normalized loss — deferred by owner decision until the additive path is shown to work.
- Beam-fusion or carrier-path changes, GPU work, performance claims, cross-dataset
  generalization, topology/densification changes, or production-default changes.
- Promoting any quantitative statement without the results audit and ARA gates.
- Migrating or rewriting historical run bundles; template-v1 evidence remains immutable and
  grandfathered.

## Selected Skills

- rtgs-core
- rtgs-task-workflow
- rtgs-experiment
- rtgs-review
- rtgs-docs-sync
- rtgs-verify
- realtime-gs-results-audit

## Experiment Contract

experiments/tasks/20260730_additive_analytic_objective_stage_frames00008_00009.json

## Current Evidence

- Registration-only; no runs and no result numbers. Design basis verified in source:
  exact-vs-proxy semantics (`field_loss.py`, `field_lifter.py` `analytic_semantics`),
  detached transmittance visibility (`field_visibility.py`), incidence gating and thin-axis
  surfels (`surfel_lift.py`, rho 0.1, 70-degree gate), the additive legacy conversion seam
  (`data/field_inputs.py`), and the point-sampled baseline loss
  (`optim/compact_trainer.py:1672`).
- Empirical cautions from `docs/EXPERIMENTS.md` motivating the protocol shape: 2026-07-17
  (latent hard-min correspondence settles in stable wrong basins) and 2026-07-20
  (a correspondence-free consensus objective improved itself while distance-to-truth
  worsened). Decisions therefore bind to held-out compact metrics, never to the training
  objective, and the occlusion story gets an explicit unweighted-analytic control arm.
- A17 is now implemented in the shared contract and checker. New tasks explicitly freeze v2;
  five named historical task ids remain on v1. V2 validates dimensioned non-heldout fitting
  histories, complete parameters/environment/run receipts and exact commands; generates static
  SVG histories, `index.html`, `README.md`, and a self-excluding SHA-256 manifest; and distinguishes
  renderable failures from results-bearing completed bundles. Focused compatibility, leakage,
  link, command, failure, and tamper tests pass.

## Minimal Plan

1. Implement and verify the shared Bundle Contract v2 prerequisite while preserving v1 evidence.
   (complete)
2. Register the draft experiment task JSON with frozen splits, seeds, stages, comparators,
   metrics, resource protocol, and honest blockers. (complete)
3. Produce additive native bundles for both frames with deterministic stage-1 tooling; build
   and swap the additive data seal.
4. Implement the surfel/visibility-weighted analytic objective and hybrid schedule with CPU
   tests behind existing seams; freeze the provisional refinement configuration.
5. Write and freeze the task driver and `run_command`; obtain the distinct prospective
   review; set the task `ready`.
6. `init-run`, execute and record the RESULT, obtain the distinct results audit, render and gate
   the final bundle, and log the audited outcome before any claim motion.

## Status

In progress

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

### Question

May Experiment Bundle Contract v2 be added as a prerequisite within RTGS-006, with the driver
turn transferred to this Codex session while the additive-objective protocol remains unchanged?

### Options

Fold the shared reporting prerequisite into RTGS-006 and transfer the driver; defer the contract
until RTGS-006 closes; silently broaden or overwrite the task without a durable decision.

### Recommendation

Fold it in explicitly: RTGS-006 is still draft and will be the first consumer, so the contract can
be CPU-contracted before any protocol review or protected run without changing scientific scope.

### Decision

(Owner, in chat.) Yes. Add Bundle Contract v2 as an explicit prerequisite, transfer the driver
turn to Codex, and preserve the additive-objective protocol.

### Date

2026-07-30

## Handoff Log

Append Driver handoffs, Reviewer verdicts, and session-completion entries in chronological order.
Use `###` for entries and `####` for their fields so entries remain nested below this section.
Never delete earlier entries. On terminal closeout, archive the complete record as
`docs/tasks/<task-id>-<slug>.md`, change the archived `Turn` to `none`, and reset this file to the
unchanged template.

### Handoff (human-directed driver transfer)

#### Objective

Transfer the active driver turn to Codex and insert the owner-adopted Bundle Contract v2 as a
pre-run prerequisite without changing the additive-objective experiment.

#### Reviewed state

`origin/main` plus the current ARA-only adoption record for N181/N182, O148, and A17. RTGS-006 and
its experiment task are registration-only, `draft`, prospectively unreviewed, and have no run root
or result numbers.

#### Changes

The owner explicitly transferred the driver turn. Administrative ownership now names Codex, and
the task success criteria and plan name the shared v2 reporting prerequisite.

#### Evidence

Owner approval in chat; A17 in `ara/logic/solution/architecture.md`; the existing v1 contract in
`experiments/README.md`, `experiments/templates/metrics.json`,
`scripts/experiment_contract.py`, and `scripts/check_results_bundle.py`.

#### Assumptions

The reporting prerequisite changes no scientific arm, data path, split, seed, primary metric,
guard, evidence phase, or result interpretation.

#### Uncertainties

No explicit handoff was available from the former driver beyond the durable registration record;
all scientific implementation and additive-data blockers remain open.

#### Review Focus

Backward compatibility for v1 evidence, fail-closed v2 schemas and links, deterministic
CPU-only rendering, and no held-out metric leakage through fitting curves.

#### Protected actions not taken

No additive data generation, prospective review, `init-run`, protected execution, result audit,
claim motion, commit, or push.

#### Recommended Next Action

Implement and verify Bundle Contract v2, then resume the additive-objective plan at additive
teacher generation.

### Handoff (Bundle Contract v2 prerequisite implemented)

#### Objective

Implement the owner-adopted experiment naming/output contract before RTGS-006 becomes
result-bearing, while preserving all historical v1 evidence and the scientific protocol.

#### Reviewed state

`origin/main` plus implementation diff SHA-256
`a95c28ab28ef17c8d9f64fe18c679780aa878cd7e5f9d8f62fec1e6ca1dc67e2`, computed with
`.agents/state/current-task.md` excluded to avoid a self-referential digest.

#### Changes

New tasks must explicitly freeze report template v2; five existing task ids are the complete v1
grandfather set. The shared contract now validates dimensioned non-heldout fitting histories,
effective configuration, environment and run receipts, completed/failed state, and exact
reproduce/report-server/viewer commands. It generates static SVG fitting charts, final metrics,
full parameters and provenance into `index.html` and `README.md`, then inventories every other
run file plus declared evidence in a SHA-256 manifest. The independent bundle gate verifies the
same links/checksums and refuses to classify a rendered failure as results-bearing. Templates,
repository/skill documentation, A17, and the RTGS-006 task version were updated in the same diff.

#### Evidence

`./scripts/verify.sh` passed: Ruff lint/format, the full CPU pytest suite, docs-sync, ARA, script
layout, agent workflow, and experiment-contract gates. Focused contract tests cover v2 output,
static fitting charts/stage markers, v1 compatibility, heldout-history rejection, canonical
server/viewer commands, missing generated files, checksum tampering, and failure-report behavior.
`git diff --check` is clean.

#### Assumptions

The standardized report server runs from the repository root on port 8765. `manifest.json`
cannot checksum itself, so it is linked from both generated documents and inventories every other
run-local file. A distinct raw-results audit writes the canonical AUDIT records before the final
report render; smoke receipts are added by rendering once, exercising the report/viewer, then
rendering again.

#### Uncertainties

The prerequisite has only driver self-review, not independent acceptance. Its generated report
was exercised through deterministic CPU fixtures and link parsing, not a browser GUI. No real
RTGS-006 data, driver, protected run, viewer smoke, or scientific result exists yet.

#### Review Focus

Challenge the explicit v1 allowlist, v2 failure semantics, tidy-history leakage guard, exact
command contract, generated Markdown/HTML escaping, self-excluding manifest completeness, and
tamper/link checks. Confirm that the audit-before-final-render sequence matches the independent
results-audit workflow.

#### Protected actions not taken

No additive teacher generation, data-seal rewrite, scientific implementation, prospective
review, `init-run`, protected execution, result/claim promotion, commit, or push.

#### Recommended Next Action

Obtain an independent review of the v2 prerequisite. After any required corrections, resume
RTGS-006 at deterministic additive teacher generation and data-seal construction; keep the task
`draft` until all scientific blockers and the prospective protocol review are closed.
