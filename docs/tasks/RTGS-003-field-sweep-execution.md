# Current Task

## Title

Execute and audit the fixed-anchor compact-field sweep

## Task ID

RTGS-003

## Role Assignment

- Driver: Codex-field-sweep-runner
- Reviewer: Codex-results-auditor
- Turn: none

## Mode

Validate

## Risk

Protected

## Maturity

- Target: Claim-ready
- Reached: Pipeline-integrated

## Goal

Execute the exact prospectively approved
`20260729_field_sweep_placement_stage_frames00008_00009` task, preserve its complete calibrated
artifact bundle, obtain an independent results audit, and report only the bounded disposition
licensed by that audit.

## Motivation

The implementation task established a runnable, sealed compact-field placement comparison but
deliberately stopped before protected outcome access. The user has now explicitly authorized the
official run and wants its visual results.

## Success Criteria

- Initialize the one canonical run root from a clean tracked source state and execute the exact
  frozen argv without protocol-bearing edits.
- Preserve all warmup and measured cells, task/source/data locks, resource receipts, metrics,
  initial/final PLYs, previews, and append-only producer evidence.
- Have the distinct reviewer recompute the frozen gates from raw cells and write the canonical
  results-audit Markdown and JSON before any interpretation or results-page rendering.
- Append the bounded disposition to `docs/EXPERIMENTS.md` and ARA without changing a default.
- Generate and HTTP-smoke-test the canonical `index.html`, smoke-test the exact viewer command,
  and pass `check-run`, `check_results_bundle.py`, focused checks, and `./scripts/verify.sh`.

## Constraints

- The task JSON, review artifact, protocol digest, data seal, splits, seeds, arms, gates, and exact
  run command are immutable after initialization.
- Reconstruction may consume calibration and compact `.rtgsv` fields only; alpha, source RGB,
  masks, legacy `SceneData`, Beam/carrier code, and held-out fitting remain forbidden.
- Frames 00008 and 00009 are outcome-exposed development/replication data. No confirmatory,
  physical-geometry, RGB-quality, GPU, speed, cross-dataset, topology, or default claim is allowed.
- Preserve failed or partial attempts below the canonical run root; never overwrite artifacts or
  create sibling `_v2`, `_final`, timestamp, or `latest` roots.
- Producer output remains unaudited until the distinct results reviewer records a disposition.

## Non-Goals

- Editing the frozen experiment protocol or treatment implementation after outcome access.
- Running CUDA/GPU variants, adding scenes, tuning gates, or changing production defaults.
- Deleting the historical branch or publishing Git changes without separate authorization.

## Selected Skills

- `rtgs-core`
- `rtgs-task-workflow`
- `rtgs-experiment`
- `realtime-gs-results-audit`
- `rtgs-review`
- `rtgs-verify`
- `research-manager`

## Experiment Contract

experiments/tasks/20260729_field_sweep_placement_stage_frames00008_00009.json

## Current Evidence

- RTGS-002 reached Pipeline-integrated maturity and was independently accepted.
- The task status is `ready`; `experiment_contract validate`, `validate-data`, and
  `review-digest` reproduce the approved digest
  `9ff7057e47e3ea6a2af0532edbecf13036a892d9805e02cfe26aee7f33732844`.
- Commit `80eb9f94a9a9c569baeaea65ac2ceae01b45a962` is synchronized on `main` and
  `origin/main`; the canonical run root is absent.
- Official source commit `ec5735d5a549147f64490e57832578e72ae51400` initialized the
  canonical run and revalidated all 55 sealed files.
- The first discarded warmup failed at refit step 0 before any completed warmup, measured cell,
  or held-out metric.
- The independent audit verdict is
  `CONFIRMED FAILED BEFORE MEASURED OUTCOMES / POSTMORTEM NARROWED`.

## Minimal Plan

1. Completed: checkpointed clean source and initialized the official run.
2. Completed: preserved the first-warmup failure and stopped before measured outcomes.
3. Completed: obtained a distinct independent failure audit and bounded disposition.
4. Completed: logged the inconclusive attempt without rendering a false result page.
5. Completed: prepared append-only closeout; a repaired run belongs to a new task id.

## Status

Inconclusive

## Human Decisions

### Question

May the prospectively approved protected compact-field experiment be initialized and executed?

### Options

- Keep the task ready but unrun.
- Run the exact frozen official task.
- Replace the protocol with a new task before execution.

### Recommendation

Run the exact frozen task, preserve every artifact, and require the independent audit before
rendering or interpretation.

### Decision

The user explicitly instructed: “ok run it”.

### Date

2026-07-30

## Handoff Log

### Handoff (official attempt failed before measurement)

#### Objective

Execute the exact approved task and preserve a claim-safe result bundle.

#### Reviewed state

The canonical run is bound to clean commit `ec5735d5a549147f64490e57832578e72ae51400`.
It contains the task lock, two generic failed-worker markers, and
`attempts/attempt-001/failure.json`; no completed warmup or measured cell exists.

#### Changes

Initialized the official run, executed the frozen command, preserved the first-warmup failure,
wrote append-only RESULT Markdown/JSON, and logged the bounded failure in `docs/EXPERIMENTS.md`.

#### Evidence

The first warmup failed at refit step 0. The postmortem displayed maximum mean/covariance
round-trip errors of about `0.0009` and `0.0234` under a frozen `0.0002` absolute gate. No
held-out validation ran. The distinct audit independently bound source, task, review, seal,
chronology, artifact absence, and float32 ULP arithmetic.

#### Assumptions

None of the failed attempt's numeric placement hypotheses were evaluated.

#### Uncertainties

Float32 round-off is strongly consistent with the failure but not replay-complete causal proof
because raw diagnostic tensors and trace were not retained.

#### Review Focus

Verify the attempt is consumed, no held-out or measured outcome exists, the postmortem is not
overstated, and a successor cannot reuse the same task id or run root.

#### Protected actions not taken

No tolerance, task, review, seal, source, failed run artifact, default, or branch was overwritten.
No result page or viewer was presented for the incomplete attempt.

#### Recommended Next Action

Open a new protected task, add an opt-in float64 CPU compute path and native-scale regression,
obtain a new prospective review, and execute a new canonical run.

### Review (independent failure audit)

#### Verdict

Inconclusive

#### Self-reviewed

No

#### Correctness

The official attempt is consumed and conclusively failed before any completed warmup, measured
cell, or held-out metric. Locked control flow cannot reach held-out validation after the step-0
refit exception.

#### Evidence Quality

Task, protocol review, data seal, clean source, exact command, task lock, failure receipt, result
pair, run inventory, chronology, hashes, and float32 ULP calculations were independently checked.

#### Simplicity

The disposition preserves the failed run and requires a successor task instead of weakening a
gate or resuming under changed source.

#### Missing Cases

The diagnostic replay retained only rounded displayed matrices and a generic marker, not raw
tensors, traceback, command, environment, or a structured diagnostic receipt.

#### Required Changes

Do not rerun this task id. Do not claim a treatment, resource, or quality result from this attempt.

#### Optional Improvements

Record exact invariant values in future failure messages/receipts and use float64 for the
native-resolution CPU comparison while retaining the original absolute gate.
