# Current Task

## Title

Execute and audit the fixed-anchor compact-field sweep

## Task ID

RTGS-003

## Role Assignment

- Driver: Codex-field-sweep-runner
- Reviewer: Codex-protocol-reviewer
- Turn: driver

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
- No protected outcome for this task has been opened.

## Minimal Plan

1. Checkpoint this execution task and initialize the official run from clean source.
2. Execute the exact frozen command while preserving progress and failure receipts.
3. Hand raw producer outputs to the distinct results reviewer and apply only audit corrections.
4. Log the bounded disposition, generate/smoke-test visuals and viewer, and gate the bundle.
5. Review, verify, archive RTGS-003 at the reached maturity, and hand off without deleting refs.

## Status

In progress

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

Append Driver handoffs, Reviewer verdicts, and session-completion entries in chronological order.
Use `###` for entries and `####` for their fields so entries remain nested below this section.
Never delete earlier entries. On terminal closeout, archive the complete record as
`docs/tasks/<task-id>-<slug>.md`, change the archived `Turn` to `none`, and reset this file to the
unchanged template.
