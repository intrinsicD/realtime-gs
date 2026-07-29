# Current Task

## Title

Repair native-scale field sweep precision and execute a successor task

## Task ID

RTGS-004

## Role Assignment

- Driver: Codex-field-sweep-f64-driver
- Reviewer: Codex-protocol-reviewer
- Turn: driver

## Mode

Stabilize

## Risk

Protected

## Maturity

- Target: Claim-ready
- Reached: Pipeline-integrated

## Goal

Add an opt-in float64 CPU compute path that preserves the fixed absolute source-projection gate at
native resolution, freeze and prospectively review a successor to the consumed field-sweep task,
then execute, independently audit, render, and smoke-test its complete result bundle.

## Motivation

The first official task failed before measurement because native-scale float32 projection
round-trip error exceeded its absolute invariant. The independent audit found the diagnosis
strongly consistent with one-to-three-ULP float32 round-off but not replay-complete causal proof.
A successor must fix precision without weakening the gate or mutating the consumed task.

## Success Criteria

- Add a default-preserving `input`/`float64` field-lift compute-dtype control and a native-scale
  regression that reproduces the float32 failure and passes the unchanged absolute gate in
  float64.
- Preserve compact observation semantics, mean residual precision, CPU-first imports, held-out
  isolation, fixed anchors, arm parity, and the production `compact_carve` default.
- Make invariant failures retain exact step, dtype, mean error, covariance error, and tolerance;
  make the successor worker retain a structured exception receipt.
- Register a new experiment task with the same data, splits, arms, seeds, metrics, gates, and
  resource protocol, changing only the shared compute dtype and failure observability.
- Obtain distinct prospective approval of the new digest without successor outcome access.
- From clean committed source, initialize and run the successor; obtain a distinct results audit
  before logging, rendering, viewer smoke, bundle gates, or interpretation.
- Pass focused tests, full CPU tests, docs sync, `./scripts/verify.sh`, canonical report checks,
  HTTP link smoke, and the exact viewer smoke.

## Constraints

- Do not change, resume, overwrite, or add outputs to the consumed 20260729 task except its
  already-preserved append-only failure record.
- Do not weaken the `0.0002` projection guardrail or tune arms, seeds, splits, gates, optimizer,
  evaluation, or decision rules after the failed warmup.
- Keep float64 opt-in; default field-lift computation must retain the input dtype.
- Reconstruction may consume calibration and compact fields only; no RGB, masks, packed alpha,
  `SceneData`, Beam/carrier path, dense trainer, or held-out fitting.
- Preserve failed successor attempts below its one canonical run root.
- Make no default, RGB-quality, physical-geometry, GPU, speed, topology, cross-dataset, or
  confirmatory claim.

## Non-Goals

- Repairing the historical raw-RGB branch or deleting Git refs.
- Changing camera calibration bytes, compact archives, the data seal, or teacher semantics.
- Introducing a general mixed-precision framework outside the field-lift boundary.
- Publishing commits or pushing branches without separate authorization.

## Selected Skills

- `rtgs-core`
- `rtgs-task-workflow`
- `rtgs-experiment`
- `realtime-gs-results-audit`
- `rtgs-review`
- `rtgs-verify`
- `research-manager`

## Experiment Contract

experiments/tasks/20260730_field_sweep_placement_f64_stage_frames00008_00009.json

## Current Evidence

- The consumed task's independent audit is
  `benchmarks/results/20260729_field_sweep_placement_stage_frames00008_00009_AUDIT.md`.
- It confirms zero completed warmups, measured cells, or held-out metrics.
- Recorded covariance differences were one to three float32 ULPs, maximum `0.0234375`, under a
  frozen `0.0002` absolute gate.
- The original task, run, result, and audit are immutable provenance.

## Minimal Plan

1. Implement opt-in float64 field computation and structured invariant failure receipts.
2. Add native-scale precision and successor-driver contract tests.
3. Freeze and validate the successor task, then obtain distinct prospective approval.
4. Commit the verified source, initialize once, and execute the exact successor command.
5. Commission the independent results audit; log only its bounded disposition.
6. Render and smoke-test the report/viewer, gate the bundle, verify, and close RTGS-004.

## Status

In progress

## Human Decisions

### Question

Should the requested experiment stop permanently after a pre-measurement precision failure, weaken
the frozen gate, or proceed through a separately reviewed successor?

### Options

- Stop with only the inconclusive failed attempt.
- Lower the source-projection gate and reuse the consumed task.
- Preserve the failure and run a new task after a default-preserving precision repair.

### Recommendation

Use a separately reviewed successor with opt-in float64 computation and the unchanged gate.

### Decision

The user authorized running the experiment. Repository policy forbids reusing the consumed task;
RTGS-004 takes the smallest claim-safe successor path within that authorization.

### Date

2026-07-30

## Handoff Log

Append Driver handoffs, Reviewer verdicts, and session-completion entries in chronological order.
Use `###` for entries and `####` for their fields so entries remain nested below this section.
Never delete earlier entries. On terminal closeout, archive the complete record as
`docs/tasks/<task-id>-<slug>.md`, change the archived `Turn` to `none`, and reset this file to the
unchanged template.

### Handoff (corrected successor protocol)

#### Objective

Independently review the opt-in float64 field-compute repair and exact successor protocol before
any successor run initialization or outcome access.

#### Reviewed state

Working-tree implementation plus successor protocol digest
`a45ee0da9f2282cdeebfa93a9321408a9d1a7ce4b64ba6a2746f61e30546a1e0`.

#### Changes

- Added opt-in float64 field computation while preserving the input-dtype default.
- Added native-scale precision regression and structured invariant/worker failure evidence.
- Added a successor task and thin driver that preserve the original split, seeds, arms, metrics,
  gates, and resource protocol.
- Removed the invalid executable dependency on the consumed, incomplete predecessor; its
  relationship remains narrative provenance only.

#### Evidence

- Ruff passes on every changed Python file.
- The 59 focused field-input, field-lifter, observation, and experiment-driver tests pass.
- `experiment_contract.py validate-data` and `review-digest` pass.
- The reviewer rejected the prior draft digest because it incorrectly encoded the consumed,
  incomplete predecessor as an executable dependency.

#### Assumptions

The smallest valid successor changes only common compute precision and failure observability; the
unchanged absolute projection gate remains the relevant invariant.

#### Uncertainties

The full native dataset run has not started. The regression establishes the repaired numerical
path but does not predict placement quality, resource use, or the frozen producer decision.

#### Review Focus

Confirm source/protocol parity, unchanged controls and guardrails, outcome isolation, successor
lifecycle validity, and whether the exact corrected digest is safe to mark `ready`.

#### Protected actions not taken

The successor was not initialized or executed, no successor output was inspected, no report was
rendered, no claim or default was changed, and no commit was pushed.

#### Recommended Next Action

If the prospective review accepts the exact digest, write the canonical review artifacts; then
return the task to the driver to record approval, verify, commit the source checkpoint, and
initialize the official run exactly once.

### Review (prospective protocol)

#### Verdict

Accepted

#### Self-reviewed

No

#### Correctness

The corrected successor protocol at digest
`a45ee0da9f2282cdeebfa93a9321408a9d1a7ce4b64ba6a2746f61e30546a1e0` is prospectively approved.
It has no invalid executable dependency on the consumed predecessor; data, splits, seeds, arms,
refit, metrics, gates, resources, held-out isolation, and no-image controls remain matched. The
only common compute change is opt-in float64, and the only added execution behavior is a structured
worker-failure receipt.

#### Evidence Quality

The reviewer independently reproduced the native-scale float32 failure and float64 pass against
the unchanged `0.0002` gate, verified crop-residual semantics and successor subprocess identity,
and passed the 59 focused tests, task/data/digest checks, Ruff, docs/ARA/script/workflow checks,
whitespace validation, and full `./scripts/verify.sh`. Outcome Access was `none`: the successor was
not initialized or executed and no successor result was inspected.

#### Simplicity

The implementation adds one default-preserving dtype control and a thin wrapper around the
consumed driver's unchanged orchestration and treatment logic. It avoids duplicating the
three-arm experiment or weakening the invariant.

#### Missing Cases

The official native-data outcomes, resource receipts, viewer/report bundle, and independent
results audit remain intentionally unavailable until after initialization and execution. This
prospective approval predicts none of those results and authorizes no default, RGB, geometry, GPU,
speed, topology, or generalization claim.

#### Required Changes

None to the reviewed protocol or implementation. The driver must record the same reviewer,
verdict, digest, and canonical Markdown artifact in the task, set it to `ready`, and revalidate
before initialization.

#### Optional Improvements

None before execution. The JSON review file is a structured companion only; the Markdown review
remains the canonical artifact bound by the experiment lock.
