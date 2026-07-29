# Current Task

## Title

Reimplement robust compact-field plane sweep as a controlled experiment

## Task ID

RTGS-002

## Role Assignment

- Driver: Codex-field-sweep-driver
- Reviewer: Codex-protocol-reviewer
- Turn: none

## Mode

Implement

## Risk

Protected

## Maturity

- Target: Pipeline-integrated
- Reached: Pipeline-integrated

## Goal

Preserve the useful discrete plane-sweep mechanism from commit `d5accb7` as a modern,
image-free compact-field placement treatment with a matched experiment driver on `main`.

## Motivation

The retired branch's implementation is incompatible with the current field-input boundary and
fails on CUDA, while the current compact-Carve placement does not implement its source-excluded
robust view trimming or coarse-to-fine search as an independently testable treatment. A bounded
modern implementation lets the old branch be archived without losing the research question.

## Success Criteria

- Add CPU-first, device/dtype-safe coarse-to-fine ray placement over compact observation fields.
- Exclude the source view in the robust arm and keep the lowest-cost frozen fraction of valid
  neighboring views without using component identity as supervision.
- Compare identical deterministic source anchors under bounded midpoint, all-view consensus
  sweep, and source-excluded robust sweep arms.
- Keep existing `FieldLifter` behavior and defaults unchanged.
- Add deterministic unit tests, a pipeline-path test, and a task-specific runnable experiment
  driver bound to one validated experiment contract.
- Pass focused tests, docs sync, the complete CPU verification gate, and the full CPU suite.

## Constraints

- Consume only `SceneFits` / `ReconstructionInputs` compact fields and calibration during
  reconstruction; held-out compact teachers are reporting-only.
- Use per-ray AABB intersections, clamp every coarse-to-fine update to the original interval,
  bound query memory, and preserve CPU-first imports.
- Do not repeat or promote the old branch's unbound quantitative claims.
- Do not delete the old branch until the replacement is verified and the user separately
  authorizes branch deletion.
- Do not execute an official results-bearing run before a distinct prospective protocol review.

## Non-Goals

- Changing the production field-placement default.
- Reintroducing RGB image sampling, legacy `SceneData`, ray-polish, or voxel merging.
- Claiming calibrated quality, speed, GPU parity, or a default-selection result in this task.
- Running the protected calibrated experiment or deleting Git refs in this implementation turn.

## Selected Skills

- `rtgs-core`
- `rtgs-task-workflow`
- `rtgs-experiment`
- `rtgs-review`
- `rtgs-docs-sync`
- `rtgs-verify`
- `realtime-gs-results-audit`
- `research-manager`

## Experiment Contract

experiments/tasks/20260729_field_sweep_placement_stage_frames00008_00009.json

## Current Evidence

- Commit `d5accb7` contains the historical image-backed cost-volume prototype.
- The branch audit found seven merge conflicts, obsolete APIs, a direct CUDA device error, and
  prose/result disagreement, so it is a code donor rather than mergeable evidence.
- Current compact-Carve performs one bounded all-view field sweep, but it includes the source
  view and does not implement source-excluded robust trimming or coarse-to-fine refinement as a
  separately selectable treatment.
- `rtgs.lift.field_sweep` now provides fixed midpoint, all-view consensus, and source-excluded
  robust coarse-to-fine arms behind an unchanged `compact_carve` default.
- Synthetic tests cover matched anchors, bounded depth, determinism, source exclusion, exact
  reprojection, post-refit held-out evaluation order, and held-out reconstruction invariance.
- The task-specific driver uses fresh single-thread workers, exact seal rehashing, locked
  provenance, `load_alpha=False`, live image/legacy/Beam denials, scoped resource receipts, and
  atomic per-cell artifacts.
- Distinct prospective review approved protocol digest
  `9ff7057e47e3ea6a2af0532edbecf13036a892d9805e02cfe26aee7f33732844` without outcome access.
- `./scripts/verify.sh` passes the complete CPU/local gate after review; no protected experiment
  was initialized or run.

## Minimal Plan

1. Completed: froze the three-arm protocol and deterministic compact-field input boundary.
2. Completed: implemented the shared fixed-anchor sweep seam and opt-in placement modes.
3. Completed: added numerical, leakage, determinism, ordering, and pipeline-path tests.
4. Completed: added and statically exercised the task-specific guarded runner without outcomes.
5. Completed: obtained distinct prospective approval and passed the complete verification gate.

## Status

Accepted

## Human Decisions

Record escalated questions and dated answers here. An answer that exists only in chat is not
durable task state. Use one block per decision:

```markdown
### Question
### Options
### Recommendation
### Decision
### Date
```

### Question

Should the incompatible historical branch be merged, dropped outright, or preserved through a
modern experiment on `main`?

### Options

- Merge the historical commit.
- Delete the branch without retaining its mechanism.
- Reimplement the mechanism through the current compact-field boundary, then archive/delete the
  branch after verification.

### Recommendation

Reimplement only the discrete sweep idea with current safety and evidence contracts; preserve the
old commit as archival provenance and remove the active branch only after explicit authorization.

### Decision

On 2026-07-29, the user selected the modern experiment path and deferred branch deletion until
after the replacement exists.

### Date

2026-07-29

## Handoff Log

### Handoff (implementation and protocol ready)

#### Objective

Replace the historical image-backed cost-volume prototype with a modern, bounded compact-field
experiment on `main`, without changing the production placement default or consuming outcomes.

#### Reviewed state

The final protocol is `ready`, has no blockers, and is bound to prospective-review digest
`9ff7057e47e3ea6a2af0532edbecf13036a892d9805e02cfe26aee7f33732844`. The worktree contains the
implementation, guarded task driver, task/review artifacts, tests, and synchronized design and
architecture documentation. The protected run directory does not exist.

#### Changes

Added a CPU fixed-anchor sweep with midpoint, all-view consensus, and source-excluded robust
coarse-to-fine modes; integrated opt-in `FieldLifter` placement selection and retained initial
semantic evaluation; removed runtime-only legacy annotation imports; froze and reviewed a
two-scene, three-seed matched experiment; and added live input guards, seal/lock checks, scoped
resource receipts, artifact publication, and producer aggregation.

#### Evidence

Fifteen focused field/driver tests pass. The live supported import route records five negative
control denials and no forbidden import/open attempt. The in-driver seal verifier rehashed 55
files and 8,373,380 bytes. `experiment_contract` validation, data validation, exact review digest,
Ruff, docs sync, ARA, script layout, agent workflow, and the full non-slow CPU test suite all pass
through `./scripts/verify.sh`.

#### Assumptions

Compact `GaussianObservationField` queries are the appropriate modern substitute for the old
prototype's image samples. Development and replication frames are outcome-exposed and therefore
support only the frozen development claim boundary. Packed compact alpha is disabled.

#### Uncertainties

The protected experiment has not run, so no placement benefit, quality, resource, or default
claim exists. GPU behavior and cross-dataset generalization remain untested. The remote historical
branch remains present until the user separately authorizes deletion.

#### Review Focus

Check fixed-anchor parity, strict source exclusion, original-interval bounds, post-fit held-out
access, seal and task-lock drift rejection, end-to-end resource scope, default preservation, and
the absence of RGB/mask/legacy/Beam access.

#### Protected actions not taken

No official run was initialized or executed, no protected RGB or mask outcome was opened, no
result or default was promoted, and no branch, commit, push, or remote ref was changed.

#### Recommended Next Action

Commit the verified source on `main`; then initialize and execute the exact frozen command from a
clean source state. Run the independent results audit before rendering or interpreting results.
Delete the historical branch only after the user confirms the replacement is sufficient.

### Review (distinct prospective and implementation-contract review)

#### Verdict

Accepted

#### Self-reviewed

No

#### Correctness

The reviewer first rejected held-out ordering, resource-scope, seal-runtime, and packed-alpha
ambiguities. After correction, the reviewer confirmed that held-out validation is post-fit,
measured resources come from the scoped worker receipt, task-lock and current data bytes are both
rehash-bound, alpha is disabled, source exclusion is explicit, and matched-arm parity fails closed.

#### Evidence Quality

The reviewer reran experiment/data/digest validation, focused field and driver tests, Ruff, and
script-layout checks without initializing a run or accessing outcomes. The driver then reran the
complete repository verification gate after the approval metadata landed.

#### Simplicity

One opt-in placement module reuses the existing compact query and fiber/refit seams. The default
path is unchanged, the experiment owns one frozen driver, and runtime guards/receipts are local to
that driver rather than adding a second global execution framework.

#### Missing Cases

No calibrated experiment outcome, independent results audit, GPU execution, or cross-dataset
confirmation exists. Those are deliberately outside this implementation task and its claim
boundary.

#### Required Changes

None for the pipeline-integrated, prospectively reviewed experiment handoff.

#### Optional Improvements

After a clean commit, run the frozen task, preserve every cell, commission the independent results
audit, and only then decide whether the mechanism or historical branch should be retired.
