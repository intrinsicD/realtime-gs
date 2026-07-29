# Prospective Protocol Review

- Task ID: `20260730_field_sweep_placement_f64_stage_frames00008_00009`
- Protocol SHA-256: `a45ee0da9f2282cdeebfa93a9321408a9d1a7ce4b64ba6a2746f61e30546a1e0`
- Reviewer: `Codex-protocol-reviewer`
- Verdict: `approved`
- Outcome Access: `none`

## Scope

This protocol may establish development and replication evidence about whether the named
source-excluded robust compact-field sweep improves placement and post-refit held-out compact-field
error over matched bounded-midpoint and all-view-consensus arms on outcome-exposed frames 00008 and
00009. It is a separately identified precision-repaired successor to the consumed 20260729 task.
It cannot establish RGB image quality, physical geometry accuracy, GPU or production performance,
cross-dataset generalization, topology utility, or a default change.

## Checks

- Confirmed the predecessor is consumed and inconclusive: its first discarded warmup failed before
  measurement, with zero completed warmups, measured cells, or held-out outcomes. Its audit
  characterizes the recorded discrepancy as strongly consistent with native-scale float32
  round-off, not replay-complete causal proof.
- Confirmed the successor has no executable `depends_on` edge. This is required because
  `init-run` accepts only complete canonical dependencies, while the failed predecessor remains
  narrative provenance and its task, run, result, and audit remain immutable.
- Recomputed the exact prospective digest above, validated the task graph, and validated the
  unchanged compact data seal. The datasets, data roles, selected bytes, train/held-out splits,
  input policy, three paired measured seeds, metrics, required charts, and resource protocol are
  identical to the approved predecessor protocol.
- Confirmed the frozen arms, fixed-anchor selection, compact loading, refit optimizer, topology
  setting, evaluation, producer decision rule, stopping behavior, and `0.0002` source-projection
  guardrail are unchanged. The only common treatment change is
  `field_lift.compute_dtype="float64"`; the only added execution guard is
  `structured_worker_failure_receipt`.
- Confirmed `FieldLiftConfig.compute_dtype` remains opt-in with default `"input"`, so incoming
  dtype and the production `compact_carve` default are unchanged. The float64 mode promotes the
  compact field inputs used by placement, refit, correspondence, and validation without mutating
  the source observations.
- Reproduced the native-scale synthetic counterexample: input/float32 computation fails at step
  zero with covariance error `0.015625` against the unchanged `0.0002` invariant, while float64
  computation completes with source-projection error approximately `7.28e-12`, also passing the
  unchanged task gate.
- Confirmed crop-local `mean_residuals` remain archived float32 corrections, are cast only when
  applied, and reconstruct the same corrected native means after field promotion. Cameras retain
  their stored calibration dtype and cast their operands into the active tensor computation.
- Confirmed only the deterministic training subset enters placement and refit. Held-out teachers
  remain ordered and reporting-only until both refit and topology complete; the existing
  held-out-perturbation and integration tests remain green.
- Confirmed the thin successor wrapper reinstalls the successor task, task path, run root,
  float64 field configuration, and wrapped worker in every process. The generated worker argv
  resolves to the successor wrapper, not the consumed driver.
- Confirmed worker exceptions retain task/cell identity, timestamp, exception type and message,
  and full traceback in a non-overwriting `failure.json`, then re-raise so orchestration remains
  fail-closed. Successful execution continues to use the predecessor's unchanged atomic
  publication, lineage checks, aggregation, and resource accounting.
- Confirmed the image-free boundary is unchanged: compact loading keeps alpha disabled, the
  worker installs the same image-file and forbidden-import guard before RTGS imports, all five
  negative controls remain required, and no RGB, mask, legacy scene, dense trainer, Beam, or
  carrier path is admitted.
- Confirmed the exact argv command names the successor driver, task, and one canonical run root.
  Focused verification passed all 59 field-input, observation, lifter, and experiment-driver
  tests; Ruff check/format, docs sync, ARA structure, script layout, agent workflow, experiment
  contract/data validation, digest recomputation, whitespace checks, and the full
  `./scripts/verify.sh` gate also passed.

## Findings

Approved prospectively. The corrected successor is runnable, isolates the shared precision repair
without altering the three-arm comparison, preserves held-out and no-image boundaries, and adds
failure observability without changing successful outcomes or decision gates. Any protocol-bearing
edit requires a new digest and prospective review before initialization; approval authorizes
execution of the frozen design, not any result or claim.

## Protected Actions Not Taken

The reviewer did not initialize or execute the successor, inspect any successor run or result
artifact, open protected RGB or mask outcomes, render a report, or access any successor outcome
while performing this review.
