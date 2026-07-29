# Prospective Protocol Review

- Task ID: `20260729_field_sweep_placement_stage_frames00008_00009`
- Protocol SHA-256: `9ff7057e47e3ea6a2af0532edbecf13036a892d9805e02cfe26aee7f33732844`
- Reviewer: `Codex-protocol-reviewer`
- Verdict: `approved`
- Outcome Access: `none`

## Scope

This protocol may establish development and replication evidence about whether the named
source-excluded robust compact-field sweep improves placement and post-refit held-out compact-field
error over matched bounded-midpoint and all-view-consensus arms on outcome-exposed frames 00008 and
00009. It cannot establish RGB image quality, physical geometry accuracy, GPU or production
performance, cross-dataset generalization, topology utility, or a default change.

## Checks

- Confirmed the compact data seal binds the calibration, manifests, and every selected `.rtgsv`
  container for both registered frames; `validate-data` passes, and the driver rehashes the
  lock-bound seal and selected bytes before execution.
- Confirmed the exact train and held-out camera IDs form disjoint complete partitions, only the
  deterministic training subset enters placement/refit, and both placement and final held-out
  validations occur after refit/topology has completed.
- Confirmed the reconstruction is invariant to perturbed held-out teacher values in the synthetic
  integration test, while the held-out metric changes as expected.
- Confirmed all three arms share the same seeded source-anchor draw, AABB bounds, primitive count,
  source covariance construction, field refit, disabled topology, validation samples, and artifact
  path; aggregate publication rejects anchor or lineage mismatch.
- Confirmed the treatment excludes the source view from neighbor scoring, retains a frozen robust
  neighbor fraction, uses the same coarse-to-fine grid as the all-view comparator, and has no
  silent treatment fallback.
- Confirmed raw RGB/mask files, packed alpha, image-capable loaders, `SceneData`, RGB training,
  Beam Fusion, carrier refinement, and carrier scheduling are excluded by frozen configuration,
  live import/open guards, negative controls, and exit receipts.
- Confirmed three paired measured seeds per scene, one discarded warmup per scene and arm,
  counterbalanced measured arm order, fresh single-thread CPU workers, and retention of every raw
  measured cell.
- Confirmed wall time and peak RSS are drawn from the scoped worker receipt after compact loading,
  validation, serialization, and atomic cell publication; these measurements are explicitly
  descriptive and support no performance claim.
- Confirmed the directional producer rule is frozen at a robust-to-midpoint final RGB
  geometric-mean ratio of at most `0.95`, at least two robust wins among three paired seeds on each
  scene, supported-track fraction of at least `0.95`, and source-projection error of at most
  `0.0002` pixels.
- Confirmed the exact argv command, run root, required PLY/history/config/boundary/resource
  artifacts, producer evidence records, pending independent results-audit state, viewer command,
  and development-only claim boundary are frozen.
- Structural verification passed: experiment task validation, data-seal validation, protocol
  digest, 15 focused synthetic/static tests, Ruff check/format, and durable script layout.

## Findings

Approved prospectively. The protocol isolates the placement mechanism with matched controls,
prevents held-out outcomes from affecting reconstruction, binds inputs and execution provenance,
and limits interpretation to the registered compact-domain development evidence. Any
protocol-bearing edit requires a new digest and prospective review before initialization.

## Protected Actions Not Taken

The reviewer did not initialize or execute the experiment, inspect a run/result artifact for this
task, open protected RGB or mask outcomes, or access any task outcome while performing this review.
