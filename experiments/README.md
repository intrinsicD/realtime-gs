# Task-first experiments

This is the active surface for new result-bearing experiments. Historical drivers and evidence under
`benchmarks/` are append-only provenance; do not rename them to make the old tree look tidy.
Use [`INDEX.md`](INDEX.md) to find both task-first and historical experiments under the current
`YYYYMMDD_<task_slug>_<data_slug>` naming scheme. Historical names in that catalog are navigation
aliases only; the linked legacy paths remain authoritative.

## The three arms

| Arm | Reconstruction inputs | Question |
|---|---|---|
| `direct_compact` | calibration + frozen 2D Gaussian bundles only | How much reconstruction-process VRAM does the no-Beam compact pipeline use? |
| `beam_fusion` | calibration + frozen 2D Gaussian bundles only | Does Beam Fusion add value under matched count, optimizer, and compute? |
| `rgb_3dgs` | RGB + lossless masks + calibration + a frozen 3D initialization | What quality/resource trade-off does image-supervised 3DGS produce? |

The RGB arm may evaluate the saved compact model against RGB/masks. That does not retroactively
put images inside the compact arm. Keep the processes and their memory receipts separate.

The current RGB arm is a **matched-initialization, image-supervised 3DGS comparison**. It is not
“Original 3DGS”: this capture has no COLMAP/SfM sparse model. Add a separately named
`original_3dgs_colmap` comparator only after a real sparse reconstruction is sealed.

## Lifecycle

1. Create `tasks/YYYYMMDD_<task_slug>_<data_slug>.json` from the template.
2. Assign one owner and freeze dependencies, data, splits, seeds, stages, comparators, metrics,
   gates, resource accounting, and the exact argv command. Keep status `draft` while any blocker
   remains.
3. Run `python scripts/experiment_contract.py validate`.
4. Rehash local inputs with `validate-data`. Change the task or its data seal before execution if
   it fails. A generated compact dataset may declare an adjacent `production_manifest` in its
   dataset record; `seal-data` then binds that provenance sidecar in addition to calibration,
   the load-bearing compact manifest, and every listed compact view.
5. Have a reviewer whose stable label differs from the owner run `review-digest`, write
   `reviews/<task_id>_PROTOCOL_REVIEW.md` from the template, and record `Outcome Access: none`.
   The reviewer must not execute the protected run or inspect sealed outcomes.
6. Copy the approved/rejected metadata into `protocol_review`, set status to `ready`/`blocked`,
   and re-run `validate`. A protocol edit requires a new digest and review.
7. Run `init-run`. Official initialization fails on a dirty tracked worktree, binds the review
   artifact hash, and never overwrites an existing run.
8. Write every attempt/cell below the one canonical `runs/<task_id>/` root. Do not create
   `_v2`, `_final`, `_failed`, timestamp, or “latest” sibling directories.
9. Produce the shared `metrics.json`, dimensioned `training_history.json`, effective config,
   environment/run/input-boundary/resource receipts, initial/final PLYs and previews, and the
   canonically named RESULT Markdown+JSON records. A failed run still writes the JSON sources it
   reached and records `status: failed`; it may omit models, previews, and evidence.
10. Perform the distinct independent results audit over the raw bundle and RESULT records; write
    the canonical AUDIT Markdown+JSON records and declare all four evidence files in `metrics.json`.
11. Run `render` once, serve the generated report, and exercise the viewer in a real browser. A
    pass requires HTTP 200 plus all local report targets, WebGL2 and a live canvas, a ready viewer,
    visible non-background scene pixels, an orbit that changes the camera, and no fatal or
    unclassified client errors. Record these facts, browser identity, classified warnings, renderer
    when exposed, and exact viewer argv in `viewer_smoke.json`, then rerun `render` so the
    final manifest includes that receipt. It generates `index.html`, the accompanying `README.md`,
    and `manifest.json`; rerun it after changing any source. Never hand-edit those three files.
12. Gate the final run with both `check-run` and `scripts/check_results_bundle.py`, then append the
    audited outcome to `docs/EXPERIMENTS.md`.

The task remains `ready` and immutable after `init-run`; run/result status lives in the run
receipts and append-only evidence, not by mutating the locked task. Task changes after a run starts
require a new task id. A protected run that terminates early uses the same root and generates a
failure report, but the bundle gate rejects it as non-results-bearing. Nested retries stay under
`runs/<task_id>/attempts/` with their own failure receipts; they do not consume new top-level names.

## Naming and file ownership

- Task/run id: `YYYYMMDD_<task_slug>_<data_slug>`.
- Prospective review: `reviews/<task_id>_PROTOCOL_REVIEW.md`.
- Task-specific driver: `scripts/experiments/<task_id>.py`.
- Local outputs: `runs/<task_id>/` (ignored except `runs/README.md`).
- Optional generated-input provenance:
  `dataset/<capture>/<frame>/gaussians2d*/production_manifest.json`, declared by the task and
  included in its data seal.
- Human evidence: `benchmarks/results/<task_id>_RESULT.md` and `<task_id>_AUDIT.md`.
- Machine evidence: the same stem with `.json`.
- Scratch space: `.scratch/<task_id>/`, never the repository root.

An agent claims one draft task by editing only that task's `owner` in its working branch. It must
not edit another active task, reuse another task's run directory, or write ad-hoc reports. Shared
code goes under `src/`; task-specific orchestration goes under `scripts/experiments/`.

## Experiment bundle contract v2

New tasks freeze `report_template_version: 2`. Older tasks with no version select v1 and remain
valid; do not migrate or rerender historical evidence merely for consistency.

Before `render`, the task driver owns these run-local sources:

- `metrics.json`: result summary, final metrics, the three frozen diagrams, artifact/evidence
  declarations, notes, and exact reproduce/report-server/orbit-viewer argv commands;
- `training_history.json`: tidy records keyed by step, wall time, stage, dataset, arm, seed, split,
  and metric id, plus metric metadata. Each dataset/arm/seed series records one ordered `start` and
  `end` boundary (step and elapsed seconds) for every frozen stage, including stages without a
  scalar metric row. Records must stay inside their stage interval. Fitting history may use only
  `train`, `validation`, or `diagnostic`; held-out/test records are rejected;
- `gaussians.config.json`: the complete effective configuration, not only CLI overrides;
- `environment.json`: Python, platform, package versions, and device identity;
- `run_receipt.json`: start/finish UTC times, status, exit code, failure phase, and message;
- `input_boundary_receipt.json` and `resource_receipt.json`;
- `viewer_smoke.json`: after the first render, a structured attestation of report targets and
  browser/WebGL readiness, visible scene pixels in a UI-free framebuffer crop, a camera-changing
  orbit interaction, an empty fatal/unclassified client-error list, classified client warnings,
  and exact viewer argv;
- completed-run models/previews and canonical RESULT/AUDIT evidence.

`render` owns and atomically regenerates:

- `index.html`, with status, claim/input boundaries, pipeline, one elapsed-time SVG per
  metric/dataset/arm/seed series, shaded stages and explicit start/end boundaries in every curve,
  final metrics/diagrams, full parameters, provenance/environment, exact commands, and relative
  links to the entire inventory;
- `README.md`, the same durable handoff in Markdown, including how to reproduce, serve the report,
  and start the orbit viewer;
- `manifest.json`, which inventories every run file (except itself) and every declared evidence
  file with scope, role, media type, byte size, and SHA-256.

The arm may choose its metric rows and values, but it may not replace the renderer or omit frozen
primary metrics/the three required diagrams on a completed run. `check-run` verifies schemas,
frozen-command equality, generated links, full inventory, and checksums. The results-bundle gate
also requires previews and the structured browser smoke; server reachability without a rendered,
orbited WebGL client is insufficient. A structurally renderable failure report is not a
results-bearing bundle.

Use `templates/task.json`, `templates/metrics.json`, `templates/training_history.json`,
`templates/run_receipt.json`, `templates/environment.json`, `templates/viewer_smoke.json`, and
`templates/protocol_review.md` as the producer examples.

## Evidence boundary still missing

Frames 00008 and 00009 have both been used in earlier development. They can support development
and replication, not an untouched confirmatory claim. A paper-strength VRAM/quality claim still
needs at least one newly sealed scene, the same hardware/software environment, multiple paired
seeds, and a real COLMAP baseline if “Original 3DGS” is named.
