# Task-first experiments

This is the active surface for new result-bearing experiments. Historical drivers and evidence under
`benchmarks/` are append-only provenance; do not rename them to make the old tree look tidy.

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
   it fails.
5. Set the task to `ready`, then run `init-run`. Official initialization fails on a dirty tracked
   worktree and never overwrites an existing run.
6. Write every attempt/cell below the one canonical `runs/<task_id>/` root. Do not create
   `_v2`, `_final`, `_failed`, timestamp, or “latest” sibling directories.
7. Produce the shared `metrics.json`, initial/final PLYs, history, config, input-boundary and
   resource receipts, previews, and the canonically named RESULT/AUDIT Markdown+JSON records.
   Generate (never hand-edit) the page with `render`.
8. Gate the run with both `check-run` and `scripts/check_results_bundle.py`, then perform the
   independent results audit and append the outcome to `docs/EXPERIMENTS.md`.

The task remains `ready` and immutable after `init-run`; run/result status lives in the run
receipts and append-only evidence, not by mutating the locked task. Task changes after a run starts
require a new task id. Failed attempts stay under
`runs/<task_id>/attempts/` with their failure receipt; they do not consume new top-level names.

## Naming and file ownership

- Task/run id: `YYYYMMDD_<task_slug>_<data_slug>`.
- Task-specific driver: `scripts/experiments/<task_id>.py`.
- Local outputs: `runs/<task_id>/` (ignored except `runs/README.md`).
- Human evidence: `benchmarks/results/<task_id>_RESULT.md` and `<task_id>_AUDIT.md`.
- Machine evidence: the same stem with `.json`.
- Scratch space: `.scratch/<task_id>/`, never the repository root.

An agent claims one draft task by editing only that task's `owner` in its working branch. It must
not edit another active task, reuse another task's run directory, or write ad-hoc reports. Shared
code goes under `src/`; task-specific orchestration goes under `scripts/experiments/`.

## Common report contract

Every arm uses report template v1 from `scripts/experiment_contract.py`. The page always contains:

- identity, decision, evidence phase, and exact claim boundary;
- reconstruction/evaluation modality boundaries;
- the ordered pipeline diagram;
- grouped numeric metrics;
- quality, resource, and stage-runtime diagrams;
- task/data/source provenance, artifacts, and the exact viewer command.

`templates/metrics.json` documents the producer schema. The arm may choose its own metric rows and
chart values, but it may not replace the template or omit one of the three diagrams.

## Evidence boundary still missing

Frames 00008 and 00009 have both been used in earlier development. They can support development
and replication, not an untouched confirmatory claim. A paper-strength VRAM/quality claim still
needs at least one newly sealed scene, the same hardware/software environment, multiple paired
seeds, and a real COLMAP baseline if “Original 3DGS” is named.
