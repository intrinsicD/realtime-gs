---
name: rtgs-experiment
description: Run a research experiment on the pipeline (compare lifting variants, sweep a hyperparameter, test a hypothesis) and log the outcome in docs/EXPERIMENTS.md. Use for any "does X help / which variant is better / try Y" request.
---

# Experiment

This is a research repo — experiments are first-class and must be reproducible and logged.

## Register the task before code or execution

For every result-bearing experiment, first read `experiments/README.md` and create:

`experiments/tasks/YYYYMMDD_<task_slug>_<data_slug>.json`

Freeze the question, claim boundary, evidence phase, data paths/seal, camera splits, seeds, input
policy, ordered stages, comparators, metrics, diagrams, resource scope, and exact argv command.
Keep the task `draft` while any blocker remains. Validate it with:

```bash
.venv/bin/python scripts/experiment_contract.py validate
.venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/<task_id>.json
```

Only a `ready` task may start. Initialize it with:

```bash
.venv/bin/python scripts/experiment_contract.py init-run experiments/tasks/<task_id>.json
```

The command refuses a dirty tracked worktree and an existing run root. A protocol change after
initialization requires a new task id. Keep attempts under the one run root; never create
`_v2`, `_final`, `_failed`, timestamp, or “latest” siblings.

## Running

Quick comparisons on synthetic scenes (works on CPU):

```bash
.venv/bin/rtgs run --scene synthetic --lifter depth --refine-iters 200
.venv/bin/rtgs bench --quick        # all variants side by side
```

Without a registered task these are scratch diagnostics only: do not retain their output, log
their numbers as an experiment, or use them in a claim.

Every R&D branch must include a local calibrated-data interaction before handoff. The supplied
object captures are directly loadable from `dataset/`; the loader preserves calibrated camera ids,
uses every eighth selected camera as held-out test data, and excludes those cameras from fitting,
lifting, and refinement:

```bash
.venv/bin/rtgs run \
  --scene dataset/2025_03_07_stage_with_fabric/frame_00008 \
  --downscale 16 --lifter carve --out runs/<name>
```

Synthetic scenes remain useful for CPU regression and mechanism gates, but do not close a
pipeline-quality/default question with synthetic-only evidence. If checkpoint or hyperparameter
selection is needed, freeze a validation subset drawn from training cameras and leave the loader's
test cameras reporting-only.

## Viewer and results-page handoff (mandatory)

Use `--out` and keep previews enabled so every results-bearing run writes `gaussians_init.ply`,
`gaussians.ply`, metrics/history JSON, calibrated comparisons, and novel-view diagnostics. Launch
and smoke-test the browser viewer before reporting the experiment:

```bash
.venv/bin/rtgs view \
  --gaussians runs/<name>/gaussians.ply \
  --scene dataset/2025_03_07_stage_with_fabric/frame_00008 \
  --downscale 16 --device cpu --rasterizer torch
```

The smoke must happen in a real browser client: confirm the viewer reached ready state, WebGL2
created at least one canvas, a UI-free framebuffer crop contains non-background scene pixels, an
orbit gesture changed the camera, and no fatal or unclassified client error was raised. Record the
browser name/version, user agent, WebGL renderer when exposed, exact viewer argv, classified
warnings, and those checks in `viewer_smoke.json` using
`experiments/templates/viewer_smoke.json`. An HTTP 200 or a working camera over a blank canvas is
not a viewer smoke.

On Ubuntu systems using Snap Firefox with NVIDIA, preserve AppArmor denials for
`snap.firefox.firefox` and `/dev/char/195:*` as packaging diagnostics, but do not infer a crash
from the denial alone: the same denial can accompany a successful WebGL2 render. Require direct
evidence such as a crash report, lost context, uncategorized exception, or blank framebuffer.
Use Chrome/Chromium or a non-Snap Firefox only when the failing client remains reproducible.

Include the exact viewer command and artifact directory in the result handoff. The orbitable WebGL
preview is qualitative; decision metrics and camera snapshots must come from the exact selected
`Rasterizer` backend. Use Torch snapshots in the current shared environment; its editable
GaussianImage `gsplat` fork is not the repository's modern 3D gsplat backend.

Every new official task freezes report template v2 and produces the machine sources documented by
`experiments/templates/`: metrics, dimensioned fitting history, complete effective parameters,
environment, run/input/resource receipts, models/previews, and RESULT/AUDIT evidence. History may
not carry held-out/test fitting observations. Record exact reproduce, report-server, and orbit
viewer argv in `metrics.json`.

After the distinct results-audit pass has written the canonical AUDIT records, render once,
smoke-test the page/viewer, write the receipt, and render again so the final manifest includes the
receipt. Do not hand-write the generated outputs:

```bash
.venv/bin/python scripts/experiment_contract.py render runs/<task_id>
.venv/bin/python scripts/experiment_contract.py check-run runs/<task_id>
```

The renderer creates `index.html`, `README.md`, and a SHA-256 `manifest.json`. The page carries the
input boundary, pipeline, static SVG fitting histories over elapsed seconds with explicit start/end
boundaries for every frozen stage in every dataset/arm/seed series, grouped final metrics,
quality/resource/stage-runtime diagrams, full parameters, environment/provenance, exact commands,
and relative links to every inventoried artifact and evidence record. Rerun it after any source or
receipt changes. Serve from the location recorded by `commands.serve_report`, require HTTP 200 for
the page and every local target, and preserve the structured browser smoke receipt. A JSON-only
handoff is incomplete; synthetic mechanism/unit checks that do not claim an official result are
exempt.

Gate the bundle before reporting:

```bash
.venv/bin/python scripts/check_results_bundle.py runs/<name>
```

It checks required artifacts/previews, report links and summary numbers, the generated Markdown
handoff, complete manifest inventory/checksums, and the structured page/WebGL/orbit browser smoke.
Pass `--no-previews` only for a legitimately preview-free run. Failed runs may render an explicit
failure report but cannot pass this gate. A bundle that does not pass is not results-bearing, and
its numbers do not go in a handoff. Historical v1 bundles remain on their frozen validation path.

Name a kept task-specific driver `scripts/experiments/<task_id>.py`. Reusable performance cases
belong in `benchmarks/run.py`; a throwaway belongs in `.scratch/<task_id>/`. Do not add another
claim-specific file at `benchmarks/` root and do not implement another HTML template.

## Logging (mandatory)

Append an entry to `docs/EXPERIMENTS.md` using its template: date, question, setup
(exact command/config + git rev), result numbers, conclusion, and follow-ups. Negative
results are logged too — they are the point of a research repo. If the experiment changes
a default hyperparameter, update the config dataclass AND note the entry that justifies it. Also
record the local `dataset/` scene/split, viewer-ready output directory, exact `rtgs view` command,
and `index.html` path; if the local-data interaction, viewer smoke, or results-page smoke did not
run, label the experiment incomplete.

If the experiment produced or changed a claim, add or update its row in `ara/logic/claims.md` in
the same commit, with a `Proof` binding to the artifact and a `Boundary` recording what the run
does *not* establish. Stage a not-yet-promoted finding as an `O<NN>` observation in
`ara/staging/observations.yaml` instead. `.venv/bin/python scripts/check_ara.py` verifies the
structure; see the "Evidence and claims" section of CLAUDE.md.

One-off task drivers go in `scripts/experiments/`, not the top level of `scripts/`.
