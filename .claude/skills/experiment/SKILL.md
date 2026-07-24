---
name: experiment
description: Run a research experiment on the pipeline (compare lifting variants, sweep a hyperparameter, test a hypothesis) and log the outcome in docs/EXPERIMENTS.md. Use for any "does X help / which variant is better / try Y" request.
---

# Experiment

This is a research repo — experiments are first-class and must be reproducible and logged.

## Running

Quick comparisons on synthetic scenes (works on CPU):

```bash
.venv/bin/rtgs run --scene synthetic --lifter depth --refine-iters 200
.venv/bin/rtgs bench --quick        # all variants side by side
```

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

Include the exact viewer command and artifact directory in the result handoff. The orbitable WebGL
preview is qualitative; decision metrics and camera snapshots must come from the exact selected
`Rasterizer` backend. Use Torch snapshots in the current shared environment; its editable
GaussianImage `gsplat` fork is not the repository's modern 3D gsplat backend.

Every official results-bearing output directory must also contain `index.html`. Generate it from
the exact saved metrics and visual artifacts, use relative links, and include the protocol,
summary, result/audit records, viewer manifest, comparison visuals, and saved models relevant to
the experiment. Bind the page in the machine summary, serve it from the repository root, require
HTTP 200 for the page and every local target, and preserve a smoke-test receipt. A JSON-only
handoff is incomplete; synthetic mechanism/unit checks that do not claim an official result are
exempt.

For sweeps, write a short script under `benchmarks/` (or a throwaway in the scratchpad if
it should not be kept) that calls `rtgs.pipeline.run_pipeline` directly with varying
config, seeds fixed.

## Logging (mandatory)

Append an entry to `docs/EXPERIMENTS.md` using its template: date, question, setup
(exact command/config + git rev), result numbers, conclusion, and follow-ups. Negative
results are logged too — they are the point of a research repo. If the experiment changes
a default hyperparameter, update the config dataclass AND note the entry that justifies it. Also
record the local `dataset/` scene/split, viewer-ready output directory, exact `rtgs view` command,
and `index.html` path; if the local-data interaction, viewer smoke, or results-page smoke did not
run, label the experiment incomplete.
