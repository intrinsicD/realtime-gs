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

Real scenes need a COLMAP dataset directory (`sparse/0` + `images/`):

```bash
.venv/bin/rtgs run --scene /path/to/colmap_dataset --lifter carve --out runs/<name>
```

For sweeps, write a short script under `benchmarks/` (or a throwaway in the scratchpad if
it should not be kept) that calls `rtgs.pipeline.run_pipeline` directly with varying
config, seeds fixed.

## Logging (mandatory)

Append an entry to `docs/EXPERIMENTS.md` using its template: date, question, setup
(exact command/config + git rev), result numbers, conclusion, and follow-ups. Negative
results are logged too — they are the point of a research repo. If the experiment changes
a default hyperparameter, update the config dataclass AND note the entry that justifies it.
