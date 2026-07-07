---
name: bench
description: Run the benchmark suite and update docs/BENCHMARKS.md. Use after performance-relevant changes (renderers, fitting loops, lifters, trainer) or when asked how fast/accurate the pipeline is.
---

# Bench

```bash
.venv/bin/python benchmarks/run.py --quick --update-docs   # CPU-sized, ~1-2 min
.venv/bin/python benchmarks/run.py --update-docs           # full config (use on GPU boxes)
```

- Results are appended as JSON to `benchmarks/results/` (one file per run, named by
  timestamp + host kind). Commit them — they are the performance history.
- `--update-docs` rewrites ONLY the block between `<!-- BENCH:BEGIN -->` and
  `<!-- BENCH:END -->` in `docs/BENCHMARKS.md`. Never hand-edit inside that block; prose
  outside it is yours to maintain.
- The harness benchmarks: 2D image fitting (it/s, PSNR), reference-renderer throughput,
  each lifting variant (runtime + init PSNR), and a short end-to-end refine
  (init→final PSNR). GPU/gsplat benches run only where CUDA is available.

## Comparing runs

Each JSON has `meta` (git rev, device, torch version) and `results` keyed by benchmark
name. To claim a speedup/regression, cite two result files. If a change alters benchmark
numbers meaningfully (>10%), mention it in the commit message and add a line to
`docs/EXPERIMENTS.md`.
