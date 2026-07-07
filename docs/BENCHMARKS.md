# Benchmarks

Performance and quality are tracked, not guessed. The harness is `benchmarks/run.py`:

```bash
.venv/bin/python benchmarks/run.py --quick --update-docs   # CPU-sized configuration
.venv/bin/python benchmarks/run.py --update-docs           # full configuration (GPU box)
.venv/bin/python benchmarks/run.py --quick --smoke         # CI smoke (tiny, no file output)
```

Each run writes `benchmarks/results/<timestamp>_<device>.json` containing `meta`
(git revision, device, torch version, config) and `results`. Commit result files — they
are the performance history of the repo. The table below is rewritten in place by
`--update-docs`; do not edit inside the markers.

Benchmarks included:

- `image2gs_fit` — stage-1 fitting throughput (iterations/s) and reached PSNR
- `render_ref` — reference rasterizer throughput (frames/s at benchmark scene size)
- `lift_<variant>` — per-variant lifting runtime and initialization PSNR (mean over views)
- `e2e_<variant>` — init PSNR → PSNR after a short refinement, with stage timings

<!-- BENCH:BEGIN -->
_No benchmark results recorded yet. Run `benchmarks/run.py --update-docs`._
<!-- BENCH:END -->

## Reading the numbers

- **Init PSNR** (after lifting, before refinement) is the headline metric for the research
  idea: better init ⇒ fewer refinement iterations to a target quality.
- **Time-to-quality** matters more than final PSNR; final PSNR after long refinement tends
  to converge across initializations.
- CPU numbers (this harness on a laptop/CI) are for *relative* comparisons between
  variants and for catching regressions. Absolute speed claims require the GPU
  configuration (M2 in the roadmap).
