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
_Last run: 2026-07-07T11:59:28+00:00 · device `cpu` · torch 2.12.1+cpu · rev `eb437bb` · scene `synthetic_g40_c12_s48`_

| benchmark | key numbers |
| --- | --- |
| `image2gs_fit` | iters_per_s: 75.84 · psnr: 30.30 · seconds: 1.58 |
| `render_ref` | fps: 458.97 · frames: 36 · seconds: 0.08 |
| `lift_depth` | seconds: 0.02 · init_psnr: 17.05 · init_n_gaussians: 1186 |
| `e2e_depth` | init_psnr: 17.05 · final_psnr: 28.53 · final_n_gaussians: 3901 · refine_seconds: 19.68 · total_seconds: 20.59 |
| `lift_gradient` | seconds: 8.79 · init_psnr: 18.05 · init_n_gaussians: 1800 |
| `e2e_gradient` | init_psnr: 18.05 · final_psnr: 25.31 · final_n_gaussians: 6546 · refine_seconds: 24.05 · total_seconds: 34.59 |
| `lift_carve` | seconds: 0.11 · init_psnr: 17.48 · init_n_gaussians: 1020 |
| `e2e_carve` | init_psnr: 17.48 · final_psnr: 29.13 · final_n_gaussians: 4201 · refine_seconds: 17.93 · total_seconds: 18.82 |
| `lift_sfm` | seconds: 0.00 · init_psnr: 19.95 · init_n_gaussians: 200 |
| `e2e_sfm` | init_psnr: 19.95 · final_psnr: 28.67 · final_n_gaussians: 1369 · refine_seconds: 5.13 · total_seconds: 5.38 |
| `lift_random` | seconds: 0.00 · init_psnr: 8.08 · init_n_gaussians: 2000 |
| `e2e_random` | init_psnr: 8.08 · final_psnr: 27.93 · final_n_gaussians: 1884 · refine_seconds: 17.64 · total_seconds: 18.22 |
<!-- BENCH:END -->

## Reading the numbers

- **Init PSNR** (after lifting, before refinement) is the headline metric for the research
  idea: better init ⇒ fewer refinement iterations to a target quality.
- **Time-to-quality** matters more than final PSNR; final PSNR after long refinement tends
  to converge across initializations.
- CPU numbers (this harness on a laptop/CI) are for *relative* comparisons between
  variants and for catching regressions. Absolute speed claims require the GPU
  configuration (M2 in the roadmap).
