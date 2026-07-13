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
- `render_ref_cpu` — CPU reference rasterizer throughput (frames/s at benchmark scene size)
- `lift_<variant>` — per-variant lifting runtime and initialization PSNR (mean over views)
- `e2e_<variant>` — init PSNR → PSNR after a short refinement, full shared-stage timing,
  time-to-quality samples, peak VRAM, and final primitive count

<!-- BENCH:BEGIN -->
_Last run: 2026-07-13T12:36:16+00:00 · device `cpu` · torch 2.12.1+cpu · rev `4c27649` · scene `synthetic_g40_c12_s48`_

| benchmark | key numbers |
| --- | --- |
| `image2gs_fit` | iters_per_s: 142.73 · psnr: 30.30 · seconds: 0.84 |
| `render_ref_cpu` | fps: 811.09 · frames: 36 · seconds: 0.04 |
| `lift_depth` | seconds: 0.02 · init_psnr: 19.08 · init_n_gaussians: 1155 · fit_seconds: 3.90 |
| `e2e_depth` | init_psnr: 19.08 · final_psnr: 31.36 · final_n_gaussians: 2788 · refine_seconds: 10.87 · fit_seconds: 3.90 · lift_seconds: 0.02 · total_seconds: 15.18 · peak_vram_mb: 0.00 · psnr_curve: [(75, 27.60728391011556), (150, 31.361705621083576)] · seconds_curve: [(75, 3.900251758284867), (150, 10.86851008515805)] |
| `lift_gradient` | seconds: 7.93 · init_psnr: 22.43 · init_n_gaussians: 1727 · fit_seconds: 3.90 |
| `e2e_gradient` | init_psnr: 22.43 · final_psnr: 30.86 · final_n_gaussians: 4315 · refine_seconds: 20.04 · fit_seconds: 3.90 · lift_seconds: 7.93 · total_seconds: 32.56 · peak_vram_mb: 0.00 · psnr_curve: [(75, 27.775246461232502), (150, 30.860476811726887)] · seconds_curve: [(75, 5.956784686073661), (150, 20.029277155175805)] |
| `lift_carve` | seconds: 0.08 · init_psnr: 20.31 · init_n_gaussians: 1396 · fit_seconds: 3.90 |
| `e2e_carve` | init_psnr: 20.31 · final_psnr: 31.91 · final_n_gaussians: 3661 · refine_seconds: 15.15 · fit_seconds: 3.90 · lift_seconds: 0.08 · total_seconds: 19.59 · peak_vram_mb: 0.00 · psnr_curve: [(75, 28.73896376291911), (150, 31.913447856903076)] · seconds_curve: [(75, 5.34015017747879), (150, 15.142141921445727)] |
| `lift_sfm` | seconds: 0.00 · init_psnr: 19.95 · init_n_gaussians: 200 · fit_seconds: 3.90 |
| `e2e_sfm` | init_psnr: 19.95 · final_psnr: 28.27 · final_n_gaussians: 1380 · refine_seconds: 2.96 · fit_seconds: 3.90 · lift_seconds: 0.00 · total_seconds: 7.01 · peak_vram_mb: 0.00 · psnr_curve: [(75, 27.280276934305828), (150, 28.2741379737854)] · seconds_curve: [(75, 0.7955154376104474), (150, 2.9626199370250106)] |
| `lift_random` | seconds: 0.00 · init_psnr: 14.11 · init_n_gaussians: 2000 · fit_seconds: 3.90 |
| `e2e_random` | init_psnr: 14.11 · final_psnr: 29.05 · final_n_gaussians: 3651 · refine_seconds: 20.34 · fit_seconds: 3.90 · lift_seconds: 0.00 · total_seconds: 24.78 · peak_vram_mb: 0.00 · psnr_curve: [(75, 26.15394401550293), (150, 29.054691473642986)] · seconds_curve: [(75, 8.220168687403202), (150, 20.324235172010958)] |
<!-- BENCH:END -->

## Reading the numbers

- **Init PSNR** (after lifting, before refinement) is the headline metric for the research
  idea: better init ⇒ fewer refinement iterations to a target quality.
- **Time-to-quality** matters more than final PSNR; final PSNR after long refinement tends
  to converge across initializations.
- CPU numbers (this harness on a laptop/CI) are for *relative* comparisons between
  variants and for catching regressions. Absolute speed claims require the GPU
  configuration (M2 in the roadmap).
