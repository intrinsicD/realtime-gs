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

For calibrated masked captures, the headline metric is held-out foreground PSNR. Full-canvas
PSNR, foreground-crop PSNR/SSIM, train metrics, primitive counts, and visual artifacts are saved
separately so black background does not inflate the result and train/test leakage is detectable.

<!-- BENCH:BEGIN -->
_Last run: 2026-07-14T09:05:16+00:00 · device `cpu` · torch 2.12.1+cpu · rev `aaef37c` · scene `synthetic_g40_c12_s48`_

| benchmark | key numbers |
| --- | --- |
| `image2gs_fit` | iters_per_s: 163.68 · psnr: 30.30 · seconds: 0.73 |
| `render_ref_cpu` | fps: 895.00 · frames: 36 · seconds: 0.04 |
| `lift_depth` | seconds: 0.01 · init_psnr: 19.08 · init_n_gaussians: 1155 · fit_seconds: 3.56 |
| `e2e_depth` | init_psnr: 19.08 · final_psnr: 31.33 · final_n_gaussians: 2884 · refine_seconds: 9.28 · fit_seconds: 3.56 · lift_seconds: 0.01 · total_seconds: 13.19 · peak_vram_mb: 0.00 · psnr_curve: [(75, 28.113994280497234), (150, 31.33353265126546)] · seconds_curve: [(75, 3.0667776335030794), (150, 9.277339012362063)] |
| `lift_hybrid` | seconds: 4.76 · init_psnr: 21.61 · init_n_gaussians: 1733 · fit_seconds: 3.56 |
| `e2e_hybrid` | init_psnr: 21.61 · final_psnr: 31.44 · final_n_gaussians: 4058 · refine_seconds: 15.98 · fit_seconds: 3.56 · lift_seconds: 4.76 · total_seconds: 24.83 · peak_vram_mb: 0.00 · psnr_curve: [(75, 27.806065877278645), (150, 31.43999195098877)] · seconds_curve: [(75, 5.4818503856658936), (150, 15.97036144323647)] |
| `lift_gradient` | seconds: 6.96 · init_psnr: 22.43 · init_n_gaussians: 1727 · fit_seconds: 3.56 |
| `e2e_gradient` | init_psnr: 22.43 · final_psnr: 31.03 · final_n_gaussians: 4291 · refine_seconds: 16.21 · fit_seconds: 3.56 · lift_seconds: 6.96 · total_seconds: 27.25 · peak_vram_mb: 0.00 · psnr_curve: [(75, 27.775740305582683), (150, 31.027061303456623)] · seconds_curve: [(75, 5.30160375405103), (150, 16.21365449205041)] |
| `lift_carve` | seconds: 0.08 · init_psnr: 20.31 · init_n_gaussians: 1396 · fit_seconds: 3.56 |
| `e2e_carve` | init_psnr: 20.31 · final_psnr: 31.87 · final_n_gaussians: 3682 · refine_seconds: 13.92 · fit_seconds: 3.56 · lift_seconds: 0.08 · total_seconds: 17.95 · peak_vram_mb: 0.00 · psnr_curve: [(75, 28.73896376291911), (150, 31.8705309232076)] · seconds_curve: [(75, 4.981697021052241), (150, 13.916375315748155)] |
| `lift_sfm` | seconds: 0.00 · init_psnr: 19.95 · init_n_gaussians: 200 · fit_seconds: 3.56 |
| `e2e_sfm` | init_psnr: 19.95 · final_psnr: 28.89 · final_n_gaussians: 1368 · refine_seconds: 2.87 · fit_seconds: 3.56 · lift_seconds: 0.00 · total_seconds: 6.55 · peak_vram_mb: 0.00 · psnr_curve: [(75, 27.280276934305828), (150, 28.890977541605633)] · seconds_curve: [(75, 0.7962553277611732), (150, 2.8722090451046824)] |
| `lift_random` | seconds: 0.00 · init_psnr: 14.11 · init_n_gaussians: 2000 · fit_seconds: 3.56 |
| `e2e_random` | init_psnr: 14.11 · final_psnr: 29.15 · final_n_gaussians: 3652 · refine_seconds: 16.42 · fit_seconds: 3.56 · lift_seconds: 0.00 · total_seconds: 20.37 · peak_vram_mb: 0.00 · psnr_curve: [(75, 26.15394401550293), (150, 29.145880063374836)] · seconds_curve: [(75, 7.66153350006789), (150, 16.4095715675503)] |
<!-- BENCH:END -->

## Reading the numbers

- **Init PSNR** (after lifting, before refinement) is the headline metric for the research
  idea: better init ⇒ fewer refinement iterations to a target quality.
- **Time-to-quality** matters more than final PSNR; final PSNR after long refinement tends
  to converge across initializations.
- CPU numbers (this harness on a laptop/CI) are for *relative* comparisons between
  variants and for catching regressions. Absolute speed claims require the GPU
  configuration (M2 in the roadmap).
