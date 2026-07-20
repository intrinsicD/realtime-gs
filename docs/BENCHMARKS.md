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
- `field_product_kernel_cpu` — deterministic CPU timing for the analytic additive
  density/RGB-numerator product-kernel discrepancy. This is a mechanism-only microbenchmark; it
  measures neither normalized/faded/affine StructSplat teacher semantics nor reconstruction
  quality, topology utility, or end-to-end field-lift performance

Focused depth-covariance research uses `benchmarks/depth_covariance_ablation.py`. It caches one
set of train-view 2D fits per seed, tunes the scalar isotropic control on training views only,
asserts covariance arms preserve non-covariance fields, and reports strict held-out metrics.

Focused fixed-correspondence research uses `benchmarks/world_position_consistency_ablation.py`.
It constructs a privileged synthetic GT-identity graph plus a degree-, endpoint-, and camera-pair-
matched derangement, reuses both bitwise across Gradient/Hybrid, and reports engagement, local
assigned-GT geometry, whole-scene utility, control attribution, and complete provenance. This is a
research harness; it does not supply a deployable matcher or change the default lifter objective.

The train-only follow-up uses `benchmarks/dense_train_position_ablation.py` and the pluggable
`rtgs.lift.matching.PositionMatcher` boundary. It freezes a raw-patch/epipolar graph using only
training RGB, calibration, and retained fitted centers, then applies a strict post-freeze synthetic
identity audit before any optimization arm. The official reference-backend graph passed coverage
floors but failed semantic precision (9.04%-11.76% versus 60%), so the harness correctly emitted a
provenance-complete stopped artifact without running or reporting the withheld utility arms.

Focused oriented-surface research uses `benchmarks/surface_plane_normal_ablation.py`. It freezes
four-neighbor cross-view planes from corrupted metric training depth, separates correct plane
normals from a within-source shuffled alignment-normal control, and performs a post-freeze clean
target audit before any Hybrid optimization. The sole official constructor passed every structural
floor but failed clean plane validity in all three seeds, so the harness emitted a stopped artifact
with all five utility arms withheld. The generic loss API remains disabled by default.

Real registered-RGB-D target validation uses `benchmarks/tum_rgbd_oriented_validity.py`. Its sealed
two-phase protocol constructs targets from 48 T-only depth views, audits them in eight disjoint V
views, and calibrates all desk thresholds mechanically from `fr1/xyz`. The sole `fr1/desk`
confirmatory run passed coverage, support, median-normal, and free-space gates but failed surface
p90 (202.11 mm), relative-depth p90 (25.19%), and low-tail normal agreement. The atomic desk seal
is consumed, Phase B is withheld, and the result must not be rerun or tuned on desk.

Signed residual attribution uses `benchmarks/tum_rgbd_signed_attribution.py`. Its nested sparse
target/dense-T visibility masks are constructed without validation depth, then independently label
behind-observed and observed-free-space residuals with target-balanced reductions and cluster
bootstrap intervals. The official `fr3/sitting_xyz` development run found sign-selective partial
occlusion enrichment but failed its frozen 2x risk-ratio and 15% relative positive-reduction
floors. Its decision manifest therefore forbids `fr3/walking_xyz` confirmation; no walking member
was opened and no optimization was authorized.

For calibrated masked captures, the headline metric is held-out foreground PSNR. Full-canvas
PSNR, foreground-crop PSNR/SSIM, train metrics, primitive counts, and visual artifacts are saved
separately so black background does not inflate the result and train/test leakage is detectable.

<!-- BENCH:BEGIN -->
_Last run: 2026-07-14T20:09:32+00:00 · device `cpu` · torch 2.9.0+cu128 · rev `2dddca4` · scene `synthetic_g40_c12_s48`_

| benchmark | key numbers |
| --- | --- |
| `image2gs_fit` | iters_per_s: 159.86 · psnr: 30.30 · seconds: 0.75 |
| `render_ref_cpu` | fps: 625.18 · frames: 36 · seconds: 0.06 |
| `lift_depth` | seconds: 0.02 · init_psnr: 21.00 · init_n_gaussians: 1155 · fit_seconds: 3.89 |
| `e2e_depth` | init_psnr: 21.00 · final_psnr: 32.95 · final_n_gaussians: 3087 · refine_seconds: 21.25 · fit_seconds: 3.89 · lift_seconds: 0.02 · total_seconds: 26.13 · peak_vram_mb: 0.00 · psnr_curve: [(75, 28.19204298655192), (150, 32.95308097203573)] · seconds_curve: [(75, 6.955220429000292), (150, 21.237210514000253)] |
| `lift_hybrid` | seconds: 11.45 · init_psnr: 21.61 · init_n_gaussians: 1734 · fit_seconds: 3.89 |
| `e2e_hybrid` | init_psnr: 21.61 · final_psnr: 32.69 · final_n_gaussians: 4040 · refine_seconds: 33.15 · fit_seconds: 3.89 · lift_seconds: 11.45 · total_seconds: 49.85 · peak_vram_mb: 0.00 · psnr_curve: [(75, 28.01260248819987), (150, 32.68544546763102)] · seconds_curve: [(75, 12.48156635199939), (150, 33.13528015999964)] |
| `lift_gradient` | seconds: 17.13 · init_psnr: 22.44 · init_n_gaussians: 1731 · fit_seconds: 3.89 |
| `e2e_gradient` | init_psnr: 22.44 · final_psnr: 32.13 · final_n_gaussians: 4159 · refine_seconds: 33.40 · fit_seconds: 3.89 · lift_seconds: 17.13 · total_seconds: 55.85 · peak_vram_mb: 0.00 · psnr_curve: [(75, 28.02474323908488), (150, 32.13338279724121)] · seconds_curve: [(75, 12.150749306000762), (150, 33.38112198300041)] |
| `lift_carve` | seconds: 0.09 · init_psnr: 20.31 · init_n_gaussians: 1396 · fit_seconds: 3.89 |
| `e2e_carve` | init_psnr: 20.31 · final_psnr: 33.07 · final_n_gaussians: 3825 · refine_seconds: 30.58 · fit_seconds: 3.89 · lift_seconds: 0.09 · total_seconds: 35.80 · peak_vram_mb: 0.00 · psnr_curve: [(75, 28.87715784708659), (150, 33.06545384724935)] · seconds_curve: [(75, 11.115793133000807), (150, 30.564637349000805)] |
| `lift_sfm` | seconds: 0.00 · init_psnr: 19.95 · init_n_gaussians: 200 · fit_seconds: 3.89 |
| `e2e_sfm` | init_psnr: 19.95 · final_psnr: 29.18 · final_n_gaussians: 1386 · refine_seconds: 6.20 · fit_seconds: 3.89 · lift_seconds: 0.00 · total_seconds: 10.41 · peak_vram_mb: 0.00 · psnr_curve: [(75, 26.831709067026775), (150, 29.177371819814045)] · seconds_curve: [(75, 1.246785763999469), (150, 6.199949501999981)] |
| `lift_random` | seconds: 0.00 · init_psnr: 14.11 · init_n_gaussians: 2000 · fit_seconds: 3.89 |
| `e2e_random` | init_psnr: 14.11 · final_psnr: 29.68 · final_n_gaussians: 4428 · refine_seconds: 39.54 · fit_seconds: 3.89 · lift_seconds: 0.00 · total_seconds: 44.97 · peak_vram_mb: 0.00 · psnr_curve: [(75, 26.088585535685223), (150, 29.67621151606242)] · seconds_curve: [(75, 16.145546818000184), (150, 39.516056276999734)] |
<!-- BENCH:END -->

## Reading the numbers

- **Init PSNR** (after lifting, before refinement) is the headline metric for the research
  idea: better init ⇒ fewer refinement iterations to a target quality.
- **Time-to-quality** matters more than final PSNR; final PSNR after long refinement tends
  to converge across initializations.
- CPU numbers (this harness on a laptop/CI) are for *relative* comparisons between
  variants and for catching regressions. Absolute speed claims require the GPU
  configuration (M2 in the roadmap).
