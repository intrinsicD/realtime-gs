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
_Last run: 2026-07-30T15:20:24+00:00 · device `cuda` · torch 2.13.0+cu130 · rev `44be09f` · scene `synthetic_g40_c12_s48`_

| benchmark | key numbers |
| --- | --- |
| `field_product_kernel_cpu` | components_per_field: 96 · field_l2_evaluations: 3 · component_pair_terms: 165888 · seconds: 0.04 · evaluations_per_s: 80.96 · l2_total: 533.22 |
| `compact_placement_csr_cpu` | components: 600 · query_points: 2048 · tile_size: 16 · nonempty_tiles: 256 · total_entries: 4094 · max_candidates: 27 · retained_payload_bytes: 20480 · component_id_dtype: int32 · evaluated_pairs: 196596 · peak_pair_chunk: 32766 · csr_build_seconds: 0.00 · grouped_seconds: 0.06 · csr_seconds: 0.00 · speedup: 24.39 · max_color_err: 0.00 · max_weight_sum_err: 0.00 · within_contract: 1 · cuda_seconds: 0.00 · cuda_speedup_vs_csr: 5.13 · cuda_max_color_err: 0.00 · cuda_max_weight_sum_err: 0.00 |
| `image2gs_fit` | iters_per_s: 111.13 · psnr: 29.57 · seconds: 1.08 |
| `image2gs_fit_batched` | views: 12 · seconds: 1.11 · serial_seconds: 3.80 · speedup_vs_serial: 3.41 · psnr_mean: 28.98 · serial_psnr_mean: 28.98 |
| `render_ref_cpu` | fps: 691.63 · frames: 36 · seconds: 0.05 |
| `lift_depth` | seconds: 0.13 · init_psnr: 21.03 · init_n_gaussians: 1138 · fit_seconds: 3.85 |
| `e2e_depth` | init_psnr: 21.03 · final_psnr: 31.89 · final_n_gaussians: 2958 · refine_seconds: 0.68 · fit_seconds: 3.85 · lift_seconds: 0.13 · total_seconds: 4.77 · peak_vram_mb: 21.00 · psnr_curve: [(75, 27.794999917348225), (150, 31.890784740447998)] · seconds_curve: [(75, 0.3461156729608774), (150, 0.6806481722742319)] |
| `lift_hybrid` | seconds: 0.29 · init_psnr: 21.87 · init_n_gaussians: 1714 · fit_seconds: 3.85 |
| `e2e_hybrid` | init_psnr: 21.87 · final_psnr: 32.60 · final_n_gaussians: 4037 · refine_seconds: 0.65 · fit_seconds: 3.85 · lift_seconds: 0.29 · total_seconds: 4.82 · peak_vram_mb: 21.00 · psnr_curve: [(75, 28.21779187520345), (150, 32.60060421625773)] · seconds_curve: [(75, 0.3248103139922023), (150, 0.6449614511802793)] |
| `lift_gradient` | seconds: 0.34 · init_psnr: 22.58 · init_n_gaussians: 1707 · fit_seconds: 3.85 |
| `e2e_gradient` | init_psnr: 22.58 · final_psnr: 31.40 · final_n_gaussians: 3963 · refine_seconds: 0.68 · fit_seconds: 3.85 · lift_seconds: 0.34 · total_seconds: 4.90 · peak_vram_mb: 21.00 · psnr_curve: [(75, 27.572185198465984), (150, 31.40077797571818)] · seconds_curve: [(75, 0.32116923574358225), (150, 0.6753093954175711)] |
| `lift_carve` | seconds: 0.05 · init_psnr: 20.44 · init_n_gaussians: 1359 · fit_seconds: 3.85 |
| `e2e_carve` | init_psnr: 20.44 · final_psnr: 32.98 · final_n_gaussians: 3918 · refine_seconds: 0.64 · fit_seconds: 3.85 · lift_seconds: 0.05 · total_seconds: 4.57 · peak_vram_mb: 21.00 · psnr_curve: [(75, 28.67591206232707), (150, 32.976891040802)] · seconds_curve: [(75, 0.31021956261247396), (150, 0.637114935554564)] |
| `lift_sfm` | seconds: 0.01 · init_psnr: 19.78 · init_n_gaussians: 200 · fit_seconds: 3.85 |
| `e2e_sfm` | init_psnr: 19.78 · final_psnr: 28.53 · final_n_gaussians: 1359 · refine_seconds: 0.62 · fit_seconds: 3.85 · lift_seconds: 0.01 · total_seconds: 4.51 · peak_vram_mb: 21.00 · psnr_curve: [(75, 26.56816593805949), (150, 28.526710828145344)] · seconds_curve: [(75, 0.30024223681539297), (150, 0.6184223648160696)] |
| `lift_random` | seconds: 0.00 · init_psnr: 14.34 · init_n_gaussians: 2000 · fit_seconds: 3.85 |
| `e2e_random` | init_psnr: 14.34 · final_psnr: 29.55 · final_n_gaussians: 4433 · refine_seconds: 0.67 · fit_seconds: 3.85 · lift_seconds: 0.00 · total_seconds: 4.55 · peak_vram_mb: 21.00 · psnr_curve: [(75, 26.053237915039062), (150, 29.5485045115153)] · seconds_curve: [(75, 0.32562865503132343), (150, 0.672127628698945)] |
<!-- BENCH:END -->

## Reading the numbers

- **Init PSNR** (after lifting, before refinement) is the headline metric for the research
  idea: better init ⇒ fewer refinement iterations to a target quality.
- **Time-to-quality** matters more than final PSNR; final PSNR after long refinement tends
  to converge across initializations.
- CPU numbers (this harness on a laptop/CI) are for *relative* comparisons between
  variants and for catching regressions. Absolute speed claims require the GPU
  configuration (M2 in the roadmap).
