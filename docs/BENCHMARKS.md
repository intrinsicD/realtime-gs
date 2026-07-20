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
_Last run: 2026-07-20T12:38:59+00:00 · device `cpu` · torch 2.13.0+cpu · rev `381a986` · scene `synthetic_g40_c12_s48`_

| benchmark | key numbers |
| --- | --- |
| `field_product_kernel_cpu` | components_per_field: 96 · field_l2_evaluations: 3 · component_pair_terms: 165888 · seconds: 0.08 · evaluations_per_s: 35.33 · l2_total: 533.22 |
| `compact_placement_csr_cpu` | components: 600 · query_points: 2048 · tile_size: 16 · nonempty_tiles: 256 · total_entries: 4094 · max_candidates: 27 · retained_payload_bytes: 20480 · component_id_dtype: int32 · evaluated_pairs: 196596 · peak_pair_chunk: 32766 · csr_build_seconds: 0.00 · grouped_seconds: 0.14 · csr_seconds: 0.01 · speedup: 17.39 · max_color_err: 0.00 · max_weight_sum_err: 0.00 · within_contract: 1 |
| `image2gs_fit` | iters_per_s: 62.11 · psnr: 30.30 · seconds: 1.93 |
| `render_ref_cpu` | fps: 303.92 · frames: 36 · seconds: 0.12 |
| `lift_depth` | seconds: 0.05 · init_psnr: 21.00 · init_n_gaussians: 1155 · fit_seconds: 9.32 |
| `e2e_depth` | init_psnr: 21.00 · final_psnr: 32.84 · final_n_gaussians: 3086 · refine_seconds: 22.65 · fit_seconds: 9.32 · lift_seconds: 0.05 · total_seconds: 33.48 · peak_vram_mb: 0.00 · psnr_curve: [(75, 28.19104274113973), (150, 32.84132480621338)] · seconds_curve: [(75, 8.76204867399997), (150, 22.64948220899987)] |
| `lift_hybrid` | seconds: 10.66 · init_psnr: 21.61 · init_n_gaussians: 1732 · fit_seconds: 9.32 |
| `e2e_hybrid` | init_psnr: 21.61 · final_psnr: 33.02 · final_n_gaussians: 4049 · refine_seconds: 31.39 · fit_seconds: 9.32 · lift_seconds: 10.66 · total_seconds: 52.75 · peak_vram_mb: 0.00 · psnr_curve: [(75, 28.212827523549397), (150, 33.0198548634847)] · seconds_curve: [(75, 11.637063362999925), (150, 31.37585457)] |
| `lift_gradient` | seconds: 15.25 · init_psnr: 22.44 · init_n_gaussians: 1729 · fit_seconds: 9.32 |
| `e2e_gradient` | init_psnr: 22.44 · final_psnr: 32.73 · final_n_gaussians: 4101 · refine_seconds: 31.51 · fit_seconds: 9.32 · lift_seconds: 15.25 · total_seconds: 57.81 · peak_vram_mb: 0.00 · psnr_curve: [(75, 28.080894470214844), (150, 32.72951396306356)] · seconds_curve: [(75, 11.33119046999991), (150, 31.507531149999977)] |
| `lift_carve` | seconds: 0.59 · init_psnr: 20.31 · init_n_gaussians: 1396 · fit_seconds: 9.32 |
| `e2e_carve` | init_psnr: 20.31 · final_psnr: 32.99 · final_n_gaussians: 3830 · refine_seconds: 31.24 · fit_seconds: 9.32 · lift_seconds: 0.59 · total_seconds: 42.48 · peak_vram_mb: 0.00 · psnr_curve: [(75, 28.73401403427124), (150, 32.992239475250244)] · seconds_curve: [(75, 13.252550595999992), (150, 31.23293237400003)] |
| `lift_sfm` | seconds: 0.00 · init_psnr: 19.95 · init_n_gaussians: 200 · fit_seconds: 9.32 |
| `e2e_sfm` | init_psnr: 19.95 · final_psnr: 29.18 · final_n_gaussians: 1386 · refine_seconds: 8.87 · fit_seconds: 9.32 · lift_seconds: 0.00 · total_seconds: 18.62 · peak_vram_mb: 0.00 · psnr_curve: [(75, 26.831709225972492), (150, 29.177371978759766)] · seconds_curve: [(75, 2.2976632480001626), (150, 8.872096607999993)] |
| `lift_random` | seconds: 0.00 · init_psnr: 14.11 · init_n_gaussians: 2000 · fit_seconds: 9.32 |
| `e2e_random` | init_psnr: 14.11 · final_psnr: 29.68 · final_n_gaussians: 4428 · refine_seconds: 37.85 · fit_seconds: 9.32 · lift_seconds: 0.00 · total_seconds: 48.60 · peak_vram_mb: 0.00 · psnr_curve: [(75, 26.088585535685223), (150, 29.676211833953857)] · seconds_curve: [(75, 15.795660714000178), (150, 37.82145288300012)] |
<!-- BENCH:END -->

## Reading the numbers

- **Init PSNR** (after lifting, before refinement) is the headline metric for the research
  idea: better init ⇒ fewer refinement iterations to a target quality.
- **Time-to-quality** matters more than final PSNR; final PSNR after long refinement tends
  to converge across initializations.
- CPU numbers (this harness on a laptop/CI) are for *relative* comparisons between
  variants and for catching regressions. Absolute speed claims require the GPU
  configuration (M2 in the roadmap).
