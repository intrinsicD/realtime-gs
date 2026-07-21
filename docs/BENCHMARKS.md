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
_Last run: 2026-07-21T19:14:24+00:00 · device `cpu` · torch 2.13.0+cpu · rev `892f448` · scene `synthetic_g40_c12_s48`_

| benchmark | key numbers |
| --- | --- |
| `field_product_kernel_cpu` | components_per_field: 96 · field_l2_evaluations: 3 · component_pair_terms: 165888 · seconds: 0.10 · evaluations_per_s: 29.80 · l2_total: 533.22 |
| `compact_placement_csr_cpu` | components: 600 · query_points: 2048 · tile_size: 16 · nonempty_tiles: 256 · total_entries: 4094 · max_candidates: 27 · retained_payload_bytes: 20480 · component_id_dtype: int32 · evaluated_pairs: 196596 · peak_pair_chunk: 32766 · csr_build_seconds: 0.00 · grouped_seconds: 0.19 · csr_seconds: 0.01 · speedup: 21.52 · max_color_err: 0.00 · max_weight_sum_err: 0.00 · within_contract: 1 |
| `image2gs_fit` | iters_per_s: 54.21 · psnr: 30.30 · seconds: 2.21 |
| `image2gs_fit_batched` | views: 12 · seconds: 7.58 · serial_seconds: 11.18 · speedup_vs_serial: 1.48 · psnr_mean: 28.28 · serial_psnr_mean: 28.28 |
| `render_ref_cpu` | fps: 273.27 · frames: 36 · seconds: 0.13 |
| `lift_depth` | seconds: 0.05 · init_psnr: 21.00 · init_n_gaussians: 1155 · fit_seconds: 11.00 |
| `e2e_depth` | init_psnr: 21.00 · final_psnr: 32.84 · final_n_gaussians: 3086 · refine_seconds: 24.76 · fit_seconds: 11.00 · lift_seconds: 0.05 · total_seconds: 37.18 · peak_vram_mb: 0.00 · psnr_curve: [(75, 28.19104274113973), (150, 32.84132480621338)] · seconds_curve: [(75, 8.859514996999678), (150, 24.693827665999834)] |
| `lift_hybrid` | seconds: 11.51 · init_psnr: 21.61 · init_n_gaussians: 1732 · fit_seconds: 11.00 |
| `e2e_hybrid` | init_psnr: 21.61 · final_psnr: 33.02 · final_n_gaussians: 4049 · refine_seconds: 37.88 · fit_seconds: 11.00 · lift_seconds: 11.51 · total_seconds: 61.85 · peak_vram_mb: 0.00 · psnr_curve: [(75, 28.212827523549397), (150, 33.0198548634847)] · seconds_curve: [(75, 13.75669389099994), (150, 37.705645754000216)] |
| `lift_gradient` | seconds: 18.00 · init_psnr: 22.44 · init_n_gaussians: 1729 · fit_seconds: 11.00 |
| `e2e_gradient` | init_psnr: 22.44 · final_psnr: 32.73 · final_n_gaussians: 4101 · refine_seconds: 38.24 · fit_seconds: 11.00 · lift_seconds: 18.00 · total_seconds: 68.89 · peak_vram_mb: 0.00 · psnr_curve: [(75, 28.080894470214844), (150, 32.72951396306356)] · seconds_curve: [(75, 13.834418529999766), (150, 38.0713196480001)] |
| `lift_carve` | seconds: 0.28 · init_psnr: 20.31 · init_n_gaussians: 1396 · fit_seconds: 11.00 |
| `e2e_carve` | init_psnr: 20.31 · final_psnr: 32.99 · final_n_gaussians: 3830 · refine_seconds: 36.57 · fit_seconds: 11.00 · lift_seconds: 0.28 · total_seconds: 49.34 · peak_vram_mb: 0.00 · psnr_curve: [(75, 28.73401403427124), (150, 32.992239475250244)] · seconds_curve: [(75, 14.30232719300011), (150, 36.44221557599985)] |
| `lift_sfm` | seconds: 0.00 · init_psnr: 19.95 · init_n_gaussians: 200 · fit_seconds: 11.00 |
| `e2e_sfm` | init_psnr: 19.95 · final_psnr: 29.18 · final_n_gaussians: 1386 · refine_seconds: 9.77 · fit_seconds: 11.00 · lift_seconds: 0.00 · total_seconds: 21.23 · peak_vram_mb: 0.00 · psnr_curve: [(75, 26.831709225972492), (150, 29.177371978759766)] · seconds_curve: [(75, 3.022302100000161), (150, 9.769549164000182)] |
| `lift_random` | seconds: 0.00 · init_psnr: 14.11 · init_n_gaussians: 2000 · fit_seconds: 11.00 |
| `e2e_random` | init_psnr: 14.11 · final_psnr: 29.68 · final_n_gaussians: 4428 · refine_seconds: 42.41 · fit_seconds: 11.00 · lift_seconds: 0.00 · total_seconds: 55.00 · peak_vram_mb: 0.00 · psnr_curve: [(75, 26.088585535685223), (150, 29.676211833953857)] · seconds_curve: [(75, 17.1123046040002), (150, 42.286336498999844)] |
<!-- BENCH:END -->

## Reading the numbers

- **Init PSNR** (after lifting, before refinement) is the headline metric for the research
  idea: better init ⇒ fewer refinement iterations to a target quality.
- **Time-to-quality** matters more than final PSNR; final PSNR after long refinement tends
  to converge across initializations.
- CPU numbers (this harness on a laptop/CI) are for *relative* comparisons between
  variants and for catching regressions. Absolute speed claims require the GPU
  configuration (M2 in the roadmap).
