# Preregistration: full `frame_00008` compact-initializer convergence suite

Date frozen: 2026-07-21, before opening any new full-26-view placement or downstream result from
the six prospective arms below.

Scientific status: **prospective descriptive comparison with historical anchors; not a fresh
confirmatory test of beam fusion and not authority for a production-default change**.

## Question and chronology

On the same checked-in 26-view compact capture, how do all repository initialization families
that can legitimately consume the available evidence compare (a) before optimization and (b)
after the identical ordinary 3DGS density/convergence schedule?

The full beam-fusion outcome and the top-K initialization-only control were already observed under
the separately frozen beam protocol. The new downstream top-K, dense+merge, easy-only,
splat-SfM, field, and random results have not been opened. Earlier seven-view/640-splat dense and
easy-only experiments and idealized synthetic splat-SfM/field tests are also known. Consequently:

- the six new executions are prospective under this document;
- the combined seven-arm ranking is development evidence, not an unbiased confirmatory selection;
- all views are fit, so no held-out, novel-view, or generalization language is permitted;
- no timing number from the sequential operator is a portable performance benchmark.

Historical anchors, fixed before this protocol:

- beam preregistration SHA-256:
  `1fa29697ccc729e4caab4a0dff4e8528d244cced13c32da53d73f24aa7c7a126`;
- beam result JSON SHA-256:
  `63afd7534f1fe7fe5d186788f24f6e8c147e487ce68b2e1ab5f9a482f4ab293d`;
- known beam outcome: 5,000 initialized, 44,222 selected, selected step 69,000;
- known top-K initialization-only outcome: 5,000 initialized; no downstream top-K result was
  available when this protocol was frozen.

## Frozen input and isolation

Input:
`dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d`.

- manifest SHA-256:
  `b1c8e256d73e2c05f3cb4797a615bdbb2639a637f12908a5c96a2a9a9f912847`;
- calibration SHA-256:
  `51b8fc396fc8447f24e325e0a525f2e7d422388790dd9a293e1a81804b265091`;
- 26 ordered cameras, 5,000 optimized 2D Gaussians per view, 130,000 total;
- all indices `0..25` enter placement when the method supports all-view evidence and all 26
  compact teacher crops enter downstream fitting;
- compact teachers and lossless packed alpha are the only image evidence;
- source RGB, source masks outside the bundles, monocular-depth models, sparse point clouds, and
  external SfM products must not be opened by the suite operator;
- no final original-RGB evaluation phase will run for this comparison.

The harness writes provenance and an executed-source snapshot before placement. Every parent
must hash-bind this protocol. Revision at freeze:
`d74c9a623cba8af4694e0112753927407c7fdab5` with a dirty worktree whose exact bytes are preserved
per run. Frozen pre-outcome implementation SHA-256 values:

- `benchmarks/full_compact_reconstruction.py`:
  `47fb0492c646766f88bc2e752870003ba4f8bd45f366880400d60b4183bc4e93`;
- `benchmarks/run_compact_initializer_suite.py`:
  `e398817f8b901c98be9177362962c13a6742ac43217d18dc73b04cf0ed9a4f0f`;
- compact carve:
  `8e735ddfa91d9ebe9e218707bd83b6392a3428e158b31ecc3e39f1898bd9e404`;
- confidence gate:
  `a19facc5076a43b8e116ce1d2f66641094caec61d2e77688040b05c7f74013dc`;
- splat-SfM:
  `1a6c01718f680c33aad9367385bee77a3a768f76abfe861173a20e2802b36b03`;
- field lift:
  `cc3e7cf77d8b2298533f2e8c9b04f343f270d952d6ffb4b4cd14347554433107`;
- random baseline:
  `ec293cdc7c60bdeddb1dd8392eba89d068285bd5a86188d2821da5c1057aacdc`.

## Complete repository-method applicability inventory

The word “all” means every implementation receives an explicit disposition. A method is executed
only when its documented inputs exist; fabricating an RGB image, depth map, or point cloud from a
different source would cease to be the same compact-data experiment.

| Repository method/family | Compact evidence only? | Disposition in this suite |
|---|---|---|
| component-center balanced top-K / `CompactCarveInitializer` | yes | new full fit |
| bounded beam fusion | yes | historical full fit, reused without rerun |
| dense all-eligible + voxel merge | yes | new full fit |
| frozen easy-only confidence gate after dense merge | yes | new full fit |
| calibrated `structure_from_splats` | yes | new full fit if its frozen gates return tracks |
| complete `FieldLifter.fit` | yes | new full fit at its bounded native 128-track scale |
| random baseline | uses only bounds | new 5,000-Gaussian lower-bound fit |
| legacy `GradientLifter` | no: dense RGB photometric targets | inapplicable, no execution |
| legacy `CarveLifter` | no: dense RGB/color-volume samples | inapplicable; compact carve above is its valid compact analog |
| `DepthLifter` | no: RGB depth inference or supplied depth maps | inapplicable, no execution |
| `HybridLifter` | no: depth plus dense RGB photometric correction | inapplicable, no execution |
| classic `SfMLifter` | no: `scene.points` plus view RGB for color | inapplicable, no execution |
| internal emergency field-placement fallback | not a public method | not an arm; any fallback in a public arm is a failure |

The inapplicable methods may be tested later in a separately named RGB/depth/SfM cohort. Their
absence cannot support a claim that a compact-native arm beats them.

## Prospective arms and frozen placement parameters

All arms use seed 0, float32 3D Gaussians, initial opacity 0.10, camera-derived center/extent when
no trusted bounds hint exists, and no post-hoc cardinality trimming. Initial count is an outcome
for native variable-count methods and must be reported before final quality.

1. **`topk`**
   - enumerate all 130,000 component centers;
   - request exactly 5,000 tracks;
   - 32 midpoint depth samples per ray, minimum 2 views, robust-view fraction 0.60, score floor
     0.01, candidate multiplier 3;
   - fallback forbidden. Expected count is exactly 5,000 or the arm fails.

2. **`dense-merge`**
   - same component-center scoring parameters as top-K;
   - retain every eligible ray rather than rank-trimming;
   - merge at an absolute 0.06-world-unit voxel, union opacity, score-weighted moment merge;
   - no cap or selection after merge. Report eligible, merged, member-count, and distinct-view
     multiplicity histograms.

3. **`easy-only`**
   - reproduce the exact dense placement and 0.06 merge above;
   - apply frozen `ClusterConfidenceConfig`: minimum view multiplicity 2, maximum RMS spread 0.50
     voxels, maximum half-max depth width 0.20, minimum best covered views 2, maximum reprojection
     residual 16 px;
   - no post-gate count matching. Report every failure-category count and kept/dropped totals.

4. **`splat-sfm`**
   - all 325 calibrated view pairs, no pair limit, source chunk 256 (arithmetic-preserving memory
     bound), minimum 2 views;
   - defaults otherwise: near 0.05, bounds scale 0.5, epipolar gate 3σ, color distance 0.35,
     size log-ratio 1.0, weights 1/1/0.5, ratio test 0.8, reprojection gate 3 px,
     triangulation angle at least 2 degrees, sigma range `1e-4` to half extent;
   - no gate relaxation if no tracks survive. No tracks is an informative arm failure.

5. **`field`**
   - complete public field path rather than its top-K placement subroutine;
   - native bounded `max_tracks=128`, all 26 training views, 32 depth samples, candidate
     multiplier 3, minimum 2 views, robust fraction 0.60, score floor 0.01;
   - `FieldRefitConfig` defaults: 40 iterations, learning rate 0.025, appearance from step 20,
     degree-1 source-anchored SH, chunk 256, visibility refresh 5;
   - one default transactional topology round, no background tracks because packed alpha exists;
   - fallback forbidden. Report placement count, final field-lift count, refit objective history,
     topology receipts, and semantic validation.

6. **`random`**
   - exactly 5,000 points uniform by volume in the sphere of radius half the camera-derived
     extent; seed 0;
   - isotropic scale `0.5 * extent / 5000^(1/3)`, gray `[0.5,0.5,0.5]`, opacity 0.10;
   - compact observations are unused in placement and remain the downstream targets.

7. **historical `beam-fusion` anchor**
   - exactly the prior frozen 5,000-Gaussian result; do not rerun or alter it;
   - its known placement and convergence receipts enter the final descriptive table with a clear
     historical label.

If any native initialization exceeds the frozen 100,000 downstream hard cap, it is reported as
incompatible with this training budget and is not trimmed. If a method raises under frozen gates,
the failure and complete traceback/log are retained; thresholds are not changed.

## Common downstream fit and convergence

Every successful prospective arm uses the existing full compact reconstruction harness with:

- all 26 native compact fit-window targets and packed alpha;
- CUDA gsplat, packed and antialiased, black background;
- 30,000-step Adam parent, seed 0, evaluation/checkpoint every 1,000;
- standard masked 3DGS objective: SSIM lambda 0.2, mask-alpha lambda 0.05, outside-alpha lambda
  0.01;
- target SH degree 3, promoted every 1,000 steps;
- gsplat DefaultStrategy with abs-grad and revised opacity;
- density start 500, stop 15,000, every 100, gradient threshold `8e-4`, split-scale fraction
  0.01, prune opacity 0.005, prune-scale fraction 0.1, hard cap 100,000;
- the same existing learning rates and 0.01 final means-LR factor.

Every successful 30k parent receives at least the 30k→40k fixed-topology polish. If the existing
joint convergence receipt says `still_improving`, run 40k→50k tail, then 50k→60k cooldown, then
60k→70k settle as needed. Each boundary reloads PLY parameters but not Adam moments, per-parameter
step counters, or RNG state, so all continuations are explicitly non-exact. Segment seeds and LR
rules remain the already frozen harness values 1/2/3/4. Stop at the first `plateau` receipt or at
70k; never extend or retune an arm after seeing its quality.

Within each segment, model selection uses the equal-view mean frozen compact training objective;
candidates within relative `1e-6` of the minimum choose the earliest step. The joint plateau
requires both existing tests:

1. five trailing 1k transitions with neither at least 0.25% objective reduction nor at least
   0.05 dB foreground-PSNR gain; and
2. the last-six-window robust trend/median rule already implemented and tested in the harness.

The prospective arm order is fixed as `topk`, `dense-merge`, `easy-only`, `splat-sfm`, `field`,
`random`. The operator may continue after a placement failure but may not reorder arms in response
to quality. Sequential wall time, peak memory, and placement time are diagnostics only because the
machine is not isolated and order is not randomized.

## Required reporting and conclusion rule

For every arm, report:

- applicability and input modality;
- exact placement config, placement seconds, input component count, initialized 3D count, and
  lineage/track/cluster diagnostics available for that method;
- initial all-fitted-view foreground PSNR, crop PSNR, crop SSIM, alpha IoU/inside/outside;
- Gaussian count at density stop, selected count, selected global step, total executed steps,
  convergence status, selected compact objective, foreground PSNR, crop SSIM, and alpha metrics;
- optimizer elapsed time, placement time, peak VRAM, checkpoint callback cost, and artifact hashes,
  all labeled local diagnostics;
- failures without imputation.

Initial and final rank are reported separately. Because counts are intentionally native, a raw
quality rank is count-confounded; counts must appear in the same table.

A converged arm is **materially better in fitted compact quality** than another only when its
selected foreground PSNR is at least 0.10 dB higher **and** its selected equal-view objective is at
least 0.25% lower. If no arm dominates the best competitor on both gates, conclude that this
single-scene suite does not identify a materially superior converged initializer under adaptive
density. If an arm wins both gates, call it the scene-specific development winner only; do not
change a default without a held-out, multi-seed, multi-scene protocol.

For practical equivalence, list arms within 0.05 dB foreground PSNR and 0.25% objective of the
best. Primitive count and local time may describe tradeoffs inside that set but cannot decide a
portable speed claim. Initial quality cannot stand in for downstream quality, and downstream
quality cannot be attributed to initialization when density growth changes cardinality materially.

## Viewer policy

No viewer runs during measured placement or optimizer execution. Checkpoint PLY writing remains
enabled and its callback time is recorded. After the suite reaches terminal state, launch one
CPU-only viewer comparing the selected development winner with its own initialization, with
`--max-viewer-gaussians` at least the larger count. This gives visual inspection without CUDA
allocation, but it is not “zero impact”: CPU, RAM, HTTP, and file I/O remain nonzero. A claim about
viewer overhead would require randomized on/off repetitions and is outside this protocol.

## Frozen command

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv-cuda/bin/python benchmarks/run_compact_initializer_suite.py \
  --out runs/all_initializers_frame00008_20260721 \
  --protocol benchmarks/results/20260721_all_initializers_frame00008_PREREG.md \
  --keep-going
```

The operator writes `suite_status.json`, never invokes `--phase evaluate`, and refuses to
overwrite a partial phase directory.
