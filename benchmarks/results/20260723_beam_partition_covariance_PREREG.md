# Masked native-anchor Beam partition covariance on Janelle — frozen protocol

Date frozen: 2026-07-23 (Europe/Berlin)

Repository revision: `c2a7e120a5cafdcf22d4bff6f5b9868b860eb1df` plus the explicitly
hashed experiment files below.

This protocol was written after synthetic mechanism tests and before executing or inspecting any
Janelle outcome from the new masked partition treatment. It is a single-scene, all-fitted-view
development experiment. It cannot authorize a production-default change.

## Question and causal controls

Beam Fusion already retains exact source identities and implied camera-space depths for its
surviving 3D Gaussians. Can the complete original 2D Gaussian density inside each reference mask
be partitioned around those surviving native 2D anchors, then used to estimate wider and more
useful 3D covariances?

No 3D Gaussian is projected into an image to discover, rematch, or move a 2D anchor. The anchors
are exactly the unique `(view, source_component)` identities in Beam Fusion's CSR lineage.

The arms are:

- `ci`: unchanged Beam Fusion covariance-intersection initialization;
- `pou-area`: retain each native contributor's original 2D anisotropy/orientation but scale its
  covariance to match the determinant of its masked partition moment;
- `pou-full`: use the full masked partition second moment, including its anisotropy/orientation.

`pou-area` is the scale/visible-area control. A gain shared by `pou-area` and `pou-full` is
evidence for larger support, not evidence that the full partition shape is useful. A gain unique
to `pou-full` is required to claim value from the partition anisotropy.

## Frozen implementation and pre-run gate

- Partition/refit implementation:
  `src/rtgs/lift/beam_partition.py`,
  SHA-256 `088956887dea77bfeb720714ee937a58ac83ff1166b2c8080fd15c516c0ad196`.
- Beam lineage-depth change:
  `src/rtgs/lift/beam_fusion.py`,
  SHA-256 `575c12fdb59ad7a430178ed5899eb9d546cddc965f50617eeed0b40fe9ca2e12`.
- Harness:
  `benchmarks/beam_partition_covariance.py`,
  SHA-256 `5ac7a63d4b8105b0b0b5f392fc541cf4dbbb7ca660f4fa6f93839b1fc07f817e`.
- Synthetic tests:
  `tests/test_beam_partition.py`,
  SHA-256 `656ab4f6b9d0129bd8e395f990ee67de8c5899b96abd77ce8a46407bf52a6986`.
- Shared covariance diagnostics:
  `benchmarks/beam_covariance_refit.py`,
  SHA-256 `8eb11a50fa9055578139985350ece981861c717e81317445863c9233c890995e`.
- Shared training/evaluation harness:
  `benchmarks/beam_convergence_dynamics.py`,
  SHA-256 `6521af11d0af8513cd6963de260786e37c9791506a0782619b0561045fe2ffa9`.
- Pre-run command:
  `.venv/bin/python -m pytest -q tests/test_beam_partition.py tests/test_beam_fusion.py
  tests/test_beam_covariance_refit.py`
  returned `31 passed`.
- Pre-run lint:
  `.venv/bin/ruff check benchmarks/beam_partition_covariance.py
  src/rtgs/lift/beam_partition.py src/rtgs/lift/beam_fusion.py
  tests/test_beam_partition.py`
  passed.
- No production initializer registration or default is changed.

Any post-freeze edit to a hashed file invalidates the confirmatory label unless it is separately
documented, re-frozen, and rerun from scratch.

## Exact masked partition

For each of the eight selected views:

1. deduplicate Beam Fusion's retained source component ids in that view; these sorted native ids
   and their exact native means are fixed anchors;
2. represent every one of the view's 5,000 original 2D Gaussians by deterministic order-5
   tensor-product Gauss-Hermite quadrature (25 standard-normal samples per Gaussian);
3. weight samples by
   `amplitude * 2*pi*sqrt(det(native_covariance)) * quadrature_weight`;
4. map each continuous sample to its native pixel with `floor(x), floor(y)` and discard it unless
   the exact packed foreground mask is true there;
5. assign each retained sample to its nearest fixed anchor mean in Euclidean native-pixel
   distance; sorted anchor id breaks an exact tie;
6. compute the density-weighted second moment about the *fixed native anchor mean*, not about a
   newly fitted centroid;
7. clamp only the numerical 2D variance floor to `1e-6 px^2`; empty partitions are errors.

Hard nearest-anchor responsibilities sum to one for every retained quadrature sample. Shared
source anchors referenced by multiple 3D tracks are partitioned only once and their one partition
moment is reused; source density is never duplicated.

For `pou-area`, if `C_native` and `C_partition` are the native and partition covariances, use
`sqrt(det(C_partition) / det(C_native)) * C_native`, which exactly matches the 2D determinant.
For `pou-full`, use `C_partition`.

The substituted 2D covariance is lifted with Beam Fusion's original ray, the newly retained exact
per-contributor implied depth, and the original depth-range half-length. Equal-weight CI precision
averaging is repeated without resolving or changing the 3D mean. Three-dimensional standard
deviations retain Beam Fusion's bounds `[1e-4, 0.5 * scene_extent]`.

## Dataset and common Beam initialization

- Compact source:
  `dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d`.
- Manifest SHA-256:
  `b1c8e256d73e2c05f3cb4797a615bdbb2639a637f12908a5c96a2a9a9f912847`.
- Object/capture: Janelle, frame `00008`.
- Selected global view indices:
  `[0, 3, 6, 9, 12, 15, 18, 21]`, corresponding to
  `C0001, C0006, C0012, C0019, C0022, C0028, C0031, C0039`.
- All eight views form Beam tracks, partitions, and training loss.
- Evaluation local indices `[0, 2, 4, 6]` correspond to
  `C0001, C0012, C0022, C0031`; they are fitted, not held out.
- Exact compact Gaussian fields are point-sampled at downscale 32; packed alpha is area
  downsampled for the training/evaluation mask. No source RGB is loaded.
- Seed `0`; exactly `800` retained 3D Gaussians or abort.
- Beam settings:
  `min_views=3`, `transverse_gate_sigma=3.0`, `fold_in_gate_sigma=3.0`,
  `max_color_distance=0.35`, `color_sigma=0.25`,
  `nms_voxel_size=scene_extent/100`, `init_opacity=0.10`, `source_chunk=256`,
  `seed_budget_multiplier=4`.

Every arm must have bit-identical 3D means, opacity, and SH/color fields, the same contributor
lineage and depth arrays, and the same count. Only quaternion/log-scale may differ.

## Common fixed-topology refinement

- `1,000` iterations per arm.
- Pure Torch reference rasterizer on CPU.
- Fixed topology: no clone, split, prune, merge, teleport, or opacity reset.
- Seed `0`; identical view sampler and learning-rate schedules.
- Means learning rate `1.6e-4 * scene_extent`, decaying to 1%; quaternion `1e-3`; scale
  `5e-3`; opacity `5e-2`; SH DC `2.5e-3`; SH rest `1.25e-4`.
- Loss: masked L1 plus `0.2 * D-SSIM`, `0.05 * mask-alpha`, and
  `0.01 * outside-alpha`; black background.
- SH target degree 3, increasing every 33 steps.
- Metrics every 25 steps; initial/final PLYs and exact Torch previews are mandatory.

Primary command:

```bash
.venv/bin/python benchmarks/beam_partition_covariance.py \
  --protocol benchmarks/results/20260723_beam_partition_covariance_PREREG.md \
  --out runs/beam_partition_covariance_20260723
```

Exact repeat:

```bash
.venv/bin/python benchmarks/beam_partition_covariance.py \
  --protocol benchmarks/results/20260723_beam_partition_covariance_PREREG.md \
  --out runs/beam_partition_covariance_20260723_repeat
```

## Frozen validity gates and metrics

The experiment is mechanically valid only if:

1. every view has at least one native anchor and no empty partition;
2. per-view partition-of-unity relative mass error is at most `1e-12`;
3. the native-covariance Beam refit reproduces stored CI covariance with maximum relative
   Frobenius error at most `1e-4`;
4. every treatment covariance is finite and SPD, all arms contain exactly 800 Gaussians, and all
   frozen non-covariance assertions pass.

Report:

- source link count, unique `(view,component)` anchor count, and duplicate-link count;
- per-view unmasked/masked mass, anchor count, partition masses, and determinant scale quantiles;
- 3D sigma/condition distributions;
- covariance reprojection residuals against both original contributor covariance and each arm's
  own partition target;
- initialization/final foreground PSNR, alpha IoU, alpha-inside, alpha-outside;
- trapezoidal fitted-view foreground-PSNR AUC across fixed checkpoints;
- first checkpoint reaching the CI final foreground PSNR.

A treatment has a promising *coverage mechanism* only if all hold:

1. initialization alpha-inside is at least 25% higher and alpha IoU at least 10% higher than CI;
2. initialization foreground PSNR is no more than 0.25 dB below CI;
3. initialization alpha-outside is at most `0.005`.

It has a promising *optimization outcome* only if final alpha IoU is no more than `0.01` below CI
and either foreground-PSNR AUC is at least 1% higher than CI or final foreground PSNR is at least
0.10 dB higher.

The full partition shape adds value beyond support area only if `pou-full` passes both gates and,
relative to `pou-area`, either:

- initialization alpha IoU is at least 2% higher with foreground PSNR no more than 0.10 dB lower
  and alpha-outside no more than `0.001` higher; or
- foreground-PSNR AUC is at least 1% higher or final foreground PSNR is at least 0.10 dB higher,
  with final alpha IoU no more than `0.01` lower.

Passing motivates a multi-scene held-out replication only. Failure is a negative result and
retains CI as the default.

## Integrity, independent audit, and viewer handoff

- Primary output: `runs/beam_partition_covariance_20260723`.
- Exact repeat: `runs/beam_partition_covariance_20260723_repeat`.
- Required artifacts: strict summary JSON, per-arm dynamics JSON, initial/final PLY, and
  initial/final exact Torch preview.
- The repeat must independently reconstruct all initializations. The audit must recompute hashes,
  counts, frozen-field equality, partition validity, trajectory summaries, and every decision
  gate from artifacts.
- Required qualitative comparison:

```bash
.venv-cuda/bin/rtgs view \
  --comparison-manifest benchmarks/results/20260723_beam_partition_covariance_VIEWER.json \
  --max-viewer-gaussians 800 --device cpu --port 8783
```

The server must be HTTP-smoke-tested and then stopped. Orbit inspection is qualitative and cannot
override exact Torch metrics.
