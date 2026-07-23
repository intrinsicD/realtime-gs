# Beam-track covariance refit on Janelle — frozen protocol

Date frozen: 2026-07-23 (Europe/Berlin)

Repository revision: `c2a7e120a5cafdcf22d4bff6f5b9868b860eb1df` plus the explicitly
hashed experiment-only files below.

This protocol was written after synthetic mechanism tests but before executing or inspecting the
Janelle treatment outcomes. It is a single-scene, all-fitted-view development experiment. It
cannot authorize a production-default change.

## Question and hypotheses

Beam Fusion already produces a CSR lineage from each retained 3D component to at most one 2D
Gaussian per contributing view. Does using those same correspondences to estimate the 3D
covariance improve:

1. covariance reprojection agreement with the contributing 2D Gaussians;
2. visible initialization coverage; and
3. convergence under otherwise identical 3DGS optimization?

The unchanged covariance-intersection result is the `ci` control. The two treatments are:

- `track-lsq`: the Splat-SfM linear covariance triangulation applied to Beam Fusion's contributor
  tracks, followed by a bounded SPD projection;
- `track-robust`: `track-lsq` followed by robust, observation-whitened gradient refinement of a
  Cholesky-factorized SPD covariance, with a weak CI prior.

The expected ordering for the direct covariance mechanism is
`track-robust <= track-lsq < ci` in whitened reprojection residual. Whether that translates into
coverage or optimization improvement is explicitly unknown.

## Frozen implementation

- Harness:
  `benchmarks/beam_covariance_refit.py`,
  SHA-256 `ebae0f5c6697690d3c8c92bc3657dffe1e6206a32970e5894e86fb603bc93336`.
- Synthetic tests:
  `tests/test_beam_covariance_refit.py`,
  SHA-256 `75db5bf9f56ebb19db6a664e43bac9e8d4b8fb0c475fe652c48d6427c23efb51`.
- Pre-run mechanism gate:
  `.venv/bin/python -m pytest -q tests/test_beam_covariance_refit.py` returned `2 passed`.
- No production initializer/configuration is changed by this experiment.

If either file changes after this freeze, the result must record the new hash and explain why; a
scientific conclusion requires re-freezing or treating the run as exploratory.

## Dataset and view roles

- Compact source:
  `dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d`.
- Manifest SHA-256:
  `b1c8e256d73e2c05f3cb4797a615bdbb2639a637f12908a5c96a2a9a9f912847`.
- Object/capture: Janelle, frame `00008`.
- Selected global view indices:
  `[0, 3, 6, 9, 12, 15, 18, 21]`, corresponding to
  `C0001, C0006, C0012, C0019, C0022, C0028, C0031, C0039`.
- All eight views construct the Beam Fusion tracks, covariance treatments, and training loss.
- Metrics are evaluated every 25 steps on local indices `[0, 2, 4, 6]`, corresponding to
  `C0001, C0012, C0022, C0031`. These are fitted views, not held-out views.
- Exact compact Gaussian observation fields are point-sampled at downscale 32; packed alpha is
  area-downsampled and used as the mask. No source RGB is loaded.

## Common Beam Fusion initialization

- Seed: `0`.
- Output count: exactly `800` 3D Gaussians; abort if the cap is not reached.
- `min_views=3`.
- `transverse_gate_sigma=3.0`.
- `fold_in_gate_sigma=3.0`.
- `max_color_distance=0.35`.
- `color_sigma=0.25`.
- `nms_voxel_size = scene_extent / 100`.
- `init_opacity=0.10`.
- `source_chunk=256`.
- `seed_budget_multiplier=4`.

All arms must have identical means, contributor assignments, SH/color coefficients, opacity, and
count at initialization. The harness asserts means, opacity, and SH bit-for-bit. Only quaternion
and log-scale fields may differ, as the representation of the treatment covariance.

## Covariance treatments

The measured 2D covariance uses each contributor's effective 2D variances and rotation in native
pixel coordinates. For a fixed 3D Beam Fusion mean and camera projection Jacobian `J_v`, both
treatments compare `J_v Sigma_3D J_v^T` against the corresponding measured `Sigma_2D,v`.

### `track-lsq`

- Solve the six independent entries of a symmetric 3D covariance by masked batched pseudoinverse,
  exactly as the current private Splat-SfM covariance solver does.
- Symmetrize and eigendecompose.
- Clamp 3D standard deviations to `[1e-4, 0.5 * scene_extent]`.
- Reconstruct quaternion/log-scale without changing any other field.

### `track-robust`

- Initialize from the bounded `track-lsq` covariance.
- Lower-triangular Cholesky parameterization with log diagonals, so covariance stays SPD.
- Objective per valid contributor:
  RMS Frobenius error of
  `Sigma_2D^(-1/2) (J Sigma_3D J^T) Sigma_2D^(-1/2) - I`.
- Huber delta: `0.25`.
- Optimizer: float64 Adam, `120` steps, learning rate `0.03`.
- Weak relative-Frobenius CI prior weight: `1e-3`.
- Eigenvalue-bound penalty weight: `1e-3`.
- Gradient-norm clip: `10`.
- Final standard-deviation clamp: `[1e-4, 0.5 * scene_extent]`.

## Common 3DGS refinement

- `1,000` iterations per arm.
- Pure Torch reference rasterizer on CPU.
- Fixed topology: no clone, split, prune, merge, teleport, or opacity reset.
- Seed `0`; same view sampler and learning-rate schedules in every arm.
- Learning rates:
  means `1.6e-4 * scene_extent` decaying to 1%, quaternion `1e-3`, scale `5e-3`,
  opacity `5e-2`, SH DC `2.5e-3`, SH-rest `1.25e-4`.
- Loss:
  masked L1 plus `0.2 * D-SSIM`, `0.05 * mask-alpha`, and `0.01 * outside-alpha`;
  black, non-random background.
- SH target degree 3, increasing every 33 steps.
- Metrics every 25 steps; initial and final PLYs and exact Torch previews are mandatory.

Command:

```bash
.venv/bin/python benchmarks/beam_covariance_refit.py \
  --protocol benchmarks/results/20260723_beam_covariance_refit_PREREG.md \
  --out runs/beam_covariance_refit_20260723
```

## Frozen metrics and decision rules

Direct covariance diagnostics use every valid contributor link:

- relative Frobenius covariance reprojection residual;
- observation-whitened RMS covariance reprojection residual;
- 3D sigma and condition-number distributions;
- raw LS non-SPD count before bounding.

Visual/pipeline metrics use the exact Torch evaluator:

- initialization and final fitted-view foreground PSNR;
- initialization and final alpha IoU and alpha-inside;
- fitted-view foreground-PSNR trajectory and trapezoidal AUC across the fixed checkpoints;
- iteration at which a treatment first reaches the control's final foreground PSNR, if it does.

A treatment is a promising covariance mechanism only if all of the following hold:

1. its median whitened covariance residual is at least 20% lower than `ci`;
2. initialization alpha IoU is at least 10% higher than `ci`, while initialization foreground
   PSNR is no more than 0.25 dB worse;
3. either foreground-PSNR AUC is at least 1% higher than `ci` or final foreground PSNR is at least
   0.10 dB higher, while final alpha IoU is no more than 0.01 below `ci`.

`track-robust` is preferred over the simpler `track-lsq` only if its median whitened residual is
at least 5% lower and it also has the better pipeline outcome under rule 3. Otherwise LSQ remains
the simpler candidate. Because this is one deterministic development scene, passing these rules
only motivates a multi-scene held-out replication; it does not change defaults. Failing any rule
is a negative result and must be logged as such.

## Integrity, audit, and viewer handoff

- Output directory:
  `runs/beam_covariance_refit_20260723`.
- A strict summary JSON, per-arm dynamics JSON, initial/final PLY, and initial/final preview are
  required.
- The independent audit must recompute field equality, PLY counts/hashes, covariance diagnostics,
  trajectory summaries, and all frozen gates from artifacts.
- Required qualitative comparison command:

```bash
.venv/bin/rtgs view \
  --comparison-manifest benchmarks/results/20260723_beam_covariance_refit_VIEWER.json \
  --max-viewer-gaussians 800 --device cpu --port 8782
```

- The server must be HTTP-smoke-tested. Orbit inspection is qualitative and cannot override the
  exact Torch metrics.
