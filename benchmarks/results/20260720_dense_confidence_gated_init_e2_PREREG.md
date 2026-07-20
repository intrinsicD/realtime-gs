# E2 preregistration: easy-only seed plus matched density control

Date frozen: 2026-07-20, after the audited I1 count reproduction and before
any downstream optimization result was opened.

## Hypothesis and arms

The frozen 60-Gaussian easy-only seed, followed by the same Adam and gsplat
DefaultStrategy density-control schedule, will recover the dropped-hard
regions and finish within the repeat-calibrated held-out quality tolerance of
the best of:

1. `topk`: balanced top-K, 172 initial Gaussians;
2. `dense_all`: dense-all + merge, 2,319 initial Gaussians;
3. `easy_only`: frozen I1 gate, 60 initial Gaussians.

A fourth `topk_repeat` execution estimates the control/control envelope. It
uses the same top-K bytes and schedule but seed `20260721` instead of
`20260720`. The three scientific arms all use seed `20260720`.

Initialization PLY SHA-256:

- top-K:
  `d83ee1e764ee6bc0d1cf7696e848df91b0a92d33ad5c9932c9e1138e8564e9fb`
- dense-all:
  `56ce5f1ac3a321f6912506dc4e2c8484c1c3b9d5930eb140b84253faf106cff7`
- easy-only:
  `1d3205755d67e6e3badd48a9d41a1329a38898e6e6178150cac25aadc57b6a9f`

## Frozen source and late-release split

Source frame:
`/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008`.
Calibration SHA-256:
`51b8fc396fc8447f24e325e0a525f2e7d422388790dd9a293e1a81804b265091`.

- Optimization views, in order:
  `C0001, C0008, C0014, C0021, C0026, C0031, C0039`.
- Validation/time-to-quality view: `C1002`.
- Final held-out view, materialized only after every optimization arm and the
  control repeat have completed: `C1004`.
- RGB and mask inputs are undistorted by the repository calibrated loader and
  evaluated at `downscale=8`.
- The harness hashes every selected RGB/mask plus calibration file before
  training and persists the manifest. C1004's bytes/path may be hash-bound
  before training but its image, mask, and camera tensors must not be decoded
  or constructed until the late-release boundary.

This split is independent of optimizer sampling. C1002 and C1004 were absent
from the seven-view compact bundle used to construct all three
initializations. C1002 may be evaluated at fixed checkpoints but never enters
the loss. C1004 is final-only.

## Frozen optimization schedule

All arms use:

- `300` full-image Adam steps at `downscale=8`;
- gsplat rasterizer on CUDA, unpacked, non-antialiased;
- `target_sh_degree=3`, SH interval `75`;
- standard masked 3DGS objective and learning rates, random background,
  `ssim_lambda=0.2`, mask alpha weight `0.05`;
- gsplat `DefaultStrategy`, `absgrad=True`, density start `25`, stop `275`,
  every `25`, gradient threshold `8e-4`;
- split-scale fraction `0.01`, prune opacity `0.005`, prune-scale fraction
  `0.1`, revised opacity enabled, no opacity reset inside the run;
- hard cap `2,319` Gaussians for every arm;
- validation/checkpoint cadence every `50` steps.

The harness resets both the global CPU and CUDA RNGs immediately before each
arm in addition to passing the arm seed to `TrainConfig`. No schedule or
threshold changes are permitted after any arm starts.

The strict compact bundle has no explicit bounds hint. E2 therefore recomputes
the exact E1 seven-camera axis-intersection fallback and passes that frozen
center/extent into every arm; C1002 geometry cannot affect the means learning
rate or density scale thresholds.

## Required evidence

For each arm:

- initial and final PLY plus SHA-256;
- complete training history, density surgery records, sampled-view sequence,
  validation foreground PSNR trajectory, primitive-count trajectory, native
  elapsed trajectory, final count, and peak VRAM;
- final seven-view compact-teacher PSNR/foreground-PSNR/SSIM;
- final training RGB foreground PSNR/crop SSIM/alpha-IoU;
- final C1002 validation and late-release C1004 held-out foreground PSNR,
  crop SSIM, and alpha-IoU;
- viewer command pairing final and initial PLY.

The CPU reference smoke uses a tiny synthetic scene, classic density control,
and the same three input cardinality orderings. It is mechanism-only.

## Decision rule

Let the control envelope be the absolute final C1004 foreground-PSNR
difference between `topk` and `topk_repeat`. The allowed quality deficit is
`min(0.1 dB, control_envelope)`.

Let `best_competitor` be the higher-C1004-foreground-PSNR arm among `topk`
and `dense_all` (ties favor the lower primitive count, then lower native
elapsed time). Easy-only wins only if:

1. its final C1004 foreground PSNR is no worse than
   `best_competitor - allowed_deficit`; and
2. its final primitive count is no greater than the best competitor's; and
3. its native optimization time is no greater than the best competitor's.

Equivalently, a count or time advantage is consequential only inside the
repeat-calibrated equal-quality band. Report the result regardless of sign.

No default changes on an E2 loss. I2/E3 opens only if a held-out regression is
also localized to the regions represented by hard-dropped training
observations; aggregate failure alone is insufficient.
