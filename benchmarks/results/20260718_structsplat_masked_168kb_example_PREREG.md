# Preregistration: masked StructSplat example under 168,000 bytes

Date frozen: 2026-07-18 (Europe/Berlin)

This is a single-view feasibility and presentation run. It does not select a repository default,
claim that all dataset views meet the size cap, or make a 3D/novel-view claim.

## Frozen input

- View: `C0014`
- RGB: `dataset/2025_03_07_stage_with_fabric/frame_00008/rgb/C0014.jpg`
- Mask: `dataset/2025_03_07_stage_with_fabric/frame_00008/mask/mask_C0014.png`
- Calibration and preprocessing: the source-bound native-resolution path in
  `benchmarks/stage1_capacity_sweep.py`
- Output: `runs/structsplat_masked_168kb_example_20260718/`

## Frozen method

- Undistort RGB bilinearly and the PNG mask with nearest-neighbor sampling, threshold the mask at
  `> 0.5`, and crop to its tight calibrated bounding rectangle.
- Compute StructSplat's structure tensor and initialization density on the masked RGB crop,
  multiply the density by the binary mask, renormalize, and initialize 5,000 `aniso_onedge`
  Gaussians with WSE sampling. Background-layer initialization is disabled.
- Optimize a fixed-count field for 1,000 Adam steps with StructSplat's native `cuda_tiled`
  renderer. The objective is `0.7 * foreground L1 + 0.3 * foreground-mask-normalized SSIM`;
  no background pixel enters either term.
- Project any optimized rounded center that exits the foreground back to its nearest foreground
  pixel and clear that center's Adam momentum.
- Use compact support with `sigma_cutoff=3` and `support_fade_alpha=1`.
- Freeze the exact native field as one integrity-checked
  `rtgs.gaussian_observation_field.v1` archive. Do not serialize source RGB or mask arrays.

## Pass/fail gates

The run passes only if:

1. all initial and final rounded Gaussian centers are inside the foreground mask;
2. the strict archive reload succeeds and has no member named as RGB, image, target, or mask;
3. the exact archive size is at most **168,000 bytes**;
4. current StructSplat `cuda_tiled` playback of the strict reload matches the live terminal render
   at maximum absolute error `<= 1e-5` and mean absolute error `<= 1e-6`;
5. the report includes foreground PSNR/weighted SSIM, raw archive-render versus masked-crop
   PSNR/SSIM, foreground/background support activity, and outside-mask RGB leakage;
6. the comparison panel shows the raw unmasked archive render. A separate mask-composited cell
   may be shown only when explicitly labeled as a foreground diagnostic.

No minimum quality threshold is frozen: the purpose is to let the user judge whether the first
strict-size example is visually sufficient. The 168,000-byte cap applies to the serialized
Gaussian archive, not to the diagnostic PNG panel.

## Interpretation boundary

A foreground-only normalized RGB Gaussian field does not contain an explicit alpha channel.
Therefore applying the mask before fitting prevents background scene content from becoming
Gaussian tokens, but it does not guarantee an exact hard silhouette after the mask is discarded.
Raw-render leakage is a decision-relevant result and must not be hidden by compositing the source
mask into the reported playback.
