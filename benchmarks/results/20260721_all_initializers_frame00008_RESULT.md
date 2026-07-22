# Full `frame_00008` compact-initializer convergence suite — result

## Outcome

**No materially superior converged initializer was identified.** Dense+merge has the highest
all-fitted-view foreground PSNR, **38.248049 dB**, but beam fusion has the lowest selected
equal-view training objective, **0.002447185**. Dense+merge leads beam by **0.360674 dB** while
having a **4.4003% higher (worse)** objective, so it fails the preregistered requirement that a
winner improve both foreground PSNR by at least 0.10 dB and objective by at least 0.25%.

The fitted-quality Pareto front is therefore `{dense-merge, beam-fusion}`. The frozen practical-
equivalence intersection is empty because no arm is simultaneously within 0.05 dB of the best
PSNR and within 0.25% of the best objective. This is a metric tradeoff, not evidence for a
production winner. Balanced top-K remains the default.

The machine-readable result is
`benchmarks/results/20260721_all_initializers_frame00008_RESULT.json` (SHA-256
`f9e64398f141c53c61816c31ed246285ab9832015199dbb4f9a7f0dd2f436953`). The independent audit is
`benchmarks/results/20260721_all_initializers_frame00008_AUDIT.{md,json}`; the audit JSON SHA-256
is `296093fbf6ee4c5917b97b7a123ea27689c8f85ec9751908126a1cfd8cf45d24`.

## Scope and applicability

This is a prospective single-real-scene **development** comparison on the checked-in compact
bundle. All 26 views were used for placement, fitting, checkpoint selection, and stopping. There
is no held-out, novel-view, multi-scene, or multi-seed evidence. Native initializer counts were
retained, so the raw ranking is count-confounded. The beam arm is a previously completed,
protocol-disclosed historical anchor; the other six arms are prospective under this suite.

| Repository initializer/family | Disposition |
| --- | --- |
| balanced component-center top-K | prospective full fit |
| dense all-eligible + voxel merge | prospective full fit |
| confidence-gated easy-only | prospective full fit |
| calibrated structure-from-splats | prospective full fit |
| complete field lift | prospective full fit |
| camera-bounds random | prospective full fit |
| bounded beam fusion | historical full-fit anchor |
| legacy gradient lift | inapplicable: requires dense RGB photometric targets |
| legacy carve lift | inapplicable: requires dense RGB/color-volume samples |
| depth lift | inapplicable: requires RGB depth inference or supplied depth maps |
| hybrid lift | inapplicable: requires depth plus dense RGB targets |
| classic SfM lift | inapplicable: requires sparse scene points plus RGB for colors |
| internal field fallback | not a public arm; a fallback would be an arm failure |

The compact dataset parent contains only `gaussians2d` plus calibration. No RGB, depth, or sparse
point evidence was fabricated to make an incompatible method appear comparable.

## Frozen setup

- Input: `dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d`.
- Manifest SHA-256:
  `b1c8e256d73e2c05f3cb4797a615bdbb2639a637f12908a5c96a2a9a9f912847`.
- Calibration SHA-256:
  `51b8fc396fc8447f24e325e0a525f2e7d422388790dd9a293e1a81804b265091`.
- Evidence: 26 calibrated views × 5,000 compact 2D Gaussians = 130,000 components, with packed
  alpha; source RGB was not opened by the prospective suite.
- Revision: `d74c9a623cba8af4694e0112753927407c7fdab5`, dirty source preserved in each parent snapshot.
- Environment: Python 3.11.15, PyTorch 2.12.0+cu132, CUDA 13.2 runtime, gsplat 1.5.3,
  StructSplat 0.1.0, RTX 4090, seed 0.
- Protocol SHA-256:
  `217a4fecceca161f4291e78e0e53b201be3e1560e33a875bd29a9fd54534aaf6`.
- Harness SHA-256:
  `47fb0492c646766f88bc2e752870003ba4f8bd45f366880400d60b4183bc4e93`.
- Suite-operator SHA-256:
  `e398817f8b901c98be9177362962c13a6742ac43217d18dc73b04cf0ed9a4f0f`.

Common downstream schedule: 30,000 native-resolution CUDA steps; gsplat DefaultStrategy density
from step 500 through the last update before 15,000, every 100 steps, under a 100,000-Gaussian
cap; masked 3DGS loss with SSIM weight 0.2, mask-alpha weight 0.05, outside-alpha weight 0.01;
degree-3 SH; packed antialiased rendering. Every arm then received non-exact, PLY-reloaded,
fixed-topology 10k segments through the first joint plateau or the 70k ceiling. All seven were
assessed at 70k and met both plateau rules there.

## Initializer parameters and realized counts

- **Top-K:** all 130,000 component centers, 32 depth samples/ray, minimum 2 views, robust-view
  fraction 0.60, score floor 0.01, candidate multiplier 3, exactly 5,000 selected. Placement
  evaluated 2,386,754,916 bounded query pairs over 4,160,000 sampled points; 128,973 candidates
  were eligible.
- **Beam fusion (historical):** all 325 view pairs and 8.125 billion 5k×5k ray pairs; minimum
  3 views; transverse and fold-in gates 3σ; color distance 0.35 and color σ 0.25; source chunk
  256; 0.0223616-unit seed voxel; 20,000 retained seed budget; exactly 5,000 outputs.
- **Dense+merge:** the same 128,973 eligible component-center lifts, no rank trimming, 0.06-unit
  voxel, union opacity, score-weighted moment merge. It produced **2,088** 3D Gaussians.
- **Easy-only:** the identical dense merge followed by minimum view multiplicity 2, RMS spread at
  most 0.50 voxel, half-max width at most 0.20, best covered views at least 2, and reprojection
  residual at most 16 px. It kept **7/2,088** clusters. Failure counts overlap: half-max width
  1,536, reprojection 1,472, view multiplicity 929, and spread 230.
- **Splat-SfM:** all 325 pairs, source chunk 256, minimum 2 views, 3σ epipolar gate, color distance
  0.35, size log-ratio 1.0, ratio test 0.8, reprojection at most 3 px, angle at least 2 degrees.
  It produced **943** tracks: 930 of length 2 and 13 of length 3.
- **Field:** complete `FieldLifter`, native `max_tracks=128`, all 26 views, 32 depth samples,
  40 refit iterations at LR 0.025, appearance from step 20, one topology round. It placed 128,
  then returned **127** after one accepted topology move.
- **Random:** exactly **5,000** gray, isotropic, uniform-volume samples in the camera-derived
  sphere (center `[0.341568, 0.141040, 2.746898]`, extent 2.236157, radius half the extent),
  isotropic scale 0.0653856 and opacity 0.10.

## Primary results

All metrics below are equal-view means over the same 26 **fitted** compact teachers. `N@15k` is
the topology after the density window and equals the selected final count because later segments
were fixed-topology. Placement and optimizer times are unrepeated, sequential, contended local
diagnostics—not benchmarks.

| Initializer | Place s | Init N | Init FG dB | Init SSIM | Init α-IoU | N@15k/final | Selected | Final FG dB | Final crop dB | Final SSIM | Final α-IoU | Objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top-K | 99.700 | 5,000 | 11.862928 | 0.771551 | 0.267402 | 43,288 | 70k | 37.299174 | 42.266771 | 0.995143 | 0.974146 | 0.002742345 |
| beam fusion† | 138.326 | 5,000 | 11.582614 | 0.729672 | 0.001988 | 44,222 | 69k | 37.887375 | 42.854971 | **0.995821** | 0.976061 | **0.002447185** |
| dense+merge | 104.763 | 2,088 | **20.754629** | **0.961306** | **0.517816** | 49,177 | 70k | **38.248049** | **43.215646** | 0.995559 | **0.976468** | 0.002554868 |
| easy-only | 109.700 | 7 | 11.046278 | 0.653397 | 0.000000 | 35,644 | 69k | 36.958743 | 41.926338 | 0.994749 | 0.974181 | 0.002905327 |
| splat-SfM | 195.111 | 943 | 11.757629 | 0.727168 | 0.000190 | 39,987 | 69k | 37.706291 | 42.673888 | 0.995128 | 0.975190 | 0.002759203 |
| field | 1,068.385 | 127 | 11.469409 | 0.717424 | 0.000547 | 39,059 | 70k | 37.240826 | 42.208422 | 0.995081 | 0.974875 | 0.002765665 |
| random | 0.002 | 5,000 | 11.176665 | 0.884538 | 0.293215 | 39,513 | 70k | 37.425717 | 42.393312 | 0.995276 | 0.975730 | 0.002680195 |

† Historical anchor disclosed before the six prospective outcomes were opened.

Initial foreground-PSNR order:
`dense-merge > top-K > splat-SfM > beam > field > random > easy-only`.

Final foreground-PSNR order:
`dense-merge > beam > splat-SfM > random > top-K > field > easy-only`.

The optimizer times ranged from 1,975.2 to 2,197.6 seconds; peak reported VRAM ranged from 3.764
to 3.801 GiB. Checkpoint-callback accounting ranged from 0.719 to 0.908 seconds. Those numbers
describe these executions only; the machine was not isolated and the arm order was not randomized.

## Method-specific observations

- Dense+merge is the clear initialization-quality leader on this scene, gaining **8.891700 dB**
  over top-K with fewer initial Gaussians. Its union-opacity merge changes both geometry and alpha,
  so this does not isolate which part causes the gain.
- Splat-SfM retained 943/10,973 prefilter tracks from 405,436 pair matches and 111,582 accepted
  union edges. Mean/max reprojection error was 0.87195/2.98633 px, mean triangulation angle
  89.8159 degrees, and mean covariance residual 0.169944. Each view still had 4,867–4,962
  unmatched splats.
- Field accepted all 40 continuous refit steps; all 127 outputs had observable rank 6 and maximum
  condition 2.3109. Source projection/color errors were at most `2.18e-11`/`2.22e-16`. Its sampled
  train-only semantic validation was density MSE 9.80293 and RGB MSE 0.0330586 over 3,328 points.
  No held-out semantic set existed because all views were fit.
- Random finishing fourth in foreground PSNR, and the 7-Gaussian easy-only seed reaching
  36.9587 dB after growing to 35,644, demonstrate that ordinary adaptive density can recover much
  of the fitted-view target regardless of initialization. They do not prove initialization is
  irrelevant for held-out geometry, convergence speed, other scenes, or smaller budgets.
- Initial count does not predict terminal count or quality here. Density expanded every arm to
  35.6k–49.2k, so final recovery cannot be causally credited to an initializer alone.

## Audit limitation affecting field

The field execution saved its final PLY/hash and aggregate accounting of seven topology proposals
with one acceptance, but the harness omitted the seven individual move receipts required by the
protocol. This is a reporting defect, not a quality-artifact failure: field initialization/final
quality and counts remain auditable, while move-level topology utility does not.

## Exact command

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv-cuda/bin/python benchmarks/run_compact_initializer_suite.py \
  --out runs/all_initializers_frame00008_20260721 \
  --protocol benchmarks/results/20260721_all_initializers_frame00008_PREREG.md \
  --keep-going
```

The operator was resume-safe and ran the frozen order
`topk, dense-merge, easy-only, splat-sfm, field, random`. It did not invoke the source-RGB
evaluation phase. No viewer ran during measured placement or optimization.

## Post-result viewer handoff

After the suite and audit completed, the fitted-PSNR leader was opened with its own initialization:

```bash
.venv-cuda/bin/rtgs view \
  --gaussians runs/all_initializers_frame00008_20260721/dense-merge/settle_60000_70000/gaussians_final.ply \
  --initial runs/all_initializers_frame00008_20260721/dense-merge/fit_0_30000/gaussians_init.ply \
  --max-viewer-gaussians 50000 --rasterizer torch --device cpu \
  --host 127.0.0.1 --port 8781 --no-open
```

The server returned HTTP 200 at `http://127.0.0.1:8781`; its launch sample used about 578 MiB RSS,
and `nvidia-smi` listed no compute process. That confirms a CPU-only viewer-server path, not zero
overhead. A local browser's WebGL process may still use the display GPU. For a future live fit,
add `--watch-checkpoints <active-phase>/checkpoints`; the suite's separate continuation phase
directories require repointing the watcher at each boundary.
