# New opt-in variants on Janelle `frame_00008` — result

Date: 2026-07-24

Status: **pool passes its stage-1 and downstream development gates; containment fails its local
gate but passes downstream; no default change**

Frozen protocol chain:
[`v1`](20260724_new_variants_frame00008_PREREG.md),
[`v2`](20260724_new_variants_frame00008_PREREG_V2.md), and
[`v3`](20260724_new_variants_frame00008_PREREG_V3.md).

Machine-readable result:
[`runs/new_variants_frame00008_20260724_v3/summary.json`](../../runs/new_variants_frame00008_20260724_v3/summary.json),
SHA-256 `f302b6eaaae6eac8dd7e0894b371f8860df03d047ebb73e037e72ee90be166e9`.

Independent audit:
[`20260724_new_variants_frame00008_AUDIT.md`](20260724_new_variants_frame00008_AUDIT.md) and
[`20260724_new_variants_frame00008_AUDIT.json`](20260724_new_variants_frame00008_AUDIT.json),
JSON SHA-256 `8d7046634f8c12ea38e792b80a8d1a39a81c3e778f7975e9293996e7a28e93f8`.

## Setup

- Janelle `frame_00008`, calibrated and undistorted at downscale 16 (333×288).
- Evenly selected cameras:
  `C0001, C0008, C0014, C0021, C0026, C0031, C0039, C1004`.
- The first seven cameras were used for fitting, carve lifting, optimization, and train-only
  checkpoint selection. `C1004` was reporting-only.
- Seed 0; native CUDA stage-1 renderer; seven independent 640-Gaussian fits per arm, 300 steps.
- Common carve lift: 32³ grid, 48 samples/ray, minimum two views, fixed frozen thresholds.
- Common refinement: 2,000 unpacked, antialiased CUDA/gsplat steps with DefaultStrategy density
  through step 900 and 1,100 recovery steps afterward.
- Single-factor stage-1 arms: unchanged baseline, pool/free-list recycling,
  `mask_coverage_weight=5.0`, and structure-tensor initialization.
- The fifth report arm is the baseline trajectory selected by train-only best PSNR.

The first protocol attempt stopped on a reporting-key typo after one unsaved baseline fit. The
second saved all stage-1 fits but stopped before optimizer step 1 because installed gsplat 1.5.3
does not support packed RGB+D with random backgrounds. Both failures are preserved. V3 changed
only the common gsplat execution layout to unpacked, refit every arm from scratch, and completed.

## Stage-1 results

Metrics are equal-view means over the seven fitted cameras. Coverage is color-independent.

| Arm | FG PSNR | Δ FG PSNR | Crop SSIM | Coverage IoU @ 0.1 | Coverage inside | Coverage outside | Frozen gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `baseline` | 24.8360 | — | 0.94681 | 0.28069 | 0.65753 | 0.09018 | control |
| `pool` | **26.0652** | **+1.2291** | 0.95544 | 0.29464 | 0.64193 | 0.07815 | **pass** |
| `mask-containment` | 14.3236 | −10.5124 | 0.74370 | **0.64985** | 0.61597 | **0.00854** | **fail** |
| `structure-tensor` | 25.7317 | +0.8957 | **0.95633** | 0.23702 | 0.64464 | 0.12509 | **fail** |

Pool kept exactly 640 live rows in a 1,280-row allocation for every view. It won foreground PSNR
on **7/7** cameras, gained **1.2291 dB**, and reduced mean outside coverage by **13.34%**, passing
all frozen clauses.

Containment cut outside coverage by **90.52%**, but it lost **10.5124 dB** foreground PSNR and
reduced inside coverage by 6.32%; weight 5.0 therefore fails its intended stage-1 usefulness gate.
The high coverage IoU reflects aggressive spill suppression and must not be read as appearance
quality.

Structure initialization won PSNR on **7/7** cameras and gained **0.8957 dB**, but raised outside
coverage by **38.71%**, beyond the 10% guardrail. Its stage-1 gate therefore fails despite the
sharper fit.

## Downstream results

Held-out metrics below use only `C1004`. The train column is the equal-view mean over the seven
fitted cameras.

| Arm | Init N | Init held-out FG dB | Final N | Final train FG dB | Final held-out FG dB | Δ held-out | Held-out α-IoU | Δ α-IoU | Frozen gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `baseline` | 418 | 15.7522 | 5,542 | **29.3592** | 22.5152 | — | **0.95043** | — | control |
| `pool` | 458 | 15.8150 | 5,548 | 29.2647 | 22.7842 | **+0.2690** | 0.94498 | −0.00545 | **pass** |
| `mask-containment` | 395 | 14.4871 | 5,710 | 29.3513 | **23.0730** | **+0.5578** | 0.94609 | −0.00434 | **pass** |
| `structure-tensor` | 410 | **16.1034** | 5,498 | 29.2395 | 22.5428 | +0.0276 | 0.94182 | −0.00861 | **fail** |
| `best-train-checkpoint` | 418 | 15.7522 | 5,542 | 29.3592 | 22.5152 | +0.0000 | 0.95043 | +0.00000 | **neutral** |

Pool is the only treatment to pass both its intended stage-1 gate and the downstream gate. Its
held-out foreground PSNR improves **0.2690 dB**, held-out alpha IoU remains within the frozen
0.01 guardrail, and train foreground PSNR changes by −0.0945 dB.

Containment is the most interesting contradiction: its stage-1 appearance fit is poor, yet the
common carve+density path ends **+0.5578 dB** better on `C1004`, with alpha IoU −0.00434 and train
PSNR −0.0079 dB. That passes the downstream gate, but it does not select weight 5.0. It may reflect
a useful geometry/topology bias, recovery from a deliberately sparse field, or single-seed
variance; a weight sweep and replication are required before interpretation.

Structure initialization has the strongest held-out initialization (+0.3512 dB versus baseline),
but its final gain is only **+0.0276 dB**, below the frozen +0.10 dB materiality floor.

Train-only checkpoint selection chose step **2,000**. The selected model and final endpoint are
bit-exact, so the new policy is neutral on this run.

## Visual comparison

The `C0014` stage-1 sheet agrees with the metrics: pool and structure-tensor are visibly sharper
than baseline, while containment is thin and dim. After refinement all held-out renders are
qualitatively close; the +0.27/+0.56 dB pool and containment advantages are subtle, not a visual
category change.

- Stage 1:
  [`stage1_contact_sheet.png`](../../runs/new_variants_frame00008_20260724_v3/stage1_contact_sheet.png)
- Training camera:
  [`reconstruction_train_C0014.png`](../../runs/new_variants_frame00008_20260724_v3/reconstruction_train_C0014.png)
- Held-out camera:
  [`reconstruction_heldout_C1004.png`](../../runs/new_variants_frame00008_20260724_v3/reconstruction_heldout_C1004.png)
- Each model directory also contains calibrated comparisons, an eight-camera animation, and
  12-frame novel-orbit and novel-elevation diagnostics.

The synchronized viewer loaded all five initial/final pairs (ten models), returned HTTP 200 with a
2,888,259-byte response, used `CUDA_VISIBLE_DEVICES=''`, and was stopped cleanly:
[`viewer receipt`](20260724_new_variants_frame00008_VIEWER_RECEIPT.json).

```bash
CUDA_VISIBLE_DEVICES='' .venv-cuda/bin/rtgs view \
  --comparison-manifest benchmarks/results/20260724_new_variants_frame00008_VIEWER.json \
  --scene /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008 \
  --downscale 16 --device cpu --max-viewer-gaussians 20000 \
  --host 127.0.0.1 --port 8784 --no-open
```

## Conclusion

Keep every production default unchanged.

The fixed-capacity pool/free-list policy is the cleanest positive result: it improves all seven
stage-1 fits and passes the one-camera held-out downstream gate at matched live count. The
containment result is promising only downstream and is too internally contradictory to promote.
Structure-tensor initialization improves the immediate fit but fails its spill guardrail and loses
materiality after refinement. Best-train checkpoint selection provides no benefit here.

The next decisive experiment is multi-seed and multi-scene, with more than one held-out camera:
replicate baseline versus pool directly, and separately sweep containment weights below 5.0 while
tracking stage-1 appearance, carve count, final topology, and held-out quality. Do not combine the
two treatments until their individual effects replicate.

No timing, memory, real-time, multi-scene, or default claim is supported. The GPU was not isolated,
arm order was fixed, and this is one scene/seed with one held-out camera.
