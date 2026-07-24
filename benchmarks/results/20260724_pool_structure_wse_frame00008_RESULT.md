# Pool + structure tensor, with and without WSE, on Janelle `frame_00008`

Date: 2026-07-24

Status: **no combined method is a balanced winner; keep defaults unchanged**

Frozen protocol chain:
[`v1`](20260724_pool_structure_wse_frame00008_PREREG.md),
[`v2`](20260724_pool_structure_wse_frame00008_PREREG_V2.md), and
[`v3`](20260724_pool_structure_wse_frame00008_PREREG_V3.md).

Machine-readable result:
[`summary.json`](../../runs/pool_structure_wse_frame00008_20260724/summary.json),
SHA-256 `83c832b920a4603937112f4ff177ca8ac4d420dc58e72e97e847e7c896e176eb`.

Independent audit:
[`20260724_pool_structure_wse_frame00008_AUDIT.md`](20260724_pool_structure_wse_frame00008_AUDIT.md)
and
[`20260724_pool_structure_wse_frame00008_AUDIT.json`](20260724_pool_structure_wse_frame00008_AUDIT.json),
JSON SHA-256 `9350deab92e73876fee55e2d7e8c50f4d1e95d3653efa93c16d238be2ee9382e`.

## Setup

The calibrated split and all common settings match the 2026-07-24 single-factor comparison:
Janelle `frame_00008` at downscale 16, seven train-only cameras and reporting-only `C1004`, seed
0, seven 640-live/1,280-capacity pooled fits per arm, 300 native-CUDA stage-1 steps, common carve
lift, and 2,000 unpacked antialiased CUDA/gsplat refinement steps.

The three arms were:

- `pool-gradient`: the previous pooled gradient initializer, rerun as the anchor;
- `pool-structure-density`: structure tensor plus oriented covariance, but no WSE—the first 640
  points from the same 2,560-point density candidate stream are kept;
- `pool-structure-wse`: the same structure path, with anisotropic WSE choosing 640 of those 2,560
  candidates.

The two structure arms differ only in `structure_sampling`. The pool lifecycle, tensor/density
fields, candidate RNG stream, covariance construction, optimizer, lift, and refinement are
otherwise identical.

V1 preflight found an import-order lint issue before any calibrated run. V2 then failed at the
first direct-script import before creating an output directory or loading the scene. Both
source-only amendments and the pre-plan failure receipt are preserved. V3 completed.

## Stage-1 results

Equal-camera means over seven training views:

| Arm | FG PSNR | Crop SSIM | Coverage IoU @ 0.1 | Coverage inside | Coverage outside |
|---|---:|---:|---:|---:|---:|
| `pool-gradient` | **26.0660** | **0.95543** | **0.29464** | **0.64192** | **0.07815** |
| `pool-structure-density` | 23.7498 | 0.94894 | 0.27064 | 0.62080 | 0.10046 |
| `pool-structure-wse` | 23.3114 | 0.94846 | 0.25461 | 0.60561 | 0.10968 |

Directly against the matched no-WSE control, WSE changed foreground PSNR by **−0.4385 dB**, won
only **3/7** paired cameras, and increased outside coverage by **9.17%**. It therefore fails its
frozen stage-1 gate.

Neither structure combination challenged pooled gradient. Density lost **2.3161 dB**, won 0/7
views, and raised outside coverage 28.55%. WSE lost **2.7546 dB**, won 0/7, and raised outside
coverage 40.34%. Both fail their anchor-relative gates.

## Lift and refinement

| Arm | Init N | Init held-out FG dB | Init α-IoU | Final N | Final train FG dB | Final held-out FG dB | Final α-IoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pool-gradient` | 458 | 15.8144 | 0.69392 | 5,578 | 29.1729 | **22.7073** | 0.94689 |
| `pool-structure-density` | 423 | 16.0306 | **0.74389** | 5,393 | 29.2993 | 22.5888 | **0.95181** |
| `pool-structure-wse` | 422 | **16.1074** | 0.72049 | 5,553 | **29.4016** | 22.7025 | 0.94716 |

WSE versus no-WSE passes the literal downstream gate: held-out foreground PSNR is
**+0.1137 dB**, alpha-IoU is **−0.00465**, and train foreground PSNR is **+0.1023 dB**. That does
not rescue its failed stage-1 claim or make the complete method a winner.

Against pooled gradient, density ends **−0.1185 dB** held out and WSE ends **−0.0047 dB**. Neither
reaches the frozen +0.10 dB materiality floor. The structure arms provide somewhat stronger
initial held-out metrics, but common density refinement erases that advantage.

The independent audit also found that the nominally unchanged pooled-gradient anchor moved from
22.7842 dB in the preceding run to 22.7073 dB here (**−0.0769 dB**) and changed final count
5,548→5,578. Native CUDA/gsplat atomics are not bit-exact across these sessions. WSE's +0.1137 dB
edge over no-WSE is therefore a marginal single-run observation, not replication-grade evidence.

## Visual comparison

The `C0014` stage-1 sheet agrees with the aggregate result: pooled gradient is cleaner; both
structure arms are blurrier, and WSE does not produce a consistent visual improvement over the
density control. After refinement the three train and held-out renders are very close, with the
remaining differences concentrated around silhouette/error edges.

- [Stage-1 sheet](../../runs/pool_structure_wse_frame00008_20260724/stage1_contact_sheet.png)
- [Training camera](../../runs/pool_structure_wse_frame00008_20260724/reconstruction_train_C0014.png)
- [Held-out camera](../../runs/pool_structure_wse_frame00008_20260724/reconstruction_heldout_C1004.png)

Each model directory also contains an eight-camera calibrated comparison plus novel-orbit and
novel-elevation animations. The synchronized CPU viewer loaded all three initial/final pairs,
returned HTTP 200, owned the listening socket without a CUDA-visible device, and shut down
cleanly:
[`viewer receipt`](20260724_pool_structure_wse_frame00008_VIEWER_RECEIPT.json).

```bash
CUDA_VISIBLE_DEVICES='' .venv-cuda/bin/rtgs view \
  --comparison-manifest benchmarks/results/20260724_pool_structure_wse_frame00008_VIEWER.json \
  --scene /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008 \
  --downscale 16 --device cpu --max-viewer-gaussians 20000 \
  --host 127.0.0.1 --port 8785 --no-open
```

## Conclusion

Do not promote either combination.

Under the fixed pool policy, removing WSE improves the immediate structure-tensor fit, while WSE
recovers a small downstream advantage over that no-WSE control. Neither structure variant beats
the simpler pooled-gradient anchor end to end, and the downstream WSE/no-WSE difference is close
to observed cross-run drift.

Keep gradient initialization and all production defaults unchanged. Retain
`structure_sampling="density"` only as an explicit research control. A future WSE claim would need
paired multi-seed repeats and more scenes/cameras; this result does not justify a hyperparameter
sweep or a pool+structure default.

No timing, memory, WSE×pool interaction, multi-scene, or generalization claim is supported.
