# Scientist pass — pooled structure-tensor WSE ablation

Date: 2026-07-24

Machine-readable audit:
[`20260724_pool_structure_wse_frame00008_AUDIT.json`](20260724_pool_structure_wse_frame00008_AUDIT.json),
SHA-256 `9350deab92e73876fee55e2d7e8c50f4d1e95d3653efa93c16d238be2ee9382e`.

Disposition: **15/15 checks passed; no balanced combined winner**

## Claim inventory and disposition

| # | Claim | Kind and scope | Evidence | Disposition |
|---|---|---|---|---|
| 1 | The no-WSE arm is a matched opt-in control | Proven mechanism/configuration | Source, focused tests, effective plan | **Confirm: structure arms differ only in `structure_sampling`; default stays WSE** |
| 2 | WSE improves pooled structure fitting | Preregistered single-scene stage-1 claim | 21 saved fits and independent replay | **Retire: −0.4385 dB, 3/7 wins** |
| 3 | WSE improves pooled structure downstream | Preregistered single-camera development claim | Exact gsplat replay | **Narrow: +0.1137 dB and guardrails pass, but local gate failed and effect is marginal** |
| 4 | Pooled structure without WSE beats pooled gradient | Preregistered combined-method claim | Stage-1 and endpoint gates | **Retire: −2.3161 dB stage 1, −0.1185 dB held out** |
| 5 | Pooled structure with WSE beats pooled gradient | Preregistered combined-method claim | Stage-1 and endpoint gates | **Retire: −2.7546 dB stage 1, −0.0047 dB held out** |
| 6 | This estimates a WSE×pool interaction | Causal interaction claim | All arms are pooled | **Not tested** |
| 7 | A default should change | Production/generalization claim | One scene, seed, held-out camera | **Not authorized** |
| 8 | Timings or allocations are performance evidence | Performance claim | Unreserved, unrepeated GPU | **Not authorized** |

## Chronology and source binding

V1 froze the treatments and gates before any calibrated outcome. Its synthetic preflight exposed
only import formatting. V2 preserved V1 but failed before plan creation on direct-script package
resolution. The tracked failure receipt states that the scene was not loaded and no arm began. V3
changed only that import fallback and its protocol path, predates the official plan, and completed.

The summary binds revision `7772f4fb63bf5b7c6540fbce7dfa3bf578bd7c11`, the dirty-tree status
and diff, the wrapper and shared harness, all relevant fit/pool/structure/lift/train/render
sources, effective dataclasses, calibration, every source RGB/mask, and every loaded tensor.
Current executed-source hashes still match.

All summary-bound artifacts exist with their exact byte counts and SHA-256 hashes.

## Isolation and control audit

- Frozen camera order and split replay exactly: seven train-only views, held-out `C1004`.
- No `C1004` stage-1 fit exists.
- All three arms use a 1,280-row pool with exactly 640 live/output Gaussians per fitted view.
- The two structure configs differ only in `structure_sampling`.
- Every training history contains 2,000 samples drawn only from local indices `[0,6]`.
- Density count changes end by step 900, followed by 1,100 recovery steps.
- Every initial/final NPZ and PLY is finite with its reported count.
- All final endpoints use the final policy; no held-out checkpoint selection occurs.

## Independent metric and gate replay

All 21 saved fits were rendered from disk. Reported stage-1 metrics replay within the frozen
tolerance, and native-CUDA outputs agree with the Torch correctness anchor at numerical precision.
All initial/final 3D metrics replay exactly through the unpacked antialiased gsplat path.

| Contrast | Stage-1 FG Δ | Wins | Outside ratio | Stage-1 gate | Held-out FG Δ | α-IoU Δ | Train FG Δ | Downstream gate |
|---|---:|---:|---:|---|---:|---:|---:|---|
| WSE vs density | **−0.4385 dB** | 3/7 | 1.0917 | **fail** | **+0.1137 dB** | −0.00465 | +0.1023 dB | pass |
| Density vs gradient | −2.3161 dB | 0/7 | 1.2855 | **fail** | −0.1185 dB | +0.00491 | +0.1264 dB | **fail** |
| WSE vs gradient | −2.7546 dB | 0/7 | 1.4034 | **fail** | −0.0047 dB | +0.00027 | +0.2287 dB | **fail** |

The downstream WSE gate is not a global pass. It is conditional on pooled structure, contradicts
the stage-1 direction, and does not beat pooled gradient. Secondary initialization and train
metrics cannot rescue the failed primary combined-method gates.

## Repeatability warning

The audit hash-bound the preceding pooled-gradient result and verified that its prior effective
configuration matches the current anchor after removing only the newly added default-valued
`structure_sampling` field.

Despite that, held-out foreground PSNR moved **−0.0769 dB**, alpha-IoU moved +0.00191, and final
count changed 5,548→5,578 across sessions. That is direct evidence that these CUDA/gsplat runs are
not bit-exact. The WSE/no-WSE endpoint delta is only +0.1137 dB, so it is too close to observed
session drift to support promotion without paired repeats.

## Visual and viewer audit

All 15 bound visuals decode: three cross-arm sheets plus calibrated, camera-path, orbit, and
elevation artifacts for each arm. Visual inspection supports only the result note's scoped
statements: pooled gradient is cleaner at stage 1; refined endpoints are close.

The CPU viewer loaded six bound PLYs, returned HTTP 200 with a 2,888,259-byte response, owned the
exact listening socket, listed no NVIDIA compute process, and stopped with its PID gone and port
closed.

## Commands checked

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv-cuda/bin/python benchmarks/pool_structure_wse_frame00008.py \
  --protocol benchmarks/results/20260724_pool_structure_wse_frame00008_PREREG_V3.md \
  --out runs/pool_structure_wse_frame00008_20260724
```

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv-cuda/bin/python benchmarks/audit_pool_structure_wse_frame00008.py
```

```bash
CUDA_VISIBLE_DEVICES='' .venv-cuda/bin/rtgs view \
  --comparison-manifest benchmarks/results/20260724_pool_structure_wse_frame00008_VIEWER.json \
  --scene /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008 \
  --downscale 16 --device cpu --max-viewer-gaussians 20000 \
  --host 127.0.0.1 --port 8785 --no-open
```

## Final decision

Keep all defaults unchanged. Retain the density-prefix mode as a research control, retire both
pool+structure combinations as promotion candidates in this scope, and treat the WSE downstream
edge as an unreplicated conditional observation.

Still unverified: paired repeatability, multi-seed and multi-scene transfer, more than one held-out
camera, WSE behavior without pooling, isolated performance, and whether the slight endpoint
difference is geometry/topology or CUDA session variance.
