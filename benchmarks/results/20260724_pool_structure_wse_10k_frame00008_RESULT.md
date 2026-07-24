# Pooled structure/WSE 10k checkpoint study — result

Date: 2026-07-24

Protocol:
[`20260724_pool_structure_wse_10k_frame00008_PREREG.md`](20260724_pool_structure_wse_10k_frame00008_PREREG.md)

Machine result:
[`runs/pool_structure_wse_10k_frame00008_20260724/summary.json`](../../runs/pool_structure_wse_10k_frame00008_20260724/summary.json),
SHA-256 `6fdabac92cd0bf1d4ad610f90083ecd75f0942c15cf16278809fa0ba46baf01b`.

Scientist pass:
[`20260724_pool_structure_wse_10k_frame00008_AUDIT.md`](20260724_pool_structure_wse_10k_frame00008_AUDIT.md)
and
[`20260724_pool_structure_wse_10k_frame00008_AUDIT.json`](20260724_pool_structure_wse_10k_frame00008_AUDIT.json).

## Outcome

The scoped 10k result favors `pool-structure-wse`. It passes the frozen balanced downstream gate
against both `pool-gradient` and the matched `pool-structure-density` no-WSE control at every saved
checkpoint. The density control fails against gradient at every checkpoint.

Held-out `C1004` foreground PSNR:

| Arm | 2k | 4k | 6k | 8k | 10k | 10k − 2k |
|---|---:|---:|---:|---:|---:|---:|
| `pool-gradient` | 22.5578 | 22.4392 | 22.3103 | 22.2166 | 22.1985 | −0.3593 |
| `pool-structure-density` | 22.1302 | 21.9920 | 21.8402 | 21.8487 | 21.7950 | −0.3352 |
| `pool-structure-wse` | **22.7824** | **22.7325** | **22.6846** | **22.5752** | **22.5687** | −0.2137 |

At 10k:

| Contrast | Held-out FG Δ | Held-out α-IoU Δ | Train FG Δ | Frozen gate |
|---|---:|---:|---:|---|
| density vs gradient | −0.4035 dB | +0.00196 | −0.0498 dB | fail |
| WSE vs gradient | **+0.3702 dB** | +0.00127 | −0.0825 dB | pass |
| WSE vs density | **+0.7737 dB** | −0.00068 | −0.0328 dB | pass |

Both WSE contrasts also pass at 8k, meeting the preregistered definition of a sustained long-run
positive development observation. The held-out trapezoidal trajectory averages are
21.7061/21.3448/22.0225 dB for gradient/density/WSE.

Training longer did not improve held-out quality. Every arm has its highest reporting-only
`C1004` checkpoint at 2k, then declines through 10k. This does **not** authorize choosing 2k:
`C1004` is held out and no train-only validation checkpoint rule was run.

The new 2k snapshots are not replays of the parent experiment's 2k endpoints. All arms start from
the exact parent initializations, but their fresh means-learning-rate schedule spans 10k rather
than 2k. The within-run checkpoint comparisons are matched; cross-schedule endpoint differences
do not estimate CUDA repeatability.

## Visual result

The held-out sheet shows all three arms converging to similar object-level reconstructions by 2k,
with persistent high-frequency silhouette, hair, hand, and dress-edge residuals thereafter. WSE's
numerical advantage is visible as a modestly cleaner reconstruction rather than a categorical
appearance change. The longer checkpoints do not show a qualitative improvement that contradicts
the declining held-out metric.

Open the required offline results page:

```text
runs/pool_structure_wse_10k_frame00008_20260724/index.html
```

It contains the exact metric table, trajectory plot, cross-arm sheets, all per-arm checkpoint
renders, and links to each saved PLY.

## Commands and handoff

Official run:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv-cuda/bin/python benchmarks/pool_structure_wse_10k_frame00008.py \
  --protocol benchmarks/results/20260724_pool_structure_wse_10k_frame00008_PREREG.md \
  --out runs/pool_structure_wse_10k_frame00008_20260724
```

Independent audit:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv-cuda/bin/python benchmarks/audit_pool_structure_wse_10k_frame00008.py
```

Checkpoint viewer:

```bash
CUDA_VISIBLE_DEVICES='' .venv-cuda/bin/rtgs view \
  --comparison-manifest \
  benchmarks/results/20260724_pool_structure_wse_10k_frame00008_VIEWER.json \
  --scene /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008 \
  --downscale 16 --device cpu --max-viewer-gaussians 20000 \
  --host 127.0.0.1 --port 8786 --no-open
```

The CPU viewer loaded all 30 initial/checkpoint model entries, returned HTTP 200, had no NVIDIA
compute process, and shut down with its PID gone and port closed.

## Decision

Confirm only the narrow result: under this fresh 10k schedule on this scene/seed/camera, WSE is
materially better than its no-WSE control and modestly better than pooled gradient. Do not change a
default: the parent stage-1 gates for both structure arms still failed, the current evidence has
one scene/seed/held-out camera, and no repeatability or performance study ran.

For a practical shorter schedule, add a train-only validation camera and preregister checkpoint
selection; do not use `C1004` to choose 2k. For promotion, repeat the full treatment on multiple
seeds, scenes, and held-out cameras.
