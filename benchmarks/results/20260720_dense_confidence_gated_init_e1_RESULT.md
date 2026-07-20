# Dense confidence-gated initialization E1 result

## Scope and chronology

This is the calibrated E1 stage from
`docs/TASK_DENSE_CONFIDENCE_GATED_INIT.md`. The hypothesis, arms, seed, metrics,
merge voxel, and decision rule were preregistered in commit `3856c5c` before
outcome access. The official run used clean commit
`68d0a6dc4557c7f9dd5023bf7ecd5cd8808659d6`.

Input was the strict seven-view compact bundle
`runs/compact_masked_bundle_640_20260717/reconstruction_inputs`, manifest
SHA-256 `6ed60cf3df1f9ca476dfabace18b8d868d63eebd041522313c5b4fd644ee2614`.
It contains 640 compact components for each of
`C0001/C0008/C0014/C0021/C0026/C0031/C0039` from calibrated Janelle
`frame_00008`; held-out camera `C1004` was excluded during acquisition.

## Official correctness-anchor run

```bash
/usr/bin/time -v .venv/bin/python benchmarks/compact_init_eval.py \
  --bundle runs/compact_masked_bundle_640_20260717/reconstruction_inputs \
  --out runs/dense_confidence_gated_init_e1_20260720 \
  --seed 0
```

| Arm | Gaussians | Mean full PSNR | Mean foreground PSNR | Mean SSIM |
|---|---:|---:|---:|---:|
| balanced top-K | 172 | 18.192647 | 18.169987 | 0.760636 |
| dense + voxel merge | 2,319 | 20.161683 | 20.141388 | 0.631656 |
| dense − top-K | +2,147 | +1.969036 dB | +1.971401 dB | −0.128979 |

Foreground-PSNR gains for
`C0001/C0008/C0014/C0021/C0026/C0031/C0039` were respectively
`+2.3093/+2.0839/+1.8828/+2.0171/+1.9592/+1.6373/+1.9102 dB`.
No view regressed.

The preregistered E1 decision is **FAIL**: the mean gain and per-view guard
pass, but `2319 / 172 = 13.4826×`, far above the required `≤2×` count.
Therefore dense+merge is not a count-controlled “better init” and the top-K
default does not change.

Artifacts:

- `init_eval.json`: SHA-256
  `7bf4ac973fe373c5b4cf7170877001041f46f9e63ccbf0e16a9b0d3d744f6ea6`
- `init_topk.ply`: SHA-256
  `d83ee1e764ee6bc0d1cf7696e848df91b0a92d33ad5c9932c9e1138e8564e9fb`
- `init_dense_merged.ply`: SHA-256
  `56ce5f1ac3a321f6912506dc4e2c8484c1c3b9d5930eb140b84253faf106cff7`

Observed local resource receipt (not a portable benchmark): wall time
`1:38:31`, maximum RSS `17,552,984 KiB`, mean CPU utilization `544%`.

## Complete cluster diagnostic and fast-backend parity replay

The landed harness discarded the full view-multiplicity histogram. After the
official outcome, reporting/progress and evaluation-backend seams were added,
without changing placement or merge. The diagnostic replay used:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
/usr/bin/time -v .venv/bin/python benchmarks/compact_init_eval.py \
  --bundle runs/compact_masked_bundle_640_20260717/reconstruction_inputs \
  --out runs/dense_confidence_gated_init_e1_fast_20260720 \
  --seed 0 --rasterizer gsplat --device cuda
```

Both replay PLY SHA-256 values are byte-identical to the official PLYs. The
cluster view-multiplicity histogram is:

| Distinct source views | Clusters |
|---:|---:|
| 1 | 1,819 |
| 2 | 406 |
| 3 | 73 |
| 4 | 16 |
| 5 | 4 |
| 6 | 1 |

Thus 78.44% of merged clusters are monocular; 500 clusters have multiplicity
at least two and 94 have multiplicity at least three.

The replay's largest absolute aggregate metric difference from the official
Torch result is `0.003812 dB`. On C0001, rendering the fit-window-adjusted
Torch camera instead of a full frame followed by a crop changes PSNR by only
`3.8e-6 dB` and SSIM by `6.0e-7`. Cropped gsplat versus cropped Torch changes
foreground PSNR by `−0.003252 dB`, SSIM by `+0.000759`, and has mean absolute
pixel error `1.19e-4`.

The fast replay observed wall time `47.18 s` and maximum RSS `1,811,176 KiB`.
Placement took `0.72 s` top-K and `0.71 s` dense; evaluations took `22.65 s`
and `21.76 s`. These are single-run local diagnostics on an RTX 3050 with no
warmup/repeats or idle-machine guarantee, not portable performance claims.

## Profiling and viewer handoff

Linux `perf` was unavailable because the host has
`kernel.perf_event_paranoid=4`. A one-view `cProfile` replay on C0001 measured:

| Stage | Seconds |
|---|---:|
| strict bundle load | 0.026 |
| PLY load | 0.0004 |
| CSR build | 0.009 |
| compact teacher render | 2.202 |
| full-frame Torch 3D render | 634.043 |
| metrics | 1.104 |

The artifact-only viewer was launched and returned HTTP 200:

```bash
.venv/bin/rtgs view \
  --gaussians runs/dense_confidence_gated_init_e1_20260720/init_dense_merged.ply \
  --initial runs/dense_confidence_gated_init_e1_20260720/init_topk.ply \
  --host 127.0.0.1 --port 8773 --no-open
```

A calibrated-snapshot viewer command could not run because this workspace's
`dataset/2025_03_07_stage_with_fabric/frame_00008` contains `gaussians2d/` but
no longer contains `rgb/`; the CLI therefore fell through to a nonexistent
COLMAP model. Quantitative E1 metrics remain bound to the strict calibrated
compact bundle.

Independent disposition:
`benchmarks/results/20260720_dense_confidence_gated_init_e1_AUDIT.md`.
