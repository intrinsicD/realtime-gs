# Janelle beam-convergence dynamics — independent replication

## Outcome

The reduced real-data screen confirms the mechanism, with narrower causal wording than the
original 2026-07-23 log entry:

1. Adaptive density does **not** erase most original beam centers. At step 1,000, 740/800
   originals remain; their mean displacement is 0.01449 world units and p90 is 0.02162.
2. Every scheduled opacity reset removes rendered support temporarily. At steps
   100/200/300/400/500, both ADC arms have no Gaussian above opacity 0.02 and alpha-IoU is
   at most 0.000287.
3. Density growth makes original beam rows a minority: Beam-ADC ends at 4,255 Gaussians,
   of which 740 (17.39%) are original rows. The controller created 4,440 child/clone rows
   cumulatively and removed 985 split parents.
4. At an equal fixed count, the **complete beam initializer package** beats the complete random
   package by 2.39419 dB foreground PSNR and 0.16342 alpha-IoU. This is not a position-only
   attribution: the arms also differ in covariance/scale, quaternion, and SH/color fields.

The production claim remains open. This is an eight-view, downscale-32, CPU Torch/classic-density
mechanism screen with fitted-view evaluation, not the 26-view CUDA gsplat schedule and not a
held-out-view experiment.

## Frozen parameters

| Parameter | Value |
| --- | --- |
| Data | `dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d` |
| Input binding | 27 files; set SHA-256 `5811b08c5d37d6d4e797e9e2aab18d9a6f420266041bb9b874ec380a43c507f2` |
| Views | Global indices `0,3,6,9,12,15,18,21` (8/26) |
| Teachers | Exact compact observation-field queries with packed-alpha masking |
| Resolution | Per-view fit windows downscaled by 32 |
| Initial count | 800 in every arm |
| Arms | `{beam, random} × {adaptive density, fixed topology}` |
| Seed / steps | 0 / 1,000 |
| Evaluation | Every 25 steps on local fitted views `0,2,4,6`; no held-out cameras |
| Loss | Masked L1 + 0.2 D-SSIM + 0.05 mask-alpha + 0.01 outside-alpha; black background |
| Renderer / device | Torch reference / CPU, PyTorch 2.12.1+cpu, 16 Torch threads |
| ADC | steps 20–500 every 4; threshold 0.003; resets every 100 to 0.011; cap 8,000 |
| Executed revision | `c2a7e120a5cafdcf22d4bff6f5b9868b860eb1df` |
| Executed harness SHA-256 | `bbfe4172958af8f1188999f0eb1d4c41dccef2299b40ff93909f65e8dcf17991` |

The protocol was already public in commit `d8948eb`; this run is a reproduction of a disclosed
development result, not a blinded confirmatory test.

## Endpoints

| Arm | Init FG PSNR | Init alpha-IoU | Final FG PSNR | Final alpha-IoU | Final count | Original survivors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| beam + ADC | 12.33030 | 0.01073 | **26.85986** | **0.95082** | 4,255 | 740 |
| random + ADC | 11.90994 | 0.29410 | 24.38014 | 0.32681 | 1,198 | 439 |
| beam + fixed | 12.33030 | 0.01073 | **25.00831** | **0.93649** | 800 | 800 |
| random + fixed | 11.90994 | 0.29410 | 22.61412 | 0.77308 | 800 | 800 |

ADC endpoint comparisons are capacity-confounded: Beam-ADC ends with 3.55× as many Gaussians as
Random-ADC. The fixed-topology comparison matches count and schedule, but not the initializer's
non-position fields.

## Reset trajectory

| Reset step | Beam FG PSNR | Beam alpha-IoU | Random FG PSNR | Random alpha-IoU |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 11.97899 | 0.000000 | 14.72097 | 0.000000 |
| 200 | 12.41935 | 0.000000 | 15.43730 | 0.000000 |
| 300 | 12.80647 | 0.000000 | 15.86595 | 0.000000 |
| 400 | 13.29001 | 0.000000 | 16.14809 | 0.000000 |
| 500 | 13.79233 | 0.000000 | 16.39559 | 0.000287 |

At all ten reset checkpoints, every Gaussian has opacity below 0.02 and the confident-set Chamfer
distance is undefined. The original harness wrote those empty-set distances as non-standard JSON
`NaN`; the audit normalizes only those 30 expected values to `null`, and the harness now emits
strict JSON prospectively.

## Repeatability and relation to the earlier entry

The two ADC arms were immediately repeated in the same environment. Every scientific trajectory
field and both final PLYs were byte-identical between runs:

| Arm | Scientific-record SHA-256 | Final-PLY SHA-256 |
| --- | --- | --- |
| beam + ADC | `18dbeaee76bdeab95bbf34b5c94e6e31064956c8d3176a3924329a736554462f` | `77c0b3c8353152efaa6c7e5f41b35766a7e6fc44d3cc48a0d0da5fa514259f60` |
| random + ADC | `41fabd611766543f2990f7292bb1c19af693e7cccf54736fe1bfe7cca510fe6d` | `84eab6a4369418ec10827cadfa6ecfb2440d83b9e7e0488724cc9de279112b43` |

The earlier untracked run's exact ADC endpoints do not reproduce: this replication has 135 fewer
Beam-ADC and 90 fewer Random-ADC Gaussians, with foreground PSNR lower by approximately 0.06 and
0.40 dB respectively. Its raw artifacts and exact executed source were not retained. The repeated
current result therefore replaces those exact ADC endpoint numbers; the qualitative reset,
survival, dilution, and fixed-topology findings remain.

## Commands and artifacts

Primary 2×2 run:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 .venv/bin/python \
  benchmarks/beam_convergence_dynamics.py \
  --out runs/beam_convergence_dynamics_replication_20260723
```

Same-environment ADC repeat:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 .venv/bin/python \
  benchmarks/beam_convergence_dynamics.py \
  --out runs/beam_convergence_dynamics_repeat2_20260723 \
  --arms beam-adc random-adc
```

Audit:

```bash
.venv/bin/python benchmarks/audit_beam_convergence_dynamics.py
```

The strict machine-readable evidence is
`benchmarks/results/20260723_beam_convergence_dynamics_REPLICATION_AUDIT.json`
(SHA-256 `6e84dee111456076948ea67570e836851af223696e5a3f573e41951f86f91ad3`).
It embeds the normalized trajectories, all source/artifact hashes, derived quantities, and
132/132 passing audit checks. The local primary and repeat directories contain 66 finite,
count-checked PLY files.

Viewer handoff:

```bash
CUDA_VISIBLE_DEVICES='' .venv-cuda/bin/rtgs view \
  --gaussians runs/beam_convergence_dynamics_replication_20260723/beam-adc/gaussians_final.ply \
  --initial runs/beam_convergence_dynamics_replication_20260723/beam-adc/gaussians_init.ply \
  --watch-checkpoints runs/beam_convergence_dynamics_replication_20260723/beam-adc \
  --max-viewer-gaussians 8000 --rasterizer torch --device cpu \
  --host 127.0.0.1 --port 8784 --no-open
```

The viewer returned HTTP 200. Its server PID used no CUDA compute allocation; one immediate sample
was 555,884 KiB RSS and 17% of one logical CPU. The server was stopped after the smoke test. This
is a visual-diagnostics handoff, not a zero-overhead or performance claim.
