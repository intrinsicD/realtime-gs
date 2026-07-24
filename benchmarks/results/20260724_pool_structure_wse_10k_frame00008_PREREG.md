# Pooled structure/WSE 10k checkpoint study on Janelle `frame_00008` — frozen protocol

Date frozen: 2026-07-24 (Europe/Berlin)

Status at freeze: the audited 2k parent outcomes are known, but no arm has been trained, rendered,
or measured under the new 10,000-step schedule. Preflight covered only static checks and synthetic
generation of the HTML/SVG structure.

## Question and scope

Does a fresh 10,000-step refinement schedule change the earlier downstream ranking among the
pooled gradient anchor, pooled structure-tensor density control, and pooled structure-tensor WSE
treatment? When do any differences appear, and do they persist through 8k and 10k?

This remains a single-scene, single-seed development comparison with one reporting-only held-out
camera. It cannot establish generalization, runtime, or a repository default.

## Fixed parent states and arms

Every arm starts from its exact audited stage-2 initialization in
`runs/pool_structure_wse_frame00008_20260724/summary.json`, SHA-256
`83c832b920a4603937112f4ff177ca8ac4d420dc58e72e97e847e7c896e176eb`:

| Arm | N | Parent initialization NPZ SHA-256 |
|---|---:|---|
| `pool-gradient` | 458 | `f73cfde8cf2c7dfeb0a8b3e32474e01f62fa8eb106e3c2af18d24fb7fe9f64c6` |
| `pool-structure-density` | 423 | `1ef84c7885d0b0404a4ac1713a224c4f01e3540d086e9151372a5d1e85d1b79c` |
| `pool-structure-wse` | 422 | `f0e41c4c57289f08c8b7101898c1f06192e0b2085b10bb877d8a315e97971abb` |

The structure arms retain the parent experiment's matched WSE/no-WSE treatment. Stage 1 and carve
lifting are not rerun, so their known outcomes cannot be reinterpreted as new 10k evidence.

## Calibrated data and split

- Raw capture:
  `/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008`.
- Calibration:
  `dataset/2025_03_07_stage_with_fabric/calibration_dome.json`.
- Loader: downscale 16, at most eight evenly selected images, undistortion and masks enabled,
  `test_every=8`.
- Frozen camera order:
  `C0001,C0008,C0014,C0021,C0026,C0031,C0039,C1004`.
- Training cameras: the first seven cameras above.
- Reporting-only held-out camera: `C1004`.

`C1004` may not influence optimization, checkpoint timing, selection, stopping, gates, or protocol
changes. It is evaluated only after training from the exact saved snapshots.

## Fresh 10k refinement schedule

Each arm is restarted from its parent initialization with a new optimizer and seed 0. This is not a
continuation from the parent's 2k endpoint. The full means-learning-rate schedule spans 10,000
steps, so the new 2k snapshot is not expected to reproduce the parent's 2k endpoint.

The parent final-refinement config is unchanged except:

```text
iterations=10000
schedule_iterations=10000
eval_every=100
checkpoint_policy=final
iteration_offset=0
```

Shared material settings remain CUDA/gsplat, unpacked antialiased rendering, random backgrounds,
masks, `outside_alpha_lambda=0.01`, `mask_alpha_lambda=0.05`, target SH degree 3 with interval 250,
and `gsplat-default` adaptive density with:

```text
start_iter=100, stop_iter=1000, every=100
grad_threshold=0.0008, absgrad=True
split_scale_frac=0.01, split_factor=1.6
prune_opacity=0.005, prune_scale_frac=0.1
max_gaussians=20000, opacity_reset_every=1000
opacity_reset_value=0.011, revised_opacity=True
```

Complete detached states are captured at exactly 2,000, 4,000, 6,000, 8,000, and 10,000 completed
steps. The returned final state must be tensor-identical to the captured 10k state. Density surgery
must stop by the configured boundary, leaving at least 9,000 steps after step 1,000 for recovery.
No best-checkpoint selection is permitted.

## Metrics and frozen interpretation

At initialization and every saved checkpoint, report equal-camera means over the seven training
cameras and the single held-out `C1004` values for foreground/crop/full PSNR, SSIM, alpha IoU,
inside/outside alpha, plus primitive count.

For each checkpoint, evaluate these three directional comparisons:

1. `pool-structure-density` versus `pool-gradient`;
2. `pool-structure-wse` versus `pool-gradient`;
3. `pool-structure-wse` versus `pool-structure-density`.

A comparison passes its balanced checkpoint gate only if all three clauses hold:

1. held-out foreground PSNR delta is at least `+0.10 dB`;
2. held-out alpha-IoU delta is at least `-0.01`;
3. train foreground PSNR delta is at least `-0.25 dB`.

The 10k endpoint is primary. A treatment may be called a **sustained long-run positive** only if
the same comparison passes at both 8k and 10k. Passing at an earlier checkpoint but not both final
checkpoints is a transient observation; a secondary metric cannot rescue a failed gate.

As a descriptive secondary trajectory summary, compute the trapezoidal foreground-PSNR average
over steps `[0,2000,4000,6000,8000,10000]` separately for train and held-out splits:

```text
trajectory_average = sum((y[i] + y[i+1]) / 2 * (step[i+1] - step[i])) / 10000
```

The trajectory average is not a selection rule and cannot override the endpoint gate. Timing and
peak-allocation fields are non-decisional because the shared GPU is neither reserved nor repeated.

## Required artifacts and visual handoff

Official output:

```text
runs/pool_structure_wse_10k_frame00008_20260724
```

The run must preserve:

- plan and summary receipts binding source, protocol, parent, inputs, environment, and artifacts;
- initial NPZ+PLY plus NPZ+PLY at 2k, 4k, 6k, 8k, and 10k for every arm;
- exact training histories and post-run metrics for every saved state;
- per-arm train/held-out checkpoint PNGs and progress GIFs;
- cross-arm `C0014` train and `C1004` held-out checkpoint sheets;
- a bound SVG quality trajectory;
- a 15-method checkpoint viewer manifest; and
- `index.html` at the run root, with relative offline links to metrics, checkpoint visuals, PLYs,
  provenance, viewer, result note, and audit.

The index page is a required result artifact, not an optional convenience. It must decode, have no
broken local links after the result/audit handoff is complete, and receive an HTTP smoke-test
receipt.

Official command:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv-cuda/bin/python benchmarks/pool_structure_wse_10k_frame00008.py \
  --protocol benchmarks/results/20260724_pool_structure_wse_10k_frame00008_PREREG.md \
  --out runs/pool_structure_wse_10k_frame00008_20260724
```

Planned viewer smoke:

```bash
CUDA_VISIBLE_DEVICES='' .venv-cuda/bin/rtgs view \
  --comparison-manifest \
  benchmarks/results/20260724_pool_structure_wse_10k_frame00008_VIEWER.json \
  --scene /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008 \
  --downscale 16 --device cpu --max-viewer-gaussians 20000 \
  --host 127.0.0.1 --port 8786 --no-open
```

Planned index smoke: serve the run root on `127.0.0.1:8790`, require HTTP 200 for `index.html` and
every local page asset, then stop the server and record the checks.

## Source binding at freeze

- Git revision: `7772f4fb63bf5b7c6540fbce7dfa3bf578bd7c11`.
- Harness:
  `benchmarks/pool_structure_wse_10k_frame00008.py`,
  SHA-256 `c163af7328cc7d3ad599101fcf4ddeb0ed90f97755cc82409883f4823c6d9c69`.
- Shared experiment helpers:
  `benchmarks/new_variants_frame00008.py`,
  SHA-256 `d0f429352a28bdb1584cc30ff9b92a7a70b94c168966a19e4785876ea7cc1e8c`.
- Audited parent summary:
  `runs/pool_structure_wse_frame00008_20260724/summary.json`,
  SHA-256 `83c832b920a4603937112f4ff177ca8ac4d420dc58e72e97e847e7c896e176eb`.

The official plan must additionally bind the complete working-tree status/diff, every executed
source file, the frozen protocol, all raw/calibration inputs, loaded tensors, and parent model
records before optimization starts.
