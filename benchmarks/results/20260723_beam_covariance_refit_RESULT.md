# Beam-track covariance refit on Janelle — result

Date: 2026-07-23

Status: complete, exactly repeated, independently audited; **neither treatment passes all
preregistered gates and no default change is authorized**.

Protocol:
[`20260723_beam_covariance_refit_PREREG.md`](20260723_beam_covariance_refit_PREREG.md)

Machine audit:
[`20260723_beam_covariance_refit_AUDIT.json`](20260723_beam_covariance_refit_AUDIT.json)

Scientist pass:
[`20260723_beam_covariance_refit_AUDIT.md`](20260723_beam_covariance_refit_AUDIT.md)

## Answer

Beam Fusion does expose useful partial correspondences: each retained 3D Gaussian has CSR lineage
to at most one fitted 2D Gaussian in each contributing view. In this run, 800 3D Gaussians had
6,029 links (7.536 views/Gaussian on average) to 4,704 unique 2D Gaussians. That is only 11.76% of
the 40,000 input components in the selected eight views; source components may also be reused by
multiple 3D outputs.

Those links did **not** yield a trustworthy 3D covariance through the tested linear solver. The
linear system is inconsistent (median relative linear residual 0.737), and 635/800 raw solutions
are not SPD. Clamping the invalid eigenvalues produces very wide, extremely anisotropic splats.
That accidental scale inflation gives dramatically better visible coverage and faster
fixed-topology optimization, but it no longer reprojects to the corresponding 2D covariances.

The robust Cholesky fit does the opposite: it removes most of the LSQ inflation, modestly improves
the median whitened reprojection error over CI, and consequently returns almost exactly to the
CI arm's near-zero initial coverage and convergence curve. More covariance gradient descent is
therefore not the missing step in this formulation.

## Frozen setup

- Dataset:
  `dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d`,
  compact manifest SHA-256
  `b1c8e256d73e2c05f3cb4797a615bdbb2639a637f12908a5c96a2a9a9f912847`.
- Eight all-fitted views:
  `C0001, C0006, C0012, C0019, C0022, C0028, C0031, C0039`.
- Beam Fusion: seed 0, 800 outputs, minimum 3 views, 3-sigma seed/fold-in gates,
  color distance 0.35/sigma 0.25, extent/100 NMS voxel, opacity 0.10.
- Covariance arms:
  unchanged CI; masked Splat-SfM pseudoinverse plus bounded SPD projection; and 120 steps of
  float64 Cholesky Adam at learning rate 0.03, Huber delta 0.25, CI-prior weight `1e-3`.
- All arms have bit-identical means, opacity, SH/color, and count. Only covariance differs.
- Refinement: Torch CPU, downscale 32, 1,000 steps, fixed topology, identical masks/loss/seed;
  no split, clone, prune, merge, teleport, or opacity reset.
- Four reported cameras are a subset of the fitted cameras. This is development evidence, not
  held-out evidence.

Official command:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python benchmarks/beam_covariance_refit.py \
  --protocol benchmarks/results/20260723_beam_covariance_refit_PREREG.md \
  --out runs/beam_covariance_refit_20260723
```

The output summary SHA-256 is
`350ce35a8cfd353a43d71e8979b4b301df3a40261c7f34c69211fdd16de83ca2`.

## Results

| Arm | Median whitened covariance residual | Median minimum / maximum 3D sigma | Initial FG PSNR | Initial alpha IoU | PSNR-AUC mean | First reaches CI-final PSNR | Final FG PSNR | Final alpha IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CI | **0.6888** | 0.00210 / 0.00980 | 12.3303 dB | 0.01073 | 21.8568 dB | 1000 | 25.0083 dB | **0.93649** |
| track-LSQ | 13.4478 | 0.00010 / 0.04499 | **14.2323 dB** | **0.55056** | **23.8475 dB** | **650** | **25.5652 dB** | 0.93466 |
| track-robust | 0.6350 | 0.00010 / 0.00855 | 12.2749 dB | 0.01047 | 21.6498 dB | never | 24.9186 dB | 0.93505 |

`track-LSQ` makes the median maximum axis 4.59× wider than CI and pins the median minimum axis to
the `1e-4` floor. Its median covariance condition number is 178,541, versus 15.0 for CI. That
changes initial alpha-inside from 0.13975 to 0.49503 and alpha IoU by +0.53983 absolute, but also
raises outside alpha from 0.00103 to 0.02538. Its PSNR AUC is +9.108% and its final foreground PSNR
is +0.5569 dB versus CI, with only -0.00184 alpha-IoU difference.

The useful render effect is nevertheless not correspondence-consistent: `track-LSQ` worsens the
median whitened residual by 18.52× relative to CI rather than reducing it. `track-robust` reduces
that median by only 7.80%, below the frozen 20% gate; its residual mean is worse than CI because of
large tails. It also loses 0.95% PSNR AUC and 0.0897 dB final PSNR.

## Preregistered decisions

| Treatment | Direct covariance gate | Initial coverage/quality gate | Pipeline gate | Overall |
| --- | --- | --- | --- | --- |
| track-LSQ | **fail** | pass | pass | **fail** |
| track-robust | **fail** | **fail** | **fail** | **fail** |

The robust arm is not preferred over LSQ: although it repairs much of LSQ's covariance residual,
it is worse in both AUC and final PSNR. Neither arm may be described as a validated physical
covariance estimator.

## Interpretation and next experiment

The experiment separates two objectives that were being conflated:

1. matching each fitted 2D Gaussian's local covariance; and
2. giving a sparse 800-splat surface enough footprint to render useful alpha.

The correspondences constrain the first. They do not require enough footprint to solve the
second, especially when only 11.76% of input 2D components appear uniquely in the retained tracks.
LSQ helps only because the invalid-SPD repair injects the second objective accidentally.

The next controlled test should therefore be an **explicit coverage prior**, not another unconstrained
covariance optimizer: preserve CI orientation/shape, compare preregistered global or
projected-footprint scale multipliers against a bounded CI/LSQ blend, and gate outside-mask alpha
as well as inside coverage. A physically named covariance arm should instead use a PSD-constrained
solver and must pass reprojection diagnostics before entering refinement. Split/merge can then be
tested only after a scale policy has independently passed the initialization gate.

## Repetition, audit, and provenance limits

The full command was repeated unchanged into
`runs/beam_covariance_refit_repeat_20260723`. All timing-free summary fields, every checkpoint
trajectory, and initial/final PLY hashes repeat exactly. The independent audit rebuilt Beam Fusion
and all contributor links from the compact payloads, recomputed covariance residuals/AUC/gates
from saved artifacts, and passed 75/75 checks.

One source-binding defect remains explicit: the official harness imported the dirty
`beam_convergence_dynamics.py`, but the preregistration omitted its separate hash. Its unchanged
post-run hash is
`6521af11d0af8513cd6963de260786e37c9791506a0782619b0561045fe2ffa9`.
The exact repeat narrows, but cannot erase, that chronology defect.

After the numerical result, the generated viewer path was found to be one parent too high. The
two-line integration-only repair is isolated in
[`20260723_beam_covariance_refit_POSTRUN_FIX.patch`](20260723_beam_covariance_refit_POSTRUN_FIX.patch);
reversing it reconstructs the preregistered/executed harness hash. It changes no numerical path or
artifact.

## Visual comparison

Initial/final Torch preview pairs are under
`runs/beam_covariance_refit_20260723/{ci,track-lsq,track-robust}/`. Qualitatively, CI and robust
start nearly black; LSQ starts visibly covered but blurred. All three become visually similar by
step 1,000, with the quantitative advantage above retained by LSQ.

Use one synchronized orbit camera for all six PLYs:

```bash
CUDA_VISIBLE_DEVICES='' .venv-cuda/bin/rtgs view \
  --comparison-manifest benchmarks/results/20260723_beam_covariance_refit_VIEWER.json \
  --max-viewer-gaussians 800 --device cpu --port 8782 --no-open
```

The base `.venv` does not contain the optional `viser` extra, so the successful smoke used the
existing `.venv-cuda` installation while forcing CPU. All six 800-splat endpoints loaded, PID
2014629 owned `127.0.0.1:8782`, HTTP returned 200, no NVIDIA compute process was listed, and the
server was stopped afterward. Receipt:
[`20260723_beam_covariance_refit_VIEWER_RECEIPT.json`](20260723_beam_covariance_refit_VIEWER_RECEIPT.json).
This WebGL view is qualitative and must not replace the exact Torch metrics.
