# Full `frame_00008` tomographic beam-fusion experiment — preregistration

Date frozen: 2026-07-21, before any full-view beam-fusion output or downstream beam-initialized
optimization result was opened.

## Question and scope

Can bounded tomographic beam fusion consume every compact 2D Gaussian in the calibrated
`frame_00008` bundle, produce a useful 5,000-primitive 3D initialization, and reach the existing
compact-training convergence criterion under the established full-reconstruction schedule?

This is a single-scene, all-view development experiment. All 26 views participate in beam fusion
and fitting. Consequently, every image metric is a fitted-view diagnostic: this protocol makes no
held-out, novel-view, generalization, or default-selection claim.

## Immutable input and arms

Input directory:
`dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d`.

- 26 calibrated compact views in manifest order;
- 5,000 optimized 2D Gaussians per view, 130,000 total;
- manifest semantic digest
  `0f86429b4cf503df3ad46ca84a9346c9ab1ada51509d90e13ae9fb241d2a8ef5`;
- calibration SHA-256
  `51b8fc396fc8447f24e325e0a525f2e7d422388790dd9a293e1a81804b265091`;
- manifest bounds center `[0.3415684700, 0.1410396844, 2.7468976974]`, extent
  `2.2361571789`.

Two initializations are evaluated against byte-identical materialized compact targets:

1. `topk_control`: the established component-center compact placer, seed 0, 32 depth samples,
   `min_views=2`, robust view fraction 0.60, minimum score 0.01, and exactly 5,000 requested 3D
   Gaussians.
2. `beam_fusion`: every one of the 325 view pairs and every 5,000-by-5,000 cross-view ray pair is
   evaluated. Gates are `min_views=3`, transverse `3.0 sigma`, maximum RGB distance `0.35`, RGB
   sigma `0.25`, and fold-in `3.0 sigma`. Covariance intersection uses equal contributor weights.
   The along-ray bounds use `near=0.05` and `bounds_scale=0.5`; covariance sigma is clamped to
   `[1e-4, 0.5*extent]`; initial opacity is 0.10. Pair matrices stream with source chunk 256.
   The strongest seed per 3D voxel is retained, where voxel size is the implementation's frozen
   default `0.01*extent` (`0.0223615718` here). The top 20,000 seed voxels (4x oversubscription)
   are folded into all views in chunks of 512; contributor-signature dedupe and final voxel NMS
   retain at most 5,000 Gaussians. No view-pair limit is applied.

If beam fusion yields fewer than 5,000 valid post-NMS components, fitting still runs and the actual
count is reported, but the initialization comparison is explicitly not count-matched.

Primary initialization diagnostic: all-26-view mean foreground PSNR. Full PSNR, crop metrics,
SSIM, per-view metrics, count, lineage/view-multiplicity histogram, unmatched components, placement
time, evaluated ray pairs, gated seed count, retained voxel count, and peak pair matrix are also
saved. A positive or negative initialization difference is descriptive for this one fitted scene.

## Frozen downstream fit and stopping rule

Only the beam initialization receives a new downstream fit. The fit never opens source RGB: each
target is a deterministic native-resolution StructSplat replay multiplied by its packed lossless
alpha. Optimization uses CUDA gsplat 1.5.x, packed and antialiased rendering, fixed black
background, masked standard 3DGS loss, target SH degree 3 activated at 1,000-step intervals, seed
0, and the repository's standard per-parameter Adam learning rates.

The parent segment runs 30,000 steps. Gsplat DefaultStrategy density control starts at step 500,
stops at 15,000, runs every 100 steps with AbsGS threshold `8e-4`, prunes below opacity `0.005` or
scale fraction `0.1`, and enforces a 100,000-Gaussian hard cap. Evaluation and viewer-ready PLY
checkpoints occur every 1,000 steps.

After 30,000, fitting continues in fixed-topology 10,000-step segments using the already landed
`polish`, `tail`, `cooldown`, and `settle` protocol, to at most global step 70,000. Each segment is
a declared non-exact restart because PLY does not preserve Adam/RNG state. The first continuation
uses the existing terminal means learning rate and multiplies other learning rates by 0.25; the
50k-to-60k and 60k-to-70k segments apply the repository's frozen additional 0.25 cooldown factors.

After each segment, compact targets alone choose the earliest checkpoint within `1e-6` relative
objective of the segment minimum. Stop when either:

- the joint convergence status is `plateau`, requiring both (a) five consecutive 1k transitions
  with neither at least 0.25% objective reduction nor at least 0.05 dB foreground-PSNR gain, and
  (b) the frozen last-six trend rule (absolute Theil-Sen slope <=0.01 dB/1k, recent median gain
  <=0.05 dB, median per-view objective improvement <=0.5%, and at most 20% of views improving by
  more than 1%); or
- the status is `regression`, which terminates without a convergence claim.

If the 70,000-step ceiling is reached with `still_improving`, report that the fit did not converge
under this budget. “Converged” means only this compact-training plateau criterion, never a global
or mathematical optimum.

The historical 2026-07-20 top-K-initialized 70k result may be shown as context, but its dirty,
non-exact recovery chain and different executed environment prohibit a causal downstream
beam-vs-top-K claim. No production default changes on this experiment.

## Viewer and performance accounting

The training callback saves an isolated PLY only at the already-frozen 1,000-step evaluation
cadence. A separate CPU viewer process may poll the checkpoint directory and replace its WebGL
model. Exact snapshots, scene RGB, and CUDA viewer rendering remain disabled while optimization is
active. Training records callback seconds separately from native optimizer elapsed time. The
watcher is designed to make interference small, not literally zero; wall time and callback time
must be reported, and no timing claim is permitted from a viewer-contended run.

Viewer command template:

```bash
.venv-cuda/bin/rtgs view \
  --gaussians runs/beam_fusion_full_frame00008_fit_20260721/gaussians_init.ply \
  --watch-checkpoints runs/beam_fusion_full_frame00008_fit_20260721/checkpoints \
  --max-viewer-gaussians 50000 --device cpu \
  --host 127.0.0.1 --port 8780 --no-open
```

## Required disposition

Save configs, input/source hashes, placement lineage, initial/final/checkpoint PLYs, target hashes,
training histories, compact metrics, convergence selections, exact commands, peak RSS/VRAM, and a
viewer HTTP smoke. Append the result to `docs/EXPERIMENTS.md`, run the repository results-audit
skill, and distinguish proven mechanism behavior, measured single-scene diagnostics, and remaining
assertions. Any non-finite tensor, cap violation, missing artifact, source drift, or viewer failure
is reported rather than repaired by changing this protocol after outcome access.
