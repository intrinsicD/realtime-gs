# Preregistration — all-26-view Beam carrier maturation

**Frozen:** 2026-07-27, before creating the outcome directory
**Status:** development/mechanism experiment; all views are fitted
**Output:** `runs/carrier_maturation_all26_frame00008_20260727/`

## Question

Does the complete ADR-002 process behave as intended when Beam Fusion uses all 26 compact Janelle
views: sequential field repair, fixed-topology convergence, three clone-every-row tangent waves
with convergence between waves, higher-SH appearance expansion, and finally ordinary 3DGS
density control followed by a fixed-topology recovery?

This run also asks where quality changes: after each repair, immediately after every clone wave,
after each recovery, and throughout optimization.

## Scope and claim boundary

All 26 cameras are used for fitting, stopping, and reporting. There are **no held-out views**.
Consequently:

- this is an all-fitted-view reconstruction and mechanism diagnostic;
- it may compare the dynamics and final fit with the earlier all-view compact-target run;
- it cannot establish generalization, novel-camera accuracy, or a production default;
- results are outcome-exposed, single-scene, and single-seed;
- no threshold or schedule may be changed after the first outcome is read.

The later error-driven cloning experiment described in ADR-002 is not run here. Clone selection is
uniform by design.

## Frozen inputs

- Compact captures:
  `dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d`
- Exact sorted dataset order, 26 cameras:
  `C0001,C0004,C0005,C0006,C0008,C0009,C0012,C0014,C0018,C0019,C0020,C0021,`
  `C0022,C0025,C0026,C0028,C0029,C0030,C0031,C0034,C0037,C0039,C1000,C1001,`
  `C1002,C1004`
- Beam input count: 26 × 5,000 = 130,000 compact 2D Gaussians.
- Native RGB/mask sources are resolved from each compact view's source binding and must match its
  stored SHA-256 digest.
- Native optimization/reporting resolution: 5328×4608.
- Compact-teacher reporting uses the deterministic StructSplat replay from
  `benchmarks/full_compact_reconstruction.py`, `reference` renderer, chunk 4096, alpha applied.
- Seed: 27027.
- Hardware/runtime: local NVIDIA RTX 4090; `.venv-cuda`; PyTorch 2.12.0+cu132; gsplat 1.5.3.

## Frozen Beam and repair pipeline

Beam Fusion:

- `min_views=3`
- `transverse_gate_sigma=3.0`
- `max_color_distance=0.35`
- `color_sigma=0.25`
- `fold_in_gate_sigma=3.0`
- `nms_voxel_size=scene_extent/100`
- `init_opacity=0.10`
- `source_chunk=256`
- `max_components=5000`
- `seed_budget_multiplier=4`

The 5,000 retained carriers are then repaired in this exact order:

1. covariance only; means, opacity, and appearance remain bit-exact;
2. optical-density opacity only; means, covariance, and appearance remain bit-exact;
3. robust amplitude-weighted SH0/color only; means, covariance, and opacity remain bit-exact;
   higher SH bands remain disabled.

`CarrierRepairConfig` uses its code defaults: 120 covariance Adam steps, 80 opacity Adam steps, and
5 robust appearance IRLS steps. The contributor CSR emitted by this Beam run is fixed throughout
repair; no correspondence or topology may change.

## Frozen maturation schedule

All dense optimization uses native masked RGB, the CUDA gsplat rasterizer with packed and
antialiased rendering, no random background, hard SH color activation, and CPU-streamed scene
targets. One camera is sampled per optimizer update. Loss is the repository's masked 3DGS loss:
weighted L1 + D-SSIM + alpha supervision and configured regularization.

Every train metric checkpoint evaluates all 26 native views. A plateau is based only on the
arithmetic mean foreground PSNR over those 26 training views. At a checkpoint, patience resets
only when PSNR exceeds the previous qualifying best by more than the frozen `min_delta`.

1. **Fixed topology / SH0**
   - optimize means, quaternion/covariance, scale/covariance, opacity, and SH0;
   - no clone, split, insert, or prune;
   - evaluate every 1,000 updates;
   - plateau: minimum 4,000 updates, six non-qualifying evaluations, `min_delta=0.01 dB`;
   - safety cap: 30,000 updates.

2. **Three uniform tangent clone waves**
   - count schedule: 5,000→10,000→20,000→40,000;
   - every row current at that boundary is cloned exactly once;
   - every parent survives;
   - child quaternion, covariance/scale, opacity, and SH are inherited;
   - child opacity scale is 1.0;
   - displacement is a zero-mean Gaussian with standard deviation 10% of the parent scale on the
     two local axes orthogonal to its shortest principal axis;
   - displacement on the shortest-axis/local-normal coordinate is exactly zero before rotation;
   - immediate post-clone state is saved and fully evaluated;
   - after each wave, optimize means, covariance, opacity, and SH0 with fixed topology;
   - recovery plateau: minimum 3,000 updates, five non-qualifying evaluations,
     `min_delta=0.01 dB`, safety cap 15,000 updates.

3. **Higher-SH appearance**
   - extend the current model to SH degree 3;
   - freeze means, quaternion, scale, and opacity;
   - optimize SH0 and higher bands;
   - plateau: minimum 3,000 updates, five non-qualifying evaluations,
     `min_delta=0.01 dB`, safety cap 15,000 updates.

4. **Standard 3DGS growth**
   - repository classic screen-gradient clone/split/prune controller;
   - 30,000 updates;
   - density events from update 500 through 15,000 every 100 updates;
   - gradient threshold `2e-4`, split scale fraction `0.01`, split factor `1.6`;
   - prune opacity `0.005`, prune scale fraction `0.1`;
   - opacity reset every 3,000 updates to `0.011`;
   - hard maximum 100,000 Gaussians;
   - all fields and SH degree 3 optimize;
   - the last 15,000 updates provide recovery after the final possible topology event.

5. **Standard fixed-topology settle**
   - no further topology changes;
   - optimize all fields at frozen low continuation learning rates:
     means `1e-7`, quaternion `1.5625e-5`, scale `7.8125e-5`, opacity `7.8125e-4`,
     SH0 `3.90625e-5`, higher SH `1.953125e-6`;
   - plateau: minimum 5,000 updates, six non-qualifying evaluations,
     `min_delta=0.005 dB`, safety cap 40,000 updates.

A phase that reaches its safety cap without the frozen plateau is reported as `max_iterations`,
not as converged. No result may silently relabel it.

## Frozen measurements

At every optimizer update:

- total loss;
- L1, D-SSIM, and alpha terms;
- their weighted contributions;
- opacity and scale regularization values;
- sampled camera and active SH degree in the phase history.

At every 1,000-update metric checkpoint:

- mean native fitted-view foreground PSNR;
- crop PSNR and crop SSIM;
- alpha IoU, alpha inside, and alpha outside;
- Gaussian count and active SH degree.

At all 14 semantic boundaries—Beam, each of the three repairs, fixed-topology plateau, each
post-clone and post-recovery pair, higher-SH plateau, standard-growth completion, and final
settle—the run saves:

- PLY;
- native 26-view per-camera plus mean/median/min/max PSNR, SSIM, LPIPS-Alex/max-512, and alpha
  diagnostics;
- equivalent deterministic compact-teacher metrics;
- carrier lineage/survival, displacement, covariance drift, opacity drift, and descendant counts;
- a calibrated C0031 target/prediction/error/alpha strip.

Checkpoint PLYs are additionally saved every 5,000 updates. Final qualitative contact-sheet and
novel-orbit artifacts use the same cameras at 1/8 resolution; these are visual aids only and do
not supply table metrics.

## Contextual comparison

The earlier all-view Beam run is a contextual target-comparable reference only:

- initialization:
  `runs/beam_fusion_full_frame00008_fit_20260721/initial_compact_metrics.json`
- selected 69k result:
  `runs/beam_fusion_full_frame00008_settle_60000_70000_20260721/compact_metrics.json`

Its reported 37.8874 dB is an optimized compact-teacher fitted-view score, not Beam initialization.
The present run optimizes native masked RGB. Only its separately reported compact-teacher column
may be numerically compared with that historical compact column.

## Frozen artifact and handoff requirements

The output must contain root initial/final PLYs, phase histories, the full per-update trajectory,
all boundary PLYs/metrics/previews, clone receipts, source hashes, protocol/config/manifest,
comparison viewer manifest, `index.html`, summary-bound metrics, and final visual artifacts.

After completion:

1. run an adversarial results audit;
2. write the result and audit records;
3. launch and smoke-test `rtgs view` on the saved comparison manifest;
4. serve `index.html` from the repository root and require HTTP 200 for it and its local targets;
5. record both exact commands in a smoke receipt;
6. pass `scripts/check_results_bundle.py`;
7. append the bounded result to `docs/EXPERIMENTS.md`.

## Frozen command

```bash
.venv-cuda/bin/python benchmarks/carrier_maturation_all26.py \
  --out runs/carrier_maturation_all26_frame00008_20260727 \
  --protocol benchmarks/results/20260727_carrier_maturation_all26_PREREG.md \
  --raw-frame /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008
```

## Implementation binding

Git HEAD at freeze: `dd84c28deb3378d57992cd10b20f08bb594f102a`

- `benchmarks/carrier_maturation_all26.py`:
  `21249b963375fd6e3cc16df23b593fc8aa0e79defd697e0f641a4bf45c96c1a3`
- `src/rtgs/lift/beam_fusion.py`:
  `575c12fdb59ad7a430178ed5899eb9d546cddc965f50617eeed0b40fe9ca2e12`
- `src/rtgs/lift/carrier_refinement.py`:
  `bbd50a727da29663e22415376c8cabf269bf561e929a76ae86a4360dbb5590f5`
- `src/rtgs/optim/carrier_schedule.py`:
  `97c089c683ba0f86bafa302f674613a34d222b8faca5d697a6366f9eb1a8d17c`
- `src/rtgs/optim/density.py`:
  `a18be4d1b425177b74db8fb4ef814e53f4b246450d3e8f5f149962328110680a`
- `src/rtgs/optim/trainer.py`:
  `228a5269d02fe33e8d5981c1fd83ee79211b24d485d10bd6be59b13ff2432fed`
- `src/rtgs/visualize.py`:
  `0cd822b475fe7a8cf4e8738d28d0ac27f1c06b4bb7cd30f5592b4eff2c5b63d7`
- `benchmarks/full_compact_reconstruction.py`:
  `9ee2688f5d4f18c46790cd572118503bb11d83b86d821ffe749930b1eb8be722`
- `docs/ADR-002-carrier-refinement.md`:
  `081e0fca8cd64829953e92e59047dca48ef3b3df12bca00b9162fb2c4844d027`
