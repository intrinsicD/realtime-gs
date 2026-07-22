# realtime-gs

Research repository testing one idea: **make 3D Gaussian Splatting (3DGS) reconstruction
fast by skipping the cold start.** Instead of initializing 3DGS from a sparse SfM point
cloud (or random points) and spending most of the optimization budget growing/placing
primitives, we:

1. **Fit every input image with 2D gaussians** (GaussianImage-style accumulated splatting —
   seconds per image, embarrassingly parallel across images).
2. **Lift the 2D gaussians into 3D** — each 2D gaussian already carries position, anisotropic
   shape, and color; only its depth (and the covariance along the ray) is missing. The main
   lifting interface exposes five named variants: A–D use the legacy RGB/depth scene contract,
   while E has a separate compact-only entry:
   - **A · `gradient`** — keep each gaussian on its camera ray and optimize per-gaussian
     depth by rendering into *other* views (multi-view photometric gradient descent).
   - **B · `depth`** — feed-forward monocular depth (Depth Anything V2 or similar) gives
     depth directly; the missing along-ray variance is estimated from the depth spread
     inside the gaussian's footprint.
   - **C · `carve`** — a voxel color-consistency volume (space-carving flavor) scores each
     gaussian's ray; gaussians from different views that land in the same cell are merged
     by moment matching.
   - **D · `hybrid`** — aligned monocular depth initializes each bounded ray, then a short
     multi-view photometric optimization corrects depth before confidence/color-aware fusion.
   - **E · `field`** — an image-free research path places source-anchored fibers from compact
     frozen 2D fields, refits an additive density/RGB-numerator proxy, and applies transactional
     topology proposals. Frozen StructSplat renderer semantics are validated separately.
   Checked-in compact captures also support RGB-free research initializers without pretending
   that their inputs match the legacy cohort: balanced top-K compact carve, dense+merge,
   confidence-gated easy-only, calibrated splat-SfM, tomographic beam fusion, complete field lift,
   and a camera-bounds random control. Classic SfM still requires sparse points; gradient/carve,
   depth, and hybrid require dense RGB and/or depth evidence absent from a compact-only bundle.
3. **Refine with standard 3DGS optimization** from this dense, structured initialization.
   CUDA runs can select gsplat's Default (AbsGS/revised opacity) or MCMC
   (relocation/teleportation + noise) strategy under a hard configurable primitive budget.

Rendering/refinement reuses the state-of-the-art CUDA stack ([gsplat](https://github.com/nerfstudio-project/gsplat))
on GPU; a pure-PyTorch reference rasterizer keeps the whole pipeline testable on CPU.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cpu
# On a GPU machine use a CUDA PyTorch wheel, then install .[cuda,depth,viewer,dev].

.venv/bin/rtgs run --scene synthetic --lifter depth   # end-to-end on a synthetic scene
.venv/bin/rtgs bench --quick                          # compare all lifting variants
./scripts/verify.sh                                   # lint + tests + docs-sync
```

The repository's checked-in captures are stored after Stage 1 rather than as RGB. Each
`frame_*/gaussians2d/` directory contains an ordered manifest and one self-contained `.rtgsv`
per camera; each file is capped at 168,000 decimal bytes and may include exact bit-packed alpha.
Load them without an image decoder or StructSplat runtime:

```python
from rtgs.data import CompactDataset

compact = CompactDataset.load(
    "dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d"
)
inputs = compact.to_reconstruction_inputs()
```

Run the native field lift directly from the same compact directory. The command loads no source
image, preserves optional packed alpha, creates an explicit deterministic held-out split, and
safely ignores pre-split bounds/points unless they are explicitly marked train-only. It saves the
3D initialization. `Path(--out).with_suffix(".field.npz")` contains field masses,
render opacity, fiber/source state, fitting/all-view correspondence visibility, gains, split
indices, and correspondences; strict
`Path(--out).with_suffix(".diagnostics.json")` contains semantic validation and diagnostics:

```bash
.venv/bin/rtgs lift-field \
  --dataset dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d \
  --heldout-stride 8 --field-args '{"max_tracks":128}' \
  --out runs/field-frame-00008/gaussians_init.ply
```

`field` is an implemented CPU-tested research path with one audited all-fitted-view development
execution, not a held-out/generalization, performance, topology-utility, or production-default
claim. Its analytic whole-plane loss is exact for additive peak-mixture density and RGB numerator
only; normalized finite-support/fade/affine StructSplat teachers are evaluated by separate bounded
deterministic sampled validation with train/held-out aggregates.

For a full compact-only comparison, the convergence harness accepts `topk`, `beam-fusion`,
`dense-merge`, `easy-only`, `splat-sfm`, `field`, and `random`. The suite operator runs each
successful arm through the same density schedule and fixed-topology plateau rule, records native
initial counts instead of silently trimming variable-count methods, and never opens source RGB:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv-cuda/bin/python benchmarks/run_compact_initializer_suite.py \
  --out runs/all_initializers_frame00008_20260721 \
  --protocol benchmarks/results/20260721_all_initializers_frame00008_PREREG.md \
  --keep-going
```

This is an all-fitted-view development comparison. It cannot establish held-out quality or make
RGB/depth/SfM-dependent methods lose by omission; the protocol contains the complete applicability
inventory and the method-specific parameters.

The audited result is deliberately a **non-winner**. Dense+merge has the highest fitted-view
foreground PSNR (38.2480 dB), while beam fusion has the lowest selected training objective
(0.002447); dense's objective is 4.40% worse, so it does not pass the frozen two-metric win rule.
Adaptive density grew every arm to 35.6k–49.2k Gaussians, and even random finished fourth by
foreground PSNR. The single-scene suite therefore identifies no materially superior converged
initializer and authorizes no default change:

| Initializer | Initial 3D Gaussians | Selected final | Fitted FG PSNR | Objective |
| --- | ---: | ---: | ---: | ---: |
| top-K | 5,000 | 43,288 @ 70k | 37.2992 | 0.002742 |
| beam fusion (historical) | 5,000 | 44,222 @ 69k | 37.8874 | **0.002447** |
| dense+merge | 2,088 | 49,177 @ 70k | **38.2480** | 0.002555 |
| easy-only | 7 | 35,644 @ 69k | 36.9587 | 0.002905 |
| splat-SfM | 943 | 39,987 @ 69k | 37.7063 | 0.002759 |
| field | 127 (128 before topology) | 39,059 @ 70k | 37.2408 | 0.002766 |
| random | 5,000 | 39,513 @ 70k | 37.4257 | 0.002680 |

See the frozen
[`protocol`](benchmarks/results/20260721_all_initializers_frame00008_PREREG.md),
[`result`](benchmarks/results/20260721_all_initializers_frame00008_RESULT.md), and independent
[`audit`](benchmarks/results/20260721_all_initializers_frame00008_AUDIT.md). All 26 views were fit,
native counts differ, timings are nonportable, and the field arm lacks the protocol-requested
individual topology move receipts; those limitations are part of the result, not footnotes to a
default-selection claim.

Compare every saved initial state against its selected final state in one orbit viewer with the
checked-in [`comparison manifest`](benchmarks/results/20260721_all_initializers_frame00008_VIEWER.json):

```bash
.venv-cuda/bin/rtgs view \
  --comparison-manifest benchmarks/results/20260721_all_initializers_frame00008_VIEWER.json \
  --max-viewer-gaussians 50000 --device cpu --port 8782
```

The **Gaussian set** selector is ordered by method, then `initial`/`final`, and includes the loaded
count in each label. Orbit once and switch entries: changing the set does not reset the browser
camera, which makes geometry changes easy to spot without running seven viewer processes. The cap
does not truncate this suite (its largest selected endpoint has 49,177 splats). All fourteen models
are loaded and prepared in host memory at startup, while only the selected set is transferred to
the WebGL view. This compact-only bundle has no source RGB, so this command intentionally provides
orbit inspection rather than calibrated reference-image snapshots.

The migration is resumable and deletes RGB/mask directories only after every bundle and source
hash verifies:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python scripts/convert_datasets_to_gaussians2d.py convert --remove-sources
.venv/bin/python scripts/convert_datasets_to_gaussians2d.py verify
```

The established `rtgs run --scene ...` path still expects an external RGB dataset. The separate
`rtgs lift-field` command consumes checked-in compact captures image-free and stops after Stage 2;
RGB-backed Stage-3 refinement remains a separate path.

For the calibrated object captures in the Janelle dataset, point `--scene` at one frame.
The loader finds `calibration_dome.json`, undistorts RGB and masks, uses every eighth camera as
held-out evaluation, and keeps evenly distributed cameras when `--max-images` is set:

```bash
python3 -m venv .venv-cuda
.venv-cuda/bin/pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu132
.venv-cuda/bin/pip install -e '.[cuda,depth,viewer,dev]'
.venv-cuda/bin/pip install -e ~/Documents/structsplat   # optional MIT stage-1 backend

.venv-cuda/bin/rtgs run \
  --scene ~/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008 \
  --downscale 16 --device cuda --fit-backend structsplat \
  --initial-gaussians 640 --max-gaussians 2000 --fit-iterations 300 \
  --lifter carve --lifter-args '{"grid_res":96}' \
  --refine-iters 7000 --density-strategy gsplat-default \
  --densify-start 200 --densify-stop 3500 --densify-every 100 \
  --max-3d-gaussians 30000 --target-sh-degree 3 --antialiased \
  --out runs/janelle-carve

# Fixed 640 control: 640 is the start, not a hard-coded ceiling.
.venv-cuda/bin/rtgs run --scene ~/Dropbox/Work/Janelle/karate/frame_00005 \
  --device cuda --fit-backend structsplat --initial-gaussians 640 \
  --max-gaussians 640 --no-adaptive-density --lifter hybrid --out runs/janelle-hybrid
```

`--initial-gaussians` and `--max-gaussians` are independent. StructSplat can grow from any
configured start until convergence or the maximum; the native backend keeps the initial count
fixed. Every `rtgs run --out ...` writes `gaussians_init.ply`, `gaussians.ply`, metrics and
training-history JSON, sampled calibrated-camera reference/init/final/error images,
`reconstruction_contact_sheet.png`, `reconstruction.gif`, an interpolated `novel_orbit.gif`,
and an off-plane `novel_elevation.gif` for geometry inspection. `rtgs refine` writes the same
metrics/history sidecars and visual diagnostics unless `--no-preview` is supplied.

For the most initialization-robust density control, substitute
`--density-strategy gsplat-mcmc`; this enables gsplat's low-opacity relocation/teleportation,
growth, and position noise. `classic` remains the CPU-compatible reference strategy. Default
strategy uses the published AbsGS-compatible gradient threshold automatically; the threshold,
opacity-reset interval, density window, and final budget remain explicit CLI controls.

Open the saved result in the interactive browser viewer (Viser is an optional Apache-2.0
dependency):

```bash
.venv-cuda/bin/rtgs view \
  --gaussians runs/janelle-carve/gaussians.ply \
  --scene ~/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008 \
  --downscale 16 --device cuda --rasterizer gsplat
```

The viewer auto-detects a sibling `gaussians_init.ply`, supports orbit navigation,
initial/final switching, significance-ranked splat-count and opacity controls, and train/test
camera filtering. Click a camera or choose it in the sidebar; **Render exact snapshot** compares
the reference RGB against a full-SH render from the selected `Rasterizer` backend. The orbitable
WebGL preview refreshes its RGB-only splats from the model's full SH coefficients as the camera
moves; the exact snapshot remains authoritative for sorting and rasterization. For a remote GPU
host, add `--host 0.0.0.0 --no-open` and use SSH port forwarding.
`--max-viewer-gaussians` is only a configurable browser-transfer cap; it never changes the
reconstruction. Training writes `gaussians.config.json`; `view` and `render` automatically reuse
its packed/antialiased render mode, with explicit `--[no-]packed` and `--[no-]antialiased`
overrides available. For multiple methods, `--comparison-manifest` accepts the strict
`rtgs.viewer-comparison.v1` schema shown above and resolves model paths relative to the manifest.

To watch a long fit without sharing the training GPU, run the viewer on CPU and point it at the
checkpoint directory:

```bash
.venv-cuda/bin/rtgs view --gaussians runs/example/gaussians_init.ply \
  --watch-checkpoints runs/example/checkpoints --device cpu
```

The viewer attempts only the newest named PLY checkpoint, leaves the last valid model visible, and
retries if it catches a file while it is still being written. `--device cpu` keeps the viewer
server off CUDA, but checkpoint I/O, CPU work, and host-memory use are nonzero. The orbit preview
still runs as WebGL in the browser. If that browser is on the training
workstation, its graphics process can still consume the display GPU. For the lowest interference,
run the server on CPU and open it from a second machine through an SSH tunnel; “zero performance
impact” requires a controlled on/off measurement and is not claimed.

The default depth checkpoint is the Apache-2.0 Depth Anything V2 Small model. Other checkpoint
names are rejected unless their code and weights have been explicitly license-verified.

## Documentation

| Doc | Contents |
| --- | --- |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module map, dataflow, backend abstractions, CLI |
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | State-of-the-art survey and what we reuse from where |
| [`docs/RESEARCH_LOOP.md`](docs/RESEARCH_LOOP.md) | Reusable three-iteration R&D prompt and execution record |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Milestones and open questions |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | How to benchmark + tracked results |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Dated experiment log (positive and negative results) |
| [`CLAUDE.md`](CLAUDE.md) | Agent guide: hard rules, commands, workflows |

## Status

Early research code. The full pipeline runs end-to-end on synthetic scenes, COLMAP datasets, and
the calibrated object-capture JSON format. The gsplat CUDA path, optional StructSplat CUDA fitter,
and Depth Anything V2 Small backend have been exercised on an RTX 4090; CPU remains the reference
and CI path.
