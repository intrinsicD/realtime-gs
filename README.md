# realtime-gs

Research repository testing one idea: **make 3D Gaussian Splatting (3DGS) reconstruction
fast by skipping the cold start.** Instead of initializing 3DGS from a sparse SfM point
cloud (or random points) and spending most of the optimization budget growing/placing
primitives, we:

1. **Fit every input image with 2D gaussians** (GaussianImage-style accumulated splatting —
   seconds per image, embarrassingly parallel across images).
2. **Lift the 2D gaussians into 3D** — each 2D gaussian already carries position, anisotropic
   shape, and color; only its depth (and the covariance along the ray) is missing. Three
   competing variants supply it:
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
overrides available.

The default depth checkpoint is the Apache-2.0 Depth Anything V2 Small model. Other checkpoint
names are rejected unless their code and weights have been explicitly license-verified.

## Documentation

| Doc | Contents |
| --- | --- |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module map, dataflow, backend abstractions, CLI |
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | State-of-the-art survey and what we reuse from where |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Milestones and open questions |
| [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) | How to benchmark + tracked results |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Dated experiment log (positive and negative results) |
| [`CLAUDE.md`](CLAUDE.md) | Agent guide: hard rules, commands, workflows |

## Status

Early research code. The full pipeline runs end-to-end on synthetic scenes, COLMAP datasets, and
the calibrated object-capture JSON format. The gsplat CUDA path, optional StructSplat CUDA fitter,
and Depth Anything V2 Small backend have been exercised on an RTX 4090; CPU remains the reference
and CI path.
