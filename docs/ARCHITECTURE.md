# Architecture

## Dataflow

```
                    ┌────────────────────────────────────────────────┐
 images  ─────────► │ stage 1  rtgs.image2gs                         │
 (per view)         │   fit.py: N 2D gaussians per image             │
                    │   renderer2d.py: differentiable accumulated    │
                    │   splatting (no sorting, GaussianImage-style)  │
                    └───────────────┬────────────────────────────────┘
                                    │ Gaussians2D per view (xy, cholesky cov, color, weight)
                    ┌───────────────▼────────────────────────────────┐
 cameras ─────────► │ stage 2  rtgs.lift   (three variants)          │
 (COLMAP, JSON,     │   gradient.py: bounded ray depth+rot+scale     │
  or synthetic)     │     multi-view descent (color frozen) + merge  │
                    │   depth.py: monocular depth backend + footprint│
                    │     variance for the along-ray sigma           │
                    │   carve.py: voxel color-consistency carving,   │
                    │     ray-tunnel placement, moment-match merging │
                    └───────────────┬────────────────────────────────┘
                                    │ Gaussians3D (means, quats, scales, opacity, SH)
                    ┌───────────────▼────────────────────────────────┐
                    │ stage 3  rtgs.optim                            │
                    │   trainer.py: L1+D-SSIM 3DGS loop              │
                    │   density.py: clone / split / prune /          │
                    │     opacity-reset (screen-space grad driven)   │
                    └───────────────┬────────────────────────────────┘
                                    ▼
                          refined Gaussians3D (.ply / .npz)
```

## Subpackages

| Package | Responsibility |
| --- | --- |
| `rtgs/core` | Shared math and containers: `gaussians2d` (xy, Cholesky cov, color, weight), `gaussians3d` (means, quats, log-scales, opacity, SH; PLY/NPZ IO), `camera` (COLMAP-convention pinhole, project/unproject/rays), `sh` (real spherical harmonics deg ≤ 3), `metrics` (PSNR, SSIM). |
| `rtgs/image2gs` | Stage 1. `renderer2d` performs sparse accumulated (sum) blending and exposes color-independent coverage; `fit` optimizes masked images with gradient-magnitude initialization. `adapters` converts native, StructSplat RS, and GaussianImage-style NPZ fields into the common Cholesky representation. |
| `rtgs/lift` | Stage 2. `base` implements projection-consistent covariance lifting and depth-surface covariance. `gradient` bounds every optimized depth to its ray/object-volume intersection, regularizes depth and scale, and merges observations with conservative opacity. `depth` lifts raw metric or aligned inverse depth. `carve` uses real masks when present and otherwise color-independent Gaussian coverage. `merge` performs moment matching. Registry: `rtgs.lift.get_lifter(name)`. |
| `rtgs/depth` | Depth estimation behind the `DepthBackend` protocol: `mock`; permissive-allowlisted Depth Anything V2 Small through `transformers` (lazy import); robust scale/shift alignment to per-view COLMAP tracks; and object-bounds alignment when calibrated captures have no sparse points. |
| `rtgs/render` | Rasterization behind the `Rasterizer` protocol: `torch_ref` (pure-PyTorch EWA splatting + depth-sorted alpha compositing; the correctness anchor, CPU-capable, fully differentiable) and `gsplat_backend` (CUDA, lazy import). `get_rasterizer("auto")` picks gsplat when CUDA is available. |
| `rtgs/optim` | Stage 3. `trainer` runs device-aware masked 3DGS optimization on training views, reports held-out views, and progressively enables SH bands. `density` consumes torch/gsplat screen gradients, preserves Adam state, and enforces a hard primitive budget. |
| `rtgs/data` | `synthetic` builds ground-truthed tests; `colmap` parses text/binary reconstructions and observation tracks; `calibrated` loads the object-capture JSON format, applies OpenCV distortion correction to RGB/masks, preserves view ids, estimates object bounds, and creates an every-eighth train/test split. |
| `rtgs/pipeline` | `pipeline.py` orchestrates stages 1–3 with timing and per-stage metrics; `compare_lifters` runs all variants on one scene. |
| `rtgs/cli` | `cli.py`, argparse-based. |

Registered lifters: `gradient`, `depth`, `carve` (plus `sfm` baseline that mimics classic
SfM-point initialization for comparison, and `random` as the lower-bound baseline).

## CLI

| Command | Purpose |
| --- | --- |
| `rtgs fit-images ...` | Stage 1 only: fit 2D gaussians to images in a directory, save `.npz` per image. |
| `rtgs lift ...` | Stage 2 only: lift fitted 2D gaussians into a 3D gaussian set. |
| `rtgs refine ...` | Stage 3 only: run 3DGS optimization from an initialization. |
| `rtgs run ...` | End-to-end on synthetic, COLMAP, or calibrated-frame data; `--fits` skips stage 1 using native/StructSplat/GaussianImage NPZ files. |
| `rtgs render ...` | Render a saved gaussian set from a camera path / dataset cameras. |
| `rtgs bench ...` | Delegates to `benchmarks/run.py` (variant comparison + micro-benchmarks). |

## Backend abstractions (hard rule: pluggable, CPU-first)

- **Rasterizer** (`rtgs.render.base.Rasterizer`): `render(gaussians3d, camera, bg) -> RenderOutput(color, alpha, depth, means2d)`. `torch_ref` is authoritative for semantics; `gsplat_backend` must match it (parity test, `@pytest.mark.cuda`). The trainer and the `gradient` lifter only speak to this interface.
- **DepthBackend** (`rtgs.depth.base.DepthBackend`): `predict(image) -> DepthPrediction(depth, kind)` where kind ∈ {`metric`, `relative`, `affine`, `inverse`}. Non-metric predictions are aligned (`rtgs.depth.align`) before lifting.

No module imports CUDA-only or heavyweight optional dependencies at import time; they are
imported inside functions and failures produce actionable error messages.

## Conventions

- Camera extrinsics are **world-to-camera** (COLMAP): `x_cam = R @ x_world + t`; `Camera.position` is the camera center in world space. +z is the viewing direction (OpenCV).
- Images are float32 tensors in `[0,1]`, shape `(H, W, 3)`; pixel `(u, v)` = (column, row); the pixel center of the top-left pixel is `(0.5, 0.5)`.
- 2D covariances are parametrized by Cholesky factors `(l11, l21, l22)` with positive diagonal (GaussianImage). `weight` is accumulated-render amplitude, not alpha; lifted observations start with independent conservative opacity. 3D covariances use unit quaternion + log-scales (3DGS).
- Colors in `Gaussians3D` are SH coefficients `(N, K, 3)`, `K = (deg+1)^2`; degree-0 stores `(rgb - 0.5)/C0` (3DGS convention).
- Everything is `torch.float32`; tests seed all RNGs.
