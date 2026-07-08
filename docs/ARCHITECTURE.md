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
 (COLMAP or         │   gradient.py: ray-constrained depth+rot+scale │
  synthetic)        │     multi-view descent (color frozen) + merge  │
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
| `rtgs/image2gs` | Stage 1. `renderer2d` renders a `Gaussians2D` set with accumulated (sum) blending, differentiably, chunked over pixels. `fit` optimizes positions/covariances/colors/weights against an image with gradient-magnitude initialization. |
| `rtgs/lift` | Stage 2. `base` holds the `Lifter` protocol and shared geometry (2D cov → 3D cov lifting, along-ray sigma estimation). Variants: `gradient` (lift with a footprint-scaled along-ray thickness — the "epsilon" `ray_thickness` knob — then optimize depth + rotation + scale along each pixel ray with color/opacity frozen, and merge redundant gaussians), `depth`, `carve`. `merge` implements moment-matched gaussian merging (used by `gradient`/`carve` and available as a generic post-process). Registry: `rtgs.lift.get_lifter(name)`. |
| `rtgs/depth` | Depth estimation behind the `DepthBackend` protocol: `mock` (ground-truth/constant, for tests and synthetic scenes), `depth_anything` (Depth Anything V2 via `transformers`, lazy import), `align` (least-squares scale/shift alignment of relative depth to sparse 3D points). |
| `rtgs/render` | Rasterization behind the `Rasterizer` protocol: `torch_ref` (pure-PyTorch EWA splatting + depth-sorted alpha compositing; the correctness anchor, CPU-capable, fully differentiable) and `gsplat_backend` (CUDA, lazy import). `get_rasterizer("auto")` picks gsplat when CUDA is available. |
| `rtgs/optim` | Stage 3. `trainer` runs the standard 3DGS optimization (Adam with per-group LRs, L1 + D-SSIM); `density` implements adaptive density control driven by screen-space positional gradients. |
| `rtgs/data` | `synthetic` builds fully ground-truthed test scenes (random 3D gaussians + ring of cameras, rendered with the reference rasterizer, GT depth included); `colmap` parses COLMAP sparse reconstructions (text and binary) into cameras + points. |
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
| `rtgs run ...` | End-to-end (fit → lift → refine) on `--scene synthetic` or a COLMAP dir. |
| `rtgs render ...` | Render a saved gaussian set from a camera path / dataset cameras. |
| `rtgs bench ...` | Delegates to `benchmarks/run.py` (variant comparison + micro-benchmarks). |

## Backend abstractions (hard rule: pluggable, CPU-first)

- **Rasterizer** (`rtgs.render.base.Rasterizer`): `render(gaussians3d, camera, bg) -> RenderOutput(color, alpha, depth, means2d)`. `torch_ref` is authoritative for semantics; `gsplat_backend` must match it (parity test, `@pytest.mark.cuda`). The trainer and the `gradient` lifter only speak to this interface.
- **DepthBackend** (`rtgs.depth.base.DepthBackend`): `predict(image) -> DepthPrediction(depth, kind)` where kind ∈ {`metric`, `relative`, `affine`}. Non-metric predictions must be aligned (`rtgs.depth.align`) before lifting.

No module imports CUDA-only or heavyweight optional dependencies at import time; they are
imported inside functions and failures produce actionable error messages.

## Conventions

- Camera extrinsics are **world-to-camera** (COLMAP): `x_cam = R @ x_world + t`; `Camera.position` is the camera center in world space. +z is the viewing direction (OpenCV).
- Images are float32 tensors in `[0,1]`, shape `(H, W, 3)`; pixel `(u, v)` = (column, row); the pixel center of the top-left pixel is `(0.5, 0.5)`.
- 2D covariances are parametrized by Cholesky factors `(l11, l21, l22)` with positive diagonal (GaussianImage). 3D covariances by unit quaternion + log-scales (3DGS).
- Colors in `Gaussians3D` are SH coefficients `(N, K, 3)`, `K = (deg+1)^2`; degree-0 stores `(rgb - 0.5)/C0` (3DGS convention).
- Everything is `torch.float32`; tests seed all RNGs.
