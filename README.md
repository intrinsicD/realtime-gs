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
3. **Refine with standard 3DGS optimization** (density control included) from this dense,
   structured initialization — the hypothesis is that far fewer iterations are needed.

Rendering/refinement reuses the state-of-the-art CUDA stack ([gsplat](https://github.com/nerfstudio-project/gsplat))
on GPU; a pure-PyTorch reference rasterizer keeps the whole pipeline testable on CPU.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cpu
# On a GPU machine additionally: .venv/bin/pip install -e '.[cuda,depth]'

.venv/bin/rtgs run --scene synthetic --lifter depth   # end-to-end on a synthetic scene
.venv/bin/rtgs bench --quick                          # compare all lifting variants
./scripts/verify.sh                                   # lint + tests + docs-sync
```

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

Early research code. The full pipeline runs end-to-end on synthetic scenes and COLMAP
datasets on CPU; GPU fast paths (gsplat rasterization, Depth Anything V2) activate
automatically when their dependencies are installed.
