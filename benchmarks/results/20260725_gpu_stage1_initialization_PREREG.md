# Stage 1 on GPU: cover-consistent initialization under the production stack — preregistration

Date: 2026-07-25 (frozen before any GPU outcome)

## Why this exists

Everything established so far is a **downscale-32 CPU** result using the reference rasterizer and
the classic density controller. Three separate things could break it at production scale:

1. **The renderer.** The CPU reference adds a fixed `0.3 px²` screen-space floor, and that floor is
   the reason an under-sized primitive supplies only ~12% of its own rendered footprint. gsplat's
   antialiasing filter is different. **Run `benchmarks/gpu_dilation_probe.py` first** — if gsplat's
   floor is much smaller, the scale-suppression mechanism does not transfer and this experiment
   measures something else.
2. **The resolution.** At downscale 32 the object spans ~40 px, so a 2 px silhouette band holds 24%
   of the pixels. The stage-0 residual decomposition is re-measured here at real resolution; its
   CPU finding that boundary error dominates is expected to shrink and must not be assumed.
3. **The topology controller.** gsplat's Default strategy is not the CPU classic one, and MCMC
   replaces clone/split outright — which could erase the clone-in-place failure entirely.

## Preregistered treatment, and why it changed

The CPU screen's treatment was `surfel` (cover extent + orientation + coverage-derived opacity).
The measured decomposition says the opacity term should be dropped:

| arm | extent | opacity | init α-out | final PSNR (fixed topology) |
|---|---|---|---:|---:|
| `ci` | small | 0.10 | 0.0037 | 21.2741 |
| `cover-iso` | cover | 0.10 | 0.0437 | **21.8594** |
| `cover-iso-op` | cover | derived | 0.1851 | 21.5367 |
| `surfel` | cover | derived | 0.1312 | 21.6450 |

The derived opacity bought nothing downstream (+0.18% AUC, −0.046 dB versus control) and is what
produces the silhouette leak. `cover-iso` was the best fixed-topology arm and stayed inside the
0.05 leakage guardrail.

**`cover-iso` is therefore the preregistered treatment here.** This is a *new* preregistration
precisely because selecting it from the run that produced it would be post-hoc; that run's own
report records it as non-selected. `cover-surfel` and `cover-surfel-op` are labelled secondaries.

## Arms

All four share bit-identical means, SH/colour, and count; only quaternions, log-scales, and
opacity differ.

| arm | covariance | opacity | status |
|---|---|---|---|
| `ci` | unchanged Beam CI | 0.10 | control |
| `cover-iso` | isotropic at the cover sigma | 0.10 | **preregistered treatment** |
| `cover-surfel` | oriented flat surfel | 0.10 | secondary: does orientation add anything? |
| `cover-surfel-op` | oriented flat surfel | coverage rule | secondary: does the leak reappear? |

## Protocol

- `frame_00009` compact bundle, **downscale 4** (~938×410 per view, versus 117×51 on CPU).
- Train views `[0,3,6,9,12,15,18,21]`; **held-out** `[1,13,25]` = `C0004, C0025, C1004`,
  reporting only. `C1004` is extrapolative and is reported separately.
- Beam Fusion: unchanged configuration, **5,000** outputs.
- gsplat rasterizer, CUDA, seed 0, 7,000 steps, SH degree 3.
- Modes: `fixed` (no topology change) and `density` (gsplat Default). `mcmc` is available and
  should be run separately — it is a different mechanism, not another seed.
- Matched hard budget: `3 × n_init = 15,000` for every arm, so no arm can win by spending more.

## Preregistered decision

Primary: `cover-iso` versus `ci`, held-out foreground PSNR at step 7,000, in the `density` mode.

- **G-GPU1 (transfers)** — `cover-iso` ≥ `ci` + 0.15 dB **and** `cover-iso` final count ≤ `ci`
  final count.
- **G-GPU2 (does not transfer)** — `|cover-iso − ci|` < 0.15 dB. The CPU result is then reported
  as a CPU-reference-rasterizer artefact and the initialization claim is withdrawn for production.
- **G-GPU3 (reverses)** — `ci` ≥ `cover-iso` + 0.15 dB. Reported as a negative result.
- **Guardrail** — final held-out outside-mask alpha ≤ 0.05 for every arm.

Secondary, reported but not gated: the stage-0 residual decomposition at this resolution for every
arm (interior holes / interior appearance / boundary / leak), the `fixed` mode, the extrapolative
camera alone, and the two secondary arms.

## Interpretation limits fixed in advance

One scene, one seed. This tests **transfer to the production stack**, not generalization across
scenes — that still needs multiple mask-bearing captures, which the repository does not currently
have. **No default change is authorized by this run under any outcome.**

## Commands

```bash
# 0. Does the mechanism exist under gsplat at all? Run this first.
python benchmarks/gpu_dilation_probe.py --out runs/gpu_dilation_probe/summary.json

# 1. The preregistered comparison.
PYTHONUNBUFFERED=1 python benchmarks/gpu_stage1_initialization.py \
  --protocol benchmarks/results/20260725_gpu_stage1_initialization_PREREG.md \
  --out runs/gpu_stage1_initialization

# 2. Separately, the MCMC mechanism (different controller, not another seed).
PYTHONUNBUFFERED=1 python benchmarks/gpu_stage1_initialization.py \
  --protocol benchmarks/results/20260725_gpu_stage1_initialization_PREREG.md \
  --out runs/gpu_stage1_mcmc --modes mcmc
```
