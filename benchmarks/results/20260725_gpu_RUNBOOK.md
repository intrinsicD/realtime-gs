# GPU runbook — stages 0, 1, and how stage 2 gets designed

Everything in this line so far is a downscale-32 CPU result. This is the order to run things on a
CUDA box, what each step decides, and why stage 2 is deliberately **not** written yet.

## Before anything: smoke-test on CPU

Every harness below runs on CPU at toy scale, so a typo costs seconds instead of GPU hours:

```bash
python benchmarks/gpu_dilation_probe.py --backends torch \
  --out /tmp/probe.json

python benchmarks/gpu_stage1_initialization.py \
  --protocol benchmarks/results/20260725_gpu_stage1_initialization_PREREG.md \
  --out /tmp/g1 --downscale 32 --n-init 300 --iterations 40 \
  --rasterizer torch --device cpu --arms ci cover-iso --modes fixed density-classic
```

`density-classic` exists only for this: it swaps gsplat's Default strategy for the CPU classic
controller so the full path is exercisable without a GPU. The frozen protocol uses `density`.

## Step 0 — does the mechanism even exist under gsplat? (minutes)

```bash
python benchmarks/gpu_dilation_probe.py --out runs/gpu_dilation_probe/summary.json
```

Renders one isotropic Gaussian at swept scales and recovers the backend's screen-space floor from
the alpha-weighted second moment. Validated against the CPU reference, where it recovers
**0.556 px** against the known **0.548 px** — so the method is sound and any gsplat number it
reports is trustworthy.

**This gates everything else.** The CPU story is: a 0.548 px floor absorbs ~88% of any scale
change, so an under-sized primitive's scale gradient is 10× too weak and it cannot grow fast enough
to become split-eligible before densification ends.

- **gsplat floor ≈ 0.5 px** → the mechanism transfers; proceed as written.
- **gsplat floor ≪ 0.5 px** → the mechanism does *not* transfer. Primitives can grow freely, and
  the CPU result may be a reference-rasterizer artefact. Still run step 1 — but the expected
  outcome flips to G-GPU2, and that is a legitimate, publishable negative.

## Step 1 — the preregistered initialization comparison (hours)

Protocol: [`20260725_gpu_stage1_initialization_PREREG.md`](20260725_gpu_stage1_initialization_PREREG.md)

```bash
PYTHONUNBUFFERED=1 python benchmarks/gpu_stage1_initialization.py \
  --protocol benchmarks/results/20260725_gpu_stage1_initialization_PREREG.md \
  --out runs/gpu_stage1_initialization
```

Defaults: `frame_00009`, downscale 4 (~938×410), 5,000 initial Gaussians, 7,000 steps, gsplat,
matched hard budget 15,000, held-out `C0004 / C0025 / C1004`.

The preregistered treatment is **`cover-iso`** — cover extent, **opacity unchanged**. That is a
change from the CPU screen's `surfel`, because the CPU data showed the derived opacity buys nothing
downstream (+0.18% AUC, −0.046 dB) and is what causes the silhouette leak (initial outside alpha
0.0437 without it, 0.1851 with it). `cover-iso` was the best fixed-topology arm there — but it was
*not* that run's preregistered treatment, so this is a fresh preregistration rather than a post-hoc
pick.

Then, separately (a different mechanism, not another seed):

```bash
PYTHONUNBUFFERED=1 python benchmarks/gpu_stage1_initialization.py \
  --protocol benchmarks/results/20260725_gpu_stage1_initialization_PREREG.md \
  --out runs/gpu_stage1_mcmc --modes mcmc
```

MCMC/relocation replaces clone/split outright. If the initialization advantage survives there too,
it is not an artefact of the clone-in-place failure.

## Step 2 — the residual decomposition at real resolution (minutes)

Step 1 runs this automatically per arm, but to inspect it directly:

```bash
python benchmarks/residual_decomposition.py \
  --run runs/gpu_stage1_initialization --downscale 4 \
  --rasterizer gsplat --device cuda \
  --models density/ci/gaussians_final.ply density/cover-iso/gaussians_final.ply
```

**This is the step that redesigns stage 2**, and it is why stage 2 is not written yet. On CPU at
downscale 32 the answer was already surprising: at convergence, interior holes hold **0.00–0.08%**
of the held-out error (robust across boundary widths and alpha thresholds), while the silhouette
band holds 47–77% and interior appearance 23–45%. Holes were 58.9% of the error *at
initialization* and optimization closed them completely.

Caveat that makes the GPU re-run mandatory: at downscale 32 the object spans ~40 px, so a 2 px
boundary band is 24% of all pixels. That share must shrink at downscale 4, and the balance between
boundary and appearance may invert.

## Stage 2 — designed from step 2's output, not before

Three branches, each needing a different tool. Picking one before measuring is exactly the mistake
this line has already made twice.

| if the dominant residual is… | then stage 2 is… | and the metric is… |
|---|---|---|
| **interior holes** | birth/split targeting: split-threshold policy, densification timing, relocation | interior hole fraction and hole-component size distribution |
| **silhouette boundary** | a silhouette-aware initialization rule — shrink or attenuate where the surface curves away from the camera; the cover condition assumes a locally planar patch and is provably wrong there | boundary-band L1 share, outside-mask alpha |
| **interior appearance** | not a geometry problem at all: SH scheduling, colour, fine optimization — and the initialization work is done | interior-covered L1 share at fixed geometry |

Whichever branch fires, freeze **SH at degree 0** during the geometry stage and raise it only
afterwards. The runs so far trained all SH bands throughout, which lets colour absorb geometric
error; once it has, geometry cannot recover it later.

## Still open, and blocked on data rather than effort

Generalization needs **multiple mask-bearing scenes**. Both other checked-in roots
(`karate/frame_00005`, `karate/frame_00060`) carry no packed alpha, so the masked coverage and
leakage protocol cannot run on them at all. If you have further calibrated captures with masks,
that is the single thing standing between this and a result that generalizes.

## Standing constraints

- No default changes. `surfel_init` is opt-in; Beam Fusion keeps CI.
- Three of five gates failed on the CPU screen (leak, attribution, and participation — the last
  one *inverted*). Those are recorded as failures and were not re-scored.
- Every harness writes `summary.json` with the protocol hash; keep the runs directory.
