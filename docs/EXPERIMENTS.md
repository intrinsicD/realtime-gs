# Experiment log

Dated, append-only log of research experiments — positive **and negative** results.
Template:

```markdown
## YYYY-MM-DD — short title
- **Question**: what hypothesis was tested?
- **Setup**: exact command(s)/config, git rev, seed(s), scene(s)
- **Result**: the numbers (paste the relevant benchmark/test output)
- **Conclusion**: what we now believe, and with how much confidence
- **Follow-ups**: next experiments this suggests (mirror into ROADMAP if substantial)
```

Rules: an experiment that changes a default hyperparameter must be linked from the code
comment at the changed default. Threshold changes in tests must cite an entry here.

---

## 2026-07-14 — Compact 2D starts, strict held-out metrics, and CUDA Janelle ablation
- **Question**: Is 640 a useful configurable *start* rather than an image-wise final cap; which
  lift gives the best object-centric initialization; and how much 3D density growth is useful?
- **Setup**: RTX 4090, PyTorch 2.12.0+cu132, gsplat 1.5.3, seed 0. Janelle
  `2025_03_07_stage_with_fabric/frame_00008` has 26 calibrated RGB/mask views; every eighth view
  gives 23 train and 3 strictly held-out views. Runs used 1/16 resolution (333×288), 300 stage-1
  iterations, 1000 refinement iterations, and foreground/crop held-out metrics. StructSplat
  compared fixed 320, fixed 640, and adaptive 640→2000 against native fixed 640. Downstream
  controls used `carve(grid_res=96)` and density disabled, unrestricted, or stopped at iteration
  300 with a 15k cap. Full machine-readable result:
  `benchmarks/results/20260714T085148Z_cuda_janelle.json`. The CPU synthetic regression benchmark
  is `benchmarks/results/20260714T090516Z_cpu.json`.
- **Result**: Mean stage-1 foreground PSNR / 23-view wall time was StructSplat fixed-320
  **27.35 dB / 13.45 s**, fixed-640 **28.60 / 14.67**, adaptive-2000 **29.41 / 15.01**, versus
  native fixed-640 **25.66 / 47.29**. Fixed-640→carve initialized only 3,015 3D splats at
  **21.98 dB** held-out foreground PSNR, versus native-640's 3,613 at **20.16 dB** and
  adaptive-2000's 5,473 at **21.32 dB**. With the short 15k-capped density schedule, fixed-640
  reached **25.67 dB foreground / 32.08 dB crop / 0.9604 crop SSIM**. Fixed-320 reached
  25.50/31.89/0.9598, retaining a small deficit from its weaker init (-0.53 dB). For the same
  adaptive-2000 init, no density / unrestricted growth / capped growth reached 25.60 at 5,473 /
  25.04 at 70,485 / **25.76 dB at 15,000** splats. Depth-seeded bounded-ray `hybrid` improved
  initialization over direct monocular `depth` from **12.68 to 20.23 dB**; under the same 15k
  cap it refined to 23.41 versus depth's 23.20 dB, but `carve` remained better and faster.
  On the synthetic CPU benchmark, hybrid initialized at 21.61 dB and refined to 31.44 dB versus
  direct depth's 19.08/31.33, confirming the integration without claiming the real-depth ranking.
- **Conclusion**: 640 is a sound default start for this scene, not a ceiling. More per-image 2D
  splats improve the isolated image fit but did not improve the 3D initialization; compact
  structured fits plus carving were better. Adaptive 2k recovered 0.09 dB more final quality,
  while fixed 640 gave the strongest initialization with fewer splats. A short hard-capped 3D
  growth phase beat both no growth and unrestricted growth; the latter overfit the training views
  and expanded to 70k–100k splats. StructSplat fixed-640 was about 3.2× faster than native
  fixed-640 and improved held-out initialization by 1.82 dB. These are one-frame, low-resolution
  findings, not a cross-dataset ranking.
- **Follow-ups**: Repeat at 1/8 and 1/4 resolution, add LPIPS-VGG and peak-VRAM/time-to-quality,
  test `quadtree_wse` and GaussianImage at matched time/count, and compare the current density
  controller with gsplat MCMC/relocation. Inspect the saved contact sheet/GIF before choosing a
  high-resolution run.

## 2026-07-13 — Geometry/device correctness pass and calibrated Janelle smoke test
- **Question**: Do projection-correct covariance, bounded ray depths, independent opacity,
  color-independent carving coverage, and corrected density/timing plumbing improve the
  initialization pipeline; and does it run on the supplied calibrated object captures?
- **Setup**: `python benchmarks/run.py --quick --update-docs`, CPU, seed 0, synthetic 12-view
  48×48 scene, 150 2D gaussians/view, 120 fit + 150 refine iterations. Real-data smoke used
  `2025_03_07_stage_with_fabric/frame_00008`, four evenly sampled views at 1/64 resolution,
  real PNG masks, 60 gaussians/view, 15 fit + 5 bounded-ray + 3 refine iterations. The capture
  inventory is 26 RGB views in each stage frame and 30/32 in the two karate frames; the stage
  frames also contain per-camera masks. Result: `benchmarks/results/20260713T123616Z_cpu.json`.
- **Result**: Synthetic init/final PSNR changed versus the 2026-07-08 tracked run: depth
  17.05/28.53 → **19.08/31.36** dB; gradient 19.41/25.38 → **22.43/30.86**; carve
  17.48/29.13 → **20.31/31.91**. Gradient lift time fell 15.43 → 7.93 s. The comparison now
  includes the shared 3.90 s all-view fit cost and time/PSNR samples. On Janelle, the bounded-ray
  init reached **23.75 dB**, short refinement reached 23.83 dB, and ray-stage loss fell
  0.0124 → 0.0073. A mock relative-inverse-depth backend exercised the no-SfM bounds alignment
  and produced 46 finite splats at 18.49 dB. Coverage threshold 0.4 reduced the synthetic
  carving median center-to-GT distance from 0.276 to 0.236 (0.3 threshold vs 0.4).
- **Conclusion**: The repaired transfer now improves all three initializers on the integration
  benchmark, and both proposed depth routes execute on the real calibration format. This is a
  regression/integration comparison, not an isolated causal ablation; GPU quality and real
  held-out full-resolution quality remain unmeasured. Actual StructSplat/GaussianImage fields can
  now skip native stage 1 through the adapter rather than being conflated with 3D opacity.
- **Follow-ups**: Create a CUDA-enabled environment for the RTX 4090 and run the full 26-view
  frame at 1/4 resolution; compare StructSplat versus native versus GaussianImage at matched
  2D PSNR and primitive count; run real Depth Anything V2 Small and a depth→bounded-ray hybrid;
  report held-out PSNR/SSIM/LPIPS and time-to-quality against SfM when sparse points are available.

## 2026-07-08 — Refined `gradient` variant: depth+rot+scale along the ray, then merge
- **Question**: The staged idea "fit 2D → lift with a thin third axis → optimize each
  gaussian along its ray for position/rotation/scale → full 3DGS". Does optimizing
  rotation+scale (not just depth) and merging redundant gaussians beat the old depth-only
  `gradient` lifter, and is a literal thin "epsilon" a good along-ray init?
- **Setup**: synthetic ring scene (12 views, 48×48, 40 GT gaussians), 150 2D gaussians/
  view, `gradient` lift 100–120 iters, refine 150 iters, `rasterizer="torch"`, seed 0,
  CPU. Compared against the old depth-only behavior and across `ray_thickness`. Rev after
  this commit.
- **Result** (lift-only): depth-only `n=1800, med-dist-to-GT=0.355, init-PSNR=17.95`;
  depth+rot+scale `0.446, 18.69`; +merge(0.01) `n=1790` (barely changed). End-to-end
  (150 refine iters): OLD depth-only `init 16.98 → final 25.30, n 1800→3564`; NEW
  depth+rot+scale +merge(0.03) `init 17.95 → final 25.85, n 1634→3408`. Coarser merge
  does reduce count (voxel-frac 0.01/0.03/0.06 → n 1792/1634/1152). Thin
  `ray_thickness=0.05` raised init-PSNR (19.85) but **worsened** geometry (med-dist 0.519).
- **Conclusion**: (1) Optimizing rotation+scale + merging gives a **small but real** gain
  (init +1.0 dB, final +0.55 dB) from fewer, better-shaped gaussians. (2) **Merge barely
  fires at a fine voxel because the geometry is scattered** (median 0.35–0.45 to GT vs
  0.19–0.21 for `carve`/`depth`) — problems ④ (redundancy) and ⑤ (under-constrained depth)
  are coupled: cross-view merging only helps *after* depth converges onto surfaces, which
  single-view-sampled photometric descent does not achieve in ~100 CPU iters. (3) A literal
  thin "epsilon" trades appearance for geometry (higher init PSNR, worse 3D placement) and
  is numerically riskier — footprint-scaled thickness is the right default; the knob is
  clamped to ≥0.05× the footprint. End-to-end the `gradient` variant is still the weakest
  (25.85 vs carve 29.13, depth 28.53) precisely because of the scattered geometry.
- **Follow-ups**: (a) GPU run with many more ray-opt iterations + gsplat — does geometry
  converge enough for merge to matter? (b) Hybrid: `depth`/`carve` init → short `gradient`
  polish (start from good geometry so the ray stage refines rather than searches).
  (c) Real-scene measurement with train/test split + matched wall-clock vs SfM-init 3DGS
  (the actual step-5 comparison; synthetic numbers here are relative-only).

## 2026-07-07 — Pipeline v1 sanity on synthetic scenes
- **Question**: Do all three lifting variants beat random initialization on synthetic
  scenes, and does refinement converge from each?
- **Setup**: `python benchmarks/run.py --quick` at the initial commit (rev `eb437bb`);
  synthetic ring scene (12 views, 48×48, 40 GT gaussians), 150 2D gaussians/view,
  150 refine iters, seed 0. Result file: `benchmarks/results/20260707T115928Z_cpu.json`.
- **Result** (init PSNR → final PSNR, dB): `gradient` 18.05 → 25.31, `carve`
  17.48 → **29.13**, `depth` (GT depth backend) 17.05 → 28.53, `sfm` baseline
  19.95 → 28.67, `random` baseline **8.08** → 27.93. Lift wall-clock on CPU:
  depth 0.02 s, carve 0.11 s, gradient 8.8 s (it renders during optimization).
- **Conclusion**: The pipeline machinery works end-to-end and every variant initializes
  9-10 dB above random. Surprises worth noting: (1) `gradient` has the best init but the
  *worst* final PSNR — it keeps all per-view gaussians on their rays (1800), densification
  then balloons the count (6546) and short refinement can't clean it up; it likely needs
  cross-view merging like `carve` has. (2) `carve` refines best despite a mid init.
  (3) The `sfm` baseline init PSNR is inflated here because synthetic "SfM points" are
  sampled directly from GT gaussians. Nothing about real scenes is concluded yet (GT
  depth flatters `depth`; real monocular depth adds scale error).
- **Follow-ups**: M2 GPU validation; add merge step to `gradient` (or hybrid B→A:
  depth init + gradient polish); revisit densification budgets for dense inits (M3).
