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
