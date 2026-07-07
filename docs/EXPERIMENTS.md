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
