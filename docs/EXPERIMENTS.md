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
- **Setup**: `python benchmarks/run.py --quick` at the initial commit; synthetic ring
  scene (12 views, 48×48, 40 GT gaussians), seed 0.
- **Result**: see `benchmarks/results/` at this revision and the table in
  docs/BENCHMARKS.md. All variants initialize well above the `random` baseline;
  `depth` (with ground-truth depth backend) gives the best init PSNR, as expected;
  `gradient` and `carve` land between `random` and `depth`; refinement improves every
  variant.
- **Conclusion**: The pipeline machinery works end-to-end. Nothing about relative variant
  quality on *real* scenes can be concluded yet (GT depth flatters variant B; real
  monocular depth has scale error the alignment step must absorb).
- **Follow-ups**: M2 GPU validation; hybrid B→A (depth init + gradient polish).
