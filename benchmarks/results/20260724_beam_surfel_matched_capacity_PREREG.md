# Matched-capacity falsification of the surfel-initialization count result — preregistration

Date: 2026-07-24 (frozen before any matched-capacity outcome was produced)

## Why this exists

The 20260724 screen reported that cover-consistent arms reach **higher held-out quality with
under half the primitives**: 22.3279 dB at 2,230 versus the control's 21.8745 dB at 5,030. That
comparison used **unmatched budgets**. Both arms ran under the same 8,000 cap and simply stopped
growing at different points, so the result is equally consistent with a much weaker reading:

> The control was merely allowed to overshoot. Density control is greedy, the control's
> under-sized primitives qualify more often (0.6125 vs 0.2550), and the extra ~2,800 primitives
> are overfitting the training views (control train-view PSNR 28.6049 vs `cover-iso`'s 27.2435).
> At a **common** budget the control might reach the same held-out quality.

This experiment is designed to **falsify the capacity claim**, not to support it.

## Question

At a matched primitive budget, does a cover-consistent initialization still beat the unchanged
covariance-intersection initialization on held-out cameras?

## Scope and its limit, stated up front

This runs on `frame_00009`, the **same root** as the screen it is testing. It is therefore a
mechanism confirmation, **not** a generalization result: it can withdraw the capacity claim, or
leave it standing on one scene, but it cannot extend it to other scenes. The two remaining
checked-in roots (`karate/frame_00005`, `karate/frame_00060`) carry no packed alpha, so the
mask-based coverage/leakage protocol cannot run on them without a different supervision regime;
a multi-scene test needs mask-bearing captures and stays queued.

## Frozen protocol

Identical to `20260724_beam_surfel_init_PREREG.md` — same root, same train views
`[0,3,6,9,12,15,18,21]`, same held-out `[1,13,25]` = `C0004, C0025, C1004`, same downscale-32
compact teachers, same Beam Fusion configuration, same 1,000-step Torch CPU refinement, same
loss — with exactly two changes:

1. **Matched hard budget.** `DensityConfig.max_gaussians = 2400 = 3 x N_init`. This is derived
   from the initialization size, which is frozen at 800 for every arm, and not from any observed
   final count. Every other density parameter is unchanged (start 20, stop 500, every 4,
   threshold 3e-3, prune 0.005/0.1, reset every 100 to 0.011).
2. **Three seeds**, 0, 1, and 2, varying the trainer/density RNG. Beam Fusion is deterministic
   and its output is computed once and shared by every arm and seed, so all arms remain
   bit-identical in means, SH/color, and count.

## Arms

| arm | covariance | opacity | status |
|---|---|---|---|
| `ci` | unchanged Beam CI | fixed 0.10 | control |
| `surfel` | oriented flat surfel | coverage rule | the preregistered treatment of the prior screen |
| `cover-iso-op` | isotropic at the cover sigma | coverage rule | **selected from the prior screen's outcome**; reported as a labeled secondary arm and cannot be treated as confirmed by the run that selected it |

## Preregistered decisions

Primary comparison is `surfel` versus `ci`, held-out pool, foreground PSNR at step 1,000.

- **M1 (capacity advantage holds)** — `surfel` >= `ci` + 0.15 dB in at least 2 of 3 seeds
  **and** `surfel` final N <= `ci` final N in those seeds. A treatment may not win by spending
  more primitives.
- **M2 (falsification)** — if `|surfel - ci| < 0.15 dB` in at least 2 of 3 seeds, the prior
  screen's "better held-out quality with under half the primitives" wording is **withdrawn** and
  rewritten to whatever the matched budget shows: either "equal quality at fewer primitives"
  (still useful, weaker) or "no capacity advantage" (the claim dies).
- **M3 (regression guard)** — if `ci` >= `surfel` + 0.15 dB in at least 2 of 3 seeds, the
  cover-consistent initialization is worse under a matched budget and the whole direction is
  reported as negative.
- **Guardrail** — final held-out alpha-outside <= 0.05 for every arm and seed, as before.

M1, M2, and M3 are mutually exclusive and exhaustive over the 2-of-3 majority; a 1/1/1 split
across seeds is recorded as **inconclusive** and closes the question on this root.

Secondary, reported but not gated: `cover-iso-op` versus `ci` on the same terms, alpha-IoU, the
capped-versus-uncapped control (`ci` at 2,400 here versus `ci` at 5,030 in the prior screen, which
measures what the cap alone costs), and per-seed final counts.

**No default change is authorized by this run under any outcome.** A default change requires
multiple mask-bearing scenes and the production CUDA gsplat strategies.

## Official command

```bash
PYTHONUNBUFFERED=1 .venv/bin/python benchmarks/beam_surfel_matched_capacity.py \
  --protocol benchmarks/results/20260724_beam_surfel_matched_capacity_PREREG.md \
  --out runs/beam_surfel_matched_capacity_20260724
```
