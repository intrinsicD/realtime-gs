# Is the control's deficit initialization scale, or a density controller mismatched to it? — preregistration

Date: 2026-07-24 (frozen before any outcome of this experiment)

## Why this exists

The scale-gradient diagnostic showed the control's primitives **do** grow — 2.25× over 1,000 steps
under a coherent gradient (99.0% growth-signed, sign consistency 0.8592) — but at ~860 steps per
doubling, while density control commits the entire primitive budget between steps 20 and 500 and
reads the scale as it is *at that moment*. Split-eligibility was 2.8% at the first density event
and still 24.0% at the last, so every event was a clone-in-place.

That makes the previous conclusion **confounded**. "The initialization's scale is wrong" and "the
density controller's schedule and absolute split threshold are mismatched to this initialization"
predict the same outcome, and the whole session has only tested the first. If repairing the
controller alone closes most of the gap, the initialization claim must be narrowed accordingly.

This experiment is designed to **narrow my own claim**, not to defend it.

## Question

With the initialization held bit-identical, can changing only the density controller — when it
runs, and what counts as large enough to split — recover the cover-consistent arm's held-out
quality?

## Arms

Five arms. Four share the **unchanged `ci` initialization** (bit-identical means, quats,
log-scales, opacity, SH); only `DensityConfig` differs. The fifth is the unchanged treatment.

| arm | initialization | density schedule | split boundary |
|---|---|---|---|
| `ci-baseline` | `ci` | start 20, stop 500 | `0.01 * extent` = 0.02229 (default) |
| `ci-late` | `ci` | **start 300, stop 780** | 0.02229 (default) |
| `ci-relsplit` | `ci` | start 20, stop 500 | **arm's own median initial sigma_max** |
| `ci-late-relsplit` | `ci` | **start 300, stop 780** | **arm's own median initial sigma_max** |
| `surfel-baseline` | `surfel` | start 20, stop 500 | 0.02229 (default) |

`start 300, stop 780` is chosen to give **exactly the same number of scheduled density events**
as the default window: `(500-20)/4 + 1 = 121` and `(780-300)/4 + 1 = 121`. The delay is therefore
not a change in how much densification is allowed, only in when it happens — by step 300 the
control's primitives have grown roughly 1.3×, and by 780 roughly 1.9×, on the measured rate.

The relative split boundary is computed **programmatically from each arm's own initialization**
(`split_scale_frac = median(sigma_max) / extent`) rather than hand-picked, so it is an
"adapt the threshold to the initializer" policy rather than a tuned constant. Reported for the
record: applying the same policy to `surfel` would give `0.01071`, within 7% of the existing
`0.01` default — i.e. the default threshold is already appropriate for a correctly scaled
initializer and mismatched only for an under-scaled one.

Everything else is frozen and identical: `frame_00009`, train views `[0,3,6,9,12,15,18,21]`,
held-out `[1,13,25]`, matched hard budget 2,400, 1,000 steps, `every=4`, gradient threshold 3e-3,
prune 0.005/0.1, opacity reset every 100 to 0.011, same loss, seeds 0/1/2.

## Preregistered decisions

Primary comparison is the **best repaired control** (the best of `ci-late`, `ci-relsplit`,
`ci-late-relsplit` by mean held-out foreground PSNR across seeds) against `surfel-baseline`,
per seed, at a 0.15 dB margin. Majority of 3 seeds decides.

- **S1 — controller explains it.** The best repaired control comes within 0.15 dB of
  `surfel-baseline` in a majority of seeds. The initialization-scale claim is **narrowed**: the
  measured advantage is substantially attributable to a controller mismatched to the
  initialization, and correcting the covariance is one of at least two sufficient repairs.
- **S2 — initialization explains it.** The best repaired control remains ≥ 0.15 dB below
  `surfel-baseline` in a majority of seeds. Controller repair is not sufficient and the
  initialization claim stands as stated.
- **S3 — both contribute.** The best repaired control beats `ci-baseline` by ≥ 0.15 dB **and**
  still trails `surfel-baseline` by ≥ 0.15 dB in a majority of seeds. Report the split explicitly
  as a decomposition, and narrow the initialization claim to the residual.

S1, S2, and S3 are evaluated in that order; a case matching none is recorded as
**inconclusive** and closes the question on this root.

Secondary, reported but not gated: each repaired arm individually versus `ci-baseline`; final
counts; the fraction of surviving originals above the split boundary at the first and last density
event; and the logged growth trajectory below.

## Caveats this run also closes

The scale-gradient diagnostic evaluated gradients at step 0 only and inferred the growth rate from
an endpoint. This harness additionally logs, at every checkpoint and for every cell:

- median `sigma_max` of the **identity-tracked surviving originals** — the growth trajectory as a
  curve rather than an average rate;
- the fraction of surviving originals above that arm's split boundary;
- a full gradient-telemetry pass (`|dL/d log s|`, `|dL/d mu|`, sign consistency across views,
  fraction growth-signed) recomputed at that point in parameter space, not just at step 0.

These are descriptive instrumentation, not gated quantities.

## Scope

Same root as everything else in this line, so this can narrow or preserve a mechanism claim and
cannot generalize it. CPU reference rasterizer, downscale 32, classic controller. **No default
change is authorized under any outcome.**

## Official command

```bash
PYTHONUNBUFFERED=1 .venv/bin/python benchmarks/beam_surfel_schedule_confound.py \
  --protocol benchmarks/results/20260724_beam_surfel_schedule_confound_PREREG.md \
  --out runs/beam_surfel_schedule_confound_20260724
```
