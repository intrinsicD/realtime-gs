# Why an under-sized primitive does not grow: a race the density schedule wins

Date: 2026-07-24

Kind: **post-hoc diagnostic** on the frozen initializations. No optimization, no selection, no
default. Machine-readable result: `runs/beam_surfel_scale_gradient_20260724/summary.json`.

## Question

If the control's primitives sit below the split boundary, why does plain gradient descent not
simply grow them until they cover the surface? Every earlier run in this line measured
*outcomes* — PSNR, alpha-IoU, counts, survival. None measured the gradient the optimizer actually
receives, which is the only instrument that can answer this.

## Measurement

For each frozen initialization, one forward and backward of the trainer's masked loss per training
view, capturing `dL/d log_scale`, `dL/d mu`, and `dL/d opacity_logit` per Gaussian. Also the
intrinsic (undilated) projected sigma in pixels, and per-Gaussian sign consistency across views
(`|mean| / RMS`), which is what decides whether Adam — which normalizes magnitude — can move a
parameter at all.

## Result

The reference rasterizer renders `Sigma_2D = J Sigma_3D J^T + 0.3 I` px², a **0.5477 px** low-pass
floor.

| | `ci` | `cover-iso` | `surfel` |
|---|---:|---:|---:|
| world sigma_max (median) | 0.00739 | 0.02387 | 0.02387 |
| intrinsic projected sigma | **0.2015 px** | 1.2736 px | 1.0326 px |
| below the 0.5477 px dilation floor | **98.0%** | 0.4% | 2.3% |
| primitive's share of rendered variance | **11.9%** | 84.4% | 78.0% |
| \|dL/d log s\| per view (median) | **4.86e-06** | 4.83e-05 | 1.93e-05 |
| \|dL/d mu\| per view (median) | 4.08e-04 | 8.34e-04 | 7.34e-04 |
| scale / position gradient ratio | **0.0126** | 0.0541 | 0.0268 |
| scale-gradient sign consistency across views | **0.8592** | 0.7916 | 0.4154 |
| fraction whose scale gradient points to **growth** | **99.0%** | 93.4% | **52.4%** |
| signed mean dL/d log s (median) | −4.69e-06 | −3.72e-05 | **−4.06e-07** |

## Where the gradient goes

**Into the low-pass filter.** The control projects to 0.2015 px against a 0.5477 px dilation floor,
so its rendered footprint is `sqrt(0.2015² + 0.3) = 0.5836 px` of which the primitive contributes
**11.9%** of the variance. Changing its own scale barely changes what the loss can see: the
relative sensitivity `sigma_own² / (sigma_own² + 0.3)` is 0.119 versus 0.78–0.84 for the cover
arms. The scale gradient is 10× smaller in absolute terms than a correctly sized primitive's, and
4.3× smaller *relative to its own position gradient* — the asymmetry the mechanism predicts,
because a dilated blob still translates at full strength when its mean moves.

## But the gradient is not the blocker — the schedule is

This is where the obvious explanation fails, and it is worth stating plainly because it was the
hypothesis this diagnostic was written to test. **The control's scale gradient is small but
excellent**: 99.0% of its primitives have a gradient pointing toward growth, at sign consistency
0.8592 across views. Adam normalizes magnitude, so a small, coherent gradient still produces a
near-full learning-rate step. The control's primitives are not stuck.

And indeed they are not: the identity-tracked run measured the surviving originals growing from
0.00739 to **0.01660** world sigma over 1,000 steps — a **2.25×** increase, `Δ log s = 0.809`, or
`8.1e-4` per step against the `5e-3` learning rate. They grow at roughly 16% of the maximal
log-space rate, which is about 860 steps to double.

The problem is **when**. Density control runs from step 20 to step 500 and makes every topology
decision in that window, reading the scale as it is *at the time*. At `8.1e-4` per step the
control's primitives have gained only `Δ log s ≈ 0.4` — about 1.5×, or sigma ≈ 0.011 — by the time
densification stops, still well under the 0.02229 split boundary. The identity-tracked
split-eligibility confirms exactly this: **2.8%** of originals eligible at the first density event,
still only **24.0%** at the last.

So the causal chain is:

1. the initialization is ~3.1× too small, and 98% of it falls below the renderer's dilation floor;
2. the low-pass filter absorbs ~88% of any scale change, shrinking the scale gradient by 10×;
3. Adam still climbs, but at ~860 steps per doubling;
4. the density controller commits the entire primitive budget between steps 20 and 500, while the
   primitives are still below the split threshold, so every event is a **clone in place**;
5. by the time the originals have grown 2.25× the topology is fixed, and the 1,633 clones — which
   *are* born at usable size and grow freely — carry the image.

Growth by gradient descent and topology decision by density control are on the same clock, and
the initialization loses the race. That is a different failure from "the gradient vanishes", and
it has a different set of fixes: correct the scale up front, delay densification until scales have
converged, or make the split threshold scale-relative rather than absolute.

## Independent evidence that the cover rule lands in the right place

The derived cover scale is not merely "bigger". It sits at the **zero crossing of the scale
gradient**: `surfel`'s signed mean `dL/d log s` is −4.06e-07, effectively zero, with only 52.4% of
primitives pointing to growth and sign consistency down at 0.4154 — the signature of a parameter
already at its optimum, where views disagree because there is nothing left to agree on. The control
sits far from that fixed point (−4.69e-06, 99.0% pointing to growth, consistency 0.8592). The
isotropic `cover-iso-op` variant overshoots slightly in the other direction (35.5% pointing to
growth, i.e. mostly shrink), consistent with an isotropic primitive over-covering where a flattened
surfel would not.

This was not designed as a check on the cover condition, and it is a stronger one than the
preregistered gates: the hexagonal-ripple derivation and the optimizer's own fixed point agree to
within a gradient that is an order of magnitude below the control's.

## Limits

One scene, one seed, the CPU reference rasterizer at downscale 32, gradients evaluated at step 0
only. `EWA_DILATION = 0.3` px² is this repository's convention; CUDA gsplat applies its own
antialiasing filter and the suppression factor there must be measured, not assumed. The growth-rate
figure is inferred from the endpoint sigma of the identity-tracked run rather than from a logged
per-step trajectory, so it is an average rate, not a curve. Sign consistency is measured across
views at a single point in parameter space, not across training steps.

## Command

```bash
.venv/bin/python benchmarks/beam_surfel_scale_gradient.py \
  --out runs/beam_surfel_scale_gradient_20260724
```
