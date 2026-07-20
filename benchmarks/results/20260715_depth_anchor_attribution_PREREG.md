# Exact sampled-confidence attribution preregistration

Written on 2026-07-15 before implementing the two new control modes or running any attribution
benchmark.

## Question and literature boundary

Does the spatial correspondence of confidence to known corrupted depth—not merely valid-prior
gating—materially improve bounded-ray geometry? DP-GS motivates confidence derived from geometric
and photometric consistency. The 2026-07-15 Scholar Inbox digest adds ExtraGS/FlowPainter as
examples of reliability-gated local guidance and MAC-Splat as evidence that 2D photometric losses
alone leave sparse-view depth/correspondence ambiguous. These papers motivate isolating the
mechanism; none provides the repository's thresholds or establishes that synthetic oracle
confidence should help this optimizer.

## Frozen arms

All arms use the same unjittered bounded-ray-fraction Smooth-L1 target, beta 0.05, automatic
median-valid-anchor stiffness, depth-prior lambda 0.01, initialization jitter, fitted 2D Gaussians,
and main optimizer random stream.

1. `valid_uniform`: anchor weight 1 for every retained ray with a valid sampled prior and 0 for an
   invalid-prior fallback.
2. `confidence`: anchor weight is the already bilinearly sampled and sanitized confidence on valid
   retained rays.
3. `confidence_shuffled`: copy the `confidence` weights, then permute them only among valid retained
   rays within each source view using a dedicated generator seeded by `seed + 1_000_003`.

The shuffled mode must not consume the generator used for depth jitter or target-view selection,
and it must not alter merge/observation confidence. The permutation includes valid zero-confidence
weights.

## Frozen data and execution

- Seeds 0, 1, 2; CPU reference rasterizer; four PyTorch threads.
- Forty-Gaussian synthetic scenes; twelve 48x48 cameras.
- Training views `[0,1,2,4,5,6,8,9,10]`; strictly held-out views `[3,7,11]`.
- 150 fitted 2D Gaussians per training view for 120 fit iterations.
- Deterministic 8x8 corruption blocks over valid training depth, selected by
  `(x // 8 + y // 8 + view_index) % 3 == 0`.
- Corrupted depth is multiplied by 1.20 in even source views and 0.80 in odd source views;
  corrupted pixels have confidence 0.05 and other valid pixels confidence 1.
- Sixty bounded-ray iterations; rotation optimization, scale optimization, merging, density
  control, and downstream refinement are disabled.
- One official three-seed run. No seed replacement, rerun, lambda/threshold/corruption change, or
  post-hoc refinement is permitted for the decision.

## Pre-metric validity assertions

The official artifact is invalid rather than replaceable if any assertion fails:

1. every arm has bit-identical step-0 means, quaternions, scales, opacity, and SH;
2. retained primitive layout, source-view boundaries, valid-prior masks, and normalized stiffness
   match across arms;
3. within every source view, sorted valid `confidence_shuffled` weights are bit-identical to sorted
   `confidence` weights, invalid weights stay zero, and sums/counts match;
4. the shuffle changes at least one sampled weight location in every view containing more than one
   distinct valid weight; and
5. setting depth-prior lambda to zero produces bit-identical multi-step outputs and loss histories
   across modes, proving the shuffle generator does not disturb the optimizer schedule.

## Primary metrics and decision rule

Ground truth is used only to construct the controlled training corruption and to report geometry.
Primary metrics are:

- strict held-out expected-depth RMSE divided by scene extent; and
- source-ray absolute relative depth-error p90 on the fixed corruption mask (not a
  post-shuffle low-confidence group).

Confidence weighting passes the material-effect test only if, versus `valid_uniform`, it:

1. reduces mean held-out depth RMSE by at least 2%;
2. reduces mean corrupted-source p90 by at least 15%;
3. wins at least two of three paired seeds on each metric; and
4. has mean held-out initialization PSNR no more than 0.10 dB worse.

The effect is attributable to correct confidence locations only if all material-effect conditions
pass, `confidence` also beats `confidence_shuffled` in at least two of three seeds on both geometry
metrics, and shuffling erases at least half of the `confidence`-versus-`valid_uniform` gain on each
metric. For lower-is-better metric `m`, define `gain(mode) = (m(valid_uniform) - m(mode)) /
m(valid_uniform)`; the erasure condition is `gain(confidence_shuffled) <= 0.5 * gain(confidence)`.

Held-out PSNR/SSIM, alpha IoU/coverage, nearest-GT-center median/p90, loss checkpoints, primitive
count, resolved stiffness, and lift time are secondary diagnostics and cannot rescue a failed
primary rule.

## Stopping rule

If any material-effect or attribution condition fails, log a null/negative result and stop
confidence-anchor loss/threshold/lambda sweeps. The next mechanism question becomes
leave-one-source-view-out or direct train-view geometric/correspondence consistency. If every
condition passes, proceed only to train-derived confidence with actual monocular depth on
calibrated views. This synthetic experiment cannot change the production `legacy` default.
