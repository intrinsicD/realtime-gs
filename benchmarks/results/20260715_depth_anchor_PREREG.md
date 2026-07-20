# Confidence-weighted bounded-ray anchor preregistration

Written before the implementation and official runs on 2026-07-15.

## Question and paper adaptation

Does a confidence-weighted, unjittered normalized-ray anchor let `HybridLifter` correct unreliable
depth priors without pulling reliable rays away from their depth seed? This adapts the confidence
gating/local geometric anchoring ideas in DP-GS and NoDrift3R to the repository's calibrated,
bounded-ray optimizer. It does not reproduce either paper's learned confidence or ray-map model.

## Paired arms

All arms reuse identical fitted 2D Gaussians, depth maps, camera splits, optimizer draws, colors,
opacity, scales, and refinement budgets.

1. `legacy`: uniform squared error in raw-logit space to the jittered initialization (current code).
2. `normalized`: uniform Smooth L1 in bounded-ray fraction space to the unjittered prior/fallback.
3. `confidence`: the normalized loss weighted by `valid_prior * sampled_confidence`.
4. `thresholded`: the normalized loss gated at confidence >= 0.5.

Smooth-L1 beta is 0.05. Primary lifting disables merging so every output retains source-observation
correspondence. A secondary fixed-budget refinement disables density control.

## Data and perturbation

- Seeds 0, 1, 2; CPU reference rasterizer; 40-Gaussian synthetic scenes.
- Twelve 48x48 cameras with train views `[0,1,2,4,5,6,8,9,10]` and held-out views `[3,7,11]`.
- 150 fitted 2D Gaussians per training view, 120 fit iterations, SH degree 0.
- Clean condition: metric GT depth, confidence 1 on valid depth.
- Corrupted condition: deterministic 8x8 low-confidence blocks covering about one third of valid
  pixels; depth is multiplied by 1.20 or 0.80 according to view parity and confidence is 0.05 in
  those blocks. Other valid pixels retain confidence 1. Invalid background remains invalid.
- Sixty bounded-ray iterations, followed by sixty equal no-density refinement iterations.

## Metrics and decision rule

Report held-out initialization/final PSNR and SSIM; source-ray absolute relative depth error at
low- and high-confidence observations (median and p90); nearest-GT-center median/p90 distance;
wall-clock; and primitive count.

The primary hypothesis is supported only if `confidence` versus `legacy`, on corrupted priors:

1. improves mean held-out initialization PSNR by at least 0.25 dB and wins at least two seeds;
2. reduces mean low-confidence source-depth p90 error by at least 15%;
3. retains at least 0.10 dB mean held-out PSNR after refinement; and
4. regresses clean held-out initialization by no more than 0.10 dB.

`normalized` isolates coordinate/unjittering effects. `thresholded` tests whether hard rejection is
preferable to continuous confidence. A default may change only if the primary rule passes and the
continuous arm is not worse than thresholding by more than 0.10 dB or 5% low-confidence p90 error.
Otherwise all useful modes remain opt-in and the negative/null result is logged.

## Pre-run amendment after implementation audit

This amendment was made before any official ablation run. Raw-logit L2 and normalized Smooth L1
have different curvature. Every normalized arm therefore uses the same automatic multiplier
`2 * beta / median(u_anchor * (1-u_anchor))^2`, computed over valid training anchors, so its local
raw-parameter curvature matches legacy L2 at a representative anchor. The resolved multiplier is
recorded per run.

The official matrix adds a confidence-shuffled corrupted condition, preserving each view's
confidence values while breaking their spatial correspondence. A confidence-specific conclusion
requires `confidence` to beat `normalized` on calibrated corruption and for at least half of that
gain to disappear after shuffling. Rotation and scale optimization are disabled in the causal lift
run; merging remains disabled. Only the corrupted condition receives short no-density refinement.
Because the synthetic confidence is a controlled causal instrument rather than a deployable
estimator, this experiment cannot by itself change the production default; it can justify retaining
the mode and proceeding to a train-only multi-view confidence experiment.
