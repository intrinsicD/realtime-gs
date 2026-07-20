# Confidence-anchor post-run protocol audit

This audit was written after the official artifact
`20260714T224800Z_cpu_depth_anchor.json`; it does not change the preregistration or the observed
numbers.

## Verdict

The primary hypothesis remains unsupported. On corrupted priors, confidence weighting trailed the
legacy anchor by 0.0577 dB at initialization and 0.0417 dB after refinement, won one of three
initialization seeds, and worsened low-confidence source-depth p90 error by 1.63%. None of those
failures depends on the shuffled condition.

The artifact field `confidence_location_causal=true` is only the result of the preregistered
directional calculation. It is not an attribution-clean causal conclusion and must not be cited as
one.

## Attribution problems found after the run

1. `normalized` averages its loss over both valid-prior anchors and unjittered fallback anchors,
   while `confidence` assigns zero anchor weight to invalid priors. Only 829/1303, 821/1293, and
   795/1262 retained observations had valid clean source-depth diagnostics across seeds. Thus this
   comparison changes valid/fallback gating as well as confidence magnitude.
2. The shuffled condition permutes confidence pixels before confidence is bilinearly sampled at
   fitted-Gaussian centers. Interpolation therefore does not preserve the sampled weight multiset.
   The retained low-confidence group changed from 261/236/233 observations in calibrated corruption
   to 104/131/116 after shuffling (730 to 351 total). The shuffled arm is not a pure location
   permutation at the optimized rays.

## Invariants that did pass audit

- Every condition and seed reports exact step-0 equality across all four arms.
- Primitive counts match across arms before and after refinement.
- The resolved normalized-loss stiffness is identical across arms within each seed/condition;
  corrupted and shuffled conditions also match exactly.
- Fitting and lifting consume only nine training cameras; views 3/7/11 are used only for held-out
  metrics. Refinement uses the scene's training split and preserves primitive ordering/count because
  density control is disabled.
- The three SHA-256 values embedded in the artifact match the exact benchmark, gradient-lifter, and
  hybrid-lifter sources that produced it.

## Narrow repair experiment

Keep the existing lambda, seeds, corruption, and no-merge lift fixed. Add a `valid-uniform` arm that
uses the same valid-prior mask as `confidence` but weight 1 for every valid retained ray. Build the
negative control only after sampling: permute the sampled confidence weights among retained valid
rays within each source view, preserving the exact multiset and denominator. Preregister a material
effect floor and paired-seed consistency for held-out depth RMSE and corrupted-source p90. Skip
refinement and learned-confidence work unless this isolated contrast produces a meaningful,
replicated gain.

## Provenance limitation

The official run records revision `2dddca4`, a dirty-worktree flag, the full command/configuration,
and hashes for the three directly changed experiment files. It does not hash every executed
dependency (for example fitting, synthetic-scene, trainer, rasterizer, and shared lift utilities),
so the JSON is audit-rich but not an independently replay-complete source snapshot. Preserve the
working-tree changes together or commit them before treating the command as archival replay.
