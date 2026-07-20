# I1 preregistration: correspondence-confidence gate

## Chronology and permitted data

This freezes I1 after the audited E1 result and before any easy-only
initialization quality or downstream E2 outcome is evaluated. Thresholds use
only E1 cluster-confidence signal distributions on the seven training compact
views. No held-out camera, easy-only render metric, optimizer result, or
density-control outcome has been opened.

Input and placement remain exactly E1:

- strict bundle:
  `runs/compact_masked_bundle_640_20260717/reconstruction_inputs`;
- bundle manifest SHA-256:
  `6ed60cf3df1f9ca476dfabace18b8d868d63eebd041522313c5b4fd644ee2614`;
- seed `0`, 48 depth samples, `min_views=2`, merge voxel `0.06`;
- dense placement PLY SHA-256:
  `56ce5f1ac3a321f6912506dc4e2c8484c1c3b9d5930eb140b84253faf106cff7`.

## Frozen classifier

A merged cluster is **easy** if and only if all conditions hold:

1. distinct source-view multiplicity is at least `2`;
2. RMS Euclidean distance of pre-merge member means from the merged mean is at
   most `0.50 × merge_voxel_size`;
3. the maximum member `candidate_half_max_widths` is at most `0.20` world-depth
   units;
4. the minimum member `candidate_best_n_covered` is at least `2`;
5. the maximum pixel reprojection residual, obtained by projecting the merged
   mean into each member's source camera and comparing with that member's
   `lineage.source_xy`, is at most `16.0 px`.

Score-margin minimum, within-cluster consensus-color variance, member count,
RMS reprojection residual, and maximum spread remain diagnostics only. They do
not change the frozen keep decision.

The thresholds are round values selected from the pre-quality E1 signal
diagnostic. Among the 500 multiplicity-≥2 clusters, cross-view quantiles were:

| Signal | p50 | p90 | max |
|---|---:|---:|---:|
| RMS spread / voxel | 0.2825 | 0.4882 | 0.7916 |
| maximum half-max width | 0.1822 | 0.3339 | 0.7522 |
| maximum reprojection residual (px) | 37.38 | 79.08 | 144.73 |

The frozen conjunction is expected to retain 60 clusters on this bundle.
That expected count is a diagnostics reproduction target, not a quality gate
and not permission to alter thresholds after seeing easy-only quality.

## Implementation gates

- CPU-first and deterministic; no CUDA import at module import time.
- Validate candidate-audit selected rows against returned dense lineage.
- Emit one typed record per merged cluster, keep mask, failure reasons,
  kept/dropped totals, and per-signal quantiles.
- Unit fixture: a three-view co-located target passes; a single-view decoy
  fails by construction.
- `benchmarks/compact_init_eval.py --gate` saves `init_easy_gated.ply`, embeds
  gate diagnostics in `init_eval.json`, and reproduces the 2,319-cluster E1
  histogram plus the expected 60 kept clusters.
- I1 does not change the top-K default and does not itself authorize E2.

## Post-implementation screen

After the code and count reproduction pass, easy-only init metrics may be
reported as exploratory diagnostics. They cannot tune this classifier.
E2 must use this frozen classifier and the separately frozen matched
optimizer/density-control protocol.
