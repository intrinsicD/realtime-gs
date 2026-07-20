# Exact sampled-confidence attribution

Official CPU artifact:
`benchmarks/results/20260715T052539Z_cpu_depth_anchor_attribution.json`.
Preregistered protocol:
`benchmarks/results/20260715_depth_anchor_attribution_PREREG.md`.
Post-run audit:
`benchmarks/results/20260715_depth_anchor_attribution_RESULT.md`.

## Protocol and invariants

Seeds 0/1/2 used 40-Gaussian synthetic scenes, twelve 48x48 cameras, nine training views,
held-out views 3/7/11, shared 150-Gaussian/view fits, and 60 bounded-ray iterations. The
`valid_uniform`, `confidence`, and `confidence_shuffled` arms shared the same unjittered normalized
Smooth-L1 anchor, lambda 0.01, beta 0.05, jitter 0.02, automatic median-valid stiffness, and
deterministic 8x8 block corruption. Refinement, merge, rotation, scale optimization, and density
control were disabled.

All seeds passed bitwise step-0 equality and two-iteration lambda-zero main-RNG equality. All 27
source-view shuffle checks preserved the exact valid sampled-weight multiset, canonical sum and
squared sum, kept invalid weights zero, and changed every nonconstant layout. Primitive counts were
1303/1293/1262 and were identical across arms within seed.

## Primary result

| metric | valid uniform | confidence | exact shuffle | confidence vs uniform |
| --- | ---: | ---: | ---: | ---: |
| held-out depth RMSE | 0.151705 | 0.149962 | 0.151792 | 1.149% lower |
| corrupted-source p90 | 0.204933 | 0.206519 | 0.205348 | 0.774% worse |
| held-out PSNR | 19.6318 dB | 19.6310 dB | 19.6328 dB | -0.0008 dB |
| held-out SSIM | 0.6108 | 0.6149 | 0.6090 | +0.0041 |

Confidence won the RMSE comparison in 3/3 seeds but missed the 2% effect floor. It won the
corrupted-p90 comparison in 1/3 seeds and missed the required 15% reduction. The PSNR safety guard
passed. Confidence beat exact shuffle in 3/3 RMSE and 2/3 p90 comparisons, but shuffle erased half
the RMSE gain only; both material-effect and location-attribution decisions are false.

Auxiliary all-source p90 improved 8.26% and nearest-GT median improved 2.06%, while nearest-GT p90
worsened 0.26%. These cannot override the frozen primary tail criterion. The clean-GT source
diagnostic includes 16/29/34 more rays than the valid anchor mask across seeds, which may dilute
sensitivity but is arm-invariant.

## Decision

Keep `legacy` as default and stop confidence-anchor loss/lambda/threshold/weighting sweeps on this
setup. The result allows a small location-sensitive expected-depth signal, not a material robust
utility claim. Pivot to leave-one-source-view-out supervision and, if needed, direct train-view
geometry/correspondence consistency.

