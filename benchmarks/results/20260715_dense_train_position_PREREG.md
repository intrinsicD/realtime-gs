# Preregistration: dense train-only patch/epipolar position consistency

## Question and literature boundary

The preceding fixed-GT-identity experiment established that the frozen position loss strongly
localizes the small set of primitives it touches, but its graph covered only 7.73%-9.43% of
retained nodes and missed every whole-scene materiality threshold. This one allowed follow-up asks:
does a substantially broader graph built only from training RGB and calibration propagate that
local mechanism into material whole-scene geometry?

Scholar Inbox on 2026-07-15 highlighted MAC-Splat's reciprocal semantic matching and robust
world-frame consistency, while the repository survey identifies EDGS's dense correspondence,
triangulation, and filtering sequence. RoMa provides a plausible optional learned backend. This
experiment implements none of those systems: it is a deterministic raw-RGB patch/epipolar CPU
reference backend behind a pluggable matcher boundary. It adds no learned descriptor, confidence-
weighted sampling, shape/appearance term, correspondence-created initializer, or densification
change. A positive result is evidence only for this synthetic repository adaptation, not SOTA,
real-scene deployability, or reproduction of
[MAC-Splat](https://arxiv.org/abs/2607.10792),
[EDGS](https://arxiv.org/abs/2504.13204), or
[RoMa](https://arxiv.org/abs/2305.15404).

## Frozen data and arms

- CPU only; Torch reference rasterizer; four Torch threads; deterministic seeds `0,1,2`.
- Per seed: 40-Gaussian synthetic scene, 12 cameras at 48x48, held-out global views
  `[3,7,11]`, and the remaining nine views physically subset before fitting/lifting/matching.
- One shared 150-Gaussian/view, 120-step stage-1 fit per seed. The retained layout is exactly the
  `min_weight=0.05` and ray/AABB-filtered layout exposed by the zero-step Gradient lifter.
- Two families: pure Gradient and deterministic corrupted-depth Hybrid, with the exact corruption,
  initial-depth, and diagnostic construction used by the preceding position experiment.
- Three paired arms in each family:
  1. `none`: inclusive photometric bounded-ray optimization, no position term;
  2. `dense_train_position`: the fixed train-only matcher graph;
  3. `degree_shuffled_position`: the exact-degree/per-camera-pair cyclic endpoint derangement.
- All arms retain 90 lift steps, `lr=0.1`, inclusive full-set rendering, legacy anchors, Gradient
  jitter/lambda `0.15/0.001`, Hybrid jitter/lambda `0.02/0.01`, no rotation/scale optimization,
  no merge, refinement, or density control.
- The only intervention is the preceding frozen uniform edge loss:
  `0.25 * mean(Huber(||mu_i-mu_j||_1 / extent, delta=0.05))`.
  There is no coefficient, delta, norm, schedule, edge-weight, or matcher-backend sweep.

## Frozen matcher

The official backend is `PatchEpipolarMatcher` (`patch_epipolar`) and is constructed once per seed
before either family runs. Its API accepts only the nine training RGB tensors, their calibrated
cameras, and detached retained `(xy, source_view_ids, source_ranges)`. It cannot accept
`SceneData`, sparse/GT points, scene bounds, GT depths/Gaussians, corrupted Hybrid depth, lifter
initial positions, or held-out images/cameras. The graph is detached, bitwise reused by both
families, uniformly weighted, and never rematched.

For every retained center, sample the clamped bilinear raw-RGB 5x5 patch at integer offsets
`dx,dy in {-2,-1,0,1,2}` and flatten its 75 values without normalization. For every unordered
training-camera pair:

1. compute the calibrated fundamental matrix with `x_right^T F x_left = 0`;
2. retain candidates whose maximum of the two point-to-epipolar-line distances is at most 2 px;
3. use raw-patch L2 distance and reciprocal nearest neighbors, with lowest retained index as the
   deterministic `argmin` tie break;
4. require both `(best + 1e-6)/(second + 1e-6) <= 0.50`; a missing second eligible candidate is
   positive infinity, while exact descriptor ties therefore fail;
5. triangulate the closest points on the two infinite unit rays and use their midpoint; require
   nonparallel denominator `>1e-8`, both line parameters positive, acute unoriented ray angle at
   least 10 degrees, depth greater than 0.05 in both cameras, and maximum midpoint reprojection
   error at most 1.5 px;
6. drop a camera-pair block with fewer than two survivors; otherwise sort by `(left,right)` and
   retain every survivor. Confidence `1-max(forward_ratio,reverse_ratio)` is diagnostic only.

The control leaves sorted left endpoints fixed and cyclically shifts right endpoints by `-1`
within each retained block. It must preserve the exact endpoint degree vector, per-block endpoint
multisets, camera-pair counts/baselines, edge count, and confidence multiset, with zero exact edge
overlap. Its worse descriptor/geometric feasibility is expected and must be serialized alongside
step-zero residual/Huber diagnostics; correspondence attribution is explicitly scoped by this
limitation.

## Transparent pre-freeze probes and topology validity

Only fits, retained zero-step layouts, training images/cameras, graph structure, and matcher
covariates were intended for threshold selection. The final matcher above produced:

| seed | retained nodes | edges | represented nodes | node coverage | blocks | min nodes/view |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1303 | 177 | 239 | 18.342% | 35 | 22 |
| 1 | 1293 | 187 | 247 | 19.103% | 35 | 21 |
| 2 | 1262 | 165 | 227 | 17.987% | 34 | 17 |

This is 1.91x-2.47x the preceding same-seed node coverage, although edge counts are similar because
the former oracle repeated a smaller node set. Before the official run, every seed must have at
least 160 edges, 220 represented nodes, 17.5% node coverage, 1.85x its preceding same-seed node
coverage, 34 camera-pair blocks, all nine training views, and at least 16 represented nodes/view.
These are validity floors, not success criteria and cannot rescue failed geometry.

One implementation pilot reported an approximate seed-0 dominant-GT edge agreement near 90% for
the already-selected final matcher settings. No threshold was changed from that inspection, but it
means semantic precision is not fully blinded; this disclosure is part of the tracked provenance.
No held-out/source quality metric or 90-step arm outcome was inspected before this freeze. After
this file is written, matcher parameters, graph gates, loss, arms, and decision thresholds cannot
change.

## Post-freeze synthetic precision audit

GT contribution labels are diagnostic only and enter after positive/control pair hashes are fixed.
The literal reference compositor labels an endpoint only when total alpha contribution is at least
0.05, its dominant contribution is at least 0.05, and dominant purity is at least 0.50. An edge is
correct only when both endpoints are labeled and share the same dominant GT Gaussian; every
unlabeled edge counts as incorrect. The matcher is semantically valid only if positive precision is
at least 60% in every seed and at least twice shuffled precision in every seed. GT labels never
filter/reweight an edge. Failure diagnoses this matcher and closes the position branch without a
threshold rerun.

## Frozen mechanism, utility, safety, and attribution gates

All gains are paired against `none`, averaged over three seeds, and lower is better unless stated.

- **Engagement**: positive-edge normalized-L1 residual p90 improves at least 25%, with 3/3 wins.
- **Local geometry**: valid represented-node distance to its post-freeze dominant GT center p90
  improves at least 20%, with at least 2/3 wins.
- **Whole-scene materiality, both families**: held-out depth RMSE improves at least 2% and all-
  source clean-depth abs-relative p90 at least 10%, each with at least 2/3 wins.
- **Hybrid additional materiality**: corrupted-source p90 improves at least 15%, with at least 2/3
  wins.
- **Safety**: held-out PSNR changes by at least -0.10 dB, foreground coverage by at least -0.02,
  and alpha IoU by at least -0.02.
- **Control attribution**: on every applicable geometry metric, the positive graph beats the
  shuffled control in at least 2/3 seeds and the shuffled gain versus `none` is no more than half
  the positive gain.

Training PSNR/SSIM, cross-only training L1, medians, nearest-GT summaries, loss histories, and
timings remain secondary diagnostics and cannot rescue a failed primary gate.

## Required invariants and provenance

- Step-zero outputs/layout/rays are identical across arms within a family.
- At position lambda zero, outputs, total/anchor histories, RNG target schedule, and rays are exact.
- Every graph index is canonical, unique, and cross-source; control degree, endpoints, blocks, and
  source-pair counts are exact; positive/control overlap is zero.
- Graph/layout hashes are identical across families; position pairs do not consume optimization
  RNG; every 90-step arm visits all nine training views with identical schedules/counts.
- Means stay inside bounded ray/AABB intervals; every output/history is finite.
- Serialize raw pairs, confidence, graph components/covariates, per-view/per-block coverage,
  control mismatch, strict GT audit, local/global metrics, source hashes, dirty-tree diff metadata,
  environment, command, and exact configuration. A tracked output refuses non-frozen config and
  refuses overwrite.

## Stopping rule

- Structural failure before outcomes invalidates the run; it does not authorize a tracked result.
- Precision failure is a matcher failure: stop position consistency and do not tune thresholds.
- Precision + coverage + engagement + local correction but failed whole-scene materiality closes
  the position-consistency branch and pivots to the Scholar-grounded local plane/shortest-axis
  normal constraint, honestly scoped first to depth-backed Hybrid.
- Materiality plus control separation in both families advances only to real/calibrated replication
  with an optional learned matcher; one family advances only that family. It still cannot change a
  production default.
- Materiality without control separation is generic graph regularization, not correspondence
  evidence. Engagement failure closes the non-engaging position intervention.
- There will be exactly one official three-seed artifact, no post-outcome matcher/loss rerun, and
  no production default change under any outcome.
