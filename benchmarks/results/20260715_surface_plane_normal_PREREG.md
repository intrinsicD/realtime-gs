# Preregistration: Hybrid cross-view local-plane pulling and shortest-axis normal alignment

Frozen: 2026-07-15T12:31:16+02:00, before implementation of either oriented-target loss and
before any 90-step arm, clean-target audit, source-geometry result, or held-out result was run.
Repository revision: `2dddca4aff59702341af9faceefa76ad2505dd83` plus the dirty worktree bound by
the eventual artifact's source hashes and Git metadata.

Independent protocol-audit amendment: 2026-07-15T12:39:37+02:00, after the pure detached-target
loss API was implemented but before the target builder, benchmark integration, clean audit, any
optimized arm, or any source/held-out result existed. It clarified the ray parameter, clean-normal
metric, synthetic-boundary scope, arm-specific decisions, and required comparison with the thick
reference. It changed no target parameter, loss, coefficient, fit, seed, schedule, or already probed
structural floor.

## Question and literature boundary

Can fixed local planes built only from corrupted training-depth oriented points supply the
cross-view geometry signal that inclusive bounded-ray photometric optimization, anchor variants,
LOSO supervision, and sparse position consistency did not? Once the center is pulled to that
plane, does aligning a Gaussian's distinct shortest covariance axis with the plane normal improve
surface geometry rather than merely rotate appearance?

Incremental Gaussian Triangulation (IGT, [arXiv 2607.10690](https://arxiv.org/abs/2607.10690))
uses RGB-D oriented points, the point-to-plane term `|n_i^T (mu_i-p_i)|`, and the sign-invariant
shortest-axis term `1-|n_g^T n_i|`; it reports coefficients 0.05 and 0.2. IGT also uses planar
opaque surfels, dynamic insertion/triangulation, pruning/densification, active/frozen regions, a
CUDA oriented-point path, and real RGB-D data. This experiment copies no code and reproduces none
of those systems. It is a fixed-target, CPU-reference repository adaptation with ray-bounded fitted
2D Gaussians. A positive result would support only this mechanism and would not establish IGT,
RGB-only Gradient, learned monocular normals, online mapping, or real-scene performance.

A same-pixel source-depth point would make the plane term largely another depth anchor because
each mean is constrained to that pixel's ray. The frozen construction therefore queries only the
other eight training views. This cross-view exclusion is a repository inference, not a paper claim.

## Frozen data and arms

- CPU reference rasterizer, four Torch threads, deterministic seeds `0,1,2`.
- Per seed: 40-Gaussian synthetic scene, twelve 48x48 cameras, global held-out views `[3,7,11]`,
  and the remaining nine views physically subset before fitting, target construction, or lifting.
- One shared 150-Gaussian/view, 120-step stage-1 fit per seed. The canonical retained layout comes
  from a zero-step Hybrid run after `min_weight=0.05`, optional mask, and ray/AABB filtering.
- Metric training depths use the preceding deterministic 8x8 block corruption of privileged
  synthetic rendered depth: valid pixels in
  one third of blocks are multiplied by 1.2 in even local views and 0.8 in odd local views. The
  target builder receives only these corrupted depth tensors, the nine training cameras, retained
  source pixels/IDs/ranges, and the same sparse-point-derived synthetic train-scene bounds already
  used by Hybrid. It cannot receive clean depth, corruption masks,
  GT Gaussians/points, held-out data, RGB, confidence, or an optimized arm.
- Hybrid uses 90 bounded-ray steps, depth learning rate 0.1, rotation learning rate 0.005, jitter
  0.02, legacy anchor coefficient 0.01, inclusive `all` photometric supervision, fixed scales,
  fixed color/SH/opacity, no merge, refinement, or density control.
- Every arm enables quaternion optimization. The thick reference uses `ray_thickness=1.0`;
  the other four use the same target-independent planar thickness `ray_thickness=0.15`. This is the
  repository's existing surface-normal thickness ratio reused to make the selected shortest axis
  physically distinct; it is not an outcome-tuned IGT value.
- Five paired arms:
  1. `thick_none`: thickness 1.0, no new loss;
  2. `surfel_none`: thickness 0.15, correct targets supplied for diagnostics, both coefficients 0;
  3. `surfel_plane`: correct targets, plane coefficient 0.05, normal coefficient 0;
  4. `surfel_plane_normal`: correct targets, plane coefficient 0.05, normal coefficient 0.2;
  5. `surfel_plane_shuffled_normal`: the same correct plane loss and a within-source shuffled
     normal only for shortest-axis alignment, with coefficients 0.05 and 0.2.
- There is no target-radius, incidence, thickness, coefficient, optimizer, schedule, loss-form,
  confidence, scale, merge, refinement, or density sweep after this freeze.

## Frozen corrupted-train-depth local-plane targets

For each corrupted local training depth map, enumerate all 48x48 pixel centers in row-major order.
At every finite depth greater than 0.05, unproject the center and retain it only when the point lies
inside the same train-scene AABB. Candidate ordering is local view index followed by row-major pixel
index. No color, confidence, derivative, clean geometry, or oriented normal enters this pool.

For each retained fitted primitive, in retained-index order:

1. Bilinearly sample its own corrupted source depth. Require it to be finite and strictly inside
   that source ray's existing `[near,far]` AABB interval, with `near>=0.05`.
2. Unproject that unjittered prior depth to a query point. Concatenate points from the other eight
   local training views and stably select the globally nearest four by `(squared distance, local
   view, row, column)`. Require at least two distinct supporting views and farthest-query distance
   at most `0.10 * scene_extent`.
3. Let target point `p` be the four-point centroid and
   `C=(1/4) sum_j (q_j-p)(q_j-p)^T`. Use the smallest-eigenvalue eigenvector of symmetric `C` as
   `n`; canonicalize its sign by making its largest-absolute coordinate positive, with coordinate
   tie order x/y/z. Require `lambda_min / sum(lambda) <= 0.05`. Do not add a hidden color, depth-
   discontinuity, normal-consistency, or per-view-nearest filter.
4. Require `|n dot (d/||d||)| >= 0.10`, where `d` is the original non-unit direction returned by
   `Camera.pixel_rays`. Compute the camera-depth ray parameter exactly as
   `t_star = n dot (p-o) / (n dot d)` and require finite `near < t_star < far`. The unit direction
   is used only for incidence; comparing a unit-ray distance to the camera-depth bounds is invalid.
5. Emit an explicit detached retained index, target point/normal, four support indices/views, and
   diagnostic distances/eigenvalues/incidence/intersection data. Targets are constructed once,
   hashed, and bitwise reused; they are never rematched or regenerated inside an arm.

The target API must validate unique increasing int64 retained indices, exact `(M,3)` point/normal
shapes, detached inputs, finite values, nonzero normals, and in-range layout indices. Its artifact
is bound to hashes of retained pixels, source IDs/ranges, corrupted depths, target indices/points/
normals, and candidate metadata. Invalid target slots are omitted rather than multiplied by zero.

## Frozen shuffled-normal control

Within every query source-view group, sort target slots by `(source_y, source_x, retained_index)`
and cyclically roll the normal tensor by `floor(group_count/2)`. Points, target indices,
applicability, support metadata, and the normals used by `L_plane` stay correct and fixed. Only the
normal supplied to `L_normal` is shuffled. The assignment has no fixed slot and must preserve the
exact alignment-normal tensor multiset within each source group, while correct and shuffled hashes
must differ. This is a location-specific shortest-axis-normal control conditional on the same
correct plane, not a control for plane pulling or an alternative target builder.

## Transparent train-input feasibility probes

Before this freeze, probes used only shared fits, corrupted training depths, training cameras,
retained layouts, bounds, and step-zero parameters. No clean-depth normal/plane audit, source error,
held-out metric, or optimized arm was inspected. An initial radius/incidence probe motivated the
final AABB-valid and reachable-plane implementation. Replaying the exact final construction gave:

| seed | retained | targets | coverage | corrupted targets / corrupted nodes | min targets/source |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1303 | 339 | 26.02% | 98 / 305 (32.13%) | 26 |
| 1 | 1293 | 318 | 24.59% | 79 / 268 (29.48%) | 24 |
| 2 | 1262 | 326 | 25.83% | 67 / 260 (25.77%) | 31 |

The farthest of four neighbors had normalized p90 0.064-0.070 and plane/ray incidence p10
0.184-0.193. PCA `lambda_mid/sum(lambda)` had minimum 0.0133-0.0149, p10 0.081-0.101, and median
0.225-0.245, so the accepted four-point sets were not line-degenerate. With the common 0.15
thickness, the second/shortest scale ratio had median 6.60 and p10 6.21-6.27, versus an ambiguous
near-tie near 1.005 at historical thickness 1.0. A final control replay must report its separation.

These probes freeze validity floors, not success criteria. Before any clean audit or optimization,
every seed must have at least 300 targets, 23% retained-node coverage, at least 60 targets in the
known corruption stratum, all nine query and candidate source views, at least 20 targets per query
source, farthest-neighbor p90 at most 0.08 extent, incidence p10 at least 0.10,
`lambda_mid/sum(lambda)` minimum at least 0.01, and second/shortest scale-ratio p10 at least 5.0.
All nine local views must appear in selected four-point support metadata, not merely in the
candidate pool. In every seed, the shuffled alignment control must preserve the exact per-source
normal multiset and have median `1-|n dot n_shuffled| >= 0.25`. Failure is structural and refuses a
tracked result; floors may not be relaxed.

## Post-freeze synthetic target audit

Only after correct/control target tensors and hashes are frozen may the harness use clean training
depth and the known corruption mask for diagnostic target validity. At each target's own source
pixel, construct the clean point and normal using the existing validity-aware surface-Jacobian
convention: bilinearly sample clean depth and its validity-aware derivatives; form
`q=[(u-cx)/fx,(v-cy)/fy,1]`, `J_u=[z/fx,0,0]+q*dD/du`, and
`J_v=[0,z/fy,0]+q*dD/dv`; then use
`n_clean=normalize(cross(J_u,J_v)) @ camera.R`. A clean target is labelable only when depth and
derivatives are finite, depth is greater than 0.05, and the cross product has norm greater than
`1e-8`. Every target must remain labelable. In every seed, both all-target and corrupted-target
subsets must have:

- target-plane-to-clean-point absolute residual p90 at most 0.10 scene extent; and
- median unoriented target-normal/clean-normal cosine at least 0.50.

The final clean-normal geometry metric is the p90 over target rows of
`1-|a_selected dot n_clean|`, using the same frozen parameter-axis index as the optimized normal
loss. Clean depths and normals never select targets or optimization parameters.

If this audit fails, the four-point cross-view target constructor is rejected and the tracked run
stops before all 90-step arms. Clean data never filters, changes, reweights, or rematches a target.

## Frozen losses and axis convention

For explicit target rows `m`, current pre-merge means `mu`, target points `p`, and normalized
target normals `n`:

`L_plane = mean_m |n_m^T (mu[index_m] - p_m)| / scene_extent`

For `(w,x,y,z)` quaternions, `R` satisfies `Sigma=R diag(s^2) R^T`; its columns are local axes.
Freeze each target's minimum-scale parameter-axis index from the common step-zero log scales and
gather `a_m = R[:, :, frozen_axis_m]` throughout training:

`L_normal = mean_m (1 - clamp(|a_m^T n_m|, max=1))`.

The two active coefficients are exactly the IGT-reported 0.05 and 0.2. Replacing paper sums with
means and normalizing plane distance by scene extent are repository adaptations needed to make the
objective invariant to target count and synthetic scene scale. The axis claim is therefore
"selected distinct minimum-scale parameter axis," not a differentiable covariance eigensystem.

The implementation records complete plane and normal histories even at zero coefficient, but adds
each loss to the objective only when its coefficient is positive. The shuffled arm always uses the
correct target normal for `L_plane` and the shuffled alignment normal only for `L_normal`.
Positive plane/normal coefficients require targets; a positive normal coefficient requires
rotation optimization.

## Required invariants and diagnostics

The official harness must fail rather than serialize arm outcomes if any invariant fails:

- no-target behavior remains exact; supplied targets at both coefficients zero preserve output
  fields, objective/anchor histories, target-view schedule, rendered counts, and ray fractions;
- the four surfel arms are step-zero identical; they have identical retained layouts, plane-target
  hashes, fixed axis indices, optimizer groups, and initial parameters. `thick_none` differs only
  in scales/quaternions induced by its preregistered thickness;
- all actual arms use identical 90-view target schedules, render counts, and primitive counts;
- targets and normals are detached and unchanged after every arm; the shuffled control preserves
  exact indices, points, correct plane normals, per-source alignment-normal multisets, and
  applicability with no fixed assignment;
- every output/history is finite; means remain on their original rays and within AABB fractions;
  all nine training views occur in the official schedule;
- serialize raw targets/control permutation, layout/prior/fit/target hashes, source and candidate
  coverage, locality/incidence/reachability, scale gaps, step-zero and final correct/control plane
  and normal residuals, quaternion change, ray saturation, clean-target audit, source/held-out
  geometry, PSNR/coverage/IoU, complete histories/timings, command/config, environment, source
  hashes, revision, dirty status, and tracked-diff provenance.

## Frozen mechanism, utility, safety, and attribution gates

All gains are three-seed paired means; lower is better unless stated. For any evaluated arm and
named baseline, the **global materiality/safety predicate** requires held-out expected-depth
RMSE/extent to improve at least 2%, all-source clean-depth absolute-relative p90 at least 10%, and
corrupted-source p90 at least 15%, each with at least 2/3 seed wins. Held-out PSNR delta must be at
least -0.10 dB and foreground-coverage and alpha-IoU deltas each at least -0.02. Both loss arms must
satisfy this full predicate separately against `surfel_none` and `thick_none`; thinning a baseline
cannot manufacture an advancing result.

The target-supported clean-depth metric projects the final pre-merge target-indexed mean into its
own source camera and computes `|z_pred-z_clean|/max(z_clean,0.05)`. Its p90 is reported over all
targets and over targets in the known corruption stratum.

The **plane-only pass** for `surfel_plane` requires all of:

- correct-target point-to-plane p90 improves at least 25% versus `surfel_none`, with 3/3 wins;
- target-supported clean-depth p90 improves at least 15% overall and 20% in the corruption stratum
  versus `surfel_none`, each with at least 2/3 wins; and
- the global materiality/safety predicate passes against both `surfel_none` and `thick_none`.

There is no shuffled-plane control. A plane-only pass is therefore evidence for this fixed cross-
view plane/depth regularizer, not for semantic plane correspondence or a universally correct local
plane assignment.

The **combined pass** for `surfel_plane_normal` requires all of:

- its correct-target point-to-plane p90 improves at least 25% versus `surfel_none`, with 3/3 wins;
- correct-target normal-loss p90 improves at least 25% versus `surfel_plane`, with 3/3 wins;
- target-supported clean-depth p90 improves at least 15% overall and 20% in the corruption stratum
  versus `surfel_none`, each with at least 2/3 wins;
- selected-axis-to-clean-source-normal p90 improves at least 20% versus `surfel_plane`, with at
  least 2/3 wins;
- the global materiality/safety predicate passes against both `surfel_none` and `thick_none`; and
- it beats `surfel_plane_shuffled_normal` on target-normal p90 in 3/3 seeds, on clean-local-normal
  p90 in at least 2/3, and on every primary global geometry metric in at least 2/3. The shuffled
  arm may preserve no more than half of the correct normal gain over `surfel_plane` and no more
  than half of each correct global gain over `surfel_none`.

`thick_none` uses rotation optimization and is not a replay of the preceding rotation-frozen
position experiment; it isolates thickness within this experiment. Neither its result nor a
`thick_none -> surfel_none` gain can rescue a loss that fails against `surfel_none`. Training
metrics, medians, nearest-GT centers, quaternion magnitude, loss curves, and runtime remain
secondary and cannot rescue a failed gate.

## Stopping and interpretation

- Structural target failure invalidates the run and creates no official result; thresholds do not
  move. Post-freeze clean-target-audit failure creates the sole tracked stopped artifact, rejects
  this four-point cross-view constructor, and runs no optimization arm.
- If the losses do not engage, close this fixed-loss adaptation without coefficient/schedule tuning.
- If engagement and clean-local-shape pass but global materiality fails, classify the mechanism as
  locally effective but globally insufficient and close synthetic plane/normal loss sweeps. The
  next valid evidence would require actual calibrated monocular/RGB-D normals, not another target
  threshold sweep on these three scenes.
- If `surfel_plane` passes but the combined arm fails normal engagement, local shape, safety, or
  global utility, retain only plane pulling as a research mechanism and reject shortest-axis
  alignment.
- If the common thin-surface reference explains a combined-versus-thick change while the combined
  arm fails against `surfel_none`, attribute the result to covariance initialization, not the new
  plane/normal losses.
- Materiality without shuffled-control separation is generic surface regularization, not evidence
  for the correct local normal assignment.
- A full pass advances Hybrid only to real/calibrated replication with a pluggable depth/normal
  backend. It does not change a production default; RGB-only Gradient remains out of scope.
- Exactly one official three-seed artifact is allowed. A tracked output requires the exact defaults,
  refuses overwrite, and may not be rerun after outcomes. No target/loss/thickness coefficient or
  gate changes are permitted in response to that artifact.
