# Inverse-projection fiber fitting, Iteration 2 — topology repair preregistration

Frozen at: 2026-07-17 (Europe/Berlin)

Status: **FROZEN BEFORE ITERATION 2 IMPLEMENTATION OR OFFICIAL EXECUTION**

## Prior evidence and question

Iteration 1e is a committed scientific `FAIL`, independently accepted as a valid negative result.
Its exact source fibers preserve their spawning 2D means/covariances to numerical precision, and
an oracle-correspondence arm recovers the synthetic 3D construction. Independent row-wise hard
minima instead converge to stable many-to-one and view-inconsistent assignments. The result and
audit are bound below:

| Artifact | SHA-256 |
| --- | --- |
| Iteration 1e result JSON | `2601a45d19d1d8a636d3c0db5ef8b14adf5f4137baaf718c86e1f80a84cecf9e` |
| Iteration 1e result note | `a108c099ee77dbf42857c3fd2e7b37d06e9d472e0c432c8a010fde7fc48880f3` |
| Iteration 1e audit JSON | `c45cdc9a67a61c34796b07388308dd4e678d268ebbcc4062419ccab7c379515a` |
| Iteration 1e audit note | `3ccaf78782521bafbce57464d678183cae51468efd6c4c6f64400944e7c147a4` |
| Iteration 1e executed sources | `cc23e3ab9e95307453e97193d71f84040a832b16b08fb4e9d231f661ecb1f5a5` |

The outcome-informed diagnostic found a large residual gap: mean non-source residual `<0.1`
selected all and only complete correct tracks in the three consumed roots, and a `0.01` world-unit
connected-component radius produced eight duplicate clusters. That diagnostic is not evidence for
this iteration. It only fixes the intervention and thresholds before fresh-root access.

The Iteration 2 question is:

> Can residual-gated pruning, source-preserving duplicate contraction, balanced rematching, and
> fixed-track refitting turn the 32 exact-source lifts into the eight hidden 3D Gaussians and
> recover all 2D correspondences on fresh synthetic scenes?

This tests the user's proposed topology mechanism. It does not replace the lift with supplied
tracks: correspondence is latent during the initial fit, pruning and cluster membership are
computed only from fitted geometry/residuals, and every retained 3D Gaussian keeps one exact
ground-truth source projection.

## Fresh identity and roots

```text
namespace:               rtgs.inverse-projection-fiber.iter2.v1
scene roots:             27688011, 27688012, 27688013
initial-depth roots:     27688111, 27688112, 27688113
observation-order roots: 27688211, 27688212, 27688213
result:                  benchmarks/results/20260717_inverse_projection_fiber_iter2_RESULT.json
artifacts:               runs/inverse_projection_fiber_iter2_official_20260717
```

An exact repository search immediately before this freeze found no occurrence of the namespace or
nine roots. Official execution may construct each root only once. Development tests must use
disjoint roots and paths.

## Frozen synthetic construction and label boundary

Retain Iteration 1e's geometry and numeric path:

- eight degree-zero anisotropic 3D Gaussians and six 64x64 calibrated ring cameras;
- fitting views `0,1,2,3`; reporting-only held-out views `4,5`;
- 32 exact inverse-projection fibers, one per fitting-view 2D Gaussian;
- EWA dilation `0.3`, depth bounds `[1.2,3.6]`, center plus `0.25` affine-invariant conic cost;
- CPU float64 Adam, learning rate `0.025`, betas `(0.9,0.999)`, epsilon `1e-8`;
- 400 initial hard-min updates followed by 200 compute-matched continuation updates; and
- checkpoints every 20 updates, with finite/SPD/depth/source-invariant sentinels.

Before candidate construction, independently permute the eight observation rows in every one of
the six views from the frozen observation-order root. The permutation is accepted exactly as drawn,
including an identity draw; it is never redrawn. Candidate code receives camera geometry, local
row indices, 2D means/covariances, and source-view indices only. Generator identities live in a
separate evaluator/control object and are forbidden from residual scoring, pruning, clustering,
representative selection, balanced rematching, and learned fixed-track refitting.

Development sentinels must show that relabeling local rows and undoing that relabeling leaves
physical costs, selections, cluster geometry, and balanced assignments unchanged. The official
result stores permutations, generator before/after states, hashes, and a candidate-input manifest.

## Frozen arms and topology algorithm

All arms share byte-identical scenes, observation permutations, initial depths, and initial fiber
geometry within each root.

### A. `hardmin_32` paired control

Run the Iteration 1e fiber-conic objective unchanged for 600 updates. For a hypothesis from source
view `s`, each other fitting view independently contributes its minimum over eight local target
rows. Freeze a complete snapshot at update 400; B and C start from independent copies of exactly
that state. Continue A with the same hard-min objective through update 600 so every relative
topology comparison uses 600 optimizer updates. Save exact float64 parameters and evaluation
arrays at both 400 and 600, every per-hypothesis/per-view cost matrix, assignments, residuals,
hidden-label evaluation arrays, and checkpoint histories.

### B. `residual_topology_8` proposed method

Starting from an independent copy of A's immutable update-400 snapshot:

1. For hypothesis `h` with source view `s`, compute

   `r_h = mean_{t in fitting views, t != s} min_j C[h,t,j]`.

All B/C residuals, complete-correct-track evaluator predicates, pruning decisions, clustering,
component metrics, and representative choices are evaluated from that update-400 topology point.
They never use A's update-600 continuation.

2. Retain `h` iff `r_h < 0.1` (strict); prune every other hypothesis.
3. Construct an undirected graph on retained hypotheses, adding an edge iff their Euclidean 3D
   center distance is strictly `<0.01`. Duplicate groups are graph connected components.
4. In each component retain exactly one source fiber: the member with minimum `r_h`; exact ties use
   `(source_view, local_source_row, original_hypothesis_index)`. Prune the other members. No moment
   averaging is permitted because it would destroy the exact source projection.
5. Continue only if there are exactly eight representatives. For each fitting view, hard-fix every
   representative spawned by that view to its exact local source row. Remove those fixed rows and
   columns, then solve a minimum-cost bijection between the remaining representatives and remaining
   observations. Use an exact dependency-free subset-DP solver with lexicographically smallest
   exact-tie resolution. Selection uses detached costs; the selected entries in the original
   differentiable matrix define the loss. If fixed source rows conflict or are not unique, reject
   the topology; never override an exact source anchor.
6. Freeze the four bijections and refit the eight representative fibers for 200 Adam updates.
   Each representative's own source observation remains exact by construction; its three other
   assigned fitting observations supply the optimization loss. No rematching or early stopping is
   allowed during recovery.

All 32 original fitting observations therefore receive exactly one learned track assignment if
the topology phase is valid. “Merge” means learned component membership plus pruning redundant
copies; the surviving Gaussian remains anchored to an actually observed source 2D Gaussian.

### C. `shuffled_residual_topology` negative control

Within each source-view group ordered by local source row `i=0..7`, set
`shift = 1 + observation_order_root % 7` and
`q[s,i] = r[s,(i+shift) % 8]`. Apply the same strict `q<0.1` threshold. This preserves the
selected count and score multiset per source view while changing which hypotheses survive. For C,
`q` also replaces `r` in representative minimum/tie selection; clustering and all geometry costs
remain unchanged. Use the same source-fixed exact-bijection and recovery code. If it does not form
exactly eight components, record a completed topology rejection and do not repair, redraw, or tune.

### D. `oracle_8` feasibility control

Use the eight view-0 source fibers from the common initialization and hidden generator identities
to fix their correct observations in the other three fitting views. Train for 600 updates, matching
the proposed method's total optimization depth. This control may access labels; no candidate API
may call it or receive its assignments.

## Weighting and appearance decision

Every row and target has uniform mass. Fitted 2D component weight is excluded from correspondence
capacity and evidence weighting: this repository has measured a `weight*color` gauge, so raw
weight is not identified as opacity, confidence, or existence probability. Color, opacity, and SH
are also excluded here. A source RGB/DC value is one directional observation, not ground truth for
all view-dependent SH coefficients.

## Frozen metrics and exact evidence

Before held-out release, record for A–C:

- exact source center/covariance residuals, depth bounds, SPD/finite flags, losses, gradients, and
  checkpoint histories;
- all `32x8` per-view geometry costs and selected local rows;
- per-hypothesis residual, retained flag, connected-component id, representative id, and reason;
- survivor count, component count/sizes, component center diameters, and representative source
  views; and
- using evaluator-only labels: survivor precision/recall for complete correct tracks, hidden-mode
  coverage, component purity/completeness, train assignment accuracy, exact four-view track
  fraction, GT center error, and affine-invariant covariance error.

Definitions are frozen as follows. For A, a complete correct track is one of 32 hypotheses whose
three non-source hard-min assignments all have the hypothesis source observation's hidden label;
exact-track fraction is the count divided by 32, and fitting assignment accuracy is correct
non-source assignments divided by 96. For B/C, an exact track is one of eight representatives
whose fixed/source assignment and three other view assignments share one hidden label;
exact-track fraction is divided by eight, and fitting assignment accuracy is correct entries among
the complete `8 representatives x 4 views = 32` bijection. Hidden-mode coverage is the number of
unique source hidden labels among retained survivors divided by eight, even if topology is later
rejected. Component purity is the fraction of components whose members share one source hidden
label; an empty component set has purity zero. Component completeness is the fraction of covered
hidden labels whose retained survivors all occur in one component. Survivor precision/recall use
A's complete-correct-track predicate as the evaluator-only positive class.

If B/C topology is rejected, primitive count, survivor/component/coverage metrics remain recorded;
its final fitting/held-out accuracy, exact-track fraction, and geometry-comparison values are
defined as `0.0` for gate arithmetic and separately carry `topology_rejected=true`. No absent value
is silently omitted from a mean.

Only after all three roots, all A/B/C fitting trajectories, every topology decision,
representative set, fitting assignment, oracle fitting trajectory, exact fitting-side array, and
their hashes are durably committed may one evaluator release views 4–5 once for reporting. Evaluate
one-to-one per-view assignments, held-out hidden-label accuracy, projected center/conic/combined
cost, 3D center median/p90, covariance median/p90, and primitive count. Held-out values cannot
select a representative, threshold, component, assignment, checkpoint, root, or arm.

For the overcomplete A control only, held-out evaluation solves four independent `8x8` bijections
per held-out view—one for each source-view group—so every one of its 32 predictions is assigned and
each observed target has capacity four overall. A held-out accuracy is therefore divided by
`32 hypotheses x 2 views = 64`, and its held-out geometry cost is the equal mean of the 32 selected
entries per view followed by the equal mean across two views. B/C/oracle use one `8x8` bijection per
held-out view and their accuracy denominator is `8 x 2 = 16`. No A held-out value enters topology
selection or the B-versus-A primary relative gates.

Unlike Iteration 1e, save exact float64 final tensors and exact evaluation arrays in NPZ files.
Every JSON stores their path, byte size, SHA-256, dtype, shape, and semantic hash. Save initial and
final PLYs only for visualization; PLY precision is not decision evidence.

All scalar geometry summaries use float64 values. For a sample sorted as `x[0]..x[n-1]`, quantile
`p` uses linear interpolation at `u=(n-1)*p`: with `a=floor(u)` and `b=ceil(u)`, return
`x[a] + (u-a)*(x[b]-x[a])`. Medians use `p=0.5` and p90 uses `p=0.9`. A component diameter is the
maximum pairwise Euclidean center distance among its members (zero for a singleton); component
diameter p90 applies that rule across all components. Held-out geometry cost is, per held-out view,
the arithmetic mean of the eight center-plus-`0.25`-conic entries for B/C/oracle or the 32 entries
from A's four source-group bijections, followed by an equal arithmetic mean across views 4 and 5.
Every cross-root “mean” is an equal arithmetic mean over the three frozen roots. The relative
center reduction is `(mean_A600 - mean_B600) / max(mean_A600, 1e-12)`.

## Frozen gates and falsification

### Gate 1 — protocol validity

Every actually optimized model must keep parameters/costs finite, covariances SPD, and loss depths
positive. Every materialized fiber must have source-center error `<=1e-6 px` and relative source-
covariance error `<=1e-5`. A is intentionally row-wise and is not required to be bijective. B/C
must have a valid source-fixed bijection and complete 200-update final tensors only conditional on
reaching exactly eight representatives; a frozen topology rejection instead requires a complete
rejection receipt and all pre-rejection arrays/hashes and remains a valid scientific failure or
negative control. The oracle must always have valid source-fixed bijections and complete tensors.
The candidate-label denial and relabeling sentinels must pass. Required exact arrays/tensors,
hashes, source/config/input manifests, initialization pairing, update counts, and checkpoints must
be complete for each reached phase. The oracle must achieve, in every root, train and held-out
accuracy `1.0`, exact-track fraction `1.0`, center p90 `<=0.01`, and covariance median `<=0.01`.
Otherwise the protocol is `INVALID`, not a scientific failure.

### Gate 2 — residual selection and contraction

The proposed method passes only if every root has:

- survivor precision `>=0.95` and recall `>=0.90` for complete correct tracks;
- all eight hidden modes covered;
- exactly eight connected components and representatives;
- component purity `1.0`; and
- component center-diameter p90 `<=0.01`.

Any root with a missing mode, extra/missing component, or impure component falsifies the frozen
topology hypothesis. No split, teleport, threshold change, nearest-count repair, or oracle fill-in
is allowed.

### Gate 3 — recovered correspondence and geometry

After 200 fixed-track updates, every proposed-method root must have:

- exactly eight primitives and exactly one assignment per original fitting observation;
- fitting assignment accuracy `1.0` and exact four-view track fraction `1.0`;
- held-out one-to-one association accuracy `>=0.95`;
- GT center p90 `<=0.05` and covariance-distance median `<=0.10`; and
- non-inferiority to its paired oracle:
  `center_p90 <= max(0.05, 1.25*oracle_center_p90)` and
  `heldout_geometry_cost <= 1.10*oracle_cost + 0.01`.

Relative to paired A at update 600, the proposed method must win center p90 and exact-track fraction
in all three roots, improve mean exact-track fraction by at least `0.20`, and reduce mean center
p90 by at least 50%. A's center p90 is over its 32 source-labelled hypotheses; B's is over its
eight representatives. If A already meets every absolute Gate-3 threshold in at least two roots,
relative topology attribution is `INCONCLUSIVE` even if B passes.

### Gate 4 — negative-control separation

The proposed method must exceed shuffled residual pruning by at least `0.25` mean hidden-mode
coverage and `0.25` mean exact-track fraction. The shuffled control must fail either eight-mode
coverage, exact component count/purity, or a final Gate-3 absolute requirement in at least two
roots. Otherwise residual identity, rather than merely retained count, is not established.

Overall scientific `PASS` requires Gates 1–4. A completed failure is retained and reported; it may
motivate splitting or unbalanced matching only in the next iteration, never amend this one.

The only supported positive claim is for the complete frozen bundle—residual pruning, spatial
duplicate contraction by source-preserving representative selection, source-fixed balanced
rematching, and fixed-track refitting. This factorial does not isolate which member of that bundle
is necessary, and it does not test splitting or a general dynamic topology lifecycle.

## Explicit exclusions and Iteration 3 boundary

Even a pass is limited to noiseless, fully visible, balanced synthetic scenes with `m=k=8` and a
known overcomplete 32-hypothesis initialization. It does not establish unknown `k`, variable or
unequal per-view counts, occlusion, compound 2D Gaussians, distractors, camera error, adaptive
thresholds, dustbin/unbalanced transport, split triggers, appearance recovery, SH, opacity,
real-scene correspondence, large-count scalability, speed, memory, CUDA, or a production default.

Iteration 3 is the only calibrated-data fit. It must perturb the surviving mechanism with fitted
2D observations, unequal/missing evidence, color/appearance, and a production-path viewer
interaction. If Iteration 2 fails because a true mode has no survivor, Iteration 3 may preregister
a residual-triggered split or relocation control. If contraction succeeds but rematching fails,
Iteration 3 may preregister unbalanced/dustbin transport. No such action is silently added here.
