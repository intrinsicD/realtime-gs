# Preregistration: fixed-match world-frame position consistency

Frozen: 2026-07-15T10:06:08+02:00, before implementation of the position loss and before any
position-loss arm was optimized. Repository revision: `2dddca4aff59702341af9faceefa76ad2505dd83`
plus the source-bound dirty worktree recorded by the official artifact.

Audit chronology: the initial freeze fixed the arms, loss, coefficient, data, global gates, and
one-run rule. Before implementation and before any position optimization, an independent graph
replay added stricter contribution parity and the non-rescuing assigned-GT local-mechanism
classification. After implementation, one untracked seed-0 three-iteration smoke exercised the
first harness. A final source audit on 2026-07-15, still before any 90-step or tracked run, found and
corrected the documentation/harness inconsistencies itemized below. The smoke is not an official
result. No arm, loss coefficient, loss delta/norm, graph threshold, numerical outcome gate, data,
seed, or one-run rule changed in response to it.

## Pre-official audit clarifications

These amendments were made at 2026-07-15T10:37:05+02:00, before the sole official run:

The last pre-amendment smoke artifact (`/tmp/rtgs_position_smoke.json`, intentionally untracked)
embedded preregistration SHA-256
`3e4a6ed1da339f0649644eda5f833746339e463e5d01cba115f46bdf15920181` and harness SHA-256
`33dc8a4a19e456f7cfe51756207f6a72eacf61d9aea329cb3112d3a7dc4b1884`. The official artifact will
bind the amended files separately; this records that the post-smoke clarifications are not being
presented as part of the original 10:06 file state.

- An intermediate feasibility replay produced 148/123/149 edges by sampling the expected-depth
  criterion at retained fitted centers. That implementation contradicted the already written graph
  rule below, which samples at projected GT centers. Replaying the written rule gives 169/140/175
  edges, 106/100/119 represented primitives, 32/31/31 non-singleton view-pair blocks, and
  27/30/33 represented GT identities for seeds 0/1/2. The written projected-GT rule is retained;
  no threshold or graph-validity floor changes.
- “Source-depth p90” means the existing absolute-relative error
  `|z_pred - z_gt| / max(z_gt, 0.05)`, not error divided by scene extent. The held-out depth RMSE
  remains extent-normalized. The earlier `/extent` wording for source errors was erroneous; the
  numerical thresholds remain 10% and 15%.
- The frozen shuffled-control rule is executable as follows, without a new similarity threshold:
  a family is classified as generic graph regularization when its material-effect gates pass but
  the already frozen per-metric attribution test fails. This branch precedes the sparse-coverage
  branch. Position-loss coefficient/delta/norm/schedule sweeps remain stopped under every outcome.
- Diagnostic-only additions serialize closest-ray feasibility, midpoint-to-assigned-GT and
  assigned-GT-to-endpoint-ray errors, family-specific step-zero correct/shuffled residuals, and
  complete histories. Cross-family retained layouts and graph hashes are now asserted rather than
  merely recorded. At lambda zero, “histories exact” means the optimization-relevant total-loss,
  anchor-loss, and target-view histories; paired arms intentionally still populate their
  topology-specific position diagnostic, but it is multiplied by zero and cannot affect the
  update. None of these diagnostics can rescue a failed primary gate.

## Question and scope

Does a single robust world-frame position-consistency term between genuinely corresponding
train-view fitted primitives supply material geometry information that inclusive photometric
bounded-ray optimization lacks? Is any benefit attributable to correct correspondence topology
rather than a generic degree-matched graph regularizer?

This is a **repo-specific, position-only oracle adaptation**, not a MAC-Splat or EDGS reproduction.
MAC-Splat applies Huber to the L1 world-coordinate discrepancy of confidence-filtered reciprocal
matches and reports an overall MAC coefficient of 0.25, but jointly trains position, shape, and
appearance and does not report its Huber delta or a position-only ablation. EDGS supports direct
correspondence-derived geometry and spatial coverage but is a triangulated initializer, not this
loss. Its code is not reused.

The positive topology below uses synthetic GT identities and alpha contributions. It is a
privileged upper-bound diagnostic: no outcome can authorize a production default or a deployable
matcher claim. GT coordinates and depths define topology/validity only and are never loss targets.
Held-out cameras, images, and depth maps are not used to fit primitives, choose pairs, or optimize.

## Pre-freeze feasibility work

Before freezing, read-only graph-construction probes ran on seeds 0/1/2 with the intended fitted
layouts, but **no position-loss arm was optimized and no arm quality metric was inspected**.
Threshold probes selected a 1.5-pixel projected-center radius, 0.10-extent visibility-depth
tolerance, 0.50 compositing-purity floor, and 10-degree acute ray-angle floor to retain a clean,
nondegenerate graph. A stricter independent pre-implementation replay added exact reference-
compositor contribution parity. The intermediate replay's fitted-center depth check produced
148/123/149 edges, 98/89/107 represented retained primitives, 31/31/32 view-pair blocks, and
26/27/31 represented GT identities for seeds 0/1/2. The final pre-official audit identified that
check as inconsistent with frozen graph step 2 and replaced these as operational feasibility counts
with the projected-GT-center counts above. No full arm was optimized between these probes. These
diagnostics set graph validity floors only; they did not select the loss coefficient or outcome
gates.

## Frozen experiment

Two families by three paired arms:

1. `none`: inclusive photometric supervision, no position term;
2. `oracle_position`: inclusive photometric supervision plus the correct fixed graph; and
3. `degree_shuffled_position`: the same term on a semantic derangement with exactly matched graph
   degree and camera-pair counts.

Shared scene and optimization protocol:

- seeds 0/1/2; 40 synthetic GT Gaussians; twelve 48x48 cameras;
- global train views `[0,1,2,4,5,6,8,9,10]`; held-out views `[3,7,11]`;
- 150 fitted 2D Gaussians per train image, 120 fit iterations, identical fitted tensors per arm;
- 90 lift iterations, depth learning rate 0.1, CPU reference renderer, four Torch threads;
- `photometric_supervision_mode="all"` in every arm;
- fixed color, opacity, rotation, and free scale parameters; merging, refinement, and density
  disabled. As in the inherited lifter, world scale still changes deterministically with optimized
  ray depth to preserve the fitted projected footprint, so the intervention is position-only in
  its added loss but does not hold derived covariance numerically constant;
- Gradient: jitter 0.15, legacy anchor lambda 0.001, no metric prior;
- Hybrid: jitter 0.02, legacy anchor lambda 0.01, deterministic metric GT depth corrupted in the
  same frozen 8x8 +20%/-20% blocks used by the preceding experiments, without confidence;
- no shape, appearance, opacity, covariance, plane, normal, or confidence consistency term.

## Frozen oracle graph

The graph is built once per seed after an iterations-zero Gradient layout pass and reused bitwise
for both families. Retained pixel coordinates and concatenated indices come from the lifter after
weight, mask, ray/AABB, and source-order filtering; the harness must not reimplement that layout.

For each training view:

1. Project all synthetic GT Gaussian centers into the training camera.
2. Retain centers in front of/in the image whose bilinearly sampled rendered expected depth is
   positive and differs from the center's camera depth by at most `0.10 * scene_extent`.
3. Form mutual-nearest pairs between retained fitted centers and eligible projected GT centers;
   require Euclidean image distance at most 1.5 pixels.
4. At the selected fitted center, reproduce the CPU reference renderer's depth-sorted alpha
   contribution for every GT Gaussian. Require the assigned identity's absolute contribution at
   least 0.05, contribution/total-alpha purity at least 0.50, and dominant contribution identity.
5. Keep at most one retained fitted primitive per GT identity per view.

For every unordered training-view pair, connect representatives sharing the same GT identity when
the acute angle between their two world-space ray lines is at least 10 degrees. Drop view-pair
blocks with fewer than two edges from both topology arms. The correct graph must contain at least
100 edges, 75 represented primitives, 24 non-singleton view-pair blocks, 20 represented GT
identities, all nine training views, and at least five retained representatives per view for every
seed; otherwise the official run is invalid and thresholds may not be relaxed after seeing outcomes.

The shuffled graph is constructed independently inside each unordered view-pair block: order edges
by GT identity and cyclically rotate the right endpoints by one. Because identities and endpoints
are unique within a block, this yields zero same-identity edges while preserving exactly:

- total edge count and uniform weights;
- every retained primitive's global endpoint degree;
- every unordered camera-pair count and camera-baseline distribution; and
- the left/right endpoint multiset within every block.

The graph tensors are detached, hashed, range-checked, cross-source, duplicate-free, and identical
across Gradient and Hybrid. No graph construction consumes the lifter's jitter/target-view RNG.

## Frozen position loss

For current pre-merge world-space means and fixed edge set `E`:

`r_k = ||mu_i - mu_j||_1 / scene_extent`

`L_pos = mean(Huber(r_k, target=0, delta=0.05))`

The two topology arms add `0.25 * L_pos` to the unchanged inclusive photometric and legacy anchor
objective. Weights are uniform. The coefficient 0.25 is a single MAC-Splat-motivated starting
point, while extent normalization and delta 0.05 are repository-specific; there is no coefficient,
delta, norm, or schedule sweep. The loss acts on bounded-ray means only and cannot move primitives
off their original rays or outside the scene AABB.

## Required invariants and diagnostics

The official harness must fail rather than serialize if any invariant fails:

- default/no-pair behavior is bit-exact; all arms are step-zero identical within family;
- at lambda zero, correct and shuffled pairs leave fields, total/anchor histories, and target
  schedules exact; topology-specific position diagnostics may differ but have zero objective
  weight;
- all actual-lambda arms share the full 90-step target schedule and final primitive count;
- retained source IDs/ranges/pixel centers and both graph tensors are identical across families;
- shuffled versus correct degree, endpoint, source-pair, pair-count, and baseline histograms are
  exact; semantic edge overlap is zero;
- all means remain within their original ray/AABB fractions; fields and histories are finite;
- every training view is sampled during the official schedule;
- complete config, command, Git/dirty state, source hashes, fitted/prior/layout/graph hashes,
  position and total-loss histories, graph coverage/purity/angle statistics, pair residual p50/p90,
  closest-ray and assigned-GT ray-feasibility errors, represented-primitive distance to its assigned
  GT center, ray-bound saturation, runtime, and raw per-seed metrics are serialized.

The correct term is considered **engaged** only if its final correct-edge normalized L1 p90 is at
least 25% lower than `none` on average and lower in all three seeds. Engagement is a non-rescuing
mechanism requirement: it cannot substitute for the utility gates below.

A separate local-geometry mechanism signal requires the correct arm's represented-primitive-to-
assigned-GT-center p90/extent to improve at least 20% over `none`, with at least 2/3 seed wins. It
also cannot rescue a failed global utility decision, but distinguishes a coverage-limited graph
from pair collapse toward incorrect midpoints.

## Frozen primary gates

For each family, `oracle_position` versus `none` must satisfy all applicable criteria:

- held-out expected-depth RMSE/extent improves at least 2%, with at least 2/3 seed wins;
- all-source clean-depth absolute-relative p90 improves at least 10%, with at least 2/3 wins;
- Hybrid corrupted-source absolute-relative p90 improves at least 15%, with at least 2/3 wins;
- held-out PSNR changes by at least -0.10 dB;
- foreground coverage and alpha IoU each change by at least -0.02; and
- the position term satisfies the engagement criterion.

Attribution additionally requires, for every applicable primary geometry metric:

- correct topology beats degree-shuffled topology in at least 2/3 seeds; and
- the shuffled arm preserves at most half of the correct arm's gain over `none`.

Training PSNR/L1, SSIM, nearest-GT center distances, pair-residual median, bound saturation, and
runtime are secondary diagnostics and cannot rescue a failed primary decision.

## Frozen interpretation and stopping rule

- If correct topology passes utility and attribution in both families, the next experiment replaces
  oracle identities with one deployable train-only matcher before any shape/appearance term.
- If exactly one family passes, only that family advances to a deployable-matcher replication.
- If the correct loss engages and the local-geometry signal passes but global utility fails, treat
  sparse graph coverage as the leading bottleneck and allow one denser train-only matcher
  experiment; do not sweep coefficient/delta/norm/schedule.
- If the correct loss engages but the local-geometry signal fails, stop position-consistency sweeps
  on this synthetic setup and pivot to the Scholar-grounded local plane/normal constraint branch.
- If correct and shuffled improve similarly, classify the effect as generic graph regularization,
  not correspondence evidence. Operationally this means material utility passes but at least one
  already frozen attribution-control criterion fails; no separate similarity threshold is added.
- If engagement or a structural invariant fails, report the tested intervention as non-engaging or
  invalid; do not tune or rerun under this preregistration.
- No oracle result changes the inclusive production default.
- Do not sweep the position-loss coefficient, delta, norm, or schedule under any outcome.

Exactly one tracked official run is allowed with the exact defaults above. Smoke tests and
lambda-zero invariant checks may run before it. The official output path is timestamped and must
not already exist; completed official outcomes may not be overwritten or silently rerun.
