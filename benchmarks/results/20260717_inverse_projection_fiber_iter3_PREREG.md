# Iteration 3 preregistration — capacity-aware soft correspondence on exact Gaussian fibers

Date frozen: 2026-07-17 (Europe/Berlin)

This is the third and final evidence iteration for the inverse-projection-fiber question begun in
`20260717_inverse_projection_fiber_iter1e_PREREG.md`. It is written after the independently
accepted Iteration 2 negative result and before any Iteration 3 optimizer outcome is generated or
examined. A development smoke may reject an invalid implementation, but may not tune the frozen
scientific roots, schedules, arm definitions, thresholds, or real split.

## Bound prior evidence and tree state

- Repository revision: `2dddca4aff59702341af9faceefa76ad2505dd83` with a dirty working tree.
- Dirty tracked binary-diff SHA-256 at freeze time:
  `e2940e30369a5532756f0e7f35a91dbfae27d07bac3cbad5c596052a83967dd3`.
- Iteration 2 result SHA-256:
  `d153706a5534a5f1d319d18b2961c944842bb01cd1573992b280c5ce096a2dfd`.
- Iteration 2 independent audit SHA-256:
  `98440bd7b14de8a959b2aac0c5172b9de31da7e21f3e58a3ba2883b2546dd55d`.
- Strict compact-bundle manifest SHA-256:
  `6ed60cf3df1f9ca476dfabace18b8d868d63eebd041522313c5b4fd644ee2614`.

The Iteration 2 mechanism failure is the premise for this experiment: post-hoc residual
contraction can remove duplicate hypotheses, but it cannot recover a hidden mode for which hard
row-minimum training left no correctly localized candidate. The proposed change moves observation
capacity and unmatched mass into the fitting loop instead of attempting to repair topology only
after fitting.

## Question and hypothesis

**Question.** Does full-covariance, dustbin-aware soft assignment on an exact source fiber retain
cross-view modes under unequal 2D decompositions better than independent hard minima, and is any
gain specifically attributable to a two-sided capacity constraint rather than softness alone?

**Primary hypothesis.** An augmented unbalanced-transport E-step with footprint-area capacities,
alternated with gradient updates of only the four exact fiber coordinates, will improve
transported parent purity and completeness over hard minima and row-softmax when different views
split the same projected 3D Gaussian into one, two, or three 2D components.

**Real-data feasibility hypothesis.** With arm choice fixed by the synthetic mechanism result, a
640-hypothesis fit on the frozen calibrated compact bundle will preserve every source mean and
covariance, remain inside development-only ray bounds, reduce validation Gaussian-set cost from
its common midpoint initialization, and produce finite viewer-ready 3D Gaussians. Real tracks are
called *inferred associations*, never true correspondences.

## Shared method contract

1. `InverseProjectionFiber` remains the geometry parameterization. Every track has exactly four
   geometric variables: camera-z depth, two tangent-ray Cholesky completion coordinates, and log
   ray scale. The source 2D center and tangent covariance block are not optimized.
2. The matching cost is the two-dimensional Bhattacharyya distance. For residual variance
   `sigma2`, `A'=A+sigma2 I`, `B'=B+sigma2 I`, `S=(A'+B')/2`, and

   `C = 1/8 d^T S^-1 d + 1/2(logdet S - 1/2 logdet A' - 1/2 logdet B')`.

   This uses center and covariance shape without the product-likelihood log-determinant incentive
   to collapse the free ray thickness. Plans are computed without autograd and detached during the
   M-step.
3. A track's own source view is excluded from association and loss. Its source observation is a
   hard equality constraint, not evidence that may inflate support.
4. The row-softmax arm appends one finite dustbin logit per track. The transport arms augment each
   `N x M` view problem with a dust row and dust column. If real source capacities sum to `A` and
   target capacities to `B`, the appended masses are `B` and `A`; both augmented marginals are
   normalized by `A+B`. Generalized log-Sinkhorn uses
   `phi=rho/(rho+epsilon)`. Real mass, conditional support, entropy, and both dust routes are saved
   separately.
5. Association capacity is neither posterior confidence nor rendering opacity. Amplitude is never
   copied into opacity. The real compact bundle has amplitudes identically one and therefore
   contributes no confidence signal. The area arm uses `sqrt(det covariance)` only as a transport
   capacity; the uniform arm is the attribution control.
6. Geometry optimization is generalized EM-inspired alternating descent, not claimed to be exact
   BCPD, variational Bayes, or a monotone ELBO: the M-step is finite-step Adam, temperature and
   residual variance are annealed, and the transport is unbalanced.
7. No trainable lateral source offset, camera correction, coherence prior, track-death prior,
   split event, opacity update, or renderer loss is permitted in the primary geometry comparison.
   Final support/death and duplicate-cluster statistics are diagnostics. This isolates assignment
   before adding topology surgery.

## Synthetic unequal-decomposition experiment

### Inputs and split

- Eight deterministic degree-zero 3D Gaussians and seven calibrated ring cameras.
- Five fitting views and two held-out views. Held-out projections and labels are materialized only
  after geometry, transports, and any support decisions are frozen.
- Official scene roots: `37688011,37688012,37688013`.
- Official depth roots: `37688111,37688112,37688113`.
- Official order/split roots: `37688211,37688212,37688213`.
- Separate development roots must not overlap those nine roots.
- Every parent projection is deterministically moment-split into `k in {1,2,3}` equal-weight
  children. View-specific `k` patterns are frozen in the benchmark source and cyclically permuted
  across parents; child centers have zero weighted offset and child covariance plus between-child
  covariance exactly recovers the parent covariance. Two deterministic outlier observations are
  added per fitting view and labeled only for evaluation.
- Every fitting observation, including outliers, seeds one source fiber. Initial depths are drawn
  once per root strictly inside `[1.2,3.6]` and reused byte-for-byte across arms.

### Arms

- **A — hardmin:** independent row-wise minimum Bhattacharyya cost, no dustbin.
- **B — row:** row-softmax over observations plus dustbin; uniform row capacity.
- **C — uot-uniform:** augmented unbalanced transport with uniform real capacities.
- **D — uot-area:** the same transport with `sqrt(det covariance)` capacities.
- **O — oracle ceiling:** parent labels fix observation support; labels never enter another arm.
- **S — shuffled-view negative control:** arm D with each non-source camera consuming the next
  fitting view's observation geometry cyclically. This changes the calibrated evidence rather
  than merely shuffling evaluator labels.

All non-oracle arms use the same initial fiber state, Bhattacharyya implementation, optimizer,
number of projections, and annealing schedule. Frozen defaults are 20 outer E-steps, two Adam
M-updates per E-step, learning rate `0.025`, exponential temperature `2.0 -> 0.10`, exponential
residual variance `1.0 -> 0.05 px^2`, dustbin cost `4.0`, 50 log-Sinkhorn iterations, and
`rho=1.0`. A correctness repair may change arithmetic only if it is applied to all affected arms
and documented before official roots are accessed.

### Metrics

Primary metrics are computed from real transported mass on non-source fitting views:

- parent purity: mass assigned between observations carrying the same hidden parent, divided by
  all inlier real mass;
- parent completeness: fraction of the eight parents receiving non-dust support in every fitting
  view in which that parent is observed;
- outlier dust recall and inlier dust false-positive rate;
- conditional association entropy and real transported mass.

Secondary metrics are held-out parent assignment, per-parent center p90, affine-invariant
covariance median, source center/covariance residual, depth-bound incidence, support-view
histogram, final proximity-cluster count/purity, wall time, and peak RSS. Hidden labels may only be
read by the evaluator after each arm's learned state is frozen.

### Synthetic decision gates

The capacity-aware claim passes only if all of the following pass:

1. **Validity:** every arm is finite; D source-center maximum is `<=1e-8 px`, source covariance
   relative-Frobenius maximum is `<=1e-8`, all depths are strictly bounded, moment-split first and
   second moments rederive within `1e-10`, and oracle parent purity/completeness are each `>=0.99`.
2. **Absolute mechanism:** D parent purity and completeness are each `>=0.90` in all three roots,
   mean outlier dust recall is `>=0.80`, and mean inlier dust false-positive rate is `<=0.20`.
3. **Soft-assignment gain:** D beats A by at least `0.15` mean parent purity, beats B by at least
   `0.05`, and wins both paired comparisons in all three roots.
4. **Capacity attribution:** D beats C by at least `0.03` mean parent purity or reduces the
   per-parent coefficient of variation of transported mass by at least 20%, with the same
   direction in at least two roots. If only C passes, conclude that two-sided assignment helped but
   footprint capacity was not established.
5. **Negative control:** D exceeds S by at least `0.15` mean parent purity and S fails either the
   `0.90` purity or completeness floor in every root.

Any failed gate makes the overall synthetic result FAIL. Secondary geometry cannot rescue an
association failure. No threshold or schedule is tuned after official outcomes.

## Calibrated compact-bundle interaction

### Frozen input and roles

Strict-load
`runs/compact_masked_bundle_640_20260717/reconstruction_inputs` on CPU. It is a recovered,
qualified exploratory bundle: seven 5328x4608 StructSplat fields, 640 components per view,
`geometry:null`, no RGB/mask/source-path fields, and no bounds hint. Its original acquisition
lifecycle remains a documented FAIL; this experiment may establish interaction with the recovered
payload, not erase that provenance.

- Development/fitting only: `C0001,C0008,C0014,C0021,C0026`.
- Validation only: `C0031,C0039`.
- Report only after all model and appearance state is frozen: `C1004` RGB/mask/calibration.

The dataset loader's default every-eighth split is forbidden here because it would place C1004 in
training. Roles are selected by exact camera ID.

### Bounds, hypotheses, and arms

- Compute the existing compact-Carve camera-axis least-squares center and extent from the five
  development cameras only. Use the cube `center +/- 0.5*extent`; intersect unnormalized world
  rays whose scalar equals camera-z depth. Record `camera_axis_fallback`, center, extent, and a
  semantic digest before the fit worker starts. Validation/report masks or images may not define
  the box.
- Select exactly 128 source anchors per development view (640 total). Divide each fitted window
  into an 8x8 grid, choose up to two observations per cell by descending footprint area with stable
  component-index ties, then fill any shortfall by global descending area. Every one of the 640
  target observations per view remains available for matching.
- Initialize each valid fiber at the midpoint of its ray interval; invalid or empty intervals are
  a protocol failure, not silently dropped.
- Run C and D with frozen synthetic hyperparameters. If the synthetic validity gate fails, the real
  run is withheld. If only one synthetic transport arm is scientifically accepted, it remains the
  primary real arm, but both C and D are retained as paired diagnostics. Validation never selects a
  hyperparameter or checkpoint.

Dense per-view plans are processed sequentially. The preregistered scope is 640 hypotheses, not
the full 3,200 development components. Full-scale, sparse epipolar candidates, track birth/death,
coherence, and iterative split/merge are explicitly unresolved scale-up questions.

### Geometry and ray-constraint diagnostics

Before validation access, freeze geometry and save initial/final PLY and NPZ state. Report:

- exact source mean/covariance residual and depth bounds;
- development and validation symmetric Gaussian-set Bhattacharyya cost;
- real/dustbin mass, conditional entropy, and number of non-source views with row-real fraction
  `>=0.20` per track;
- effective supported-track count and proximity-cluster count at a fixed `0.01*extent` radius;
- wall time, peak RSS, and primitive count;
- per-camera association-weighted residual components parallel and perpendicular to the local
  epipolar direction. A systematic mean perpendicular residual exceeding `0.25` pooled target
  standard deviations is flagged as camera/model mismatch. It does not authorize per-Gaussian
  lateral freedom; the proposed follow-up is shared per-camera pose/intrinsic correction.

Real geometry feasibility passes only if source center is `<=1e-4 px`, source covariance relative
error `<=1e-6`, all states are finite/bounded, mean validation set cost improves by at least 5%
from the common midpoint state, neither validation view regresses by more than 10%, and at least
50% of tracks have support in two non-source development views. These gates establish only that
the constrained fitter interacts coherently with this payload.

### Exploratory source-anchored appearance phase

Geometry and transports are frozen before appearance. Query each exact normalized compact field at
the frozen projected track center. A degree-one SH module is parameterized as a particular solution
plus the null space of the source-view SH row, so preactivation at the source direction equals the
source field query exactly. Optimize only null-space coefficients against non-source development
queries, weighted by detached real support. Opacity remains an independent fixed `0.1`; compact
amplitudes are not copied or learned as alpha.

Report source equality, development/validation query-color RMSE, negative-preactivation incidence,
and C1004 masked Torch-render PSNR after the model is frozen. Raw compact component coefficients
are not physical isolated colors; the normalized field query is the image-space target. Appearance
is exploratory and cannot rescue a failed geometry gate or establish a renderer-equivalence,
opacity, SH-recovery, or pipeline-quality claim.

### Evidence and handoff

The result must include exact configuration, environment, roots, input hashes, per-arm arrays,
model/plan hashes, initial/final PLYs, timing/resources, and a machine-readable statement of every
gate. Fitting must run without opening source RGB or masks; C1004 release is separately receipted
after final model hashes. Launch and smoke-test:

```bash
.venv/bin/rtgs view \
  --gaussians runs/inverse_projection_fiber_iter3_real_20260717/gaussians.ply \
  --initial runs/inverse_projection_fiber_iter3_real_20260717/gaussians_init.ply \
  --scene dataset/2025_03_07_stage_with_fabric/frame_00008 \
  --downscale 16 --device cpu --rasterizer torch --no-open
```

The viewer is qualitative. Exact Torch snapshots and metrics, not WebGL appearance, decide the
reported interaction. An independent `realtime-gs-results-audit` pass is required before any
result enters the research or roadmap conclusions.

## Claim boundary

Even a PASS supports only: exact source-fiber preservation, improved synthetic many-to-many
association under this generator, and bounded calibrated interaction on 640 hypotheses. It does
not establish true real correspondences, full-scene coverage, a production default, full 3,200
scaling, occlusion correctness, opacity semantics, physical component color, Stage-3 quality,
pose robustness, GPU speed, or superiority to standard 3DGS/MVS. A FAIL is logged without tuning
and used to identify whether the remaining blocker is cost, capacity, source coverage, calibration,
visibility, or topology.
