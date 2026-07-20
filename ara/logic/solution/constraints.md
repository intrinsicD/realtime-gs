# Constraints

## R01: Merge-free primary covariance comparison
- **Constraint**: Disable voxel merging for the causal covariance-only comparison because merge
  weights depend on Gaussian volume and can otherwise change means, color, and opacity.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`benchmarks/depth_covariance_ablation.py`]
- **Evidence**: [N04, N05, N06]
- **From staging**: O03

## R02: Permute confidence after retained-ray sampling
- **Constraint**: A spatial confidence negative control must permute sampled weights among valid
  retained rays within each source view, preserving the exact multiset and denominator. Pixel-space
  shuffling before bilinear sampling changes the optimized-ray weight distribution.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`benchmarks/depth_anchor_ablation.py` (protocol to repair)]
- **Evidence**: [N21, `benchmarks/results/20260715_depth_anchor_AUDIT.md`]
- **From staging**: O24
- **Implemented by**: [`src/rtgs/lift/gradient.py`, `benchmarks/depth_anchor_attribution.py`]
- **Validated by**: [N24, `benchmarks/results/20260715T052539Z_cpu_depth_anchor_attribution.json`]

## R03: Scope the confidence-anchor stopping rule
- **Constraint**: Stop synthetic bounded-ray confidence loss, lambda, threshold, and weighting
  sweeps under the frozen protocol, without generalizing the negative result to train-derived
  observation uncertainty or direct cross-view consistency mechanisms.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`benchmarks/depth_anchor_attribution.py`, `docs/ROADMAP.md`]
- **Evidence**: [N24, N25, N26, `benchmarks/results/20260715_depth_anchor_attribution_RESULT.md`]
- **From staging**: O26

## R04: Globally balance matched source-exclusion exposure
- **Constraint**: A matched non-self dropout control for LOSO must match each target's removed
  primitive count and frozen scalar-opacity sum, exclude no target-own primitive, and exclude every
  primitive exactly once across the target schedule. This controls count, opacity, and global
  exposure, but not group topology, visibility, projected alpha, color, or spatial coverage.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`src/rtgs/lift/gradient.py`, `benchmarks/cross_view_supervision_ablation.py`]
- **Evidence**: [N27, `benchmarks/results/20260715_cross_view_supervision_PREREG.md`, `benchmarks/results/20260715T062601Z_cpu_cross_view_supervision.json`]
- **From staging**: O28

## R05: Scope and stop the sparse position-loss branch
- **Constraint**: The sparse-oracle outcome authorizes no production default, deployability claim,
  or coefficient/delta/norm/schedule sweep. Test exactly one denser train-only matcher with the same
  loss; if local engagement still does not propagate globally, pivot to plane/normal consistency.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`benchmarks/world_position_consistency_ablation.py`, `docs/ROADMAP.md`]
- **Evidence**: [N32, N33, N34, N35, `benchmarks/results/20260715_world_position_consistency_RESULT.md`]
- **From staging**: O31

## R06: Scope degree-matched shuffled-topology attribution
- **Constraint**: Matching edge count, endpoint degree, per-block endpoints, camera-pair counts, and
  baselines does not match closest-ray feasibility or initial residual/gradient magnitude. Claims
  from this control must be phrased as correct topology versus this structural derangement.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`benchmarks/world_position_consistency_ablation.py`]
- **Evidence**: [N32, `ara/evidence/tables/world_position_consistency.md`, `benchmarks/results/20260715_world_position_consistency_RESULT.md`]
- **From staging**: O33

## R07: Stop the rejected raw-patch position branch without threshold tuning
- **Constraint**: A failed strict matcher-precision gate withholds the position-optimization arms
  and authorizes no patch, ratio, epipolar, reprojection, confidence, coefficient, delta, norm, or
  schedule sweep. Scope the rejection to this raw-patch backend and pivot to depth-backed local
  plane/normal consistency rather than generalizing to learned matching.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`benchmarks/dense_train_position_ablation.py`, `docs/ROADMAP.md`]
- **Evidence**: [N37, N38, N39, N40, `benchmarks/results/20260715_dense_train_position_RESULT.md`]
- **From staging**: O35

## R08: Count unlabeled matcher endpoints as semantic failures
- **Constraint**: A dense fitted-primitive matcher audit must count low-contribution or unlabeled
  endpoints as false, not omit them from precision. Conditional labeled-pair precision can be a
  secondary diagnostic but cannot rescue a failed strict graph-validity gate.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`benchmarks/dense_train_position_ablation.py`]
- **Evidence**: [N37, N38, `ara/evidence/tables/dense_train_position.md`]
- **From staging**: O37

## R09: Withhold surface-loss arms after target-validity failure
- **Constraint**: A failed post-freeze clean oriented-point gate withholds every plane/normal
  optimization arm and authorizes no target-builder, audit-threshold, loss-weight, or schedule
  tuning on the revealed seeds. Scope rejection to the four-neighbor corrupted-depth constructor;
  later evidence requires an independent calibrated metric-depth/RGB-D oriented-point source.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`benchmarks/surface_plane_normal_ablation.py`, `docs/ROADMAP.md`]
- **Evidence**: [N42, N43, N44, N45, `benchmarks/results/20260715_surface_plane_normal_RESULT.md`]
- **From staging**: O39

## R10: Withhold oriented utility after failed real-RGB-D transfer
- **Constraint**: The failed one-shot `fr1/desk` surface/depth/normal tail gates withhold every
  Phase-B plane/normal arm and authorize no desk threshold relaxation, view deletion, V-depth
  residual filtering, grid densification, or TUM-backend promotion. A revisit requires new
  development/confirmatory sequences and a separately preregistered attribution protocol.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`benchmarks/tum_rgbd_oriented_validity.py`, `docs/ROADMAP.md`]
- **Evidence**: [N47, N48, N49, N50, `benchmarks/results/20260715_tum_rgbd_oriented_validity_RESULT.md`]
- **From staging**: O43

## R11: Compare same-pixel sensor-plane loss with an ordinary depth anchor
- **Constraint**: On a bounded ray `mu=o+t*d`, a same-pixel registered point/normal plane produces
  `|n^T(mu-p)|=|n^T d| |t-t_star|`; when the source camera/timestamp matches, `t_star` is sensor
  depth. Any later utility claim must call this incidence-weighted sensor-depth regularization and
  include an ordinary extra-depth anchor control rather than treating it as unrestricted 3D plane
  pulling.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`src/rtgs/lift/surface.py`, `benchmarks/tum_rgbd_oriented_validity.py`]
- **Evidence**: [N46, N49, N50, `benchmarks/results/20260715_tum_rgbd_oriented_validity_PREREG.md`]
- **From staging**: O44

## R12: Stop signed attribution before the reserved walking confirmation
- **Constraint**: A failed `fr3/sitting_xyz` development occlusion gate forbids decoding or
  inspecting the reserved `fr3/walking_xyz` archive under this protocol, creating a confirmation
  attempt seal, tuning visibility density/tolerance/effect floors, or running any plane/normal or
  ordinary-depth utility arm.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`benchmarks/tum_rgbd_signed_attribution.py`]
- **Evidence**: [N52, N53, N54, `benchmarks/results/20260715_tum_rgbd_signed_attribution_DECISION.json`, `benchmarks/results/20260715_tum_rgbd_signed_attribution_RESULT.md`]
- **From staging**: O46

## R13: Preserve established compositing order when expanding visibility
- **Constraint**: A support-expansion audit must preserve the baseline relative compositing order,
  including exact-depth ties, and insert only newly visible primitives according to the renderer's
  depth rule. Otherwise measured render differences confound visibility-set expansion with
  reordering. Any invariant failure stops before an outcome-bearing arm.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`src/rtgs/render/torch_ref.py`, `benchmarks/visibility_margin_ablation.py`]
- **Evidence**: [N64, `benchmarks/results/20260715_visibility_margin_PHASE_A_ATTEMPT.json`, `benchmarks/results/20260715_visibility_margin_iter2_PREREG.md`]
- **From staging**: O56

## R14: Prove native-precision representation feasibility before optimizer arms
- **Constraint**: A representation or gauge intervention must pass a prospectively defined
  feasibility contract at the intervention's native dtype before outcome-bearing optimizer arms
  run. An algebraically exact equivalence does not justify float64-exact thresholds after a
  non-idempotent float32 transform; any precision margin or representation change must be fixed
  analytically or with outcome-neutral probes, never relaxed from hidden arm behavior.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`benchmarks/quaternion_gauge_ablation.py`, `tests/test_quaternion_gauge_ablation.py`]
- **Evidence**: [N81, N82, `benchmarks/results/20260716T030759Z_cpu_quaternion_gauge_iter2_invalid_AUDIT.md`]
- **From staging**: O64

## R15: Treat implementation completeness as necessary but not sufficient evidence
- **Constraint**: A claim-admitting sealed fit experiment must reject malformed/non-finite raw
  evidence, bind exact decision populations and metric inputs, preserve truthful failure prefixes,
  bind CLI/seal/runtime source and environment state, and support independent raw recomputation.
  A boolean implementation-complete flag alone cannot authorize a scientific result.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`benchmarks/stage1_fit_parameterization.py`, `tests/test_stage1_fit_parameterization.py`]
- **Evidence**: [N88, N90, `benchmarks/results/20260716_stage1_fit_parameterization_IMPLEMENTATION_REVIEW.md`, `benchmarks/results/20260716T101608Z_cpu_stage1_fit_parameterization_AUDIT.md`]
- **From staging**: O71

## R16: Isolate source RGB from compact refinement
- **Constraint**: Once compact StructSplat observations exist, the refinement data model must not
  contain source-RGB tensors or paths and must not use original RGB for optimization or model
  selection. Preserve originals in the dataset and expose selected references only to a separate
  post-training evaluator/viewer that loads one view at a time after configurations are frozen.
- **Provenance**: user
- **Crystallized via**: verbal-affirmation
- **Evidence**: [N99, N103, N104]
- **From staging**: O80
- **Implementation boundary**: The current `SceneData`/`Trainer` contract eagerly carries images,
  so compact training requires a distinct observation-only scene interface. N106 confirms that
  this boundary starts immediately after Stage 1: current Carve reads `scene.images` and must gain
  an exact compact-field query path before the new branch can claim end-to-end compliance.

## R17: Preserve captured compact-field semantics instead of converted initialization semantics
- **Constraint**: Compact supervision must consume the losslessly recorded StructSplat field
  parameters and blend/support/filter/viewport semantics. Converted `Gaussians2D` archives clamp
  colors and omit normalized epsilon, finite support/fade, affine-color, and producer-specific
  filtering semantics, so they remain initialization-only and cannot silently become teachers.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`src/rtgs/core/observation2d.py`, `src/rtgs/image2gs/structsplat_backend.py`, `src/rtgs/data/reconstruction_inputs.py`]
- **Evidence**: [N108, N113, `tests/test_structsplat_observation.py`, `tests/test_observation2d.py`]
- **Scope**: Current parity evidence covers the captured arrays/semantics and complete CPU fixture
  pixel grids. It does not establish CUDA or arbitrary-continuous-coordinate StructSplat parity,
  replay-complete producer provenance, performance, or reconstruction quality.
- **From staging**: O86

## R18: Do not repair a consumed calibrated lifecycle with post-failure evidence
- **Constraint**: A terminal once-only calibrated result remains failed even if later diagnostics
  render the same PLYs or a future harness fixes the failure mechanism. Any success claim requires
  a fresh preregistered namespace whose actual spawned bound worker completes gsplat/CUDA snapshots
  and the launched-process-owned HTTP viewer smoke.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`benchmarks/compact_point_training.py`, `tests/test_compact_point_training.py`]
- **Evidence**: [N120, N121, `runs/compact_point_training_20260716/CALIBRATED_FAILURE_AUDIT.md`, `runs/compact_point_training_20260716/postfailure_abi_diagnostic.json`, `runs/compact_point_training_20260716/postfailure_viewer_diagnostic.json`]
- **From staging**: O101

## R19: Treat compact working-set hardening as partial scaling evidence only
- **Constraint**: Preflight must precede working-set/init transfer to the configured device,
  compact optimization must omit unused global geometry from its device-tensor working set, and
  non-CPU overlap counting must use bounded chunks rather than copying whole teachers. These fixes
  do not authorize production-scale claims while all teachers remain resident, CUDA queries are
  unindexed, eager index state remains, or backward saved state scales with the outer microbatch
  times visible Gaussians.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`src/rtgs/optim/compact_trainer.py`, `tests/test_compact_trainer.py`]
- **Evidence**: [N118, N121, `docs/ARCHITECTURE.md`, `docs/RESEARCH.md`]
- **From staging**: O102

## R20: Do not treat a rank-K association tensor as raw-fragment identity
- **Constraint**: Pairwise or multi-view factorization is bookkeeping only after the latent
  membership model is defined. Unequal simultaneous fragments, occlusion/compositing, unknown
  track count, and non-conserved fitted mass require visibility, group/mass semantics, and explicit
  birth/death/merge topology; a near-one-hot rank-K tensor alone cannot make independently fitted
  fragments physical tracks.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`src/rtgs/lift/field_loss.py`, `src/rtgs/lift/field_lifter.py`,
  `src/rtgs/lift/field_topology.py`, `docs/DESIGN_field_lift.md`]
- **Evidence**: [N135, N136, N145, N146]
- **From staging**: O120

## R21: Exclude unverified optional geometry from held-out field fitting
- **Constraint**: `SceneFits` must declare a complete disjoint train/held-out partition. When
  held-out views exist, optional sparse points and bounds may enter placement only with explicit
  train-only provenance; otherwise fitting derives its search region from training cameras while
  held-out compact observations remain reporting/correspondence inputs.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`src/rtgs/data/field_inputs.py`, `src/rtgs/lift/field_lifter.py`,
  `tests/test_field_inputs.py`, `tests/test_field_lifter.py`]
- **Evidence**: [N94, N104, N145, N146]
- **From staging**: O126
- **Boundary**: This closes the native `SceneFits` field path only. The generic
  `ReconstructionInputs.from_scene` provenance concern in O91 remains open.

## R22: Separate additive analytic optimization from frozen StructSplat semantics
- **Constraint**: Closed-form product-kernel discrepancies may be called exact only for additive
  whole-plane peak-Gaussian density and RGB numerator. Normalized blending, finite support/fade,
  epsilon, and affine teacher color must be evaluated through the immutable teacher query equation
  in a separate bounded validator; proxy movement is not a normalized-renderer quality result.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`src/rtgs/lift/field_loss.py`, `src/rtgs/lift/field_validation.py`,
  `src/rtgs/lift/field_lifter.py`, `tests/test_field_loss.py`, `tests/test_field_validation.py`]
- **Evidence**: [N113, N145, N146, `docs/DESIGN_field_lift.md`]
- **From staging**: O127

## R23: Qualify local performance diagnostics before portability claims
- **Constraint**: A performance number is portable or causal only when source state, dependency
  versions, device/host state, warmup, repetitions, and comparable inputs are bound. Dirty-worktree
  quick receipts and single-run mixed CPU/GPU profiles may support mechanism diagnosis and parity
  checks but must remain explicitly local diagnostics.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Code ref**: [`benchmarks/results/20260720T213644Z_cpu_AUDIT.md`,
  `benchmarks/results/20260720_dense_confidence_gated_init_e1_RESULT.md`,
  `benchmarks/results/20260720_dense_confidence_gated_init_e2_RESULT.md`]
- **Evidence**: [N156, N157, `ara/evidence/tables/20260720_dense_confidence_gated_init.md`]
- **From staging**: O131
