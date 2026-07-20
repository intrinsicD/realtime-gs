# Claims

## C01: Validity-aware depth derivatives prevent invalid-boundary scale explosion
- **Statement**: On depth maps where zero/NaN denotes invalid background, validity-aware finite
  differences materially reduce pathological surface covariance scale/conditioning relative to
  central differences across invalid samples.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: Controlled replicated scenes with invalid depth boundaries show no
  conditioning/quality improvement, or the validity-aware implementation worsens finite valid
  surface derivatives.
- **Proof**: [N04, N05, `benchmarks/results/20260714T195655Z_cpu_depth_covariance_iter2_raw.json`, `benchmarks/results/20260714T195727Z_cpu_depth_covariance_iter2_robust.json`]
- **Dependencies**: []
- **Tags**: depth, covariance, numerical-stability, initialization
- **From staging**: O01

## C02: Synthetic covariance-mode ranking is protocol-dependent
- **Statement**: Across the tested three-seed synthetic protocols, footprint variance does not
  beat a train-selected global isotropic sigma by the predeclared 0.25 dB initialization margin,
  and the leading mode changes between robust initialization and merge+density refinement.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: Reproduction of the same configs shows a stable mode exceeding the
  margin across initialization, clean refinement, perturbed depth, and merge+density recovery.
- **Proof**: [N04, N05, N06, N08, `docs/EXPERIMENTS.md`]
- **Dependencies**: [C01]
- **Tags**: ablation, depth, covariance, synthetic, held-out
- **From staging**: O04

## C03: Exact synthetic confidence weighting is not a material robust anchor improvement
- **Statement**: Under the frozen three-seed exact-attribution protocol, sampled confidence does
  not meet the preregistered materiality criteria versus valid-prior-uniform anchoring, despite a
  small seed-consistent held-out depth-RMSE improvement.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A source-exact replay of the frozen protocol passes both the 2%
  held-out depth-RMSE and 15% corrupted-source p90 gates with at least two of three seed wins and
  satisfies the exact-shuffle attribution gate.
- **Proof**: [N24, N25, `benchmarks/results/20260715T052539Z_cpu_depth_anchor_attribution.json`, `ara/evidence/tables/depth_anchor_attribution.md`]
- **Dependencies**: []
- **Tags**: depth, confidence, bounded-ray, attribution, synthetic, negative-result
- **From staging**: O25

## C04: Synthetic own-source exclusion is not a material geometry improvement
- **Statement**: Under the frozen three-seed synthetic Gradient and corrupted-depth Hybrid
  protocols, excluding target-own fitted splats from the photometric training render does not meet
  the preregistered material and robust geometry criteria versus inclusive supervision.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A source-exact replay of the frozen protocol passes every relevant
  held-out RMSE and source-tail effect floor with at least two of three seed wins while preserving
  the PSNR, coverage, and IoU guards.
- **Proof**: [N27, N28, `benchmarks/results/20260715T062601Z_cpu_cross_view_supervision.json`, `ara/evidence/tables/cross_view_supervision.md`]
- **Dependencies**: []
- **Tags**: photometric-supervision, source-provenance, bounded-ray, attribution, synthetic, negative-result
- **From staging**: O27

## C05: Sparse fixed-position consistency localizes matched nodes but misses global utility
- **Statement**: Under the frozen three-seed synthetic Gradient and corrupted-depth Hybrid
  protocols, the correct fixed-pair position loss materially localizes represented ray-bounded
  primitives but does not meet the preregistered whole-scene held-out and source-tail utility gates.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A source-exact replay fails either local engagement/localization gate,
  or passes every applicable global materiality gate with the saved raw seeds and decision rules.
- **Proof**: [N32, N33, `benchmarks/results/20260715T084557Z_cpu_world_position_consistency.json`, `ara/evidence/tables/world_position_consistency.md`]
- **Dependencies**: []
- **Tags**: correspondence, position-consistency, bounded-ray, synthetic, coverage, mixed-result
- **From staging**: O30

## C06: Raw patch/epipolar matching broadens coverage but fails strict semantic validity
- **Statement**: Under the frozen three-seed synthetic protocol, the deterministic train-only raw
  patch/epipolar matcher passes its structural density floors but fails the preregistered strict
  semantic precision floor in every seed, so its graph is not a valid input to the position loss.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A source-exact replay of the frozen matcher and strict compositor
  audit reaches at least 60% precision in every seed or fails the reported structural floors.
- **Proof**: [N37, N38, `benchmarks/results/20260715T094311Z_cpu_dense_train_position.json`, `ara/evidence/tables/dense_train_position.md`]
- **Dependencies**: [C05]
- **Tags**: correspondence, matching, epipolar, synthetic, coverage, precision, negative-result
- **From staging**: O34

## C07: Structurally plausible corrupted-depth plane targets fail clean validity
- **Statement**: Under the frozen three-seed synthetic protocol, four-neighbor cross-view PCA
  targets built from corrupted metric training depth pass every structural floor but fail the
  preregistered clean point-to-plane validity gate in every seed.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A source-exact replay fails a reported structural floor, or reaches
  all-target and corrupted-target clean plane p90 at most 0.10 of extent in every seed.
- **Proof**: [N42, N43, `benchmarks/results/20260715T110342Z_cpu_surface_plane_normal.json`, `ara/evidence/tables/surface_plane_normal.md`]
- **Dependencies**: []
- **Tags**: oriented-points, planes, normals, depth-corruption, synthetic, validity-gate, negative-result
- **From staging**: O38

## C08: Registered RGB-D local normals fail confirmatory cross-view tail transfer
- **Statement**: Under the frozen TUM `fr1/xyz` development to `fr1/desk` confirmatory protocol,
  registered metric depth and deterministic five-point normals provide broad eligible and
  two-view-supported target populations but fail the transferred surface-p90, relative-depth-p90,
  and p10-normal-cosine validity gates.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A source-exact replay of the sealed implementation and artifacts
  passes all nine mechanically transferred gates, or fails the reported eligibility/support
  populations or decision reconstruction.
- **Proof**: [N47, N48, `benchmarks/results/20260715T143959Z_cpu_tum_rgbd_oriented_validity_xyz.json`, `benchmarks/results/20260715T144052Z_cpu_tum_rgbd_oriented_validity_desk.json`, `ara/evidence/tables/tum_rgbd_oriented_validity.md`]
- **Dependencies**: [C07]
- **Tags**: oriented-points, rgb-d, normals, cross-view, real-data, validity-gate, negative-result
- **From staging**: O41

## C09: Dense T-only visibility provides partial but sub-threshold occlusion attribution
- **Statement**: Under the frozen TUM `fr3/sitting_xyz` protocol, a stride-8 T-only construction
  z-buffer preferentially removes behind-observed residuals and improves target depth p90, but
  fails the preregistered target-balanced positive-reduction and risk-ratio magnitude floors.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A source-exact reconstruction fails the reported nesting/support or
  sign-selectivity invariants, or passes both the frozen 15% relative positive-reduction and 2x
  target-balanced risk-ratio gates.
- **Proof**: [N52, N53, `benchmarks/results/20260715T160300Z_cpu_tum_rgbd_signed_attribution_sitting.json`, `ara/evidence/tables/tum_rgbd_signed_attribution.md`]
- **Dependencies**: [C08]
- **Tags**: rgb-d, visibility, occlusion, signed-residual, real-data, partial-result
- **From staging**: O45

## C10: Dense-visible sitting contradictions increase with capture-time separation
- **Statement**: In the frozen `fr3/sitting_xyz` audit after dense T-only visibility,
  target-balanced far-minus-near contradiction increases 11.19 percentage points, and the
  preregistered pose-conditioned sensitivity remains positive at 10.01 points.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: A source-exact reconstruction gives a nonpositive target-cluster
  bootstrap lower bound or a nonpositive estimable four-cell pose-conditioned effect.
- **Proof**: [N52, N55, `benchmarks/results/20260715T160300Z_cpu_tum_rgbd_signed_attribution_sitting.json`, `ara/evidence/tables/tum_rgbd_signed_attribution.md`]
- **Dependencies**: [C09]
- **Tags**: rgb-d, temporal, rigidity, visibility, pose-sensitivity, real-data
- **From staging**: O47

## C11: The hard SH color floor is not a material bottleneck in the frozen CPU protocol
- **Statement**: In the frozen three-seed synthetic fixed-topology audit, negative spherical-
  harmonic color preactivations and their recoverable blocked-gradient mass were too rare to pass
  any materiality gate; pooled negative incidence was 0.336527%, recoverable mass was 0.090828%,
  and fixed SMU-1 recovered mass was 0.025266%, so no candidate trained.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A source-exact replay passes the frozen per-seed or pooled incidence
  and recoverable-gradient gates, or an independently preregistered domain shows material blocked
  mass and utility without selecting SMU parameters from these outcomes.
- **Proof**: [N60, N61, `benchmarks/results/20260715T192112Z_cpu_sh_activation_iter2_audit.json`, `benchmarks/results/20260715T192112Z_cpu_sh_activation_iter2_audit_AUDIT.md`, `ara/evidence/tables/smooth_support_audits.md`]
- **Dependencies**: []
- **Tags**: spherical-harmonics, activation, SMU, gradient-gate, synthetic, negative-result
- **From staging**: O53

## C12: A material kernel-tail mechanism does not imply held-out utility
- **Statement**: The frozen CPU q=[12,16) support annulus passed its local loss-gradient screen,
  but the fixed C=12, W=4 C1 taper and hard-forward gradient attribution control changed diffuse
  common-hard foreground PSNR by -0.014483 dB and -0.018470 dB on average, respectively, with
  zero of three seed wins for either arm.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A source-exact replay fails the mechanism screen or shows either
  frozen candidate passing the preregistered held-out utility rule and safety guardrails.
- **Proof**: [N62, N63, `benchmarks/results/20260715T202218Z_cpu_kernel_support_taper_iter2_audit.json`, `benchmarks/results/20260715T202917Z_cpu_kernel_support_taper_iter2_ablation.json`, `benchmarks/results/20260715T202917Z_cpu_kernel_support_taper_iter2_ablation_AUDIT.md`, `ara/evidence/tables/smooth_support_audits.md`]
- **Dependencies**: []
- **Tags**: rasterization, kernel-support, smooth-tail, attribution, synthetic, negative-result
- **From staging**: O54

## C13: Three-sigma visibility truncation is immaterial in the frozen CPU protocol
- **Statement**: In the frozen three-seed synthetic fixed-topology audit, the detached 3-sigma
  image cull omitted four of 2,480,463 genuine q<12 Gaussian-pixel pairs, with effective-mass
  fraction 1.646359e-8 and render-delta/residual 3.986964e-8; every materiality gate failed and
  support-safe training was forbidden.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A source-exact replay fails the reported ordering/target/support
  invariants or passes the frozen materiality rule, or an independently preregistered workload
  demonstrates material missed q<12 support without selecting the margin from this result.
- **Proof**: [N64, N65, `benchmarks/results/20260715T213132Z_cpu_visibility_margin_iter2_audit.json`, `benchmarks/results/20260715T213132Z_cpu_visibility_margin_iter2_audit_AUDIT.md`, `ara/evidence/tables/smooth_support_audits.md`]
- **Dependencies**: []
- **Tags**: rasterization, visibility, culling, support, synthetic, negative-result
- **From staging**: O55

## C14: Production-scale Carve grouping is insufficient for the frozen equal-count utility test
- **Statement**: In the frozen three-seed CPU synthetic Carve audit, production voxel grouping
  compressed raw primitives by only 2.34%-2.68%; multi-member cells, exposed mass, and compression
  each failed every seed's materiality gates, so equal-count moment-versus-prune refinement was
  withheld and merge utility remains untested.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A source-exact replay fails construction/parity or passes the frozen
  materiality rule, or an independently preregistered regime produces material collisions without
  selecting its grid scale from these outcomes.
- **Proof**: [N71, N72, `benchmarks/results/20260715T232244Z_cpu_carve_merge_controls_iter2_audit.json`, `benchmarks/results/20260715T232244Z_cpu_carve_merge_controls_iter2_audit_AUDIT.md`, `ara/evidence/tables/20260716_stage1_carve_multiscale_quaternion.md`]
- **Dependencies**: []
- **Tags**: carve, merging, materiality, synthetic, stopped-before-utility
- **From staging**: O61

## C15: Current downstream Stage-1 semantics depend on a non-identifiable weight/color gauge
- **Statement**: In the frozen three-seed CPU synthetic contract audit, all 54 product-preserving
  Stage-1 source renders remained equivalent while both tested representatives materially changed
  coverage, retention, unmerged Depth, and unmerged Carve outputs in every seed and the raw-sum
  pool. The independently qualified evidence establishes an interface problem but selects no
  canonical representative and demonstrates no held-out utility.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A source-exact tensor replay violates source-render equivalence or
  fails the frozen per-seed/pool materiality decisions, or a corrected invariant downstream
  boundary shows that the reported dependence came from an artifact error.
- **Proof**: [N76, `benchmarks/results/20260716T003140Z_cpu_stage1_weight_gauge_audit.json`, `benchmarks/results/20260716T003140Z_cpu_stage1_weight_gauge_audit_AUDIT.md`, `ara/evidence/tables/20260716_stage1_carve_multiscale_quaternion.md`]
- **Dependencies**: []
- **Tags**: stage-1, representation, gauge, lifter-interface, qualified-evidence
- **From staging**: O62

## C16: The exact 24-to-48 fixed-topology schedule fails quality and exposure efficiency
- **Statement**: In the frozen three-seed CPU synthetic comparison, camera-blocked,
  loss-pyramid-blocked, and exposure-matched camera-interleaved refinement lost foreground-PSNR
  AUC and final PSNR in every seed; the camera arms used 62.5% of full raster-pixel exposure but
  failed quality noninferiority, and blocked ordering failed attribution.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A source-exact replay passes any frozen quality/exposure/attribution
  decision, or an independently preregistered different schedule succeeds without tuning scale or
  boundaries from this result.
- **Proof**: [N74, N75, `benchmarks/results/20260716T003735Z_cpu_multiscale_refinement.json`, `benchmarks/results/20260716T003735Z_cpu_multiscale_refinement_AUDIT.md`, `ara/evidence/tables/20260716_stage1_carve_multiscale_quaternion.md`]
- **Dependencies**: []
- **Tags**: multiscale, fixed-topology, coarse-to-fine, synthetic, negative-result
- **From staging**: O63

## C17: Removing the local null coordinate did not improve the bounded Stage-1 fit
- **Statement**: In the frozen deterministic CPU-synthetic, 150-component, 120-update N78 scope,
  current 9p Adam had a 0.122921 pooled null-energy ratio, yet the bounded unit-weight 8p arm lost
  every appearance-only and joint seed: mean final-PSNR deltas were -1.796120 dB and -1.501525 dB,
  respectively, and joint non-inferiority failed. Local projected motion is therefore not
  sufficient evidence that removing the redundant coordinate improves this fit.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: The bound raw/source replay fails, any frozen metric or decision is
  recomputed differently, or the exact preregistered arm passes its appearance or joint gate.
  Success of a separately preregistered parameterization or variable-projection solve does not
  falsify this scoped statement.
- **Proof**: [N90, N91, `benchmarks/results/20260716T101608Z_cpu_stage1_fit_parameterization_AUDIT.md`, `benchmarks/results/20260716T101608Z_cpu_stage1_fit_parameterization_SCIENTIST_REVIEW.json`, `ara/evidence/tables/20260716_stage1_fit_parameterization.md`]
- **Dependencies**: []
- **Tags**: stage-1, parameterization, gauge, adam, null-motion, synthetic, negative-result
- **From staging**: O72

## C18: Sparse CPU point rendering matches the frozen dense-reference scope
- **Statement**: On the complete frozen synthetic pixel grids, `TorchPointRasterizer` matches the
  dense CPU renderer's color, alpha, depth, global visible order, five 3D parameter-gradient
  families, and retained screen-space gradients within the sealed tolerances; on the authorized
  calibrated interaction it matches 4,096 frozen C0001 pixel-center replacement draws from the
  existing 835-Gaussian PLY.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: A bound source/fixture replay exceeds the frozen absolute/relative
  gates, changes visible order, admits proposal/lineage filtering, or fails a required empty/global
  compositor invariant.
- **Proof**: [`benchmarks/results/20260716_point_rasterizer_parity_RESULT.json`, `runs/point_rasterizer_parity_20260716/calibrated_parity.json`, `benchmarks/results/20260716_point_rasterizer_parity_AUDIT.md`, N116]
- **Dependencies**: []
- **Tags**: point-rasterizer, parity, sparse-rendering, cpu, compact-supervision
- **From staging**: O95
- **Boundary**: This does not establish full-image calibrated parity, active nonzero off-grid
  coordinate gradients, compact optimization, quality, memory, speed, density control, or
  CUDA/gsplat parity.

## C19: Fixed-attempt discrete Gaussian proposals preserve uniform pixel risk
- **Statement**: On the exact finite-pixel fixture, the implemented uniform/Gaussian rejection
  mixture with recorded proposal probability and rejected null attempts retained in the fixed
  denominator reproduces the uniform discrete-pixel expectation and remains invariant to attempt
  microchunking within the frozen numerical tolerance.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: Exact enumeration differs from target risk, proposal branch
  accounting fails, a rejection is resampled/dropped from the denominator, the importance formula
  differs from target/proposal probability, or microchunking changes the estimator beyond `2e-12`.
- **Proof**: [`benchmarks/results/20260716_point_rasterizer_parity_RESULT.json`, `benchmarks/results/20260716_point_rasterizer_parity_AUDIT.md`, `tests/test_observation2d.py`, N116]
- **Dependencies**: []
- **Tags**: importance-sampling, discrete-pixels, compact-teacher, unbiased-risk
- **From staging**: O96
- **Boundary**: This is an estimator-identity result; proposal variance, convergence, quality, and
  scaling on real compact teachers remain unmeasured.

## C20: Gaussian teacher proposals do not improve the frozen fixed-topology protocol materially
- **Statement**: In the sealed three-seed CPU synthetic fixed-topology comparison, the discrete-
  pixel Gaussian mixture lost normalized log-AUC direction in all three seeds and had geometric-
  mean `G_AUC=1.0245665262`, while the continuous-area Gaussian mixture had
  `G_AUC=0.9910818462` and favorable direction in all three seeds but missed the preregistered
  `0.95` materiality floor; the global decision is `NO_GLOBAL_SAMPLING_WIN`.
- **Status**: supported
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: Bound RAW recomputation changes either aggregate, seed direction,
  gate classification, or the global decision, or a source-exact replay violates the frozen
  topology, attempt, risk-domain, RNG, teacher, optimizer-clock, or RGB-access invariants.
- **Proof**: [`benchmarks/results/20260716_compact_point_training_RAW.json`, `benchmarks/results/20260716_compact_point_training_RESULT.json`, `benchmarks/results/20260716_compact_point_training_AUDIT.md`, N119]
- **Dependencies**: [C19]
- **Tags**: compact-teacher, importance-sampling, fixed-topology, cpu, synthetic, negative-result
- **From staging**: O98
- **Boundary**: This does not compare pixel and area arms as the same risk, authorize a proposal
  default, or answer calibrated, density-enabled, quality, speed, memory, or CUDA behavior.

## C21: Residual responsibility may improve matched one-wave birth allocation
- **Statement**: Under the frozen iter2 protocol, native front-to-back compositing responsibility
  weighted by compact-teacher residual may allocate a matched 32-birth budget more usefully than
  native gradient ranking or a within-stratum uniform shuffle because it attributes underfit to
  contributing parents.
- **Status**: untested
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: The source-bound official iter2 result fails its preregistered
  distinctness or responsibility-attribution mechanism gate, fails its held-out utility rule, or
  violates any matched-count, stratum, replay, source, runtime, RGB-denial, or topology invariant.
- **Proof**: [N79, N130, N131, `benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_PREREG.md`, `benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_IMPLEMENTATION_REVIEW.md`, `benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_FAILURE_AUDIT.md`]
- **Dependencies**: []
- **Tags**: compact-teacher, density-control, responsibility, gradient-ranking, matched-births
- **From staging**: O66
- **Boundary**: The implementation review and focused structural CUDA smoke establish readiness
  only. The official iter2 seal command later failed before publication and permanently consumed
  that lifecycle without an official root, score, comparison, utility metric, or scientific
  decision. Testing this hypothesis now requires a fresh preregistered successor namespace.

## C22: Raw-fragment exact fibers plus capacity-aware transport do not pass the synthetic release gate
- **Statement**: In the completed root of the consumed final inverse-projection-fiber experiment,
  exact source equalities and numerically valid UOT did not make independently split 2D fragments
  reliable latent 3D tracks: UOT-area reached `0.5468` purity, `0.25` completeness, and only
  `0.2730/0.0560` track/observation outlier recall, while UOT-uniform also failed every applicable
  per-root acceptance floor. The calibrated interaction is therefore withheld.
- **Status**: supported
- **Provenance**: ai-executed
- **Crystallized via**: empirical-resolution
- **Falsification criteria**: Independent recomputation of the bound root-0 NPZ changes any
  reported acceptance operand or shows either transport arm meets all frozen per-root floors.
  Success of a separately preregistered stable-track or moment-aggregate construction does not
  falsify this raw-fragment-scoped claim.
- **Proof**: [`benchmarks/results/20260717_inverse_projection_fiber_iter3_SYNTHETIC_ATTEMPT.json`,
  `runs/inverse_projection_fiber_iter3_synthetic_20260717/root_0/evidence.npz`,
  `benchmarks/results/20260717_inverse_projection_fiber_iter3_FAILURE_AUDIT.json`,
  `benchmarks/results/20260717_inverse_projection_fiber_iter3_FAILURE_AUDIT.md`]
- **Dependencies**: []
- **Tags**: inverse-projection-fiber, correspondence, unbalanced-transport, topology, synthetic,
  negative-result, failed-execution
- **From staging**: O115
- **Boundary**: The official transaction is consumed and incomplete: only root 0 completed, no
  top-level RESULT exists, and three-root means, capacity attribution, real-data behavior,
  appearance, quality, GPU behavior, and performance remain unresolved. Root 0 alone rejects both
  all-root real-release candidates, so omitted roots cannot restore that release.

## C23: Two EWA views leave one general covariance coordinate unobservable
- **Statement**: For a 3D Gaussian with a triangulated mean and ordinary EWA covariance projection
  `C_v = A_v S A_v^T`, the stacked design from two generic calibrated views has rank five, not six.
  If `d_v` is view v's world ray, `d_1 d_2^T + d_2 d_1^T` is a nonzero shared null mode. Three
  generic views can raise the design to rank six and isolate the covariance.
- **Status**: supported
- **Provenance**: ai-executed
- **Crystallized via**: artifact-commitment
- **Falsification criteria**: A valid derivation or source-exact CPU case shows generic rank six for
  two ordinary EWA views, or shows the stated shared mode has nonzero projection through either
  view while both Jacobians annihilate their own ray directions.
- **Proof**: [N138, `src/rtgs/lift/inverse_projection_fiber.py`,
  `tests/test_inverse_projection_fiber.py::test_covariance_projection_design_has_rank_five_then_six`,
  `benchmarks/results/20260717_inverse_projection_fiber_iter1_PREREG.md`]
- **Dependencies**: []
- **Tags**: inverse-projection-fiber, covariance, observability, ewa, tomography, rank
- **From staging**: O119
- **Boundary**: This claim concerns the Jacobian-based EWA covariance used by this repository after
  fixing the mean. It does not cover additional physical amplitude/thickness measurements, exact
  nonlinear finite-support projection, or priors that supply the missing scalar.
