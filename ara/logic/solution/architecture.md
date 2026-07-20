# Architecture Decisions

## A01: Pluggable bounded-ray anchor semantics
- **Design**: `GradientLifter` selects legacy, normalized, continuous-confidence, or thresholded
  anchor semantics through one mode while `HybridLifter` forwards the same option. The legacy mode
  remains the default, so the pipeline and backend interfaces require no fork.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [N19, N22, `docs/ARCHITECTURE.md`]
- **Code ref**: [`src/rtgs/lift/gradient.py`, `src/rtgs/lift/hybrid.py`]
- **From staging**: O15
- **Attribution controls**: `valid_uniform` and `confidence_shuffled` preserve validity/layout and
  isolate sampled-weight placement without changing the `legacy` default.

## A02: Source-aware photometric supervision controls
- **Design**: `GradientLifter` selects inclusive `all`, `leave_one_source_out`, or globally balanced
  `matched_nonself_dropout` for training renders while leaving the full final output and other loss
  terms unchanged. `HybridLifter` forwards the same option; `all` preserves the production default.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [N27, N29, `docs/ARCHITECTURE.md`, `ara/evidence/tables/cross_view_supervision.md`]
- **Code ref**: [`src/rtgs/lift/gradient.py`, `src/rtgs/lift/hybrid.py`]
- **From staging**: O29

## A03: Explicit fixed-pair position-consistency research API
- **Design**: `GradientLifter.lift_with_position_pairs` accepts a detached validated cross-source
  `(E,2)` tensor and optionally adds extent-normalized Huber-after-L1 world-position consistency.
  `HybridLifter` predicts aligned priors and forwards the same pairs. The zero coefficient default
  preserves legacy outputs, histories, and random schedules.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [N32, N34, `docs/ARCHITECTURE.md`, `ara/evidence/tables/world_position_consistency.md`]
- **Code ref**: [`src/rtgs/lift/gradient.py`, `src/rtgs/lift/hybrid.py`, `tests/test_lift.py`]
- **From staging**: O32

## A04: Train-only position matcher boundary
- **Design**: `PositionMatcher` accepts only train RGB, calibrated cameras, and a detached retained
  center layout, and returns a fixed positive pair graph plus an exact-degree cyclic control. The
  deterministic `PatchEpipolarMatcher` is a rejected CPU research reference; the boundary can host
  a future optional learned backend without exposing GT/depth/held-out state or changing lifters.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [N37, N39, `docs/ARCHITECTURE.md`, `ara/evidence/tables/dense_train_position.md`]
- **Code ref**: [`src/rtgs/lift/matching.py`, `tests/test_matching.py`, `benchmarks/dense_train_position_ablation.py`]
- **From staging**: O36

## A05: Explicit retained-indexed oriented-point research API
- **Design**: `OrientedPointTargets` binds detached retained indices to fixed world points and
  separate plane/alignment normals. `GradientLifter` optionally adds extent-normalized absolute
  point-to-plane and sign-invariant selected-axis normal losses, freezing the selected minimum-
  scale axis from step zero; `HybridLifter` forwards the same targets. Both coefficients default
  to zero, preserving legacy outputs and schedules.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [N42, N44, `docs/ARCHITECTURE.md`, `ara/evidence/tables/surface_plane_normal.md`]
- **Code ref**: [`src/rtgs/lift/surface.py`, `src/rtgs/lift/gradient.py`, `src/rtgs/lift/hybrid.py`, `tests/test_surface.py`, `tests/test_lift.py`]
- **From staging**: O40

## A06: View-keyed oriented-point prediction and canonicalization boundary
- **Design**: `OrientedPointBackend` returns immutable-provenance geometry and normals with explicit
  geometry kind, normal frame, validity, and optional confidence for one stable view ID.
  Canonicalization validates ownership/content and emits detached world-space maps; deterministic
  registered-depth normal estimation is reusable, while the failed TUM backend remains isolated in
  its benchmark rather than entering production backend selection.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [N47, N49, `docs/ARCHITECTURE.md`, `ara/evidence/tables/tum_rgbd_oriented_validity.md`]
- **Code ref**: [`src/rtgs/lift/surface.py`, `tests/test_surface.py`, `benchmarks/tum_rgbd_oriented_validity.py`, `tests/test_tum_rgbd_oriented_validity.py`]
- **From staging**: O42

## A07: Sealed nested-visibility attribution harness
- **Design**: `tum_rgbd_signed_attribution.py` binds source and implementation hashes before PNG
  decode, reuses the sealed oriented target mechanics, constructs nested sparse and dense T-only
  z-buffers without accepting V depth, labels signed residuals afterward, and writes an append-only
  development decision that fail-closes walking confirmation. It remains benchmark-only and does
  not alter production lifters.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [N52, N54, `benchmarks/results/20260715_tum_rgbd_signed_attribution_PREDECODE_SEAL.json`, `ara/evidence/tables/tum_rgbd_signed_attribution.md`]
- **Code ref**: [`benchmarks/tum_rgbd_signed_attribution.py`, `tests/test_tum_rgbd_signed_attribution.py`]
- **From staging**: O48

## A08: Explicit Stage-1 appearance-to-lifter semantic boundary
- **Design**: Stage 1 uses additive, order-independent `weight*color` accumulation, whereas the
  current lifters separately consume scalar weight for coverage/retention/opacity and color for
  appearance before Stage 3 switches to depth-sorted alpha compositing. The interface must treat
  scalar amplitude and observed color as explicit semantics rather than assume an arbitrary
  factorization is identifiable. The production fit default remains unchanged; N78 now has an
  opt-in fit-time seam but no scientific result, and the N77 downstream semantic factorial remains
  open.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [N76, C15, `docs/ARCHITECTURE.md`, `ara/evidence/tables/20260716_stage1_carve_multiscale_quaternion.md`]
- **Code ref**: [`src/rtgs/image2gs/renderer2d.py`, `src/rtgs/image2gs/fit.py`, `src/rtgs/lift/base.py`, `src/rtgs/lift/depth.py`, `src/rtgs/lift/carve.py`]
- **From staging**: O58
- **Boundary**: The qualified audit authorizes neither a canonical representative nor a default
  change; the semantic factorial and fit-time parameterization remain open experiments.
- **Outcome update (2026-07-16)**: N90 supersedes only the earlier open-status wording for the
  fit-time branch. The bounded 8p arm failed appearance and joint gates; the semantic boundary and
  no-default conclusion remain.

## A09: Default-preserving Stage-1 fit-time appearance seam
- **Design**: Native Stage 1 keeps `weight_color_9p` as its bit-exact default and permits an opt-in
  `unit_weight_bounded_8p` coordinate map initialized from the same additive RGB amplitude. A
  shared-initialization entry point, disabled geometry freeze, and detached diagnostic snapshots
  expose paired raw parameters, gradients, optimizer state, schedule, target, and renders without
  allowing callbacks to mutate the fit or CPU RNG state. StructSplat rejects the candidate before
  importing its optional backend.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [N78, N84, `docs/ARCHITECTURE.md`, `docs/EXPERIMENTS.md`]
- **Code ref**: [`src/rtgs/image2gs/fit.py`, `tests/test_stage1_fit_seam.py`, `benchmarks/stage1_fit_parameterization.py`, `tests/test_stage1_fit_parameterization.py`]
- **From staging**: O68
- **Boundary**: The production seam passed an outcome-blind contract audit, but the confirmatory
  harness remains `IMPLEMENTATION_COMPLETE=False`. No seal, attempt, official seed, quality result,
  conditioning result, efficiency claim, or default authorization exists.
- **Outcome update (2026-07-16)**: N90 completed the hardened harness, seal, once-only run, and
  independent raw/scientist audit. The bounded 8p arm was negative and failed joint
  non-inferiority, so this seam remains opt-in research infrastructure and `weight_color_9p`
  remains the production default.

## A10: GPS-inspired compact teacher sampling boundary
- **Design**: Frozen per-view StructSplat fields provide native-coordinate training samples without
  materializing dense RGB. A seeded proposal mixes weighted in-image Gaussian-mass samples with
  uniform foreground, boundary, and background strata. The first arm evaluates the exact
  normalized StructSplat color at each coordinate; an optional lower-cost arm carries the sampled
  component's RGB and uses MSE, whose conditional expected gradient targets the normalized field.
  The 3D student remains an exact differentiable, depth-sorted sparse alpha compositor evaluated at
  the same coordinates. Fixed topology establishes the supervision mechanism before standard
  split/clone/prune density control is enabled.
- **Provenance**: ai-suggested
- **Crystallized via**: verbal-affirmation
- **Evidence**: [N99, N101, N102, `https://jorisar.nl/gaussian_point_splatting/gaussian_point_splatting.pdf`]
- **From staging**: O79
- **Boundary**: GPS's opacity correction, Poisson counts, pixel rounding, and stochastic atomic
  visibility are not transferred because they neither match normalized StructSplat amplitude
  semantics nor provide the refinement gradients required by Stage 3. N104 prohibits source-RGB
  access during refinement; N105 tests lift-time lineage as an optional soft prior without making
  it the compositing semantics. Under N107/N108, Gaussian mass is a proposal rather than the loss
  definition: recorded propensities or fixed strata must prevent component fragmentation from
  silently changing the image-space objective.

## A11: Independent compact-observation and 3D-population cardinalities
- **Design**: Treat the full converged per-view 2D teacher count `N2D`, explicit initialization
  budget `B_init`, dynamic refined count `N3D(t)`, and ray/patch batch `K` as independent. A seeded
  selection policy may lift all teacher components or a spatially balanced coreset, but unselected
  components remain in the immutable teacher field. Subsequent global sparse-ray refinement may
  split, clone, or prune 3D splats without changing the teacher or loss definition. Optional
  lineage is a sparse weighted sidecar/DAG over selected source observations and descendants, not
  a dense correspondence matrix or correctness dependency.
- **Provenance**: user
- **Crystallized via**: verbal-affirmation
- **Evidence**: [N99, N105, N107, N108, N109]
- **From staging**: O83, O87
- **Scaling boundary**: Teacher lookup and student rendering must be indexed by local support and
  streamed by view. Count sweeps must separately vary `N2D`, `B_init`, `N3D(t)`, and `K`; quality
  or compute changes from one axis cannot be attributed to another.
- **Canonical stage notation**: For view `i`, Stage 1 maps `N_init,i^2D` to
  `N_opt,i^2D`; RGB-free lifting maps the full teacher collection, optionally through a selected
  initialization coreset, to `N_init^3D`; sampled global refinement maps that model to the
  independently determined `N_opt^3D` viewer population. Omit `i` only for an explicitly shared
  2D count.
- **Implementation update (2026-07-16)**: N113 implements the frozen teacher/data contract and
  N114 implements standalone compact initialization; N116 supplies the audited CPU sparse point
  renderer and discrete-risk prerequisite. Sampled global refinement, dynamic `N_opt^3D`,
  calibrated quality evaluation, and production integration remain open.

## A12: Source-proposed, all-view-scored compact Carve initializer
- **Design**: `CompactCarveInitializer` consumes `ReconstructionInputs`, draws a fixed
  view-balanced candidate pool from full compact-field mass, searches bounded source rays, and
  scores each depth with coverage-weighted queries to every teacher. It returns the requested
  `N_init^3D` on sufficient support or fails closed. The parent view/component determines the ray
  and initial lateral covariance, but does not select a teacher target or rendering subset;
  lineage is retained as a sidecar for later controlled experiments.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [N106, N107, N108, N112, N113, N114, `docs/EXPERIMENTS.md`]
- **Code ref**: [`src/rtgs/lift/compact_carve.py`, `src/rtgs/lift/base.py`, `tests/test_compact_carve.py`]
- **Invariant boundary**: An exact co-located split into identical components whose amplitudes sum
  to the original preserves the tested scores and initialized geometry. Non-coincident or
  otherwise non-identical fragmentation is not covered.
- **Scaling boundary**: Point batches and reference-backend point-component temporaries are capped.
  Tile-index state remains proportional to component-tile overlaps, custom backends must honor the
  chunk contract, and runtime/peak RSS/full-resolution/CUDA behavior are unmeasured.
- **From staging**: O90

## A13: Separate sparse point-rendering and compact-training boundary
- **Design**: Keep dense image rasterization behind `Rasterizer` and selected-coordinate 3D
  evaluation behind the separate `PointRasterizer`. A sparse renderer forms one camera-global
  visible set and depth order, composites every visible Gaussian at every requested coordinate,
  streams point/Gaussian chunks, and exposes retained screen-space gradients without accepting
  teacher proposal or lineage ids. Compact teacher sampling is a separate concern with explicit
  continuous-area or discrete-pixel risk semantics and fixed-attempt importance correction.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [N102, N105, N108, N115, N116, `docs/ARCHITECTURE.md`]
- **Code ref**: [`src/rtgs/render/point_base.py`, `src/rtgs/render/torch_points.py`, `src/rtgs/core/observation2d.py`, `tests/test_point_render.py`, `tests/test_point_rasterizer_parity.py`]
- **Implementation boundary**: The CPU point renderer and discrete proposal are implemented and
  independently audited. `CompactTrainer`, fixed-topology optimization, topology control,
  CLI/pipeline integration, and a sparse CUDA/gsplat backend do not yet exist.
- **From staging**: O92
- **Outcome update (2026-07-17)**: N119/N120 now implement the separate fixed-topology
  `CompactTrainer` over `ReconstructionInputs` and `PointRasterizer`; it queries each compact
  teacher independently, globally composites all eligible 3D Gaussians, and exposes four explicit
  proposal/risk modes without RGB or lineage supervision. N121 preflights before working-set/init
  transfer and constructs a teacher/camera-only device-tensor working set. Dynamic topology,
  sparse indexed CUDA teacher/student backends, and CLI/pipeline integration remain absent.
- **Additional code ref**: [`src/rtgs/optim/compact_trainer.py`, `tests/test_compact_trainer.py`, `benchmarks/compact_point_training.py`]
- **Additional staging**: O100

## A14: Integrity-bound compact dataset views with optional exact alpha
- **Design**: Each checked-in view is one strict `.rtgsv` container holding the fitted
  `GaussianObservationField`, calibrated camera, source and preprocessing provenance, and optional
  lossless bit-packed crop alpha. The complete container—not only its teacher payload—is capped at
  168,000 decimal bytes. A frame-level manifest binds view order, hashes, sizes, alpha presence,
  calibration, and a bounds hint; strict loaders reject unsafe members, schema drift, digest
  mismatch, and cap violations without importing StructSplat or Pillow.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [N140, N141, N143, N144]
- **Code ref**: [`src/rtgs/data/compact_views.py`, `scripts/convert_datasets_to_gaussians2d.py`, `tests/test_compact_views.py`, `tests/test_convert_datasets_to_gaussians2d.py`]
- **From staging**: O124
- **Dataset boundary**: Exact alpha is present only for the 52 views whose source frames supplied
  authoritative masks; the 62 unmasked views omit it. Original raster files are no longer a
  checked-in fallback, so future 3D fitting must consume compact observations and cameras.

## A15: Decomposition-invariant field lifting with explicit topology
- **Design**: `FieldLifter` consumes `SceneFits`, places a bounded subset of source components on
  exact inverse-projection fibers, and continuously refits projected additive density/RGB-numerator
  fields without assigning reference components to tracks. Detached center transmittance and
  per-view gains modulate fitting; covariance observability pins underdetermined free-column
  coordinates. An immutable scheduler proposes prune, representative-fiber merge, split, and
  residual birth moves and accepts only strict additive-proxy-plus-parsimony decreases. Dense
  visibility-gated product-kernel correspondences are emitted after fitting as bookkeeping.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [N136, N138, N145, N146, `docs/DESIGN_field_lift.md`]
- **Code ref**: [`src/rtgs/data/field_inputs.py`, `src/rtgs/lift/field_lifter.py`,
  `src/rtgs/lift/field_refit.py`, `src/rtgs/lift/field_topology.py`,
  `src/rtgs/lift/field_visibility.py`, `tests/test_field_lifter.py`]
- **From staging**: O125
- **Boundary**: The path is registered and exposed through `run_field_pipeline` and
  `rtgs lift-field`, but remains research-only. Unit tests and bounded compact-input smokes establish
  mechanism/integration behavior only; calibrated reconstruction quality, topology utility,
  runtime/memory scaling, CUDA behavior, and default status remain open.
