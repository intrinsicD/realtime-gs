# Heuristics

## H01: Three-pass falsification loop
- **Rationale**: Predeclared controls prevent post-hoc ranking; evidence-driven prompt revisions
  isolate mechanisms; replication and a final perturbation/production interaction distinguish a
  narrow correctness fix from a protocol-specific winner.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: medium
- **Code ref**: [`docs/RESEARCH_LOOP.md`, `benchmarks/depth_covariance_ablation.py`]
- **From staging**: O14

## H02: Gate smooth-boundary candidates on material incidence before training
- **Rationale**: A mathematical dead zone is not automatically an optimization bottleneck. Measure
  affected incidence and loss-directed recoverable mass before training a smooth substitute; if
  the local mechanism passes, compare a smooth-forward arm with a hard-forward gradient-only
  attribution control. Audit detached culling separately from the continuous kernel because a
  smooth kernel cannot restore candidates removed before evaluation.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`benchmarks/sh_activation_ablation.py`, `benchmarks/kernel_support_taper_ablation.py`, `benchmarks/visibility_margin_ablation.py`]
- **From staging**: O52

## H03: Keep global compositing normative and anneal lineage assistance
- **Rationale**: A teacher component's lifted descendant is construction provenance rather than a
  permanent physical identity: normalized 2D bases overlap, Carve merges many sources, and 3D
  density control creates and removes descendants. Train and evaluate through the global
  depth-sorted compositor; test parent-only rendering only as a mechanism control, and permit
  lineage to influence early credit assignment only as a weak annealed prior that reaches zero
  before unrestricted density control.
- **Provenance**: ai-suggested
- **Crystallized via**: verbal-affirmation
- **Sensitivity**: high
- **Evidence**: [N105, N107]
- **From staging**: O82

## H04: Solve oracle aggregate geometry before learned association
- **Rationale**: Moment-merging known child membership isolates decomposition from geometry.
  Triangulate aggregate centers first, subtract declared EWA dilation, report the weighted
  covariance design's rank/condition/null basis, and solve PSD-constrained covariance least
  squares. Use at least three generic views for an unregularized solve; a two-view arm must expose
  its one-dimensional null coordinate and fill it only from an explicit prior. This baseline
  separates topology, observability, and optimizer failures but cannot infer topology itself.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: high
- **Code ref**: [`src/rtgs/lift/field_measurement.py`,
  `src/rtgs/lift/field_observability.py`, `tests/test_field_measurement.py`,
  `tests/test_field_observability.py`]
- **Evidence**: [N138, N139, N146]
- **From staging**: O121

## H05: Cache exact compact evaluation targets across candidates
- **Rationale**: Compact teachers and support masks are invariant when candidates share
  `ReconstructionInputs`. Render each teacher once, crop the calibrated camera to the exact scored
  fit window, and reuse the target tensors while only rerendering candidate 3D Gaussians. Keep
  progress opt-in so silent library calls perform no progress timing or record allocation.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: medium
- **Code ref**: [`src/rtgs/lift/compact_init_eval.py`, `src/rtgs/render/torch_ref.py`,
  `benchmarks/compact_init_eval.py`, `tests/test_compact_init_eval.py`, `tests/test_render.py`]
- **Evidence**: [N156, `ara/evidence/tables/20260720_dense_confidence_gated_init.md`]
- **From staging**: O130
- **Boundary**: Prepared targets retain about 652.5 MB on this seven-view bundle. The CPU Torch
  renderer remains the semantic anchor; gsplat replay is a fast diagnostic with measured small
  numerical drift, not bit-exact backend equivalence.

## H06: Flatten exact compact queries before adding hierarchy or CUDA
- **Rationale**: Replace Python-fragmented tile candidate evaluation with one canonical flattened
  CSR pair stream, bounded vectorized evaluation, and deterministic reductions. This preserves
  candidate identity while removing interpreter overhead; only a post-CSR profile should justify
  hierarchy or CUDA complexity.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Sensitivity**: medium
- **Code ref**: [`src/rtgs/core/observation2d.py`, `src/rtgs/lift/compact_carve.py`,
  `benchmarks/run.py`, `tests/test_observation_csr.py`]
- **Evidence**: [N150, N157, `benchmarks/results/20260720T123859Z_cpu.json`,
  `benchmarks/results/20260720T213644Z_cpu_AUDIT.md`]
- **From staging**: O128
- **Boundary**: The latest 26.6x quick ratio is not a causal current-worktree claim. Full
  26-view/130,000-component production placement timing remains separately open.
