# Stage-1 weight/color gauge implementation review

Verdict: PASS

## Scope and disposition

This is an independent, outcome-free implementation review against the complete frozen
preregistration. I did not run the seal command, the official `audit` action, an official fit, or
any official seed. I did not inspect or create a scientific outcome. The review authorizes only
sealing the reviewed implementation after the repository-wide verification gate passes; it does
not authorize a default change or any scientific claim.

No blocking or major finding remains. The implementation is additive benchmark/test code and
does not alter a production default.

## Bound review snapshot

- Preregistration: `benchmarks/results/20260716_stage1_weight_gauge_PREREG.md`, SHA-256
  `ec2bdaea7362649392da915af2d44e7aa47a8a1825546f8487f6afa3067b9489`.
- Harness: `benchmarks/stage1_weight_gauge_audit.py`, SHA-256
  `86dc68315e3a7f6bae9099edf2af1d2fbb7608e67e5e059478ecd0b746f0d1b1`.
- Focused tests: `tests/test_stage1_weight_gauge_audit.py`, SHA-256
  `d3d1d9a94f26d9bf60264cc9b3bc129d1acc1b348047a1d6a8fdda530976a196`.
- Direct contract sources reviewed: `src/rtgs/core/gaussians2d.py`
  (`6cbd61b1c4c39fcf0abd376d19d14b4130402e3952f1805973a02a85a64ac6df`),
  `src/rtgs/core/gaussians3d.py`
  (`d417a4a103ae7ea1e3f4a7799c2b709597014b8966acb0e72b2bd447a0ad0ba5`),
  `src/rtgs/data/scene.py`
  (`3fa557f03bab5eb7666476968e0a70ff3e5639d6e24251807905691df36004c3`),
  `src/rtgs/data/synthetic.py`
  (`b2b16f02a92c89003439062085e39d1f5ced2cc9ebaf5b8874cf80c0fd4d70b2`),
  `src/rtgs/image2gs/fit.py`
  (`2a9b76d41e83cc444fa98b3a0f3aa45eb8b6032806fa3d899377acfd98257e18`),
  `src/rtgs/image2gs/renderer2d.py`
  (`3228530ff5be088e1598ab4a220597b9ade45c466b8452425951ee72fe9f0523`),
  `src/rtgs/lift/base.py`
  (`b19fa04733c42c1bb5c210e1ac2fced73b7fcbfe7b0d7521ec62fd1d68ba503d`),
  `src/rtgs/lift/depth.py`
  (`19e2e59d8c8a32d1dcc1b86b79364c09f845c493956cb12ea94597fadd874021`),
  `src/rtgs/lift/carve.py`
  (`35135d6c93de3a836c9f9843fdfc63e3c08d973a080a9743372d3ee057a829eb`),
  `src/rtgs/depth/mock.py`
  (`fcc9ce42dc0f73ff387144d3f0ff614eb0ac4cbcef5a28fca881159ee9f97a1e`),
  `src/rtgs/render/torch_ref.py`
  (`61716787329e85a186982f81c2a89cb270255473ca26688c409191a1b53bd86e`), and
  `src/rtgs/core/sh.py`
  (`554f3a25e25c7312248a98c15685e9bf805c85a81a96f56e13e1481619eb4687`).

The future seal, rather than this note, remains authoritative for the complete all-source and
all-test path set. The harness recomputes that set, verifies every digest, and refuses drift.

## Requirement trace

### Chronology, split isolation, and preparation

- The preregistration digest is a hard-coded protocol check. The CLI exposes only the
  outcome-free `seal` action and the once-only `audit` action.
- Each full twelve-view synthetic scene is constructed once, physically subset to the exact nine
  training indices, and then discarded. Subsequent fitting, transforms, renders, and lifters see
  only the subset. The lifter-facing `SceneData` explicitly sets `gt_gaussians=None`, retains only
  the permitted world priors, and pins the training-only center/full-diameter extent in the scene
  cache. No held-out view is subsequently addressed.
- Input, camera, depth, optional mask, point, point-visibility, bounds, center/extent, fit-field,
  history, ordering, view-map, and aggregate hashes are formed before gauge construction. The
  native fit is called once per seed and the fitted shape, finiteness, range, center bounds, and
  positive Cholesky diagonals are checked explicitly.
- All three seeds are prepared and transformed before the first source render. All 54 transformed
  source-view equivalence checks finish before coverage, retention, or either lifter can run.
  Source-equivalence failure returns an invalid payload with no downstream decision or statistic.

### Gauges, equivalence, and raw reductions

- `identity`, `unit_weight`, and `peak_color` implement the frozen formulas in source order.
  `xy/chol` are bit-identical, zero-amplitude behavior is exact, transformed fields remain bounded,
  and amplitude absolute/relative tolerances are checked before rendering.
- Source renders use the specified additive renderer, black background, no clamp, no gradient, and
  row chunk 64. Finiteness, maximum absolute error, float64 L1 numerator/denominator, and floored-MSE
  PSNR are checked exactly. Render hashes and raw values are exposed only after the global gate.
- Coverage uses the unchanged density definition and strict `>0.40` crossing rule. Retention uses
  only strict `weight>0.05`. Per-view, seed-tagged per-seed, and pooled key sets and float64 raw sums
  are retained. Materiality helpers implement the frozen conjunctive thresholds without arm or
  seed selection, and pooling requires the same named transform in two of three seeds plus the raw
  pool.

### Independent Depth reconstruction

- Every gauge gets one fresh `GroundTruthDepth` backend and one unmerged production lift with the
  exact frozen configuration.
- The side reconstruction independently rebuilds finite-depth, `z>0.05`, strict-weight, optional
  mask, and confidence masks, emits immutable `(seed, local_view, original_component)` keys in
  production order, and checks count plus means/covariance/opacity/SH parity.
- Shared identity/transform keys enforce the preregistered means/covariance/opacity geometry
  control. SH changes remain diagnostic. Raw color, alpha, and accumulated-depth render sums use
  the frozen hard Torch renderer and are pooled before ratios.

### Carve sidecar and parity

- Every gauge receives exactly one ordinary unmerged `CarveLifter.lift` call. The sidecar copies
  the current bounds, coverage, seen/covered counts, color moments, hull, consistency, ray tunnel,
  score, placement, depth-variance, and ray-sigma expressions without invoking a second lifter.
- Every source key has the required nullable record. Emitted keys are the view-ordered,
  ascending-original-index `keep_indices[placed]` sequence. Coverage maps, all named volume
  tensors, source records, and outputs are hashed.
- Production/sidecar ordered count, means, covariance, opacity, and SH parity is required for all
  three gauges before any comparison; quaternion sign is correctly excluded. Shared-key tunnel
  score and selected-depth deltas, geometry/appearance deltas, and all nine frozen renders are
  serialized.
- In addition to the checked-in masked toy, I ran an in-memory no-mask toy probe to exercise the
  production Gaussian coverage branch. It emitted three primitives and returned exact zero parity
  error for means, covariance, opacity, and SH.

### Seal, marker, routing, and safety

- Seal creation runs exactly the five frozen verification subprocesses in order, captures literal
  commands, exit codes, complete stdout/stderr and their hashes, and refuses any failure or source
  change. It prepares no official seed or outcome.
- The seal covers the preregistration, harness, `pyproject.toml`, every repository Python source,
  and every Python test. Runtime checks both the full sealed set and every loaded repository source
  before the atomic attempt marker and checks them again after execution.
- The official path is validated, the invalid sibling is derived by the one frozen terminal suffix
  replacement, both JSON possibilities and both notes are preflighted, and the marker binds all
  four paths before scientific work. Exclusive creation prevents overwrite. Valid and invalid
  routing are mutually exclusive.
- Empty Depth/Carve outputs and parity failures produce no backend decision. The final decision
  explicitly has no Phase B and never authorizes a default change.

## Commands and evidence

```text
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m pytest -q tests/test_stage1_weight_gauge_audit.py
...........                                                              [100%]
11 passed

.venv/bin/python -m ruff check \
  benchmarks/stage1_weight_gauge_audit.py tests/test_stage1_weight_gauge_audit.py
All checks passed!

.venv/bin/python -m ruff format --check \
  benchmarks/stage1_weight_gauge_audit.py tests/test_stage1_weight_gauge_audit.py
2 files already formatted

env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -  # in-memory, three-camera, no-mask Carve sidecar parity probe
{'output_count': 3, 'parity': {'means': 0.0, 'covariance': 0.0,
 'opacity': 0.0, 'sh': 0.0}}
```

Repository-wide verification is intentionally left to the preregistered seal command. This review
did not consume the official attempt.
