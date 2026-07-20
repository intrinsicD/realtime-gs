# Implementation review: compact footprint occupancy scalar ablation

Date: 2026-07-17T02:10:11+02:00 (Europe/Berlin)

Verdict: **PASS — implementation and outcome-free protocol are ready; execution remains gated on
explicit root authorization.**

## Bound artifacts

- Preregistration:
  `benchmarks/results/20260717_compact_occupancy_scalar_PREREG.md`, SHA-256
  `cccb15c35d83854c74893a471a3a1d5803339a2608fea739acce4fd6ec878293`.
- Harness: `benchmarks/compact_occupancy_scalar_ablation.py`, SHA-256
  `3be876d49885b5008baaf3afa42843da884e5ae144f05d3d2974b10b65f510a4`.
- Focused tests: `tests/test_compact_occupancy_scalar_ablation.py`, SHA-256
  `8b5f8918fabdcfe7a5e2c59fdaba1dec4228c03af64e5f45f0721bde3a376c89`.
- Deterministic compact-Carve source: `src/rtgs/lift/compact_carve.py`, SHA-256
  `810fd03f3ab057756ad5d730a93b7ee5b204956003b0fd31602612ff7373edc1`.
- Compact-Carve tests: `tests/test_compact_carve.py`, SHA-256
  `9f7c666044b3b10625717e93bc4bc7223c80d76aca2693e28019109ec710ebc6`.
- Reused helper harness: `benchmarks/compact_masked_lift_screen.py`, SHA-256
  `90b27af700ffe572bbafe4efbf93aca03d169b9a1b52725fbffd92b8b5443bf0`.
- Runtime transitive-source aggregate: SHA-256
  `02f910f711dacd7328e76528cac357abd6fed5598f0b3d792bc7440b35157f40`.

No official output directory exists and no Stage-A or Stage-B official metric was queried while
performing this review.

## Root-review findings and dispositions

1. **CPU center nondeterminism — resolved in core and isolated in protocol.** The compact-Carve
   helper uses explicit `gelsd` rank detection, explicit `gels` for full-rank systems, and the
   `gelsd` minimum-norm solution for deficient systems. Four CPU tests cover repeated full-rank,
   rank-deficient, near-degenerate, and explicit-driver behavior. PyTorch intra/inter-op threads
   are additionally pinned to one before plan creation. The camera-only center and extent are
   evaluated once, bound in `plan.json`, and injected as `bounds_hint` into one Stage-B input
   clone. All lift arms and failure diagnostics consume that clone. A protocol test repeats
   `_center_and_extent` on the clone and requires exact equality, then source-checks that every
   Stage-B call uses the clone.
2. **Outcome-free chronology — resolved.** The once-only output directory receives `plan.json`
   before `load_masks`, scalar construction, proxy construction, indexing, or metric queries.
   Tensor/proxy hashes observed after the seal go to `observed_metadata.json`. The selector accepts
   only the tuning metric object and is serialized before report metrics are queried.
3. **Scalar math and selection tests — resolved.** Six focused tests cover deterministic
   antithetic samples and their frozen hash; LSE constant endpoints, bounds, and invalid inputs;
   precision-guard rejection and deterministic tie-breaking; selector lack of a report-metric
   argument; seal/report source ordering; and injected center/extent reuse.
4. **Center replay wording/gate — resolved.** Exact proxy tensors and query semantics must replay
   the prior center proxy bit-for-bit. Cross-run PLY bit equality is not claimed. Means/covariance
   differences are recorded and compared only with the predeclared `1e-5`/`5e-3` geometric
   tolerances informed by the separate determinism audit. Scientific contrasts remain within one
   pinned-thread process.
5. **Mass-normalization interpretation — resolved.** The plan explicitly states that per-view
   `r=A*D/M` cancels a common positive scalar multiplier and therefore tests relative component
   occupancy distribution, not scalar magnitude.
6. **Memory scaling — resolved.** Proxy indices/backends are constructed, queried, and released
   one scalar variant at a time. Stage B likewise retains only one proxy index set per arm; all
   seven variants are never indexed simultaneously.
7. **Split claim — resolved.** The report set is described only as metric-not-passed-to-selector.
   The preregistration discloses the earlier all-view footprint-sizing diagnostic and that Stage B
   consumes all seven fields; no held-out novel-view claim is permitted.
8. **RGB boundary — resolved.** Stage B color comes from exact masked compact fields. Source-image
   RGB opens are denied across Stage A and Stage B; masks supply occupancy/evaluation only.

## Checks executed

```text
python -m ruff format --check src/rtgs/lift/compact_carve.py \
  tests/test_compact_carve.py benchmarks/compact_occupancy_scalar_ablation.py \
  tests/test_compact_occupancy_scalar_ablation.py
=> 4 files already formatted

python -m ruff check src/rtgs/lift/compact_carve.py tests/test_compact_carve.py \
  benchmarks/compact_occupancy_scalar_ablation.py \
  tests/test_compact_occupancy_scalar_ablation.py
=> All checks passed

PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_compact_carve.py tests/test_compact_occupancy_scalar_ablation.py
=> 39 passed

PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_compact_occupancy_scalar_ablation.py
=> 6 passed

.venv/bin/python -m py_compile benchmarks/compact_occupancy_scalar_ablation.py \
  tests/test_compact_occupancy_scalar_ablation.py src/rtgs/lift/compact_carve.py \
  tests/test_compact_carve.py
=> PASS

git diff --check -- src/rtgs/lift/compact_carve.py tests/test_compact_carve.py \
  benchmarks/compact_occupancy_scalar_ablation.py \
  tests/test_compact_occupancy_scalar_ablation.py \
  benchmarks/results/20260717_compact_occupancy_scalar_PREREG.md \
  benchmarks/results/20260717_compact_occupancy_scalar_IMPLEMENTATION_REVIEW.md
=> PASS
```

An additional in-memory preflight strict-loaded both seven-view bundles, pinned both Torch thread
counts to one, produced the frozen center/extent/box, constructed all seven scalar variants, and
confirmed exact prior-center proxy tensor replay. It wrote no official result.

## Authorization boundary

The reviewed command is exactly:

```bash
PYTHONPATH=src .venv/bin/python -u benchmarks/compact_occupancy_scalar_ablation.py \
  --anchor-bundle runs/compact_masked_bundle_640_20260717/reconstruction_inputs \
  --out runs/compact_occupancy_scalar_ablation_20260717
```

Do not execute until the root agent confirms the final hashes above. Do not overwrite a partial or
failed output, alter a beta/guard/split/seed/tolerance after access, or interpret report views as
novel-view evidence.
