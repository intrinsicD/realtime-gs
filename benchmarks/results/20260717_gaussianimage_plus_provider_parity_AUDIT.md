# Independent results audit: GaussianImage++ provider parity

Date: 2026-07-17T03:12:10+02:00 (Europe/Berlin)

Verdict: `QUALIFIED PASS`

The official `PASS` is valid for its preregistered mechanism-only claim: on the exact frozen
GaussianImage++ binary, software stack, RTX 3050, synthetic fields, and one 160×120 checkpoint,
the CPU checkpoint adapter and dense tile reference reproduce the native direct-covariance
additive renderer within the frozen tolerances. The deterministic finite-SPD filter selects 626 of
639 components without repair, and the native worker reproduces the exact filtered field within
the same gates.

There is no blocker to retaining that bounded result. There **are** hard blockers to any broader
promotion: this run did not fit an image, use calibrated full-resolution views, establish that
filtering preserves source-image quality, test memory or speed, compare providers, initialize or
refine 3D Gaussians, or render a 3D viewer result. In particular, the 13-component filter changes
570 of 19,200 pixels by more than `1e-6`, with maximum clamped-channel change `0.3718417883`.
That is evidence that filtering is observable, not evidence that it is harmless.

## Claim disposition

| # | Claim | Kind and scope | Evidence | Disposition |
|---|---|---|---|---|
| 1 | The CPU adapter/reference matches the native renderer on the seven rendered synthetic semantic cases. | Measured; synthetic; exact frozen CUDA binary and RTX 3050 | Official result plus preserved synthetic input/native NPZs | **Confirm.** Parameters, projected means, radii, hits, and tile candidate sets agree; every image gate passes. |
| 2 | The raw 639-component checkpoint diagnostic matches the native renderer. | Measured; one 160×120 real checkpoint; diagnostic only | `raw_checkpoint_diagnostic` and `checkpoint/raw_native.npz` | **Confirm, narrow.** It is one checkpoint on one frozen GPU/software path, not a provider-valid or general checkpoint claim. |
| 3 | Stored-float32 finite-SPD filtering deterministically retains 626 components, preserves order, performs no repair, and the exact filtered field matches native re-rendering. | Measured; adapter/provider-eligibility mechanism | `filtered_checkpoint_provider`, filtered input/native NPZs, mask hash | **Confirm.** The independently recomputed mask hash is `ba4c6f9f…6540`; the filtered parity arm passes. |
| 4 | The implementation safely rejects a tile containing more than the native kernel's complete first 256 candidates. | Proven for the frozen sentinel and guarded code path | `tile_cap_sentinel`, source, focused tests | **Confirm.** Independent projection gives 257 candidates and CPU render rejects before dispatch. |
| 5 | SPD filtering preserves source-image quality or is suitable as-is for Stage 1. | Not measured | No source RGB or quality target was available | **Withhold.** The nonzero render change makes a later quality experiment mandatory. |
| 6 | GaussianImage++ is preferable to StructSplat, scales to full resolution, or improves 3D fitting/refinement. | Not measured | None in this experiment | **Withhold.** No provider comparison, full-resolution fit, calibrated lift, optimization, or viewer evidence exists. |
| 7 | The recorded worker timings or memory values establish performance. | Asserted only if cited beyond diagnostics | Worker metadata | **Retire/forbid.** GPU-idle state, warmup, repeats, and an aggregation rule were not preregistered. |

## Chronology and source binding

The lifecycle is ordered and append-only:

- repaired preregistration SHA-256:
  `8b8494bebd3829abdffaf00e7ed905137fc928b1d081e1ae567fb61b267bdae3`;
- historical implementation review retained `FAIL`, then the repaired exact-hash addendum recorded
  `PASS` at 2026-07-17T03:03:55+02:00;
- implementation seal timestamp: 2026-07-17T01:05:09+00:00;
- one-shot attempt timestamp: 2026-07-17T01:06:25+00:00;
- terminal result timestamp: 2026-07-17T01:06:39+00:00.

The attempt predates every preserved worker input/output by filesystem time. The result binds the
attempt file SHA-256 exactly, and attempt and result both bind the seal file and preregistration.
Strict JSON parsing found no NaN or infinity.

| Binding | Independently checked value |
|---|---|
| Seal file SHA-256 | `c37ec5c4229d49e3b78d6f030f7b4293beb18f1e519e42f4a251b1146196d13b` |
| Seal canonical payload SHA-256 | `77445ede1d67aaef9a3ac1540aa21e249b773ba9c8d485d80e02875dc8a2b1e6` |
| Attempt file SHA-256 | `6a032783f5745eab9ef731c3d9aa86cf966590cb70ee9bd8eb5bf8544f99eb34` |
| Result file SHA-256 | `871583747b9a0dd0fe05437d2464ac04e0dcd22c99ef52ef0a79e31231f92810` |
| Sealed source aggregate | `b7dec8eb1c742c9e99f1670e2610c5060c3a6e45fc83c1789bf986d394189215` |
| External aggregate | `b797c34cba616509d51e22d649dc753ac1f906da4cb4739993a0bac53d7fd0b4` |
| Frozen checkpoint | `ad611facd72e813dece1b95c3268dbfd82f8af01cdb5ad67e1c7675cc670794b` |
| Frozen `csrc.so` | `9b57b7e0531a50d87c529d3541fbf370f9d85455836ac0cf5414c01ce48ac222` |

All six sealed implementation/review file hashes and their aggregate match the current bytes. The
external GaussianImage++ checkout remains clean at commit
`549cfaab2b400248f685c12782a180f3cfc038b0`; all bound foreign source, extension, preload, and
checkpoint hashes still equal the seal. The official source bundle is currently untracked in a
dirty parent worktree, but the seal preserves the exact executed bytes; Git revision alone must
not be presented as sufficient provenance.

## Independent numeric recomputation

I did not replay the sealed one-shot CUDA command. I recomputed all gates from the preserved input
and native-output artifacts, first through a fresh CPU evaluation of the sealed reference and then
through a separately written NumPy projection/raster calculation based directly on the frozen
foreign formulas. Both passes preserve the official decision.

| Arm | Components | Max tile population | Official raw max abs | Official raw mean abs | Result |
|---|---:|---:|---:|---:|---|
| Seven rendered synthetic cases | 1–3 | 0–3 | at most `1.1920929e-7` | at most `3.6057297e-9` | PASS |
| Raw checkpoint diagnostic | 639 | 134 | `3.4570694e-6` | `3.9558220e-8` | PASS |
| Filtered provider field | 626 | 127 | `9.5367432e-7` | `3.8649148e-8` | PASS |
| Tile-cap sentinel | 257 | 257 | not rendered | not rendered | expected rejection PASS |

For every rendered arm, projected means have zero maximum error, parameters are byte-value exact,
radii and hit counts are exact, and complete candidate sets are exact. The maximum absolute conic
error on both checkpoint arms is `7.2479248e-5`; it satisfies the preregistered combined
`atol=2e-6, rtol=5e-6` gate. All raw/clamped image comparisons satisfy
`atol=1e-5, rtol=1e-5`, and every raw mean error is below `1e-6`.

The separate NumPy implementation also passes every endpoint. Its independently calculated raw
maximum errors were `1.0728836e-6` for the raw checkpoint and `9.5367432e-7` for the filtered
field; small differences from the Torch-reference error summaries reflect different CPU
exponential arithmetic and do not approach a frozen gate.

The nine worker records have zero return codes; their stdout/stderr hashes, output hashes, actual
input hashes, command-line expected hashes, worker-reported hashes, and NPZ metadata hashes all
agree. All report the exact isolated environment: Torch `2.9.0+cu128`, CUDA `12.8`, and
`NVIDIA GeForce RTX 3050` capability 8.6.

The filter accounting also recomputes exactly:

- raw components: 639;
- retained finite-SPD components: 626;
- removed components: 13 (`2.034%`);
- pixels with any clamped-channel change greater than `1e-6`: 570/19,200 (`2.969%`);
- raw-to-filtered clamped maximum absolute change: `0.3718417883`;
- raw-to-filtered clamped mean absolute channel change: `0.00053289445`.

## Protocol and implementation findings

No result-blocking protocol deviation was found. The repaired lifecycle creates the attempt before
checkpoint access; both parent and worker hash a single in-memory byte buffer before decoding; the
worker output reports the observed hash; invalid terminal JSON is reduced to a finite `FAIL`; and
the 257-candidate case is not sent to the incomplete native path. Focused regressions cover these
repairs.

The exact foreign code path independently inspected in this audit uses direct packed covariance,
integer pixel coordinates, the reviewed covariance inverse/eigenvalue radius, additive
`min(1, opacity*exp(-sigma))`, the `1/255` cutoff, no non-empty-render background term, and the
three-channel kernel's first-batch termination. That supports the implemented semantics. The
binary is content-hashed, but this experiment contains no reproducible build receipt proving that
the binary was compiled from the adjacent source files; therefore the strongest defensible claim
is parity with the exact `csrc.so`, not with every build of the source commit.

No source RGB reader, trainer, optimizer, or fitting path exists in the audited harness/worker.
This audit did not syscall-trace the already-consumed official run, so the access conclusion is
source/control-flow based. The only real-data input in scope is the compact checkpoint.

The parent CPU environment was not serialized into the official seal/result. This audit used
Python 3.12.9, NumPy 2.1.3, and Torch `2.9.0+cu128`; preserved native NPZs and exact source hashes
made numeric recomputation possible. Capture parent Python/NumPy/Torch versions in future seals if
portable replay completeness is claimed.

## Checks executed

```text
# strict lifecycle/hash/source/external/worker-output verification and exact metric recomputation
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python - <<'PY'
  ...audit-only recomputation...
PY
=> all lifecycle/source/external checks true; all recomputed arms PASS

# separately implemented NumPy direct-covariance projection and raster calculation
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python - <<'PY'
  ...independent projection/raster audit...
PY
=> ALL_PASS True

PYTHONPATH=. CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m pytest -q tests/test_gaussianimage_plus_provider_parity.py
=> 22 passed

.venv/bin/python -m ruff check benchmarks/gaussianimage_plus_provider_parity.py \
  benchmarks/gaussianimage_plus_native_worker.py \
  tests/test_gaussianimage_plus_provider_parity.py
=> All checks passed

.venv/bin/python -m ruff format --check benchmarks/gaussianimage_plus_provider_parity.py \
  benchmarks/gaussianimage_plus_native_worker.py \
  tests/test_gaussianimage_plus_provider_parity.py
=> 3 files already formatted

git diff --check -- <provider-parity scoped files>
=> PASS
```

The full repository verification was not run in this isolated audit because unrelated concurrent
research changes are still in progress. The root session must run the repository-wide gate before
handoff. No CUDA worker was replayed because the official phase is one-shot; CUDA evidence comes
from the hash-bound preserved worker outputs. No viewer was expected or produced because the
preregistered experiment is a 2D renderer/adapter mechanism gate.

## Evidence required for promotion

Before GaussianImage++ becomes a Stage-1 provider or supports any downstream claim, a separate
preregistered experiment must sequentially fit selected calibrated views at full resolution,
preserve variable per-view optimized counts, export only compact 2D Gaussian fields before RGB is
closed, quantify the quality cost of deterministic filtering, lift those fields into 3D, and show
the resulting 3D Gaussians in the exact gsplat viewer. A general renderer-adapter claim would also
need multiple independently selected checkpoints/configurations (including larger tile
populations) and preferably a reproducible native-extension build receipt.
