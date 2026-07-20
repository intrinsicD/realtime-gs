# Implementation review addendum 1: GaussianImage++ provider parity

Date: 2026-07-17T03:03:55+02:00 (Europe/Berlin)

Verdict: `PASS`

This addendum independently reviews the repaired implementation while retaining the initial FAIL
review as immutable history. All four blockers are resolved in the executed control flow and
covered by focused CPU regressions. The implementation is ready to seal; the official `run`
command remains prohibited until explicit root-agent authorization.

## Exact repaired files

- `benchmarks/gaussianimage_plus_provider_parity.py` — SHA-256
  `dc0780c44faec093784d60c3bcf684648ea005c66bfaa04297b814d8862d6daa`
- `benchmarks/gaussianimage_plus_native_worker.py` — SHA-256
  `c715b58fa04c09173a5bb755f13a0aa1796ba89deb034d3c7feafb95a08ee2d7`
- `tests/test_gaussianimage_plus_provider_parity.py` — SHA-256
  `2bd9073c87caab79b5136a3f9e63e410d78da590f6fd0bd3a4bdbe7d76031207`
- `benchmarks/results/20260717_gaussianimage_plus_provider_parity_PREREG.md` — SHA-256
  `8b8494bebd3829abdffaf00e7ed905137fc928b1d081e1ae567fb61b267bdae3`
- Historical review
  `benchmarks/results/20260717_gaussianimage_plus_provider_parity_IMPLEMENTATION_REVIEW.md`
  — SHA-256
  `f15e6e4553d2bea5f919ca03fa0b1621a03b2fb8ca0a96e7053feaa757376752`

This addendum is intentionally not self-hashed. The seal command binds it together with the five
files above.

## Blocker dispositions

1. **Checkpoint chronology — resolved.** `command_run()` verifies the seal and sealed sources,
   then exclusively creates `ATTEMPT` before calling `execute_official_run()`. Its pre-attempt
   external verifier checks only the sealed checkpoint path/hash literals plus current static
   foreign source, binary, environment, and preload bindings. `REAL_CHECKPOINT` is absent from the
   pre-attempt required-file/hash path. An independent guarded invocation of
   `verify_external_bindings_pre_attempt()` raised on any attempted checkpoint hash and completed
   with `pre_attempt_checkpoint_reads=0`. Fixture construction and every checkpoint byte read,
   hash, and decode occur inside the post-marker execution path.

2. **Point-of-use identity and TOCTOU defense — resolved.** The parent adapter reads the checkpoint
   once into bytes, hashes that buffer against the seal-bound SHA-256, and decodes the same
   `BytesIO` buffer only after equality. Before worker launch the parent again requires the path
   bytes to have the expected hash. The isolated worker independently reads once, hashes before
   decode, rejects a mismatch, and decodes that same in-memory buffer. Its output metadata reports
   the observed input hash and the parent requires exact equality. The filtered provider NPZ uses
   the same expected-input contract. A change before either consumer is rejected; a change after a
   consumer's single read cannot alter that consumer's decoded bytes.

3. **Endpoint wording — resolved.** The preregistered aim and claim boundary now consistently call
   the CPU endpoint a “dense integer-pixel tile reference renderer.” No sparse-query or
   coordinate-list capability is claimed.

4. **Finite terminal failure — resolved.** `write_terminal_result()` validates the complete result
   with strict `allow_nan=False` JSON before exclusive creation. A non-finite or otherwise
   non-serializable candidate is replaced by a finite `FAIL` receipt carrying attempt, seal, and
   preregistration bindings. The injected-NaN regression proves that an existing attempt receives
   exactly such a parseable terminal result rather than becoming orphaned.

## Semantic and lifecycle recheck

- CPU projection still matches the reviewed float32 direct-covariance inverse, eigenvalue floor,
  circular long-axis support, short-axis clip, C/CUDA truncation, integer lattice, radii, hits, and
  tile enumeration semantics.
- CPU rendering still matches the reviewed additive `min(1, opacity*exp(-sigma))` path, strict
  negative-sigma and `1/255` cutoff behavior, one final clamp, global no-intersection background,
  and black untouched pixels on any non-empty render.
- The 257-component sentinel is rejected before render dispatch. All rendered arms retain the
  complete-set `<=256` tile guard required by the foreign kernel's first-batch behavior.
- The raw checkpoint remains diagnostic and participates in overall status. The provider-valid arm
  preserves only finite stored-float32 components satisfying positive diagonals and determinant,
  preserves order, performs no covariance repair, and re-renders the exact filtered NPZ.
- The isolated worker remains render-only, explicitly imports and hashes the bundled `csrc.so`,
  checks the clean foreign commit, Python prefix, Torch/CUDA labels, sole `libstdc++` preload, and
  never imports an image decoder, trainer, optimizer, or source RGB.
- Every rendered synthetic arm, raw diagnostic, filtered provider arm, parameter/projection/tile
  identity, and raw/clamped image tolerance remains part of overall status. Claim limits still
  forbid image-fit quality, calibrated-data, source-quality, speed/memory, provider preference,
  and downstream 3D claims.
- The preregistration hash literal matches the repaired file. The seal requires both the historical
  exact FAIL verdict and this addendum's exact PASS verdict, re-runs focused verification, seals all
  six files, and refuses preexisting seal/attempt/result artifacts.

## Current external bindings

- Clean GaussianImage++ commit:
  `549cfaab2b400248f685c12782a180f3cfc038b0`
- Static external aggregate SHA-256:
  `341b3bf71472064d4133cdd60ffd067d3f11803b2c1ea13cfa4a11cf4d7a1003`
- Full seal-time external aggregate SHA-256:
  `b797c34cba616509d51e22d649dc753ac1f906da4cb4739993a0bac53d7fd0b4`
- Frozen checkpoint SHA-256:
  `ad611facd72e813dece1b95c3268dbfd82f8af01cdb5ad67e1c7675cc670794b`
- Bundled `csrc.so` SHA-256:
  `9b57b7e0531a50d87c529d3541fbf370f9d85455836ac0cf5414c01ce48ac222`
- System `libstdc++.so.6` SHA-256:
  `1fd75fe70354a416d75aef22bcae68c47bd25d20e2d0568c30b1a9838cf62f11`

The foreign checkout was clean during this addendum review. No source RGB was opened.

## Checks executed

```text
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

.venv/bin/python -m py_compile benchmarks/gaussianimage_plus_provider_parity.py \
  benchmarks/gaussianimage_plus_native_worker.py \
  tests/test_gaussianimage_plus_provider_parity.py
=> PASS
```

No seal, official attempt, official result, worker output, viewer artifact, implementation edit, or
experiment execution was created during this addendum review.
