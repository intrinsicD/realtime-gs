# Independent implementation-review addendum: Quaternion Retry-2 pre-seal recovery

Verdict: PASS

Reviewed outcome-blind on `2026-07-16T05:05:09+02:00`, after a pre-seal full-verification
failure and its narrow test/provenance repair, but before any successful Retry-2 seal write,
attempt marker, official seed preparation, optimizer arm, render comparison, or scientific
result.

This append-only addendum does not replace or edit the original implementation review. Together,
the frozen base review and this addendum authorize only creation of a fresh full-verification seal.
They do not establish Phase-A materiality, authorize Phase B, select a quaternion policy, or
support a production/default claim.

## Frozen protocol and base-review binding

The scientific protocol and base review remain byte-identical:

```text
fe201606b878cb29b4502a283dde78a30c3d2dab9a0efa091c83be3b95bfe4f3  benchmarks/results/20260716_quaternion_gauge_iter2_PREREG.md
f23708072e6746e7e0e714020d3e6d0a31bf132150e2fa57b14a9f1a63bac818  benchmarks/results/20260716_quaternion_gauge_iter2_IMPLEMENTATION_REVIEW.md
f1ba26d2520e6f78731b404babe0e091f2341d16ab5e30607b25ba32692c764e  benchmarks/results/20260716_quaternion_gauge_PREREG.md
7528d22e0daa909f8f67e8d73b0269de5f9b4bf21b1677a0d2341361be1ecd8d  benchmarks/results/20260716T015517Z_cpu_quaternion_gauge_invalid_AUDIT.md
```

The original and Retry-2 preregistrations still control every hypothesis, scene, seed, input,
fit/lift configuration, diagnostic subset, perturbation, arm, optimizer, schedule, checkpoint,
metric, threshold, decision gate, and interpretation boundary. This addendum introduces no
scientific or outcome-dependent change. It records only recovery from a failed implementation
verification before the seal/attempt boundary.

## Failed-seal chronology and disposition

The base review authorized this then-current implementation snapshot:

```text
e7819814b469a55dbca89c0e7e853f5dcb0cd6974fdc6fcb972588ad3cd60bb2  benchmarks/quaternion_gauge_ablation.py
5e831558459187db8ea69f2ad97fd9c4e1a66efdb6f4ae0c4223f484489fa51f  tests/test_quaternion_gauge_ablation.py
```

The subsequent official seal command entered the seal's required repository-wide verification
sequence and stopped fail-closed during the non-slow pytest command. The failing source-domain test
computed `loaded_source_hashes()` inside the already-populated pytest interpreter and asserted a
strict loaded-source subset. Its answer therefore depended on unrelated modules imported earlier
in the full suite. This was import-order/test-process pollution, not a failure of the sealed-source
algorithm in a fresh harness process and not a scientific result.

The exception occurred inside `create_seal()` before it returned a payload to the append-only
writer. At addendum review time all of the following remained absent:

```text
benchmarks/results/20260716_quaternion_gauge_iter2_SEAL.json
benchmarks/results/20260716_quaternion_gauge_iter2_SEAL_RESULT.md
benchmarks/results/20260716_quaternion_gauge_iter2_PHASE_A_ATTEMPT.json
```

No Retry-2 seal or marker was created or consumed; no official scene, seed, arm, trajectory,
checkpoint, AUC, materiality decision, or Phase-B clearance was exposed. Under the frozen
chronology, the failed verification required repair and a fresh independent review but did not
consume the scientific namespace.

## Exact accepted recovery snapshot

The final reviewed recovery bytes are:

```text
1c18438ff76330c58f5a78519cf8c833e4375ecb279ffb05460664ae1baa7d62  benchmarks/quaternion_gauge_ablation.py
60d3648747d4d7803b83b6bebc8e741b159ef415acc1983c2060137458c6df52  tests/test_quaternion_gauge_ablation.py
```

The Retry-2 protected production/contract bytes remain unchanged:

```text
26105043c13453b7904c6ec8626cce2622f580485ffb335f99ec59a52a3a1d36  tests/test_quaternion_gauge.py
3bb73a2071ff3525c07c0d1a57387ecccb5b5f16a3cc18398091b2606752053f  src/rtgs/optim/trainer.py
7426f166742203b907c992abc24c0d7503a0da7783eb59ccb2515c51e5735b2c  pyproject.toml
```

Static inspection found no change to the shared float64 removed-gradient helper, its four evidence
call sites, the native float32 projection copied into `q.grad`, independent float32 Adam replay,
Trainer, policy equations, scientific configurations, materiality/utility/safety calculations,
thresholds, artifact types, or output namespace. The exact Phase-A result binding schema remains
the previously reviewed 13-key schema; no result field was added, removed, or reinterpreted.

## Fresh-interpreter source-domain repair

The sole test-mechanism repair moves the loaded-source-domain probe into a fresh Python subprocess.
The subprocess imports only the real harness and its dependencies, calls the real
`loaded_source_hashes()`, and returns its path map and canonical aggregate as strict JSON. The
parent test independently checks that aggregate, the proper-subset relationship to the full sealed
domain, and the different full/loaded aggregates. Ambient modules imported by pytest before the
test can no longer affect the probe.

Before this addendum exists, the toy test supplies one prospective placeholder digest only for the
missing addendum path while still hashing every existing file through the real function. Once this
addendum is written, both parent and subprocess take the ordinary real-file branch. This mechanism
does not construct an official scene or seed and cannot expose an experiment outcome.

## Append-only addendum provenance

The harness now defines the sole addendum path as:

```text
benchmarks/results/20260716_quaternion_gauge_iter2_IMPLEMENTATION_REVIEW_ADDENDUM_1.md
```

The addendum is fail-closed and transitively bound through every execution layer:

1. `_sealed_paths()` includes the base review and this addendum.
2. `loaded_source_hashes()` explicitly includes both review files even if import discovery would
   not find Markdown resources.
3. addendum verification requires the exact file and an exact `Verdict: PASS` line, returning its
   byte hash;
4. seal creation captures both review records before full verification, re-reads both after it,
   and refuses any review/source/aggregate drift;
5. the seal payload stores the addendum path/hash explicitly, while its canonical digest and
   source aggregate independently commit to the same bytes;
6. seal loading recomputes the addendum record and requires exact equality;
7. the verified-seal summary carried into fresh markers contains the addendum record; and
8. Phase-A marker validation requires the addendum in the loaded-source subset, in addition to
   checking that subset against the full sealed source map and recomputing its aggregate.

The existing 13-key result schema deliberately continues to carry the base
`implementation_review_path`/`implementation_review_sha256`. Its `seal_sha256`,
`seal_file_sha256`, and `source_aggregate` fields bind the seal's explicit addendum record and the
addendum bytes transitively, so no post-preregistration result-schema expansion was needed.

Focused fail-closed tests cover a missing addendum, a non-PASS addendum, byte tampering, missing
seal binding, wrong addendum path, wrong stored hash, disagreement with the current addendum
record, omission from a marker's loaded-source map with a recomputed aggregate, and the fresh-
interpreter full-versus-loaded source-domain relation. The prior seal-file binding and exact-key
tamper tests remain intact.

## Verification evidence

The following outcome-free checks were run against the accepted recovery hashes above:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m pytest -q \
  tests/test_quaternion_gauge_ablation.py tests/test_quaternion_gauge.py
```

Result: exit `0`; 54 focused tests passed.

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m pytest -q -m 'not slow'
```

Result: exit `0`; the complete non-slow CPU suite passed, with only the suite's expected skips.
This run exercises the repaired source-domain test in the full-suite import context that caused
the pre-seal failure.

```text
.venv/bin/python -m ruff check \
  benchmarks/quaternion_gauge_ablation.py tests/test_quaternion_gauge_ablation.py
.venv/bin/python -m ruff format --check \
  benchmarks/quaternion_gauge_ablation.py tests/test_quaternion_gauge_ablation.py
git diff --check -- \
  benchmarks/quaternion_gauge_ablation.py tests/test_quaternion_gauge_ablation.py
```

Result: all exited `0`; Ruff reported all checks passed and both files formatted.

These reviewer checks do not replace the seal's frozen repository-wide verification sequence.
After this addendum is finalized, the fresh seal must rerun and bind repository-wide Ruff, format,
the full non-slow suite, docs sync, and diff check against the final addendum-inclusive source map.

## Limitations and authorization boundary

I did not invoke the harness `seal`, `audit`, or `run` actions; did not create or consume a Retry-2
marker; did not construct an official scene or seed; did not execute an official optimizer arm;
did not inspect an unavailable old arm outcome; and did not modify any harness, test, production,
documentation, or ARA file. This review writes only this addendum.

PASS means the pre-seal, outcome-neutral import-order failure has a narrowly scoped, independently
reviewed, append-only recovery whose final bytes and provenance are suitable for a fresh seal
attempt. It is not scientific evidence. Any protocol, base-review, addendum, source, environment,
old-artifact, or required-absence drift must stop sealing/execution. Any eventual Phase-A artifact
still requires an independent scientist audit and strict machine clearance before Phase B; any
Phase-B result requires another audit before interpretation.
