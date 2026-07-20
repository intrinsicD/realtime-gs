# Independent implementation review: quaternion radial-gauge Retry-2

Verdict: PASS

Reviewed outcome-blind on `2026-07-16T04:50:03+02:00` after the Retry-2 preregistration was
frozen and before any Retry-2 seal, attempt marker, official seed preparation, optimizer arm,
render comparison, or scientific result.

This verdict is implementation authorization only. It does not establish Phase-A materiality,
authorize Phase B, select a quaternion policy, or support a production/default claim. A fresh seal
must still execute and record the complete frozen verification sequence before any official
Retry-2 action.

## Scope and protocol binding

The reviewed implementation is bound to these exact prospective protocol bytes:

```text
fe201606b878cb29b4502a283dde78a30c3d2dab9a0efa091c83be3b95bfe4f3  benchmarks/results/20260716_quaternion_gauge_iter2_PREREG.md
f1ba26d2520e6f78731b404babe0e091f2341d16ab5e30607b25ba32692c764e  benchmarks/results/20260716_quaternion_gauge_PREREG.md
```

The original preregistration remains normative for every scientific choice. Static inspection
found no Retry-2 change to the frozen seeds, scenes, training/held-out identities, native fit,
Depth-surface lift, anisotropic subset, perturbation, radial scales, Phase-A arms, optimizer or LR,
schedules, checkpoints, metrics, thresholds, materiality conjunction, Phase-B arms, utility/safety
gates, preference rule, or interpretation boundary. Retry-2 changes only removed-gradient
diagnostic arithmetic, its verification, and append-only provenance/output namespaces.

The consumed invalid attempt and its independent disposition remain byte-bound:

```text
146193dc0783b01d5fada9608e276845a1aea6e8e44ba4ed53772adc47ef4ad8  benchmarks/results/20260716_quaternion_gauge_SEAL.json
c6a7c663edff15114c11b714ed6342e1ebd1e72b535a565e6d3861ce9e7868dc  benchmarks/results/20260716_quaternion_gauge_PHASE_A_ATTEMPT.json
8381979a9b6fba958e34d8a2d2e4210dc783ede808edd2fa88faddf3b4b53739  benchmarks/results/20260716T015517Z_cpu_quaternion_gauge_invalid.json
34adccfe91650cd821dc99c0f6c4cdf7e5668ac4b89faa0e4ad4466c95d56a61  benchmarks/results/20260716T015517Z_cpu_quaternion_gauge_invalid_RESULT.md
7528d22e0daa909f8f67e8d73b0269de5f9b4bf21b1677a0d2341361be1ecd8d  benchmarks/results/20260716T015517Z_cpu_quaternion_gauge_invalid_AUDIT.md
```

Read-only provenance validation confirmed all five hashes and confirmed the required absence of
the consumed namespace's valid Phase-A JSON/note/reviews, old Phase-B marker, and old Phase-B
output/note. The old invalid artifact exposes no optimizer arms, trajectories, checkpoints, AUC,
materiality decision, or Phase-B clearance, so no scientific outcome was available to tune this
repair.

## Reviewed source snapshot

The exact accepted Retry-2 implementation bytes are:

```text
e7819814b469a55dbca89c0e7e853f5dcb0cd6974fdc6fcb972588ad3cd60bb2  benchmarks/quaternion_gauge_ablation.py
5e831558459187db8ea69f2ad97fd9c4e1a66efdb6f4ae0c4223f484489fa51f  tests/test_quaternion_gauge_ablation.py
```

The pre-repair snapshot used for comparison was:

```text
fd58d01ade1dcd8582acd915b1eb4478df2fc52e105d2ede1b51079d68cdc747  benchmarks/quaternion_gauge_ablation.py
e8c33135be51c56a5335d0e410b63f8bc5c3ea13020e799ce66279a9d905456a  tests/test_quaternion_gauge_ablation.py
```

The Retry-2 preregistration's protected Trainer and focused contract remain byte-identical:

```text
26105043c13453b7904c6ec8626cce2622f580485ffb335f99ec59a52a3a1d36  tests/test_quaternion_gauge.py
3bb73a2071ff3525c07c0d1a57387ecccb5b5f16a3cc18398091b2606752053f  src/rtgs/optim/trainer.py
7426f166742203b907c992abc24c0d7503a0da7783eb59ccb2515c51e5735b2c  pyproject.toml
```

No production source was changed by Retry-2. In particular, the Trainer policy seam, default
`current` behavior, policy equations, callbacks, and density guard were not modified.

## Diagnostic repair and optimizer non-interference

The shared `removed_gradient_diagnostics_float64` helper implements the frozen order exactly:
raw inputs must be matching finite float32 `(N,4)` tensors; both tensors are promoted to float64;
the quaternion is then normalized; signed dot, absolute numerator, clamped gradient-norm
denominator, fraction, and projected diagnostic are calculated in float64. It rejects invalid
shape, dtype, device, finiteness, and zero/near-zero quaternion rows. `ACTIVE_NORM` remains
`1e-12`, and exact tensor equality remains required by evidence validation.

AST and source inspection found exactly four production/validation call sites:

1. step-zero prerequisite production;
2. Phase-A projection-step production;
3. serialized Phase-A projection-step recomputation; and
4. serialized step-zero prerequisite recomputation.

Both validators reconstruct the serialized raw quaternion and gradient as float32 before calling
the helper. Stored numerators, denominators, fractions, projected step-zero evidence, and ordered
diagnostic hashes are recomputed from the helper. No duplicate removed-gradient formula remains in
those evidence paths.

The actual optimizer path remains independent and native float32. The producing arm still computes
`unit = normalize(q_old)` in float32 and copies
`raw_gradient - unit * sum(unit * raw_gradient)` into `q.grad`. The independent Phase-A replay
still reconstructs the raw gradient as float32, evaluates the same native expression, and replays
Adam without calling the diagnostic helper. The helper's float64 projected tensor is never copied
to `q.grad`, an optimizer parameter, an Adam buffer, `q_star`, `q_new`, loss, or replay state.
Static inspection and the observational-purity/replay tests found the native projection, Adam
configuration, policy ordering, policy equations, and optimizer-state behavior unchanged.

## Fail-closed provenance and namespace review

All active Retry-2 paths, output regexes, marker types, result types, seal type, and scientist-review
type use the frozen `quaternion_gauge_iter2` namespace. References to the original namespace are
limited to immutable consumed-artifact bindings, required-absence guards, and negative rejection
tests. The old seal, marker, and output cannot satisfy a Retry-2 gate.

Seal creation verifies the original and Retry-2 preregistration hashes, independent review,
consumed artifact hashes, required absences, source hashes, environment, and full verification. It
checks the protocol/provenance/review/source snapshot both before and after verification and embeds
the consumed-attempt provenance in the fresh non-self-referential seal. Seal loading revalidates
those bindings and absences against current bytes.

Phase-A markers bind the verified fresh seal, exact output prefix and all four preflighted output
paths, full sealed-source aggregate, and the loaded-source subset. Result bindings now have an
exact 13-key schema covering both preregistrations, implementation review, seal path, canonical
seal digest, byte-level `seal_file_sha256`, source aggregate, retry-provenance digest, and consumed
marker path/hash. Phase-B authorization checks that exact key set and every seal-derived value,
then separately checks the canonical Phase-A marker path, marker byte hash, marker payload/output
binding, raw Phase-A recomputation, derived human audit, and strict machine-review clearance.

An independent adversarial pass initially found that the first Retry-2 candidate serialized
`seal_file_sha256` without validating it during Phase-B authorization. No review artifact or
official action had been created. The implementation was corrected before this PASS: the exact
binding schema now compares `seal_file_sha256`, rejects missing or extra binding keys, and retains
the canonical marker path/hash checks. Dedicated tests prove tampering only that field, deleting
it, or adding an unexpected field all fail closed.

## Verification evidence

The following outcome-free checks were run against the accepted hashes above:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m pytest -q \
  tests/test_quaternion_gauge_ablation.py tests/test_quaternion_gauge.py
```

Result: exit `0`; 52 focused tests passed.

The focused suite covers the adversarial normalize-order distinction, an independent explicit
float64 reference for all four helper outputs, invalid input rejection, JSON round-trip through the
real raw validator, separate numerator/fraction/hash tampering, diagnostic observational purity,
native float32 Adam replay, exactly four shared-helper sites, Retry-2 namespace rejection, consumed
artifact hashes/absence, fresh seal provenance, exact 13-key result bindings, and the pre-existing
Phase-A/Phase-B raw recomputation guards.

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m pytest -q -m 'not slow'
```

Result: exit `0`; the complete non-slow CPU suite passed, with only the suite's expected skips.

```text
.venv/bin/python -m ruff check \
  benchmarks/quaternion_gauge_ablation.py tests/test_quaternion_gauge_ablation.py
.venv/bin/python -m ruff format --check \
  benchmarks/quaternion_gauge_ablation.py tests/test_quaternion_gauge_ablation.py
git diff --check -- \
  benchmarks/quaternion_gauge_ablation.py tests/test_quaternion_gauge_ablation.py
```

Result: all exited `0`; Ruff reported all checks passed and both files formatted.

These reviewer checks do not replace the seal command's required repository-wide Ruff, format,
non-slow test, docs-sync, and diff verification. The fresh seal must run and bind that exact full
sequence after this review file is finalized.

## Limitations and authorization boundary

I did not invoke the harness `seal`, `audit`, or `run` actions; did not create or consume a Retry-2
marker; did not construct an official scene or seed; did not execute an official optimizer arm;
did not inspect an unavailable old arm outcome; and did not modify any source, test, production,
documentation, or ARA file. This review writes only this implementation-review artifact.

PASS means the outcome-neutral repair is sufficiently specified, isolated, tested, provenance-
bound, and fail-closed to proceed to the fresh full-verification seal. It is not scientific
evidence. Any source, protocol, review, environment, old-artifact, or required-absence drift must
stop sealing/execution. Any eventual Phase-A artifact still requires an independent scientist
audit and strict machine clearance before Phase B, and any Phase-B result requires another audit
before interpretation.
