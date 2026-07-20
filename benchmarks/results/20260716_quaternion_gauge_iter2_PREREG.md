# Quaternion radial-gauge Retry-2 preregistration

Status: **FROZEN before Retry-2 implementation, pilot, seal, or official execution**

Freeze time: `2026-07-16T04:23:56+02:00`

This is an append-only, outcome-neutral repair preregistration. It does not amend, replace, or
reinterpret the original scientific protocol. It authorizes one fresh attempt only after the
implementation and gates below pass. It records no Quaternion Phase-A optimizer-arm result,
materiality decision, or Phase-B result.

## 1. Normative inheritance and precedence

The complete scientific protocol is incorporated unchanged from:

```text
f1ba26d2520e6f78731b404babe0e091f2341d16ab5e30607b25ba32692c764e  benchmarks/results/20260716_quaternion_gauge_PREREG.md
```

That exact file remains normative for every scientific choice: hypothesis, estimand, scene
construction, seeds, inputs, fitting and lifting configuration, diagnostic subset and perturbation,
arms, radial scales, optimizer, schedules, checkpoints, metrics, thresholds, reductions,
invariants, Phase-A materiality rule, Phase-B utility/safety/preference rules, interpretation
boundary, and prohibition on changing a production default. Retry-2 MUST fail closed if that file's
byte hash differs.

This delta controls only:

1. the arithmetic order of removed-gradient *diagnostic evidence* specified in Section 4;
2. the adversarial verification needed to prove that repair and optimizer non-interference; and
3. fresh Retry-2 provenance, marker, and output namespaces.

If this file appears to change a scientific choice, the original file controls and Retry-2 MUST
stop before sealing. There is no permission to tune, relax, add, delete, or reinterpret a scientific
arm, threshold, equality, tolerance, schedule, seed, input, metric, or decision rule.

For avoidance of doubt, the inherited experiment still uses exactly seeds `[0,1,2]`; the original
40-Gaussian, 12-camera, 48-pixel synthetic scenes; training views
`[0,1,2,4,5,6,8,9,10]`; held-out views `[3,7,11]`; the original native fit and metric-depth surface
lift; the frozen top-128 anisotropic diagnostic rows; the 20-degree perturbation; radial scales
`[0.25,1,4]`; all five original 40-step Phase-A policies; Phase-A checkpoints
`[0,10,20,30,40]`; all three original 120-step Phase-B arms and cyclic arm orders; and Phase-B
checkpoints `[0,30,60,90,120]`. All original exact configurations and decision thresholds are read
from the hash-bound original rather than restated here.

## 2. Consumed attempt and retry authority

The prior namespace is consumed and immutable. Retry-2 binds these exact bytes:

```text
146193dc0783b01d5fada9608e276845a1aea6e8e44ba4ed53772adc47ef4ad8  benchmarks/results/20260716_quaternion_gauge_SEAL.json
c6a7c663edff15114c11b714ed6342e1ebd1e72b535a565e6d3861ce9e7868dc  benchmarks/results/20260716_quaternion_gauge_PHASE_A_ATTEMPT.json
8381979a9b6fba958e34d8a2d2e4210dc783ede808edd2fa88faddf3b4b53739  benchmarks/results/20260716T015517Z_cpu_quaternion_gauge_invalid.json
34adccfe91650cd821dc99c0f6c4cdf7e5668ac4b89faa0e4ad4466c95d56a61  benchmarks/results/20260716T015517Z_cpu_quaternion_gauge_invalid_RESULT.md
7528d22e0daa909f8f67e8d73b0269de5f9b4bf21b1677a0d2341361be1ecd8d  benchmarks/results/20260716T015517Z_cpu_quaternion_gauge_invalid_AUDIT.md
```

The final independent audit has exact `Verdict: PASS` only for invalid-artifact integrity. It
confirms that the old JSON exposes only preparation and prerequisite evidence for seeds 0/1/2,
with no arms, optimizer trajectories, checkpoints, AUC, materiality decision, valid Phase-A result,
or Phase-B authorization. It diagnoses a producer/validator floating-point ordering mismatch and
explicitly permits an append-only diagnostic-only retry under the conditions adopted here.

The old failure is fixed evidence, not a pilot. No value from stripped in-memory arms may be
recovered, inspected, reconstructed, used to tune Retry-2, or cited as an outcome. The old seal,
marker, invalid JSON, invalid note, and audit MUST remain byte-identical. The old seal or marker
MUST NOT be reused. The old namespace MUST NOT be repaired, completed, overwritten, renamed,
deleted, or used to authorize Phase B.

At Retry-2 seal time the harness MUST verify the five hashes above and verify that the consumed old
namespace still has no valid `20260716T015517Z_cpu_quaternion_gauge_audit.json`, no corresponding
valid-result note or machine clearance, no old Phase-B attempt marker, and no old Phase-B output.
Any contradiction stops Retry-2 before execution.

## 3. Frozen causal diagnosis

The consumed producer computed a diagnostic numerator using
`float64(normalize_float32(q_old))`, while validation reconstructed the serialized values and used
`normalize_float64(float64(q_old))`. Exact equality was then required. These mathematically
equivalent orders can differ at the last bits. The actual projection passed to Adam and the Adam
replay both used the same native-float32 equation and were not implicated.

This diagnosis authorizes no tolerance substitution and no conclusion about gauge materiality or
candidate utility. Retry-2 retains exact raw equality; it makes producer and validator derive the
diagnostic by one shared operation order.

## 4. Sole arithmetic repair

The implementation MUST define one shared pure helper named
`removed_gradient_diagnostics_float64(quaternions, raw_gradient)`. Both native production and every
serialized-evidence validator for removed-gradient diagnostics MUST call it. No duplicate formula
is allowed for those fields.

Inputs MUST first be or be reconstructed as the raw native `torch.float32` arrays with matching
shape `(N,4)`. In particular, validation MUST use
`torch.tensor(serialized_value, dtype=torch.float32)` before calling the helper; it MUST NOT parse
the JSON array directly as float64. The helper then performs exactly this order:

```python
q64 = quaternions.detach().to(dtype=torch.float64)
g64 = raw_gradient.detach().to(dtype=torch.float64)
unit64 = torch.nn.functional.normalize(q64, dim=-1)
signed_dot64 = (unit64 * g64).sum(dim=-1)
numerator64 = signed_dot64.abs()
denominator64 = torch.linalg.vector_norm(g64, dim=-1).clamp_min(ACTIVE_NORM)
fraction64 = numerator64 / denominator64
projected64 = g64 - unit64 * signed_dot64.unsqueeze(-1)
```

Before reduction it MUST reject mismatched/non-`(N,4)` shapes, non-finite inputs, and quaternion
rows at or below the already-frozen `MIN_QUATERNION_NORM`. `ACTIVE_NORM` remains exactly `1e-12`.
No epsilon, clamp, dtype, device transfer, reduction order, comparison, or tolerance may change.

The helper returns the four float64 tensors `numerator`, `denominator`, `fraction`, and
`projected_gradient`. Existing step-zero fields map to those names. Existing per-step projection
fields map `numerator`, `denominator`, and `fraction` to `removed_numerator`,
`removed_denominator`, and `removed_fraction`; their diagnostic hash remains the ordered
`tensor_collection_hash` of those three mapped float64 tensors. The helper's float64
`projected_gradient` is diagnostic evidence only where the original protocol already stores it.

Producer and validator MUST use the same helper output for serialization, exact tensor equality,
maximum reduction, threshold comparison, and diagnostic hashes. Hash field names and ordering stay
as in the original artifact schema. Exact `torch.equal` validation remains mandatory. One-field
numeric or hash tampering MUST still fail closed.

This helper may replace the duplicated removed-gradient diagnostic calculations in both the
step-zero prerequisite and Phase-A projection-step paths. It MUST NOT alter any other diagnostic,
metric, invariant, or decision calculation.

## 5. Native optimizer and replay invariance

The actual gradient projection remains exactly the existing native-float32 equation:

```python
unit32 = torch.nn.functional.normalize(q_old32, dim=-1)
projected32 = raw_gradient32 - unit32 * (
    unit32 * raw_gradient32
).sum(dim=-1, keepdim=True)
```

Only `projected32` may be copied into `q.grad`. The float64 helper output MUST NEVER be copied into,
aliased with, or used to calculate `q.grad`, an optimizer parameter, an Adam buffer, `q_star`,
`q_new`, a loss, or replay state. The existing float32 Adam configuration, call ordering, policy
application, and float32 replay expression remain unchanged and bit-exact. No Trainer or production
source change is authorized by this retry.

The pre-repair implementation is bound for audit comparison:

```text
fd58d01ade1dcd8582acd915b1eb4478df2fc52e105d2ede1b51079d68cdc747  benchmarks/quaternion_gauge_ablation.py
e8c33135be51c56a5335d0e410b63f8bc5c3ea13020e799ce66279a9d905456a  tests/test_quaternion_gauge_ablation.py
26105043c13453b7904c6ec8626cce2622f580485ffb335f99ec59a52a3a1d36  tests/test_quaternion_gauge.py
3bb73a2071ff3525c07c0d1a57387ecccb5b5f16a3cc18398091b2606752053f  src/rtgs/optim/trainer.py
```

Implementation may change only the harness and its focused ablation test as required by Sections
4, 6, and 7. `tests/test_quaternion_gauge.py`, `src/rtgs/optim/trainer.py`, and all other production
sources MUST retain the hashes above. The independent implementation review MUST compare the
pre/post harness and explicitly attest that the native projection, Adam update, policy equations,
and replay equations are unchanged.

## 6. Mandatory adversarial tests

Before a seal, focused CPU tests MUST add and pass all of the following without constructing an
official scene or seed:

1. A deterministic, non-axis-aligned float32 `(N,4)` adversarial `q` and finite nonzero `g` for
   which `F.normalize(q, dim=-1).to(float64)` is not tensor-equal to
   `F.normalize(q.to(float64), dim=-1)`. The test MUST assert that this ordering distinction is
   actually present, so an insensitive fixture is invalid.
2. An independent scalar/explicit float64 reference that first promotes the raw float32 `q` and
   `g`, then normalizes/dots/norms/clamps/divides. Every shared-helper output MUST be tensor-equal
   to that reference. At least one helper numerator or fraction MUST differ from the forbidden
   normalize32-then-promote diagnostic path.
3. A producer-style payload built from the helper, JSON round-tripped, reconstructed as float32,
   and accepted by the real raw-evidence validator using exact equality and the real diagnostic
   hash.
4. Separate copies with exactly one numerator value changed, exactly one fraction value changed,
   and only the diagnostic hash changed. Each MUST be rejected fail-closed by the real validator.
5. An observational-purity test with two identical float32 parameters and Adam states. Calling the
   diagnostic helper before the frozen native projection in one branch MUST leave raw inputs
   unchanged, produce tensor-equal projected32 gradients, and produce tensor-equal `q_star` and
   ordered Adam state tensors/hashes versus the branch that does not call it.
6. A replay test confirming the real validator/replay still reconstructs the raw gradient as
   float32, applies the frozen native projection expression, and exactly reproduces the stored
   float32 `q_star` and Adam state. Diagnostic float64 tensors must not enter this path.

Existing fail-closed, source-binding, namespace, full Phase-A recomputation, and Trainer-current
regression tests remain required. Passing a toy test is implementation evidence only, never an
experimental outcome.

## 7. Fresh Retry-2 namespace and exact constants

The updated harness MUST use these exact constants:

```text
PREREGISTRATION = benchmarks/results/20260716_quaternion_gauge_iter2_PREREG.md
IMPLEMENTATION_REVIEW = benchmarks/results/20260716_quaternion_gauge_iter2_IMPLEMENTATION_REVIEW.md
DEFAULT_SEAL = benchmarks/results/20260716_quaternion_gauge_iter2_SEAL.json
PHASE_A_ATTEMPT = benchmarks/results/20260716_quaternion_gauge_iter2_PHASE_A_ATTEMPT.json
PHASE_B_ATTEMPT = benchmarks/results/20260716_quaternion_gauge_iter2_PHASE_B_ATTEMPT.json
```

`PREREGISTRATION_SHA256` MUST be this file's externally reported post-write SHA-256. The fresh seal
note is `benchmarks/results/20260716_quaternion_gauge_iter2_SEAL_RESULT.md`.

The exact Phase-A prefix grammar is
`^\d{8}T\d{6}Z_cpu_quaternion_gauge_iter2$`, directly under `benchmarks/results/`. A fresh prefix
`<UTC>_cpu_quaternion_gauge_iter2` derives exactly:

```text
<prefix>_audit.json
<prefix>_audit_RESULT.md
<prefix>_invalid.json
<prefix>_invalid_RESULT.md
```

The exact Phase-B output grammar is
`^\d{8}T\d{6}Z_cpu_quaternion_gauge_iter2_ablation\.json$`, directly under
`benchmarks/results/`; its note is `<stem>_RESULT.md`.

Use these exact Retry-2 artifact types:

```text
quaternion_gauge_iter2_implementation_seal
quaternion_gauge_iter2_phase_a_attempt
quaternion_gauge_iter2_phase_a_audit
quaternion_gauge_iter2_phase_a_invalid
quaternion_gauge_iter2_phase_a_scientist_review
quaternion_gauge_iter2_phase_b_attempt
quaternion_gauge_iter2_phase_b_ablation
```

Bindings are prospective and stage-specific. The fresh seal MUST bind the original and Retry-2
preregistrations, fresh implementation review, sealed sources/source aggregate, environment, full
verification, effective configurations, and the five immutable old-artifact bindings in Section 2;
it does not self-bind or bind a future marker. Each fresh marker MUST bind the already-verified
fresh seal and that stage's exact inputs and preflighted outputs. Each result and its derived review
MUST bind the fresh seal and the relevant already-consumed marker; the strict machine Phase-A
review additionally binds its derived human audit and valid Phase-A artifact exactly as inherited
from the original protocol. Artifact validators MUST require Retry-2 types and paths exactly; no
original-attempt type, seal, marker, output, or review may satisfy a Retry-2 gate.

All writes are exclusive/atomic and append-only. Preflight MUST reject if any derived result/note
or the relevant fresh marker already exists. A failed Retry-2 consumes its Phase-A marker and
namespace; another attempt would require another prospectively frozen preregistration, review,
seal, markers, and output namespace.

## 8. Chronology and execution gates

The only authorized order is:

1. Freeze this file and externally record its SHA-256.
2. Implement only Sections 4, 6, and 7.
3. Obtain a fresh outcome-blind implementation review at the exact path above with an exact
   `Verdict: PASS`. It must bind the old failure/audit, this file, the source diff, all invariant
   hashes, the adversarial tests, and the unchanged float32 optimizer/replay path.
4. Run the original full CPU verification suite unchanged: Ruff check, Ruff format check,
   `pytest -q -m "not slow"`, docs sync, and `git diff --check`.
5. Create the fresh seal and note. The seal must bind sources, environment, full verification,
   effective configurations, original protocol hash, Retry-2 preregistration hash, review hash,
   and old-artifact bindings. No scientific action may precede the completed seal write.
6. Run one fresh Phase A under its fresh marker/output prefix.
7. Independently audit any valid Phase-A artifact. The human audit path derived from
   `<prefix>_audit.json` is `<prefix>_audit_AUDIT.md`; the exact machine review path is
   `<prefix>_audit_AUDIT.json` and must use the Retry-2 scientist-review type and exact binding
   schema inherited from the original.
8. Run Phase B only if the unchanged raw Phase-A recomputation authorizes it, the derived fresh
   human audit exists and is hash-bound, and its strict fresh machine-review JSON gives exact
   `phase_b_execution_clearance: true`. Phase B must create its fresh marker before constructing
   any official seed.

An invalid Retry-2 artifact cannot authorize Phase B. A passing implementation review or seal is
not a scientific outcome. A valid Phase A still has no production-default authority. The original
Phase-B interpretation and post-result audit requirements remain unchanged.

## 9. Frozen commands

After implementation review PASS, use the existing harness CLI with only fresh paths:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  benchmarks/quaternion_gauge_ablation.py seal \
  --output benchmarks/results/20260716_quaternion_gauge_iter2_SEAL.json

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  benchmarks/quaternion_gauge_ablation.py audit \
  --seal benchmarks/results/20260716_quaternion_gauge_iter2_SEAL.json \
  --output-prefix benchmarks/results/<UTC>_cpu_quaternion_gauge_iter2

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  benchmarks/quaternion_gauge_ablation.py run \
  --seal benchmarks/results/20260716_quaternion_gauge_iter2_SEAL.json \
  --phase-a benchmarks/results/<PHASE_A_UTC>_cpu_quaternion_gauge_iter2_audit.json \
  --review benchmarks/results/<PHASE_A_UTC>_cpu_quaternion_gauge_iter2_audit_AUDIT.json \
  --output benchmarks/results/<FRESH_UTC>_cpu_quaternion_gauge_iter2_ablation.json
```

`<UTC>`, `<PHASE_A_UTC>`, and `<FRESH_UTC>` are literal placeholders here, not authorized
filenames. Each execution must substitute a real UTC timestamp matching the frozen grammar.

## 10. Stop conditions and frozen self-review

Stop before sealing or execution if any of these occurs: original or retry prereg hash drift; old
artifact/audit hash drift; reuse or existence of a consumed marker/namespace; any scientific
configuration or decision diff; any Trainer/production-source diff; normalize-before-promote in a
removed-gradient diagnostic; float64 diagnostic influence on native projection/Adam/replay; relaxed
equality/tolerance; missing adversarial or tamper test; failed full verification; non-PASS
implementation review; or source/environment drift during seal/execution.

Executability self-review at freeze: the repair has one named helper and a total operation order;
serialized dtype reconstruction is explicit; diagnostic and optimizer paths are separated; exact
schemas, constants, regexes, artifact types, commands, hashes, chronology, and fail-closed gates are
specified; and the original protocol supplies every unchanged scientific detail by exact hash.
No outcome was consulted or generated while writing this file.
