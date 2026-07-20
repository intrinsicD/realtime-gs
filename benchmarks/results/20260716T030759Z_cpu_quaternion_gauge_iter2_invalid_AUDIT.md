# Independent scientist audit: Quaternion Retry-2 invalid Phase A

Verdict: PASS

Retry disposition: **PERMISSIBLE WITH CONDITIONS**

This PASS applies only to the integrity, provenance, and fail-closed disposition of the consumed
invalid artifact. It does **not** validate a Phase-A optimizer result, establish radial-gauge
materiality, select a quaternion policy, authorize Phase B, or support a production/default claim.
The Retry-2 Phase-A namespace is consumed and immutable.

## Claim disposition

| # | Claim | Evidence scope | Disposition |
|---|---|---|---|
| 1 | The Retry-2 attempt was prospectively sealed, uniquely marked, and wrote the bound invalid JSON/note pair. | Seal, marker, invalid artifact, note, exact hashes, chronology, and source/environment validation | **Confirm** |
| 2 | The retained preparation and prerequisite records for seeds 0/1/2 are internally valid. | Raw-recomputable serialized fields, reductions, hashes, schedules, selection, perturbation, and gradients | **Narrow**: all raw-recomputable fields pass; training/render pixel tensors are not serialized, so their producer hashes cannot be independently replayed from this artifact alone |
| 3 | The post-optimizer construction invariant failed because native float32 canonicalization changes direction at a scale far above the inherited float64 covariance tolerance. | Deterministic step-zero reconstruction from retained preparation fields plus sealed source inspection | **Confirm** for every seed/scale at step zero; later checkpoint magnitudes are not serialized and are not reconstructed |
| 4 | Phase A established whether ambient Adam's radial gauge is material or whether a candidate is better. | No arm, trajectory, checkpoint, AUC, reduction, or decision record is retained | **Not authorized** |
| 5 | Phase B may execute. | Invalid artifact type; no valid Phase-A pair, human/machine clearance, Phase-B marker, or Phase-B output exists | **Forbidden** |
| 6 | A further append-only validity retry is scientifically permissible. | Invalid disposition exposes binary construction failures but no materiality/utility outcome | **Narrow**: permissible only under the prospective repair conditions below; the original `2e-12` contract cannot be silently relaxed in this namespace |

## Exact evidence and bindings

The following bytes were independently hashed during this audit:

```text
f1ba26d2520e6f78731b404babe0e091f2341d16ab5e30607b25ba32692c764e  benchmarks/results/20260716_quaternion_gauge_PREREG.md
fe201606b878cb29b4502a283dde78a30c3d2dab9a0efa091c83be3b95bfe4f3  benchmarks/results/20260716_quaternion_gauge_iter2_PREREG.md
f23708072e6746e7e0e714020d3e6d0a31bf132150e2fa57b14a9f1a63bac818  benchmarks/results/20260716_quaternion_gauge_iter2_IMPLEMENTATION_REVIEW.md
f27d5b69ecd658a03dc8685f60097b42dfabf5f258d11521151141db49c2bdf3  benchmarks/results/20260716_quaternion_gauge_iter2_IMPLEMENTATION_REVIEW_ADDENDUM_1.md
1c18438ff76330c58f5a78519cf8c833e4375ecb279ffb05460664ae1baa7d62  benchmarks/quaternion_gauge_ablation.py
60d3648747d4d7803b83b6bebc8e741b159ef415acc1983c2060137458c6df52  tests/test_quaternion_gauge_ablation.py
26105043c13453b7904c6ec8626cce2622f580485ffb335f99ec59a52a3a1d36  tests/test_quaternion_gauge.py
3bb73a2071ff3525c07c0d1a57387ecccb5b5f16a3cc18398091b2606752053f  src/rtgs/optim/trainer.py
e8169892c708678b91ea59589f76363b9512d6a48c70e718c37eea41b5b78f6c  benchmarks/results/20260716_quaternion_gauge_iter2_SEAL.json
09334199fdd5cf92a1ab2d1e3abd60aba0bb2690b9f6a5c5f58bb5d9a9337cc9  benchmarks/results/20260716_quaternion_gauge_iter2_PHASE_A_ATTEMPT.json
56df44d380ede52dba568b068685d9ffd1dbd625fe9ef92e8f31559660e0af0b  benchmarks/results/20260716T030759Z_cpu_quaternion_gauge_iter2_invalid.json
36649c101f9e859508edb167388dc9db4310cd812245f6d48045faa7483761c4  benchmarks/results/20260716T030759Z_cpu_quaternion_gauge_iter2_invalid_RESULT.md
```

The verified seal has canonical payload digest
`11915807a334a80f23e4313adb74696e684242847c44175e01add525b961861b`, full sealed-source
aggregate `dee5d8a21ac66f23216057daa914f15acc2ced4b3712ed97060d6b8b1b992414`, and verification
record digest `118808e8d77b0886669e1681d82b4ddcd9c301495973ece9c30ee7fd98ce2f95`. The marker's loaded-source
subset aggregate is `ea349f3585668cc96960c894fbddb3a1b5b987f566954c823a2bc710de710e19`;
every loaded path/hash is a matching subset of the sealed domain. The invalid artifact's retry
provenance binding is
`cd484a670f1f2410538591216fd0a51c21bb5a58f76c7d67c8fb876e1017940b`.

The sealed harness's `load_and_verify_seal()` passed after reconstructing the official CPU
environment. It independently rechecked both preregistration hashes, both implementation-review
records, the canonical seal digest, byte-level source map and aggregate, the prior consumed-attempt
bindings/required absences, frozen defaults, and the environment fingerprint. The marker binds the
exact command, fresh output prefix, all four derived output paths, seal summary, source domain,
and environment. The invalid artifact has the exact 13-key binding schema, and every seal-derived
value plus the marker byte hash matches.

## Chronology and isolation

The observable chronology is monotonic:

| Event | Time |
|---|---|
| Retry-2 preregistration frozen | `2026-07-16T04:23:56+02:00` |
| Base implementation review PASS | `2026-07-16T04:50:03+02:00` |
| Pre-seal recovery addendum PASS | `2026-07-16T05:05:09+02:00` |
| Fresh seal written | `2026-07-16T03:07:42+00:00` |
| Fresh Phase-A marker consumed | `2026-07-16T03:08:05+00:00` |
| Invalid artifact written | `2026-07-16T03:08:48+00:00` |

The seal records successful repository-wide Ruff, format, non-slow CPU pytest, docs-sync, and
`git diff --check` before the scientific marker. Its environment is CPU execution with
`CUDA_VISIBLE_DEVICES=""`, four Torch/OMP/MKL threads, deterministic algorithms enabled,
Torch `2.9.0+cu128`, and no loaded gsplat, StructSplat, or StructSplat adapter module. The dirty
repository is acceptable here because the seal preserves the exact executed source map and
aggregate rather than relying on revision identity alone.

The invalid JSON's top-level key set is exactly
`artifact_type, bindings, environment, failure, seeds, timestamp_utc`. Its three seed records have
exactly `seed, preparation, prerequisites`; no serialized `arms`, trajectories, checkpoints, AUC,
materiality or Phase-A decision exists anywhere in the tree. The failure string itself exposes 72
binary validity labels: 45 current-policy normalized-copy failures (three seeds x three scales x
five checkpoints) and 27 canonical-policy step-zero-equivalence failures (three seeds x three
scales x three policies). Those labels are construction-validity evidence, not optimizer quality
or materiality evidence.

The invalid note is byte-for-byte the note derived by the sealed harness. The bound valid Phase-A
JSON/note, both scientist-review files, the Retry-2 Phase-B marker, and every Retry-2 Phase-B
JSON/note are absent. The authorization code additionally requires a valid
`quaternion_gauge_iter2_phase_a_audit` artifact and strict machine clearance, so this invalid type
cannot enter Phase B even if renamed or supplied manually.

## Independent recomputation of retained evidence

Using only the invalid JSON and the exact sealed harness, I reconstructed every raw-recomputable
preparation field: initialization/target tensors and hashes, float64 covariance, anisotropy and
ranked selection, perturbation axes/norms/delta/product, schedules, aggregate hashes, and all
serialized step-zero gradient reductions. `validate_phase_a_preparation()` and
`recompute_prerequisite_validity()` returned `passed: true` with no failures for all three seeds.

| Seed | Init rows | Eligible / selected | Active gradient rows at `0.25/1/4` | Max tangent residual | Removed-gradient max | Scaled-gradient max |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 845 | 845 / 128 | 127 / 127 / 127 | `4.9001901212e-7` | `4.9001901212e-7` | `0` |
| 1 | 854 | 854 / 128 | 127 / 127 / 127 | `2.9820469631e-7` | `2.9820469634e-7` | `0` |
| 2 | 825 | 825 / 128 | 127 / 127 / 127 | `4.6755400402e-7` | `4.6755400403e-7` | `0` |

The retained radial/antipodal representation records recompute to pass. Their stored rotation,
covariance, and render reductions are all zero, their pooled reductions are internally exact, and
their hashes have valid SHA-256 form. Separate in-memory tamper probes changed one target
covariance value, one removed-gradient ratio, only the projected-gradient hash, and only the
top-level prerequisite pass flag; the real validators rejected each with the corresponding raw-
evidence mismatch.

Evidence boundary: training images/cameras, fit histories, and representation render pixels are
not serialized in the invalid artifact; only their hashes and raw scalar render reductions are.
Accordingly, their pixel-level producer hashes cannot be regenerated without reconstructing an
official seed, which this outcome-blind audit deliberately did not do. This does not weaken the
invalid disposition, but it narrows the retained prerequisite claim to internal/raw-recomputable
validity rather than a standalone replay-complete render package.

## Failure diagnosis

The inherited original protocol requires Phase-A step-zero and normalized-copy covariance
equivalence at both maximum absolute and relative Frobenius error `<=2e-12`. The candidate policies
perform entry canonicalization in native float32:

```text
q_candidate = F.normalize(c * q_perturbed, dim=-1)
```

The covariance audit then promotes each stored float32 tensor to float64 and normalizes again in
`quat_to_rotmat`. A second float32 normalization is not idempotent in direction. Reconstructing the
mandatory step-zero candidate from each retained `q_perturbed` gives:

| Seed | Max direction delta after common float64 normalization | Covariance max abs error | Relative covariance error | Frozen gate |
|---:|---:|---:|---:|---:|
| 0 | `1.0770374703e-8` | `6.1272389593e-10` | `4.4032617238e-10` | fail |
| 1 | `2.0618824148e-8` | `2.0513717704e-9` | `1.2022659046e-9` | fail |
| 2 | `1.5078977900e-8` | `9.5304763560e-10` | `7.2764233996e-10` | fail |

These values are identical across `c=0.25,1,4` because those power-of-two scalings are exact in
float32 before the same normalization. Every reconstructed covariance error exceeds `2e-12` by a
large margin; covariance alone therefore forces all 27 canonical construction failures even if
every render reduction passes. The same deterministic calculation proves the step-0
current-policy normalized-copy audit must fail for all seeds/scales.

The remaining current-policy checkpoint tensors and their equivalence records were intentionally
stripped, so their exact error magnitudes cannot be audited or reconstructed. Static source
inspection shows that every such checkpoint uses the same `F.normalize(float32 q)` followed by
the same float64 covariance comparison, which is consistent with the 45 binary failure labels,
but this audit does not infer or recreate hidden checkpoint values. No optimizer metric is needed
for the diagnosis: the retained step-zero construction already proves that the numerical contract
is infeasible for the mandatory native-float32 canonicalization on all three official inputs.

This is distinct from the first attempt's removed-gradient producer/validator ordering mismatch.
Retry-2 correctly repaired that path: the retained step-zero removed-gradient evidence now
recomputes exactly through the shared promote-before-normalize float64 helper. The new failure is
in the separately inherited physical-equivalence tolerance.

## Scientific and execution boundary

No valid Phase-A materiality outcome exists. In particular, nothing here supports a claim about:

- current ambient Adam being materially gauge-dependent or gauge-insensitive;
- entry canonicalization, unit retraction, tangent-displacement retraction, or gradient projection
  improving or harming optimization;
- any self-target AUC, final covariance, effective learning-rate, or radial-displacement threshold;
- held-out utility, real-scene behavior, CUDA/gsplat parity, runtime, memory, density interaction,
  export semantics, or a production default.

Phase B is unconditionally forbidden. No human audit of this invalid artifact can substitute for
the missing valid Phase-A result and strict machine-readable execution clearance.

## Conditions for any further retry

A further append-only attempt is scientifically permissible because the retained artifact exposes
only prerequisites and binary construction failures, not the materiality/utility outcomes that a
repair could tune toward. It is **not** permissible to edit or reuse Retry-2, or to call the repair
"scientifically unchanged" while silently changing the inherited `2e-12` gate. The original
preregistration explicitly freezes that threshold and forbids tuning it after failure.

Before another official action, a new prospective preregistration must:

1. bind the exact Retry-2 preregistration, reviews/addendum, seal, marker, invalid JSON/note, and
   this audit; retain all prior namespaces byte-identically;
2. define a fresh preregistration, implementation review, seal, attempt markers, artifact types,
   and output namespace;
3. explicitly amend only the construction/normalized-copy numerical contract (or the exact
   canonicalization representation) and justify it from float32 error analysis, retained
   preparation fields, and/or nonofficial generic probes—not from unavailable checkpoint
   magnitudes or hidden optimizer outcomes;
4. preserve all seeds, scenes, arms, optimizers, schedules, checkpoints, materiality metrics,
   thresholds, Phase-B utility/safety gates, and the production default unless separately and
   prospectively justified;
5. add an outcome-free pre-optimizer feasibility check for the actual entry canonicalization so
   this deterministic step-zero failure is caught before all arm trajectories run;
6. retain exact raw-evidence recomputation, tamper rejection, source/environment binding, and
   diagnostic separation from native float32 optimizer state; and
7. obtain a new independent implementation PASS and fresh full-verification seal before a single
   fresh Phase-A attempt. Any valid result still needs a separate scientist audit and strict
   machine clearance before Phase B.

The numerical contract should be fixed once with an analytic or broad outcome-neutral precision
margin; it must not be iteratively relaxed until an official trajectory passes.

## Checks actually executed

- Read `CLAUDE.md`, the complete `realtime-gs-results-audit` skill, both quaternion
  preregistrations, both Retry-2 implementation reviews, the seal, marker, invalid note, relevant
  harness validators, focused tests, quaternion math, and optimizer seam hashes.
- Recomputed SHA-256 for every artifact/source listed above and scanned for forbidden valid
  Phase-A/review/Phase-B paths.
- Strict-loaded the seal, marker, and invalid JSON; checked exact key sets, output derivation,
  canonical aggregates, source subset, artifact bindings, note derivation, environment, and
  absence of outcome fields.
- The first read-only seal-verifier invocation correctly rejected an audit process that had not
  enabled deterministic algorithms. Re-running with the official environment and
  `torch.use_deterministic_algorithms(True)` passed all seal checks.
- Recomputed all raw-recomputable preparation/prerequisite evidence for seeds 0/1/2 and ran four
  independent tamper probes; all untampered records passed and all tampered records failed closed.
- Reconstructed only the deterministic step-zero canonicalization/covariance invariant from the
  retained fields. No official seed, fit, lift, optimizer arm, or hidden checkpoint was rerun.
- Ran
  `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_quaternion_gauge_ablation.py tests/test_quaternion_gauge.py`:
  all 54 focused CPU tests passed.

No CUDA/GPU test, performance benchmark, Phase-A `audit` action, Phase-B `run` action, official
seed reconstruction, documentation edit, production-source edit, test edit, seal creation, or
machine-clearance write was performed.
