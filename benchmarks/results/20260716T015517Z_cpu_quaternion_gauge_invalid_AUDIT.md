# Independent scientist audit: consumed invalid Quaternion Phase-A artifact

Verdict: PASS

Audited outcome-blind on `2026-07-16T04:05:03+02:00` under the repository
`realtime-gs-results-audit` procedure.

This verdict is only for the integrity and fail-closed disposition of the invalid artifact. It is
not a valid Phase-A result, materiality conclusion, candidate result, or implementation success.

- Phase-A scientific outcome: **none authorized**.
- Phase-B execution: **forbidden**.
- Production/default implication: **none**.
- Append-only retry: **scientifically permissible only under the outcome-neutral conditions below**.

I did not invoke the harness `seal`, `audit`, or `run` actions; did not construct or rerun an
official seed; did not create, reuse, delete, or modify an attempt marker; did not inspect an
unavailable optimizer-arm outcome; and did not change any source, test, protocol, threshold, gate,
documentation, or ARA file.

## Claim disposition

| # | Claim | Kind and scope | Bound evidence | Disposition |
|---|---|---|---|---|
| 1 | The JSON/note are the authentic output of the consumed sealed Phase-A attempt. | Proven artifact integrity | Invalid JSON/note, Phase-A marker, seal, preregistration, implementation review | **Confirm** |
| 2 | The attempt stopped fail-closed and exposes no valid result or materiality decision. | Proven schema/namespace property | Invalid JSON keys and siblings; harness exception branch | **Confirm** |
| 3 | All retained preparation, representation, and step-zero-gradient prerequisites are internally valid. | Measured serialized prerequisite evidence for seeds 0/1/2 | Raw invalid JSON revalidated by the sealed harness | **Confirm**, prerequisite scope only |
| 4 | The reported failure is a producer/validator numeric-representation mismatch, not evidence about the quaternion hypothesis. | Code-path causal diagnosis plus nonofficial toy reproduction | Sealed harness lines 1864-1885, 2210-2238, and 2755-2764 | **Confirm**; no official arm value is inferred |
| 5 | The current attempt can authorize Phase B. | Scientific/gating claim | Invalid artifact and absent valid audit/review | **Retire** for this consumed namespace |

No blocking, major, or minor integrity finding remains for the invalid-artifact disposition.

## Seal, source, marker, and artifact integrity

Strict duplicate-key/non-finite JSON loading passed for the seal, marker, and invalid artifact. The
seal's unsigned canonical digest independently recomputes to
`1dd0659e2941fa7c5acf504dade6f4fc8d07c404dd4aaa17c4e61f8faa0cc26f`, exactly its stored
`sha256`. Its 75 unique sealed paths exactly match its source-hash mapping, every current file
digest matches, and the mapping independently canonicalizes to source aggregate
`d170a2463d17d679cc6a4b4839c3e4ec6600ad71453086d8f9313f70747f911b`.

The marker contains 38 loaded source paths. They form a proper subset of the sealed mapping, every
loaded digest matches its sealed counterpart, the preregistration/review/harness/`pyproject.toml`
entries are present, and the mapping independently canonicalizes to
`ec53c70b1a2a962dd4c871430dd8ed4f8fd50b678df20b8eb8d89f686caafe39`. The marker's full
sealed aggregate, embedded verified-seal object, environment fingerprint, canonical output prefix,
four derived output paths, and literal command all validate through the real authorization-side
checker.

The invalid artifact's exact ten-field binding object matches:

- the frozen preregistration and final PASS implementation-review paths and byte hashes;
- the seal's canonical digest, byte hash, path, and source aggregate;
- the canonical Phase-A marker path and byte hash; and
- the current artifact/note namespace recorded before marker consumption.

The seal, marker, and invalid artifact environment records are exact-equal: CPU execution, CUDA
hidden, deterministic algorithms enabled, four Torch/OMP/MKL threads where frozen, and optional
gsplat/StructSplat modules absent. The dirty repository state is not hidden: the seal records Git
revision `2dddca4aff59702341af9faceefa76ad2505dd83`, `dirty=true`, tracked-diff SHA-256
`446db64fe8345e68e812acd5b946ac857fccb33085c71795da9679d32625e184`, the exact source map,
and the successful five-command verification record.

Chronology is ordered correctly. The final preregistration clarification preceded the final
implementation PASS review; the seal was created at `2026-07-16T01:55:04+00:00`, the marker at
`2026-07-16T01:55:27+00:00`, and the invalid artifact at
`2026-07-16T01:56:10+00:00`. The marker is consumed and remains present.

The human result note is byte-for-byte equal to `invalid_phase_a_note()` for the stored payload.
Its statement is appropriately narrow: the invariant failed and no materiality conclusion is
authorized.

## Failure boundary and evidence exposure

The artifact has exactly these top-level keys:

```text
artifact_type, bindings, environment, failure, seeds, timestamp_utc
```

Its type is exactly `quaternion_gauge_phase_a_invalid`, and its failure is exactly:

```text
stage: optimizer_or_reduction
message: stored projected-gradient fractions differ from raw q/gradient
```

Seeds are exactly `[0,1,2]`. Each seed has only `seed`, `preparation`, and `prerequisites`; there is
no `arms`, trajectory, checkpoint, AUC, decision, materiality, effective-configuration, or
`phase_b_authorized` field. The valid Phase-A audit JSON/note, Phase-B attempt marker, and Phase-B
ablation output are absent.

The sealed producer first completed all three preparations and global prerequisites, then executed
the optimizer arms in memory, and then entered `_phase_a_invariants()`. The exception was raised
inside raw step validation before `recompute_phase_a_decision()` could run. The caught
`optimizer_or_reduction` branch deliberately stripped all in-memory arms and wrote only the
preparation/prerequisite records. `main()` does not print arm metrics. Thus no valid result,
materiality decision, or optimizer trajectory is exposed by the artifact, note, or CLI path.

All serialized prerequisite evidence was independently recomputed through
`validate_phase_a_preparation()` and `recompute_prerequisite_validity()`. For each of seeds 0, 1,
and 2, the reconstruction validated the full initialization/target fields and covariance hashes,
training hashes, diagnostic selection, perturbation, schedule, four representation records and
their per-view/pooled reductions, three raw step-zero gradient records, removed-gradient record,
and scaled-gradient identities. Each seed independently returned exactly
`{"passed": true, "failures": []}`. This confirms prerequisites only; it does not recover or imply
any stripped optimizer-arm outcome.

## Causal diagnosis: diagnostic arithmetic representation mismatch

The exact error string has one throwing site: `derive_projection_removed_fractions()`. The producer
and validator apply mathematically equivalent operations in a different floating-point order.

The producer at the gradient-projection step does:

```text
u32 = normalize(q_old_float32)
n_producer = abs(dot(float64(u32), float64(raw_gradient32)))
d_producer = norm(float64(raw_gradient32))
fraction_producer = n_producer / d_producer
```

The validator reconstructs the serialized float32 arrays directly as float64 and does:

```text
q64 = float64(q_old_float32)
g64 = float64(raw_gradient32)
n_validator = abs(dot(normalize(q64), g64))
d_validator = norm(g64)
fraction_validator = n_validator / d_validator
```

In floating-point arithmetic,
`float64(normalize_float32(q)) != normalize_float64(float64(q))` for ordinary rows. The
denominators agree, but the numerators and fractions need not be bit-identical. The validator then
uses `torch.equal`, so even a harmless last-bit difference raises the stored failure before checking
the diagnostic hash.

A two-row nonofficial tensor probe reproduced the exact exception. It observed maximum absolute
differences of about `4.00e-11` in the numerator and `3.02e-9` in the fraction. In the same probe,
the actual native-float32 projected gradient
`g32-u32*dot(u32,g32)` replayed bit-for-bit. The sealed arm producer and its Adam replay likewise
both apply the actual projection in float32. Therefore this failure identifies inconsistent
serialization/validation arithmetic for a diagnostic reduction; it is not evidence of an invalid
quaternion projection, failed scientific threshold, arm ordering problem, optimizer difference,
materiality result, or candidate result.

The exact official failing row arrays are intentionally absent from the fail-closed artifact, so no
official numeric magnitude is asserted here. The causal classification rests on the unique sealed
throw site, the deterministic operation-order mismatch immediately upstream, the matching
float32 optimizer replay path, and an exact tiny-tensor reproduction of the same exception.

## Append-only retry disposition

An append-only retry is scientifically permissible because no arm outcome or materiality decision
was serialized or exposed and the required change is diagnostic arithmetic only. It is not
permissible to repair or overwrite this namespace, reuse either the seal or consumed marker, or
continue to Phase B from this artifact.

A valid retry must do all of the following:

1. Preserve this seal, marker, invalid JSON/note, and audit unchanged.
2. Freeze an explicit retry preregistration before execution, cite this failure and audit, and use
   fresh seal, Phase-A marker, and output names.
3. Make producer and validator call one shared projected-gradient diagnostic helper. Consistent
   with the frozen rule that audit reductions use float64, that helper should reconstruct the raw
   float32 `q_old` and gradient, promote both to float64, then normalize, dot, norm, clamp, divide,
   serialize, and hash in that one order.
4. Leave the actual gradient-projection arm operation in native float32 exactly unchanged. Do not
   alter `q.grad`, Adam, entry state, arm order, scene, seeds, subset, perturbation, radial scales,
   schedule, checkpoints, LR, thresholds, metrics, or decision gates.
5. Add a nonofficial adversarial unit test with a float32 quaternion for which
   `normalize32-then-promote` differs from `promote-then-normalize`. Require the shared helper's
   producer payload to pass raw recomputation, and require one-field numerator/fraction/hash
   tampering to fail closed.
6. Obtain a fresh independent implementation review and seal only after the exact full verification
   passes. Any fresh valid Phase A must receive its own independent scientist audit and strict
   machine clearance before Phase B can be considered.

No threshold relaxation, tolerance substitution for the frozen raw equality, reuse of hidden arm
values, or post-outcome scientific change is justified.

## Commands and limitations

Read-only commands included:

```text
sha256sum <invalid-json> <invalid-note> <phase-a-marker> <seal> <seal-note> <prereg> <review> <harness> <focused-tests> <trainer> <pyproject>
jq <schema/key-only summaries> <invalid-json> <phase-a-marker> <seal>
rg -n <failure/projection/invalid-path symbols> benchmarks/quaternion_gauge_ablation.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python - <<'PY'
# strict load; load_and_verify_seal; real marker validator; exact binding/environment/note checks;
# validate_phase_a_preparation and recompute_prerequisite_validity for serialized seeds only
PY
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python - <<'PY'
# two-row nonofficial producer/validator arithmetic reproduction; no scene or official seed
PY
git diff --check
```

The first manual binding assertion used an absolute marker path while the protocol correctly stores
the canonical repository-relative path. A key-by-key diagnostic showed that as the sole difference;
the corrected exact-path assertion and the real marker validator both passed. This was a reviewer
probe construction error, not an artifact finding.

No official arm was rerun, no stripped arm value was inspected or reconstructed, and no GPU/CUDA
claim was tested. The seal's recorded full verification is provenance evidence, not GPU evidence.

## Hash binding

```text
8381979a9b6fba958e34d8a2d2e4210dc783ede808edd2fa88faddf3b4b53739  benchmarks/results/20260716T015517Z_cpu_quaternion_gauge_invalid.json
e796f5e4cfd7bf686d7db2ee0c691c86c415c0bbeebc68d01d809030c1e963a4  invalid JSON canonical payload
34adccfe91650cd821dc99c0f6c4cdf7e5668ac4b89faa0e4ad4466c95d56a61  benchmarks/results/20260716T015517Z_cpu_quaternion_gauge_invalid_RESULT.md
c6a7c663edff15114c11b714ed6342e1ebd1e72b535a565e6d3861ce9e7868dc  benchmarks/results/20260716_quaternion_gauge_PHASE_A_ATTEMPT.json
35fd623acbcdde0bd18ed092e23220de20c5e8795aef4b8b4707470a1aae2aa5  Phase-A marker canonical payload
146193dc0783b01d5fada9608e276845a1aea6e8e44ba4ed53772adc47ef4ad8  benchmarks/results/20260716_quaternion_gauge_SEAL.json
1dd0659e2941fa7c5acf504dade6f4fc8d07c404dd4aaa17c4e61f8faa0cc26f  seal unsigned canonical digest
d0e2df31f25fd1d1eda631c9a9a75270b5df6668d7c8ebd9109c149bbe7d57f7  benchmarks/results/20260716_quaternion_gauge_SEAL_RESULT.md
f1ba26d2520e6f78731b404babe0e091f2341d16ab5e30607b25ba32692c764e  benchmarks/results/20260716_quaternion_gauge_PREREG.md
dfda96d46ad2405195b2036526dea7023835ca775021e9c34e53781cfd0843d7  benchmarks/results/20260716_quaternion_gauge_IMPLEMENTATION_REVIEW.md
fd58d01ade1dcd8582acd915b1eb4478df2fc52e105d2ede1b51079d68cdc747  benchmarks/quaternion_gauge_ablation.py
e8c33135be51c56a5335d0e410b63f8bc5c3ea13020e799ce66279a9d905456a  tests/test_quaternion_gauge_ablation.py
26105043c13453b7904c6ec8626cce2622f580485ffb335f99ec59a52a3a1d36  tests/test_quaternion_gauge.py
3bb73a2071ff3525c07c0d1a57387ecccb5b5f16a3cc18398091b2606752053f  src/rtgs/optim/trainer.py
1e8c3d7d532fa47f11e7766f88872ca714fe06b428948ae8098655802fcc4995  tests/test_optim.py
7426f166742203b907c992abc24c0d7503a0da7783eb59ccb2515c51e5735b2c  pyproject.toml
d170a2463d17d679cc6a4b4839c3e4ec6600ad71453086d8f9313f70747f911b  sealed source-map aggregate
ec53c70b1a2a962dd4c871430dd8ed4f8fd50b678df20b8eb8d89f686caafe39  loaded source-map aggregate
a7873c546b2ca73a726df6096696ead8c35d6c32e47e5bf0640e915ef9ead994  sealed verification-record aggregate
```

The SHA-256 of this audit note is reported externally after the append-only write.
