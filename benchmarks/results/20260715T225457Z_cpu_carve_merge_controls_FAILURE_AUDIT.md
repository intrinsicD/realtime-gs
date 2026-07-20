# Independent failure audit: Carve merge-controls Phase A

## Disposition

The official Carve merge-controls Phase-A attempt claimed at
`2026-07-15T22:55:03+00:00` exited with status `1` during seed 1. The failure is an
in-memory binary64 reduction-representation defect in the audit harness. It is not a frozen
materiality-gate outcome, does not authorize Phase B, and does not permit reuse of the claimed
attempt or its absent output namespace.

This note is append-only failure evidence. It is not a Phase-A result artifact and contains no
reconstructed or remembered scientific measurements.

## Bound evidence

- Base preregistration:
  `benchmarks/results/20260715_carve_merge_controls_PREREG.md`
- Base preregistration SHA-256:
  `4eda7a69442bddc25cd5edce85125942f91adc52f3d62806f050a64b854b3efe`
- Failed implementation seal:
  `benchmarks/results/20260715_carve_merge_controls_SEAL.json`
- Failed implementation-seal SHA-256:
  `a802d14170944944e2cee0b766a44f635aad59bdf2412bd771629b06e4d0d923`
- Failed sealed-source aggregate:
  `21ca5d47a4cad54c8cdf446339f174febc48018f6cee45b193569aebd40694cf`
- Failed sealed harness SHA-256:
  `edd7b35b113670c2cb097e69cb3976b8e65c0227cb742eb5d37b87a7a9f3546c`
- Failed attempt marker:
  `benchmarks/results/20260715_carve_merge_controls_PHASE_A_ATTEMPT.json`
- Failed attempt-marker SHA-256:
  `4e784e9626bf9d3025be1e8ed2c362ba75471538a02b2babae93257af3cf7b5c`
- Claimed output:
  `benchmarks/results/20260715T225457Z_cpu_carve_merge_controls_audit.json`
- Marker-bound environment: Python `3.12.9`, Torch `2.9.0+cu128`, CPU device,
  `CUDA_VISIBLE_DEVICES=""`, four Torch/OMP/MKL threads, and deterministic algorithms.

The marker records this argv:

```text
/home/alex/Documents/realtime-gs/.venv/bin/python benchmarks/carve_merge_controls_ablation.py audit --seal benchmarks/results/20260715_carve_merge_controls_SEAL.json --output benchmarks/results/20260715T225457Z_cpu_carve_merge_controls_audit.json
```

The following artifacts were confirmed absent after exit:

- `benchmarks/results/20260715T225457Z_cpu_carve_merge_controls_audit.json`
- `benchmarks/results/20260715T225457Z_cpu_carve_merge_controls_audit_RESULT.md`
- `benchmarks/results/20260715_carve_merge_controls_PHASE_B_ATTEMPT.json`

At `2026-07-16T00:57:22+02:00`, all 70 paths in the failed seal still reproduced the
sealed hashes and source aggregate exactly. The failed source must remain recoverable through its
seal plus a reverse-applicable repair record or equivalent immutable source snapshot.

## Retained process transcript

Exit status: `1`.

Standard output:

```text
Phase A: preparing seed 0
Phase A: preparing seed 1
```

Standard error:

```text
Traceback (most recent call last):
  File "/home/alex/Documents/realtime-gs/benchmarks/carve_merge_controls_ablation.py", line 1841, in <module>
    raise SystemExit(main())
  File "/home/alex/Documents/realtime-gs/benchmarks/carve_merge_controls_ablation.py", line 1829, in main
    payload = run_phase_a(args.seal, args.output)
  File "/home/alex/Documents/realtime-gs/benchmarks/carve_merge_controls_ablation.py", line 1448, in run_phase_a
    runs.append(phase_a_seed(seed))
  File "/home/alex/Documents/realtime-gs/benchmarks/carve_merge_controls_ablation.py", line 922, in phase_a_seed
    record["gate"] = phase_a_seed_gate_from_raw_evidence(record)
  File "/home/alex/Documents/realtime-gs/benchmarks/carve_merge_controls_ablation.py", line 957, in phase_a_seed_gate_from_raw_evidence
    raise RuntimeError("Phase-A reported residual differs from per-view raw sums")
RuntimeError: Phase-A reported residual differs from per-view raw sums
```

No other stdout, stderr, numeric outcome, or persisted partial payload was retained from the
producing process.

## Root cause

`materiality_render_audit` creates each per-view positive binary64 residual, immediately adds it
to a running Python-float accumulator with `denominator += residual`, and records the identical
float in the per-view evidence. It accumulates each control numerator with the same explicit
left-fold pattern.

`phase_a_seed_gate_from_raw_evidence` instead recomputes the denominator and both control
numerators from those recorded per-view floats with Python 3.12's built-in `sum`. Python 3.12
uses a compensated float summation path, so it can differ in the least significant bit from the
producer's ordered `0.0; += value` fold. The validator then requires exact equality. Seed 1
reached the first exact comparison and raised before returning its record.

The defect covers the complete materiality-reduction chain:

1. the raw residual denominator;
2. the moment-versus-voxel-representative numerator;
3. the moment-versus-global-budget-prune numerator; and
4. both ratios derived from those totals.

Repairing or relaxing only the denominator comparison would leave the two numerator comparisons
and derived ratios exposed to the same defect. A tolerance would also be broader than necessary.

## Access boundary

Seed 0 completed `phase_a_seed` in memory. Seed 1 completed fitting, raw and parity Carve
construction, arm construction and audits, train and held-out initialization evaluation, and
materiality rendering before the bookkeeping exception. Its record was not returned. Seed 2 did
not start.

The process did not reach the cross-seed `phase_a_decision`, loaded-source end check, returned
Phase-A payload, or exclusive artifact writer. No numeric measurements were printed, persisted,
reconstructed, or inspected after the failure. Nevertheless, the fixed official marker is
consumed and the retry must be disclosed as a retry rather than an unopened original attempt.

## Scientific claim disposition

| Claim | Disposition | Reason |
|---|---|---|
| Phase A passed | Unverified; do not claim | No three-seed decision or result artifact exists. |
| Phase A failed a frozen scientific gate | Unverified; do not claim | The exception preceded the gate result and was representational. |
| The attempt failed on inconsistent float reduction implementations | Confirmed | Transcript and sealed source identify the exact producer/validator mismatch. |
| Phase B is authorized | Rejected | There is no Phase-A pass artifact or independent result clearance. |
| The original marker/output may be retried | Rejected | The append-only once-only attempt is consumed despite the absent output. |
| A narrowly repaired Retry-2 is admissible | Conditional | Only under all constraints below. |

## Frozen Retry-2 requirements

Retry-2 may correct only this representation mismatch. It must preserve every scientific choice in
the base preregistration, including scenes, split, seeds, fit and Carve configurations, raw tensor,
arms, counts, metrics, thresholds, gates, stopping rules, Phase-B configuration, and interpretation.

The only admissible materiality-accounting repair is one shared explicit ordered Python-float
left-fold used by both the producer and validator:

```python
def left_fold_float64(values):
    total = 0.0
    for value in values:
        total += float(value)
    return total
```

For the denominator and both control numerators, the producer must collect the ordered per-view
terms and apply that helper; the validator must apply the same helper to the serialized terms in
the same frozen training-view order. Both ratios must be derived from those shared totals.
Built-in `sum`, `math.fsum`, reordered reduction, tolerance-based acceptance, threshold
movement, or post-outcome fallback is forbidden in this path. This preserves the failed
producer's original bit-level aggregates rather than replacing them with a different summation
policy.

Before execution, Retry-2 must also:

1. freeze an append-only Retry-2 preregistration binding this note, the base preregistration,
   failed seal, failed source aggregate, and failed attempt marker;
2. preserve a reverse-applicable exact repair diff or equivalent immutable snapshot of the failed
   sealed harness and tests;
3. add a positive-valued adversarial unit test, such as nine `0.1` terms, proving that the
   explicit left fold can differ from Python 3.12 `sum` and that producer and validator remain
   bit-identical for the denominator, both numerators, and ratios;
4. retain the existing tamper, raw-evidence recomputation, fail-closed, split, seal, and
   once-only-marker tests;
5. pass full CPU verification before creating a fresh implementation seal;
6. use fresh Retry-2-specific seal, Phase-A marker, Phase-B marker, audit output, ablation output,
   and companion-note namespaces, without overwriting or reusing any failed-attempt path;
7. bind the fresh seal and attempt markers to the Retry-2 preregistration and this audit's
   SHA-256, as well as the historical hashes above;
8. recreate seeds 0, 1, and 2 from scratch without reusing any failed-process in-memory state; and
9. keep Phase B blocked unless Retry-2 Phase A completes, passes every unchanged frozen gate, and
   receives a separate independent `realtime-gs-results-audit` clearance bound to its exact JSON,
   seal, source aggregate, and review artifact.

Any Retry-2 execution failure consumes its fresh marker and requires another append-only failure
disposition. No failure repair may be used to tune the scientific protocol.

## Post-review chronology correction

Appended at `2026-07-16T01:19:12+02:00`, before a Retry-2 seal or execution and without access to
any scientific outcome. Independent implementation review identified an impossible ordering in
requirement 1 above: the Retry-2 preregistration was already frozen before this independent failure
note was written, so that earlier file cannot literally bind this later note without a post-freeze
amendment.

The admissible chronology is therefore clarified, not scientifically changed. The already-frozen
Retry-2 preregistration (SHA-256
`fd4361ab1a53a22760db72e99614abb04206c1b639602e0015d8debde91c1203`) binds and
incorporates the base protocol, failed seal, failed source aggregate, and consumed marker. The
fresh Retry-2 implementation seal and both fresh attempt markers must bind the final SHA-256 of
this later independent audit, as requirement 7 already states. This correction supersedes only
the words “binding this note” in requirement 1; every repair, test, namespace, source, gate,
stopping, and independent-clearance requirement remains unchanged.
