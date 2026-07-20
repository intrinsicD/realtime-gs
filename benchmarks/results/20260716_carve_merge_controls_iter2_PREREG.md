# Preregistration: Carve merge-controls provenance retry 2

## Chronology and failed-attempt boundary

Frozen at `2026-07-16T00:56:46+02:00`, after the first official Phase-A process failed
closed but before changing implementation, running a diagnostic, reconstructing an in-memory
record, or observing any scientific count, ratio, metric, hash, gate, or candidate outcome.

Pre-implementation provenance clarification at `2026-07-16T01:01:18+02:00`, still before any
repair or scientific execution. Independent review required the exact Phase-B command, a
retry-specific seal artifact type, and explicit runtime checks that the failed JSON/note remain
absent; it also corrected the first paragraph's initially postdated minute timestamp to the
file-birth second above. These amendments only make the already-frozen recovery executable and
auditable. They change no computation, condition, gate, or interpretation.

The scientific protocol remains the original
`benchmarks/results/20260715_carve_merge_controls_PREREG.md` (SHA-256
`4eda7a69442bddc25cd5edce85125942f91adc52f3d62806f050a64b854b3efe`). Its first
complete implementation was sealed as
`benchmarks/results/20260715_carve_merge_controls_SEAL.json` (SHA-256
`a802d14170944944e2cee0b766a44f635aad59bdf2412bd771629b06e4d0d923`). The official
Phase-A command consumed
`benchmarks/results/20260715_carve_merge_controls_PHASE_A_ATTEMPT.json` (SHA-256
`4e784e9626bf9d3025be1e8ed2c362ba75471538a02b2babae93257af3cf7b5c`) and named
`benchmarks/results/20260715T225457Z_cpu_carve_merge_controls_audit.json`.

The process printed only `Phase A: preparing seed 0`, `Phase A: preparing seed 1`, and a
traceback. While validating the seed-1 record, it raised `RuntimeError: Phase-A reported residual
differs from per-view raw sums`. It exited before preparing seed 2, aggregating a decision,
verifying end-of-run provenance, writing JSON, writing a companion note, or printing any
scientific value. The named JSON and note do not exist. Phase B was never claimed or run. No
partial Python state survived process exit, and no partial outcome may be reconstructed from logs,
reruns, temporary instrumentation, or the deterministic setup.

## Frozen representation-only diagnosis and repair

The failure is in a fail-closed evidence validator, not in fitting, lifting, arm construction,
rendering, or a scientific gate. `materiality_render_audit` forms each aggregate with an explicit
left-to-right float64/Python-float fold (`total = total + value`). The raw-evidence validator
re-forms the same mathematical sum with Python's built-in `sum` and then requires bit equality.
Those two floating reduction implementations are not guaranteed to round bit-identically; seed 1
exposed the mismatch. The per-view raw evidence was therefore rejected before serialization even
though no evidence of data tampering was reported.

Retry 2 makes exactly one semantic repair:

1. add a small deterministic helper that starts from Python float `0.0` and applies
   `total = total + float(value)` in the serialized view order;
2. use that helper both when forming the reported materiality totals and when recomputing them
   from per-view evidence before authorization; and
3. retain exact equality between the two results, so altered per-view values, totals, or ratios
   still fail closed.

This preserves the original accumulator's order and numerical result. It is not permission to use
Python's built-in `sum`, `math.fsum`, reorder views, introduce a tolerance, round a value, change
dtype, omit raw evidence, or weaken a tamper check for these three materiality reductions. Focused
tests must cover a finite sequence for which another summation implementation can round
differently, exact left-fold reproduction, and continued rejection of a changed per-view
numerator or reported aggregate.

No scene, split, seed, stage-1 fit, Carve configuration, parity lift, voxel key, moment formula,
control construction, tie rule, count, schedule, renderer, metric, checkpoint, training order,
threshold, materiality criterion, success/safety gate, stopping rule, or interpretation changes.
The held-out callback, raw metric numerators, float64 moment audit, and raw-evidence authorization
logic in the first seal remain mandatory. The failed run cannot supply a warm cache, fitted tensor,
raw tensor, arm, denominator, selected threshold, or prior expectation to Retry 2.

## Fresh seal, review, and append-only namespace

Before the retry, the repaired harness and tests must pass the complete original verification
gate and a new independent implementation review. The new seal must bind both preregistrations,
the immutable predecessor seal and attempt marker with the hashes above, the repaired harness,
all repository-owned loaded source, tests, environment, revision, and dirty diff. Runtime must
verify the fixed predecessor hashes and new source aggregate before claiming the new marker. Seal
creation and every runtime phase must also require that both failed-attempt targets remain absent:
`benchmarks/results/20260715T225457Z_cpu_carve_merge_controls_audit.json` and
`benchmarks/results/20260715T225457Z_cpu_carve_merge_controls_audit_RESULT.md`.

Fresh artifacts are:

- seal: `benchmarks/results/20260716_carve_merge_controls_iter2_SEAL.json`;
- seal artifact type: `carve_merge_controls_iter2_implementation_seal`;
- Phase-A marker: `benchmarks/results/20260716_carve_merge_controls_iter2_PHASE_A_ATTEMPT.json`;
- Phase-B marker: `benchmarks/results/20260716_carve_merge_controls_iter2_PHASE_B_ATTEMPT.json`;
- Phase-A output: `<UTC>_cpu_carve_merge_controls_iter2_audit.json`;
- Phase-B output, only after the unchanged all-seed Phase-A gate and an independently bound
  scientist clearance: `<UTC>_cpu_carve_merge_controls_iter2_ablation.json`;
- Phase-A machine review artifact type:
  `carve_merge_controls_iter2_phase_a_scientist_review`.

The result artifact types must likewise be retry-specific:
`carve_merge_controls_iter2_phase_a_audit` and
`carve_merge_controls_iter2_phase_b_ablation`. The old seal, marker, and absent named result remain
untouched. Refuse every overwrite. An interrupted Retry-2 attempt requires another append-only
preregistration and namespace.

Official commands retain the original CPU environment:

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python benchmarks/carve_merge_controls_ablation.py seal --output benchmarks/results/20260716_carve_merge_controls_iter2_SEAL.json`

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python benchmarks/carve_merge_controls_ablation.py audit --seal benchmarks/results/20260716_carve_merge_controls_iter2_SEAL.json --output <UTC>_cpu_carve_merge_controls_iter2_audit.json`

If and only if the unchanged Phase-A decision passes and an independent
`realtime-gs-results-audit` review binds and clears that exact result, the original Phase-B
protocol may run with the exact command:

`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python benchmarks/carve_merge_controls_ablation.py ablate --seal benchmarks/results/20260716_carve_merge_controls_iter2_SEAL.json --audit <UTC>_cpu_carve_merge_controls_iter2_audit.json --phase-a-review <bound-review.json> --output <UTC>_cpu_carve_merge_controls_iter2_ablation.json`

Every valid Retry-2 result receives a new
independent scientist audit before any documentation, ARA claim, follow-up selection, or default
decision.
