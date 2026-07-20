# Compact residual-responsibility birth allocation iter2 — lifecycle failure audit

Date: 2026-07-17

Verdict: **UNAVAILABLE — official lifecycle consumed; namespace permanently closed**

## Finding

The exact preregistered public seal command was executed once:

```text
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33 \
  .venv/bin/python benchmarks/compact_responsibility_birth_allocation_iter2.py seal
```

It returned nonzero from the pre-publication verification stage with:

```text
seal verification failed
```

The failing verification item was the full repository check, `scripts/verify.sh`. Its wider test
collection encountered ambient/order-sensitive test behavior. This is a lifecycle and
implementation-verification failure, not a frozen scientific-gate outcome.

The preregistration permits no retry, resume, alternate output, replacement worker, or seed
substitution. It also states that any lifecycle failure yields `UNAVAILABLE`. The failed official
command therefore consumes iter2 even though it failed before publishing a seal or attempt marker.

## Bound pre-attempt evidence

- iter2 preregistration:
  `benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_PREREG.md`
- preregistration SHA-256:
  `e0be823718b1b074d0c720d1cccf8800a18bd72580877fb1e1f44c30dcb5806c`
- independent preregistration review:
  `benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_PREREG_REVIEW.md`
- preregistration-review SHA-256:
  `59b60d6516ee3547978bb41cf5faa51fc2353f262c136feec71fc6a14def22a5`
- independent implementation review:
  `benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_IMPLEMENTATION_REVIEW.md`
- implementation-review SHA-256:
  `217c443e4a2f17291653b7a742d9702b79dfc492875930bc101fbb56d6e96e52`
- implementation-reviewed source aggregate:
  `79c8f374e416a93a6572d262a09dfa41b4bd851d15596f49c5ac80e3ffa5b5de`

The implementation review's earlier affected four-suite PASS was outcome-free and only made the
reviewed source eligible to attempt sealing. It did not supersede the broader verification
required inside the subsequently executed official seal command.

## Chronology and publication boundary

The harness first checked namespace absence and both independent reviews, ran the static and
dynamic pre-marker root-use proofs, captured its binding state, and then invoked three
verification items: the affected four-suite pytest command, `scripts/verify.sh`, and
`git diff --check`. At least `scripts/verify.sh` returned nonzero, so the harness raised
`ProtocolInvalid("seal verification failed")`.

The harness creates and validates the executed-source archive only after all verification items
return zero. It publishes the seal only after that archive and a final binding check succeed.
Consequently, this attempt stopped before either publication. Phase A is reachable only through a
valid published seal and creates its exclusive marker before any matching official random root can
reach a generator. That code was never reached.

No machine-readable command receipt was published because the verification records still existed
only in the failing process. In particular, the absent executed-source archive means that no
later working-tree state may be represented as an exact archive of the failed process.

## Confirmed absent artifacts

The following frozen paths were checked after the failure and were absent:

- `benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_SEAL.json`
- `benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_EXECUTED_SOURCES.tar`
- `benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_PHASE_A_ATTEMPT.json`
- `benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_PHASE_A_RESULT.json`
- `benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_PHASE_A_AUDIT.json`
- `benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_PHASE_B_ATTEMPT.json`
- `benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_RESULT.json`
- `runs/compact_responsibility_birth_allocation_iter2_20260717`

No official training root (`78101..78103`), evaluation-bank root (`78201..78203`), split-noise
root (`78301..78303`), or shuffle root (`78401..78403`) reached its matching generator, schedule,
sampler, bank, trainer, score, selection, split, evaluator, or worker. No Phase-A score,
eligibility set, stratum, selection, bank, arm, checkpoint metric, utility metric, terminal
decision, viewer output, timing result, or quality result was produced or observed.

## Scientific claim disposition

| Claim | Disposition | Reason |
| --- | --- | --- |
| The iter2 seal passed. | Rejected. | The official seal command returned nonzero before publication. |
| Iter2 Phase A or Phase B passed or failed a scientific gate. | Unavailable; do not claim. | Neither phase began and no result artifact exists. |
| Responsibility allocation improves or harms fitting. | Unavailable; do not claim. | No official arm or outcome was produced. |
| The reviewed implementation was eligible to attempt sealing. | Retained only for the reviewed pre-seal snapshot. | The independent implementation review passed, but broader official verification later failed. |
| Correcting test ordering or ambient isolation reopens iter2. | Rejected. | A repair cannot reverse consumption of a once-only official lifecycle command. |

## Permanent disposition

The complete `compact_responsibility_birth_allocation_iter2` lifecycle is terminally closed. Its
seal command, official and focused roots, seed-domain literals, artifact paths, run directory, and
result stem may not be rerun, repaired in place, reused, overwritten, or pooled with a successor.
Post-failure test-order corrections are development evidence for a fresh namespace only.

Any scientifically equivalent successor requires an append-only preregistration with fresh,
pairwise-disjoint official and focused roots, fresh seed domains, fresh harness/test/artifact/run
paths, fresh independent reviews, and a fresh once-only lifecycle. This audit authorizes no such
execution by itself.

