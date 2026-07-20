# Compact residual-responsibility birth allocation iter3 — initial preregistration review

Date: 2026-07-17

Verdict: FAIL

Unresolved findings: 2

## Scope and bindings

This is an independent, outcome-free review of:

```text
benchmarks/results/20260717_compact_responsibility_birth_allocation_iter3_PREREG.md
SHA-256 352133e2830d921af272c472cfe41b3d7114643627fd7d585b4bef8ac2613f81
```

The review also bound:

- the imported scientific preregistration at
  `e6f34080320459f74b0c6f20634c94697b74bffe4bfb6cb807f6e35fcc8a3427`;
- the iter2 preregistration at
  `e0be823718b1b074d0c720d1cccf8800a18bd72580877fb1e1f44c30dcb5806c`; and
- the iter2 lifecycle failure audit at
  `b0992cf6a190b9ac9f9bde5701b09abb05af8617c0a6234182355cf49f80b0fa`.

Exact-word repository-wide searches over source, tests, documentation, ARA, historical result
artifacts, and run metadata found every frozen iter3 random root and both seed-domain literals
only in the iter3 preregistration. The iter3 harness, focused test, reviews, seal, executed-source
archive, phase markers/results/audit, final result, visualizer, and run directory were absent
before that file was frozen. The preregistration is therefore accepted as the first repository
occurrence of its fresh root, domain, and lifecycle namespace. No root-bearing generator,
schedule, sampler, trainer, bank, split, shuffle, evaluator, or worker was executed during this
review.

The scientific question, arms, matched birth budget, inputs, optimizer, score formulas, topology
operations, checkpoints, gates, estimands, claim boundary, and stopping rule are imported without
an identified scientific change. That is not sufficient for PASS because the following two
lifecycle-evidence findings remain unresolved.

## Finding 1 — the frozen iter2 failure chronology conflates distinct evidence

The directly observed official-command evidence is:

- the exact preregistered iter2 `seal` command was invoked once;
- its exit code was `1`;
- stdout was empty; and
- stderr ended with the traceback through `create_seal()` line 2832 and
  `ProtocolInvalid: seal verification failed`.

The failing process did not publish its per-command verification receipts. An immediate
post-failure execution of the embedded preload-bound `scripts/verify.sh` command stopped in Ruff
on the old failed-namespace harness and test. Only after those mechanical lint corrections did a
later full verification expose three ambient/order-dependent test-isolation failures.

The iter3 preregistration instead states that the full verification encountered
ambient/order-sensitive test behavior as the cause of the original nonzero command. That merges
the direct generic traceback, the immediate Ruff diagnostic, and the later post-failure test
diagnostics into one chronology. It overstates what the failed process itself recorded.

Required resolution: an append-only, outcome-free preregistration addendum must preserve all three
events separately, label the Ruff and order-dependence evidence as post-failure diagnostics, and
state that the original process exposed only the generic verification failure because no
machine-readable verification receipt survived.

## Finding 2 — no exclusive seal-attempt marker or bounded seal-failure receipt is frozen

The iter2 seal command consumed its once-only namespace before publishing a seal, attempt marker,
executed-source archive, or machine-readable failure receipt. The iter3 preregistration correctly
retains the rule that any failed official command consumes iter3 even when no artifact is
published, but its frozen lifecycle repeats the same evidence gap: it names a successful seal and
an optional prose failure audit, without an entry-time exclusive seal-attempt artifact or a
bounded machine-readable seal-failure artifact.

Required resolution: an append-only, outcome-free preregistration addendum must freeze:

1. an iter3 `SEAL_ATTEMPT` path and schema, exclusively created and strictly re-read before any
   fallible seal verification;
2. an iter3 `SEAL_FAILURE` path and schema, exclusively published on any nonzero verification,
   timeout, exception, drift, archive, or publication failure; and
3. exact receipt fields sufficient to bind the attempt, command, working directory, relevant
   preload/environment identity, timestamps, exit/exception state, ordered verification commands,
   return codes, stdout/stderr byte counts, hashes and bounded tails, and the available
   source/input/runtime entry binding.

Successful seal publication must bind the attempt marker and require the failure path to remain
absent. Failure publication must never authorize sealing or a scientific phase, and inability to
publish the bounded failure receipt must still leave the namespace consumed.

## Claim disposition

| Claim | Disposition |
| --- | --- |
| The iter3 roots, domains, and lifecycle paths were fresh when preregistered. | Confirmed for the reviewed preregistration and repository state. |
| The iter3 document reproduces the iter2 failure chronology exactly. | Rejected pending Finding 1. |
| The iter3 lifecycle guarantees a durable receipt for a pre-publication seal failure. | Rejected pending Finding 2. |
| The scientific protocol may be implemented or sealed now. | Withheld; preregistration review is FAIL. |

No implementation migration, focused-root execution, implementation review, seal, official root,
Phase A, Phase B, result, visualization, scientific claim, or default change is authorized by
this review. A later independent addendum review may return PASS only after both findings are
resolved append-only and no outcome-bearing iter3 mechanism has run.
