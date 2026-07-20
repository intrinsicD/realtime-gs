# Compact residual-responsibility birth allocation iter2 — failure-audit addendum 1

Date: 2026-07-17

Disposition: **CORRECTED CHRONOLOGY; TERMINAL `UNAVAILABLE` UNCHANGED**

## Immutable artifacts bound

This is an append-only correction to:

```text
benchmarks/results/20260717_compact_responsibility_birth_allocation_iter2_FAILURE_AUDIT.md
SHA-256 b0992cf6a190b9ac9f9bde5701b09abb05af8617c0a6234182355cf49f80b0fa
```

The original audit is not edited because the already-frozen iter3 preregistration binds that exact
hash:

```text
benchmarks/results/20260717_compact_responsibility_birth_allocation_iter3_PREREG.md
SHA-256 352133e2830d921af272c472cfe41b3d7114643627fd7d585b4bef8ac2613f81
```

This addendum also binds the imported scientific preregistration:

```text
benchmarks/results/20260717_compact_responsibility_birth_allocation_PREREG.md
SHA-256 e6f34080320459f74b0c6f20634c94697b74bffe4bfb6cb807f6e35fcc8a3427
```

Its lines 937--942 explicitly state that any failed official command consumes the namespace, that
a bounded failure result should be written when possible with subprocess tails, that partial
evidence must not be deleted or overwritten, and that any repair requires a fresh append-only
preregistration, seeds, seal, marker, and output namespace. Those rules, rather than access to a
scientific outcome, make iter2 terminal.

## Exact official process receipt

The exact official command remained:

```text
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33 \
  .venv/bin/python benchmarks/compact_responsibility_birth_allocation_iter2.py seal
```

Its observed top-level process receipt was:

```text
exit status: 1
stdout: empty
final stderr traceback line:
ProtocolInvalid: seal verification failed
```

No machine-readable verification receipt was published. The harness retained completed child
command records only in memory and raised before its result print, executed-source archive, or
seal writer.

## Corrected immediate failure cause

The original audit incorrectly attributes the official failure to ambient/order-sensitive test
behavior. The immediate official verification failure was earlier: `scripts/verify.sh` stopped at
its first `==> ruff check` stage. A direct outcome-free reproduction identified exactly five Ruff
findings:

1. `benchmarks/compact_responsibility_birth_allocation.py`: import sorting `I001`;
2. the same historical failed harness: unused `argparse` import `F401`;
3. the same historical failed harness: unused `sys` import `F401`;
4. the same historical failed harness: overlong line 1832 `E501`; and
5. `tests/test_compact_responsibility_birth_allocation.py`: import sorting `I001`.

Both affected files belong to the older, already-failed namespace but are included by the
repository-wide Ruff command. Because `scripts/verify.sh` uses `set -e`, it stopped before its
format check, non-slow pytest collection, and docs-sync stages. Thus no order-dependent pytest
failure occurred inside the official iter2 seal process.

The seal harness had already run its separate affected four-suite pytest subprocess before the
full verifier. It also invokes `git diff --check` after `scripts/verify.sh` while building the
three in-memory verification records, then rejects if any return code is nonzero. Nothing from
those records was persisted after the generic exception.

## Later discoveries are separate post-failure evidence

Only after the five mechanical Ruff findings were corrected did a later development-only full
pytest run expose three order-dependent test-isolation defects:

1. `test_rgb_guard_denies_importlib_bypass_and_counts_attempt` asserted forbidden-module absence
   after `_without_preloaded_rgb_modules` had already restored ambient modules; the assertion now
   occurs inside that context.
2. `test_iter2_runtime_binding_has_no_unbound_local_sources` treated unrelated modules loaded by
   the full suite as part of its isolated fixture; it now monkeypatches origins and local sources
   to the expected bound closure for that focused receipt check.
3. In `test_unbound_and_shadowed_loaded_modules_are_rejected`, ambient origin violations could
   preempt the intended non-`rtgs` source-closure branch; only that branch now temporarily
   monkeypatches `module_origin_violations` to `()`. The `rtgs` unbound and shadowed-origin checks
   remain unmasked.

These post-failure discoveries explain why additional test-isolation changes were needed before a
fresh successor, but they did not cause the already-consumed official command to return 1. The
subsequent preload-bound full-verification PASS is also development evidence only and cannot
repair or reopen iter2.

## Post-failure source-aggregate status

The independent implementation review binds the pre-failure 44-file aggregate:

```text
79c8f374e416a93a6572d262a09dfa41b4bd851d15596f49c5ac80e3ffa5b5de
```

After the mechanical and test-isolation corrections, a read-only recomputation through
`reviewed_source_hashes()` produced:

```text
8e1a0483e7f80b2ca84f4c86c60dbe8a994cbe584f55fde9f9b12e82781aedf1
```

The current aggregate therefore does not equal the aggregate named by the iter2 implementation
review, and `implementation_review_passed()` returns `False`. This is expected post-failure drift,
not evidence that the failed reviewed snapshot had a scientific defect. It independently prevents
the changed tree from masquerading as the reviewed iter2 implementation, while the consumed
lifecycle rule already forbids any retry.

## Scope of correction

This addendum supersedes only the original audit's attribution of the immediate official failure
to order-sensitive tests and makes the top-level process receipt and post-failure aggregate drift
explicit. All other original findings remain:

- iter2 is permanently consumed;
- seal, executed-source archive, phase markers, phase results, final result, and run directory
  were absent;
- no official root reached a generator or scientific mechanism;
- no score, bank, arm, metric, decision, quality result, or viewer result existed; and
- no positive, negative, causal, convergence, scaling, or default claim is permitted.

No official command or root was rerun to prepare this addendum.
