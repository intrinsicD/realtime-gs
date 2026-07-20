# Compact residual-responsibility birth allocation iter3 — preregistration addendum 1

Frozen at: 2026-07-17T08:04:49+02:00

Status: **FROZEN, AWAITING INDEPENDENT ADDENDUM REVIEW, NOT IMPLEMENTED, NOT SEALED, NOT RUN**

## Purpose and immutable bindings

This append-only addendum resolves the two lifecycle-evidence findings in the independent initial
FAIL review. It does not edit the base preregistration, change a scientific choice, authorize
implementation, or use a random root.

| Artifact | SHA-256 |
| --- | --- |
| base iter3 preregistration | `352133e2830d921af272c472cfe41b3d7114643627fd7d585b4bef8ac2613f81` |
| independent iter3 initial FAIL review | `dab05011c2531a837873ca2f286ac86a2a580c688951d61688a466cc0a3e76ac` |
| imported scientific preregistration | `e6f34080320459f74b0c6f20634c94697b74bffe4bfb6cb807f6e35fcc8a3427` |
| original iter2 failure audit | `b0992cf6a190b9ac9f9bde5701b09abb05af8617c0a6234182355cf49f80b0fa` |
| iter2 failure-audit addendum 1 | `f75b7943b4bf29b38d27599839e5c174ee9bf1ee98174f0695a56638feecb386` |

The initial review is:

```text
benchmarks/results/20260717_compact_responsibility_birth_allocation_iter3_PREREG_REVIEW_INITIAL_FAIL.md
```

It contains exact `Verdict: FAIL` and `Unresolved findings: 2`. Its Finding 1 requires separation
of the directly observed iter2 failure from later diagnostics. Its Finding 2 requires an exclusive
entry-time seal-attempt marker and a bounded machine-readable seal-failure receipt.

The base preregistration remains normative except where this addendum explicitly corrects its
failure chronology or extends its seal lifecycle. All roots, domains, arms, formulas, budgets,
inputs, steps, metrics, gates, stopping rules, and claim boundaries remain byte-for-byte frozen by
the base document.

## Corrected iter2 chronology

The directly observed iter2 official-process evidence was only:

```text
command:
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33 \
    .venv/bin/python benchmarks/compact_responsibility_birth_allocation_iter2.py seal
exit status: 1
stdout: empty
final stderr traceback line:
  ProtocolInvalid: seal verification failed
```

The failed process published no per-command verification receipt. Therefore the process itself
establishes only a generic pre-publication seal-verification failure.

An immediate outcome-free post-failure reproduction of the embedded preload-bound
`scripts/verify.sh` stopped at `==> ruff check` with five mechanical findings in the old failed-
namespace harness and test: `I001`, two `F401` findings, and `E501` in
`benchmarks/compact_responsibility_birth_allocation.py`, plus `I001` in
`tests/test_compact_responsibility_birth_allocation.py`.

Only after those mechanical Ruff findings were corrected did a later development full pytest run
expose the three ambient/order-dependent test-isolation defects described and hash-bound in the
iter2 failure-audit addendum. Those later defects did not cause the already-failed official
process to return 1. The later corrected full-verification PASS is development evidence only.

This chronology supersedes the base iter3 preregistration's statement that ambient/order-sensitive
tests were the immediate cause. It changes no terminal disposition: imported scientific-
preregistration lines 937--942 make a failed official command consume its namespace, so iter2
remains permanently `UNAVAILABLE`.

## New frozen seal lifecycle paths

The following paths join every base-preregistered iter3 lifecycle and namespace-cleanliness check:

```text
seal attempt:
  benchmarks/results/20260717_compact_responsibility_birth_allocation_iter3_SEAL_ATTEMPT.json
seal failure:
  benchmarks/results/20260717_compact_responsibility_birth_allocation_iter3_SEAL_FAILURE.json
```

Both paths were absent immediately before this addendum was frozen. Naming them here is inert:
neither file may be created before the later, independently authorized official seal command.

At external seal-command entry, namespace cleanliness requires the seal attempt, seal failure,
seal, executed-source archive, both phase markers, both phase results, Phase-A audit, final result,
failure-audit path, visualizer outputs, and run directory all to be absent. Any present path
rejects entry without overwrite. The existence of a seal-attempt marker permanently forbids a
second seal attempt, regardless of whether a seal or failure receipt exists.

## Exclusive seal-attempt claim

After argument parsing and the read-only all-path absence check, but before review validation,
root-use proofs, binding construction, verification, archive creation, or any other fallible seal
work, the official `seal` command must exclusively create:

```text
artifact_type: compact_responsibility_birth_iter3_seal_attempt_v1
status: CLAIMED
scientific_decision: UNAVAILABLE
timestamp_utc: RFC-3339 UTC
command:
  argv: exact argv strings
  literal: exact preregistered public command
  cwd: exact absolute repository working directory
process:
  executable: exact absolute Python executable
  pid: descriptive integer
environment:
  LD_PRELOAD: exact value
  preload_path: resolved absolute path
  preload_sha256: SHA-256 of resolved preload bytes
  PYTHONPATH: exact value or null
  focused_test_environment_key_at_entry: exact value or null
protocol:
  preregistration_path
  preregistration_sha256
  preregistration_addendum_path
  preregistration_addendum_sha256
  passing_preregistration_review_path
  passing_preregistration_review_sha256
  implementation_review_path
  implementation_review_sha256
namespace:
  every frozen lifecycle path
  fresh official and focused root sets as inert integers
```

`scientific_decision=UNAVAILABLE` in the attempt marker means that claiming a seal attempt never
authorizes a scientific phase. It is not a negative result.

Creation uses an exclusive no-overwrite regular-file write, atomic complete JSON bytes, and a file
plus parent-directory sync. The command strictly reopens, parses, schema-validates, and hashes the
attempt marker as its immediately next operation. It compares the reread object with the intended
canonical object exactly. Any creation, sync, reread, parse, schema, or equality failure returns
nonzero and consumes iter3; it never deletes, rewrites, or retries the attempt path.

After a valid reread, internal seal namespace validation requires exactly this attempt marker to
be present with its claimed SHA-256 while every other iter3 output remains absent. No official
root may reach a generator during seal creation.

## Ordered verification transcript

The seal wrapper maintains an append-only in-memory transcript from immediately after the attempt
reread. Each verification item is appended as soon as its subprocess completes or times out.
The ordered items remain unchanged:

1. the focused iter3 plus point-render, compact-trainer, and optimizer pytest suites;
2. preload-bound `scripts/verify.sh`; and
3. `git diff --check`.

For every started item, the transcript records:

```text
ordinal
command argv
absolute cwd
started_at_utc
finished_at_utc
timeout_seconds or null
timed_out
returncode or null
exception type/message or null
stdout:
  byte_count
  sha256 of complete raw bytes
  tail_source_byte_count
  tail_utf8
stderr:
  byte_count
  sha256 of complete raw bytes
  tail_source_byte_count
  tail_utf8
```

Each tail is decoded with UTF-8 replacement from at most the final 8192 raw bytes. Hashes and byte
counts always cover the complete raw stream, not the bounded tail. A subprocess-spawn exception
records the exception and canonical empty-stream byte count/hash/tails. A timeout records all
bytes captured before termination and leaves `returncode=null` unless the subprocess API reports
an exact code. Timing is descriptive and cannot alter a gate, root, or scientific decision.

The wrapper does not discard a nonzero item. It completes only the same later diagnostic items
that the base seal already ordered, unless a timeout, process-spawn failure, or host interruption
makes continuation impossible. Any nonzero, timeout, exception, or missing item makes successful
seal publication forbidden.

## Mandatory bounded seal-failure receipt

After the attempt marker is valid, every later nonzero verification, timeout, exception, source/
input/runtime drift, review rejection, root-proof failure, archive failure, binding failure, seal
publication failure, or post-publication verification failure must enter one outer failure
handler. Before returning nonzero, that handler exclusively publishes:

```text
artifact_type: compact_responsibility_birth_iter3_seal_failure_v1
status: FAIL
scientific_decision: UNAVAILABLE
timestamp_utc: RFC-3339 UTC
command:
  argv
  literal
  cwd
seal_attempt:
  path
  sha256
exception:
  type
  message
  bounded_traceback_tail
failure_stage: exact enumerated lifecycle stage
verification:
  ordered list of every started verification record
binding:
  available: true | false
  stage: last successfully captured binding stage or null
  canonical_sha256: digest or null
  value: complete last successfully captured source/input/runtime binding or null
  unavailable_reason: string or null
artifact_inventory:
  captured_at_utc
  capture_boundary: immediately before exclusive SEAL_FAILURE publication
  entries:
    each frozen iter3 lifecycle path:
      state: absent | regular_file | directory | other | inspection_error
      sha256: regular-file digest when readable, otherwise null
      error: type/message or null
```

The receipt additionally repeats the exact attempt-bound preregistration, addendum, passing
preregistration review, and implementation-review paths and hashes. Its bounded traceback tail is
at most 8192 UTF-8 bytes under the same decoding rule. Verification stream records use the exact
complete hashes, byte counts, and bounded tails specified above.

The inventory samples `SEAL_FAILURE` itself as absent immediately before its exclusive
publication; after publication, the handler strictly reopens, parses, schema-validates, hashes,
and exactly compares the receipt. It must also confirm that the attempt marker still equals its
claimed bytes and hash. It may print only a canonical locator containing failure status, failure
path, failure SHA-256, and attempt SHA-256 before exiting 1.

Failure publication uses exclusive no-overwrite atomic complete JSON, file and parent-directory
sync, and immediate strict reread. The handler never removes a partial archive, seal, marker, log,
or run directory. If inventory collection, failure-receipt creation, sync, or reread itself fails,
the command preserves the original exception plus the publication exception on stderr and exits
nonzero. Inability to publish the bounded receipt does not reopen the namespace: the durable
attempt marker still consumes iter3, and no retry or alternate failure path is permitted.

Uncatchable process termination or host/storage loss cannot be promised a failure receipt. The
exclusive attempt marker is the durable fail-closed evidence for that case.

## Successful seal binding

A successful seal is permitted only when:

- the attempt marker strictly revalidates and its SHA-256 is unchanged;
- every ordered verification item exists and returned exactly zero without timeout or exception;
- source/input/runtime bindings are unchanged across verification and archive creation;
- the executed-source archive exists and strictly revalidates;
- `SEAL_FAILURE` is absent; and
- every base-preregistered seal invariant passes.

The seal payload must add:

```text
seal_attempt:
  path: benchmarks/results/20260717_compact_responsibility_birth_allocation_iter3_SEAL_ATTEMPT.json
  sha256: exact attempt-marker SHA-256
```

It also persists the complete ordered verification records defined above. Seal publication remains
exclusive and is followed by strict reread. `verify_seal()` rejects unless the attempt marker
matches the seal binding, the failure path is absent, every verification record is successful,
and every existing base seal check passes.

Every Phase-A/Phase-B authorization path calls `verify_seal()` and therefore rejects a missing or
changed attempt marker or any present `SEAL_FAILURE`. Phase markers and results must repeat the
seal-attempt SHA-256 in addition to their existing seal bindings.

## Once-only and review requirements

There is no retry, resume, attempt replacement, copied attempt, alternate failure output,
SEAL_FAILURE deletion, implicit latest-file lookup, or second seal command. A failed official
seal command consumes iter3 even if it fails before verification, cannot publish its bounded
failure receipt, or leaves a partial seal/archive.

Before implementation migration or focused execution, a new independent outcome-free addendum
review must bind:

- the base preregistration and this addendum;
- the initial FAIL review;
- the original iter2 failure audit and its addendum;
- every fresh root/domain/path and both new seal paths; and
- the exact chronology and receipt schemas above.

It must write exact `Verdict: PASS` with `Unresolved findings: none` at the base-preregistered
review path:

```text
benchmarks/results/20260717_compact_responsibility_birth_allocation_iter3_PREREG_REVIEW.md
```

The later implementation review must prove the exclusive attempt/failure behavior with
development-only temporary paths, including nonzero verification, timeout, exception, binding
drift, archive failure, seal-publication failure, failure-publication failure, tamper, preexisting
attempt/failure paths, and successful seal binding. Those tests may use only focused or unrelated
development roots and may never create either official lifecycle file.

This addendum changes lifecycle evidence durability only. It authorizes no implementation,
generator, root, seal, phase, result, visualization, claim, or default.
