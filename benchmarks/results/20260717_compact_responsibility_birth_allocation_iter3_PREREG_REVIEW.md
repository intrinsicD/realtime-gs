# Compact residual-responsibility birth allocation iter3 — preregistration addendum review

Date: 2026-07-17

Verdict: PASS

Unresolved findings: none

## Scope and immutable bindings

This is the independent, outcome-free addendum review required by the frozen iter3 lifecycle. It
binds:

| Artifact | SHA-256 |
| --- | --- |
| base iter3 preregistration | `352133e2830d921af272c472cfe41b3d7114643627fd7d585b4bef8ac2613f81` |
| initial independent FAIL review | `dab05011c2531a837873ca2f286ac86a2a580c688951d61688a466cc0a3e76ac` |
| iter3 preregistration addendum 1 | `b96dfbe572563c18fd319e665f7adf7bad0408a4347585c7118a1c4b9277ec8b` |
| original iter2 lifecycle failure audit | `b0992cf6a190b9ac9f9bde5701b09abb05af8617c0a6234182355cf49f80b0fa` |
| iter2 failure-audit addendum 1 | `f75b7943b4bf29b38d27599839e5c174ee9bf1ee98174f0695a56638feecb386` |
| imported scientific preregistration | `e6f34080320459f74b0c6f20634c94697b74bffe4bfb6cb807f6e35fcc8a3427` |
| iter2 preregistration | `e0be823718b1b074d0c720d1cccf8800a18bd72580877fb1e1f44c30dcb5806c` |

The base preregistration remains immutable. Addendum 1 supersedes only its iter2 failure-cause
chronology and extends only its seal-evidence lifecycle. The scientific question, arms, inputs,
matched birth budget, score arithmetic, topology operations, optimizer, checkpoints, banks,
metrics, thresholds, gates, estimands, claim limits, and scientific stopping rules remain
unchanged.

## Resolution of the initial FAIL findings

| Initial finding | Independent disposition |
| --- | --- |
| The frozen chronology conflated the directly observed generic seal failure, the immediate Ruff diagnosis, and later order-dependent tests. | Resolved. The iter2 failure-audit addendum and iter3 preregistration addendum separately bind exit status 1 with empty stdout and the generic final traceback, the immediate five-finding Ruff reproduction, and the three later post-failure test-isolation discoveries. They explicitly state that only Ruff caused the reproduced immediate verifier stop and that later tests did not cause the consumed official process. |
| No exclusive entry-time seal-attempt marker or bounded machine-readable seal-failure receipt was frozen. | Resolved. Addendum 1 freezes distinct `SEAL_ATTEMPT` and `SEAL_FAILURE` paths, schemas, exclusive write/sync/reread rules, command/environment/protocol bindings, ordered subprocess receipts, complete stream hashes and byte counts with bounded tails, lifecycle-stage and exception evidence, binding and artifact inventory, success binding, failure behavior, and no-retry semantics. |

The attempt marker is created only after a read-only all-path absence check and before review
validation, root-use proofs, binding construction, subprocess verification, archive creation, or
other fallible seal work. Once present, it permanently consumes the seal attempt. The failure
handler publishes its bounded receipt for every catchable later failure; inability to publish that
receipt still leaves the durable attempt marker and namespace consumed. Successful seal
publication is forbidden unless all verification items pass, bindings remain exact, the executed
source archive revalidates, the failure path is absent, and the final seal binds the unchanged
attempt marker.

These requirements close the evidence hole that allowed the failed iter2 process to terminate
with only a generic traceback. Their implementability, schema validation, exclusive-write
behavior, exception coverage, tamper rejection, and success/failure authorization paths remain
mandatory subjects of the later implementation review; they are not assumed from prose alone.

## Freshness and chronology

Repository-wide exact-word searches over source, tests, documentation, ARA, historical result
artifacts, and run metadata found the frozen iter3 roots and both seed-domain literals only in the
base iter3 preregistration. The earlier proposed candidate values are already declared
contaminated and forbidden there.

At review time:

- the iter3 harness and focused test did not exist;
- both new seal-attempt and seal-failure files were absent;
- the seal, executed-source archive, Phase-A marker/result/audit, Phase-B marker, final result,
  visualizer outputs, and run directory were absent; and
- no iter3 root had reached a generator, schedule, sampler, trainer, bank, split, shuffle,
  evaluator, worker, score, selection, state, metric, or result.

The root sets are pairwise distinct within iter3 and disjoint from the failed, iter2, focused, and
explicitly contaminated sets named by the frozen documents. The iter3 domain and lifecycle paths
are fresh. No random mechanism or official command was executed during either preregistration
review.

## Claim and authorization disposition

| Claim | Disposition |
| --- | --- |
| The iter3 preregistration is a fresh, outcome-neutral successor. | Confirmed for the bound documents and reviewed repository chronology. |
| The two findings in the initial FAIL review remain unresolved. | Rejected; both are closed by addendum 1 as detailed above. |
| The iter3 seal lifecycle now requires durable once-only attempt evidence and bounded catchable-failure evidence. | Confirmed as a frozen implementation requirement, not yet as implemented behavior. |
| Responsibility allocation improves or harms compact fitting. | Withheld; no iter3 score, bank, arm, metric, or result exists. |
| Iter3 may be sealed or run now. | Rejected. A separate independent implementation review must first bind the complete migrated source and return exact PASS with no unresolved findings. |

This PASS authorizes implementation migration and development-only focused verification under the
frozen restrictions. It does not authorize creation of either official seal lifecycle file,
official-root use, seal execution, Phase A, Phase B, a result, visualization, a scientific claim,
or a default change.

Before sealing, the independent implementation review must verify the complete source closure,
scientific-semantic identity with iter2, root/path/domain substitution, source/input/runtime/RGB
binding, the prescribed preload-bound affected and full verification, and every development-only
seal-attempt/failure/success case frozen by addendum 1. Any unresolved implementation finding
withholds sealing and cannot be repaired after an official iter3 command.
