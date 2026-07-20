# Iteration 1d preregistration review — FAIL

Reviewed: 2026-07-17, before any iter1d implementation or official execution.

Reviewed preregistration SHA-256:
`91d3e1c601c6eeb41f4c828e1f600c5dd7a1f52c818754745548130d9b35fe9c`.

Recommendation: **FAIL; close iter1d before implementation and official execution.**

## Blocking findings

1. The receipt-path contract promised preservation under arbitrary repeated non-cooperating
   races. Linux provides atomic `RENAME_EXCHANGE`, but no unlink/rename-if-device-and-inode
   primitive. An entry can be swapped after verification and before cleanup, so deleting only
   the prepared inode while preserving every unknown inode is impossible as an absolute
   guarantee. The protocol must define a cooperative/single-writer or bounded fault model and
   cease cleanup when recovery becomes uncertain.
2. Explicit namespace arguments did not mechanically bind every schema, status, phase, root,
   fallback, recovery manifest, prepared payload, and fault fixture to one receipt domain. A
   single immutable domain object must drive every disk producer.
3. Aggregate, terminal, and lifecycle publication lacked a frozen authoritative order and
   validity predicate. Lifecycle must be the final commit marker; partial or contradictory
   files authorize no scientific claim.
4. A multi-file source hash is not an atomic snapshot. The no-writer threat boundary must begin
   when the final closure read starts, not only after it returns, unless a filesystem snapshot
   or cooperating lock is used.

All iter1c history hashes, the fresh iter1d roots/namespace, unchanged scientific protocol,
mutation-aware root attempt, durable initialization design, and claim boundary otherwise
passed review. No iter1d implementation, generator, official root, result, metric, gate, or
scientific outcome was constructed or inspected.
