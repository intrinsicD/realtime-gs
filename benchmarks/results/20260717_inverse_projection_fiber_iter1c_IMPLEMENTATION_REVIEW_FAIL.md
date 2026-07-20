# Iteration 1c pre-root implementation review — FAIL

Reviewed: 2026-07-17, before any iter1c official root was constructed.

Recommendation: **FAIL; official execution remains blocked.**

The reviewer confirmed that the scientific objectives, reductions, association semantics,
gates, denominators, 400-update schedule, cumulative-RSS label, fresh official roots, static
checks, and focused test commands conform. The following implementation defects must be
repaired and independently re-reviewed before official execution.

## High-severity blockers

1. The exact iter1b protocol imported by iter1c was neither included in the runtime source
   closure nor checked against the three SHA-256 values frozen in the iter1c preregistration.
2. Paired initialization applied the free arm's numerical covariance tolerance to fiber arms;
   it did not enforce the fiber contract's byte-identical realized mean and covariance hashes.
3. A false combined validity sentinel could continue to scientific aggregation and publish
   `FAIL`, although every sentinel failure is required to publish `INVALID`.
4. INVALID publication replaced any existing aggregate or terminal path, so a late unowned
   collider could be overwritten instead of preserved.
5. Official GT, camera, projected-target, source, and depth hashes remained in memory until
   successful aggregation; an early post-root rank/fitting failure therefore lacked a durable
   per-replicate input receipt.

## Medium-severity blockers

6. Preflight checked the ambient default dtype but not the ambient default device, despite the
   CPU-only protocol and CPU provenance claim.
7. The verification receipt used an official namespace/schema while recording development
   roots, contrary to the rule that development receipts use the development namespace and a
   schema containing `development`.
8. The start/end source check was followed by fresh reads of preregistration and verification
   files, leaving a source-drift race between the checked closure and top-level hashes.
9. Atomic file helpers fsynced file contents but not containing directories after link, rename,
   or unlink mutations.

All nine findings are implementation/lifecycle conformance repairs. None changes the frozen
scientific question, arms, data-generating process, objective, optimizer, metric, gate, or claim
boundary. No official iter1c scene root, depth root, rank result, fit, metric, gate, or result
artifact was constructed or inspected during this review.
