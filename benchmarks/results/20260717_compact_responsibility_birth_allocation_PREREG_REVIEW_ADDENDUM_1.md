# Compact residual-responsibility birth allocation — amendment 1 review

Verdict: PASS

Unresolved findings: none

## Bound artifacts and concurrency chronology

- Pre-amendment preregistration:
  `cb384fb560cffae23550b6b4975a3fb439c0a05bb6997a079696830587b11bb9`.
- Initial independent FAIL review, now preserved verbatim at
  `20260717_compact_responsibility_birth_allocation_PREREG_REVIEW_INITIAL_FAIL.md`:
  `804036c7fdcd1c82a163f7551c34a134d4e6cd4a0f6bd4d00ceb851ff8550b66`.
- Concurrent premature PASS preserved at
  `20260717_compact_responsibility_birth_allocation_PREREG_REVIEW.md`:
  `2ec29eeb5b0d5824bc7ec3c234fe4f01fa8c23a9fcb8dc164fccd54395c6d214`.
- Amended preregistration reviewed here:
  `e6f34080320459f74b0c6f20634c94697b74bffe4bfb6cb807f6e35fcc8a3427`.

The initial FAIL was published first at the designated review path. A concurrent reviewer then
replaced that path with the preserved premature PASS before amendment 1 existed. Amendment 1 was
appended afterward, preserved the original FAIL bytes at the new append-only path, invalidated
the premature PASS as authorization, and required this fresh addendum review. This re-review
started only after all four hashes above existed. At review time the experiment harness,
implementation review, seal, both attempt markers, both phase results, Phase-A audit, and run
directory were absent.

## Closure check

Amendment 1 fully resolves the three FAIL findings:

1. It freezes atom types, integer/string encodings, separators, domain prefix, exact
   `evaluation_bank` label, ordered view/measure parts, and bank metadata.
2. It defines `primary_C` and `safety_C` explicitly and supplies an ordered, mutually exclusive,
   exhaustive terminal decision map, including mixed uniform-risk failures.
3. It freezes the final-PLY count, dtype, finite-value, elementwise tolerance, and comparison
   arithmetic without a result-dependent reduction or field-specific repair.

These clarifications do not alter an arm, score, quota, stream root, surgery, gate threshold, or
claim boundary, and they introduce no contradiction with the frozen VJP, score, phase, or
once-only lifecycle contracts. No implementation was performed and no official or focused seed,
schedule, bank, score, selection, training state, surgery, or outcome was generated or
inspected.
