# Iteration 1c repaired-implementation re-review — FAIL

Reviewed: 2026-07-17, before any iter1c official root was constructed.

Recommendation: **FAIL; retire the iter1c namespace and all six official roots.**

Reviewed sealed hashes:

- harness: `7f3c1214fa56f8cb40bea2bf120a877d1d4e4206553e059cb8f92d32498ed7ae`
- focused tests: `99a17adeb1bb733351fbdde34b6b046f08af3707feb552fc9a1a637ae6a81445`
- verification receipt: `4fd3afb79cb6758baa30b7ae505885d71356800f0f2b553a454771046f8677c5`

## High-severity blockers

1. Development failure-injection tests emitted official-schema receipts under the official
   namespace `rtgs.inverse-projection-fiber.iter1c.v1`, including false
   `ROOTS_STARTED` claims. Representative files occur under pytest sessions `1943`, `1944`,
   and `1945`; the complete retained inventory is in
   `20260717_inverse_projection_fiber_iter1c_POLLUTION_INVENTORY.md`. This directly violates
   the frozen rule that development tests may never emit the official namespace or schema and
   is materially the same restart condition that retired iter1b.
2. `_replace_json_if_owned` checked SHA/device/inode and then performed unconditional
   `os.replace`. Root-free fault injection inserted an external collider inside that gap; the
   helper returned success and overwrote it. Ownership capture separately hashed and then
   statted the path, allowing a same-content replacement between those operations to be
   captured as owned.
3. The lifecycle transition could commit `ROOTS_STARTED` and then raise during directory fsync
   or post-replacement ownership capture. Because `roots_started=True` was assigned only after
   helper return, that committed transition could skip durable INVALID publication.

## Medium-severity blockers

4. Raw scene inputs were durable before rank, but common-constructor and per-arm initialization
   hashes/deltas remained in memory. An initialization-equivalence failure occurred before the
   failing receipt was serialized, contrary to the frozen all-deltas-and-hashes requirement.
5. The final source check preceded result canonicalization and publication. A source edit in
   that window could escape detection, and the INVALID builder could re-hash after a detected
   drift and lose the first observed mismatching closure.

## Verified conforming

Historical iter1b hashes, strict fiber versus numerical free initialization semantics, combined
sentinel rejection, raw input receipts, ambient dtype/device checks, objectives, reductions,
controls, optimizer schedule, metrics, gates, ordinary 45-file development closure, and parent
directory fsync coverage otherwise conformed. All 84 focused/parity tests passed, but that same
test session exposed the namespace pollution.

No iter1c official scene root, depth root, rank result, fit, metric, gate, or scientific outcome
was constructed or inspected. Independent root-free review used only development roots,
including `91/92` and `101/102`.
