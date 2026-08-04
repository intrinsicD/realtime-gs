# Prospective Protocol Review

- Task ID: `20260801_paper_three_provider_fullres_stage_frame00008`
- Protocol SHA-256: `77b4e3a0cf137b21162ed9a52294aebce72a52d7fb8be0b44816640b54d02a2b`
- Reviewer: `Popper`
- Verdict: `rejected`
- Outcome Access: `none`

## Scope

This V3 review covers only the owner-authorized bounded repair of Hegel's three V2 blockers: exact
reviewed-tree binding with metadata-only descendants, one-shot structured failure publication for
every trusted post-initialization root phase, and a live guard/receipt for the first canonical
compact load. It also verifies that the integrated BENCH-019 and optimizer-mechanism lineages are
inside the reviewed base. It does not assess any RTGS-008 outcome. Even an approved run on this one
outcome-exposed frame could support only a source-bound within-frame description of
provider-native fidelity and visible signatures; it could not support provider superiority,
generalization, compact-VRAM superiority, a production default, Original 3DGS/COLMAP performance,
or paper-level quality.

## Checks

- Verified clean review branch `rtgs/008-review-v3` at frozen package
  `78f84197664bdcbaf8566b1ce23882567f928717`. Reviewed implementation checkpoint
  `73dcc5f6ea4c9adb2ee99abd1184333e74b333e2` is its ancestor, and the package changes only the
  three currently declared review-metadata paths.
- Recomputed protocol digest
  `77b4e3a0cf137b21162ed9a52294aebce72a52d7fb8be0b44816640b54d02a2b` and independently
  rehashed all 85 unique sealed files: 78 `.rtgsv` plus seven JSON files, exactly 24,831,997
  bytes, with no mismatch. Data-seal SHA-256 is
  `645e2991232232db4425daa14a10fe5239263532d18060cf530859857b23507a`.
- Confirmed V3 repair commit `e2795fe868b02d33eacdfe55700b097432e9e2df`, accepted BENCH-019
  lineage `d8b1cc926b72255affedde46889cbd4e7f66fd5c`, optimizer integration
  `3f86eb689a383149b19e1dc80b73cab0cf6214d1`, and optimizer mechanism commit
  `ab356672cc2c32b329d85d8ac7e32f3e95ce1f22` are all ancestors of the reviewed base.
- Preserved Hegel's V2 rejection byte-for-byte at
  `experiments/reviews/20260801_paper_three_provider_fullres_stage_frame00008_PROTOCOL_REVIEW_V2_REJECTED.md`,
  SHA-256 `c64a4b413ef80846fd0efd9ca644c25995025552cfeb4de495dfe7d0a4c8f815`.
- Passed all 18 focused paper-driver and initializer tests. The committed-descendant negative
  control correctly accepts the reviewed base and allowlisted review metadata while rejecting a
  committed behavior change and a widened allowlist. Binding, sealed-data, and environment
  exceptions inside the current `try` reach the root publisher, and the publisher emits the
  expected diagnostic sources in its isolated test.
- Verified the first canonical compact load is now performed by `_initialize_arm` under a live
  `NoImageGuard`; its field semantics and passing denial record are read from the initializer
  receipt and embedded as `first_compact_load_guard` in the canonical input-boundary receipt. The
  fresh-process negative control makes an image attempt in that first load fail.
- Ran two additional outcome-free adversarial counterexamples. An uncommitted edit to a
  behavior-bearing tracked source file leaves `HEAD` and the task lock unchanged and is accepted
  by `_source_binding_passes`. Separately, malformed task JSON at an otherwise initialized-looking
  exact root raises before the root `try` and produces zero `_publish_failed_run` calls.

## Findings

The guarded-first-load blocker is closed, and the V3 changes improve the other two boundaries, but
the protocol remains rejected because their advertised fail-closed guarantees are still not exact:

1. **Blocking / high -- execution can use a dirty, unreviewed source tree.**
   `_source_binding_passes` validates ancestry, `base..HEAD` paths, and equality between `HEAD` and
   the commit recorded by `init-run`, but it never checks the current index/worktree. The official
   cleanliness fields are historical values captured in `task.lock.json`. In a temporary Git
   counterexample, the reviewed base passed; changing tracked `src/behavior.py` without committing
   left `source_commit` equal to `HEAD`, and `_source_binding_passes` still returned true. The
   canonical Python process would execute those dirty bytes. Recheck current tracked and untracked
   source cleanliness at every root/worker binding, or bind and compare the live execution-tree
   bytes, before calling this exact reviewed-tree execution.
2. **Blocking / high -- post-initialization root failure publication still has pre-`try` gaps.**
   `_orchestrate` resolves paths, requires the run directory and lock, and loads the task JSON
   before entering its failure-publication `try`. A missing lock after initialization therefore
   raises directly, and a malformed task produced `JSONDecodeError` with zero publisher calls in
   the independent counterexample. This is precisely the drift/binding class for which the failed
   root receipt is needed. Move every operation after resolving the exact canonical root under one
   exception boundary and make the publisher capable of emitting a minimal schema-valid failure
   record when task or lock loading itself fails.

The scientific design, data seal, provider semantics, integrated lineage, first-load repair, and
in-`try` binding/data/environment failure tests are otherwise coherent. No implementation repair
is authorized by this review, and the protected matrix must not be initialized or executed.

## Protected Actions Not Taken

The reviewer did not invoke `init-run`, invoke the canonical run or worker, train a cell, render a
downstream model, or access any RTGS-008 downstream metric, preview, report, viewer, or result
artifact. Outcome Access remained `none`. Checks were limited to frozen protocol/source/input
files, outcome-free unit tests, static inspection, and isolated temporary counterexamples.
