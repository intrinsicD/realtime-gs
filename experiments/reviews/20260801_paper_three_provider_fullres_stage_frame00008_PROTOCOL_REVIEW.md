# Prospective Protocol Review

- Task ID: `20260801_paper_three_provider_fullres_stage_frame00008`
- Protocol SHA-256: `77b4e3a0cf137b21162ed9a52294aebce72a52d7fb8be0b44816640b54d02a2b`
- Reviewer: `Hegel`
- Verdict: `rejected`
- Outcome Access: `none`

## Scope

This V3 review covers only the owner-authorized bounded repair of Hegel's three V2 blockers:
complete reviewed-tree binding with metadata-only descendants, structured failed-v2 publication
for every trusted post-initialization root phase, and a live guard/receipt for the first canonical
compact load. It also verifies that the accepted BENCH-019 and optimizer-mechanism lineages are
inside the reviewed base. It does not assess any RTGS-008 outcome. Even an approved run on this
single outcome-exposed frame could support only a source-bound within-frame description of
provider-native fidelity and visible signatures; it could not support provider superiority,
generalization, compact-VRAM superiority, a production default, Original 3DGS/COLMAP performance,
or paper-level quality.

## Checks

- Verified clean review branch `rtgs/008-review-v3` at frozen package
  `78f84197664bdcbaf8566b1ce23882567f928717`. Reviewed implementation checkpoint
  `73dcc5f6ea4c9adb2ee99abd1184333e74b333e2` is its ancestor; its tree is
  `c955838ea5d9369fbde5190eaba9e9b6338c82af`, and the freeze changes only three paths from the
  task-declared eight-path metadata allowlist.
- Recomputed protocol digest
  `77b4e3a0cf137b21162ed9a52294aebce72a52d7fb8be0b44816640b54d02a2b`. Contract and data
  validation pass. Independently rehashed all 85 unique regular sealed files: 78 `.rtgsv` plus
  seven JSON files, exactly 24,831,997 bytes, with no mismatch. Data-seal SHA-256 is
  `645e2991232232db4425daa14a10fe5239263532d18060cf530859857b23507a`.
- Confirmed accepted RTGS-009 commit `3992a3473751d961e2e7b526e47a64732bd93e4b`, accepted
  RTGS-010/BENCH-019 commit `d8b1cc926b72255affedde46889cbd4e7f66fd5c`, optimizer mechanism
  commit `ab356672cc2c32b329d85d8ac7e32f3e95ce1f22`, and optimizer integration
  `3f86eb689a383149b19e1dc80b73cab0cf6214d1` are all ancestors of the reviewed base. The
  BENCH-019 source files match the accepted RTGS-010 tip, and the RTGS-008 driver/focused-test
  bytes at the reviewed base match repair commit `e2795fe868b02d33eacdfe55700b097432e9e2df`.
- Preserved the V1 and V2 rejected artifacts byte-for-byte at SHA-256
  `bed74d7d261cc1c829485a6e3c647c0a2ef99bd2ef50a8a38ec0b05bbd1316dc` and
  `c64a4b413ef80846fd0efd9ca644c25995025552cfeb4de495dfe7d0a4c8f815`, respectively.
- Verified the committed-tree guard accepts the reviewed base and allowed review metadata while
  rejecting a clean committed behavior change, a widened allowlist, and a source-lock mismatch.
  An independent temporary-Git counterexample then changed a tracked behavior file without
  committing it: `HEAD` and the lock stayed unchanged, the worktree became dirty, and
  `_source_binding_passes` still returned true.
- Independently exercised binding, sealed-data, and environment exceptions in isolated temporary
  roots. Each exception inside the current root `try` reached the one-shot publisher and emitted
  schema-valid failed-v2 sources. A separate initialized-looking temporary root with malformed task
  JSON raised before that `try` and emitted no failed-root publication; a missing lock is rejected
  by the same pre-`try` boundary.
- Verified the first canonical compact load is now performed by `_initialize_arm` under a live
  `NoImageGuard`. Its field semantics and passing denial record are read from the initializer
  receipt and embedded as `first_compact_load_guard` in the canonical input-boundary receipt. The
  fresh-process negative control makes an attempted `.png` open in that first load fail.
- Passed all 18 focused paper-driver/initializer tests and the 59-test broader outcome-free
  experiment-contract, compact-view, initializer, density, and paper-driver suite. The final full
  repository verification gate passed with the rejected V3 metadata installed.
- A concurrent outcome-blind Popper draft independently recorded the same dirty-tree and pre-`try`
  counterexamples. The exact uncommitted draft had SHA-256
  `73bcfebe5c5887f33e4ba35408e2965f825107b5c1228d0de57521f3446276ef`. The frozen metadata
  allowlist does not authorize a separate V3 artifact path, so this canonical Hegel review records
  that forensic digest rather than widening the reviewed source boundary.

## Findings

The guarded-first-load blocker is closed, and the repair improves the other two boundaries, but
the protocol remains rejected because their advertised fail-closed guarantees are not exact:

1. **Blocking / high -- live execution bytes are not bound to the reviewed tree.**
   `_source_binding_passes` validates ancestry, `base..HEAD` paths, and equality between `HEAD` and
   the commit recorded by `init-run`, but it never checks the current index/worktree. The
   `official_clean_source` fields are historical values captured in `task.lock.json`. An
   uncommitted tracked behavior change therefore passes and can be imported by the canonical
   Python process; an untracked module on an importable execution path has the same gap. Recheck
   live tracked and untracked execution-tree cleanliness at root and worker binding, or bind and
   compare the live execution-tree bytes, before calling this exact reviewed-tree execution.
2. **Blocking / high -- post-initialization failure publication still starts after root metadata
   preconditions.** `_orchestrate` resolves paths, requires the run directory and lock, and loads
   task JSON before entering its failure-publication `try`. Missing/corrupt task or lock metadata
   can therefore leave an initialized attempt without the promised root failure receipt and v2
   diagnostic sources. Move every operation after resolving the exact canonical root under one
   exception boundary and make the publisher capable of emitting a minimal schema-valid failure
   record when task or lock loading itself fails.

The scientific design, data seal, provider semantics, integrated lineage, first-load repair, and
in-`try` binding/data/environment publication are otherwise coherent. No implementation repair is
authorized by this review, and the protected matrix must not be initialized or executed.

## Protected Actions Not Taken

The reviewer did not invoke `init-run`, invoke the canonical run or worker, create or enumerate the
official RTGS-008 run root, train a cell, render a downstream model, or access any downstream
metric, preview, report, viewer, model, or result artifact. Outcome Access remained `none`. Checks
were limited to frozen protocol/source/input files, outcome-free unit tests, static inspection,
and isolated temporary counterexamples.
