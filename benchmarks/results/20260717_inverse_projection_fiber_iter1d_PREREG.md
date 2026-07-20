# Inverse-projection fiber fitting, iteration 1d — transaction-corrected preregistration

Frozen at: 2026-07-17 (Europe/Berlin)

Status: **FROZEN BEFORE ITER1D IMPLEMENTATION OR OFFICIAL EXECUTION**

## Bound history and restart reason

Iteration 1d imports the complete scientific question, geometry, losses, reductions, arms,
optimizer, sentinels, metrics, gates, and claim boundary from iter1b through iter1c. Only the
receipt namespace, atomic-publication, durable-initialization, and final-source-check contracts
below change.

| Historical artifact | SHA-256 |
| --- | --- |
| closed iter1c preregistration | `a05da9a9386fd37abac6de4c7ef441a5e96665e59c8d55f82dbffe1a114d3de4` |
| iter1c preregistration review (PASS) | `77ad2e1d712eda4262a41cbb0c7ea24f3fbd1adc92717e39ac49327cd6a2c0a0` |
| iter1c first implementation review (FAIL) | `71daedf2d39201ced29c4eac2d0bc33dd538920d7309a0b70121c938a8f50944` |
| iter1c repaired verification receipt | `4fd3afb79cb6758baa30b7ae505885d71356800f0f2b553a454771046f8677c5` |
| iter1c repaired-implementation re-review (FAIL) | `87615350ffa76bd560fb60a1d5d6819e839184629761a2748c4504758c734ad3` |
| iter1c pollution inventory | `f49c490bc707523ba9fcd1f5fadac1247fcc7f1b3c2b454b6f8892ced312d2b9` |

The reviewed iter1c harness and focused tests had hashes
`7f3c1214fa56f8cb40bea2bf120a877d1d4e4206553e059cb8f92d32498ed7ae` and
`99a17adeb1bb733351fbdde34b6b046f08af3707feb552fc9a1a637ae6a81445`.

No iter1c official root, rank result, fit, metric, gate, or scientific outcome existed. Iter1c
is nevertheless retired because development tests wrote official iter1c terminal/lifecycle
receipts under `/tmp`, directly violating its namespace rule. Independent fault injection also
showed a check-then-replace race, a potentially committed-but-unreported `ROOTS_STARTED`
transition, non-durable failing initialization evidence, and a final-source-check publication
window. No iter1c file, namespace, root, stream, receipt, or metric may be pooled with iter1d.

## Fresh namespace and roots

```text
official namespace:    rtgs.inverse-projection-fiber.iter1d.v1
development namespace: rtgs.inverse-projection-fiber.development.iter1d.v1

scene roots:            17686011, 17686012, 17686013
initial-depth roots:    17686111, 17686112, 17686113
```

An exact repository search found no occurrence of these literals before this file. The six
iter1d roots may be constructed only by the reviewed official harness and only once.

## Scientific protocol imported unchanged

The following definitions remain byte-for-byte normative from the SHA-bound iter1b/iter1c
protocols:

- eight exact synthetic Gaussians, six 64x64 ring cameras, optimization views `0..3`, held-out
  views `4..5`, and 32 source hypotheses;
- exact camera-depth mean fibers, three-coordinate Schur-complement covariance fibers, and the
  paired free world-mean/Cholesky-SPD control;
- stopped symmetric-Mahalanobis center cost, affine-invariant conic cost, weight `0.25`, hard
  minimum, source weight `25`, oracle, and cyclic shuffled control;
- deterministic full-batch CPU float64 Adam at `0.025`, exactly 400 updates, and checkpoints at
  `0,20,...,400`;
- rank, construction, finite-difference, duplicate-invariance, checkpoint, and strict paired
  initialization sentinels;
- post-fit nearest association, denominators 96/64/32, GT-ID errors, track fractions,
  condition/depth/runtime diagnostics, and all six scientific gates; and
- exclusion of RGB, SH, opacity, visibility/occlusion, global track partitioning, topology,
  merge, split, prune, and teleport.

Iteration 1d still tests only idealized hypothesis-wise geometry association. It authorizes no
real-data, appearance, topology, or global “true correspondence” claim.

## Receipt-domain separation

Every receipt-producing function takes an explicit namespace. Exactly two domains are allowed:

- official functions emit schemas containing `iter1d`, the official namespace, and the true
  root-transition state;
- development functions emit schemas containing `development`, the fresh development
  namespace, and `root_consumption_status = DEVELOPMENT_ONLY`.

Tests may hold official strings in memory to validate rejection, but no development test or
negative fixture may write an official namespace, official schema, official root literal, or
`ROOTS_STARTED` claim to disk. The verification run uses one fresh dedicated pytest base
directory and recursively scans every emitted JSON/text file afterward. Any official iter1d
namespace, schema, or root literal in that development tree is a verification failure and
permanently retires iter1d before official execution.

## Atomic owned-path transaction

Official execution is Linux-only and must fail before roots unless libc exposes
`renameat2(..., RENAME_EXCHANGE)`. JSON ownership is captured from one `O_NOFOLLOW` file
descriptor: hash bytes and `fstat` that same descriptor, then verify that the path still names
the captured device/inode.

Updating an owned JSON path must not use check-then-`os.replace`:

1. prepare and fsync the complete replacement in the same directory;
2. atomically exchange replacement and target with `RENAME_EXCHANGE`;
3. inspect the displaced directory entry;
4. accept only when its hash/device/inode equals the stored ownership;
5. otherwise exchange back, fsync, verify the unexpected entry was restored at its original
   path, and delete only the process's own prepared inode; and
6. if verification is disrupted by another mutation, retain every unknown inode under a
   recovery name and fail `INVALID` rather than deleting or overwriting it.

Fault injection must cover a different-content collider and a same-content/different-inode
collider arriving immediately before the first exchange, plus ownership capture replacement
between hashing and path verification. Unknown symlinks, directories, unreadable entries, and
multiple-race recovery uncertainty are preserved and fail closed.

Every mkdir, link, exchange, rollback, unlink, and recovery rename is followed by containing
directory fsync. No helper may report a path as durably updated until its file and directory
entries have been fsynced.

## Conservative root-transition semantics

Set `root_transition_attempted = true` immediately before attempting the atomic
`ROOTS_STARTED` lifecycle exchange, before any generator is called. From that instruction
onward, any exception publishes official INVALID receipts and permanently consumes the iter1d
namespace/roots even when the generator was never reached.

An exchange helper error records whether the prepared replacement currently owns the public
path and carries that replacement identity to the caller. Thus a transition that committed but
failed during fsync/capture can still be safely invalidated. INVALID receipts distinguish
`ROOT_TRANSITION_ATTEMPTED` from `OFFICIAL_GENERATORS_CONSUMED` while never permitting a retry.

## Durable paired-initialization receipts

After each durable raw-input receipt and before rank/fitting:

1. write a per-scene common-constructor receipt containing initial depths, common mean/covariance
   hashes, and raw-input-receipt path/hash;
2. for every arm, compute the complete initialization-equivalence receipt and write it
   exclusively before testing `pass`, constructing an optimizer, or training; and
3. add every receipt path/hash and the failing receipt, if any, to terminal INVALID provenance.

Fiber realized mean/covariance hashes must equal the common hashes. Free realized mean maximum
absolute error remains exactly zero and covariance relative-Frobenius maximum remains
`<=1e-14`; all source projection tolerances remain unchanged. All hashes, deltas, and checks are
durable even when initialization itself invalidates the run.

## Final source closure and prepared publication

The complete source closure includes the generic scientific implementation, iter1d wrapper,
all inherited protocol/review/pollution artifacts, iter1d preregistration/review, focused tests,
and exact verification receipt. Bound historical hashes are checked before artifact creation.

Construct the complete result with candidate end hashes equal to the immutable start map,
canonicalize it, and fsync prepared result/terminal/lifecycle payloads before the final source
read. Immediately before the first public link/exchange, hash the entire closure once. If it
differs from start, publish INVALID and serialize that first observed mismatching map; do not
replace it with a later re-hash after a revert. If it matches, publish only the already-prepared
bytes; no source file is read or result payload reserialized afterward.

The unavoidable threat boundary is a non-cooperating writer changing a source after the final
hash instruction itself. Publication performs no source-dependent computation after that
instruction and records start plus candidate end maps identically.

## Lifecycle, paths, and invalidation

Preflight retains all iter1c safeguards: ambient default dtype exactly float32, ambient default
device CPU, existing real output parents, absent/disjoint/non-nested output paths, complete
source availability, exact historical digests, and a PASS development receipt whose roots are
disjoint from official roots.

After `root_transition_attempted`, every sentinel failure, non-finite value, source drift,
collision, fsync failure, or output failure independently attempts terminal, aggregate, and
lifecycle INVALID publication. Unowned entries are never replaced. Publication errors and
recovery paths are serialized without traceback, secrets, or RGB. There is no retry, resume,
overwrite, repair-in-place, checkpoint selection, or outcome-conditioned rerun.

## Iteration boundary

Iter1b, iter1c, and iter1d are implementation/lifecycle restarts before any scientific outcome;
together they remain Iteration 1 of exactly three. Iteration 2 may be specified only after one
valid iter1d outcome is executed and independently results-audited. Iteration 3 remains the sole
calibrated-data fit.
