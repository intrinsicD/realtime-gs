# Inverse-projection fiber fitting, iteration 1e — implementable transaction preregistration

Frozen at: 2026-07-17 (Europe/Berlin)

Status: **FROZEN BEFORE ITER1E IMPLEMENTATION OR OFFICIAL EXECUTION**

## Bound history and paper-only restart

Iteration 1e imports the complete scientific protocol from iter1b/iter1c through the closed
iter1d preregistration. It changes only transaction threat boundaries, receipt-domain typing,
and the three-file commit rule.

| Historical artifact | SHA-256 |
| --- | --- |
| closed iter1d preregistration | `91d3e1c601c6eeb41f4c828e1f600c5dd7a1f52c818754745548130d9b35fe9c` |
| iter1d preregistration review (FAIL) | `e9ecb31f6ed4738a902b38aa2b87dca53ec4ccbcc725c003a82954bee7791598` |

The iter1d review verified all transitive iter1b/iter1c hashes, scientific definitions, and
freshness. It failed before implementation because arbitrary repeated non-cooperating races
cannot satisfy conditional inode cleanup on Linux, receipt fields were not driven by one
immutable domain object, and lifecycle was not explicitly the only commit marker. No iter1d
code, generator, root, result, metric, gate, or scientific outcome existed. Iter1d is closed and
cannot be pooled with iter1e.

## Fresh namespace, roots, and development tree

```text
official namespace:    rtgs.inverse-projection-fiber.iter1e.v1
development namespace: rtgs.inverse-projection-fiber.development.iter1e.v1

scene roots:            17687011, 17687012, 17687013
initial-depth roots:    17687111, 17687112, 17687113

verification base:      /tmp/rtgs_ipf_iter1e_verify_20260717_001
```

Exact searches found no prior repository occurrence of the namespaces or roots and no existing
verification-base entry before this file. The official roots may be constructed only once by a
reviewed official harness.

## Scientific protocol imported unchanged

Iter1e retains exactly:

- eight synthetic degree-zero Gaussians, six 64x64 ring cameras, four optimization views, two
  held-out views, and 32 source hypotheses;
- exact mean/covariance inverse-projection fibers and the paired free Cholesky-SPD control;
- stopped center, affine-invariant conic, `0.25` geometry weighting, hard-min, `25` free-source
  weighting, oracle, and cyclic shuffled objectives;
- deterministic CPU float64 Adam at `0.025`, 400 updates, checkpoints `0,20,...,400`;
- all construction, derivative, duplicate, rank, checkpoint, and strict initialization
  sentinels;
- all post-fit association rules, denominators, metrics, six gates, and cumulative-RSS labels;
  and
- the exclusion of RGB, SH, opacity, visibility/occlusion, global track partitioning, topology,
  merge, split, prune, and teleport.

This remains idealized hypothesis-wise geometry association only. It authorizes no real-data,
appearance, topology, or global “true correspondence” claim.

## Explicit threat boundary

Official execution is a **single-writer transaction**. From preflight artifact-directory
creation through final lifecycle commit, no other process is authorized to mutate the reserved
aggregate path, artifact tree, recovery names, or scientific source closure. Sequential
pre-existing/late collisions are detected and preserved.

The required concurrency fault model is exactly one injected name mutation at each tested seam,
followed by a quiescent writer. Arbitrary repeated non-cooperating directory races are outside
the guarantee because Linux has no unlink/rename-if-device-and-inode syscall. If observations
suggest more than the bounded mutation or recovery becomes uncertain, the harness performs no
further mutation of contested names, retains every observable entry, records last-observed
facts, publishes best-effort INVALID elsewhere, and authorizes no claim.

The source-closure single-writer interval begins when the final one-pass closure observation
reads its first byte and ends at lifecycle commit. A multi-file hash is not an atomic snapshot;
uncoordinated source writes during that interval are outside the guarantee. This boundary is a
precondition, not a claim that such writes can always be detected.

## Immutable receipt domain

One frozen `ReceiptDomain` object drives every disk producer and contains:

- protocol label, official/development label, namespace, schema family;
- the only permitted root-consumption statuses;
- permitted roots (official) or a prohibition on all official roots (development); and
- whether official phases and commit states are permitted.

The domain is required for config, provenance, lifecycle, raw inputs, common constructor,
per-arm initialization, arm result, aggregate, terminal, fallback/recovery manifest, prepared
payload, verification, and every fault-injection artifact. Schemas and statuses are derived,
not supplied independently.

Official schemas contain `iter1e`; development schemas contain `development_iter1e`, use the
fresh development namespace, contain no official root literal, and always set
`root_consumption_status = DEVELOPMENT_ONLY`. Tests may construct official-shaped dictionaries
in memory to test rejection but may not write them.

## Ownership capture and retained exchange entries

Official preflight requires Linux and libc `renameat2(..., RENAME_EXCHANGE)`; there is no
fallback. All name operations use a held parent-directory file descriptor and single-component
names.

Ownership capture opens with `O_RDONLY|O_CLOEXEC|O_NOFOLLOW`, `fstat`s, hashes with `pread` from
that same descriptor, `fstat`s again, rejects metadata changes/non-regular files, then verifies
the path still names the same device/inode. The expected hash is compared last.

Every complete prepared JSON entry is named from creation as
`.<target>.recovery.<128-bit-nonce>.prepared`, written fully, file-fsynced, and directory-fsynced.
Prepared/recovery entries are **never unlinked after a link or exchange**, even on success; they
remain hashed transaction evidence. This removes the impossible conditional-unlink promise.

For an owned update:

1. exchange prepared and target atomically and directory-fsync;
2. verify the public name identifies the prepared inode and snapshot the displaced recovery
   entry without following links;
3. accept only if the displaced SHA/device/inode matches stored ownership, retaining that old
   owned entry at the recovery name;
4. otherwise exchange back, directory-fsync, verify the public entry matches the displaced
   snapshot and the recovery entry matches the prepared inode, retain both, and fail; and
5. on disruption, stop contested-name mutation and report
   `YES_AFTER_EXCHANGE`, `NO_AFTER_ROLLBACK`, or `UNKNOWN_AFTER_DISRUPTION` as a last-observed
   fact, never an absolute current-state claim.

For an absent target, publish with an exclusive hard link from the prepared recovery entry,
directory-fsync, and verify the public path identifies that inode. `EEXIST` preserves the
collider. Unknown public entries remain at their path; recording that path as a recovery
location does not imply moving it.

Fault tests cover same/different-content regular colliders immediately before exchange,
same-FD capture replacement, symlink/directory/unreadable entries, mutation before rollback and
before recovery verification, exchange/fsync/capture failures, and event ordering. Each test
uses one mutation at its named seam followed by quiescence.

## Conservative root transition and durable initialization

`root_transition_attempted = true` is set immediately before the `ROOTS_STARTED` exchange.
Any exception after that instruction attempts official INVALID and permanently retires the
namespace even if no generator ran. Mutation errors carry the prepared identity and transaction
report so the caller can safely use a last-observed ownership candidate. Immediately before the
first generator invocation, set `official_generators_consumed = true` conservatively.

INVALID distinguishes `ROOT_TRANSITION_ATTEMPTED` from `OFFICIAL_GENERATORS_CONSUMED`; neither
permits retry.

Per scene, the durable order is:

1. raw-input receipt;
2. common-constructor receipt containing initial depths/hash, common mean/covariance hashes,
   and raw-receipt path/hash;
3. rank sentinel; and
4. for each arm, a complete initialization receipt—including CPU/float64 and all equivalence
   checks—written before checking `pass`, creating Adam, or training.

Where possible, initialization exceptions become bounded error-form receipts. INVALID includes
all raw/common/per-arm descriptors and the complete failing receipt. A failing initialization
must leave optimizer/training spies untouched.

## Prepared result and final source observation

Before the final source observation:

1. construct the complete result with candidate start/end source maps both equal to immutable
   start provenance;
2. canonicalize result bytes and compute its hash;
3. construct/canonicalize terminal bytes cross-referencing the result hash;
4. construct/canonicalize final lifecycle bytes cross-referencing result and terminal hashes;
5. create, file-fsync, and directory-fsync all three recovery-qualified prepared entries; and
6. open all required directory descriptors.

Then perform exactly one complete closure observation. Preserve it immutably as
`first_source_observation`, including partial hashes and per-path read errors. If it differs from
start or contains an error, publish INVALID using that first observation and never call the
source-hash helper again, even after a revert.

If it matches, no payload is reserialized and no source is reread: exclusively link result,
exclusively link terminal, then exchange lifecycle last. Every mutation is directory-fsynced.
A later publication failure uses the saved matching observation for INVALID without rereading
sources.

## Authoritative commit and claim predicate

Lifecycle is the **only commit marker** and is published last. A scientific outcome is valid
only when all of the following hold:

1. lifecycle has a terminal scientific status `PASS` or `FAIL` (never pending/INVALID);
2. aggregate and terminal are durable regular files;
3. all three share the official domain, namespace, scientific status, source maps, and root
   status;
4. terminal's aggregate hash and lifecycle's aggregate/terminal hashes match the files;
5. terminal phase is `complete`; and
6. the transaction validator reports no recovery uncertainty or publication error.

Until that final lifecycle exchange and validation, result/terminal files are uncommitted
transaction debris and authorize no metric, gate, or claim. Any partial, contradictory, or
collided publication is INVALID regardless of embedded scientific status.

## Development verification

The exact fresh base directory is `/tmp/rtgs_ipf_iter1e_verify_20260717_001`. The focused pytest
command uses it exclusively. After pytest, recursively scan every path name and every regular
file byte payload; symlinks or non-regular entries fail verification. Forbidden development-tree
content is the official iter1e namespace, official schema family, six official root literals,
and official transition/root-status literals. The receipt stores scanned file hashes/count and
zero forbidden matches using only the development domain.

## Iteration boundary

Iter1b through iter1e are pre-outcome implementation/paper restarts and together remain
Iteration 1 of exactly three. Iteration 2 may be specified only after one committed iter1e
outcome is independently results-audited. Iteration 3 remains the sole calibrated-data fit.
