# Inverse-projection fiber fitting, iteration 1c — implementation-corrected preregistration

Frozen at: 2026-07-17 (Europe/Berlin)

Status: **FROZEN BEFORE ITER1C IMPLEMENTATION OR OFFICIAL EXECUTION**

## Bound history and restart reason

Iteration 1c imports the complete scientific question, geometry, losses, reductions, arms,
optimizer, sentinels, metrics, gates, and claim boundary from the closed iter1b protocol, except
for the explicit lifecycle and initialization corrections below.

| Historical artifact | SHA-256 |
| --- | --- |
| closed iter1b preregistration | `f0b93ade31925cacff17d0101340b4bbda76ac6fbaee68f631c2e14d0b2824a1` |
| independent iter1b preregistration review (PASS) | `4adc3dc4ff14d64083df12d79c463b85c9289ff78f94861d92f2f174859de235` |
| independent iter1b implementation review (FAIL) | `d731d7ce4ee221fea2a625dc576167a8c15c37909ad9a7343bbb1c87751c0a2f` |

No iter1b official root was constructed. An unrelated-root two-update smoke produced development
receipts under the iter1b namespace in `/tmp`; that pollution and a false byte-identity
requirement close iter1b permanently. No iter1b file, root, stream, receipt, or metric may be
pooled with iter1c.

## Fresh namespace and roots

```text
official namespace:    rtgs.inverse-projection-fiber.iter1c.v1
development namespace: rtgs.inverse-projection-fiber.development.v1

scene roots:            17685011, 17685012, 17685013
initial-depth roots:    17685111, 17685112, 17685113
```

An exact repository search found no occurrence of these literals before this file. Development
tests must use unrelated roots, the development namespace, and schemas containing
`development`; they may never emit an iter1c namespace or official schema.

## Scientific protocol imported unchanged

The following iter1b definitions remain normative without modification:

- eight exact synthetic Gaussians, six 64x64 ring cameras, optimization views `0..3`, held-out
  views `4..5`, and 32 source hypotheses;
- the exact camera-depth mean fiber and three-coordinate Schur-complement covariance fiber;
- free world means and exp-diagonal Cholesky-SPD covariance control;
- stopped symmetric-Mahalanobis center cost, affine-invariant conic cost, weight `0.25`, hard-min
  latent matching, fixed source weight `25`, oracle, and cyclic shuffled control;
- deterministic full-batch CPU float64 Adam at `0.025`, exactly 400 updates, and checkpoints at
  steps `0,20,...,400`;
- official rank/conditioning, exact construction, finite-difference, duplicate-invariance, and
  checkpoint sentinels;
- post-fit ordinary-nearest association, denominators 96/64/32, GT-ID projected errors, track
  fractions, condition/depth/runtime diagnostics, and every primary gate; and
- the restriction to idealized hypothesis-wise association, excluding RGB, appearance,
  opacity, topology, global track partitioning, and any real-data claim.

The finite-difference sentinel holds the stopped center metric fixed across `+/-1e-6`
perturbations; this is the numerical meaning of `stop_gradient`.

## Corrected paired-initialization contract

Every arm receives byte-identical constructor input tensors for mean and covariance. The fiber
arms realize those tensors byte-identically. The free arm must factor the common covariance into
its Cholesky/log coordinates, which entails a float64 round-trip. Scientific fairness therefore
uses numerical, not bytewise, equivalence of the realized state.

Before any optimizer is created, for every replicate and every arm:

1. hash and compare the common constructor-input mean and covariance tensors;
2. require realized mean maximum absolute difference `== 0`;
3. require per-row realized covariance relative Frobenius maximum `<=1e-14`;
4. require source-projection center maximum difference `<=1e-10 px`; and
5. require source-projection covariance relative Frobenius maximum `<=1e-12`.

All deltas and hashes are serialized. Failure is `INVALID`. Unrelated development scenes showed
a maximum free Cholesky covariance round-trip error of `4.58e-16`, leaving more than a 20x
margin to the frozen threshold.

## Frozen root-stream and dtype precondition

`make_gt_gaussians` is default-dtype-sensitive. Before any official generator, camera, scene,
or depth draw is constructed, require exactly:

```text
torch.get_default_dtype() == torch.float32
```

A mismatch fails before official-root consumption and may be corrected because no official
stream has begun. Do not silently change the ambient dtype. Generate the established float32 GT
and camera streams first, then promote their parameters/tensors to float64 for projection and
optimization. Serialize the precondition and resulting tensor hashes.

## Exclusive lifecycle and durable invalidation

Preflight occurs before any official-root construction:

- `--out` and `--artifacts-dir` must both be absent, disjoint, and neither equal to nor nested
  inside the other;
- reject every reserved path collision;
- create the exclusive artifact root and write immutable config, start provenance, and a
  `ROOTS_NOT_STARTED` lifecycle receipt; and
- bind the complete scientific source closure and this preregistration by SHA-256.

Immediately before the first official root reaches a generator, atomically replace the lifecycle
receipt with `ROOTS_STARTED`. From that point onward, every exception, failed sentinel,
non-finite value, source drift, artifact collision, or output failure must attempt to write:

```text
<artifacts-dir>/terminal.json   status = INVALID
--out                           status = INVALID
```

The receipt records the phase, exception class/message, completed arm list, partial artifact
paths/hashes, start/end source hashes, and root-consumption status. It must not contain a Python
traceback, environment secrets, or RGB. The command returns nonzero after publishing INVALID.
There is no retry, repair-in-place, resume, overwrite, checkpoint selection, or outcome-driven
rerun in this namespace. Successful completion writes both receipts with the scientific gate
status (`PASS` or `FAIL`, never `INVALID`).

## Source and input closure

Hash at start and again immediately before terminal publication:

- the harness, this preregistration, inverse-projection fiber module, shared projection module,
  synthetic generator, camera, 3D Gaussian, and SH implementation;
- every other local Python module imported into the scientific path; and
- the focused test file and exact verification-command receipt.

Any start/end mismatch is `INVALID`.

For every official replicate serialize hashes of:

- GT means and covariances after the frozen float32 stream is promoted;
- every camera's intrinsics, image dimensions, `R`, and `t`;
- each view's projected target means, covariances, and depths;
- ordered source view/component IDs and source means/covariances; and
- initial depths plus common constructor-input and realized per-arm geometry.

These are provenance, not extra fitting inputs.

## Resource-label correction

Linux `ru_maxrss` is process-global and cumulative. Label it
`process_peak_rss_bytes_so_far`; do not call it an arm-local peak or compare arms causally by
that field. Wall time remains per arm.

## Iteration boundary

This lifecycle correction does not consume an evidence-driven research iteration because no
official scientific outcome existed. Iter1c remains iteration 1 of exactly three. Iteration 2
may be specified only after iter1c is executed and audited; iteration 3 remains the sole
calibrated-data fit.

No capability, global correspondence, topology, appearance, or real-data claim is authorized by
this correction protocol.
