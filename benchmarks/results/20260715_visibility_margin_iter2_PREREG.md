# Coarse visibility-margin retry: order-preserving support-safe extension

## Chronology and incorporated protocol

Frozen at `2026-07-15T23:15:00+02:00`, before implementing the retry fix, recomputing any
incidence statistic, or training a support-safe arm. This retry incorporates every scientific
choice, scene, seed, metric, threshold, target-parity prerequisite, all-in-front `U` audit,
four-cell Phase-B evaluation, and interpretation rule from
`benchmarks/results/20260715_visibility_margin_PREREG.md` at SHA-256
`1a0d9ec8c211a678898a699650fab2e2ab4c146c4d82df801e40622ab551767a`, except for the explicitly
listed attribution repair and fresh artifact namespace below.

The first implementation was sealed by
`benchmarks/results/20260715_visibility_margin_SEAL.json` at SHA-256
`92396fc86621d432cf0b53be6e37376578b2f4271ab979141c7bdc117f8b1b99`. Its official Phase-A
attempt is permanently recorded at
`benchmarks/results/20260715_visibility_margin_PHASE_A_ATTEMPT.json` with SHA-256
`13f3b8515d2a8657c6cb12230c55f0c60d2c253694914ff8c91cfb056490c149`. The attempt completed the
diffuse seed-0 current run, then stopped during diffuse seed-1 initialization at the frozen
current-order-versus-filtered-safe-order invariant. It stopped before a JSON/result note was
created, before any gate statistic or quality metric was printed, and before any candidate arm ran;
the named failed output and companion note are absent.

Failure-only diagnosis recreated diffuse seed 1 without computing or printing incidence/quality.
In train view 9, current visibility contained 826 Gaussians and support-safe visibility 827. Two
current indices swapped at adjacent order positions because their float32 camera depths were
exactly equal (`2.2596030235290527`) and adding one element changed the unspecified tie behavior of
`torch.argsort`. This is an attribution defect: a margin experiment must not silently reorder the
already-visible compositing sequence.

## Frozen representation-only repair

The current/default `3.0` path and its existing `torch.argsort(z[current_indices])` expression stay
unchanged. For a margin strictly larger than `3.0`, the Torch reference renderer will:

1. recompute the established current-visible subset using the same detached `3.0*sigma` envelope;
2. obtain its exact established order with the unchanged default `torch.argsort` call;
3. separately depth-order only the newly admitted indices;
4. concatenate established-current order before new order and apply a **stable** depth sort to that
   already ordered list.

Thus filtering the expanded order back to current indices must be exactly equal to the established
current order, including exact-depth ties. A newly admitted primitive at exactly the same depth is
placed after established primitives at that depth. This deterministic convention changes no
current primitive order and has no effect when the expanded set admits nothing. It is not an
independent candidate, sorting sweep, or claimed renderer improvement; it is the minimum repair
needed to isolate the visibility-set intervention. Margins below `3.0` are outside this experiment
and retain the ordinary ordering path.

Add CPU tests that construct an exact-depth tie whose old expanded set changed filtered order,
prove the retry preserves established order, prove default output remains bit-exact, and prove an
expanded render is bit-exact to current whenever no Gaussian is newly admitted. Also assert the
complete expanded sequence is monotonically nondecreasing in depth. The Phase-A depth-order
invariant remains mandatory; it is not relaxed.

## Replay, gates, and fresh artifacts

Create a fresh complete seal after independent protocol, implementation, and harness review. Replay
all six current Phase-A runs from scratch; no cached fit, initialization, training, incidence, or
metric from the failed attempt may enter the result. The target-generation parity prerequisite must
again run before every fit. Phase B remains authorized only by the unchanged final-diffuse validity
and material gates in the incorporated protocol and requires a new exact-hash scientist review.
If authorized, its candidate uses the order-preserving support-safe renderer for both training and
matched evaluation; the common-current and forward-only cells retain their incorporated meanings.
Any positive interpretation is narrowly about support-safe visibility with this baseline-preserving
tie extension, not a pure margin-only effect for primitives at exactly equal depth.

Fresh fixed paths are:

- seal `benchmarks/results/20260715_visibility_margin_iter2_SEAL.json`;
- Phase-A marker `benchmarks/results/20260715_visibility_margin_iter2_PHASE_A_ATTEMPT.json`;
- Phase-B marker `benchmarks/results/20260715_visibility_margin_iter2_PHASE_B_ATTEMPT.json`;
- Phase-A output `<UTC>_cpu_visibility_margin_iter2_audit.json`;
- Phase-B output `<UTC>_cpu_visibility_margin_iter2_ablation.json`.

The retry seal/runtime must bind the incorporated protocol, first seal, consumed first marker, their
exact hashes, and absence of both failed output paths. Any further failure consumes the iter2 marker
and requires another append-only preregistration/namespace. No outcome authorizes changing the
margin, scene framing, resolution, near plane, support cutoff, loss, schedule, gates, or seeds.
