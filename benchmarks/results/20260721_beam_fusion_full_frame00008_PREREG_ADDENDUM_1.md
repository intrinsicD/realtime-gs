# Beam-fusion full `frame_00008` preregistration — Addendum 1

Frozen: 2026-07-21 after the uninterrupted 0-to-30k parent completed, but before any 30k-to-40k
continuation optimization or checkpoint metric existed.

Parent protocol:
`20260721_beam_fusion_full_frame00008_PREREG.md` (the byte-bound original is unchanged).

## Uninterrupted-parent continuation plumbing

The landed `polish` preflight encoded the historical 2026-07-20 run's accidental interruption and
required a non-exact 4k-to-30k recovery receipt plus legacy-v1 target lineage. The present parent is
cleaner: it ran steps 1 through 30,000 uninterrupted and wrote deterministic-v2 targets at entry.
It therefore correctly failed that recovery-specific preflight even though its scientific
continuation schedule is the one frozen in the parent protocol.

The implementation may add a separate fail-closed clean-parent preflight that requires exactly:
the all-view/non-smoke 30k config, uninterrupted history with 30,000 losses and global offset 0,
the bound final PLY/count, deterministic-v2 target hashes, the frozen SH/checkpoint schedule, and
no recovery fields. `polish` then replays every target twice and requires exact deterministic
identity before constructing the same 30,001-to-40,000 optimizer segment. The receipt labels the
parent `uninterrupted`; the child remains a non-exact restart because PLY lacks Adam/RNG state.

No gate, learning rate, seed, target, metric, selection rule, stopping rule, Gaussian count, or
maximum horizon changes. The already observed parent and initialization results cannot be used to
tune the continuation. Tests must cover both historical recovered-parent and new uninterrupted-
parent preflights before execution.
