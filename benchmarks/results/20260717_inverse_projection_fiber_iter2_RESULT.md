# Inverse-projection fiber fitting, Iteration 2 — result

## Outcome

**Scientific status: FAIL.** The once-only official transaction committed normally and Gate 1
passed on all three roots. This is a negative result for the frozen residual-prune, spatial-
contraction, source-fixed rematch, and fixed-track-refit bundle—not a harness failure.

The official computation ran on CPU in float64 through the hash-pinned
`benchmarks/inverse_projection_fiber_iter2_launcher.py`. The isolated outer bootstrap captured the
launcher and implementation review through held descriptors, required launcher SHA-256
`d2a2eef01596d742bb7e554c3b97dc55aa5c0e8089d131d6893e966752333229` and review SHA-256
`d4655faec8711d5f3a2e29b07f08e11fa0a5c9bc5c3de512123f37cf32179fae`, injected only those pins
and the held workspace descriptor, then compiled the captured launcher bytes with `python3 -I -B`.
The first bootstrap invocation rejected an over-broad executable archive before reservation or
root access; after tests/support files were separated from the executable allowlist, the only
official transaction was `05b9478e90b04d0b9c17f8ba5202c085`.

The committed machine-readable result is
`benchmarks/results/20260717_inverse_projection_fiber_iter2_RESULT.json`, SHA-256
`d153706a5534a5f1d319d18b2961c944842bb01cd1573992b280c5ce096a2dfd`. The exact executed source
archive is `runs/inverse_projection_fiber_iter2_official_20260717/EXECUTED_SOURCES.zip`, SHA-256
`373545e0da7e05e2a78d8d83118a0cbb898dcfc40190ba4729ee08bfc7a90cec`.

## Primary results

| Root bundle | Survivor precision / recall | Hidden-mode coverage | Representatives | Topology | Fit / held-out / exact-track accuracy | Center p90 |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `27688011/27688111/27688211` | `0.9048 / 1.0000` | `1.000` | `8` | accepted | `1.0 / 1.0 / 1.0` | `3.103e-7` |
| `27688012/27688112/27688212` | `1.0000 / 1.0000` | `1.000` | `8` | accepted | `1.0 / 1.0 / 1.0` | `2.955e-7` |
| `27688013/27688113/27688213` | `1.0000 / 1.0000` | `0.875` | `7` | rejected | n/a | n/a |

Gate 2 therefore failed root 0's frozen `>=0.95` precision floor and root 2's eight-mode/count
requirements; only root 1 passed every selection/contraction check. Root 2 had no retained
candidate for hidden mode 2, so source-preserving merge/prune could not reconstruct the missing
track. In the two accepted roots, source-fixed rematching and 200-step refitting did recover exact
four-view tracks and perfect held-out association, with covariance-distance medians
`1.255e-6` and `1.381e-6`.

The 32-hypothesis hard-min control had mean exact-track fraction `0.53125`; proposed topology,
counting the rejected root as no recovered track, reached `0.666667`. The gain `0.135417` missed
the frozen `0.20` requirement. The shuffled control rejected all three roots and had zero exact
tracks, but its mean hidden-mode coverage was the same as proposed (`0.958333`), so Gate 4's
identity-separation requirement failed. Gates 2, 3, and 4 all failed.

Do not report the result JSON's three-root relative center reduction. Rejection placeholders encode
zero geometry for the third root, which makes that aggregate spuriously favorable. Only the two
accepted-root center errors above are interpretable.

## Validity and diagnosis

All three roots independently rederive from exact typed `input_evidence.npz`, `fit_evidence.npz`,
and `heldout_evidence.npz` with zero false checks and zero scalar-summary mismatches. The oracle
reaches perfect train/held-out/track accuracy in every root. Each root was constructed once,
held-out data was released once after the pre-held-out commit, learned-state hashes were unchanged
after release, and every source/projection/finite/SPD/checkpoint invariant passed. The worker exited
zero with no stderr or process stragglers; lifecycle is `COMMITTED` and all nine root IDs are
`CONSUMED`.

The result narrows the failure mechanism. When residual selection leaves one viable candidate per
hidden mode, contraction plus balanced source-fixed rematching is strong: it solved two roots
exactly. It is not a reliable topology estimator, however. Across disjoint development and
official roots the same frozen thresholds produced representative counts `8/8/9` and `8/8/7`.
False positives can sometimes be contracted harmlessly, but a hidden mode with no complete
hard-min candidate cannot be recovered by later merge/prune. The next method must couple
assignment capacity and track survival during fitting rather than making a one-shot unary residual
decision afterward.

## Scope

This was a three-root, noiseless, balanced, synthetic component experiment. It did not test unequal
2D decompositions, visibility/occlusion, appearance or SH, RGB rendering quality, uncertain camera
poses, real images, GPU behavior, speed, or memory. It establishes neither "true" correspondence
on real data nor a production default. See the independent scientist pass in
`benchmarks/results/20260717_inverse_projection_fiber_iter2_AUDIT.md`.
