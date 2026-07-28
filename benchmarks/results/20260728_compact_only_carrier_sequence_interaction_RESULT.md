# Compact-only carrier sequence interaction — result — 2026-07-28

## Status

**Complete; independently audited `PASS_WITH_SCOPE_LIMITS`.**

The final compact-only development sequence is:

1. fit-only Beam Fusion;
2. renderer-aware symmetric covariance repair;
3. compact degree-zero phase 1 with all parameter families trainable;
4. strict fitted-Gaussian visual-hull projected-center pruning; and
5. compact degree-zero phase 2 with means frozen.

No RGB, dense mask, packed alpha, dense trainer, topology growth, opacity/appearance repair,
legacy covariance repair, soft support penalty, clone-all, or SH3 stage is retained.

## Frozen decisions

All ratios are paired over roots `282701,282702,282703` and aggregated geometrically.

| Question | `J_Q` ratio | `J_U` ratio | Decision |
| --- | ---: | ---: | --- |
| Post-phase-2 prune / unpruned phase 2 | 1.0383 | 1.0229 | Viable within 5% |
| Pre-prune → phase 2 / unpruned phase 2 | 1.0269 | 1.0164 | Viable within 5% |
| Pre-prune → phase 2 / post-phase-2 prune | 0.9890 | 0.9936 | Select pre-prune order |
| Joint SH3 / degree-zero after pre-prune | 0.9770 | 0.9921 | Fail 5% materiality |
| Separate SH3 / degree-zero after pre-prune | 0.9703 | 0.9873 | Fail 5% materiality |

Both containment orders remove 5.32–5.42% of rows and end with exactly zero fitting-view
`q>9` or near-plane center violations. Pre-pruning allows the fixed-means second phase to recover
quality and is 1.10% better in `J_Q` than pruning only at the end.

The selected arm has geometric validation:

- `J_Q = 0.00422499`;
- `J_U = 0.00786318`; and
- 4,729–4,734 retained rows.

Relative to the unpruned selected phase-2 anchor, the explicit containment invariant costs 2.69%
`J_Q` and 1.64% `J_U`.

## SH disposition

Both SH3 variants improve numerically in all three roots:

- joint SH3 improves `J_Q` by 2.30%;
- a separate SH3-only stage improves `J_Q` by 2.97%.

Neither reaches the preregistered 5% materiality floor. The compact development pipeline
therefore remains degree zero and avoids an extra stage and larger coefficient tensor.

## Containment meaning

For each retained center `x_i` and fitting view `v`, the policy enforces:

`depth_v(x_i) > near`

and

`min_j (pi_v(x_i)-mu_vj)^T C_vj^-1 (pi_v(x_i)-mu_vj) <= 9`.

Means are frozen afterward, making this fitting-view center invariant exact.

This prevents carriers whose centers fall outside the positive-amplitude fitted 2D Gaussian
support union in any fitting view. It does not detect a floater inside the multi-view visual
hull, prove physical surface occupancy, or constrain the entire infinite-support mathematical
Gaussian. A finite-footprint erosion criterion was not tested and would penalize legitimate
boundary carriers.

## Provenance

- The source tree, parent policy-closure artifacts, input containers, and evaluation banks were
  sealed before execution and remained byte-identical.
- All 27 arm result/model receipts, 15 training histories, parent lineage, paired samples,
  strict-prune subsets, immutable compact teachers, fixed topology, and frozen-mean motion were
  independently checked.
- The live no-image boundary recorded zero image opens/imports and all three negative controls
  passed.
- Packed alpha was not loaded and held-out cameras were not evaluated.

## Claim boundary

This result selects the single-scene compact-only development pipeline. It does not establish
exact source masks, absence of visual-hull-interior floaters, cross-scene quality, general
VRAM/runtime savings, production readiness, or a publication claim.

## Artifacts

- Preregistration: `20260728_compact_only_carrier_sequence_interaction_PREREG.md`
- Harness: `benchmarks/compact_only_carrier_sequence_interaction.py`
- Run: `runs/compact_only_carrier_sequence_interaction_20260728`
- Independent audit: `20260728_compact_only_carrier_sequence_interaction_AUDIT.md`
- Machine-readable audit: `20260728_compact_only_carrier_sequence_interaction_AUDIT.json`
