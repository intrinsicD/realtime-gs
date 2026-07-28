# Compact-only carrier sequence interaction — independent audit — 2026-07-28

Status: **PASS_WITH_SCOPE_LIMITS**.

## Referee disposition

| Question | Disposition | Independently recomputed evidence |
| --- | --- | --- |
| Did the run remain compact-only through every stage? | **Confirm within the executable boundary.** | All 91 pre-sealed source files and 23 parent artifacts remain byte-identical. The guard recorded zero image opens/imports, packed alpha was not loaded, compact containers match, and held-out cameras were not evaluated. |
| Is post-phase-2 strict pruning viable? | **Yes.** | Pruned/unpruned `J_Q=1.0383` (+3.83%), `J_U=1.0229`; zero fitting-view violations and 5.32–5.42% rows removed. |
| Is pre-phase-2 strict pruning viable? | **Yes, and selected.** | Prune→phase2/unpruned `J_Q=1.0269` (+2.69%), `J_U=1.0164`; zero violations. It improves on post-pruning: `J_Q=0.9890`, `J_U=0.9936`. |
| Does joint SH3 earn inclusion in phase 2? | **No.** | Joint-SH3/degree-zero after pre-pruning: `J_Q=0.9770` (-2.30%), 3/3 wins, but below the frozen 5% materiality floor. |
| Does a separate SH3 phase earn a third stage? | **No.** | Separate-SH3/degree-zero: `J_Q=0.9703` (-2.97%), 3/3 wins, but below 5%. |
| Does the sequence establish no free floaters? | **Only outside the fitted-view visual hull.** | Every retained center has `q<=9` and positive depth in every fitting view, and frozen means preserve that invariant. Interior-hull floaters remain unidentifiable. |
| Does this prove the broad VRAM claim? | **No.** | It proves an image-free compact execution and records absolute resources; it does not provide a controlled dense baseline or multi-scene scaling evidence. |

## Audited selected sequence

1. Fit-only Beam Fusion; no free birth.
2. Renderer-aware symmetric covariance repair only.
3. Compact all-parameter degree-zero phase 1.
4. Strict fitted-Gaussian visual-hull projected-center pruning.
5. Compact degree-zero phase 2 with means frozen.

No opacity repair, appearance repair, legacy covariance repair, soft support penalty, clone,
split, insertion, densification, or SH3 stage is retained.

The selected arm has validation geometric `J_Q=0.00422499` and
`J_U=0.00786318` with 4,729–4,734 rows. Relative to the unpruned selected
phase-2 anchor, containment costs 2.69% `J_Q` and 1.64% `J_U`.

## Receipt findings

- All 27 arm result/model receipts and 15 training histories match their digests.
- Parent phase-1 and phase-2 semantic lineage, strict-prune subsets, SH expansion, source/proposal
  immutability, paired point streams, fixed topology, and exact frozen-mean motion were
  independently checked.
- Every contained arm has exactly zero fitting-view center-support and near-plane violations.
- The run is single-scene development evidence. It selects the repository development sequence,
  not a cross-scene, production, or publication claim.

## Commands checked

```text
.venv/bin/python benchmarks/compact_only_carrier_sequence_interaction.py
.venv/bin/python benchmarks/audit_compact_only_carrier_sequence_interaction.py
```
