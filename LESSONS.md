# LESSONS.md — pre-flight before any results-bearing run

Companion to the task-first protocol in `CLAUDE.md` (Hard Rules 7–10). The protocol freezes
*what* we run; this file is about making sure the frozen configuration *tests what we
claim*. Shared with `~/Documents/SplatDiffuseLBO/LESSONS.md` (same checklist, both repos).

## Pre-flight check — part of the protocol-review packet, verified before the digest

1. **Units and scales** of every weight, threshold and tolerance, against our data (scene in
   mesh units, depth 400–600, `h`, dB). Numbers copied from papers/examples are converted.
2. **What the library computes**, read from the implementation: quantity, normalization,
   schedule, gradient paths, and whether advertised features fire (gsplat's opacity reset
   with `reset_every` never fires — `src/rtgs/optim/strategies.py`).
3. **Dry mechanism check with gradients** on a case with known answer (sphere, offset
   plane, two sheets, duplicated splats), as CPU tests.
4. **Teacher before student:** any term pulling towards an estimate (field normal, depth
   normal, fitted `S`) first shows the estimate beats the quantity it replaces, on the data.
5. **Degenerate minima named** (clones, collapse, shrink, wrong-but-consistent field) and
   either excluded or caught by a hard gate; "any seed violates" = the arm fails.
6. **Un-gameable metrics:** gates in fixed units, coverage and signed displacement beside
   distances, subset counts, structural diagnostics of downstream operators.
7. **Smoke on the real path** per arm with the final configuration; smoke numbers are
   disclosed as pilot exposure in the PREREG.
8. **Budgets and guards** per stage (GPU cells, CPU evaluation), timeout ≠ failure.

## Lessons (dated)

- **2026-10-07 — `20261007_jet_consistency_prior_tosca_cat0` (rejected).** kNN prior had a
  collapse minimum served by densification clones (64–73 % near-duplicates); 3DGS primitives
  were not flat, so the regularized "normal" was undefined. Items 3, 5 and the primitive
  check would have caught both before 11 GPU cells.
- **2026-10-08 — v2 PREREG, `λ_d = 100`.** Copied from the 2DGS paper code; gsplat's
  distortion is L1 in raw depth units, the control arm died in the 600-step smoke. Items 1–2.
- **2026-10-07 — review written on the reviewer's behalf.** The reviewer reads the generated
  task fields; the review file is written after that, by nobody else.
