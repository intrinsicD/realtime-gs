# CLAIMS & QUESTIONS — What is publishable here, and under which outcome

**Date:** 2026-07-25
**Sits above:** `20260725_init_value_program_PREREG.md`
**Purpose:** the PREREG asks validation questions ("is my pipeline broken, how do I fix it").
This document holds the *research* questions — the ones whose answers matter to people who will
never open this repository. Every PREREG experiment either feeds one of the claims below or is
internal machinery. Machinery is necessary; it is not a contribution, and it must never drift into
the paper as if it were one.

---

## 1. The two paper candidates

The program can produce two distinct papers. They must not be blended into one manuscript that
half-argues both — that is the classic shape of a rejection.

### Paper A — Systems/Method
**Claim:** *Image-free 3DGS reconstruction from per-view 2D Gaussian captures: k× more views held
at fixed VRAM, at quality parity with RGB-backed training.*

- **The one question it answers:** can the source images be discarded after a per-view 2D fit
  without losing reconstruction quality — and what does that buy in memory and scale?
- **What carries it:** E11 (memory/throughput, the headline numbers) + a quality-parity result
  under P-fix on standard benchmarks + E1c (the lifted geometry is accurate — supporting, not
  headline).
- **What kills it:** a quality gap to RGB-backed training that exceeds the noise floor on held-out
  views. Parity is the load-bearing word; "slightly worse but smaller" is a much weaker paper and
  should be recognized as such early, not after writing.
- **Honest framing if init adds nothing:** the init is then *free* (it falls out of the compact
  representation anyway) and the paper claims memory, not convergence. That version survives a
  quality tie; the convergence version does not.

### Paper B — Analysis
**Claim:** *Initialization value in 3DGS is a function of the constraint-to-capacity ratio (Ω) and
of the optimizer's trust in the init; positional accuracy is causally the driver, and standard
schedules are optimizers for bad initializations.*

- **The one question it answers:** when, why, and by what mechanism does initialization matter in
  3DGS — as opposed to *whether* one particular initializer beats another on one benchmark.
- **What carries it:** the gap-vs-Ω collapse figure (E4/E5/E6), the dose–response figure (E4b),
  the interaction plot (E12), and the lineage survival/attribution analysis (E2/E3) as the
  mechanistic illustration.
- **What kills it:** nothing, structurally — every outcome of those experiments is a finding,
  including "init never matters above Ω = X" and "there is no interaction." That robustness is
  what makes it the safer paper. Its risk is different: **novelty erosion.** It stands only if the
  related-work gap (§3) survives a fresh literature pass.

### Decision rule between them (pre-declared)
- Beam fusion wins downstream under the PREREG's E4 decision rule → **Paper A is the main paper**,
  with Paper B's figures as its analysis section. Strongest combined form.
- Beam fusion does not win downstream → **Paper B is the paper**, standalone; Paper A's E11 result
  goes to a workshop/systems venue as a short paper. Do not force the method paper if the
  convergence claim failed — reviewers detect a systems paper wearing a science costume.
- The 0.5–0.8 middle band of the E4 rule → Paper A with the memory framing, Paper B separate.

---

## 2. The three research questions (Paper B's spine)

**Q1 — Regime.** Under what conditions (primitive budget, view count, resolution — unified as Ω)
can initialization have any effect in 3DGS at all?
*Why it generalizes:* it explains why the literature's init comparisons disagree with each other —
they were run at different, unreported Ω. One curve that reconciles contradictory published
results is a contribution regardless of which initializer anyone uses.
*Fed by:* E4, E5, E6. *Deliverable:* the single gap-vs-Ω figure.

**Q2 — Cause.** Is positional accuracy the causal driver of init value, or a correlate of
distribution/count/scale statistics?
*Why it generalizes:* every existing comparison is between whole initializers — confounded by
construction. A dose–response manipulation on one initializer isolates the variable. To our
knowledge nobody has done this; **that "to our knowledge" is load-bearing and must be verified
(§3), not assumed.**
*Fed by:* E4b. *Deliverable:* T@τ vs. σ, two noise models, random baseline as the upper anchor.

**Q3 — Interaction.** Do different initializations require different optimization — is the
standard 3DGS schedule an optimizer *for bad inits*?
*Why it generalizes:* if true, every init comparison run under the default schedule (i.e. nearly
all of them) systematically undercounts good initializations. That is a methodological point about
the field's evaluation practice, not about our pipeline.
*Fed by:* E12. *Deliverable:* interaction contrasts per schedule factor; the S4 crossover
(position-LR helps accurate init, hurts random) is the flagship signature if it appears.

Everything else in the PREREG — E1, E2, E3, E7, E8, E9 — is validation and repair. Its outputs
appear in the paper only as: design-choice justifications (one sentence + citation to supplement),
ablation rows, and the lineage figure. **E2/E3's lineage attribution is the exception:** "is an
init refined or rejected by the optimizer" is a measurement no one else reports and is promotable
to a first-class analysis figure in Paper B.

---

## 3. Novelty ledger — claims about the literature that must be verified before writing

The gap this program aims at: *existing work compares initializers as black boxes, without budget
conditioning, without causal manipulation, without optimizer interaction.* Each row below is an
assumption behind that sentence. A fresh literature pass (the field turns over in months; anything
remembered from before ~mid-2026 is stale) must confirm or kill each row. A killed row narrows the
claim; it does not necessarily kill a paper — but discovering it during review does.

| # | Assumed gap | Known nearest neighbors to check against | If the row falls |
| --- | --- | --- | --- |
| N1 | No published dose–response manipulation of init accuracy in 3DGS | RAIN-GS and successors (deliberately coarse init); any 2025/26 init-analysis papers | Q2 shrinks to the along-ray/isotropic contrast, which is likely still open |
| N2 | No Ω-style unification of when init matters | sparse-view 3DGS literature (view-count axis exists there); budget-constrained/compact-GS papers (budget axis exists there); check whether anyone crossed the axes | Q1 becomes "we complete the missing cells," weaker but viable |
| N3 | No systematic init × schedule interaction study | 3DGS-MCMC (schedule replacement, init-robustness claims); AbsGS/Pixel-GS/Taming-3DGS (densification criteria); check their ablations for init-conditioning | Q3 narrows to the specific factors not yet crossed with init |
| N4 | No image-free (2D-fit-only) training pipeline with quality parity | compressed/streaming-3DGS work; any "fit-then-discard" or feature-space-supervision papers | Paper A's framing shifts from "first" to the measured k× numbers, which survive regardless |
| N5 | Dense-prior inits (DUSt3R/MASt3R-style) are evaluated as endpoints, not analyzed for *why* they help | InstantSplat lineage and successors | Strengthens the need for Q2 rather than weakening it — but they must appear as baselines, not be absent |

**Rule:** no manuscript text before this table has a verdict per row, with citations, dated.

---

## 4. What the PREREG does not cover but publication requires

1. **Standard benchmarks.** MipNeRF360 + Tanks&Temples (or the then-current equivalent) for
   whichever paper is written. Object captures alone do not survive review; the synthetic scene is
   for diagnostics only. This means the Stage-1 capture pipeline must run on those datasets —
   budget that engineering now, it is on the critical path of *both* papers.
2. **Community metrics throughout.** SSIM and LPIPS alongside PSNR in every reported cell, not
   only in E5. Geometry error (depth/Chamfer vs. reference) wherever a reference exists — PSNR is
   blind to floaters, and floaters are the failure mode init is supposed to prevent.
3. **External baselines as arms, not variants.** Minimum: dense COLMAP/MVS init, one
   DUSt3R/MASt3R-style init, 3DGS-MCMC as a full method column, and the then-current default 3DGS
   with AbsGS-style densification. Our seven internal initializers are conditions; these are
   comparisons.
4. **Compute disclosure.** Tuning budgets (P-adapt), total GPU-hours, hardware — reviewers
   increasingly ask, and the tuning-fairness story of §1.5 is a strength only if the budget is
   stated.
5. **The two metrics this program invented need names and defense.** Lift efficiency (ΔLE) and
   alpha-mass attribution (A) will each need a half-column of motivation in the paper; they are
   minor contributions in their own right and should be presented as such, not smuggled in.

---

## 5. One-line status ledger (update as results land; no prose, no spin)

| Claim | Status | Evidence |
| --- | --- | --- |
| Q1 regime curve | pending | — |
| Q2 causality | pending | — |
| Q3 interaction | pending | — |
| A: VRAM k× | pending | — |
| A: quality parity | pending | — |
| N1–N5 verdicts | pending | — |

The discipline this table enforces: a paper is written when its rows are green, not when
enthusiasm peaks. If A's parity row goes red, Paper A's convergence framing is dead that day —
write it in the table and move on to the memory framing. The table is the antidote to the thing
this whole program exists to prevent: deciding the conclusion first and auditing the evidence
later.
