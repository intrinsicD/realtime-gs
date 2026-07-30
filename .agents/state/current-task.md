# Current Task

## Title

LaTeX paper skeleton for the two-part VRAM / Beam Fusion publication plan

## Task ID

RTGS-003

## Role Assignment

- Driver: Claude-paper-draft-driver
- Reviewer: Claude-paper-draft-driver
- Turn: driver

## Mode

Implement

## Risk

Standard

## Maturity

- Target: Not applicable
- Reached: Not applicable

## Goal

Produce a buildable LaTeX working draft in `paper/` that implements
`docs/PAPER_PLAN_beam_fusion.md` as a full paper skeleton in the owner's single-thread
storyline (fields, standard reconstruction with random/SfM initialization, memory protocol,
tomographic initialization, carrier refinement, experiments with closing ablations), with
red `TODO:` markers for everything still to be shown (including exact figure
specifications), the owner's style rules, and well-explained black text for implemented
mechanism and evidence-bound results.

## Motivation

The human owner requested the full paper program as proper `.tex` files with red text for
open evidence and required figures. The plan documents (`PAPER_PLAN_beam_fusion.md`,
`THREE_ARM_EXPERIMENT_PROGRAM.md`, `20260725_claims_and_questions.md`) define claims and
discipline but no manuscript structure; a skeleton that encodes the claim firewall and the
open-obligation ledger makes the remaining experimental program explicit and reviewable.

## Success Criteria

- `paper/main.tex` compiles cleanly (no undefined references/citations, no overfull boxes).
- Single-thread structure per the owner decision of 2026-07-30, with the claim separation
  preserved as evaluation rules (reconstruction result established with standard
  initializations alone).
- Owner style rules hold in the running text: no em dashes, semicolons, colons, or
  marketing vocabulary, every red item prefixed `TODO:`, negative results only where an
  obvious approach does not apply and an alternative is given.
- Every quantitative statement in black is copied from an audited artifact and carries an
  evidence note to `ara/logic/claims.md` rows or sealed results; development scope is
  labelled as such; nothing is promoted beyond its ledger status.
- Every pending item is red (`\toshow`/`ToShowBlock`) and tracked in an obligations
  appendix; every required figure has a red slot with a concrete production spec.
- `./scripts/verify.sh` passes.

## Constraints

- Documentation-only change: no `src/`, `tests/`, `benchmarks/`, or `ara/` edits.
- Claim wording must not exceed `ara/logic/claims.md` status (Hard Rules 8–9 discipline
  applied to the new `paper/` surface).
- Build artifacts stay untracked (`paper/.gitignore`).

## Non-Goals

- Running any experiment, benchmark, or figure production.
- Changing claims, defaults, or the experiment registry.
- Submission formatting (venue style files) or a verified bibliography — the bib is marked
  as reconstructed-from-memory (obligation O-13 in the draft).

## Selected Skills

- rtgs-core
- rtgs-task-workflow
- rtgs-review

## Experiment Contract

None

## Current Evidence

- Compile receipt (v2 restructure, 2026-07-30): tectonic 0.15.0, 25 pages, BibTeX resolves
  all citations, 0 overfull boxes, no undefined references (local build; no LaTeX in CI, so
  the draft is not gated). Style scan clean: no em/en dashes or `--` in text, no prose
  semicolons or colons outside `TODO:` prefixes, no banned vocabulary.
- Content bindings: ara claims C15, C18–C20, C23, C24, C28–C31;
  `docs/ADR-002-carrier-refinement.md`; `docs/THREE_ARM_EXPERIMENT_PROGRAM.md`;
  `docs/PAPER_PLAN_beam_fusion.md`; `benchmarks/results/20260725_init_value_program_PREREG.md`
  (E11 protocol).
- Verification: ruff check/format, docs_sync, check_ara, check_script_layout,
  check_agent_workflow, and experiment-contract validation all pass; the CPU suite passes
  with one deselected test
  (`tests/test_compact_point_training.py::test_linux_listener_binding_requires_the_exact_process`),
  which fails identically on the unmodified base commit 80eb9f9 in this sandboxed
  container (viewer-listener /proc inspection) and is therefore environment-caused, not a
  regression of this docs-only change.

## Minimal Plan

1. Extract claim status, math, and numbers from the plan documents, ADR-002, the claim
   ledger, and `src/rtgs/lift/beam_fusion.py` / `optim/compact_trainer.py`. (done)
2. Write `paper/` skeleton: preamble with red-marker macros, 14 section files, bib,
   Makefile, README, figure list. (done)
3. Compile, fix layout (robust `\repopath`, obeyspaces, table widths), visual check. (done)
4. Integrate: `CLAUDE.md` map line, `PAPER_PLAN` pointer, `paper/.gitignore`. (done)
5. Self-review per rtgs-review, run `./scripts/verify.sh`, commit, push. (done)

## Status

Provisionally accepted (self-reviewed)

## Human Decisions

### Question

How should the draft be structured and styled after the owner's review of v1?

### Options

Keep the two-part (VRAM claim / Beam Fusion) structure, or restructure to the owner's
single-thread storyline. Keep or change the marker and style conventions.

### Recommendation

Follow the owner's instructions verbatim and keep the claim separation as evaluation rules
rather than document structure.

### Decision

(Owner, in chat.) Single-thread storyline: 2D multiview Gaussian fields to 3D Gaussian
field with the standard approach works, initialization shown with random and SfM (no Beam
Fusion), then the tomography approach arises from inverting the rendering, then the
problems of the naive approach with solutions, then full results, ablation study at the
end. Style rules: never "embarrassingly parallel", no em dashes, no semicolons, no colons,
no marketing, "novel" avoided, "to the best of our knowledge" acceptable, "We" acceptable,
paper stays in English. Every red item gets a "TODO: " prefix. Do not name the result
"VRAM claim". Plain headline/subsection titles. No negative results in the paper except
where an obvious or existing idea does not apply here and an alternative is given. Approved
phrases 1-3 of the style probe unchanged, phrase 4 re-punctuated, phrases 5 and 7 become
TODOs. The owner will additionally send one of their papers for a fine-grained style pass.

### Date

2026-07-30

## Handoff Log

### Handoff (Claude-paper-draft-driver, 2026-07-29)

#### Objective

Deliver the requested two-part LaTeX paper draft (Part I VRAM claim, Part II Beam Fusion)
with red open-evidence markers and figure specifications, integrated into the repository's
documentation surface without touching code or claims.

#### Reviewed state

Started from `claude/paper-vram-fusion-beam-zi5lt4` at 80eb9f9 with a clean tree and the
unchanged task template. Read `CLAUDE.md`, `PAPER_PLAN_beam_fusion.md`,
`THREE_ARM_EXPERIMENT_PROGRAM.md`, `20260725_claims_and_questions.md`, ADR-002,
`ara/logic/claims.md`, the 2026-07-28 experiment log entry, E11 in the init-value PREREG,
and the Beam Fusion / observation-field / compact-trainer sources.

#### Changes

New `paper/` directory: `main.tex`, `preamble.tex` (ToShow red macros, figure slots,
evidence notes), `sections/00–13` (abstract, intro, related work with N1–N5 red ledger,
setting/arms/decision rule, Part I: capture model + gauge hazard, point-sampled unbiased
supervision, VRAM claim + NVML protocol, experiments with baseline/main-table skeletons;
Part II: beam back-projection/gating/CI-fusion math, rank-5 observability proposition,
carrier refinement with corrected covariance objective and containment, matched three-way
experiment design + ablation grid; limitations, conclusion, obligations appendix O-1–O-15
+ figure production list), `references.bib` (marked unverified), `Makefile`, `README.md`,
`figures/README.md`, `.gitignore`. One map line in `CLAUDE.md`; one pointer block in
`docs/PAPER_PLAN_beam_fusion.md`; this task record.

#### Evidence

Local tectonic build: 27 pages, all citations resolved, 0 overfull hboxes, no undefined
references; page renders visually checked (red markers, evidence notes, tables, math).
Every black number traced to its ledger row (C15, C18–C20, C23, C24, C28–C31) during
drafting. All verify.sh gates green except one pre-existing environment-caused test
failure (viewer-listener /proc inspection), shown to fail identically on the unmodified
base commit; the suite passes with that single test deselected (details in Current
Evidence).

#### Assumptions

The 168 kB-per-view capture cap and the 26-view / 19-fitting / 7-validation split of the
development scene are as recorded in the PREREG and the 2026-07-28 experiment entry; the
draft's memory arithmetic is presented as format arithmetic, not measurement. Paper lives
in `paper/` (not `docs/`) as a new top-level surface; docs-sync does not gate `.tex`.

#### Uncertainties

Bibliography metadata is from memory and explicitly marked unverifiable (O-13). No LaTeX
distribution exists in CI, so future edits are not compile-gated; the build receipt is
local-only. Whether the owner wants `main.pdf` tracked (currently gitignored). The
sandboxed session container cannot run the viewer-listener ownership test (pre-existing,
base-commit-reproduced); hosted CI should be consulted as the authoritative full-suite
gate for this branch.

#### Review Focus

Claim hygiene: does any black sentence exceed its ledger row (development vs confirmatory,
single-scene scope, post-fit VRAM boundary)? Are all pending items actually red? Do the
quoted numbers match the cited artifacts?

#### Protected actions not taken

No experiment execution, no claim-ledger edits, no default changes, no results-bearing
artifacts touched, no PR opened.

#### Recommended Next Action

Human pass over the draft (structure, tone, part ordering), then start on the producible
figure slots (Figs. 2, 8, 9, 11 are producible from existing artifacts) and obligation O-3
(preregistration of the parity floor and Arm-1 schedule).

### Review (Claude-paper-draft-driver, self-review, 2026-07-29)

#### Verdict

Accepted

#### Self-reviewed

Yes

#### Correctness

Diff is documentation-only; no code, tests, benchmarks, or ledger files touched. Verified
each quoted number against its source row (C30 ratios/win counts, C28/C29 dB and survival
values, C24 dB/count factors, C20 AUC ratios, C18/C19 scope, E11 cap, ADR-002 stage list
and 380-update phases, Beam defaults from `BeamFusionConfig`). LaTeX builds clean.

#### Evidence Quality

Black statements carry evidence notes to on-disk artifacts; development-only scope is
stated in text wherever C25–C31-class evidence is used; all pending claims are red and
mirrored in the obligations appendix. The draft never states the VRAM or parity headline
as established.

#### Simplicity

Standard article class with minimal, widely available packages; macros are small; no
custom style files. Two-part structure follows the plan document one-to-one.

#### Missing Cases

CI does not compile the draft (no TeX in the environment) — acceptable for a docs-only
artifact, noted in Uncertainties. Bibliography verification and the novelty-ledger pass
are deliberately left open as O-13 rather than silently trusted.

#### Required Changes

None.

#### Optional Improvements

Track `main.pdf` via a release artifact or CI build once a TeX toolchain is available;
produce the four already-producible figures; consider a venue-style variant of
`main.tex` once a target venue is chosen.

### Handoff (Claude-paper-draft-driver, 2026-07-30, v2 restructure)

#### Objective

Rewrite the draft per the owner's review of v1: single-thread storyline, owner style
rules, `TODO:` prefixes on all red items, no "VRAM claim" naming, plain titles, and no
negative-result reporting except as design justification.

#### Reviewed state

v1 draft at commit 7f9dada with PR #19 open. Owner decision recorded above in Human
Decisions.

#### Changes

Restructured `paper/sections/` from the two-part layout into twelve single-thread files
(overview, fields, reconstruction with random/SfM initialization, memory measurement,
tomography, refinement, experiments ending in the ablation study, limitations, conclusion,
TODO appendix). Removed the part headers, claim boxes, and the carrier-history subsection.
Rewrote all prose without em dashes, semicolons, prose colons, or marketing vocabulary.
Replaced the marker macros with `\todo`/`TodoBlock`/`\tbd`/`\figslot`, all rendering a
`TODO:` prefix. Renamed obligations O-1..O-15 to TODOs T-1..T-15. Reworded evidence notes
to `[evidence ...]` without a colon. Retained kept negative findings only as
design-justification sentences (naive Gaussian product, one-sided repair residual, opacity
gauge, density-preserving clone) and moved their numbers into the ablation table. Updated
`paper/README.md`, `paper/figures/README.md`, the `CLAUDE.md` map line, and the
`PAPER_PLAN` pointer.

#### Evidence

tectonic build of the restructured draft: 25 pages, all citations resolved, 0 overfull
boxes, no undefined references. Automated style scan over `sections/` and `main.tex`
returned zero em/en dashes, zero `--` sequences, zero prose semicolons, zero prose colons
outside `TODO:` prefixes, and no banned vocabulary. Structural checkers re-run before
commit (see Current Evidence).

#### Assumptions

Bibliography titles keep their original punctuation because they are citations, not our
prose. Numeric ranges avoid dashes by using "to". The claim-separation requirement of
`PAPER_PLAN_beam_fusion.md` survives as evaluation rules inside the single-thread
structure, which the updated `PAPER_PLAN` pointer records.

#### Uncertainties

The owner announced a sample paper of their own for a fine-grained style pass. Tone and
rhythm may change again once it arrives. Whether the ablation table's retained
development-verdict numbers match the owner's no-negative-results preference is for the
owner to confirm on read-through.

#### Review Focus

Style-rule compliance in rendered output, faithfulness of the restructure to the owner's
storyline, and that no black statement gained strength during the rewrite.

#### Protected actions not taken

No experiment execution, no claim-ledger edits, no default changes, no results-bearing
artifacts touched, no new PR opened (PR #19 updates on push).

#### Recommended Next Action

Owner read-through of the restructured draft, then the fine-grained style pass once the
owner's sample paper arrives, then the producible figures (Figs. 1, 2, 6, 7, 8, 11).

### Review (Claude-paper-draft-driver, self-review, 2026-07-30, v2)

#### Verdict

Accepted

#### Self-reviewed

Yes

#### Correctness

Docs-only diff. All quoted numbers re-checked against their ledger rows during the
rewrite; none changed value or scope. The restructure preserves every equation, the
proposition, and all evidence bindings. LaTeX builds clean and all cross-references
resolve after the section renames.

#### Evidence Quality

Development-only scope labels survived the rewrite. The removed carrier-history section
contained C28/C29 negative narratives, which remain in the repository ledger and are no
longer claimed or contradicted by the paper. Ablation-table verdicts cite C30/C31
unchanged.

#### Simplicity

Fewer files, plain conventional section titles, one storyline, smaller macro surface
(`\todo`, `TodoBlock`, `\tbd`, `\figslot`).

#### Missing Cases

The owner's sample-paper style pass is pending by design. CI still does not compile the
draft.

#### Required Changes

None.

#### Optional Improvements

After the sample-paper pass, consider tightening the remaining long sentences in the
overview and refinement sections.
