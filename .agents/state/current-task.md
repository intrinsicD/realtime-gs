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
`docs/PAPER_PLAN_beam_fusion.md` as a full paper skeleton: Part I carries the systems/VRAM
claim, Part II the Beam Fusion approach, with red markers for everything still to be shown
(including exact figure specifications) and well-explained black text for implemented
mechanism and evidence-bound results.

## Motivation

The human owner requested the full paper program as proper `.tex` files with red text for
open evidence and required figures. The plan documents (`PAPER_PLAN_beam_fusion.md`,
`THREE_ARM_EXPERIMENT_PROGRAM.md`, `20260725_claims_and_questions.md`) define claims and
discipline but no manuscript structure; a skeleton that encodes the claim firewall and the
open-obligation ledger makes the remaining experimental program explicit and reviewable.

## Success Criteria

- `paper/main.tex` compiles cleanly (no undefined references/citations, no overfull boxes).
- Two claim-separated parts with the Arm-1/Arm-2 firewall and predeclared decision rules.
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

- Compile receipt: tectonic 0.15.0, 27 pages, BibTeX resolves all citations, 0 overfull
  boxes, no undefined references (local build; no LaTeX in CI, so the draft is not gated).
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

None.

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
