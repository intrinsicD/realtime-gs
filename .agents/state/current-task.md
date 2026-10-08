# Current Task

## Title

Jet-consistency (Weingarten) prior for 3DGS training on TOSCA cat0

## Task ID

RTGS-030

## Role Assignment

- Driver: Claude-Code-Fable-5.1-driver
- Reviewer: Alexander Dieckman
- Turn: reviewer

## Mode

Validate

## Risk

Protected

## Maturity

- Target: Calibrated
- Reached: Calibrated

## Goal

Implement the opt-in jet-consistency prior (`rtgs.optim.jet_prior`, `--jet-lambda/--jet-order`) and run the preregistered screen.

## Motivation

SplatDiffuseLBO measured noisy trained splat normals (16.5 deg median) and a thick shell on the cat; the PREREG asks whether a Weingarten consistency term fixes both during training.

## Success Criteria

PREREG decision recorded with every prediction marked; bundle passes check-run and the results-bundle gate; verify.sh passes; defaults unchanged.

## Constraints

Host RAM < 8 GB; one GPU job at a time; companion repository read-only. ("No commits" superseded 2026-10-08, see Human Decisions.)

## Non-Goals

Default changes; real-capture confirmation; running the SplatDiffuseLBO operator.

## Selected Skills

- rtgs-experiment
- rtgs-verify

## Experiment Contract

experiments/tasks/20261007_jet_consistency_prior_tosca_cat0.json

## Current Evidence

`benchmarks/results/20261007_jet_consistency_prior_tosca_cat0_RESULT.md` (reject; neighbour collapse), self-audit `..._AUDIT.md`, claim C54.

## Minimal Plan

Done: implementation, data preparation, contract, 11 cells, evaluation, report. Open: reviewer reading of the generated protocol fields and the result; independent audit; deferred operator metric.

## Status

In review

## Human Decisions

Record escalated questions and dated answers here. An answer that exists only in chat is not
durable task state. Use one block per decision:

```markdown
### Question
### Options
### Recommendation
### Decision
### Date
```

### Question
Commit the uncommitted RTGS-030 work, and which regularizer does the next registration test?
### Options
(a) v2.1 field-normal prior as registered (no shape operator); (b) an explicit 2-jet prior on 2DGS surfels.
### Recommendation
Commit on a branch first; the v1 protocol-review record is marked invalid as review evidence.
### Decision
Commit on branch `rtgs-030-jet-prior`. The next registration tests the 2-jet explicitly (jet2 vs
jet1 vs base) on 2DGS surfels so the 1-jet is well defined and trained; v2.1 is not run as the
test of this idea. Collapse protection, teacher check of S and the operator metric budget come
before the PREREG.
### Date
2026-10-08 (Alexander Dieckman, in chat with the coordinator session)

## Handoff Log

Append Driver handoffs, Reviewer verdicts, and session-completion entries in chronological order.
Use `###` for entries and `####` for their fields so entries remain nested below this section.
Never delete earlier entries. On terminal closeout, archive the complete record as
`docs/tasks/<task-id>-<slug>.md`, change the archived `Turn` to `none`, and reset this file to the
unchanged template.

### Handoff (2026-10-07, result for the reviewer)

#### Objective
Reviewer reading of the jet-prior implementation, the generated protocol fields and the result.

#### Reviewed state
Uncommitted worktree on 27c6ce5; protocol digest `03e3b661ebaa33203df3d2f8786858d68633d1f336461257b403647cfb9c35a1`.

#### Changes
`src/rtgs/optim/jet_prior.py`, `Trainer.train(jet_prior=...)`, `rtgs refine|run --jet-lambda/--jet-order`, `run_pipeline(jet_prior=...)`; data preparation, evaluation and task driver under `scripts/experiments/`; task, seal, review record, RESULT/AUDIT, C54, EXPERIMENTS entry.

#### Evidence
`runs/20261007_jet_consistency_prior_tosca_cat0/` (check-run OK, results-bundle gate OK); `tests/test_jet_prior.py`.

#### Assumptions
The paraboloid sign follows the companion code convention; the PREREG sign is treated as a slip.

#### Uncertainties
Self-audit only; operator metric deferred; the digest approval record was written by the driver from the reviewer's content approval.

#### Review Focus
The deviations listed in the protocol review record, and the neighbour-collapse reading of the failure.

#### Protected actions not taken
No commits; no default changes; nothing modified in the companion repository; no rerun or retuning after outcomes.

#### Recommended Next Action
Reviewer reads RESULT.md and the protocol review record; if the collapse reading holds, register a new task with a minimum-radius neighbourhood.

### Session note (2026-10-08, coordinator)

#### Changes
Worktree committed on `rtgs-030-jet-prior`; v1 PROTOCOL_REVIEW annotated as not an independent
review; new negative control `test_knn_prior_collapse_minimum_from_clones`; SplatDiffuseLBO
SPEC3D §11.1-11.2 sign text corrected to the code convention (companion commit 7ed8601).

#### Evidence
Mutation check: flipping the estimator sign or the height-residual sign in `jet_residuals` fails
`test_sphere_residual_small_and_shape_operator`. Collapse: with 8 copies per splat the 16-NN
r_nu of a random normal field drops to ~1e-15 (one copy: 0.61 -> 0.45 only).

#### Recommended Next Action
Teacher check of estimated S from trained 2DGS splats against mesh S (vs S = 0), and an
operator-metric budget under 8 GB, before a v3 PREREG.

### Plan reviews (2026-10-08, coordinator: Claude session; author of plan and P1 code)

#### Object
`docs/TASK_jet2_synthetic_sphere_ellipsoid.md` rev. 1 (round 1) and rev. 2 (exchange; SHA-256
652189ee7cf0fe50f8b647c7f67cf8b23093e81a275ff2335ef27f2875169398 per Codex).

#### Reviewers
Codex codex-cli 0.161.0, requested gpt-6-astra / xhigh, thread 01a11b57-2016-7921-8ca1-1d2686f66b47,
read-only sandbox (probe OK); recursion protection instruction-only (other MCP/plugin tools not
disabled). Claude Code 2.1.294, claude-opus-5-5 / max, session 11cc7558-abe8-46f9-9df8-6ce919fce3ee,
tools Read/Grep/Glob, recursion protection enforced; could not read SplatDiffuseLBO in round 1 and
realtime-gs in the exchange round (directory permissions of the callee session).

#### Verdicts
Round 1: Codex `revise`, Claude `revise`. Exchange: Codex "revise for P1/P2 as written"; Claude
"revise" (P1 may proceed once its test list covers the failure modes and the estimator fixes are
in the design). Full answers kept in the session scratch (/tmp/peer, not durable); the binding
requirements are carried into plan rev. 3. Disagreement "sphere cannot separate jet1/jet2" settled
by an analytic check (cap test), not by argument.
