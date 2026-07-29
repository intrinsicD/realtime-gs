# Current Task

## Title

Mature the repository-native agent workflow

## Task ID

RTGS-001

## Role Assignment

- Driver: Codex
- Reviewer: Codex
- Turn: driver

## Mode

Stabilize

## Risk

Protected

## Maturity

- Target: CPU-contracted
- Reached: CPU-contracted

## Goal

Finish a bounded, mechanically enforced agent workflow that coordinates substantial work,
keeps skill discovery and CI aligned, and requires prospective review of exact experiment
protocols without duplicating the repository's roadmap, experiment, or claim authorities.

## Motivation

The repository already has strong calibrated-data, results-bundle, experiment, audit, and ARA
gates, but it lacked durable ordinary-task state, mechanically checked Driver/Reviewer/Turn
semantics, an always-first router, prospective protocol review, exact skill-mirror coverage, and
one canonical local/CI verification path.

## Success Criteria

- The active/archive task schema, role/turn/verdict rules, maturity states, task links, skill
  mirror, canonical guide redirect, settings exposure, and CI parity are structurally checked.
- All registered experiment tasks use the prospective-review schema, and `init-run`, run locks,
  report rendering, and run validation bind the exact reviewed protocol.
- The core, task-workflow, and research-ideation skills validate and are discoverable through
  exact `.agents/skills` symlinks.
- Documentation, templates, tests, PR prompts, and repository commands describe one coherent
  workflow with explicit authority boundaries.
- Focused tests and `./scripts/verify.sh` pass.

## Constraints

- Preserve CPU-first operation, append-only result history, and existing roadmap/experiment/ARA
  authority.
- Do not execute experiments, consume protected outcomes, change defaults, or publish claims.
- Do not import wholesale backlog/task-graph or epistemic-runtime systems from peer repositories.
- Preserve unrelated existing repository evidence and local data.

## Non-Goals

- Running benchmarks or research experiments.
- Replacing `docs/ROADMAP.md`, `experiments/tasks/`, `docs/EXPERIMENTS.md`, or `ara/`.
- Cryptographically authenticating agent identity or adding a parallel orchestration service.
- Opening a PR, changing release/issue state, or representing publication as independent review.

## Selected Skills

- `rtgs-core`
- `rtgs-task-workflow`
- `skill-creator`
- `rtgs-docs-sync`
- `rtgs-review`
- `rtgs-verify`
- `github:yeet`
- `research-manager`

## Experiment Contract

None

## Current Evidence

- Read-only comparison completed against `agent-kit`, `IntrinsicEngine`, `structsplat`, and
  `prospect`.
- `docs/AGENT_WORKFLOW.md`, the task checker, three skills, and task/archive scaffolding exist.
- All registered experiment tasks and the template use schema 2 with a pending prospective-review
  envelope.
- Sixteen focused workflow/experiment-contract tests pass; all three new skills validate; the
  research-portfolio validator self-test passes.
- `./scripts/verify.sh` passes end to end with an explicit CPU-only test boundary.

## Minimal Plan

1. Complete the experiment-review schema and migrate registered tasks/templates.
2. Wire workflow checks into scripts, settings, CI, docs, and the PR surface.
3. Add focused structural tests and validate all three skills.
4. Self-review the complete diff, run docs sync and full verification, and record a provisional
   handoff because no independent reviewer is available.
5. With explicit user authorization, publish the exact verified provisional state to `main`.

## Status

Provisionally accepted (self-reviewed)

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

Should the provisionally self-reviewed RTGS-001 worktree be committed directly to `main` and
pushed before a distinct reviewer promotes or archives the task?

### Options

- Keep the verified work local until independent review.
- Publish the verified provisional state while preserving its explicit review limitation.

### Recommendation

Publish only with explicit human authorization, retain the provisional task status, and avoid
representing publication as independent technical acceptance.

### Decision

On 2026-07-29, the user explicitly instructed Codex to commit all changes, integrate them on
`main`, and push them to the remote. The repository was already on `main`, so no synthetic merge
commit or branch indirection is required while `main` remains synchronized with `origin/main`.

### Date

2026-07-29

## Handoff Log

### Resume after requested pause

#### Objective

Complete the preserved workflow-maturity migration without treating the partial tree as green.

#### Reviewed state

Dirty worktree recorded on 2026-07-29 before resumption; `scripts/experiment_contract.py` had the
first schema patch applied and `.tmp_experiment_contract_2.patch` remained unapplied.

#### Changes

Opened this task record and resumed inspection only.

#### Evidence

`git status --short --branch`, the full changed-file diff, canonical guide, selected skills, and
the prepared patch were read before further editing.

#### Assumptions

`RTGS-001` is unused because `docs/tasks/` contains no archived task records.

#### Uncertainties

Focused tests may expose schema or compatibility issues in the prepared experiment patch.

#### Review Focus

Fail-closed protocol review, state-machine correctness, migration compatibility, CI parity, and
avoiding duplicate authority.

#### Protected actions not taken

No experiment, benchmark, result promotion, default change, commit, push, or external action.

#### Recommended Next Action

Complete the smallest coherent schema slice, then test it before broad documentation changes.

### Handoff

#### Objective

Deliver a mature repository-native agent workflow without weakening or duplicating the existing
experiment, results, calibrated-data, viewer, or ARA evidence gates.

#### Reviewed state

At the review point on 2026-07-29, the tracked binary diff SHA-256 was
`e3ac406f9a104a6d409485faebec8d4bfcf96cf761269b6bffa301ace6ace42f`; the sorted
untracked-path/content manifest SHA-256 was
`2627ebf01060b74bd10ce3106871548f97f2d96ea3e0e77e63e5ecb5193d09e3`. Final task-state and
research-manager metadata are expected append-only changes after that review point.

#### Changes

- Added one durable active-task/archive workflow with checked roles, turns, maturity, handoffs,
  verdicts, bounded revision loops, and provisional self-review.
- Added the always-first `rtgs-core` router, `rtgs-task-workflow`, and adversarial
  `rtgs-research-ideation` skill with exact Codex discovery symlinks.
- Added prospective experiment review bound to a stable protocol digest, structured no-outcome
  review artifact, task/run lock, artifact hash, command, data seal, and canonical report.
- Added the workflow checker, human PR prompts, settings exposure, script-layout registration,
  canonical CI delegation to `verify.sh`, and an explicit CPU-only verification boundary.
- Migrated all registered experiment tasks and templates to schema 2 and removed pause-era scratch
  patches.

#### Evidence

- `16 passed`: focused `tests/test_agent_workflow.py` and `tests/test_experiment_contract.py`.
- `experiment_contract: OK` for all registered tasks/programs.
- Three `quick_validate.py` skill passes and research-portfolio validator `self-test: PASS`.
- `./scripts/verify.sh`: Ruff, format, full non-slow CPU suite, docs sync, ARA, script layout,
  agent workflow, and experiment contracts all pass.
- `git diff --check` passes.

#### Assumptions

Stable Driver/Reviewer labels are cooperative provenance rather than authenticated identity.
The empty task archive makes `RTGS-001` the next unused repository task ID.

#### Uncertainties

No independent reviewer has reproduced the checks. CUDA-specific tests were deliberately excluded
from the CPU verification gate and were not needed because no runtime implementation changed.
The prospective-review outcome-access boundary is structurally declared and artifact-bound, not
cryptographically provable.

#### Review Focus

Check that administrative status is the only non-review protocol field excluded from the digest;
that the active task is not a second roadmap; that CI cannot omit a local gate; and that no
documentation overstates self-review, CPU evidence, or prospective approval.

#### Protected actions not taken

No experiment or benchmark was executed, no sealed outcome was consumed, no result/default/claim
was promoted, and no commit, push, PR, branch operation, or external change was made.

#### Recommended Next Action

A distinct reviewer or human should inspect this exact state, rerun the strongest checks, append
an independent verdict, and archive `RTGS-001` only if accepted.

### Review (self-review)

#### Verdict

Accepted

#### Self-reviewed

Yes

#### Correctness

The task/status/turn and review state machine is fail-closed; exact skill mirrors and canonical
authority paths are checked; prospective approval is invalidated by every protocol mutation but
the administrative status transition; and run validation rejects task, review artifact, command,
data, schema, and source-metadata drift.

#### Evidence Quality

Focused positive and negative structural tests cover identity normalization, full review shape,
revision escalation, follow-up linkage, digest stability, outcome-access declaration, report
rendering, and lock drift. The complete repository verification then passed on the intended CPU
surface.

#### Simplicity

The design retains one active task file and one archive, delegates canonical verification to one
script, and links rather than recreates roadmap, experiment, result, and ARA authorities.

#### Missing Cases

Independent acceptance and CUDA-specific execution are absent. No archived real task exists yet,
although archive naming and terminal-state behavior are structurally checked.

#### Required Changes

None for provisional self-reviewed acceptance.

#### Optional Improvements

After independent review, archive this record and reset the active slot; use the first real task
closeout as an additional forward test of archive ergonomics.
