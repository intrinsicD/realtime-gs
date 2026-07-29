# Agent workflow

This document explains how repository instructions, task coordination, research protocols, and
evidence records fit together. `CLAUDE.md` remains the canonical repository contract; this is the
expanded operating model for substantial agent work.

## Authority map

Each surface owns a different question:

| Surface | Authority |
|---|---|
| `CLAUDE.md` | Repository invariants, commands, layout, and skill routing |
| `.agents/state/current-task.md` | The one active substantial task, its roles, turn, handoffs, and verdict |
| `docs/tasks/` | Closed task and review history |
| `docs/ROADMAP.md` | Research milestones and unresolved research questions |
| `experiments/tasks/*.json` | Frozen result-bearing protocols |
| `runs/<task_id>/task.lock.json` | Exact task, data, source, command, and prospective-review binding for one run |
| `docs/EXPERIMENTS.md` | Append-only experiment interpretation |
| `ara/logic/claims.md` | Claim status, falsification rule, and proof binding |

Do not copy status between these surfaces as free prose. Link them. A result-bearing engineering
task names its experiment contract; the experiment contract names its data and command; the run
lock freezes both; the audit disposes of the claim in `ara/` and `docs/EXPERIMENTS.md`.

## What counts as substantial

A task is substantial when it changes behavior, an interface, a dependency, a default, repository
policy, durable state, or a result/claim artifact. Formatting and typo-only edits are exempt.

Before substantial work:

1. Read `CLAUDE.md` and inspect the worktree.
2. Read `.agents/state/current-task.md`.
3. Continue that record only when its Task ID and goal match the request. Otherwise, do not
   overwrite an active task; finish, hand off, or explicitly supersede it first.
4. Fill the task record with observable success criteria, non-goals, the smallest skill sequence,
   risk, and intended maturity.
5. Read the selected skills and relevant code, tests, decisions, and experiment records.

The active record is a coordination layer, not permission to commit, push, run protected data, or
make external changes. Those actions still require the authority implied by the user request and
the repository rules.

## Risk and independent review

Use `Protected` for:

- result-bearing work or capability/default claims;
- architecture or public-contract decisions that are expensive to reverse;
- release-critical or destructive migrations; and
- any task whose failure could silently invalidate evidence.

Everything else is `Standard`. Both classes receive a review. A Driver may self-review when no
second reviewer is available, but only the provisional status is valid. `Accepted` and
`Accepted with follow-up` require a structured Review block with `Self-reviewed: No` and distinct
Driver and Reviewer labels. These labels make responsibility and handoff reconstruction
checkable; they do not cryptographically prove identity.

Two consecutive `Revision required` verdicts without acceptance trigger human escalation rather
than an unbounded third revision loop.

## Maturity vocabulary

Use maturity to prevent a green CPU test from being reported as a working research capability.
The levels are cumulative for implementation and results work:

| Level | What it establishes |
|---|---|
| `Scaffolded` | A seam, API, or fail-closed structure exists; behavior is not yet established |
| `CPU-contracted` | Deterministic CPU/reference tests exercise the promised contract |
| `Pipeline-integrated` | The canonical CLI or pipeline path actually exercises the feature |
| `Calibrated` | A frozen local calibrated scene, saved artifacts, exact metrics, and viewer receipt pass |
| `Claim-ready` | The applicable preregistration, independent results audit, and ARA proof binding pass |
| `Retired` | The replaced path and its stale docs/registrations are removed |
| `Not applicable` | Documentation or policy work for which the ladder is irrelevant |

If a task closes below its target, use `Accepted with follow-up` and name the follow-up task.
Closing at `Scaffolded` is valid only when that endpoint is an explicit non-goal or a named task
owns the next maturity level.

## Handoff and verdict format

Driver handoff:

```markdown
### Handoff

#### Objective
#### Reviewed state
Commit or exact dirty-tree/source digest.

#### Changes
#### Evidence
#### Assumptions
#### Uncertainties
#### Review Focus
#### Protected actions not taken
#### Recommended Next Action
```

Reviewer response:

```markdown
### Review

#### Verdict
Accepted / Accepted with follow-up / Revision required / Rejected / Inconclusive

#### Self-reviewed
Yes / No

#### Correctness
#### Evidence Quality
#### Simplicity
#### Missing Cases
#### Required Changes
#### Optional Improvements
```

Treat a producer-authored handoff as untrusted orientation. Recompute important hashes, rerun the
relevant checks, inspect negative controls, and state which protected data or one-shot action was
not consumed during review.

## Result-bearing task path

Result-bearing work has two reviews with different timing:

1. A prospective protocol review before an official run. The experiment task records a distinct
   reviewer, an `approved` verdict, an exact protocol digest, and a review artifact under
   `experiments/reviews/`. The digest excludes only administrative `status` and the review
   envelope, so changing the owner, blockers, data, command, metrics, controls, or any other
   protocol field invalidates approval. The review artifact records exact task/digest/reviewer/
   verdict fields and `Outcome Access: none`; `init-run` hashes the artifact into the run lock.
2. The existing `realtime-gs-results-audit` referee pass after outcomes exist. It recomputes raw
   results, checks provenance and controls, and confirms, narrows, retires, or leaves unresolved
   every claim.

The prospective reviewer must not execute the protected run or inspect sealed outcomes. Approval
only means the protocol is fit to execute; it is not evidence that the method works.

## Research ideation path

Open-ended method discovery starts with `rtgs-research-ideation`. It maps the repository frontier,
generates independent idea lanes, audits prior art, and returns falsifiable candidates with cheap
killing tests. It does not silently create roadmap commitments or modify production code.

After the user selects a candidate:

1. open or update the active task;
2. record a consequential, hard-to-reverse decision in an ADR when warranted;
3. create the experiment contract before outcome-bearing implementation or execution;
4. run `rtgs-experiment` or `rtgs-bench`;
5. run `realtime-gs-results-audit`, `rtgs-review`, docs sync, and verification.

## Structural enforcement

`python scripts/check_agent_workflow.py` validates:

- the active/archived task schema, status/turn pair, maturity, review verdict, and identity
  separation, including complete structured handoff/review fields;
- experiment-task links from the active task;
- skill names and exact `.claude/skills` ↔ `.agents/skills` discovery coverage;
- the canonical `AGENTS.md` redirect; and
- the pull-request review surface and CI invocation of `./scripts/verify.sh`, preventing human,
  local, and hosted gates from drifting.

The checker is part of `./scripts/verify.sh`. The pull-request template exposes the same decisions
to human review.

For periodic health review, inspect recent substantial changes together rather than only one diff:
look for silent scope growth, half-finished paths described as shipped, documented-but-untested
behavior, stale defaults or task state, unnecessary abstraction, untracked shims, orphaned skills,
and stronger prose than the ARA evidence. Run such a review after a workflow escape or before a
release; do not add a calendar ceremony when the structural checks and per-task reviews suffice.

## Comparative adoption record

This workflow deliberately integrates mechanisms rather than copying another repository:

| Source | Integrated here | Deliberately not copied |
|---|---|---|
| `agent-kit` | Driver/Reviewer/Turn state, durable handoff, terminal verdict, provisional self-review, bounded revision loop | Parallel backlog/state/ideas authorities and a wholesale installer payload |
| `IntrinsicEngine` | Always-first core routing, explicit maturity, PR review prompts, CI/mirror drift enforcement | Large dependency task graph, generated session brief, engine-specific CI matrix, fixed weekly ceremony |
| `structsplat` | Repository-prefixed task and research-ideation skills, task-policy validation, exact discovery-mirror check | Its renderer/rate-accounting task taxonomy and duplicate `tasks/INDEX.md` backlog |
| `prospect` | Untrusted prospective handoff, distinct pre-execution reviewer, exact protocol digest, outcome-access boundary | Prospect’s epistemic object model and maintainer-owned one-shot authorization runtime |

The existing calibrated-scene, result-bundle, viewer, experiment, and ARA gates remain
authoritative. The additions close coordination and prospective-review gaps without weakening or
replacing those stronger repository-specific controls.
