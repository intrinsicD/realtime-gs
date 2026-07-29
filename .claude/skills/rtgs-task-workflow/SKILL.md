---
name: rtgs-task-workflow
description: Frame, coordinate, hand off, independently review, and close substantial realtime-gs work through the single durable task record. Use when work changes behavior, interfaces, dependencies, defaults, policy, durable repository state, or result/claim artifacts; when resuming another agent's work; when a task needs success criteria or maturity boundaries; or when recording a review verdict or human decision.
---

# Task workflow

The active record is `.agents/state/current-task.md`; closed records live in `docs/tasks/`.
`docs/AGENT_WORKFLOW.md` defines the authority boundaries and maturity vocabulary.

## Start or resume

1. Read `CLAUDE.md`, the active record, and the relevant code/evidence.
2. If the active Task ID matches the request, confirm its recorded `Turn` permits you to act.
3. If the active record is populated for different work, do not overwrite it. Finish, hand off,
   explicitly supersede, or seek a human decision.
4. For a new task, copy the unchanged template in place and allocate the next unused `RTGS-NNN`
   across `docs/tasks/` and the active record.
5. Fill every field. State observable success criteria, non-goals, exact verification, the smallest
   useful plan, risk, target/reached maturity, selected skills, and current evidence.
6. Use `Protected` for results/default/capability claims, hard-to-reverse architecture, releases,
   destructive migration, or evidence-critical behavior.
7. Link a result-bearing task to one exact `experiments/tasks/<task_id>.json`. The experiment JSON
   remains the protocol authority; do not duplicate its arms, seeds, gates, or command in prose.

Do not create a task merely to satisfy ceremony. Formatting and typo-only changes are trivial.

## Roles and turn

- The Driver scopes and implements the smallest viable patch, collects evidence, self-audits, and
  writes the handoff.
- The Reviewer starts from the reviewed state, distrusts the handoff, reproduces important checks,
  hunts counterexamples and unnecessary complexity, and records a verdict.
- `Turn` is a durable protocol marker, not a filesystem lock. Agents sharing a worktree act
  serially.
- Use stable labels supplied by the user/tool. Labels are provenance, not authenticated identity.

Status and turn must agree:

- `Not started`, `In progress`, `Revision required` → `driver`
- `In review` → `reviewer`
- `Blocked on human decision` → `human`
- terminal/provisional dispositions → `driver`

## Implement

1. Work against the recorded success criteria only.
2. Preserve buildability and CPU-first imports after each meaningful slice.
3. Add tests for behavior; add parity and backend-labelled evidence when applicable.
4. Update docs and task maturity in the same change.
5. For a result-bearing path, obtain the exact prospective protocol review before `init-run`; do
   not let the prospective reviewer consume the protected run or sealed outcomes. Freeze the
   owner and protocol while status is `draft`, run `review-digest`, write the canonical
   `experiments/reviews/<task_id>_PROTOCOL_REVIEW.md`, then record approval and change status to
   `ready`. Any protocol edit requires another review.
6. Run focused checks, then `./scripts/verify.sh`.
7. Append a Driver handoff; never replace earlier log entries.

Use this handoff shape:

```markdown
### Handoff

#### Objective
#### Reviewed state
#### Changes
#### Evidence
#### Assumptions
#### Uncertainties
#### Review Focus
#### Protected actions not taken
#### Recommended Next Action
```

Bind the reviewed state to a commit or an exact source/diff digest. Name weak points and skipped
hardware explicitly.

## Review

Review the claim or behavior, not just formatting:

1. Restate the required outcome and maturity target.
2. Inspect the whole diff and exact reviewed state.
3. Reproduce the strongest meaningful checks.
4. Search for counterexamples, leakage, hidden fallback, state drift, false completion, and a
   simpler implementation.
5. For results, use `realtime-gs-results-audit`; a generic code review cannot replace it.
6. Append:

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

An independent acceptance uses distinct Driver/Reviewer labels and `Self-reviewed: No`. If only
the Driver is available, set both labels to that agent, `Self-reviewed: Yes`, and status
`Provisionally accepted (self-reviewed)`. Do not archive it as accepted until another reviewer or
a human promotes it.

After two consecutive `Revision required` verdicts, record the unresolved choice under Human
Decisions and set `Blocked on human decision`; do not start an unbounded third loop.

## Close

- `Accepted`: reached the target maturity.
- `Accepted with follow-up`: name a distinct `RTGS-NNN` that owns the maturity shortfall.
- `Rejected`: preserve what was ruled out; do not merge rejected implementation merely to retain
  its task record.
- `Inconclusive`: continue with the cheapest resolving test or close with the missing evidence.
- `Superseded`: name the replacement/reason; no fabricated reviewer verdict is needed.

For terminal closeout, copy the complete record to
`docs/tasks/RTGS-NNN-<lowercase-slug>.md`, change only archived `Turn` to `none`, reset the active
file to its exact template, and run:

```bash
python scripts/check_agent_workflow.py
./scripts/verify.sh
```

Rejected/inconclusive implementation should use a metadata-only closeout commit on the default
branch rather than merging the rejected code. Never perform branch, commit, merge, push, or
cherry-pick operations unless the user's request authorizes them.
