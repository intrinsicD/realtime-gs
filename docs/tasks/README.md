# Agent task archive

`realtime-gs` keeps one durable active task in `.agents/state/current-task.md`. Completed,
rejected, inconclusive, and superseded records are archived here as
`RTGS-NNN-<lowercase-slug>.md`.

This archive is execution and handoff history, not a second backlog:

- `docs/ROADMAP.md` remains the authority for research milestones and open research questions.
- `experiments/tasks/*.json` remains the immutable protocol authority for result-bearing runs.
- `ara/logic/claims.md` remains the authority for claims and their evidence.
- `docs/EXPERIMENTS.md` remains the append-only interpretation ledger.

Use the `rtgs-task-workflow` skill to open, hand off, review, and close a task. The structural
contract is enforced by `python scripts/check_agent_workflow.py`.

An independently accepted task has distinct Driver and Reviewer labels and a structured Review
verdict. When only one agent is available, it may self-review, but the record stays
`Provisionally accepted (self-reviewed)` in the active slot until a different reviewer or a human
accepts it. Labels are cooperative provenance, not authenticated identity.
