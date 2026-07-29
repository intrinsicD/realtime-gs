# Prospective protocol reviews

Every result-bearing experiment needs a review before `init-run`. The reviewer must have a stable
label different from the task owner and must not execute the protected run or inspect sealed
outcomes. Approval says the frozen design is fit to execute; it says nothing about the result.

Use this sequence:

1. Freeze the owner, data, split, seeds, stages, comparators, metrics, gates, resource protocol,
   blockers, and exact command while the task remains `draft`.
2. Run
   `python scripts/experiment_contract.py review-digest experiments/tasks/<task_id>.json`.
3. Copy `experiments/templates/protocol_review.md` to
   `experiments/reviews/<task_id>_PROTOCOL_REVIEW.md`.
4. A distinct reviewer completes the five machine-readable fields and all four narrative sections
   without consuming protected outcomes.
5. Record the same reviewer, verdict, digest, and artifact path in the task's
   `protocol_review`. Set task status to `ready` after approval or `blocked` after rejection.
6. Run `python scripts/experiment_contract.py validate` again. Any protocol change invalidates
   the review and requires a new digest and review.

The digest excludes only the administrative task `status` and the `protocol_review` envelope.
Owner, blockers, data, command, metrics, controls, and every other task field remain bound. At
`init-run`, the run lock also hashes this review artifact so it cannot be edited silently after
execution starts.
