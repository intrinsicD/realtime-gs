# Current Task

## Title
Fresh-checkout verification for the work handoff

## Task ID
RTGS-017

## Role Assignment

- Driver: Codex-delivery
- Reviewer: Codex-delivery
- Turn: none

## Mode
Stabilize

## Risk
Standard

## Maturity

- Target: CPU-contracted
- Reached: CPU-contracted

## Goal
Make the existing verification gate reproducible in a clean work checkout.

## Motivation
The user requested both repositories on remote main for use at work. Hosted CI exposed two
pre-existing local-environment dependencies after the code push.

## Success Criteria
The CPU verification gate passes without ignored run folders, using the same PyTorch release
as the completed local gate. Preserve the pinned clamp-gradient assertion and frozen experiment
source, protocols, results and data seals. Commit and push the bounded repair.

## Constraints
No reconstruction rerun or production-source change. Do not modify old evidence to satisfy tests.
The user cancelled the private archive upload and prohibits additional image or mask uploads.

## Non-Goals
Changing renderer derivatives, proving compatibility with every PyTorch release, changing the
frozen tomography study, or promoting a research claim.

## Selected Skills

- rtgs-core
- rtgs-task-workflow
- rtgs-review
- rtgs-docs-sync
- rtgs-verify

## Experiment Contract
None

## Current Evidence
GitHub run 34067068370 installed torch 2.14.0+cpu and failed the preserved zero-boundary clamp
assertion. It also failed an assertion that an ignored historical run folder exists. The local
2.9.0+cu128 gate passed. Raw hosted failure is preserved in the local work-handoff bundle.
These are environment and test-portability observations, not reconstruction results.

## Minimal Plan
Pin CI and the documented CPU setup to the exercised torch 2.9.0 release. Replace the local-folder
assertion with binding to the tracked historical result. Run the full gate in a clean checkout,
self-review the bounded diff, and push the repair.

## Status
Superseded

## Human Decisions
2026-09-07: The user explicitly requested commit, merge to main and push for both repositories
for use at work. Repairing fresh-checkout verification is within that delivery scope.

## Handoff Log

### Start — 2026-09-07
The original code and evidence commits are already pushed. Preserve them and add a narrow
portability commit. No separate reviewer or new experiment is selected.

### Handoff (2026-09-07)

#### Objective
Make the CPU verification gate portable to a fresh checkout for the authorized work handoff.
#### Reviewed state
731beda plus the CI/README torch pin, verification skill note and tracked-result assertion.
#### Changes
Pin CPU verification to torch 2.9.0; replace the ignored historical run-directory assertion with
its tracked result/task binding. No source, protocol, original result or tolerance changes.
#### Evidence
Focused clean-checkout workflow, source-boundary, clamp-gradient and protocol tests pass after
binding PYTHONPATH to the clean checkout. The first auxiliary checkout run detected its shared
editable install pointing at the original tree; these source-origin failures were not suppressed.
The corrected full gate is the pre-commit requirement. Its receipt is stored separately.
#### Assumptions
The exercised PyTorch release is the CPU verification baseline, not all-version compatibility.
#### Uncertainties
Post-push hosted verification remains pending. The private archive upload still needs approval.
#### Review Focus
Retain exact gradient/source guards and old evidence; check clean-checkout imports and CI pin.
#### Protected actions not taken
No reconstruction rerun, scientific claim/default promotion, source change or independent acceptance.
#### Recommended Next Action
Pass the corrected full gate and push this bounded repair under the existing user authorization.

### Review (2026-09-07)

#### Verdict
Accepted
#### Self-reviewed
Yes
#### Correctness
The environment pin preserves the exercised derivative contract. The protocol assertion now
binds to tracked historical evidence without requiring private local execution outputs.
#### Evidence Quality
Hosted failure and focused local reproduction; no test thresholds, source guards or old result
bytes changed. Full corrected clean-checkout verification remains a pre-commit condition.
#### Simplicity
One installer pin and one portable evidence assertion; no numerical workaround.
#### Missing Cases
Newer PyTorch compatibility and post-push hosted confirmation.
#### Required Changes
Pass the full gate before commit and preserve the provisional review scope.
#### Optional Improvements
A separate dependency-upgrade task may investigate changed boundary-gradient semantics.

### Final local verification (2026-09-07)
The full corrected clean-checkout gate and the no-write CI benchmark smoke passed. Logs remain
local in `.scratch/work-handoff-20260907/`; they are outside the public code commit. Frozen
experiment source and data seals remain unchanged. Hosted post-pin confirmation is pending;
this task retains its explicit provisional self-review status.

### Delivery instruction (2026-09-07)
The user declined the 8.16 GB upload and instructed no additional images or masks. The empty
private draft release was removed without uploading any asset. Preserve earlier pending-upload
entries as history; the current transfer decision is cancellation. Hosted CI for the CPU repair
at commit 16dde3c passed; this does not change the task's provisional self-review disposition.

### Coordination successor (2026-09-07)
The delivered verification repair remains provisionally self-reviewed as recorded above.
RTGS-018 supersedes this coordination record for the user-requested cross-repository
implementation and independent review session. This archive does not promote the old review.
