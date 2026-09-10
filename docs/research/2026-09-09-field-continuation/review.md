# Independent documentation review: RTGS-022

Reviewer: Codex-continuation-reviewer, distinct from Driver Codex-continuation-discussion.
Date: 2026-09-09. Recorded from the reviewer's returned verdict.

## Verdict

Accepted for the documentation/recommendation scope of RTGS-022. This is not experiment
approval or validation of a proposed remedy.

## Self-reviewed

No.

Reviewed hashes:

- Continuation note: `655cc17f9e7fa995a36d2b9c4f12d32d19302eacf204db66b4f89118137cd6d9`
- Collaboration receipt: `8081cbc0a68f0dc8317086383b9a76f2b347ddf79d118fbfc08e3cd6701d3c7b`

## Correctness

The proposed ordering is defensible: measure actual target residuals and matched-state
gradient differences, then select controlled interventions. Paired parameter states,
geometry/topology distinctions, oracle-information labels and splice artifacts are
handled explicitly.

The final note correctly narrows three remaining Claude statements: Adam moments affect
updates rather than ordinary loss gradients; an occlusion probe cannot uniquely identify
causes or assume a suitable camera exists; an incorrect observation model need not fail
on every phantom.

The tomography control uses the correct normalized parallel-ray marginal and separate
density/projection evaluation. Field-only inputs and declared learned priors are
distinguished. Failed RTGS-021 gates remain failed; no causal, high-quality or end-to-end
success is implied.

## Evidence Quality

Both saved prompts and answers match their receipt hashes; both recorded processes
exited successfully. The discussion is clearly distinguished from an independent audit
of private outcomes. Primary-source spot checks support the R2-Gaussian normalization,
known-pose COLMAP route, and volumetric rasterizer limitations.

C47, the RTGS-021 audit MD/JSON and its frozen protocol are byte-identical to HEAD.
`git diff --check`, `check_agent_workflow.py` and `check_ara.py` passed. The reviewer did
not rerun reconstruction or the full test suite.

## Simplicity

One bounded diagnostic is recommended first. Later interventions are conditional,
without prescribing a broad implementation program or reviving retired methods.

## Missing Cases

Execution details remain appropriately unapproved: exact state/view selection, gradient
normalization and near-zero tolerances, intervention budgets, stopping rules, and fresh
confirmation allocation require a prospective protocol. These omissions do not block
this discussion artifact.

## Required Changes

None.

## Optional Improvements

When converting the recommendation into a protocol, report gradient comparisons within
parameter groups using declared parameterizations and near-zero handling; avoid
interpreting raw norms across differently scaled groups as relative causal importance.

The Driver owns final verification, current-turn research recording and task closure.
