# Current Task

## Title

Discuss the post-RTGS-021 continuation with Claude

## Task ID

RTGS-022

## Role Assignment

- Driver: Codex-continuation-discussion
- Reviewer: Codex-continuation-reviewer
- Turn: none

## Mode

Decide

## Risk

Standard

## Maturity

- Target: Not applicable
- Reached: Not applicable

## Goal

Deliver a bounded continuation recommendation after discussing public mechanisms and experimental design with Claude and relating the critique to the already audited local results.

## Motivation

The user asks how to continue after the failed reconstruction-quality prerequisites, while retaining the tomography question.

## Success Criteria

Complete a real two-way Claude discussion; preserve its scope and responses; distinguish observed outcomes from hypotheses; recommend a smallest diagnostic and explicit conditional next steps; retain the frozen experiment result and gates.

## Constraints

Discussion and documentation only. Claude receives public mathematics and general test designs without private repository/capture context. Preserve all completed runs, protocols, scores and audits. Existing no-additional-image/mask upload instruction persists. No new commit or push requested in this discussion turn.

## Non-Goals

No reconstruction execution, new protocol approval, production change, causal success claim, roadmap commitment or automatic reopening of RTGS-021 D/E.

## Selected Skills

- rtgs-core
- rtgs-task-workflow
- rtgs-review
- rtgs-verify
- research-manager at end

## Experiment Contract

None

## Current Evidence

Both public Claude rounds and exact receipts are saved. Codex-continuation-reviewer independently accepted the bounded recommendation at the recorded note/receipt hashes. Current-turn proposals remain staged; C47 and frozen experiments are unchanged. Final repository verification is recorded in docs/research/2026-09-09-field-continuation/verification.json.
## Minimal Plan

Review the existing decision tree; obtain and challenge Claude's general critique; verify primary sources; write a concise continuation note; review, verify and record only current-turn proposals.

## Status

Accepted

## Human Decisions

2026-09-09: The user says “Now discuss with Claude how we should continue.” This authorizes discussion; it does not select or authorize execution of a proposed follow-up experiment.

## Handoff Log

### Start (2026-09-09)

Both repositories were clean and up to date immediately before the request. realtime-gs is at 66afd22 and StructSplat at 03ae7d4. This is a focused reassessment of the existing plan after RTGS-021, not a new novelty portfolio or a results-bearing run. Claude ListAgents reports no reachable sessions, so the public-only CLI discussion uses the same isolation boundary as RTGS-020.

### Handoff

#### Objective
Deliver the completed Claude discussion and a conditional continuation recommendation without implying experimental approval or high-quality success.

#### Reviewed state
Base commit 66afd22fb4b1eebf86904c21cac19ccf61c9fca3. Final discussion note SHA-256 655cc17f9e7fa995a36d2b9c4f12d32d19302eacf204db66b4f89118137cd6d9; collaboration receipt SHA-256 8081cbc0a68f0dc8317086383b9a76f2b347ddf79d118fbfc08e3cd6701d3c7b. Source, original protocols, results and audits remain unchanged.

#### Changes
One focused continuation note, its navigation link, both public Claude prompts/responses and their process/model receipts. Two reviewer suggestions clarify matched parameter states and region-splice artifacts. No runnable experiment driver or default was changed.

#### Evidence
C47 and the independent RTGS-021 audit anchor all local outcome statements. Two public Claude CLI requests completed with success, no permission denials and reported Opus 5 answers plus auxiliary Haiku usage. Public primary sources support the forward-model, sampling and calibrated matching context.

#### Assumptions
The user requested discussion, not selection or execution. Private repository/capture context remains outside Claude's prompt. A source-mask intervention is additional information; a learned prior is declared separately from runtime field-only inputs.

#### Uncertainties
The causes of the failed quality gates remain unidentified. No target-gradient diagnostic, parameter refit, topology factorial, new baseline, tomography control or field-only geometry comparison was executed. The old D/E prerequisite remains failed.

#### Review Focus
Confirm that provenance is accurate, the final note does not repeat Claude's withdrawn causal claims, diagnostic comparisons isolate their declared variables as far as possible, and confirmation/data boundaries remain explicit.

#### Protected actions not taken
No new reconstruction run, experimental approval, production modification, private upload, commit or push.

#### Recommended Next Action
Complete the distinct documentation review. Record only current-turn hypotheses/proposals in the research trace, run the repository verification gate, and hand the recommendation to the user without launching it.

### Review (2026-09-09)

#### Verdict

Accepted

#### Self-reviewed

No

#### Correctness

Accepted for the documentation/recommendation scope of RTGS-022. This is not experiment
approval or validation of a proposed remedy.

Reviewed hashes:

- Continuation note: `655cc17f9e7fa995a36d2b9c4f12d32d19302eacf204db66b4f89118137cd6d9`
- Collaboration receipt: `8081cbc0a68f0dc8317086383b9a76f2b347ddf79d118fbfc08e3cd6701d3c7b`

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

#### Evidence Quality

Both saved prompts and answers match their receipt hashes; both recorded processes
exited successfully. The discussion is clearly distinguished from an independent audit
of private outcomes. Primary-source spot checks support the R2-Gaussian normalization,
known-pose COLMAP route, and volumetric rasterizer limitations.

C47, the RTGS-021 audit MD/JSON and its frozen protocol are byte-identical to HEAD.
`git diff --check`, `check_agent_workflow.py` and `check_ara.py` passed. The reviewer did
not rerun reconstruction or the full test suite.

#### Simplicity

One bounded diagnostic is recommended first. Later interventions are conditional,
without prescribing a broad implementation program or reviving retired methods.

#### Missing Cases

Execution details remain appropriately unapproved: exact state/view selection, gradient
normalization and near-zero tolerances, intervention budgets, stopping rules, and fresh
confirmation allocation require a prospective protocol. These omissions do not block
this discussion artifact.

#### Required Changes

None.

#### Optional Improvements

When converting the recommendation into a protocol, report gradient comparisons within
parameter groups using declared parameterizations and near-zero handling; avoid
interpreting raw norms across differently scaled groups as relative causal importance.

The Driver owns final verification, current-turn research recording and task closure.

### Discussion closeout (2026-09-09)

The accepted scope is a completed discussion and conditional recommendation. N238/N239
and O173-O175 record only this request; no historical observation was promoted and no
experiment was launched. The recommendation, public response receipts and independent
review are linked above. Final repository checks and their exact log are saved under
`docs/research/2026-09-09-field-continuation/verification.json` before handoff.
