# Current Task

## Title

Research high-quality field-only 3D reconstruction and the tomography connection

## Task ID

RTGS-020

## Role Assignment

- Driver: Codex-field-reconstruction-research
- Reviewer: Codex-field-reconstruction-research
- Turn: none

## Mode

Explore

## Risk

Protected

## Maturity

- Target: Not applicable
- Reached: Not applicable

## Goal

Produce a source-backed, code-grounded research proposal with Claude's public-literature critique and a discriminating next experiment for high-quality reconstruction from 2D Gaussian fields.

## Motivation

The user asks whether medical tomography offers a route from 2D densities to a high-quality 3D Gaussian reconstruction after the mixed BENCH-019 result and the earlier blurred tomography screen.

## Success Criteria

Explain the forward models and identifiability limits; distinguish existing code from proposed changes; inspect primary literature; preserve the two existing audits; provide a practical architecture, alternatives and a test that separates compression, initialization and capacity. Record the actual scope and outcome of Claude collaboration.

## Constraints

Research and documentation only. Preserve current code, defaults and completed experiment artifacts. No new image, mask, model or capture uploads. Private repository context is excluded from Claude's public research task following automatic review rejection. No commit or push is requested.

## Non-Goals

No new reconstruction execution, protocol approval, claimed quality gain, physical-density proof, or production integration.

## Selected Skills

- rtgs-core
- rtgs-task-workflow
- rtgs-research-ideation
- structsplat-core and structsplat-research-ideation for cross-repository orientation
- rtgs-verify
- research-manager at end of turn

## Experiment Contract

None

## Current Evidence

docs/RESEARCH_2026-09-07_FIELD_RECONSTRUCTION.md; existing C42–C46 and their cited audits. Code inspected at realtime-gs 8a71505 and structsplat 03ae7d4. Public primary literature is linked in the research note. Two public Claude research rounds completed; the report adjudicates disagreements and links both responses and the provenance receipt. No independent repository review is claimed.

## Minimal Plan

Inspect prior results and code; obtain public mathematical critique from Claude; verify primary sources; synthesize a bounded plan and alternatives; check documentation and record research provenance.

## Status

Superseded

## Human Decisions

2026-09-07: User explicitly requests coworking with Claude and research into high-quality 3DGS reconstruction from 2D Gaussians, suggesting medical tomography. User clarifies their expectation that tomography reconstructs 3D density from 2D densities. This authorizes research; it does not endorse any resulting hypothesis.

## Handoff Log

### Start (2026-09-07)

The previous active slot was the blank template. No earlier task is superseded. The existing public Claude connector has no agent types. A separate Claude CLI request uses a fresh temporary directory, safe mode, no session persistence and public WebSearch/WebFetch tools only. No repository content is included in its prompt.

### Research delivery (2026-09-07)

The requested research synthesis and two-round public Claude discussion are complete. Report SHA-256: ba6cf4bfb77ded3fa5d289c28dc9efa49dba26abc6a9fd17c897bbab9f94d00a. The report proposes a target-information comparison followed by field-derived geometry controls; these remain hypotheses. Claude withdrew unsupported matching and covariance claims; Codex rejects the follow-up's unsupported geometry-error bound. No source change, new reconstruction, result promotion, commit or push occurred. The initial full repository check found only two tests reporting the same missing navigation link to this new report; the link is repaired. Final verification is recorded in docs/research/2026-09-07-field-reconstruction/verification.json.

The generic research-manager epilogue identified this turn's research question and untested recommendation. An automatic approval review rejected a broader ledger update including unrelated older observations; no historical research ledger entry was changed. Current-request discussion and hypotheses are preserved in this report and task record.

### Handoff (2026-09-07)

#### Objective
Deliver the requested mathematical and implementation-grounded research proposal.
#### Reviewed state
Report digest and source commits are bound in Research delivery and Current Evidence above; Claude dialogue digests are in the collaboration receipt.
#### Changes
Current research report, navigation, public dialogue and this RTGS-020 task only. Historical ledgers and all executable source remain unchanged.
#### Evidence
Primary literature, existing C42–C46 audits, code inspection and two public Claude research rounds. Portfolio and navigation checks pass; final repository verification is recorded separately.
#### Assumptions
Calibrated static views; candidate reconstruction receives only fields and cameras after fitting. Reference-image initialization is a labelled diagnostic, not a field-only success.
#### Uncertainties
No new high-quality reconstruction result, generalization evidence or speed measurement. The suggested experiment is unexecuted.
#### Review Focus
Forward-model correctness, conditional covariance evidence, target parity, capacity and initialization confounds.
#### Protected actions not taken
No source change, new experiment, ready protocol, method/default promotion, historical ledger edit, private data upload, commit or push.
#### Recommended Next Action
Use the report to freeze a separate target-information experiment with prospective independent review.

### Review (2026-09-07)

#### Verdict
Accepted
#### Self-reviewed
Yes
#### Correctness
The report preserves the CT/RGB distinction and exact-target equivalence without promising unique geometry or success.
#### Evidence Quality
Claude contributed public reasoning, not an independent repository review. The final report corrects its remaining overclaims. Existing quantitative conclusions remain bound to their original audits.
#### Simplicity
Reuse existing field queries, compact trainer and density adapter; diagnose information before adding another lift.
#### Missing Cases
Execution, high-capacity quality, GPU parity, performance and multi-capture generalization remain future work.
#### Required Changes
None to the research synthesis. Final verification outcome must be recorded before handoff.
#### Optional Improvements
After the information screen, test field-derived patch geometry and covariance/coverage controls.

### Final verification (2026-09-07)

The complete ./scripts/verify.sh gate passed with exit code 0 after the navigation fix. The final log and hashed receipt are in docs/research/2026-09-07-field-reconstruction/verification.json. Portfolio structure, all 12 local report links and git diff --check pass. Research synthesis is delivered; the proposed reconstruction experiment remains unexecuted.

### Supersession (2026-09-08)

The user approved executing the proposed experiments. RTGS-021 continues this research with prospective experiments; the completed research synthesis and its self-review remain preserved above.
