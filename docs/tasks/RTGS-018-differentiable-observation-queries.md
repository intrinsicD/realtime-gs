# Current Task

## Title
Differentiable empty observation queries and portable verification fixtures

## Task ID
RTGS-018

## Role Assignment

- Driver: Codex-rtgs-engineering
- Reviewer: Codex-cross-repo-review
- Turn: none

## Mode
Stabilize

## Risk
Protected

## Maturity

- Target: CPU-contracted
- Reached: CPU-contracted

## Goal
Make reference and indexed empty-support observation queries return differentiable zeros.

## Motivation
The cross-repository observation seam must remain usable by coordinate-based optimization
when no components contribute. Detached zero accumulators can make backward fail.

## Success Criteria
Reproduce the failure before changing code; preserve zero coordinate derivatives for empty
fields, empty point batches, and indexed queries with no candidate tiles; retain ordinary value
and gradient parity and checkpoint behavior. Make legacy runtime-guard unit tests independent
of the host library installation using actual temporary ABI files and hashes; preserve negative
controls. Run focused tests, the full verification gate,
and the slow CPU tests. Obtain a distinct implementation review bound to the patch digest.

## Constraints
Preserve CPU-first imports, compositor equations, historical protocols and evidence.
Keep images, masks, datasets, credentials, and saved models out of external review payloads.

## Non-Goals
A new lifting method, formal result-bearing experiments, quality/performance claims, default
promotion, CUDA execution, or changes to historical results. This is a CPU contract repair.

## Selected Skills

- rtgs-core
- rtgs-task-workflow
- rtgs-review
- rtgs-docs-sync
- rtgs-verify

## Experiment Contract
None

## Current Evidence
Independent inspection and deterministic failure reproductions are in progress. A temporary
verification environment uses the repository-pinned PyTorch 2.9.0 CPU release.

## Minimal Plan
Reproduce the missing graph, retain a zero dependency on query coordinates where accumulation
has no graph, add boundary and backend parity tests, supply portable ABI test fixtures, verify,
and hand off for independent review.

## Status
Accepted

## Human Decisions
2026-09-07: User requested implementation, experiments, improvements, testing, and review of
StructSplat and realtime-gs with separate Codex/Claude agents. Claude network review requires
specific payload approval after automatic review rejected the initial launch; approval pending.

## Handoff Log

### Start (2026-09-07)
RTGS-017 is archived as coordination-superseded while preserving its provisional review history.
This task targets CPU-contracted engineering; no calibrated research branch or formal run opens.

### Verification scope (2026-09-07)
The initial full gate exposed host-specific ABI assertions, loopback socket sandbox restrictions,
and a Conda MKL/OpenMP import conflict. The driver is isolating the test ABI prerequisites with
temporary files and real hashes; historical benchmark guards, frozen inputs, and result bytes
remain unchanged. Verification uses MKL_THREADING_LAYER=GNU and the pinned temporary environment.

### Handoff (2026-09-07)

#### Objective
Retain zero coordinate derivatives for empty observation queries and make ABI unit fixtures
portable without modifying historical experiment enforcement.
#### Reviewed state
Source/test diff SHA-256: 7dca95b5b892ab99b76c8ac6bdea37c77e29fa4c9bb2bbee05a646a5e8c0b203.
Scope: observation2d.py and the five changed tests shown by git diff; base main 31f0aca.
#### Changes
Attach zero coordinate dependency only after an otherwise detached accumulation. Replace
host ABI file assumptions in unit tests with temporary bytes and real hashes; test tampering.
#### Evidence
New empty-query tests fail before the fix (64 failing variants); 183 focused observation tests
pass afterward. Six existing ABI tests failed on the absent host file; the repaired subset and
two new guard cases pass (8 tests). Full verification is running with the pinned CPU environment.
#### Assumptions
CPU support-index semantics are unchanged. Temporary ABI bytes are unit fixtures, never an
actual loaded library or authorization to execute historical protocols.
#### Uncertainties
Full gate and slow tests are pending. CUDA is unavailable in this session.
#### Review Focus
Independent zero-gradient, mixed-output, checkpoint and inference probes; ensure ABI file/hash
and sole-preload negative controls still detect drift. Inspect ARA C44 scope and task archive.
#### Protected actions not taken
No formal experiment, held-out data access, reconstruction run, old evidence mutation, commit,
push or external upload. Claude code review awaits explicit payload approval.
#### Recommended Next Action
Review the frozen source/tests independently, reconcile final gate receipts, and close the
CPU contract record at its tested maturity.

### Review (2026-09-07)

#### Verdict
Accepted
#### Self-reviewed
No
#### Correctness
Codex-cross-repo-review found no blocker in the bound six-file diff. Empty accumulations retain
zero coordinate gradients; connected graphs retain ordinary behavior. Temporary ABI unit
fixtures preserve actual path/hash, preload ordering, missing-file and changed-file checks.
#### Evidence Quality
The reviewer independently reproduced 183 observation/CSR/checkpoint tests and eight ABI tests,
and checked noncontiguous/extreme-coordinate empty-query counterexamples. Root's final complete
verify.sh passed with torch 2.9.0+cpu and Ruff 0.15.20. The slow-only command selected no tests;
explicit collection confirmed zero slow tests and 2,126 deselected, so the normal gate exercised
all selected CPU tests. Claims are bounded to ARA C44 and its cited deterministic fixtures.
#### Simplicity
One conditional post-accumulation helper and local test fixtures; no new production dependency.
#### Missing Cases
CUDA execution and calibrated reconstruction effects are outside this CPU-contract scope.
#### Required Changes
None. The reviewer-required full gate and slow-suite inventory are complete.
#### Optional Improvements
A separate task may validate component_chunk types consistently across query backends.

### Final verification (2026-09-07)
Final code/test diff remained 7dca95b5b892ab99b76c8ac6bdea37c77e29fa4c9bb2bbee05a646a5e8c0b203.
Full gate receipt: /tmp/structsplat-rtgs-collab/rtgs-verify-final.log; slow selection receipt:
/tmp/structsplat-rtgs-collab/rtgs-slow-collection.log. Independent nine-file hash/command receipt:
/tmp/structsplat-rtgs-collab/independent_review_receipt.json. These are local execution receipts;
tracked tests and this exact review boundary are the durable evidence. No commit or push made.
Claude network review remains unperformed pending the specifically requested code-payload approval.

### Claude static review (2026-09-07)

The user subsequently approved sending the prepared code-only payload to Claude/Anthropic.
Claude (`claude-fable-5-1`, tools disabled) completed an independent static review and reported:
"No blocking findings within this patch scope." Claude executed no tests. The successful patch
packet SHA-256 is 66de54dc3b22162caccb5664e484d0f1082d41c063eb603c6522a464cdbc1527;
the verbatim review SHA-256 is
6aecbb0b1a517f94edceaf7758ca6ec381cf729f0d3970733df3500655b81ad7.
Local review and invocation receipts are in /tmp/structsplat-rtgs-collab/claude-review.md
and claude-review-receipt.json. An earlier full-packet attempt returned unusable tool-call text
with tools disabled and is not counted as a review. No datasets, images, masks, credentials,
saved models, or experimental artifacts were included.

Claude confirmed the zero-gradient attachment and shape/broadcasting behavior. The driver
resolved its nonblocking surrounding-context questions against the complete source: each
affected runtime guard checks preload existence/hash before frozen-runtime comparison, with
missing paths short-circuited safely; hashlib and _GroupedObservationIndexReference imports
already exist. No realtime-gs source or tests changed after the accepted full verification.
Claude's separate StructSplat variance-endpoint fixture suggestion was applied and independently
reviewed in CORE-020. The exact RTGS-018 review boundary and CPU validation above remain current.
