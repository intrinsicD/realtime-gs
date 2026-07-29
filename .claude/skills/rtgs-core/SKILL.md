---
name: rtgs-core
description: Orient and route any substantial work in the realtime-gs repository. Use at the start of a session, before touching source, tests, scripts, experiments, docs, tasks, skills, or CI, when deciding where work belongs, or whenever repository policy, verification, evidence maturity, or the agentic workflow is relevant.
---

# realtime-gs core router

Treat `CLAUDE.md` as the canonical contract. This skill is a compact entry point and routing aid,
not a second policy source. If any text here disagrees with `CLAUDE.md`, follow `CLAUDE.md`.

## Session start

1. Read `CLAUDE.md` completely.
2. Inspect `git status --short --branch`; preserve unrelated or user-owned changes.
3. Read `.agents/state/current-task.md`.
4. If a populated task matches the request, continue it and respect its `Turn`. If it does not
   match, do not overwrite it; hand it off, close it, or ask for direction when superseding it
   would change scope.
5. Read only the specialist skills and repository docs required by the touched scope.
6. Inspect the implementation, tests, decisions, protocols, and evidence before editing.

Substantial work changes behavior, an interface, dependency, default, policy, durable state, or
claim/result artifact. Use `rtgs-task-workflow` for it. Pure formatting and typo fixes are exempt.

## Hard boundaries

- Keep imports CPU-first. CUDA, `gsplat`, `transformers`, and other optional/GPU dependencies stay
  lazy and guarded.
- Add backends through the declared rasterizer/depth interfaces; do not fork pipeline logic.
- Keep tests deterministic and quality thresholds as floors.
- Treat CPU, CUDA, calibrated-pipeline, and claim-ready evidence as different maturity classes.
- Do not make result/default/capability claims without the experiment, audit, and ARA gates.
- Preserve append-only result artifacts and the one canonical run root per experiment task.
- Keep task-specific drivers in `scripts/experiments/`; durable repository tooling belongs at the
  top of `scripts/` only when the script-layout checker names it.

## Routing

| Work | Skill |
|---|---|
| Open, continue, hand off, review, or close substantial work | `rtgs-task-workflow` |
| Generate novel research directions or cross-domain transfers | `rtgs-research-ideation` |
| Design or run a result-bearing experiment | `rtgs-experiment` |
| Run or update tracked performance benchmarks | `rtgs-bench` |
| Audit outcomes, claims, defaults, or confirmatory evidence | `realtime-gs-results-audit` |
| Review a non-trivial diff | `rtgs-review` |
| Reconcile documented and implemented structure | `rtgs-docs-sync` |
| Run the complete local/CI gate | `rtgs-verify` |

A common research flow is:

`rtgs-core` → `rtgs-task-workflow` → `rtgs-research-ideation` (when discovery is needed) →
`rtgs-experiment`/`rtgs-bench` → `realtime-gs-results-audit` → `rtgs-review` →
`rtgs-docs-sync` → `rtgs-verify`.

## Evidence maturity

Use the task record's vocabulary precisely:

- `Scaffolded`: seam exists, behavior unproven.
- `CPU-contracted`: deterministic reference tests exercise the contract.
- `Pipeline-integrated`: canonical CLI/pipeline path exercises it.
- `Calibrated`: frozen local data, artifacts, metrics, and viewer receipt pass.
- `Claim-ready`: prospective protocol review, results audit, and ARA proof pass.
- `Retired`: replaced path and stale registrations/docs are removed.

Never report a higher level from evidence that establishes only a lower one.

## Gate

Before reporting completion, run the applicable focused checks, review the whole diff, then run:

```bash
./scripts/verify.sh
```

CI must call the same script; `scripts/check_agent_workflow.py` enforces that parity.
