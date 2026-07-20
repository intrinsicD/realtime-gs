---
name: realtime-gs-results-audit
description: Adversarial referee pass ("scientist pass") over realtime-gs claims, experiments, benchmarks, and causal or confirmatory evidence. Use before quantitative or capability claims enter README.md or docs/, after an experiment or benchmark session, before changing a default or closing a roadmap question, when reviewing a results-bearing PR or commit, or whenever preregistered, held-out, real-data, or GPU numbers lack an independent audit. Do not use to propose or run a new experiment; use the experiment or bench skill for that.
license: MIT
metadata:
  version: "1.0.0"
---

# Results Audit ("scientist pass")

> **Provenance.** Distilled 2026-07-15 from this repository's own audit
> history: the depth-covariance screen that falsified its expected winner and
> rejected a final-step density evaluation; the confidence-anchor post-run
> audit that exposed two attribution confounds and incomplete source binding;
> matcher, local-plane, TUM-transfer, and signed-occlusion gates that withheld
> downstream utility claims; and GPU runs whose timings were non-decisional
> under contention.

## Stance

Act as a referee, not the producing session. Hunt for errors in protocol
chronology, data isolation, code paths, controls, accounting, and wording.
Reviewer and repair sessions are not exempt. Deliver corrections, narrowed or
retired claims, and truthful append-only dispositions, never reassurance.
Report "everything checks out" only with the claim table filled and every row
bound to independently checked evidence.

## When to run

- Before claims enter `README.md`, `docs/BENCHMARKS.md`, `docs/RESEARCH.md`,
  `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, or `ara/logic/claims.md`.
- After an official artifact or result note lands in `benchmarks/results/`.
- Before changing a default, promoting a research backend, opening a
  confirmatory phase, or declaring a roadmap question resolved.
- On request: "audit", "referee pass", "scientist pass", "verify the claims".
- Periodically sample old positive results and rejected protocols.

## Procedure

### 1. Inventory the claims

Sweep the relevant docs, `ara/PAPER.md`, `ara/logic/claims.md`,
`docs/EXPERIMENTS.md`, `ara/evidence/`, and matching JSON, `*_PREREG.md`,
`*_RESULT.md`, `*_AUDIT.md`, seal, threshold, acquisition, and decision
artifacts. Build:

| # | Claim | Kind + scope | Evidence path | Protocol/source bound? | Executed where/when? |
|---|---|---|---|---|---|

Kind is proven, measured, or asserted; scope includes synthetic/real,
development/confirmatory, initialization/refinement, and CPU/GPU. An asserted
quantitative claim, a result without a dated experiment/ARA row, or a paper idea
described as this repository's result is already a finding.

### 2. Audit chronology, isolation, and preregistration

Verify hypotheses, arms, seeds, train/held-out split, metrics, minimum effects,
falsification rules, and stopping gates were frozen before outcome access.
Accept later amendments only when timestamps and access audit show the affected
outcomes were still unseen.

Initialization, fitting, matching, depth alignment, thresholds, checkpoint
choice, and default selection consume training data only. Held-out RGB/depth is
reporting-only. Synthetic GT may audit a frozen graph after construction only
when its API cannot receive GT. For development/confirmatory work, verify archive
SHA-256, acquisition record, payload allowlists, disjoint T/V/H roles, protocol
and implementation hashes, and atomic once-only seals. A consumed or failed
confirmatory attempt stays consumed; unopened data remains unopened.

### 3. Bind evidence to exact source and execution

Every numeric claim needs a machine-readable artifact, human result note, and
exact command. Check revision, dirty state, tracked-diff/source-tree hashes,
loaded source hashes, effective config, environment, and input/tensor hashes.
A dirty run is usable only when its exact executed source is preserved. The
general `benchmarks/run.py` history is not automatically a replay-complete causal
package. Never overwrite an official result or reconstruct numbers from memory.

### 4. Re-execute independent checks

For replayable harnesses copy the official command from the matching result note
to a fresh scratch output; do not replay a one-shot sealed phase. Recompute gates,
percentage directions, paired wins, intervals, and effect sizes directly from raw
JSON. Run the strongest applicable repository checks:

```bash
CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh
.venv/bin/pytest -q
git diff --check
```

For sealed RGB-D audit code, include its focused CPU tests. If the claim is the
tracked benchmark table, run `.venv/bin/python benchmarks/run.py --quick
--update-docs`; otherwise do not rewrite that generated block merely to audit an
unrelated experiment.

### 5. Audit invariants and controls

Check step-0 equality, shared fitted tensors, train/test indices, optimizer
schedules, primitive counts, RNG streams, target histories, and source hashes
across arms. Confirm shuffled/negative controls preserve the intended sampled
multiset, denominator, graph degree/exposure, code path, and budget. A local
mechanism improvement is not global materiality. A constructor/matcher that
fails its validity gate cannot proceed to optimization, and secondary metrics
cannot rescue a failed primary gate.

### 6. Audit configuration, metrics, and accounting

Verify device/import path, PyTorch/gsplat versions, rasterizer, fit backend,
lifter, covariance/anchor/supervision mode, scene/resolution/masks, fit/lift/
refine iterations, merge/density schedule, recovery steps, primitive counts/cap,
and SH degree. Reject silent backend fallbacks or shadowed installations.

Recompute full/foreground/crop PSNR, SSIM/LPIPS, alpha IoU/leakage, depth RMSE
and tails, geometry errors, counts, time, VRAM, ratios, and speedups from raw
values. Preserve paired seeds, frozen effect floors, win counts, and safety
gates. Density surgery on the final evaluation step is invalid without recovery;
train-selected hyperparameters never use held-out cameras.

### 7. Enforce performance and environment honesty

CPU verification, skipped CUDA tests, CUDA execution, and GPU timing are distinct
evidence classes. Green CPU CI does not prove gsplat parity, CUDA behavior,
real-time throughput, or GPU memory. Performance evidence requires a named idle
GPU, warmup, repeats, aggregation rule, software versions, and cited raw files.
A contended GPU run may support quality diagnostics but not timing or speedup.

### 8. Dispose of every claim in the same change

For each row: **confirm**, **narrow**, or **retire**. Update public prose,
`ara/logic/claims.md` and matching evidence, append-only
`docs/EXPERIMENTS.md`, the result/audit note, and `docs/ROADMAP.md` status as
applicable in the same commit. Preserve rejected/superseded artifacts. Regenerate
the marked `docs/BENCHMARKS.md` block only through `benchmarks/run.py
--update-docs`. Change a production default only after its preregistered
multi-seed material gate and required production-path interaction pass.

### 9. Report

End with the claim table, corrections, protocol/provenance findings, commands
actually executed, skipped CUDA/GPU work, still-unverified claims and why, and
the exact new evidence needed to promote each row.

## Anti-patterns (hard nos)

- Tuning gates, arms, or tolerances after seeing outcomes.
- Using held-out/sealed data for fitting, selection, repair, or stopping.
- Calling a dirty partial-source-hash artifact replay-complete.
- Overwriting an official JSON, seal, threshold, or decision artifact.
- Promoting a failed constructor into a utility run.
- Treating contended-GPU timing or green CPU CI as GPU-performance evidence.
- Lowering a test quality floor without a dated experiment justification.
- Reporting remembered numbers without reopening their raw source.

## Repository anchors

`docs/EXPERIMENTS.md` — append-only result ledger · `docs/RESEARCH_LOOP.md` —
three-iteration workflow · `benchmarks/results/` — protocols, JSON, audits,
seals, and decisions · `ara/logic/claims.md`, `ara/evidence/` — bounded claims
and proof · experiment-specific benchmark harnesses — fail-closed execution ·
`benchmarks/run.py`, `docs/BENCHMARKS.md` — tracked history ·
`tests/test_pipeline.py` — held-out leakage guard · `tests/test_tum_rgbd_*.py`
— sealed-protocol checks · `scripts/verify.sh` — CPU gate · `CLAUDE.md` —
backend, determinism, benchmark, and docs rules.
