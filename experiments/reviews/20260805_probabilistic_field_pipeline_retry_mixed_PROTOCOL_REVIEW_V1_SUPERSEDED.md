# Prospective Protocol Review

- Task ID: `20260805_probabilistic_field_pipeline_retry_mixed`
- Protocol SHA-256: `91e2c662d079742e99b7336d2f5a5de7ed4a55e733653e5537a6090570b9ce0d`
- Source-tree SHA-256: `e1c442a54585aa67fbfa1e57c6bbb73c77eb5a61c95889a38a33f8941383651a`
- Reviewer: `Codex-probabilistic-field-protocol-reviewer`
- Verdict: `approved`
- Outcome Access: `none`

## Scope

This was an outcome-blind prospective review of the exact infrastructure-retry task, source
binding, predecessor failure provenance, eleven sealed Gaussian2D datasets, 549-cell plan,
scientific controls, execution chronology, run-root isolation, and import-guard repair. I read the
latest retry handoff, all four preserved predecessor rejection reviews, and the predecessor's
approved v5 review before checking the retry independently.

If executed successfully, this retry has the same evidence scope as the approved predecessor: it
may provide deterministic synthetic mechanism evidence and development-only operability/utility
observations for native-control and all-candidate lifting over a deterministic
512-component-per-view proxy of the eleven sealed fields. It cannot establish complete-field
fidelity, source-RGB reconstruction quality, true globally coupled multi-marginal OT,
GPS-Gaussian reproduction, GPU or real-time performance, cross-scene generality, a production
default, or reconstruction accuracy from independent-half agreement.

## Checks

- Recomputed the retry review digest with
  `.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260805_probabilistic_field_pipeline_retry_mixed.json`.
  It exactly matched
  `91e2c662d079742e99b7336d2f5a5de7ed4a55e733653e5537a6090570b9ce0d`.
- Recomputed the length-prefixed binding over `scripts/experiment_contract.py`, the retry driver,
  and every `src/rtgs/**/*.py` file. All 102 bound files produced
  `e1c442a54585aa67fbfa1e57c6bbb73c77eb5a61c95889a38a33f8941383651a`, exactly matching the
  retry task.
- Restricted predecessor-run inspection to four authorized infrastructure receipts:
  `task.lock.json`, root `failure.json`, `synthetic/failure.json`, and `run_receipt.json`. The
  lock binds the approved predecessor protocol
  `ae3b9cbb11157480f33c1520244582409c0f09cd7cad7f6637cde161e8811044`, its approved review
  artifact, the same data seal, and the predecessor command. No model, metric, synthetic result,
  calibrated result, preview, report, or viewer artifact was inspected.
- Reconstructed the failure chronology from those receipts. The predecessor lock began at
  `2026-08-05T10:46:07.611324+00:00`. During `synthetic_initialization`, importing `torch` passed
  Python's valid `fromlist=None` into the live guard; the guard attempted to iterate `None` and
  raised `TypeError` at `2026-08-05T10:46:13.835044+00:00`. Its structured receipt records
  `completed_cells=0`. The parent coordinator then recorded an orchestration failure at
  `2026-08-05T10:46:13.842739+00:00` with `measured_cell_count=0`, followed by the failed run
  receipt at `2026-08-05T10:46:13.843520+00:00`. This is zero-cell infrastructure provenance,
  not retry outcome access.
- Diffed the complete predecessor and retry drivers. The only deltas are the new `TASK_ID` and
  replacing iteration over `fromlist` with iteration over `(fromlist or ())`; the additional
  changed lines are formatting of that same expression. No worker, cell, metric, gate, resource,
  aggregation, serialization, report, or viewer behavior changed.
- Diffed the complete task files. Retry-specific changes are confined to task identity and slug,
  draft/pending review state, dependency on the failed predecessor, an infrastructure-retry title
  and claim-boundary explanation, retry driver/task/run paths, and the resulting source binding.
  After removing exactly those identity/lifecycle/provenance/path fields, both scientific task
  bodies have the identical canonical SHA-256
  `e008f7051f0eeacd95a59492ada176a18cefb0ef4821053ed5f8183e96ca5246`.
- Confirmed both tasks reference the exact same data-seal file, whose SHA-256 is
  `20e719d89628375c515db94102abf6e5018dbd6d686d0633235407fad5c7deb6`. Contract and data
  validation returned `OK`. The unchanged seal contains eleven datasets and 309 files totaling
  204,306,829 bytes: 296 compact `.rtgsv` files, eleven compact manifests, and two calibration
  files, with no image file. The dataset roles, manifest paths, train/held-out partitions, three
  seeds, two calibrated arms, comparators, invariant gates, stopping rules, input guards, and
  CPU resource protocol are unchanged.
- Inspected the outcome-free retry plan. It contains 549 unique cells with the same frozen stage
  counts: 324 exact-shape, 60 association, 81 support-mask, 6 topology, 6 schedule, 6
  independent-half, and 66 calibrated cells. Schedule cells retain five cleanup iterations; the
  calibrated matrix remains eleven datasets by seeds `80501`, `80502`, and `80503` by
  `native_controls` and `all_candidate_mechanisms`. Canonical serialization of the complete old
  and retry cell arrays produced the same SHA-256
  `1ca991e48c5bf43c0ef32b8e83af5d9a693434992e990020f382d2b86753eb89`.
- Confirmed retry path isolation. The task command, driver constants, and runtime path resolver all
  name only `experiments/tasks/20260805_probabilistic_field_pipeline_retry_mixed.json`,
  `scripts/experiments/20260805_probabilistic_field_pipeline_retry_mixed.py`, and
  `runs/20260805_probabilistic_field_pipeline_retry_mixed`. The retry task remains `draft` with a
  pending review, and both its canonical review path and run root were absent before this artifact
  was written.
- Independently loaded the retry driver in a clean Python subprocess, entered `NoImageGuard`, and
  executed `__import__("torch", globals(), locals(), None, 0)`. It returned the `torch` module.
  The guard receipt reported `passed=true`, all three negative-control denials, zero unexpected
  denied imports or paths, and no forbidden module loaded. This exercises the exact predecessor
  failure surface without invoking an experiment cell.
- Ran
  `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_fiber_correspondence.py tests/test_probabilistic_field_experiment_protocol.py tests/test_probabilistic_field_pipeline.py tests/test_experiment_contract.py tests/test_field_refit.py`;
  all 71 focused outcome-free tests passed, including the clean-subprocess retry guard regression
  and unchanged 549-cell plan contract. The task/workflow checker also passed.
- Ran `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh`. Ruff, format, the complete non-slow CPU suite,
  docs sync, ARA, script layout, agent workflow, and experiment-contract checks all passed. Only
  the two already documented PyTorch warnings were emitted.
- Recomputed the protocol and source digests after verification and reconfirmed that the retry run
  root and canonical review path remained absent before writing this review.

## Findings

The exact retry protocol and source binding above are **approved** for execution. The predecessor
failed before cell 1 because its task-local import wrapper rejected a valid Python import call
shape. The retry fixes exactly that infrastructure defect, preserves the approved scientific
protocol and complete producer plan, and isolates all future artifacts under the new task/run
identity. The live clean-subprocess counterexample now passes without weakening the guard's
negative controls. No concrete retry blocker remains.

The lifecycle chronology is therefore: predecessor v5 prospective approval; predecessor
initialization; zero-cell infrastructure failure; creation of this new dependent draft task and
new vacant run root; this outcome-blind retry approval; then, only after the owner records this
exact approval and transitions the task to `ready`, guarded retry initialization and execution may
begin. Any change to a digest-bearing retry task field or bound source byte invalidates this
approval and requires another prospective review. Approval establishes fitness to retry; it does
not establish successful execution, favorable metrics, method superiority, or any scientific
outcome.

## Protected Actions Not Taken

I did not invoke retry `init-run`, execute the retry coordinator, run an official-seed synthetic
cell or calibrated worker, access or create any retry outcome, inspect any predecessor artifact
beyond the four named infrastructure receipts, render a report, create or open an index page,
launch an orbit viewer, update the task/index/current-task state, or interpret a quantitative
result. Retry Outcome Access remained `none` throughout.
