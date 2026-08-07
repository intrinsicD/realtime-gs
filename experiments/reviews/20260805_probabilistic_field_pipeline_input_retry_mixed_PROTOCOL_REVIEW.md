# Prospective Protocol Review

- Task ID: `20260805_probabilistic_field_pipeline_input_retry_mixed`
- Protocol SHA-256: `20a71565f43d9b6879ec8382ef00c0cbe27846669814a0f1509ccb13a3ba89d1`
- Source-tree SHA-256: `2d997a21d4c923ede25555994bafa9cec01bb7c741edd0e6da93fadf37bcbe71`
- Reviewer: `Codex-probabilistic-field-protocol-reviewer`
- Verdict: `approved`
- Outcome Access: `none`

## Scope

This was an outcome-blind prospective review of the new input-contract retry. I reviewed the exact
task/source binding, sealed compact metadata, prior failure/progress receipts, all task and driver
diffs against the failed retry, the 549-cell plan, fail-closed input checks, run-root isolation,
and verification evidence. I read the latest input-retry handoff fully before checking the frozen
bytes independently.

If executed successfully, this task retains the prior scientific scope: deterministic synthetic
mechanism evidence and development-only operability/utility observations for native-control and
all-candidate lifting over a deterministic 512-component-per-view proxy of the eleven sealed
fields. It cannot establish complete-field fidelity, source-RGB reconstruction quality, true
globally coupled multi-marginal OT, GPS-Gaussian reproduction, GPU or real-time performance,
cross-scene generality, a production default, or reconstruction accuracy from independent-half
agreement.

## Checks

- Recomputed the protocol digest with
  `.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260805_probabilistic_field_pipeline_input_retry_mixed.json`.
  It exactly matched
  `20a71565f43d9b6879ec8382ef00c0cbe27846669814a0f1509ccb13a3ba89d1`.
- Recomputed the length-prefixed binding over `scripts/experiment_contract.py`, the input-retry
  driver, and every `src/rtgs/**/*.py` file. All 102 bound files produced
  `2d997a21d4c923ede25555994bafa9cec01bb7c741edd0e6da93fadf37bcbe71`, exactly matching the
  task.
- Restricted prior-run inspection to the failed retry's root `failure.json`,
  `run_receipt.json`, and discarded-warmup `failure.json`. I did not open
  `synthetic_results.json`, any prior metric/result payload, any scratch output, or a report/model
  artifact. The root receipt records `measured_cell_count=0`; the failing command is the discarded
  `stage_00008_default`, seed `80500`, all-candidate warmup. Reaching that command in the frozen
  coordinator establishes only progress past the synthetic subprocess, not permission to inspect
  or reuse its payload.
- Reconstructed the exact failure chronology from those receipts. The failed retry began at
  `2026-08-05T11:14:41.437603+00:00`; its warmup raised at
  `2026-08-05T11:17:00.351649+00:00` during `field_fit` with
  `observation and fiber AA dilation must agree`; the root orchestration failure and final failed
  receipt followed at `2026-08-05T11:17:00.544509+00:00` and
  `2026-08-05T11:17:00.545354+00:00`. The actual first-warmup exception was the AA-convention
  mismatch. The per-dataset byte-cap issue was separately identified from sealed metadata and
  would have blocked later full-resolution bundles; it was not the exception thrown by that
  168000-byte-contract warmup.
- Diffed the complete task files. Beyond the new task/driver/run identity, draft/pending lifecycle,
  input-retry title and provenance boundary, and resulting source binding, the only new protocol
  fields are `calibrated_followup.compact_view_byte_caps` and
  `calibrated_followup.projection_dilation=0.0`. After removing exactly those fields and the retry
  identity/provenance/path envelope, the old and new scientific task bodies have the same
  canonical SHA-256 `fc0589dbfb1913340f9f66b327f173929b3269407db8ec334548153ed752dea4`.
  The question, hypothesis, datasets, roles, train/held-out splits, seeds, comparators, metrics,
  invariant and utility gates, stopping rules, resource protocol, and all other frozen
  configuration are unchanged.
- Diffed the complete drivers. The only changes are the new `TASK_ID` and the bounded input-
  contract implementation: exact cap/dilation validation, two calibrated plan factors, explicit
  calibrated `FieldLiftConfig.projection_dilation`, per-dataset `CompactDataset.load(byte_cap=...)`,
  a loaded-observation AA equality gate, and recording of cap/configured/observed values in each
  successful input receipt. No synthetic generator, treatment, control, decision, metric, hard
  invariant, aggregation, resource, report, or viewer logic changed.
- Independently read only the integrity-bound metadata from every compact bundle. All 296 sealed
  observation archives declare `aa_dilation=0.0`. Each dataset is internally uniform and exactly
  matches the task map: six datasets declare a 168000-byte cap and five full-resolution datasets
  declare an 8388608-byte cap. The map covers all and only the eleven frozen dataset IDs, and every
  bundle's manifest byte count is at or below its declared cap. These are producer-declared loader
  safety contracts; they do not truncate teachers, select components, or form a performance arm.
- Confirmed fail-closed chronology in the new worker. Task compilation rejects missing/extra cap
  IDs, non-positive/non-integer caps, and negative or non-finite dilation. `CompactDataset.load`
  receives the selected task cap and independently requires it to equal each bundle's embedded
  cap. Immediately after load, the worker forms the set of realized observation AA values and
  raises unless it is exactly `[projection_dilation]`; this occurs before split materialization,
  `SceneFits`, configuration construction, or any field fit. Both calibrated arms receive the
  same frozen zero dilation. Successful input receipts retain the declared cap, realized AA set,
  and configured dilation.
- Exercised outcome-free negative task variants in memory. Missing dataset cap coverage, a zero
  cap, negative dilation, and non-finite dilation were all rejected. Both calibrated arm configs
  resolved to `projection_dilation=0.0`.
- Ran contract and data validation; both returned `OK`. The unchanged data seal has SHA-256
  `20e719d89628375c515db94102abf6e5018dbd6d686d0633235407fad5c7deb6` and contains eleven
  datasets and 309 files totaling 204,306,829 bytes: 296 `.rtgsv` bundles, eleven compact
  manifests, and two calibration files, with no image file.
- Compared the old and new outcome-free plans by `cell_id`. All 549 identifiers and every
  stage/arm/seed/source/prerequisite are unchanged. All 483 synthetic cells are identical. The 66
  calibrated cells preserve every prior factor and add only `compact_view_byte_cap` and
  `projection_dilation`. Stage counts remain 324 exact-shape, 60 association, 81 support-mask, 6
  topology, 6 schedule, 6 independent-half, and 66 calibrated; seeds remain
  `80501`/`80502`/`80503`, arms remain `native_controls`/`all_candidate_mechanisms`, and the
  component cap remains 512. The driver's canonical cell SHA-256 is exactly
  `8a3749c00b73f8a791d2cae6e91e37c81c2b75b87b6245cbca82acf339000a00`.
- Confirmed the exact command and path guards name only the new task, new driver, and one canonical
  `runs/20260805_probabilistic_field_pipeline_input_retry_mixed` root. The task remains `draft`
  with a pending review, and both that run root and this canonical review path were absent before
  the review was written.
- Ran
  `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_fiber_correspondence.py tests/test_probabilistic_field_experiment_protocol.py tests/test_probabilistic_field_pipeline.py tests/test_experiment_contract.py tests/test_field_refit.py`;
  all 72 focused outcome-free tests passed. The task/workflow checker also passed.
- Ran `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh`. Ruff, format, the complete non-slow CPU suite,
  docs sync, ARA, script layout, agent workflow, and experiment-contract checks all passed. Only
  the two already documented PyTorch warnings were emitted.
- Recomputed the task, source, and cell digests after verification and reconfirmed that the new run
  root and canonical review path remained absent before this artifact was written.

## Findings

The exact input-retry protocol and source binding above are **approved** for execution. The two
new values are exact, sealed-input contracts rather than tuned scientific treatments: loader caps
mirror the producer declarations without changing data or the 512-component fitting proxy, and
zero projection dilation matches every sealed observation with a pre-fit fail-closed equality
check. The new plan exposes both values per calibrated cell and successful input receipts preserve
their realized evidence. No hypothesis, treatment, control, threshold, seed, arm, split, metric,
gate, or resource rule drifted. No concrete protocol blocker remains.

The lifecycle chronology is: the original task failed before cell 1 on an import-wrapper defect;
the first retry repaired that defect, completed its synthetic phase, and then failed in the one
discarded calibrated warmup on the AA mismatch before any measured calibrated cell; sealed-
metadata inspection also identified the latent per-dataset byte-cap contract; the failed roots
remain immutable; this new task freezes both exact input conventions under a new vacant identity;
and this outcome-blind review approves only that frozen retry. Only after the owner records this
exact approval and transitions the task to `ready` may initialization and execution begin.

Any further digest-bearing task edit or bound-source change invalidates this approval. Approval
establishes fitness to execute; it does not establish successful execution, favorable metrics,
method superiority, or any scientific outcome.

## Protected Actions Not Taken

I did not initialize or execute the input retry, create its run root or lock, invoke an official
synthetic cell or calibrated worker, access prior synthetic/metric outcomes, inspect scratch
outputs, render a report, create or open an index page, launch an orbit viewer, update the
task/index/current-task state, or interpret a quantitative result. Outcome Access remained `none`
throughout.
