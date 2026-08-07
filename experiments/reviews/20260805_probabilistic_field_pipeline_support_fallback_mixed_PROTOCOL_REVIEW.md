# Prospective Protocol Review

- Task ID: `20260805_probabilistic_field_pipeline_support_fallback_mixed`
- Protocol SHA-256: `f4e5b6bf7cd75d55d14d8cbdc019f2c996e7f6ccad7960f1143d1885b00d16e3`
- Source-tree SHA-256: `13db64f01e211791ff67fcccfcd3f7e923761a5c8cc707763dcf56259a36b663`
- Reviewer: `Codex-probabilistic-field-protocol-reviewer`
- Verdict: `approved`
- Outcome Access: `none`

## Scope

This was an outcome-blind prospective review of the exact empty-support fallback task. I reviewed
the frozen task/source binding, the unchanged scientific matrix, exact exception boundary,
whole-cell retry and independent-half behavior, Torch RNG reset, requested/effective-mask
provenance, successful and rejected terminal validation, timing inclusion, no-imputation
aggregation, report/orbit presentation, run-root isolation, and repository verification. I read
the final handoff and independently rebound every frozen digest before creating this artifact.

If executed successfully, this task remains development-only evidence over deterministic
512-component-per-view proxies of eleven sealed Gaussian2D fields. It may measure synthetic
mechanism behavior and calibrated operability, quality, convergence, resource use, and
presentation artifacts under the frozen controls. It cannot establish complete-field fidelity,
source-RGB reconstruction quality, spatial resolution, true globally coupled multi-marginal OT,
GPS-Gaussian reproduction, GPU or real-time performance, cross-scene generality,
production-default suitability, or accuracy from independent-half agreement. A fallback cell is
explicitly unmasked operability evidence and is ineligible as hard-versus-probability mask
evidence.

## Checks

- Recomputed the protocol digest with
  `.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260805_probabilistic_field_pipeline_support_fallback_mixed.json`.
  It exactly matched
  `f4e5b6bf7cd75d55d14d8cbdc019f2c996e7f6ccad7960f1143d1885b00d16e3`.
- Recomputed the length-prefixed binding over `scripts/experiment_contract.py`, this task's
  driver, and every `src/rtgs/**/*.py` file. All 102 bound files produced
  `13db64f01e211791ff67fcccfcd3f7e923761a5c8cc707763dcf56259a36b663`, exactly matching the
  frozen source binding.
- Ran repository contract validation and task data validation; both returned `OK`. The unchanged
  data seal has SHA-256
  `20e719d89628375c515db94102abf6e5018dbd6d686d0633235407fad5c7deb6`, covers eleven
  datasets and 309 files totaling 204,306,829 bytes, and contains 296 compact `.rtgsv` files,
  thirteen JSON files, and no image path. An independent filesystem discovery found exactly the
  same eleven `gaussians2d*/manifest.json` field sets named by the task.
- Restricted predecessor inspection to the already failed completion run's permitted failure and
  run receipts. The root failure SHA-256 is
  `06e07156e4c2a937fa5d1dbd402c1d9b04b625877d1ce23f9188328ebf474104`, the run-receipt
  SHA-256 is `506db34780753d91145550713273d3402be023ab02e83f2d04b5bcb4104395b0`, and the terminal
  empty-support failure SHA-256 is
  `0718cf26f83df954302063afb597f978bdf4606bd50a3a5531853fa9222106ae`. I did not open any
  predecessor successful summary, metric payload, model, report, or synthetic result.
- Recomputed the expanded plan and canonical payload. The 549-cell list SHA-256 is
  `1af09dabc5de411ff09acdef30efa5da17e010f6fcad2b450b0dc08d31f005fc`; the plan-payload
  SHA-256 is `5596baf2728b73bf283c4185a3cc00ec4ec2a036dcea37bc05851d149a807d91`.
  Stage counts remain 324 exact-shape, 60 association, 81 support-mask, 6 topology, 6 schedule,
  6 independent-half, and 66 calibrated cells.
- Compared every plan cell with the failed completion task by `cell_id`. All 483 synthetic cells
  are exactly equal. The 66 calibrated cells differ only by the seven declared fallback
  disclosures: policy, exact exception type/message, retry limit, RNG reset rule, interpretation,
  and the strengthened continuation-receipt contract hash. Datasets, roles, splits, seeds,
  stages, comparators, charts, data seal, synthetic generation, mask test, mechanism arms,
  pipeline, hard invariant gates, and decision rules remain equal. `depends_on` is correctly empty
  because a canonically failed predecessor cannot satisfy initialization dependency checks.
- Diffed the drivers structurally. Of 81 shared top-level definitions, 71 are unchanged. Added or
  changed definitions are confined to task-scoped fallback execution/provenance, calibrated
  receipt validation, outcome aggregation, and report/viewer presentation. No synthetic
  generator, scientific treatment, decision rule, or hard-invariant implementation changed.
- Exercised the retry boundary independently with reviewer-authored immutable fit fixtures. Exact
  base `ValueError: support-mask policy rejected every field-placement source` produced exactly
  `probability -> none`, reset Torch to the cell seed, and reran the primary plus both independent
  halves under `none`; all three results received identical provenance. A subclass with the exact
  message, a different `ValueError` message, and a different exception type each escaped after one
  call. A repeated exact error escaped after exactly two calls (`hard`, then `none`).
- Confirmed timing scope directly in the worker: field-fit timing starts before the requested
  attempt and ends only after fallback execution and provenance enforcement, so an eligible hard
  gate after retry retains both attempts in fit/process timing. A retry failure is not converted
  into an eligible continued terminal.
- Attacked successful provenance with coherent and one-surface mutations. Exact records pass;
  changed effective mode, retry count, trigger, interpretation, config, or cross-record value
  fails. JSON boolean/integer aliases, integer/float aliases, altered null/value states, float RNG
  seeds, numeric fallback metrics with the wrong JSON type, and typed diagnostic aliases all fail.
  The centralized record validator enforces exact keys and types before canonical JSON equality.
- Exercised the hard-gate failure path with a generated rejected model. Failure, input-boundary,
  resource, and requested/effective-config receipts carried one identical task-derived fallback
  record, and the structured loader accepted the exact terminal. Mutating the fallback or
  effective config on any individual surface, including a boolean retry-count alias, prevented
  continuation. A consistently disclosed non-fallback hard-gate terminal remained eligible.
- Exercised mixed accepted/rejected presentation with one accepted representative, one rejected
  unmasked-fallback seed, and one rejected mask-retained seed in the same arm. All three appear in
  the comparison manifest; rejected entries are seed-qualified and labeled `presentation-only`
  plus effective fallback status. Both rejected records appear in the presentation receipt,
  report notes serialize their exact provenance and rejection, and both seed-qualified PLY pairs
  appear in the report artifact list. An all-rejected two-arm counterexample selected six unique
  viewer entries without hiding a seed.
- Exercised an all-failure dataset summary with six calibrated failures, including one rejected
  fallback. The only curve values were zero calibrated-success indicators; all comparison cards
  were explicitly unavailable successful-cell counts; failure notes were presentation-only and
  retained fallback status; history contained only six zero success records; and an empty
  conditional metric returned JSON null with denominator zero. No failed quality, runtime,
  resource, convergence, or topology value was imputed.
- Confirmed that all 38 statically produced successful-cell metric IDs, including resource/stage
  additions and fallback curves, have report metadata. Successful fallback fraction is computed
  over all 66 attempts exactly as frozen, while rejected fallback cells remain separately visible
  in failure/report/presentation records and excluded from mask-mode interpretation.
- Ran
  `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_fiber_correspondence.py tests/test_probabilistic_field_experiment_protocol.py tests/test_probabilistic_field_pipeline.py tests/test_experiment_contract.py tests/test_field_refit.py`;
  all 89 focused tests passed. The seven support-fallback tests passed independently as well.
- Ran `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh`. Ruff, format, the complete non-slow CPU test
  suite, docs sync, ARA, script layout, agent workflow, and experiment-contract checks all passed.
  Only the two documented pre-existing PyTorch warnings were emitted.
- Recomputed protocol, source, data, cell-list, and plan-payload bindings after verification.
  Immediately before this artifact was written, the task remained `draft` with a pending review,
  `depends_on` remained empty, and both this canonical review path and
  `runs/20260805_probabilistic_field_pipeline_support_fallback_mixed` were absent.

## Findings

The exact support-fallback protocol and source binding above are **approved** for execution. The
new behavior is narrowly fail-closed: only one exact base exception can trigger one whole-cell
retry; the retry is visibly unmasked; the primary and independent halves cannot mix mask modes;
the failed attempt remains timed; all scientific hard gates still apply; and any second or
different failure aborts. Successful and rejected outputs carry type-strict requested/effective
provenance, rejected models remain presentation-only, and fallback cells cannot be used as masked
evidence. No scientific treatment, comparator, threshold, gate, seed, split, input, or decision
rule changed.

Five prospective defects were found and repaired in superseded freezes before this approval: a
rejected fallback could lose provenance after a hard gate; a failed predecessor was listed as an
unreachable executable dependency; three successful diagnostic fields were not validated; a
successful representative could hide a rejected fallback from report/viewer presentation; and
Python bool/int equality could bypass the promised JSON-exact record check. The final frozen bytes
pass direct counterexamples for every defect. No concrete protocol blocker remains.

Only after the owner records this exact approval and transitions the task to `ready` may the
canonical run be initialized and executed. Any digest-bearing task edit or bound-source change
invalidates this approval. Approval establishes fitness to execute; it does not establish
successful completion, favorable metrics, method superiority, browser/report usability, orbit
viewer availability, or any scientific outcome.

## Protected Actions Not Taken

I did not transition the task to `ready`, initialize or execute the official support-fallback run,
create its run root or lock, invoke an official synthetic or calibrated worker, inspect any
successful predecessor outcome, attach or read an official result/audit payload, render or open an
official index page, launch an orbit viewer, change a library default, make a scientific claim,
commit, push, or publish. Temporary counterexamples contained only reviewer-authored fixtures and
were deleted after validation. Outcome Access remained `none` throughout.
