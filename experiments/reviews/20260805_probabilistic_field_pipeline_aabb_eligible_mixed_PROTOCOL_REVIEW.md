# Prospective Protocol Review

- Task ID: `20260805_probabilistic_field_pipeline_aabb_eligible_mixed`
- Protocol SHA-256: `7237f73f45b3515b8496b6a629a1e728414f634bee61b7b74391dbbfb189db06`
- Source-tree SHA-256: `0cca172519fe6840f2051b3683dac8336c01caecbb0d3dbd24e2e04d0b1b4c47`
- Pinned base-driver SHA-256:
  `9d453b967b09005b63d3bef6aac48b817ac841ea2b7faa3593d4f980e4310169`
- Reviewer: `Codex-probabilistic-field-protocol-reviewer`
- Verdict: `approved`
- Outcome Access: `none`

## Scope

This was the second distinct, outcome-blind prospective review of the forward-AABB-eligible
successor. The first freeze was rejected before execution because its administrative review
blocker made the reviewed digest impossible to transition to `ready`, and because its eligibility
receipt did not bind per-view eligible/rejected counts to actual capped candidate capacities. I
reviewed the repaired lifecycle, exact task/source/base bindings, unchanged scientific plan,
train-only bounds-before-draw implementation, deterministic capacity-balanced selection,
insufficient-capacity failure, native/candidate preprocessing parity, diagnostic validation,
subprocess identity, and immutable predecessor chronology.

If executed successfully, this remains development-only evidence over deterministic
512-component-per-view proxies of eleven sealed Gaussian2D fields. It may measure the frozen
synthetic mechanisms and calibrated compact-field operability, quality, convergence, resource,
and presentation behavior. It cannot establish complete-field fidelity, source-RGB
reconstruction quality, spatial resolution, true globally coupled multi-marginal OT,
GPS-Gaussian reproduction, GPU or real-time performance, cross-scene generality,
production-default suitability, or accuracy from independent-half agreement.

## Checks

- Recomputed the protocol digest independently and with `scripts/experiment_contract.py`; both
  produced `7237f73f45b3515b8496b6a629a1e728414f634bee61b7b74391dbbfb189db06`.
  The task is `draft`, its review fields are pending, `depends_on` and `blockers` are empty, and
  the successor run root was absent.
- Recomputed the length-prefixed binding over `scripts/experiment_contract.py`, the successor
  wrapper, and every `src/rtgs/**/*.py` file. All 102 files produced
  `0cca172519fe6840f2051b3683dac8336c01caecbb0d3dbd24e2e04d0b1b4c47`.
- Recomputed the pinned support-fallback base-driver byte digest as
  `9d453b967b09005b63d3bef6aac48b817ac841ea2b7faa3593d4f980e4310169`.
  The wrapper checks this digest before import, and its task contract requires the exact base path,
  algorithm, and hash.
- Recomputed the data-seal digest as
  `20e719d89628375c515db94102abf6e5018dbd6d686d0633235407fad5c7deb6`.
  Repository contract validation and task data validation both returned `OK`.
- Recomputed the 549-cell list digest as
  `1af09dabc5de411ff09acdef30efa5da17e010f6fcad2b450b0dc08d31f005fc`
  and the full plan-payload digest as
  `205eb30228c3c2ac4d976fac9f7d91263703e35b379cface83555a59a5b3a75b`.
  All 549 cell dictionaries are exactly equal to the approved pinned base plan: 324 exact-shape,
  60 association, 81 mask, six topology, six schedule, six independent-half, and 66 calibrated
  cells.
- Re-ran the first rejected lifecycle counterexample. Converting an in-memory task copy from
  `draft`/pending review to `ready`/approved review leaves the protocol digest exactly unchanged
  because `blockers` is already empty and only administrative status/review fields change.
- Re-ran the first rejected receipt counterexample. With actual capped capacities `[2,2,2]`, the
  unchanged-total redistribution eligible `[1,3,1]` plus rejected `[1,0,0]` is rejected because
  one optimized view claims three candidates from capacity two.
- Attacked `n_init_3d` with zero, negative, boolean, and float values; attacked all aggregate
  scalars and per-view lists with bool/int/float aliases, negatives, wrong lengths, and inconsistent
  sums; and attacked coherent false totals against the actual capacity vector. Every mutation was
  rejected while one exact receipt passed.
- Attacked `target_component_counts_used` with the wrong container type, insufficient/empty
  length, boolean/float entries, and zero/negative capacities. Attacked optimized views with
  bool/float entries, duplicates, negative and out-of-range indices, and length mismatch. Every
  mutation failed closed. A realized reviewer-authored tiny `FieldLiftResult` passed and its
  selected candidate total exactly matched the indexed capped component counts.
- Mutated every eligibility-policy field, outcome-input list, base binding, source-binding hash,
  source-binding algorithm/scope, driver identity, task identity, and task/run/root command path;
  added extra keys to the exact policy/base/source records. Every mutation was rejected.
- Confirmed all synthetic aggregate, synthetic-cell, calibrated worker, and warmup commands invoke
  the successor wrapper and successor task/run root. The imported base module carries the same
  patched file/task/run/driver identity.
- Confirmed native and candidate arms share placement mode, 64-track cap, eight selected training
  views, 512-component cap, depth/robust-sweep controls, projection dilation, and cell seed. Mask,
  association, topology, and refit differences remain the predeclared downstream treatments.
- Confirmed the source computes the train-only search AABB before eligibility and the draw,
  filters component-center rays by a positive-depth AABB interval, allocates the exact requested
  count with the existing stable capacity-aware quota rule, and uses the frozen local CPU seed over
  eligible top-mass pools. A reviewer counterexample proved insufficient global capacity raises
  the exact failure before `torch.randperm` can execute; deterministic repeats retained the exact
  count and lineage.
- Restricted predecessor inspection to immutable protocol and permitted failure chronology. The
  base task/review remain unchanged; root failure, run-receipt, and terminal failure hashes remain
  `aaebf3f69e5ff43ba1bf48b8e5f9088c26324af96337fef0b6c5d749ccdb7dd6`,
  `388913389fb878283786568082b12698d83f7d3535d2fda589d87926ce72c7cd`, and
  `f3e9d1352a920c8d2994f19fddd63b1012a054d89723626a5f1bdb90cdc73763`.
  No predecessor successful summary, metric, model, report, or viewer was opened.
- All nine focused fixed-sweep/successor tests passed. The broader correspondence, protocol,
  probabilistic-pipeline, experiment-contract, refit, and field-lifter suite passed. The complete
  CPU suite, including slow tests, passed with only the two documented PyTorch warnings.
- Ran `./scripts/verify.sh`. Ruff, format, the complete non-slow CPU suite, docs sync, ARA,
  script layout, agent workflow, and experiment-contract checks all passed. `git diff --check`
  was clean.
- Recomputed every protocol/source/base/cell/plan/data binding after verification. Immediately
  before this artifact was written, the task remained `draft` and pending, `blockers` remained
  empty, and neither this canonical path nor the successor run root existed.

## Findings

The exact v2 protocol and source bindings above are **approved** for execution. The eligibility
repair is deterministic and arm-independent: train-only geometry defines the search volume,
ineligible rays cannot enter quota allocation or the seeded draw, sufficient capacity preserves
the requested pre-mask anchor count, and insufficient capacity aborts rather than reducing it.
The successful-fit receipt binds exact total and per-view eligibility counts to each optimized
view's actual capped candidate capacity with strict JSON types and ranges before primary or
independent-half hard-invariant evaluation.

Both v1 blockers are resolved without changing a scientific cell. The task can transition from
draft/pending to ready/approved without protocol-digest drift, and impossible per-view capacity
receipts now fail closed. No concrete prospective blocker remains.

Only after the Driver records this exact approval and transitions the task to `ready` may the
canonical run be initialized and the exact command executed. Any digest-bearing task edit, bound
source change, or pinned base-driver byte change invalidates this approval. Approval establishes
fitness to execute; it does not establish successful completion, favorable metrics, method
superiority, browser/report usability, orbit-viewer availability, or any scientific outcome.

## Protected Actions Not Taken

I did not edit the task into `ready`, fill its review record, initialize or execute the successor
run, invoke an official synthetic or calibrated worker, access a successor outcome, inspect a
successful predecessor outcome, attach or read result/audit payloads, render or open an official
index page, launch an orbit viewer, change a default, make a scientific claim, commit, push, or
publish. Reviewer-authored counterexamples were in-memory or tiny deterministic fixtures and
created no retained result. Outcome Access remained `none` throughout.
