# Prospective Protocol Review

- Task ID: `20260805_probabilistic_field_pipeline_retry_mixed`
- Protocol SHA-256: `03722e78b6b303bd87b60da1d8ac61b210a5cb4e4ecf52a49431139f64a78418`
- Source-tree SHA-256: `e1c442a54585aa67fbfa1e57c6bbb73c77eb5a61c95889a38a33f8941383651a`
- Reviewer: `Codex-probabilistic-field-protocol-reviewer`
- Verdict: `approved`
- Outcome Access: `none`

## Scope

This was an outcome-blind prospective review of the retry task's v2 task-only amendment. The
review asked whether removing an invalid executable dependency on the zero-cell failed predecessor
is contract-correct and whether every scientific, data, source, command, resource, gate, and
549-cell field remains identical to the superseded retry V1 approval. I read the latest v2 handoff
and the complete preserved V1 approval before checking the amendment independently.

If executed successfully, the amended retry retains the V1 evidence scope: deterministic
synthetic mechanism evidence and development-only operability/utility observations for
native-control and all-candidate lifting over a deterministic 512-component-per-view proxy of the
eleven sealed fields. It cannot establish complete-field fidelity, source-RGB reconstruction
quality, true globally coupled multi-marginal OT, GPS-Gaussian reproduction, GPU or real-time
performance, cross-scene generality, a production default, or reconstruction accuracy from
independent-half agreement.

## Checks

- Recomputed the amended review digest with
  `.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260805_probabilistic_field_pipeline_retry_mixed.json`.
  It exactly matched
  `03722e78b6b303bd87b60da1d8ac61b210a5cb4e4ecf52a49431139f64a78418`.
- Recomputed the length-prefixed binding over `scripts/experiment_contract.py`, the retry driver,
  and every `src/rtgs/**/*.py` file. All 102 bound files still produced
  `e1c442a54585aa67fbfa1e57c6bbb73c77eb5a61c95889a38a33f8941383651a`, exactly matching both
  the amended task and retry V1. No bound source byte changed.
- Confirmed the V1 approval is preserved at
  `experiments/reviews/20260805_probabilistic_field_pipeline_retry_mixed_PROTOCOL_REVIEW_V1_SUPERSEDED.md`
  with SHA-256 `f096e6fd1e0a3ac113eb028180db77254d7dfd66990141ecd3cd639ee1f067c5`.
  Its reviewed protocol digest was
  `91e2c662d079742e99b7336d2f5a5de7ed4a55e733653e5537a6090570b9ce0d`.
- Audited the exact protocol delta in memory. The amended task has `depends_on=[]`. Restoring only
  V1's `depends_on=["20260805_probabilistic_field_pipeline_mixed"]` while leaving every other
  digest-bearing field untouched reproduces the V1 digest
  `91e2c662d079742e99b7336d2f5a5de7ed4a55e733653e5537a6090570b9ce0d` exactly. Therefore
  `depends_on` is the sole protocol-bearing amendment. The only non-protocol lifecycle changes are
  the required reset from recorded V1 approval/`ready` to `status="draft"` with a null,
  `verdict="pending"` review envelope and relocation of V1 out of the canonical review path.
- Confirmed the refused-initialization ordering from the handoff and contract implementation.
  `init_run` validates the task, requires `status="ready"`, verifies the data seal, and validates
  every `depends_on` run before it computes the retry run path, checks for collisions, builds the
  lock, calls `run.mkdir`, or writes `task.lock.json`. Dependency failure raises the exact
  `dependency ... is not a complete canonical run` error at that earlier boundary. Reaching that
  boundary after V1 approval establishes that task/review/status/seal validation had passed; the
  later root-creation statements were not reached.
- Rechecked the dependency semantics. `depends_on` is an executable prerequisite checked by
  `validate_run` against a complete canonical run bundle; it is not a general provenance edge.
  The predecessor terminated during synthetic initialization with zero completed/measured cells
  and did not form such a complete bundle, so it cannot satisfy this prerequisite. The retry does
  not consume a predecessor result. Removing the edge is therefore the minimal contract-correct
  amendment, while the predecessor task ID, import failure, zero-cell boundary, and unchanged-
  protocol statement remain explicit in the retry claim boundary and durable handoffs.
- Confirmed the refused attempt created no retry directory. Before this review, neither
  `runs/20260805_probabilistic_field_pipeline_retry_mixed` nor a retry `task.lock.json`, worker,
  cell, metric, result, or outcome existed. The canonical v2 review path was also vacant.
- Ran contract and sealed-data validation; both returned `OK`. The task still references the
  exact same data-seal file with SHA-256
  `20e719d89628375c515db94102abf6e5018dbd6d686d0633235407fad5c7deb6`: eleven datasets and
  309 files totaling 204,306,829 bytes, comprising 296 compact `.rtgsv` files, eleven compact
  manifests, and two calibration files, with no image file. Dataset roles and paths, every
  train/held-out split, seeds `80501`/`80502`/`80503`, both calibrated arms, comparators, metrics,
  invariant gates, stopping rules, input guards, and CPU resource accounting remain unchanged.
- Inspected the outcome-free amended plan. It contains 549 unique cells with the unchanged stage
  counts: 324 exact-shape, 60 association, 81 support-mask, 6 topology, 6 schedule, 6
  independent-half, and 66 calibrated cells. Schedule cells retain five cleanup iterations; the
  calibrated matrix remains eleven datasets by three seeds by `native_controls` and
  `all_candidate_mechanisms`. Canonical serialization of all cells still has SHA-256
  `1ca991e48c5bf43c0ef32b8e83af5d9a693434992e990020f382d2b86753eb89`, exactly matching V1.
- Confirmed the exact command and all task/driver/run paths are unchanged from retry V1 and still
  target only `scripts/experiments/20260805_probabilistic_field_pipeline_retry_mixed.py`,
  `experiments/tasks/20260805_probabilistic_field_pipeline_retry_mixed.json`, and the one vacant
  `runs/20260805_probabilistic_field_pipeline_retry_mixed` root.
- Ran
  `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_fiber_correspondence.py tests/test_probabilistic_field_experiment_protocol.py tests/test_probabilistic_field_pipeline.py tests/test_experiment_contract.py tests/test_field_refit.py`;
  all 71 focused outcome-free tests passed. The workflow checker also passed with the reviewer
  turn active.
- Ran `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh`. Ruff, format, the complete non-slow CPU suite,
  docs sync, ARA, script layout, agent workflow, and experiment-contract checks all passed. Only
  the two already documented PyTorch warnings were emitted.
- Recomputed both digests after verification and reconfirmed that the retry run root and canonical
  review path remained absent before this artifact was written.

## Findings

The exact amended protocol and unchanged source binding above are **approved** for execution. V1
correctly approved the import-guard repair and unchanged scientific producer, but its subsequent
initialization attempt exposed a task-graph misuse before any retry root or outcome existed. The
v2 amendment removes only that invalid executable prerequisite, preserves the failed predecessor
as explicit zero-cell provenance, and changes no result-producing treatment, control, input,
metric, gate, resource rule, command, or source byte. No concrete amendment blocker remains.

The explicit lifecycle chronology is: the original predecessor failed before cell 1; retry V1 was
prospectively approved with Outcome Access `none`; the owner recorded V1 approval and reached
`init-run` dependency validation; `init-run` refused the incomplete predecessor before creating
the retry root or lock; the V1 review was preserved as superseded; the task was amended only at
`depends_on` and reset to draft/pending; this outcome-blind v2 review approved the amended digest.
Only after the owner records this exact approval and returns the task to `ready` may guarded
initialization be attempted again.

Any further digest-bearing task edit or bound-source change invalidates this approval. Approval
establishes fitness to initialize the amended retry; it does not establish successful execution,
favorable metrics, method superiority, or any scientific outcome.

## Protected Actions Not Taken

I did not invoke `init-run`, create the retry root or lock, execute the retry coordinator, run an
official-seed synthetic cell or calibrated worker, access any retry outcome, render a report,
create or open an index page, launch an orbit viewer, update the task/index/current-task state, or
interpret a quantitative result. Outcome Access remained `none` throughout.
