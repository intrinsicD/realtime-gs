# Prospective Protocol Review

- Task ID: `20260805_probabilistic_field_pipeline_mixed`
- Protocol SHA-256: `d169a381e1f3f36a1cd962a3f0d81d1437e61b26ccecc2927da425d03b24b45b`
- Reviewer: `Codex-probabilistic-field-protocol-reviewer`
- Verdict: `rejected`
- Outcome Access: `none`

## Scope

I reviewed the exact development-phase protocol for deterministic synthetic mechanism tests and
the image-free native-control versus all-candidate comparison over the eleven sealed local
Gaussian2D field sets. If corrected and executed, this protocol could support bounded mechanism
evidence and development-only operability/utility observations for a deterministic
512-component-per-view teacher proxy. It could not establish complete-field fidelity, source-RGB
quality, a true multi-marginal OT method, GPS-Gaussian reproduction, GPU or real-time performance,
cross-scene generality, production-default suitability, or accuracy from independent-half
agreement.

## Checks

- Independently recomputed the protocol digest with
  `.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260805_probabilistic_field_pipeline_mixed.json`.
  It exactly matched
  `d169a381e1f3f36a1cd962a3f0d81d1437e61b26ccecc2927da425d03b24b45b`.
- Ran the outcome-free structural and data checks. Both
  `.venv/bin/python scripts/experiment_contract.py validate` and
  `.venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260805_probabilistic_field_pipeline_mixed.json`
  returned `OK`.
- Independently inspected `--inspect-plan`. It contains 549 unique cell identifiers with the
  declared counts: 324 shape, 60 association, 81 mask, 6 topology, 6 schedule, 6
  independent-half, and 66 calibrated cells. The calibrated cells exactly cover all eleven
  frozen datasets, three seeds, and two arms.
- Compared the task inventory with every `dataset/**/gaussians2d*/manifest.json`: all eleven and
  only those eleven manifests are listed. The refreshed compact seal binds 309 files
  (296 `.rtgsv`, eleven compact manifests, and two calibration files), totaling 204,306,829
  bytes. It contains no source RGB or external mask file.
- Rechecked every split against its compact manifest. All view identifiers are covered exactly
  once and train/held-out sets are disjoint. The nine Stage bundles have embedded alpha in all
  views; both Karate bundles are unmasked. The worker checks that optimized indices are a subset
  of training indices and disjoint from held-out indices before held-out reporting.
- Inspected the live input guard. It denies image-suffix opens and forbidden image/legacy/Beam
  imports through negative controls, while `CompactDataset` verifies the sealed manifest and
  bundle digests without Pillow. The deterministic component cap is opt-in, uses an 8x8 spatial
  stratum followed by global mass-area filling, retains original/used counts and a selection
  digest, and leaves the repository default uncapped.
- Inspected the generated per-dataset report schema. It requires exact coverage of frozen
  datasets, curves for every finite worker-summary metric across all seeds and both arms, filtered
  optimizer/stage histories, artifacts, and one orbit command per child page. Individual child
  `index.html` writes use a temporary file before replacement.
- Ran the focused outcome-free suite with
  `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q tests/test_probabilistic_field_experiment_protocol.py tests/test_probabilistic_field_pipeline.py tests/test_experiment_contract.py`;
  all 44 tests passed.
- Ran `CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh`. Ruff and format passed, but verification
  stopped at the non-slow CPU suite with
  `FAILED tests/test_agent_workflow.py::test_repository_agent_workflow_is_valid`: the exact
  reported problem was `.agents/state/current-task.md: status 'In progress' requires Turn
  'driver', found 'reviewer'`.
- The bounded calibrated workload (512 teacher components per view, at most 64 tracks, at most
  eight training views, ten refit iterations, one CPU thread, one warmup, and 66 measured
  subprocesses) is plausibly executable on the named CPU environment. That boundedness does not
  repair the measurement and protocol-binding defects below.

## Findings

The protocol is rejected before outcome access. The following are execution blockers, and any
repair changes the reviewed implementation/protocol surface and therefore requires a new digest
and prospective review.

1. **The reviewed source is not cryptographically bound for the planned development run.** The
   protocol digest covers the task object, not driver contents. `init_run(..., development=True)`
   records `git diff --binary HEAD`, which omits untracked files; the task-specific driver and
   other new implementation files are currently untracked. The driver then validates the task,
   seal, command, and lock but no driver/source digest. Consequently the exact source reviewed
   here could change without invalidating either the task digest or run binding. Commit the
   reviewed source or add and verify a complete source/driver hash that includes untracked files
   before initialization.

2. **The schedule treatment is not the frozen treatment and its speed samples are not fresh
   processes.** Each expanded schedule cell declares `final_cleanup_iterations: 10`, the task's
   general pipeline freezes five cleanup iterations, but `_pipeline_synthetic_cell` hardcodes
   three. Moreover all 483 synthetic cells run sequentially inside one `_synthetic_worker`
   process. The schedule decision nevertheless uses those timings as the preregistered
   fresh-process refit comparison. Launch each timed schedule repeat in a fresh subprocess and
   make the cleanup count agree exactly in the task, cell plan, and executed configuration.

3. **The association decision rule and negative control are not implemented as registered.** The
   registered rule requires treatment comparisons at matched accepted-parent coverage and an
   explicit dustbin/capacity-balance gate. `_synthetic_decisions` compares unstratified medians
   without matching coverage, and neither the synthetic record nor the calibrated record computes
   or enforces `dustbin_capacity_balance_tolerance`. The nominal shuffled-candidate negative
   control does not corrupt the candidate identities or projection gate; it computes the same
   plan and only shifts ground-truth labels during scoring. Implement the matched-coverage rule,
   record and enforce the declared balance residual, and corrupt the actual candidate/gate input
   for the negative control.

4. **The mask factorial does not execute its registered field-level Pareto test.** It is a
   48-element support-vector toy rather than a held-out Gaussian-field evaluation, yet its vector
   error is labeled `heldout_field_density_mse`/`heldout_field_rgb_mse` and
   `render_opacity_equal` is set to the constant `True`. The decision compares probability only
   with hard masking, uses precision-times-coverage plus density error, and ignores the `none`
   comparator, separate precision and coverage coordinates, and RGB error. This does not enforce
   the registered Pareto-nondominance rule against both hard and unmasked controls. Exercise the
   actual support/mass/opacity path on frozen synthetic Gaussian fields and implement all declared
   Pareto coordinates and comparators.

5. **Several other frozen gates are recorded in prose but are not decision-enforced.** The shape
   decision checks only known-parent covariance relative Frobenius error, not both errors named in
   its registered rule. The declared shape stage says view count is swept, but every case uses the
   same four training cameras. `split_density_mass_tolerance`,
   `split_optical_thickness_tolerance`, and the source covariance relative gate are not emitted
   and checked by the experiment decision layer. `_synthetic_decisions` reduces invariants to
   transport finiteness, positive real mass, and one source-projection scalar; `_orchestrate`
   proceeds to calibrated cells even if one of those returned invariant booleans is false. Bind
   every declared gate to a measured value and fail closed before calibrated execution on hard
   invariant failure, while keeping failed utility mechanisms diagnostic-only as registered.

6. **Rollback semantics currently swallow failures that the stopping rules classify as hard.**
   The calibrated association uses `failure_policy='rollback'`, and `_run_association` catches all
   `RuntimeError` and `ValueError` failures without classifying invalid transport/source-invariant
   failures separately. The task says a cell stops immediately on invalid transport balance,
   source projection, or a non-finite tensor. Narrow rollback to eligible utility/operability
   failures or revise and re-review the stopping rule; hard invariant violations must terminate
   the cell with a structured receipt.

7. **Failure and aggregation publication are neither structured nor atomic as promised.** A
   worker exception writes only `WORKER_FAILED.txt`, not the declared structured JSON failure
   receipt. A top-level subprocess or aggregation exception has no `try/finally` path that writes
   `run_receipt.json` with `status: failed`. Aggregation publishes dataset directories,
   presentations, machine results, root copies, configuration, metrics, and repository evidence
   incrementally with no aggregate transaction; a late exception leaves partial canonical
   outputs, while the no-overwrite helpers make the same run root non-resumable. Implement an
   atomic aggregate staging/commit boundary and canonical structured failure receipts for worker,
   orchestration, and aggregation failures.

8. **The measured resource scope differs from the frozen resource protocol.** `scope_seconds` is
   captured before `os.replace(temporary, output)`, so the reported worker `wall_seconds` excludes
   publication even though publication is explicitly inside the frozen scope. The frozen
   `output_bytes` host metric is not emitted, and the resource aggregation does not report the
   declared repeat min/max summaries. Time the exact registered boundary and serialize every
   frozen resource field before using wall time in a speed decision.

9. **Repository verification evidence is presently false/stale.** The current task handoff says
   full verification passes and also claims fresh-process synthetic workers and atomic
   aggregation. Static inspection contradicts the latter two claims, and the independent
   `./scripts/verify.sh` invocation fails on the task workflow state quoted above. Correct the
   durable task state and rerun the complete verification gate after all protocol repairs.

The data inventory, split isolation, deterministic bounded proxy, claim boundary, and child-report
design are otherwise suitable foundations. They do not make this exact frozen producer fit to
execute.

## Protected Actions Not Taken

I did not initialize or execute the canonical run, invoke any calibrated worker, open any sealed
result or outcome artifact, inspect or create
`runs/20260805_probabilistic_field_pipeline_mixed`, render a results page, launch a viewer, update
the task/index/current-task state, or make any quantitative interpretation. Outcome access
remained `none` throughout this review.
