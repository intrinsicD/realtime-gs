# BENCH019 coordinator source review

Reviewer: Codex-cross-repo-review
Self-reviewed: No
Verdict: accepted within coordinator source scope; integration and execution approval pending
Outcome access: none; review used source, draft protocol and synthetic temporary fixtures only

## Exact reviewed source

scripts/experiments/20260907_bench019_local_stage_frames00008_00009.py
SHA256: 88fe5da19fde2419b40c1355ebbe8116e293ce5bc2a2b928c42a5b796c72046e

tests/test_bench019_local_driver.py
SHA256: fd4a705fcb528038c6635cc8af80642cffc70a5dd521069b3edf921123a187b0

The experiment task remains a changing draft, external/internal source bindings are being prepared, and report-helper glue is not covered by this acceptance. This is not a canonical prospective protocol approval or permission to init-run/fit.

## Findings and disposition

1. External source completeness: initially, listed hashes could remain unchanged after a clean StructSplat commit added a behavior-bearing file. Corrected with prospective safe relative patterns and exact observed-path set comparison before per-file hash checking. The new test rejects an added source module and permits unrelated metadata. Resolved in reviewed source.
2. Resource scope: initially, an uncontended NVML receipt claimed compilation excluded although worker clocks include lazy import/JIT/cache setup. Corrected to label the inclusive worker boundary; task scope also states that warmups do not prove compilation absent later. Resolved.
3. Earliest failure receipt: initially, malformed/missing task JSON made the exception handler reread the same broken task and fail before recording anything. Corrected to derive the permitted canonical root from constant TASK_ID and use the initialized lock for the failure start time. A malformed-task regression now proves failure JSON and failed run receipt are written. Resolved. Minimal v2 failure report sources remain integration work.
4. Phase-two publication race: the coordinator releases workers as soon as the canonical frozen protocol exists. Finalizing directly there before committing required Struct task/review metadata could trigger its clean-worktree guard and invalidate the run. The driver agreed and added the prospective lifecycle: finalize under a staging filename, complete metadata-only commit and clean/source checks, then atomically publish the canonical filename. This operational requirement must remain in the final digest-bound task.

No remaining blocking finding in the reviewed coordinator code scope.

## Correctness assessment

The coordinator constrains canonical root and worker matrix; checks both current clean worktrees, immutable ready RTGS task bytes/protocol/review/raw seal, exact RTGS HEAD, live internal/external behavior envelopes; and verifies generated inventory plus exact Struct protocol/split/family/seed/schedule bindings before phase-two dispatch. Artifact validation uses the existing frozen-protocol API with allow_review=False. The top-level schedule respects the two-phase barrier, excludes warmups from primary matrix, gates remaining primary cells on two A/A comparisons, and uses frozen representative previews without model selection. Complete failure/report acceptance still depends on the pending integration path.

The frozen StructSplat v1 exporter expects 18 primary plus one A/A row, while this local coordinator produces two A/A pairs. Preserve the extra contained replay and its pass/fail receipt as separately bound local evidence unless the exact protocol schema explicitly represents it; do not silently feed 20 rows to the unchanged 19-row expected set or discard the additional gate evidence. The report owner was informed.

## Independent verification

Ran from /home/alex/Documents/realtime-gs:

CUDA_VISIBLE_DEVICES= MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=/home/alex/Documents/structsplat/src:/home/alex/Documents/realtime-gs/src /tmp/structsplat-rtgs-collab/venv/bin/python -m pytest -q tests/test_bench019_local_driver.py

Final result: 21 tests passed.

Eight additional synthetic guard probes exercised a matching baseline, dirty RTGS, dirty StructSplat, different RTGS HEAD, source-envelope error, changed raw-seal bytes, changed review bytes and changed task bytes. Baseline passed and every mutation failed. Git/raw-source helpers were controlled mocks; no protected files were opened.

Three independent synthetic orchestration probes exercised the actual run_experiment schedule with mocked workers/protocol validation (no models or metrics generated):

- Missing phase-two approval: exactly six input-worker dispatches, then timeout; zero downstream workers and downstream directory absent.
- Failed A/A: six inputs, six warmups, four measured/replay fitting workers and four evaluation workers; gate fails and no remaining primary cell/report runs.
- Successful synthetic schedule: six inputs, 26 fitting workers (six warmups, 18 unique primary cells, two A/A replays), 20 evaluation workers, one A/A gate, two frozen representative preview workers and one report publication. Exactly two primary cells precede the A/A gate; no duplicate primary key.

## Remaining integration conditions

Finish and review report-helper success/failure glue, exact source inventories and resolved final task bytes. Preserve the staging-publication lifecycle and current/initial Struct commit provenance. Run required full repository gates and retain CUDA tiny-fit/Trainer diagnostics. Then obtain canonical independent approval of the exact RTGS protocol digest before init-run/fitting, followed by outcome-unseen StructSplat finalization after phase 1. This review neither inspects outcomes nor grants those approvals.
