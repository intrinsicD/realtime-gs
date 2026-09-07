# BENCH019 task-local helper source review

Reviewer: Codex-cross-repo-review
Self-reviewed: No
Scope: implemented Stage-1 input helper and fixed downstream worker; preliminary integration design
Verdict: accepted within helper scope; complete execution approval remains pending
Outcomes accessed: No. No local capture, generated field, or protected downstream outcome was inspected or produced.

## Reviewed state

- src/rtgs/bench019_local_inputs.py SHA256 674ebfe13c831739e5a6af09500856e1d5b306323728289739cb7dc84ab52466
- tests/test_bench019_local_inputs.py SHA256 cff67506e1dc0214f535606e5e7f1aa651d660b7e905175ea4fbdf3948d69631
- src/rtgs/bench019_local_downstream.py SHA256 5f220c101cb002f436697dce9059ff7210ae283ec3f54a599adc89870755d8eb
- tests/test_bench019_local_downstream.py SHA256 d70847eabc7623e7a592f62a920a836ddfbe6f57696cde66f3c12b0b24382e29

The experiment task is actively being revised by the driver and is not approved by this note. A planning-time task hash in bench019-helper-review-probes.json identifies the draft observed during probes; it is not an execution-approved digest.

## Correctness

No blocking source findings within these helper boundaries. Acquisition uses one float soft-matted target/crop, native mask=None to avoid an extra crop or weighted objective, normalized L2/SSIM=0/no weighting, and otherwise identical normalized configurations with containment as the declared geometric difference. Fixed counts/horizons and observer checkpoint semantics are checked. Deterministic full-canvas sample coordinates are shared across families; support explicitly requires query.valid and weight_sum>1e-8, which avoids treating crop validity as support or alpha. Cold serialized observation semantics, camera framing, source/config provenance and array results are checked. The current mask-alpha==0.5 guard explicitly rejects mismatch with the compact binary-mask metadata contract; calibrated nearest-resized 8-bit masks are the declared supported inputs.

The downstream helper cold-loads only the exact eight training fields on CPU and validates row count, blend semantics, manifest membership and absence of mask-derived bounds. FieldSweep uses its existing CPU implementation and a fixed 256-row budget. RGB Trainer explicitly requires CUDA, disables topology changes and internal evaluation, and saves the final 1000-update model. Training/heldout loads are separate. Saved model arrays replay exactly. Final reporting uses binary >=0.5 regions, the soft-matted black target, display-clamped RGB, mean per-view PSNR and auditable sufficient statistics. The metrics are appropriate to the explicit matted-target definition; they are not interchangeable with raw-RGB foreground PSNR.

Refinement retains the frozen Trainer random-background masked L1/DSSIM plus alpha objective. Equal Stage-1 L2 does not imply downstream L2. This must remain visible in task/report prose and resolved configurations. Peak Torch memory is reset before field loading and Trainer resets are disabled. The refinement clock includes fixed checkpoint serialization and is labeled accordingly.

## Independent evidence

Ran from /home/alex/Documents/realtime-gs:

CUDA_VISIBLE_DEVICES= MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=/home/alex/Documents/structsplat/src:/home/alex/Documents/realtime-gs/src /tmp/structsplat-rtgs-collab/venv/bin/python -m pytest -q tests/test_bench019_local_downstream.py

Result: 16 CPU tests passed, 1 CUDA diagnostic skipped.

Same environment, tests/test_bench019_local_inputs.py:
Result: 12 CPU tests passed, 3 CUDA diagnostics skipped.

Additional independent probes: 128 randomized foreground/IoU/exterior cases with noncontiguous tensor views checked against a NumPy scalar-formula oracle, including exact threshold values; 24 deterministic pixel-sampling cases checked against an independently constructed exhaustive 3x3 neighborhood boundary oracle with exterior padding. All passed. Source hashes and counts are recorded in /tmp/structsplat-rtgs-collab/bench019-helper-review-probes.json.

Producer-reported real CUDA tiny-fit/Trainer evidence must be retained by the driver and combined with required full gates; this reviewer did not independently rerun GPU tests.

## Simplicity and ownership

The helpers reuse the existing fitters, observation export/serialization, calibrated loader, FieldSweep, Trainer and rasterizer. Historical portfolio validation is untouched. The coordinator, rather than these helpers, owns prospective approval, sealed input paths, process boundaries, comprehensive failure receipts, NVML monitoring, phase barriers and report generation. Helper success therefore cannot authorize standalone protected execution.

## Required integration gates before execution approval

1. Finish coordinator/report source and meaningful tests; inspect their exact frozen diff separately. Complete required repository gates and preserve GPU diagnostic receipts. Freeze final task definitions and remove unresolved draft blockers only when actually satisfied.
2. Bind all behavior-bearing source in both repositories, including Python, CUDA/C++, headers, runtime/protocol/report scripts and environment/build configuration. Exclude the task and review record from their own source digest to avoid circularity; bind their exact bytes separately. A source hash over only Python is incomplete for this CUDA comparison.
3. Start phase 1 from clean identified source commits and immutable approved RTGS task/review/raw seal. Require current clean worktrees and identical behavior-byte envelopes at phase/coordinator/worker entry and exit. Keep original and current commit identities in receipts. A later metadata-only StructSplat task-review commit is acceptable only under the prospectively frozen policy, with unchanged executable/configuration envelopes and unchanged RTGS ready task/lock/review.
4. Do not rely on existing validators for live-source enforcement: StructSplat _validate_repository validates recorded metadata, not today's HEAD/status; RTGS _locked_task validates source_commit shape, not today's equality. The coordinator must explicitly enforce the declared current-clean and byte-equality policy. The existing RTGS source binding checker covers only the repository rooted at its call; external StructSplat source needs its own enforced envelope.
5. Produce phase-1 artifacts only under inputs/, freeze their complete inventory and supported-metric files, and keep downstream absent/empty until independent StructSplat prepare-review/finalize completes. The finalized envelope must bind the exact immutable ready RTGS task, generated fields, metrics, schedule and source metadata. Test missing/changed review, changed raw/generated/source bytes and attempts to dispatch phase 2 early. No outcome-guided regeneration or in-place repairs.
6. Freeze all post-input Struct protocol design choices now (families, metrics/signs/priority, two frames, three paired seeds, two A/A checks, materiality and insufficiency). Postphase1 finalization may resolve hashes and provenance only, not choose a more favorable comparison after inspecting field quality.
7. Preserve the complete four-stage experiment scope even if task.stages contains only refinement for the shared report's per-seed history schema. Shared Stage-1 acquisition belongs once in clearly linked per-view histories/resource records, with lift/evaluation timings separately scoped and visible; do not duplicate Stage-1 measurements across seeds or imply its omission from the history schema removes it from total cost. Report-helper fixtures must demonstrate this distinction.
8. Canonical execution approval requires a distinct approved exact RTGS protocol digest after implementation review. Final scientific acceptance additionally requires independent raw-results audit, complete paired/A-A cells and both report validators including real RTGS browser/viewer evidence. This note grants none of those later approvals.

## Optional improvements

No helper-only change requested. Keep future additions bounded to concrete integration findings. The experiment remains a two-frame, one-capture development comparison of fitting pipelines at one fixed downstream capacity and horizon; no general surrogate/default/equivalence claim follows.
