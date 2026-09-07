# Prospective Protocol Review

- Task ID: `20260907_bench019_local_stage_frames00008_00009`
- Protocol SHA-256: `0df467f6418273465167297edc866c7d526c78aac11e8c0f97fe6e409f7cf2c5`
- Reviewer: `Codex-cross-repo-review`
- Verdict: `approved`
- Outcome Access: `none`

## Scope

Approve the exact implemented two-phase protocol for a local, within-capture development comparison of three specified fitting pipelines on frames 00008/00009. This initial approval permits clean init-run and training-only Phase 1 acquisition. Downstream execution remains barred until a separate outcome-unseen StructSplat review approves the exact generated-field envelope. No general-surrogate, equivalence, speed, compression, compact-only memory or default-promotion claim is approved.

All scientific choices for both phases are frozen now. Phase 2 authoring may resolve hashes, exact generated semantics and clean source provenance only; it cannot change families, data roles, settings, predictors, responses, A/A tolerances or decision rules after fields exist.

## Checks

- Raw seal independently revalidated against source files; no images decoded. Both frames use the same frozen eight training and three held-out IDs. Acquisition and refinement loaders select training views only; held-out sources are reporting-only.
- Complete resolved Stage 1 and downstream dataclasses independently compared to the implemented constructors; exact match. Shared float soft-matted crop, unweighted Stage 1 L2, fixed512 rows/1000 updates, native mask=None and paired normalized containment control are explicit. Native-source versus declared-observation drift is descriptive; cold serialization is a hard gate.
- Fixed CPU FieldSweep with 256 rows and CUDA-only SH0 refinement with 1000 updates, no densification/early selection, final checkpoint, three paired seeds, six separate warmups and two A/A pairs are fully specified. Refinement retains the declared random-background masked L1/DSSIM/alpha objective. Unsupported midpoint fallback is recorded without retuning.
- Predictor signs/priority, full-canvas matted-target scoring, alpha/support distinctions, raw denominators and aggregation are exact. Resource bytes include consumed manifests/archives; Stage 1 acquisition is charged once. Warmups/A/A are excluded from primary time/memory aggregation. Lazy import/JIT/cache work is included when incurred; contention/NVML absence cannot support speed or substitute process-memory claims.
- Development materiality requires positive paired PSNR differences for every seed, mean gain at least 0.25 dB in both frames and mean alpha-IoU loss at most 0.02 per frame. General validity remains not_evaluated_insufficient_scope for one capture, regardless of within-frame rankings. Missing/failed cells and failed A/A invalidate comparison; no tolerance tuning or outcome-guided regeneration.
- Live RTGS source envelope independently verified: 147 files, aggregate SHA256 `7c60dc19b92cae29698d196835953e6b6e4e984a5653ee398e32bfc7897f9a71`. External StructSplat envelope: 248 files, exact paths and hashes independently verified; canonical external-binding JSON SHA256 `c863f4e1d232bf3b46dc996db02d1eb5b9cb6872bb8433d4e550e9cb13b669a5`. Both include Python/CUDA/C++/headers and build configuration. Exact ready-task/review/raw-seal bytes are locked separately, avoiding self-referential source hashes.
- Both worktrees must be clean before acquisition and at guarded boundaries. RTGS HEAD must equal the init-run lock. Only the prospectively allowed Struct metadata-only review commit may advance between phases, with unchanged executable envelopes and recorded old/new provenance. Finalize under a staging filename, commit required review metadata and confirm clean sources, then atomically publish the canonical frozen protocol.
- Independent focused verification: 60 CPU tests passed across the two helpers, coordinator and report tests; four CUDA diagnostics skipped in this review's pinned CPU environment. Additional independent NumPy metric/morphology/paired-comparison oracles, guarded-state probes and complete synthetic scheduling/publication probes passed. The complete RTGS CPU/structural verify gate passed with absolute PYTHONPATH; the relative-PYTHONPATH environment failure was resolved without source changes. No slow tests were collected. Driver-retained real CUDA tiny-fit/Trainer and prior kernel/adapter diagnostics support the selected backend; they do not establish calibrated quality.
- Strict portable 19-row export preserves 18 primary plus nativeA/A; the extra contained replay remains separately bound local evidence. Shared v2 report and both corrected nested-manifest validators passed synthetic rendering, omission and tamper checks. Failures retain machine-readable records and empty/unavailable sources rather than invented metrics. Producer writes RESULT only; separate raw-results audit and real browser/viewer smoke remain mandatory.

Final reviewed changed runtime SHA256 values:

- `src/rtgs/bench019_local_inputs.py`: `674ebfe13c831739e5a6af09500856e1d5b306323728289739cb7dc84ab52466`
- `src/rtgs/bench019_local_downstream.py`: `5f220c101cb002f436697dce9059ff7210ae283ec3f54a599adc89870755d8eb`
- `src/rtgs/bench019_local_report.py`: `10c7d640e39e4a504162f27c704c05d9fd6f25b71ef7a2c267061d248de09968`
- `scripts/experiments/20260907_bench019_local_stage_frames00008_00009.py`: `57c50de238c3836b688d11233f02c46f1a05b746af6fbf84cee2bea3bad7bd67`
- `scripts/experiment_contract.py`: `15f129398aabb46a50d435c76fec27b4df6c75e6d08d2e7d78a5a96657b25959`
- `scripts/check_results_bundle.py`: `0f5e1293a4c6651d1fed80237bd1319f44ed2ca268d81cb27ce87d2e54f2fedf`

## Findings

No blocking findings remain in the exact approved protocol and source scope. Review findings on external-source additions, truthful timing boundaries, malformed-task failure receipts and nested field-manifest coverage were corrected and independently tested. Claude's prospective design feedback is retained with explicit disposition; its unsupported capacity prediction is not treated as evidence.

The accepted StructSplat initial metadata commit is `6ff898e8682cc7d932d540b818b3f0d6a4a0e529`; it changes task/index/session metadata only and keeps production sources and historical evidence unchanged. The RTGS driver must now commit the approved source/task/review metadata cleanly before init-run. Initial approval does not waive the separate Phase 2 review or final results/report/browser acceptance gates.

## Protected Actions Not Taken

This reviewer did not initialize or execute the protected run, fit any local image, inspect generated fields or calibrated downstream outcomes, tune from held-out observations, change protocol choices, or write an outcome-audit approval. Only source, raw seals, task metadata, synthetic test artifacts and verification logs were inspected. This change records approval and sets administrative ready/review metadata only; the protocol digest is unchanged.
