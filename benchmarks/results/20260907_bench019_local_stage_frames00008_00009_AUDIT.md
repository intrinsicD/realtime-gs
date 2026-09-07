# 20260907_bench019_local_stage_frames00008_00009: independent results audit

Accepted as bounded development evidence. Neither normalized family meets the complete frozen two-frame materiality rule. No general Stage1 surrogate, default change, or speed claim is supported. Final report/browser delivery gates remain separate completion requirements.

Reviewer: `Codex-cross-repo-review`, distinct from the producing driver. No fitting, selection, threshold adjustment or result rewriting occurred during this audit. The prospective reviews preceded their respective execution phases; outcomes were accessed only after choices were frozen.

| Family minus native additive | Frame 8 mean PSNR delta | Frame 9 mean PSNR delta | Complete materiality rule |
|---|---:|---:|---|
| structsplat_normalized | -2.852263dB | -2.708337dB | Failed |
| structsplat_contained | +0.896715dB | +0.312826dB | Failed |

Contained frame 9 seed 19001 regressed by **0.238384dB**, so its positive frame means do not satisfy the required positive change for every seed. The contained alpha-IoU mean changes were -0.004087 and +0.005864. Uncontained alpha IoU changed -0.066112 and -0.059634. Failed materiality is bounded to this capacity/horizon and does not establish equivalence or reject a general hypothesis.

All 18 primary cells, both A/A replays, six warmups and six Stage1 acquisitions completed. Each frame/family used eight 512-row training fields and a fixed 256-row SH0 lift/refinement endpoint after 1000 updates. Training histories contain only the frozen training views, no held-out evaluations, no density changes and the prescribed final checkpoint. Shared training-view schedules, camera geometry, field hashes, resolved configurations and all checkpoint descriptors were checked.

Independent CPU arithmetic reproduced 60 per-view raw-statistic records, all 20 cell aggregates, paired-seed means/ranges/sample standard deviations, both A/A gates and resource exclusions. Native A/A PSNR/IoU differences were 0.000760810dB/0.0000452883; contained differences were 0.005841467dB/0.000179833, within the frozen 0.05dB/0.01 tolerances.

Independent CUDA replay cold-loaded all 18 primary final models and rendered all 54 held-out views on **RTX 4090, torch 2.13.0+cu130, CUDA 13.0**, with the frozen gsplat settings. Independent NumPy float64 scoring differed by at most 3.55e-15dB PSNR, with zero IoU/leakage difference and exact binary counts. The maximum raw squared-error-sum difference was 4.55e-13. The predeclared diagnostic tolerances were 1e-5dB/1e-6IoU/1e-7leakage; experimental tolerances were unchanged. Sandbox CUDA was unavailable; the successful replay used approved host execution.

Camera-only least-squares bounds were independently reconstructed, and all 20 saved initial models were checked against persisted rays/depths, the original AABB and unsupported midpoint fallback. Uncontained fallback fractions were 46.1–50.4%, native 16.4–20.7%, and contained 3.9–6.6%. Every frame/seed family spread exceeds 10 percentage points. These differences constrain mechanism attribution: the evidence compares the specified fitting families interacting with fixed FieldSweep and RGB refinement, rather than isolating a compositor equation.

The portable Struct export has 19 rows: 18 primary plus native A/A. The additional contained A/A remains in the 20-cell local audit. Independent ranks/correlations and held-out-frame top-choice agreements match the report: foreground PSNR and boundary MAE each have pooled within-frame Spearman -1; support IoU has +0.75. The two frames share one capture, below the frozen minimum 3, so the decision remains `question_unavailable`, `selected_predictor=null`. One-capture bootstrap endpoints are degenerate and convey no generalization uncertainty. Existing manifest `claim_ready=true` records a frozen/complete/A/A-valid export; it is not scientific promotion.

Stage1 is charged once per frame/family; warmups and A/A repeats are excluded from primary resource aggregates. Complete consumed field bytes include the manifest and eight archives with camera/alpha/framing payloads. Counts are not a codec rate or compression claim. Three of 20 measured/replay worker receipts sampled a foreign compute process. All timing remains descriptive under the frozen contention/JIT policy; own-process NVML, Torch allocated/reserved and total-device figures are kept distinct.

| Claim or interpretation | Disposition | Bound evidence |
|---|---|---|
| execution: All frozen 18 primary cells, two A/A replays, six warmups and six shared Stage1 acquisitions completed at fixed budgets. | confirm | `audit/bench019-independent-cpu-audit.json` |
| contained: Contained StructSplat meets the complete materiality rule. | narrow | `audit/bench019-independent-cpu-audit.json` |
| uncontained: Uncontained normalized family underperforms native additive in this frozen pipeline. | confirm | `audit/bench019-independent-cpu-audit.json` |
| cold_replay: Saved final-model metrics are independently reproducible on the named CUDA renderer. | confirm | `audit/bench019-independent-gpu-replay.json` |
| surrogate: Stage1 predictors establish a general downstream surrogate or independent replication. | retire | `audit/bench019-independent-statistics-geometry.json` |
| mechanism: The comparison isolates compositor equations or containment alone. | narrow | `audit/bench019-independent-statistics-geometry.json` |
| performance: Observed local timings establish a speedup or compact-only GPU memory claim. | retire | `audit/bench019-independent-cpu-audit.json` |
| bytes: Reported field bytes count complete consumed training manifests and archives. | confirm | `audit/bench019-independent-cpu-audit.json` |
| provenance: Execution was bound prospectively to exact source, task, raw inputs and generated fields across the two phases. | confirm | `audit/bench019-independent-cpu-audit.json` |
| manifest_label: Struct manifest claim_ready=true permits scientific promotion. | narrow | `structsplat_report/decision.json` |

The interpretation table includes prohibited inferences; it does not imply that every unsupported interpretation appeared in producer prose.

Exact execution provenance: RTGS `f165d353ff9e9174c1ac1c936ed7855cc7a25abb`; Struct initial `6ff898e8682cc7d932d540b818b3f0d6a4a0e529`, followed by allowed metadata-only `3ada86049863a6a7293cf22f47a31f9f728a9445`. Source byte envelopes, raw seals, all 144 generated input files, the committed/run protocol equality and immutable ready task were reverified.

Actual audit commands, environments, complete source/checkpoint hashes and numerical recalculations are recorded in the sibling AUDIT JSON and canonical run `audit/` receipts. Struct `scripts/check_report_bundle.py` passed. CPU/full verification and actual CUDA preflight receipts from source acceptance remain under `preflight/`; unchanged executable code did not require repeating the full suites.

No blocking evidence finding remains. Shared RTGS report rendering/manifest validation and real browser/WebGL/orbit smoke must pass before overall task closeout; their completion is not asserted by this scientific audit. Promoting beyond this development scope requires a new prospective multi-capture protocol and appropriate uncontended resource measurements. No new experiment is authorized by this audit.
