# Independent results audit: 20260908_field_teacher_information_stage_frame00008

**Verdict: accepted with limits for execution and bounded development results.** The teacher qualifies, but the photograph baseline fails its adequacy gate and the high-field arm fails the paired requirements. D/E are unavailable. This leaves high-quality end-to-end field-only reconstruction unresolved.

Reviewer: Codex-protocol-reviewer (distinct from Driver Codex-field-experiments). Created 2026-09-08T21:00:40.121340+00:00. Final report/browser/manifest delivery gates remain pending. The prospective review remains unchanged.

## Bound result

| Arm | Foreground PSNR dB | Full PSNR dB | Boundary PSNR dB | Crop LPIPS | Alpha IoU |
|---|---:|---:|---:|---:|---:|
| A: photographs | 24.782436 | 35.811775 | 20.784817 | 0.114609 | 0.957174 |
| B: 100k fields | 24.316375 | 32.889365 | 19.278671 | 0.173856 | 0.745162 |
| C: low-budget fields | 18.552894 | 22.703759 | 11.516209 | 0.582649 | 0.494741 |

Each seed is the mean of the four frozen reporting views; each arm is the mean of three seeds. Teacher qualification uses all 22 training views: foreground PSNR 36.4558623921 dB and crop LPIPS 0.0473696128, passing the frozen >=30 dB and <=0.08 mean requirements.

A foreground means are 8101: 24.750247 dB, 8102: 24.770340 dB, 8103: 24.826721 dB. All fail the >=25 dB requirement. A LPIPS <=0.15 and IoU >=0.90 pass in all seeds. Its recognizable-subject visual criterion is met, but the numeric failure remains decisive.

B foreground proximity (within 0.5 dB of A) passes seeds 8101/8103 and fails 8102. Boundary proximity within 0.5 dB, LPIPS within +0.02 and IoU within -0.02 fail in every seed. B also visibly regresses on outline and detail. No secondary metric rescues the prerequisite.

## Independent computations and visual checks

All nine final models were independently rerendered at the four fixed reporting cameras after official evaluation. The largest discrepancy from official PSNR is 0.000002723 dB; LPIPS agrees exactly; alpha IoU differs by at most 0.000000029. Independent float64 PSNR reductions, all per-view/seed/arm means and inclusive gate components agree with the official decision. Raw 36-row audit values and official cell records are embedded in the companion AUDIT JSON.

All nine contact sheets were inspected. For every arm and seed, fixed frames 0, 4, 8, 12, 16 and 20 of both novel-orbit and novel-elevation animations were inspected in six arm-comparison montages. This is sampled animation inspection, not a claim to have viewed every frame. Image and animation hashes are preserved in the companion JSON.

- A retains a recognizable head, limbs, clothing outline and more cloth texture. Residual coarse face/feet/fine-cloth detail and angular geometry limit quality.
- B retains the subject and central garment texture, but has persistent haze, wisps or streaks around the head, garment and legs, with softer contours across all seeds.
- C has severe blur, mottling, broad opaque halos or ghosting and weak face/clothing detail.

The explicitly posthoc alpha>=0.5 check separates thresholded excess opacity from missing coverage. Mean per-view precision/recall are A 0.972716/0.983525, B 0.747409/0.996096, C 0.494793/0.999814. False-positive area divided by ground-truth foreground area is A 0.027713, B 0.343796, C 1.025683. Thus B’s IoU deficit is arithmetically dominated by opacity outside the fixed silhouette, consistent with the visible halos. These counts identify no cause and are not physical geometry truth.

## Protocol, execution and recovery

Protocol digest `b26db0f9b00acba58a86516f95b8cdac218ba5a1067cee253622f4b0e41e11d8`; source envelope `d83422a66c3a2bb6137ce0a4059d94554dbb97a7941cd7f0c8002b65b1134382`; source commit `8a715051cc789cc525e880a653a45158d65b5ab1` with exact dirty source preserved. Final audit rehashed 119 live/snapshot files and all 105 sealed inputs. All 66 target caches remain unchanged from the preparation audit.

Preparation independently matched 22 cameras and source RGB/mask hashes, recomputed 44 teacher PSNR rows, and checked all 44 parity records. Maximum direct/index discrepancy is 7.153e-7, CPU/CUDA 3.577e-7 and decoded-sample discrepancy 4.366e-11. Teacher LPIPS aggregation was checked from raw records; teacher LPIPS forwards were not repeated.

All nine final cells have 8,000 finite updates, frozen effective configs, paired camera sampling, original initialization hashes, expected SH/density schedules and finite NPZ/PLY models. No heldout metric appears in fitting history; fitting read guards show no dataset opens. The original 8,001-point initialization multiset is shared across all seeds, with deterministic row permutations. RGB and mask-derived bounds/initialization make B/C field-supervised refinement.

The last successful fit ended at run time 3734.023046 s; official evaluation starts at 3752.032324 s. Last-cell completion receipt 2026-09-08T20:43:59.619038+00:00 precedes evaluator launch 2026-09-08T20:44:04.960363+00:00. The independent rerender is audit-only and did not select a checkpoint or alter fitting.

Two interrupted workers returned exit 143, consistent with SIGTERM; initiating causes remain unconfirmed. The first four completed cells survived the first interruption, and eight survived the second. The unfinished ninth had only initialization/configuration and a log through step 500. Its partial directory/log/start record were quarantined and hash-checked. One independently approved restart used the original seed, initialization and full 8,000-step schedule; PTY changed execution transport only. The preserved retry permission is experiments/README.md:64-68, with separate review and authorization receipts under audit_checks/ and attempts/.

**At least 500 discarded updates occurred.** The exact interrupted step count and failed-attempt resource costs are unknown. Successful-cell counters omit this work; no equal-total-compute or performance claim is accepted. The canonical reproduce argv describes a fresh execution; actual continuation child commands remain recorded under attempts/. Planned presentation annotations are reviewed separately and may be added without changing frozen source, models, scores, gates or RESULT.

## Viewer warning disposition

The saved B/8101 model has 25,088 finite rows; no row was filtered for display, and converted covariance values are finite. Saved Chromium/SwiftShader telemetry shows WebGL2, a ready Gaussian renderer, changed camera pose and one renderer with frustum culling disabled. Independently counted subject pixels in the UI-free rectangle x=100, y=200, w=850, h=590 are 52,229 before and 58,575 after camera movement.

The raw console.error about THREE.computeBoundingSphere NaN is retained. Installed Viser supplies a finite two-component quad position attribute while the shader reads separate 3D centers; the committed targeted frustum-culling workaround explains this warning without invalid model tensors. Clock deprecation and software-WebGL notices are also retained. This supports functionality of the tested client, not hardware performance or reconstruction adequacy.

## Claim disposition

| Claim | Disposition | Evidence and boundary |
|---|---|---|
| K1: Nine fixed final cells and the frozen evaluator completed with preserved source and data. | confirm | 119 source/snapshot and 105 sealed input hashes; nine invariant records, successful receipts, recovery chronology. |
| K2: The high-capacity 2D teacher meets its mean training-view qualification. | confirm | 36.4558623921 dB foreground PSNR and 0.0473696128 crop LPIPS; PSNR independently recomputed, LPIPS raw-row mean checked. |
| K3: The photograph baseline is adequate for the high-quality conditional test. | retire | All three A mean foreground PSNR values are below 25 dB. LPIPS, IoU and recognizable-subject visual criterion cannot rescue the failed foreground requirement. |
| K4: The high-field arm approaches the matched photograph arm within every frozen margin. | retire | B boundary PSNR, LPIPS and alpha IoU margins fail in every seed; foreground margin also fails seed 8102; visual outline/detail regressions. |
| K5: High-field refinement scores better than the low-budget acquisition family. | narrow | All three seeds and listed metrics favor B; C jointly changes row count, 2D fit budget, cap and acquisition seeds, so this does not isolate row count. |
| K6: This establishes high-quality end-to-end reconstruction using only 2D fields. | retire | A fails adequacy; B fails paired gates; RGB/mask-derived initialization was shared. The broader capability question remains unresolved, and D/E are unavailable. |
| K7: This tests medical-style physical-density tomography or proves Gaussian fields cannot reconstruct 3D. | retire | Targets are view appearance under the specified rendering model; no physical-density or ground-truth geometry test. |
| K8: The B silhouette deficit at alpha >= 0.5 is dominated by excess opacity outside the fixed mask. | narrow | B precision 0.747409, recall 0.996096, FP/GT 0.343796; A precision 0.972716, recall 0.983525, FP/GT 0.027713. Consistent visible halos do not identify their cause. |
| K9: Successful-cell timing or memory establishes a performance advantage or equal total compute. | retire | Contended GPU, two interrupted attempts and at least 500 discarded updates; exact failed-attempt cost is unknown. |
| K10: The saved representative B model renders and responds to camera movement in the tested browser. | narrow | Finite 25088-row model, WebGL2 renderer ready, before/after camera and framebuffer changes. Known Viser quad warning retained; tested Chromium/SwiftShader only. |

## Limits and pending handoff

- Previously exposed single capture, 22 training and 4 reporting-only views, three refinement seeds, downscale 8; not confirmatory generalization or full-resolution evidence.
- All seeds use the same 8001-point geometry/attribute multiset, with deterministic row permutations. B/C consume RGB/mask-derived initialization and train-only bounds.
- Nine training masks touch image edges. The all-view in-frame hull may omit geometry outside any camera; the failed A gate does not isolate initialization versus optimizer quality.
- C is an acquisition-family comparison, not an isolated row-count or byte-cap ablation.
- The worker did not retain per-target tensor-load hashes. Cached target bytes first hashed by the reviewer after preparation remain unchanged; source guards and read-boundary instrumentation are scoped evidence, not tamper-proof isolation.
- Teacher LPIPS forwards were not independently repeated; all 22 raw-view records and qualification aggregation were checked. Final heldout LPIPS was independently rerendered for all 36 rows.
- Two SIGTERM-consistent interruptions have unconfirmed causes. The final cell restarted once from the original initialization and full schedule; at least 500 updates were discarded, with exact count and failed-attempt resource costs unknown.
- The audit initially tried an algebraically equivalent ideal quadrature. Float32 sampling ties changed 2 and 4 mask pixels in two views; that attempt stopped at an assertion. Preserved final audit uses the frozen operator and records the numerical discrepancy without changing official scores or gates.
- Visual review inspected all nine contact sheets and six fixed frames from each of 18 orbit/elevation animations, not every animation frame or physical geometry truth.
- Runtime/VRAM fields cover successful fitting cells only under contention. No total-compute equality, throughput, speedup or hardware-WebGL performance claim.
- Final report rendering, report browser/link checks, manifest and completed-bundle validation remain the Driver delivery stage; this audit accepts execution/results only.

Actual audit commands and all immutable evidence hashes are recorded in the companion JSON. The scripts under the canonical run audit_checks/ preserve the computations. Final GPU work was limited to fixed-model metric rendering; no fitting or conditional experiment was rerun. Full host verification and selected CUDA parity tests were Driver prelaunch evidence. The Driver must finish report rendering, browser/link checks, manifest validation and bounded docs/ARA recording before terminal task acceptance.

Canonical machine audit: `20260908_field_teacher_information_stage_frame00008_AUDIT.json` (SHA-256 `61e791a053e3c0f086b160f3f6aed284c991bb3add3bb3728a45da12187fa6dd`). Numeric producer record: `20260908_field_teacher_information_stage_frame00008_RESULT.json`; raw comparison and replay evidence: `runs/20260908_field_teacher_information_stage_frame00008/`.
