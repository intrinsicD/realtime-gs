# RTGS-021: field-target information screen

Provenance: ai-executed. Development result, independently audited; see C47.

| Input | Foreground PSNR | Full PSNR | Boundary PSNR | Crop LPIPS | Alpha IoU |
|---|---:|---:|---:|---:|---:|
| Photographs | 24.782436 | 35.811775 | 20.784817 | 0.114609 | 0.957174 |
| 100k/view fields | 24.316375 | 32.889365 | 19.278671 | 0.173856 | 0.745162 |
| Low-budget fields | 18.552894 | 22.703759 | 11.516209 | 0.582649 | 0.494741 |

Each seed averages four frozen held-out views; each arm averages three seeds. Inputs are one previously exposed calibrated Stage capture, frame_00008, downscale 8 (666x576), 22 training views and heldout C0001/C0018/C0029/C1002. All arms start from the same 8001-point training-RGB/mask visual-hull geometry; seeds permute its rows. The existing CUDA gsplat Trainer runs 8,000 steps per final cell, with the same loss, schedule and density cap. B/C are field-supervised refinement.

The high teacher passes its mean training-view qualification: foreground PSNR36.455862 dB and crop LPIPS0.047370. Every A seed misses the 25 dB foreground gate while passing its LPIPS/IoU gates and recognizable-subject visual criterion. Every B seed fails paired LPIPS/boundary/IoU margins and visible outline/detail preservation. D/E were not run. C also changes acquisition budget/seed settings, so B-versus-C does not isolate primitive count.

The independent fixed-model rerender agrees within 2.723e-6 dB PSNR, exact LPIPS and 2.9e-8IoU. Its explicitly posthoc alpha>=0.5 analysis finds A precision/recall 0.972716/0.983525 and B 0.747409/0.996096. False-positive area relative to GT foreground is 2.7713% for A and 34.3796% for B. Visible halos and these counts locate excess opacity; they do not identify its cause or establish physical geometry.

Two SIGTERM-consistent interruptions have unconfirmed causes. Complete models were preserved; one partial ninth cell was quarantined and restarted unchanged after independent approval. At least 500 discarded updates and unknown failed-attempt costs are excluded from successful-cell resource counters. Timing is descriptive on a contended RTX3050. No performance, full-resolution, generalization, physical-density, strict field-only or default claim.

Protocol: `20260908_field_teacher_information_stage_frame00008`; digest `b26db0f9b00acba58a86516f95b8cdac218ba5a1067cee253622f4b0e41e11d8`; source envelope `d83422a66c3a2bb6137ce0a4059d94554dbb97a7941cd7f0c8002b65b1134382`. Exact dirty source at commit 8a715051cc789cc525e880a653a45158d65b5ab1 is preserved in the run;119 source/snapshot and 105 input hashes pass.

- Canonical human/machine result: `benchmarks/results/20260908_field_teacher_information_stage_frame00008_RESULT.md` and `_RESULT.json`.
- Canonical independent audit: `benchmarks/results/20260908_field_teacher_information_stage_frame00008_AUDIT.md` and `_AUDIT.json`.
- Raw scores: `runs/20260908_field_teacher_information_stage_frame00008/comparison.json`; independent36-row replay: `runs/20260908_field_teacher_information_stage_frame00008/audit_checks/final_metric_audit.json`.
- Report/models/previews: `runs/20260908_field_teacher_information_stage_frame00008/index.html`; browser receipt: `runs/20260908_field_teacher_information_stage_frame00008/viewer_smoke.json`.
- Final delivery/repository checks: `ara/evidence/tables/20260908_field_teacher_information/verification_receipt.json`.

The shared report, browser link check, visible WebGL scene/orbit and both run/bundle gates passed. The first report browser attempt's three missing protocol links were preserved, then fixed with byte-identical HTTP mirrors. Known nonfatal Viser quad and SwiftShader notices are retained. Final repository check results are recorded separately in the referenced receipt before handoff.
