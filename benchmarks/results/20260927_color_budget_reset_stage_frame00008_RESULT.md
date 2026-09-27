# Training budget and intended opacity reset for field-only distillation colour quality on Janelle frame 00008

18 paired development cells completed. Held-out foreground PSNR inside the mask, field-only: 23.489 dB at 8000 steps, 22.769 dB at 30000 steps, 22.731 dB at 30000 steps with the intended reset; photographs 23.878 / 23.261 dB.

H1 longer budget (30000 vs 8000 steps, field-only): reject. H2 intended opacity reset at 30000 steps: inconclusive. Visual adequacy and any promotion require the independent audit; no default changes.

| Condition | foreground_psnr | crop_lpips | outside_alpha_mass | floater_fraction | interior_alpha |
|---|---:|---:|---:|---:|---:|
| nb_8k_up | 23.488748 | 0.145397 | 0.001135 | 0.000834 | 0.999002 |
| nb_8k_rs | 23.552471 | 0.145665 | 0.001102 | 0.000849 | 0.999054 |
| nb_30k_up | 22.768617 | 0.165498 | 0.000927 | 0.000789 | 0.999061 |
| nb_30k_rs | 22.731253 | 0.166686 | 0.000942 | 0.000817 | 0.998985 |
| ph_8k_up | 23.878357 | 0.129246 | 0.001095 | 0.000845 | 0.999033 |
| ph_30k_rs | 23.260886 | 0.140215 | 0.000881 | 0.000785 | 0.998997 |

Development screen on one previously outcome-exposed frame (RTGS-021/024/025/026 used it), the RTGS-025 split (22 training, 4 held-out views), three paired seeds, downscale 8. The 30000-step budget also extends densification to step 15000 (the 3DGS convention); 8000-step cells keep densification to 6000, as in RTGS-025/026. The intended reset is the RTGS-027 opt-in callback on gsplat's own iteration clock; upstream gsplat 1.5.3 never resets. Training packed alpha is mask-derived; held-out masks are evaluation-only, so alpha metrics are out-of-sample. Targets are decoded with the exact CPU tile index. Timings on a possibly shared local GPU are descriptive. No default, SOTA, generalization, physical-geometry or speed claim. Before protocol review, a non-protocol driver smoke (60/120 iterations, reset period 20, two training views standing in for held-out, no real held-out access) exposed training-view teacher fidelity, reset events and a meaningless tiny-budget gate value to the Driver; no threshold changed afterwards.

Numeric gates precede the independent audit and do not assert visual adequacy. Raw paired values and source lock: `benchmarks/results/20260927_color_budget_reset_stage_frame00008_RESULT.json`. Report: `runs/20260927_color_budget_reset_stage_frame00008/index.html`.
