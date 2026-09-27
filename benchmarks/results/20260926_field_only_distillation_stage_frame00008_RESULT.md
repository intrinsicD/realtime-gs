# Field-only photometric distillation with mask-restricted color and silhouette supervision on Janelle frame 00008

21 paired development cells completed. Held-out foreground PSNR inside the mask: field-only 23.492 dB, photographs 23.831 dB, mask-contained teacher 23.350 dB.

The photograph reference failed its frozen adequacy gate; H1 and H2 are inconclusive and the trainer/initializer must be resolved first.

| Condition | foreground_psnr | crop_lpips | outside_alpha_mass | floater_fraction | interior_alpha |
|---|---:|---:|---:|---:|---:|
| nb_rand_ms | 23.491974 | 0.145762 | 0.001138 | 0.000829 | 0.999105 |
| nb_hull_ms | 23.488124 | 0.145750 | 0.001020 | 0.000844 | 0.999300 |
| nb_rand_pm | 22.067391 | 0.170215 | 0.000436 | 0.000408 | 0.981121 |
| mc_rand_ms | 23.350365 | 0.140380 | 0.001003 | 0.000773 | 0.999173 |
| gi_rand_ms | 23.618332 | 0.131946 | 0.001016 | 0.000776 | 0.999165 |
| ph_rand_ms | 23.831118 | 0.128898 | 0.001078 | 0.000816 | 0.999110 |
| ph_hull_ms | 23.921834 | 0.124561 | 0.000938 | 0.000774 | 0.999343 |

Development screen on one previously outcome-exposed frame (RTGS-021/024 used it), one train/held-out split, three paired seeds and one fixed 8000-step budget at downscale 8. Packed training alpha is mask-derived and is used for supervision and the hull; this is field-plus-silhouette, not strictly Gaussian-only reconstruction. Held-out photographs and masks are evaluation-only. no_boundary retains 5000-8592 Gaussians per view versus 11000 for mask_contained and gaussianimage, and uses a different topology schedule outcome, so H2 compares teacher families, not containment alone. No default, SOTA, generalization, physical-geometry or speed claim. Before protocol review, a non-protocol driver smoke on training views only exposed training-view teacher fidelity to the Driver; no reconstruction or held-out outcome was observed and no threshold changed afterwards.

Numeric gates precede the independent audit and do not assert visual adequacy. Raw paired values and source lock: `benchmarks/results/20260926_field_only_distillation_stage_frame00008_RESULT.json`. Report: `runs/20260926_field_only_distillation_stage_frame00008/index.html`.
