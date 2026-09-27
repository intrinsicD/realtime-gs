# All-mask silhouette-hull floater relocation for field-only distillation on Janelle frame 00008 (exact CPU-index decoding retry)

12 paired development cells completed. Held-out foreground PSNR inside the mask: field-only 23.158 dB without and 23.221 dB with relocation; photographs 23.718 / 23.699 dB.

H1 all-mask floater relocation versus the masked objective alone: inconclusive. No production model is trained; the frozen rule is not met. Visual adequacy and any promotion require the independent audit; no default changes.

| Condition | foreground_psnr | crop_lpips | outside_alpha_mass | floater_fraction | interior_alpha | hull_rejected_fraction |
|---|---:|---:|---:|---:|---:|---:|
| nb_ms | 23.157871 | 0.167610 | 0.001782 | 0.001470 | 0.998727 | 0.054677 |
| nb_ms_reloc | 23.220908 | 0.164909 | 0.001314 | 0.001026 | 0.998808 | 0.001825 |
| ph_ms | 23.717539 | 0.142658 | 0.001695 | 0.001403 | 0.998775 | 0.044078 |
| ph_ms_reloc | 23.698802 | 0.141619 | 0.001368 | 0.001077 | 0.998893 | 0.001171 |

Development screen on one previously outcome-exposed frame (RTGS-021/024/025 used it), one split with two held-out colour views, three paired seeds and one 8000-step budget at downscale 8. The masks of all 26 views, including both held-out views, define the visual hull by the user's design, so held-out alpha, floater and hull-rejection metrics are in-sample; only held-out colour is a novel-view measurement. Training packed alpha is mask-derived; the field arms are field-plus-silhouette and the photograph arms photograph-plus-silhouette. No default, SOTA, generalization, physical-geometry or speed claim. The conditional all-view production model has no held-out view and is not evidence. Relocation also re-places the random initialization: most random start points lie outside the hull and move to its surface at the first event, so H1 measures the method as a whole (hull-guided placement plus ongoing relocation), not ongoing relocation alone. Before protocol review, a non-protocol driver smoke (60 iterations, two training views standing in for held-out, no real held-out colour) exposed relocation counts, hull size and a meaningless 60-iteration gate value to the Driver; no threshold changed afterwards. This task repeats the approved design of 20260927_silhouette_relocation_stage_frame00008, whose single run failed closed at its CUDA-versus-CPU decoder-parity gate (1.889e-4 at 1 of 512 sites of training view C0026, a near-empty normalized-blend region) before any initialization, fitting cell or held-out access; the only design change is exact CPU-index decoding. The Driver diagnosed that site after the failure; no reconstruction outcome exists from the predecessor.

Numeric gates precede the independent audit and do not assert visual adequacy. Raw paired values and source lock: `benchmarks/results/20260927_silhouette_relocation_index_decode_stage_frame00008_RESULT.json`. Report: `runs/20260927_silhouette_relocation_index_decode_stage_frame00008/index.html`.
