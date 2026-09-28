# Field-only distillation improvement portfolio (SH degree, regularization, budget factors, resolution, view count) on Janelle frame 00008

27 paired development cells completed. Held-out foreground PSNR inside the mask: nb_base 22.099 dB, nb_sh1 22.453 dB, nb_sh0 22.563 dB, nb_reg 23.092 dB, nb_30k_d6 21.169 dB, nb_30k_d6_lr 21.541 dB, nb_ds4 24.040 dB, nb_v11 21.243 dB, ph_base 22.197 dB.

Per-arm verdicts against nb_base (nb_sh1: pass, nb_sh0: pass, nb_reg: pass, nb_30k_d6: reject, nb_30k_d6_lr: reject, nb_ds4: pass). This is a screen of six simultaneous comparisons; any pass needs a separately registered confirmation. No default changes.

| Condition | foreground_psnr | crop_lpips | outside_alpha_mass | floater_fraction | interior_alpha |
|---|---:|---:|---:|---:|---:|
| nb_base | 22.099386 | 0.171175 | 0.000928 | 0.000689 | 0.998175 |
| nb_sh1 | 22.453205 | 0.165942 | 0.001140 | 0.000729 | 0.998332 |
| nb_sh0 | 22.563181 | 0.163769 | 0.001251 | 0.000726 | 0.998476 |
| nb_reg | 23.091652 | 0.149513 | 0.001303 | 0.000838 | 0.994045 |
| nb_30k_d6 | 21.168798 | 0.198008 | 0.000724 | 0.000625 | 0.997489 |
| nb_30k_d6_lr | 21.541142 | 0.187843 | 0.000831 | 0.000697 | 0.998265 |
| nb_ds4 | 24.039986 | 0.130982 | 0.000944 | 0.000781 | 0.999186 |
| nb_v11 | 21.242834 | 0.191648 | 0.001235 | 0.000946 | 0.997481 |
| ph_base | 22.197498 | 0.153528 | 0.000933 | 0.000726 | 0.998387 |

Development screen of six simultaneous treatment comparisons on one previously outcome-exposed frame (RTGS-021/024/025/026/028 used it), the RTGS-025 split (22 training / 4 held-out views), three paired seeds. A pass is a screening signal that needs a separately registered confirmation. The frame has no further calibrated photographs, so more views cannot be tested directly; the half-view arm only estimates view-count sensitivity. Every model is scored under two operators: primary, a downscale-4 render box-averaged to the downscale-8 grid (its pixel centres coincide with the reference quadrature sites in position only); second, the point-sampled downscale-8 render used in RTGS-025/026/028. gsplat's classic-mode 2D dilation of 0.3 px^2 acts in the render's own pixel units, so only nb_ds4 is scored with its training render under the primary operator; nb_ds4 is therefore gated under both operators. Primary-operator numbers are not comparable to earlier tasks. Training packed alpha is mask-derived; held-out masks are evaluation-only, so alpha metrics are out-of-sample. Targets are decoded with the exact CPU tile index. Upstream gsplat 1.5.3 performs no opacity reset in any arm. Timings are descriptive. No default, SOTA, generalization, physical-geometry or speed claim. Before protocol review, a non-protocol driver smoke (60/120 iterations, two training views standing in for held-out, no real held-out access) exposed training-view teacher fidelity and meaningless tiny-budget values to the Driver; no threshold changed afterwards.

Numeric gates precede the independent audit and do not assert visual adequacy. Raw paired values and source lock: `benchmarks/results/20260928_field_only_portfolio_stage_frame00008_RESULT.json`. Report: `runs/20260928_field_only_portfolio_stage_frame00008/index.html`.
