# Separate 2D target information from 3D reconstruction initialization

Nine paired development cells completed. Heldout foreground PSNR: photographs 24.782 dB, 100k fields 24.316 dB, low-budget fields 18.553 dB.

Photograph baseline failed the frozen adequacy gate; resolve baseline geometry/solver quality before interpreting field-only viability.

| Arm | Foreground PSNR | Full PSNR | Boundary PSNR | Crop LPIPS | Alpha IoU |
|---|---:|---:|---:|---:|---:|
| rgb | 24.782436 | 35.811775 | 20.784817 | 0.114609 | 0.957174 |
| field_high | 24.316375 | 32.889365 | 19.278671 | 0.173856 | 0.745162 |
| field_low | 18.552894 | 22.703759 | 11.516209 | 0.582649 | 0.494741 |

Development diagnostic on one previously exposed calibrated capture at downscale eight. A/B/C share an RGB/mask-derived initialization; B/C are field-supervised refinement, not end-to-end field-only reconstruction. No physical-density, generalization, full-resolution, original-COLMAP-3DGS, speed or VRAM advantage claim. Failed A or teacher qualification leaves the high-quality field-only question unresolved.

Numeric gate computation precedes independent audit; it does not assert visual adequacy. Raw paired values and source lock: `benchmarks/results/20260908_field_teacher_information_stage_frame00008_RESULT.json`. Reproduce argv is bound in that record. Report: `runs/20260908_field_teacher_information_stage_frame00008/index.html`.
