# Full-resolution Stage-1 teacher fidelity diagnostic

This post-failure, non-decision-bearing diagnostic isolates the seven frozen StructSplat
observation fields from the later 2D-to-3D lift and compact optimizer. Source RGB and masks are
used only for Stage-1 evaluation; they do not enter initialization or refinement.

| Quantity | Result |
|---|---:|
| views | 7 |
| native resolution | 5328x4608 |
| pixels per view | 24,551,424 |
| Gaussians per teacher | 640 |
| pixels per Gaussian | 38,361.6 |
| mean full-image PSNR | 19.485666 dB |
| full-image PSNR range | 17.731439--21.734425 dB |
| mean foreground PSNR | 18.826633 dB |
| foreground PSNR range | 17.446824--19.723822 dB |
| mean foreground pixel fraction | 6.0882% |
| mean component centers on foreground | 9.1071% |
| mean exact nonzero-render support | 97.3810% |
| mean foreground nonzero-render support | 95.4323% |
| mean median / p90 / p99 scale | 41.900 / 142.174 / 302.846 px |
| worst random point-query/CUDA-raster absolute error | 2.772e-6 |
| worst reproduced acquisition-PSNR discrepancy | 3.104e-6 dB |

The archive-to-render comparison finds no crop-offset, half-pixel, RS covariance, amplitude, or
color-preservation error. All fields were intentionally fitted over the full native canvas with
`mask=None`; their `fit_window` is `[0,0,5328,4608]`. The visible broad blur and unsupported black
holes are already present in the 2D teachers. Only roughly 38--78 of 640 component centers per
view lie on the foreground mask. This diagnoses the current 640-Gaussian, 100-step, unmasked
full-frame acquisition as an insufficient teacher-quality regime; it does not establish the
quality ceiling of StructSplat, a masked/cropped fit, a larger variable `N_opt^2D`, or another 2D
Gaussian producer.

Exact artifacts:

- `runs/compact_point_training_20260716/stage1_teacher_audit/all_views_metrics.json`
  (`sha256=3ba10e81495257a6b7679b83a70c358c2e8cb6b41627af293dba907f8c63cfc8`)
- `runs/compact_point_training_20260716/stage1_teacher_audit/all_views_target_vs_teacher.png`
  (`sha256=ecddf9d1c1ee8c67c330553e02f05f6fe57554a63be5a5c838139e87f83fa76d`)
- `runs/compact_point_training_20260716/stage1_teacher_audit/C0001_metrics.json`
  (`sha256=c5fa37354da953d62d8e69e2b29a0a31bc3d9ad2afd57fb8e4aaa59222157e8f`)
- `runs/compact_point_training_20260716/stage1_teacher_audit/C0001_teacher_audit_panel.png`
  (`sha256=2cf5cb32e19487d3f8a7f12957ae986a8369e59468b784e8d2b55122a0aa3374`)

