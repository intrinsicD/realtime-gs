# Confidence-weighted bounded-ray anchor ablation

Official CPU artifact: `benchmarks/results/20260714T224800Z_cpu_depth_anchor.json`.
Preregistered protocol: `benchmarks/results/20260715_depth_anchor_PREREG.md`.
Post-run attribution audit: `benchmarks/results/20260715_depth_anchor_AUDIT.md`.

## Protocol

Seeds 0/1/2 used 40-Gaussian synthetic scenes, twelve 48x48 cameras, nine training views,
three held-out views, shared 150-Gaussian/view fits, 60 merge-free bounded-ray steps, and
60 no-density refinement steps for corrupted priors. Arms were step-0-identical legacy,
normalized, continuous-confidence, and thresholded anchors. Corrupted 8x8 blocks multiplied
depth by 1.2 or 0.8 and assigned confidence 0.05; a pixel-shuffled condition was also run.

## Primary corrupted-prior result

| arm | init PSNR | init SSIM | low-confidence p90 | final PSNR | final SSIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| legacy | 19.689 | 0.5863 | 0.2066 | 24.526 | 0.8711 |
| normalized | 19.675 | 0.6050 | 0.2081 | 24.471 | 0.8729 |
| confidence | 19.631 | 0.6149 | 0.2100 | 24.484 | 0.8768 |
| thresholded | 19.626 | 0.6152 | 0.2087 | 24.481 | 0.8761 |

Confidence versus legacy changed initialization PSNR by -0.0577 dB, won one of three seeds,
worsened low-confidence p90 by 1.63%, and changed final PSNR by -0.0417 dB. Only the clean
regression guard passed (-0.0900 dB); the preregistered primary hypothesis failed.

## Attribution audit

Confidence reduced mean held-out depth RMSE versus normalized by 0.394% under calibrated
corruption, while pixel-shuffled confidence worsened it by 0.929%. This direction cannot be
treated as causal attribution: normalized includes invalid-prior fallback anchors that confidence
excludes, and shuffling pixels before bilinear sampling changed the optimized-ray weight
distribution. Low-confidence retained-ray counts changed from 261/236/233 by seed to
104/131/116 after shuffling. The primary failure is independent of this flawed control.

## Decision

Retain `legacy` as default and keep the new modes as opt-in controls. The narrow repair is a
valid-prior-uniform comparison plus a permutation of already sampled retained-ray weights that
preserves the exact per-view multiset. Production confidence requires a train-derived estimator
and real calibrated monocular-depth validation.
