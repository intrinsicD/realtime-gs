# Depth covariance probes

Both probes ran on git revision `2dddca4` with the pure-PyTorch CPU rasterizer, deterministic
seed 0, shared stage-1 fits within each probe, merging disabled, and no source edits by the
executing sub-agent.

## Tiny held-out probe

Setup: 25 GT Gaussians, 8 cameras at 32x32, train views 0-5, test views 6-7, 120 fitted
Gaussians/view for 120 steps, opacity 0.1, and 40 SH0 refinement steps without densification.

| mode | n | init test PSNR | init test SSIM | final test PSNR | final test SSIM | condition p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| surface | 423 | 20.3144 | 0.4097 | 25.2934 | 0.7888 | 1,488,925.5 |
| footprint | 423 | 21.4551 | 0.5976 | 25.5721 | 0.8939 | 19.46 |
| one-pixel z/f | 423 | 21.3361 | 0.8025 | 25.5666 | 0.9256 | 3.16 |

## Canonical-size global-sigma pilot

Setup: 40 GT Gaussians, 12 cameras at 48x48, train views `[0,1,2,4,5,6,8,9,10]`, test
views `[3,7,11]`, 150 fitted Gaussians/view for 120 steps, init-only. Footprint RMS sigma was
0.14299 (0.09515 of scene extent). The global sigma grid was selected by training metrics only.

| mode | global sigma | train PSNR | test PSNR | test SSIM | condition p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| surface | n/a | 19.7837 | 19.5836 | 0.3707 | 843,277.7 |
| footprint | n/a | 20.6288 | 20.4495 | 0.5876 | 14.93 |
| global 0.5x RMS | 0.07150 | 21.0167 | 20.9494 | 0.7496 | 17.30 |
| global 1x RMS | 0.14299 | 20.6443 | 20.4377 | 0.6145 | 69.21 |
| global 2x RMS | 0.28599 | 18.4843 | 18.1511 | 0.3092 | 276.83 |

## Official three-iteration ablation

Setup: three seeds, 40 GT Gaussians, 12 cameras at 48x48, nine train/three held-out views,
150 fitted Gaussians/view for 120 steps. Isotropic sigma was selected on training views only.
Complete paired samples and configs are stored in `benchmarks/results/`; source provenance and
effective commands are bound by `20260714_depth_covariance_REPLAY.md`.

| protocol | isotropic PSNR | footprint PSNR | surface PSNR | surface p99 condition |
| --- | ---: | ---: | ---: | ---: |
| raw clean initialization | 20.985 | 20.722 | 19.744 | 1,758,524 |
| robust clean initialization | 20.985 | 21.080 | 21.195 | 721 |
| robust clean, 60-step final | 26.206 | 26.251 | 26.040 | n/a |
| perturbed initialization | 20.940 | 20.881 | 21.086 | 647 |
| perturbed, 60-step final | 26.002 | 25.901 | 25.918 | n/a |
| merge+density + 20-step recovery final | 27.047 | 26.485 | 26.280 | n/a |

The 80-step merge+density run is retained as a dead end: density surgery occurred on the final
step, so its collapsed final metrics were rejected for covariance ranking.
