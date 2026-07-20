# Uniform depth-anchor baseline

Read-only CPU probes run on 2026-07-15 with CUDA hidden and four PyTorch threads. The full-scene
probe began from tracked `GradientLifter` source SHA-256
`f0d88462380cdc2a1da99d613109f71aad199addd3b6c1375b6666f3d3b5dfcb`, before concurrent
confidence-mode work appeared in the shared worktree. The new working-tree implementation retained
`legacy` as its default, and a direct post-change unit check confirmed that this default gives the
same anchor loss for confidence 1.0 and 0.01.

## Causal confidence check

Setup: seed 0, 25 GT Gaussians, eight 32x32 views (six train/two held out), 120 fitted
Gaussians/view for 120 steps, 40 bounded-ray steps, depth jitter 0.02, anchor lambda 0.01, and
merging disabled. Valid GT depth was multiplied by 1.12; valid pixels were assigned either
confidence 1.0 or 0.01. Of 701 retained observations, 389 (55.49%) consumed a valid corrupted
prior, with median absolute corruption 0.278 world units.

High- and low-confidence corrupted priors produced bit-identical pre-merge results:

- maximum mean difference: 0
- maximum covariance difference: 0
- maximum difference over all 40 loss samples: 0
- the same 12-step check through `HybridLifter` also had zero mean/history difference

Thus confidence reached only merge weights in the measured implementation; it did not modulate the
uniform bounded-ray anchor.

## Seed-0 diagnostic

| arm | held-out PSNR | SSIM | median GT distance | median move from corrupt init |
| --- | ---: | ---: | ---: | ---: |
| clean, uniform anchor | 21.030 | 0.7152 | 0.2017 | 0.0448 |
| corrupt high confidence | 20.842 | 0.7028 | 0.2222 | 0.1245 |
| corrupt low confidence | 20.842 | 0.7028 | 0.2222 | 0.1245 |
| corrupt low, no-anchor proxy | 21.066 | 0.7243 | 0.2189 | 0.2175 |

The no-anchor arm is a mechanism probe, not a proposed default.

## Three-seed replication

The same 25-Gaussian/eight-view setup used seeds 0/1/2. Corrupted-prior initialization started at
19.632 ± 0.566 dB held-out PSNR. After 40 bounded-ray steps:

| metric | uniform anchor | no-anchor proxy | proxy minus uniform |
| --- | ---: | ---: | ---: |
| held-out PSNR | 20.954 ± 0.402 | 21.092 ± 0.348 | +0.138 ± 0.085 dB |
| held-out SSIM | 0.6908 ± 0.0118 | 0.7201 ± 0.0138 | +0.0293 ± 0.0102 |
| median GT distance | 0.2051 ± 0.0162 | 0.2002 ± 0.0165 | -0.0049 ± 0.0094 |
| median move from init | 0.0908 ± 0.0293 | 0.2052 ± 0.0110 | +0.1144 ± 0.0185 |

Every seed improved held-out PSNR without the uniform anchor, but the mean gain is below the
preregistered 0.25 dB threshold for the forthcoming confidence-aware arm.

## Runtime for harness sizing

- Seed-0 causal probe: 6.91 seconds wall, including one fit, six lift arms, and evaluation;
  peak RSS 784,500 KiB.
- Three-seed replication: 18.45 seconds wall; peak RSS 784,508 KiB.
- Per seed, the 120-step six-view fit averaged 2.22 seconds and one 40-step lift arm averaged
  1.48-1.51 seconds under concurrent machine load.
- At this tiny size, four anchor modes across three seeds should remain below one minute when fits
  are reused. The preregistered 12-view/48x48, two-condition, 60-lift/60-refine design will be
  materially larger and should continue caching fits per seed.
