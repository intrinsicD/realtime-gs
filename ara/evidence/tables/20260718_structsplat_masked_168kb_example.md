# Mask-gated StructSplat C0014 feasibility evidence

Date: 2026-07-18

| Measure | Value | Scope |
|---|---:|---|
| Exact teacher bytes | 150,492 | One C0014 archive |
| Margin below decimal cap | 17,508 | Cap = 168,000 bytes |
| Gaussian count | 5,000 | Fixed count |
| Foreground PSNR | 36.878836 dB | Clamped RGB, source mask weights |
| Foreground weighted SSIM | 0.9019588 | Source mask weights |
| Raw render vs masked-crop PSNR | 17.875642 dB | No mask compositing |
| Raw render vs masked-crop SSIM | 0.7297999 | No mask compositing |
| Rounded centers outside mask | 0 | Strict-reloaded archive |
| Finite-support mask IoU | 0.6032109 | Weight threshold > 1e-8 |
| Outside pixels above 1/255 | 31.1769% | Tight-crop background |
| Live/archive maximum error | 0 | Recorded terminal vs strict reload |
| zlib-compressed bit mask | 7,226 bytes | Post-run diagnostic only |

Forensic sources:

- `runs/structsplat_masked_168kb_example_20260718/result.json`
- `benchmarks/results/20260718_structsplat_masked_168kb_example_RESULT.md`
- `benchmarks/results/20260718_structsplat_masked_168kb_example_AUDIT.md`

Disposition: one-view archive-integrity and cap feasibility confirmed; exact mask-free silhouette,
subjective visual acceptance, production packaging, and whole-dataset conversion remain untested.
