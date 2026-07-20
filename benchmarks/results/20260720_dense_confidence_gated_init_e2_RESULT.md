# E2 result: easy-only seed plus matched density control

Date: 2026-07-20
Disposition: **easy-only loses the preregistered held-out gate**

## Frozen protocol

Preregistration:
[`20260720_dense_confidence_gated_init_e2_PREREG.md`](20260720_dense_confidence_gated_init_e2_PREREG.md)
(SHA-256
`9a7107a3314f17b514c64d7aa91d656e81535b75fc2f032d795a8547547d9f9e`).

- Train: `C0001, C0008, C0014, C0021, C0026, C0031, C0039`
- Validation/time-to-quality: `C1002`
- Late-release held-out: `C1004`
- Resolution: source RGB/masks at `downscale=8` (`666×576`)
- Schedule: 300 Adam steps, gsplat DefaultStrategy, density every 25
  steps from 25 through 275, hard cap 2,319, validation every 50 steps
- Main seed: `20260720`; top-K repeat seed: `20260721`

All main arms used the same sampled training-view sequence. C1004 was decoded
only after all four optimizer executions completed.

## Held-out result and decision

| Execution | Init→final count | C1004 fg PSNR | C1004 crop SSIM | C1004 alpha IoU | Native time |
|---|---:|---:|---:|---:|---:|
| top-K | 172→178 | 11.2280 dB | 0.629810 | 0.0000 | 2.213 s |
| dense-all | 2,319→2,319 | **14.9079 dB** | **0.786987** | **0.5729** | 2.298 s |
| easy-only | 60→1,229 | 12.7332 dB | 0.691707 | 0.2766 | **2.191 s** |
| top-K repeat | 172→172 | 11.2351 dB | 0.630166 | 0.0000 | 2.229 s |

The top-K control/control envelope is `0.007099 dB`, tighter than the nominal
0.1 dB tolerance. Easy-only is `2.174663 dB` below dense-all, so it fails the
equal-quality condition despite using 1,090 fewer final Gaussians and 0.107 s
less native optimization time.

Preregistered decision:

- quality inside repeat-calibrated band: **false**
- primitive count no greater than dense-all: true
- native time no greater than dense-all: true
- **easy-only wins: false**

No default change is authorized.

## Recovery and trajectories

| Execution | Initial→final C1004 fg PSNR | Initial→final compact-teacher fg PSNR |
|---|---:|---:|
| top-K | +0.0246 dB | +0.0241 dB |
| dense-all | +2.2432 dB | +1.9172 dB |
| easy-only | +1.1671 dB | +1.1927 dB |

Validation foreground-PSNR / primitive-count trajectories:

| Step | top-K | dense-all | easy-only |
|---:|---:|---:|---:|
| 50 | 11.318 / 172 | 13.713 / 2,319 | 11.937 / 60 |
| 100 | 11.297 / 201 | 14.014 / 2,319 | 11.797 / 157 |
| 150 | 11.297 / 205 | 14.376 / 2,319 | 11.735 / 278 |
| 200 | 11.304 / 175 | 14.497 / 2,319 | 11.949 / 537 |
| 250 | 11.306 / 173 | 14.442 / 2,319 | 12.390 / 959 |
| 300 | 11.342 / 178 | 14.838 / 2,319 | 13.033 / 1,229 |

Easy-only was still improving and growing at the final checkpoint. The result
therefore rejects this exact 300-step schedule; it does not prove that a
longer or explicitly budget-filling schedule cannot recover.

## Runtime and artifacts

The sealed harness took `103.50 s` end to end and peaked at `2,566,164 KiB`
RSS. Native optimizer time was only about 2.2 s per arm; repeated exact
full-resolution compact-teacher evaluation dominated total runtime. Peak
reported CUDA allocation was 0.121–0.125 GiB.

After the sealed result, the evaluator gained an immutable teacher/support
cache for repeated candidates. On the same calibrated bundle, preparing all
seven targets took `14.309 s`; two cached top-K evaluations took `7.548 s`
and `7.106 s`. Both repetitions were identical and their aggregate
PSNR/foreground-PSNR/SSIM deltas from the frozen I1 GPU result were exactly
zero. Boolean support masks reduced retained target storage from 803,098,288
to `652,517,359` bytes; the final diagnostic peaked at `2,269,648 KiB` RSS.
Based on the observed uncached 20–21 s candidate evaluations, this should
remove roughly 38–43 s from a four-arm replay. That projection is not an
official E2 rerun or a portable benchmark.

Raw result SHA-256:
`1990a5e9510e83da5a94f5d8684700149e6bba6e77bba9eee0960fef5bf91e32`.

Final PLY SHA-256:

- top-K:
  `ddf2197ccc9684180da2d91e73219fe1cca26c1f27d3c4ea26f9276d4fe002c3`
- dense-all:
  `f90c6b384f8d56acf95d13fd7596a4c115ad77dc67d19eafef88ae3285c3c970`
- easy-only:
  `da7290fb78620802098c751f5304f0b85e0a2cb9b64569266adeadc45fec40e5`
- top-K repeat:
  `f2fe5b265dbc61c635ebea68779fd778883bdac3f8a6f82839e308c4a2a81845`

Reproduction:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
/usr/bin/time -v .venv/bin/python \
  benchmarks/dense_confidence_gated_init_e2.py \
  --out runs/dense_confidence_gated_init_e2_20260720
```

Viewer:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
.venv/bin/rtgs view \
  --gaussians runs/dense_confidence_gated_init_e2_20260720/easy_only/final.ply \
  --initial runs/dense_confidence_gated_init_e2_20260720/easy_only/initial.ply \
  --scene /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008 \
  --downscale 8 --rasterizer gsplat --device cuda \
  --host 127.0.0.1 --port 8775 --no-open
```

The calibrated viewer returned HTTP `200` with `2,888,259` bytes.

## Conditional follow-up

E2 demonstrates an aggregate held-out regression but does not localize it to
the hard-dropped training regions. Under the task's unlock rule, I2/E3 remains
closed. A separately preregistered localization diagnostic or longer
budget-filling control would be needed before attributing the deficit to
missing hard correspondences.
