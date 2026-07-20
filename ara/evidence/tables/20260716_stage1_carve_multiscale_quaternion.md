# 2026-07-16 representation, allocation, scale, and coordinate evidence

All entries below are CPU synthetic evidence under their own frozen protocols. They do not
establish real-scene, CUDA/gsplat, runtime, memory, or production-default claims.

## Carve exact-count materiality

| Seed | Raw count | Moment count | Compression | Multi-member cells | Exposed raw mass |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1156 | 1125 | 2.68% | 29 | 5.19% |
| 1 | 1160 | 1129 | 2.67% | 31 | 5.34% |
| 2 | 1155 | 1128 | 2.34% | 27 | 4.68% |

Every seed missed the frozen 50-cell, 15%-exposure, and 10%-compression floors. Construction and
parity checks passed, but Phase B was forbidden. This says the production grid did not create a
material equal-count intervention; it does not compare merge utility with pruning.

- Result: `benchmarks/results/20260715T232244Z_cpu_carve_merge_controls_iter2_audit.json`,
  SHA-256 `1e1142b4a4301b7f05546f62d5868c64e976183b549dd305775fca43753a29cc`.
- Audit: `benchmarks/results/20260715T232244Z_cpu_carve_merge_controls_iter2_audit_AUDIT.md`,
  SHA-256 `190a43465ac1108a7f4964766ac32e7b7cb890ff5df15486cac937cf66fd2d74`.
- Exploration: N71-N72; claim C14.

## Stage-1 weight/color gauge contract

| Representative | Maximum source RGB error | Coverage delta / reference | Depth render delta / signal | Carve render delta / signal |
| --- | ---: | ---: | ---: | ---: |
| Unit weight | 1.7881393e-7 | 0.520168 | 0.581622 | 0.589632 |
| Peak color | 1.1920929e-7 | 0.705005 | 2.022173 | 1.077597 |

All 54 transformed source renders passed equivalence and all 4050 components changed both weight
and color. Both representatives passed every frozen downstream materiality gate in 3/3 seeds and
the raw-sum pool. The scientist verdict is qualified because decision-grade reductions and hashes,
not raw tensors, are retained. The evidence identifies gauge-dependent interface semantics but
does not choose a representative or show held-out benefit.

- Result: `benchmarks/results/20260716T003140Z_cpu_stage1_weight_gauge_audit.json`, SHA-256
  `e001d6efdfcf0beea30ae578069d6057350e47b3f3516ad95f216ae495793791`.
- Audit: `benchmarks/results/20260716T003140Z_cpu_stage1_weight_gauge_audit_AUDIT.md`, SHA-256
  `871c3235954f1025b05641385d70cd33c6160d200f74a26fb322dc20e390dfd6`.
- Exploration: N76; claim C15; architecture A08.

## Fixed-topology 24-to-48 multiscale refinement

| Candidate | Mean foreground-PSNR AUC delta | Mean final PSNR delta | Seed AUC wins | Raster-pixel exposure |
| --- | ---: | ---: | ---: | ---: |
| Camera blocked | -0.338645 dB | -0.263247 dB | 0/3 | 62.5% |
| Loss-pyramid blocked | -0.088758 dB | -0.203262 dB | 0/3 | 100% |
| Camera interleaved | -0.345927 dB | -0.734998 dB | 0/3 | 62.5% |

All candidates lost AUC and final PSNR in every seed. Reduced raster-pixel exposure is accounting,
not speed, and both camera arms failed quality noninferiority. Blocked-minus-interleaved mean AUC
was +0.007282 dB, below the attribution gate. The literal 12-to-24-to-48 N68 proposal was not run.

- Result: `benchmarks/results/20260716T003735Z_cpu_multiscale_refinement.json`, SHA-256
  `343263f3193871dbdae4f390d46ba9c305cb9c38bfead0dd5c7bc97448ce35fa`.
- Audit: `benchmarks/results/20260716T003735Z_cpu_multiscale_refinement_AUDIT.md`, SHA-256
  `c736a0de3160f61f8b1df9113783576fab2f706d100687df34c0bac1a06cd394`.
- Exploration: N73-N75; claim C16.

## Quaternion radial-gauge attempts

Neither attempt retained an optimizer outcome. The first failed on inconsistent diagnostic
arithmetic order. Retry-2 repaired that path, then failed the inherited construction contract:

| Seed | Direction delta | Covariance max error | Relative covariance error | Gate |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.0770374703e-8 | 6.1272389593e-10 | 4.4032617238e-10 | 2e-12 |
| 1 | 2.0618824148e-8 | 2.0513717704e-9 | 1.2022659046e-9 | 2e-12 |
| 2 | 1.5078977900e-8 | 9.5304763560e-10 | 7.2764233996e-10 | 2e-12 |

Native float32 canonicalization is not direction-idempotent under a second float64 normalization,
so every seed/scale necessarily failed the covariance gate. Both invalid artifacts strip arms,
trajectories, checkpoints, AUC, and materiality. Phase B never ran; no quaternion policy or default
claim is authorized.

- First invalid audit: `benchmarks/results/20260716T015517Z_cpu_quaternion_gauge_invalid_AUDIT.md`,
  SHA-256 `7528d22e0daa909f8f67e8d73b0269de5f9b4bf21b1677a0d2341361be1ecd8d`.
- Retry-2 invalid result: `benchmarks/results/20260716T030759Z_cpu_quaternion_gauge_iter2_invalid.json`,
  SHA-256 `56df44d380ede52dba568b068685d9ffd1dbd625fe9ef92e8f31559660e0af0b`.
- Retry-2 audit: `benchmarks/results/20260716T030759Z_cpu_quaternion_gauge_iter2_invalid_AUDIT.md`,
  SHA-256 `b4492303d9dd688e1685eb886c90cbf94ceeefc698c48bb38667ea1cfd57d866`.
- Exploration: N80-N82; constraint R14. No claim node is created.

