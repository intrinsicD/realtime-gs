# New opt-in variants on Janelle `frame_00008` — frozen protocol v3

Date frozen: 2026-07-24 (Europe/Berlin)

This document amends and otherwise incorporates protocol v2
[`20260724_new_variants_frame00008_PREREG_V2.md`](20260724_new_variants_frame00008_PREREG_V2.md),
SHA-256 `2230b2f8a0b78918308f6fc850586e616e24447acb333fc002e8c243e0ba4b90`,
which in turn incorporates v1. All arms, inputs, split roles, seeds, fit/lift/training
hyperparameters, metrics, thresholds, interpretation gates, artifacts, and scope restrictions
remain frozen except for the execution-layout correction below.

## Why v3 exists

The complete v2 stage-1 phase ran and saved all 28 fits plus the `C0014` contact sheet. It then
stopped on the first baseline refinement render before optimizer step 1:

- `runs/new_variants_frame00008_20260724_v2/plan.json`, SHA-256
  `e8849d3222d48b60962fbee68f8a759b83686710c69b5b1eceec34aa84bdcc9f`;
- `runs/new_variants_frame00008_20260724_v2/failure.json`, SHA-256
  `7d0841fdd600687755f3fc20ca50b48b6722f56f279db959a5e0bb18583bbac2`;
- failure: installed gsplat 1.5.3 asserted on a packed RGB+D random-background shape
  (`torch.Size([1, 4])`).

All stage-1 numbers are now development outcomes already seen. Their claim gates were frozen in
v1 and are not changed here. No downstream optimizer update, checkpoint result, final model, or
held-out final metric has been produced.

## Frozen correction

Set `TrainConfig.packed=False` and use the same unpacked layout for exact evaluation and visual
renders. Keep:

- gsplat backend;
- antialiasing enabled;
- random backgrounds enabled;
- RGB+D rendering;
- every scientific arm, tensor, loss, optimizer, density setting, iteration, seed, and metric
  unchanged.

Packing is an execution/storage layout seam, not a treatment. A CUDA-synthetic smoke using the
same `gsplat-default + absgrad + random background + unpacked + antialiased` combination completed
three training/evaluation steps successfully before this amendment was frozen.

The v3 run starts from scratch, refits all stage-1 arms, and may not load v2 fits or state. The new
output is `runs/new_variants_frame00008_20260724_v3`.

Corrected harness: `benchmarks/new_variants_frame00008.py`, SHA-256
`d0f429352a28bdb1584cc30ff9b92a7a70b94c168966a19e4785876ea7cc1e8c`.

## Official v3 command

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv-cuda/bin/python benchmarks/new_variants_frame00008.py \
  --protocol benchmarks/results/20260724_new_variants_frame00008_PREREG_V3.md \
  --out runs/new_variants_frame00008_20260724_v3
```

The viewer manifest path and viewer command from v1 remain unchanged; the generated manifest must
point only to v3 artifacts.
