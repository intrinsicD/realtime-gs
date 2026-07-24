# Pooled structure-tensor WSE ablation — frozen protocol V3

Date frozen: 2026-07-24 (Europe/Berlin)

This source-only amendment incorporates
[`20260724_pool_structure_wse_frame00008_PREREG_V2.md`](20260724_pool_structure_wse_frame00008_PREREG_V2.md),
SHA-256 `e9de68c509d5b0ee21eddf08e55180834f357713351d23a46bc1172ce9fff4f6`,
which incorporates V1.

The V2 official command failed at the harness's first import with
`ModuleNotFoundError: No module named 'benchmarks'`. The output directory was not created, the
scene was not loaded, no arm started, and no calibrated outcome was seen. The append-only receipt
is
[`20260724_pool_structure_wse_frame00008_ATTEMPT_V2_FAILURE.json`](20260724_pool_structure_wse_frame00008_ATTEMPT_V2_FAILURE.json),
SHA-256 `c1f41b750ec77decc2ff276cfea4978b959a5e3a24eac91393e8ac7bec783d6d`.

V3 changes only:

1. add an import fallback for direct `python benchmarks/<script>.py` execution, where Python puts
   `benchmarks/` rather than the repository root on `sys.path`;
2. change the harness's default protocol path from V2 to this V3 file.

Every question, arm, implementation treatment, input, split role, seed, count, fit/lift/refinement
setting, metric, threshold, gate, artifact, interpretation restriction, and output namespace from
V1 remains frozen unchanged.

Corrected harness:
`benchmarks/pool_structure_wse_frame00008.py`,
SHA-256 `9daef81cfcfdb12d2cc3afa786ffb5f798d6492660284250ec09a2ba8ad5efd1`.

Official command:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv-cuda/bin/python benchmarks/pool_structure_wse_frame00008.py \
  --protocol benchmarks/results/20260724_pool_structure_wse_frame00008_PREREG_V3.md \
  --out runs/pool_structure_wse_frame00008_20260724
```
