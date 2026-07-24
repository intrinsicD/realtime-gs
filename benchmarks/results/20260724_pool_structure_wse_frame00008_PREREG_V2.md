# Pooled structure-tensor WSE ablation — frozen protocol V2

Date frozen: 2026-07-24 (Europe/Berlin)

This source-only amendment incorporates
[`20260724_pool_structure_wse_frame00008_PREREG.md`](20260724_pool_structure_wse_frame00008_PREREG.md),
SHA-256 `616f3691c90e714270dd9c20daf87d48571e8257db8f150594abc90694a9c03d`.

No calibrated Janelle outcome from any new arm has been rendered, measured, or inspected. The V1
preflight ran only the stated CPU/synthetic mechanism checks and then stopped because Ruff required
the new harness's local `benchmarks` import to be separated from the installed `rtgs` import.

V2 changes only:

1. the import-block whitespace/order required by Ruff;
2. the harness's default protocol path from V1 to this V2 file.

Every question, arm, implementation, input, split role, seed, count, fit/lift/refinement setting,
metric, threshold, gate, artifact, interpretation restriction, and output namespace from V1
remains frozen unchanged.

Corrected harness:
`benchmarks/pool_structure_wse_frame00008.py`,
SHA-256 `726d2765e6feb647e12148cec6c279543a0a69f2e998a1efec93367a0d797ca9`.

Official command:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv-cuda/bin/python benchmarks/pool_structure_wse_frame00008.py \
  --protocol benchmarks/results/20260724_pool_structure_wse_frame00008_PREREG_V2.md \
  --out runs/pool_structure_wse_frame00008_20260724
```
