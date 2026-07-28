# Preregistration V2 — all-26-view Beam carrier maturation

**Frozen:** 2026-07-27, before Beam Fusion or any model-quality metric was computed  
**Status:** execution-only correction to the V1 protocol  
**Output:** `runs/carrier_maturation_all26_frame00008_20260727/`

This document incorporates the complete frozen protocol in
`20260727_carrier_maturation_all26_PREREG.md`, SHA-256
`ba0723230267d761ebb2e91ae0d516616228e212c96ef4b0460f094186b2f2d4`.

## Reason for V2

The first launch loaded and source-hash-verified the 26 compact captures and native RGB/masks, then
stopped while preparing compact evaluation targets:

```text
ModuleNotFoundError: No module named 'benchmarks'
```

It stopped before Beam Fusion, carrier repair, model rendering, training, or any model-quality
metric. The preserved pre-outcome receipt is
`runs/carrier_maturation_all26_frame00008_20260727_failed_preoutcome_v1/`.

The only correction inserts the repository root into `sys.path` immediately before importing the
existing deterministic compact-target replay helper. It does not change any input, model,
algorithm, hyperparameter, stopping rule, metric, analysis rule, artifact, or claim boundary.

## V2 implementation binding

- corrected `benchmarks/carrier_maturation_all26.py`:
  `e2e1710038029462a50de55217429db6813131c7f483593a29436ac4c928e545`
- all other hashes remain exactly those frozen in V1:
  - Beam Fusion:
    `575c12fdb59ad7a430178ed5899eb9d546cddc965f50617eeed0b40fe9ca2e12`
  - carrier repair:
    `bbd50a727da29663e22415376c8cabf269bf561e929a76ae86a4360dbb5590f5`
  - carrier schedule:
    `97c089c683ba0f86bafa302f674613a34d222b8faca5d697a6366f9eb1a8d17c`
  - density:
    `a18be4d1b425177b74db8fb4ef814e53f4b246450d3e8f5f149962328110680a`
  - trainer:
    `228a5269d02fe33e8d5981c1fd83ee79211b24d485d10bd6be59b13ff2432fed`
  - visualization:
    `0cd822b475fe7a8cf4e8738d28d0ac27f1c06b4bb7cd30f5592b4eff2c5b63d7`
  - compact replay:
    `9ee2688f5d4f18c46790cd572118503bb11d83b86d821ffe749930b1eb8be722`
  - ADR-002:
    `081e0fca8cd64829953e92e59047dca48ef3b3df12bca00b9162fb2c4844d027`

## Frozen V2 command

```bash
.venv-cuda/bin/python benchmarks/carrier_maturation_all26.py \
  --out runs/carrier_maturation_all26_frame00008_20260727 \
  --protocol benchmarks/results/20260727_carrier_maturation_all26_PREREG_V2.md \
  --raw-frame /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008
```
