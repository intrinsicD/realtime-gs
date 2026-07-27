# ADR-002 carrier refinement on full-resolution masked Janelle — frozen protocol v3

**Frozen:** 2026-07-27, before the harness entered `main()` or read any Janelle data/outcome
**Supersedes:** v1 and `20260727_carrier_refinement_fullres_PREREG_V2.md`
**Evidence class:** outcome-exposed, single-scene, single-seed development/mechanism evidence

This protocol incorporates every scientific and operational setting from v1
(`38e7e258a36188337c2a926e2d33759b34431556e342e747ceaa45383ddd84aa`) and the
non-scientific direct-execution intent of v2
(`ce0147cb669f467e1e9287b9ee686892ea7291c83be46adb75816d1b06f94dd6`).

## Reason for revision

The v2 smoke again failed during import, before `main()`: the fallback imported
`init_value_masked_screen.py`, whose own absolute sibling import could not resolve under direct
script execution. No compact/raw source was opened, no arm ran, and no outcome was exposed.

V3 removes this runtime benchmark-to-benchmark dependency. It embeds exactly:

- validation IDs `C0031`, `C1000`, `C1002`;
- report-only IDs `C1001`, `C1004`;
- the frozen mechanical split: remove reserved IDs, sort, take
  `train8[i] = pool[(i*n)//8]`, then `train3 = train8[0], train8[3], train8[6]`.

These constants and rule are already stated verbatim in v1. There is no scientific change. All
resolution, seed, initialization, repair, arm, budget, metric, gate, restriction, dependency hash,
and artifact requirements remain exactly those in v1.

## Implementation binding

Harness SHA-256:
`5e8c653a4a690f91df4256f2a29491a14c39e921e6bc71949bde590aeafde144`

All other implementation hashes are inherited unchanged from v1/v2.

## Official command

```bash
PYTHONUNBUFFERED=1 .venv-cuda/bin/python benchmarks/carrier_refinement_fullres.py \
  --protocol benchmarks/results/20260727_carrier_refinement_fullres_PREREG_V3.md \
  --out runs/carrier_refinement_fullres_frame00008_20260727
```
