# ADR-002 carrier refinement on full-resolution masked Janelle — frozen protocol v4

**Frozen:** 2026-07-27 after the first official v3 attempt stopped in result instrumentation
**Supersedes:** v1–v3
**Evidence class:** outcome-exposed, single-scene, single-seed development/mechanism evidence

V4 incorporates every scientific setting, restriction, gate, and artifact requirement from v1
(`38e7e258a36188337c2a926e2d33759b34431556e342e747ceaa45383ddd84aa`) plus the purely
operational import amendments in v2/v3.

## Observed failure and disclosure

The v3 official attempt completed and saved `beam-only`, then trained `beam-standard` for the
frozen 160 updates. It stopped before evaluating/saving that arm because `_run_standard` asked the
lineage tracker for an “initial” snapshot *after* density surgery had already reindexed the tracker
to final rows:

```text
RuntimeError: lineage state and Gaussian row count disagree
```

The console exposed the already-saved `beam-only` development endpoint:

- foreground PSNR 11.541 dB;
- crop SSIM 0.8774;
- masked-crop LPIPS 0.3261;
- 2,400 primitives.

No comparison arm endpoint was produced, so this did not expose the central contrast. Regardless,
frame 00008 was already explicitly outcome-exposed before v1.

## Sole v4 change

V4 records the immutable initial lineage summary immediately after tracker construction, before
calling `Trainer.train`, and reuses that summary after training. This changes:

- no initialization;
- no random draw;
- no renderer call;
- no optimizer or density decision;
- no iteration budget;
- no saved model or image metric.

It only makes the already-preregistered initial/final lineage diagnostic serializable when row
count changes.

The failed v3 directory must be retained under a clearly named `*_failed_v3` path. The full v4
matrix reruns every arm from scratch, including `beam-only`; no v3 number is copied into v4.

## Implementation binding

Harness SHA-256:
`09a84a30f1783018c5963a701803ec5eae03efe07aee951870c69c89fbdce98f`

All other implementation hashes remain those frozen in v1.

## Official v4 command

```bash
PYTHONUNBUFFERED=1 .venv-cuda/bin/python benchmarks/carrier_refinement_fullres.py \
  --protocol benchmarks/results/20260727_carrier_refinement_fullres_PREREG_V4.md \
  --out runs/carrier_refinement_fullres_frame00008_20260727
```
