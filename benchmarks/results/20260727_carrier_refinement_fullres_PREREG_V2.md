# ADR-002 carrier refinement on full-resolution masked Janelle — frozen protocol v2

**Frozen:** 2026-07-27, before any Janelle source or outcome was read by the new harness
**Supersedes:** `20260727_carrier_refinement_fullres_PREREG.md`
**Evidence class:** outcome-exposed, single-scene, single-seed development/mechanism evidence

This protocol incorporates every scientific and operational setting from v1, whose SHA-256 is
`38e7e258a36188337c2a926e2d33759b34431556e342e747ceaa45383ddd84aa`.

## Reason for revision

The first smoke command failed during Python module import:

```text
ModuleNotFoundError: No module named 'benchmarks'
```

The failure occurred before `main()`, before compact/raw data loading, and before any result metric.
No arm ran and no outcome was exposed. V2 makes only two operational changes:

1. add the repository-standard fallback import used when a benchmark is executed as a direct
   script rather than a module;
2. point the harness's default protocol path to this v2 file.

All inputs, split roles, 5328×4608 resolution, seed, initializations, repair settings, 13 arms,
160-update budgets, metrics, LPIPS hashes, interpretation gates, evidence restrictions, required
artifacts, and unavailable-SfM-baseline treatment remain exactly as frozen in v1.

## Revised implementation binding

- `benchmarks/carrier_refinement_fullres.py`: populated by the SHA-256 recorded immediately below
  after this file is added and before the second smoke command.
- All other implementation/document hashes are exactly those listed in v1:
  - `src/rtgs/lift/carrier_refinement.py`:
    `bbd50a727da29663e22415376c8cabf269bf561e929a76ae86a4360dbb5590f5`
  - `src/rtgs/optim/carrier_schedule.py`:
    `62de83cdd953a33fb1299ad023b2fb3c57b4f04aa872b44f017e39dae9280f69`
  - `src/rtgs/optim/density.py`:
    `5c64ea245de69a63317c3c4564017ea82066184963af2984c7d89928af601ef8`
  - `src/rtgs/optim/trainer.py`:
    `49c6493b1d9fe5ac45abdda1f3a16802dfbce112cc9f50d96f0590504804ce23`
  - `tests/test_carrier_refinement.py`:
    `7cdc7376631d64adbb05c3e20cc716d8786884fbbd1e4788e5d56f0b4ccab7fb`

Harness SHA-256:
`fceb622e59bac93f921017a2113417a7ab6da7e12bf01ad427a0482d7f0c92cd`

## Official command

```bash
PYTHONUNBUFFERED=1 .venv-cuda/bin/python benchmarks/carrier_refinement_fullres.py \
  --protocol benchmarks/results/20260727_carrier_refinement_fullres_PREREG_V2.md \
  --out runs/carrier_refinement_fullres_frame00008_20260727
```
