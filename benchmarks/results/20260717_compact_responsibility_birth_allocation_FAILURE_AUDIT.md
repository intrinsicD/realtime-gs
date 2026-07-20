# Compact residual-responsibility birth allocation — lifecycle failure audit

Date: 2026-07-17

Verdict: **UNAVAILABLE — namespace permanently closed before sealing**

## Finding

During implementation, `tests/test_optim.py` used the frozen official split-noise root
`77201` as the seed of a real `torch.Generator`. The root agent then executed that test before
the exclusive Phase-B attempt marker existed. This violates the preregistered rule that official
roots may reach a generator only in their matching phase after its attempt marker.

The violating command was:

```text
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33 \
  .venv/bin/python -m pytest -q \
  tests/test_point_render.py tests/test_compact_trainer.py tests/test_optim.py
```

The selected-birth test constructed `torch.Generator().manual_seed(77201)` and consumed two
standard-normal draws. No model-quality result, selection, evaluation bank, Phase-A score, or
Phase-B arm was observed, but the lifecycle rule is intentionally stricter than outcome access.

## Bound state at discovery

- amended preregistration SHA-256:
  `e6f34080320459f74b0c6f20634c94697b74bffe4bfb6cb807f6e35fcc8a3427`
- passing amended preregistration review SHA-256:
  `93b1858be05f75a32ba17e07fc208c1bd2ea3369720ad49adaf9b6ac5db91ee5`
- official split root consumed before marker: `77201`
- seal: absent
- Phase-A attempt marker: absent
- Phase-A result: absent
- Phase-B attempt marker: absent
- final result: absent

## Consequence

The complete `20260717_compact_responsibility_birth_allocation` namespace is terminally invalid.
It may not be sealed, run, repaired, retried, or used for a scientific claim. Its only scientific
decision is `UNAVAILABLE`. The partial harness and tests remain non-result-bearing implementation
material.

The test seed was changed after discovery to development-only root `63892`. That correction does
not reopen this namespace. Re-executing the same scientific design requires a fresh append-only
preregistration, fresh official and focused roots, fresh review, fresh seal, fresh markers, and a
fresh output namespace.

No quality, convergence, allocation, or scaling inference is permitted from this failure.
