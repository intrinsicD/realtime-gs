# Independent scientist audit — visibility-margin Phase A retry

## Disposition

**Phase-B execution clearance: FAIL.** All three final diffuse seeds pass the frozen validity
floor, but all three fail the materiality gate and the raw-sum pool also fails. The protocol
therefore forbids support-safe training; no passing scientist-review JSON was created and the
Phase-B command must not run.

This disposition is limited to fixed-topology CPU synthetic, depth-initialized refinement with the
Torch reference renderer. It does not establish real-scene behavior, density-control interaction,
near-plane behavior, gsplat/CUDA semantics or performance, or the value of other culling schemes.

## Evidence binding

- Official Phase-A JSON: `benchmarks/results/20260715T213132Z_cpu_visibility_margin_iter2_audit.json`,
  SHA-256 `cacbf8782cf803e27f6715bfe7dd673d0be4f4eabfb51dc920838339e1b08785`.
- Retry protocol: `benchmarks/results/20260715_visibility_margin_iter2_PREREG.md`, SHA-256
  `5769708fda257e82f96f70574817984e94604a5e4e3e90a1c3400d068aa92129`; incorporated original
  protocol SHA-256 `1a0d9ec8c211a678898a699650fab2e2ab4c146c4d82df801e40622ab551767a`.
- Iter2 seal: `benchmarks/results/20260715_visibility_margin_iter2_SEAL.json`, SHA-256
  `5b53648c58e04d1d714d95db3dece183e8b12c23c8b39d289ef6b1ded19f3dda`, source aggregate
  `fb4330d696af327bff8f143edaf6935927085412d42fb1bfc29ee608138cfce0`.
- Once-only iter2 Phase-A marker SHA-256:
  `cc9ca41bd21ae8980265f40189565cf84e7ae25c3d5dde32fbbaffdbfda4282e`.

## Independent raw recomputation

Every checkpoint summary was independently reconstructed from its nine raw per-view records:
`U=I union M`, count/mass partitions, three frozen q bins, exposure identities, objective sums,
and render/residual sums. All 108 checkpoint-view records and all stored summaries matched. The
final diffuse decision recomputed as follows (fractions are shown as percentages):

| Seed | U pairs | M pairs | Missed pairs | Missed effective mass | Render delta / residual | Exposures | Valid | Material |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| 0 | 836,383 | 4 | 0.000478250% | 0.000004720% | 0.000012473% | 2 | PASS | FAIL |
| 1 | 836,496 | 0 | 0% | 0% | 0.000000215% | 0 | PASS | FAIL |
| 2 | 807,584 | 0 | 0% | 0% | 0% | 0 | PASS | FAIL |
| pooled | 2,480,463 | 4 | 0.000161260% | 0.000001646% | 0.000003987% | 2 | PASS | FAIL |

The frozen material minima were 0.05% missed pairs, 0.05% missed effective mass, 0.1% render
delta/residual, 100 missed pairs, and three distinct exposures, with at least 2/3 seed passes plus
a pooled pass. The pool missed those quantitative minima by roughly 310x, 30,370x, 25,082x, and
25x respectively, and also had only two exposures. Its four missed pairs were fully accounted for
in the preregistered `10 <= q < 11` (three) and `11 <= q < 12` (one) bins. Initialization and the
reporting-only view-dependent condition cannot rescue the frozen final-diffuse gate.

## Validity, provenance, and isolation

- Strict JSON parsing passed. All 74 sealed paths still matched their exact hashes; all 40 loaded
  repository sources were a hash-matching subset of the seal. The seal records passing Ruff,
  non-slow CPU tests, and docs-sync in the frozen CPU environment.
- The original seal and consumed first-attempt marker matched their retry-bound hashes. The first
  failed JSON/result-note paths remain absent. The first attempt's exact-depth-tie failure is
  represented append-only; the retry preserves the established current order while stably
  inserting only newly admitted primitives. The iter2 marker predates the result, and no iter2
  Phase-B marker existed during this review.
- All six run identities, 120-step schedules, train/held-out indices, SH schedules, configs, and
  fixed primitive counts matched the protocol. Same-seed diffuse/view-dependent schedules were
  identical. The normal training arm used margin `3.0`; support-safe renders were no-grad
  counterfactuals only.
- All 72 ground-truth parity records (six runs by twelve views) had exact zero current-versus-safe
  color, alpha, and depth errors and exact zero stored-target/depth errors. Thus target generation
  did not encode the culling difference.
- Fitting, lifting, training, and the gate use only the nine training views. Held-out views
  `[3,7,11]` enter only post-training reporting. Source inspection and completed assertions bind
  the all-in-front `U`, disjoint/complete `I/M` partition, `q >= 9-1e-5` shell, support-safe
  coverage, near-plane invariance, and current-order preservation.

Independent checks executed after artifact creation:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m pytest -q \
  tests/test_visibility_margin.py tests/test_visibility_margin_ablation.py
24 passed

git diff --check
pass

fail-closed validate_phase_a_audit(official artifact, exact seal)
pass; phase_b_authorized = false
```

## Claim disposition

| Claim | Disposition | Reason |
| --- | --- | --- |
| The current 3-sigma cull materially truncates hard `q<12` support in this setup. | Retire | Only 4 of 2,480,463 pooled final diffuse support pairs were missed, with negligible mass/render effect. |
| The exact `sqrt(12)` margin improves refinement. | Untested | Phase B is forbidden; no candidate training ran. |
| The current `3.0` margin remains the default. | Confirm | The material gate failed and no default change is authorized. |
| The result transfers to real scenes, density control, gsplat/CUDA, or performance. | Unsupported | Those evidence classes were excluded. |

No threshold, margin, resolution, near-plane, support-cutoff, seed, or framing sweep is authorized
by this negative result. A different culling hypothesis requires a new preregistered question.
