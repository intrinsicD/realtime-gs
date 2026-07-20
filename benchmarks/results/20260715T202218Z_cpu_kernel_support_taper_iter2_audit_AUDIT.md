# Independent scientist audit: kernel-support Phase A retry 2

## Disposition

**PASS for retry-specific Phase-B execution clearance.** The retry is a deterministic scientific
replay of the independently audited first Phase A. Clearance applies only to the two frozen arms
and the fresh iter2 namespace; it does not revive or overwrite the consumed first Phase-B attempt.

Bindings:

- retry audit SHA-256:
  `57421f39ff5d983ac37bc63e2c1eabe1a9528a6ed4415001d52a3ee9bce76609`
- retry seal SHA-256:
  `e6b551222e7242ebf3d44a3fa9d7ede0b41daf39c19f2593a9a8406b5d266097`
- retry source aggregate:
  `4f13421bfb570e8e42570bb97f39aa88bb90c2c8d822864f4272fbb68786e674`
- retry preregistration SHA-256:
  `49eaadad6a62e6d7e5dc3696e4650dfccb693ec99c1348d9afb186f9b4ecac08`

## Independent recomputation

Raw per-step additive recomputation exactly reproduced every stored seed summary and the pooled
decision. The diffuse gate passed in all three seeds and pooled:

| Seed | Eligible | Annulus upstream | Recoverable | Recovered / active | Recovered / boundary |
|---:|---:|---:|---:|---:|---:|
| 0 | 16,164,687 | 41.2645% | 25.5507% | 0.263861% | 5.72836% |
| 1 | 16,427,099 | 40.7169% | 24.8590% | 0.255989% | 5.45598% |
| 2 | 15,699,101 | 40.3301% | 23.5441% | 0.236565% | 5.12382% |
| pooled | 48,290,887 | 40.7745% | 24.6717% | 0.252269% | 5.43819% |

All scientific run fields matched the first audit exactly: scene, fitted-set, initialization and
final tensor hashes; effective configs; target and SH schedules; raw diagnostic records; losses;
checkpoints; hard held-out metrics; summaries; and decision. Only fit/training/wall-clock elapsed
fields differed. This is stronger than threshold agreement and confirms the retry did not change
the experiment.

## Provenance and validity

- The original protocol, seal, passing Phase A, independent review, and consumed Phase-B marker
  matched the historical hashes frozen in the retry seal.
- The original failed output
  `benchmarks/results/20260715T201746Z_cpu_kernel_support_taper_ablation.json` and its result note
  remain absent. The original Phase-B marker remains present at SHA-256
  `0c3b1e96ab56680db64758c9e2ceb17a5c53bb5f950bfd416d1165b08433e3c1`.
- No iter2 Phase-B marker existed during this review. The retry preregistration preceded its seal;
  its atomic Phase-A marker preceded the audit result.
- Every current sealed path and loaded repository source matched the retry seal. Strict JSON,
  environment, split, six run identities, 120-step schedules, fixed counts, q-bin accounting,
  finite denominators, and hard-gradient invariants passed.
- The only common sealed implementation files changed from the first seal were the harness and its
  focused harness test. Repository renderer/trainer/scientific source hashes were unchanged. The
  new test covers canonical tuple/list schedule equivalence and historical artifact validation.
- Focused verification passed: `15 passed` for the two kernel-support test files. The retry seal
  records passing Ruff, full non-slow CPU tests, and docs-sync.

The first failure is credibly representation-only: Python tuple/list equality rejected
semantically identical JSON checkpoint schedules after one candidate training loop, before an
output artifact or candidate evaluation was produced. Canonical JSON schedule comparison repairs
that invariant without changing numerical execution. No candidate result is established by this
audit; real-scene, density-control, gsplat/CUDA, speed, and default-change claims remain withheld.
