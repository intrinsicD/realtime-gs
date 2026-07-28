# Compact-only carrier stage ablation — independent audit — 2026-07-28

Status: **PASS_WITH_SCOPE_LIMITS**.

## Referee disposition

| Claim | Disposition | Independently recomputed evidence |
| --- | --- | --- |
| The run used compact 2D Gaussian/camera tensors rather than RGB/masks. | **Confirm, executable-boundary scope.** | The live receipt passed with zero image opens, zero forbidden imports, and all three negative controls; packed alpha was configured as unaccessed. Focused tests separately make alpha member reads/decodes fatal. This is not an OS-level information-flow proof. |
| Legacy covariance repair is needed. | **Retire.** | Final validation `J_Q` factorial marginal 1.5163 (+51.63%); 0/3 paired wins. |
| Legacy opacity repair is needed. | **Retire.** | Final validation `J_Q` factorial marginal 1.0978 (+9.78%); 0/3 paired wins. Its target is non-identifiable fitted-mixture amplitude. |
| Legacy appearance repair is needed. | **Retire as a stage.** | Final validation `J_Q` factorial marginal 0.9988 (-0.12%), far below the preregistered 5% materiality floor. |
| Renderer-aware symmetric covariance repair is better than legacy C. | **Confirm on this scene.** | `J_Q` ratio 0.2386 (-76.14%), 3/3 wins; `J_U` ratio 0.3845. |
| Corrected appearance should be retained with corrected covariance. | **Retire as material.** | CA vs C changes `J_Q` by -0.08% while changing `J_U` by +0.18%. The mechanical CA winner is below any stage-necessity threshold. |
| Updating only means is sufficient. | **Reject.** | Means-only vs all-parameter Beam: `J_Q` 2.4876 (+148.76%), `J_U` 1.6335. |
| Means can be frozen. | **Narrow confirm for the Beam initializer.** | Means-fixed vs all-parameter Beam: `J_Q` 0.9993, `J_U` 0.9935; it passes the 2% gate. This interaction was not tested after corrected covariance. |
| The sampled support barrier removes outside centers/free floaters. | **Retire.** | Beam support vs parent `J_Q` 1.0001; mean validation outside fraction 0.006817 vs 0.006800. A floater inside the visual hull remains undetectable. |
| The run proves the VRAM selling claim. | **Narrow to an absolute measurement.** | Maximum compact training allocation 74.42 MiB, reservation 80.00 MiB, with 5,000 Gaussians and all compact teachers resident. There is no controlled dense baseline, idle-GPU timing protocol, scene scaling, or generalization evidence. |

## Recomputed ordering

- `corrected_CA_all`: validation geometric `J_Q=0.00549373`.
- `corrected_C_all`: validation geometric `J_Q=0.00549789`.
- `beam_all`: validation geometric `J_Q=0.01492036`.
- `beam_means_fixed`: validation geometric `J_Q=0.01491001`.
- `beam_means_only`: validation geometric `J_Q=0.03711604`.

The numerical CA/C ordering does not justify retaining appearance repair: its `J_Q` edge is
0.076% and `J_U` is worse.
The bounded minimal schedule supported for another experiment is corrected covariance followed by
compact all-parameter optimization; a corrected-covariance × means-freeze interaction remains
open.

## Protocol and provenance findings

1. All 48 arm receipts are complete; each retains 5,000 rows, immutable teacher/proposal digests,
   paired within-root point samples, and bit-exact zero motion for frozen parameter families.
2. Only fit and validation metrics occur in checkpoint receipts. The held-out finalist set and
   selection hash match the frozen rule; held-out numbers remain descriptive.
3. The producing manifest did not freeze the Git revision, dirty diff, command, or an executed
   source archive. Post-run file hashes and mtimes found no relevant file newer than the estimated
   start, but that is not a cryptographic execution seal. This development result is therefore
   not replay-complete or confirmatory.
4. Held-out proposal banks were generated before training as explicitly frozen, while held-out
   target risks were not computed until selection. Because held-out compact fields were resident
   in the same process, this is not an unopened/one-shot sealed-data design.
5. The GPU resource numbers are allocator diagnostics only. The run did not record an idle-device
   control and cannot support speed or comparative-memory claims.

## Commands checked

```text
.venv/bin/python benchmarks/compact_only_carrier_stage_ablation.py
.venv/bin/python benchmarks/audit_compact_only_carrier_stage_ablation.py
.venv/bin/pytest -q tests/test_compact_views.py tests/test_compact_only_carrier_stage_ablation.py tests/test_observation2d.py tests/test_compact_trainer.py tests/test_carrier_refinement.py tests/test_lift.py
```

Full repository verification is deferred until the result, compact pipeline changes, docs, and
later-stage protocol are all reconciled in one final verification pass.
