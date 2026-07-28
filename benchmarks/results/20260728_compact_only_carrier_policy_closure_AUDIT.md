# Compact-only carrier policy closure — independent audit — 2026-07-28

Status: **PASS_WITH_SCOPE_LIMITS**.

## Referee disposition

| Question | Disposition | Independently recomputed evidence |
| --- | --- | --- |
| Did every stage use only compact 2D Gaussian fields and cameras? | **Confirm within the executable boundary.** | The pre-execution source seal still matches all 90 bound source files. The live guard records zero image opens/imports, all three negative controls passed, packed alpha was not loaded, all compact containers match their sealed digests, and held-out cameras were not evaluated. |
| May phase-1 means be frozen? | **No.** | Means-frozen/all geometric ratios: `J_Q=1.0224` (+2.24%), `J_U=1.0027`. `J_Q` misses the 2% non-inferiority gate. |
| Is means-only optimization sufficient? | **No, decisively.** | Means-only/all: `J_Q=4.1823` (+318.23%), `J_U=2.6712`. |
| Is a second fixed-topology phase necessary? | **Yes on this scene.** | Continue/stop: `J_Q=0.7170` (-28.30%), 3/3 wins; `J_U=0.8029`. |
| May means be frozen in phase 2? | **Yes on this scene.** | Means-frozen/all phase-2 ratios: `J_Q=1.0147` (+1.47%), `J_U=1.0116` (+1.16%), both within 2%. |
| Does higher SH help? | **Yes as a phase-2 alternative; incremental value remains untested.** | SH3-only/stop: `J_Q=0.9159` (-8.41%), 3/3 wins; `J_U=0.9521`. It ranks behind degree-zero continuation and was not applied after continuation. |
| Is clone-all necessary? | **No.** | Versus the matched half-budget continuation, legacy clone has `J_Q=0.9754`, 2/3 wins; preserving clone has `J_Q=0.9699`, 3/3 wins. Neither reaches the 5%/3-root gate, so doubling 5,000 rows to 10,000 is unjustified. |
| Should the corrected clone replace legacy copy? | **The math is safer, but no stage replacement is justified.** | Preserving/legacy final `J_Q=0.9944` and `J_U=0.9960`; the 2% replacement gate fails. Independent tensor reconstruction confirms the preserving operator's coincident optical-density and tangent second-moment equations. |
| Can compact masks constrain outside centers? | **Yes, as strict fitted-Gaussian visual-hull center pruning.** | It removes 5.37% of rows and leaves exactly zero fitting-view `q>9`/near-plane violations. Static validation ratios are `J_Q=1.0404`, `J_U=1.0354`, within the frozen 5% gate. Means-frozen recovery preserves containment and improves over stop (`J_Q=0.8069`). |
| Does this establish “no free-floating Gaussians”? | **No.** | It proves projected-center membership in every fitting-view Gaussian support union. A floater inside that visual hull remains observationally indistinguishable; full-footprint containment was neither required nor tested. |
| Does this establish the general VRAM claim? | **No; it establishes a compact-only execution path and absolute resource receipts.** | Peak RSS was 2.72 GiB for the entire research process. No controlled dense baseline, scene scaling, or generalization protocol was run. |

## Bounded policy supported by this audit

1. Fresh fit-only Beam Fusion with no free birth.
2. Renderer-aware symmetric covariance repair only.
3. Compact fixed-topology phase 1 with all degree-zero parameter families trainable.
4. A second compact fixed-topology phase; means may be frozen.
5. No opacity repair, appearance repair, soft support penalty, or clone-all stage.
6. Strict fitting-view projected-center pruning is a viable safety stage, but its exact placement
   relative to the selected full phase 2 and any incremental SH stage still requires a sealed
   interaction test.

The best measured arm is `continue_all_380`
(`J_Q=0.00405465`), while the lower-motion
phase-2 choice `continue_means_fixed_380` is within the preregistered non-inferiority margin
(`J_Q=0.00411440`).

## Provenance and scope

- All 33 arm receipts, model archives, history digests, source hashes, input-container hashes,
  paired sample streams, checkpoint identities, frozen-family motion, clone math, prune subsets,
  and frozen decision gates were independently recomputed.
- The closure run is source-sealed before execution and remains byte-identical at audit time.
- The run is single-scene development evidence. It authorizes a bounded compact-only development
  policy and another interaction experiment, not a production, cross-scene, or publication claim.
- Held-out cameras were not evaluated, so this result does not spend another held-out decision.

## Commands checked

```text
.venv/bin/python benchmarks/compact_only_carrier_policy_closure.py
.venv/bin/python benchmarks/audit_compact_only_carrier_policy_closure.py
```

Full repository verification remains deferred until the compact pipeline, final policy
interaction, documentation, and tests are reconciled together.
