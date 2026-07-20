# Independent results audit: Stage-1 semantic-factorial utility

Verdict: PASS (unqualified)

Reviewed at `2026-07-16T06:54:34Z` by the independent N77 Phase-B results-audit
session (`/root/n77_phase_b_audit`). I did not preregister, implement, review, seal, or execute the
harness; did not participate in the Phase-A review or Phase-B attempt; and did not run an official
seed. I followed the repository `realtime-gs-results-audit` procedure and recomputed every
decisive Phase-B quantity from the raw NPZ with `allow_pickle=False`. No decision below is taken
on trust from the result JSON.

There are no blocking, major, or qualifying findings. The artifact is a valid negative result for
the proposed **joint** `(m,o)` repair: it materially improves the frozen Depth path but fails
non-inferiority in the frozen Carve path, so `repair_utility_survives=false`, the current boundary
must be retained, and no default change is authorized. The positive source-observation-color
effect is an attribution result and research lead, not an authorization to select the color-only
arm after seeing this outcome.

## Claim disposition

| # | Claim | Kind and scope | Evidence | Disposition |
|---|---|---|---|---|
| 1 | The complete Phase-B execution and evidence-validity gate passed. | Measured; deterministic CPU synthetic; three fresh seeds; exact matched capacity; fixed topology. | Bound JSON and raw NPZ named below. | **Confirm.** All raw, isolation, count, schedule, and reporting gates independently pass. |
| 2 | The full candidate `m_amp__rgb_obs` is materially better than `w_fit__c_fit` for Depth. | Measured held-out result under the frozen Depth lifter and refinement. | 18 final held-out render cells for the two primary Depth arms, reduced first by camera then by seed. | **Confirm at the frozen scope.** Mean PSNR is `+3.1272303263 dB`; all three seeds improve materially. |
| 3 | The full candidate is non-inferior for Carve. | Preregistered primary claim. | Matching Carve held-out cells and raw reductions. | **Retire for this protocol.** Mean PSNR is `-2.2053135766 dB`, mean SSIM is `-0.0400165584`, and every seed fails the PSNR non-inferiority floor. |
| 4 | The joint invariant semantics survive as a general cross-backend repair or justify a default change. | Cross-backend/default claim. | Frozen conjunctive decision. | **Retire.** `repair_utility_survives=false`, `cross_backend_material_improvement=false`, and `default_change_authorized=false`. |
| 5 | Color routing is the Depth driver; scalar, color, and their interaction are material Carve drivers. | Factorial attribution within this exact experiment. | Raw four-arm per-seed final metrics and frozen driver rule. | **Confirm, narrowly.** These are total downstream effects, not physical-semantic truth or permission to select a new arm. |
| 6 | The result establishes physical opacity/albedo, real-data transfer, CUDA/gsplat behavior, or speed/memory performance. | Asserted beyond-scope claim. | No applicable evidence. | **Remain unverified and unclaimed.** |

## Chronology, source binding, and artifact routing

- The preregistration and its independent review hash to
  `f53146f12894d5e804baf699b0ba0df51d5768ef708884f5a0343c523d96e1ce` and
  `72596ee50731e6b8c55e4e54f83ce53339f85e0672c0dd55483959759b822e7a`.
  The outcome-blind implementation review hashes to
  `11bf25ac5461e50d88cc412320afd30edc4a4af51c10e0a3afc67589fefd7065` and says exact
  `Verdict: PASS`.
- Seal creation, Phase-A marker/result/review, and Phase-B marker/result are strictly ordered at
  `06:17:31`, `06:18:01`, `06:18:24`, `06:31:50`, `06:36:51`, and `06:39:29` UTC on
  2026-07-16. The Phase-B marker was therefore claimed only after the independent Phase-A machine
  review said `PASS`, `recomputed_phase_a_pass=true`, and `phase_b_authorized=true`.
- The seal SHA-256 is
  `d07131a90357627e0589016bf9b10c88a804af689482e29ba7576a200b7d8adb`.
  All 79 sealed files still match their byte sizes and hashes; the independently regenerated
  source collection is
  `bea80b126dc1caad55f61e05bd52a07c0d8ed32aea85fe8cf6d37c9bfb34b8e2`.
  The harness and focused-test hashes are
  `6baf5455da4f3901ff97e305ba498ea91c956157baf74392c7c1c1622d27e4a7` and
  `28841fb5e4bd482647dfd68b6f0328613211b30f5f76c86e650349cb9d2953e6`.
  The dirty revision `2dddca4aff59702341af9faceefa76ad2505dd83` is usable here because the seal
  preserves the complete tracked diff, untracked-file collection, exact executed sources, and
  passing verification outputs rather than claiming a clean checkout.
- The seal records the exact five-command verification sequence, five zero return codes, and
  correctly hashed stdout/stderr. Its snapshot digest and 35-file loaded-source subset also
  recompute exactly.
- The Phase-B marker SHA-256 is
  `82d1017b92e70c6aaaa5e80cd5689375bb9d3fd8d6b802f45bd394ea11701e0a`.
  It binds the seal, source collection, Phase-A JSON/raw/audit/review, exact sole command,
  environment, and all six prospective valid/invalid paths. Only the valid JSON/NPZ/note triple
  exists; no invalid sibling exists and resume is forbidden.
- The official run is CPU-only with empty `CUDA_VISIBLE_DEVICES`, four OMP/MKL/Torch threads,
  Torch `2.9.0+cu128`, NumPy `2.1.3`, and deterministic algorithms enabled. The sealed scientific
  path also installs the frozen socket/DNS/child-process guard.

## Raw archive and isolation recomputation

The result JSON, raw NPZ, and result note hash to:

```text
005eabffc062e158c1ca510865fa40be799733bc5f9bc6c4c3444fff63fc0d9c  JSON
6c639b3758fb1564225ee02caa897e9700dbb399b67ab0ee2cc267d13ebc0ae9  RAW.npz
c477120e5a1d97d54e68a0aa50b6c8c00823cec1473e336b6ed6bdebc8cf9899  RESULT.md
```

The NPZ has exactly 13,944 unique, uncompressed ZIP members. All arrays load with
`allow_pickle=False`; all are numeric or boolean; no object array exists; and all 9,968 floating
arrays are finite. The arrays contain 244,206,579 uncompressed data bytes. I recomputed the frozen
little-endian dtype/shape/data digest for every member and matched every sorted JSON manifest row.
The independently recomputed collection digest is
`a5eab9669990b6a343a8747832dca5c67edc61d09b4ab78fd8a2702bbddc1c2e`, equal to the
JSON and result note.

The raw member insertion order supplies an additional structural chronology check. The last
global lift/capacity member is index 5,383 and the first refinement member is 5,384; the last
refinement member is 11,503 and the first held-out target/render member is 11,504; completion
starts at 13,940. This agrees with the sealed two-pass implementation: all six capacity cells and
24 matched initializations exist before the first optimizer step, and all 24 final models and
pre-unlock digests exist before the one global held-out unlock.

The pre-unlock input namespace contains only nine local training views with mapping
`[0,1,2,4,5,6,8,9,10]`. Held-out originals `[3,7,11]` occur only in the post-unlock target/render
namespaces. The sealed `HeldOutGuard` raises on every accessor before its one-way global unlock;
the guarded payload is not passed to fit, lift, capacity selection, Trainer, schedule generation,
checkpointing, or pre-unlock hashing.

## Fits, arms, ordinary lifts, and exact capacity

- Seeds are exactly `[4409,5519,6637]`. There are 27 fitted source views and exactly 150 finite,
  bounded components in each, for 4,050 total. All source keys, positive Cholesky diagonals, fit
  histories, field-minimal scenes, split mappings, and the frozen native 120-iteration fit config
  match raw evidence. The default fit remains `weight_color_9p` with geometry trainable.
- The identity `a`, `m`, and `h` fields replay exactly from fitted fields. The `00/10/01/11` arm
  geometry and scalar/color identities are bit-exact. Both scalar and color interventions change
  100% of components in each seed and in the pooled set, exceeding the frozen 10% identifiability
  floor without consulting quality.
- Independent dense evaluation of the frozen coverage equation reproduces all 108 identity-arm
  coverage maps within `1.7881393432617188e-07` maximum absolute error. Every one of the 108
  Carve common-coverage references points to the canonical manifest name and content digest; no
  relabeled or duplicated coverage domain is used.
- Exactly 24 ordinary unmerged production lifts exist, one per seed/backend/arm. Counts range
  from 364 to 1,160. Depth component indices and Carve placed indices reconstruct every ordered
  source-key list exactly. Production versus independent means, covariance, opacity, and SH have
  maximum absolute error zero in all 24 cells. An independent quaternion/scale covariance replay
  agrees within `7.720201641858715e-08` maximum absolute error.
- All frozen `rho=m*L11*L22` values replay exactly in float64. All 5,400 backend-tagged tie-break
  digests, availability masks, complete ranks, selected prefixes, selected masks, canonical
  source-key concatenations, and production-field subsets are exact. The six capacity cells are:

| Seed | Backend | Per-view quotas | Total |
|---|---|---|---:|
| 4409 | Depth | `[88,92,91,79,94,88,85,92,87]` | 796 |
| 4409 | Carve | `[55,46,51,53,56,52,51,49,51]` | 464 |
| 5519 | Depth | `[90,94,95,85,85,96,84,92,84]` | 805 |
| 5519 | Carve | `[42,44,41,44,42,39,35,39,38]` | 364 |
| 6637 | Depth | `[75,84,96,94,88,89,85,83,83]` | 777 |
| 6637 | Carve | `[48,38,50,53,48,51,47,51,52]` | 438 |

Every per-view quota is at least 8 and every total is at least 270. All four matched arms have the
same count vector within a cell. Natural color-only arms have the same emitted keys as their
current-color counterparts; scalar routing changes availability. After common-`rho` matching, the
current-scalar versus `m`-scalar selected-set Jaccards are `0.9925/0.6571`, `1.0000/0.4707`, and
`0.9974/0.6404` for Depth/Carve on the three seeds. This is an intended treatment-mediated
mechanism, but it is important for interpreting the Carve scalar effect.

## Fixed-topology refinement and held-out reporting

- All 24 raw train configs match the frozen CPU Torch, degree-0, no-density, no-mask,
  no-random-background setup. Trainer seeds independently regenerate as
  `2_044_090/2_044_091`, `2_055_190/2_055_191`, and `2_066_370/2_066_371` for Depth/Carve.
  Every 120-entry expected schedule, recorded schedule, and history schedule is bit-exact; all
  nine training views occur and the four arms share one schedule per seed/backend.
- Exactly five checkpoints exist at steps `[0,30,60,90,120]` for every model. I checked all 720
  checkpoint model-field arrays and all 1,080 checkpoint training-render cells. Step 0 equals the
  matched initialization; step 120 equals the final model; all optimizer/checkpoint count vectors
  remain fixed; every training checkpoint PSNR/SSIM and camera mean independently reduces with
  zero error.
- All 24 natural, matched, final, selected-key, schedule, count, and parameter-delta relations are
  exact. The 24 preregistered pre-unlock natural/matched/final/source-key/schedule digests all
  recompute exactly. Raw completion is `(phase_code=11, completed_seeds=3,
  completed_lifts=24, completed_models=24)`, where code 11 is the sealed `complete` phase.
- Post-unlock evidence contains exactly 216 render cells:
  `3 seeds x 2 backends x 4 arms x 3 states x 3 cameras`. Every raw color, alpha, accumulated
  depth, clamped color, target-valid mask, predicted depth, and target is complete and finite. For
  every cell I independently recomputed full-canvas float32 PSNR, frozen 11x11 SSIM, float64 MSE
  and alpha sum, normalized depth RMSE, target-depth coverage, and the unweighted three-camera
  seed mean. All raw and JSON reductions match with zero error. The minimum final held-out PSNR
  over all arms/backends/seeds/cameras is `24.5721473694 dB`, so the 10 dB floor passes.

## Factorial estimands and frozen decisions

All camera values were first averaged within seed; the three seeds remain the replicate unit. The
primary full-candidate differences independently reduce to:

| Backend | Per-seed PSNR differences (dB) | Mean / worst (dB) | Mean / worst SSIM | Non-inferior | Material improvement |
|---|---|---|---|---|---|
| Depth | `[+3.5517686,+3.1276487,+2.7022737]` | `+3.1272303 / +2.7022737` | `+0.02481648 / +0.02292619` | yes | yes |
| Carve | `[-1.8543193,-2.4084511,-2.3531704]` | `-2.2053136 / -2.4084511` | `-0.04001656 / -0.05259609` | no | no |

The three-seed mean PSNR factorial effects are:

| Backend | Scalar main effect | Color main effect | Interaction | Frozen material drivers |
|---|---:|---:|---:|---|
| Depth | `-0.0000587 dB` | `+3.1272890 dB` | `-0.0015772 dB` | color only |
| Carve | `-4.5319131 dB` | `+2.3265995 dB` | `-2.5163723 dB` | scalar, color, interaction |

The corresponding mean SSIM effects are Depth
`(-0.00002401,+0.02484049,-0.00001725)` and Carve
`(-0.06576312,+0.02574656,-0.00669592)`. Every per-seed estimand, paired seed list,
mean/minimum/maximum summary, same-sign driver count, non-inferiority conjunct, materiality
conjunct, cross-backend conjunction, and PSNR-floor decision matches the raw reconstruction.

The exact scientific interpretation is therefore asymmetric. Replacing fitted color with sampled
source-observation RGB is a strong positive attribution signal in both frozen backends, and it
fully explains the Depth gain. Replacing the routed scalar with peak amplitude is essentially
neutral for Depth but strongly harmful for Carve; the negative Carve interaction makes the joint
candidate worse than the current/current arm despite the positive color effect. Because Carve's
selected source identities also change substantially, its scalar effect is the intended **total
downstream effect** through coverage, retention, tunnel placement, matched selection, and
refinement—not a direct-opacity coefficient and not proof that `m` is physically wrong.

The preregistered joint repair consequently fails as a general repair. Keep `w_fit__c_fit` as the
current boundary and do not tune or retry this namespace. A future color-only proposal would need
its own outcome-independent protocol, real-data transfer evidence, and interaction checks before
any integration/default decision; N77 does not authorize selecting `w_fit__rgb_obs` post hoc.

## Commands and skipped evidence

Commands actually executed from the repository root included:

```text
sha256sum <preregistration, reviews, seal, markers, Phase-A authorization, utility JSON/RAW/note>
sed / rg / jq <protocol, seal, marker, result, and sealed harness inspections>
.venv/bin/python <independent allow_pickle=False member, manifest, finiteness, and binding audit>
.venv/bin/python <independent fit, arm, dense-coverage, lift-parity, rho/rank/quota/selection audit>
.venv/bin/python <independent schedule, config, checkpoint, topology, chronology, and digest audit>
.venv/bin/python <independent 216-cell held-out metric, factorial, driver, and decision audit>
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_stage1_semantic_factorial.py tests/test_stage1_fit_seam.py
```

The focused verification passed 59/59 tests. I did not replay the one-shot official command,
construct or fit an official seed, update production/defaults/docs/ARA, run CUDA or gsplat, or
collect timing/memory data. Those omissions preserve the one-shot protocol and the stated claim
boundary. No real-data, CUDA, performance, or physical-semantics claim is promoted by this audit.
