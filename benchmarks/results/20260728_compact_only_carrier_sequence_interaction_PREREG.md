# Compact-only carrier sequence interaction — preregistration — 2026-07-28

## Status and provenance

Status at creation: **FROZEN BEFORE ANY SEQUENCE-INTERACTION ARM OUTCOME**.

This is a third single-scene development experiment. It uses only conclusions and audited
snapshots from:

- `20260728_compact_only_carrier_stage_ablation_AUDIT.md`; and
- `20260728_compact_only_carrier_policy_closure_AUDIT.md`.

No held-out metric is produced. The prior held-out descriptive values are not used.

The audited policy closure established:

- phase 1 must train all degree-zero parameter families;
- a second fixed-topology phase is necessary;
- means may be frozen in phase 2;
- clone-all is unnecessary;
- strict projected-center pruning is viable immediately after phase 1; and
- SH3 helps as an alternative to phase 2 but has not been tested after or jointly with the
  selected phase.

This protocol resolves only the two remaining ordering interactions needed for a compact-only
development pipeline.

## Questions

1. Does strict projected-center pruning remain within its validation budget when placed after
   the selected full phase 2?
2. Does pruning before phase 2 allow the means-frozen optimizer to recover enough quality while
   preserving exact fitting-view center containment?
3. At the same 380-update phase-2 budget, does jointly training SH3 improve materially over
   degree-zero phase 2?
4. If joint SH3 does not suffice, does a separate SH3-only phase add enough incremental value to
   justify a third optimization stage?

## Absolute no-image boundary

Every operation and evaluation may consume only:

- sealed fitted 2D Gaussian observation tensors;
- calibrated cameras and bounds;
- sealed/audited 3D Gaussian snapshots; and
- numeric/hash receipts.

The harness must use `CompactDataset.load(..., load_alpha=False)`. Packed alpha must not be read
or decoded. No RGB, dense mask, image raster, `SceneData`, dense `Trainer`, Pillow, OpenCV, or
image-capable repository data path may be imported or opened. The live denial boundary and its
three negative controls are mandatory.

## Parent binding and source seal

The harness must bind before execution:

- this preregistration digest;
- the parent policy-closure result and audit digests;
- every parent phase-1 and selected phase-2 model/result digest;
- Git revision and dirty status;
- exact command and Python/Torch/CUDA environment;
- SHA-256 and byte count of every `src/rtgs/**/*.py` file and all producing harnesses; and
- compact input manifests/containers and evaluation-bank digests.

The intentionally dirty tree is represented by exact file hashes, not a clean-tree claim.

## Roots, cameras, and fixed evaluation

Reuse the audited policy-closure roots and its sealed validation banks:

- parent phase-1 roots: `282701,282702,282703`;
- phase-2 sampling roots: `283701,283702,283703`;
- new SH-only roots: `285701,285702,285703`;
- evaluation-bank roots: `284701,284702,284703`.

Use the same split:

- fitting: `0,1,2,3,5,6,8,9,11,12,13,14,17,18,19,20,22,24,25`;
- validation: `4,10,16,21`;
- held-out, not evaluated: `7,15,23`.

Each validation metric is evaluated on the already sealed 8,192-attempt uniform and proposal
banks. All phase-2 training arms within a root share the exact 380-step view schedule and point
attempts. The two SH-only continuations within a root share their exact schedule and attempts.

## Bound starting states

For each root, load the audited `corrected_C_all_380` phase-1 model from
`runs/compact_only_carrier_policy_closure_20260728`.

Also bind and load its audited `continue_means_fixed_380` result as the unpruned selected
phase-2 anchor. Do not rerun or select among the prior arms.

## Containment operator

Use the exact audited strict projected-center criterion. Retain row `i` iff, for every fitting
view `v`:

`depth_v(x_i) > near`

and

`q_iv = min_j (pi_v(x_i)-mu_vj)^T C_vj^-1 (pi_v(x_i)-mu_vj) <= 9`.

Only positive-amplitude fitted 2D Gaussians participate. Record removed identities and hashes.
Every containment arm must assert zero fitting-view `q>9` and near-plane violations after its
final operation.

Means remain frozen after pruning. Therefore later training cannot invalidate projected-center
containment.

This is fitted-Gaussian visual-hull center containment only. It is not exact source-mask
containment, full-footprint containment, surface occupancy, or proof of no interior floaters.

## Arms

All phase-2 training uses the audited compact configuration: 380 balanced-cycle updates, 256
area-Gaussian proposal attempts, proposal-attempt target, 0.25 uniform fraction, CUDA float32,
black background, hard 3-sigma EWA support, and the same learning rates/chunks.

| Arm | Starting state and operation |
| --- | --- |
| `unpruned_phase2_d0` | Immutable audited `continue_means_fixed_380` anchor |
| `unpruned_phase2_d0_then_prune` | Strict-prune the anchor; no recovery |
| `prune_then_phase2_d0` | Strict-prune phase 1, then 380 means-frozen degree-zero updates |
| `prune_then_phase2_sh3_joint` | Strict-prune phase 1, expand to SH3, then the same 380 means-frozen updates with quaternion, scale, opacity, SH0, and SH-rest trainable |
| `phase2_sh3_joint_unpruned` | Expand unpruned phase 1 to SH3, then the same 380 means-frozen joint updates |
| `phase2_sh3_joint_then_prune` | Strict-prune final `phase2_sh3_joint_unpruned`; no recovery |
| `prune_then_phase2_d0_then_sh3` | Start from final `prune_then_phase2_d0`, expand to SH3, then 380 SH-only updates with geometry and opacity frozen |
| `unpruned_phase2_d0_then_sh3` | Start from the audited unpruned anchor, expand to SH3, then the same 380 SH-only updates |
| `unpruned_phase2_d0_then_sh3_then_prune` | Strict-prune final `unpruned_phase2_d0_then_sh3`; no recovery |

There is no topology growth, insertion, clone, split, opacity reset, free birth, legacy repair,
appearance repair, or soft support penalty.

## Frozen metrics and resource accounting

Report equal-view validation `J_Q` and `J_U`, fitting/validation support diagnostics, row count,
semantic hashes, and exact parameter motion. Training receipts must record source/proposal
immutability, point-sample identities, PyTorch peak allocated/reserved bytes, resident compact
teacher bytes, 3D parameter bytes, chunk maxima, and non-decisional elapsed time.

No held-out camera may occur in any metric receipt.

## Frozen decisions

All ratios are paired by root and aggregated geometrically.

1. **Post-phase-2 pruning viability.** `unpruned_phase2_d0_then_prune` is viable only if it has
   zero fitting-view violations, removes at most 10% of rows, and its validation `J_Q` and `J_U`
   are each at most 5% worse than `unpruned_phase2_d0`.
2. **Pre-phase-2 pruning viability.** `prune_then_phase2_d0` is viable only if it has zero
   fitting-view violations, removes at most 10% of rows, and its validation `J_Q` and `J_U` are
   each at most 5% worse than `unpruned_phase2_d0`.
3. **Containment order.** If both are viable, choose pre-phase-2 pruning only if both its metrics
   are no worse than 2% relative to post-phase-2 pruning. Otherwise choose the viable arm with
   lower geometric `J_Q`; ties within 0.5% choose post-phase-2 pruning because it avoids training
   on a transformed starting set.
4. **Joint SH3 materiality.** Under the selected containment order, compare
   `prune_then_phase2_sh3_joint` with `prune_then_phase2_d0` for pre-pruning, or
   `phase2_sh3_joint_then_prune` with `unpruned_phase2_d0_then_prune` for post-pruning. Retain
   joint SH3 only if it improves validation `J_Q` by at least 5% in all three roots, worsens
   geometric `J_U` by at most 2%, and preserves zero fitting-view violations.
5. **Separate SH3 materiality.** Under the selected order, compare
   `prune_then_phase2_d0_then_sh3` with `prune_then_phase2_d0`, or
   `unpruned_phase2_d0_then_sh3_then_prune` with `unpruned_phase2_d0_then_prune`. A separate
   SH3-only phase is justified only if it improves validation `J_Q` by at least 5% in all roots,
   worsens geometric `J_U` by at most 2%, and preserves zero fitting-view violations.
6. **Joint versus separate SH.** If both SH choices pass, choose joint SH3 if it is within 2% of
   the separate contained SH arm on both metrics; otherwise choose the lower-`J_Q` passing arm.
7. The final bounded policy is the shortest sequence satisfying containment whose additional
   stages pass their materiality gates. No purely numerical winner below a frozen materiality
   threshold earns a stage.

## Claim boundary

This experiment may select the compact-only single-scene development sequence implemented by
the repository. It cannot establish no interior floaters, exact masks, cross-scene quality,
general VRAM/runtime savings, production readiness, or a publication claim.
