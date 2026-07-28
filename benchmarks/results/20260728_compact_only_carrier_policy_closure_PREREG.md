# Compact-only carrier policy closure — preregistration — 2026-07-28

## Status and provenance

Status at creation: **FROZEN BEFORE ANY POLICY-CLOSURE ARM OUTCOME**.

This is a second single-scene development experiment. Its arms are chosen from validation-only
conclusions in:

- `20260728_compact_only_carrier_stage_ablation_RESULT.md`; and
- `20260728_compact_only_carrier_stage_ablation_AUDIT.md`.

The first experiment's held-out numbers were already descriptively unlocked. They are not used
to choose this protocol, and this experiment will not query or report held-out risk at all.

The first result established, on this scene, that:

- legacy covariance and opacity repair are harmful;
- appearance repair is immaterial;
- corrected renderer-aware covariance repair is material;
- means-only optimization is bad from raw Beam;
- means-frozen optimization is non-inferior from raw Beam but untested after corrected
  covariance; and
- the sampled soft support barrier does not reduce support violations.

This protocol closes only the remaining policy interactions needed to replace the old dense
carrier schedule with a compact-only fixed/topology decision. It is not confirmatory.

## Questions

1. After corrected covariance repair, may means remain fixed, or must all parameter families
   move? Is means-only still insufficient?
2. Does a second fixed-topology compact phase add material value after 380 updates?
3. If a later phase exists, may means be frozen during it?
4. Does compact-only higher-order SH improve enough to justify a separate appearance phase?
5. Does the old clone-all stage help under compact supervision, and does an
   optical-density/moment-preserving tangent clone improve on copied-opacity cloning?
6. Can strict Gaussian-field visual-hull pruning guarantee fitting-view projected-center
   containment without unacceptable validation loss?

## Absolute no-image boundary

The complete process, including initialization, every later optimization phase, topology
construction, pruning, recovery, and evaluation, may consume only:

- fitted 2D Gaussian observation tensors;
- calibrated cameras and bounds;
- generated 3D Gaussian tensors; and
- numeric/hash receipts.

It must use `CompactDataset.load(..., load_alpha=False)`. Packed alpha must not be read or
decoded. No source RGB, mask, dense teacher replay, `SceneData`, dense `Trainer`, Pillow, OpenCV,
or image raster may be imported or opened. The live denial boundary and its three negative
controls remain mandatory.

The harness must freeze before execution:

- the preregistration digest;
- Git revision and dirty status;
- SHA-256 of every executed repository source;
- command, Python/Torch/CUDA environment, input containers, repair/arm configs, and evaluation
  banks.

Because the tree is intentionally dirty research work, the exact source hashes—not a clean Git
claim—are the execution binding.

## Cameras and roots

Use the same global split as the first experiment:

- fitting indices: `0,1,2,3,5,6,8,9,11,12,13,14,17,18,19,20,22,24,25`;
- validation indices: `4,10,16,21`;
- held-out indices: `7,15,23`.

Held-out cameras must not be evaluated, ranked, or reported.

Use fresh paired roots:

- phase-1 training: `282701,282702,282703`;
- later-phase/recovery: `283701,283702,283703`; and
- evaluation banks: `284701,284702,284703`.

Within a root, all phase-1 arms share the exact view schedule and point attempts. All later
training arms share their exact later-phase schedule and attempts. Evaluation banks contain
8,192 uniform and 8,192 proposal attempts for each fitting/validation view and root, using the
same fixed-attempt definitions as the first experiment.

## Fresh compact initialization

Rerun fit-only Beam Fusion from the 19 fitting cameras with the first protocol's exact config and
5,000-carrier cap. Do not reuse the prior unsealed Beam snapshot.

Run corrected covariance repair only:

`P_iv = J_iv Sigma_i J_iv^T + 0.3 I`

and

`r_iv = sqrt(mean(log(lambda(C_iv^-1/2 P_iv C_iv^-1/2))^2))`.

Use 120 Adam steps, learning rate `0.03`, Huber delta `0.25`, prior `1e-3`, world sigma bounds
`[1e-4, 0.5*extent]`, and aspect ratio at most 100. Do not run opacity or appearance repair.

## Phase-1 optimizer

Use 380 balanced-cycle updates, 256 area-Gaussian proposal attempts, proposal-attempt target,
uniform fraction 0.25, CUDA float32, degree-zero SH, black background, hard 3-sigma EWA support,
and the first experiment's chunk sizes and nonzero Adam rates.

Three phase-1 arms begin from the same corrected-covariance snapshot:

| Arm | Trainable families |
| --- | --- |
| `corrected_C_all_380` | means, quaternion, scale, opacity, SH0 |
| `corrected_C_means_fixed_380` | quaternion, scale, opacity, SH0 |
| `corrected_C_means_only_380` | means only |

Every zero-rate family must remain bit-exact.

## Later-stage arms

Every arm below starts from its paired root's final `corrected_C_all_380` state. A later
`CompactTrainer` invocation intentionally starts fresh Adam state, matching the old carrier
schedule's phase boundary.

| Arm | Operation |
| --- | --- |
| `stop_380` | no later operation; immutable anchor |
| `continue_all_380` | 380 more updates, all degree-zero families trainable |
| `continue_means_fixed_380` | 380 more updates with means frozen |
| `higher_sh_only_380` | expand to degree 3; train SH0/SH-rest only; freeze means, quaternion, scale, opacity |
| `legacy_clone_all_recover_190` | clone every row once with the copied-opacity legacy operator, then 190 all-family degree-zero updates |
| `mass_tangent_clone_all_recover_190` | clone every row once with the preserving operator below, then the same 190 updates |
| `strict_prune` | remove every row violating any fitting-view support union; no recovery |
| `strict_prune_recover_190` | the same prune, then 190 updates with means frozen |

No insertion, free birth, opacity reset, stochastic split, or ordinary density strategy is
allowed.

## Matched clone operators

Both clone arms transform every one of the 5,000 phase-1 rows into a parent/child pair in the
same physical row order and at the same two positions.

For each Gaussian, select its largest stored covariance axis `e`, let its standard deviation be
`sigma`, and use deterministic tangent displacement `delta=0.5 sigma`:

- retained parent position: `mu`;
- child position: `mu + delta e`.

The largest covariance axis is used because the shortest axis is the surfel-normal candidate;
the operation must not move a child along the thin normal.

The legacy arm copies quaternion, scale, opacity, and SH to the child and leaves the parent
unchanged. Its coincident limit changes alpha from `a` to `1-(1-a)^2`.

The preserving arm uses the same positions and appearance but updates both rows:

- `a' = 1 - sqrt(1-a)`, preserving combined optical transmittance in the coincident limit;
- `sigma' = sqrt(sigma^2-(delta/2)^2)` along `e`, preserving the pair's covariance about its
  shared first moment; and
- all other covariance axes and SH coefficients are copied.

This is not exact image preservation after a finite offset and alpha compositing order; it is the
matched optical-density/second-moment correction to the legacy operator. The comparison must not
be described as isolating opacity alone.

Both arms have exactly 10,000 rows before recovery and share every recovery sample.

## Strict projected-center containment

For each 3D center `x_i` and fitting view `v`, compute:

`q_iv = min_j (pi_v(x_i)-mu_vj)^T C_vj^-1 (pi_v(x_i)-mu_vj)`

over positive-amplitude fitted 2D Gaussians. A row is retained iff:

- its depth exceeds the point renderer near plane in every fitting view; and
- `q_iv <= 9` in every fitting view.

This is deterministic center containment in the fitting-view Gaussian visual hull. The recovery
arm freezes means, so containment cannot be undone. Both arms must re-evaluate and assert zero
fitting-view violations after the operation.

This criterion still cannot detect a floater inside the visual hull and does not require full
footprint containment. It must not be called a proof of surface occupancy or “no floaters.”

## Frozen evaluation and resource accounting

At every arm's final state report equal-view validation `J_Q` and `J_U`, fitting and validation
support diagnostics, row count, exact parameter motion, and semantic hashes. Phase-1 also records
steps `0,95,190,380`; later optimizer arms record their start, midpoint, and final checkpoint.

Record:

- PyTorch peak allocated/reserved bytes separately for each training phase;
- resident compact-teacher bytes and 3D parameter bytes;
- point/Gaussian/query chunk maxima;
- elapsed time as non-decisional;
- zero forbidden image/alpha/dense-path access; and
- containment removed-row identities and hashes.

No held-out metric is produced.

## Frozen decisions

All ratios are paired by root and aggregated geometrically.

- Phase-1 means-frozen is non-inferior only if validation `J_Q` and `J_U` are each at most 2%
  worse than `corrected_C_all_380`.
- Phase-1 means-only uses the same 2% gate; failure rejects “only update means.”
- A second fixed phase is necessary only if `continue_all_380` improves `J_Q` by at least 5% in
  all three roots and worsens `J_U` by at most 2%.
- If a second phase is necessary, means may be frozen only if
  `continue_means_fixed_380` is within 2% of `continue_all_380` on both metrics.
- Higher SH is necessary only if it improves `J_Q` by at least 5% in all roots and worsens `J_U`
  by at most 2% versus `stop_380`.
- A clone stage is necessary only if it improves `J_Q` by at least 5% in all roots and worsens
  `J_U` by at most 2% versus both `stop_380` and the half-budget `continue_all_380` checkpoint.
- The preserving clone replaces legacy copy only if its final `J_Q` improves by at least 2% and
  its `J_U` is no worse than 2%.
- Strict pruning is viable only if it proves zero fitting-view center violations, removes no
  more than 10% of rows, and keeps both validation metrics within 5% of `stop_380`. Recovery may
  rescue quality but must preserve zero fitting-view violations.

The resulting bounded carrier policy is the earliest/simplest sequence whose required gates pass.
A lower numerical risk without its materiality gate does not justify another stage.

## Claim boundary

This experiment may choose a compact-only development policy for this repository to test across
more scenes. It cannot establish exact source-mask containment, absence of visual-hull-interior
floaters, production quality, general speed/VRAM savings, multi-scene transfer, or a published
default.
