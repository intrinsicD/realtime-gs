# Driver response to the V1 prospective review

- Task ID: `20260926_field_only_distillation_stage_frame00008`
- Rejected digest: `ef904cd749a80de5f25b9c6706cc705617aebfe56bd0b16912695626a99e44af`
- Revised digest: `05048698cc34656fd562c8f60d534037745d99acf91b9cac6712a5f314b23f61`
- Driver: `Claude-Code-Opus-5.5-driver`
- Review record: `20260926_field_only_distillation_stage_frame00008_PROTOCOL_REVIEW_V1_REJECTED.md`
- Source binding: 118 files, aggregate `587419c468566af2a1a3181f992012a00eff253a5942950e3bfd90be74dab206`

No hypothesis, threshold, seed, split, comparator, metric, iteration budget or execution order
changed. No held-out view or reconstruction outcome was observed.

## Required changes

- **B1.** Confirmed: `importlib.metadata.version("realtime-gs")` raises `PackageNotFoundError`;
  the distribution is `rtgs` 0.1.0. (a) The environment record now queries `rtgs` and guards every
  lookup. (b) `write_environment` is separate from `snapshot_source` and runs on every start that
  lacks `environment.json`. (c) Snapshot, environment, source guard, entry data guard and
  preflight all run inside the coordinator `try`, so a start-up failure writes
  `execution_failure.json` and the failure report. (d) `preflight()` requires CUDA, imports
  gsplat and runs one LPIPS(alex, pretrained) forward on CUDA before the first protected worker,
  recording `preflight.json`. Unit test: `test_environment_record_uses_installed_distributions`.
- **B2.** `preprocessing.target_operator` now states both regimes: colour ignored where the soft
  alpha fraction is zero, and outside-mask target colour entering in proportion to the fraction
  at 0.25/0.5/0.75 boundary pixels. It names the family difference in that band (contained teacher
  black, uncontained teachers extrapolated, photographs room background) as in scope for H2 and
  required in the audit.
- **B3.** `decision_policy.h2_teacher_containment` now writes the mirrored two-part reject
  condition exactly as `gates()` evaluates it.

## Optional changes

- **R1.** Adopted. The audit hook resolves paths (and the target-cache root) before comparison;
  `publish()` refuses any run root other than `runs/<task_id>` (unit-tested).
- **R2.** Adopted. `prepare` aborts if any training pixel with soft alpha > 0 has decoded
  coverage < 1 for any family and records the count per family.
- **R3.** Adopted. `prepare` requires each field view's recorded source photograph and mask
  SHA-256 to equal the sealed digests.
- **R4.** Adopted. `reset_cuda_peak_stats` is false in all six resolved configs; the resource scope
  states the resulting peak semantics.
- **R5.** Adopted. H1/H2 inequalities are evaluated in written form (`a >= b - margin`), and the
  decision-policy text says so.
- **R6.** Adopted. `training_masks` now describes the per-view checks performed once in `prepare`
  on behalf of all arms.
- **R7.** Not adopted. The six non-gating cells keep the second uncontained family and the
  photograph initializer reference visible; the full matrix is about one GPU-hour.
- **R8.** Adopted. A fitting-worker timeout writes a `timed_out` receipt before the failure report.

## Verification

Focused CPU tests in the two task files: 16 pass and the CUDA decode-parity test self-skips on
CPU (it passes on the local GPU); `./scripts/verify.sh` passes on the committed
tree. A repeated non-protocol GPU smoke in `.scratch/` (60 iterations, one seed, two training
views standing in for held-out) exercised preflight, the environment record, the new prepare
checks (0 clipped mask pixels, all source digests matching), all seven fit conditions, evaluation
in a fresh process and the gate code. Its training-view teacher values equal the previously
disclosed ones.
