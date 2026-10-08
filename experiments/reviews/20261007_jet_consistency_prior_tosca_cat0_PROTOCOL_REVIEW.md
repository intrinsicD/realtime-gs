# Prospective Protocol Review

- Task ID: `20261007_jet_consistency_prior_tosca_cat0`
- Protocol SHA-256: `03e3b661ebaa33203df3d2f8786858d68633d1f336461257b403647cfb9c35a1`
- Reviewer: `Alexander Dieckman`
- Verdict: `approved`
- Outcome Access: `none`

## Scope

Development screen of the jet-consistency (Weingarten) prior on 160 pyrender views of TOSCA cat0
(128 train / 32 held-out), arms base / jet1 / jet2 x seeds 9561-9563 plus the descriptive
lambda arms 0.01 and 1.0 (seed 9561). The question, regularizer, data, arms, metrics,
predictions and decision policy are those of
`benchmarks/results/20261007_jet_consistency_prior_tosca_cat0_PREREG.md`.

How this record came about, stated precisely. The reviewer approved the PREREG content on
2026-10-07 ("Ich habe das PREREG gelesen. Und ich bin einverstanden.", quoted in the PREREG
status line) and asked that the digest approval be recorded once the task's structural fields
were generated with the repository tooling. The task owner generated those fields and wrote this
record on that instruction; it binds the content approval to the digest above. The reviewer has
not separately read the generated JSON fields; the points below are the ones that go beyond the
PREREG text and should be checked by the reviewer when reading the result.

## Checks

- Data roles: reconstruction reads RGB, masks and calibration of the 128 training views and the
  seeded random initialization only; held-out views and the source mesh open only in `evaluate`
  (audit-hook guard in the driver, probed in a pipeline smoke). Data seal
  `experiments/data/20261007_jet_consistency_prior_tosca_cat0.json` (322 files); mesh copies
  bound by `dataset/external/tosca_cat0/source/PROVENANCE.json`.
- Arms differ only in `jet_prior_configs`; `resolved_training_configs` are identical except the
  seed (gsplat-default, 15000 steps, densification 500-7500, SH 1, masks on).
- Decision rule in `decision_policy.per_arm` restates the PREREG thresholds (4 deg, 30 %, -0.2 dB,
  every seed); reject is written as "worse on the normal median in every seed, or worse on the
  centre median in every seed".
- Geometry evaluator validated on the companion model before any training (reproduces 16.5 deg
  median, 49.5 deg p90, 0.0279 splat units, Zhou-Laehner 38.21 %).

## Findings

Approved on the basis of the content approval above, with these generated deviations from the
PREREG wording, all fixed before any outcome of this task existed:

1. The SplatDiffuseLBO operator metric (PREREG primary metric 3) is deferred
   (`deferred_metrics`): `splat_lbo_cat.py` takes no input path and its v3 assembly needs more
   than 11 GB host RAM, above this run's 8 GB limit. The Zhou-Laehner operator on the exported
   centres is computed; the operator prediction stays open.
2. Sign of the paraboloid residual: with `S` fitted from `P_i(nu_j - nu_i) = S s_j` (`S = I/R` on
   a sphere with outward normals, the companion code convention), the consistent height model is
   `z = -1/2 s^T S s`; the PREREG's `z_j - 1/2 s_j^T S_i s_j` is implemented as
   `z_j + 1/2 s_j^T S_i s_j`.
3. RGB stored as JPEG quality 100, 4:4:4 (the contract's canonical RGB pattern), not PNG.
4. Development (non-official) run: no commits are made in this session, so `init-run` uses
   `--development` on a dirty worktree; the coordinator archives the bound source.

## Protected Actions Not Taken

The reviewer did not execute the protected run and has seen no outcome of this task. (The owner
ran a 600-step pipeline smoke in `.scratch/` before freezing to exercise the driver; the protocol
fields were written before that smoke and were not changed afterwards except for removing the
jet parameters from `TrainConfig`, an implementation refactor that keeps their values.)
