# P3 packet, Layer A (intent-free): jet2 analytic pilot

Packet for `experiments/tasks/20261008_jet2_prior_2dgs_analytic_{sphere,ellipsoid}.json`
(status draft; the `pilot` section and the fields below are what this packet freezes).

## Code state

- Repository: `realtime-gs`, branch `rtgs-030-jet-prior`. Candidate commit: the commit that adds
  this packet (recorded in the manifest command output and in the review files). Worktree clean at
  start; every run writes below `.scratch/` only (pilot mode refuses other roots).
- Companion: `~/Documents/SplatDiffuseLBO` at commit `7ed8601` (read-only; used through
  `scripts/experiments/analytic_operator_companion.py` with `~/miniconda3/bin/python3.12`).
- Diff of this work against `main`: `git diff --no-ext-diff --no-textconv main...HEAD`.

## Code paths

| Role | Path |
|---|---|
| Driver (pilot, teacher, operator, calibrate) | `scripts/experiments/20261008_jet2_prior_2dgs_analytic.py` |
| Reused fit / guards / init / pre-flight orchestration | `scripts/experiments/20261008_jet_field_prior_2dgs_tosca_cat0.py`, `scripts/experiments/20261007_jet_consistency_prior_tosca_cat0.py` |
| Prior | `src/rtgs/optim/jet2_prior.py` (`Jet2Prior`, `jet2_residuals`, `exact_radius_pairs`) |
| Trainer / 2DGS backend | `src/rtgs/optim/trainer.py`, `src/rtgs/render/gsplat_2dgs_backend.py` |
| Geometry evaluation | driver: `closest_point`, `surface_frame`, `curvature`, `analytic_geometry`, `ideal_surfels` |
| Spectrum | `scripts/experiments/analytic_operator_companion.py` + driver `spectrum` |
| Data | `scripts/experiments/prepare_analytic_surfaces.py`; `dataset/external/analytic_{sphere,ellipsoid}/` |
| Deterministic checks | `tests/test_jet2_prior.py`, `tests/test_jet2_analytic_geometry.py`, `tests/test_jet2_eval_controls.py` |

## Data

- Renders: 160 views, 512², sealed by `experiments/data/20261008_jet2_prior_2dgs_analytic_{sphere,ellipsoid}.json`
  (322 files each); split 128 train / 32 held-out (task `splits`).
- Evaluation-only reference: `dataset/external/analytic_*/source/reference.json` (bound by
  `source/PROVENANCE.json`): sphere `l(l+1)` with multiplicities, area `4 pi`; ellipsoid
  lambda_0..lambda_64 and area of ContinuousLevelSetLBOGS E12 (P1 FEM level 7).
- Surfaces: `sum_k x_k^2/a_k^2 = 1`, axes `(1,1,1)` and `(1,0.75,0.5)` (task `datasets[0].surface_axes`).

## Resolved configuration per cell

All P3 cells are arm `base`: task `resolved_training_configs.base.<seed>`,
`surfel_regularization_configs.base` (depth distortion 0.01 from 3000, normal consistency 0.05
from 7500, `depth_scale` = `scene_scale.values.<seed>`), `jet2_prior_configs.base` (weight 0:
no prior). 15000 iterations, densification 500-7500. The teacher cell is the same configuration
stopped at iteration 7500 (`schedule_iterations` 15000). The Jet2Prior estimator used for the
curvature metric reads `jet2_prior_configs.jet2` (`radius` 3.0, `min_mass` 0.5,
`min_conditioning` 0.02) at evaluation time.

## Run plan (cells, seeds, order, budget)

Per surface `S` in (sphere, ellipsoid), in this order, one GPU job at a time:

| # | Command (from the repository root) | Cells | GPU / CPU budget |
|---|---|---|---|
| 1 | `.venv/bin/python scripts/experiments/20261008_jet2_prior_2dgs_analytic.py teacher --task experiments/tasks/20261008_jet2_prior_2dgs_analytic_S.json --run-dir .scratch/20261008_jet2_prior_2dgs_analytic_S/p3_teacher` | base 9561 to 7500 | 15 min |
| 2 | `... pilot --pilot --task <same> --run-dir .scratch/20261008_jet2_prior_2dgs_analytic_S/p3_pilot` | base 9561, 9562, 9563; evaluate; operator on the three models; calibrate | 3 x 30 min GPU; operator 60 min/model guard; calibration grid below |
| 3 | `... pilot --pilot --repeat --task <same> --run-dir .scratch/20261008_jet2_prior_2dgs_analytic_S/p3_repeat` | base 9561 again; evaluate; operator | 30 min GPU; 60 min operator |

Calibration grid (`pilot.calibration`): ideal surfels from `ideal_surfels(axes, N, seed=11,
tilt_deg)` for N in (4000, 15000) x tilt in (0, 5, 10) deg; geometry via `analytic_geometry`
and spectrum via `spectrum`, per row.

Stop conditions: a failed process stops the remaining plan for that surface (report, no rerun
without a new packet); an operator `timeout` is recorded as unevaluated; RSS watchdog 20 GB.
Total budget: about 2.5 h GPU, about 6 h CPU.

## Outputs (all under the run directories)

`teacher_check.json` (`analytic_geometry` of the step-7500 model), `cells/base/<seed>/` with
`receipt.json`, `history.json`, `effective_config.json`, `evaluation.json` (appearance and
`analytic_geometry`), `operator.json` (spectrum, status, structural gates, relative errors vs
reference); `calibration.json` and `calibration/<row>/operator.json`.

## Manifest

```
python3 ~/.claude/skills/experiment-preflight/manifest.py \
  --exclude experiments/reviews/20261008_jet2_prior_2dgs_analytic_P3 --exclude .scratch \
  --exclude runs \
  dataset/external/analytic_sphere dataset/external/analytic_ellipsoid \
  experiments/reviews/20261008_jet2_prior_2dgs_analytic_P3/layer-a.md \
  experiments/reviews/20261008_jet2_prior_2dgs_analytic_P3/layer-b.md \
  > experiments/reviews/20261008_jet2_prior_2dgs_analytic_P3/manifest.txt
```
Packet digest = SHA-256 of `manifest.txt`. The companion repository is bound by its commit.
