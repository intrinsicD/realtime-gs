# P3 packet, Layer A (intent-free): jet2 analytic pilot — revision 2

Packet for `experiments/tasks/20261008_jet2_prior_2dgs_analytic_{sphere,ellipsoid}.json`
(status draft). This packet freezes the task's `pilot` section and the code paths below.
Revision 2 replaces revision 1 (digest `d4924957…dd9a`, reviews REVISE/REVISE).

## Code state

- Repository `realtime-gs`, branch `rtgs-030-jet-prior`, at the commit recorded as line 1 of
  `manifest.txt`. Review files are written next to this packet but **not committed before the
  runs** (a commit would change line 1). The pilot refuses to start unless `git status
  --porcelain --untracked-files=no` is empty in both repositories, and records both HEADs at
  start and end (`start_record.json`, `end_record.json`); it fails if a HEAD changed.
- Companion `~/Documents/SplatDiffuseLBO` at `7ed8601` (read-only), via
  `scripts/experiments/analytic_operator_companion.py` and `~/miniconda3/bin/python3.12`.
- Diff for review: `git diff --no-ext-diff --no-textconv main...HEAD`.

## Code paths

| Role | Path |
|---|---|
| Driver: pilot orchestration, teacher, evaluate, operator, calibrate, surprise checks, records | `scripts/experiments/20261008_jet2_prior_2dgs_analytic.py` (`pilot_main`, `teacher`, `evaluate`, `operator`, `calibrate`, `surprises`, `record`) |
| Reused fit / guards / init / prepare | `scripts/experiments/20261008_jet_field_prior_2dgs_tosca_cat0.py` (`fit`, `initialize`, `appearance`, `collapse_fraction`), `scripts/experiments/20261007_jet_consistency_prior_tosca_cat0.py` (`prepare`, `access_guard`, `data_guard`, `load_views`) |
| Prior (estimator used for the curvature metric) | `src/rtgs/optim/jet2_prior.py` |
| Trainer / 2DGS backend | `src/rtgs/optim/trainer.py`, `src/rtgs/render/gsplat_2dgs_backend.py` |
| Geometry | driver `closest_point` (bisection, singular branch), `surface_frame`, `curvature`, `analytic_geometry`, `ideal_surfels` |
| Spectrum | `scripts/experiments/analytic_operator_companion.py` (companion configuration **v2**, §16.3) + driver `spectrum` |
| Data | `scripts/experiments/prepare_analytic_surfaces.py`; `dataset/external/analytic_{sphere,ellipsoid}/` |
| Deterministic checks | `tests/test_jet2_prior.py`, `tests/test_jet2_analytic_geometry.py`, `tests/test_jet2_eval_controls.py` (output: `pytest.txt` here) |

## Data

Renders: 160 views, 512², sealed by `experiments/data/20261008_jet2_prior_2dgs_analytic_*.json`
(322 files each, checked at start and end); split 128 train / 32 held-out. Visual check:
`render_contact_sheet.png` here (views C0001, C0080, C0160 of each surface). Evaluation-only
reference `dataset/external/analytic_*/source/reference.json` (bound by `PROVENANCE.json`).
Surfaces `sum_k x_k^2/a_k^2 = 1`, axes `(1,1,1)` and `(1,0.75,0.5)`.

## Resolved configuration

Fits: arm `base` only (`resolved_training_configs.base.<seed>`, `surfel_regularization_configs.base`,
`jet2_prior_configs.base` weight 0): 15000 iterations, densification 500-7500, depth distortion
0.01·D/s from 3000, normal consistency 0.05 from 7500. Teacher fits: the same, stopped at 7500
(`schedule_iterations` 15000). Curvature metric: the Jet2Prior estimator with
`jet2_prior_configs.jet2` (radius 3, min_mass 0.5, min_conditioning 0.02) on the evaluated model.
Spectrum: companion v2 — `σ_n = max(σ_min, sqrt(σ_t1 σ_t2)/50)`, trial width `σ_χ,i = 0.75 h_i`,
companion estimator `shape_estimated` with `S_i = 0` where `|S_i|σ_t1,i > 1` (row `flat`:
`S ≡ 0`), numba sparse `measure="closed"`, analytic outward orientation, λ_0..λ_31 by shift-invert,
structural gate G15a; valid status = G15a passed.

## Run plan (task `pilot`)

One command per surface S, sphere first, then ellipsoid; one GPU job at a time:

```
.venv/bin/python scripts/experiments/20261008_jet2_prior_2dgs_analytic.py pilot --pilot \
  --task experiments/tasks/20261008_jet2_prior_2dgs_analytic_S.json \
  --run-dir .scratch/20261008_jet2_prior_2dgs_analytic_S/p3
```

Order inside (each phase a fresh process with the hard timeout `pilot.timeouts_seconds`):
start record → `calibration/` (8 rows, CPU) → `teacher/` (base 9561 and 9562 to step 7500,
`teacher-eval`) → `pilot/` (base 9561, 9562, 9563; evaluate; **surprise check**; operator;
**surprise check**) → `repeat/` (base 9561 again; same sequence) → end record.

Calibration rows (`pilot.calibration.rows`, seed 11, coverage seed 1011): N = 4000 and 15000 ×
tilt 0/5/10 deg; N = 15000 tilt 0 with log-σ spread 0.5; N = 15000 tilt 0 with `S ≡ 0` (v2flat).

Stops (`pilot.surprise`, enforced by `surprises`): after evaluate — any base model with normal
median > 15 deg, held-out PSNR < 25 dB or silhouette IoU < 0.95; after operator — status
`invalid` (G15a), `memory`, `rss_watchdog`, `error`, `killed_timeout`, missing output, or peak
RSS > 16 GB. A failing phase process stops the plan. Operator `timeout` (assembly deadline or
60-min guard) is recorded as unevaluated and does not stop.

Budget per surface (expected from the technical runs / hard bound): calibration 0.5 h / 4 h;
teacher fits 2 × 4 min / 2 × 60 min; pilot + repeat fits 4 × 8 min / 4 × 60 min; evaluation
2 × 5 min / 2 × 60 min; operator 4 × 10 min / 4 × 70 min. Both surfaces together: about 1 h GPU
and 2 h CPU expected; hard bound 24 h.

## Outputs

`start_record.json`, `end_record.json`, `stopped.json` (if stopped); `calibration/calibration.json`
(+ per row `gaussians.ply`, `operator.json`); `teacher/teacher_check.json`; per fit
`cells/base/<seed>/{receipt,history,effective_config,evaluation,operator}.json`;
`surprises_{evaluate,operator}.json`.

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
