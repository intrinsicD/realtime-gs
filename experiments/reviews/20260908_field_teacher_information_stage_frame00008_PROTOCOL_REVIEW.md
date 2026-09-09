# Prospective Protocol Review

- Task ID: `20260908_field_teacher_information_stage_frame00008`
- Protocol SHA-256: `b26db0f9b00acba58a86516f95b8cdac218ba5a1067cee253622f4b0e41e11d8`
- Reviewer: `Codex-protocol-reviewer`
- Verdict: `approved`
- Outcome Access: `none`

## Scope

Approve this frozen development comparison of photograph, high-capacity field, and existing
low-budget field targets under a shared training-only RGB/mask visual-hull initialization.
The approval covers one previously exposed capture, three paired seeds, the specified
downscale-eight target operator, and final-iteration evaluation. It does not establish
high-quality reconstruction, field-only initialization, physical density recovery,
generalization, or a resource advantage. D/E require their own prospective task and approval
after every prerequisite, including independent visual adequacy, passes.

## Checks

- Read the canonical repository guide, experiment lifecycle and experiment/results-audit
  skills. Independently inspected the entire task-owned driver and report helper, their
  focused tests, and the relevant camera, calibrated-loader, compact-field query, trainer,
  density-strategy and preview APIs. Source inspection included the installed gsplat split
  operator to check its use of global random state.
- The task freezes 22 training cameras and four reporting-only cameras, with three paired
  seeds and a fixed nine-cell execution order. Preparation loads training views individually;
  initialization uses their masks and colors. All final fitting models must be saved before
  the evaluation process opens reporting views. The fitting process is fresh for every cell.
- High and low inputs retain their original bytes and production provenance. The high archive
  cap is 8,388,608 bytes; the low archive cap is 168,000 bytes. Both use additive native field
  semantics. C changes acquisition count, fitting budget and fit seed together and is correctly
  labelled a teacher-family comparison, not a row-count-only causal effect.
- Full-resolution camera equality is checked before common scaling. Photographs are
  undistorted at full resolution and premultiplied by their binary masks before the same
  four-site quadrature used for field colors. Field alpha payloads are not decoded. Fitting
  scenes carry no masks, so A/B/C use the same unmasked L1/SSIM objective and black background.
  CPU direct/index, CPU/CUDA and decoded-site parity have frozen sites and tolerances.
- Loader bounds are computed from training cameras/masks only. The hull uses center plus/minus
  half the full-diameter extent, aborts on occupied cube faces, and requires at least 1,000
  shell points. Each paired seed loads the same checksummed initialization in all three arms.
  Reference colors averaged across training projections are an explicit initialization choice;
  adequacy is tested rather than assumed.
- The implementation uses the frozen 8,000-step gsplat optimizer and density schedule, with
  a 100,000-primitive cap and SH degree three. Densification stops sufficiently before final
  evaluation. Python, NumPy, global Torch and CUDA generators are seeded after warmup; the
  trainer separately seeds its private camera generator. CUDA bitwise determinism is not
  claimed. Resolved dataclass configurations match the task for all three seeds.
- LPIPS uses the pretrained Alex backbone without a random-weight or alternate-metric
  fallback. Foreground/boundary metrics use fixed reference-mask regions, full metrics retain
  background error, crops derive only from the reference mask, and alpha IoU uses a fixed
  threshold. The report validates exact per-view coverage, finite metrics, completed cells and
  8,000 executed steps, and recomputes per-view means before paired and group aggregation.
- Teacher qualification precedes baseline adequacy and paired B gates. Inclusive margins are
  evaluated directly as B versus A plus/minus the margin. Numeric success explicitly awaits
  independent visual/results audit; low-budget performance cannot rescue a failed prerequisite.
- CUDA import/execution failures, nonfinite training values, OOM and fitting-worker timeout
  are failures, with no CPU fallback or metric-based early stopping. Resource measurements are
  descriptive on the shared GPU. The complete matrix stops on an execution failure.
- Native flat loss history is mapped to its actual checkpoint observations. Stage boundaries
  use recorded run-wall timestamps; shared preparation/init costs are identified and must not
  be counted repeatedly as independent resource costs. RESULT JSON and Markdown reject
  conflicting existing evidence. The shared renderer owns final HTML/Markdown/manifest output.
- Development source preservation remains mandatory before the first protected worker. The
  protocol itself binds a live-verified 116-file source envelope with aggregate SHA-256
  `d83422a66c3a2bb6137ce0a4059d94554dbb97a7941cd7f0c8002b65b1134382`.

## Independently executed checks

All commands below were read-only or synthetic unit/mechanism checks and accessed no protected
capture reconstruction outcomes:

```text
.venv/bin/python scripts/experiment_contract.py validate
.venv/bin/python scripts/experiment_contract.py validate-data experiments/tasks/20260908_field_teacher_information_stage_frame00008.json
.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/20260908_field_teacher_information_stage_frame00008.json
.venv/bin/python scripts/experiments/20260908_field_teacher_information_stage_frame00008.py selftest
.venv/bin/python -m pytest -q tests/test_field_teacher_information_report.py
```

Registry and data validation passed. The synthetic selftest passed quadrature, projection,
silhouette-band, equal-target loss/parameter-gradient and five forbidden-input-open probes.
All three focused report tests passed. An independent configuration check found no source-binding
errors and exact equality between resolved live dataclass configurations and all three frozen
configurations. The owner is separately responsible for completing the full verification gate,
preserving the GPU preflight receipt, and creating the canonical run snapshot. This prospective
approval is not a statement that those execution/handoff gates have already passed.

Reviewed task-owned source identities:

| File | SHA-256 |
|---|---|
| `scripts/experiments/20260908_field_teacher_information_stage_frame00008.py` | `8b4336df15f9f8232b6bfc7ac394034018ecc93518c0603dc79f11b6e36ac302` |
| `scripts/experiments/20260908_field_teacher_information_stage_frame00008_report.py` | `739ed845b5127f5c28422b061ab7b26f60581b55788ddb8753dc156e6a775006` |
| `tests/test_field_teacher_information_report.py` | `08ac8ebef9df13e3b734bdd9ef80f0bec21496410289b5a21d91466c78848e61` |

## Findings

Before run initialization, an operational amendment changed the viewer command to explicit
port 8879 because the default port was occupied. This record supersedes its unlaunched
`e996ea6721bc447d6b48c784a4e657cc076ea2a113ce338463f5e0a1d7f957a4` review. The reviewer
rechecked the generated command, exact protocol digest and live source-binding equality after
the amendment; experimental controls and inference gates did not change. The report server's
separate port 8765 must be available at handoff without disrupting unrelated processes.

Approved after the driver encoded the prospective corrections concerning low archive cap,
full-diameter bounds and hull-face rejection, exact target/mask sampling, family-comparison
wording, fixed execution/parity controls, inclusive gate arithmetic, native history shape,
complete metric coverage, conflicting evidence, global topology RNG seeding and source binding.
The reviewed source implements the stated comparison and preserves the prerequisite hierarchy.
Any change to the bound source or substantive protocol requires a new digest and review before
initialization. The independent results audit, report/viewer smoke and bundle gates remain
necessary before reporting a completed results-bearing experiment.

## Protected Actions Not Taken

The reviewer did not initialize a run, execute preparation/initialization/fitting/evaluation on
the protected capture, inspect protected reconstruction outcomes, select checkpoints, alter
existing data/field archives, or write the owner task record. Acquisition provenance and camera
metadata were read solely to assess the prospective design. This review has no outcome access.
