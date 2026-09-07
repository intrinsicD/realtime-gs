# Continue at home — BENCH-019 local CUDA work (2026-09-07)

The user authorized committing this completed work, merging it into both main branches and
pushing both repositories. This supersedes the earlier local-only Git delivery scope. Existing
restrictions on additional source-image, mask, model and capture-archive uploads remain intact.

From each home checkout, switch to main and pull the delivered commits:

```bash
git switch main
git pull --ff-only
```

The numerical fixes, experiment implementation, frozen protocols, independent reviews, raw
numeric rows, result summaries and final verification receipts are tracked. Large datasets,
generated fields/models and the full HTML/preview bundle remain on the CUDA workstation at:

`/home/alex/Documents/realtime-gs/runs/20260907_bench019_local_stage_frames00008_00009/`

That canonical experiment completed and must not be rerun into the existing directory. The
source commits bound during execution are RTGS f165d35 and StructSplat 6ff898e, with the allowed
metadata-only transition to StructSplat 3ada860 before downstream execution. Delivery commits
add evidence and documentation; they do not redefine those measured source identities.

The matched run used two exposed Stage frames, three fitting families, three paired downstream
seeds, 512 2D rows/training view and fixed 256 3D rows with 1,000 RGB-refinement updates. Contained
StructSplat improved frame mean foreground PSNR by 0.896715/0.312826 dB, but one frame 9 seed
regressed by 0.238384 dB, so it failed the full consistency/materiality rule. Uncontained normalized
lost all six seed pairs. Initializer midpoint fallback differs substantially between families;
this comparison cannot isolate compositor equations. The representative reconstructions are
blurred at this capacity. Defaults remain unchanged, and the broader BENCH-019 surrogate
question remains unavailable with one capture.

Both repository verification scripts passed. StructSplat passed 2,495 portable tests; independent
CUDA replay reproduced 54 held-out views from 18 saved models. Both report validators, the Struct
portable checker, both real viewer/orbit/blank-control checks and all 1,404 final local HTTP links
passed. Browser smoke used explicitly recorded software WebGL; quality was scored with CUDA.

For the durable scientific findings, read `results-audit.md`/`results-audit.json` in StructSplat's
`ara/evidence/bench019-local-stage-20260907/`, or the canonical BENCH-019 AUDIT files under
realtime-gs `benchmarks/results/`. `final-verification.json` binds the final gate receipts in
both evidence directories. The exact viewer commands are in realtime-gs `docs/EXPERIMENTS.md`
and the workstation report. The completed RTGS task is archived as
`docs/tasks/RTGS-019-local-bench019-development.md`; StructSplat BENCH-019 remains partial for
its general multi-capture question.

A later study could separate the initializer-support effect from fitting-family differences
and test independent captures. It requires a fresh prospective protocol and independent review;
this completed result does not select a new default or authorize a confirmation run by itself.
