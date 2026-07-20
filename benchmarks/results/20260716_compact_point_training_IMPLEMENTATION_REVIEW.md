# Compact point-training implementation review

Verdict: PASS

Unresolved findings: none

## Outcome-blind bindings

This independent implementation review binds the frozen preregistration
`benchmarks/results/20260716_compact_point_training_PREREG.md` at SHA-256
`865f86d35805c265d27caf4b5f6e02b99e4679f53162dbdd23c17681354065ea`.

It binds the 114-path reviewed implementation manifest returned by the outcome-free
`source_hashes()` helper in `benchmarks/compact_point_training.py` at aggregate SHA-256
`56ef22b6ebb849f21508035f473abf6f5f6e20532e7985aa22723deb18e41d44`.
That reviewed aggregate deliberately excludes this review file; the later seal operation must
bind the review itself and rerun the preregistered full verification, avoiding a circular review
hash.

No official seed was invoked, no official fixture was constructed, no official or calibrated
outcome artifact was accessed, no dataset RGB was decoded, and no calibrated acquisition was
started during this review.

## Scope and findings

The review covered the compact observation equations and proposals, strict RGB-free bundle
boundary, point renderer and trainer, CompactCarve handoff, official harness authorization and
statistics, one-shot artifact lifecycle, calibrated source/input bindings, and exact viewer
integration. It also inspected the development-only tests that exercise all nine prerequisite
gates.

The nine gates are executable and outcome-free:

1. Continuous and discrete fixed-attempt estimators preserve rejection nulls, include all
   attempts in the denominator, and cover the differentiable all-null case.
2. Off-grid coordinate, mean, and log-scale gradients meet the frozen materiality and float64
   central-difference tolerances.
3. The global compositor responds materially to an unrelated visible Gaussian independently of
   proposal component metadata.
4. Pre-sampled loss and all five effective parameter-family gradients agree across the frozen
   outer, point, and Gaussian chunk interventions.
5. View schedules are mode-independent, and a deliberate 97-draw intervention at one step cannot
   perturb the next step's seed, coordinates, activity mask, importance weights, or component IDs.
6. All four proposal modes keep fixed 3D topology, immutable teacher digests, finite state, fixed
   attempt counts, and aligned Adam clocks, including the all-invisible case.
7. Fresh-process training strictly reloads the RGB-free bundle while `open`, `os.open`, PIL,
   calibrated loaders, and all cached `SceneData` aliases are proven-denied. All seven literal
   archive caps fail before `np.load`; exact nested keys, identifier grammar, ordinary files,
   symlink rejection, and resolved containment are covered.
8. Native crop translation, reference/indexed teacher parity, and chunked pixel/area evaluation
   parity are covered.
9. Module import is outcome-free. Every helper that exposes official literal arrays, cameras,
   seeds, configuration, or fixture construction requires a marker-bound authorization object;
   an absent or structurally incomplete marker cannot authorize construction.

The result implementation recomputes normalized log-AUCs and labels from a strict digest-bound
RAW reload. Development tests cover the hand calculation, frozen label precedence, neutral and
mixed-seed guards, and rejection of inconsistent RAW bindings.

The lifecycle-only amendment is implemented without changing scientific semantics. Before PASS
RAW commit, a caught failure produces FAIL RAW and RESULT artifacts with
`decision=MECHANISM_FAIL`. After an exclusively created, flushed, and directory-fsynced PASS RAW,
that RAW remains immutable; a later caught failure produces only a FAIL RESULT that binds the
committed RAW, records `failure_phase=post_raw_commit`, and contains no decision. Fault injection
covers both sides of the commit boundary, partial writes are removed, and bound strict reloads
reject replaced bytes.

The calibrated path remains non-decision-bearing and fail-closed. It binds the exact plan, seal,
source aggregate, full RGB/mask directory selection, calibration, StructSplat provider, and
installed gsplat package before use. Acquisition exports only strict 2D-teacher and camera data.
The fresh trainer and bounded teacher evaluator rebind that acquisition and deny dataset/RGB
access. Observation/archive/index preflight precedes CompactCarve index construction, and those
same capped indexes are reused by training. The literal `evaluate_checkpoint_risks=False`
exception proves zero exhaustive evaluator calls while preserving detached snapshots, hashes,
and callback events at `(0,10,20,40)`. Initial/final PLY hashes freeze before held-out RGB decode.
Exact native-resolution initial and final snapshots execute through
`rtgs.render.gsplat_backend.GsplatRasterizer` on CUDA, after which the unchanged preregistered
viewer command is launched and its branded HTTP endpoint and frozen inputs are rechecked.

## Review verification

The reviewer ran the following non-official checks against the bound snapshot; each returned
success:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m pytest -q tests/test_observation2d.py tests/test_reconstruction_inputs.py tests/test_compact_trainer.py tests/test_compact_point_training.py tests/test_structsplat_observation.py tests/test_viewer.py tests/test_compact_carve.py

.venv/bin/python -m ruff check benchmarks/compact_point_training.py tests/test_compact_point_training.py tests/test_compact_trainer.py tests/test_reconstruction_inputs.py tests/test_observation2d.py tests/test_structsplat_observation.py tests/test_viewer.py tests/test_compact_carve.py src/rtgs/optim/compact_trainer.py src/rtgs/data/reconstruction_inputs.py src/rtgs/core/observation2d.py src/rtgs/viewer.py src/rtgs/lift/compact_carve.py

.venv/bin/python -m ruff format --check benchmarks/compact_point_training.py tests/test_compact_point_training.py tests/test_compact_trainer.py tests/test_reconstruction_inputs.py tests/test_observation2d.py tests/test_structsplat_observation.py tests/test_viewer.py tests/test_compact_carve.py src/rtgs/optim/compact_trainer.py src/rtgs/data/reconstruction_inputs.py src/rtgs/core/observation2d.py src/rtgs/viewer.py src/rtgs/lift/compact_carve.py

.venv/bin/python -m py_compile benchmarks/compact_point_training.py src/rtgs/optim/compact_trainer.py src/rtgs/data/reconstruction_inputs.py src/rtgs/core/observation2d.py src/rtgs/viewer.py src/rtgs/lift/compact_carve.py

git diff --check
```

The later `seal` operation remains responsible for executing and binding the exact focused test,
repository-wide Ruff checks, and full test suite required by the preregistration. This review does
not substitute for or pre-claim that one-shot verification step.

## Claim boundary

This PASS applies only to executability and protocol fidelity for the frozen bounded synthetic
experiment and the frozen seven-view, 640-teacher-per-view, 835-Gaussian fixed-topology calibrated
integration. It makes no outcome, quality, speed, memory, novel-view, or default-method claim.

Production-scale aggregate budgets, lazy/indexed GPU teacher residency, bounded-backward
rendering, and 3D split/clone/merge/prune density control remain intentionally outside this
experiment. They are future engineering and research requirements, not unresolved findings in
the frozen protocol reviewed here.
