# realtime-gs — agent guide

Research repository testing one idea: **make 3D Gaussian Splatting (3DGS) reconstruction fast
by first fitting every input image with 2D gaussians, then lifting those 2D gaussians into 3D
as the initialization for standard 3DGS optimization.**

Pipeline (see `docs/ARCHITECTURE.md` for the full design):

```
images ──► [1] image2gs: fit compact 2D gaussians per image (native or StructSplat)
       ──► [2] lift: 2D→3D, five competing variants
              A. lift.gradient — multi-view photometric gradient descent on per-ray depths
              B. lift.depth    — feed-forward monocular depth (Depth Anything V2 / mock)
              C. lift.carve    — voxel color-consistency carving + merging along ray tunnels
              D. lift.hybrid   — aligned depth seed + bounded-ray photometric correction
              E. lift.field    — image-free compact-field proxy refit + topology research path
       ──► [3] optim: standard RGB-backed 3DGS, or the separate compact-only fixed-topology
                      carrier path (2D fields only; no SceneData/image handover)
```

## Hard rules (do not break these)

1. **CPU-first testability.** No module may require CUDA at import time. `gsplat`,
   `transformers`, and any GPU-only dependency are imported lazily inside functions and
   guarded. The pure-PyTorch reference rasterizer (`rtgs.render.torch_ref`) is the
   correctness anchor; the full test suite must pass on a CPU-only machine.
2. **Backends are pluggable.** Dense rasterizers implement `rtgs.render.base.Rasterizer`;
   sparse point rasterizers implement `rtgs.render.point_base.PointRasterizer`; depth estimators
   implement `rtgs.depth.base.DepthBackend`. New fast paths go behind these interfaces — never
   fork pipeline logic per-backend.
3. **Determinism in tests.** Every test seeds RNGs (helpers in `tests/conftest.py`).
   Quality thresholds in tests are deliberately loose; do not tighten them to "current
   behavior" — they encode floors, not snapshots.
4. **Docs stay in sync.** `python scripts/docs_sync.py` is part of verification and CI.
   If you add/remove a subpackage, CLI command, or skill, update `docs/ARCHITECTURE.md`
   (and this file's pipeline sketch if it changed).
5. **Benchmarks are tracked, not vibes.** Performance claims go through
   `python benchmarks/run.py` (JSON in `benchmarks/results/`, human table in
   `docs/BENCHMARKS.md` via `--update-docs`). Never hand-edit the generated table block.
6. **Experiments are logged.** Research findings (a variant works/doesn't, a
   hyperparameter matters) get a dated entry in `docs/EXPERIMENTS.md`.
7. **Local data and viewer handoff are mandatory.** Every new R&D branch must exercise a
   calibrated scene under `dataset/` before it is considered complete. Synthetic scenes remain
   valid for deterministic unit tests and mechanism screens, but synthetic-only evidence cannot
   close a pipeline-quality or default question. Results-bearing runs save `--out` artifacts and
   previews and the v2 experiment bundle: dimensioned fitting history, effective parameters,
   environment/run receipts, generated relative-link `index.html` and `README.md`, a checksummed
   manifest, and smoke-test receipts for both the page and an `rtgs view` command covering the
   saved initial/final or checkpoint Gaussians. Treat the WebGL view as a diagnostic;
   quantitative decisions use exact rasterizer metrics on a frozen train/validation/test protocol.
   Gate the bundle with `python scripts/check_results_bundle.py <run_dir>` — it verifies the
   artifacts, previews, results page links, and viewer receipt. A run that does not pass it is
   not a results-bearing run.
8. **Claims live in `ara/`.** Any quantitative or capability statement that enters `README.md`
   or `docs/` must have a row in `ara/logic/claims.md` bound to evidence on disk. See
   "Evidence and claims" below. `python scripts/check_ara.py` is part of verification.
9. **Experiments are task-first and use one report.** Before writing a result-bearing driver or
   starting a run, register `experiments/tasks/YYYYMMDD_<task_slug>_<data_slug>.json` and freeze
   inputs, splits, seeds, stages, comparators, metrics, resource scope, and command. Initialize
   `runs/<task_id>/` through `scripts/experiment_contract.py` only after a distinct prospective
   reviewer approves the exact protocol digest without outcome access. Never overwrite the run
   root or create `_v2`/`_final`/`latest` siblings. Freeze `report_template_version: 2` in every
   new task. Generate `index.html`, `README.md`, and `manifest.json` from the shared metrics,
   history, configuration, environment, and receipt schemas; never hand-edit generated files.
   Historical v1 bundles and benchmark/result paths are append-only provenance and remain valid.
10. **Substantial work has one durable task record.** Behavior, interface, dependency, default,
    policy, durable-state, and claim/result changes use `.agents/state/current-task.md`, with
    closed records in `docs/tasks/`. Driver, Reviewer, Turn, status, maturity, handoff, and verdict
    must agree. Self-review may reach only `Provisionally accepted (self-reviewed)`; independent
    acceptance requires distinct labels. This coordination layer links to, and never replaces,
    the roadmap, experiment registry, experiment log, or ARA ledger.

## Commands

```bash
# one-time setup (CPU box; on a GPU box add: pip install -e '.[cuda,depth]')
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]' \
    --extra-index-url https://download.pytorch.org/whl/cpu

./scripts/verify.sh          # complete CPU/local/CI structural gate (before every commit)
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q             # full CPU suite
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q -m "not slow"  # verify/CI suite
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/python scripts/docs_sync.py     # docs↔code consistency check
.venv/bin/python scripts/check_ara.py     # ara/ claim-ledger structural check
.venv/bin/python scripts/check_script_layout.py           # scripts/ vs scripts/experiments/
.venv/bin/python scripts/check_agent_workflow.py           # task/skills/CI authority gate
.venv/bin/python scripts/experiment_contract.py validate  # task/program structural gate
.venv/bin/python scripts/experiment_contract.py review-digest experiments/tasks/<task_id>.json
.venv/bin/python scripts/check_results_bundle.py <run_dir>  # Hard Rule 7 bundle gate
.venv/bin/python benchmarks/run.py --quick --update-docs   # refresh benchmarks
.venv/bin/rtgs --help                     # CLI: fit-images / lift / lift-field / refine / run / render / view / bench
```

## Repository map

```
src/rtgs/
  bench019.py, bench019_portfolio.py, bench019_adapters.py, bench019_predictors.py
               passive cross-repository receipts, pre-outcome source inventory, deterministic
               development adapters, and exact-semantics sampled predictors; no reconstruction
               execution, confirmation outcome access, or BENCH-019 analysis
  core/        gaussians2d/3d, observation2d (+ experimental CUDA query ext), camera, sh,
               metrics — shared math & containers
  image2gs/    stage 1: differentiable 2D splatting (serial + fused batch_views),
               native/StructSplat fitting, adapters; experimental CUDA ext in cuda_backend.py
  lift/        stage 2: gradient/depth/hybrid/carve/field, probabilistic_pipeline,
               compact_carve, beam_fusion,
               carrier_refinement (ADR-002 renderer-aware covariance repair; legacy
               opacity/SH0 repair controls are retired from the carrier path),
               surfel_init (cover-consistent covariance/opacity post-process),
               surfel_lift (ADR-XXXX closed-form covariance/opacity/colour), field_* and merge
  depth/       DepthBackend protocol, mock (tests), depth_anything (lazy), align (scale/shift)
  render/      dense Rasterizer (torch CPU ref, gsplat CUDA); sparse PointRasterizer (torch CPU)
  optim/       stage 3: RGB trainer.py; RGB-free fixed-topology compact_trainer.py;
               CPU classic density.py; CUDA gsplat strategies.py; active_set.py (opt-in
               priority-ranked update masking); ADR-YYYY init_density.py
               (three-channel appearance-preserving growth) + init_trust.py (trust schedule);
               carrier_schedule.py (ADR-002 compact fixed-topology two-phase maturation +
               strict fitting-view projected-center containment)
  data/        scenes/loaders plus compact_views.py capped view bundles; field_inputs.py
               explicit compact train/heldout seam; reconstruction_inputs.py fixed-topology seam
  carrier_pipeline.py  compact-only Beam -> corrected covariance -> fixed-topology carrier
               sequence; accepts ReconstructionInputs only
  pipeline.py  strict-split legacy orchestration + image-free run_field_pipeline and opt-in
               run_probabilistic_field_pipeline; lazily
               reexports the compact run_carrier_pipeline; visualize.py previews;
               viewer.py browser UI; live.py igsv live-training bridge; cli.py CLI including lift-field
tests/         CPU-only pytest suite; conftest.py has seeding + tiny-scene fixtures
experiments/   active task/program registry, prospective reviews, local-data seals, and shared
               task/review/metrics templates
benchmarks/    reusable run.py plus immutable legacy drivers and append-only result/audit evidence
docs/          ARCHITECTURE, RESEARCH (SOTA survey), RESEARCH_LOOP, ROADMAP, BENCHMARKS,
               EXPERIMENTS, AGENT_WORKFLOW.md; tasks/ closed agent-work records;
               DESIGN_field_lift.md + DESIGN_probabilistic_field_pipeline.md design notes;
               TASK_* per-protocol notes;
               THREE_ARM_EXPERIMENT_PROGRAM.md active claim-arm design and math;
               ADR-XXXX-surfel-lift.md and ADR-YYYY-init-preserving-densification.md (the
               init-parameter and schedule decisions); ADR-002-carrier-refinement.md
               (compact-only carrier policy, corrected math, stage ablations, and containment);
               20260725_claims_and_questions.md (which outcome makes which paper, above the
               init-value PREREG); PAPER_PLAN_beam_fusion.md (working title, contributions,
               required experiments/ablations/metrics)
ara/           claim + evidence ledger (see "Evidence and claims" below)
dataset/       calibrated local scenes required by Hard Rule 7
paper/         LaTeX working draft of the paper (compact 2D fields -> standard 3DGS ->
               tomographic init + refinement); red TODO markers = open evidence; see
               paper/README.md

scripts/       verify.sh plus the checkers (docs_sync, check_ara, check_script_layout,
               check_agent_workflow, check_results_bundle, experiment_contract); registered task
               drivers live in scripts/experiments/
.claude/skills/  rtgs-core, rtgs-task-workflow, rtgs-research-ideation, rtgs-verify, rtgs-bench,
               rtgs-docs-sync, rtgs-experiment, rtgs-review, realtime-gs-results-audit — task
               recipes
.agents/skills/  Agent Skills/Codex discovery symlinks for repo-prefixed skills
.agents/state/   one active substantial-task record; not a second backlog
.github/         hosted verification and the human pull-request review template
```

Skill names are prefixed (`rtgs-*`, `realtime-gs-*`) so they do not collide with another
repository's skills when several repos are open in one agent session. An unprefixed skill name
here would be shadowed by, or shadow, a sibling repository's skill of the same name.

## Skills (load by task)

| When you are… | Load |
|---|---|
| Starting or routing substantial repository work | `.claude/skills/rtgs-core/SKILL.md` |
| Opening, resuming, handing off, reviewing, or closing a substantial task | `.claude/skills/rtgs-task-workflow/SKILL.md` |
| Generating or auditing open-ended research directions | `.claude/skills/rtgs-research-ideation/SKILL.md` |
| Verifying a change | `.claude/skills/rtgs-verify/SKILL.md` |
| Running tracked benchmarks | `.claude/skills/rtgs-bench/SKILL.md` |
| Running a research experiment | `.claude/skills/rtgs-experiment/SKILL.md` |
| Reconciling docs and code | `.claude/skills/rtgs-docs-sync/SKILL.md` |
| Reviewing a diff before commit | `.claude/skills/rtgs-review/SKILL.md` |
| Auditing claims, evidence, or a results-bearing change | `.claude/skills/realtime-gs-results-audit/SKILL.md` |

Typical substantial flow: `rtgs-core` → `rtgs-task-workflow` → the task-specific skill →
`rtgs-review` → `rtgs-docs-sync` + `rtgs-verify`. Result-bearing work inserts
`rtgs-experiment` (run it) → `realtime-gs-results-audit` (referee it) before claim or default
promotion.

Run a results audit after every official experiment or benchmark session, before a
quantitative claim/default change, and before opening a confirmatory phase.

## Evidence and claims (`ara/`)

`ara/` is the Agent-Native Research Artifact: this repository's claim and evidence ledger. It is
not optional bookkeeping — it is where a number becomes a claim you are allowed to repeat.

```
ara/PAPER.md              layer index; start here
ara/logic/claims.md       the claim ledger: C<NN> rows, each with a falsification criterion
ara/logic/problem.md      the research problem statement
ara/logic/concepts.md     crystallized concepts
ara/logic/solution/       constraints, architecture, heuristics
ara/staging/observations.yaml  O<NN> observations awaiting promotion to a claim
ara/trace/                exploration_tree.yaml (N<NN> nodes) + pm_reasoning_log.yaml
ara/evidence/             raw proof index and tables
```

**When to touch it.** Before a quantitative or capability statement enters `README.md` or
`docs/`, it needs a `ara/logic/claims.md` row whose `Proof` cites a tracked artifact
(`benchmarks/results/*.json`, an `ara/evidence/` table, or a test node). A claim row carries
nine fields: `Statement`, `Status`, `Provenance`, `Crystallized via`, `Falsification criteria`,
`Proof`, `Dependencies`, `Tags`, `From staging`, plus an optional `Boundary` recording what the
evidence does *not* establish.

`Status` starts with a disposition word — `supported`, `refuted`, `untested`, `unavailable`,
`hypothesis`, `superseded`, `withdrawn` — optionally followed by a qualifier. A `supported` or
`refuted` claim must cite at least one artifact path that exists on disk.

`python scripts/check_ara.py` enforces the structure: required layer files, PAPER.md index
targets, claim-field completeness, status vocabulary, dependency resolution, proof-path
existence, and staging-ID resolution. It cannot judge whether a sentence overstates its
artifact — that is `rtgs-review` step 3, and `realtime-gs-results-audit` for anything promoted.

Negative results get rows too. A `refuted` claim with a clean falsification record is the most
valuable thing in the ledger.

## Working style for agents

- Start substantial work with `rtgs-core`, read `.agents/state/current-task.md`, and use
  `rtgs-task-workflow` to continue or open the one matching task without overwriting another.
- Before committing: `./scripts/verify.sh` must pass. CI (`.github/workflows/ci.yml`) runs
  the same steps on CPU.
- Before experiment code: create the task file, freeze an owner, obtain a distinct prospective
  review under `experiments/reviews/`, and run `python scripts/experiment_contract.py validate`.
  One agent owns one task id; write all local outputs below its exact run root and generate its
  page with the shared renderer. See `experiments/README.md`.
- Adding a lifting variant: implement `rtgs.lift.base.Lifter`, register it in
  `rtgs.lift.get_lifter`, add a pipeline test in `tests/test_pipeline.py`, a benchmark
  entry in `benchmarks/run.py`, and a row in `docs/ARCHITECTURE.md`.
- GPU-only work (gsplat parity, CUDA kernels) must ship with a CPU-skipped test
  (`@pytest.mark.cuda`) and a CPU-reference counterpart test where feasible.
- Keep test scenes tiny (≤64×64 images, ≤300 gaussians, ≤200 iters). The suite must stay
  under ~3 minutes on a 4-core CPU box.
- Before closing a research branch, run its frozen production-path interaction on a calibrated
  frame in `dataset/`, preserve held-out cameras for reporting only, save the viewer-ready PLYs
  and previews, launch the viewer, and report the exact viewer command with the metrics.
- Literature context and "what we reuse from where" lives in `docs/RESEARCH.md` — read it
  before redesigning any stage.

`AGENTS.md` (for other agent harnesses) points here; this file is canonical.
