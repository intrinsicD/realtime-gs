# realtime-gs — agent guide

Research repository testing one idea: **make 3D Gaussian Splatting (3DGS) reconstruction fast
by first fitting every input image with 2D gaussians, then lifting those 2D gaussians into 3D
as the initialization for standard 3DGS optimization.**

Pipeline (see `docs/ARCHITECTURE.md` for the full design):

```
images ──► [1] image2gs: fit 2D gaussians per image (GaussianImage-style)
       ──► [2] lift: 2D→3D, three competing variants
              A. lift.gradient — multi-view photometric gradient descent on per-ray depths
              B. lift.depth    — feed-forward monocular depth (Depth Anything V2 / mock)
              C. lift.carve    — voxel color-consistency carving + merging along ray tunnels
       ──► [3] optim: standard 3DGS refinement + density control (gsplat on GPU)
```

## Hard rules (do not break these)

1. **CPU-first testability.** No module may require CUDA at import time. `gsplat`,
   `transformers`, and any GPU-only dependency are imported lazily inside functions and
   guarded. The pure-PyTorch reference rasterizer (`rtgs.render.torch_ref`) is the
   correctness anchor; the full test suite must pass on a CPU-only machine.
2. **Backends are pluggable.** Rasterizers implement `rtgs.render.base.Rasterizer`; depth
   estimators implement `rtgs.depth.base.DepthBackend`. New fast paths go behind these
   interfaces — never fork pipeline logic per-backend.
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

## Commands

```bash
# one-time setup (CPU box; on a GPU box add: pip install -e '.[cuda,depth]')
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]' \
    --extra-index-url https://download.pytorch.org/whl/cpu

./scripts/verify.sh          # lint + format check + tests + docs-sync  (run before every commit)
.venv/bin/pytest -q                       # tests only
.venv/bin/pytest -q -m "not slow"         # what CI runs
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/python scripts/docs_sync.py     # docs↔code consistency check
.venv/bin/python benchmarks/run.py --quick --update-docs   # refresh benchmarks
.venv/bin/rtgs --help                     # CLI: fit-images / lift / refine / run / render / bench
```

## Repository map

```
src/rtgs/
  core/        gaussians2d, gaussians3d, camera, sh, metrics — shared math & containers
  image2gs/    stage 1: differentiable 2D splatting + per-image fitting
  lift/        stage 2: base utilities + gradient.py / depth.py / carve.py / merge.py
  depth/       DepthBackend protocol, mock (tests), depth_anything (lazy), align (scale/shift)
  render/      Rasterizer protocol, torch_ref (CPU reference), gsplat_backend (CUDA)
  optim/       stage 3: trainer.py (3DGS loop), density.py (clone/split/prune)
  data/        synthetic.py (test scenes with GT), colmap.py (real scenes)
  pipeline.py  end-to-end orchestration;  cli.py  argparse CLI
tests/         CPU-only pytest suite; conftest.py has seeding + tiny-scene fixtures
benchmarks/    run.py harness + results/*.json
docs/          ARCHITECTURE, RESEARCH (SOTA survey), ROADMAP, BENCHMARKS, EXPERIMENTS
scripts/       verify.sh, docs_sync.py
.claude/skills/  verify, bench, docs-sync, experiment — task recipes for agents
```

## Working style for agents

- Before committing: `./scripts/verify.sh` must pass. CI (`.github/workflows/ci.yml`) runs
  the same steps on CPU.
- Adding a lifting variant: implement `rtgs.lift.base.Lifter`, register it in
  `rtgs.lift.get_lifter`, add a pipeline test in `tests/test_pipeline.py`, a benchmark
  entry in `benchmarks/run.py`, and a row in `docs/ARCHITECTURE.md`.
- GPU-only work (gsplat parity, CUDA kernels) must ship with a CPU-skipped test
  (`@pytest.mark.cuda`) and a CPU-reference counterpart test where feasible.
- Keep test scenes tiny (≤64×64 images, ≤300 gaussians, ≤200 iters). The suite must stay
  under ~3 minutes on a 4-core CPU box.
- Literature context and "what we reuse from where" lives in `docs/RESEARCH.md` — read it
  before redesigning any stage.

`AGENTS.md` (for other agent harnesses) points here; this file is canonical.
