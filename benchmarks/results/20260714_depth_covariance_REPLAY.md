# Depth-covariance experiment replay manifest

The six ablation runs were executed while the worktree evolved from clean revision `2dddca4`.
Their JSON records preserve the effective config, but `HEAD + git_dirty=true` alone cannot recover
that code. This manifest binds the results to a replayable final research patch.

## Source state

Start from clean revision `2dddca4`, then apply:

```bash
git apply benchmarks/results/20260714_depth_covariance_replay.patch
```

- Replay patch SHA-256: `2430b366a80e947233cc5afe474babbf9823cd6bcddfc2a3d0a263642220cd4f`
- `benchmarks/depth_covariance_ablation.py`: `67f5cb0beb6da3bd8be27e2261a66478b6f7049d717aad01516feabee9eb4912`
- `src/rtgs/lift/base.py`: `b19fa04733c42c1bb5c210e1ac2fced73b7fcbfe7b0d7521ec62fd1d68ba503d`
- `src/rtgs/lift/depth.py`: `19e2e59d8c8a32d1dcc1b86b79364c09f845c493956cb12ea94597fadd874021`

The patch's reverse application was checked against the completed worktree. Later script changes
only made robust gradients the default, tightened validation/provenance, and trimmed recorded
history; the replay commands below pass the effective gradient mode explicitly.

Environment: Python 3.12.9, PyTorch 2.9.0+cu128 with CUDA hidden, four CPU threads. Wall-clock
timings can vary; the claims use paired quality/geometry metrics.

## Effective replay commands

Iteration 1, raw-gradient seed-0 screen:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python benchmarks/depth_covariance_ablation.py \
  --seeds 0 --modes isotropic footprint surface --condition clean --refine-iters 0 \
  --no-merge --no-densify --no-robust-depth-gradients --output /tmp/iter1.json
```

Iteration 2, raw three-seed replication:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python benchmarks/depth_covariance_ablation.py \
  --seeds 0 1 2 --modes isotropic footprint surface --condition clean --refine-iters 0 \
  --no-merge --no-densify --no-robust-depth-gradients --output /tmp/iter2_raw.json
```

Iteration 2, validity-aware clean-depth refinement:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python benchmarks/depth_covariance_ablation.py \
  --seeds 0 1 2 --modes isotropic footprint surface --condition clean --refine-iters 60 \
  --no-merge --no-densify --robust-depth-gradients --output /tmp/iter2_robust.json
```

Iteration 3, perturbed-depth causal run:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python benchmarks/depth_covariance_ablation.py \
  --seeds 0 1 2 --modes isotropic footprint surface --condition blur-noise \
  --refine-iters 60 --no-merge --no-densify --robust-depth-gradients \
  --output /tmp/iter3_noise.json
```

Iteration 3 negative protocol result (density surgery on the final step):

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python benchmarks/depth_covariance_ablation.py \
  --seeds 0 1 2 --modes isotropic footprint surface --condition blur-noise \
  --refine-iters 80 --merge --densify --robust-depth-gradients \
  --output /tmp/iter3_no_recovery.json
```

Iteration 3 corrected production interaction (20 recovery steps):

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python benchmarks/depth_covariance_ablation.py \
  --seeds 0 1 2 --modes isotropic footprint surface --condition blur-noise \
  --refine-iters 100 --merge --densify --robust-depth-gradients \
  --output /tmp/iter3_recovery.json
```

Canonical post-change regression:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python benchmarks/run.py --quick --update-docs
```
