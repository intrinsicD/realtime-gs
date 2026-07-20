# CPU baseline and replay audit

Audit time: 2026-07-14T23:00+02:00. Source revision `2dddca4`; the research worktree was
already dirty. No source, documentation, benchmark-result, or test file was changed by this
audit.

## Environment

- CPU: Intel Core i9-11900KF, 8 physical cores / 16 hardware threads, 62 GiB RAM.
- Python 3.12.9; PyTorch 2.9.0+cu128; PyTorch intra-op/inter-op threads 8/8.
- An RTX 3050 was visible normally, so CPU checks explicitly used
  `CUDA_VISIBLE_DEVICES=''`.
- The virtual environment uses system site packages; CPU results therefore bind to this
  environment rather than a hermetic lockfile.

## Verification

```bash
/usr/bin/time -f 'VERIFY_WALL=%e VERIFY_USER=%U VERIFY_SYS=%S VERIFY_MAXRSS_KB=%M' \
  env CUDA_VISIBLE_DEVICES='' ./scripts/verify.sh
```

Ruff lint passed, all 58 files passed format checking, docs sync passed, and tests reported
95 passed / 5 CUDA-skipped out of 100 collected. Wall/user/system time was
20.88/148.42/2.30 seconds; peak RSS was 926,292 KiB. A standalone
`pytest -m 'not slow' -ra` confirmation took 20.52 seconds wall (19.22 seconds reported by
pytest) and produced the same 95/5 result. No failures occurred.

## No-write CPU smoke

```bash
/usr/bin/time -f 'SMOKE_WALL=%e SMOKE_USER=%U SMOKE_SYS=%S SMOKE_MAXRSS_KB=%M' \
  env CUDA_VISIBLE_DEVICES='' .venv/bin/python benchmarks/run.py --quick --smoke
```

The smoke configuration was 15 GT Gaussians, six 24x24 views, 40 fitted Gaussians, 25 fit
steps, and 15 refinement steps. It completed in 2.66 seconds wall (9.22 user, 0.23 system;
792,972 KiB peak RSS). Stage-1 fitting reached 15.394 dB at 42.71 iterations/s; the reference
renderer reached 1,057.73 frames/s. Depth initialized at 21.220 dB with 188 primitives and
ended at 21.713 dB in 0.534 seconds total. The synthetic-SfM baseline initialized at 23.535 dB
with 200 primitives and ended at 28.763 dB in 0.529 seconds total. Synthetic SfM is privileged
by GT-derived points, so this is a regression signal rather than a real-scene ranking.

## Canonical workload bottlenecks

The source-bound CPU result `benchmarks/results/20260714T200932Z_cpu.json` reports the
40-Gaussian, twelve-view, 48x48, 120-fit/150-refine workload. Depth is the fastest learned
route at 26.13 seconds total (21.00 to 32.95 dB); carve reaches the best final score at 33.07 dB
in 35.80 seconds. Hybrid and gradient cost 49.85 and 55.85 seconds. Refinement costs depth
21.25 seconds, carve 30.58, and random 39.54; gradient lifting itself costs 17.13 seconds.
Final primitive counts are SfM 1,386, depth 3,087, carve 3,825, hybrid 4,040, gradient 4,159,
and random 4,428.

## Replay integrity

The replay patch and three source files exactly matched every SHA-256 value printed in
`benchmarks/results/20260714_depth_covariance_REPLAY.md`. Both checks exited zero:

```bash
git apply --check --reverse benchmarks/results/20260714_depth_covariance_replay.patch
git apply --check --cached benchmarks/results/20260714_depth_covariance_replay.patch
```

The patch therefore reverses from the completed worktree and applies to the clean index at
revision `2dddca4` as claimed.
