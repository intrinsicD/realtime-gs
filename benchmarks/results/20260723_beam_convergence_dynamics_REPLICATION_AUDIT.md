# Janelle beam-convergence dynamics — independent results audit

## Verdict

**Accept the reset/survival/dilution mechanism as a reproducible reduced-scale Janelle result;
narrow the initializer and production claims.** Two current ADC executions are scientifically
identical, the fixed-topology endpoints reproduce the prior rounded values, and all 66 saved PLYs
reload finite with their recorded counts. Periodic opacity resets really do collapse rendered
support, most original beam rows survive with small displacement, and those rows become a minority
after density growth.

Do not call the fixed-topology delta a position-only effect: beam and random differ in means,
quaternions, scales/covariances, and SH/color. Do not retain the original exact ADC endpoint counts:
their git-ignored raw artifacts and executed untracked harness were not preserved, and the current
bit-repeatable run differs. No result here tests the full 26-view CUDA gsplat schedule, held-out
cameras, generalization, convergence to a global optimum, or a production default.

The machine-readable audit SHA-256 is
`6e84dee111456076948ea67570e836851af223696e5a3f573e41951f86f91ad3`.

## Claim table

| Claim | Kind and scope | Disposition | Independent evidence |
| --- | --- | --- | --- |
| Opacity resets repeatedly collapse Beam-ADC support. | Measured, real-data reduced CPU screen | **Confirm.** | At steps 100/200/300/400/500, all opacities are below 0.02, the confident set is empty, and alpha-IoU is 0.000 for Beam-ADC. |
| The reset mechanism is beam-specific. | Causal | **Retire.** | Random-ADC has the same five empty-confident-set resets and alpha-IoU ≤0.000287. |
| Refinement destroys the beam initialization geometrically. | Mechanism | **Retire for this screen.** | 740/800 originals survive; their final displacement is 0.01449 mean / 0.02162 p90 world units. |
| Density control dilutes the original beam rows. | Accounting | **Confirm narrowly.** | Beam grows 800→4,255; 740 originals are 17.39% of the endpoint. Surgery accounting independently closes with 4,440 cumulative newborn rows and 985 removed split parents. “Original” excludes descendants. |
| Beam beats random under fixed topology. | Comparative, full initializer packages | **Confirm.** | Equal 800-count endpoints differ by +2.39419 dB FG PSNR and +0.16342 alpha-IoU. |
| Correct position alone causes the fixed-topology gain. | Causal attribution | **Retire / not isolated.** | Only count, optimizer, targets, and schedule are matched. Means, quaternions, scales, and SH differ; opacity alone is equal. |
| The originally logged exact ADC endpoints are reproducible. | Reproducibility | **Retire.** | Two current executions agree exactly with each other but end at 4,255/1,198 rather than 4,390/1,288. Original raw artifacts are absent. |
| The qualitative ADC dynamics are reproducible. | Reproducibility, same environment | **Confirm.** | Complete normalized scientific-record hashes and final PLY hashes match exactly across both current ADC runs. |
| This explains production gsplat convergence or held-out quality. | Production/generalization | **Not tested.** | Eight fitted views, downscale 32, Torch CPU/classic density, 1,000 steps, and no held-out cameras. |

## Protocol, chronology, and isolation

The protocol and outcome were already public in commit `d8948eb`; this is not a blinded
preregistration. The replication launched from clean revision
`c2a7e120a5cafdcf22d4bff6f5b9868b860eb1df` using the then-current harness SHA-256
`bbfe4172958af8f1188999f0eb1d4c41dccef2299b40ff93909f65e8dcf17991`. The checked-in compact
bundle is bound by a 27-file set digest
`5811b08c5d37d6d4e797e9e2aab18d9a6f420266041bb9b874ec380a43c507f2`.

All eight selected views enter training. Four of those same fitted views supply checkpoint
metrics. There is no validation or held-out role, so the result is correctly limited to mechanism
diagnosis. The beam placement uses data-derived colors/covariances while random starts gray and
isotropic; the 2×2 holds each initializer fixed across density modes, making the within-initializer
ADC-vs-fixed mechanism contrast useful, but it does not isolate placement across initializers.

The original logged run named revision `cef1b2c` plus an untracked harness and kept results only in
a git-ignored run directory. Those bytes are now absent. The current committed harness may well
match the final untracked source, but there is no preserved execution-time hash proving that.
Consequently the prior exact ADC counts are unbound historical observations, not a replay target.

## Verification

Ruff check, Ruff format check, docs-sync, `git diff --check`, and the 114-test focused collection
covering beam dynamics, beam fusion, optimization, and full compact recovery pass (optional
CUDA/dependency cases skip normally). The complete `./scripts/verify.sh` CPU gate reaches 100% but
is not green: it retains exactly the same 16 unrelated fail-closed failures documented by the
2026-07-21 audits:

- nine historical harness checks reject the current system `libstdc++`, PyTorch source, or frozen
  native ABI binding; and
- seven G2SR diagnostic checks require the absent historical
  `runs/compact_masked_bundle_640_20260717/reconstruction_inputs/manifest.json`.

No threshold, source binding, or artifact requirement was weakened. None of the failures is in the
new dynamics harness, audit, optimizer, beam-fusion, or recovery paths.

## Independent checks

The audit reopened every raw JSON and PLY rather than copying console output. It verified:

- the frozen 2×2 parameters, view indices, seed, checkpoint grid, density schedule, and reset steps;
- identical initialization metrics across ADC/fixed modes for each initializer;
- endpoint bindings between each trajectory and summary;
- all 44 primary and 22 repeat PLYs finite, with valid opacity/quaternions and exact recorded counts;
- density-surgery accounting for clone, split, prune, birth, removal, and final count;
- exact same-environment ADC scientific trajectories and final PLY hashes;
- input and local artifact manifests; and
- all reported deltas from raw values.

All 132 checks pass. The primary raw `dynamics.json` files contain 30 non-standard `NaN` values:
three Chamfer fields at each of five empty-confident-set resets in each ADC arm. No model tensor or
metric is non-finite. The audit accepts only this exact pattern and normalizes it to `null` in the
strict tracked JSON. The harness now returns `None` for an empty Chamfer set and serializes with
`allow_nan=False`; focused tests cover both empty and nonempty cases.

## Corrections to interpretation

The robust result is a sawtooth caused by explicit opacity resets, not disappearance and
rediscovery of the original beam centers. The density controller does add a much larger
descendant/newborn population, but strict row identity labels all split children and clones as
newborns. Therefore “17% originals” proves population dilution, not that 83% of final geometry was
learned independently of the beam parents.

The beam package's equal-count advantage is real in this screen, but “surface placement causes
+2.4 dB” is too strong. A placement-only factorial would need matched color/SH, opacity,
scale/covariance, and orientation while swapping only centers (and ideally the reciprocal field
swaps). The production question additionally needs the full compact bundle, gsplat
DefaultStrategy, a matched top-K arm, and reporting-only held-out cameras.

## Viewer and skipped work

The successful viewer command in the result note returned HTTP 200 on `127.0.0.1:8784`. The CPU
viewer process had no `nvidia-smi` compute allocation and was stopped after the smoke. Its CPU/RAM
use is nonzero, so no performance claim follows.

No full 26-view/70k CUDA fit, matched top-K dynamics arm, held-out RGB evaluation, novel-view
geometry test, GPU timing, or global-convergence test was run. Those remain the exact evidence
needed to promote this mechanism screen.
