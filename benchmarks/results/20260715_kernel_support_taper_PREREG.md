# Hard kernel-support incidence and C1 taper preregistration

## Chronology, literature grounding, and scope

This protocol was frozen at `2026-07-15T21:34:00+02:00`, before implementing or computing any
kernel-support diagnostic and before running any taper training arm. It follows the failed,
independently audited SH color-floor gate: the next distinct hard operation in the
semantics-defining reference rasterizer is the per-pixel EWA support
`exp(-q/2) * 1[q < 12]`, where `q` is squared Mahalanobis distance.

The 2026-07-12 through 2026-07-15 Scholar Inbox digest contained no paper directly testing a
smooth spatial-support replacement in a 3D Gaussian splatting rasterizer. Grassmannian Splatting I
([arXiv:2607.10489](https://arxiv.org/abs/2607.10489)) uses a smooth clamp for a different Schur
denominator and explicitly retains a standard 3DGS rasterizer; it is continuity precedent, not
evidence for this intervention. SplatCtrl
([arXiv:2607.08948](https://arxiv.org/abs/2607.08948)) uses continuous Gaussian distance fields for
control/collision queries, not raster support. No external threshold, result, or implementation is
reused.

This is a fixed-topology, visible-set-conditioned, CPU synthetic mechanism experiment. It cannot
establish real-scene benefit, CUDA/gsplat parity or speed, density-control compatibility, benefit
beyond the reference renderer's detached 3-sigma visibility gate, or a production-default change.
The near-plane test, alpha cap, depth order, and visibility culling remain unchanged and are not
smoothed in this experiment.

An independent pre-implementation protocol audit found no validity-critical change to the
outward-only C1 construction, frozen gates, or interpretation. It clarified that the current
trainer propagates color and alpha training losses through the kernel; depth is an evaluation
metric, not an active training-loss path. Before sealing, a focused test must also establish that
the pinned PyTorch alpha clamp propagates the expected gradient at its zero boundary. These are
factual/implementation checks made before any diagnostic outcome, not changes to the experiment.

## Question

Does the hard `q < 12` kernel support suppress a material amount of loss-directed gradient in the
immediately adjacent annulus during production-shaped refinement, and, only if so, can a tightly
bounded C1 compact tail improve held-out learning without changing the established interior kernel?

## Frozen kernel definitions

Let `C = 12`, `W = 4`, `t = (q-C)/W`, and
`S(t) = 1 - 3*t^2 + 2*t^3`. For finite squared Mahalanobis distance `q`:

1. `hard`: `h(q) = exp(-q/2)` for `q < C`, and zero otherwise. This is the unchanged default.
2. `c1_taper`: `s(q) = exp(-q/2)` for `q < C`; `exp(-q/2) * S(t)` for
   `C <= q < C+W`; and zero otherwise.
3. `hard_forward_c1_taper_gradient`: return
   `h(q).detach() + (s(q) - s(q).detach())` using one cached `s(q)` tensor.

The taper is C1 at both boundaries, is bit-identical to hard for `q < 12`, has compact support at
`q >= 16`, and adds at most `exp(-6) = 0.0024787521766663585` kernel weight. The straight-through
arm has exactly the hard forward value and the taper derivative; because the functions and their
derivatives agree for `q < 12`, it adds gradient only in the new annulus. It is an attribution
control, not proposed renderer semantics. No width or shape sweep is permitted.

The hard default must remain bit-exact. Non-hard modes are opt-in research controls supported only
by the Torch reference backend under this protocol; gsplat must reject them explicitly.

## Frozen data, initialization, and refinement

Seeds are `0, 1, 2`. The experiment reuses the complete scene, fit, lift, split, and refinement
definitions of `benchmarks/results/20260715_sh_activation_PREREG.md` at SHA-256
`5353c4aa37c13e280f0bf3761679424e0bb5e17b4e942a7ff36275e84be88c1f`, including:

- 40 GT Gaussians, twelve 48x48 ring cameras, training views
  `[0,1,2,4,5,6,8,9,10]`, and held-out views `[3,7,11]`;
- native stage-1 fitting with 150 Gaussians/image for 120 iterations and the pinned metric-depth
  `DepthLifter` surface-covariance initialization;
- paired `diffuse` and `view_dependent` scene conditions, with target generation and its
  preactivation validity assertion unchanged;
- 120 CPU Torch-reference refinement iterations, density disabled, the exact per-field learning
  rates and loss/configuration values in that protocol, and four Torch threads.

Every condition/seed reuses one bit-identical fitted set and one bit-identical lifted initialization
across arms. The sampled training-view and active-SH-degree schedules must match exactly. The
`diffuse` condition is the primary gate because the spatial-support mechanism does not require a
specialized color target; `view_dependent` is a prespecified replication/guardrail.

## Phase A: hard-support audit

Exactly one official hard-only audit is permitted. During the actual hard training backward pass,
the reference renderer exposes each visible Gaussian/pixel `q`, hard kernel `h(q)`, and their
retained gradients. Let `u = dL/dh(q)`, `r = dL/dq`, `I = {0 <= q < 12}`,
`B = {8 <= q < 12}`, and `A = {12 <= q < 16}`. For every sampled training step, aggregate in
float64:

- observation counts in `I`, `B`, `A`, and frozen q bins
  `[0,2), [2,4), [4,6), [6,8), [8,10), [10,12), [12,13), [13,14), [14,15), [15,16)`;
- `annulus_incidence = count(A) / count(I union A)`;
- `annulus_upstream_fraction = sum_A |u| / sum_(I union A) |u|`;
- `recoverable_annulus_fraction = sum_(A and u<0) |u| / sum_A |u|`, where `u<0` means increasing
  the currently zero kernel weight is locally loss-reducing;
- `recovered_total_ratio = sum_(A and u<0) |u * s'(q)| / sum_I |u * h'(q)|`;
- `recovered_boundary_ratio = sum_(A and u<0) |u * s'(q)| / sum_B |u * h'(q)|`;
- the same raw masses by q bin, iteration, view, and loss-summed kernel contribution; distinct
  Gaussian/view exposure counts may be reported as a secondary, non-gating cancellation check.

`u` is the gradient of the kernel tensor after opacity, alpha clamp, transmittance, and the active
color/alpha loss paths have propagated into it; thus it already includes whether a pair can affect
the actual training loss. Depth remains evaluation-only. Ratios pool raw sums, never seed
percentages. A zero/non-finite denominator is invalid.

Validity invariants, not outcome gates: all audited tensors and summaries are finite; `q >= -1e-6`;
the hard kernel is exactly zero for `q >= 12`; `r` is exactly zero there; for `q < 12`, `r` agrees
with `u * (-0.5*exp(-q/2))` within absolute `1e-6` and relative `1e-5`; every hard-training render
and parameter is finite. Diagnostic collection may retain tensors only until that step is reduced.

Phase B is authorized only when the `diffuse` aggregate meets every condition in at least two of
three seeds and in the three-seed pooled result:

1. `annulus_upstream_fraction >= 0.01`;
2. `recoverable_annulus_fraction >= 0.10`;
3. `recovered_total_ratio >= 0.001`;
4. `recovered_boundary_ratio >= 0.05`;
5. every training view was sampled; and
6. at least 100,000 `I union A` observations were audited per seed.

The `view_dependent` condition is fully reported but cannot rescue a failed diffuse gate. If the
gate fails, stop: do not run either candidate, alter `W`, lower a threshold, add seeds, perturb the
initialization, or change the loss under this protocol.

## Phase B: paired taper ablation

If and only if Phase A passes and an independent scientist-review manifest confirms the exact
artifact and seal, one Phase-B run recreates both conditions/seeds, verifies their scene/fitted/init
hashes against Phase A, and trains `c1_taper` and
`hard_forward_c1_taper_gradient`. The sealed Phase-A hard runs are the baseline and are not rerun.

Required invariants include bit-identical initial Gaussian fields, optimizer inputs, schedules and
primitive counts; exact hard/straight-through step-zero forward equality; taper/hard equality for
all `q<12`; maximum kernel deviation no greater than `exp(-6)+1e-7`; finite parameters/renders/
histories; unchanged hard defaults; and no held-out input to fitting, lifting, training, selection,
or stopping.

Final parameters are evaluated with one common hard reference renderer as the primary comparison;
matched-taper metrics are secondary. Held-out foreground PSNR, masked/cropped SSIM, alpha IoU,
foreground coverage, and normalized expected-depth RMSE use exactly the definitions in the
incorporated SH protocol and are averaged over held-out views then equally over seeds.

On `diffuse`, `c1_taper` versus hard must satisfy all of:

1. mean held-out common-hard foreground PSNR gain at least `0.10 dB`;
2. common-hard foreground PSNR improves in at least two of three seeds;
3. mean SSIM regression no more than `0.002` and no-seed regression more than `0.005`;
4. normalized expected-depth RMSE regression no more than `2%`; and
5. alpha IoU and foreground coverage regression no more than `0.02`.

On `view_dependent`, mean held-out common-hard foreground PSNR may regress by at most `0.10 dB`
and no seed by more than `0.25 dB`. This guardrail cannot rescue the diffuse primary gate.
Attribution additionally requires the hard-forward control to improve diffuse common-hard PSNR in
at least two seeds and retain at least half of the taper's mean gain. Training loss, matched-taper
metrics, runtime, and individual checkpoints are secondary and cannot rescue a failed primary or
attribution gate.

## Frozen interpretation and stopping

- Phase-A failure: the immediately adjacent support tail is not material under this CPU synthetic
  fixed-topology protocol; do not tune the taper. Inspect the separate detached visibility/culling
  gate next, without combining interventions.
- Taper utility without hard-forward support: classify any gain as a forward support change, not
  evidence that restored annulus gradients caused it.
- Hard-forward support without taper utility: the mechanism exists, but this C1 forward taper is
  rejected.
- Full utility and attribution pass: record a promising reference-renderer mechanism and require
  separately preregistered CUDA semantics/performance, density-enabled interaction, and real-scene
  replication before any default change.

No result authorizes a width/shape, learning-rate, loss, schedule, iteration, seed, cutoff, or
visibility-margin sweep. Permitted positive wording is limited to: "A fixed C1 tail of maximum
kernel weight `exp(-6)` improved fixed-topology CPU synthetic refinement under common-hard held-out
evaluation, with evidence consistent with recovering loss-directed gradient immediately outside
the reference kernel support."

## Artifacts and commands

Before Phase A, the complete audit and candidate implementation, decision logic, tests, this
protocol, and loaded repository source aggregate are sealed after focused and full CPU verification.
The seal covers every `src/rtgs/**/*.py` and `tests/**/*.py`, this protocol, the harness, and
`pyproject.toml`. It records exact source hashes, Python/PyTorch/platform/CPU/thread environment,
determinism, and the incorporated SH protocol hash. Loaded repository Python must be a sealed path.

Phase A and Phase B have separate fixed atomically exclusive attempt markers. Outputs and companion
notes are append-only and collision-checked before an attempt is claimed. A strict Phase-A review
JSON must bind the exact audit SHA-256, seal SHA-256, and source aggregate before Phase B. The
harness recomputes all summaries and gates from raw JSON. Any sealed-source/protocol change after
Phase A invalidates the artifact and cannot authorize Phase B.

Artifacts bind command, UTC time, Git revision/status/tracked-diff hash, environment, source and
loaded-source hashes, preregistration hash, scene/input/fitted/init/final tensor hashes, split,
effective configs, full schedules, raw summaries, decisions, wall time, and a SHA-256 note. JSON is
strict (`allow_nan=False`) and no output may be overwritten.

Official command forms are:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  benchmarks/kernel_support_taper_ablation.py seal --output \
  benchmarks/results/20260715_kernel_support_taper_SEAL.json

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  benchmarks/kernel_support_taper_ablation.py audit --seal \
  benchmarks/results/20260715_kernel_support_taper_SEAL.json --output \
  benchmarks/results/<UTC>_cpu_kernel_support_taper_audit.json

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  benchmarks/kernel_support_taper_ablation.py ablate --audit \
  benchmarks/results/<UTC>_cpu_kernel_support_taper_audit.json --phase-a-review \
  benchmarks/results/<UTC>_cpu_kernel_support_taper_audit_AUDIT.json --seal \
  benchmarks/results/20260715_kernel_support_taper_SEAL.json --output \
  benchmarks/results/<UTC>_cpu_kernel_support_taper_ablation.json
```

Smoke tests may use tiny configs and `/tmp` outputs but may not inspect an official outcome or
change a frozen scientific choice. Every official result receives an independent scientist audit
before docs/ARA claims or any continuation decision.
