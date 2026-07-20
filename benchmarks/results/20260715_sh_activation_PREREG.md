# SH color-floor incidence and SMU-1 preregistration

## Chronology and scope

This protocol was frozen before any SH-floor incidence, suppressed-gradient statistic, or
activation-arm quality result was computed. Read-only code inspection established that learned
spherical-harmonic (SH) color is floored by `max(x, 0)` in the semantics-defining renderer, but it
did not establish how often the floor is active or whether smoothing it helps.

The 2026-07-12 through 2026-07-15 Scholar Inbox digest contained no paper directly testing a
smooth replacement for the 3DGS SH nonnegativity floor. Grassmannian Splatting I uses a soft clamp
for continuity in a different geometric denominator; that is motivation for measuring continuity,
not evidence for this color intervention. No external code or reported threshold is reused.

This is a fixed-topology CPU synthetic mechanism experiment. It cannot establish real-scene
benefit, CUDA/gsplat parity or speed, density-control compatibility, general 3DGS superiority, or a
production-default change.

## Pre-outcome terminology amendment (2026-07-15T18:34:08Z)

After this protocol was first written, the user clarified that by a smooth maximum they meant the
Smooth Maximum Unit (SMU) family. This clarification arrived before any audit statistic or arm
outcome was computed. Biswas et al., *Smooth Maximum Unit: Smooth Activation Function for Deep
Networks Using Smoothing Maximum Technique* (CVPR 2022, arXiv:2111.04682), call their erf-based
approximation from below `SMU` and their square-root approximation from above `SMU-1`. The frozen
`squareplus` formula below is exactly SMU-1 with the leaky branch fixed to `alpha=0` and
`mu=2*delta=2/255`. It is therefore renamed `smu1`; the numerical function, calibration, gates,
seeds, and all other protocol choices are unchanged. No exact-SMU arm is added because at
`alpha=0` it can produce small negative forward values, whereas this renderer's activation is the
nonnegative radiance floor being studied.

An independent protocol audit, also before any outcomes, tightened the attribution control. The
original straight-through expression inherited SMU-1's slight positive-side gradient attenuation,
so it was not specific to dead negative colors. The frozen control below instead keeps the identity
gradient for `x >= 0` and adds only SMU-1's negative-side gradient. This is a design correction, not
an outcome-dependent change. The same audit added an implementation seal and explicit floating
point tolerance to the already intended invariants.

## Question

Does the hard nonnegative SH color floor suppress a material amount of loss-directed gradient
during production-shaped refinement, and, only if so, can a tightly calibrated smooth maximum
improve held-out learning without harming geometry or a view-independent-color guardrail?

## Frozen activation definitions

Let `x` denote RGB after SH evaluation and the standard `+0.5` shift, before nonnegativity.

1. `hard`: `h(x) = max(x, 0)`, the unchanged default.
2. `smu1`: `s(x) = 0.5 * (x + sqrt(x^2 + mu^2))`.
3. `hard_forward_smu1_negative_gradient`: let
   `q(x) = where(x < 0, s(x), x)`, then return
   `h(x).detach() + (q(x) - q(x).detach())` using one cached `q(x)` tensor.

`mu = 2/255` (equivalently `delta = mu/2 = 1/255`) is fixed without a sweep. This is SMU-1 with
`alpha=0`, specialized to smooth `max(x, 0)`. For every finite `x`, SMU-1 is positive and smooth,
has a nonzero negative-side derivative, and differs from hard by at most `delta`. Because
front-to-back compositing weights sum to at most one, the fixed-parameter per-pixel RGB difference
is also bounded by `delta` in exact arithmetic. The straight-through arm has exactly the hard
forward value, identity derivative for positive inputs, and the SMU-1 derivative for negative
inputs; it is a negative-side attribution control, not a proposed renderer semantic.

The hard default must remain bit-exact. All non-hard modes are opt-in research controls.

## Frozen data and initialization

Seeds are `0, 1, 2`. Each condition uses 40 GT Gaussians, twelve 48x48 ring cameras, native
stage-1 fitting with 150 Gaussians per training image for 120 iterations, and the fixed global
split:

- training views `[0,1,2,4,5,6,8,9,10]`;
- held-out views `[3,7,11]`.

Only training images are fitted. Each condition/seed reuses one bit-identical fitted set and one
bit-identical pinned `DepthLifter` initialization across activation arms. The lifter consumes
synthetic metric training depth, uses the pinned robust surface covariance and normal merge, and
produces degree-0 SH which the trainer pads to degree 3.

The fully pinned stage-1 configuration is `FitConfig(n_gaussians=150,
max_gaussians=5000, iterations=120, backend="native", adaptive_density=True,
growth_waves=5, relocate_fraction=0, structsplat_renderer="auto", lr=0.01,
grad_init_mix=0.7, row_chunk=64, log_every=50, convergence_patience=0,
convergence_tol=0.05, convergence_check_every=25)`. Native fitting is fixed-count, so its adaptive
and maximum fields are recorded no-ops. The pinned lifter is `DepthLifter` with the synthetic
metric-depth backend, `sh_degree=0`, `min_weight=0.05`, `init_opacity=0.1`,
`normal_thickness=0.15`, `covariance_mode="surface"`, `isotropic_sigma=None`,
`robust_depth_gradients=True`, `merge=True`, and `merge_voxel_frac=0.01`.

Two paired scene conditions are frozen:

1. `diffuse`: the existing synthetic scene with degree-0, view-independent GT color. This is the
   production-shaped guardrail and intentionally contains no true higher-band target signal.
2. `view_dependent`: identical geometry, opacity, cameras, sparse points, and base RGB, but degree-1
   target SH with
   `sh[i,3,c] = -0.12 * (-1)^(i+c) / C1` and all other non-DC coefficients zero. Thus target RGB is
   `base_rgb + 0.12 * (-1)^(i+c) * direction_x`. Before any fit or training, the harness must assert
   that every GT Gaussian/camera/channel preactivation is finite and in `[0.03, 0.97]`. Target
   generation therefore never needs the hard floor while still requiring view-dependent SH.

Images and depths for the second condition are rerendered from its frozen GT with the reference
renderer. Geometry is unchanged, so the sparse-point input remains valid.

## Frozen refinement

Every arm uses:

- `TrainConfig(iterations=120, rasterizer="torch", device="cpu", densify=False)`;
- `lr_means=1.6e-4`, `lr_quats=1e-3`, `lr_scales=5e-3`, `lr_opacity=5e-2`,
  `lr_sh=2.5e-3`, and `lr_sh_rest=1.25e-4`, with the repository's per-field Adam
  implementation at `eps=1e-15` and means decay to 1% across the run;
- `ssim_lambda=0.2`, `target_sh_degree=3`, `sh_degree_interval=30`, `eval_every=30`;
- `use_masks=False`, `random_background=False`, `outside_alpha_lambda=0.01`,
  `mask_alpha_lambda=0.05`, `opacity_reg=None`, `scale_reg=None`, `packed=False`, and
  `antialiased=False`; `validate_render_finite=True` fails on any non-finite training color,
  alpha, or depth, and the mask/background coefficients are recorded no-ops;
- four Torch threads and the arm's scene seed as trainer seed.

Degrees 0, 1, 2, and 3 are active for 30 iterations each. Density is disabled to isolate color
activation under fixed primitive topology. The complete sampled training-view schedule is recorded
and must match exactly across arms.

## Phase A: hard-floor audit

Exactly one official hard-only audit is permitted. During the actual hard training backward pass,
the reference renderer exposes the visible preactivation `x`, activated color `h(x)`, and both
retained gradients. Let `u = dL/dh(x)` be the upstream activated-color gradient. For every sampled
training step with active SH degree at least 1, aggregate:

- `negative_fraction = count(x < 0) / count(x)`;
- `blocked_fraction = sum(|u| * 1[x < 0]) / sum(|u|)`;
- `recoverable_fraction = sum(|u| * 1[x < 0 and u < 0]) / sum(|u|)`, where `u < 0` means an
  identity-gradient descent step would raise the dead color toward the target;
- `smu1_recovered_fraction = sum(|u| * s'(x) * 1[x < 0 and u < 0]) / sum(|u|)`;
- positive-side SMU-1 attenuation, both as retained L1 mass
  `sum(|u| * s'(x) * 1[x > 0]) / sum(|u| * 1[x > 0])` and by frozen raw-margin bins
  `(0,0.01), [0.01,0.02), [0.02,0.05), [0.05,0.10), [0.10,inf)`;
- recoverable-gradient mass by frozen raw-margin bins
  `(-inf,-0.10), [-0.10,-0.05), [-0.05,-0.02), [-0.02,-0.01), [-0.01,0)`;
- counts, raw negative magnitude summaries, RGB channel, active degree, iteration, and view.

The actual hard preactivation gradient must be exactly zero wherever `x < 0`; positive finite
entries must agree with the upstream gradient. These are validity invariants, not outcome gates.
Every numerator and denominator is accumulated in float64. A seed or pool with zero or non-finite
total upstream-gradient L1 denominator is invalid. Pooled fractions are ratios of the three seeds'
raw summed numerators and denominators, not averages of seed-level percentages.

Phase B is authorized only when the `view_dependent` aggregate over degree-1-or-higher steps meets
all of the following in at least two of three seeds and in the three-seed pooled result:

1. visible negative-channel fraction is at least 1%;
2. recoverable blocked-gradient L1 fraction is at least 5%;
3. the fixed SMU-1 would recover at least 0.5% of total upstream gradient L1 mass;
4. every training view was sampled in the audited window; and
5. at least 10,000 visible Gaussian/channel observations were audited per seed.

The diffuse condition is reported but cannot open Phase B by itself. If this gate fails, stop: do
not run either candidate arm, change `delta`, change the gate, perturb initialization, or replace
seeds under this protocol.

## Phase B: paired activation ablation

If and only if the Phase-A decision authorizes it, exactly one official Phase-B run recreates each
condition/seed and verifies its scene, fitted-tensor, lifted-initialization, config, and hard-audit
hashes before training the two candidate arms. The Phase-A hard arm is the frozen baseline and is
not silently rerun or replaced.

Required invariants:

- initial Gaussian fields and optimizer inputs are bit-identical across all arms;
- hard and straight-through forward colors/renders are bit-identical at step zero;
- SMU-1's step-zero per-Gaussian RGB deviation from hard is at most `1/255 + 1e-7`;
- its rendered RGB deviation is at most `(1/255) * max_measured_contribution_sum + 1e-6`,
  where the harness records that contribution sum and also requires it to be at most `1 + 1e-6`;
- target-view schedules, active-degree schedules, initial/final primitive counts, and evaluation
  checkpoints match exactly;
- all parameters, renders, losses, and diagnostic histories are finite;
- default `eval_sh`, `TorchRasterizer`, and `TrainConfig` behavior remain hard and unchanged;
- fitting/lifting never receives held-out images, depths, or metrics.

### Primary evaluation

Final parameters from every arm are evaluated first with one common hard reference renderer. This
is the primary portability comparison and prevents a matched smooth forward from defining its own
success metric. Matched-activation train/held-out metrics are secondary.

For each held-out view, the frozen foreground is the GT hard-render alpha mask `alpha > 0.05`.
Primary PSNR is binary-foreground-weighted RGB PSNR after clamping prediction and target to `[0,1]`;
full-canvas and crop PSNR are secondary. SSIM is computed after masking both images by that GT mask
and cropping them with the repository's `masked_crop(..., margin_fraction=0.05)`. Predicted support
is hard-render alpha `> 0.05`; alpha IoU is intersection over union and foreground coverage is
intersection over GT-foreground count. Expected depth is `depth / clamp_min(alpha, 1e-6)` and its
RMSE divided by scene extent is evaluated on the intersection of predicted and GT supports. An
empty GT mask, empty depth intersection, or non-finite metric invalidates the arm. Metrics are first
averaged equally over the three held-out views within each seed, then seed means are averaged
equally. No pixel- or view-count-weighted pooling defines an outcome gate.

On `view_dependent`, `smu1` versus `hard` must satisfy every criterion:

1. mean held-out hard-render PSNR improves by at least 0.25 dB;
2. held-out hard-render PSNR improves in at least two of three seeds;
3. mean held-out SSIM regresses by no more than 0.002 and no seed regresses by more than 0.005;
4. expected-depth RMSE over scene extent regresses by no more than 2%; and
5. alpha IoU and foreground coverage each regress by no more than 0.02.

On `diffuse`, mean held-out hard-render PSNR may regress by at most 0.10 dB and no seed may regress
by more than 0.25 dB. This guardrail cannot rescue failure of the view-dependent primary gate.

Dead-gradient attribution additionally requires `hard_forward_smu1_negative_gradient` to improve
view-dependent held-out hard-render PSNR in at least two of three seeds and preserve at least half
of SMU-1's mean gain over hard. Training loss, matched-smooth metrics, full histories,
time-to-quality, raw SH margins, nearest-GT distances, and runtime are secondary and cannot rescue
a failed primary or attribution gate.

## Frozen interpretation and stopping

- Phase-A gate failure: the hard floor is not material under this protocol; stop smooth-color
  activation work and return to the separately proposed hard-support taper diagnostic.
- SMU-1 utility without straight-through support: classify the gain as forward remapping or
  bias, not evidence that dead-gradient recovery caused it.
- Straight-through support without SMU-1 utility: the suppressed-gradient mechanism exists,
  but this smooth nonnegative forward is rejected.
- Full utility and attribution pass: record a promising fixed-topology CPU synthetic mechanism and
  require a separately preregistered CUDA/gsplat parity, density-enabled interaction, and real
  calibrated replication before any default change.
- No outcome authorizes a `delta`, learning-rate, SH schedule, iteration-count, seed, loss, or
  threshold sweep under this protocol.

Permitted positive wording is limited to: "SMU-1 training with a maximum fixed-parameter
perturbation of one 8-bit level improved fixed-topology CPU synthetic depth-initialized refinement,
with evidence consistent with recovering gradients
suppressed by the SH color floor." No stronger claim is permitted.

## Artifacts and commands

Before Phase A, the complete audit and candidate implementation, decision logic, focused tests,
preregistration, and loaded repository source aggregate are sealed in a manifest after focused and
full repository verification pass. Phase A and Phase B must assert the same seal. Any code or
protocol change after Phase A invalidates it and requires a new protocol and complete rerun;
Phase-B code may not be implemented or edited after opening the audit result.

The seal covers every `src/rtgs/**/*.py` and `tests/**/*.py` file plus this protocol, the harness,
and `pyproject.toml`; the harness rejects any altered seal-controlled path list and any loaded
repository source outside or different from that set. Seal, Phase A, and Phase B must also match on
Python/PyTorch, platform and CPU identity, Torch thread and deterministic settings,
`CUDA_VISIBLE_DEVICES=''`, `OMP_NUM_THREADS=4`, and `MKL_NUM_THREADS=4`.

Phase A and Phase B each have one repository-fixed, atomically exclusive attempt marker. Output and
companion-note collisions are checked before an attempt is claimed or any scene is prepared; a new
output filename cannot open a second attempt. The Phase-A scientist-review manifest must be strict
JSON with a passing verdict and must bind the exact audit SHA-256, seal SHA-256, and sealed source
aggregate. Before Phase B, the harness recomputes every diagnostic summary and the frozen gate from
the strict audit JSON rather than trusting its stored decision.

The harness must refuse to overwrite an output, serialize JSON with `allow_nan=False`, and bind the
full command, UTC timestamp, Git revision/status/tracked-diff hash, loaded repository source hashes
and aggregate, Python/PyTorch/platform/CPU/thread environment, deterministic settings, preregistration
hash, scene/image/camera/GT/fitted/init/final tensor hashes, split, effective configs, complete
schedules, raw per-seed diagnostics, summaries, decisions, wall time, and output SHA-256 in its
human result note.

Official commands have the forms:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  benchmarks/sh_activation_ablation.py seal --output \
  benchmarks/results/20260715_sh_activation_SEAL.json

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  benchmarks/sh_activation_ablation.py audit --seal \
  benchmarks/results/20260715_sh_activation_SEAL.json --output \
  benchmarks/results/<UTC>_cpu_sh_activation_audit.json

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  benchmarks/sh_activation_ablation.py ablate --audit \
  benchmarks/results/<UTC>_cpu_sh_activation_audit.json --phase-a-review \
  benchmarks/results/<UTC>_cpu_sh_activation_audit_AUDIT.json --seal \
  benchmarks/results/20260715_sh_activation_SEAL.json --output \
  benchmarks/results/<UTC>_cpu_sh_activation_ablation.json
```

Smoke tests may use tiny configs and untracked `/tmp` outputs. They cannot change any frozen
official default or inspect an official arm outcome before its phase is authorized.

Every official result is logged with its date in `docs/EXPERIMENTS.md` and receives an independent
scientist audit under the repository's result-audit protocol. Phase A must pass that audit before
its gate decision can authorize Phase B; no research, roadmap, or default claim is updated before
the applicable audit.
