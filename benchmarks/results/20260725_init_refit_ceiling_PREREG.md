# Do the Beam Fusion means carry the value? A refit-ceiling 2x2 — preregistration

Date: 2026-07-25 (frozen before any outcome of this protocol, and before the
`20260725_init_cost_to_target` run was read)

## Why this exists

The initialization work so far repaired the non-mean parameters with an **analytic** rule: a cover
sigma at a fixed ratio, optionally an oriented frame, optionally a coverage-derived opacity. Every
variant landed small (`cover-iso` +0.271 dB fixed-topology, +0.071 dB at the density endpoint), and
repairing *more* parameters made it worse (`cover-surfel-op` −0.091 dB).

Two very different explanations fit that pattern, and nothing on disk separates them:

- **E1 — the estimator is bad.** The means are good; the analytic cover rule is simply a poor way
  to set scales, rotations, opacity, and colour. A better estimate would unlock the initialization.
- **E2 — the means do not carry the value.** No setting of the non-mean parameters helps much,
  because position is not what limits this pipeline.

This protocol separates them by replacing the analytic rule with a **direct optimization** of the
non-mean parameters against the training images, means held exactly frozen. That is the ceiling of
"good means, everything else correct" — it cannot be beaten by any analytic estimator of the same
parameters. Verified mechanism: `lr_means = 0.0` with `densify = False` moves means by exactly
`0.0` while scales, opacity, and SH train.

## The 2x2

Crossing **means** with **how the non-mean parameters are set** is what makes the two explanations
separable. `random` means come from the same construction as the cost-to-target protocol: uniform
in the Beam Fusion bounding box, grey SH, isotropic extent at the median nearest-neighbour spacing.

| arm | means | non-mean parameters | role |
|---|---|---|---|
| `ci` | Beam Fusion | Beam Fusion's own | repository default reference |
| `cover-iso` | Beam Fusion | analytic cover rule | current best analytic arm |
| `ci-refit` | Beam Fusion | **optimized, means frozen** | E1's ceiling |
| `random` | uniform in box | analytic isotropic | no-prior reference |
| `random-refit` | uniform in box | **optimized, means frozen** | the separator |

`random-refit` is the arm that makes this protocol decisive. If optimizing the non-mean parameters
lifts random points just as much as it lifts Beam Fusion points, then the refit — not the means —
is doing the work, and E2 holds.

## Protocol

- `frame_00009`, downscale 4 (~938x410), train `[0,3,6,9,12,15,18,21]`, held-out `[1,13,25]`
  reporting only, `C1004` extrapolative and reported separately. Identical to the two preceding
  protocols.
- gsplat rasterizer, CUDA, seed 0, `n_init = 5,000`, SH degree 3, matched hard budget 15,000.
- **Phase A (refit arms only):** 300 steps, `lr_means = 0.0`, `densify = False`.
- **Phase B (all arms):** the remaining steps under the controller, using `iteration_offset` so the
  schedules see global step numbers.
- **Total is 7,000 steps for every arm.** The refit arms get 300 refit + 6,700 controlled; the
  non-refit arms get 7,000 controlled. The refit is spent *from* the budget, never added to it —
  otherwise the refit arms would be bought a head start rather than earning one.
- Controller: **`mcmc`** is primary (it was the strongest controller measured: +0.62 dB on the
  control arm over DefaultStrategy). `density` is run as a secondary for comparability with the
  two preceding protocols.
- Held-out metrics every 250 steps, in global step numbers.

## Outcome variables

There are **two co-primaries**, because the `20260725_init_cost_to_target` run showed they can
disagree: in its primary cell `cover-iso` reached 21.0 dB in 500 steps against `ci`'s 750 (a step
win) but took 14.5 s against `ci`'s 10.6 s (a time loss), because its primitives are ~4x larger in
sigma and cost ~2x more per step to rasterize. A protocol that scores only steps would call that a
success; the stated goal is wall-clock speed, so scoring only steps would be measuring the wrong
thing.

- **P1, steps to target**: `steps_to_target(arm, T)`, the first evaluated global step at which
  held-out foreground PSNR >= T, for T in {19.0, 20.0, 21.0} dB. Global steps include Phase A.
- **P2, train seconds to target**: the training wall-clock at that same crossing, taken from the
  trainer's own `history["elapsed"]`, which **excludes** checkpoint-callback time. Evaluation
  cadence is identical across arms, but eval cost is not part of what an initialization changes and
  must not be charged to it. Phase A seconds count toward the refit arms' total.

Secondary: held-out foreground PSNR at step 7,000, and the primitive count at each crossing.

## Preregistered decision

Both comparisons are evaluated on the **`mcmc`** controller at **T = 21.0 dB**, and each is scored
**twice** — once on P1 (steps) and once on P2 (train seconds), using the same thresholds. A gate is
reported as passed only when it passes on **both**; a split verdict (steps win, time loss, or the
reverse) is reported as **split** and promotes nothing, because it means the arm trades one cost
for the other rather than reducing cost.

**R1 — is the analytic rule leaving value on the table?** `ci-refit` versus `cover-iso`, step ratio
`steps_to_target(ci-refit) / steps_to_target(cover-iso)`:

- **G-R1a (estimator was the problem, E1)** — ratio <= 0.75.
- **G-R1b (no)** — ratio in (0.75, 1.33).
- **G-R1c (refit hurts)** — ratio >= 1.33.

**R2 — do the means carry the value?** `ci-refit` versus `random-refit`, step ratio
`steps_to_target(ci-refit) / steps_to_target(random-refit)`. Both arms have their non-mean
parameters optimized, so a difference is attributable to the means:

- **G-R2a (means are material)** — ratio <= 0.5, or `random-refit` fails to reach 21.0 dB within
  7,000 steps while `ci-refit` reaches it.
- **G-R2b (means are not material, E2)** — ratio in (0.75, 1.33).
- **G-R2c (random means are better)** — ratio >= 1.33.

A ratio in (0.5, 0.75] is recorded as "directional, sub-threshold" and promotes nothing.

**Guardrail** — final held-out outside-mask alpha <= 0.05 for every arm.

## What each outcome would mean

- **G-R1a**: the cover rule is a bad estimator and the initialization line is worth continuing —
  with a better estimator, not a better heuristic.
- **G-R1b and G-R2b together**: neither the estimator nor the means are the limiting factor. The
  lifting stage's contribution to final quality would be effectively refuted on this scene, and the
  pipeline thesis would have to rest on something other than initialization quality.
- **G-R2a**: Beam Fusion means are doing real work once the other parameters are set properly —
  which would make the analytic rule, not the lift, the thing to replace.

## Interpretation limits fixed in advance

One scene, one seed, one capture. 300 refit steps is a single point on a curve that was not swept;
a null under R1 bounds what *this* refit length achieves, not every possible refit. SH degree 3 is
retained for comparability. **No default change is authorized by this run under any outcome.**

## Commands

```bash
PYTHONUNBUFFERED=1 python benchmarks/gpu_init_refit_ceiling.py \
  --protocol benchmarks/results/20260725_init_refit_ceiling_PREREG.md \
  --out runs/gpu_init_refit_ceiling
```
