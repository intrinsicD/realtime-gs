# Initialization as a cost reduction, not a quality ceiling — preregistration

Date: 2026-07-25 (frozen before any outcome of this protocol)

## Why this exists

`20260725_gpu_stage1_initialization_PREREG.md` returned its preregistered null: under gsplat
DefaultStrategy at a 3x budget, `cover-iso` beat `ci` by +0.071 dB held-out foreground at step
7,000, inside the 0.15 dB band. Two facts from that same run say the null is about the *outcome
variable*, not about initialization:

1. The effect exists and is large where the controller cannot spend budget — `fixed` topology
   gave +0.271 dB, and at step 1,000 `cover-iso` led by +0.254 dB while holding **7,406**
   primitives against the control's **9,500**.
2. The controller dwarfs the initialization. Swapping DefaultStrategy for MCMC moved the *control*
   arm by +0.62 dB (21.077 -> 21.699), roughly 9x the largest initialization effect measured.

Adaptive density control rebuilds the answer by birth given enough budget, which is why a better
seed converges to the same ceiling. Quality-at-a-fixed-step-count is therefore the one measure
that structurally cannot see an initialization benefit.

This protocol is follow-up (a) of the 2026-07-24 birth-attribution entry, which named it and was
never run: *"measure convergence cost directly — steps and wall-clock to a fixed held-out target —
since 'less optimization in total' is implied by these numbers but was never the measured
quantity."*

## The gap this closes

Every arm of every initialization experiment so far (`ci`, `cover-iso`, `cover-surfel`,
`cover-surfel-op`) is Beam Fusion with **bit-identical means**; `build_initializations` asserts it.
The repository has therefore never compared Beam Fusion against *not* lifting. The claim "the
lifted 2D gaussians are a good initialization" is currently unfalsified because it is untested.
This protocol adds the missing no-prior control.

## Arms

Means now differ between arms — that is the point of this protocol, and it is the difference from
the previous one.

| arm | means | covariance | colour | status |
|---|---|---|---|---|
| `random` | uniform in scene bounds, seed 0 | isotropic at median NN spacing | grey (SH DC 0.5) | no-prior control |
| `ci` | Beam Fusion | unchanged Beam CI | Beam Fusion | repository default |
| `cover-iso` | Beam Fusion | isotropic at the cover sigma | Beam Fusion | best fixed-topology arm |

`random` receives the same count, the same initial opacity (0.10), and an isotropic extent set
from its own median nearest-neighbour spacing, so it is handicapped only by lacking geometric and
photometric prior — not by extent or budget.

## Protocol

- `frame_00009` compact bundle, downscale 4 (~938x410), identical to the previous protocol.
- Train views `[0,3,6,9,12,15,18,21]`; held-out `[1,13,25]` = `C0004, C0025, C1004`, reporting
  only. `C1004` is extrapolative and reported separately.
- gsplat rasterizer, CUDA, seed 0, 7,000 steps, SH degree 3, `n_init = 5,000`.
- Controllers: `density` (gsplat Default) and `mcmc`, both at the matched hard budget 15,000.
- Held-out metrics evaluated every 250 steps (finer than the previous 500) so first-crossing
  resolution is 250 steps.

## Outcome variables

**Primary — cost to target.** `steps_to_target(arm, T)` = the first evaluated step at which
held-out foreground PSNR >= T, for the preregistered ladder **T in {19.0, 20.0, 21.0} dB**.
First-crossing is used precisely because held-out quality decays after its peak under
DefaultStrategy; a crossing is unaffected by the later decline. `inf` if never reached.

Reported alongside each crossing: wall-clock seconds, primitive count, and the same three
quantities for the `mcmc` controller.

**Secondary — quality at a constrained budget.** Held-out foreground PSNR at hard budgets
**1.0x** (5,000, no densification headroom) and **1.5x** (7,500), at step 7,000. This is where the
`fixed`-topology result predicts the initialization should matter most.

## Preregistered decision

Primary comparison A — does the cover-consistent extent reduce cost? `cover-iso` versus `ci`,
`density` controller, at **T = 21.0 dB**, on the step ratio `steps_to_target(cover-iso) /
steps_to_target(ci)`:

- **G-C1 (init reduces cost)** — ratio <= 0.75.
- **G-C2 (no cost benefit)** — ratio in (0.75, 1.33).
- **G-C3 (reversal)** — ratio >= 1.33.

Primary comparison B — is Beam Fusion better than no prior at all? `ci` versus `random`, same
controller and target:

- **G-B1 (lifting is material)** — `steps_to_target(ci) <= 0.5 * steps_to_target(random)`, or
  `random` fails to reach 21.0 dB within 7,000 steps while `ci` reaches it.
- **G-B2 (lifting is not material)** — ratio in (0.75, 1.33).
- **G-B3 (lifting hurts)** — ratio >= 1.33.

A ratio in (0.5, 0.75] for comparison B is recorded as "directional, sub-threshold" and promotes
nothing.

**Guardrail** — final held-out outside-mask alpha <= 0.05 for every arm, as before.

## What each outcome would mean

- **G-C1 and G-B1**: the initialization claim survives with its outcome variable corrected — the
  contribution is convergence cost, not final fidelity. This would be the first evidence in the
  repository that lifting beats not lifting.
- **G-B2**: Beam Fusion is not distinguishable from random points under a densifying controller.
  That would make the entire lift stage's value proposition rest on constrained-budget or
  fixed-topology regimes, and the pipeline thesis would need restating.
- **G-C2 with G-B1**: lifting matters, the covariance refinement does not.

## Interpretation limits fixed in advance

One scene, one seed, one capture. This measures cost transfer on the production stack, not
generalization across scenes. SH degree 3 is retained for comparability with the previous protocol
even though colour is known to absorb geometric error; an SH-0 replication is a follow-up, not
part of this decision. **No default change is authorized by this run under any outcome.**

## Commands

```bash
PYTHONUNBUFFERED=1 python benchmarks/gpu_init_cost_to_target.py \
  --protocol benchmarks/results/20260725_init_cost_to_target_PREREG.md \
  --out runs/gpu_init_cost_to_target
```
