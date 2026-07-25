# Who carries the final image — the initialization or the newborns? (post-hoc diagnostic)

Date: 2026-07-24

Kind: **post-hoc diagnostic**, not a preregistered comparison. It localizes a mechanism and
selects nothing — no rule, no default, no hyperparameter.

Machine-readable result: `runs/beam_surfel_birth_attribution_20260724/summary.json`

## Question

The matched-capacity result showed the cover-consistent initialization wins at an equal primitive
budget, but not *why*. Two very different stories fit the same endpoint: the initial Gaussians are
refined into the answer and births only patch residual holes; or the initial Gaussians are
vestigial and the answer is rebuilt by birth, making the initialization a mere seeding heuristic.

## Method

Every initial row's physical identity is carried through clone/split/prune surgery. A clone
appends a newborn and keeps its parent; a split **replaces** the parent with two newborns, so a
split original stops counting as a surviving original — the strict reading, and the one that makes
"the originals are still here" falsifiable. Three subsets of the *same* final model are then
rendered on the held-out cameras: everything, surviving originals only, newborns only.

`ci` and `surfel`, seed 0, matched budget 2,400, 1,000 steps, otherwise the frozen protocol.

**Alpha compositing is not additive, so the two subsets are not a decomposition of the whole.**
Removing primitives changes occlusion and total alpha, which depresses subset PSNR structurally.
The statements this design *can* falsify are "originals alone already reach the endpoint" and
"newborns alone already reach the endpoint", and alpha-IoU is the more interpretable subset
statistic than PSNR.

## Result

Split boundary is `0.01 * extent = 0.02229` world sigma: at or below it the controller can only
**clone in place**; above it, it **splits**, displacing each child by a draw from the parent's own
covariance.

| | `ci` | `surfel` |
|---|---:|---:|
| final composition | 767 originals + 1,633 newborns | 639 originals + 1,648 newborns |
| held-out FG PSNR — all | 21.678 | **22.186** |
| held-out FG PSNR — originals only | 13.458 | 14.182 |
| held-out FG PSNR — newborns only | **21.459** | 20.649 |
| held-out α-IoU — all | 0.9205 | 0.9241 |
| held-out α-IoU — originals only | 0.4385 (767 prims) | **0.7355 (639 prims)** |
| held-out α-IoU — newborns only | **0.9250** | 0.8413 |
| survivor displacement, world | 0.01515 median / 0.02287 p90 | 0.01843 / 0.02955 |
| survivor displacement, in units of its own initial sigma | **2.015 median / 4.338 p90** | **0.815 / 1.387** |
| final median sigma_max — originals / newborns | 0.01660 / **0.03135** | **0.05847** / 0.02915 |
| final median opacity — originals / newborns | 0.0317 / 0.0737 | 0.0535 / 0.0487 |
| originals split-eligible, first density event | **2.8%** | **63.2%** |
| originals split-eligible, last density event | 24.0% | 97.2% |

## Reading

**In the control the answer is rebuilt by birth.** Newborns alone reach 21.459 dB and α-IoU
0.9250, matching the complete model (21.678 / 0.9205) — the α-IoU is even marginally *higher*
without the originals. The 767 surviving originals alone give 13.458 dB and 0.4385, barely above
the 11.798 dB initialization. They also end **smaller** than their own descendants (0.01660 vs
0.03135) and at **lower** opacity (0.0317 vs 0.0737). The beam initialization functions as a
seeding heuristic whose content is then discarded.

**In the treatment neither subset alone reproduces the endpoint.** Originals give 14.182 / 0.7355,
newborns 20.649 / 0.8413, the whole 22.186 / 0.9241 — both subsets are materially below the whole,
which is the signature of collaboration rather than replacement. The comparison that is not
confounded by count is the originals' α-IoU: **0.7355 from 639 primitives versus 0.4385 from 767**.
The treatment's originals hold substantially more of the object with *fewer* primitives. They also
remain the largest primitives in the model (0.05847 vs the newborns' 0.02915), i.e. the
load-bearing surface, with the newborns adding detail on top.

**The positional initialization is preserved in one arm and abandoned in the other.** Absolute
motion is similar — 0.01515 versus 0.01843 world units, both roughly 0.4x the 0.0409 median
inter-primitive spacing. What differs is whether that motion stays inside the primitive's own
support: the control's survivors move a median **2.0x their own sigma** (p90 4.3x), leaving their
own footprint entirely, while the treatment's move **0.8x** (p90 1.4x) and stay inside it. A
primitive whose optimum lies outside its own support has almost no gradient pointing there and has
to be dragged; one whose optimum lies inside is refined smoothly. Note this normalization is
mechanically kinder to the larger primitives, so the honest statement is the conditional one, not
"the treatment moved less".

**Split was structurally unavailable to the control.** At the first density event only **2.8%** of
the control's originals were above the split boundary; even after 500 steps of scale growth it
reached only 24.0%. Everything else could only be **cloned in place** — a duplicate of a too-small
Gaussian at the same location, which adds optical mass but does not fill space. In the treatment
**63.2%** were split-eligible immediately, rising to **97.2%**. Split is the operator that
displaces children by a draw from the parent's covariance, and for a surfel that draw lies in the
tangent plane; it is the mechanism by which good positions actually propagate into surface
coverage, and in the control it was closed.

This is also why the earlier participation counts pointed the "wrong" way. The control's originals
met the densification criterion far more often (0.6125 vs 0.2550) precisely because they were
under-sized — the screen-space positional gradient scales as `1/sigma` — and each of those firings
produced a clone in place rather than a split. High participation was the symptom.

## Limits

One scene, one seed, one device, CPU reference rasterizer, downscale 32, classic CPU controller
rather than production CUDA gsplat. Subset renders are self-consistent sub-models, not an additive
attribution. Split-eligibility is read from public controller state before each scheduled round and
is a bound on what the controller *can* do to a row, not a count of what it did. No claim about
wall-clock or steps-to-target is made here: convergence speed was not the measured quantity.

## Command

```bash
.venv/bin/python benchmarks/beam_surfel_birth_attribution.py \
  --out runs/beam_surfel_birth_attribution_20260724
```
