# Cover-consistent surfel initialization on Janelle `frame_00009` — result

Date: 2026-07-24

Status: complete and independently audited (**70/70 checks**). The preregistered treatment
**fails three of five gates**, one of them in the direction opposite to the hypothesis. **No
default change is authorized.** The screen nevertheless produces a clean, well-attributed
mechanism separation that neither prior Beam covariance experiment could see.

Protocol: [`20260724_beam_surfel_init_PREREG.md`](20260724_beam_surfel_init_PREREG.md)

Machine-readable result: `runs/beam_surfel_init_20260724/summary.json`

Audit: `runs/beam_surfel_init_20260724/audit.json` (recomputed by
[`benchmarks/audit_beam_surfel_init.py`](../../benchmarks/audit_beam_surfel_init.py))

Results page: `runs/beam_surfel_init_20260724/index.html`

## Answer

Beam Fusion's covariance is not "approximate"; it answers a different question. Covariance
intersection estimates **where a component is**. The renderer needs **how much surface a
primitive covers**. Measured on the fresh `frame_00009` root and replicated on `frame_00008`:

| quantity | frame_00009 | frame_00008 |
|---|---:|---:|
| mean \|cos\| between fused short axis and kNN surface normal | 0.531 | — |
| mean \|cos\| between fused long axis and mean view direction | 0.540 | — |
| random-orientation baseline | 0.500 | 0.500 |
| median contributing-camera arc | 161.0° | — |
| fused sigma_max / *smallest* contributor footprint | 1.660 | 1.750 |
| fused sigma_max / *median* contributor footprint | 0.443 | 0.551 |
| Spearman(sigma_max, smallest footprint) | 0.564 | 0.685 |
| Spearman(sigma_max, median footprint) | 0.399 | 0.500 |
| fused sigma_max / kNN-3 spacing | **0.182** | **0.256** |
| 2D inputs per 3D output | 50.0 | 50.0 |

Three independent defects, none of which is "the covariance is imprecise":

1. **Orientation carries no surface information.** The contributing cameras span a median 161°
   arc, so no direction is systematically under-triangulated; the residual ~4:1 anisotropy is
   set by heterogeneous per-view footprints. Alignment with a local surface normal sits at the
   random baseline.
2. **A precision mean is dominated by the sharpest observation.** `Lambda = mean_k Lambda_k`
   makes every component inherit the tightest 2D Gaussian in its track, not the typical one —
   1.66–1.75× the smallest contributor footprint but 0.44–0.55× the median.
3. **The covariance never learned about decimation.** 40,000 fitted 2D Gaussians reduce to 800
   outputs, but the covariance still describes the un-decimated observation scale. The widest
   axis reaches 0.18–0.26× the distance to the component's own nearest neighbours, so the
   primitives cannot touch, let alone tile: 6.58 would have to overlap to reach alpha 0.5 at
   opacity 0.10.

## Treatment

`rtgs.lift.surfel_init.reconcile_covariances` keeps means, SH/color, and count bit-identical and
rebuilds extent, orientation, and opacity from derived conditions:

- kNN PCA over the component centers supplies the local frame, spacing, and planarity;
- `sigma_t = max(0.5 * spacing, resolution_floor)`, where `0.5` is fixed by the hexagonal-cover
  ripple (measured by direct summation: 81.5% at 0.30, 13.3% at 0.40, **1.25% at 0.50**, 0.31%
  at 0.55) and the floor is the per-component median contributor footprint from Beam's own CSR
  lineage. The floor was the binding constraint for **323 of 800** components, so the rule is
  genuinely two-sided rather than a rescaled kNN heuristic;
- `sigma_n` is the measured out-of-plane spread, raised to the localization sigma along that
  normal, then capped at `0.5 * sigma_t`;
- opacity inverts `1 - (1 - o)^S` for the local overlap `S = 2 pi sigma_t^2 / cell area`.

Realized on this root: median `sigma_t` 0.02387 (3.119× the prior median `sigma_max`), median
`sigma_n` 0.01181, median overlap 1.814 kernel units, median opacity 0.719. The control's own
sigma implies an overlap far below one, so its derived opacity saturates the 0.95 clamp — which
is what the `ci-op` arm is.

## Setup

- Fresh root `dataset/2025_03_07_stage_with_fabric/frame_00009`.
- Train views `[0,3,6,9,12,15,18,21]` = `C0001, C0006, C0012, C0019, C0022, C0028, C0031,
  C0039`; these alone feed Beam Fusion and refinement.
- **Held-out** `[1,13,25]` = `C0004, C0025, C1004`; reporting only. `C1004` is extrapolative.
- Beam Fusion seed 0, 800 outputs, minimum 3 views, 3-sigma gates, color 0.35/0.25, extent/100
  NMS voxel, opacity 0.10 — the unchanged 20260721/20260723 configuration. 5,848 contributor
  links, 7.31 views per component.
- Refinement: Torch CPU reference rasterizer, downscale 32, 1,000 steps, seed 0, identical loss
  and schedules. `fixed` = no topology change; `adc` = the classic controller frozen in the
  20260723 convergence-dynamics screen (start 20, stop 500, every 4, threshold 3e-3, cap 8,000).

Official command:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python benchmarks/beam_surfel_init.py \
  --protocol benchmarks/results/20260724_beam_surfel_init_PREREG.md \
  --out runs/beam_surfel_init_20260724
```

## Results — fixed topology, held-out cameras

| arm | init PSNR | init α-IoU | init α-in | init α-out | PSNR AUC | final PSNR | final α-IoU | final α-out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ci` | 11.7978 | 0.00213 | 0.1316 | 0.0037 | 19.4044 | 21.2741 | 0.9065 | 0.0488 |
| `ci-op` | 14.4893 | 0.64970 | 0.6448 | 0.0286 | 19.4395 | 21.2281 | 0.9045 | 0.0524 |
| `cover-iso` | 13.8767 | 0.51420 | 0.4768 | 0.0437 | **21.0774** | **21.8594** | **0.9261** | 0.0437 |
| `cover-iso-op` | **17.4458** | 0.69500 | **0.9235** | 0.1851 | 21.0256 | 21.5367 | 0.9221 | 0.0426 |
| `surfel` | 16.9586 | **0.73315** | 0.8908 | 0.1312 | 20.8934 | 21.6450 | 0.9247 | 0.0432 |

**The two defects have opposite effects, and only one of them matters for optimization.**
Opacity alone (`ci-op`) transforms the step-0 picture — alpha IoU 0.00213 → 0.64970, initial
PSNR +2.69 dB — and then contributes essentially nothing: AUC +0.18%, final PSNR −0.046 dB
versus the control. Extent alone (`cover-iso`) produces a *worse* initial image than `ci-op`
yet carries the entire optimization gain: AUC **+8.62%**, final **+0.585 dB**, final alpha IoU
+0.0196. The covariance was blocking refinement; the opacity was only blocking the initial
render. The 20260723 optical-thickness probe could not distinguish these because it was
render-only and never optimized.

Orientation is a second-order but consistent effect. At identical extent and opacity rule,
`surfel` leaks less than `cover-iso-op` (initial alpha-outside 0.1312 vs 0.1851) with higher
coverage (0.73315 vs 0.69500), which is what a tangent-aligned surfel should do at a silhouette
— it projects to a thin streak where an isotropic blob projects to a disc.

## Results — classic density control, held-out cameras

| arm | PSNR AUC | final PSNR | final α-IoU | final α-out | **final N** | qualified frac | cloned | split | originals alive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ci` | 18.7920 | 21.8745 | 0.9246 | 0.0350 | 5030 | 0.6125 | 3043 | 1188 | 767 |
| `ci-op` | 19.0800 | 21.9724 | 0.9277 | 0.0353 | 4699 | **0.7200** | 2990 | 910 | 774 |
| `cover-iso` | 19.9640 | 22.2455 | 0.9286 | 0.0455 | **2156** | 0.1725 | 704 | 653 | 675 |
| `cover-iso-op` | **20.0140** | **22.3279** | **0.9298** | 0.0424 | 2230 | 0.2000 | 767 | 668 | 668 |
| `surfel` | 19.9450 | 22.1855 | 0.9241 | 0.0441 | 2287 | 0.2550 | 818 | 673 | 639 |

**The premise that the initial Gaussians "do not participate in densification" is false, and
the truth is more useful.** The control's original rows qualify for densification *more* often
than the treatment's (0.6125 vs 0.2550), because the classic criterion is the screen-space
positional gradient and `dG/dmu` scales as `1/sigma`: an under-sized primitive sitting on a
residual has a *steeper*, not a weaker, positional gradient. `ci-op` — small sigma at the
saturated 0.95 opacity — maximizes both factors and qualifies most of all (0.7200).

So density control was not ignoring the badly-scaled initialization; it was **compensating for
it by multiplication**. The control needed 5,030 primitives to reach 21.8745 dB held out. The
cover-consistent arms reach **higher quality with 43–46% of that count**: `cover-iso-op` ends at
22.3279 dB with 2,230 (+0.4534 dB at 0.443× the primitives).

The training views confirm this is capacity, not skill: the control ends **higher** on train
views (28.6049 dB vs `cover-iso`'s 27.2435) while ending **lower** on held-out views. The extra
~2,800 primitives buy training-view fit. On the extrapolative camera `C1004` the ordering
matches the held-out pool (`ci` 19.9158, `cover-iso-op` 20.5739, `surfel` 20.4537).

## Preregistered decisions

| gate | criterion | measured | verdict |
|---|---|---|---|
| G1 coverage | init α-IoU ≥ 0.25 **and** init α-out ≤ 0.05 | 0.73315 / 0.13125 | **FAIL** |
| G2 initial quality | init PSNR ≥ `ci` + 1.0 dB | +5.1607 dB | PASS |
| G3 optimization | AUC ≥ +3% **and** final PSNR ≥ `ci` − 0.1 dB | +7.6738% / +0.3710 dB | PASS |
| G4 participation | qualified fraction ≥ 5× `ci` | 0.2550 vs 0.6125 (**0.42×**) | **FAIL** |
| G5 attribution | init α-IoU ≥ `ci-op` + 0.10 | +0.08346 | **FAIL** |
| guardrail | final α-out ≤ 0.05, every `adc` arm | worst `cover-iso` 0.04553 | PASS |

Three failures, reported as frozen and not re-scored:

- **G1** fails on leakage, not coverage. The cover extends a primitive roughly `sigma_t` past
  the silhouette, and at the derived opacity that reads as a halo. It is trained away — every
  `adc` arm ends within the 0.05 guardrail — but the initialization itself is not mask-clean.
  Note that `cover-iso` alone sits at 0.0437 initial alpha-outside, inside the guardrail; it was
  not the preregistered treatment and **is not selected here**.
- **G4** fails in the *opposite* direction to the hypothesis, which is the screen's most
  informative outcome. The gate was written on the assumption that the initialization was being
  ignored by density control. It was not; it was being amplified.
- **G5** fails narrowly (+0.083 against +0.10). It was written against initial alpha-IoU, which
  turned out to be the metric on which opacity dominates. The measurement that actually
  separates covariance from opacity is the AUC/final-quality/count triple above, where `ci-op`
  contributes +1.5% AUC and `cover-iso` +6.2% at half the primitives. That reading is **not** a
  preregistered gate and does not substitute for one.

**No default change is authorized.** Beam Fusion keeps CI; `surfel_init` stays opt-in.

## Scope limits

One scene, one seed, one CPU device, downscale 32, 8 of 26 views, 800 initial components, the
Torch reference rasterizer with the classic CPU controller rather than CUDA gsplat, and 1,000
steps. Two of the three held-out cameras are interpolative. Nothing here establishes a
cross-scene ranking, a production-topology result, or a performance claim.

## Provenance, chronology, and defects

- Protocol hash `5430a4569085b63bee6d673f9a7fc2ac6c421da3589dca561d3b177228cac248`, unchanged
  since the run and independently confirmed against `summary.json` by the audit.
- **Harness hash differs between execution and commit** and is recorded as such: executed
  `542d45599cd97596cf0e19831adec34910a2367b8fd89956c0aec8b98ea4f7bf`, committed
  `6f2baad9dcd1937b0594c9495ad6706df81f5b73bef936e7500238ee7a56eeef`. The only change is
  `write_viewer_manifest`, which gained a `modes` list so the manifest covers both modes instead
  of the first. That function is called at `main()` line 687, *after* `summary.json` is written
  at line 685, so it cannot reach any measurement, artifact, or metric. The executed version
  emitted a `fixed`-only manifest, which was regenerated post-run over all ten (mode, arm) pairs.
- `src/rtgs/lift/surfel_init.py` received **docstring- and comment-only** edits after the run
  was launched (an Adam rate-limit correction, a cap-ordering clarification, and the 1.75→
  1.66–1.75 ratio fix). The audit's rule-reproduction checks rebuild the saved arms' sigmas and
  opacities from the current module source and pass, which is the operative proof that no
  executable path changed.
- The preregistration carries three explicit amendments: a diagnostic-number correction made
  before any run outcome was read, the G4 wording bound to the quantity the harness measures,
  and disclosure that a 50-iteration two-arm smoke exposed the `ci`/`surfel` **initial** held-out
  numbers after the gates were frozen. No gate was changed after any outcome was seen.
- Installing the optional `viewer` extra for the mandated viewer smoke replaced the editable
  install with a regular one, so `rtgs` briefly resolved to `site-packages` instead of `src/`.
  That happened while the `adc` arms were running. The shadowing copy was verified
  **byte-identical** to `src/rtgs` (83 files, `diff -rq` clean) before it was removed and the
  editable install restored, so no executed code differed; already-imported modules in the
  running process were unaffected in any case.
- The pre-existing test `tests/test_compact_point_training.py::test_linux_listener_binding_
  requires_the_exact_process` fails in this container because `/proc/<pid>/net/tcp6` does not
  exist (IPv6 disabled). The commit touches none of the files involved. Every other lint,
  format, docs-sync, and test check passes.

## Next preregistered experiments

1. ~~**The count result is the finding worth confirming.**~~ **Done, same root, on
   2026-07-24.** The budgets compared above are *unmatched*, so this reading was tested against
   the alternative that the control was merely allowed to overshoot. Under a matched hard budget
   of 2,400 (= 3 x `N_init`) over three seeds, `surfel` beat `ci` by **+0.5074 / +0.6450 /
   +0.8049 dB** held out while never reaching the cap, and the capped control *lost* 0.196 dB
   versus its own uncapped endpoint: verdict `M1_CAPACITY_ADVANTAGE_HOLDS`, 3/3 seeds, guardrail
   passed. At matched capacity the treatment also leads on train views, removing the overfit
   ambiguity noted above. See
   [`20260724_beam_surfel_matched_capacity_RESULT.md`](20260724_beam_surfel_matched_capacity_RESULT.md).
   What remains genuinely open is **generalization**: multiple mask-bearing scenes and the
   production CUDA gsplat strategies. The two other checked-in roots have no packed alpha.
2. **Separate the leakage.** The silhouette halo is a real property of a cover rule applied to a
   surface that curves away from the camera. Test a mask-aware or curvature-aware shrink at the
   silhouette as its own treatment with an initial alpha-outside gate; do not select
   `cover-iso` post hoc from this run.
3. **Re-derive the participation question.** Now that the criterion is known to fire hardest on
   under-sized primitives, the useful gate is not "how much densification" but "how much
   densification per dB held out". Preregister that ratio directly.

## Visual comparison

Initial/final preview pairs (teacher left, render right, held-out `C0004`) are under
`runs/beam_surfel_init_20260724/{fixed,adc}/<arm>/`. Synchronized orbit over all ten
initial/final pairs:

```bash
.venv/bin/rtgs view \
  --comparison-manifest benchmarks/results/20260724_beam_surfel_init_VIEWER.json \
  --device cpu --port 8794 --no-open
```

The smoke loaded all 20 models, PID 16059 owned `127.0.0.1:8794`, HTTP returned 200, no CUDA
device was present, every relative link on the results page resolved, and the server was stopped
and the port released afterwards. Receipt:
[`20260724_beam_surfel_init_VIEWER_RECEIPT.json`](20260724_beam_surfel_init_VIEWER_RECEIPT.json).
The WebGL view is qualitative; all numbers above come from the exact Torch CPU rasterizer.
