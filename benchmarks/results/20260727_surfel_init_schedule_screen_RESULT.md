# ADR-XXXX / ADR-YYYY development screen — init parameters and the schedule that keeps them

**Status: DEVELOPMENT-ONLY — NOT CONFIRMATORY.**
Stage frames `frame_00008`/`frame_00009` are assigned "debugging and hypothesis generation only"
by §1.2 of the master protocol (`20260725_init_value_program_PREREG.md`): they are
outcome-exposed, because earlier experiments in this repository selected variants on them.
Nothing in this document may enter `README.md`, `docs/`, or `ara/logic/claims.md` as a method
claim. The confirmatory phase remains blocked at `..._REVIEW_02_FAIL.md`.

These two frames were used because they are the **only** compact bundles in this repository that
carry packed alpha, and ADR-XXXX §3 solves against exactly that alpha. The screen cannot be run
anywhere else at all.

## Run identity

| Field | Value |
| --- | --- |
| Run dir | `runs/surfel_init_schedule_screen/` |
| Harness | `benchmarks/surfel_init_schedule_screen.py` (+ `_report.py`, `_bundle.py`) |
| ADRs implemented | `docs/ADR-XXXX-surfel-lift.md`, `docs/ADR-YYYY-init-preserving-densification.md` |
| Design | Screen A: 2 scenes × 7 arms × 3 seeds, fixed topology. Screen B: 2 scenes × 2 inits × 3 growth × 2 trust × 2 seeds |
| Cells | 82 |
| Optimization | 1500 updates, downscale 4, masked supervision, gsplat/CUDA |
| Wall clock | 131.7 min, RTX 3050 |
| Endpoint | median per-view **foreground-weighted PSNR on held-out views**, target = each view's own compact 2D fit composited on black |
| Materiality | 0.25 dB, paired per seed |

`Q@0` and `init alpha IoU` are measured **before any optimizer step**. The IoU is the ADR-XXXX
A2 gate: intersection over union of the lifted state's rendered alpha with the ground-truth
foreground at threshold 0.5, on the fitted views.

---

## Screen A — the init parameters (fixed topology, 3 seeds)

| scene | arm | init α-IoU | Q@0 | Q@1500 | ΔQ vs `ci` | wins | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| frame_00008 | `ci` | 0.119 | 11.62 | 18.45 | — | — | reference |
| frame_00008 | `beam-cover` | 0.699 | 15.48 | 18.01 | −0.27 | 0/3 | material loss |
| frame_00008 | `surfel-rho0.05` | 0.692 | 17.87 | 16.48 | −2.32 | 0/3 | material loss |
| frame_00008 | `surfel-rho0.1` | 0.691 | 17.91 | 17.04 | −1.41 | 0/3 | material loss |
| frame_00008 | `surfel-rho0.25` | 0.688 | 18.11 | 17.29 | −1.29 | 0/3 | material loss |
| frame_00008 | `surfel-rho0.1-fixedalpha` | 0.378 | 13.03 | 18.50 | +0.14 | 0/3 | inside the band |
| frame_00008 | `random-matched` | 0.360 | 14.42 | 18.76 | +0.31 | 2/3 | material gain |
| frame_00009 | `ci` | 0.011 | 11.05 | 16.57 | — | — | reference |
| frame_00009 | `beam-cover` | 0.809 | 15.46 | 15.46 | −1.34 | 0/3 | material loss |
| frame_00009 | `surfel-rho0.05` | 0.768 | 15.87 | 15.14 | −1.25 | 0/3 | material loss |
| frame_00009 | `surfel-rho0.1` | 0.765 | 15.88 | 15.02 | −1.13 | 0/3 | material loss |
| frame_00009 | `surfel-rho0.25` | 0.772 | 15.92 | 14.87 | −1.28 | 0/3 | material loss |
| frame_00009 | `surfel-rho0.1-fixedalpha` | 0.471 | 12.67 | 16.27 | −0.24 | 0/3 | inside the band |
| frame_00009 | `random-matched` | 0.328 | 13.14 | 16.29 | −0.28 | 0/3 | material loss |

### Finding A1 — the construction moves its own mediator by a large margin

ADR-XXXX does exactly what it was designed to do at the initialization. Alpha IoU rises from
0.119 to 0.691 on `frame_00008` and from **0.011 to 0.765** on `frame_00009`; held-out Q@0 rises
by 6.3 dB and 4.8 dB. The "accurate means whose rendered state is empty" failure the ADR was
written to fix is fixed. The A2 gate itself (IoU ≥ 0.8) is **missed** on both scenes — 0.691 and
0.765 — so the construction is close to, but short of, its own acceptance bar.

### Finding A2 — none of that survives optimization, and §3 is why

Every arm carrying the §3 transmittance solve is a **material loss** against the repository
default, on both scenes, at 0/3 seeds: −1.13 to −2.32 dB. Turning §3 off and using the incumbent
constant α = 0.10 (`surfel-rho0.1-fixedalpha`) lands **inside the 0.25 dB band on both scenes**
(+0.14, −0.24). The §1/§2/§4 construction — rotation, scale, colour — is therefore
downstream-neutral, and the opacity solve alone costs 1.3–1.5 dB.

That is a clean dissociation in the single factor the ADR isolates: §3 is **+4.9 dB at
initialization and −1.5 dB at the end**. Under `ara` constraint R25 a repair must move the
mediator *and* improve downstream fixed-count held-out quality; §3 does the first and reverses
the second, so it is not selectable.

### Finding A3 — the ρ sweep does not rescue it

E7's sweep surface is monotone on `frame_00008` (0.05 → 16.48, 0.1 → 17.04, 0.25 → 17.29: thinner
is worse) and flat on `frame_00009` (15.14 / 15.02 / 14.87). No tested ρ reaches the comparator on
either scene. ρ is not the free parameter that makes this construction work.

### Diagnostics the flags produced (`surfel-rho0.1`)

| quantity | frame_00008 | frame_00009 |
| --- | ---: | ---: |
| `flag_scale_disagree` (>3× between views) | 48.9% | 53.9% |
| `flag_normal_disagree` (PCA vs 2D-axis normal >30°) | 89.1% | 86.1% |
| `flag_line_like` (median 2D aspect >8:1) | 48.8% | 67.5% |
| `merge_candidate` (α at the lower bound) | 30.1% | 34.7% |
| σ_tangent / spacing (median) | 0.468 | 0.457 |
| NNLS relative residual | 0.223 | 0.184 |
| linearization within the ≲0.7 validity bound | 94.4% | 94.4% |

Two of these are worth naming. **Roughly half the primitives disagree by more than 3× between
views on their own tangential scale**, which is the ADR's own stated estimator of the beam-fusion
false-correspondence rate — the lift is being asked to build covariance on correspondences that
are wrong about half the time. And the measured σ_tangent/spacing sits at ~0.46, just under the
0.5 that `rtgs.lift.surfel_init` derives as the smallest ratio that covers a surface: the
*measured* footprints are systematically too small to tile after decimation, which is a real
tension between "recover what Stage 1 measured" and "cover the surface".

---

## Screen B — the schedule, crossed with the init (2 seeds)

Each cell is the median held-out gain over that init's own `none/notrust` baseline. The last
column is the difference between the two inits' effects; that difference *is* the ADR-YYYY
thesis.

| level | frame_00008 surfel | frame_00008 random | **interaction** | frame_00009 surfel | frame_00009 random | **interaction** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `none/trust` | −1.10 | +0.49 | **−1.58** | −0.48 | −0.10 | −0.38 |
| `classic/notrust` | +1.27 | −0.27 | **+1.54** | −0.44 | −0.19 | −0.25 |
| `classic/trust` | +0.43 | −1.32 | +1.75 | −0.03 | −0.68 | +0.66 |
| `init-preserving/notrust` | +0.29 | −0.85 | +1.14 | −1.21 | −1.03 | −0.18 |
| `init-preserving/trust` | −0.70 | −1.15 | +0.45 | −1.51 | −1.87 | +0.35 |

### Finding B1 — the init-preserving controller loses to vanilla ADC in every cell

`init-preserving/notrust` is below `classic/notrust` in all four (scene × init) comparisons:
17.26 vs 18.24 and 17.33 vs 17.91 on `frame_00008`; 14.28 vs 15.05 and 15.16 vs 16.00 on
`frame_00009`. This is despite the ADR-YYYY operators being *measurably* gentler: on the unit
fixture their events preserve the render at **42.6 dB (clone) and 47.2 dB (split)** against
**28.3 dB and 32.8 dB** for the vanilla operators run through the same harness.

So appearance-preserving densification events are not what makes densification work. The
vanilla operators' perturbation — the thing ADR-YYYY §2/§3 was written to eliminate — is doing
something useful, most plausibly exploration. The ADR's design invariant ("growth changes
capacity, never the current solution") is implementable and is not the winning property.

### Finding B2 — the trust schedule is refuted in direction

ADR-YYYY §5 predicts the S4 crossover: trust helps an accurate init and hurts a random one.
Measured under fixed topology, trust is **negative for the surfel init on both scenes** (−1.10,
−0.48) and better for the random control on `frame_00008` (+0.49). The interaction is −1.58 dB
where it is large, i.e. the **opposite sign** from the prediction. Trusting position, tangential
scale and rotation while leaving opacity and SH free lets the appearance parameters fit to a
geometry that is not allowed to correct itself.

### Finding B3 — the one positive interaction does not replicate

On `frame_00008`, vanilla densification is strongly init-conditioned: +1.27 dB for the accurate
init versus −0.27 dB for random, an interaction of **+1.54 dB**. That is exactly the kind of
signature E12 is designed to detect. It **does not replicate** on `frame_00009`, where the same
contrast is −0.25 dB. One of two scenes is not a result. This reproduces the scene-dependence
already recorded as Finding 3 of the 2026-07-26 Karate screen, on different scenes and a
different metric surface.

---

## Verdicts

| Component | Mediator | Downstream | Disposition |
| --- | --- | --- | --- |
| ADR-XXXX §1/§2/§4 (rotation, scale, colour) | large gain | inside band, both scenes | **neutral** — implementable, no measured value |
| ADR-XXXX §3 (transmittance opacity NNLS) | large gain | −1.13 to −2.32 dB, 0/6 seed-wins | **refuted downstream** |
| ADR-YYYY §1–§4 (init-preserving growth) | event invariance +14 dB | below vanilla in 4/4 comparisons | **refuted downstream** |
| ADR-YYYY §5 (trust schedule) | applies exactly as specified | negative for the accurate init, both scenes | **refuted in direction** |
| "Densification value is init-conditioned" | — | +1.54 dB on one scene, −0.25 on the other | **not replicated** |

No default changes. Both ADRs remain `proposed`, and both remain implemented, tested, and
selectable so the levels stay available to a future E12 grid.

## What this screen does not establish

- No comparison against structure-from-motion of any kind (no source RGB; the COLMAP comparator
  is unavailable in this repository).
- No held-out RGB quality — the target surface is each view's own 2D fit, so these numbers bound
  agreement with Stage 1, not with the camera.
- No geometry or coverage claim (no independent reference surface).
- Two scenes, outcome-exposed, three seeds (Screen A) and two seeds (Screen B). The Screen B
  factorial has no tuning budget per arm, so it tests the ADRs' *specified* levels, not the best
  achievable version of each (PREREG §1.5 P-adapt was not run).
- The 1500-update budget at 3 training views and N=2400 is a low-Ω regime; train PSNR rises to
  33.5 dB while held-out sits near 18, so all arms are in the over-parameterized corner where the
  Ω thesis predicts initialization should matter least.

## Bundle

`python scripts/check_results_bundle.py runs/surfel_init_schedule_screen` → OK. The
representative bundle at the run root is `frame_00008 / surfel-rho0.1 / none/notrust / seed
26001`, selected by a fixed outcome-independent rule (lexicographically first scene, anchor arm,
fixed topology, lowest seed). It is a viewer handoff and a preview, **not** a result.

```
.venv/bin/rtgs view --comparison-manifest runs/surfel_init_schedule_screen/comparison.json \
    --rasterizer gsplat --device cuda
```

`comparison.json` carries all 19 named initial/final pairs — every Screen-A arm and every
Screen-B schedule level on the representative scene — under one orbit camera. Viewer receipt:
`viewer_smoke.json` (served). Results page: `runs/surfel_init_schedule_screen/index.html`.

## Next actions

1. The §3 target is a **binary foreground mask**, so `−log(1 − Â)` asks for a transmittance the
   geometry provably cannot reach; 42% of primitives sit at a bound. Before re-running §3, decide
   whether the ADR's alpha target should be the packed silhouette at all, or the Stage-1
   *accumulated coverage*, and amend the ADR rather than the code.
2. The 49–54% `flag_scale_disagree` rate is a beam-fusion correspondence problem, not a lift
   problem. It bounds what any covariance construction on this lineage can achieve and should be
   attacked first.
3. The measured σ_tangent/spacing ≈ 0.46 versus the 0.5 cover condition is a decidable design
   question: measurement and cover disagree by a known factor after decimation.
