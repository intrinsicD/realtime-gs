# Scientist pass — ADR-XXXX / ADR-YYYY surfel init and schedule screen

Date: 2026-07-27 (post-hoc referee session, after commit `095313d`)

Machine-readable audit:
[`20260727_surfel_init_schedule_screen_AUDIT.json`](20260727_surfel_init_schedule_screen_AUDIT.json),
SHA-256 `856b9b0d4f5a93486e139dac29d8bbd27d429ad0760fd25978b0febfd10ba49b`.

Disposition: **all four screen findings confirmed at development scope; one citation erratum;
raw-bundle binding is limited; three staged observations promoted as scoped claims C25–C27.**

This audit ran in a fresh CPU-only workspace: the gitignored `runs/surfel_init_schedule_screen/`
bundle is absent, so no GPU cell is replayed. What is independently checked here is the internal
consistency of the published tables across their two redundant representations, the analysis
conventions against the harness source, the chronology and isolation structure, the CPU-provable
acceptance criteria, and the one figure the result reuses from a deterministic CPU fixture —
which reproduces exactly.

## Claim inventory and disposition

| # | Claim | Kind and scope | Disposition |
|---|---|---|---|
| 1 | ADR-XXXX moves its mediator (alpha IoU 0.119→0.691, 0.011→0.765; Q@0 +6.3/+4.8 dB) | Measured, development, mediator-level | **Confirm** (table-internal; A2 gate ≥0.8 correctly reported missed at 0.691/0.765) |
| 2 | Every §3-carrying arm is a material held-out loss on both scenes, 0/3 seed-wins each (−1.13 to −2.32 dB) | Measured, development, downstream | **Confirm → C25** |
| 3 | §3-off (`surfel-rho0.1-fixedalpha`) lands inside the ±0.25 dB band on both scenes; §1/§2/§4 downstream-neutral | Measured, development, single-factor dissociation | **Confirm** (part of C25; "costs 1.3–1.5 dB" is the median-difference estimator; paired gives 0.9–1.6 dB, same sign and materiality) |
| 4 | ρ sweep does not rescue §3 | Measured, development | **Confirm** (monotone f8, flat f9, no level reaches `ci`) |
| 5 | Init-preserving controller below vanilla ADC in 4/4 scene×init comparisons despite 14 dB gentler events | Measured, development, 2 seeds | **Confirm → C26** (event invariance 42.6/28.3 and 47.2/32.8 dB re-executed exactly on CPU) |
| 6 | Trust schedule negative for the accurate init on both scenes; interaction −1.58 dB, opposite of the §5 prediction | Measured, development, 2 seeds | **Confirm → C27** |
| 7 | Init-conditioned densification value (+1.54 dB f8) | Non-replicated (−0.25 dB f9) | **Correctly withheld** — stays out of the ledger |
| 8 | Wall clock 131.7 min on RTX 3050 | Context, unrepeated GPU | **Not performance evidence; none claimed** |
| 9 | No default changes; both ADRs stay `proposed` | Disposition | **Confirm** (ADR headers verified) |

## Chronology, isolation, and preregistration

The governing master protocol (`20260725_init_value_program_PREREG.md`) merged at `d6876bc` on
2026-07-25, before the screen commit `095313d` (2026-07-27). The confirmatory phase remains
blocked at `20260725_init_value_program_PREREG_REVIEW_02_FAIL.md` and was not touched. Sealed
report-only cameras `C1001`/`C1004` are excluded from the harness's evaluation set.

Isolation is structural in the harness: `make_masked_scene` orders train views first and
initializers receive `train_local` only; the endpoint is `evaluate_views_masked` over the fixed
`VALIDATION` tuple (`C0031`, `C1000`, `C1002`); no selection, stopping rule, or tuning reads the
evaluation views. Implementation acceptance criteria (N165) were measured on synthetic ground
truth and fixtures before any screen outcome existed; intra-commit chronology beyond that rests
on run artifacts that are no longer present, which is acceptable at development scope only.

**Erratum (citation).** The RESULT header attributes the frame-role assignment ("debugging and
hypothesis generation only") to "§1.2 of the master protocol
(`20260725_init_value_program_PREREG.md`)". The governing PREREG's §1.2 is the noise-floor gate
and never names the Stage frames. The role table is §1.2 of the **withdrawn** 2026-07-26 rewrite
(`20260726_init_value_program_REWRITE_WITHDRAWN.md`), whose header states "Nothing here governs
any claim". The substantive point survives on independent grounds — the Stage frames are
outcome-exposed because earlier experiments (e.g. the 2026-07-24 `frame_00008` studies) selected
variants on them, and the screen's development-only self-scope is *more* restrictive than
anything the governing PREREG demands for these frames — so no claim is inflated. The RESULT is
append-only and stays uncorrected; this audit is the record.

## Evidence binding

The raw 82-cell bundle (`runs/surfel_init_schedule_screen/`, per-cell JSON, PLYs, previews,
`index.html`, viewer receipt) is gitignored and absent from a fresh clone. The trace records that
it passed `scripts/check_results_bundle.py` at run time; that receipt cannot be re-verified here.
The durable, tracked evidence for the numbers is the RESULT note itself plus the harness source,
tests, and trace nodes — so every claim promoted from this screen carries the replay boundary
explicitly. Recommendation for future screens: also track a compact per-cell summary JSON under
`benchmarks/results/` (the 2026-07-24 house pattern), which would have made this audit's numeric
recomputation exact instead of representation-level.

## Independent checks executed

- **Two-representation cross-check (Screen B).** The gain table and Finding B1's absolute medians
  are published independently; `classic/notrust − init-preserving/notrust` agrees exactly in all
  four scene×init comparisons (0.98, 0.58, 0.77, 0.84 dB).
- **Interaction arithmetic.** All 10 interaction cells recompute from the effect columns within
  0.01 dB (two-decimal rounding).
- **Verdict machinery.** `contrast()` defines ΔQ as the median of per-seed paired deltas vs `ci`
  and the verdict rule as gain (wins ≥ n−1 ∧ median ≥ +0.25), loss (median ≤ −0.25), else inside
  band; all 12 non-reference Screen-A verdicts reproduce. The apparent ΔQ vs median-difference
  mismatch (e.g. `beam-cover` −0.27 vs −0.44) is this documented pairing convention, not an error.
- **Cell accounting.** 42 (Screen A) + 48 (Screen B factorial) − 8 (`none/notrust` cells
  explicitly reused from Screen A) = 82, matching the stated count; the reuse is valid because
  `growth=none` never consumes the 6000 budget and iterations are shared.
- **CPU re-execution.** `pytest tests/test_surfel_lift.py tests/test_init_preserving_density.py`
  → exit 0. The deterministic A1 fixture reproduces the event-invariance figures **exactly**:
  clone 42.6 vs 28.3 dB, split 47.2 vs 32.8 dB.
- **`beam-cover` attribution.** Confirmed in the harness that `beam-cover` is the incumbent
  `rtgs.lift.surfel_init` reconciliation, not an ADR-XXXX §3 arm, so "every arm carrying the §3
  solve" correctly denotes the three ρ arms.

Commands: `pytest -q tests/test_surfel_lift.py tests/test_init_preserving_density.py`; a
seed-free re-execution of the A1 fixture via the test module's own helpers; `git show`/`grep`
inspection of harness, splits, PREREG, and withdrawn rewrite. Skipped for cause: GPU cell replay
and `check_results_bundle.py` (no CUDA device; bundle absent).

## What promotion requires next

The promoted claims C25–C27 are development-only. To move any of them past that scope:
non-outcome-exposed alpha-bearing scenes, ≥3 paired seeds throughout, per-arm tuning
(PREREG §1.5 P-adapt) before schedule levels are compared, a tracked machine-readable summary,
and the governing program's E0 noise floor measured first. Amending ADR-XXXX §3's alpha target
(binary silhouette vs Stage-1 accumulated coverage) is an ADR amendment, not a code patch, per
the RESULT's own next-actions list.
