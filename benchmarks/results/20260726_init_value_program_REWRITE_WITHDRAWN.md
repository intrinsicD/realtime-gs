> **WITHDRAWN — NOT THE GOVERNING PROTOCOL.**
>
> This is the uncommitted 2026-07-26 rewrite of `20260725_init_value_program_PREREG.md`
> (sha256 `e8f75dd72425abcb5b2edf33a88ba7c456b69576b97500083fae064ebae5f8cc`). It replaced the
> committed program's Part 0 instrumentation (M0-M5) and experiments (E0-E11) with a
> Karate/Stage gated protocol, deleting the clone/split census (E3, hypothesis H-C3) and the
> split-inheritance repair (E9) among others. The author withdrew it on 2026-07-26 and restored
> the committed version at `7d62f50` (sha256
> `2336b6362b0602a488fc10ed74b526eba91f8fd05a19db621e6cc42a1cef2408`) as governing.
>
> Preserved only because artifacts cite its hash: both preregistration reviews
> (`..._PREREG_REVIEW_INITIAL_FAIL.md`, `..._PREREG_REVIEW_02_FAIL.md`) reviewed *this* text, and
> the 2026-07-26 development and masked screens recorded it as their protocol. Those reviews and
> runs are therefore reviews and runs of a withdrawn document. Nothing here governs any claim.

---

# PREREG — Initialization Value Program (repaired master protocol)

**Original freeze:** 2026-07-25

**Protocol repair:** 2026-07-26

**Repository base at repair:** `d6876bc1f1e78cbcc32f712c3a9d4706fa94ec0e`

**Status:** **AMENDED / DEVELOPMENT-ONLY / NOT AUTHORIZED FOR CONFIRMATORY RUNS**

**Prior independent review:**
[`20260725_init_value_program_PREREG_REVIEW_INITIAL_FAIL.md`](20260725_init_value_program_PREREG_REVIEW_INITIAL_FAIL.md)
— verdict `FAIL — PENDING PROTOCOL REPAIR`.

**Next required reviews:** a fresh non-author review of this exact master-protocol hash; a separate
non-author `PASS` on the executable validation addendum before the validation screen; and another
non-author `PASS` on the phase-specific confirmatory addendum before any confirmatory test payload
is opened. The author of this repair cannot issue any of these passes.

**Results location:** results never enter this file. Development and confirmation use separate
`*_RESULT.md`, run directories, audits, and ARA dispositions.

This amendment was written **after** initialization results existed. It does not preserve the
original claim that the program was frozen before any relevant data existed, and it does not
retroactively license any previous run. Existing results are pilot evidence only.

---

## 0. What this protocol is trying to learn

The program asks four questions that must not be collapsed into one score:

1. **Optimizer value:** under the same low absolute primitive count, does the Stage 1 → lift
   initializer reach a fixed held-out quality sooner than standard train-only sparse SfM?
2. **Budgeted quality:** after the same fixed number of optimizer updates and with fixed topology,
   does it produce better held-out quality than sparse SfM and a count-matched random initializer?
3. **Mechanism:** if there is a gain, is it explained by additional surface coverage, by more
   accurate means, by render-state packaging, or by an interaction among them?
4. **Systems value:** after all required preprocessing is charged, is the complete path faster or
   smaller under a stated workload without breaking numerical or visual guardrails?

The confirmatory thesis is deliberately narrow:

> On unseen calibrated, mask-bearing scenes in a few-view, low-count regime, the frozen
> `beam-cover` initializer improves both end-to-end time-to-target and held-out quality at a fixed
> absolute count relative to train-only fixed-pose COLMAP SfM, and the same comparisons against a
> count-matched random initializer show that the gain comes from initialization information rather
> than merely from optimizer or run-order variance.

The thesis is not a geometry claim, not a universal 3DGS claim, and not a claim about unbounded
densification. Coverage and systems claims require their own evidence licences below.

---

## 1. Outcome-access and evidence ledger

### 1.1 Information available before this repair

The repair author had access to at least the following outcomes or outcome summaries:

| Date | Evidence family | Permitted use here |
| --- | --- | --- |
| 2026-07-21 | `20260721_all_initializers_frame00008_*` | pilot motivation only |
| 2026-07-23 | `20260723_beam_covariance_refit_*` and convergence replication | pilot mechanism evidence only |
| 2026-07-24 | beam surfel init, matched-capacity, birth-attribution, and scale-gradient results | candidate selection and diagnostic design only |
| 2026-07-25 | GPU Stage 1 initialization, cost-to-target, and refit-ceiling local runs | budget/timing design only |
| 2026-07-26 | initial independent review named in the header | protocol repair requirements |

The known pilot pattern includes a step-count win that was not a wall-clock win, a positive
matched-capacity signal on one Stage frame, and refit variants that did not establish a downstream
gain. Those observations motivate this design; none is confirmatory evidence.

### 1.2 Dataset roles

| Role | Inputs | Outcome use |
| --- | --- | --- |
| Pilot/development | Stage frames `frame_00008` and `frame_00009`, all existing run artifacts | debugging and hypothesis generation only |
| Validation/development | Karate frames `frame_00005` and `frame_00060` | implementation validation and a fixed screen; never a public method claim |
| Confirmation | exactly three newly acquired calibrated scenes satisfying §2.3 | only source of a new pipeline claim |
| Mechanism reference | synthetic scenes or independent active-depth/scan references satisfying §8 | coverage/mechanism claims only |

Stage data are outcome-exposed. Karate has also been used in smoke/development runs and its current
compact manifests do not contain packed alpha masks. It is therefore unsuitable for the masked
confirmatory endpoint. No Stage or Karate result may be pooled with confirmation, used to estimate
a confirmatory effect, or described as replication of that effect.

### 1.3 Frozen local manifest identities

These identities prevent silent substitution; they do not promote the inputs to confirmation.

| Input | Manifest SHA-256 | Semantic digest |
| --- | --- | --- |
| Stage `frame_00008` | `b1c8e256d73e2c05f3cb4797a615bdbb2639a637f12908a5c96a2a9a9f912847` | `0f86429b4cf503df3ad46ca84a9346c9ab1ada51509d90e13ae9fb241d2a8ef5` |
| Stage `frame_00009` | `c31f976e016b3f681ac7aed528bae660ae77f315f37cf2128024fdef5a413262` | `3718027565b7660c3cc62a54d94a7ddf3718f3030dbaae402534f3530bd3731f` |
| Karate `frame_00005` | `01b2a84b654f13d8969534719e4216db433f14ac336d4c59729617634f0adafd` | `7bdc584ca0cdcf0924a340b42bb8fe7964796e2dd450c5b7107521225a4187bd` |
| Karate `frame_00060` | `7d0b478d345c0fa36006482bf4f8c2e9bf5ca9b95b2662f4e4b280725da68fea` | `57774f1c6b4644b44d3ecd5542bfd9026763b9e49168a7c509095f2cc5fea125` |

---

## 2. Separation of development, validation, and test

### 2.1 General isolation rule

Every initializer, count adapter, covariance/refit stage, optimizer, checkpoint decision, and model
selection operation may read **training-role inputs only**. Validation inputs may be read only by
the validation scorer. Confirmatory test RGB, masks, 2D fits, derived features, and metrics remain
sealed until all arms have finished and every checkpoint is immutable.

The rule is enforced structurally, not by convention:

- A split manifest enumerates every permitted file by scene, role, view ID, size, and SHA-256.
- Training is launched from a staged directory containing only training-role payloads.
- Loaders accept an explicit allow-list and fail closed on an undeclared view or path.
- Every process writes a machine-readable file-access receipt containing all loaded input paths.
- The test scorer is a separate command and environment. It receives immutable checkpoint hashes
  only after a one-time unlock recorded in the run ledger.
- No training, refit, arm replacement, hyperparameter change, checkpoint deletion, or rerun is
  permitted after the first confirmatory test payload is opened.

A unit/integration test must deliberately inject a held-out path and show that the training command
fails before the first validation or confirmation run is authorized.

### 2.2 Validation split: fixed now, non-confirmatory

Karate validation uses full-frame metrics because packed alpha is absent. The role split is fixed:

| Scene | 3-view train set | Additional views in the nested 8-view train set | validation-only | report-only |
| --- | --- | --- | --- | --- |
| `frame_00005` | `C0005,C0010,C0021` | `C0022,C0039,C0025,C0030,C0037` | `C0031,C1000,C1002` | `C1001,C1004,C1005` |
| `frame_00060` | `C0000,C0037,C0020` | `C0001,C0024,C0021,C0039,C0022` | `C0031,C1000,C1002` | `C1001,C1004,C1005` |

The remaining views are unused. Validation-only views determine the go/no-go gate in §7.1.
Report-only views are opened once after the validation decision is frozen and diagnose overfit;
they cannot be used to change an arm, cell, threshold, or recorded decision, and they cannot carry
a claim. Their predeclared guardrail may only block progression.

### 2.3 Confirmation split: not yet available

Confirmation requires exactly three scenes that meet all of these eligibility rules before any
method is run:

- not previously used by this initialization line, including unpublished local runs;
- original RGB, calibrated intrinsics/extrinsics, and foreground alpha/masks are present;
- at least 12 usable views with adequate calibrated angular diversity;
- an independent active-depth/scan reference exists if a real-scene coverage claim is desired;
- eligibility is based only on acquisition and calibration QC, never on method output or COLMAP
  success;
- the acquisition manifest records provenance, all payload hashes, calibration hashes, and any
  exclusions made before method execution.

For each scene, the confirmatory addendum must enumerate exactly three training view IDs and six or
more sealed test view IDs. Selection may use camera poses only: start from the lexicographically
first eligible camera, then repeatedly select the camera maximizing minimum Euclidean camera-center
distance, breaking ties lexicographically. The first three selected cameras are training views,
the next six are test views, and all remaining views are unused. RGB, masks, Stage 1 fits,
features, and method outputs must not be inspected to choose the split. The explicit IDs in the
addendum govern over the algorithm.

Until those scenes exist and their addendum has a fresh non-author `PASS`, **the confirmatory phase
does not exist and no confirmatory command is authorized**.

---

## 3. Frozen arms and what each comparison means

### 3.1 Primary arms

1. **`beam-cover` (candidate).** Beam Fusion means and appearance; cover-consistent isotropic
   extent; initial opacity `0.10`; no covariance refit. This candidate was selected using pilot
   evidence and is fixed before validation. Its exact resolved config and source hash are frozen in
   the validation addendum and may not change for confirmation without restarting validation.
2. **`colmap-sfm-fixed-pose` (standard comparator).** Feature extraction, matching, and sparse
   triangulation use original training RGB only and the supplied training-camera calibration.
   Validation/test images, features, tracks, dense MVS, and all-view COLMAP models are forbidden.
   COLMAP version, command line, database hash, image allow-list, mapper/triangulator settings, and
   point-cloud hash are required artifacts.
3. **`random-count-matched` (information control).** The existing seeded random initializer,
   restricted to the same train-derived bounding volume and exact count as `beam-cover` in that
   scene/cell. It receives no validation/test-derived bounds.

`ci` is retained only as a validation/mechanism ablation. It is not eligible to replace
`beam-cover` after results are seen and is not a confirmatory primary arm.

### 3.2 Count and topology semantics

The primary regime is **fixed topology**:

- absolute requested count `N` is the same for every arm in a cell;
- if a structured arm produces at least `N` native roots, a frozen deterministic weighted
  farthest-point adapter selects exactly `N` using training data only;
- the adapter never invents, duplicates, jitters, or fills missing points;
- if an arm produces fewer than `N`, the run is recorded `insufficient_initial_points`; it is not
  silently substituted, and that scene cannot pass the all-arm quantitative headline gate;
- after initialization, birth, clone, split, merge, and prune are disabled; final count must equal
  initial count exactly.

Any later **capped dynamic** systems run is a different regime: topology changes are allowed but
`N(t) <= cap` at every recorded step. It must report the entire population trajectory and may not
be described as exact-count or fixed-topology evidence. No document may simultaneously promise
exact final count and permit ordinary pruning/birth.

### 3.3 Baseline availability is fail-closed

The repository does not currently contain the original Karate RGB/COLMAP source payload needed to
execute `colmap-sfm-fixed-pose`, and no reviewed train-only SfM initializer is yet registered.
Therefore validation is blocked until the implementation and inputs pass §6. A missing or failed
COLMAP arm is reported as baseline unavailability, not converted into a coverage win. The headline
comparison requires an executable comparator on all three confirmatory scenes.

---

## 4. Outcomes and estimands

### 4.1 Evaluation unit and aggregation

The atomic paired unit is `(scene, split, seed, arm, N)`. Confirmatory seeds are exactly
`26001,26002,26003,26004,26005`; the same seed is used across arms. Validation uses the first three.
Arm execution order is a deterministic seed-keyed Latin-square rotation so thermal/run-order drift
does not align with one arm.

For each checkpoint, metrics are computed per held-out view and then median-aggregated within the
scene/seed. Comparisons are paired by scene and seed. Across seeds report median, MAD,
`1.4826 * MAD`, minimum, and maximum. Across scenes report every scene separately; a pooled mean
cannot satisfy a gate.

Five paired seeds are a feasibility/stability design, not a claim of asymptotic precision: the
`4/5` rule tolerates at most one seed reversal. Requiring the same material direction on all three
scenes guards against a one-scene result but does not estimate performance over a scene
population. The program is intentionally unable to license small effects below the frozen
materiality thresholds; exact contrasts expose the remaining uncertainty.

### 4.2 Co-primary endpoints

Let `Q_a(s,k,j)` be median masked foreground PSNR for arm `a`, scene `s`, seed `k`, checkpoint
`j`. Higher is better. Checkpoints are frozen at optimizer updates
`j in {0,100,200,...,1500}`. All checkpoint renders are scored only after test unlock.

The scorer clamps prediction and target RGB to `[0,1]`. Per-view foreground MSE is the sum of
RGB squared errors weighted by the continuous ground-truth alpha/mask in `[0,1]`, divided by three
times the mask-weight sum; `Q = -10 log10(max(MSE, 1e-12))`. Crop SSIM uses the repository's
masked foreground bounding box plus a `5%` image-extent margin. Alpha IoU thresholds both the
ground-truth mask and predicted alpha at `0.05`. LPIPS uses the frozen AlexNet LPIPS v0.1 weights
on the same masked crop; the addendum records the package and weight-file hash. Every arm uses one
identical, addendum-frozen resolution and color transform within a phase. Resolution is not swept
and no resolution-generalization claim is permitted.

**Endpoint Q — held-out quality at fixed count and updates**

`Q1500_a(s,k) = Q_a(s,k,1500)`.

The material paired margin is `0.25 dB`. SSIM, LPIPS, alpha IoU, fitted-view PSNR, and iteration
curves are secondary/guardrail metrics and cannot rescue a failed co-primary endpoint.

**Endpoint T — synchronized end-to-end seconds-to-target**

For each scene/seed, define the target only after immutable test scoring as:

`tau(s,k) = Q_colmap(s,k,1500) - 0.25 dB`.

This formula is frozen now. The target is derived after all models are immutable, no model is
selected with it, and no rerun follows test unlock. It therefore measures how quickly each frozen
arm reaches near-final standard-comparator quality without using test data during fitting.

`T@tau` is the cumulative synchronized wall time from arm-specific raw training inputs to the first
checkpoint `j` satisfying both:

1. `Q_a(s,k,j) >= tau(s,k)`, and
2. the next two scheduled checkpoints each satisfy `Q_a >= tau(s,k) - 0.10 dB`.

An improving value above `tau + 0.10` is not a failure. A crossing at the last two checkpoints
cannot establish persistence and is right-censored. A run that never establishes persistence is
right-censored at its observed end; it is never assigned a finite time, zero, or the budget limit.
Report reachability first and time conditional on reaching second. Pairwise time ratios are defined
only when both arms reach. For a decision involving a censored comparator, a paired time success is
defined as: the candidate reaches and either (a) the comparator is censored, or (b) both reach and
`T_candidate / T_comparator <= 0.75`. Candidate censoring and joint censoring are not successes.

### 4.3 Timing boundary

End-to-end time includes every arm-specific operation required to produce and optimize the initial
3D Gaussians:

- `beam-cover`: Stage 1 fitting on train views, lift/fusion, count adaptation, packaging, transfer,
  renderer setup, and optimizer time to the checkpoint;
- COLMAP: train-image feature extraction, matching, fixed-pose triangulation, count adaptation,
  packaging, transfer, renderer setup, and optimizer time;
- random: train-derived bound construction, sampling, packaging, transfer, renderer setup, and
  optimizer time.

Only manifest parsing, common calibration validation, test scoring, result bundling, and viewer
generation are excluded and timed separately. Cached arm-specific preprocessing is forbidden for
the primary timing endpoint. A cached/post-Stage-1 timer may be reported only as a labelled
subsystem diagnostic.

Wall time uses a monotonic host clock around the complete process. GPU phases additionally use GPU
events, with device synchronization immediately before the start and after the stop. Runs execute
one at a time on the frozen device, after one unmeasured warm-up per executable, with power/clock
state and background-process receipt recorded. Iterations-to-target is reported as a diagnostic,
never called speed.

### 4.4 Direction-correct reference-gap diagnostics

An oracle/reference run is a diagnostic reference, not a mathematical ceiling or bound. For a
higher-is-better quality metric:

`G_Q = Q_ref - Q_random`, `R_Q(a) = (Q_a - Q_random) / G_Q`.

For lower-is-better time, when every involved arm reaches:

`G_T = T_random - T_ref`, `R_T(a) = (T_random - T_a) / G_T`.

The ratios are undefined if the denominator has the wrong sign or magnitude below the corresponding
materiality threshold (`0.25 dB` for quality, `10%` of `T_random` for time), or if a required time
is censored. Values outside `[0,1]` are permitted and reported; they demonstrate that the reference
was not a bound. No ratio supports the claim that no initializer can matter.

### 4.5 No pooled Omega decision variable

View count, absolute primitive count, resolution, optimizer updates, wall time, and population
trajectory are recorded as separate factors. The former pooled `Omega` variable is retired from
hypothesis tests and decision rules because its proposed factor directions were contradictory.
Exploratory normalized load measures may be plotted descriptively but cannot govern a claim.

---

## 5. Claims and frozen decision rules

All rules below are conjunctions; a favorable secondary metric cannot compensate for a failed
condition. There is no post-hoc “best structured arm” or “best cell.”

### 5.1 Confirmatory headline gate

At the fixed confirmatory anchor (`3` train views, `N=2400`, fixed topology, `1500` updates), the
headline thesis passes only if all of the following hold:

1. all three primary arms execute at exactly `N=2400` on every confirmatory scene;
2. on **each of three scenes**, at least `4/5` paired seeds have
   `Q1500_beam-cover - Q1500_colmap >= 0.25 dB`, and the scene median does too;
3. the same quality rule holds against `random-count-matched` on each scene;
4. on **each scene**, all five `beam-cover` runs reach `tau` and at least `4/5` paired seeds satisfy
   the paired time-success rule in §4.2 against COLMAP; if at least three pairs are uncensored, their
   scene-median ratio must also be `<= 0.75`;
5. the same candidate-reach and paired time-success rule holds against random; if fewer than three
   comparator pairs are uncensored, only target-reach dominance—not a percentage speedup—is
   licensed;
6. scene-median SSIM is not worse by more than `0.005`, LPIPS is not worse by more than `0.010`,
   and alpha IoU is not worse by more than `0.020` versus either comparator;
7. no numerical-validity, split-isolation, provenance, or result-bundle gate fails.

If quality passes but time does not, the only licensed wording is “better quality after 1500 fixed
updates in the tested regime.” If time passes but quality does not, the only licensed wording is
“reaches the fixed comparator-derived target sooner in the tested regime.” The broad “valuable
initializer” headline requires both.

The thresholds are practical materiality rules, not estimates of seed noise and not p-values.
Exact paired contrasts and uncertainty summaries are always published. No multiplicity correction
is needed for the headline because every co-primary/comparator/scene condition must pass; secondary
and subgroup findings are explicitly exploratory.

### 5.2 Coverage claim gate

“Improves surface coverage” requires a known or genuinely independent reference surface that was
not used by Stage 1, lift, SfM, count adaptation, selection, or training. Dense COLMAP/MVS generated
from the same images is a shared-bias diagnostic and cannot satisfy this gate.

At `epsilon = 0.01 * reference_scene_diagonal`, report:

- completeness: fraction of 100,000 fixed reference-surface samples within `epsilon` of a center;
- accuracy: fraction of Gaussian centers within `epsilon` of the reference surface;
- their harmonic mean, with the exact fixed reference sample hash.

A coverage claim requires, on every one of three independent-reference scenes, a scene-median
completeness gain of at least `5` percentage points over COLMAP, no accuracy loss greater than `1`
point, and downstream passage of at least one primary endpoint. Geometry alone never licenses an
optimizer-value claim. COLMAP failure alone never licenses a coverage claim.

### 5.3 Systems claim gate

A systems claim must name its boundary. “Compact supervision is faster/smaller” may refer only to
the post-Stage-1 loader/storage path and must report its excluded preprocessing. “The pipeline is
faster” must use §4.3 end-to-end time. Storage bytes include all metadata, indexes, and required
calibration. A systems win additionally requires output equivalence within the numerical tolerances
and all visual guardrails in §5.1. Functional or quality regressions cannot be dismissed because a
system metric improved.

### 5.4 Null and narrowing outcomes

- One scene or fewer than five confirmatory seeds: diagnostic only.
- Any opened test followed by repair/rerun: confirmation invalid; acquire new sealed scenes.
- Comparator unavailable on any confirmatory scene: no quantitative headline; report robustness
  limitation separately.
- Effect only on Stage/Karate: pilot result; no general claim.
- Coverage gain without downstream gain: geometry finding only; H-P2 is not supported.
- Downstream gain without independent-reference coverage gain: initializer value may stand, but the
  coverage mechanism remains unresolved.
- A failure on one confirmatory scene defeats the all-scene headline; pooled averages cannot rescue
  it.

---

## 6. Implementation and provenance gates

Before the 96-run validation screen, all items below must exist in one validation addendum and pass
a fresh non-author review:

1. exact source commit, dirty-diff archive, resolved config JSON, environment lock, package/CUDA/
   driver/COLMAP versions, hardware identity, and every command line;
2. recovered original Karate RGB and calibration with file hashes and provenance;
3. a registered, pluggable `colmap-sfm-fixed-pose` initializer that imports and tests CPU-first;
4. fail-closed split staging plus positive and negative isolation tests;
5. deterministic weighted farthest-point count adapter with tie-breaking tests;
6. fixed-topology enforcement that aborts on any population change;
7. synchronized component and end-to-end timers with a timing-boundary test;
8. checkpoint hashing and a scorer that cannot mutate or resume training;
9. a machine-readable decision program implementing §5, tested on pass, fail, missing-arm, wrong
   count, and censored-time fixtures;
10. deterministic result bundles satisfying Hard Rule 7 and
    `python scripts/check_results_bundle.py <run_dir>`.

Every results bundle must contain at minimum:

- prereg/addendum/review hashes and the decision-program hash;
- input/split/file-access receipts and all payload hashes;
- per-arm resolved configs, commands, environment, source archive, and dirty diff;
- component timers, synchronization receipts, checkpoint hashes, population trajectories, and raw
  per-view metrics;
- machine-readable paired contrasts, censoring state, decision result, previews, relative-link
  `index.html`, and viewer receipt;
- failed and excluded runs, with the predeclared reason and no deletion.

No silent fallback, auto-selected backend, changed seed, substituted scene, reduced run count,
retuned threshold, or undocumented cache is allowed. A gate failure stops the phase.

---

## 7. Development and confirmatory execution

### 7.1 Fixed validation screen

The validation screen is exactly:

- scenes: Karate `frame_00005`, `frame_00060`;
- train views: the exact nested `3` and `8` view sets in §2.2;
- absolute counts: `N in {2400,5000}`;
- arms: `beam-cover`, `ci`, `colmap-sfm-fixed-pose`, `random-count-matched`;
- seeds: `26001,26002,26003`;
- topology: fixed;
- optimizer checkpoints: `0,100,...,1500`;
- evaluation for the go/no-go decision: validation-only full-frame views;
- total: `2 * 2 * 2 * 4 * 3 = 96` training runs, with no hidden extra arm.

The anchor is fixed in advance as `3` views and `N=2400`; no screen cell replaces it. Other cells
measure whether the direction changes with view count or count and are descriptive.

Because Karate lacks packed alpha, define validation `Qval` as ordinary full-canvas RGB PSNR with
the same clamp, data range, view-median, and checkpoint cadence as §4.2. Define
`tau_val(s,k) = Qval_colmap(s,k,1500) - 0.25 dB`; validation timing uses the same persistence and
censoring rules with `Qval` substituted for `Q`. These quantities are never pooled with or renamed
as the masked confirmatory endpoints.

Proceed to confirmation only if, at the anchor on both Karate scenes, `beam-cover` is not worse than
either primary comparator by more than `0.25 dB` in scene-median `Qval1500`, and it shows at least
one of: `>=0.25 dB` median `Qval1500` gain or `<=0.80` median end-to-end `T@tau_val` ratio against both
comparators. This is a development viability gate, not a claim threshold. Report-only views are
then opened once to diagnose selection overfit. Recompute the same anchor rule with report-only
`Qval` in place of validation-only `Qval`; failure blocks confirmation and sends the program to
§8, without changing the recorded validation decision or selecting an alternative.

### 7.2 Confirmatory run count and ordering

After validation passes, acquire the three fresh scenes and write the confirmatory addendum. The
primary confirmation is exactly:

`3 scenes * 3 arms * 5 paired seeds * 1 anchor = 45 training runs`.

The addendum freezes scene/view IDs and hashes, exact source/config/commands, device, randomized
run order, warm-up, output paths, and expected bundle inventory. The primary count may not be
reduced and no ablation may be inserted. Synthetic or systems diagnostics have separately stated
counts and separate bundles; they are not part of the 45.

All 45 runs and checkpoints finish before the one-time test unlock. The scorer evaluates every
immutable checkpoint in one batch. The decision program runs without hand editing. Any change
after unlock invalidates the phase rather than creating a second chance on the same test data.

---

## 8. Experiments that identify how to improve the method

These experiments are developmental diagnostics. They explain a failure or nominate one repair;
they do not themselves establish the confirmatory thesis.

### 8.1 D1 — render-transfer audit

`P_2D` is renamed the **2D self-reconstruction baseline**. It is not called a ceiling. For the
same train views, record the baseline render and the immediately lifted 3D render before training.
The **render-transfer gap** is the signed loss or metric difference between those two renders.

Decompose the transfer without changing means: projection, covariance mapping, opacity/compositing,
appearance/SH, clipping, and renderer conventions. Each stage emits a render and hash. A large gap
localizes packaging loss but does not prove that means are useful, that the optimizer benefits, or
that closing the gap will improve held-out quality.

### 8.2 D2 — matched-topology synthetic 2x2 factorial

On hashed synthetic scenes with known camera, masks, surface, and root correspondences, use the same
root IDs and exact count in all cells:

- means: `{beam-lifted, correspondence-matched reference means}`;
- packaging: `{current analytic packaging, train-only fixed-topology packaging refit}`.

Packaging is recomputed conditionally for each mean set; it is never copied from the reference
means. The refit may use train renders only. Run the complete 2x2 with the five paired confirmatory
seeds and report mean main effect, packaging main effect, interaction, render-transfer gap,
coverage/accuracy, and downstream held-out endpoints. This can separate bad means from bad
packaging and reveal an interaction; replacing both at once cannot.

### 8.3 D3 — coverage mechanism

Use the independent-reference metrics and gate in §5.2. Also report the same metrics for dense MVS
as a labelled shared-input diagnostic. Repeat at `3` and `8` train views and `N=2400`; do not pool
those factors. If coverage improves but downstream quality/time does not, stop treating coverage as
the causal lever.

### 8.4 D4 — lineage and optimizer interaction

Record for every initial root: descendants, births, prunes, first-split time, final opacity/mass,
gradient norms, and visible-pixel contribution. A lineage association is descriptive. It may direct
a repair only after one frozen one-factor intervention changes the proposed mediator and then
changes a downstream held-out endpoint in the predicted direction.

### 8.5 Repair queue and reset rule

Only one repair family may be tested per child preregistration:

1. covariance/scale packaging;
2. opacity/appearance packaging;
3. split/birth policy;
4. count adaptation.

The child prereg names one intervention, one mediator, one downstream endpoint, scenes, seeds,
commands, and a decision rule before execution. Trying several variants and promoting the best is
exploration; it must be labelled as such and followed by a new validation pass. A candidate change
after confirmatory test access requires three new sealed confirmatory scenes.

Numerical SPD and downstream optimizer utility are separate licences. An SPD covariance may be a
useful coverage prior without being a physically supported covariance estimate. Claims of physical
validity still require reprojection/observability checks and must retain known rank/conditioning or
few-view limitations; a downstream win does not erase them.

---

## 9. Amendment-to-review closure matrix

| Initial review finding | Repair in this amendment |
| --- | --- |
| Lower-is-better headroom had reversed signs and undefined censoring | §4.2 and §4.4 define signed quality/time gaps, persistence, right-censoring, and undefined cases |
| `Omega` encoded contradictory factor directions | §4.5 retires it from decisions and keeps factors separate |
| Held-out data were reused for selection and claims | §§1–2 establish pilot/validation/fresh-test roles, fail-closed staging, and one-time unlock |
| Noise and aggregation were undefined | §4.1 fixes paired seeds, view/seed/scene aggregation, MAD scale, and §5 uses material conjunction gates |
| Iterations-to-target was called speed and timing boundaries were incomplete | §§4.2–4.3 make synchronized end-to-end seconds primary and iterations diagnostic |
| Train-only isolation and COLMAP were not executable/fail-closed | §§2, 3.3, and 6 block runs until inputs, implementation, receipts, and negative tests exist |
| `P_2D`, oracle, and GT substitutions were over-interpreted | §§4.4 and 8.1–8.2 rename references and require a matched-topology factorial |
| Optimizer utility was conflated with physical covariance validity | final paragraph of §8.5 keeps the licences separate |
| Exact configs/counts/cost were under-frozen and M4 contradicted pruning | §§3.2, 6, and 7 distinguish fixed/capped semantics, require child hashes/commands, and state exact run counts |

---

## 10. Stop conditions and interpretation boundary

Stop the current phase immediately on split leakage, payload/hash mismatch, silent fallback,
non-finite state, population mismatch, timing-boundary violation, missing required arm, or incomplete
bundle. Preserve the failed run.

This program can support claims only for the tested scene class, three-view anchor, count, optimizer,
hardware/timing boundary, and implementation hashes. Generalization to other resolutions, view
counts, dynamic densification, outdoor scenes, or devices requires new evidence. No default changes,
README capability claim, roadmap closure, or ARA claim promotion may occur until the confirmatory
bundle and its independent audit pass in the same change.

The next legal action is not an experiment. It is a fresh non-author review of this repaired file,
followed—only on `PASS`—by implementation of §6 and a separately hashed validation addendum.
