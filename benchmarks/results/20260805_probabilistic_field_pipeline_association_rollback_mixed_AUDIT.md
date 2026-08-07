# Probabilistic Gaussian-field all-dataset experiment — independent results audit

Date: 2026-08-06 (Europe/Berlin)  
Auditor: `Codex-probabilistic-field-protocol-reviewer`  
Verdict: **ACCEPTED FOR BOUNDED DEVELOPMENT INTERPRETATION; THREE MECHANISM CLAIMS RETIRED**

## Referee disposition

The immutable producer run is complete and internally consistent enough to support the frozen,
development-only synthetic decisions and calibrated operability accounting. Independent
recomputation from raw cells exactly reproduces all producer aggregates and predicates. The
synthetic hard invariants pass; rank-aware covariance recovery and probability support pass their
isolated rules at `3/3` seeds. Association, topology selection, and progressive scheduling each
fail at `0/3` seeds and are retired for this protocol. The combined calibrated arm is descriptive
only and cannot rescue those failures.

All 66 calibrated cells have exactly one terminal: 59 successes and seven structured candidate
failures. Native controls succeed `33/33`; the all-candidate arm succeeds `26/33`. Failed cells
have no summary or quality/runtime metric, appear in history only as `cell_success=0`, and are
represented by explicitly rejected presentation models. No failed value is imputed and no native
result is substituted. Exactly two successful cells use the frozen whole-cell unmasked-support
fallback, giving `2/66 = 0.030303030303030304`; those cells establish unmasked operability only.

| Claim | Disposition | Independently checked evidence |
| --- | --- | --- |
| The result is bound to the approved task, review, data, command, and result-producing source. | **Confirm, development scope.** | Task, protocol, prospective review, 309-file data seal, 102-file source tree, successor driver, pinned base driver, and command hashes match the lock. The run was deliberately dirty and development-only; the exact scientific source bytes are preserved by the dedicated bindings, while the broader dirty-worktree digest is not itself a source snapshot. |
| The frozen schedule completed without hiding failures. | **Confirm.** | The separate 483-cell synthetic matrix, one discarded warmup, and all 66 measured terminals exist. Root receipts report 59 successes, seven failures, and exit zero. The warmup is absent from measured aggregates and histories. |
| Every synthetic hard invariant passed. | **Confirm.** | Maximum fixed-point residual is `7.894663323071427e-08`; maximum candidate-gate mass is `0`; maximum source-mean error is `1.7763568394002505e-14`; maximum relative source-covariance error is `1.0072114908675744e-15`; both split errors are `0`; all isolation/finiteness gates pass. |
| Rank-aware full covariance passes the frozen exact-shape rule. | **Confirm at synthetic exact-field scope.** | It wins the two covariance comparisons in all `3/3` seeds over all 27 eligible full-rank strata per seed. This does not establish physical-geometry accuracy on calibrated captures. |
| Probability support passes the frozen corrupted-mask rule. | **Confirm at controlled synthetic-mask scope.** | It is not dominated by hard or no support in 7, 4, and 8 nonzero-corruption strata for seeds 80501, 80502, and 80503, respectively. The two calibrated unmasked fallbacks are excluded from mask-mode evidence. |
| Field-mass-capacity transport improves association. | **Retire for this protocol.** | It has `0/3` seed wins. Its matched-coverage score beats row-softmax and the shuffled negative but loses the field-no-association control at every seed. Calibrated combined results cannot rescue this failed isolated gate. |
| Projection-nonlinearity topology selection improves held-out density at matched count. | **Retire for this protocol.** | It has `0/3` seed wins. Candidate and native density MSE are exactly equal at every seed and candidate wall time is slightly higher. |
| Progressive views improve frozen time-to-quality. | **Retire for this protocol.** | Refit time is 15.8%, 16.9%, and 17.4% lower across the three fresh-process pairs, but density/RGB endpoint deviations violate the frozen 1% tolerance; seed wins are `0/3`. |
| Transactional rollback lets the matrix continue with zero-success semantics. | **Confirm with provenance narrowing.** | One candidate terminal has exactly the four missing-transport gates expected after rollback; the bound wrapper-to-base traceback proves that exact rollback diagnostics passed in memory before the base hard gate. The original caught exception type/message and `association_failure` string were not serialized, so this is path-proven rollback consistency, not a directly replayable exception receipt. |
| Forward-AABB eligibility is auditable for every realized fit. | **Narrow.** | All 59 successful primary summaries serialize internally consistent total/per-view counts. Bound control flow validates primary and half results before successful publication and validates each failed primary before its hard gate. Failed-primary and independent-half AABB counters are not retained, so their exact counts cannot be independently reconstructed from artifact bytes. |
| The calibrated arm establishes quality or performance superiority. | **Retire.** | It is a descriptive 512-component-per-view proxy with unequal successful denominators after seven candidate failures. The host is recorded only as `x86_64`; hostname, CPU model detail, load, and idleness are absent. Timings and RSS are raw host-local diagnostics, not general performance evidence. |
| Report and orbit-viewer behavior is established. | **Not yet evaluated.** | Eleven comparison manifests and their referenced PLY files exist and hash correctly, but this audit intentionally did not render pages, open visual assets, launch viewers, or perform browser smoke. |

## Protocol, source, and chronology binding

- Task SHA-256: `42ec54bea2636c8adce9e4bae175cc9e847a32b25cbea02b73ba3b04949dff43`.
- Protocol SHA-256: `e57d58112fd6f95467e8ddacdb4daad7fc9d83ed48b8b9f336a32b1966a92e87`.
- Prospective review SHA-256:
  `7c0b99c864ff7fecb0e3f5e3180c615f4e157ea70d0d010c78ce29e3238d65f4`.
- Data-seal SHA-256:
  `20e719d89628375c515db94102abf6e5018dbd6d686d0633235407fad5c7deb6`
  over 309 unique files and 204,306,829 bytes; contract revalidation returned `OK`.
- Result-producing 102-file source-tree SHA-256:
  `b17dec48edf2f07469dc9f6b197d062e4c0ee59698a05772c3674ad1fdf9b2eb`.
- Successor-driver SHA-256:
  `9a53815e2e0f17c2b40c9c67295c319eec4fff163541012804438717d5801bff`.
- Pinned base-driver SHA-256:
  `9d453b967b09005b63d3bef6aac48b817ac841ea2b7faa3593d4f980e4310169`.
- Task-lock SHA-256:
  `d73b1d145385ae74a2ecd6f293e5a828e908ed6d156d1932ebc82fc66a40ae7e`.
- Locked source commit: `36630c7fef14c0907134d2f3c532be3da4a0c43e`; the lock records
  `source_dirty=true`, development mode, and dirty-state digest
  `57bcdda05300658eb579db8db1fc71c89558d662b984a72494c2f880b8f12355`.
- Exact command: `.venv/bin/python scripts/experiments/20260805_probabilistic_field_pipeline_association_rollback_mixed.py --task experiments/tasks/20260805_probabilistic_field_pipeline_association_rollback_mixed.json --run runs/20260805_probabilistic_field_pipeline_association_rollback_mixed`.

The lock was created at `2026-08-06T05:42:15.590692+00:00`. The discarded warmup published
before the first measured terminal. The run receipt closed at
`2026-08-06T12:33:43.927020+00:00`, after 24,688.336328 seconds, and the atomic aggregate commit
receipt followed at `2026-08-06T12:33:43.929106+00:00`.

The dirty-state digest includes unrelated tracked changes and an untracked-file manifest at lock
time, but does not store those bytes. It cannot now be regenerated after append-only result and
research records were added. This does not leave the result-producing algorithm ambiguous: the
task, contract, successor, pinned base driver, and every `src/rtgs/**/*.py` byte are separately
bound and currently reproduce their reviewed digests. It does prevent calling the wider dirty
workspace a replay-complete source archive.

## Cardinality and failure accounting

The synthetic factorial independently expands to exactly:

| Stage | Cells |
| --- | ---: |
| Exact shape recovery | 324 |
| Re-componentized association | 60 |
| Support-mask factorial | 81 |
| Topology factorial | 6 |
| Schedule factorial | 6 |
| Independent-half stability | 6 |
| **Total** | **483** |

The measured calibrated grid is exactly 11 datasets × 3 seeds × 2 arms = 66 terminals. The seven
failures are all in `all_candidate_mechanisms`:

| Dataset | Seed | Frozen terminal gate |
| --- | ---: | --- |
| `stage_00008_native_fullres` | 80501 | `transport real mass` |
| `stage_00008_structsplat_no_boundary_fullres` | 80502 | `transport real mass` |
| `stage_00008_structsplat_no_boundary_fullres` | 80503 | `transport real mass` |
| `stage_00009_native_fullres` | 80501 | `transport real mass` |
| `stage_00009_native_fullres` | 80502 | `transport real mass` |
| `stage_00009_native_fullres` | 80503 | `transport real mass` |
| `karate_00060_default` | 80501 | `transport plan missing, transport real mass, transport fixed point, candidate gate` |

Every failure has the exact frozen failure/boundary/resource key sets, matching task/dataset/seed/
arm/warmup context, live clean guard, one identical no-fallback record across failure, boundary,
resource, and config, two hash-bound rejected PLYs, no preservation error, finite CPU resource
receipts, and no summary. The last row is rollback-consistent. The other six are committed-
association results rejected for insufficient real mass; the successful candidate cells all
serialize `association_status=committed` and pass every transport gate.

The only successful support fallbacks are the native and candidate cells for
`stage_00008_structsplat_no_boundary_fullres`, seed 80501. Both record exact initial
`ValueError: support-mask policy rejected every field-placement source`, one whole-cell retry,
RNG reset to 80501, effective mask `none`, three checked/fallback fits, and the frozen
`unmasked_operability_only_not_mask_mode_evidence` interpretation. No failed cell used fallback.

## Independent synthetic recomputation

| Mechanism | Seed wins | Frozen result | Referee disposition |
| --- | ---: | --- | --- |
| Rank-aware shape | 3/3 | Pass | Confirm, exact synthetic fields only |
| Field-mass association | 0/3 | Fail | Retire |
| Probability support | 3/3 | Pass | Confirm, controlled corruption only |
| Projection-nonlinearity topology | 0/3 | Fail | Retire |
| Progressive schedule | 0/3 | Fail | Retire |

Association matched-coverage score medians for field-mass / row-softmax / no-association /
shuffled-negative are, by seed:

- 80501: `0.20070238160231874 / 0.19656624661642264 / 0.3312623245933365 / 0.066828356249104`;
- 80502: `0.18572323696038903 / 0.16677843080714255 / 0.27088168261793133 / 0.08551847448424887`;
- 80503: `0.19795736287979787 / 0.19190871362820458 / 0.33146357474378885 / 0.07633689227612572`.

The schedule's progressive/all refit-time ratios are `0.8418741680509578`,
`0.8308721426115305`, and `0.8257891014556084`, but density endpoint deviations are
`17.0516%`, `6.9933%`, and `8.3994%`; RGB deviations are `10.4894%`, `0.0807%`, and `4.8675%`.
Each seed therefore fails the joint time-to-quality rule. Topology density errors are bitwise
equal between arms at every seed, while candidate/native wall ratios are `1.00581`, `1.01796`,
and `1.01081`.

## Calibrated aggregates and no-imputation check

All producer aggregates below reproduce exactly from raw terminal records:

| Metric | Value | Denominator/scope |
| --- | ---: | --- |
| Known-parent world-center RMSE | `0.02185523095957925` | Synthetic rank-aware cells |
| Known-parent covariance relative error | `0.009127745810987228` | Synthetic rank-aware cells |
| Association precision × coverage | `0.19277893157559217` | Synthetic field-mass cells |
| Support precision × coverage | `0.39127218934911245` | Synthetic probability cells |
| Calibrated cell success fraction | `0.8939393939393939` | 59/66 attempts |
| Successful unmasked-fallback fraction | `0.030303030303030304` | 2/66 attempts |
| Candidate held-out density MSE | `0.8221686179262926` | 26 successful candidate cells |
| Candidate held-out RGB-numerator MSE | `0.03164297257995046` | 26 successful candidate cells |
| Candidate refit wall time | `4.861169292009436` s | 26 successful candidate cells |
| Candidate final count | `63` | 26 successful candidate cells |
| Independent-half center distance | `0.06372770358118819` | 19 successful seed-80501 cells, both arms |

The last metric is stability only. It is not accuracy or resolution. The candidate conditional
metrics have a different success process from native controls and must not be reported as an
unqualified arm superiority result.

No-imputation checks passed on every surface:

- each failed root and dataset cell lacks `summary` and all quality/runtime fields;
- the 2,426-row history gives each failure only one zero-valued success indicator;
- conditional medians use successful cells only and carry explicit denominators;
- root charts and all dataset curves omit failed quality/runtime points while retaining zero
  success points;
- all seven rejected models are separately labeled presentation-only in comparison manifests;
- no failed candidate borrows a native or another-seed terminal.

## Isolation, AABB, artifact, and resource checks

All 1,842 repeated compact-file receipt records across the 66 workers match current byte counts,
SHA-256 values, and entries in the 309-file seal. Every worker allows only calibration and
`gaussians2d`, records no external mask/image or held-out training access, passes all three
negative controls, has zero denied real paths/imports, and exits without forbidden modules.
Successful summaries reproduce held-out indices from the frozen splits; optimized primary views
are training-only. Every retained independent-half pair is disjoint, its union is the original
training set, and neither half touches held-out views.

All 59 directly serialized primary AABB receipts have exact integer types, eight-or-fewer
per-view capacities, eligible + rejected = candidate counts, totals equal the selected component
capacity, and at least the requested 64 eligible anchors. These primary receipts cover 241,664
candidate components. Successful seed-80501 cells report three invariant-checked fits and other
successful seeds report one, totaling 97 successful primary/half results. The bound successor
validates AABB diagnostics before association and base invariants for every successful result and
for the first failing result. Nevertheless, only successful primary counters are serialized;
failed-primary and half-fit counter arrays are not independently available. Promotion evidence
should retain one per-fit AABB receipt rather than rely on bound control flow.

The audit parsed all 192 PLY files structurally without rendering them. Vertex payload sizes match
their headers, every float is finite, and every quaternion is nonzero. It verified all 66 root
boundary/resource receipt hashes, all 22 successful-only resource summaries, 111 dataset
presentation artifact hashes, all 11 comparison manifests, 28 referenced methods, and all root
representative copies. Visual PNG/GIF bytes were hashed but not opened or interpreted.

Every worker records one CPU and Torch thread, `cuda_used=false`, CUDA availability/device count
as inventory only, positive RSS, and finite load/fit/serialization/publication/process timings.
The environment is Linux 6.14, Python 3.12.9, NumPy 2.1.3, and Torch 2.9.0+cu128 with workspace
source. Successful process times span `14.803075918054674` to `937.7126565260114` seconds;
successful median RSS is 1,353,060,352 bytes. These are descriptive diagnostics. `x86_64` is not
a named host or meaningful CPU model, and no load/idleness receipt exists, so calibrated timing,
RSS, schedule speed, real-time, or general performance claims are unsupported.

## Producer artifact inventory

Before audit publication the canonical run root contains 565 producer files totaling 9,786,401
bytes. SHA-256 of compact canonical JSON over every sorted relative path, byte count, and file
SHA-256 is `0e12cb1e4b4435d0344e64e1fe288c285e35daf887c16a7fff9ceabc59a46bad`.

| Artifact | SHA-256 |
| --- | --- |
| `task.lock.json` | `d73b1d145385ae74a2ecd6f293e5a828e908ed6d156d1932ebc82fc66a40ae7e` |
| `run_receipt.json` | `09166b5701445f555931fdf131be250d0c91b0f80c4537078482f2508106c3b2` |
| `aggregate_commit_receipt.json` | `eb08a88b2b0332ac520e263ea0a2511f95824910f2d1215909b43dc4dfca2f2d` |
| `cell_results.json` | `bdc4c1e558a74dfc772d82a3f260a228ace23194f264a516f9013c09f8d6f270` |
| `metrics.json` | `ad5c43c8fd1e2eb8ffe1e2831a864c88430648505c63255958beaeafb67580a5` |
| `training_history.json` | `76f3b6bf80b82fbff7fb1e06a572dd4302dce806b7589927741285a7590495f4` |
| `input_boundary_receipt.json` | `5a4ef48927e451401194b404e8f293d35d01f4830752c665aa60b571e88ad9e7` |
| `resource_receipt.json` | `ec160527994ee88d36cff321a9b2e15d775a9c1f76cdc25b48882e62dbe5c1ad` |
| `synthetic/synthetic_results.json` | `7d2bae229d2b0d3a5210d5bdc3521b932ee55e412608453cc97e68db29ef0fe0` |
| producer result JSON | `a5aa2cfaefd74895abff600411a40d4c0bc2de1949871225ec0aa83c1b3f65b5` |
| producer result Markdown | `63dc629a4e5d4698453f546a6bbf8f64bdb3172643d1a2b205f7190d2330656b` |

## Evidence boundary and required downstream wording

This result supports only:

- the frozen exact-synthetic rank-aware covariance decision;
- the frozen controlled-corruption probability-support decision;
- rejection of the tested association, topology, and schedule mechanisms under their exact
  isolated rules;
- descriptive operability of native (`33/33`) and all-candidate (`26/33`) pipelines over eleven
  sealed, capped Gaussian2D field sets;
- narrow operation of the explicit unmasked fallback (`2/66`) and rollback-consistent
  zero-success continuation (`1/66`).

It does not establish complete-field fidelity, source-RGB reconstruction quality, physical
geometry, spatial resolution, GPS-Gaussian reproduction, true globally coupled multi-marginal OT,
cross-scene generality, GPU behavior, real-time speed, memory superiority, production readiness,
or a default change. It does not establish independent-half accuracy. Any public or ARA wording
must preserve these boundaries and the three failed mechanism decisions.

## Checks performed

- Independently recomputed task/protocol/review/data/source/driver/base/result hashes and the
  565-file canonical run inventory.
- Independently expanded the full 483 + 66 plan and recomputed every invariant, paired seed rule,
  aggregate median, denominator, fallback count, and success/failure indicator from raw JSON.
- Checked all 66 terminal directories, exact failure contracts, support-fallback provenance,
  input hashes, split isolation, AABB arithmetic, resource timing relations, histories, root
  consolidation, dataset curves, presentation-copy hashes, and PLY structure.
- `experiment_contract.py validate-data` and `experiment_contract.py validate` passed;
  `git diff --check` passed.
- The focused AABB/rollback/fallback/held-out/pipeline suite produced 27 passes and one lifecycle
  assertion failure. The failing test still expects the pre-review task status `draft`; the
  approved and executed task is correctly `ready`. Re-running the same focused selection without
  that stale assertion passed all 27 selected tests.

The stale lifecycle assertion must be updated and the complete `./scripts/verify.sh` gate rerun
before task closeout. That source/test repair is downstream of, and does not mutate, this immutable
producer result.

## Protected actions not taken

This audit did not rerun the experiment, mutate any producer artifact, edit the frozen task or
prospective review, change a default, render `index.html`, open PNG/GIF visuals, launch an orbit
viewer, perform browser smoke, claim visual usability, commit, push, or publish. CUDA/GPU work was
not executed.
