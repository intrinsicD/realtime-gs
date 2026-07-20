# Compact occupancy-point refinement factorial — failure audit — 2026-07-17

## Verdict

**PROTOCOL FAIL; SCIENTIFICALLY INDETERMINATE.** The exclusive official attempt was correctly
consumed and terminated, but it stopped while constructing the third evaluation-bank archive.
No factorial arm, optimizer update, checkpoint evaluation, primary contrast, secondary contrast,
or decision gate ran. The immutable result's `decision="NO_REFINEMENT_TARGET_PROMOTION"` is
therefore **retired as a scientific decision**. It may be retained only as the conservative
operational fact `promotion_authorized=false`: a failed attempt cannot authorize a follow-up, but
it provides no evidence that the proposal-attempt target or balanced schedule fails.

The immediate cause is a native-resolution float32 endpoint bug in the continuous uniform branch
of `GaussianPointProposal.sample`. A value strictly below one can round to the fitted window's
exclusive upper coordinate after the float32 multiply/add. The official invariant correctly
rejected that bank. This is not a CUDA optimizer, point-rasterizer, RGB-denial, compact-teacher,
camera, or 3D-initialization failure; execution had not reached a worker.

The original preregistration, review, seal, attempt, terminal result, and partial bank archives
must remain immutable. A repaired experiment requires a new append-only namespace, new seeds, a
new review and seal, and a second once-only attempt.

## Claims and dispositions

| # | Claim | Kind and scope | Evidence | Disposition |
| --- | --- | --- | --- | --- |
| 1 | The official Cartesian product completed and can answer D versus B. | Asserted, calibrated single-scene fixed-topology CUDA development experiment | Terminal result plus partial run directory | **Retire.** Status is `FAIL`; no worker exists and no metric was computed. |
| 2 | `NO_REFINEMENT_TARGET_PROMOTION` is the preregistered scientific outcome. | Asserted decision | `RESULT.json` has the string but no `decision_metrics`, gates, or workers | **Retire as scientific; narrow to fail-closed operations.** It means only that this failed attempt authorizes nothing. |
| 3 | The one-shot lifecycle was consumed and failed closed without overwriting official artifacts. | Proven protocol/lifecycle fact | Linked seal, attempt, result hashes; exclusive files; terminal traceback | **Confirm.** The failed attempt cannot be retried under the same namespace. |
| 4 | The failure occurred before arm execution. | Proven execution-scope fact | Two bank files, absent third bank and manifest, empty worker directory, source chronology | **Confirm.** Zero of twelve workers ran. |
| 5 | The implemented float32 continuous uniform branch always produces fully active fitted-window draws and bounds importance by the inverse uniform fraction. | Asserted implementation invariant, especially native-resolution translated windows | Sealed sampler source, failed invariant, endpoint arithmetic | **Retire in this unconditional form.** It holds only when the affine float32 transform remains strictly below each upper bound; the implementation needs an endpoint-safe repair. The discrete-pixel proposal is not implicated. |
| 6 | The two completed proposal-bank archives pass the active-mass guards. | Measured partial diagnostic | Immutable banks 76501 and 76502 | **Narrow to non-decisional diagnostics.** Their 14 active fractions span `0.991943359375` to `0.997802734375`, ratio `1.0059069653`, but the required 21-bank population is incomplete. |
| 7 | The pre-seal implementation review's bank section had no remaining blocker. | Asserted review conclusion | Passing review versus realized official bank failure | **Superseded for this point.** Its source/lifecycle checks remain useful, but its uniform-bank conclusion missed the native-resolution float32 endpoint case. |

No official-factorial outcome claim currently appears in `README.md`, `docs/EXPERIMENTS.md`,
`docs/ROADMAP.md`, `ara/PAPER.md`, or `ara/logic/claims.md`. `docs/RESEARCH.md` does make the
broader implementation-level statement that the continuous proposal's uniform branch covers the
background and bounds importance. That statement must be qualified as the ideal/intended
estimator, or deferred until the endpoint-safe implementation is reviewed. This audit does not
reopen the separate, artifact-bound 2026-07-16 synthetic comparison; it does reject extrapolating
an unconditional native-resolution float32 sampler guarantee from it.

## Immutable chronology

All times below are UTC. Filesystem times are used only to order the append-only artifacts; JSON
timestamps are used where available.

| Time | Event and independently checked consequence |
| --- | --- |
| `2026-07-17T01:19:52.284180696Z` | Preregistration file mtime. It freezes three training seeds, three evaluation seeds, four arms, and the fully-active uniform-bank invariant. |
| `2026-07-17T02:03:57.690514306Z` | Passing implementation-review file mtime. The review reports 48 focused tests and no official namespace artifacts at completion. |
| `2026-07-17T02:05:07Z` | Seal timestamp. Its payload digest, current source hashes, and review/preregistration links validate. |
| `2026-07-17T02:05:46Z` | Exclusive attempt timestamp. The attempt links exactly to the seal, preregistration, and frozen config. The run directory and empty `workers/` directory were created immediately afterward. |
| `2026-07-17T02:05:47.126049093Z` | Complete immutable evaluation-bank archive for seed 76501 published. |
| `2026-07-17T02:05:48.011037052Z` | Complete immutable evaluation-bank archive for seed 76502 published. |
| before `2026-07-17T02:05:48.797218898Z` | Seed 76503 bank construction reached the uniform-bank guard and raised `ProtocolInvalid("uniform evaluation bank is not fully active/direct")`. Because an archive is published only after all seven views are built, `banks_76503.npz` does not exist. |
| `2026-07-17T02:05:48Z` | Terminal result timestamp (`mtime 02:05:48.797218898Z`). It records `status="FAIL"`, the exception and traceback, and the exact seal/attempt/preregistration links. |

`bank_manifest.json` is absent, `banks_76503.npz` is absent, and `workers/` contains zero entries.
In the sealed control flow, every bank must be completed and the manifest published before the
first worker subprocess is launched. Those facts establish that no official training seed was
executed and no optimization outcome was accessed.

## Failure localization and root cause

### What the immutable artifacts prove

`generate_bank_archive` calls `GaussianPointProposal.sample(..., uniform_fraction=1.0)` and then
requires all three of the following for a uniform bank: every attempt is active, every coordinate
is inside the fitted window, and every proposal component id is `-1`. With uniform fraction one,
PyTorch uniform deviates are below one, so the branch selector makes every attempt uniform;
`active` is copied from that selector and component ids remain initialized to `-1`. The only
reachable failed predicate is therefore `inside_fit_window`.

The sampler computes each continuous coordinate in the field's float32 dtype:

```text
xy = rand_float32 * fit_size + fit_origin
```

while `valid_domain` checks the mathematically half-open interval with strict upper bounds:

```text
fit_origin <= xy < fit_origin + fit_size
```

Those two operations are not closed under the intended half-open interval in float32. A post-fail
localization by the producing session reported evaluation seed 76503, view `C0039`, uniform
attempt index 2051, coordinate `[4881.0, 4050.303955078125]`, and fitted window
`[1282, 1936, 3599, 2225]`. The x coordinate is exactly the exclusive upper bound
`1282 + 3599 = 4881`, so `inside_fit_window=false`. The immutable result itself records only the
guard and traceback, not this seed/view/index; this audit therefore treats the exact localization
as a post-failure diagnostic and independently checks its causal arithmetic rather than claiming
it was sealed output.

An excluded, seed-free arithmetic check reproduced the mechanism. The largest float32 below one
is `0.9999999403953552` (bits `0x3f7fffff`); the producing session reported that exact raw value
at the failing index, rather than an RNG value equal to one. For the reported x window:

- the exact-real gap from the upper bound after multiplication is `0.00021451711654663086`;
- the float32 predecessor of `4881.0` is `4880.99951171875`, giving an ULP of
  `0.00048828125` and half-ULP `0.000244140625`; and
- because the exact-real gap is smaller than half an ULP, float32 multiply/add rounds the result
  to `4881.0`, which fails the strict `< 4881` check.

This explains both the rare incidence and why the small synthetic pre-seal fixture passed. The
factorial test generated a bank on an `8 x 6` synthetic window using evaluation seed 76501; it did
not exercise translated native-resolution windows, force the largest representable deviate, or
test an affine result at the exclusive endpoint. It also invoked an official evaluation seed in a
synthetic focused test, so the preregistration's unqualified word “fresh” is technically too broad
even though that test exposed no official teacher/model metric and did not cause this failure.
Future official seeds must never be used by focused tests.

### Consequences for the estimator

Simply accepting the out-of-window attempt would be incorrect. In a mixed proposal, `active` can
remain true while `valid_domain` removes the uniform term from `proposal_density`; this can break
the advertised inverse-uniform-fraction importance bound. The correct repair is to make uniform
coordinate construction half-open in the field's actual dtype, not to weaken the bank guard,
resample a failed official attempt, drop the attempt, or normalize by the active count.

## Artifact integrity and partial data

The seal's internal `seal_payload_sha256` recomputes exactly. Every sealed source-path hash still
matches the current corresponding file. The attempt's seal, preregistration, and config links
recompute, and the result's attempt, seal, and preregistration links recompute. The two NPZ files
pass ZIP integrity and the sealed `load_bank_archive` tensor/metadata hash validation.

| Artifact | SHA-256 | Disposition |
| --- | --- | --- |
| Preregistration | `72553e528cbd12185b3845e63ab5367d4e78af3711acfc850383bebd7519f2bf` | Immutable, consumed protocol |
| Implementation review | `3aef740057e5c16bb822b400aef8acfdbd601d8ecb52843770b8950592e971f3` | Immutable, superseded only on endpoint coverage |
| Seal | `8d3299b1c67f1d7aa125423846d96104556d82864115f0f6489335646f66451c` | Immutable; internal payload SHA validates |
| Attempt | `11c75fd2257041b344481a052fed96267ea78031d7f29d5b651c71bf7a6fe763` | Immutable and consumed |
| Terminal result | `d8030691fba7ebba3a77783473bffd538a1ca4640930e70d311cd6f6c454f520` | Immutable protocol failure; scientific decision unavailable |
| Sealed harness | `7099a1662a01909efab5c6effd66b9a4182b87b9b50112ef57acc9db2ca4de64` | Exact executed source |
| Sealed continuous sampler | `6e729ee825497a954d3653fc9bab3823fb7d6473a1337b869aa4f33a01c8806e` | Contains endpoint-unsafe affine transform |
| Focused factorial tests | `b14abd4910fab5159a6019b818402b03ce9638d4e987853906f816a6ee262f2f` | Misses forced endpoint/native-window case |
| Bank 76501 | `5f8c630dfae0138138ef952103e816f829fea372cf148009e84797bc8017d989` | Valid but partial/non-decisional; 1,259,916 bytes |
| Bank 76502 | `8d5ae934da9e10b7c2a0260fb13ebdb47fe6658e101383f20124ff54797f7d83` | Valid but partial/non-decisional; 1,259,959 bytes |

The completed banks' semantic hashes are respectively
`8253259d8ce6ea872824ab7044c84284720d0b7452045870d9afd474247f30b4` and
`86f5476026a75750fdf28b5d3c588339f9ceb72274f27a91c9b2806828f249a4`.
All 14 completed uniform view banks have 4,096 active, inside, direct attempts. These are integrity
facts only; neither partial bank may enter a repaired experiment or any aggregate.

## Required append-only repair experiment

1. **Preserve the failure.** Do not edit, remove, rename, or reuse the original preregistration,
   review, seal, attempt, result, run directory, or two bank archives. Link this audit from the new
   preregistration and experiment ledger.
2. **Repair the primitive, not the guard.** Centralize continuous uniform-coordinate construction
   and clamp each affine result to the dtype predecessor of the exclusive upper bound (for example,
   `nextafter(upper, lower)`) while retaining the lower bound, fixed attempt count, target density,
   null semantics, and original RNG consumption. Do not resample, discard, or reinterpret the
   boundary attempt.
3. **Add deterministic boundary tests.** Inject/construct the largest float32 below one and test
   both axes on translated official-scale windows. Require all uniform draws to be active, direct,
   strictly inside, finite, and positive-density. For mixed proposals, require finite importance
   and the `1/eta` bound. Test float32 and float64, zero and translated origins, and native-scale
   widths/heights without depending on a lucky RNG seed.
4. **Separate test and official RNG domains.** Focused tests must use test-only seeds and may not
   call any new official training/evaluation seed. The official wrapper can enforce its seed set,
   but the testable bank builder needs an explicitly separate test namespace.
5. **Improve failure receipts.** If a bank invariant fails, record seed, view, kind, failing
   predicate, first failing index, coordinate, fitted window, and tensor hashes in an append-only
   failure diagnostic. A failed terminal result should use `scientific_decision="UNAVAILABLE"`
   (or equivalent) plus `promotion_authorized=false`, not a label shared with an evaluated negative
   result.
6. **Create an independent `iter2` lifecycle.** Freeze the same scientific arms, budgets, metrics,
   and thresholds unless a pre-outcome rationale explicitly changes them. Use fresh training and
   evaluation seeds, a new run directory and artifact prefix, a fresh independent implementation
   review, a new exact source/input/runtime seal, and one new exclusive attempt. Reusing even the
   untouched original training seeds would blur the consumed attempt; fresh seed sets are the
   clean repair.
7. **Regenerate all banks.** Do not copy either partial bank. The repaired attempt must build and
   hash all 21 uniform/proposal view pairs before launching any arm, then run all twelve fresh
   bounded workers and recompute the original decision arithmetic.
8. **Audit before claiming.** Only a `status="PASS"` iter2 result with all gates present can yield
   either scientific decision. Run a new independent results audit before density follow-up,
   default changes, documentation outcome prose, or gsplat/viewer visualization claims.

## Checks actually performed

- Read `CLAUDE.md` and `.claude/skills/realtime-gs-results-audit/SKILL.md` in full.
- Read the preregistration, implementation review, seal, attempt, terminal result, sealed harness,
  sampler/query implementation, focused tests, partial run directory, and relevant public/ARA
  claim surfaces.
- Recomputed all official JSON linkage hashes and the seal payload digest; checked every sealed
  source-file hash against the live corresponding bytes.
- Loaded only the two already-published bank archives through the sealed integrity checker and
  recomputed their active counts; ran `unzip -t` on both.
- Performed only seed-free float32 endpoint arithmetic using an explicitly constructed largest
  representable value below one.
- Ran `git diff --check` (pass at audit time).

No seal/run/worker operation was invoked. No official training or evaluation seed was replayed,
no bank was regenerated, no CUDA/GPU work ran, and no partial metric was promoted. The focused
factorial test suite was deliberately not rerun because its current bank-roundtrip test invokes an
official evaluation seed; its passing pre-seal transcript remains bound in the immutable seal.
