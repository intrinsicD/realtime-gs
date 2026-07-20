# Compact occupancy-point refinement factorial iter3 — independent results audit — 2026-07-17

## Verdict

**PASS — CONFIRM `AUTHORIZE_DENSITY_FOLLOWUP`, with the preregistered strict scope.**

The immutable iter3 result is complete, internally linked, source/input/runtime bound, RGB-free,
and arithmetically correct. Independent recomputation from the three bank archives, twelve worker
JSON files, twelve complete histories, checkpoint records, and raw artifact hashes reproduces all
seven authorizing D/B gates exactly. The decision is therefore confirmed as:

```text
status=PASS
scientific_decision=AUTHORIZE_DENSITY_FOLLOWUP
promotion_authorized=true
```

This authorizes only a new, separately preregistered RGB-free variable-\(N_{\mathrm{opt}}^{3D}\)
density-control experiment. It does **not** authorize a production default, a balanced-schedule
claim, a source-RGB-equivalence claim, a novel-view claim, a scaling claim, or a speed/memory
claim.

## Claim table

| # | Claim | Kind and scope | Independently checked evidence | Disposition |
| --- | --- | --- | --- | --- |
| 1 | Iter3 completed the frozen four-arm, three-seed factorial. | Proven lifecycle fact; one calibrated scene, CUDA, fixed topology. | Three banks precede all twelve PASS workers; every worker has 140 steps and four checkpoints; immutable result links to the attempt and seal. | **Confirm.** |
| 2 | The endpoint-safe continuous-uniform path is valid on this native-resolution compact bundle. | Measured mechanism fact. | All 86,016 uniform attempts are active, direct, finite, and strictly half-open; all raw tensor hashes and generator-seed derivations match. | **Confirm narrowly.** This is not a general sampler proof beyond the reviewed implementation and inputs. |
| 3 | D improves final proposal-attempt risk over B. | Preregistered primary development contrast, D/B on \(J_Q\). | Per-seed D/B ratios `0.7816781153`, `0.7799140251`, `0.7705785062`; geometric ratio `0.7773749194`. | **Confirm.** Gate `<=0.95` passes. |
| 4 | D improves checkpoint-curve proposal-attempt risk over B. | Preregistered log-AUC contrast. | Per-seed AUC-derived ratios `0.8860028699`, `0.8816811726`, `0.8762084839`; geometric ratio `0.8812883921`. | **Confirm.** Gate `<=0.97` passes. |
| 5 | The D/B improvement is seed-consistent. | Preregistered paired-win gate. | D has lower final \(J_Q\) in all three seeds. | **Confirm.** `3/3`, gate `>=2/3` passes. |
| 6 | D does not materially harm uniform-area risk relative to B under the frozen safety rule. | Preregistered safety claim on equal-view final \(J_U\). | Per-seed D/B ratios `0.9431259244`, `0.9571952741`, `0.9437474863`; geometric ratio `0.9480007454`. | **Confirm under the frozen aggregate gate.** Geometric `<=1.05` and every seed `<=1.10` pass. There was no preregistered per-view \(J_U\) gate. |
| 7 | Proposal-bank active mass is adequate and comparable. | Preregistered bank-population guard. | Twenty-one raw fractions span `0.991943359375`–`0.99853515625`; max/min `1.00664533596`. | **Confirm.** Both `>=0.95` and `<=1.03` gates pass. |
| 8 | Proposal-attempt targeting, rather than scheduling alone, is the supported mechanism. | Measured secondary factorial interpretation. | C/A final \(J_Q\) geometric ratio `0.7814970779`, AUC ratio `0.8779345587`, with all six per-seed ratios below one; schedule-only B/A and D/C AUC ratios are `0.9993200569` and `1.0031376000`. | **Confirm narrowly as a development interpretation.** The primary authorizing result remains D/B. |
| 9 | `balanced_cycle` improves refinement. | Secondary schedule claim. | B/A and D/C effects are small, AUC directions are mixed across seeds, and no schedule-specific promotion gate was preregistered. | **Withhold.** Do not describe balanced scheduling as established or select it as a default on this evidence. |
| 10 | The result authorizes a density-control follow-up. | Preregistered stage-transition decision. | Every primary, safety, active-mass, pairing, step-zero, binding, worker, and RGB invariant passes. | **Confirm narrowly.** It authorizes opening the exact follow-up defined below, not its outcome. |
| 11 | Iter3 proves variable-count quality, scalability, ordinary-3DGS superiority, source-RGB equivalence, novel-view quality, or production readiness. | General capability/performance assertions. | Outside this single-scene, same-camera, fixed-835-Gaussian teacher-risk protocol. | **Reject / still unverified.** |
| 12 | The post-result gsplat contact sheet and live viewer close the required diagnostic handoff. | Post-result visualization fact, non-decision-bearing. | Twenty-eight native 5328x4608 PNGs and a contact sheet hash-match the receipt; zero source-RGB/dataset opens; live endpoint returned HTTP 200. | **Confirm as diagnosis only.** The images are not metrics and cannot change the decision. |

No positive iter3 outcome claim was present in `README.md`, `docs/`, `ara/PAPER.md`, or
`ara/logic/claims.md` at audit time. Any later documentation must preserve the dispositions above.

## Immutable artifacts and chronology

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Preregistration | 10,175 | `5b3f721307b2f85446a1862584406ec9383ea63a75ff93585b7840f244861ef8` |
| Implementation review | 11,650 | `308e225b8f3bfbd9611ce4a45db40a1ccf3a0d294fc371e7d491333d62e294d5` |
| Seal | 17,746 | `0dd59fbab95b317b60e02f04343311e51022a0fac1c40105af62c3be29ac8d0e` |
| Exclusive attempt | 397 | `270f3a972d0931d6e51c37574da245a9e76f203aaab347254537d3195b18f155` |
| Bank manifest | 47,829 | `3f25005768c3089d59bb4bc1028e2239a8a62209a4cc5d5371958dc14261afd3` |
| Terminal result | 360,639 | `c0a278a8cc41f12632be121b14937f9fc2a2a03cd03716bae96b5bd9d6510116` |
| Executed-source archive | 542,720 | `ec23f809d2b5b7e720d90a4f4630c5fe5115021c5dca3ef548244a3302399875` |

The seal's internal payload digest recomputes. Its preregistration and implementation-review links
match. The attempt links to that seal and its config hash; the result links to the exact attempt,
seal, preregistration, review, and bank manifest.

Chronology is fail-closed:

| UTC | Event |
| --- | --- |
| `2026-07-17T02:57:49Z` | Iter3 preregistration existed before review and outcome access. |
| `2026-07-17T03:22:11Z` | Independent implementation review existed before sealing. |
| `2026-07-17T03:23:19Z` | Seal published. |
| `2026-07-17T03:23:35Z` | Once-only attempt token published. |
| `2026-07-17T03:23:36Z`–`03:23:38Z` | Three fresh bank archives published. |
| `2026-07-17T03:23:38Z` | Bank manifest published before any worker. |
| `2026-07-17T03:23:43Z`–`03:24:47Z` | Twelve workers published in the frozen seed/arm order. |
| `2026-07-17T03:24:47Z` | Terminal PASS result published. |
| `2026-07-17T03:27:10Z`–`03:27:58Z` | Non-decision-bearing gsplat plan and receipt published after the result. |

The decision-bearing run subset contains exactly 88 files, 9,771,247 bytes. An audit-generated
canonical `{path,bytes,sha256}` manifest has aggregate
`7b979ab3b0ebb78b42b7f89907509cfce2738dba4f3eb45eb68af31f09b525e8`.
This aggregate is an audit snapshot, not a field retroactively inserted into the result.

Because the repository is dirty and much of the experiment source is untracked, hashes alone
would not be durable preservation. Before this audit was published, the still-matching 26-file
source set was archived append-only. Independent tar inspection found exactly 26 ordinary-file
members, no links or extras; every member matches `seal.source_hashes`, and their canonical
aggregate is the sealed
`a88ed18968bc2d6f5439af729134ea83a342e34e21399ee4adc7562647631bfc`.
The archive was created post-result and is bound here rather than being falsely represented as an
original result field.

## Independent decision recomputation

The checkpoint abscissae are `(0,0.25,0.5,1)`. I recomputed each AUC as the trapezoidal integral
of `log(max(J_Q,1e-12))`, then exponentiated the D-minus-B log-AUC difference.

| Seed | B final \(J_Q\) | D final \(J_Q\) | D/B final \(J_Q\) | D/B AUC \(J_Q\) | B final \(J_U\) | D final \(J_U\) | D/B final \(J_U\) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 76801 | 0.005802279823 | 0.004535515156 | 0.7816781153 | 0.8860028699 | 0.004672441034 | 0.004406700270 | 0.9431259244 |
| 76802 | 0.006127326875 | 0.004778788166 | 0.7799140251 | 0.8816811726 | 0.004818275314 | 0.004612030360 | 0.9571952741 |
| 76803 | 0.005817722787 | 0.004483012135 | 0.7705785062 | 0.8762084839 | 0.004369694321 | 0.004123888031 | 0.9437474863 |

The raw \(J_Q\) curves used for the AUC calculation were:

```text
76801 B=[0.019639866663,0.013393198293,0.009578300695,0.005802279823]
      D=[0.019639866663,0.012728680743,0.008455983195,0.004535515156]
76802 B=[0.019659005628,0.013639814228,0.009729090475,0.006127326875]
      D=[0.019659005628,0.012926743946,0.008506511103,0.004778788166]
76803 B=[0.019327597282,0.013371526563,0.009464072486,0.005817722787]
      D=[0.019327597282,0.012690064536,0.008196556714,0.004483012135]
```

| Frozen gate | Threshold | Recomputed | Result |
| --- | ---: | ---: | --- |
| Geometric D/B final \(J_Q\) | `<=0.95` | `0.7773749194491785` | PASS |
| Geometric D/B AUC-derived \(J_Q\) | `<=0.97` | `0.8812883921241239` | PASS |
| Strict final-\(J_Q\) wins | `>=2/3` | `3/3` | PASS |
| Geometric D/B final \(J_U\) | `<=1.05` | `0.9480007453723027` | PASS |
| Every-seed D/B final \(J_U\) | `<=1.10` | max `0.9571952741292606` | PASS |
| Every proposal active fraction | `>=0.95` | min `0.991943359375` | PASS |
| Proposal active max/min | `<=1.03` | `1.0066453359586511` | PASS |

Every one of the 21 seed/view final-\(J_Q\) ratios is below one, spanning
`0.5863871692`–`0.8434028079`. Final per-view \(J_U\) ratios span
`0.8714837093`–`1.0867345760`; therefore one view worsened by about 8.7% even though every
preregistered equal-view/per-seed safety gate passed. This is a caveat, not a failed gate.

The secondary factorial contrasts independently reproduce:

| Contrast | Geometric final \(J_Q\) ratio | Geometric AUC \(J_Q\) ratio | Interpretation |
| --- | ---: | ---: | --- |
| C/A, target change under iid | `0.7814970779` | `0.8779345587` | Large, seed-consistent proposal-target effect. |
| B/A, schedule change under uniform | `0.9834283279` | `0.9993200569` | Small/mixed; no schedule conclusion. |
| D/C, schedule change under proposal | `0.9782410438` | `1.0031376000` | Small/mixed; no schedule conclusion. |

## Banks, accounting, and invariants

- All three NPZ file hashes, sizes, embedded metadata, semantic digests, exact member allowlists,
  294 tensor-array descriptors, shapes, dtypes, and content hashes match the manifest.
- The 42 uniform/proposal generator seeds independently match the first-eight-little-endian-byte
  SHA-256 derivation and are unique.
- All 86,016 uniform attempts are active, inside, direct (`component_id=-1`), finite,
  non-negative-density, and strictly half-open. The nearest observed coordinate is
  `0.00634765625` pixel below an exclusive upper endpoint.
- All 86,016 proposal attempts are retained: 85,665 active and 351 null. Active rows imply
  inside-window rows; inactive joint density is zero.
- Every checkpoint's per-view `loss_sum / 4096` exactly reproduces its \(J_U\) or \(J_Q\), and
  every top-level checkpoint risk is the equal mean of seven views. \(J_Q\) was never divided by
  active count.
- Every arm has \(m_{\mathrm{init},i}^{2D}=m_{\mathrm{opt},i}^{2D}=640\) for seven views,
  \(\sum_i m_{\mathrm{opt},i}^{2D}=4480\), and
  \(N_{\mathrm{init}}^{3D}=N_{\mathrm{opt}}^{3D}=835\). This verifies accounting, not scaling.
- All twelve histories have exactly 140 updates. All six Adam groups report 140 steps; teacher
  and proposal digests are unchanged; trainer-internal checkpoint evaluation remains disabled.
- B and D each contain exactly twenty complete seven-view cycles per seed.

### Step zero and paired streams

All twelve step-zero snapshot files share semantic SHA-256
`4f6a7295f37ea98b2c3fdcdb41b1a93a5feefe5a8e2520474445594e46dde67c`.
Within each seed, all four arms have byte-identical step-zero checkpoint metrics:

| Seed | Canonical step-zero metric SHA-256 |
| ---: | --- |
| 76801 | `9ff48dd97a0d307b39850c4865bb4e1c81344fc2c40fb9b6266b646a6bd5d253` |
| 76802 | `2dfa70a0869fca4d2dfedd49e7ade9c13bad4fdca8c6b3e8125b141d24d8524e` |
| 76803 | `6da0810abacc203705932f4485b5ae51c47caec1f6ecc7d6044aa000c005505c` |

For every seed, A/C and B/D match all 140 steps on view index/name, sample seed, coordinates,
active and inside flags, component IDs, proposal density, and joint density. In each of the six
paired histories, `target_density_sha256` and `importance_sha256` differ on all 140 steps. Thus
the target mode changed while the paired sampled stream remained fixed.

## Source, inputs, runtime, and RGB boundary

All 26 live source files still match the seal at audit time. Independently recomputed input
aggregates are:

| Input | Files | Aggregate SHA-256 |
| --- | ---: | --- |
| Compact teacher bundle | 8 | `56a02fbdf3f4f2d61d9358f486c90f6c963449c0642533859395b0c6e2f21db7` |
| Center-occupancy proxy | 9 | `73e070fdfab42147501f94561a47681f79d26b7ff98450e31d4bf0a8d6084176` |
| Consumed iter2 run | 11 | `52643df0cd254f6fe48701929bcddf3fe2b23e36391e3d54f9870ac2fc6739ee` |
| Common initialization PLY | 1 | `0cf0340117739bb4b0491ff9c90d8d4b622b57a57f6bf8e6a3cfc9984b5c416e` |

Every prior-attempt provenance file also matches its sealed size and hash. Parent entry/exit and
all 24 worker entry/exit receipts equal the independently reconstructed expected receipt:

```text
semantic_sha256 = dd4454ce2e83ec5df3fbc4560d42351947204309170ff6983ca85b7a4050dc61
source_aggregate = a88ed18968bc2d6f5439af729134ea83a342e34e21399ee4adc7562647631bfc
runtime_sha256   = 165f117243cbc93fa5374cc2b58c257d96ee5d62ca0e49b57f49dc7902772489
config_sha256    = d3b5902ff4d4cbbc53a290b0d2cee5692421422a36c524771d09195eeab77652
```

The bound environment is Python 3.12.9, PyTorch `2.9.0+cu128`, CUDA 12.8, gsplat 1.5.3, and an
NVIDIA GeForce RTX 3050 (capability 8.6, driver 590.48.01). The exact preload and normalized,
narrowly validated PyTorch instantiator path record match across seal, parent, and workers.
These facts establish execution provenance, not idle-GPU performance.

The parent plus twelve worker denial records each report all three live negative controls firing,
zero source-RGB opens, zero forbidden imports, and no forbidden module crossing. Total independent
negative-control denials are 39. Compact 2D-Gaussian teacher colors in the point banks are allowed
protocol inputs; source images and calibrated RGB loaders were not accessed. This audit also did
not open source RGB.

## Failure history

The two earlier namespaces remain scientifically indeterminate and are not pooled:

1. Iter1 stopped during bank generation on the float32 exclusive-upper-endpoint bug. Zero workers
   ran. Its `NO_REFINEMENT_TARGET_PROMOTION` string was previously retired as a scientific
   decision.
2. Iter2 completed all banks and one arm-A optimization, then failed on an overly literal raw
   `sys.path` receipt after normal Adam/TorchDynamo initialization. No checkpoint metric was
   persisted or reconstructed and no paired contrast existed.

Iter3 uses fresh train seeds `76801..76803`, evaluation seeds `76901..76903`, banks, worker
processes, namespace, seal, and attempt. Its successful evidence neither rehabilitates nor
silently reuses either failed attempt.

## Post-result gsplat diagnostic

The post-result receipt is
`runs/compact_occupancy_refinement_factorial_iter3_20260717/visualization_seed_76801/RECEIPT.json`,
SHA-256 `607205ed1f7d99d076bf89aaa63f4fcb0a4b06c2dc075031605d54f8247a0d69`.
Independent file inspection confirms:

- 28/28 PNGs are 5328x4608, one for each seed-76801 arm/camera pair;
- gsplat 1.5.3, `packed=false`, `antialiased=false`;
- all four input PLY hashes equal the official worker records;
- source-RGB/dataset open attempts are zero;
- contact sheet SHA-256
  `add70e4e396a80fd47cef249df94afe25cd839e7d7e3e150d4c58afb96ffd199`;
  and
- `http://127.0.0.1:8879/` returned HTTP 200 and the `rtgs` process owned the listening socket at
  audit time.

The receipt explicitly marks itself `decision_bearing=false`. I did not rerender, inspect source
RGB, or derive any quantitative claim from the PNGs.

## Exact authorized next experiment

The authorized next step is a **new compact RGB-free residual-responsibility density-control
factorial**, not reuse of the older synthetic RGB residual-density preregistration.

Its causal baseline and all density arms must share the now-supported point-training regime:
`proposal_attempt` targeting, the same compact-teacher query semantics, matched point/view streams,
and the same 835-Gaussian initialization. Retaining `balanced_cycle` is allowed to preserve arm D
exactly, but must not be justified as a proven schedule improvement.

The new preregistration should isolate four topology policies:

1. `fixed`: no surgery;
2. `birth`: split high sampled-residual-responsibility parents;
3. `death`: prune low sampled responsibility/support primitives; and
4. `birth_death`: combine the two policies with independently frozen wave quotas.

Residual responsibility must be computed only from compact-teacher point samples and detached
per-Gaussian point-render responsibilities; source RGB remains forbidden. Freeze before any
official outcome: fresh train/evaluation/test seeds, score arithmetic, persistent identities,
split/prune rules, event steps, maximum and minimum counts, optimizer-state surgery, per-wave
quotas, matched RNG streams, checkpoints, count/quality accounting, and material-effect/safety
gates. The last surgery must have preregistered recovery updates before evaluation. Record the
complete \(N_{\mathrm{opt}}^{3D}\) trajectory without forcing a constant or equal final count.

The authorizing comparison is `birth_death` versus `fixed` on fresh fixed-bank \(J_Q\), with
uniform-area \(J_U\) safety and explicit count accounting; birth-only and death-only arms identify
which operation caused the effect. A passing result would still be single-scene development
evidence and would require another independent audit before multi-scene/provider transfer.

## Checks and commands actually executed

Read-only independent Python programs, without importing or invoking the official harness,
performed:

- strict/canonical JSON parsing and all lifecycle/self/link hashes;
- all 26 source hashes, five input-section hashes, directory manifests, runtime/config receipts,
  and module origins;
- all three NPZ allowlists, 294 tensor hashes, generator seeds, endpoint predicates, active/null
  counts, and metadata semantic hashes;
- all worker/history/snapshot/PLY/process hashes, reductions, step-zero equality, paired streams,
  schedules, AUCs, gates, secondary contrasts, and per-view ratios;
- all executed-source tar members and hashes; and
- all visualization receipt, plan, PNG-header dimension, PLY, source, worker, extension, and
  contact-sheet hashes.

The repository checks actually run were:

```bash
RTGS_FACTORIAL_FOCUSED_TEST=1 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.33 \
.venv/bin/python -m pytest -q \
  tests/test_compact_occupancy_refinement_factorial.py \
  tests/test_compact_trainer.py \
  tests/test_observation2d.py
# 94 passed

.venv/bin/python -m ruff check \
  benchmarks/compact_occupancy_refinement_factorial.py \
  tests/test_compact_occupancy_refinement_factorial.py \
  src/rtgs/core/observation2d.py tests/test_observation2d.py \
  src/rtgs/optim/compact_trainer.py tests/test_compact_trainer.py \
  src/rtgs/data/__init__.py src/rtgs/optim/__init__.py

.venv/bin/python -m ruff format --check \
  benchmarks/compact_occupancy_refinement_factorial.py \
  tests/test_compact_occupancy_refinement_factorial.py \
  src/rtgs/core/observation2d.py tests/test_observation2d.py \
  src/rtgs/optim/compact_trainer.py tests/test_compact_trainer.py \
  src/rtgs/data/__init__.py src/rtgs/optim/__init__.py

git diff --check
curl --silent --show-error --max-time 3 \
  --output /dev/null --write-out 'viewer_http=%{http_code}\n' \
  http://127.0.0.1:8879/
ss -ltnp '( sport = :8879 )'
```

All passed. No seal/run/worker command was replayed, no official seed or bank generator was
invoked, no optimization or render was rerun, and no source RGB was opened. Full repository
`./scripts/verify.sh` is intentionally left to the producing session after documentation and ARA
updates; GPU timings were not treated as performance evidence.

## Remaining caveats

- Single scene, one compact producer/configuration, seven same-camera teacher risks, three seeds.
- Fixed topology at 835 Gaussians; no density operation was tested.
- \(m_{\mathrm{opt},i}^{2D}=640\) for every observed view; variable-\(m_i\) interfaces were
  accounted for but not empirically stressed.
- Metrics are compact-teacher sampled RGB-channel MSE risks, not source-image PSNR/SSIM/LPIPS or
  held-out novel-view metrics.
- No source-RGB equivalence, geometry accuracy, ordinary-3DGS comparison, multi-scene transfer,
  convergence proof, quality-per-byte result, or production default.
- Worker elapsed time and memory fields are descriptive only: there was no idle-GPU timing
  protocol, warmup/repeat aggregation, or performance gate.
- The viewer/contact sheet is a post-result visual diagnostic and cannot validate the quantitative
  decision.
