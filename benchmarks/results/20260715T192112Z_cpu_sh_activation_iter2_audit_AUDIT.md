# Independent scientist audit — SH color-floor Phase A

## Disposition

**Phase-B execution clearance: FAIL.** The frozen Phase-A materiality gate failed in all three
view-dependent seeds and in the pooled result. The protocol therefore permanently forbids both
candidate arms; no passing review manifest may be created and the `ablate` command must not run.
This corrects the pre-review wording in the companion result note: Phase B is no longer merely
waiting for independent review.

The disposition is limited to this CPU, fixed-topology, synthetic, depth-initialized
Torch-reference diagnostic. It is not evidence about SMU-1 candidate quality, real scenes,
CUDA/gsplat behavior or parity, density-control interaction, speed, memory, general 3DGS quality,
or a production default.

## Evidence binding

- Official audit JSON:
  `benchmarks/results/20260715T192112Z_cpu_sh_activation_iter2_audit.json`, SHA-256
  `67431510a5620b383db729d4877bb2c2b581eb81270f2110f0bd47b0d561f4ae`.
- Retry protocol: `benchmarks/results/20260715_sh_activation_iter2_PREREG.md`, SHA-256
  `d5558e188aae81187b7bb7906995aef99b0acc4d15a2c425b5bce91851995d6e`; it incorporates the
  original frozen scientific protocol at SHA-256
  `5353c4aa37c13e280f0bf3761679424e0bb5e17b4e942a7ff36275e84be88c1f`.
- Implementation seal: `benchmarks/results/20260715_sh_activation_iter2_SEAL.json`, SHA-256
  `403ce133922f57fa45a3374be34cb92a85fb043d0a1a6ce188c82fc808370de0`, sealed-source aggregate
  `9a1106d8eb9e90b4ace168367b46256f82434a89592004fd205a8d11738f758a`.
- Once-only retry marker:
  `benchmarks/results/20260715_sh_activation_iter2_PHASE_A_ATTEMPT.json`, SHA-256
  `fd3e2c06dcb6437f2a41ff3df762892a04bc364ea8be4b1c618fe96f3dbf9910`.
- Official command: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  .venv/bin/python benchmarks/sh_activation_ablation.py audit --seal
  benchmarks/results/20260715_sh_activation_iter2_SEAL.json --output
  benchmarks/results/20260715T192112Z_cpu_sh_activation_iter2_audit.json`.

## Independent recomputation

The decision was recomputed from the 90 eligible per-step diagnostic rows in each
view-dependent run, rather than trusting the stored summary or decision. Every recomputed ratio
matches the stored value to an absolute tolerance of `1e-15`.

| Frozen requirement | Seed 0 | Seed 1 | Seed 2 | Raw-sum pooled | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Negative channels >= 1% | 0.516717057% | 0.245288236% | 0.243934689% | 0.336526905% | fail, 0/3 seeds + pool |
| Recoverable gradient >= 5% | 0.107542448% | 0.142238015% | 0.017500317% | 0.090827994% | fail, 0/3 seeds + pool |
| Fixed-SMU-1 recovered gradient >= 0.5% | 0.037962190% | 0.030719679% | 0.006035015% | 0.025265713% | fail, 0/3 seeds + pool |
| Observations >= 10,000 | 226,236 | 224,226 | 219,321 | 669,783 | pass |
| Every training view sampled | yes | yes | yes | yes | pass |

The pooled numerator counts are 2,254 negative channels among 669,783 observations. Recomputed
seed passes are `[false, false, false]`; pooled pass and `phase_b_authorized` are both `false`.
Diffuse-condition diagnostics are reporting-only and cannot rescue this decision.

## Provenance findings

The first once-only attempt at `2026-07-15T19:14:37Z` is preserved by
`benchmarks/results/20260715_sh_activation_PHASE_A_ATTEMPT.json` (SHA-256
`af764e81d6afd36736fe95835553b795ba90b43b9ab5f6c14d6afe8ea92029c3`). It completed the six
hard-arm trainings, then failed closed before serialization because the verifier treated loaded
Pillow modules under `.venv` as unsealed repository source. No first-attempt audit artifact exists,
and the recorded console output exposed no diagnostic fraction or quality metric.

The retry protocol states that only this classifier changed. That chronology is credible but not
independently reconstructible: the first-attempt harness content/diff was not archived, while the
old seal and attempt marker were retained outside the retry's sealed-path set. Their current hashes
and the incorporated original protocol hash match the retry record, but they do not recover the old
harness source. In addition, the retry's environment fingerprint binds Python, PyTorch, platform,
CPU/thread settings, and deterministic flags but not the Pillow version, despite Pillow being the
external package that triggered the retry. These limitations narrow provenance; they do not affect
the arithmetic direction or authorize Phase B.

## Claim disposition

| Claim | Disposition | Reason |
| --- | --- | --- |
| The hard SH floor is materially active in this protocol | retire | All three incidence/material-gradient gates failed in every seed and pooled. |
| SMU-1 or the hard-forward control improves refinement | untested | Phase B was stopped before either candidate was trained. |
| The hard activation remains the repository default | confirm | The sealed default-semantic checks report `hard`; no default change was authorized. |
| The result transfers to CUDA, real scenes, density-enabled training, or performance | unsupported | None of those evidence classes was executed. |

No further evidence is needed to enforce this protocol's stop. A new scientific question requires
a new preregistration and evidence; it cannot reopen this consumed Phase B.
