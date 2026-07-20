# Independent scientist audit: kernel-support Phase B retry 2

## Disposition

**Evidence validity: PASS. Frozen primary hypothesis: FAIL.** The official Phase-B artifact is
internally consistent, bound to the cleared retry-specific Phase A and sealed source, and its
stored summaries and decision reproduce exactly from raw held-out per-view values. Under the
preregistered fixed-topology CPU synthetic protocol, reject both the fixed `C=12`, `W=4` C1
forward taper and the hard-forward/taper-gradient attribution control. The hard default must not
change.

The Phase-A mechanism screen remains positive: the adjacent annulus contains material
loss-directed gradient by its frozen incidence gates. Phase B shows that exposing that gradient
with these two prespecified arms does not improve common-hard held-out quality. This is not
evidence that every support smoothing is ineffective, and it is not real-scene, density-enabled,
gsplat/CUDA, speed, or production evidence.

## Claim dispositions

| Claim | Evidence | Disposition |
|---|---|---|
| Phase A authorized the frozen candidate arms | Raw iter2 diagnostic records; bound Phase-A review | **Confirm.** All three diffuse seeds and the pooled result pass. |
| The C1 taper improves diffuse held-out foreground PSNR by at least 0.10 dB | Common-hard final evaluation in the Phase-B JSON | **Retire.** Mean gain is `-0.014483 dB`, with `0/3` seed wins. |
| Restored annulus gradients causally improve diffuse held-out PSNR | Hard-forward/taper-gradient control | **Retire.** Mean gain is `-0.018470 dB`, with `0/3` seed wins; both attribution gates fail. |
| The taper stays inside the prespecified quality guardrails | Common-hard final evaluation | **Confirm.** Every SSIM/depth/alpha/coverage/view-dependent guardrail passes, but guardrails cannot rescue the failed utility gate. |
| The result supports a renderer default change or broader 3DGS claim | This CPU synthetic fixed-topology experiment | **Withhold.** No real data, density control, gsplat/CUDA parity, or performance evidence was run. |

## Independent raw recomputation

All summaries below were recomputed from the three raw held-out view records for each seed. Hard
baselines came from the exact Phase-A artifact bound into Phase B; candidate parameters were
evaluated with the common hard reference renderer as preregistered.

| Diffuse seed | C1 taper PSNR gain (dB) | Hard-forward control gain (dB) | C1 taper SSIM delta |
|---:|---:|---:|---:|
| 0 | -0.018741 | -0.028500 | -0.000448 |
| 1 | -0.013265 | -0.013335 | -0.000430 |
| 2 | -0.011443 | -0.013576 | -0.000281 |
| mean | **-0.014483** | **-0.018470** | **-0.000386** |

The remaining frozen C1-taper guardrails also reproduce exactly:

- normalized expected-depth RMSE regression: `+0.00410576` (`+0.4106%`, limit `+2%`);
- alpha-IoU delta: `+0.00294248` (lower limit `-0.02`);
- foreground-coverage delta: `-0.000398106` (lower limit `-0.02`);
- view-dependent foreground-PSNR deltas: `[-0.009253, +0.011937, -0.011356] dB`, mean
  `-0.002891 dB` (mean limit `-0.10 dB`, per-seed limit `-0.25 dB`).

Thus the two utility criteria and both attribution criteria fail, while all seven safety/
replication guardrails pass. Recomputing the complete decision dictionary produced the artifact's
`primary_hypothesis_pass: false` exactly.

Matched-taper evaluation is correctly secondary. On the same taper-trained final parameters it
adds only `+0.006407 dB` mean diffuse foreground PSNR relative to common-hard evaluation. It does
not reverse the candidate's negative common-hard gain and cannot rescue the primary decision.

## Phase-A and execution bindings

The iter2 Phase-A raw additive recomputation reproduced every per-seed diagnostic summary, q-bin
summary, and pooled decision. Its diffuse pooled values are:

| Eligible observations | Annulus upstream | Recoverable | Recovered / active | Recovered / boundary |
|---:|---:|---:|---:|---:|
| 48,290,887 | 40.7745% | 24.6717% | 0.252269% | 5.43819% |

All three seeds pass and Phase B is authorized. The strict review JSON binds the exact Phase-A
SHA-256 `57421f39ff5d983ac37bc63e2c1eabe1a9528a6ed4415001d52a3ee9bce76609`, seal SHA-256
`e6b551222e7242ebf3d44a3fa9d7ede0b41daf39c19f2593a9a8406b5d266097`, and source aggregate
`4f13421bfb570e8e42570bb97f39aa88bb90c2c8d822864f4272fbb68786e674`.

The Phase-B attempt marker predates the output and binds that Phase A, review, and seal exactly.
The output SHA-256 is
`f44f3f3fa69fd6bdf67e8da61f90fec952ffb2a38c577a216ff256af0e2263fd`; the companion result
note reports the same digest and command. JSON parsing rejected non-standard numeric constants and
duplicate keys; every numeric leaf is finite. The 40 loaded repository-source hashes are a
matching subset of the 69 sealed paths, and every current sealed path still matches the seal.

The first Phase-B attempt remains consumed. Its marker is preserved at SHA-256
`0c3b1e96ab56680db64758c9e2ceb17a5c53bb5f950bfd416d1165b08433e3c1`, and its attempted JSON
and result note remain absent. The retry protocol was frozen before the iter2 seal, Phase-A rerun,
fresh review, and fresh Phase-B attempt. Comparing the first and iter2 seals shows only the
harness, its focused test, and the added retry protocol changed; repository renderer, trainer, and
other scientific source did not. The old console transcript was not archived, so the statement
that no first-attempt candidate metric was inspected is supported by the preregistered chronology
and absent output rather than an independently replayable log.

## Invariants, isolation, and controls

- All 12 required run identities are present once and in frozen order: two conditions, three
  seeds, and two candidate arms.
- A fresh deterministic recreation independently matched all six Phase-A scene, fitted-set, and
  initialization hashes, along with initial counts and preparation configurations. The sealed
  Phase-B harness checks those hashes before either candidate arm.
- Candidate configs differ from their Phase-A baselines only by the prespecified support mode and
  disabling diagnostic retention. Every run uses the CPU Torch renderer, 120 iterations, fixed
  topology, seed-matched initialization, and no density strategy.
- Each candidate has the exact Phase-A sampled-view, active-SH-degree, primitive-count, and
  checkpoint schedule. All 120 sampled views are training views; held-out views `[3,7,11]` occur
  only in final reporting. Final counts equal initial counts for every arm.
- The step-zero kernel bound is exactly `exp(-6)`, the hard-forward control/taper gradient error is
  zero, and the sealed check enforces exact hard-forward rendered color/alpha/depth equality.
  As an artifact-level corroboration, every hard-forward arm's matched and common-hard per-view
  metrics are bit-identical.
- The result uses final parameters only. Held-out metrics are not used for fitting, lifting,
  training, checkpoint choice, stopping, or arm selection.

One auditability limitation is non-decisional: candidate records store
`preparation_hashes_verified: true` rather than repeating the actual candidate-side preparation
hashes. The sealed comparison and fresh independent recreation support the invariant, but future
artifacts should serialize those three hashes directly.

## Checks executed

- Independent strict-JSON, finiteness, SHA-256, seal/loaded-source, retry-history, attempt-marker,
  raw per-view mean, cross-seed summary, frozen-decision, schedule, config, count, step-zero, and
  Phase-A raw-diagnostic recomputation scripts.
- Fresh recreation of all six scene/fitted/initialization preparations under four CPU threads and
  deterministic PyTorch; all hashes matched.
- `CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q
  tests/test_kernel_support_taper.py tests/test_kernel_support_taper_ablation.py` — `15 passed`.
- `git diff --check` — passed.

The iter2 seal itself records passing Ruff, format, full non-slow CPU tests, and docs-sync. This
audit did not rerun a one-shot candidate phase, CUDA tests, gsplat, density control, real data, or
performance measurements.
