# Independent scientist audit: kernel-support Phase A

## Disposition

**PASS for Phase-B execution clearance.** The frozen diffuse gate passes independently in all
three seeds and in the raw-mass pooled result. This clearance authorizes only the two already
sealed candidate arms, `c1_taper` and `hard_forward_c1_taper_gradient`. It is not evidence that
either arm improves reconstruction, and it does not authorize tuning the width, cutoff, loss,
schedule, seeds, or thresholds.

Bound evidence:

- Phase-A JSON SHA-256:
  `6380dc0b92043db608f6ba056c1cbaa2509e4eeba62a7b20e3f0fb7eacdde59c`
- implementation-seal JSON SHA-256:
  `f35827d362318d4eb55d637cdadb77c5a97deb68fd62e6f04e231e9c39702184`
- sealed source aggregate:
  `03e5fee097a32af425852416b2da1313b91dd0914de471c6c35f51ba416e466c`
- preregistration SHA-256:
  `c78a74ea67a4a0d327b8ef884006dc8ad5781da9a632f557c2e9f370a8868a58`

## Independently recomputed diffuse gate

All percentages and ratios below were recomputed from the 120 raw per-step diagnostic records in
each seed, using additive counts and `math.fsum` over raw masses rather than the stored seed
percentages.

| Seed | Eligible observations | Annulus upstream | Recoverable annulus | Recovered / active | Recovered / boundary | Gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 16,164,687 | 41.2645% | 25.5507% | 0.263861% | 5.72836% | PASS |
| 1 | 16,427,099 | 40.7169% | 24.8590% | 0.255989% | 5.45598% | PASS |
| 2 | 15,699,101 | 40.3301% | 23.5441% | 0.236565% | 5.12382% | PASS |
| pooled | 48,290,887 | 40.7745% | 24.6717% | 0.252269% | 5.43819% | PASS |

The frozen minima are 100,000 eligible observations per seed, 1% annulus upstream, 10%
recoverable annulus, 0.1% recovered/active, and 5% recovered/boundary, with every training view
sampled. All nine training views were sampled in every seed. The pooled annulus incidence was
24.8469% (11,998,786 of 48,290,887 eligible Gaussian/pixel observations). The prespecified
view-dependent replication also passed in all three seeds, but was not used to rescue the diffuse
decision.

## Validity and provenance checks

- The preregistration predates the implementation seal; the atomic Phase-A attempt marker predates
  the result. No Phase-B attempt marker existed during this review.
- All 68 sealed files still matched their recorded SHA-256 values. The recomputed source aggregate,
  seal hash, incorporated SH-protocol hash, and preregistration hash matched the artifact exactly.
- All 39 loaded repository Python/protocol paths were a hash-matching subset of the sealed paths;
  the loaded-source aggregate also matched.
- Strict JSON parsing found no non-standard numeric constants. The recorded execution used CPU,
  four Torch threads, deterministic algorithms, empty `CUDA_VISIBLE_DEVICES`, and the same
  Python/PyTorch/platform fingerprint as the seal.
- The six run identities were exactly the two frozen conditions crossed with seeds 0, 1, and 2.
  Scene, fitted-set, initialization, schedule, final-Gaussian, and held-out metric values matched
  the preceding hard SH-audit baselines; only the diagnostic collector changed. This independently
  confirms that collection did not alter the hard forward/training trajectory.
- Raw schedules contained 120 steps per run, used the frozen global train views, covered every
  train view, followed the frozen SH-degree schedule, retained a fixed primitive count, and used
  only checkpoints 30/60/90/120. Fitting, lifting, and training receive the training subset only;
  held-out views are evaluated after training and do not enter the Phase-A gate.
- Every raw q-bin count summed to its eligible count. Every per-step and per-seed ratio denominator
  was positive. All hard-support invariants held: no q below `-1e-6`, no nonzero hard kernel or q
  gradient at `q >= 12`, and no active-region chain-rule violation. Maximum active-region errors
  were below the frozen absolute/relative tolerances.
- Source inspection confirmed the diagnostic upstream is retained after opacity, alpha clamp,
  transmittance, and active color/alpha loss propagation. The sign convention is correct:
  `u < 0` combined with the taper's negative annulus derivative yields a descent update toward
  smaller q. Depth remains evaluation-only.

Focused verification executed:

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m pytest -q \
  tests/test_kernel_support_taper.py tests/test_kernel_support_taper_ablation.py
14 passed

git diff --check
pass
```

The harness's fail-closed `validate_phase_a_audit` and default-semantics checks also passed against
the official artifact and seal. The seal itself records a passing full non-slow CPU suite, Ruff
check/format check, and docs-sync check.

## Claim table and limitations

| Claim | Kind and scope | Evidence | Disposition |
|---|---|---|---|
| The adjacent `12 <= q < 16` annulus carries material loss-directed gradient under this protocol. | Measured; fixed-topology CPU synthetic hard-reference training. | Raw Phase-A histories and recomputation above. | Confirm, within scope. |
| The frozen Phase-A gate authorizes the two candidate arms. | Proven protocol decision. | Preregistration, exact audit, strict review manifest. | Confirm. |
| A C1 taper improves held-out quality. | Not yet measured. | No Phase-B result exists. | Withhold. |
| The mechanism transfers to real scenes, density control, gsplat/CUDA, speed, or production defaults. | Unsupported. | These paths were excluded by protocol. | Withhold. |

Two non-critical reporting limitations remain. The generated Phase-A result note labels the source
aggregate as “Implementation seal”; the actual seal-file hash is listed above, and both values are
unambiguous in JSON. Also, the environment fingerprint binds Python, PyTorch, platform, threads,
and determinism but is not a complete third-party package lock or CPU-model inventory. Neither
limitation affects this mechanism gate; no portability or performance claim is made.
