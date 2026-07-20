# Surface plane/normal target gate

## Protocol and artifact

- Frozen protocol: `benchmarks/results/20260715_surface_plane_normal_PREREG.md`
- Sole official artifact: `benchmarks/results/20260715T110342Z_cpu_surface_plane_normal.json`
- Artifact SHA-256: `07694e7388971936f918dd6c6d187df781e03f4bc44430f3099ea165728a9fd2`
- Result audit: `benchmarks/results/20260715_surface_plane_normal_RESULT.md`
- Scope: Hybrid only; seeds 0/1/2; corrupted metric training depth; four cross-view neighbors;
  post-freeze clean target audit before five planned optimization arms.

## Structural target evidence

| seed | retained | targets | coverage | corrupt targets | min/source | radius p90 | incidence p10 | mid-eigen ratio min | scale gap p10 | shuffle median |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1,303 | 339 | 26.02% | 98 | 26 | 0.07017 | 0.18511 | 0.01332 | 6.5368 | 0.4601 |
| 1 | 1,293 | 318 | 24.59% | 79 | 24 | 0.06383 | 0.18378 | 0.01488 | 6.5357 | 0.4984 |
| 2 | 1,262 | 326 | 25.83% | 67 | 31 | 0.06648 | 0.19264 | 0.01408 | 6.5223 | 0.4583 |

Every structural floor passed in every seed, including all nine query/candidate/support views,
at least 300 targets, at least 23% coverage, at least 60 corrupted targets, and at least 20 targets
per query source.

## Post-freeze clean audit

Required: plane residual p90 at most 0.10 of extent and median absolute clean-normal cosine at
least 0.50 in both all-target and corrupted-target strata.

| seed | all plane p90 | all cosine median | corrupt plane p90 | corrupt cosine median | pass |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 0 | 0.17451 | 0.58694 | 0.24711 | 0.64439 | no |
| 1 | 0.16042 | 0.48165 | 0.23910 | 0.37262 | no |
| 2 | 0.16475 | 0.52651 | 0.27077 | 0.49066 | no |

All 983 targets were labelable. The plane criterion failed both strata in every seed; normal
agreement was mixed. The gate therefore stopped before initialization invariants and all five
90-step arms. Every run has empty `invariants` and `arms`, and the aggregate `summary` is empty.

## Bound conclusion

Reject the four-neighbor corrupted-depth PCA target constructor without tuning. This evidence does
not evaluate the implemented point-to-plane or shortest-axis losses with valid oriented points and
does not reproduce or refute Incremental Gaussian Triangulation. The generic zero-default API is
retained; the next admissible evidence requires an independent calibrated metric-depth/RGB-D
oriented-point backend and a new pre-optimization validity audit.
