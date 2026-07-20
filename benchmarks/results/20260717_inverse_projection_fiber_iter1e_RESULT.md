# Inverse-projection fiber fitting, Iteration 1e — result

## Outcome

**Scientific status: FAIL.** The once-only Iteration 1e transaction committed normally; this is
a negative mechanism result rather than a harness failure. Exact inverse projection preserved
each hypothesis's source 2D Gaussian, but independent row-wise hard-min fitting did not recover
the hidden cross-view tracks or their 3D geometry.

The official command was:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m benchmarks.inverse_projection_fiber_iter1e \
  --out benchmarks/results/20260717_inverse_projection_fiber_iter1e_RESULT.json \
  --artifacts-dir runs/inverse_projection_fiber_iter1e_official_20260717
```

The process used the result-bound clean environment, CPU float64 PyTorch, one thread per numeric
runtime, scene roots `17687011,17687012,17687013`, initial-depth roots
`17687111,17687112,17687113`, 400 Adam updates, and four fitting plus two held-out cameras. The
machine-readable result SHA-256 is
`2601a45d19d1d8a636d3c0db5ef8b14adf5f4137baaf718c86e1f80a84cecf9e`.

## Primary results

| Root | Source center max (px) | Source covariance rel. max | Train association | Held-out association | Correct tracks | 3D center p90 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17687011 | `7.11e-15` | `7.85e-16` | `0.6250` | `0.6094` | `0.5938` | `0.6388` |
| 17687012 | `3.55e-15` | `7.24e-16` | `0.8333` | `0.8438` | `0.8125` | `0.2941` |
| 17687013 | `3.55e-15` | `4.96e-16` | `0.5729` | `0.5938` | `0.5000` | `0.5575` |

The oracle-correspondence arm used identical initial geometry and reached `1.0` train,
held-out, and track accuracy in every root, with center p90
`3.90e-8,1.15e-8,2.42e-8`. Thus the parameterization and optimizer can recover the construction
when the assignments are correct. The shuffled control's mean center p90 was `0.4991693`, versus
`0.4967925` for fiber-conic; the preregistered relative improvement was only `0.0047616`, not the
required `0.50`. Its held-out accuracy advantage was `0.2239583`, also below `0.50`.

All source-invariant, rank, SPD, depth, finite, initialization, and 378 checkpoint observations
passed. Gates 2, 3, 4, and 6 failed. Gate 5 is `UNINTERPRETABLE`: the free control's source
center drift was `0.1563`–`0.4445` px and covariance drift `0.0311`–`0.0402`, so weight 25 did
not isolate free versus exact-source geometry.

## Diagnosis and next hypothesis

The optimizer converged to stable wrong assignments rather than failing numerically. The hard-min
objective gives each hypothesis an independent target choice, with no target capacity and no
shared cross-view track. Post-outcome diagnosis found both many-to-one target occupancy and
view-inconsistent tracks.

An outcome-informed feasibility check—not confirmatory evidence—found that a mean non-source
residual `<0.1` selected the correct complete tracks with 100% precision and recall in these three
used roots. Spatially grouping those survivors within `0.01` world units produced eight clusters
in each root. These thresholds are frozen only as the hypothesis for fresh Iteration 2 roots:
residual-gated pruning, source-preserving duplicate contraction, balanced rematching of every
original 2D observation, then fixed-track refitting.

## Scope and artifacts

This iteration used noiseless synthetic component geometry only. It did not test RGB, component
weight, opacity, SH, visibility, occlusion, unknown counts, split/merge/prune actions, real data,
GPU behavior, speed, or memory. The exact executed 50-file source closure was preserved before
post-run repairs at
`benchmarks/results/20260717_inverse_projection_fiber_iter1e_EXECUTED_SOURCES.tar`, SHA-256
`cc23e3ab9e95307453e97193d71f84040a832b16b08fb4e9d231f661ecb1f5a5`; every archived file
matches the immutable result manifest.

The qualitative viewer smoke test returned HTTP 200 with:

```bash
.venv/bin/rtgs view \
  --gaussians runs/inverse_projection_fiber_iter1e_official_20260717/scene_17687011/fiber_conic/gaussians.ply \
  --initial runs/inverse_projection_fiber_iter1e_official_20260717/scene_17687011/fiber_conic/gaussians_init.ply \
  --device cpu --rasterizer torch --host 127.0.0.1 --port 8891 --no-open
```

The viewer was post-result and non-decision-bearing. See the independent audit in
`benchmarks/results/20260717_inverse_projection_fiber_iter1e_AUDIT.md`.
