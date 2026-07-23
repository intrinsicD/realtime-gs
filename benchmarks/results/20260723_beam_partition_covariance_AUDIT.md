# Scientist pass — masked native-anchor Beam partition covariance

Date: 2026-07-23

Machine-readable audit:
[`20260723_beam_partition_covariance_AUDIT.json`](20260723_beam_partition_covariance_AUDIT.json)

Disposition: **70/70 checks passed; coverage claim retired; convergence claim narrowed**

## Claim inventory and disposition

| # | Claim | Kind and scope | Evidence | Disposition |
|---|---|---|---|---|
| 1 | Beam survivors identify existing source 2D Gaussians without 3D projection matching | Proven implementation invariant; CPU | Bound source, replayed CSR/depth digest, focused tests | **Confirm** |
| 2 | All original 2D Gaussian density inside the masks is partitioned among those anchors | Measured approximation; eight fitted Janelle views | Per-view mass receipts and replay | **Confirm only at frozen order-5 quadrature resolution** |
| 3 | Partition covariances increase initial visible coverage | Preregistered fitted-view development claim | Initial alpha-inside/IoU gates | **Retire** |
| 4 | Partition covariances improve fixed-topology convergence | Measured single-scene fitted-view CPU claim | Raw trajectories, exact repeat | **Confirm narrowly** |
| 5 | Full partition shape is better than determinant-only scaling | Causal arm comparison within the same screen | `pou-full` versus `pou-area` | **Narrow to +1.87% AUC; overall gate failed** |
| 6 | Partition moments recover physical 3D covariance | Asserted interpretation | No 3D covariance ground truth; conflicting target residuals | **Not established** |
| 7 | The implementation should replace CI/defaults | Production capability/default claim | Single all-fitted-view CPU scene | **Not authorized** |

## Source, chronology, and input binding

- Frozen base revision:
  `c2a7e120a5cafdcf22d4bff6f5b9868b860eb1df`.
- The protocol SHA-256 is
  `550abb9b931fb60644c0851c5ac488de969e1bbe1f0d6ed598cae353d8290562`.
- All six implementation/test/helper hashes listed in the preregistration still match.
- Filesystem chronology places the frozen protocol before every measured dynamics artifact.
- Compact manifest SHA-256 matches
  `b1c8e256d73e2c05f3cb4797a615bdbb2639a637f12908a5c96a2a9a9f912847`;
  every listed compact-view payload hash and byte count was checked.
- Both official runs used eight fitted views, downscale 32, seed 0, 800 outputs, and 1,000
  fixed-topology CPU steps. There is no held-out test set.

The audit independently replayed Beam Fusion and the partition construction from bound source and
data. It reproduced all three initial PLY files byte-for-byte. The replayed lineage contains 800
components, 6,029 links, and 4,704 unique `(view, source_component)` anchors; its CSR plus exact
depth digest is
`2fd11f5482a28a85bb520f14b475c139e2a3b88a6847bf404ee50e5e3d71a4de`.
The CI initial PLY is also byte-identical to the preceding Beam covariance experiment's CI control.

## Invariants and validity gates

All frozen mechanism gates pass:

- every view has anchors and positive masked partition mass;
- maximum partition-of-unity relative mass error is `2.70e-16` (gate `1e-12`);
- native-covariance/depth round-trip maximum relative error is `1.23e-6`
  (gate `1e-4`);
- all saved initial/final models contain 800 finite SPD Gaussians;
- means, opacity, and SH/color are bit-identical across initial arms;
- primary and repeat checkpoint grids are complete;
- timing-free scientific dynamics and all six initial/final PLYs repeat exactly.

The official run directories do not persist raw per-anchor tensor tables. This is a provenance
limitation, not a failed check: the audit regenerated the tables and their aggregate diagnostics
exactly, while both complete runs independently regenerated identical initial artifacts. Future
experiments should save an NPZ receipt for anchors, masses, covariances, and CSR depths directly.

## Independent metric and gate recomputation

| Arm | Init alpha-inside change | Init alpha-IoU change | PSNR-AUC change | Final FG-PSNR change | Coverage gate | Optimization gate | Both |
|---|---:|---:|---:|---:|---|---|---|
| `pou-area` | −12.17% | −41.80% | **+4.86%** | +0.0899 dB | fail | pass | **fail** |
| `pou-full` | +8.77% | −17.46% | **+6.82%** | **+0.1180 dB** | fail | pass | **fail** |

The coverage failure cannot be rescued by the secondary trajectory metrics. `pou-full` does
improve initial foreground PSNR by 0.2442 dB and reaches the CI final foreground PSNR at step 950
instead of 1000, but the preregistered question required materially higher visible coverage.

Both subordinate `pou-full`-versus-`pou-area` shape clauses pass: full shape improves AUC by
1.872%, has higher initial IoU/PSNR at acceptable extra outside alpha, and preserves final IoU.
The overall shape decision remains false because it explicitly required `pou-full` to pass both
coverage and optimization first.

Covariance diagnostics were recomputed from the saved PLYs and replayed contributor targets:

- CI median whitened residual: 0.6888 against native contributors.
- `pou-area`: 0.7839 against its own determinant target and 0.6555 against native contributors.
- `pou-full`: 0.5523 against its own partition target but 1.0778 against native contributors.

This establishes internal consistency with the newly defined partition targets, not physical 3D
covariance. No ground-truth covariance or held-out projection evidence exists.

## Referee findings

1. **The new anchor interpretation is correct.** Anchor discovery uses exact CSR component ids;
   the preserved depth is used only to lift the already identified 2D covariance.
2. **“All density” needs a qualifier.** The method integrates the Gaussian mixture with 25
   deterministic quadrature samples per component and a nearest-pixel mask lookup. It is not an
   exact continuous masked integral or full pixel rasterization.
3. **The treatment is not simple upscaling.** Median determinant-matching scalar covariance
   multipliers are below one in every view, while the maximum reaches 21,290×. Existing 3D Beam
   sigma bounds prevent unbounded output, but any future blend or production test must explicitly
   control these tails.
4. **The intended coverage mechanism is falsified.** Initial alpha IoU falls in both treatments.
   The positive result is optimization conditioning on fitted views.
5. **No generalization or production claim follows.** All evaluated cameras contribute to
   initialization and fitting; topology is fixed; CUDA/gsplat split/merge behavior is untested.
6. **CPU elapsed time is non-decisional.** No speed, real-time, or GPU-memory claim is supported.

## Commands checked

Primary:

```bash
.venv/bin/python benchmarks/beam_partition_covariance.py \
  --protocol benchmarks/results/20260723_beam_partition_covariance_PREREG.md \
  --out runs/beam_partition_covariance_20260723
```

Exact repeat:

```bash
.venv/bin/python benchmarks/beam_partition_covariance.py \
  --protocol benchmarks/results/20260723_beam_partition_covariance_PREREG.md \
  --out runs/beam_partition_covariance_20260723_repeat
```

Independent audit:

```bash
.venv/bin/python benchmarks/audit_beam_partition_covariance.py
```

Viewer smoke:

```bash
CUDA_VISIBLE_DEVICES='' .venv-cuda/bin/rtgs view \
  --comparison-manifest benchmarks/results/20260723_beam_partition_covariance_VIEWER.json \
  --max-viewer-gaussians 800 --device cpu --port 8783 --no-open
```

The viewer returned HTTP 200, was bound by PID and artifact hashes, and was stopped afterward.
An unrelated process was using the GPU, but the viewer had `CUDA_VISIBLE_DEVICES=''`; no
performance conclusion is drawn.

## Final decision

Keep `rtgs.lift.beam_partition` as an opt-in research mechanism and preserve CI as the default.
The initial-coverage hypothesis is closed negatively for this consumed all-fitted-view frame. A
new promotion attempt needs multiple scenes/seeds, untouched held-out cameras, saved raw partition
receipts, and a production gsplat topology arm. It should compare unchanged CI directly with
`pou-full`; any tail clamp, soft/geodesic responsibility, covariance blend, or split/merge schedule
is a new treatment and must be frozen before outcomes.
