# Independent audit: N78 gsplat visualization replay

Reviewer: Codex independent replay referee (`/root/n78_gsplat_replay_audit`)  
Reviewed at: `2026-07-16T11:44:48Z`  
Verdict: **PASS, visualization scope only; host-install-bound provenance limitation**

The two figures and replay manifest are accurate post-hoc visualizations of the official N78
terminal 2D fits through the installed GaussianImage gsplat CUDA sum kernel. A fresh replay and
an independent raw-to-CUDA reconstruction reproduced every published frame metric exactly. This
does not extend the official CPU result to the repository's 3D `GsplatRasterizer`, lifting,
held-out views, real images, timing, memory, or a default change.

## Evidence and independent replay

| Item | Audited SHA-256 or value | Finding |
|---|---|---|
| Official raw NPZ | `028c93f350b30b61debebd5bf0706ff128f2c54faaee04614d1ee12191a3aeb7` | exact; matches the official result and prior audit |
| Visualization script | `33464bd9e14d8f500a3e5295a9141fc0697fe85ad1b4b2189026e44d5e41f110` | exact |
| Replay manifest | `145df135bc8e5ff28552621ee242ad5c2218bc6529cb195875a650c349046f11` | strict JSON; all reported values finite |
| Replay note | `1c71e5979994ecdfe9bb0a6ea65a21c601082b3f2f6f5c13973619e01e16a754` | scope and numbers agree with the manifest |
| Overview PNG | `1107efe22441a8289e5ca455aab5a1ea6955383190d37ad10619a12ea9179029` | fresh replay byte-identical; 1290x1614 RGB |
| All-view PNG | `712c8bee320e334d615ed8ce8767ed6e4cdb69e45b2384e8b01a588014e507b1` | fresh replay byte-identical; 1627x3742 RGB |

The fresh `/tmp` manifest was exactly equal to the official replay manifest after removing only
the expected dynamic creation time, output paths, and command arguments. Directly reopening the
raw NPZ established 108 unique terminal fit identities and 54 complete current/candidate pairs;
paired targets were bit-exact. An independent loop invoked the two gsplat APIs for all 108 fits
and recomputed every frame mapping, target PSNR, native-parity PSNR, maximum error, and mean
error. All non-floating fields were exact and the maximum numeric difference from the published
manifest was `0.0`.

| Replay/native comparison | PSNR min / median / mean / max | Global mean absolute error | Maximum absolute error |
|---|---:|---:|---:|
| Display-clamped | 66.3690856 / 68.0236629 / 68.0134501 / 69.6382574 dB | 0.0001981976 | 0.0044225454 |
| Unclamped | 66.3479911 / 68.0178102 / 67.9955783 / 69.6382574 dB | 0.0001989462 | 0.0044225454 |

All six independently recomputed seed summaries were exact. Candidate-minus-current mean gsplat
PSNR was `-1.684630`, `-1.936527`, and `-1.762772` dB in the appearance-only block and
`-1.346322`, `-1.726999`, and `-1.422911` dB in the joint block. The current arm was higher for
all six seed means and all 54 individual source-image pairs. This confirms only that the visual
replay preserves the official result's direction.

The overview was inspected visually and by exact pixel extraction. Its 30 image panels are the
target, both replayed arms, and both x4 absolute-error images for fixed `local_view == 0` across
all six seeds. The exhaustive sheet's 162 panels are exactly the target/current/candidate arrays
for all 6 seeds x 9 selected source views. Nearest-neighbor enlargement and border overwrites
were accounted for in the pixel check. There is no hidden qualitative subset in the exhaustive
figure.

## Renderer and representation audit

The executed renderer was `gsplat` 1.1.3 from
`/home/alex/Documents/GaussianImage_plus/gsplat/gsplat/__init__.py`, Torch `2.9.0+cu128`, CUDA
12.8, on an NVIDIA GeForce RTX 3050. The adapter calls
`project_gaussians_2d_covariance` and `rasterize_gaussians_plus`; it does not call the
repository's 3D backend. Inspection of the loaded Python and CUDA sources confirmed:

- the saved pixel-center coordinates are shifted by -0.5 because the native renderer samples
  at `(j+0.5, i+0.5)` while this CUDA kernel samples `(j, i)`;
- `[l00^2, l00*l10, l10^2+l11^2]` is the correct covariance conversion;
- saved `built_amplitude` with unit opacity gives additive RGB Gaussian contributions;
- projection uses `clip_coe=sqrt(12)`; and
- the CUDA kernel drops contributions below `1/255`, whereas the native renderer retains all
  contributions with Mahalanobis `q < 12`.

The measured nonzero parity error is therefore expected and is reported rather than hidden. The
high agreement validates these images as a faithful visualization on this installation; it is
not proof of exact backend parity. Lifting the fits merely to call the 3D alpha-compositing API
would add unmeasured depth, opacity, ordering, and compositing choices, so the note correctly
refuses to present such a conversion as the original fit.

## Claim disposition

| # | Claim | Kind and scope | Disposition |
|---:|---|---|---|
| 1 | All 108 saved terminal fits were replayed through a gsplat CUDA renderer. | measured, installed GaussianImage 2D sum fork | **confirm** |
| 2 | The published native-agreement metrics and six seed summaries are accurate. | measured, post-hoc GPU replay of CPU-synthetic source fits | **confirm**; every value independently reproduced |
| 3 | The visual replay preserves the official result's current-over-candidate direction. | measured, six seed means / 54 source-image pairs | **confirm narrowly**; it is not new optimization evidence |
| 4 | The PNGs show the declared fixed overview subset and exhaustive population. | measured figure content | **confirm** |
| 5 | This is the repository's 3D gsplat backend or evidence about 3D/held-out/real-time behavior. | capability implication | **retire**; the artifacts explicitly deny it |
| 6 | The adapter is the closest semantics-preserving gsplat option available here. | implementation judgment | **narrow** to the inspected installed APIs and representation; not a universal gsplat claim |

## Limitation and required promotion evidence

The manifest records the external gsplat package version, module path, Torch/CUDA versions,
device, and `LD_PRELOAD`, but it does not hash the external GaussianImage source files or loaded
CUDA extension binary. The figures reproduced byte-for-byte on the same host, but durable
cross-machine replay remains **host-install-bound**. A promoted portable artifact should record
the extension path and SHA-256 plus a source-tree or immutable package/revision hash (and ideally
the Pillow/font inputs that affect PNG bytes). Cross-GPU determinism was not tested.

The source data are 48x48 synthetic source-view fits with 150 Gaussians at terminal step 120.
Only the exhaustive sheet covers all terminal images; the compact overview intentionally uses
one fixed local view per seed. No fitting trajectory, novel view, lift, refinement, performance,
or memory measurement is shown.

## Commands and checks executed

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv/bin/python benchmarks/visualize_stage1_fit_parameterization.py \
  --output-prefix /tmp/n78_gsplat_replay_audit

.venv/bin/python -m ruff check benchmarks/visualize_stage1_fit_parameterization.py
.venv/bin/python -m ruff format --check benchmarks/visualize_stage1_fit_parameterization.py
.venv/bin/python -m py_compile benchmarks/visualize_stage1_fit_parameterization.py
git diff --check
```

The audit additionally ran independent inline Python checks that reopened the NPZ with
`allow_pickle=False`, rebuilt all identity pairs, rerendered all 108 frames through the loaded
CUDA APIs, recomputed all frame and aggregate metrics, verified the six summaries, normalized
the fresh manifest, and extracted every image tile from both PNGs. The original result note and
independent N78 scientific audit were reopened; the once-only N78 fit was not rerun and no
official artifact was overwritten. Focused Ruff, format, byte-code compilation, and
`git diff --check` all passed.
