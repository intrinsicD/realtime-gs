# Full `frame_00008` beam-fusion experiment — result

## Outcome

Bounded beam fusion is feasible on the full compact bundle and its downstream fit reached the
frozen compact-target plateau rule, but it did **not** improve the preregistered initialization
metric. It initialized exactly 5,000 3D Gaussians and scored 11.5826 dB mean foreground PSNR,
0.2803 dB below the count-matched top-K control. Ordinary 3DGS densification then grew the model
to 44,222 Gaussians; after 70,000 executed steps, the frozen selector chose step 69,000 at
37.8874 dB and the independent joint stopping rule reported `plateau`.

This is a real, single-scene, all-view development result. Every one of the 26 views was used for
initialization, fitting, checkpoint selection, and stopping. It supports no held-out,
novel-view, generalization, causal downstream, or default-change claim.

## Frozen setup

| Axis | Value |
| --- | --- |
| Input | `dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d` |
| Compact inputs | 26 views × 5,000 = 130,000 2D Gaussians |
| Requested/actual beam initialization | 5,000 / 5,000 3D Gaussians |
| Pair evaluation | all 325 view pairs; all 8,125,000,000 cross-view ray pairs |
| Beam gates | minimum 3 views; transverse 3.0σ; RGB distance ≤0.35; RGB σ=0.25; fold-in 3.0σ |
| Bounds/covariance | near 0.05; bounds scale 0.5; σ clamped to `[1e-4, 1.1180786]` world units |
| Bounded reduction | source chunk 256; voxel 0.02236157; 20,000 seed budget; fold chunk 512 |
| Initialization opacity | 0.10 |
| Fit | CUDA gsplat 1.5.3, packed+antialiased, black background, SH degree 3, seed 0 |
| Density | DefaultStrategy, steps 500–15,000 every 100, grad 8e-4, prune 0.005/0.1, cap 100k |
| Schedule | 30k parent, then fixed-topology 10k segments to the first joint plateau, max 70k |

The preregistration SHA-256 is
`1fa29697ccc729e4caab4a0dff4e8528d244cced13c32da53d73f24aa7c7a126`. Addendum 1,
frozen after the uninterrupted 30k parent but before continuation output, only authorized a
fail-closed clean-parent preflight; its SHA-256 is
`19d4f643f02beb47fd7abc7cdd8171ff01d52e67444949ae4763f7fa52bf237b`.

## Initialization results

| Initializer | 3D count | Placement | FG PSNR | Crop PSNR | Crop SSIM | Alpha IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-K control | 5,000 | 91.6647 s | 11.8629 | 16.8305 | 0.77155 | 0.26740 |
| Beam fusion | 5,000 | 138.3255 s | 11.5826 | 16.5502 | 0.72967 | 0.00199 |
| Beam − top-K | 0 | +46.6608 s | **−0.2803** | −0.2803 | −0.04188 | −0.26541 |

Beam fusion admitted 345,109,938 gated pair seeds into 743,844 occupied seed voxels, retained
20,000 seeds for fold-in, and reduced 19,583 eligible components to 5,000. The final components
each had 18–26 contributing views (21.6722 on average), but only 44,928 of the 130,000 distinct
input splats occurred in a final contributor signature. Thus “all inputs were consumed” means all
ray pairs were evaluated, not that every input splat survived the bounded output reduction.

The top-K placement time is below both the linked task's 300-second gate and 180-second target on
this workstation. That single diagnostic does not close the CSR task: this run did not carry the
task's frozen slow-reference lineage, exact discrete-parity audit, repeated microbenchmark,
retained-CSR payload, or peak-RSS breakdown.

## Fit and convergence

| Stage | Global steps | Topology | Optimizer elapsed | Decision |
| --- | ---: | --- | ---: | --- |
| Parent | 1–30,000 | density active through 15k | 963.863 s | 44,222 Gaussians |
| Polish | 30,001–40,000 | fixed | 301.841 s | still improving; select 40k |
| Tail | 40,001–50,000 | fixed | 296.196 s | still improving; select 50k |
| Cooldown | 50,001–60,000 | fixed, 0.25 LR factor | 287.955 s | still improving; select 60k |
| Settle | 60,001–70,000 | fixed, 0.25 LR factor | 313.793 s | plateau; select 69k |

Total native optimizer time was 2,163.649 seconds (36.06 minutes); peak recorded CUDA allocation
was 3.788 GiB. Each PLY boundary is explicitly non-exact because it does not retain Adam moments,
per-parameter step counters, or RNG state. All 26 compact targets replayed exactly at every
boundary, and topology stayed fixed at 44,222 after step 15,000.

At the selected 69k model, all-26 fitted compact-target metrics were 37.8874 dB foreground PSNR,
42.8550 dB crop PSNR, 0.995821 crop SSIM, and 0.976061 alpha IoU. The last-six Theil-Sen slope was
0.00334 dB/1k, only 11.54% of views improved by more than 1%, and all four frozen window
conditions passed. The five-transition rule also plateaued with ten consecutive non-material
transitions. “Converged” here means this training-target plateau only, not a global optimum.

The reporting-only source-RGB pass was opened after model selection. Its all-view fitted
foreground PSNR was 33.6586 dB. It is not held-out evidence and was not used to select or stop.

## Live viewer

The training viewer ran as a separate CPU process, polling 1,000-step PLY checkpoints and never
owning a CUDA allocation. Across all 70 checkpoint saves, training-side callback time totaled
0.8960 seconds, 0.0414% of native optimizer time. A watcher sample used 11.4% of one logical CPU
and 680,532 KiB RSS; therefore the correct answer is **low observed interference, not zero cost**.
There was no controlled viewer-on/off timing repeat, so no training-speed claim is made.

The selected 69k result is currently served at `http://127.0.0.1:8780`; an independent HTTP smoke
returned 200 and 2,888,259 bytes. For another run:

```bash
.venv-cuda/bin/rtgs view \
  --gaussians runs/beam_fusion_full_frame00008_fit_20260721/gaussians_init.ply \
  --watch-checkpoints runs/beam_fusion_full_frame00008_fit_20260721/checkpoints \
  --max-viewer-gaussians 50000 --device cpu \
  --host 127.0.0.1 --port 8780 --no-open
```

The watcher leaves the last loadable model visible and retries if it observes a PLY while the
writer is still producing it. Exact snapshots and source-image loading should remain disabled
during fitting if CUDA isolation is the goal.

One UI behavior was corrected after the formal fit: the executed watcher loaded each full
checkpoint and raised the slider maximum, but left the displayed count at the initial 5,000 unless
the user expanded it. The final code automatically follows a growing count when the user was
already showing the whole prior model. That post-run version passed a separate HTTP smoke against
the saved 70k checkpoint directory, but it did not receive a new live-training overhead trial.

## Commands and artifacts

The two initialization commands were:

```bash
/usr/bin/time -v .venv-cuda/bin/python benchmarks/full_compact_reconstruction.py \
  --phase fit --initializer topk --init-only \
  --out runs/beam_fusion_full_frame00008_topk_init_20260721 \
  --preregistration benchmarks/results/20260721_beam_fusion_full_frame00008_PREREG.md \
  --fit-mode all --max-tracks 5000 --depth-samples 32 --min-views 2 \
  --robust-view-fraction 0.60 --min-placement-score 0.01 --init-opacity 0.10 --seed 0

/usr/bin/time -v .venv-cuda/bin/python benchmarks/full_compact_reconstruction.py \
  --phase fit --initializer beam-fusion \
  --out runs/beam_fusion_full_frame00008_fit_20260721 \
  --preregistration benchmarks/results/20260721_beam_fusion_full_frame00008_PREREG.md \
  --fit-mode all --max-tracks 5000 --beam-min-views 3 \
  --beam-transverse-gate-sigma 3.0 --beam-max-color-distance 0.35 \
  --beam-color-sigma 0.25 --beam-fold-in-gate-sigma 3.0 --beam-source-chunk 256 \
  --beam-seed-budget-multiplier 4 --init-opacity 0.10 --iterations 30000 \
  --eval-every 1000 --density-strategy gsplat-default --densify-start 500 \
  --densify-stop 15000 --densify-every 100 --max-gaussians 100000 \
  --prune-opacity 0.005 --prune-scale-frac 0.1 --seed 0
```

Continuation used `--phase polish`, `tail`, `cooldown`, and `settle` in order, with each prior
output passed as `--parent-out`; the exact commands are stored in the machine-readable result.

The selected model is
`runs/beam_fusion_full_frame00008_settle_60000_70000_20260721/gaussians_final.ply`, SHA-256
`733843ae79e4464bb5c43d2174a17d329ebaf31ee45bf03cec4d1ada70699c63`. The dirty parent run
preserved and rehashed 55 executed source files plus its binary working-tree patch; the source
manifest SHA-256 is `9a1129cd2cd237a7aacc4ec0757ed0a42145286879bbfad6065a6d43ce7c7ae9`.

See `20260721_beam_fusion_full_frame00008_RESULT.json` for raw-path bindings and the independent
scientist pass in `20260721_beam_fusion_full_frame00008_AUDIT.md`.
