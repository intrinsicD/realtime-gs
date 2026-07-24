# Scientist pass — new opt-in variants on Janelle `frame_00008`

Date: 2026-07-24

Machine-readable audit:
[`20260724_new_variants_frame00008_AUDIT.json`](20260724_new_variants_frame00008_AUDIT.json),
SHA-256 `8d7046634f8c12ea38e792b80a8d1a39a81c3e778f7975e9293996e7a28e93f8`.

Disposition: **15/15 checks passed; pool claim confirmed narrowly; containment claim split;
structure/checkpoint claims retired**

## Claim inventory and disposition

| # | Claim | Kind and scope | Evidence | Disposition |
|---|---|---|---|---|
| 1 | Pool/free-list recycling works at fixed live count | Proven/measured mechanism; seven fitted real views | Saved fits/history, source binding, replay | **Confirm: capacity 1,280; 640 live/output in every view** |
| 2 | Pool improves stage-1 fit quality | Preregistered single-scene development claim | Seven paired camera metrics | **Confirm narrowly: +1.2291 dB, 7/7 wins, spill reduced** |
| 3 | Pool improves downstream held-out quality | Preregistered single-scene/one-camera development claim | Saved NPZ, exact gsplat replay, frozen gate | **Confirm narrowly: +0.2690 dB, alpha guardrail passes** |
| 4 | Containment weight 5.0 is a useful stage-1 rule | Preregistered local treatment claim | Coverage and appearance gates | **Retire: −10.5124 dB and inside-coverage gate fails** |
| 5 | Containment can improve downstream reconstruction | Measured single-scene/one-camera development claim | Common carve/refinement, exact replay | **Confirm only as a surprising +0.5578 dB observation requiring replication** |
| 6 | Structure-tensor initialization is useful at stage 1 | Preregistered treatment claim | Seven paired fits and spill guardrail | **Retire: +0.8957 dB but +38.71% outside coverage** |
| 7 | Structure initialization materially improves the endpoint | Preregistered downstream claim | Held-out endpoint | **Retire: +0.0276 dB is below +0.10 dB floor** |
| 8 | Train-only best-checkpoint selection improves the result | Paired model-selection claim | Shared history and tensor equality | **Retire: selected final step 2,000 exactly** |
| 9 | Any option should become a default | Production/generalization claim | One scene, seed, and held-out camera | **Not authorized** |
| 10 | Timings or allocation are performance evidence | Performance claim | Contended, fixed-order, unrepeated run | **Not authorized** |

## Chronology, source, and input binding

The v1 and v2 failures are preserved append-only with their original plans and hashes. V1 failed
on a reporting-key lookup before saving a numeric outcome. V2 completed stage 1, then failed
before optimizer step 1 on gsplat's packed RGB+D random-background shape assertion. V3 froze the
unpacked execution layout before downstream outcomes and reran all stage-1 arms from scratch.

The v3 protocol predates `plan.json`; its SHA-256 is
`649f93fca12c71437c7a44fd742935d1e7d446ade9ea7950913444ef8395816d`.
The plan and summary bind that protocol, revision
`7772f4fb63bf5b7c6540fbce7dfa3bf578bd7c11`, the effective dataclasses, the relevant source
files, calibration, every RGB/mask file, and every loaded RGB/mask tensor. Current hashes still
match the executed source.

All **124** summary-bound artifacts exist with their exact byte counts and SHA-256 hashes.

## Split isolation and execution invariants

- Frozen order and split replay exactly:
  seven training cameras followed by held-out `C1004`.
- No `C1004` stage-1 fit exists.
- All four histories contain 2,000 sampled views, each in local indices `[0,6]`.
- Checkpoint selection records exactly `[0,1,2,3,4,5,6]`; test index 7 is absent.
- Every stage-1 fit contains 640 finite, valid Gaussians.
- All saved 3D NPZ/PLYs are finite with their reported counts.
- Baseline and `best-train-checkpoint` share the same history and bit-exact initial/final tensors.
- Density changes occurred at steps 200–900; every arm then received 1,100 recovery steps. No
  final-step surgery contaminates evaluation.

## Independent metric and gate replay

All 28 saved stage-1 fits were re-rendered from disk. Maximum reported-versus-replayed metric
error was `3.81e-6`. The independent Torch-reference versus native-CUDA stage-1 render maximum
absolute pixel difference was `2.98e-7` (mean absolute difference averaged across renders
`1.29e-9`).

All initial/final 3D metrics were recomputed from exact NPZs through the frozen unpacked,
antialiased gsplat path. Maximum metric error was **0.0**.

| Arm | Stage-1 FG Δ | Stage-1 wins | Outside change | Stage-1 gate | Held-out final FG Δ | Held-out α-IoU Δ | Train FG Δ | Downstream gate |
|---|---:|---:|---:|---|---:|---:|---:|---|
| `pool` | **+1.2291 dB** | **7/7** | −13.34% | **pass** | **+0.2690 dB** | −0.00545 | −0.0945 dB | **pass** |
| `mask-containment` | −10.5124 dB | — | **−90.52%** | **fail** | **+0.5578 dB** | −0.00434 | −0.0079 dB | **pass** |
| `structure-tensor` | +0.8957 dB | 7/7 | **+38.71%** | **fail** | +0.0276 dB | −0.00861 | −0.1197 dB | **fail** |
| `best-train-checkpoint` | — | — | — | — | +0.0000 dB | +0.00000 | +0.0000 dB | **neutral/fail** |

The downstream containment result does not rescue its failed local claim or select weight 5.0.
It is a different observation about the complete carve+density interaction. It must be replicated
and decomposed before a causal mechanism is assigned.

## Visual and viewer audit

All 23 bound visual artifacts decode: three cross-arm PNGs plus, for each of five report arms, a
calibrated contact sheet, calibrated-path GIF, novel-orbit GIF, and novel-elevation GIF. The
synchronized CPU viewer loaded ten bound PLYs, returned HTTP 200, had no NVIDIA compute process,
and shut down with its PID stopped and port closed.

Visual inspection supports only the qualitative statements in the result note: pool and structure
are sharper at stage 1, containment is dimmer/sparser, and refined endpoint differences are
subtle. Visuals do not override the gates.

## Commands checked

Official v3 run:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv-cuda/bin/python benchmarks/new_variants_frame00008.py \
  --protocol benchmarks/results/20260724_new_variants_frame00008_PREREG_V3.md \
  --out runs/new_variants_frame00008_20260724_v3
```

Independent replay/audit:

```bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  .venv-cuda/bin/python benchmarks/audit_new_variants_frame00008.py
```

Viewer smoke:

```bash
CUDA_VISIBLE_DEVICES='' .venv-cuda/bin/rtgs view \
  --comparison-manifest benchmarks/results/20260724_new_variants_frame00008_VIEWER.json \
  --scene /home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008 \
  --downscale 16 --device cpu --max-viewer-gaussians 20000 \
  --host 127.0.0.1 --port 8784 --no-open
```

## Final decision

Keep all defaults unchanged.

Pool is the only balanced positive treatment in this development run. Replicate it across seeds,
scenes, and multiple held-out cameras before any promotion. Treat containment as a separate
follow-up requiring a lower-weight sweep and mechanism decomposition. Do not combine pool and
containment until both individual effects replicate. Structure initialization and checkpoint
selection do not advance from this run.

Still unverified: multi-seed variance, multi-scene transfer, more than one held-out camera,
packed/unpacked parity for this exact workload, isolated performance, and whether the containment
endpoint gain comes from geometry, capacity, topology, or chance.
