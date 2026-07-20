# Leave-one-source-view-out supervision result

This is the post-run audit for
`20260715T062601Z_cpu_cross_view_supervision.json`. The experiment was frozen in
`20260715_cross_view_supervision_PREREG.md` before the official run.

## Validity

- Command: `CUDA_VISIBLE_DEVICES='' .venv/bin/python
  benchmarks/cross_view_supervision_ablation.py --output
  benchmarks/results/20260715T062601Z_cpu_cross_view_supervision.json`
- Revision: `2dddca4aff59702341af9faceefa76ad2505dd83`, with dirty status, tracked-diff
  hash, every loaded source hash, environment, full command, and effective config embedded.
- Seeds 0/1/2 used the frozen Gradient/Hybrid × all/LOSO/matched-dropout design.
- Every family/seed passed bitwise step-0 equality, source-label/range equality, the two-step
  zero-learning-rate output/schedule check, full target-view coverage, and finite serialization.
- LOSO excluded exactly the target source group. The globally balanced matched control excluded
  the same per-target primitive count and scalar opacity sum, excluded no target-source primitive,
  and excluded every primitive exactly once across targets.
- Paired arms shared target schedules and final full-output counts: 1303/1293/1262 by seed. Final
  held-out and source diagnostics rendered or inspected the complete unmerged output.
- All embedded source hashes still match the official run files. This post-run audit is not part of
  the embedded hash set.

The artifact is valid for the preregistered decision. The matched control equalizes submitted
count, scalar opacity, and per-primitive exclusion exposure, but not coherent group topology,
visibility, projected alpha, color, or spatial coverage. That limits a positive attribution claim;
it does not weaken the observed all-versus-LOSO utility failure.

## Primary result

### GradientLifter

| Metric | All | LOSO | Matched non-self | LOSO vs all | Wins | Required | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Held-out depth RMSE | 0.154307 | 0.154077 | 0.154787 | 0.149% lower | 2/3 | >=2%, >=2/3 | no |
| All-source depth p90 | 0.211965 | 0.213057 | 0.207754 | 0.515% worse | 2/3 | >=10%, >=2/3 | no |
| Held-out PSNR | 19.6894 dB | 19.6849 dB | 19.7531 dB | -0.0045 dB | -- | >=-0.10 dB | yes |

Coverage remained 1.0 and alpha IoU changed by +0.00005. Material effect and matched-control
attribution are both false.

### Corrupted-depth HybridLifter

| Metric | All | LOSO | Matched non-self | LOSO vs all | Wins | Required | Pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Held-out depth RMSE | 0.150330 | 0.150344 | 0.151052 | 0.009% worse | 1/3 | >=2%, >=2/3 | no |
| All-source depth p90 | 0.163228 | 0.167002 | 0.170615 | 2.312% worse | 0/3 | >=10%, >=2/3 | no |
| Corrupted-source p90 | 0.205832 | 0.208639 | 0.209914 | 1.364% worse | 1/3 | >=15%, >=2/3 | no |
| Held-out PSNR | 19.6725 dB | 19.6875 dB | 19.6877 dB | +0.0150 dB | -- | >=-0.10 dB | yes |

Coverage remained 1.0 and alpha IoU changed by +0.00034. LOSO beat matched dropout on the three
geometry metrics in 2/3 seeds, but that cannot rescue failure against the inclusive utility arm.
Material effect and matched-control attribution are both false.

## Secondary mechanism signals

LOSO optimized the common source-excluded training objective slightly: cross-only train L1 fell by
0.21% for Gradient and 0.57% for Hybrid. Nearest-GT median distance improved by 1.72% and 1.31%,
respectively. These typical-case signals did not propagate to held-out expected-depth or source-tail
accuracy. Gradient source-depth median improved 1.48%, whereas Hybrid source-depth median worsened
9.61%. LOSO reduced lift time by 12.29% for Gradient and 6.07% for Hybrid because each training
render submitted fewer primitives; matched dropout showed the same computational mechanism.

## Decision

The experiment changed the optimized cross-view objective but did not reveal a material or robust
depth signal. Under the frozen synthetic protocols, own-view reconstruction is not the limiting
bottleneck that explains the geometry tail. Apply the preregistered stopping rule:

- keep inclusive `all` supervision as the default;
- retain LOSO and matched dropout only as reproducible research controls;
- stop LOSO/dropout/schedule sweeps on this setup; and
- pivot to one direct robust world-frame position-consistency term between fixed train-view
  matches while keeping depths ray-bounded.

The next experiment should test position consistency alone. Shape/appearance agreement and local
plane/normal constraints remain later branches. This result is scoped to synthetic pure Gradient
and deterministic corrupted-metric Hybrid; it does not establish clean-prior or real-data harm.

