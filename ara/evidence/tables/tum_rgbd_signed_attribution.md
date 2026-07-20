# TUM RGB-D signed attribution evidence

## Frozen source and isolation

- Development: official TUM `rgbd_dataset_freiburg3_sitting_xyz`, SHA-256
  `05c071672cda22a668860a935124737a4eb4fa772cbad372e73d5a99ce4be205`.
- Reserved confirmation: `rgbd_dataset_freiburg3_walking_xyz`, SHA-256
  `1459e9488ac0e61a2ec80dfbc35cfb77942f6d8eabded1c8d26a70be650d0e1d`.
- Predecode implementation aggregate:
  `75b0dbb6a84548852dd6de4c5623557c78168ccc305d93ff1a4664f4f4347f3d`; 21 focused tests passed.
- Sitting decoded exactly 48 T plus eight V depth members once, zero RGB and zero H depth.
- The decision records `confirmation_authorized=false`; no walking attempt seal exists and no
  walking archive member or manifest was opened.

## Development metrics

| metric | sparse | dense T-only |
| --- | ---: | ---: |
| Depth-valid pairs | 176,950 | 155,416 |
| Supported targets | 27,525 | 27,135 |
| Target-balanced positive rate | 0.116744 | 0.102500 |
| Target-balanced negative rate | 0.063469 | 0.064680 |
| Target-balanced contradiction | 0.180213 | 0.167180 |
| Median-relative-depth p90 | 0.052262 | 0.048085 |

Dense retention was 0.878305. It removed 21,534 pairs: removed positive rate 0.301059 and
negative rate 0.028977. Positive removed recall was 0.324816; negative removed recall 0.055182.
Across 10,473 targets with removed and retained observations, `E+=0.138130`, `E-=-0.007429`,
and risk ratio was 1.719507.

Bootstrap intervals (p05-p95, 1,000 target-cluster replicates):

- `E+`: 0.131480-0.144362.
- common-target positive reduction: 0.011629-0.013039.
- log positive risk ratio: 0.516729-0.565032 (ratio 1.6767-1.7595).
- temporal far-minus-near contradiction: 0.106346-0.117862.

The development gate passed every support, nesting, bootstrap-sign, selectivity, negative-tail,
and p90 safeguard. It failed:

- positive reduction: 0.014244 versus required
  `max(0.01,0.15*0.116744)=0.017512`;
- positive risk ratio: 1.719507 versus required 2.0.

Dense-visible far-minus-near contradiction was 0.111943 across 11,884 paired targets. The
four-cell pose-conditioned sensitivity was estimable at 0.100082.

## Forensic bindings

- Artifact: `benchmarks/results/20260715T160300Z_cpu_tum_rgbd_signed_attribution_sitting.json`
  (SHA-256 `ca62497b71cef78bf046d5d4b463a2a71652ac1fc80ca4ce41e3b85b17cb5dfb`).
- Decision: `benchmarks/results/20260715_tum_rgbd_signed_attribution_DECISION.json`.
- Protocol: `benchmarks/results/20260715_tum_rgbd_signed_attribution_PREREG.md`.
- Result audit: `benchmarks/results/20260715_tum_rgbd_signed_attribution_RESULT.md`.
- Exploration: N52-N56; claims C09-C10; constraint R12; architecture A07.
