# Source-aware photometric supervision

Official CPU artifact:
`benchmarks/results/20260715T062601Z_cpu_cross_view_supervision.json`.
Preregistered protocol:
`benchmarks/results/20260715_cross_view_supervision_PREREG.md`.
Post-run audit:
`benchmarks/results/20260715_cross_view_supervision_RESULT.md`.

## Protocol and invariants

Seeds 0/1/2 used 40-Gaussian synthetic scenes, twelve 48x48 cameras, nine training views,
held-out views 3/7/11, shared 150-Gaussian/view fits, and 90 bounded-ray lift iterations. Pure
Gradient and deterministically corrupted-metric Hybrid each compared inclusive `all`,
`leave_one_source_out`, and `matched_nonself_dropout`. Colors, opacity, shape, rotation, merge,
refinement, and density control were fixed or disabled as preregistered.

All six arms per seed passed bitwise step-0 equality, paired 90-step target schedules, two-step
zero-learning-rate output/schedule equality, source-range and full-output-count equality, finite
serialization, and source-bound provenance checks. LOSO removed exactly the target source group.
The matched control removed the same per-target primitive count and frozen scalar-opacity sum,
never removed a target-own primitive, and excluded every primitive exactly once globally. Final
primitive counts were 1303/1293/1262 by seed. Three independent read-only audits exactly
recomputed the summaries, decisions, mask layouts, and embedded hashes.

## Primary result

| family / metric | all | LOSO | matched non-self | LOSO vs all | wins | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Gradient held-out depth RMSE | 0.154307 | 0.154077 | 0.154787 | 0.149% lower | 2/3 | fail: >=2% |
| Gradient all-source p90 | 0.211965 | 0.213057 | 0.207754 | 0.515% worse | 2/3 | fail: >=10% |
| Gradient held-out PSNR | 19.6894 dB | 19.6849 dB | 19.7531 dB | -0.0045 dB | -- | safety pass |
| Hybrid held-out depth RMSE | 0.150330 | 0.150344 | 0.151052 | 0.009% worse | 1/3 | fail: >=2% |
| Hybrid all-source p90 | 0.163228 | 0.167002 | 0.170615 | 2.312% worse | 0/3 | fail: >=10% |
| Hybrid corrupted-source p90 | 0.205832 | 0.208639 | 0.209914 | 1.364% worse | 1/3 | fail: >=15% |
| Hybrid held-out PSNR | 19.6725 dB | 19.6875 dB | 19.6877 dB | +0.0150 dB | -- | safety pass |

Coverage remained 1.0 and alpha IoU changed by +0.00005 for Gradient and +0.00034 for Hybrid.
Both materiality and matched-control attribution decisions are false. The matched control equalizes
count, scalar opacity, and global exclusion exposure, not coherent topology, visibility, projected
alpha, color, or spatial coverage; this caveat limits a positive attribution claim but does not
weaken the failed all-versus-LOSO utility result.

LOSO reduced common cross-only training L1 by 0.21% for Gradient and 0.57% for Hybrid and improved
nearest-GT median by 1.72%/1.31%. These mechanism checks show that the intervention changed the
intended objective, but the changes did not reach held-out expected-depth or robust tail metrics.

## Decision

Keep inclusive `all` as the production default and retain LOSO/matched dropout only as reproducible
research controls. Stop source-exclusion and schedule sweeps on this frozen setup. The result is
limited to synthetic pure Gradient and deterministically corrupted-metric Hybrid and does not show
universal photometric-supervision failure or real-data harm. Pivot to one robust world-frame
position-consistency term between fixed train-view matches, with a shuffled-match control that
preserves pair count and graph degree; defer shape, appearance, confidence, and plane/normal terms.
