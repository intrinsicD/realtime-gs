# Smooth-boundary and support evidence

## SH nonnegative color floor

The frozen three-seed CPU fixed-topology Phase-A audit found pooled negative preactivation
incidence of 0.336527%, recoverable blocked-gradient mass of 0.090828%, and fixed SMU-1 recovered
mass of 0.025266%. Every per-seed and pooled gate failed. Phase B was forbidden, so neither the
SMU-1 forward arm nor hard-forward/SMU-gradient control trained.

- Result: `benchmarks/results/20260715T192112Z_cpu_sh_activation_iter2_audit.json`, SHA-256
  `67431510a5620b383db729d4877bb2c2b581eb81270f2110f0bd47b0d561f4ae`.
- Independent audit: `benchmarks/results/20260715T192112Z_cpu_sh_activation_iter2_audit_AUDIT.md`,
  SHA-256 `8c0d5e552b7e2b18b54c382348872bebdd4237c08c625acd8a1f55b46400e24f`.
- Exploration: N60-N61; claim C11; heuristic H02.

## C1 kernel-support taper

The adjacent q=[12,16) annulus passed the frozen loss-directed mechanism screen. In the authorized
three-seed Phase B, the C=12, W=4 C1 taper changed diffuse common-hard foreground PSNR by
-0.014483 dB on average and its hard-forward gradient-only control changed it by -0.018470 dB;
both won zero of three seeds and all safety guards passed. The hard q<12 default remains.

- Mechanism result: `benchmarks/results/20260715T202218Z_cpu_kernel_support_taper_iter2_audit.json`,
  SHA-256 `57421f39ff5d983ac37bc63e2c1eabe1a9528a6ed4415001d52a3ee9bce76609`.
- Utility result: `benchmarks/results/20260715T202917Z_cpu_kernel_support_taper_iter2_ablation.json`,
  SHA-256 `f44f3f3fa69fd6bdf67e8da61f90fec952ffb2a38c577a216ff256af0e2263fd`.
- Independent audit: `benchmarks/results/20260715T202917Z_cpu_kernel_support_taper_iter2_ablation_AUDIT.md`,
  SHA-256 `e8735c671a66c7942985d5b1800085b12f0916099d6aca99c18946ab891abdaf`.
- Exploration: N62-N63; claim C12; heuristic H02.

## Three-sigma visibility margin

The first sealed attempt failed closed on exact-depth ordering. The retry preserved the baseline
3-sigma code path and its established relative order while adding only newly visible candidates.
All target, q<12 support, ordering, finite, view, and count invariants passed. The diffuse pool
missed four of 2,480,463 q<12 pairs across two Gaussian/view exposures; missed effective-mass
fraction was 1.646359e-8 and render-delta/residual was 3.986964e-8. All materiality gates failed,
so Phase B was forbidden and the 3-sigma default remains.

- Result: `benchmarks/results/20260715T213132Z_cpu_visibility_margin_iter2_audit.json`, SHA-256
  `cacbf8782cf803e27f6715bfe7dd673d0be4f4eabfb51dc920838339e1b08785`.
- Independent audit: `benchmarks/results/20260715T213132Z_cpu_visibility_margin_iter2_audit_AUDIT.md`,
  SHA-256 `21c262aad36f02cf9a6520d50c2d2a867a22758e0486daa35094cdd78b9eb928`.
- Exploration: N64-N65; claim C13; constraint R13.

## Subsequent controlled question

The smooth-boundary branch is closed for this setup. The subsequent Carve exact-count protocol ran
only through Phase A: production grouping compressed 2.34%-2.68% of primitives and failed every
seed's materiality floors, so moment-versus-prune utility remained untested. The audited result and
the later representation/multiscale/coordinate evidence are indexed in
`tables/20260716_stage1_carve_multiscale_quaternion.md`.
