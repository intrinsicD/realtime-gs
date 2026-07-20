# kernel_support_taper_phase_b_ablation

- Timestamp (UTC): `2026-07-15T20:36:22+00:00`
- JSON artifact: `benchmarks/results/20260715T202917Z_cpu_kernel_support_taper_iter2_ablation.json`
- JSON SHA-256: `f44f3f3fa69fd6bdf67e8da61f90fec952ffb2a38c577a216ff256af0e2263fd`
- Command: `/home/alex/Documents/realtime-gs/.venv/bin/python benchmarks/kernel_support_taper_ablation.py ablate --audit benchmarks/results/20260715T202218Z_cpu_kernel_support_taper_iter2_audit.json --phase-a-review benchmarks/results/20260715T202218Z_cpu_kernel_support_taper_iter2_audit_AUDIT.json --seal benchmarks/results/20260715_kernel_support_taper_iter2_SEAL.json --output benchmarks/results/20260715T202917Z_cpu_kernel_support_taper_iter2_ablation.json`
- Implementation seal: `4f13421bfb570e8e42570bb97f39aa88bb90c2c8d822864f4272fbb68786e674`

## Frozen outcome decision

- Primary hypothesis pass: `False`
- C1 taper mean foreground PSNR gain: `-0.014483 dB`
- Hard-forward control mean gain: `-0.018470 dB`

The common hard renderer defines primary evaluation. This result is limited to fixed-topology CPU synthetic depth-initialized refinement.
