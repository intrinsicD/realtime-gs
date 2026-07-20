# SH color-floor incidence and SMU-1 preregistration — provenance retry 2

## Incorporation and chronology

This protocol incorporates every scientific definition, seed, scene, initialization, activation,
metric, threshold, gate, stopping rule, and claim boundary from
`benchmarks/results/20260715_sh_activation_PREREG.md` at SHA-256
`5353c4aa37c13e280f0bf3761679424e0bb5e17b4e942a7ff36275e84be88c1f`.
Those numerical choices are unchanged.

The first sealed Phase-A attempt began at `2026-07-15T19:14:37Z` and is permanently recorded by
`benchmarks/results/20260715_sh_activation_PHASE_A_ATTEMPT.json` at SHA-256
`af764e81d6afd36736fe95835553b795ba90b43b9ab5f6c14d6afe8ea92029c3`. After all six hard-arm
trainings, the harness failed closed before serialization while checking loaded-source provenance:
Pillow modules inside the repository-local `.venv` were incorrectly classified as unsealed
repository source. The process emitted only condition/seed progress and the provenance exception;
it wrote no audit artifact, printed no diagnostic fraction or quality metric, and no outcome was
inspected. The old attempt marker and seal are retained and must not be deleted, overwritten, or
used as evidence.

This retry was frozen at `2026-07-15T19:16:41Z`, after diagnosing that implementation failure and
before inspecting any SH-floor statistic. It fixes only the provenance classifier: loaded Python
under repository-owned `src/`, `tests/`, `benchmarks/`, or `scripts/` is checked against the seal,
while environment packages under `.venv/` are bound by the already frozen Python/PyTorch/platform
environment fingerprint rather than treated as repository source. A focused regression test covers
the loaded-source verifier. No training, data, activation, aggregation, decision, or interpretation
logic changes.

## Retry-specific immutable artifacts

- Preregistration: this file, while also sealing the incorporated prior protocol.
- Implementation seal:
  `benchmarks/results/20260715_sh_activation_iter2_SEAL.json`.
- Phase-A once-only marker:
  `benchmarks/results/20260715_sh_activation_iter2_PHASE_A_ATTEMPT.json`.
- Phase-B once-only marker:
  `benchmarks/results/20260715_sh_activation_iter2_PHASE_B_ATTEMPT.json`.
- Official audit and ablation outputs must use fresh `iter2` filenames and may not reuse any first
  attempt path.

The complete implementation and both protocols must pass the same full verification and receive a
new seal before the retry. Phase B remains forbidden unless the retry's frozen Phase-A gate passes
and an independent scientist-review manifest binds the exact retry audit and retry seal.

## Official retry commands

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  benchmarks/sh_activation_ablation.py seal --output \
  benchmarks/results/20260715_sh_activation_iter2_SEAL.json

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  benchmarks/sh_activation_ablation.py audit --seal \
  benchmarks/results/20260715_sh_activation_iter2_SEAL.json --output \
  benchmarks/results/<UTC>_cpu_sh_activation_iter2_audit.json

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  benchmarks/sh_activation_ablation.py ablate --audit \
  benchmarks/results/<UTC>_cpu_sh_activation_iter2_audit.json --phase-a-review \
  benchmarks/results/<UTC>_cpu_sh_activation_iter2_audit_AUDIT.json --seal \
  benchmarks/results/20260715_sh_activation_iter2_SEAL.json --output \
  benchmarks/results/<UTC>_cpu_sh_activation_iter2_ablation.json
```
