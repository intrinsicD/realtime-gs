# Hard kernel-support C1 taper — provenance retry 2

## Incorporation and chronology

This retry incorporates every scientific definition, equation, width, scene, seed, split,
configuration, diagnostic, gate, arm, metric, threshold, claim boundary, and stopping rule from
`benchmarks/results/20260715_kernel_support_taper_PREREG.md` at SHA-256
`c78a74ea67a4a0d327b8ef884006dc8ad5781da9a632f557c2e9f370a8868a58`. None changes.

The first sealed Phase A passed and was independently cleared. Its preserved artifacts are:

- implementation seal SHA-256
  `f35827d362318d4eb55d637cdadb77c5a97deb68fd62e6f04e231e9c39702184`;
- Phase-A audit SHA-256
  `6380dc0b92043db608f6ba056c1cbaa2509e4eeba62a7b20e3f0fb7eacdde59c`;
- strict independent review SHA-256
  `3f8f404912d0ef1e30e605ef6f7c194d3ec07e38f517ca1493166ff03a182919`.

The first once-only Phase-B attempt began after that clearance. It recreated diffuse seed 0 and
completed the `c1_taper` training loop, then failed before evaluating, aggregating, serializing, or
printing any candidate result. The invariant compared the candidate's in-memory
`list[tuple[int,int]]` SH checkpoint schedule directly with the semantically identical
JSON-restored `list[list[int]]` baseline. Python container-type inequality falsely reported an SH
schedule mismatch. Console output contained only the seed/condition progress line, an unrelated
tensor-to-scalar warning from a step-zero bound, and the exception. No candidate metric, loss,
history, parameter, or quality outcome was exposed or inspected. The consumed marker is preserved
at SHA-256 `0c3b1e96ab56680db64758c9e2ceb17a5c53bb5f950bfd416d1165b08433e3c1`;
the attempted output does not exist and may not be recreated under the old namespace.

This retry was frozen at `2026-07-15T22:18:44+02:00`, after diagnosing that representation-only
failure and before any candidate outcome access. The sole executable change is to compare the
active-SH-degree and primitive-count checkpoint schedules by canonical JSON hash, under which
tuples and lists share the same serialized representation. Target-view schedules already consist
of scalar lists and remain exact. No training, rendering, initialization, diagnostic, gate,
evaluation, interpretation, or numerical choice changes.

## Retry requirements

The first protocol, seal, Phase-A artifact/review, and consumed Phase-B marker remain append-only
historical evidence but cannot authorize this retry. Before any new outcome execution:

1. the harness must verify the exact hashes and presence of all four preserved artifacts above and
   verify absence of the old failed output;
2. focused and full CPU verification must pass;
3. the complete implementation and both support protocols receive a new source/environment seal;
4. Phase A is rerun completely under a fresh once-only marker and fresh output, despite the
   scientific definitions being unchanged; and
5. a new independent scientist review must recompute and bind the retry audit before Phase B.

The new harness/test code must also exercise tuple/list canonical schedule equivalence. A smoke
test may validate the invariant without training or inspecting candidate quality. The old Phase-A
numbers are known, so its rerun is a deterministic provenance check under unchanged frozen gates,
not a new threshold-selection opportunity.

## Retry-specific artifacts and commands

- Preregistration: this file, while also sealing the incorporated first protocol.
- Seal: `benchmarks/results/20260715_kernel_support_taper_iter2_SEAL.json`.
- Phase-A marker:
  `benchmarks/results/20260715_kernel_support_taper_iter2_PHASE_A_ATTEMPT.json`.
- Phase-B marker:
  `benchmarks/results/20260715_kernel_support_taper_iter2_PHASE_B_ATTEMPT.json`.
- Official JSON outputs must contain `kernel_support_taper_iter2` and use fresh UTC names.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  benchmarks/kernel_support_taper_ablation.py seal --output \
  benchmarks/results/20260715_kernel_support_taper_iter2_SEAL.json

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  benchmarks/kernel_support_taper_ablation.py audit --seal \
  benchmarks/results/20260715_kernel_support_taper_iter2_SEAL.json --output \
  benchmarks/results/<UTC>_cpu_kernel_support_taper_iter2_audit.json

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python \
  benchmarks/kernel_support_taper_ablation.py ablate --audit \
  benchmarks/results/<UTC>_cpu_kernel_support_taper_iter2_audit.json --phase-a-review \
  benchmarks/results/<UTC>_cpu_kernel_support_taper_iter2_audit_AUDIT.json --seal \
  benchmarks/results/20260715_kernel_support_taper_iter2_SEAL.json --output \
  benchmarks/results/<UTC>_cpu_kernel_support_taper_iter2_ablation.json
```

Every seal/source/environment/review/strict-JSON/append-only requirement and every positive or
negative interpretation from the incorporated protocol remains in force. No result from this
retry can change a default without the separately required CUDA, density-enabled, and real-scene
confirmations.
