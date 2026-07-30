# Fixed-anchor compact-field placement successor

## Forensic binding

| Item | Binding |
| --- | --- |
| Original consumed task | `20260729_field_sweep_placement_stage_frames00008_00009` |
| Original disposition | Failed in the first discarded warmup; zero measured outcomes |
| Successor task | `20260730_field_sweep_placement_f64_stage_frames00008_00009` |
| Protocol SHA-256 | `a45ee0da9f2282cdeebfa93a9321408a9d1a7ce4b64ba6a2746f61e30546a1e0` |
| Locked source | `a69337346fbecd156c20211abd638f976e327d62` |
| Measured design | Two named same-capture frames × three paired seeds × three placement arms |
| Audit Markdown | `benchmarks/results/20260730_field_sweep_placement_f64_stage_frames00008_00009_AUDIT.md` |
| Audit Markdown SHA-256 | `f89beafae1c995de28a2df3a28bf67c8cbc27b2c0b37a6009c174d3874f084ad` |
| Corrected audit JSON SHA-256 | `b6775322b80972dc88be02001974c34b2d82f44414be1caa2ae259fc847f7d95` |
| Reader handoff | `runs/20260730_field_sweep_placement_f64_stage_frames00008_00009/index.html` |

The successor changes the shared field-compute dtype to opt-in float64 and adds structured failure
evidence. It retains the predecessor's data, split, paired seeds, anchors, arms, refit, topology,
evaluation, resource protocol, and frozen decision rule.

## Independently recomputed result

| Measure | Bounded midpoint | All-view consensus | Source-excluded robust |
| --- | ---: | ---: | ---: |
| Pooled final held-out compact RGB MSE, geometric mean | `0.026755888660351994` | `0.02580366611690199` | `0.024017792485982237` |
| Median measured CPU wall time, seconds | `52.37958948701271` | `52.65091423000558` | `52.75939034897601` |
| Median measured peak RSS, MiB | `3236.466796875` | `3234.86328125` | `3233.79296875` |

| Frozen rule operand | Observed | Disposition |
| --- | ---: | --- |
| Robust / midpoint pooled final RGB ratio, required `≤ 0.95` | `0.8976637924784316` | Pass |
| Robust wins over midpoint, frame 00008, required `≥ 2/3` | `3/3` | Pass |
| Robust wins over midpoint, frame 00009, required `≥ 2/3` | `3/3` | Pass |
| Minimum robust supported-track fraction, required `≥ 0.95` | `0.9921875` | Pass |
| Maximum measured projection invariant, required `≤ 0.0002` | `5.820766091346741e-11` | Pass |

The robust/all-view pooled ratio is `0.9307899264070091`, but the scene-level comparison reverses:
robust/all-view is `0.8402792176410981` with `3/3` wins on frame 00008 and
`1.0310499997047535` with `1/3` wins on frame 00009.

## Evidence boundary

- This is outcome-exposed development and same-capture replication evidence against immutable
  compact Gaussian-field teachers, not source RGB.
- It supports the frozen robust-versus-midpoint result on the two named frames.
- It does not support a claim that robust beats both controls on both scenes.
- It does not establish rendered RGB quality, physical geometry, GPU behavior, speed advantage,
  topology utility, cross-dataset generalization, or a production-default change.
- The projection diagnostic combines mean-coordinate and covariance-entry discrepancies, so its
  frozen numerical threshold is an implementation invariant rather than a physical pixel unit.
- The canonical report and exact initial/final Viser command passed HTTP and result-bundle smoke
  checks; no subjective visual assessment was recorded.
