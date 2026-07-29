# Fixed-anchor compact-field sweep — failed official attempt

- **Task:** `20260729_field_sweep_placement_stage_frames00008_00009`
- **Status:** failed before measured outcomes
- **Source:** clean commit `ec5735d5a549147f64490e57832578e72ae51400`
- **Protocol:** `9ff7057e47e3ea6a2af0532edbecf13036a892d9805e02cfe26aee7f33732844`
- **Run root:** `runs/20260729_field_sweep_placement_stage_frames00008_00009`

## Execution

The exact frozen command initialized the canonical run and verified all 55 sealed compact inputs
(8,373,380 bytes). The first discarded warmup began on `frame_00008`, seed `290900`, arm
`bounded_midpoint`. At refit step 0, `fit_field_fibers` raised:

```text
RuntimeError: fiber optimizer violated the exact source projection
```

No warmup completed, no measured cell started, and no held-out validation or comparison metric was
produced.

## Postmortem

An outcome-free debugger replay of the same failed warmup stopped at the invariant. The path was
`float32`; the frozen absolute gate was `0.0002`. The displayed maximum source-mean and covariance
round-trip errors were respectively `0.0009` and `0.0234`. The largest covariance term was about
`83,616`, making the observed covariance discrepancy consistent with native-resolution float32
round-off rather than treatment behavior. This diagnosis does not license weakening the frozen
gate.

## Disposition

This official attempt is **inconclusive due to implementation failure**. It supports no arm,
quality, resource, geometry, generalization, GPU, or default conclusion. The run and failed worker
directory remain append-only. Any repaired execution must use a new task id, source state,
protocol digest, and prospective review.
