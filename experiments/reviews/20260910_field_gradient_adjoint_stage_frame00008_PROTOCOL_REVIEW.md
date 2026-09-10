# Prospective Protocol Review

- Task ID: `20260910_field_gradient_adjoint_stage_frame00008`
- Protocol SHA-256: `7f2e833b172e88592f7103aaaabd112b6a8112f4aceff10b5f4a1d528e5a341f`
- Reviewer: `Codex-continuation-reviewer`
- Verdict: `approved`
- Outcome Access: `none`
- Source binding: `122` files, aggregate SHA-256 `f29e6e8343b63a6285ced94c7578c93bc240329ec45715fbdd0f3e490fd8c7b1`
- Reviewed draft task file SHA-256: `853a5d40d375b1d4a85ce9619f6a505a344479ca7999f56924b10371aabe7a7f`
- Review date: `2026-09-10` (Europe/Berlin)

## Scope

Approve the exact implemented development protocol. It first investigates numerical
repeat variation with the photo target at one previously exposed state/view, then,
only after its fixed controls pass, measures cached target residuals and local
gradient differences at nine predeclared saved states and 22 training cameras.
No optimization or topology update is permitted.

Outcome Access means no calibrated outcomes of this new task were generated or
inspected by the reviewer. Synthetic algebra/test outputs are pre-execution controls.
Prior RTGS-021 exposure and the predecessor's failed numerical control were known
when designing this follow-up and remain expressly disclosed. That predecessor stays
failed with its original thresholds, source and partial evidence.

## Checks

- Recomputed the exact protocol digest and 122-file live source binding. Independently
  rehashed all 76 listed cache/state files and byte counts without decoding arrays.
  The predecessor's frozen source envelope still matches. The new canonical run
  directory was absent during review.
- Inspected the new driver and report helper, relevant immutable predecessor helpers,
  SSIM implementation, gsplat wrapper, input guards and synthetic tests. Scope includes
  photo-only phase access, exact target paths, fixed graph/adjoint reuse, local opacity
  mapping, nonzero-adjoint checks, repeat storage and report publication.
- Ran `.venv/bin/python -m pytest -q tests/test_field_gradient_adjoint.py tests/test_field_gradient_adjoint_report.py`:
  35 passed, one CUDA test skipped in this reviewer environment. These tests use
  synthetic data and temporary report fixtures; no capture computation occurred.
- Inspected the Driver's separate CUDA synthetic selftest receipt at
  `/tmp/20260910_field_gradient_adjoint_cuda_selftest.json`: identity control, all 18
  distinct-target component/group comparisons, full-loss combination and opacity
  chain control pass; the receipt declares no capture inputs. The reviewer did not
  independently rerun that GPU command.
- `experiment_contract.py validate`, focused Ruff checks and `git diff --check` passed.

## Findings

The implementation follows the fixed-Jacobian chain rule
`delta = J^T(h_field - h_photo)`. It propagates the difference directly and derives
the field gradient from the photo gradient plus delta, using lossless float32
component records and float64 repeat means/reductions. Synthetic independent target
backward and full-loss comparisons test nonzero algebra; a successful zero-input
control is not used as its substitute.

Fresh whole-chain repeats, fixed-image adjoint repeats and fixed-graph nonzero VJP
repeats distinguish the declared numerical stages. The initial gates remain fixed
and stop before field measurements upon failure. Ordinary parameter-VJP repeat
variation is descriptive, not a retroactive pass of the predecessor's gate. At every
measured site, paired nonzero VJP repeats retain the prospectively declared noise
flags. These flags provide observed precision context only; they do not bound shared
systematic error, prove statistical significance, or certify an aggregate gradient.

Report review initially found omitted consistency checks. Before this final binding,
the helper was corrected to validate fixed-image render/loss gates, unchanged-input
flags, ordered pair coverage, every null-VJP component/group, nonzero fixed adjoints,
and required checksum/size-bound raw-array evidence. Synthetic counterexample tests
now cover the corrected failures. Defined/total cosine counts are displayed separately
from repeat-resolved/total counts; the two can differ near zero. No protocol precision
flag was altered to make them coincide.

Four-stage diagnostic history is distinguished from fitting history. Completed RESULT
conflicts are checked before publication mutations; failure publication preserves raw
observations and does not fabricate missing metrics. Existing PLYs/previews remain
labelled as saved snapshots. Final raw-array reductions, report/browser checks,
manifest/bundle validation and independent results audit remain delivery obligations.

The final protocol includes lossless storage, 12 GB free-space preflight, a 10 GB
output cap, 2 GB write reserve, state/total time limits and fail-closed stopping.
Administrative input hashing is distinct from capture decoding. The worker excludes
raw datasets, heldout inputs and field-target decoding during the photo-only stage.
No new reconstruction-quality, field-only, physical-density, causal-remedy or
performance claim is authorized by completing this diagnostic.

## Protected Actions Not Taken

The reviewer did not initialize or execute the protected run, decode calibrated
targets/models, compute capture residuals or gradients, inspect heldout outcomes,
modify old protocols/source/results, upload private material, commit or push. Only
the designated review records were written. Any later load-bearing source or
protocol change requires a fresh exact-digest review before execution.
