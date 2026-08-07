# Probabilistic Gaussian-field all-dataset experiment — producer result

- Task: `20260805_probabilistic_field_pipeline_association_rollback_mixed`
- Status: `pending_independent_audit`
- Calibrated cell attempts: `66`
- Calibrated cells passing all hard gates: `59`
- Retained structured field-fit failures: `7`
- Successful cells using the unmasked-support fallback: `2`
- Dataset field sets: `11`
- Per-view teacher cap: `512` components

## Synthetic mechanism decisions

- association: `False` (0 / 3 seed wins)
- mask: `True` (3 / 3 seed wins)
- schedule: `False` (0 / 3 seed wins)
- shape: `True` (3 / 3 seed wins)
- topology: `False` (0 / 3 seed wins)

## Boundary

Transactional-association-rollback successor created after the approved `20260805_probabilistic_field_pipeline_aabb_eligible_mixed` run completed the unchanged 483-cell synthetic matrix, discarded warmup, and twenty-one measured attempts, then failed closed at measured cell 22/66 (`karate_00060_default`, seed 80501, all candidate mechanisms) on exact `RuntimeError: a supported projection left the valid camera domain during M-step`. That run preserved nineteen successful terminals and two eligible `hard invariant violation: transport real mass` terminals before aborting; its root failure SHA-256 is `63206958f3fd963e277d8487000f9b76383ae5a8aa9120b4235a65cbd32216d4`, run-receipt SHA-256 is `0d1abb1b84bba0d3d72edede63cf3582e31e22a665961ef9ab25448fc97758fa`, and terminal failure SHA-256 is `07a2ff8e0fbe8cfcf2926a56b77ea8ac40704f509b38a53fca35b4324671396b`; it and every predecessor remain immutable. No quality, convergence, aggregate, report, or viewer outcome from that run was accessed. This task reruns the complete matrix and changes one candidate-association behavior only: `FieldAssociationConfig.failure_policy` becomes `rollback`, so RuntimeError or ValueError from the private association clone returns the untouched placement with exact rolled-back diagnostics and is then rejected by the unchanged missing/non-finite/insufficient-transport hard gates. Rollback cannot produce a successful candidate, impute a metric, substitute a native result, or make a rejected model claim-eligible; rejected models remain presentation-only. Native arms, synthetic cells, forward-AABB eligibility, exact empty-support retry, thresholds, seeds, splits, input/component caps, optimizers, metrics, aggregation, and decision rules are unchanged. Any exception outside private association rollback, the reviewed empty-support retry, and complete structured hard-invariant continuation still aborts the root. Evidence remains development-only over the sealed eleven-field deterministic 512-component-per-view proxy and can establish neither complete-field fidelity, source-RGB reconstruction quality, spatial resolution, true globally coupled multi-marginal OT, GPS-Gaussian reproduction, GPU or real-time performance, cross-scene generality, production-default suitability, nor accuracy from independent-half agreement. Prior exposure is immutable failure chronology only and does not authorize quality or runtime tuning. Calibrated results cannot rescue a failed synthetic invariant or mechanism gate.

These are producer measurements, not audited claims. The result must not be rendered or interpreted until a distinct independent results audit checks raw cells, aggregation, input guards, approximations, and viewer artifacts.
