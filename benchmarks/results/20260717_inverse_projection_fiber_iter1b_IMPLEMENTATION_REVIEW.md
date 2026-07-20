# Iteration 1b pre-official implementation review

Reviewed: 2026-07-17, after focused development tests and before any official root.

Recommendation: **FAIL; official execution blocked**.

## Blocking findings

1. The free arm received the common covariance tensor, but its Cholesky/log round-trip changed
   the realized covariance bits. The harness checked byte identity only for fiber arms, contrary
   to the preregistered byte-identical realized initialization. On an unrelated development
   scene, the maximum relative covariance difference was `4.58e-16`; this is numerical
   equivalence, not byte identity.
2. `make_gt_gaussians` is sensitive to the ambient Torch default dtype. The harness neither
   forced nor rejected a non-float32 ambient stream, so the same root could construct a different
   scene.
3. A six-arm, two-update smoke using unrelated roots wrote receipts under the official iter1b
   namespace in `/tmp/ipf_iter1b_dev_smoke_1784295894727576340`. It produced no official root or
   outcome, but polluted namespace exclusivity.
4. A failure after official root construction would leave partial artifacts without an
   exclusive terminal `INVALID` receipt, despite the no-retry rule.

## Provenance hardening required

- Re-hash scientific sources at the end and invalidate source drift.
- Reject nested/colliding output paths before root construction.
- Hash GT means/covariances, camera tensors, and every projected target tensor.
- Label `ru_maxrss` as process-global cumulative rather than an arm-local peak.

## Verified aspects

Static compilation, Ruff, seven focused geometry tests, objective/reduction semantics,
stop-gradient finite differences, exactly 400 updates, validity sentinels, evaluation
denominators, gate formulas, and the claim boundary passed review. The reviewer constructed no
official root or result.
