# Inverse-projection fiber fitting, Iteration 2 — independent results audit

## Verdict

**ACCEPT VALID NEGATIVE.** The official transaction committed once, all frozen roots are consumed,
and the scientific status `FAIL` is supported by independently reopened raw evidence. No
transaction, chronology, held-out-isolation, source-binding, or gate-rederivation blocker was
found.

## Claim table

| Claim | Kind and scope | Evidence | Disposition |
| --- | --- | --- | --- |
| Source fibers remain exact | measured, noiseless synthetic | result Gate 1 and raw NPZ rederivation | confirm, synthetic only |
| Residual topology reliably recovers eight tracks | asserted across three official roots | Gate 2 `FAIL/PASS/FAIL` | retire for this protocol |
| Accepted topology can recover exact tracks | measured on two accepted roots | fit/held-out NPZs | narrow to 2/3 roots |
| Residual identity separates from shuffled | causal synthetic control | equal mean coverage, Gate 4 fail | retire |
| Three-root center reduction is approximately 100% | aggregate metric | result JSON | reject as non-reportable; rejected placeholder is zero |
| Real/appearance/occlusion/unequal-count/GPU capability | asserted extension | none | not tested |

## Provenance and transaction checks

- Transaction `05b9478e90b04d0b9c17f8ba5202c085` is `COMMITTED`; result, terminal,
  lifecycle, root-state, attempt, and handoff receipts agree and all nine roots are `CONSUMED`.
- Result SHA-256 is
  `d153706a5534a5f1d319d18b2961c944842bb01cd1573992b280c5ce096a2dfd`.
- Executed source archive SHA-256 is
  `373545e0da7e05e2a78d8d83118a0cbb898dcfc40190ba4729ee08bfc7a90cec`;
  all 45 members match the attempt manifest and all 44 reviewed source hashes match.
- Every root bundle was constructed once. Held-out data was released exactly once after the
  pre-held-out barrier. Learned-state hashes are identical before and after held-out access.
- The worker returned zero, wrote no stderr, left a quiescent process group, and had no killed
  stragglers.
- The earlier archive-allowlist rejection occurred before artifact creation, result reservation,
  transaction creation, or root access. It is not a consumed scientific retry.

## Independent rederivation

The audit loaded the sealed executed code and recomputed every gate input from each root's exact
typed input, fit, and held-out NPZ. All roots returned `pass=true`, with zero false evidence checks
and zero scalar-summary mismatches. Proposed conditional acceptance was `[true,true,false]` and
shuffled acceptance `[false,false,false]`, exactly matching the committed result. Gate 1 is valid.

Gate 2 fails because root 0's survivor precision is `0.904762`, root 1 passes, and root 2 has seven
representatives with `0.875` coverage and no hidden-mode-2 candidate. Accepted roots 0 and 1 both
reach perfect fit, held-out, and exact-track accuracy; root 2 is rejected. Mean track improvement
over hard-min is `0.135417`, below the frozen `0.20` floor. Proposed and shuffled mean coverage are
both `0.958333`, so Gate 4 fails despite shuffled rejecting all roots and proposed winning track
fraction.

## Reporting correction and remaining evidence

Do not quote the result's `0.999999642` relative center reduction: rejected topology uses zero-valued
placeholder geometry, invalidating that cross-root aggregate. Report accepted-root center p90 only
(`3.103e-7`, `2.955e-7`). No CUDA/GPU work or one-shot replay was performed in this read-only audit.

Promotion requires a fresh protocol in which association capacity and track survival are coupled
during fitting, followed by unequal-decomposition, occlusion/dustbin, appearance, and calibrated
real-data tests. Those are new experiments, not repairs to this consumed result.
