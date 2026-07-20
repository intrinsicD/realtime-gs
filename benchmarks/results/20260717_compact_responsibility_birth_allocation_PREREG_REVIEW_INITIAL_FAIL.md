# Compact residual-responsibility birth allocation — preregistration review

Reviewed preregistration SHA-256:
`cb384fb560cffae23550b6b4975a3fb439c0a05bb6997a079696830587b11bb9`.

Verdict: FAIL

Unresolved findings: 3

1. **Freeze the evaluation-bank seed derivation.** The text specifies only “a new named
   evaluation derivation” and does not give its domain label or byte serialization. The general
   `domain_seed` pseudocode also applies `utf8` to integer roots/parts without defining their
   representation. State the exact label, ordered parts, and encoding (for example, decimal
   `str(value).encode("utf-8")`) for uniform and proposal banks.
2. **Make the terminal decision map exhaustive.** “R beats G/U” is undefined: it can mean the
   three primary `J_Q` gates or all five primary-plus-safety gates. Mixed `J_U` failures can
   consequently match an allocation-negative label and `UNIFORM_RISK_TRADEOFF`. Define named
   per-comparator primary and safety booleans and an ordered, mutually exclusive truth table.
3. **Give the final PLY round-trip gate a number.** “Within the existing strict tolerance” does
   not identify a tolerance in the cited iter3 helper, which only records maximum errors. Freeze
   exact per-field `atol`/`rtol` values (and comparison arithmetic) before any official result.

Apart from these closure defects, the literal-color-basis VJP is valid for the current Torch
point compositor, the inactive-attempt arithmetic is consistent, the four strata match clone and
split quotas, the `835 -> 867` row/ID arithmetic is coherent, and preserving survivor Adam
moments with zero newborn moments and scalar step 35 is implementable through the existing
density-surgery pattern. No official or focused seed, schedule, bank, score, selection, training
step, or outcome was generated or inspected during this review.
