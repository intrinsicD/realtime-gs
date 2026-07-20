# Iteration 3 preregistration addendum 1 — valid-ray anchor eligibility

Date frozen: 2026-07-18 (Europe/Berlin), before any Iteration 3 fit or optimizer outcome.

This prospective input-feasibility repair amends only the ordering of the real-data anchor
eligibility and selection steps in
`20260717_inverse_projection_fiber_iter3_PREREG.md` (SHA-256
`59f0de21da20bb5785e2c5f14c89fc82114fed2d5945c704115d64b9fb3c27c8`). Synthetic roots,
arms, schedules, costs, gates, the real camera split, the 128-per-development-view budget, and all
reporting thresholds remain unchanged.

## Pre-fit observation

The preregistration first selected 128 spatial anchors and then required every selected ray to
intersect the development-camera-axis cube. A development-only input preflight, which loaded no
validation fields and no RGB or masks, showed that cube intersection is a genuine eligibility
property rather than a universal invariant:

| Development view | valid camera-z intervals | total observations |
| --- | ---: | ---: |
| C0001 | 419 | 640 |
| C0008 | 514 | 640 |
| C0014 | 526 | 640 |
| C0021 | 379 | 640 |
| C0026 | 562 | 640 |

The development-only fallback is unchanged: center
`[-0.05634030990516604,0.14126563808000317,2.425396529878651]`, extent
`1.3664202678061008`, and cube `center +/- 0.5*extent`. Rays use the unnormalized camera direction
`((u-cx)/fx,(v-cy)/fy,1)`, transformed to world coordinates, so the intersection parameter is
camera-z depth. Near depth remains `0.05`.

The original order could therefore select an invalid ray arbitrarily and terminate the real
interaction despite every view having at least 379 eligible observations. That would test anchor
ordering, not the preregistered correspondence mechanism.

## Frozen repair

For each development view:

1. compute all 640 camera-z ray/cube intervals from the already-frozen development-only bounds;
2. define eligibility as finite `far > near >= 0.05`;
3. fail closed unless the view has at least 128 eligible observations;
4. apply the original 8x8, up-to-two-per-cell, descending-footprint-area stable selection only to
   eligible observations, then fill any shortfall from remaining eligible observations by the
   same global stable area order;
5. require exactly 128 unique selected anchors per view and retain all 640 observations, including
   non-intersecting ones, as correspondence targets.

No observation is dropped based on an optimizer residual, validation behavior, color, amplitude,
or outcome. The selected-set indices, all 640 eligibility masks, interval endpoints, fallback
bounds, and semantic hashes must be saved in the real-run evidence. The valid-count floor is the
predeclared anchor budget itself (128), not a fitted threshold.

## Claim effect

This repair authorizes the bounded 640-hypothesis calibrated interaction; it does not make the
camera-axis cube a ground-truth object bound. Non-intersecting target observations are an explicit
diagnostic of incomplete fallback coverage, and the result must report their fraction. No
full-scene coverage, validation, correspondence, or production claim is gained by this addendum.
