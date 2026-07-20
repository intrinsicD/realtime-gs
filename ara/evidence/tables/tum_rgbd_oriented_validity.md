# TUM registered-RGB-D oriented-point validity

## Protocol and artifacts

- Frozen protocol: `benchmarks/results/20260715_tum_rgbd_oriented_validity_PREREG.md`
- Development: `benchmarks/results/20260715T143959Z_cpu_tum_rgbd_oriented_validity_xyz.json`
  (`384bc416bcd9c8539aa85a16da533fd2d024124bd549e80f337d6951eb206834`)
- Confirmatory: `benchmarks/results/20260715T144052Z_cpu_tum_rgbd_oriented_validity_desk.json`
  (`febc92898cb2c8f60ea4bad4c699017067e79e9bee325b210b0edb636b2b32a7`)
- Frozen thresholds: `benchmarks/results/20260715_tum_rgbd_oriented_validity_THRESHOLDS.json`
  (`e3ad0adfcd7909bcf64dabaa829dee87dab57b5cb52da7d0ae2f5ff5f9264324`)
- One-shot seal: `benchmarks/results/20260715_tum_rgbd_oriented_validity_CONFIRMATORY_SEAL.json`
  (`ade8c68318388343d5d306becbcbe2e3b055d3ec536e2e1f692f4804eb863978`)
- Result audit: `benchmarks/results/20260715_tum_rgbd_oriented_validity_RESULT.md`
- Scope: official TUM `fr1/xyz` development and sole `fr1/desk` confirmation; 48 T construction,
  eight V audit, eight H sealed views; no RGB, H payload, optimization, or utility arm.

## Frozen metrics and decision

| metric | development | confirmatory | threshold | pass |
| --- | ---: | ---: | ---: | :---: |
| `A` | 0.70036 | 0.68441 | >= 0.49026 | yes |
| `A_min` | 0.64417 | 0.59750 | >= 0.32208 | yes |
| `S` | 0.86686 | 0.80359 | >= 0.52012 | yes |
| `S_10` | 0.73198 | 0.63593 | >= 0.36599 | yes |
| `R90` | 0.02497 m | 0.20211 m | <= 0.04245 m | no |
| `D90` | 0.03104 | 0.25186 | <= 0.05000 | no |
| `C50` | 0.91516 | 0.88720 | >= 0.76516 | yes |
| `C10` | 0.67197 | 0.50262 | >= 0.52197 | no |
| `F` | 0.04503 | 0.06247 | <= 0.07755 | yes |

Development constructed 40,341 eligible targets and 34,970 oriented-supported targets. Desk
constructed 39,422 and supported 31,679. Desk surface/depth medians were 15.42 mm and 2.14%, but
p90s were 202.11 mm and 25.19%; the stopped decision is driven by a distributed heavy tail, not an
empty population. Of supported desk targets, 6,279/31,679 exceeded the surface limit,
7,392/33,048 exceeded 5% relative depth, and 3,518/31,679 missed the normal-cosine floor.

## Bound conclusion

Reject the exact registered-depth target/visibility protocol as a transferable prerequisite and
withhold Phase B. Retain the generic CPU-tested oriented backend/canonicalization and zero-default
loss APIs; keep the TUM backend harness-local. Do not tune the consumed desk case. Any revisit
requires new sequences, signed discrepancy evidence, construction-only occlusion/rigidity
controls, and an ordinary extra-depth utility control.
