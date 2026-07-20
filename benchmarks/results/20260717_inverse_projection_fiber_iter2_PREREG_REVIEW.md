# Iteration 2 topology-repair preregistration — independent review

Reviewed on 2026-07-17 before implementation or fresh-root access.

Verdict: **PASS**.

The initial review failed on compute matching, source-anchor/bijection consistency, held-out
release timing, shuffled-control semantics, conditional validity, metric definitions, and claim
scope. The failed review is preserved at
`benchmarks/results/20260717_inverse_projection_fiber_iter2_PREREG_REVIEW_INITIAL_FAIL.md`,
SHA-256 `363a692c9a8cd5849f603d42ab329915b9577c008098b45b4a7217e8d1cdd20a`.

The amended protocol closes every issue:

- A, B, C, and oracle each receive 600 optimizer updates; B/C and every topology-side metric
  branch only from A's immutable update-400 snapshot.
- Source-view assignments are fixed to the representatives' actual spawning rows before the
  remaining exact bijection is solved.
- Held-out views release once, after every fitting-side decision across all roots is durable.
- Shuffled score direction, selection use, rejection values, exact-track/component denominators,
  float64 quantiles, cross-root aggregation, and conditional validity are explicit.
- The possible positive claim covers only the complete prune/contraction/rematch/refit bundle,
  not splitting or a general unknown-count topology lifecycle.

Freshness checks found no occurrence outside the preregistration of the namespace or nine roots.
Prior Iteration 1 artifact hashes match. No Iteration 2 outcome was accessed.

Frozen preregistration SHA-256:
`95adcf0f9d03761ca57bb36444a051f5c581e21e06eb399a355437bda9f6d28e`.

The final amendment explicitly gives A four source-group `8x8` held-out bijections (32 selected
entries/view; denominator 64 across two views), while B/C/oracle use one `8x8` bijection (eight
entries/view; denominator 16). The global held-out cost definition uses the same populations before
equal averaging across views 4–5. The reviewer passed this amendment before implementation or
outcome access.

The final hash differs from the immediately preceding PASS only by semantic-preserving Markdown
line reflow in the cross-root aggregation paragraph; the reviewer explicitly recertified it.
