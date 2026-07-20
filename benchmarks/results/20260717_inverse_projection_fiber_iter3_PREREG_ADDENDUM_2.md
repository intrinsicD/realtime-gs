# Iteration 3 preregistration addendum 2 — consistent augmented-dustbin cost

Date frozen: 2026-07-18 (Europe/Berlin), before any official Iteration 3 root or real-data fit.

This prospective arithmetic repair amends only the augmented transport kernel in
`20260717_inverse_projection_fiber_iter3_PREREG.md` (SHA-256
`59f0de21da20bb5785e2c5f14c89fc82114fed2d5945c704115d64b9fb3c27c8`) and follows valid-ray
Addendum 1 (SHA-256
`f4ef57320edf1e099c24033753bf3e939d2c87fcf6b927b65bd5d6af213c91fc`). No official synthetic
root has been constructed or run, the frozen real bundle has not been opened by the Iteration 3
runner, and no scientific threshold, root, schedule, arm, split, or gate changes here. Development
smokes used only non-official roots.

## Pre-outcome implementation finding

The first augmented Sinkhorn implementation assigned the declared dustbin cost `d` to the real-to-
dust and dust-to-real edges but overwrote the dust-to-dust completion edge with zero cost. In the
balanced equal-capacity 1-by-1 case, the matched plan then costs `(c + 0)/2`, while the two
unmatched routes cost `d`; consequently the real match is indifferent at `c = 2d`. Row-softmax is
indifferent at its declared dustbin cost `c = d`. Leaving the zero completion would therefore give
arms C/D/S an effective unmatched threshold twice that of arm B and confound the preregistered
softness-versus-two-sided-capacity comparison.

The shared-method contract says that the augmented problem has a dust row and dust column with a
single finite dustbin cost. It does not authorize a free dust-to-dust edge. The implementation
review therefore treats this as the preregistration's already-authorized correctness repair, not a
hyperparameter change.

## Frozen repair

All edges incident to either augmented dustbin, including the bottom-right completion, now carry
the same frozen cost `d = 4.0`. Equivalently, for the 1-by-1 balanced equal-capacity problem the
augmented cost is

```text
[[c, d],
 [d, d]]
```

and real support is exactly one half at `c = d`, greater below it, and smaller above it. The repair
is shared byte-for-byte by UOT-uniform C, UOT-area D, and shuffled-view S. Hardmin A, row-softmax B,
oracle O, the Bhattacharyya cost, marginal construction, `rho`, temperature schedule, and all gates
remain unchanged.

The repaired `src/rtgs/lift/fiber_correspondence.py` has SHA-256
`58c14b49520d7213c28a3d70b514b4b22140bdac2351c6c8dba8c6ac01c26a4a`. The regression source
`tests/test_fiber_correspondence.py` has SHA-256
`b5510597fa732af8029d3e6a8addbd2666beba86bdc6ce4e01571cc0a2ec4a56` and includes the analytic
1-by-1 crossover test. The repaired core, exact fiber, source-anchored SH, and real-harness focused
set passed 51 CPU tests; Ruff passed on the repaired core and real-harness files.

## Claim effect

This repair restores the intended common meaning of `dustbin_cost=4.0` across the row and
transport arms. It establishes no association, geometry, real-data, topology, appearance, or
performance outcome. Any subsequent failure remains a failure under the original frozen gates.
