# Compact-only carrier policy closure — result — 2026-07-28

## Status

**Complete; independently audited `PASS_WITH_SCOPE_LIMITS`.**

This is a single-scene development result. It used only fitted 2D Gaussian observation tensors,
calibrated cameras, and 3D Gaussian tensors. Source RGB, dense masks, packed alpha, `SceneData`,
and the dense trainer were denied throughout the run. Held-out cameras were not evaluated.

The producing run was sealed before execution against the preregistration, command, Git state,
all repository Python sources used by the process, and all compact input containers. The
independent audit found the seal unchanged and recomputed all arm receipts and frozen gates.

## Decision summary

The supported bounded carrier policy is:

1. fit-only Beam Fusion;
2. renderer-aware symmetric covariance repair;
3. 380 compact fixed-topology updates with means, rotation, scale, opacity, and SH0 trainable;
4. a second 380-update compact fixed-topology phase, in which means may be frozen; and
5. no legacy covariance, opacity, appearance, sampled soft-support, or clone-all stage.

Strict fitted-Gaussian visual-hull pruning is viable as a projected-center containment stage, but
this experiment did not settle whether it should occur immediately before or after the selected
full second phase. Higher SH helped as an alternative phase but was not tested incrementally
after the selected phase. Those two interactions remain before freezing the implementation
policy.

## Recomputed validation evidence

All ratios are paired by root and aggregated geometrically over roots
`282701,282702,282703`.

| Question | `J_Q` ratio | `J_U` ratio | Frozen decision |
| --- | ---: | ---: | --- |
| Phase-1 means-frozen / all | 1.0224 | 1.0027 | Fail 2% non-inferiority |
| Phase-1 means-only / all | 4.1823 | 2.6712 | Reject |
| Second all-family phase / stop | 0.7170 | 0.8029 | Necessary; 3/3 `J_Q` wins |
| Phase-2 means-frozen / all | 1.0147 | 1.0116 | Pass 2% non-inferiority |
| SH3-only alternative / stop | 0.9159 | 0.9521 | Material alternative; 3/3 wins |
| Legacy clone / half-budget continuation | 0.9754 | 0.9997 | Fail 5% and 3-root gate |
| Preserving clone / half-budget continuation | 0.9699 | 0.9957 | Fail 5% gate |
| Preserving / legacy clone | 0.9944 | 0.9960 | Fail 2% replacement gate |
| Strict prune / stop | 1.0404 | 1.0354 | Viable within 5% |
| Strict prune + 190 recovery / stop | 0.8069 | 0.8565 | Viable; containment preserved |

The selected phase-2 means-freeze tradeoff costs 1.47% `J_Q` and 1.16% `J_U` relative to
all-family continuation while preventing further center motion.

## Absolute validation ranking

| Arm | Geometric validation `J_Q` | Geometric validation `J_U` | Rows |
| --- | ---: | ---: | ---: |
| `continue_all_380` | 0.00405465 | 0.00764806 | 5,000 |
| `continue_means_fixed_380` | 0.00411440 | 0.00773645 | 5,000 |
| `mass_tangent_clone_all_recover_190` | 0.00423145 | 0.00787160 | 10,000 |
| `legacy_clone_all_recover_190` | 0.00425546 | 0.00790346 | 10,000 |
| `strict_prune_recover_190` | 0.00456287 | 0.00815864 | 4,729–4,734 |
| `higher_sh_only_380` | 0.00517945 | 0.00906943 | 5,000 |
| `corrected_C_all_380` / `stop_380` | 0.00565505 | 0.00952596 | 5,000 |
| `corrected_C_means_fixed_380` | 0.00578162 | 0.00955197 | 5,000 |
| `strict_prune` | 0.00588356 | 0.00986311 | 4,729–4,734 |
| `corrected_C_means_only_380` | 0.02365091 | 0.02544549 | 5,000 |

## Containment result

Strict pruning removed 5.32%, 5.36%, and 5.42% of phase-1 rows. The retained centers had:

- zero `q > 9` violations against the positive-amplitude fitted 2D Gaussian union in every
  fitting view; and
- zero near-plane violations in every fitting view.

Means were frozen during recovery, so both properties remained exact afterward.

This is not a proof of “no free-floating Gaussians.” It cannot detect a center inside the
multi-view visual hull but away from the physical surface. Requiring an entire mathematical
Gaussian to lie inside a mask is impossible because Gaussian support is infinite; even a finite
3-sigma footprint criterion would erode boundaries and was not tested here.

## Math disposition

- The corrected covariance residual matches the point renderer:
  `P = J Σ Jᵀ + 0.3 I`, followed by the symmetric log-eigenvalue residual
  `sqrt(mean(log(lambda(C^-1/2 P C^-1/2))^2))`.
- The old one-sided whitened matrix residual biases scale shrinkage and is retired.
- Fitted 2D mixture amplitude does not identify 3D opacity, so opacity repair is retired.
- Cross-view fitted-mixture amplitude also makes legacy appearance repair gauge-dependent; its
  corrected alternative was empirically immaterial and is retired as a separate stage.
- Compact point-color risk correctly supervises rendered color at sampled coordinates, but color
  alone retains alpha/color gauge freedom; multiple parameter families must move in phase 1.
- Copied-opacity clone-all changes coincident alpha from `a` to `1-(1-a)^2`. The preserving
  operator fixes coincident optical density and the selected-axis second moment, but neither clone
  arm earns a stage.

## Resources and claim boundary

The whole policy-closure research process peaked at 2.72 GiB RSS. Per-arm CUDA allocator receipts
and compact-teacher/3D tensor byte counts are retained in the run. These are absolute diagnostics,
not a controlled dense-vs-compact comparison and not evidence for a general VRAM reduction claim.

The result supports making the development carrier path structurally compact-only. General VRAM,
quality, runtime, and cross-scene claims still require controlled multi-scene evidence.

## Artifacts

- Preregistration: `20260728_compact_only_carrier_policy_closure_PREREG.md`
- Producing harness: `benchmarks/compact_only_carrier_policy_closure.py`
- Run: `runs/compact_only_carrier_policy_closure_20260728`
- Independent audit: `20260728_compact_only_carrier_policy_closure_AUDIT.md`
- Machine-readable audit: `20260728_compact_only_carrier_policy_closure_AUDIT.json`
