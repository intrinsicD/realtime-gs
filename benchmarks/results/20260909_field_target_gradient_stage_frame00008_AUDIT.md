# Independent failure audit: target-gradient diagnostic

**Verdict: failed execution; scientific question inconclusive.** The worker correctly stopped at
the frozen equal-target numerical control. The original task remains consumed and failed; no
partial target-gradient or residual result is promoted. Do not change its tolerances, resume its
remaining states, or issue a completed RESULT for this run.

Reviewer: Codex-protocol-reviewer, distinct from Driver Codex-gradient-driver. Audit date:
2026-09-10. Protocol: `2246a380965a67c7d487ebae11ba607c42829e115e82a788b5bfe51e50c57426`.
Source envelope: `247d75336fe7f1d03563a245ea30895646d4c1c557b5b01ed3e3e08654e7d317`.
Run: `runs/20260909_field_target_gradient_stage_frame00008/`.

## Confirmed failure

The sixth declared state, `field_high_final_8102`, failed its `C0004` identity control before
any measured target-pair rows were produced for that state. The first five states each have
22 measured view pairs and passing identity controls: 110 partial pairs total. The remaining
three states were not started. No presentation phase, new fitted model, heldout evaluation,
optimizer step or topology update occurred. Canonical RESULT files are absent, appropriately.

Only the unweighted DSSIM gradient in raw saved quaternion coordinates exceeds the frozen
elementwise rule `abs(a-b) <= 1e-8 + 1e-4 * max(abs(a),abs(b))`:

| Failed-control statistic | Recorded value |
|---|---:|
| Violating quaternion-gradient elements | 6 / 103,372 |
| Difference L2 norm | 2.0406131924e-7 |
| Difference norm / reference norm | 2.8809211330e-5 |
| Cosine | 0.9999999995854738 |
| Maximum absolute difference | 1.0244548321e-7 |
| Maximum excess over elementwise tolerance | 3.3221694068e-8 |

All three recorded scalar losses agree exactly between the two identity paths. Other component
and parameter-group checks pass, including the weighted total gradient. These facts cannot
override the preregistered requirement that every component/group pass. A high cosine or small
relative norm does not retroactively replace the elementwise control.

The audit independently recomputed loss differences, relative-norm arithmetic, component/group
coverage and the overall pass/fail decision from the preserved JSON. It checked element counts
against the model row count and group dimensions. **The original two identity gradient arrays,
render buffers and image-space loss adjoints were not saved.** Therefore the six-element count
and recorded maxima cannot be independently recounted from element arrays. They are observations
from the verified source, bound to immutable failure JSON. Exactly equal scalar losses do not
establish exactly equal render pixels or equal intermediate adjoints.

## Provenance and integrity

The reviewer independently verified all 118 live and preserved source files, 76 cached input/state
hashes, the locked task/review/raw-seal copies, 21 preserved untracked files and the exact dirty
source-state digest. It also rehashed the administrative raw seal's 81 files, including heldout
bytes, without decoding or scoring them. Both Driver administrative receipts pass. All 231 guarded
worker reads are members of the 76-file allowlist; no denied/raw-source open is recorded. All six
visited model hashes match the sealed inputs; the five complete states record unchanged tensors
and retain correctly hashed mean-gradient archives.

The final review preceded initialization; the lock binds review SHA-256
`a179edb2b89c6167d50a24ba16103d2de0ded489c5682392a074ed2a8e3dcd78` and source commit
`66afd22fb4b1eebf86904c21cac19ccf61c9fca3` with preserved dirty state. The earlier rejected
initialization was a review-heading format check, repaired before creation of this run. It was
not a capture measurement or a numerical retry.

The execution log contains 110 measured-view lines and the explicit control-failure traceback,
with no preceding captured warning or CUDA/OOM error. Recorded environment is Torch 2.9.0,
gsplat 1.5.3, CUDA 12.8 on an NVIDIA GeForce RTX 3050. Inspection using the actual `.venv`
interpreter resolves the supported gsplat wheel; the loaded binary path was not separately
hashed in the worker receipt. No backend fallback or package-mixing explanation is established.

The failed state's own interval is retained. The global gradient interval stops at the fifth
complete state; it must not be presented as covering the failed sixth-state control. Process wall
and peak receipts include failed-control work and failure integrity checks. No preview phase
was reached and no fitting occurred. Shared-machine resource numbers support no performance
claim. No threshold, source or input changed after the failure, and no numerical retry was run
by this audit.

## Interpretation and smallest next check

The identity paths receive the same target through separate fresh render/backward graphs.
The observed disagreement therefore invalidates the declared numerical-repeatability control;
it is not evidence for a target-induced effect. The installed gsplat rasterization backward
uses floating-point atomic accumulation across image tiles, providing a plausible source of
repeat variation. This code observation does not locate the failure: render differences,
SSIM image-adjoint computation and renderer parameter VJPs were not separately recorded.
No cause of the previous reconstruction halos is identified.

The smallest useful continuation is a **new, separately registered and prospectively reviewed
numerical-control task**, explicitly selected after exposure to this failure. Keep this run intact.
Anchor that task to the already-exposed `field_high_final_8102` model and cached RGB `C0004`
view; no additional cameras, teacher comparisons or optimization are needed to begin. Freeze a
small repeat count and execution budget, save the relevant arrays, and distinguish:

1. Repeated rendered pixels from the fixed state.
2. Repeated L1/DSSIM image-space adjoints on one fixed rendered image.
3. Repeated parameter vector-Jacobian products for one fixed image-space adjoint.

This would measure where repeat variation enters. In the same new protocol, synthetic tests can
check the alternative direct-difference construction: for shared rendered image `I=f(theta)`,
form `h_B-h_A`, where `h_T=dL_T/dI`, and apply a **single** renderer backward
`J_f(theta)^T (h_B-h_A)`. This avoids subtracting two separately accumulated parameter-gradient
vectors. It does not automatically remove variation in separately computed SSIM adjoints; retain
independent identical-target loss paths and an image-adjoint control. Simply reusing the same
loss tensor or hardcoding a zero difference would make that control uninformative.

Use independent synthetic algebra/directional checks and retain the zero-input VJP behavior
before any revised teacher diagnostic. A new measured noise envelope must be reported as new
development evidence, not used to declare this failed attempt successful. Do not widen the old
tolerance to just cover the observed excess. A later full target diagnostic would need a frozen
measurement rule and its own approval; this audit does not authorize that execution automatically.

## Claim disposition

| Claim | Disposition | Evidence and limit |
|---|---|---|
| The frozen worker stopped on its numerical control. | Confirm | Sixth-state failure JSON, traceback and failed receipt agree. |
| The same-target repeat satisfies all frozen tolerances. | Retire | DSSIM/quats has six recorded violations. |
| All nine states completed. | Retire | Five complete, one failed before measurements, three unstarted. |
| Partial residual/gradient rows establish the target-error hypothesis or a halo cause. | Retire | Incomplete matrix and failed numerical prerequisite; raw evidence retained without scientific promotion. |
| The disagreement is consistent with finite-precision repeat variation. | Narrow | Identical target paths and small recorded differences; exact numerical source is unlocalized. |
| A common-render direct adjoint difference or repeat-noise protocol repairs the diagnostic. | Narrow | A proposed separately reviewed resolving test, not an observed repair. |
| Source/input boundaries were preserved. | Confirm | Independent source/cache/raw-seal hashes and recorded read allowlist pass, within the stated instrumentation scope. |
| Reconstruction capability, tomography validity or performance changed. | Retire | No fitting, density truth, completed scientific result or performance comparison occurred. |

## Audit execution and handoff

`python3 runs/20260909_field_target_gradient_stage_frame00008/audit_checks/failure_audit.py`
performed the independent hash, chronology, coverage and failure-arithmetic checks. Its machine
receipt is `audit_checks/failure_checks.json`; source and evidence hashes are embedded in the
companion canonical AUDIT JSON. The audit read the execution/source records and local gsplat
CUDA source. It did not render, backpropagate, run a repeat, recompute partial teacher metrics,
alter tolerances, perform fitting, upload private data or modify frozen files.

The failure page may link this AUDIT and raw failure evidence. It must remain marked failed,
with empty completed metrics/charts and no RESULT capability claims. Any completed-bundle gate
should continue to reject this run as non-results-bearing; that expected rejection is not a
reason to relabel it successful. Driver-owned report/link checks and bounded docs/ARA disposition
remain separate from this scientific failure audit.
