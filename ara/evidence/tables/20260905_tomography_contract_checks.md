# Tomography contract correction — 2026-09-05

The RGB field-refit objective now normalizes by cached target RGB field energy. Exact co-located
half-weight splitting preserves the combined density/RGB objective and its prediction gradients.
An opt-in soft source-footprint mode releases five relative geometry coordinates and reports
source drift. Hard mode keeps its original three trainable parameters and source equality.

Source binding: base `2ebe52cc28a1b0e812a6c436d49225fac4ee943f`; source/test patch SHA-256
`2ec04c6c9e44f3efa5408342a0992ad1e231357ee3d6dd4e98c31ecf3d3d4a75`. Per-file hashes, exact full-suite command, failures,
and the Git-based frozen-protocol audit are in `20260905_tomography_contract_checks.json`.
This is a bounded implementation record, not a reconstruction experiment or a performance result.

| Contract | Evidence | Disposition |
|---|---|---|
| Split-invariant combined loss and all prediction gradients | `tests/test_field_refit.py`, including actual RGB refit and cached-energy controls | Supported on deterministic CPU fixtures; C42 |
| Black targets and explicit legacy replay, including zero iterations | `tests/test_field_refit.py` | Finite fixture gradients and explicit legacy scale preserved |
| Identical soft/hard initialization, SPD geometry, finite-difference gradients and subset state | `tests/test_field_refit.py` | Opt-in geometry contract; no preferred tether or quality claim |
| Public lifter and saved geometry round trip | `tests/test_field_lifter.py`, `tests/test_field_cli_integration.py` | Pipeline-integrated; soft mode requires fixed topology |
| CI is not covariance inversion | `tests/test_beam_fusion.py`, existing carrier/covariance repair tests | Three orthogonal long-beam counterexample; no universal uncertainty claim |
| Faster or better dome reconstruction | No matched calibrated experiment | Unestablished |

All 155 tests in the affected contract modules pass. The complete CPU suite,
including slow tests, has 1987 passed, 24 skipped and
2 failures. Both failures concern the immutable
`20260806_gaussian2d_image_refinement_janelle_frame00008` protocol:

- `test_registered_three_arm_program_is_valid`: its frozen source binding already differs at the
  untouched starting commit (106 bound files versus 103 frozen files). The current source also
  differs. This was checked from Git bytes, not assumed to predate the patch.
- `test_exact_owner_selected_six_folder_matrix_is_frozen`: adding the three refit controls changes
  the effective-configuration digest. Removing only those new serialized fields reproduces the
  frozen digest. This guard correctly prevents treating the changed executable/configuration as
  that historical protocol. The task JSON, old results, and review digest were preserved.

An earlier test invocation disabled Python bytecode generation and caused three strict PyTorch
runtime-directory failures; normal bytecode generation resolves those failures. The final suite
uses the normal environment. During implementation, regression tests also caught missing gradients
in old hard-fiber optimizers; keeping inactive source coordinates as buffers restored that API.
The saved standard Gaussian loader converts to float32, so its round-trip test compares at the
loader's documented dtype. No quality threshold was lowered.

A scratch execution on three existing calibrated training fields exercised the public pipeline
with two soft refit steps and explicitly capped inputs. It loaded no source images, masks, or
held-out fields and passed finite/SPD/output checks. No model, quality metric, viewer claim, or
scientific result is retained from that smoke. GPU execution and a formal quality comparison did
not run.

The compact carrier pipeline already enforces Beam followed by renderer-aware covariance repair;
this patch retains it. EWA projection is local affine, center visibility is spatially approximate,
and view gains still fit density alone. Normalized RGB denominators are not asserted to be
physical line-integral density. Native compact teacher color remains the renderer-faithful target.

For a controlled geometry-constraint comparison, select the same fields, cameras, initialization,
seed, count, schedule, weights and `topology_rounds=0` in both arms; change only
`refit.source_constraint` and record `source_anchor_weight`. RGB normalization must be identical.
Soft mode leaves SH anchored at its initial source direction and retains existing visibility,
observability, mass and opacity policies. A new result-bearing comparison needs its own prospective
protocol; the historical protocol must not be refreshed to accept this patch.

Review is provisional and self-authored. The full gate is not green, and no commit or push was
made. The remaining protocol lifecycle work is separate from the implemented correction.
