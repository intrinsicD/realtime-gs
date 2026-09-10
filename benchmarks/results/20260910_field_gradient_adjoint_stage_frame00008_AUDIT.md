# Independent results audit — RTGS-024

- Task ID: `20260910_field_gradient_adjoint_stage_frame00008`
- Reviewer: `Codex-protocol-reviewer`; independent of the new driver and final prospective reviewer.
- Verdict: **accepted with narrowed claims**. The frozen diagnostic completed and its raw reductions agree. No reconstruction-quality prerequisite is reopened.
- Protocol SHA-256: `7f2e833b172e88592f7103aaaabd112b6a8112f4aceff10b5f4a1d528e5a341f`
- Source binding: 122 files, `f29e6e8343b63a6285ced94c7578c93bc240329ec45715fbdd0f3e490fd8c7b1`.
- Run: `2026-09-09T22:48:56.866257+00:00` to `2026-09-09T22:53:14.539302+00:00`, exit 0. This is development evidence on an already exposed capture.

## Findings

The new numerical controls distinguish the repeat variation from a target difference. All six whole-chain renders and image adjoints are exactly equal; six fixed-image adjoint evaluations are also exactly equal. Nevertheless, repeated nonzero VJPs on one fixed graph and one fixed adjoint differ. At the exposed `field_high_final_8102/C0004` probe, DSSIM/quaternion gradients exceed the predecessor's elementwise threshold in all 15 fixed-VJP pairs, with at most 12 violating elements and maximum absolute difference `4.44473699e-7`. This demonstrates repeat variation in the nonzero parameter-VJP path for this probe. It does not retroactively repair RTGS-023 or prove the exact mechanism of its original failure. The six new numerical zero controls and nine state zero controls pass from their saved arrays.

The target discrepancy is concentrated near and outside the silhouette. Values below are arithmetic means over the 22 training views, with RGB scaled to [0,1]. Shares are means of per-view shares, not a pooled-pixel statistic.

| Fixed mask region | Per-region MAE | Signed RGB bias | Share of absolute target error |
|---|---:|---:|---:|
| Interior | 0.00344643 | +0.00115561 | 6.7431% |
| Boundary | 0.04856269 | +0.04698458 | 39.1003% |
| Exterior | 0.00206723 | +0.00206723 | 54.1566% |

The boundary has much higher error per pixel; the exterior occupies most of the canvas. Boundary plus exterior account for 93.2569% of mean per-view absolute error. The teacher has positive bias and lower mean local contrast at the boundary (0.0885762 versus photo 0.1078694). This establishes a difference in the cached target signal. It does not localize its effect in parameter space or establish the cause of the reconstructed halo. Low/highpass values are not an orthogonal energy partition.

| Saved-state kind | Total means-gradient cosine | Relative gradient difference | Views passing all three repeat flags |
|---|---:|---:|---:|
| initial | 0.273494 | 1.034623 | 66/66 |
| rgb_final | 0.352577 | 1.835698 | 66/66 |
| field_high_final | 0.415299 | 0.911465 | 66/66 |

These are means over views and then the three inherited seeds, using `g_field = g_photo + direct_delta`. Relative difference divides by the photo-gradient norm. They are local within-state comparisons; vectors across different state topologies are never compared. All 198 total-means comparisons pass the prospectively fixed observed-repeat flags. Across all components and groups, 3,168 comparisons pass all three flags; the other 396 are structural zero/null cases in initial quaternion and inactive SHN coordinates. Both the null values and unresolved counts are preserved. Two repeats and a ten-times threshold are descriptive precision context, not a confidence interval or bound on shared systematic error. The stored gradients of the view mean have no separate aggregate precision certificate.

## Independent checks and provenance

Independent NumPy code was prepared and tested on generated arrays before outcome access. After execution authorization it read checksum-bound arrays without importing the experiment driver/report, rendering, running backward, or optimizing. It recomputed all 45 numerical repeat pairs, 15 zero/null controls, 198 two-repeat sites (9,504 float32 arrays), nine mean archives (486 float64 arrays), all component totals, direct/derived gradients, nulls, precision flags, view/seed/kind reductions, and all 22 residual region calculations. Morphology and full-image reflected box filters were implemented independently. All 241 NPZ evidence hashes/dtypes passed; 77,603 scalar reconciliations passed. Reconciliation tolerances handle float64 reduction order only; experimental gate counts and boolean flags were recomputed using the original thresholds, unchanged.

The audit independently rehashed 122 live and preserved source files, the task/review seals, the exact dirty source-state record and 31 preserved untracked files. All 76 cached inputs and all 81 administrative raw-seal files match. Raw-seal inspection was opaque byte hashing only. The worker records 244 allowed reads and no denials: the photo-only numerical phase opens exactly the selected model, C0004 photo cache and camera metadata; other cached target decoding starts after the numerical gate. Entry/exit verification and ordered stage intervals agree. All nine saved/effective tensor digests, parameter shapes, original model hashes, opacity endpoints and zero endpoint gradients agree. The selected preview NPZs are exact existing snapshot copies. There were no fitting or topology updates, retries, or heldout numerical evaluation.

Loss scalars remain source-bound producer evaluations; their recorded repeat differences were checked without replaying every loss computation. The repository source envelope excludes third-party binary contents; recorded versions are PyTorch 2.9.0, gsplat 1.5.3 and CUDA 12.8 on an RTX 3050. Source inspection confirms the common-render adjoint-difference path and effective-alpha chain. RTGS-023 source and canonical failed audit hashes remain unchanged. Its partial outcome exposure is explicitly disclosed in this new protocol.

The producer's full host `./scripts/verify.sh` log ends in `verify OK`. Earlier sandbox failures were the four existing localhost-socket checks and remain preserved; CPU verification is not CUDA-performance evidence. The independent reviewer did not repeat protected GPU work. Resource counters are descriptive and correctly scoped to the shared-GPU worker; no time or speed comparison is supported.

## Visual and delivery disposition

The reviewer inspected the 22-view contact sheet, fixed quarter-cycle samples from the reconstruction/orbit/elevation GIFs, and the actual in-app viewer screenshots. The existing selected model is visible and retains soft exterior structures/halo; these are unchanged model previews, not a newly trained result. The viewer console warning remains disclosed and its finite-model/two-component-quad explanation is consistent with source inspection; no clean-console claim is made.

One presentation defect was corrected: the first browser-clipped framebuffer images included controls and changed text selection, so the claim that their 11,306 changed pixels were UI-free is retired. The original evidence is preserved. Independently extracting native-PNG rectangle x400:900, y250:530 from the full screenshots excludes controls and axis, shows a visible orientation change, and gives 33,238 pixels with maximum channel change >2. See `verification/viewer-visual-check-corrected.json`. No numerical camera-vector claim is made.

Shared report rendering, actual report-browser checks and final bundle/manifest checks are pending this canonical scientific audit and remain separate delivery obligations. This avoids requiring a report that cannot render until this audit exists.

## Claim dispositions

| ID | Claim and evidence scope | Disposition |
|---|---|---|
| C1 | Observed nonzero VJP repeat variation at one exposed photo probe; raw repeats/control arrays | Confirm at that probe; old run remains failed |
| C2 | Cached target residuals around/outside silhouette, 22 training views | Confirm signal discrepancy; no causal gradient localization |
| C3 | Local photo/field gradient disagreement, nine frozen states and all parameter groups | Confirm with observed-repeat and coordinate limits |
| C4 | Existing model visible and viewer scene changes | Confirm corrected native-PNG proof; retire original UI-free count |
| C5 | Remedy, new high-quality reconstruction, physical density, field-only recovery, or reopening D/E | Not established; no fitting or heldout quality evaluation |
| C6 | Performance improvement | Not established on this descriptive shared-GPU run |

Each claim's source/execution binding and raw artifact paths are recorded in the companion audit JSON and `audit_checks/raw_reduction_artifacts.json`. No result/source file was rewritten by the audit.

## Smallest next test supported

A useful next counterfactual would change only the teacher target outside the existing frozen training mask: replace those pixels with the corresponding cached photo values, preserve teacher pixels inside the mask, and hold saved states, cameras, coordinates and loss weights fixed. A separately registered protocol could compare its direct local gradients with original-teacher and photo gradients using the same numerical controls. This would test the contribution of outside-mask signal to gradient disagreement without assuming that it caused the halo. It is a post-outcome proposal, not an executed remedy or permission to launch.

Freeze that comparison and stopping criteria before generating the counterfactual. Even improved local agreement would require an independently specified, authorized paired reconstruction-quality trial with an adequate photo control and heldout reporting before any quality claim. The failed RTGS-021 prerequisites remain unavailable.

## Commands and evidence

Executed: `python3 /tmp/rtgs024_independent_reductions.py` on generated arrays before outcome access; then `python3 /tmp/rtgs024_audit.py` and `python3 /tmp/rtgs024_provenance_audit.py` on completed evidence. Exact script copies and machine-readable checks are preserved under the canonical run's `audit_checks/`. No new GPU experiment, fitting, threshold change, heldout decode or private upload was performed. The producer owns the remaining report/bundle/documentation handoff.
