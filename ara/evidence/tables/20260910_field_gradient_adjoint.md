# RTGS-024: common-state target-gradient diagnostic

Provenance: ai-executed. Independently audited development evidence; C48.

The cached 2D field targets change the local fitting gradient substantially, even though their aggregate image error is small. This measures a signal mismatch, not a reconstruction repair or the cause of the previous halo.

| Frozen state kind | Total position-gradient cosine | Relative difference / photograph norm |
|---|---:|---:|
| Initial | 0.2734942779082579 | 1.0346225361814916 |
| Photograph final | 0.3525774196982357 | 1.8356982755182543 |
| High-field final | 0.41529879283317644 | 0.9114650811478158 |

Each row averages 22 views and then seeds 8101/8102/8103. Comparisons are within an identical saved state; different topologies are not vector-compared. All 198 total-position sites are defined and pass the three frozen observed-repeat flags. Initial quaternion/inactive SHN group cosines are undefined and retained. Two repeats and a ten-times threshold are descriptive sampled precision, not confidence intervals or a certificate against systematic error; view-mean vector aggregates have no separate precision certificate.

| Frozen mask region | Mean regional MAE | Mean signed RGB bias | Mean per-view share of absolute error |
|---|---:|---:|---:|
| Interior | 0.0034464262647930254 | 0.0011556066674591442 | 0.06743056826081906 |
| Boundary | 0.048562690612496136 | 0.04698457830775294 | 0.39100328333071865 |
| Exterior | 0.002067225923652148 | 0.002067225923652148 | 0.5415661484084623 |

Boundary plus exterior account for 93.256943% of mean per-view error. The exterior is large and its photograph target is black. These are not pooled-pixel shares and do not attribute the parameter difference to any one region.

The first attempt, RTGS-023, stopped at its frozen same-target DSSIM/quaternion gate after five states/110 pairs. Its partial metrics remain inconclusive and no RESULT was created. Its six-element failure count is producer/source-bound because old raw identity arrays were not retained. The separately registered RTGS-024 method followed that exposure and passed distinct design/source review before running. Its photo-only probe found identical rendered pixels and image adjoints but variation in repeated nonzero parameter VJPs. This is consistent with backward repeat variation, without proving the old failure's exact mechanism or retroactively passing it.

RTGS-024 evaluates the original full-canvas unclamped 0.8 L1 + 0.2 D-SSIM loss, using the preserved CUDA renderer and effective-alpha chain. Direct adjoint differences use one render graph; raw parameter VJPs are stored twice for photograph and difference, and field gradients are derived by addition. All six numerical and nine state zero controls pass. The independent audit re-reduced 241 NPZ archives and 77,603 scalar checks, and verified 122 source files, 76 cached inputs and 81 opaque administrative raw hashes. No optimization/topology update, heldout decoding or new quality scoring occurred.

Protocol digest: `7f2e833b172e88592f7103aaaabd112b6a8112f4aceff10b5f4a1d528e5a341f`; source aggregate: `f29e6e8343b63a6285ced94c7578c93bc240329ec45715fbdd0f3e490fd8c7b1`. Exact task/result/audit/report hashes: `ara/evidence/tables/20260910_field_gradient_adjoint/artifact_bindings.json`. Canonical RESULT/AUDIT: `benchmarks/results/20260910_field_gradient_adjoint_stage_frame00008_RESULT.json`, `_RESULT.md`, `_AUDIT.json`, `_AUDIT.md`; raw evidence: `runs/20260910_field_gradient_adjoint_stage_frame00008/audit_checks/`. The audited frozen run lasted 2026-09-09T22:48:56.866257+00:00 through 22:53:14.539302+00:00. Shared-GPU counters support no performance claim.

Final report: `runs/20260910_field_gradient_adjoint_stage_frame00008/index.html`. Actual browser checks find 18 inline SVG plots, no horizontal overflow or report console errors. All 593 local targets return HTTP 200 and served/local HTML hashes match. Run and bundle validators pass. The viewer shows the unchanged selected high-field/8101 model. Independent native scene-only ROI verification finds 33,238 changed pixels after orbit and 10,401 chromatic pixels initially. Original mis-scaled clips and their retired UI-free count remain preserved. Numeric camera vectors were unavailable. Known nonfatal Viser quad/deprecation console messages remain disclosed in the browser receipt. Final delivery and repository checks live in `ara/evidence/tables/20260910_field_gradient_adjoint/verification_receipt.json`, outside the recursively hashed run; the canonical scientific audit's earlier delivery-pending wording remains chronological history.

The proposed next step is a separately frozen gradient-only outside-mask photograph replacement with original-teacher, photograph and sham controls. It adds photograph/mask information and is an attribution probe, not a field-only method. No new counterfactual or fitting experiment was launched. C47, its failed prerequisites and unavailable D/E remain unchanged; no high-quality reconstruction, physical-density or generalization conclusion follows.
