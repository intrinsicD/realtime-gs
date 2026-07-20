# Stage-1 semantic factorial utility result (VALID)

- JSON: `20260716T063637Z_cpu_stage1_semantic_factorial_utility.json`
- JSON SHA-256: `005eabffc062e158c1ca510865fa40be799733bc5f9bc6c4c3444fff63fc0d9c`
- Raw NPZ: `20260716T063637Z_cpu_stage1_semantic_factorial_utility_RAW.npz`
- Raw NPZ SHA-256: `6c639b3758fb1564225ee02caa897e9700dbb399b67ab0ee2cc267d13ebc0ae9`
- Raw collection SHA-256: `a5eab9669990b6a343a8747832dca5c67edc61d09b4ab78fd8a2702bbddc1c2e`
- Raw array count: `13944`

The JSON and this note bind the completed uncompressed raw archive. Quantitative claims require the independent review specified by the preregistration.

## Frozen decision

```json
{
  "by_backend": {
    "Depth": {
      "mean_psnr_difference_db": 3.1272303263346366,
      "worst_psnr_difference_db": 2.702273686726887,
      "mean_ssim_difference": 0.02481647994783187,
      "worst_ssim_difference": 0.022926191488901737,
      "noninferior": true,
      "material_improvement": true
    },
    "Carve": {
      "mean_psnr_difference_db": -2.20531357659234,
      "worst_psnr_difference_db": -2.4084510803222656,
      "mean_ssim_difference": -0.040016558435228146,
      "worst_ssim_difference": -0.052596092224121094,
      "noninferior": false,
      "material_improvement": false
    }
  },
  "repair_utility_survives": false,
  "cross_backend_material_improvement": false,
  "decision_validity_gates_pass": true,
  "all_final_heldout_psnr_at_least_10_db": true,
  "default_change_authorized": false,
  "independent_results_audit_required": true
}
```
