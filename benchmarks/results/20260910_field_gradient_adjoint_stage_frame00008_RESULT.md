# Common-render image-adjoint target-gradient diagnostic

Completed training-only target residuals and common-state gradient measurements on all 22 cameras at nine predeclared saved states. The new photo-only numerical gate and all nine common-render zero-effect controls passed. Saved model snapshots were rendered without fitting or topology changes.

Development diagnostic complete; interpret region and within-group gradients with the independent audit. These observations do not identify a causal remedy or reopen the failed reconstruction-quality prerequisites.

| Saved state | Means cosine, mean over views | Relative gradient difference | Resolved/total views | Defined/total cosines |
|---|---:|---:|---:|---:|
| initial_8101 | 0.273519 | 1.034608 | 22/22 | 22/22 |
| rgb_final_8101 | 0.352422 | 1.892350 | 22/22 | 22/22 |
| field_high_final_8101 | 0.397546 | 0.922174 | 22/22 | 22/22 |
| initial_8102 | 0.273472 | 1.034632 | 22/22 | 22/22 |
| rgb_final_8102 | 0.334206 | 1.829891 | 22/22 | 22/22 |
| field_high_final_8102 | 0.432651 | 0.900099 | 22/22 | 22/22 |
| initial_8103 | 0.273492 | 1.034627 | 22/22 | 22/22 |
| rgb_final_8103 | 0.371104 | 1.784854 | 22/22 | 22/22 |
| field_high_final_8103 | 0.415700 | 0.912123 | 22/22 | 22/22 |

| Region | Mean per-view MAE | Mean signed RGB bias | Mean absolute-error share |
|---|---:|---:|---:|
| interior | 0.003446 | 0.001156 | 0.067431 |
| boundary | 0.048563 | 0.046985 | 0.391003 |
| exterior | 0.002067 | 0.002067 | 0.541566 |

One previously exposed calibrated capture, cached downscale-eight training targets, three existing seeds and saved initial/final states. No optimization, optimizer-update replay, topology change, heldout evaluation, high-quality reconstruction, end-to-end field-only recovery, physical density, causal failure attribution or speed claim. Gradients use the stated reconstructed saved-state coordinates; exported opacity is not a preserved raw optimizer logit. This follow-up is designed after the failed numerical control in 20260909_field_target_gradient_stage_frame00008; its five partial states and failed sixth control were exposed. The old run remains failed and is not completed or retuned. Common-render derived field gradients use g_field = g_photo + J_transpose(h_field-h_photo); they are a first-order saved-state diagnostic, not a replay of independently accumulated historical GPU gradients.

- Previously exposed capture; local first-order development diagnostics, not new reconstruction quality or physical-density evidence.
- Masks define regions only. Loss and gradients use the full cached training canvas. Raw-seal checks hash heldout bytes; the measurement worker never decodes them.
- Step is a measured-view ordinal. Parameters never change: the curves are not fitting histories. Global diagnostic stage intervals are repeated for context and must not be summed across series.
- Root PLYs and previews show the prospectively selected existing snapshots. PNG/GIF previews are display-clamped; numerical loss uses unclamped renderer output.
- Effective-opacity gradients are mapped by alpha*(1-alpha) to local logit coordinates. Historical raw logits and Adam updates are unavailable and are not reconstructed.
- Undefined cosine bars are omitted; raw summaries retain null counts. Cosines and relative deltas are null near zero. Parameter-group norms have different units; they cannot rank causal importance. Mean per-view statistics and statistics of the mean gradient are reported separately.
- Common-render gradients use g_field = g_photo + J^T(h_field - h_photo), with two nonzero VJP repeats per site. Float64 means derive from saved float32 component arrays.
- Zero-effect controls cover C0004 per state and test null/routing behavior only. They do not certify nonzero differences. The predecessor remains failed; ordinary parameter repeat differences are descriptive and never count as passing its gate.
- Resolved/total counts beside cosines use all three frozen observed-repeat flags. Two repeats and the ten-times rule provide sampled precision context, not a confidence interval or global accuracy bound. Weak effects and angular comparisons remain unresolved. Mean-gradient arrays have no independent aggregate precision certificate.
- Shared-GPU resources cover measurement/previews through receipts, before publication; administrative raw-seal checks and report generation are excluded. No speed claim.
- Lowpass and highpass residual summaries are not an orthogonal energy partition. No regional gradient localization or intervention was performed.

Raw view rows, all six parameter groups, component statistics and source identities are retained in the companion RESULT JSON. Independent audit is separate.
