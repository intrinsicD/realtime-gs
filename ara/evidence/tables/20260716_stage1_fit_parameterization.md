# N78 Stage-1 fit-time appearance parameterization

## Scope and provenance

The once-only deterministic CPU-synthetic comparison used three appearance-only seeds, three
joint-fit seeds, nine selected source views per seed, 150 fixed components, and 120 Adam updates.
It compared the current `weight_color_9p` fit with the bounded `unit_weight_bounded_8p` arm from a
common forward initialization. It did not test real images, held-out/novel views, Stage 2, Stage 3,
CUDA, throughput, memory, or compression.

| Artifact | SHA-256 |
|---|---|
| Preregistration | `d1440fde596667fd59e996113dd4ffa4414e23e8c783a401343d7476f00afb22` |
| Implementation review | `fd3721787296768beb866b73fe603c66351cb934b27ecbfb8dc30498cb96145d` |
| Seal | `6aeee81f97409f0dfdfdd6af84f7e26970a09f1b79b35eaaa0acfbfcf25a33a0` |
| Once-only attempt | `edaf0f8a05ea8677ebf077004e1e231de126e27e7fd90a699dbaa92dce17d867` |
| Result JSON | `041873c09c949a47d12c6ae05553d9722a9756050c12478855d55dd45f9c4315` |
| Raw NPZ | `028c93f350b30b61debebd5bf0706ff128f2c54faaee04614d1ee12191a3aeb7` |
| Raw collection | `383e8372d81c78f263111126071ad6b55fd617a70f139fc456401536fae4e352` |
| Human scientist audit | `cab58480120a965b8dfaeafb5eb2273a93ccd43f0c686752f6e7c296fdf2d99c` |
| Machine scientist review | `10bd2fbbfe3474fa1d50a9ffdd39fa4e7a90320c238f005c0fdfa18ccd0357ae` |

The independent review rehashed all 360 finite raw arrays, regenerated all 54 source targets and
864 checkpoint renders, independently reconstructed the Adam/chain/SVD/null/saturation equations,
and reproduced all frozen decisions. The exact verdict is `pass`; default change is unauthorized.

## Decisive results

All deltas are bounded-8p minus current-9p, with the seed as replicate.

| Block | Mean PSNR AUC delta | Mean final PSNR delta | Mean final SSIM delta |
|---|---:|---:|---:|
| Appearance-only | -1.330662 dB | -1.796120 dB | -0.037330 |
| Joint fit | -1.292971 dB | -1.501525 dB | -0.048419 |

| Frozen decision or diagnostic | Value |
|---|---:|
| `appearance_curve_improved` | false |
| Global null-energy ratio | 0.1229208797 |
| Null-large fraction | 0.9295082305 |
| `null_update_material` | true |
| Current/candidate global weak rows | 0 / 0 of 32,400 per arm |
| `candidate_saturation_guard_passed` | true |
| `fit_time_redundant_coordinate_interference_consistent` | false |
| `joint_stage1_noninferior` | false |
| `joint_stage1_material_improvement` | false |

## Disposition

Material local motion along the current parameterization's exact null direction is real, but it
did not predict a benefit from removing that coordinate: the bounded 8p arm lost every
appearance-only and joint seed. Retain the current 9p default and close only this exact bounded
candidate in the frozen scope without tuning. The result does not establish globally wasted
optimization and does not answer the separate variable-projection question.

