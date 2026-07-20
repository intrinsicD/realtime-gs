# Dense confidence-gated initialization chain

Date: 2026-07-20
Scope: one calibrated Janelle `frame_00008` split; E1/I1 compact teachers and E2 late-release C1004

## Frozen chain

| Stage | Arms / decision surface | Result |
|---|---|---|
| E1 | balanced top-K vs dense-all+voxel-merge; exact seven-view compact-teacher metrics | Dense gained +1.971401 dB mean foreground PSNR with every view positive, but 2,319/172 = 13.4826× count failed the ≤2× gate. |
| I1 | multiplicity/cohesion/depth-sharpness/covered-view/reprojection classifier | Frozen thresholds retained exactly 60/2,319 clusters; same-view quality was exploratory. |
| E2 | top-K, dense-all, easy-only, top-K repeat; 300 gsplat-Default steps; cap 2,319; C1004 late release | C1004 foreground PSNR: dense 14.907903, easy 12.733240, top-K 11.227999, repeat 11.235098. Easy missed dense by 2.174663 dB outside the 0.007099 dB repeat envelope. |

Easy-only ended at 1,229 Gaussians versus dense-all at 2,319 and remained on a rising
validation/count trajectory at step 300. This rejects the exact frozen schedule, not all longer
or budget-filling schedules. No spatial localization tied the deficit to hard-dropped regions, so
the conditional I2/E3 correspondence branch remained closed and balanced top-K remained default.

## Performance diagnosis

- Host Linux `perf` was unavailable because `kernel.perf_event_paranoid=4`.
- One-view `cProfile`: compact teacher 2.202 s, full-frame Torch render 634.043 s, metrics 1.104 s.
- Fit-window camera rendering plus gsplat replay: 47.18 s wall and 1,811,176 KiB RSS versus the
  98m31s / 17,552,984 KiB CPU correctness-anchor run; PLY hashes were exact and aggregate metric
  drift from Torch was at most 0.003812 dB.
- Post-result invariant-target cache: 14.309 s preparation, then 7.548 s and 7.106 s repeated
  top-K evaluations; both matched the frozen GPU aggregate metrics exactly. Boolean support masks
  retained 652,517,359 bytes and the process peaked at 2,269,648 KiB RSS.
- All timings are local diagnostic observations without idle-host repeats; they are not portable
  performance claims.

## Forensic bindings

- E1 raw SHA-256: `7bf4ac973fe373c5b4cf7170877001041f46f9e63ccbf0e16a9b0d3d744f6ea6`
- I1 raw SHA-256: `9980d91536a622808acd33076c7325707385d160d5c375363cafbc24d60986c4`
- E2 raw SHA-256: `1990a5e9510e83da5a94f5d8684700149e6bba6e77bba9eee0960fef5bf91e32`
- E2 preregistration SHA-256:
  `9a7107a3314f17b514c64d7aa91d656e81535b75fc2f032d795a8547547d9f9e`
- Canonical result and scientist-audit records:
  `benchmarks/results/20260720_dense_confidence_gated_init_{e1,i1,e2}_{RESULT,AUDIT}.md`
- Implementation:
  `src/rtgs/lift/compact_confidence_gate.py`,
  `src/rtgs/lift/compact_init_eval.py`,
  `benchmarks/compact_init_eval.py`,
  `benchmarks/dense_confidence_gated_init_e2.py`
