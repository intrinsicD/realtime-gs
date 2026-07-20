# Seven-view masked compact-teacher acquisition

The acquired artifact is a frozen exploratory Stage-1 input, not a passed end-to-end lifecycle.
The original command fitted and exported all seven views and atomically saved/strict-loaded the
bundle, then terminally failed a false-positive manifest substring check. A separate verifier
validated the existing bytes without refitting, rewriting, lifting, or viewing them. Independent
audit verdict: **QUALIFIED**.

| Quantity | Result |
|---|---:|
| ordered training views | C0001,C0008,C0014,C0021,C0026,C0031,C0039 |
| passed worker seeds | 0,1,2,3,4,5,6 |
| held-out C1004 | absent from fits, metrics, and payload |
| native canvas | 5328x4608 |
| components/view | 640 initial, 640 optimized |
| total compact components | 4,480 |
| geometry payload | none |
| equal-view mean training-foreground PSNR | 20.813050 dB |
| independent sampled query/raster worst max error | 6.855e-7 |
| manifest SHA-256 | 6ed60cf3df1f9ca476dfabace18b8d868d63eebd041522313c5b4fd644ee2614 |
| bundle aggregate SHA-256 | 3920f3ae05e60e1ce10a78544958b5c395109221963ff658204ce6d8c8f2efbf |

Per-view same-training-image foreground PSNRs were 21.000776, 22.638847, 21.903775,
18.998952, 23.028162, 18.595116, and 19.525726 dB. These are acquisition QA, not held-out or
novel-view metrics. The strict bundle has seven six-member NPZ archives (metadata, means,
log-scales, rotations, colors, amplitudes), `geometry:null`, and no dense RGB/mask/source-path
field. RGB/mask provenance and previews intentionally remain outside `reconstruction_inputs/`.

The plan binds all 53 selected realtime-gs sources and the exact dirty StructSplat source. Its
acquisition harness hash remains `6a11d589...`; the separate recovery verifier is `22affae1...`.
One plan field is known false: `effective_structsplat.init.seed=0` is recorded for all views while
the bound worker passed seeds 0--6. The external fit digest does not cover InitConfig. Therefore
the exact output hashes are usable as frozen identity, but fully correct effective-init,
deterministic, paired-improvement, held-out, 3D, performance, capacity, and default claims are
forbidden.

Primary evidence:

- `runs/compact_masked_bundle_640_20260717/plan.json`
- `runs/compact_masked_bundle_640_20260717/teacher_acquisition.json`
- `runs/compact_masked_bundle_640_20260717/recovery_result.json`
- `runs/compact_masked_bundle_640_20260717/AUDIT.md`
- `runs/compact_masked_bundle_640_20260717/masked_teacher_contact_sheet.png`

