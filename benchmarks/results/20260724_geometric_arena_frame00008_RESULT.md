# Geometric Stage-3 arena — Janelle `frame_00008`

This note reports the frozen single-scene storage experiment. The adjacent independent audit controls the final interpretation.

| arm | final N | native 10k s | density events s | peak allocated MiB | held-out FG PSNR | α-IoU |
|---|---:|---:|---:|---:|---:|---:|
| dynamic-a | 5424 | 39.887230 | 0.048930 | 45.793 | 22.314209 | 0.953317 |
| geometric | 5395 | 40.605433 | 0.057516 | 48.032 | 22.344547 | 0.956855 |
| dynamic-b | 5337 | 41.521786 | 0.055348 | 45.688 | 22.134048 | 0.954818 |

## Frozen decision reduction

- Disposition: `REJECT_CURRENT_ARENA_CORRECTNESS`
- Storage correctness: `False`
- Density-event material win: `False`
- End-to-end win: `False`
- Default change authorized: `False`

## Evidence boundary

One scene, one seed, one device, and two dynamic timing controls do not establish a general performance claim. Held-out C1004 was reporting-only. The arena currently covers `gsplat-default`; it does not test fixed-budget MCMC relocation.

- Results page: `runs/geometric_arena_frame00008_20260724/index.html`
- Viewer manifest: `benchmarks/results/20260724_geometric_arena_frame00008_VIEWER.json`
