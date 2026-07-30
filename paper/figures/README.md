# Figure sources

Produced figures land here and replace the red `\figplaceholder` boxes in the sections.
The authoritative specification for each figure is the red box + caption at its usage site;
Appendix "Figure Production List" (`sections/13_appendix_obligations.tex`) is the
consolidated index.

Quick status (see the appendix for full specs):

| Figure | File (proposed) | Producible today? |
|---|---|---|
| Fig. 1 teaser / pipeline + claim boundary | `teaser.pdf` | yes (draw; thumbnails from `dataset/` captures) |
| Fig. 2 capture visualization | `capture_grid.pdf` | yes (checked-in captures + viewer tooling) |
| Fig. 3 capture rate–distortion | `capture_rd.pdf` | needs Stage-1 sweep |
| Fig. 4 sample-budget sweep | `sample_budget.pdf` | needs frozen Arm-1 schedule |
| Fig. 5 **headline** VRAM vs. views | `vram_views.pdf` | needs resource harness (O-1) + fresh scene (O-4) |
| Fig. 6 qualitative parity grid | `parity_grid.pdf` | needs main experiment (O-6) |
| Fig. 7 Beam geometry schematic | `beam_geometry.pdf` | yes (TikZ) |
| Fig. 8 initializer comparison renders | `init_visual.pdf` | yes (development artifacts under `runs/`) |
| Fig. 9 carrier stage renders | `carrier_stages.pdf` | yes (2026-07-28 run artifacts) |
| Fig. 10 time-to-quality curves | `time_to_quality.pdf` | needs matched comparison (O-8) |
| Fig. 11 Beam diagnostics panel | `beam_diagnostics.pdf` | yes (run diagnostics dicts) |

Keep sources (TikZ, plotting scripts) next to the produced PDFs; plotting scripts that read
run artifacts belong in `scripts/experiments/` per repository layout policy.
