# Figure sources

Produced figures land here and replace the red `\figslot` boxes in the sections. The
authoritative specification for each figure is the red box + caption at its usage site;
Appendix "Figure Production List" (`sections/12_appendix_todo.tex`) is the consolidated
index.

Quick status (see the appendix for full specs):

| Figure | File (proposed) | Producible today? |
|---|---|---|
| Fig. 1 pipeline overview + measurement boundary | `teaser.pdf` | yes (draw; thumbnails from `dataset/` captures) |
| Fig. 2 capture visualization | `capture_grid.pdf` | yes (checked-in captures + viewer tooling) |
| Fig. 3 capture compression curve | `capture_rd.pdf` | needs fit sweep (T-15) |
| Fig. 4 sample-budget sweep | `sample_budget.pdf` | needs frozen direct-path schedule (T-3) |
| Fig. 5 **headline** GPU memory vs. views | `memory_views.pdf` | needs resource harness (T-1) + fresh scene (T-4) |
| Fig. 6 tomography geometry schematic | `beam_geometry.pdf` | yes (TikZ) |
| Fig. 7 initializer comparison renders | `init_visual.pdf` | yes (development artifacts under `runs/`) |
| Fig. 8 refinement stage renders | `carrier_stages.pdf` | yes (2026-07-28 run artifacts) |
| Fig. 9 qualitative held-out comparison | `parity_grid.pdf` | needs main experiment (T-6) |
| Fig. 10 initialization risk trajectories | `time_to_quality.pdf` | needs matched comparison (T-8) |
| Fig. 11 diagnostics panel | `beam_diagnostics.pdf` | yes (run diagnostics dicts) |

Keep sources (TikZ, plotting scripts) next to the produced PDFs; plotting scripts that read
run artifacts belong in `scripts/experiments/` per repository layout policy.
