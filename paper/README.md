# Paper draft — Compact 2D Gaussian Captures for Memory-Efficient 3DGS

LaTeX working draft implementing `docs/PAPER_PLAN_beam_fusion.md` as a full paper skeleton.

Storyline (one thread, owner decision 2026-07-30):

1. Per-view 2D Gaussian fields replace the images (`sections/04_fields.tex`).
2. A standard 3DGS optimizer reconstructs from the fields alone, initialized from random
   points or SfM (`05_reconstruction.tex`).
3. Memory is measured under a controlled protocol (`06_memory.tex`).
4. Rendering projects 3D Gaussians to 2D Gaussians, so initialization can also be obtained
   by inverting the projection — the tomographic initializer Beam Fusion
   (`07_tomography.tex`).
5. The naive inversion fails in identifiable ways; the carrier refinement addresses them
   (`08_refinement.tex`).
6. Experiments show the full pipeline and end with the ablation study
   (`09_experiments.tex`).

The reconstruction result is established with the standard initializations alone, so a weak
tomographic-initializer outcome cannot weaken it (evaluation rules in `03_overview.tex`).

## Draft conventions

- **Black text** is explained and evidence-bound. Statements backed by repository artifacts
  carry gray `[evidence …]` notes pointing at `ara/logic/claims.md` rows, ADRs, or sealed
  results under `benchmarks/results/`. Single-scene development evidence is always labelled
  as such in the text.
- **Red text** is a TODO and always starts with `TODO:` (`\todo{…}`, `TodoBlock`
  environments, `\tbd` table cells). Red framed boxes inside figure floats are figure slots
  specifying exactly which figure is needed and from which data.
- Appendix A (`sections/12_appendix_todo.tex`) tracks every red item as TODOs T-1 … T-15
  plus the consolidated figure production list.
- Style rules for the running text (owner decision 2026-07-30): no em dashes, no
  semicolons, no colons (the `TODO:` prefix is the one exception), no marketing vocabulary,
  and negative results appear only where an obvious approach does not apply and an
  alternative is given. Citation titles in `references.bib` keep their original punctuation.
- `references.bib` is reconstructed from memory and must be verified before submission
  (T-13), as must the literature gaps N1–N5 in the related-work section.

Toggle the gray evidence notes off with `\evidencenotesfalse` in `preamble.tex` for a clean
read.

## Build

Requires a standard TeX Live (pdflatex + bibtex + natbib/cleveref). No LaTeX distribution is
installed in the repository CI/dev containers; build locally:

```bash
cd paper
make            # latexmk if available, else pdflatex+bibtex sequence
make clean
```

## Editing rules (repository policy)

- The draft must never promote a claim beyond its `ara/logic/claims.md` status: turning a
  red passage black requires the artifact, an independent audit, and a ledger row first
  (Hard Rules 8–9 in `CLAUDE.md`).
- Numbers quoted in black are copied from audited artifacts; if a result is re-run, update
  the number and its evidence note together.
- The figure slots name their data sources; produced figures go to `figures/` and replace
  the `\figslot` box in place.
