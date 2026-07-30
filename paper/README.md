# Paper draft — Compact 2D Gaussian Captures for Memory-Efficient 3DGS

LaTeX working draft implementing `docs/PAPER_PLAN_beam_fusion.md` as a full paper skeleton.

Two claim-separated parts, per the plan's firewall rule:

- **Part I** (`sections/04–07`): the systems / VRAM claim — image-free compact-field
  supervision, the memory-measurement protocol, and the (pending) controlled experiment.
- **Part II** (`sections/08–10`): Beam Fusion — tomographic initialization, carrier
  refinement, and the (pending) matched Beam-versus-no-Beam evaluation.

## Draft conventions

- **Black text** is explained and evidence-bound. Statements backed by repository artifacts
  carry gray `[evidence: …]` notes pointing at `ara/logic/claims.md` rows, ADRs, or sealed
  results under `benchmarks/results/`. Single-scene development evidence is always labelled
  as such in the text.
- **Red text** (`\toshow{…}`, `ToShowBlock` environments) marks everything that still has to
  be shown, measured, frozen, or verified. Red framed boxes inside figure floats are figure
  slots specifying exactly which figure/image is needed and from which data.
- Appendix A (`sections/13_appendix_obligations.tex`) tracks every red item as obligations
  O-1 … O-15 plus the consolidated figure production list.
- `references.bib` is reconstructed from memory and must be verified before submission
  (obligation O-13); the same goes for the novelty ledger N1–N5 in the related-work section.

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

- The draft must never promote a claim beyond its `ara/logic/claims.md` status: turning a red
  passage black requires the artifact, an independent audit, and a ledger row first
  (Hard Rules 8–9 in `CLAUDE.md`).
- Numbers quoted in black are copied from audited artifacts; if a result is re-run, update
  the number and its evidence note together.
- The figure slots name their data sources; produced figures go to `figures/` and replace the
  `\figplaceholder` box in place.
