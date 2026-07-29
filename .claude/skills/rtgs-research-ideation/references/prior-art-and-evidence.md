# Prior-art and evidence audit

## Search procedure

Use current primary sources when novelty or state-of-the-art context matters:

- papers and official supplements;
- authors' project pages and repositories;
- theses, patents, and workshop material when relevant;
- older terminology and mathematical formulations;
- recipient, donor, and bridge-field searches for transfers.

Record queries/sources and a cutoff date. Search exact names, synonyms, functional descriptions,
equations/operators, and claimed failure modes. Inspect the closest work rather than relying on
titles or abstracts.

For each candidate, report facet-level overlap:

- same problem;
- same primitives;
- same mechanism;
- same objective;
- same evidence program;
- same prediction;
- what irreducible delta remains.

Classify as likely known, known components/new relationship, apparently unexplored, apparently
transformational, or insufficient evidence. State the strongest threat and uncertainty.

## New-evidence programs

A useful discovery experiment can create observations not present in the initial corpus. Specify:

- varied factor and controlled factors;
- observable and why it identifies the mechanism;
- surprising outcome that would change the model;
- strongest conventional explanation;
- negative/shuffled/ablated control;
- leakage and implementation-bug checks;
- raw artifact needed for independent recomputation.

Do not let an easy metric stand in for the actual claim. In this repository:

- fitted-view PSNR does not establish held-out geometry;
- a viewer image does not replace exact rasterizer metrics;
- CPU green does not establish CUDA behavior or throughput;
- synthetic correctness does not establish calibrated utility;
- a local mechanism delta does not establish end-to-end materiality;
- a post-run threshold cannot become a preregistered gate.

Design the evidence ladder before implementing the expensive path.
