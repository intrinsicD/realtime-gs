# Audit — 20261007_jet_consistency_prior_tosca_cat0

Self-audit by the task owner; **not an independent audit** (none was available in this session).

- Recomputed the per-seed gate from `runs/20261007_jet_consistency_prior_tosca_cat0/comparison.json`: jet2 and jet1 lose on the normal
  median (−22.8 to −27.1°) and the centre median (×2.76–3.14) in every seed → PREREG reject.
- Geometry evaluator validated before training on the companion model `data/cat_point_cloud.ply`
  (reproduced 16.52° median, 49.45° p90, 0.0279 splat units, Zhou–Lähner 38.21 %).
- `run_receipt.json` completed; data seal checked at entry and exit (`input_integrity_*.json`);
  held-out views and the mesh were guarded out of fitting (audit hook, receipts per cell).
- Preview of the selected base/9561 model inspected: the reconstruction matches the references.
- The neighbour-collapse explanation in the RESULT is a post-hoc diagnostic on seed 9561 models.
- Open: SplatDiffuseLBO operator metric (deferred); an independent audit.
