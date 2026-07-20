# Concepts

## K01: Global isotropic ray sigma
- **Definition**: One world-space along-ray standard deviation held constant across all lifted
  observations. It is distinct from the depth- and Gaussian-size-dependent minor-footprint floor.
- **Provenance**: ai-suggested
- **Crystallized via**: artifact-commitment
- **Evidence**: [`src/rtgs/lift/depth.py`, `benchmarks/depth_covariance_ablation.py`]
- **From staging**: O02

## K02: Domain-qualified Gaussian cardinalities
- **Definition**: Use initialization/optimized status as a subscript and representation domain as
  a superscript. For view `i`, `N_init,i^2D -> N_opt,i^2D` denotes adaptive 2D fitting; the compact
  multi-view collection then yields `N_init^3D -> N_opt^3D` through RGB-free initialization and
  sampled 3D refinement. The view index may be omitted only for an explicitly shared 2D count.
- **Provenance**: user-revised
- **Crystallized via**: definition-usage
- **Evidence**: [N107, N109, N111]
- **From staging**: O89
