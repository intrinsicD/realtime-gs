"""rtgs: fast 3D Gaussian Splatting via per-image 2D gaussian fitting and 2D-to-3D lifting.

Pipeline: ``image2gs`` (stage 1, images -> 2D gaussians) -> ``lift`` (stage 2, 2D -> 3D,
three variants) -> ``optim`` (stage 3, standard 3DGS refinement). See docs/ARCHITECTURE.md.
"""

__version__ = "0.1.0"
