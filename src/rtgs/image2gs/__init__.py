"""Stage 1: represent each input image as a set of 2D gaussians (GaussianImage-style)."""

from rtgs.image2gs.fit import FitConfig, fit_image
from rtgs.image2gs.renderer2d import render_gaussians_2d

__all__ = ["FitConfig", "fit_image", "render_gaussians_2d"]
