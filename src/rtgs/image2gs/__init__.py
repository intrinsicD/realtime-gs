"""Stage 1: represent each input image as a set of 2D gaussians (GaussianImage-style)."""

from rtgs.image2gs.adapters import load_gaussians2d
from rtgs.image2gs.fit import FitConfig, fit_image, fit_views
from rtgs.image2gs.native_observation import (
    native_gaussians_to_observation,
    native_observation_to_gaussians,
    render_native_observation_crop,
)
from rtgs.image2gs.renderer2d import render_gaussians_2d

__all__ = [
    "FitConfig",
    "fit_image",
    "fit_views",
    "load_gaussians2d",
    "native_gaussians_to_observation",
    "native_observation_to_gaussians",
    "render_gaussians_2d",
    "render_native_observation_crop",
]
