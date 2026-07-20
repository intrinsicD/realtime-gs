"""Shared math and containers: gaussian sets, cameras, spherical harmonics, metrics."""

from rtgs.core.camera import Camera
from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import (
    GaussianObservationField,
    GaussianObservationIndex,
    GaussianObservationIndexStats,
    GaussianPixelProposal,
    GaussianPointProposal,
    ObservationQuery,
    ObservationQueryBackend,
    ObservationSamples,
    fixed_attempt_mean,
)

__all__ = [
    "Camera",
    "GaussianPointProposal",
    "GaussianPixelProposal",
    "GaussianObservationField",
    "GaussianObservationIndex",
    "GaussianObservationIndexStats",
    "Gaussians2D",
    "Gaussians3D",
    "ObservationQuery",
    "ObservationQueryBackend",
    "ObservationSamples",
    "fixed_attempt_mean",
]
