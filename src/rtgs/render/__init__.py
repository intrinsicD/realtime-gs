"""Rasterization backends behind a common protocol (torch reference / gsplat CUDA)."""

from rtgs.render.base import (
    DEFAULT_VISIBILITY_MARGIN_SIGMA,
    KernelSupportDiagnostics,
    RenderOutput,
    SHColorDiagnostics,
    get_rasterizer,
)
from rtgs.render.point_base import PointRasterizer, PointRenderOutput
from rtgs.render.projection import (
    EWA_DILATION,
    EWA_NEAR,
    EWAProjection,
    project_covariances_ewa,
    project_gaussians_ewa,
)
from rtgs.render.torch_points import TorchPointRasterizer

__all__ = [
    "DEFAULT_VISIBILITY_MARGIN_SIGMA",
    "EWA_DILATION",
    "EWA_NEAR",
    "EWAProjection",
    "KernelSupportDiagnostics",
    "PointRasterizer",
    "PointRenderOutput",
    "RenderOutput",
    "SHColorDiagnostics",
    "TorchPointRasterizer",
    "get_rasterizer",
    "project_covariances_ewa",
    "project_gaussians_ewa",
]
