"""Sparse point-rasterization contract.

Unlike :class:`rtgs.render.base.Rasterizer`, this surface evaluates one camera at an
explicit flat list of image coordinates rather than materializing a complete image.  The
pure-Torch implementation is the initial correctness anchor; accelerated implementations
must preserve its global visibility, depth ordering, and compositing semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D


@dataclass
class PointRenderOutput:
    """Result of rasterizing one camera at ``S`` explicit pixel coordinates.

    ``depth`` is alpha-weighted accumulated camera-space depth. ``means2d`` is the
    projected-center tensor used by autograd; its retained gradient is the screen-space
    signal consumed by density-control research. ``visible`` maps those rows back to the
    input Gaussian set and is ordered front to back. ``compositing_color_basis`` is
    default-off; when requested it is the activated, front-to-back color tensor consumed
    by compositing, exposed for a single vector-Jacobian product without materializing
    point-by-Gaussian weights.
    """

    color: torch.Tensor  # (S,3)
    alpha: torch.Tensor  # (S,)
    depth: torch.Tensor  # (S,)
    visible: torch.Tensor | None = None  # (V,), front-to-back input-gaussian indices
    means2d: torch.Tensor | None = None  # (V,2), retained screen centers
    compositing_color_basis: torch.Tensor | None = None  # (V,3), opt-in activated colors


class PointRasterizer(Protocol):
    """Evaluates a 3D Gaussian set at explicit full-canvas image coordinates."""

    def render_points(
        self,
        gaussians: Gaussians3D,
        camera: Camera,
        xy: torch.Tensor,
        background: torch.Tensor | None = None,
        sh_degree: int | None = None,
        collect_compositing_color_basis: bool = False,
    ) -> PointRenderOutput:
        """Rasterize one camera at coordinates shaped ``(S,2)`` in ``(u,v)`` order."""
        ...
