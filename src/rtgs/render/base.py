"""Rasterizer protocol and backend registry.

The reference implementation (`torch_ref`) defines the semantics; fast backends
(`gsplat`) must match it (parity tests marked `cuda`). Pipeline code only ever talks to
this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D


@dataclass
class RenderOutput:
    """Result of rasterizing one view.

    ``depth`` is alpha-weighted accumulated depth (divide by ``alpha`` for expected
    depth). ``means2d`` is the projected-center tensor participating in the autograd
    graph — after ``loss.backward()`` its ``.grad`` holds screen-space positional
    gradients (the densification signal); ``visible`` maps its rows to gaussian indices.
    """

    color: torch.Tensor  # (H, W, 3)
    alpha: torch.Tensor  # (H, W)
    depth: torch.Tensor  # (H, W)
    means2d: torch.Tensor | None = None  # (V,2) or (1,N,2), retained screen centers
    visible: torch.Tensor | None = None  # (V,) indices into the input gaussian set


class Rasterizer(Protocol):
    """Renders a 3D gaussian set from a camera."""

    def render(
        self,
        gaussians: Gaussians3D,
        camera: Camera,
        background: torch.Tensor | None = None,
        sh_degree: int | None = None,
    ) -> RenderOutput:
        """Rasterize; ``sh_degree`` limits SH evaluation (None = use all bands)."""
        ...


def get_rasterizer(name: str = "auto", device: torch.device | str | None = None) -> Rasterizer:
    """Return a rasterizer backend: 'torch' (reference), 'gsplat' (CUDA), or 'auto'.

    'auto' picks gsplat when both the package and a CUDA device are available,
    otherwise the reference implementation. Supplying ``device`` also prevents a
    CUDA-capable host from selecting gsplat for explicitly CPU-resident data.
    """
    if name == "auto":
        requested = None if device is None else torch.device(device)
        wants_cuda = requested is None or requested.type == "cuda"
        name = "gsplat" if wants_cuda and _gsplat_available() else "torch"
    if name == "torch":
        from rtgs.render.torch_ref import TorchRasterizer

        return TorchRasterizer()
    if name == "gsplat":
        from rtgs.render.gsplat_backend import GsplatRasterizer

        return GsplatRasterizer()
    raise ValueError(f"unknown rasterizer '{name}' (expected 'auto', 'torch' or 'gsplat')")


def _gsplat_available() -> bool:
    try:
        import gsplat  # noqa: F401
        import torch as _torch

        return _torch.cuda.is_available()
    except ImportError:
        return False
