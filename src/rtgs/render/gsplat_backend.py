"""CUDA rasterization via gsplat (Apache-2.0, nerfstudio-project/gsplat).

Lazy import: this module loads without gsplat installed; constructing the backend
without gsplat + CUDA raises with an actionable message. Semantics must match
``rtgs.render.torch_ref`` (parity test: tests/test_render.py::test_gsplat_parity,
marked cuda).
"""

from __future__ import annotations

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.render.base import RenderOutput


class GsplatRasterizer:
    """Rasterizer backed by gsplat.rasterization (requires CUDA)."""

    def __init__(self) -> None:
        try:
            import gsplat  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "gsplat is not installed; run `pip install -e '.[cuda]'` on a CUDA machine "
                "or use get_rasterizer('torch')"
            ) from e
        if not torch.cuda.is_available():
            raise RuntimeError("gsplat backend requires a CUDA device; use get_rasterizer('torch')")

    def render(
        self,
        gaussians: Gaussians3D,
        camera: Camera,
        background: torch.Tensor | None = None,
        sh_degree: int | None = None,
    ) -> RenderOutput:
        """Rasterize one view with gsplat; returns the same contract as torch_ref."""
        from gsplat import rasterization

        g = gaussians
        device = g.means.device
        degree = g.sh_degree if sh_degree is None else min(sh_degree, g.sh_degree)
        viewmat = camera.viewmat.to(device)[None]
        k = camera.K.to(device)[None]
        bg = None
        if background is not None:
            bg = background.to(device)[None]

        colors, alphas, meta = rasterization(
            means=g.means,
            quats=g.quats,
            scales=g.scales,
            opacities=g.opacity,
            colors=g.sh[:, : (degree + 1) ** 2, :],
            viewmats=viewmat,
            Ks=k,
            width=camera.width,
            height=camera.height,
            sh_degree=degree,
            backgrounds=bg,
            render_mode="RGB+D",
            packed=False,
        )
        color = colors[0, :, :, :3]
        depth = colors[0, :, :, 3]
        alpha = alphas[0, :, :, 0]
        means2d = meta.get("means2d")
        if means2d is not None:
            means2d = means2d[0] if means2d.ndim == 3 else means2d
        radii = meta.get("radii")
        visible = None
        if radii is not None:
            r = radii[0] if radii.ndim > 2 else radii
            visible = (
                (r > 0).any(-1).nonzero(as_tuple=True)[0]
                if r.ndim == 2
                else (r > 0).nonzero(as_tuple=True)[0]
            )
        return RenderOutput(color=color, alpha=alpha, depth=depth, means2d=means2d, visible=visible)
