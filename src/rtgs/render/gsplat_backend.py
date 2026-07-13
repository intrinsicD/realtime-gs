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
            absgrad=True,
        )
        color = colors[0, :, :, :3]
        depth = colors[0, :, :, 3]
        alpha = alphas[0, :, :, 0]
        means2d = meta.get("means2d")
        # Keep gsplat's original tensor: the renderer graph uses this leaf directly, while a
        # post-hoc ``means2d[0]`` view does not receive ``.grad``.  DensityController handles
        # the leading camera dimension.
        if means2d is not None and means2d.requires_grad:
            means2d.retain_grad()
        radii = meta.get("radii")
        visible = None if radii is None else _visible_indices(radii)
        return RenderOutput(color=color, alpha=alpha, depth=depth, means2d=means2d, visible=visible)


def _visible_indices(radii: torch.Tensor) -> torch.Tensor:
    """Normalize gsplat's scalar/axis radius layouts to Gaussian indices."""
    # gsplat versions expose either [C,N] scalar radii or [C,N,2] axis radii.
    r = radii[0] if radii.ndim >= 2 and radii.shape[0] == 1 else radii
    visible = (r > 0).all(-1) if r.ndim == 2 else r > 0
    return visible.nonzero(as_tuple=True)[0]
