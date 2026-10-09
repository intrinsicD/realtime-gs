"""2D Gaussian surfel rasterization via gsplat's ``rasterization_2dgs`` (Huang et al. 2024).

The model stays a ``Gaussians3D``: columns 0 and 1 of ``log_scales`` are the surfel's tangent
scales, column 2 is ignored by the kernel (the trainer pins it to ``-inf`` so densification and
export see a flat primitive: ``sigma_min = 0``). ``R[:, 2]`` is the surfel normal.

Besides colour/alpha/depth (the ``RenderOutput`` contract; ``depth`` is alpha-accumulated) the
backend returns the 2DGS geometry maps, all in world coordinates and camera-facing:
``normals`` (alpha-weighted rendered normals), ``normals_from_depth`` (finite-difference normals
of the rendered expected depth), ``distortion`` (gsplat's L1 depth distortion
``sum_ij w_i w_j |z_i - z_j|`` in camera-depth units) and ``median_depth``.
gsplat 1.5.3 composites each surfel's *centre* depth (not the per-pixel ray-surfel
intersection of the 2DGS paper code), so depth, depth normals and distortion are sheet-scale
quantities: an isolated tilted surfel has a fronto-parallel depth normal, a tiled sheet recovers
its tilt with a small bias toward the viewing direction (``tests/test_gsplat_2dgs.py``). The
normal term is therefore a *centre-depth consistency* proxy for the paper's depth-normal
consistency. ``surfel_terms`` turns the maps into the two regularizers as gsplat's
``examples/simple_trainer_2dgs.py`` does, except that the distortion is divided by a frozen
scene scale ``depth_scale`` (gsplat's distortion is L1 in raw camera depth, so its weight is
unit-dependent; the example's weight assumes a unit-normalized scene). gsplat's
densification gradient for 2DGS lives in ``meta["gradient_2dgs"]``; it is exposed under
``strategy_info["means2d"]`` so the unchanged ``DefaultStrategy`` (and the classic controller
via ``RenderOutput.means2d``) read it.

Lazy import: the module loads on CPU-only machines; constructing the backend needs gsplat + CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.render.base import DEFAULT_VISIBILITY_MARGIN_SIGMA, RenderOutput


@dataclass(frozen=True)
class SurfelRegularization:
    """2DGS regularizer weights; a term is active for ``step > *_start``.

    Defaults of the starts are gsplat's example (3000 / 7000); ``depth_scale`` divides the
    distortion (camera-depth units) so ``depth_distortion`` is a unit-free weight.
    """

    depth_distortion: float = 0.0  # lambda_d on mean_pixels(distortion) / depth_scale
    normal_consistency: float = 0.0  # lambda_n on mean_pixels(1 - n . n_centre_depth)
    distortion_start: int = 3000
    normal_start: int = 7000
    depth_scale: float = 1.0

    def weights(self, step: int) -> tuple[float, float]:
        return (
            self.depth_distortion if step > self.distortion_start else 0.0,
            self.normal_consistency if step > self.normal_start else 0.0,
        )


def surfel_terms(out: RenderOutput, depth_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """``(distortion / depth_scale, centre-depth normal error)`` per-pixel means of a render."""
    if out.normals is None or out.normals_from_depth is None or out.distortion is None:
        raise RuntimeError("2DGS regularizers require --rasterizer gsplat-2dgs")
    depth_normals = out.normals_from_depth * out.alpha.detach()[..., None]
    normal_error = 1.0 - (out.normals * depth_normals).sum(-1)
    return out.distortion.mean() / depth_scale, normal_error.mean()


class Gsplat2DGSRasterizer:
    """Rasterizer backed by ``gsplat.rasterization_2dgs`` (requires CUDA)."""

    def __init__(
        self,
        *,
        packed: bool = False,
        absgrad: bool = False,
        antialiased: bool = False,
        sh_color_activation: str = "hard",
        sh_smu1_mu: float | None = None,  # unused: hard SH activation only
        collect_sh_color_diagnostics: bool = False,
        kernel_support_mode: str = "hard",
        collect_kernel_support_diagnostics: bool = False,
        visibility_margin_sigma: float = DEFAULT_VISIBILITY_MARGIN_SIGMA,
    ) -> None:
        if antialiased or sh_color_activation != "hard" or kernel_support_mode != "hard":
            raise NotImplementedError("gsplat-2dgs supports classic, hard-SH, hard-kernel only")
        if collect_sh_color_diagnostics or collect_kernel_support_diagnostics:
            raise NotImplementedError("diagnostics are defined only by the torch reference")
        if visibility_margin_sigma != DEFAULT_VISIBILITY_MARGIN_SIGMA:
            raise NotImplementedError("non-default visibility margins are torch-reference only")
        try:
            import gsplat
        except ImportError as e:
            raise RuntimeError("gsplat is not installed; run `pip install -e '.[cuda]'`") from e
        if not hasattr(gsplat, "rasterization_2dgs"):
            raise RuntimeError("this gsplat does not expose rasterization_2dgs (need >= 1.4)")
        from rtgs.render.gsplat_backend import _remove_shadowed_gsplat_editable_finders

        _remove_shadowed_gsplat_editable_finders(gsplat)
        if not torch.cuda.is_available():
            raise RuntimeError("gsplat-2dgs backend requires a CUDA device")
        self.packed = packed
        self.absgrad = absgrad

    def render(
        self,
        gaussians: Gaussians3D,
        camera: Camera,
        background: torch.Tensor | None = None,
        sh_degree: int | None = None,
    ) -> RenderOutput:
        from gsplat import rasterization_2dgs

        from rtgs.render.gsplat_backend import _visible_indices

        g = gaussians
        device = g.means.device
        degree = g.sh_degree if sh_degree is None else min(sh_degree, g.sh_degree)
        colors, alphas, normals, depth_normals, distortion, median, meta = rasterization_2dgs(
            means=g.means,
            quats=g.quats,
            scales=g.scales,
            opacities=g.opacity,
            colors=g.sh[:, : (degree + 1) ** 2, :],
            viewmats=camera.viewmat.to(device)[None],
            Ks=camera.K.to(device)[None],
            width=camera.width,
            height=camera.height,
            sh_degree=degree,
            backgrounds=None if background is None else background.to(device)[None],
            render_mode="RGB+ED",
            packed=self.packed,
            absgrad=self.absgrad,
            distloss=True,
            depth_mode="expected",
        )
        alpha = alphas[0, :, :, 0]
        gradient = meta["gradient_2dgs"]
        if gradient.requires_grad:
            gradient.retain_grad()
        info = {**meta, "means2d": gradient}
        if self.packed and meta.get("gaussian_ids") is not None:
            visible = torch.unique(meta["gaussian_ids"])
        else:
            visible = _visible_indices(meta["radii"])
        return RenderOutput(
            color=colors[0, :, :, :3],
            alpha=alpha,
            depth=colors[0, :, :, 3] * alpha,  # expected -> accumulated (the base contract)
            means2d=gradient,
            visible=visible,
            strategy_info=info,
            normals=normals[0],
            normals_from_depth=depth_normals.reshape(normals[0].shape),
            distortion=distortion[0, :, :, 0],
            median_depth=median[0, :, :, 0],
        )
