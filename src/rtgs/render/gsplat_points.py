"""CUDA point rasterization through batched one-pixel gsplat cameras.

Each requested full-canvas coordinate becomes a one-pixel camera with an intrinsics shift that
maps that coordinate to the pixel center ``(0.5, 0.5)``. All micro-cameras share the original
extrinsics, focal lengths, and view-dependent SH direction. gsplat therefore performs the same
front-to-back 3D Gaussian compositing without materializing the native-resolution image.

The packed projection exposes one screen-space row per local ``(query, Gaussian)`` overlap.
``density_gaussian_ids`` maps those rows back to physical Gaussian rows; the density controller
reduces their gradients before applying the established clone/split/prune policy.
"""

from __future__ import annotations

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.sh import (
    DEFAULT_SMU1_MU,
    SH_COLOR_ACTIVATIONS,
    activate_sh_color,
    eval_sh_preactivation,
)
from rtgs.render.base import (
    DEFAULT_VISIBILITY_MARGIN_SIGMA,
    _validate_visibility_margin_sigma,
)
from rtgs.render.point_base import PointRenderOutput
from rtgs.render.projection import EWA_DILATION, EWA_NEAR, project_gaussians_ewa
from rtgs.render.torch_ref import KERNEL_SUPPORT_MODES


class GsplatPointRasterizer:
    """Accelerated CUDA point rasterizer backed by packed gsplat micro-cameras."""

    def __init__(
        self,
        *,
        absgrad: bool = False,
        antialiased: bool = False,
        sh_color_activation: str = "hard",
        sh_smu1_mu: float = DEFAULT_SMU1_MU,
        kernel_support_mode: str = "hard",
        visibility_margin_sigma: float = DEFAULT_VISIBILITY_MARGIN_SIGMA,
    ) -> None:
        visibility_margin_sigma = _validate_visibility_margin_sigma(visibility_margin_sigma)
        if visibility_margin_sigma != DEFAULT_VISIBILITY_MARGIN_SIGMA:
            raise NotImplementedError(
                "GsplatPointRasterizer supports only visibility_margin_sigma=3.0"
            )
        if kernel_support_mode not in KERNEL_SUPPORT_MODES:
            choices = ", ".join(KERNEL_SUPPORT_MODES)
            raise ValueError(
                f"unknown kernel support mode '{kernel_support_mode}' (expected {choices})"
            )
        if kernel_support_mode != "hard":
            raise NotImplementedError(
                "non-hard point-kernel support is defined only by the torch reference"
            )
        if sh_color_activation not in SH_COLOR_ACTIVATIONS:
            choices = ", ".join(SH_COLOR_ACTIVATIONS)
            raise ValueError(
                f"unknown SH color activation '{sh_color_activation}' (expected {choices})"
            )
        if not torch.isfinite(torch.tensor(sh_smu1_mu)) or sh_smu1_mu <= 0:
            raise ValueError("sh_smu1_mu must be finite and positive")
        self.absgrad = bool(absgrad)
        self.antialiased = bool(antialiased)
        self.sh_color_activation = sh_color_activation
        self.sh_smu1_mu = float(sh_smu1_mu)
        self.visibility_margin_sigma = visibility_margin_sigma

    @staticmethod
    def _validate_xy(xy: torch.Tensor, gaussians: Gaussians3D) -> torch.Tensor:
        if not isinstance(xy, torch.Tensor):
            raise TypeError("xy must be a torch.Tensor")
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (S,2)")
        if not xy.is_floating_point():
            raise TypeError("xy must be floating point")
        if xy.device != gaussians.means.device:
            raise ValueError("xy and gaussians must be on the same device")
        if xy.requires_grad:
            raise ValueError("GsplatPointRasterizer treats query coordinates as fixed samples")
        if not bool(torch.isfinite(xy).all()):
            raise ValueError("xy must be finite")
        return xy.to(dtype=gaussians.means.dtype)

    @staticmethod
    def _global_visible(gaussians: Gaussians3D, camera: Camera) -> torch.Tensor:
        """Match the torch point anchor's camera-wide coarse visibility and depth order."""

        with torch.no_grad():
            projection = project_gaussians_ewa(
                gaussians,
                camera,
                dilation=EWA_DILATION,
                near=EWA_NEAR,
            )
            covariance = projection.covariances2d
            eig_max = (
                0.5 * (covariance[:, 0, 0] + covariance[:, 1, 1])
                + (
                    0.25 * (covariance[:, 0, 0] - covariance[:, 1, 1]).square()
                    + covariance[:, 0, 1].square()
                ).sqrt()
            )
            radii = 3.0 * eig_max.clamp_min(1e-8).sqrt()
            visible = (projection.depth > EWA_NEAR) & camera.in_image(
                projection.means2d,
                margin=radii,
            )
            rows = visible.nonzero(as_tuple=True)[0]
            return rows[torch.argsort(projection.depth[rows])]

    def render_points(
        self,
        gaussians: Gaussians3D,
        camera: Camera,
        xy: torch.Tensor,
        background: torch.Tensor | None = None,
        sh_degree: int | None = None,
        collect_compositing_color_basis: bool = False,
    ) -> PointRenderOutput:
        """Rasterize explicit full-canvas points without a dense image allocation."""

        if collect_compositing_color_basis:
            raise NotImplementedError(
                "GsplatPointRasterizer does not expose the optional compositing color basis"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("GsplatPointRasterizer requires CUDA")
        if gaussians.means.device.type != "cuda":
            raise ValueError("GsplatPointRasterizer requires CUDA-resident Gaussians")
        if gaussians.means.dtype != torch.float32:
            raise ValueError("GsplatPointRasterizer supports float32 Gaussians")
        xy = self._validate_xy(xy, gaussians)
        visible = self._global_visible(gaussians, camera)
        if xy.shape[0] == 0:
            empty = gaussians.means.new_empty
            return PointRenderOutput(
                color=empty((0, 3)),
                alpha=empty((0,)),
                depth=empty((0,)),
                visible=visible,
            )

        try:
            import gsplat
        except ImportError as error:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "GsplatPointRasterizer requires gsplat>=1.4; install the CUDA extra"
            ) from error
        if not hasattr(gsplat, "rasterization"):
            raise RuntimeError("the imported gsplat package has no rasterization API")

        count = int(xy.shape[0])
        viewmats = camera.viewmat.to(gaussians.means)[None].expand(count, -1, -1).contiguous()
        intrinsics = camera.K.to(gaussians.means)[None].expand(count, -1, -1).clone()
        intrinsics[:, 0, 2] += 0.5 - xy[:, 0]
        intrinsics[:, 1, 2] += 0.5 - xy[:, 1]

        degree = (
            gaussians.sh_degree if sh_degree is None else min(int(sh_degree), gaussians.sh_degree)
        )
        if self.sh_color_activation == "hard":
            colors_arg = gaussians.sh[:, : (degree + 1) ** 2, :]
            raster_sh_degree: int | None = degree
        else:
            directions = torch.nn.functional.normalize(
                gaussians.means - camera.position.to(gaussians.means),
                dim=-1,
            )
            preactivation = eval_sh_preactivation(
                degree,
                gaussians.sh,
                directions,
            )
            colors_arg = activate_sh_color(
                preactivation,
                self.sh_color_activation,
                smu1_mu=self.sh_smu1_mu,
            )
            raster_sh_degree = None

        rendered, alphas, metadata = gsplat.rasterization(
            means=gaussians.means,
            quats=gaussians.quats,
            scales=gaussians.scales,
            opacities=gaussians.opacity,
            colors=colors_arg,
            viewmats=viewmats,
            Ks=intrinsics,
            width=1,
            height=1,
            near_plane=EWA_NEAR,
            eps2d=EWA_DILATION,
            sh_degree=raster_sh_degree,
            backgrounds=None,
            render_mode="RGB+D",
            packed=True,
            absgrad=self.absgrad,
            rasterize_mode="antialiased" if self.antialiased else "classic",
        )
        color = rendered[:, 0, 0, :3]
        alpha = alphas[:, 0, 0, 0]
        depth = rendered[:, 0, 0, 3]
        if background is not None:
            bg = background.to(gaussians.means)
            if bg.shape != (3,):
                raise ValueError("background must have shape (3,)")
            color = color + (1.0 - alpha)[:, None] * bg[None, :]

        means2d = metadata.get("means2d")
        gaussian_ids = metadata.get("gaussian_ids")
        if means2d is None or gaussian_ids is None:
            raise RuntimeError("packed gsplat point render omitted density metadata")
        if means2d.shape != (gaussian_ids.numel(), 2):
            raise RuntimeError("packed gsplat point metadata is inconsistent")
        if means2d.requires_grad:
            means2d.retain_grad()
        return PointRenderOutput(
            color=color,
            alpha=alpha,
            depth=depth,
            visible=visible,
            means2d=means2d,
            density_gaussian_ids=gaussian_ids,
        )


__all__ = ["GsplatPointRasterizer"]
