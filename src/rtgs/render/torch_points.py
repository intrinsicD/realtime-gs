"""Pure-PyTorch sparse point rasterizer with dense-reference semantics.

The implementation intentionally duplicates the projection and visibility equations in
``torch_ref``.  It streams both query points and globally depth-sorted Gaussians, bounding
every point--Gaussian temporary while carrying front-to-back transmittance exactly across
Gaussian chunks.
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
from rtgs.render.projection import project_gaussians_ewa
from rtgs.render.torch_ref import (
    _DILATION,
    _MAX_ALPHA,
    _NEAR,
    KERNEL_SUPPORT_MODES,
    kernel_support_weight,
)


class TorchPointRasterizer:
    """CPU-first reference point rasterizer.

    Only the dense renderer's default coarse-visibility margin is supported.  Query-local
    primitive filtering would change the global compositor and is deliberately absent.
    """

    def __init__(
        self,
        point_chunk: int = 4096,
        gaussian_chunk: int = 4096,
        *,
        sh_color_activation: str = "hard",
        sh_smu1_mu: float = DEFAULT_SMU1_MU,
        kernel_support_mode: str = "hard",
        visibility_margin_sigma: float = DEFAULT_VISIBILITY_MARGIN_SIGMA,
    ) -> None:
        for name, value in (("point_chunk", point_chunk), ("gaussian_chunk", gaussian_chunk)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if sh_color_activation not in SH_COLOR_ACTIVATIONS:
            choices = ", ".join(SH_COLOR_ACTIVATIONS)
            raise ValueError(
                f"unknown SH color activation '{sh_color_activation}' (expected {choices})"
            )
        if not torch.isfinite(torch.tensor(sh_smu1_mu)) or sh_smu1_mu <= 0:
            raise ValueError("sh_smu1_mu must be finite and positive")
        if kernel_support_mode not in KERNEL_SUPPORT_MODES:
            choices = ", ".join(KERNEL_SUPPORT_MODES)
            raise ValueError(
                f"unknown kernel support mode '{kernel_support_mode}' (expected {choices})"
            )
        visibility_margin_sigma = _validate_visibility_margin_sigma(visibility_margin_sigma)
        if visibility_margin_sigma != DEFAULT_VISIBILITY_MARGIN_SIGMA:
            raise NotImplementedError(
                "TorchPointRasterizer currently supports only visibility_margin_sigma=3.0"
            )
        self.point_chunk = point_chunk
        self.gaussian_chunk = gaussian_chunk
        self.sh_color_activation = sh_color_activation
        self.sh_smu1_mu = float(sh_smu1_mu)
        self.kernel_support_mode = kernel_support_mode
        self.visibility_margin_sigma = visibility_margin_sigma

    def render_points(
        self,
        gaussians: Gaussians3D,
        camera: Camera,
        xy: torch.Tensor,
        background: torch.Tensor | None = None,
        sh_degree: int | None = None,
        collect_compositing_color_basis: bool = False,
    ) -> PointRenderOutput:
        """Evaluate ``gaussians`` at finite full-canvas coordinates ``xy``.

        Dense-render parity is defined at half-integer pixel centers. Other coordinates use
        the same continuous EWA equation and camera-wide visibility set, but do not imply a
        dense-image or CUDA parity claim.
        """
        g = gaussians
        xy = self._validate_xy(xy, g)
        device = g.means.device

        projection = project_gaussians_ewa(g, camera, dilation=_DILATION, near=_NEAR)
        z = projection.depth
        in_front = z > _NEAR
        uv_all = projection.means2d
        cov2d = projection.covariances2d

        eig_max = (
            0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
            + (0.25 * (cov2d[:, 0, 0] - cov2d[:, 1, 1]) ** 2 + cov2d[:, 0, 1] ** 2).sqrt()
        )
        radii = 3.0 * eig_max.clamp_min(1e-8).sqrt()
        visible = in_front & camera.in_image(uv_all, margin=radii.detach())
        vis_idx = visible.nonzero(as_tuple=True)[0]

        bg = (
            torch.zeros(3, device=device, dtype=g.means.dtype)
            if background is None
            else background.to(g.means)
        )
        if vis_idx.numel() == 0:
            zero = torch.zeros(xy.shape[0], device=device, dtype=g.means.dtype)
            return PointRenderOutput(
                color=bg[None, :].expand(xy.shape[0], 3).clone(),
                alpha=zero,
                depth=zero.clone(),
                means2d=None,
                visible=vis_idx,
            )

        order = torch.argsort(z[vis_idx])
        vis_idx = vis_idx[order]
        means2d = uv_all[vis_idx]
        if means2d.requires_grad:
            means2d.retain_grad()

        cov2d_v = cov2d[vis_idx]
        z_v = z[vis_idx]
        opacity_v = g.opacity[vis_idx]
        degree = g.sh_degree if sh_degree is None else min(sh_degree, g.sh_degree)
        directions = torch.nn.functional.normalize(
            g.means[vis_idx] - camera.position.to(device), dim=-1
        )
        preactivation = eval_sh_preactivation(degree, g.sh[vis_idx], directions)
        colors_v = activate_sh_color(
            preactivation,
            self.sh_color_activation,
            smu1_mu=self.sh_smu1_mu,
        )

        det = (cov2d_v[:, 0, 0] * cov2d_v[:, 1, 1] - cov2d_v[:, 0, 1] ** 2).clamp_min(1e-12)
        i00 = cov2d_v[:, 1, 1] / det
        i01 = -cov2d_v[:, 0, 1] / det
        i11 = cov2d_v[:, 0, 0] / det

        if xy.shape[0] == 0:
            return PointRenderOutput(
                color=g.means.new_empty((0, 3)),
                alpha=g.means.new_empty((0,)),
                depth=g.means.new_empty((0,)),
                means2d=means2d,
                visible=vis_idx,
            )

        color_parts: list[torch.Tensor] = []
        alpha_parts: list[torch.Tensor] = []
        depth_parts: list[torch.Tensor] = []
        for point_start in range(0, xy.shape[0], self.point_chunk):
            point_end = min(point_start + self.point_chunk, xy.shape[0])
            color, alpha, depth = self._render_chunk(
                xy[point_start:point_end],
                means2d,
                i00,
                i01,
                i11,
                opacity_v,
                colors_v,
                z_v,
                bg,
            )
            color_parts.append(color)
            alpha_parts.append(alpha)
            depth_parts.append(depth)
        return PointRenderOutput(
            color=torch.cat(color_parts),
            alpha=torch.cat(alpha_parts),
            depth=torch.cat(depth_parts),
            means2d=means2d,
            visible=vis_idx,
            compositing_color_basis=colors_v if collect_compositing_color_basis else None,
        )

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
        if not bool(torch.isfinite(xy).all()):
            raise ValueError("xy must be finite")
        return xy.to(dtype=gaussians.means.dtype)

    def _render_chunk(
        self,
        xy: torch.Tensor,
        means2d: torch.Tensor,
        i00: torch.Tensor,
        i01: torch.Tensor,
        i11: torch.Tensor,
        opacity: torch.Tensor,
        colors: torch.Tensor,
        depths: torch.Tensor,
        background: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Composite one point chunk while streaming the global Gaussian order."""
        count = xy.shape[0]
        accumulated_color = xy.new_zeros((count, 3))
        accumulated_alpha = xy.new_zeros((count,))
        accumulated_depth = xy.new_zeros((count,))
        incoming_transmittance = xy.new_ones((count,))
        terminal_background_weight: torch.Tensor | None = None

        for gaussian_start in range(0, means2d.shape[0], self.gaussian_chunk):
            gaussian_end = min(gaussian_start + self.gaussian_chunk, means2d.shape[0])
            means_chunk = means2d[gaussian_start:gaussian_end]
            dx = xy[:, None, :] - means_chunk[None, :, :]
            q = (
                dx[..., 0] ** 2 * i00[None, gaussian_start:gaussian_end]
                + 2.0 * dx[..., 0] * dx[..., 1] * i01[None, gaussian_start:gaussian_end]
                + dx[..., 1] ** 2 * i11[None, gaussian_start:gaussian_end]
            )
            gaussian_weight = kernel_support_weight(q, self.kernel_support_mode)
            alpha = (opacity[None, gaussian_start:gaussian_end] * gaussian_weight).clamp(
                0.0, _MAX_ALPHA
            )
            inclusive_local = torch.cumprod(1.0 - alpha + 1e-10, dim=1)
            exclusive_local = torch.cat(
                [torch.ones_like(inclusive_local[:, :1]), inclusive_local[:, :-1]], dim=1
            )
            transmittance = incoming_transmittance[:, None] * exclusive_local
            contribution = alpha * transmittance

            accumulated_color = (
                accumulated_color + contribution @ colors[gaussian_start:gaussian_end]
            )
            accumulated_alpha = accumulated_alpha + contribution.sum(dim=1)
            accumulated_depth = (
                accumulated_depth + contribution @ depths[gaussian_start:gaussian_end]
            )

            if gaussian_end == means2d.shape[0]:
                # The dense anchor deliberately omits +1e-10 from the terminal factor.
                terminal_background_weight = transmittance[:, -1] * (1.0 - alpha[:, -1])
            incoming_transmittance = incoming_transmittance * inclusive_local[:, -1]

        if terminal_background_weight is None:
            raise RuntimeError("non-empty visibility stream produced no terminal Gaussian")
        accumulated_color = (
            accumulated_color + terminal_background_weight[:, None] * background[None, :]
        )
        return accumulated_color, accumulated_alpha, accumulated_depth
