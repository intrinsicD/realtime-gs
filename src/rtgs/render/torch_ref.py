"""Pure-PyTorch reference rasterizer: EWA projection + depth-sorted alpha compositing.

This is the semantics-defining, fully differentiable implementation of standard 3DGS
rendering (Zwicker EWA splatting + front-to-back compositing). It is O(pixels x visible
gaussians) and meant for tests, CPU pipelines, and small scenes — the gsplat backend is
the fast path. Chunked over pixel rows to bound memory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

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
    KernelSupportDiagnostics,
    RenderOutput,
    SHColorDiagnostics,
    _validate_visibility_margin_sigma,
)
from rtgs.render.projection import EWA_DILATION, project_gaussians_ewa

_NEAR = 0.05
KERNEL_SUPPORT_CUTOFF = 12.0  # Mahalanobis^2 cutoff, matches ~3.5 sigma
KERNEL_SUPPORT_TAPER_WIDTH = 4.0
_CUTOFF = KERNEL_SUPPORT_CUTOFF
_DILATION = EWA_DILATION  # backwards-compatible private alias
_MAX_ALPHA = 0.999
KERNEL_SUPPORT_MODES = ("hard", "c1_taper", "hard_forward_c1_taper_gradient")


@dataclass(frozen=True)
class TorchRenderProgress:
    """Bounded row-chunk progress for long reference renders."""

    completed_rows: int
    total_rows: int
    visible_gaussians: int
    elapsed_seconds: float


TorchRenderProgressCallback = Callable[[TorchRenderProgress], None]


class TorchRasterizer:
    """Reference implementation of the Rasterizer protocol."""

    def __init__(
        self,
        row_chunk: int = 32,
        *,
        sh_color_activation: str = "hard",
        sh_smu1_mu: float = DEFAULT_SMU1_MU,
        collect_sh_color_diagnostics: bool = False,
        kernel_support_mode: str = "hard",
        collect_kernel_support_diagnostics: bool = False,
        visibility_margin_sigma: float = DEFAULT_VISIBILITY_MARGIN_SIGMA,
        progress_callback: TorchRenderProgressCallback | None = None,
    ):
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
        if collect_kernel_support_diagnostics and kernel_support_mode != "hard":
            raise ValueError("kernel-support diagnostics require the hard support mode")
        self.row_chunk = row_chunk
        self.sh_color_activation = sh_color_activation
        self.sh_smu1_mu = float(sh_smu1_mu)
        self.collect_sh_color_diagnostics = collect_sh_color_diagnostics
        self.kernel_support_mode = kernel_support_mode
        self.collect_kernel_support_diagnostics = collect_kernel_support_diagnostics
        self.visibility_margin_sigma = _validate_visibility_margin_sigma(visibility_margin_sigma)
        self.progress_callback = progress_callback

    def render(
        self,
        gaussians: Gaussians3D,
        camera: Camera,
        background: torch.Tensor | None = None,
        sh_degree: int | None = None,
    ) -> RenderOutput:
        """Rasterize one view. See RenderOutput for the output contract."""
        g = gaussians
        h, w = camera.height, camera.width
        device = g.means.device

        projection = project_gaussians_ewa(g, camera, dilation=_DILATION, near=_NEAR)
        z = projection.depth
        in_front = z > _NEAR
        uv_all = projection.means2d
        cov2d = projection.covariances2d

        # Visibility: in front and a configurable spectral-radius envelope intersects the image.
        eig_max = (
            0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
            + (0.25 * (cov2d[:, 0, 0] - cov2d[:, 1, 1]) ** 2 + cov2d[:, 0, 1] ** 2).sqrt()
        )
        if self.visibility_margin_sigma == DEFAULT_VISIBILITY_MARGIN_SIGMA:
            # Keep the established/default cull expression bit-exact.
            radii = 3.0 * eig_max.clamp_min(1e-8).sqrt()
        else:
            radii = self.visibility_margin_sigma * eig_max.clamp_min(1e-8).sqrt()
        visible = in_front & camera.in_image(uv_all, margin=radii.detach())
        vis_idx = visible.nonzero(as_tuple=True)[0]
        if vis_idx.numel() == 0:
            bg = (
                torch.zeros(3, device=device, dtype=g.means.dtype)
                if background is None
                else background.to(g.means)
            )
            color = bg[None, None, :].expand(h, w, 3).clone()
            zero = torch.zeros(h, w, device=device, dtype=g.means.dtype)
            return RenderOutput(
                color=color, alpha=zero, depth=zero.clone(), means2d=None, visible=vis_idx
            )

        # Depth sort (front to back).  Expanding the coarse visibility set must
        # not change the established order of primitives that were already
        # visible at the default margin, including unspecified exact-depth ties.
        if self.visibility_margin_sigma > DEFAULT_VISIBILITY_MARGIN_SIGMA:
            current_radii = 3.0 * eig_max.clamp_min(1e-8).sqrt()
            current_visible = in_front & camera.in_image(uv_all, margin=current_radii.detach())
            newly_visible = visible & ~current_visible

            vis_idx = current_visible.nonzero(as_tuple=True)[0]
            order = torch.argsort(z[vis_idx])
            vis_idx = vis_idx[order]

            new_idx = newly_visible.nonzero(as_tuple=True)[0]
            new_order = torch.argsort(z[new_idx])
            new_idx = new_idx[new_order]

            vis_idx = torch.cat((vis_idx, new_idx))
            order = torch.argsort(z[vis_idx], stable=True)
            vis_idx = vis_idx[order]
        else:
            order = torch.argsort(z[vis_idx])
            vis_idx = vis_idx[order]

        means2d = uv_all[vis_idx]
        if means2d.requires_grad:
            means2d.retain_grad()
        cov2d_v = cov2d[vis_idx]
        z_v = z[vis_idx]
        opa_v = g.opacity[vis_idx]

        # View-dependent colors from SH.
        degree = g.sh_degree if sh_degree is None else min(sh_degree, g.sh_degree)
        dirs = torch.nn.functional.normalize(g.means[vis_idx] - camera.position.to(device), dim=-1)
        preactivation = eval_sh_preactivation(degree, g.sh[vis_idx], dirs)
        colors_v = activate_sh_color(
            preactivation,
            self.sh_color_activation,
            smu1_mu=self.sh_smu1_mu,
        )  # (V,3)
        sh_color_diagnostics = None
        if self.collect_sh_color_diagnostics:
            if preactivation.requires_grad:
                preactivation.retain_grad()
            if colors_v.requires_grad:
                colors_v.retain_grad()
            sh_color_diagnostics = SHColorDiagnostics(
                preactivation=preactivation,
                activated=colors_v,
                gaussian_indices=vis_idx,
            )

        # Analytic 2x2 inverse.
        det = (cov2d_v[:, 0, 0] * cov2d_v[:, 1, 1] - cov2d_v[:, 0, 1] ** 2).clamp_min(1e-12)
        i00 = cov2d_v[:, 1, 1] / det
        i01 = -cov2d_v[:, 0, 1] / det
        i11 = cov2d_v[:, 0, 0] / det

        bg = (
            torch.zeros(3, device=device, dtype=g.means.dtype)
            if background is None
            else background.to(g.means)
        )
        xs = torch.arange(w, device=device, dtype=g.means.dtype) + 0.5

        color_rows, alpha_rows, depth_rows = [], [], []
        q_diagnostic_chunks: list[torch.Tensor] = []
        kernel_diagnostic_chunks: list[torch.Tensor] = []
        collect_kernel_diagnostics = (
            self.collect_kernel_support_diagnostics and torch.is_grad_enabled()
        )
        render_started = perf_counter() if self.progress_callback is not None else None
        for r0 in range(0, h, self.row_chunk):
            r1 = min(r0 + self.row_chunk, h)
            ys = torch.arange(r0, r1, device=device, dtype=g.means.dtype) + 0.5
            pix = torch.stack(
                [xs[None, :].expand(r1 - r0, w), ys[:, None].expand(r1 - r0, w)], dim=-1
            ).reshape(-1, 2)  # (P,2)
            d = pix[:, None, :] - means2d[None, :, :]  # (P,V,2)
            q = (
                d[..., 0] ** 2 * i00[None, :]
                + 2.0 * d[..., 0] * d[..., 1] * i01[None, :]
                + d[..., 1] ** 2 * i11[None, :]
            )
            gauss = kernel_support_weight(q, self.kernel_support_mode)
            if collect_kernel_diagnostics:
                if not q.requires_grad or not gauss.requires_grad:
                    raise RuntimeError(
                        "kernel-support diagnostics require a gradient-enabled render"
                    )
                q.retain_grad()
                gauss.retain_grad()
                q_diagnostic_chunks.append(q)
                kernel_diagnostic_chunks.append(gauss)
            alpha = (opa_v[None, :] * gauss).clamp(0.0, _MAX_ALPHA)  # (P,V) sorted near->far
            trans = torch.cumprod(1.0 - alpha + 1e-10, dim=1)
            trans = torch.cat([torch.ones_like(trans[:, :1]), trans[:, :-1]], dim=1)
            contrib = alpha * trans  # (P,V)
            color = (
                contrib @ colors_v + (trans[:, -1] * (1.0 - alpha[:, -1]))[:, None] * bg[None, :]
            )
            acc = contrib.sum(dim=1)
            depth = contrib @ z_v
            color_rows.append(color)
            alpha_rows.append(acc)
            depth_rows.append(depth)
            if self.progress_callback is not None:
                assert render_started is not None
                self.progress_callback(
                    TorchRenderProgress(
                        completed_rows=r1,
                        total_rows=h,
                        visible_gaussians=int(vis_idx.numel()),
                        elapsed_seconds=perf_counter() - render_started,
                    )
                )

        kernel_support_diagnostics = None
        if collect_kernel_diagnostics:
            kernel_support_diagnostics = KernelSupportDiagnostics(
                q_chunks=q_diagnostic_chunks,
                kernel_chunks=kernel_diagnostic_chunks,
                gaussian_indices=vis_idx,
            )
        return RenderOutput(
            color=torch.cat(color_rows).reshape(h, w, 3),
            alpha=torch.cat(alpha_rows).reshape(h, w),
            depth=torch.cat(depth_rows).reshape(h, w),
            means2d=means2d,
            visible=vis_idx,
            sh_color_diagnostics=sh_color_diagnostics,
            kernel_support_diagnostics=kernel_support_diagnostics,
        )


def _hard_kernel(q: torch.Tensor) -> torch.Tensor:
    """Established compact EWA kernel; keep the default expression bit-exact."""
    return torch.exp(-0.5 * q.clamp_max(4 * _CUTOFF)) * (q < _CUTOFF)


def _c1_taper_kernel(q: torch.Tensor) -> torch.Tensor:
    """Frozen C1 compact extension from q=12 to q=16."""
    hard = _hard_kernel(q)
    t = ((q - _CUTOFF) / KERNEL_SUPPORT_TAPER_WIDTH).clamp(0.0, 1.0)
    taper = 1.0 - 3.0 * t.square() + 2.0 * t.pow(3)
    tail = torch.exp(-0.5 * q.clamp_max(4 * _CUTOFF)) * taper
    return torch.where(
        q < _CUTOFF,
        hard,
        torch.where(q < _CUTOFF + KERNEL_SUPPORT_TAPER_WIDTH, tail, 0.0),
    )


def kernel_support_weight(q: torch.Tensor, mode: str = "hard") -> torch.Tensor:
    """Evaluate one frozen kernel-support research mode."""
    if mode == "hard":
        return _hard_kernel(q)
    smooth = _c1_taper_kernel(q)
    if mode == "c1_taper":
        return smooth
    if mode == "hard_forward_c1_taper_gradient":
        hard = _hard_kernel(q)
        return hard.detach() + (smooth - smooth.detach())
    choices = ", ".join(KERNEL_SUPPORT_MODES)
    raise ValueError(f"unknown kernel support mode '{mode}' (expected {choices})")
