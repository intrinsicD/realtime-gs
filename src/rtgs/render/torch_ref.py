"""Pure-PyTorch reference rasterizer: EWA projection + depth-sorted alpha compositing.

This is the semantics-defining, fully differentiable implementation of standard 3DGS
rendering (Zwicker EWA splatting + front-to-back compositing). It is O(pixels x visible
gaussians) and meant for tests, CPU pipelines, and small scenes — the gsplat backend is
the fast path. Chunked over pixel rows to bound memory.
"""

from __future__ import annotations

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.sh import eval_sh
from rtgs.render.base import RenderOutput

_NEAR = 0.05
_CUTOFF = 12.0  # Mahalanobis^2 cutoff, matches ~3.5 sigma
_DILATION = 0.3  # screen-space low-pass (pixel^2), as in 3DGS/gsplat 'classic' mode
_MAX_ALPHA = 0.999


class TorchRasterizer:
    """Reference implementation of the Rasterizer protocol."""

    def __init__(self, row_chunk: int = 32):
        self.row_chunk = row_chunk

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

        means_cam = camera.world_to_cam(g.means)  # (N,3)
        z = means_cam[:, 2]
        in_front = z > _NEAR

        # Project centers.
        uv_all, _ = camera.project(g.means)

        # EWA: Sigma_2D = J R Sigma R^T J^T (+ dilation)
        cov_world = g.covariance()
        r_wc = camera.R.to(device)
        cov_cam = r_wc @ cov_world @ r_wc.T
        zs = z.clamp_min(_NEAR)
        jac = torch.zeros(g.n, 2, 3, device=device, dtype=torch.float32)
        jac[:, 0, 0] = camera.fx / zs
        jac[:, 0, 2] = -camera.fx * means_cam[:, 0] / zs**2
        jac[:, 1, 1] = camera.fy / zs
        jac[:, 1, 2] = -camera.fy * means_cam[:, 1] / zs**2
        cov2d = jac @ cov_cam @ jac.transpose(-1, -2)
        cov2d = cov2d + _DILATION * torch.eye(2, device=device)

        # Visibility: in front of camera and 3-sigma footprint intersects the image.
        eig_max = (
            0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
            + (0.25 * (cov2d[:, 0, 0] - cov2d[:, 1, 1]) ** 2 + cov2d[:, 0, 1] ** 2).sqrt()
        )
        radii = 3.0 * eig_max.clamp_min(1e-8).sqrt()
        visible = in_front & camera.in_image(uv_all, margin=radii.detach())
        vis_idx = visible.nonzero(as_tuple=True)[0]
        if vis_idx.numel() == 0:
            bg = torch.zeros(3, device=device) if background is None else background
            color = bg[None, None, :].expand(h, w, 3).clone()
            zero = torch.zeros(h, w, device=device)
            return RenderOutput(
                color=color, alpha=zero, depth=zero.clone(), means2d=None, visible=vis_idx
            )

        # Depth sort (front to back).
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
        colors_v = eval_sh(degree, g.sh[vis_idx], dirs)  # (V,3)

        # Analytic 2x2 inverse.
        det = (cov2d_v[:, 0, 0] * cov2d_v[:, 1, 1] - cov2d_v[:, 0, 1] ** 2).clamp_min(1e-12)
        i00 = cov2d_v[:, 1, 1] / det
        i01 = -cov2d_v[:, 0, 1] / det
        i11 = cov2d_v[:, 0, 0] / det

        bg = torch.zeros(3, device=device) if background is None else background
        xs = torch.arange(w, device=device, dtype=torch.float32) + 0.5

        color_rows, alpha_rows, depth_rows = [], [], []
        for r0 in range(0, h, self.row_chunk):
            r1 = min(r0 + self.row_chunk, h)
            ys = torch.arange(r0, r1, device=device, dtype=torch.float32) + 0.5
            pix = torch.stack(
                [xs[None, :].expand(r1 - r0, w), ys[:, None].expand(r1 - r0, w)], dim=-1
            ).reshape(-1, 2)  # (P,2)
            d = pix[:, None, :] - means2d[None, :, :]  # (P,V,2)
            q = (
                d[..., 0] ** 2 * i00[None, :]
                + 2.0 * d[..., 0] * d[..., 1] * i01[None, :]
                + d[..., 1] ** 2 * i11[None, :]
            )
            gauss = torch.exp(-0.5 * q.clamp_max(4 * _CUTOFF)) * (q < _CUTOFF)
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

        return RenderOutput(
            color=torch.cat(color_rows).reshape(h, w, 3),
            alpha=torch.cat(alpha_rows).reshape(h, w),
            depth=torch.cat(depth_rows).reshape(h, w),
            means2d=means2d,
            visible=vis_idx,
        )
