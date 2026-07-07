"""Real spherical harmonics up to degree 3, using the 3DGS/gsplat coefficient convention.

Colors in :class:`~rtgs.core.gaussians3d.Gaussians3D` are SH coefficients; degree 0 stores
``(rgb - 0.5) / C0`` so that a zero-degree gaussian evaluates back to its RGB color.
"""

from __future__ import annotations

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def num_sh_bases(degree: int) -> int:
    """Number of SH basis functions for a given degree: (degree + 1)^2."""
    return (degree + 1) ** 2


def sh_degree_from_bases(k: int) -> int:
    """Inverse of :func:`num_sh_bases`; raises for non-square k."""
    degree = int(round(k**0.5)) - 1
    if num_sh_bases(degree) != k:
        raise ValueError(f"{k} is not a valid SH basis count (must be a square)")
    return degree


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB in [0,1] to degree-0 SH coefficients (3DGS convention)."""
    return (rgb - 0.5) / C0


def sh_to_rgb(sh_dc: torch.Tensor) -> torch.Tensor:
    """Convert degree-0 SH coefficients back to RGB (clamped to [0,1])."""
    return (sh_dc * C0 + 0.5).clamp(0.0, 1.0)


def eval_sh(degree: int, sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate SH colors.

    Args:
        degree: SH degree to evaluate (0..3); ``sh`` may carry more bases, extras ignored.
        sh: coefficients, shape (N, K, 3) with K >= (degree+1)^2.
        dirs: unit view directions, shape (N, 3).

    Returns:
        RGB colors, shape (N, 3), clamped to be non-negative after the +0.5 shift.
    """
    if sh.shape[1] < num_sh_bases(degree):
        raise ValueError(
            f"sh has {sh.shape[1]} bases, degree {degree} needs {num_sh_bases(degree)}"
        )
    result = C0 * sh[:, 0]
    if degree >= 1:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        result = result - C1 * y * sh[:, 1] + C1 * z * sh[:, 2] - C1 * x * sh[:, 3]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = (
            result
            + C2[0] * xy * sh[:, 4]
            + C2[1] * yz * sh[:, 5]
            + C2[2] * (2.0 * zz - xx - yy) * sh[:, 6]
            + C2[3] * xz * sh[:, 7]
            + C2[4] * (xx - yy) * sh[:, 8]
        )
    if degree >= 3:
        result = (
            result
            + C3[0] * y * (3.0 * xx - yy) * sh[:, 9]
            + C3[1] * xy * z * sh[:, 10]
            + C3[2] * y * (4.0 * zz - xx - yy) * sh[:, 11]
            + C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[:, 12]
            + C3[4] * x * (4.0 * zz - xx - yy) * sh[:, 13]
            + C3[5] * z * (xx - yy) * sh[:, 14]
            + C3[6] * x * (xx - 3.0 * yy) * sh[:, 15]
        )
    return (result + 0.5).clamp_min(0.0)
