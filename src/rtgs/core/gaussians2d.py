"""Container for a set of 2D gaussians fitted to a single image.

The position/Cholesky/color basis follows GaussianImage (Zhang et al., ECCV 2024): the 2x2
covariance is stored via ``L = [[l11, 0], [l21, l22]]`` with positive diagonal, which keeps it
symmetric positive definite by construction. Upstream GaussianImage fixes raster opacity to one
and directly optimizes a three-vector accumulated color. This repository extends that eight-
parameter representation by factoring accumulated RGB into a [0,1]^3 ``color`` and a [0,1]
``weight``. The extra scalar is not an alpha-compositing opacity; lifters use a conservative
independent 3D opacity prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class Gaussians2D:
    """N 2D gaussians in pixel coordinates of one image."""

    xy: torch.Tensor  # (N, 2) centers in pixels
    chol: torch.Tensor  # (N, 3) cholesky factors (l11, l21, l22), l11 > 0, l22 > 0
    color: torch.Tensor  # (N, 3) RGB in [0, 1]
    weight: torch.Tensor  # (N,) amplitude in [0, 1]

    def __post_init__(self) -> None:
        n = self.xy.shape[0]
        if self.chol.shape != (n, 3) or self.color.shape != (n, 3) or self.weight.shape != (n,):
            raise ValueError("inconsistent Gaussians2D field shapes")

    @property
    def n(self) -> int:
        """Number of gaussians."""
        return self.xy.shape[0]

    def cholesky_matrix(self) -> torch.Tensor:
        """(N,2,2) lower-triangular Cholesky factors."""
        l11, l21, l22 = self.chol[:, 0], self.chol[:, 1], self.chol[:, 2]
        zero = torch.zeros_like(l11)
        return torch.stack([torch.stack([l11, zero], -1), torch.stack([l21, l22], -1)], dim=-2)

    def covariance(self) -> torch.Tensor:
        """(N,2,2) covariance matrices Sigma = L @ L^T."""
        lmat = self.cholesky_matrix()
        return lmat @ lmat.transpose(-1, -2)

    def inverse_covariance(self) -> torch.Tensor:
        """(N,2,2) inverse covariances, computed analytically from the Cholesky factor."""
        l11, l21, l22 = self.chol[:, 0], self.chol[:, 1], self.chol[:, 2]
        # L^-1 = [[1/l11, 0], [-l21/(l11 l22), 1/l22]]; Sigma^-1 = L^-T L^-1
        a = 1.0 / l11
        b = -l21 / (l11 * l22)
        c = 1.0 / l22
        i00 = a * a + b * b
        i01 = b * c
        i11 = c * c
        return torch.stack([torch.stack([i00, i01], -1), torch.stack([i01, i11], -1)], dim=-2)

    def detach(self) -> Gaussians2D:
        """Detached copy (no autograd history)."""
        return Gaussians2D(
            self.xy.detach().clone(),
            self.chol.detach().clone(),
            self.color.detach().clone(),
            self.weight.detach().clone(),
        )

    def to(self, device: torch.device | str) -> Gaussians2D:
        """Return a copy on ``device``."""
        return Gaussians2D(
            self.xy.to(device),
            self.chol.to(device),
            self.color.to(device),
            self.weight.to(device),
        )

    def save_npz(self, path: str | Path) -> None:
        """Save to a compressed .npz archive."""
        np.savez_compressed(
            Path(path),
            xy=self.xy.detach().cpu().numpy(),
            chol=self.chol.detach().cpu().numpy(),
            color=self.color.detach().cpu().numpy(),
            weight=self.weight.detach().cpu().numpy(),
        )

    @staticmethod
    def load_npz(path: str | Path) -> Gaussians2D:
        """Load from :meth:`save_npz` output."""
        data = np.load(Path(path))
        return Gaussians2D(
            xy=torch.from_numpy(data["xy"]).float(),
            chol=torch.from_numpy(data["chol"]).float(),
            color=torch.from_numpy(data["color"]).float(),
            weight=torch.from_numpy(data["weight"]).float(),
        )
