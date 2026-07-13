"""Adapters for permissively licensed external 2D Gaussian image representations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rtgs.core.gaussians2d import Gaussians2D


def load_gaussians2d(path: str | Path, source: str = "auto") -> Gaussians2D:
    """Load native, StructSplat, or common GaussianImage-style ``.npz`` fields.

    StructSplat uses integer-centered pixels and rotation/scale covariance, while this repo uses
    half-integer centers and Cholesky covariance. Its optional ``opacities`` are logits. Geometry
    is converted exactly; RGB is clamped because normalized StructSplat colors can be unbounded.
    """
    data = np.load(Path(path))
    keys = set(data.files)
    if source == "auto":
        if {"xy", "chol", "color", "weight"} <= keys:
            source = "native"
        elif {"means", "log_scales", "rotations", "colors"} <= keys:
            source = "structsplat"
        else:
            source = "gaussianimage"
    if source == "native":
        return Gaussians2D(
            xy=_tensor(data["xy"]),
            chol=_tensor(data["chol"]),
            color=_tensor(data["color"]),
            weight=_tensor(data["weight"]),
        )

    means = _first(data, "means", "xy", "positions")
    colors = _first(data, "colors", "color", "rgb").clamp(0, 1)
    if source == "structsplat":
        means = means + 0.5
    if "log_scales" in keys:
        scales = _tensor(data["log_scales"]).exp()
    else:
        scales = _first(data, "scales", "scale", "scaling")
    angles = _first(data, "rotations", "rotation", "angles", "angle").reshape(-1)
    covariance = _rs_covariance(scales, angles)
    eye = torch.eye(2, dtype=covariance.dtype, device=covariance.device)
    chol_matrix = torch.linalg.cholesky(covariance + 1e-8 * eye)
    chol = torch.stack([chol_matrix[:, 0, 0], chol_matrix[:, 1, 0], chol_matrix[:, 1, 1]], dim=-1)

    if "opacities" in keys:
        weight = torch.sigmoid(_tensor(data["opacities"])).reshape(-1)
    elif "opacity" in keys:
        weight = _tensor(data["opacity"]).reshape(-1).clamp(0, 1)
    elif "weight" in keys:
        weight = _tensor(data["weight"]).reshape(-1).clamp(0, 1)
    else:
        weight = torch.ones(means.shape[0])
    return Gaussians2D(means, chol, colors, weight)


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array).copy()).float()


def _first(data, *names: str) -> torch.Tensor:
    for name in names:
        if name in data.files:
            return _tensor(data[name])
    raise ValueError(f"2D Gaussian archive is missing one of: {', '.join(names)}")


def _rs_covariance(scales: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    if scales.ndim == 1:
        scales = scales[:, None].expand(-1, 2)
    c, s = torch.cos(angles), torch.sin(angles)
    rotation = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], dim=-2)
    rs = rotation * scales[:, None, :]
    return rs @ rs.transpose(-1, -2)
