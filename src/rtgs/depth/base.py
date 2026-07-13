"""DepthBackend protocol: pluggable monocular depth estimation.

Backends declare what they output: 'metric' depth (meters/world units), 'relative'
(arbitrary monotonic), or 'affine' (correct up to scale+shift). Non-metric predictions
must be aligned to the scene scale (``rtgs.depth.align``) before lifting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch

DepthKind = Literal["metric", "relative", "affine", "inverse"]


@dataclass
class DepthPrediction:
    """A per-pixel depth map plus its scale semantics."""

    depth: torch.Tensor  # (H, W), positive
    kind: DepthKind
    confidence: torch.Tensor | None = None  # (H, W) in [0,1], optional


class DepthBackend(Protocol):
    """Predicts depth for a single (H, W, 3) image in [0,1]."""

    def predict(self, image: torch.Tensor) -> DepthPrediction:
        """Estimate depth for one image."""
        ...
