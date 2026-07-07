"""Test/synthetic depth backends: ground-truth and constant depth.

These keep the depth-lifting variant fully testable without model weights. The
ground-truth backend is also the honest upper bound for what a perfect feed-forward
depth model could deliver — useful for ablations.
"""

from __future__ import annotations

import torch

from rtgs.depth.base import DepthPrediction


class GroundTruthDepth:
    """Serves precomputed depth maps (e.g., from the synthetic scene generator).

    ``kind`` is configurable so tests can exercise the alignment path by serving GT
    depth distorted by a known scale/shift.
    """

    def __init__(self, depths: list[torch.Tensor], kind: str = "metric"):
        self._depths = list(depths)
        self._kind = kind
        self._cursor = 0

    def predict(self, image: torch.Tensor) -> DepthPrediction:
        """Return the next depth map (call order must match view order)."""
        depth = self._depths[self._cursor]
        self._cursor = (self._cursor + 1) % len(self._depths)
        if depth.shape != image.shape[:2]:
            raise ValueError("GT depth shape does not match image")
        return DepthPrediction(depth=depth, kind=self._kind)  # type: ignore[arg-type]


class ConstantDepth:
    """Predicts a constant depth everywhere — the degenerate baseline."""

    def __init__(self, value: float, kind: str = "metric"):
        self.value = value
        self._kind = kind

    def predict(self, image: torch.Tensor) -> DepthPrediction:
        """Return a constant map at the configured depth."""
        h, w = image.shape[:2]
        return DepthPrediction(
            depth=torch.full((h, w), float(self.value)),
            kind=self._kind,  # type: ignore[arg-type]
        )
