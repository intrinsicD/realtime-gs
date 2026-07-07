"""Depth Anything V2 backend via HuggingFace transformers (lazy, optional).

Uses the Small checkpoint by default — it is the only Apache-2.0-licensed size
(Base/Large are CC-BY-NC, see docs/RESEARCH.md §Depth). Output is relative inverse
depth; we convert to a monotone depth proxy and mark it 'affine' so the pipeline
routes it through scale/shift alignment against sparse points.
"""

from __future__ import annotations

import torch

from rtgs.depth.base import DepthPrediction

_DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthAnythingV2:
    """Monocular relative depth from Depth Anything V2 (requires `.[depth]` extra)."""

    def __init__(self, model_name: str = _DEFAULT_MODEL, device: str | None = None):
        try:
            from transformers import pipeline
        except ImportError as e:
            raise RuntimeError(
                "transformers is not installed; run `pip install -e '.[depth]'` "
                "or use a mock/GT depth backend"
            ) from e
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipe = pipeline("depth-estimation", model=model_name, device=device)

    def predict(self, image: torch.Tensor) -> DepthPrediction:
        """Estimate affine-invariant depth for one (H, W, 3) image in [0,1]."""
        import numpy as np
        from PIL import Image as PILImage

        arr = (image.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        result = self._pipe(PILImage.fromarray(arr))
        # The pipeline returns relative *inverse* depth (higher = closer); invert into a
        # monotone depth proxy. Affine alignment absorbs the unknown scale/shift.
        inv = torch.as_tensor(np.array(result["depth"], dtype="float32"), dtype=torch.float32)
        inv = inv / inv.max().clamp_min(1e-6)
        depth = 1.0 / (inv + 0.05)
        return DepthPrediction(depth=depth, kind="affine")
