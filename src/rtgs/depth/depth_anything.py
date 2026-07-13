"""Depth Anything V2 backend via HuggingFace transformers (lazy, optional).

Uses the Small checkpoint by default — it is the only Apache-2.0-licensed size
(Base/Large are CC-BY-NC, see docs/RESEARCH.md §Depth). Output is relative inverse
depth; the raw model tensor is marked ``inverse`` so alignment happens in disparity space.
"""

from __future__ import annotations

import torch

from rtgs.depth.base import DepthPrediction

_DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
_PERMISSIVE_MODELS = {_DEFAULT_MODEL}


class DepthAnythingV2:
    """Monocular relative depth from Depth Anything V2 (requires `.[depth]` extra)."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str | None = None,
        allow_unverified_license: bool = False,
    ):
        if model_name not in _PERMISSIVE_MODELS and not allow_unverified_license:
            raise ValueError(
                f"checkpoint '{model_name}' is not in the permissive-license allowlist; "
                "pass allow_unverified_license=True only after verifying its code and weights"
            )
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
        from PIL import Image as PILImage

        arr = (image.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        result = self._pipe(PILImage.fromarray(arr))
        # ``depth`` is an 8-bit PIL visualization. Keep the raw model tensor instead and let
        # the alignment layer fit its affine ambiguity in inverse-depth space.
        inv = torch.as_tensor(result["predicted_depth"], dtype=torch.float32).squeeze()
        if inv.shape != image.shape[:2]:
            inv = torch.nn.functional.interpolate(
                inv[None, None], size=image.shape[:2], mode="bicubic", align_corners=False
            )[0, 0]
        return DepthPrediction(depth=inv.clamp_min(1e-6), kind="inverse")
