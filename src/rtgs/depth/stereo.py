"""CPU-testable contracts and geometry for calibrated symmetric stereo depth.

The external GPS-Gaussian adapter is deliberately optional.  This module contains the typed
request/result seam and the exact flow-to-inverse-depth post-processing used by that adapter, so
signs, pixel centers, cycle consistency, and confidence remain testable without its repository,
checkpoint, SciPy, or CUDA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RectifiedStereoRequest:
    """One rectified pair prepared for a symmetric stereo backend.

    Images use ``(3,H,W)`` layout and already contain the backend-specific normalization.  Support
    maps are continuous values in ``[0,1]``.  Pixel array index ``(row, column)`` represents camera
    coordinate ``(column + 0.5, row + 0.5)``.
    """

    left_image: torch.Tensor
    right_image: torch.Tensor
    left_support: torch.Tensor
    right_support: torch.Tensor
    focal_pixels: float
    baseline_world: float
    maximum_cycle_error_px: float = 1.0
    confidence_decay_tau_px: float = 1.0

    def validate(self) -> None:
        """Reject malformed or ambiguous stereo inputs."""

        if self.left_image.ndim != 3 or self.left_image.shape[0] != 3:
            raise ValueError("left_image must have shape (3,H,W)")
        if self.right_image.shape != self.left_image.shape:
            raise ValueError("left and right images must have identical (3,H,W) shape")
        spatial = self.left_image.shape[1:]
        if self.left_support.shape != spatial or self.right_support.shape != spatial:
            raise ValueError("support maps must match image spatial dimensions")
        tensors = (
            self.left_image,
            self.right_image,
            self.left_support,
            self.right_support,
        )
        if any(not value.is_floating_point() for value in tensors):
            raise TypeError("stereo images and support maps must be floating point")
        if any(value.device != self.left_image.device for value in tensors):
            raise ValueError("stereo images and support maps must share one device")
        if any(value.dtype != self.left_image.dtype for value in tensors):
            raise ValueError("stereo images and support maps must share one dtype")
        if any(not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("stereo images and support maps must be finite")
        if bool(((self.left_support < 0) | (self.left_support > 1)).any()) or bool(
            ((self.right_support < 0) | (self.right_support > 1)).any()
        ):
            raise ValueError("stereo support must lie in [0,1]")
        for name in (
            "focal_pixels",
            "baseline_world",
            "maximum_cycle_error_px",
            "confidence_decay_tau_px",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class StereoDepthPrediction:
    """Bidirectional rectified inverse depth with explicit derived confidence."""

    left_inverse_depth: torch.Tensor
    right_inverse_depth: torch.Tensor
    left_confidence: torch.Tensor
    right_confidence: torch.Tensor
    left_cycle_error_px: torch.Tensor
    right_cycle_error_px: torch.Tensor
    left_valid: torch.Tensor
    right_valid: torch.Tensor
    left_flow_px: torch.Tensor
    right_flow_px: torch.Tensor
    diagnostics: dict[str, object]

    def validate(self, spatial_shape: tuple[int, int] | None = None) -> None:
        """Validate map shapes, dtypes, devices, and finite valid entries."""

        floating = (
            self.left_inverse_depth,
            self.right_inverse_depth,
            self.left_confidence,
            self.right_confidence,
            self.left_cycle_error_px,
            self.right_cycle_error_px,
            self.left_flow_px,
            self.right_flow_px,
        )
        masks = (self.left_valid, self.right_valid)
        shape = self.left_inverse_depth.shape
        if len(shape) != 2 or (spatial_shape is not None and shape != spatial_shape):
            raise ValueError("stereo prediction maps have the wrong spatial shape")
        if any(value.shape != shape for value in (*floating, *masks)):
            raise ValueError("all stereo prediction maps must share one shape")
        if any(not value.is_floating_point() for value in floating):
            raise TypeError("stereo numeric prediction maps must be floating point")
        if any(value.dtype != torch.bool for value in masks):
            raise TypeError("stereo valid maps must be boolean")
        if any(value.device != floating[0].device for value in (*floating, *masks)):
            raise ValueError("all stereo prediction maps must share one device")
        if any(not bool(torch.isfinite(value).all()) for value in floating):
            raise ValueError("stereo prediction maps must be finite")
        for inverse_depth, confidence, valid in (
            (self.left_inverse_depth, self.left_confidence, self.left_valid),
            (self.right_inverse_depth, self.right_confidence, self.right_valid),
        ):
            if bool((inverse_depth[valid] <= 0).any()):
                raise ValueError("valid inverse depth must be positive")
            if bool(((confidence < 0) | (confidence > 1)).any()):
                raise ValueError("confidence must lie in [0,1]")


class StereoDepthBackend(Protocol):
    """Predict calibrated symmetric inverse depth for one rectified pair."""

    def predict_pair(self, request: RectifiedStereoRequest) -> StereoDepthPrediction:
        """Return one depth prediction without mutating the request."""
        ...


def _flow_map(value: torch.Tensor, shape: tuple[int, int], *, name: str) -> torch.Tensor:
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.shape != shape or not value.is_floating_point():
        raise ValueError(f"{name} must have shape {shape} or (1,{shape[0]},{shape[1]})")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _warp_horizontal(value: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``value`` at ``(u + flow, v)`` without clamping pixel coordinates."""

    height, width = flow.shape
    dtype, device = flow.dtype, flow.device
    y = torch.arange(height, dtype=dtype, device=device)[:, None] + 0.5
    x = torch.arange(width, dtype=dtype, device=device)[None, :] + 0.5
    target_x = x + flow
    target_y = y.expand_as(target_x)
    grid = torch.stack(
        (2.0 * target_x / width - 1.0, 2.0 * target_y / height - 1.0),
        dim=-1,
    )
    sampled = F.grid_sample(
        value[None, None],
        grid[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0]
    in_bounds = (
        (target_x >= 0.5)
        & (target_x <= width - 0.5)
        & (target_y >= 0.5)
        & (target_y <= height - 0.5)
    )
    return sampled, in_bounds


def stereo_depth_from_bidirectional_flow(
    request: RectifiedStereoRequest,
    left_flow_px: torch.Tensor,
    right_flow_px: torch.Tensor,
) -> StereoDepthPrediction:
    """Apply the frozen GPS flow-to-inverse-depth and cycle-confidence equations.

    GPS flow is ``x_reference - x_main``.  With common rectified principal points,
    ``Tf_x_left=-fB`` and ``Tf_x_right=+fB``.  The official ``flow2depth`` therefore returns
    inverse depth ``rho = flow / Tf_x``; metric axial depth is its reciprocal.
    """

    request.validate()
    shape = tuple(request.left_support.shape)
    left_flow = _flow_map(left_flow_px, shape, name="left_flow_px").to(request.left_image)
    right_flow = _flow_map(right_flow_px, shape, name="right_flow_px").to(request.left_image)

    scale = request.focal_pixels * request.baseline_world
    left_inverse = left_flow / (-scale)
    right_inverse = right_flow / scale

    sampled_right_flow, left_in_bounds = _warp_horizontal(right_flow, left_flow)
    sampled_left_flow, right_in_bounds = _warp_horizontal(left_flow, right_flow)
    sampled_right_support, _ = _warp_horizontal(request.right_support, left_flow)
    sampled_left_support, _ = _warp_horizontal(request.left_support, right_flow)
    left_error = (left_flow + sampled_right_flow).abs()
    right_error = (right_flow + sampled_left_flow).abs()

    left_valid = (
        left_in_bounds
        & (request.left_support >= 0.5)
        & (sampled_right_support >= 0.5)
        & torch.isfinite(left_inverse)
        & (left_inverse > 0)
        & torch.isfinite(left_error)
        & (left_error <= request.maximum_cycle_error_px)
    )
    right_valid = (
        right_in_bounds
        & (request.right_support >= 0.5)
        & (sampled_left_support >= 0.5)
        & torch.isfinite(right_inverse)
        & (right_inverse > 0)
        & torch.isfinite(right_error)
        & (right_error <= request.maximum_cycle_error_px)
    )
    left_confidence = (
        torch.exp(-left_error / request.confidence_decay_tau_px)
        * request.left_support.clamp(0, 1)
        * sampled_right_support.clamp(0, 1)
    )
    right_confidence = (
        torch.exp(-right_error / request.confidence_decay_tau_px)
        * request.right_support.clamp(0, 1)
        * sampled_left_support.clamp(0, 1)
    )
    left_inverse = torch.where(left_valid, left_inverse, torch.zeros_like(left_inverse))
    right_inverse = torch.where(right_valid, right_inverse, torch.zeros_like(right_inverse))
    left_error = torch.where(left_valid, left_error, torch.zeros_like(left_error))
    right_error = torch.where(right_valid, right_error, torch.zeros_like(right_error))
    left_confidence = torch.where(
        left_valid,
        left_confidence,
        torch.zeros_like(left_confidence),
    )
    right_confidence = torch.where(
        right_valid,
        right_confidence,
        torch.zeros_like(right_confidence),
    )
    prediction = StereoDepthPrediction(
        left_inverse_depth=left_inverse,
        right_inverse_depth=right_inverse,
        left_confidence=left_confidence,
        right_confidence=right_confidence,
        left_cycle_error_px=left_error,
        right_cycle_error_px=right_error,
        left_valid=left_valid,
        right_valid=right_valid,
        left_flow_px=left_flow,
        right_flow_px=right_flow,
        diagnostics={
            "schema": "rtgs.symmetric_stereo_depth.v1",
            "focal_pixels": request.focal_pixels,
            "baseline_world": request.baseline_world,
            "tf_x_left": -scale,
            "tf_x_right": scale,
            "maximum_cycle_error_px": request.maximum_cycle_error_px,
            "confidence_decay_tau_px": request.confidence_decay_tau_px,
            "left_valid_pixels": int(left_valid.sum()),
            "right_valid_pixels": int(right_valid.sum()),
        },
    )
    prediction.validate(shape)
    return prediction


__all__ = [
    "RectifiedStereoRequest",
    "StereoDepthBackend",
    "StereoDepthPrediction",
    "stereo_depth_from_bidirectional_flow",
]
