"""Calibrated oriented-point inputs, fixed targets, and surface losses.

Backend predictions explicitly state their geometry and normal coordinate frames.  A
canonicalizer converts registered per-pixel inputs into detached world-space maps without
giving the backend ownership of optimizer tensors.  Targets then use retained primitive
indices, world-space points, and unoriented world-space normals.  Validation binds them to
one retained layout before they enter an optimization loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import quat_to_rotmat

GeometryKind = Literal["camera_z_depth", "camera_points", "world_points"]
NormalFrame = Literal["camera", "world"]

_GEOMETRY_KINDS = {"camera_z_depth", "camera_points", "world_points"}
_NORMAL_FRAMES = {"camera", "world"}


@dataclass(frozen=True)
class OrientedPointProvenance:
    """Stable identity for one backend prediction and its frozen configuration."""

    view_id: str
    backend_name: str
    backend_version: str
    config_id: str


@dataclass(frozen=True)
class OrientedPointPrediction:
    """Registered per-pixel geometry and normals in explicitly declared frames."""

    geometry: torch.Tensor
    normals: torch.Tensor
    geometry_kind: GeometryKind
    normal_frame: NormalFrame
    provenance: OrientedPointProvenance
    valid: torch.Tensor | None = None
    confidence: torch.Tensor | None = None


@dataclass(frozen=True)
class CanonicalOrientedPointMap:
    """Validated detached world-space oriented points for one calibrated view."""

    points_world: torch.Tensor
    normals_world: torch.Tensor
    valid: torch.Tensor
    confidence: torch.Tensor | None
    provenance: OrientedPointProvenance


class OrientedPointBackend(Protocol):
    """Predict registered metric geometry and normals for one stable calibrated view."""

    def predict(
        self,
        image_shape: tuple[int, int],
        camera: Camera,
        *,
        view_id: str,
    ) -> OrientedPointPrediction:
        """Return one prediction keyed by ``view_id`` without relying on call order."""
        ...


@dataclass(frozen=True)
class OrientedPointTargets:
    """Sparse fixed surface targets keyed by retained primitive index."""

    indices: torch.Tensor
    points: torch.Tensor
    plane_normals: torch.Tensor
    alignment_normals: torch.Tensor | None = None


def _is_attached(value: torch.Tensor) -> bool:
    return value.requires_grad or value.grad_fn is not None


def _validate_image_shape(image_shape: tuple[int, int], camera: Camera) -> tuple[int, int]:
    if (
        not isinstance(image_shape, tuple)
        or len(image_shape) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) for value in image_shape)
        or any(value <= 0 for value in image_shape)
    ):
        raise ValueError("image_shape must be a positive (height, width) integer tuple")
    height, width = image_shape
    if (height, width) != (camera.height, camera.width):
        raise ValueError("image_shape must match the calibrated camera resolution")
    if not all(math.isfinite(value) for value in (camera.fx, camera.fy, camera.cx, camera.cy)):
        raise ValueError("camera intrinsics must be finite")
    if camera.fx <= 0 or camera.fy <= 0:
        raise ValueError("camera focal lengths must be positive")
    if not bool(torch.isfinite(camera.R).all()) or not bool(torch.isfinite(camera.t).all()):
        raise ValueError("camera extrinsics must be finite")
    return height, width


def _validate_provenance(
    provenance: OrientedPointProvenance,
    *,
    expected_view_id: str,
    expected_config_id: str,
) -> None:
    if not isinstance(provenance, OrientedPointProvenance):
        raise TypeError("prediction provenance must be an OrientedPointProvenance instance")
    for name, value in (
        ("view_id", provenance.view_id),
        ("backend_name", provenance.backend_name),
        ("backend_version", provenance.backend_version),
        ("config_id", provenance.config_id),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"provenance {name} must be a non-empty string")
    if not isinstance(expected_view_id, str) or not expected_view_id:
        raise ValueError("expected_view_id must be a non-empty string")
    if not isinstance(expected_config_id, str) or not expected_config_id:
        raise ValueError("expected_config_id must be a non-empty string")
    if provenance.view_id != expected_view_id:
        raise ValueError("prediction provenance view_id mismatch")
    if provenance.config_id != expected_config_id:
        raise ValueError("prediction provenance config_id mismatch")


def _require_prediction_tensor(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...],
    *,
    floating: bool = True,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"prediction {name} must be a torch.Tensor")
    if value.shape != shape:
        raise ValueError(f"prediction {name} must have shape {shape}")
    if floating and not value.is_floating_point():
        raise ValueError(f"prediction {name} must use a floating dtype")
    if _is_attached(value):
        raise ValueError(f"prediction {name} must be detached")


def _pixel_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rows, columns = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    return torch.stack([columns, rows], dim=-1)


def _source_geometry_valid(
    geometry: torch.Tensor,
    geometry_kind: GeometryKind,
    camera: Camera,
) -> torch.Tensor:
    if geometry_kind == "camera_z_depth":
        return torch.isfinite(geometry) & (geometry > 0)
    finite = torch.isfinite(geometry).all(dim=-1)
    if geometry_kind == "camera_points":
        return finite & (geometry[..., 2] > 0)
    safe_world = torch.where(finite[..., None], geometry, torch.zeros_like(geometry))
    camera_points = camera.world_to_cam(safe_world.reshape(-1, 3)).reshape_as(safe_world)
    return finite & torch.isfinite(camera_points).all(dim=-1) & (camera_points[..., 2] > 0)


def canonicalize_oriented_point_prediction(
    prediction: OrientedPointPrediction,
    camera: Camera,
    image_shape: tuple[int, int],
    *,
    expected_view_id: str,
    expected_config_id: str,
    device: torch.device | str,
    dtype: torch.dtype,
) -> CanonicalOrientedPointMap:
    """Validate one registered prediction and clone it into a safe world-space map.

    Camera-z geometry is optical-axis depth.  Array element ``(row, column)`` is unprojected
    at repository pixel coordinate ``(column + 0.5, row + 0.5)``.  Camera-frame normals use
    the row-vector transform ``n_world = n_camera @ R``; their signs remain immaterial.
    """
    if not isinstance(prediction, OrientedPointPrediction):
        raise TypeError("prediction must be an OrientedPointPrediction instance")
    height, width = _validate_image_shape(image_shape, camera)
    if prediction.geometry_kind not in _GEOMETRY_KINDS:
        raise ValueError(f"unknown prediction geometry_kind {prediction.geometry_kind!r}")
    if prediction.normal_frame not in _NORMAL_FRAMES:
        raise ValueError(f"unknown prediction normal_frame {prediction.normal_frame!r}")
    _validate_provenance(
        prediction.provenance,
        expected_view_id=expected_view_id,
        expected_config_id=expected_config_id,
    )
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("canonical prediction dtype must be floating point")

    geometry_shape = (
        (height, width) if prediction.geometry_kind == "camera_z_depth" else (height, width, 3)
    )
    _require_prediction_tensor("geometry", prediction.geometry, geometry_shape)
    _require_prediction_tensor("normals", prediction.normals, (height, width, 3))
    if prediction.valid is not None:
        _require_prediction_tensor("valid", prediction.valid, (height, width), floating=False)
        if prediction.valid.dtype != torch.bool:
            raise ValueError("prediction valid must use torch.bool")
    if prediction.confidence is not None:
        _require_prediction_tensor("confidence", prediction.confidence, (height, width))
    prediction_tensors = [prediction.normals]
    if prediction.valid is not None:
        prediction_tensors.append(prediction.valid)
    if prediction.confidence is not None:
        prediction_tensors.append(prediction.confidence)
    if any(value.device != prediction.geometry.device for value in prediction_tensors):
        raise ValueError("prediction tensors must share one device")

    geometry_valid = _source_geometry_valid(prediction.geometry, prediction.geometry_kind, camera)
    normal_finite = torch.isfinite(prediction.normals).all(dim=-1)
    safe_source_normals = torch.where(
        normal_finite[..., None], prediction.normals, torch.zeros_like(prediction.normals)
    )
    normal_norm = safe_source_normals.norm(dim=-1)
    normal_valid = normal_finite & torch.isfinite(normal_norm) & (normal_norm > 0)
    inferred_valid = geometry_valid & normal_valid
    source_valid = inferred_valid if prediction.valid is None else prediction.valid
    if bool((source_valid & ~geometry_valid).any()):
        raise ValueError("valid prediction geometry must be finite, positive, and in front")
    if bool((source_valid & ~normal_valid).any()):
        raise ValueError("valid prediction normals must be finite and nonzero")
    if prediction.confidence is not None:
        confidence_valid = torch.isfinite(prediction.confidence) & (
            (prediction.confidence >= 0) & (prediction.confidence <= 1)
        )
        if bool((source_valid & ~confidence_valid).any()):
            raise ValueError("valid prediction confidence must be finite and in [0, 1]")

    resolved_device = torch.device(device)
    valid_out = source_valid.detach().to(device=resolved_device).clone()
    geometry_out = prediction.geometry.detach().to(device=resolved_device, dtype=dtype).clone()
    normals_out = prediction.normals.detach().to(device=resolved_device, dtype=dtype).clone()
    geometry_out = torch.where(
        valid_out[..., None] if geometry_out.ndim == 3 else valid_out,
        geometry_out,
        torch.zeros_like(geometry_out),
    )
    normals_out = torch.where(valid_out[..., None], normals_out, torch.zeros_like(normals_out))

    if prediction.geometry_kind == "camera_z_depth":
        pixels = _pixel_grid(
            height,
            width,
            device=resolved_device,
            dtype=dtype,
        )
        points_world = camera.unproject(pixels.reshape(-1, 2), geometry_out.reshape(-1)).reshape(
            height, width, 3
        )
    elif prediction.geometry_kind == "camera_points":
        points_world = camera.cam_to_world(geometry_out.reshape(-1, 3)).reshape(height, width, 3)
    else:
        points_world = geometry_out
    if prediction.normal_frame == "camera":
        normals_world = normals_out @ camera.R.to(normals_out)
    else:
        normals_world = normals_out

    finite_world_points = torch.isfinite(points_world).all(dim=-1)
    safe_world_points = torch.where(
        finite_world_points[..., None], points_world, torch.zeros_like(points_world)
    )
    postcast_camera_points = camera.world_to_cam(safe_world_points.reshape(-1, 3)).reshape_as(
        safe_world_points
    )
    postcast_geometry_valid = (
        finite_world_points
        & torch.isfinite(postcast_camera_points).all(dim=-1)
        & (postcast_camera_points[..., 2] > 0)
    )
    if bool((valid_out & ~postcast_geometry_valid).any()):
        raise ValueError(
            "valid prediction geometry must remain finite and positive after conversion"
        )
    world_normal_norm = normals_world.norm(dim=-1)
    postcast_normal_valid = (
        torch.isfinite(normals_world).all(dim=-1)
        & torch.isfinite(world_normal_norm)
        & (world_normal_norm > 0)
    )
    if bool((valid_out & ~postcast_normal_valid).any()):
        raise ValueError("valid prediction normals must remain finite and nonzero after conversion")
    normals_world = normals_world / world_normal_norm.clamp_min(torch.finfo(dtype).tiny)[..., None]
    points_world = torch.where(valid_out[..., None], points_world, torch.zeros_like(points_world))
    normals_world = torch.where(
        valid_out[..., None], normals_world, torch.zeros_like(normals_world)
    )

    confidence_out = None
    if prediction.confidence is not None:
        confidence_out = (
            prediction.confidence.detach().to(device=resolved_device, dtype=dtype).clone()
        )
        if bool((valid_out & ~torch.isfinite(confidence_out)).any()):
            raise ValueError("valid prediction confidence must remain finite after conversion")
        confidence_out = torch.where(valid_out, confidence_out, torch.zeros_like(confidence_out))
    return CanonicalOrientedPointMap(
        points_world=points_world.detach().clone(),
        normals_world=normals_world.detach().clone(),
        valid=valid_out.detach().clone(),
        confidence=None if confidence_out is None else confidence_out.detach().clone(),
        provenance=prediction.provenance,
    )


def estimate_registered_depth_normals(
    depth: torch.Tensor,
    camera: Camera,
    *,
    pixel_offset: int = 2,
    min_depth: float = 0.3,
    max_depth: float = 5.0,
    max_abs_depth_delta: float = 0.04,
    max_relative_depth_delta: float = 0.02,
    min_cross_norm: float = 1e-8,
    min_abs_incidence: float = 0.20,
    orient_toward_camera: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate deterministic camera-space normals from a registered five-depth stencil.

    The horizontal and vertical tangents join samples ``pixel_offset`` pixels on either
    side of each center.  A center is valid only when all five depths lie in range, every
    neighbor agrees with the center under the absolute-or-relative threshold, the tangent
    cross product is nondegenerate, and its absolute ray incidence passes the floor.
    Invalid output normals are safe zeros.
    """
    if not isinstance(depth, torch.Tensor):
        raise TypeError("depth must be a torch.Tensor")
    if depth.ndim != 2 or depth.shape != (camera.height, camera.width):
        raise ValueError("depth must have shape matching the calibrated camera")
    if not depth.is_floating_point():
        raise ValueError("depth must use a floating dtype")
    if _is_attached(depth):
        raise ValueError("depth must be detached")
    _validate_image_shape((depth.shape[0], depth.shape[1]), camera)
    if not isinstance(pixel_offset, int) or isinstance(pixel_offset, bool) or pixel_offset <= 0:
        raise ValueError("pixel_offset must be a positive integer")
    if depth.shape[0] <= 2 * pixel_offset or depth.shape[1] <= 2 * pixel_offset:
        raise ValueError("depth map must be larger than the five-point stencil")
    if (
        not math.isfinite(min_depth)
        or not math.isfinite(max_depth)
        or not 0 < min_depth < max_depth
    ):
        raise ValueError("depth range must be finite, positive, and increasing")
    for name, value in (
        ("max_abs_depth_delta", max_abs_depth_delta),
        ("max_relative_depth_delta", max_relative_depth_delta),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not math.isfinite(min_cross_norm) or min_cross_norm <= 0:
        raise ValueError("min_cross_norm must be finite and positive")
    if not math.isfinite(min_abs_incidence) or not 0 <= min_abs_incidence <= 1:
        raise ValueError("min_abs_incidence must be finite and in [0, 1]")
    if not isinstance(orient_toward_camera, bool):
        raise ValueError("orient_toward_camera must be boolean")

    height, width = depth.shape
    pixels = _pixel_grid(height, width, device=depth.device, dtype=depth.dtype)
    camera_points = torch.stack(
        [
            (pixels[..., 0] - camera.cx) / camera.fx * depth,
            (pixels[..., 1] - camera.cy) / camera.fy * depth,
            depth,
        ],
        dim=-1,
    )
    offset = pixel_offset
    center_depth = depth[offset:-offset, offset:-offset]
    neighbor_depths = (
        depth[offset:-offset, : -2 * offset],
        depth[offset:-offset, 2 * offset :],
        depth[: -2 * offset, offset:-offset],
        depth[2 * offset :, offset:-offset],
    )
    finite_and_in_range = (
        torch.isfinite(center_depth) & (center_depth >= min_depth) & (center_depth <= max_depth)
    )
    depth_delta_limit = torch.maximum(
        torch.full_like(center_depth, max_abs_depth_delta),
        max_relative_depth_delta * center_depth,
    )
    for neighbor_depth in neighbor_depths:
        finite_and_in_range &= (
            torch.isfinite(neighbor_depth)
            & (neighbor_depth >= min_depth)
            & (neighbor_depth <= max_depth)
            & ((neighbor_depth - center_depth).abs() <= depth_delta_limit)
        )

    left_points = camera_points[offset:-offset, : -2 * offset]
    right_points = camera_points[offset:-offset, 2 * offset :]
    upper_points = camera_points[: -2 * offset, offset:-offset]
    lower_points = camera_points[2 * offset :, offset:-offset]
    center_points = camera_points[offset:-offset, offset:-offset]
    tangent_u = right_points - left_points
    tangent_v = lower_points - upper_points
    cross = torch.linalg.cross(tangent_u, tangent_v, dim=-1)
    cross_norm = cross.norm(dim=-1)
    normals = cross / cross_norm.clamp_min(min_cross_norm)[..., None]
    ray_norm = center_points.norm(dim=-1)
    unit_ray = center_points / ray_norm.clamp_min(torch.finfo(depth.dtype).tiny)[..., None]
    signed_incidence = (normals * unit_ray).sum(dim=-1)
    valid_core = (
        finite_and_in_range
        & torch.isfinite(cross).all(dim=-1)
        & torch.isfinite(cross_norm)
        & (cross_norm > min_cross_norm)
        & torch.isfinite(signed_incidence)
        & (signed_incidence.abs() >= min_abs_incidence)
    )
    if orient_toward_camera:
        normals = torch.where((signed_incidence > 0)[..., None], -normals, normals)
    normals = torch.where(valid_core[..., None], normals, torch.zeros_like(normals))

    normal_map = torch.zeros(height, width, 3, dtype=depth.dtype, device=depth.device)
    valid_map = torch.zeros(height, width, dtype=torch.bool, device=depth.device)
    normal_map[offset:-offset, offset:-offset] = normals
    valid_map[offset:-offset, offset:-offset] = valid_core
    return normal_map.detach(), valid_map.detach()


def _validate_vector_rows(name: str, value: torch.Tensor, count: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != (count, 3):
        raise ValueError(f"{name} must have shape (M, 3)")
    if _is_attached(value):
        raise ValueError(f"{name} must be detached")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _normalized_normals(name: str, value: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must remain finite after dtype conversion")
    norms = value.norm(dim=-1)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
        raise ValueError(f"{name} must contain only nonzero normals")
    normalized = value / norms[:, None]
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError(f"{name} normalization must remain finite")
    return normalized


def validate_oriented_point_targets(
    targets: OrientedPointTargets,
    n_retained: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> OrientedPointTargets:
    """Validate and clone targets for a retained layout, device, and floating dtype."""
    if not isinstance(targets, OrientedPointTargets):
        raise TypeError("targets must be an OrientedPointTargets instance")
    if not isinstance(n_retained, int) or isinstance(n_retained, bool) or n_retained <= 0:
        raise ValueError("n_retained must be a positive integer")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError("target dtype must be floating point")

    indices = targets.indices
    if not isinstance(indices, torch.Tensor):
        raise TypeError("indices must be a torch.Tensor")
    if indices.dtype != torch.long:
        raise ValueError("indices must use torch.int64")
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("indices must be a non-empty 1D tensor")
    if _is_attached(indices):
        raise ValueError("indices must be detached")
    if int(indices[0]) < 0 or int(indices[-1]) >= n_retained:
        raise ValueError("indices contain an out-of-range retained primitive index")
    if indices.numel() > 1 and not bool((indices[1:] > indices[:-1]).all()):
        raise ValueError("indices must be explicitly increasing and unique")

    count = int(indices.numel())
    _validate_vector_rows("points", targets.points, count)
    _validate_vector_rows("plane_normals", targets.plane_normals, count)
    if targets.alignment_normals is not None:
        _validate_vector_rows("alignment_normals", targets.alignment_normals, count)

    resolved_device = torch.device(device)
    indices_out = indices.detach().to(device=resolved_device).clone()
    points_out = targets.points.detach().to(device=resolved_device, dtype=dtype).clone()
    if not bool(torch.isfinite(points_out).all()):
        raise ValueError("points must remain finite after dtype conversion")
    plane_out = targets.plane_normals.detach().to(device=resolved_device, dtype=dtype).clone()
    plane_out = _normalized_normals("plane_normals", plane_out)
    alignment_out = None
    if targets.alignment_normals is not None:
        alignment_out = (
            targets.alignment_normals.detach().to(device=resolved_device, dtype=dtype).clone()
        )
        alignment_out = _normalized_normals("alignment_normals", alignment_out)
    return OrientedPointTargets(
        indices=indices_out,
        points=points_out,
        plane_normals=plane_out,
        alignment_normals=alignment_out,
    )


def _check_loss_targets(
    targets: OrientedPointTargets,
    n_items: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    count = targets.indices.numel()
    if targets.indices.dtype != torch.long or targets.indices.shape != (count,) or count == 0:
        raise ValueError("validated target indices must be a non-empty 1D int64 tensor")
    if targets.indices.device != device:
        raise ValueError("target indices must be on the optimized tensor device")
    if int(targets.indices.min()) < 0 or int(targets.indices.max()) >= n_items:
        raise ValueError("target index is out of range for the optimized tensor")
    for name, value in (
        ("points", targets.points),
        ("plane_normals", targets.plane_normals),
    ):
        if value.shape != (count, 3) or value.device != device or value.dtype != dtype:
            raise ValueError(f"validated {name} disagree with the optimized tensor")
    if targets.alignment_normals is not None and (
        targets.alignment_normals.shape != (count, 3)
        or targets.alignment_normals.device != device
        or targets.alignment_normals.dtype != dtype
    ):
        raise ValueError("validated alignment_normals disagree with the optimized tensor")


def local_plane_loss(
    means: torch.Tensor,
    targets: OrientedPointTargets,
    scene_extent: float,
) -> torch.Tensor:
    """Mean absolute point-to-plane distance normalized by scene extent."""
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape (N, 3)")
    if not math.isfinite(scene_extent) or scene_extent <= 0:
        raise ValueError("scene_extent must be finite and positive")
    _check_loss_targets(targets, means.shape[0], device=means.device, dtype=means.dtype)
    normals = _normalized_normals("plane_normals", targets.plane_normals)
    offsets = means[targets.indices] - targets.points
    signed_distance = (offsets * normals).sum(dim=-1)
    return signed_distance.abs().mean() / scene_extent


def shortest_axis_normal_loss(
    quats: torch.Tensor,
    axis_indices: torch.Tensor,
    targets: OrientedPointTargets,
) -> torch.Tensor:
    """Align each selected Gaussian rotation column to an unoriented target normal."""
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError("quats must have shape (N, 4)")
    if not bool(torch.isfinite(quats).all()):
        raise ValueError("quats must contain only finite values")
    _check_loss_targets(targets, quats.shape[0], device=quats.device, dtype=quats.dtype)
    count = targets.indices.numel()
    if not isinstance(axis_indices, torch.Tensor):
        raise TypeError("axis_indices must be a torch.Tensor")
    if axis_indices.dtype != torch.long or axis_indices.shape != (count,):
        raise ValueError("axis_indices must be a length-M int64 tensor")
    if axis_indices.device != quats.device:
        raise ValueError("axis_indices must be on the quaternion device")
    if bool(((axis_indices < 0) | (axis_indices > 2)).any()):
        raise ValueError("axis_indices values must be in [0, 2]")

    rotations = quat_to_rotmat(quats[targets.indices])
    gather_index = axis_indices[:, None, None].expand(-1, 3, 1)
    axes = rotations.gather(dim=2, index=gather_index).squeeze(-1)
    normals = (
        targets.plane_normals if targets.alignment_normals is None else targets.alignment_normals
    )
    normals = _normalized_normals("alignment_normals", normals)
    cosine = (axes * normals).sum(dim=-1).abs().clamp(max=1.0)
    return (1.0 - cosine).mean()
