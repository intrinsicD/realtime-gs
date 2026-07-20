"""CPU-first geometry for flow-tracked 2D Gaussian correspondences.

This module contains the network-independent part of G²SR-style tracking: construct five
deterministic sigma points, sample a caller-supplied optical-flow field, recover a local affine
warp, reject inconsistent round trips, and triangulate matched centers.  It deliberately does not
load images or import an optical-flow model; any flow producer remains an external, pluggable
stage-1 backend.

Pixel coordinates follow :class:`rtgs.core.camera.Camera`: ``(u, v)`` addresses column and row,
and the top-left pixel center is ``(0.5, 0.5)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from rtgs.core.camera import Camera


@dataclass(frozen=True)
class SigmaPoints2D:
    """Five deterministic points and the covariance factor used to construct them."""

    points: torch.Tensor  # (..., 5, 2)
    factor: torch.Tensor  # (..., 2, 2), principal covariance axes in the columns
    eigenvalues: torch.Tensor  # (..., 2), descending
    valid: torch.Tensor  # (...)


@dataclass(frozen=True)
class FlowSamples:
    """Bilinear flow samples with an explicit fail-closed validity mask."""

    values: torch.Tensor  # (..., 2); zero where invalid
    valid: torch.Tensor  # (...)


@dataclass(frozen=True)
class AffineEstimate2D:
    """Least-squares local affine map from centered reference to target points."""

    linear: torch.Tensor  # (..., 2, 2), column-vector convention
    translation: torch.Tensor  # (..., 2)
    valid: torch.Tensor  # (...)
    rank: torch.Tensor  # (...)
    design_condition: torch.Tensor  # (...)
    determinant: torch.Tensor  # (...)
    rms_residual: torch.Tensor  # (...)


@dataclass(frozen=True)
class FlowConsistency:
    """Forward/backward center-flow round-trip diagnostics."""

    target_points: torch.Tensor  # (..., 2)
    forward: torch.Tensor  # (..., 2)
    backward: torch.Tensor  # (..., 2)
    roundtrip_error: torch.Tensor  # (...)
    valid: torch.Tensor  # (...)


@dataclass(frozen=True)
class GaussianCorrespondences2D:
    """Flow-warped 2D Gaussians and all correspondence validity diagnostics."""

    means: torch.Tensor  # (..., 2)
    covariances: torch.Tensor  # (..., 2, 2)
    valid: torch.Tensor  # (...)
    sigma_points: SigmaPoints2D
    forward_samples: FlowSamples
    affine: AffineEstimate2D
    consistency: FlowConsistency


@dataclass(frozen=True)
class DltTriangulation:
    """Batched calibrated-DLT result with geometric diagnostics."""

    points_world: torch.Tensor  # (..., 3)
    valid: torch.Tensor  # (...)
    observation_valid: torch.Tensor  # (..., V)
    observation_count: torch.Tensor  # (...)
    depths: torch.Tensor  # (..., V)
    cheiral: torch.Tensor  # (..., V)
    reprojection_error: torch.Tensor  # (..., V)
    max_observed_reprojection_error: torch.Tensor  # (...)
    singular_values: torch.Tensor  # (..., 4), descending
    rank: torch.Tensor  # (...)
    condition_number: torch.Tensor  # (...); s_max / s_third
    nullspace_gap: torch.Tensor  # (...); s_third / s_min
    algebraic_residual: torch.Tensor  # (...); s_min
    homogeneous_w: torch.Tensor  # (...)


def _require_float_tensor(name: str, value: torch.Tensor, trailing_shape: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape[-len(trailing_shape) :] != trailing_shape:
        raise ValueError(f"{name} must end in shape {trailing_shape}")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating-point dtype")


def five_sigma_points(
    means: torch.Tensor,
    covariances: torch.Tensor,
    *,
    min_eigenvalue: float = 0.0,
) -> SigmaPoints2D:
    """Build the center and ``±sqrt(2)`` principal covariance-factor axes.

    For ``covariance = L @ L.T``, the returned order is center, ``+sqrt(2)*L[:,0]``,
    ``-sqrt(2)*L[:,0]``, ``+sqrt(2)*L[:,1]``, and ``-sqrt(2)*L[:,1]``.  ``L`` is the
    symmetric eigensystem factor with the larger-variance axis first.  Invalid, asymmetric, or
    non-positive-definite inputs are marked invalid and receive NaN points rather than being
    silently repaired.
    """
    _require_float_tensor("means", means, (2,))
    _require_float_tensor("covariances", covariances, (2, 2))
    if covariances.shape[:-2] != means.shape[:-1]:
        raise ValueError("means and covariances must have identical leading dimensions")
    if means.device != covariances.device:
        raise ValueError("means and covariances must be on the same device")
    if not math.isfinite(min_eigenvalue) or min_eigenvalue < 0:
        raise ValueError("min_eigenvalue must be finite and non-negative")

    dtype = torch.promote_types(means.dtype, covariances.dtype)
    if dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    means_work = means.to(dtype=dtype)
    covariances_work = covariances.to(dtype=dtype)
    finite = torch.isfinite(means_work).all(dim=-1) & torch.isfinite(covariances_work).all(
        dim=(-2, -1)
    )
    scale = covariances_work.abs().amax(dim=(-2, -1)).clamp_min(1.0)
    symmetry_tolerance = 64.0 * torch.finfo(dtype).eps * scale
    symmetric = (covariances_work - covariances_work.transpose(-1, -2)).abs().amax(
        dim=(-2, -1)
    ) <= symmetry_tolerance

    identity = torch.eye(2, dtype=dtype, device=means.device)
    safe_covariances = torch.where(
        (finite & symmetric)[..., None, None],
        0.5 * (covariances_work + covariances_work.transpose(-1, -2)),
        identity,
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(safe_covariances)
    eigenvalues = eigenvalues.flip(-1)
    eigenvectors = eigenvectors.flip(-1)
    positive_definite = (eigenvalues > min_eigenvalue).all(dim=-1)
    valid = finite & symmetric & positive_definite

    safe_eigenvalues = eigenvalues.clamp_min(0.0)
    factor = eigenvectors * safe_eigenvalues.sqrt().unsqueeze(-2)
    spread = math.sqrt(2.0)
    first_axis = spread * factor[..., :, 0]
    second_axis = spread * factor[..., :, 1]
    points = torch.stack(
        [
            means_work,
            means_work + first_axis,
            means_work - first_axis,
            means_work + second_axis,
            means_work - second_axis,
        ],
        dim=-2,
    )
    nan = torch.full((), torch.nan, dtype=dtype, device=means.device)
    points = torch.where(valid[..., None, None], points, nan)
    factor = torch.where(valid[..., None, None], factor, nan)
    eigenvalues = torch.where(valid[..., None], eigenvalues, nan)
    return SigmaPoints2D(
        points=points,
        factor=factor,
        eigenvalues=eigenvalues,
        valid=valid,
    )


def sample_flow_bilinear(flow: torch.Tensor, points: torch.Tensor) -> FlowSamples:
    """Sample a dense ``(H, W, 2)`` displacement field at half-pixel-center coordinates.

    A point is valid only when it is finite, lies inside the closed image-center rectangle, and
    its interpolated flow is finite.  Invalid values are returned as zero with ``valid=False``;
    unlike the legacy image sampler, this function never turns out-of-frame tracks into
    border-clamped correspondences.
    """
    _require_float_tensor("flow", flow, (2,))
    _require_float_tensor("points", points, (2,))
    if flow.ndim != 3:
        raise ValueError("flow must have shape (H, W, 2)")
    if points.device != flow.device:
        raise ValueError("flow and points must be on the same device")
    height, width = flow.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("flow must have a non-empty spatial domain")

    dtype = torch.promote_types(flow.dtype, points.dtype)
    if dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    flow_work = flow.to(dtype=dtype)
    points_work = points.to(dtype=dtype)
    finite_points = torch.isfinite(points_work).all(dim=-1)
    safe_points = torch.where(
        finite_points[..., None],
        points_work,
        points_work.new_tensor([0.5, 0.5]),
    )
    in_bounds = (
        (safe_points[..., 0] >= 0.5)
        & (safe_points[..., 0] <= width - 0.5)
        & (safe_points[..., 1] >= 0.5)
        & (safe_points[..., 1] <= height - 0.5)
    )

    x = (safe_points[..., 0] - 0.5).clamp(0, width - 1)
    y = (safe_points[..., 1] - 0.5).clamp(0, height - 1)
    x0 = x.floor().long()
    y0 = y.floor().long()
    x1 = (x0 + 1).clamp_max(width - 1)
    y1 = (y0 + 1).clamp_max(height - 1)
    fx = (x - x0.to(dtype)).unsqueeze(-1)
    fy = (y - y0.to(dtype)).unsqueeze(-1)
    v00 = flow_work[y0, x0]
    v01 = flow_work[y0, x1]
    v10 = flow_work[y1, x0]
    v11 = flow_work[y1, x1]
    values = (
        v00 * (1.0 - fx) * (1.0 - fy)
        + v01 * fx * (1.0 - fy)
        + v10 * (1.0 - fx) * fy
        + v11 * fx * fy
    )
    valid = finite_points & in_bounds & torch.isfinite(values).all(dim=-1)
    values = torch.where(valid[..., None], values, torch.zeros_like(values))
    return FlowSamples(values=values, valid=valid)


def solve_local_affine(
    reference_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    sample_valid: torch.Tensor | None = None,
    rcond: float | None = None,
    min_abs_determinant: float = 1e-8,
) -> AffineEstimate2D:
    """Solve ``target_delta ~= reference_delta @ linear.T`` around point zero.

    The first point is the center and defines translation.  Remaining valid points form the
    least-squares design.  The batched SVD solution handles missing samples explicitly and reports
    rank/conditioning instead of relying on a backend-specific ``lstsq`` driver.
    """
    _require_float_tensor("reference_points", reference_points, (2,))
    _require_float_tensor("target_points", target_points, (2,))
    if reference_points.shape != target_points.shape:
        raise ValueError("reference_points and target_points must have the same shape")
    if reference_points.ndim < 2 or reference_points.shape[-2] < 3:
        raise ValueError("at least three 2D points, including the center, are required")
    if reference_points.device != target_points.device:
        raise ValueError("reference_points and target_points must be on the same device")
    if sample_valid is not None:
        if sample_valid.shape != reference_points.shape[:-1] or sample_valid.dtype != torch.bool:
            raise ValueError("sample_valid must be a bool tensor matching the point dimensions")
        if sample_valid.device != reference_points.device:
            raise ValueError("sample_valid and points must be on the same device")
    if rcond is not None and (not math.isfinite(rcond) or rcond <= 0):
        raise ValueError("rcond must be finite and positive when provided")
    if not math.isfinite(min_abs_determinant) or min_abs_determinant < 0:
        raise ValueError("min_abs_determinant must be finite and non-negative")

    dtype = torch.promote_types(reference_points.dtype, target_points.dtype)
    if dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    reference = reference_points.to(dtype=dtype)
    target = target_points.to(dtype=dtype)
    point_finite = torch.isfinite(reference).all(dim=-1) & torch.isfinite(target).all(dim=-1)
    row_valid = point_finite if sample_valid is None else point_finite & sample_valid
    center_valid = row_valid[..., 0]
    safe_reference = torch.where(point_finite[..., None], reference, torch.zeros_like(reference))
    safe_target = torch.where(point_finite[..., None], target, torch.zeros_like(target))
    reference_delta = safe_reference[..., 1:, :] - safe_reference[..., :1, :]
    target_delta = safe_target[..., 1:, :] - safe_target[..., :1, :]
    design_valid = row_valid[..., 1:] & center_valid[..., None]
    design = torch.where(
        design_valid[..., None],
        reference_delta,
        torch.zeros_like(reference_delta),
    )
    response = torch.where(design_valid[..., None], target_delta, torch.zeros_like(target_delta))

    u, singular_values, vh = torch.linalg.svd(design, full_matrices=False)
    max_singular = singular_values[..., 0]
    if rcond is None:
        tolerance = max(reference_points.shape[-2] - 1, 2) * torch.finfo(dtype).eps * max_singular
    else:
        tolerance = rcond * max_singular
    nonzero = singular_values > tolerance[..., None]
    rank = nonzero.sum(dim=-1)
    inverse = torch.where(nonzero, singular_values.reciprocal(), torch.zeros_like(singular_values))
    projected_response = u.transpose(-1, -2) @ response
    solution = vh.transpose(-1, -2) @ (inverse[..., :, None] * projected_response)
    linear = solution.transpose(-1, -2)
    translation = safe_target[..., 0, :] - safe_reference[..., 0, :]
    residual = design @ solution - response
    residual_squared = residual.square().sum(dim=-1)
    valid_row_count = design_valid.sum(dim=-1).clamp_min(1)
    rms_residual = (residual_squared.sum(dim=-1) / valid_row_count).sqrt()
    smallest_singular = singular_values[..., -1]
    design_condition = torch.where(
        smallest_singular > tolerance,
        max_singular / smallest_singular,
        torch.full_like(max_singular, torch.inf),
    )
    determinant = torch.linalg.det(linear)
    valid = (
        center_valid
        & (rank == 2)
        & torch.isfinite(linear).all(dim=(-2, -1))
        & torch.isfinite(translation).all(dim=-1)
        & torch.isfinite(determinant)
        & (determinant.abs() > min_abs_determinant)
    )
    linear = torch.where(valid[..., None, None], linear, torch.zeros_like(linear))
    translation = torch.where(valid[..., None], translation, torch.zeros_like(translation))
    return AffineEstimate2D(
        linear=linear,
        translation=translation,
        valid=valid,
        rank=rank,
        design_condition=design_condition,
        determinant=determinant,
        rms_residual=rms_residual,
    )


def warp_gaussians_2d(
    means: torch.Tensor,
    covariances: torch.Tensor,
    linear: torch.Tensor,
    translation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``mean' = mean + translation`` and ``cov' = A cov A.T``."""
    _require_float_tensor("means", means, (2,))
    _require_float_tensor("covariances", covariances, (2, 2))
    _require_float_tensor("linear", linear, (2, 2))
    _require_float_tensor("translation", translation, (2,))
    leading = means.shape[:-1]
    if (
        covariances.shape[:-2] != leading
        or linear.shape[:-2] != leading
        or translation.shape[:-1] != leading
    ):
        raise ValueError("all Gaussian warp tensors must have identical leading dimensions")
    devices = {means.device, covariances.device, linear.device, translation.device}
    if len(devices) != 1:
        raise ValueError("all Gaussian warp tensors must be on the same device")
    warped_means = means + translation
    warped_covariances = linear @ covariances @ linear.transpose(-1, -2)
    warped_covariances = 0.5 * (warped_covariances + warped_covariances.transpose(-1, -2))
    return warped_means, warped_covariances


def forward_backward_consistency(
    reference_points: torch.Tensor,
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    *,
    max_error: float = 3.0,
) -> FlowConsistency:
    """Check standard center round trips ``f(x) + b(x + f(x))``."""
    _require_float_tensor("reference_points", reference_points, (2,))
    if not math.isfinite(max_error) or max_error < 0:
        raise ValueError("max_error must be finite and non-negative")
    forward = sample_flow_bilinear(forward_flow, reference_points)
    target_points = reference_points.to(forward.values) + forward.values
    backward = sample_flow_bilinear(backward_flow, target_points)
    roundtrip = forward.values + backward.values
    error = roundtrip.norm(dim=-1)
    valid = forward.valid & backward.valid & torch.isfinite(error) & (error <= max_error)
    error = torch.where(
        forward.valid & backward.valid,
        error,
        torch.full_like(error, torch.inf),
    )
    return FlowConsistency(
        target_points=target_points,
        forward=forward.values,
        backward=backward.values,
        roundtrip_error=error,
        valid=valid,
    )


def track_gaussians_2d(
    means: torch.Tensor,
    covariances: torch.Tensor,
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    *,
    max_roundtrip_error: float = 3.0,
    require_all_sigma_points: bool = True,
) -> GaussianCorrespondences2D:
    """Track 2D Gaussians through caller-supplied flow without loading an RGB model."""
    sigma_points = five_sigma_points(means, covariances)
    forward_samples = sample_flow_bilinear(forward_flow, sigma_points.points)
    target_sigma_points = sigma_points.points.to(forward_samples.values) + forward_samples.values
    affine = solve_local_affine(
        sigma_points.points,
        target_sigma_points,
        sample_valid=forward_samples.valid,
    )
    warped_means, warped_covariances = warp_gaussians_2d(
        means.to(affine.linear),
        covariances.to(affine.linear),
        affine.linear,
        affine.translation,
    )
    consistency = forward_backward_consistency(
        means,
        forward_flow,
        backward_flow,
        max_error=max_roundtrip_error,
    )
    support_valid = (
        forward_samples.valid.all(dim=-1)
        if require_all_sigma_points
        else (forward_samples.valid.sum(dim=-1) >= 3)
    )
    valid = sigma_points.valid & support_valid & affine.valid & consistency.valid
    return GaussianCorrespondences2D(
        means=warped_means,
        covariances=warped_covariances,
        valid=valid,
        sigma_points=sigma_points,
        forward_samples=forward_samples,
        affine=affine,
        consistency=consistency,
    )


def _camera_matrices(
    cameras: Sequence[Camera],
    like: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rotations = []
    translations = []
    intrinsics = []
    for view_index, camera in enumerate(cameras):
        scalar_values = (camera.fx, camera.fy, camera.cx, camera.cy)
        if not all(math.isfinite(value) for value in scalar_values):
            raise ValueError(f"camera {view_index} has non-finite intrinsics")
        if camera.fx == 0 or camera.fy == 0:
            raise ValueError(f"camera {view_index} has a zero focal length")
        rotation = camera.R.to(like)
        translation = camera.t.to(like)
        if not torch.isfinite(rotation).all() or not torch.isfinite(translation).all():
            raise ValueError(f"camera {view_index} has non-finite extrinsics")
        rotations.append(rotation)
        translations.append(translation)
        intrinsics.append(like.new_tensor([camera.fx, camera.fy, camera.cx, camera.cy]))
    return (
        torch.stack(rotations),
        torch.stack(translations),
        torch.stack(intrinsics),
    )


def triangulate_centers_dlt(
    cameras: Sequence[Camera],
    image_points: torch.Tensor,
    *,
    observation_mask: torch.Tensor | None = None,
    require_in_image: bool = True,
    min_views: int = 2,
    min_depth: float = 0.0,
    max_condition_number: float = 1e8,
    max_reprojection_error: float | None = None,
    homogeneous_epsilon: float | None = None,
) -> DltTriangulation:
    """Triangulate batched corresponding centers with calibrated linear DLT.

    ``image_points`` has shape ``(..., V, 2)`` for ``V == len(cameras)``.  Invalid or masked
    observations contribute zero rows and are excluded from all acceptance diagnostics.  The DLT
    equations use normalized camera coordinates and world-to-camera ``[R | t]`` matrices, exactly
    matching :class:`Camera`'s OpenCV ``+z``-forward convention.

    ``condition_number`` is ``s_max / s_third``: it measures whether the three constrained
    homogeneous directions are supported without treating the desired one-dimensional nullspace as
    a failure.  ``nullspace_gap = s_third / s_min`` separately reports how isolated that solution
    is.
    """
    if len(cameras) < 2:
        raise ValueError("DLT triangulation requires at least two cameras")
    _require_float_tensor("image_points", image_points, (len(cameras), 2))
    if image_points.ndim < 2:
        raise ValueError("image_points must have shape (..., V, 2)")
    if observation_mask is not None:
        mask_has_wrong_shape = observation_mask.shape != image_points.shape[:-1]
        if mask_has_wrong_shape or observation_mask.dtype != torch.bool:
            raise ValueError("observation_mask must be a bool tensor with shape (..., V)")
        if observation_mask.device != image_points.device:
            raise ValueError("observation_mask and image_points must be on the same device")
    if not 2 <= min_views <= len(cameras):
        raise ValueError("min_views must lie between two and the camera count")
    if not math.isfinite(min_depth):
        raise ValueError("min_depth must be finite")
    if not math.isfinite(max_condition_number) or max_condition_number <= 0:
        raise ValueError("max_condition_number must be finite and positive")
    if max_reprojection_error is not None and (
        not math.isfinite(max_reprojection_error) or max_reprojection_error < 0
    ):
        raise ValueError("max_reprojection_error must be finite and non-negative")

    dtype = image_points.dtype
    if dtype in {torch.float16, torch.bfloat16}:
        dtype = torch.float32
    points = image_points.to(dtype=dtype)
    if homogeneous_epsilon is None:
        homogeneous_epsilon = 64.0 * torch.finfo(dtype).eps
    if not math.isfinite(homogeneous_epsilon) or homogeneous_epsilon <= 0:
        raise ValueError("homogeneous_epsilon must be finite and positive")

    rotations, translations, intrinsics = _camera_matrices(cameras, points)
    finite = torch.isfinite(points).all(dim=-1)
    observation_valid = finite
    if observation_mask is not None:
        observation_valid = observation_valid & observation_mask
    if require_in_image:
        in_image = []
        for view_index, camera in enumerate(cameras):
            uv = points[..., view_index, :]
            in_image.append(
                (uv[..., 0] >= 0.5)
                & (uv[..., 0] <= camera.width - 0.5)
                & (uv[..., 1] >= 0.5)
                & (uv[..., 1] <= camera.height - 0.5)
            )
        observation_valid = observation_valid & torch.stack(in_image, dim=-1)

    safe_points = torch.where(
        finite[..., None],
        points,
        points.new_zeros(()).expand_as(points),
    )
    normalized_x = (safe_points[..., 0] - intrinsics[:, 2]) / intrinsics[:, 0]
    normalized_y = (safe_points[..., 1] - intrinsics[:, 3]) / intrinsics[:, 1]
    projection = torch.cat([rotations, translations[..., None]], dim=-1)
    first_rows = normalized_x[..., None] * projection[:, 2] - projection[:, 0]
    second_rows = normalized_y[..., None] * projection[:, 2] - projection[:, 1]
    design = torch.stack([first_rows, second_rows], dim=-2)
    design = torch.where(
        observation_valid[..., None, None],
        design,
        torch.zeros_like(design),
    )
    row_norm = design.norm(dim=-1, keepdim=True)
    design = torch.where(row_norm > 0, design / row_norm.clamp_min(1e-30), design)
    design = design.flatten(start_dim=-3, end_dim=-2)

    _, singular_values, vh = torch.linalg.svd(design, full_matrices=False)
    homogeneous = vh[..., -1, :]
    homogeneous_w = homogeneous[..., 3]
    safe_w = torch.where(
        homogeneous_w.abs() > homogeneous_epsilon,
        homogeneous_w,
        torch.ones_like(homogeneous_w),
    )
    points_world = homogeneous[..., :3] / safe_w[..., None]
    finite_solution = torch.isfinite(points_world).all(dim=-1)

    max_singular = singular_values[..., 0]
    tolerance = max(2 * len(cameras), 4) * torch.finfo(dtype).eps * max_singular
    rank = (singular_values > tolerance[..., None]).sum(dim=-1)
    third_singular = singular_values[..., -2]
    smallest_singular = singular_values[..., -1]
    condition_number = torch.where(
        third_singular > tolerance,
        max_singular / third_singular,
        torch.full_like(max_singular, torch.inf),
    )
    nullspace_gap = torch.where(
        smallest_singular > tolerance,
        third_singular / smallest_singular,
        torch.full_like(third_singular, torch.inf),
    )

    camera_points = torch.einsum("vij,...j->...vi", rotations, points_world) + translations
    depths = camera_points[..., 2]
    safe_depths = torch.where(
        depths.abs() > torch.finfo(dtype).tiny,
        depths,
        torch.full_like(depths, torch.finfo(dtype).tiny),
    )
    projected_u = intrinsics[:, 0] * camera_points[..., 0] / safe_depths + intrinsics[:, 2]
    projected_v = intrinsics[:, 1] * camera_points[..., 1] / safe_depths + intrinsics[:, 3]
    projected = torch.stack([projected_u, projected_v], dim=-1)
    reprojection_error = (projected - safe_points).norm(dim=-1)
    reprojection_error = torch.where(
        finite_solution[..., None] & finite,
        reprojection_error,
        torch.full_like(reprojection_error, torch.inf),
    )
    cheiral = torch.isfinite(depths) & (depths > min_depth)
    all_observed_cheiral = (cheiral | ~observation_valid).all(dim=-1)
    observed_error = torch.where(
        observation_valid,
        reprojection_error,
        torch.zeros_like(reprojection_error),
    )
    max_observed_error = observed_error.amax(dim=-1)
    observation_count = observation_valid.sum(dim=-1)
    valid = (
        (observation_count >= min_views)
        & (rank >= 3)
        & (homogeneous_w.abs() > homogeneous_epsilon)
        & finite_solution
        & torch.isfinite(condition_number)
        & (condition_number <= max_condition_number)
        & all_observed_cheiral
    )
    if max_reprojection_error is not None:
        valid = valid & (max_observed_error <= max_reprojection_error)

    return DltTriangulation(
        points_world=points_world,
        valid=valid,
        observation_valid=observation_valid,
        observation_count=observation_count,
        depths=depths,
        cheiral=cheiral,
        reprojection_error=reprojection_error,
        max_observed_reprojection_error=max_observed_error,
        singular_values=singular_values,
        rank=rank,
        condition_number=condition_number,
        nullspace_gap=nullspace_gap,
        algebraic_residual=smallest_singular,
        homogeneous_w=homogeneous_w,
    )


__all__ = [
    "AffineEstimate2D",
    "DltTriangulation",
    "FlowConsistency",
    "FlowSamples",
    "GaussianCorrespondences2D",
    "SigmaPoints2D",
    "five_sigma_points",
    "forward_backward_consistency",
    "sample_flow_bilinear",
    "solve_local_affine",
    "track_gaussians_2d",
    "triangulate_centers_dlt",
    "warp_gaussians_2d",
]
