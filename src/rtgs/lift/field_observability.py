"""Linear mean/covariance recovery and observability from calibrated projections.

Every calibrated 2D center contributes two linear equations for the world mean.  Once that
mean is fixed, every view contributes three linear equations
``C_v = A_v Sigma A_v^T`` for the six unique entries of the symmetric world covariance.
Two generic views have rank five, not six.  This module reports that null space explicitly and
refuses an underdetermined solve unless the caller supplies a covariance prior that fixes it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from rtgs.core.camera import Camera
from rtgs.lift.inverse_projection_fiber import covariance_projection_design


def covariance_to_vector(covariance: torch.Tensor) -> torch.Tensor:
    """Map symmetric ``(...,3,3)`` covariance matrices to ``(...,6)`` design coordinates."""

    if covariance.shape[-2:] != (3, 3):
        raise ValueError("covariance must end in shape (3,3)")
    if not covariance.is_floating_point() or not bool(torch.isfinite(covariance).all()):
        raise ValueError("covariance must be finite and floating point")
    scale = covariance.detach().abs().amax().clamp_min(1.0)
    tolerance = 64.0 * torch.finfo(covariance.dtype).eps * scale
    if float((covariance - covariance.transpose(-1, -2)).abs().amax().detach()) > float(tolerance):
        raise ValueError("covariance must be symmetric")
    return torch.stack(
        [
            covariance[..., 0, 0],
            covariance[..., 1, 1],
            covariance[..., 2, 2],
            covariance[..., 0, 1],
            covariance[..., 0, 2],
            covariance[..., 1, 2],
        ],
        dim=-1,
    )


def vector_to_covariance(vector: torch.Tensor) -> torch.Tensor:
    """Invert :func:`covariance_to_vector` for tensors ending in shape ``(6,)``."""

    if vector.shape[-1:] != (6,):
        raise ValueError("vector must end in shape (6,)")
    if not vector.is_floating_point() or not bool(torch.isfinite(vector).all()):
        raise ValueError("vector must be finite and floating point")
    result = vector.new_zeros((*vector.shape[:-1], 3, 3))
    result[..., 0, 0] = vector[..., 0]
    result[..., 1, 1] = vector[..., 1]
    result[..., 2, 2] = vector[..., 2]
    result[..., 0, 1] = result[..., 1, 0] = vector[..., 3]
    result[..., 0, 2] = result[..., 2, 0] = vector[..., 4]
    result[..., 1, 2] = result[..., 2, 1] = vector[..., 5]
    return result


def _view_weights(
    weights: torch.Tensor | None,
    *,
    count: int,
    like: torch.Tensor,
) -> torch.Tensor:
    if weights is None:
        result = like.new_ones(count)
    else:
        result = torch.as_tensor(weights, dtype=like.dtype, device=like.device)
    if result.shape != (count,):
        raise ValueError("view_weights must have shape (V,)")
    if not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
        raise ValueError("view_weights must be finite and non-negative")
    if not bool((result > 0).any()):
        raise ValueError("at least one view weight must be positive")
    return result


@dataclass(frozen=True)
class CovarianceObservabilityReport:
    """SVD diagnostics for one weighted projected-covariance design."""

    design: torch.Tensor
    weighted_design: torch.Tensor
    view_weights: torch.Tensor
    singular_values: torch.Tensor
    rank: int
    tolerance: float
    condition_number: float
    observable_condition_number: float
    null_basis: torch.Tensor

    @property
    def full_rank(self) -> bool:
        return self.rank == 6


@dataclass(frozen=True)
class CovarianceSolveResult:
    """Prior-aware weighted solve followed by a symmetric PSD projection."""

    covariance: torch.Tensor
    raw_covariance: torch.Tensor
    coefficients: torch.Tensor
    raw_coefficients: torch.Tensor
    projected_covariances: torch.Tensor
    weighted_residual_norm: torch.Tensor
    report: CovarianceObservabilityReport
    used_null_prior: bool


@dataclass(frozen=True)
class MeanTriangulationReport:
    """SVD diagnostics for one weighted calibrated-center design."""

    design: torch.Tensor
    target: torch.Tensor
    weighted_design: torch.Tensor
    weighted_target: torch.Tensor
    view_weights: torch.Tensor
    singular_values: torch.Tensor
    rank: int
    tolerance: float
    condition_number: float
    null_basis: torch.Tensor

    @property
    def full_rank(self) -> bool:
        return self.rank == 3


@dataclass(frozen=True)
class MeanTriangulationResult:
    """World mean recovered from calibrated centers with geometric diagnostics."""

    mean: torch.Tensor
    projected_means: torch.Tensor
    depths: torch.Tensor
    reprojection_errors: torch.Tensor
    weighted_residual_norm: torch.Tensor
    report: MeanTriangulationReport


def triangulate_projected_mean(
    projected_means: torch.Tensor,
    cameras: Sequence[Camera],
    *,
    view_weights: torch.Tensor | None = None,
    rcond: float | None = None,
) -> MeanTriangulationResult:
    """Triangulate one world mean by weighted calibrated linear least squares.

    ``projected_means[v]`` is a pixel center in camera ``v``.  The two rows per view are
    formed in normalized camera coordinates from the stored world-to-camera transform:

    ``(R[0] - x_n R[2]) X = -(t[0] - x_n t[2])`` and likewise for ``y_n``.

    This inhomogeneous form exposes the three-dimensional observability directly.  It is
    deterministic, supports confidence weights, and rejects a rank-deficient camera layout
    instead of returning an arbitrary point on the unresolved ray/null space.
    """

    views = len(cameras)
    if views < 2:
        raise ValueError("mean triangulation requires at least two cameras")
    if projected_means.shape != (views, 2):
        raise ValueError("projected_means must have shape (V,2)")
    if not projected_means.is_floating_point() or not bool(torch.isfinite(projected_means).all()):
        raise ValueError("projected_means must be finite and floating point")
    weights = _view_weights(view_weights, count=views, like=projected_means)

    rows: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for view_index, camera in enumerate(cameras):
        intrinsics = (camera.fx, camera.fy, camera.cx, camera.cy)
        if not all(math.isfinite(value) for value in intrinsics):
            raise ValueError(f"camera {view_index} has non-finite intrinsics")
        if camera.fx == 0 or camera.fy == 0:
            raise ValueError(f"camera {view_index} has a zero focal length")
        rotation = camera.R.to(projected_means)
        translation = camera.t.to(projected_means)
        if not bool(torch.isfinite(rotation).all()) or not bool(torch.isfinite(translation).all()):
            raise ValueError(f"camera {view_index} has non-finite extrinsics")
        x_normalized = (projected_means[view_index, 0] - camera.cx) / camera.fx
        y_normalized = (projected_means[view_index, 1] - camera.cy) / camera.fy
        rows.extend(
            [
                rotation[0] - x_normalized * rotation[2],
                rotation[1] - y_normalized * rotation[2],
            ]
        )
        targets.extend(
            [
                -(translation[0] - x_normalized * translation[2]),
                -(translation[1] - y_normalized * translation[2]),
            ]
        )

    design = torch.stack(rows)
    target = torch.stack(targets)
    row_scale = weights.sqrt().repeat_interleave(2)
    weighted_design = design * row_scale[:, None]
    weighted_target = target * row_scale
    u, singular_values, vh = torch.linalg.svd(weighted_design, full_matrices=True)
    largest = singular_values[0] if singular_values.numel() else projected_means.new_tensor(0.0)
    if rcond is None:
        relative_tolerance = max(weighted_design.shape) * torch.finfo(weighted_design.dtype).eps
    else:
        relative_tolerance = float(rcond)
        if not math.isfinite(relative_tolerance) or relative_tolerance < 0:
            raise ValueError("rcond must be finite and non-negative")
    tolerance_tensor = largest * relative_tolerance
    rank = int((singular_values > tolerance_tensor).sum())
    null_basis = vh[rank:].transpose(0, 1)
    condition_number = (
        float((singular_values[0] / singular_values[-1]).detach()) if rank == 3 else math.inf
    )
    report = MeanTriangulationReport(
        design=design,
        target=target,
        weighted_design=weighted_design,
        weighted_target=weighted_target,
        view_weights=weights,
        singular_values=singular_values,
        rank=rank,
        tolerance=float(tolerance_tensor.detach()),
        condition_number=condition_number,
        null_basis=null_basis,
    )
    if not report.full_rank:
        raise ValueError("projected mean design is rank deficient")

    u_reduced = u[:, :3]
    v_reduced = vh[:3].transpose(0, 1)
    mean = v_reduced @ ((u_reduced.transpose(0, 1) @ weighted_target) / singular_values[:3])
    projected: list[torch.Tensor] = []
    depths: list[torch.Tensor] = []
    for camera in cameras:
        pixels, depth = camera.project(mean[None])
        projected.append(pixels[0])
        depths.append(depth[0])
    fitted_means = torch.stack(projected)
    fitted_depths = torch.stack(depths)
    reprojection_errors = torch.linalg.vector_norm(fitted_means - projected_means, dim=-1)
    residual = weighted_design @ mean - weighted_target
    return MeanTriangulationResult(
        mean=mean,
        projected_means=fitted_means,
        depths=fitted_depths,
        reprojection_errors=reprojection_errors,
        weighted_residual_norm=torch.linalg.vector_norm(residual),
        report=report,
    )


def analyze_covariance_observability(
    mean: torch.Tensor,
    cameras: Sequence[Camera],
    *,
    view_weights: torch.Tensor | None = None,
    rcond: float | None = None,
) -> CovarianceObservabilityReport:
    """Report rank, spectrum, condition, and right null basis for one fixed mean."""

    if not cameras:
        raise ValueError("at least one camera is required")
    if mean.shape != (3,) or not mean.is_floating_point() or not bool(torch.isfinite(mean).all()):
        raise ValueError("mean must be a finite floating tensor with shape (3,)")
    weights = _view_weights(view_weights, count=len(cameras), like=mean)
    design = covariance_projection_design(mean.expand(len(cameras), 3), cameras)
    row_scale = weights.sqrt().repeat_interleave(3)
    weighted = design * row_scale[:, None]
    _u, singular_values, vh = torch.linalg.svd(weighted, full_matrices=True)
    largest = singular_values[0] if singular_values.numel() else mean.new_tensor(0.0)
    if rcond is None:
        relative_tolerance = max(weighted.shape) * torch.finfo(weighted.dtype).eps
    else:
        relative_tolerance = float(rcond)
        if not math.isfinite(relative_tolerance) or relative_tolerance < 0:
            raise ValueError("rcond must be finite and non-negative")
    tolerance_tensor = largest * relative_tolerance
    rank = int((singular_values > tolerance_tensor).sum())
    null_basis = vh[rank:].transpose(0, 1)
    if rank:
        observable_condition = float((singular_values[0] / singular_values[rank - 1]).detach())
    else:
        observable_condition = math.inf
    condition = observable_condition if rank == 6 else math.inf
    return CovarianceObservabilityReport(
        design=design,
        weighted_design=weighted,
        view_weights=weights,
        singular_values=singular_values,
        rank=rank,
        tolerance=float(tolerance_tensor.detach()),
        condition_number=condition,
        observable_condition_number=observable_condition,
        null_basis=null_basis,
    )


def _dilations(
    dilation: torch.Tensor | float,
    *,
    count: int,
    like: torch.Tensor,
) -> torch.Tensor:
    result = torch.as_tensor(dilation, dtype=like.dtype, device=like.device)
    try:
        result = result.expand(count)
    except RuntimeError as error:
        raise ValueError("dilation must be scalar or have shape (V,)") from error
    if result.shape != (count,):
        raise ValueError("dilation must be scalar or have shape (V,)")
    if not bool(torch.isfinite(result).all()) or bool((result < 0).any()):
        raise ValueError("dilation must be finite and non-negative")
    return result


def _project_psd(covariance: torch.Tensor, minimum_eigenvalue: float) -> torch.Tensor:
    floor = float(minimum_eigenvalue)
    if not math.isfinite(floor) or floor < 0:
        raise ValueError("minimum_eigenvalue must be finite and non-negative")
    symmetric = 0.5 * (covariance + covariance.transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    projected = (eigenvectors * eigenvalues.clamp_min(floor).unsqueeze(-2)) @ eigenvectors.T
    return 0.5 * (projected + projected.T)


def solve_projected_covariance(
    mean: torch.Tensor,
    cameras: Sequence[Camera],
    projected_covariances: torch.Tensor,
    *,
    view_weights: torch.Tensor | None = None,
    dilation: torch.Tensor | float = 0.0,
    prior_covariance: torch.Tensor | None = None,
    minimum_eigenvalue: float = 0.0,
    rcond: float | None = None,
) -> CovarianceSolveResult:
    """Recover one world covariance by weighted linear least squares.

    ``projected_covariances`` may include a known isotropic EWA dilation, which is removed
    before solving and restored in the returned fitted projections.  A rank-deficient design
    raises unless ``prior_covariance`` is supplied; in that case only the design null coordinates
    are copied from the prior.  The final covariance is projected onto the PSD cone.
    """

    views = len(cameras)
    if projected_covariances.shape != (views, 2, 2):
        raise ValueError("projected_covariances must have shape (V,2,2)")
    if (
        not projected_covariances.is_floating_point()
        or projected_covariances.device != mean.device
        or projected_covariances.dtype != mean.dtype
        or not bool(torch.isfinite(projected_covariances).all())
    ):
        raise ValueError("projected covariances must be finite and share mean dtype/device")
    scale = projected_covariances.detach().abs().amax().clamp_min(1.0)
    tolerance = 64.0 * torch.finfo(projected_covariances.dtype).eps * scale
    symmetry_error = (projected_covariances - projected_covariances.transpose(-1, -2)).abs().amax()
    if float(symmetry_error.detach()) > float(tolerance):
        raise ValueError("projected_covariances must be symmetric")

    report = analyze_covariance_observability(
        mean,
        cameras,
        view_weights=view_weights,
        rcond=rcond,
    )
    dilation_values = _dilations(dilation, count=views, like=mean)
    intrinsic = projected_covariances.clone()
    intrinsic[:, 0, 0] = intrinsic[:, 0, 0] - dilation_values
    intrinsic[:, 1, 1] = intrinsic[:, 1, 1] - dilation_values
    target = torch.stack(
        [intrinsic[:, 0, 0], intrinsic[:, 0, 1], intrinsic[:, 1, 1]],
        dim=-1,
    ).reshape(-1)
    weighted_target = target * report.view_weights.sqrt().repeat_interleave(3)

    _u, singular_values, vh = torch.linalg.svd(report.weighted_design, full_matrices=True)
    rank = report.rank
    if rank:
        u_reduced = _u[:, :rank]
        v_reduced = vh[:rank].transpose(0, 1)
        raw_coefficients = v_reduced @ (
            (u_reduced.transpose(0, 1) @ weighted_target) / singular_values[:rank]
        )
    else:
        raw_coefficients = mean.new_zeros(6)

    used_prior = False
    if not report.full_rank:
        if prior_covariance is None:
            raise ValueError(
                "projected covariance design is rank deficient; an explicit prior is required"
            )
        if (
            prior_covariance.shape != (3, 3)
            or prior_covariance.device != mean.device
            or prior_covariance.dtype != mean.dtype
        ):
            raise ValueError("prior_covariance must have shape (3,3) and share mean dtype/device")
        prior_vector = covariance_to_vector(prior_covariance)
        raw_coefficients = raw_coefficients + report.null_basis @ (
            report.null_basis.transpose(0, 1) @ (prior_vector - raw_coefficients)
        )
        used_prior = True

    raw_covariance = vector_to_covariance(raw_coefficients)
    covariance = _project_psd(raw_covariance, minimum_eigenvalue)
    coefficients = covariance_to_vector(covariance)
    fitted_intrinsic = report.design @ coefficients
    fitted_rows = fitted_intrinsic.reshape(views, 3)
    fitted = mean.new_zeros((views, 2, 2))
    fitted[:, 0, 0] = fitted_rows[:, 0] + dilation_values
    fitted[:, 0, 1] = fitted[:, 1, 0] = fitted_rows[:, 1]
    fitted[:, 1, 1] = fitted_rows[:, 2] + dilation_values
    weighted_residual = (
        report.design @ coefficients - target
    ) * report.view_weights.sqrt().repeat_interleave(3)
    return CovarianceSolveResult(
        covariance=covariance,
        raw_covariance=raw_covariance,
        coefficients=coefficients,
        raw_coefficients=raw_coefficients,
        projected_covariances=fitted,
        weighted_residual_norm=torch.linalg.vector_norm(weighted_residual),
        report=report,
        used_null_prior=used_prior,
    )


__all__ = [
    "CovarianceObservabilityReport",
    "CovarianceSolveResult",
    "MeanTriangulationReport",
    "MeanTriangulationResult",
    "analyze_covariance_observability",
    "covariance_to_vector",
    "solve_projected_covariance",
    "triangulate_projected_mean",
    "vector_to_covariance",
]
