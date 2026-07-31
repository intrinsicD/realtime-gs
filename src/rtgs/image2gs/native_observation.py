"""Freeze native additive Stage-1 fits as lossless compact observation fields."""

from __future__ import annotations

import math

import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.image2gs.renderer2d import render_gaussians_2d

_NATIVE_SIGMA_CUTOFF = math.sqrt(12.0)


def native_gaussians_to_observation(
    gaussians: Gaussians2D,
    *,
    canvas_size: tuple[int, int],
    fit_window: tuple[int, int, int, int] | None = None,
    view_id: str,
    n_init: int | None = None,
    producer_version: str | None = None,
    producer_source_digest: str | None = None,
    fit_config_digest: str | None = None,
) -> GaussianObservationField:
    """Convert a crop-local native fit into an additive frozen field.

    ``gaussians.xy`` uses the native renderer's half-integer pixel convention in the local
    ``fit_window``.  The stored means use full-canvas coordinates, while ``mean_residuals`` keeps
    the exact crop-local float32 coordinates recoverable after adding a large native-resolution
    offset.

    The observation keeps the native renderer's additive ``weight * color`` factorization,
    zero support fade, zero AA dilation, and its :math:`q < 12` support radius.  The compact
    observation query uses the repository's rounded-AABB field contract; this conversion does
    not claim bit-exact replay of CUDA's hard elliptical cutoff at support-boundary pixels.
    """

    canvas_height, canvas_width = canvas_size
    if canvas_height <= 0 or canvas_width <= 0:
        raise ValueError("canvas_size must contain positive height and width")
    if fit_window is None:
        fit_window = (0, 0, canvas_width, canvas_height)
    if len(fit_window) != 4:
        raise ValueError("fit_window must contain x, y, width, and height")
    fit_x, fit_y, fit_width, fit_height = fit_window
    if (
        fit_x < 0
        or fit_y < 0
        or fit_width <= 0
        or fit_height <= 0
        or fit_x + fit_width > canvas_width
        or fit_y + fit_height > canvas_height
    ):
        raise ValueError("fit_window must lie inside canvas_size")
    if gaussians.n <= 0:
        raise ValueError("native observations require at least one Gaussian")
    tensors = (gaussians.xy, gaussians.chol, gaussians.color, gaussians.weight)
    if any(not tensor.is_floating_point() for tensor in tensors):
        raise TypeError("native Gaussian tensors must be floating point")
    if any(
        tensor.device != gaussians.xy.device or tensor.dtype != gaussians.xy.dtype
        for tensor in tensors
    ):
        raise ValueError("native Gaussian tensors must share dtype and device")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors):
        raise ValueError("native Gaussian tensors must be finite")
    if bool((gaussians.chol[:, (0, 2)] <= 0).any()):
        raise ValueError("native Gaussian Cholesky diagonals must be positive")
    if bool(((gaussians.color < 0) | (gaussians.color > 1)).any()):
        raise ValueError("native Gaussian colors must lie in [0, 1]")
    if bool(((gaussians.weight < 0) | (gaussians.weight > 1)).any()):
        raise ValueError("native Gaussian weights must lie in [0, 1]")
    local_bounds = (
        (gaussians.xy[:, 0] >= 0)
        & (gaussians.xy[:, 0] < fit_width)
        & (gaussians.xy[:, 1] >= 0)
        & (gaussians.xy[:, 1] < fit_height)
    )
    if not bool(local_bounds.all()):
        raise ValueError("native Gaussian means must lie inside the local fit_window")

    covariance = gaussians.covariance()
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    if bool((eigenvalues <= 0).any()):
        raise ValueError("native Gaussian covariance must be positive definite")
    first_axis = eigenvectors[:, :, 0]
    rotations = torch.atan2(first_axis[:, 1], first_axis[:, 0])

    offset = gaussians.xy.new_tensor([fit_x, fit_y])
    native_means = gaussians.xy + offset
    mean_residuals = None
    if gaussians.xy.dtype == torch.float32:
        provider_origin = gaussians.xy.new_tensor([fit_x + 0.5, fit_y + 0.5])
        provider_local_means = gaussians.xy - 0.5
        mean_residuals = provider_local_means - (native_means - provider_origin)

    return GaussianObservationField(
        width=canvas_width,
        height=canvas_height,
        means=native_means,
        log_scales=0.5 * eigenvalues.log(),
        rotations=rotations,
        colors=gaussians.color,
        amplitudes=gaussians.weight,
        mean_residuals=mean_residuals,
        blend_mode="additive",
        sigma_cutoff=_NATIVE_SIGMA_CUTOFF,
        support_fade_alpha=0.0,
        aa_dilation=0.0,
        view_id=view_id,
        fit_window=fit_window,
        n_init=gaussians.n if n_init is None else n_init,
        provider="native",
        producer_version=producer_version,
        producer_source_digest=producer_source_digest,
        fit_config_digest=fit_config_digest,
    )


def native_observation_to_gaussians(
    field: GaussianObservationField,
) -> Gaussians2D:
    """Recover the native additive GaussianImage-style field in crop coordinates.

    The compact observation stores covariance eigenvalues and an axis angle rather than the
    producer's Cholesky factor.  Rebuilding any Cholesky factor of that same covariance preserves
    the native additive renderer exactly up to ordinary floating-point factorization error.
    """

    if field.provider != "native":
        raise ValueError("native replay requires a provider='native' observation")
    if (
        field.blend_mode != "additive"
        or field.color_grads is not None
        or field.filter_variance is not None
        or field.support_fade_alpha != 0.0
        or field.aa_dilation != 0.0
        or not math.isclose(field.sigma_cutoff, _NATIVE_SIGMA_CUTOFF)
    ):
        raise ValueError("observation semantics do not match the native additive renderer")

    variances = field.scales().square()
    cos = torch.cos(field.rotations)
    sin = torch.sin(field.rotations)
    covariance_xx = cos.square() * variances[:, 0] + sin.square() * variances[:, 1]
    covariance_xy = cos * sin * (variances[:, 0] - variances[:, 1])
    covariance_yy = sin.square() * variances[:, 0] + cos.square() * variances[:, 1]
    floor = torch.finfo(field.dtype).tiny
    chol_11 = covariance_xx.clamp_min(floor).sqrt()
    chol_21 = covariance_xy / chol_11
    chol_22 = (covariance_yy - chol_21.square()).clamp_min(floor).sqrt()
    return Gaussians2D(
        xy=field.local_means() + 0.5,
        chol=torch.stack([chol_11, chol_21, chol_22], dim=-1),
        color=field.colors,
        weight=field.amplitudes,
    )


def render_native_observation_crop(
    field: GaussianObservationField,
    *,
    renderer: str = "auto",
    row_chunk: int = 64,
) -> torch.Tensor:
    """Render one frozen native observation on its crop without StructSplat."""

    _, _, width, height = field.fit_window
    return render_gaussians_2d(
        native_observation_to_gaussians(field),
        height,
        width,
        row_chunk=row_chunk,
        renderer=renderer,
    )


__all__ = [
    "native_gaussians_to_observation",
    "native_observation_to_gaussians",
    "render_native_observation_crop",
]
