"""Closed-form losses for the analytic Gaussian fields beneath compact observations.

The basis used here is a *peak Gaussian*

``G(x; mu, Sigma) = exp(-0.5 (x-mu)^T Sigma^-1 (x-mu))``.

This is intentionally not described as an exact loss on rendered StructSplat RGB.  Current
StructSplat artifacts use normalized blending, finite support rectangles, and optionally a
support fade.  For those artifacts this module compares the underlying untruncated density
``D = sum_i a_i G_i`` and RGB numerator ``N = sum_i a_i c_i G_i``.  Both are useful,
decomposition-invariant analytic proxies; ``N / (D + epsilon)`` is the rendered normalized
color and does not itself admit the pairwise product-kernel expansion used below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from rtgs.core.observation2d import GaussianObservationField


def _validate_geometry(
    name: str,
    means: torch.Tensor,
    covariances: torch.Tensor,
) -> None:
    if means.ndim != 2 or means.shape[1] != 2:
        raise ValueError(f"{name} means must have shape (N,2)")
    if covariances.shape != (means.shape[0], 2, 2):
        raise ValueError(f"{name} covariances must have shape (N,2,2)")
    if not means.is_floating_point() or not covariances.is_floating_point():
        raise TypeError(f"{name} geometry must be floating point")
    if means.device != covariances.device or means.dtype != covariances.dtype:
        raise ValueError(f"{name} means and covariances must share dtype/device")
    if not bool(torch.isfinite(means).all()) or not bool(torch.isfinite(covariances).all()):
        raise ValueError(f"{name} geometry must be finite")
    if means.shape[0] == 0:
        return
    symmetric_error = (covariances - covariances.transpose(-1, -2)).abs().amax()
    scale = covariances.detach().abs().amax().clamp_min(1.0)
    tolerance = 64.0 * torch.finfo(covariances.dtype).eps * scale
    if float(symmetric_error.detach()) > float(tolerance):
        raise ValueError(f"{name} covariances must be symmetric")
    if bool((torch.linalg.cholesky_ex(covariances).info != 0).any()):
        raise ValueError(f"{name} covariances must be positive definite")


def _validate_chunk_size(chunk_size: int) -> int:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    return chunk_size


@dataclass(frozen=True)
class AnalyticGaussianField2D:
    """An untruncated scalar density and RGB-numerator mixture over shared 2D geometry.

    ``density_amplitudes`` are the peak coefficients ``a_i``.  RGB coefficients already
    include density amplitude, i.e. row ``i`` is ``a_i * c_i``.
    """

    means: torch.Tensor
    covariances: torch.Tensor
    density_amplitudes: torch.Tensor
    rgb_amplitudes: torch.Tensor

    def __post_init__(self) -> None:
        _validate_geometry("field", self.means, self.covariances)
        count = self.means.shape[0]
        if self.density_amplitudes.shape != (count,):
            raise ValueError("density_amplitudes must have shape (N,)")
        if self.rgb_amplitudes.shape != (count, 3):
            raise ValueError("rgb_amplitudes must have shape (N,3)")
        for name, value in (
            ("density_amplitudes", self.density_amplitudes),
            ("rgb_amplitudes", self.rgb_amplitudes),
        ):
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating point")
            if value.device != self.means.device or value.dtype != self.means.dtype:
                raise ValueError(f"{name} must share field geometry dtype/device")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        if bool((self.density_amplitudes < 0).any()):
            raise ValueError("density_amplitudes must be non-negative")

    @property
    def n(self) -> int:
        return int(self.means.shape[0])

    @classmethod
    def from_observation(
        cls,
        field: GaussianObservationField,
        *,
        dtype: torch.dtype | None = None,
        affine_color_policy: Literal["reject", "center"] = "reject",
    ) -> AnalyticGaussianField2D:
        """Adapt the analytic mixture beneath a frozen observation.

        Compact support, support fade, normalized division, and affine component colors are not
        part of this proxy.  Affine colors are rejected rather than silently reduced to their
        center values.
        """

        if not isinstance(field, GaussianObservationField):
            raise TypeError("field must be GaussianObservationField")
        if affine_color_policy not in {"reject", "center"}:
            raise ValueError("affine_color_policy must be 'reject' or 'center'")
        if field.color_grads is not None and affine_color_policy == "reject":
            raise ValueError("analytic field proxy does not yet support affine component colors")
        target_dtype = field.dtype if dtype is None else dtype
        if target_dtype not in {torch.float32, torch.float64}:
            raise TypeError("dtype must be torch.float32 or torch.float64")
        means = field.native_means(dtype=target_dtype)
        variances = field.effective_variances().to(dtype=target_dtype)
        rotation = field.rotations.to(dtype=target_dtype)
        cos = rotation.cos()
        sin = rotation.sin()
        covariance = torch.stack(
            [
                cos.square() * variances[:, 0] + sin.square() * variances[:, 1],
                cos * sin * (variances[:, 0] - variances[:, 1]),
                cos * sin * (variances[:, 0] - variances[:, 1]),
                sin.square() * variances[:, 0] + cos.square() * variances[:, 1],
            ],
            dim=-1,
        ).reshape(-1, 2, 2)
        density = field.amplitudes.to(dtype=target_dtype)
        rgb = density[:, None] * field.colors.to(dtype=target_dtype)
        return cls(
            means=means,
            covariances=covariance,
            density_amplitudes=density,
            rgb_amplitudes=rgb,
        )


@dataclass(frozen=True)
class FieldL2Terms:
    """Separate squared-L2 integrals for density and RGB numerator."""

    density: torch.Tensor
    rgb_numerator: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.density + self.rgb_numerator


def peak_gaussian_product_integrals(
    first_means: torch.Tensor,
    first_covariances: torch.Tensor,
    second_means: torch.Tensor,
    second_covariances: torch.Tensor,
) -> torch.Tensor:
    """Return every exact whole-plane integral ``integral G_i(x) H_j(x) dx``.

    The returned shape is ``(N,M)``.  Unlike a normalized Gaussian-density product, the
    determinant prefactor retains the areas implied by peak-amplitude primitives.
    """

    _validate_geometry("first", first_means, first_covariances)
    _validate_geometry("second", second_means, second_covariances)
    if first_means.device != second_means.device or first_means.dtype != second_means.dtype:
        raise ValueError("both Gaussian batches must share dtype/device")
    if first_means.shape[0] == 0 or second_means.shape[0] == 0:
        return first_means.new_zeros((first_means.shape[0], second_means.shape[0]))

    summed = first_covariances[:, None] + second_covariances[None, :]
    delta = first_means[:, None] - second_means[None, :]
    solved = torch.linalg.solve(summed, delta.unsqueeze(-1))
    quadratic = (delta.unsqueeze(-2) @ solved).squeeze(-1).squeeze(-1)
    first_logdet = torch.linalg.slogdet(first_covariances).logabsdet[:, None]
    second_logdet = torch.linalg.slogdet(second_covariances).logabsdet[None, :]
    summed_logdet = torch.linalg.slogdet(summed).logabsdet
    log_prefactor = math.log(2.0 * math.pi) + 0.5 * (first_logdet + second_logdet - summed_logdet)
    return torch.exp(log_prefactor - 0.5 * quadratic)


def mixture_inner_product(
    first_means: torch.Tensor,
    first_covariances: torch.Tensor,
    first_amplitudes: torch.Tensor,
    second_means: torch.Tensor,
    second_covariances: torch.Tensor,
    second_amplitudes: torch.Tensor,
    *,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Return the exact L2 inner product of scalar or vector peak-Gaussian mixtures.

    Amplitudes may have shape ``(N,)`` or ``(N,C)``.  Vector mixtures use the Euclidean
    channel inner product inside the spatial integral.
    """

    chunk_size = _validate_chunk_size(chunk_size)
    _validate_geometry("first", first_means, first_covariances)
    _validate_geometry("second", second_means, second_covariances)
    if first_means.device != second_means.device or first_means.dtype != second_means.dtype:
        raise ValueError("both Gaussian batches must share dtype/device")

    first_features = first_amplitudes[:, None] if first_amplitudes.ndim == 1 else first_amplitudes
    second_features = (
        second_amplitudes[:, None] if second_amplitudes.ndim == 1 else second_amplitudes
    )
    if first_features.ndim != 2 or first_features.shape[0] != first_means.shape[0]:
        raise ValueError("first_amplitudes must have shape (N,) or (N,C)")
    if second_features.ndim != 2 or second_features.shape[0] != second_means.shape[0]:
        raise ValueError("second_amplitudes must have shape (M,) or (M,C)")
    if first_features.shape[1] != second_features.shape[1]:
        raise ValueError("mixture amplitude channel counts must agree")
    for name, value, geometry in (
        ("first_amplitudes", first_features, first_means),
        ("second_amplitudes", second_features, second_means),
    ):
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if value.device != geometry.device or value.dtype != geometry.dtype:
            raise ValueError(f"{name} must share its geometry dtype/device")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")

    result = first_means.new_zeros(())
    for first_start in range(0, first_means.shape[0], chunk_size):
        first_stop = min(first_start + chunk_size, first_means.shape[0])
        for second_start in range(0, second_means.shape[0], chunk_size):
            second_stop = min(second_start + chunk_size, second_means.shape[0])
            kernel = peak_gaussian_product_integrals(
                first_means[first_start:first_stop],
                first_covariances[first_start:first_stop],
                second_means[second_start:second_stop],
                second_covariances[second_start:second_stop],
            )
            channel_inner = first_features[first_start:first_stop] @ second_features[
                second_start:second_stop
            ].transpose(0, 1)
            result = result + (kernel * channel_inner).sum()
    return result


def field_l2(
    first: AnalyticGaussianField2D,
    second: AnalyticGaussianField2D,
    *,
    chunk_size: int = 256,
) -> FieldL2Terms:
    """Return exact analytic L2 terms for density and RGB-numerator proxy fields."""

    if not isinstance(first, AnalyticGaussianField2D) or not isinstance(
        second, AnalyticGaussianField2D
    ):
        raise TypeError("first and second must be AnalyticGaussianField2D")
    density = (
        mixture_inner_product(
            first.means,
            first.covariances,
            first.density_amplitudes,
            first.means,
            first.covariances,
            first.density_amplitudes,
            chunk_size=chunk_size,
        )
        - 2.0
        * mixture_inner_product(
            first.means,
            first.covariances,
            first.density_amplitudes,
            second.means,
            second.covariances,
            second.density_amplitudes,
            chunk_size=chunk_size,
        )
        + mixture_inner_product(
            second.means,
            second.covariances,
            second.density_amplitudes,
            second.means,
            second.covariances,
            second.density_amplitudes,
            chunk_size=chunk_size,
        )
    )
    rgb = (
        mixture_inner_product(
            first.means,
            first.covariances,
            first.rgb_amplitudes,
            first.means,
            first.covariances,
            first.rgb_amplitudes,
            chunk_size=chunk_size,
        )
        - 2.0
        * mixture_inner_product(
            first.means,
            first.covariances,
            first.rgb_amplitudes,
            second.means,
            second.covariances,
            second.rgb_amplitudes,
            chunk_size=chunk_size,
        )
        + mixture_inner_product(
            second.means,
            second.covariances,
            second.rgb_amplitudes,
            second.means,
            second.covariances,
            second.rgb_amplitudes,
            chunk_size=chunk_size,
        )
    )
    return FieldL2Terms(density=density, rgb_numerator=rgb)


__all__ = [
    "AnalyticGaussianField2D",
    "FieldL2Terms",
    "field_l2",
    "mixture_inner_product",
    "peak_gaussian_product_integrals",
]
