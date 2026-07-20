"""Lossless RGB-free observations represented by frozen 2D Gaussian fields.

The compact refinement path must preserve the renderer that produced a fitted field rather than
coerce it into :class:`rtgs.core.gaussians2d.Gaussians2D`, whose bounded colors and scalar weight
serve initialization semantics.  This module is the CPU correctness anchor for arbitrary point
queries and deterministic Gaussian-mixture proposals.  Faster indexed/CUDA implementations must
match these equations.

Coordinates follow the repository camera convention: the top-left pixel center is ``(0.5, 0.5)``.
StructSplat's integer-centered means therefore enter this contract with ``+0.5`` already applied.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

_SCHEMA = "rtgs.gaussian_observation_field.v1"
_PROVIDERS = frozenset({"structsplat", "synthetic_fixture"})
_METADATA_KEYS = frozenset(
    {
        "schema",
        "width",
        "height",
        "blend_mode",
        "epsilon",
        "sigma_cutoff",
        "support_fade_alpha",
        "aa_dilation",
        "view_id",
        "fit_window",
        "n_init",
        "n_opt",
        "provider",
        "kernel_semantics",
        "producer_version",
        "producer_source_digest",
        "fit_config_digest",
        "coordinate_convention",
        "arrays",
        "semantic_digest",
    }
)
_REQUIRED_ARRAYS = frozenset({"means", "log_scales", "rotations", "colors", "amplitudes"})
_OPTIONAL_ARRAYS = frozenset({"color_grads", "filter_variance", "mean_residuals"})
_ARRAY_DESCRIPTOR_KEYS = frozenset({"dtype", "shape", "sha256"})


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _strict_json_object(payload: bytes, *, label: str) -> dict:
    """Decode a JSON object while rejecting duplicate keys hidden by normal ``json.loads``."""

    def object_pairs(pairs: list[tuple[str, object]]) -> dict:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    value = json.loads(payload, object_pairs_hook=object_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_exact_keys(value: object, expected: frozenset[str], *, label: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{label} keys are not exact (missing={missing}, extra={extra})")
    return value


@dataclass(frozen=True)
class ObservationQuery:
    """Values returned by :meth:`GaussianObservationField.query`."""

    color: torch.Tensor  # (S,3)
    numerator: torch.Tensor  # (S,3)
    weight_sum: torch.Tensor  # (S,)
    valid: torch.Tensor  # (S,), inside the fitted observation window


@dataclass(frozen=True)
class ObservationSamples:
    """A deterministic proposal batch with enough information for importance correction."""

    xy: torch.Tensor  # (S,2), continuous coordinates or discrete pixel centers
    proposal_component_ids: torch.Tensor  # (S,), -1 for uniform samples
    proposal_density: torch.Tensor  # (S,), marginal density/probability q(x)
    joint_density: torch.Tensor  # (S,), joint density/probability of the realized active draw
    target_density: torch.Tensor  # (S,), uniform target measure (zero for null attempts)
    importance: torch.Tensor  # target_density / proposal_density
    inside_fit_window: torch.Tensor  # (S,)
    active: torch.Tensor  # (S,), contributes to a fixed-attempt estimator; false is null
    risk_measure: str  # continuous area and discrete pixels are deliberately distinct risks


def fixed_attempt_mean(values: torch.Tensor, samples: ObservationSamples) -> torch.Tensor:
    """Importance-weight ``values`` and divide by all attempts, including explicit nulls."""
    if values.shape[0] != samples.importance.shape[0]:
        raise ValueError("values and samples must have the same leading dimension")
    weights = samples.importance.reshape(
        samples.importance.shape + (1,) * (values.ndim - samples.importance.ndim)
    )
    return (values * weights).mean(dim=0)


@dataclass(frozen=True)
class GaussianObservationField:
    """A self-describing frozen 2D Gaussian renderer used as an RGB-free teacher.

    ``log_scales`` and ``rotations`` retain the native RS parameterization. StructSplat's affine
    color basis includes frozen ``filter_variance`` but not reference-renderer AA dilation;
    rendered covariance includes both. ``amplitudes`` are normalized-renderer weights, not 3D
    alpha. Colors are intentionally unbounded.
    """

    width: int
    height: int
    means: torch.Tensor  # (N,2), repository half-integer coordinate convention
    log_scales: torch.Tensor  # (N,2)
    rotations: torch.Tensor  # (N,)
    colors: torch.Tensor  # (N,3), unbounded
    amplitudes: torch.Tensor  # (N,), non-negative
    color_grads: torch.Tensor | None = None  # (N,2,3), local affine color
    filter_variance: torch.Tensor | None = None  # (N,), added isotropic variance
    blend_mode: str = "normalized"
    epsilon: float = 1e-8
    sigma_cutoff: float = 3.0
    support_fade_alpha: float = 0.0
    aa_dilation: float = 0.0
    view_id: str | None = None
    fit_window: tuple[int, int, int, int] | None = None  # full-canvas (x, y, width, height)
    n_init: int | None = None
    provider: str = "structsplat"
    producer_version: str | None = None
    producer_source_digest: str | None = None
    fit_config_digest: str | None = None
    mean_residuals: torch.Tensor | None = None  # (N,2), exact crop-local float32 correction
    _scales: torch.Tensor = field(init=False, repr=False, compare=False)
    _effective_variances: torch.Tensor = field(init=False, repr=False, compare=False)
    _color_scales: torch.Tensor = field(init=False, repr=False, compare=False)
    _conics: torch.Tensor = field(init=False, repr=False, compare=False)
    _radii: torch.Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        n = int(self.means.shape[0])
        if self.width <= 0 or self.height <= 0:
            raise ValueError("observation dimensions must be positive")
        if self.fit_window is None:
            object.__setattr__(self, "fit_window", (0, 0, self.width, self.height))
        if len(self.fit_window) != 4 or any(
            not isinstance(value, int) for value in self.fit_window
        ):
            raise ValueError("fit_window must contain four integers")
        fit_x, fit_y, fit_width, fit_height = self.fit_window
        if (
            fit_x < 0
            or fit_y < 0
            or fit_width <= 0
            or fit_height <= 0
            or fit_x + fit_width > self.width
            or fit_y + fit_height > self.height
        ):
            raise ValueError("fit_window must be a non-empty rectangle inside the canvas")
        if self.n_init is not None and self.n_init <= 0:
            raise ValueError("n_init must be positive when supplied")
        if self.provider not in _PROVIDERS:
            raise ValueError(
                "schema v1 provider must be exactly 'structsplat' or 'synthetic_fixture'"
            )
        if self.view_id is not None and not isinstance(self.view_id, str):
            raise TypeError("view_id must be a string when supplied")
        if self.producer_version is not None and not isinstance(self.producer_version, str):
            raise TypeError("producer_version must be a string when supplied")
        for name in ("producer_source_digest", "fit_config_digest"):
            digest = getattr(self, name)
            if digest is not None and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if (
            self.means.shape != (n, 2)
            or self.log_scales.shape != (n, 2)
            or self.rotations.shape != (n,)
            or self.colors.shape != (n, 3)
            or self.amplitudes.shape != (n,)
        ):
            raise ValueError("inconsistent GaussianObservationField shapes")
        if self.color_grads is not None and self.color_grads.shape != (n, 2, 3):
            raise ValueError("color_grads must have shape (N,2,3)")
        if self.filter_variance is not None and self.filter_variance.shape != (n,):
            raise ValueError("filter_variance must have shape (N,)")
        if self.mean_residuals is not None:
            if self.mean_residuals.shape != (n, 2):
                raise ValueError("mean_residuals must have shape (N,2)")
            if self.mean_residuals.dtype != torch.float32:
                raise TypeError("mean_residuals must use float32")
            if self.mean_residuals.device != self.means.device:
                raise ValueError("mean_residuals must share the observation device")
            if not bool(torch.isfinite(self.mean_residuals).all()):
                raise ValueError("mean_residuals must be finite")
        tensors = [
            self.means,
            self.log_scales,
            self.rotations,
            self.colors,
            self.amplitudes,
        ]
        tensors.extend(x for x in (self.color_grads, self.filter_variance) if x is not None)
        if any(not tensor.is_floating_point() for tensor in tensors):
            raise TypeError("observation tensors must be floating point")
        if any(tensor.device != self.means.device for tensor in tensors):
            raise ValueError("observation tensors must share one device")
        if any(tensor.dtype != self.means.dtype for tensor in tensors):
            raise ValueError("observation tensors must share one dtype")
        if self.means.dtype not in {torch.float32, torch.float64}:
            raise TypeError("observation tensors must use float32 or float64")
        if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors):
            raise ValueError("observation tensors must be finite")
        if bool((self.amplitudes < 0).any()):
            raise ValueError("amplitudes must be non-negative")
        if self.filter_variance is not None and bool((self.filter_variance < 0).any()):
            raise ValueError("filter_variance must be non-negative")
        if self.blend_mode not in {"normalized", "additive"}:
            raise ValueError("blend_mode must be 'normalized' or 'additive'")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not math.isfinite(self.sigma_cutoff) or self.sigma_cutoff <= 0:
            raise ValueError("sigma_cutoff must be finite and positive")
        if not 0.0 <= self.support_fade_alpha <= 1.0:
            raise ValueError("support_fade_alpha must be in [0,1]")
        if not math.isfinite(self.aa_dilation) or self.aa_dilation < 0:
            raise ValueError("aa_dilation must be finite and non-negative")

        for name in (
            "means",
            "log_scales",
            "rotations",
            "colors",
            "amplitudes",
            "mean_residuals",
            "color_grads",
            "filter_variance",
        ):
            tensor = getattr(self, name)
            if tensor is not None:
                object.__setattr__(self, name, tensor.detach().clone())

        scales = self.log_scales.exp()
        color_variances = scales.square()
        if self.filter_variance is not None:
            color_variances = color_variances + self.filter_variance[:, None]
        effective_variances = (color_variances + self.aa_dilation).clamp_min(1e-12)
        color_scales = color_variances.clamp_min(1e-12).sqrt()
        inv = effective_variances.reciprocal()
        cos = torch.cos(self.rotations)
        sin = torch.sin(self.rotations)
        conics = torch.stack(
            [
                cos.square() * inv[:, 0] + sin.square() * inv[:, 1],
                cos * sin * (inv[:, 0] - inv[:, 1]),
                sin.square() * inv[:, 0] + cos.square() * inv[:, 1],
            ],
            dim=-1,
        )
        var_x = cos.square() * effective_variances[:, 0] + sin.square() * effective_variances[:, 1]
        var_y = sin.square() * effective_variances[:, 0] + cos.square() * effective_variances[:, 1]
        radii = self.sigma_cutoff * torch.sqrt(torch.stack([var_x, var_y], dim=-1))
        derived = (
            scales,
            color_variances,
            effective_variances,
            color_scales,
            inv,
            conics,
            radii,
        )
        if any(not bool(torch.isfinite(tensor).all()) for tensor in derived):
            raise ValueError("derived observation tensors must be finite")
        if bool((scales <= 0).any()):
            raise ValueError("derived observation scales must be positive")
        # ``radii`` is converted to int64 below.  Reject absurd-but-finite float inputs before
        # conversion can wrap to a small/negative support and accidentally evade index caps.
        if bool((radii.ceil() > 2**62).any()):
            raise ValueError("derived observation radii exceed the safe integer range")
        object.__setattr__(self, "_scales", scales)
        object.__setattr__(self, "_effective_variances", effective_variances)
        object.__setattr__(self, "_color_scales", color_scales)
        object.__setattr__(self, "_conics", conics)
        object.__setattr__(self, "_radii", radii.ceil().long().clamp_min(1))

    @property
    def n(self) -> int:
        """Number of frozen 2D components."""
        return int(self.means.shape[0])

    @property
    def device(self) -> torch.device:
        return self.means.device

    @property
    def dtype(self) -> torch.dtype:
        return self.means.dtype

    def to(self, device: torch.device | str) -> GaussianObservationField:
        """Copy the field to ``device`` without changing its renderer semantics."""
        return GaussianObservationField(
            width=self.width,
            height=self.height,
            means=self.means.to(device),
            log_scales=self.log_scales.to(device),
            rotations=self.rotations.to(device),
            colors=self.colors.to(device),
            amplitudes=self.amplitudes.to(device),
            mean_residuals=(
                None if self.mean_residuals is None else self.mean_residuals.to(device)
            ),
            color_grads=None if self.color_grads is None else self.color_grads.to(device),
            filter_variance=(
                None if self.filter_variance is None else self.filter_variance.to(device)
            ),
            blend_mode=self.blend_mode,
            epsilon=self.epsilon,
            sigma_cutoff=self.sigma_cutoff,
            support_fade_alpha=self.support_fade_alpha,
            aa_dilation=self.aa_dilation,
            view_id=self.view_id,
            fit_window=self.fit_window,
            n_init=self.n_init,
            provider=self.provider,
            producer_version=self.producer_version,
            producer_source_digest=self.producer_source_digest,
            fit_config_digest=self.fit_config_digest,
        )

    def scales(self) -> torch.Tensor:
        """Unfiltered local RS scales used by optional affine colors."""
        return self._scales

    def effective_variances(self) -> torch.Tensor:
        """RS-axis variances after frozen covariance filtering and AA dilation."""
        return self._effective_variances

    def color_scales(self) -> torch.Tensor:
        """RS scales used by StructSplat's optional local affine-color basis."""
        return self._color_scales

    def conics(self) -> torch.Tensor:
        """Unique entries ``(a,b,c)`` of each effective inverse covariance."""
        return self._conics

    def radii(self) -> torch.Tensor:
        """Integer AABB radii matching StructSplat's clipped support rectangles."""
        return self._radii

    def local_means(
        self,
        component_ids: torch.Tensor | slice | int | None = None,
    ) -> torch.Tensor:
        """Component means in the crop-local integer-centered provider coordinates.

        ``means`` remains the schema-v1 full-canvas half-integer tensor. A float32 addition of a
        large crop offset can discard low-order mean bits, so new StructSplat archives optionally
        carry the exact correction needed after translating that tensor back to crop coordinates.
        Residual-free schema-v1 archives retain their historical subtraction exactly.
        """
        means = self.means if component_ids is None else self.means[component_ids]
        fit_x, fit_y, _, _ = self.fit_window
        origin = means.new_tensor([fit_x + 0.5, fit_y + 0.5])
        local = means - origin
        if self.mean_residuals is not None:
            residuals = (
                self.mean_residuals if component_ids is None else self.mean_residuals[component_ids]
            )
            local = local + residuals.to(local.dtype)
        return local

    def native_means(
        self,
        component_ids: torch.Tensor | slice | int | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Component means in full-canvas half-integer coordinates.

        Request ``dtype=torch.float64`` to preserve the exact corrected crop-local float32 value
        when adding a native-resolution offset. The default keeps the field dtype. For legacy
        fields without residuals this returns the stored schema-v1 tensor (optionally cast), which
        preserves their prior behavior.
        """
        target_dtype = self.dtype if dtype is None else dtype
        if target_dtype not in {torch.float32, torch.float64}:
            raise TypeError("native mean dtype must be torch.float32 or torch.float64")
        means = self.means if component_ids is None else self.means[component_ids]
        if self.mean_residuals is None:
            return means.to(dtype=target_dtype)
        local = self.local_means(component_ids).to(dtype=target_dtype)
        fit_x, fit_y, _, _ = self.fit_window
        origin = torch.tensor(
            [fit_x + 0.5, fit_y + 0.5],
            dtype=target_dtype,
            device=self.device,
        )
        return local + origin

    def support_centers(
        self,
        component_ids: torch.Tensor | slice | int | None = None,
    ) -> torch.Tensor:
        """Rounded AABB centers in full-canvas half-integer pixel coordinates.

        StructSplat rounds means in the coordinate frame in which a field was fitted.  A cropped
        fit therefore has to be translated back to that local frame *before* rounding: ties-to-
        even rounding is not translation invariant for odd crop offsets.
        """
        means = self.local_means(component_ids)
        fit_x, fit_y, _, _ = self.fit_window
        origin = means.new_tensor([fit_x + 0.5, fit_y + 0.5])
        return torch.round(means) + origin

    def support_pixels(
        self,
        component_ids: torch.Tensor | slice | int | None = None,
    ) -> torch.Tensor:
        """Rounded AABB centers as integer full-canvas pixel indices."""
        means = self.local_means(component_ids)
        fit_x, fit_y, _, _ = self.fit_window
        integer_origin = torch.tensor(
            [fit_x, fit_y],
            dtype=torch.long,
            device=means.device,
        )
        return torch.round(means).long() + integer_origin

    def component_weight(self, xy: torch.Tensor, component_ids: torch.Tensor) -> torch.Tensor:
        """Evaluate paired component weights at paired pixel coordinates."""
        xy = self._validate_xy(xy)
        component_ids = torch.as_tensor(component_ids, device=self.device, dtype=torch.long)
        if component_ids.shape != (xy.shape[0],):
            raise ValueError("component_ids must have shape (S,)")
        if bool(((component_ids < 0) | (component_ids >= self.n)).any()):
            raise ValueError("component_ids are out of range")
        return self._paired_values(xy, component_ids)[0]

    def component_color(self, xy: torch.Tensor, component_ids: torch.Tensor) -> torch.Tensor:
        """Evaluate paired constant or affine component colors without blending."""
        xy = self._validate_xy(xy)
        component_ids = torch.as_tensor(component_ids, device=self.device, dtype=torch.long)
        if component_ids.shape != (xy.shape[0],):
            raise ValueError("component_ids must have shape (S,)")
        if bool(((component_ids < 0) | (component_ids >= self.n)).any()):
            raise ValueError("component_ids are out of range")
        return self._paired_values(xy, component_ids)[1]

    def query(self, xy: torch.Tensor, component_chunk: int = 4096) -> ObservationQuery:
        """Query the exact frozen field at pixel coordinates without materializing an image.

        This reference path bounds temporary storage by ``S * component_chunk``. It intentionally
        remains simple; a static support index will replace all-component scans in the fast path.
        """
        xy = self._validate_xy(xy)
        if component_chunk <= 0:
            raise ValueError("component_chunk must be positive")
        count = xy.shape[0]
        numerator = torch.zeros(count, 3, dtype=self.dtype, device=self.device)
        denominator = torch.zeros(count, dtype=self.dtype, device=self.device)
        for start in range(0, self.n, component_chunk):
            end = min(start + component_chunk, self.n)
            component_ids = torch.arange(start, end, device=self.device)
            weights, pixel_colors = self._cross_values(xy, component_ids)
            denominator = denominator + weights.sum(dim=1)
            numerator = numerator + (weights[..., None] * pixel_colors).sum(dim=1)
        if self.blend_mode == "normalized":
            color = numerator / (denominator[:, None] + self.epsilon)
        else:
            color = numerator
        return ObservationQuery(
            color=color,
            numerator=numerator,
            weight_sum=denominator,
            valid=self.valid_domain(xy),
        )

    def query_weight_sum(self, xy: torch.Tensor, component_chunk: int = 4096) -> torch.Tensor:
        """Query only the normalized renderer denominator, avoiding color work."""
        xy = self._validate_xy(xy)
        if component_chunk <= 0:
            raise ValueError("component_chunk must be positive")
        denominator = torch.zeros(xy.shape[0], dtype=self.dtype, device=self.device)
        for start in range(0, self.n, component_chunk):
            component_ids = torch.arange(
                start,
                min(start + component_chunk, self.n),
                device=self.device,
            )
            denominator = denominator + self._cross_weights(xy, component_ids).sum(dim=1)
        return denominator

    def valid_domain(self, xy: torch.Tensor) -> torch.Tensor:
        """Whether coordinates lie in the fitted window, not merely the source canvas."""
        xy = self._validate_xy(xy)
        fit_x, fit_y, fit_width, fit_height = self.fit_window
        return (
            (xy[:, 0] >= fit_x)
            & (xy[:, 0] < fit_x + fit_width)
            & (xy[:, 1] >= fit_y)
            & (xy[:, 1] < fit_y + fit_height)
        )

    def pixel_centers(self, *, full_canvas: bool = False) -> torch.Tensor:
        """Pixel centers in the fitted window (or the full canvas) in row-major order."""
        fit_x, fit_y, fit_width, fit_height = self.fit_window
        x0, y0 = (0, 0) if full_canvas else (fit_x, fit_y)
        width, height = (self.width, self.height) if full_canvas else (fit_width, fit_height)
        y, x = torch.meshgrid(
            torch.arange(y0, y0 + height, dtype=self.dtype, device=self.device) + 0.5,
            torch.arange(x0, x0 + width, dtype=self.dtype, device=self.device) + 0.5,
            indexing="ij",
        )
        return torch.stack([x, y], dim=-1).reshape(-1, 2)

    def query_patch(
        self,
        top_left: tuple[float, float],
        size: tuple[int, int],
        component_chunk: int = 4096,
    ) -> ObservationQuery:
        """Query a regular pixel-center patch through the same point-query equation."""
        patch_height, patch_width = size
        if patch_height <= 0 or patch_width <= 0:
            raise ValueError("patch dimensions must be positive")
        x0, y0 = top_left
        y, x = torch.meshgrid(
            torch.arange(patch_height, dtype=self.dtype, device=self.device) + y0,
            torch.arange(patch_width, dtype=self.dtype, device=self.device) + x0,
            indexing="ij",
        )
        result = self.query(
            torch.stack([x, y], dim=-1).reshape(-1, 2),
            component_chunk=component_chunk,
        )
        return ObservationQuery(
            color=result.color.reshape(patch_height, patch_width, 3),
            numerator=result.numerator.reshape(patch_height, patch_width, 3),
            weight_sum=result.weight_sum.reshape(patch_height, patch_width),
            valid=result.valid.reshape(patch_height, patch_width),
        )

    def save_npz(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Atomically store an integrity-checked archive with no declared RGB payload field."""
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite observation archive: {path}")
        metadata = {
            "schema": _SCHEMA,
            "width": self.width,
            "height": self.height,
            "blend_mode": self.blend_mode,
            "epsilon": self.epsilon,
            "sigma_cutoff": self.sigma_cutoff,
            "support_fade_alpha": self.support_fade_alpha,
            "aa_dilation": self.aa_dilation,
            "view_id": self.view_id,
            "fit_window": list(self.fit_window),
            "n_init": self.n_init,
            "n_opt": self.n,
            "provider": self.provider,
            "kernel_semantics": "structsplat_rs_rounded_aabb_v1",
            "producer_version": self.producer_version,
            "producer_source_digest": self.producer_source_digest,
            "fit_config_digest": self.fit_config_digest,
            "coordinate_convention": "half_integer_pixel_centers",
        }
        arrays: dict[str, np.ndarray] = {
            "means": self.means.detach().cpu().numpy(),
            "log_scales": self.log_scales.detach().cpu().numpy(),
            "rotations": self.rotations.detach().cpu().numpy(),
            "colors": self.colors.detach().cpu().numpy(),
            "amplitudes": self.amplitudes.detach().cpu().numpy(),
        }
        if self.mean_residuals is not None:
            arrays["mean_residuals"] = self.mean_residuals.detach().cpu().numpy()
        if self.color_grads is not None:
            arrays["color_grads"] = self.color_grads.detach().cpu().numpy()
        if self.filter_variance is not None:
            arrays["filter_variance"] = self.filter_variance.detach().cpu().numpy()
        metadata["arrays"] = {
            name: {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": _array_sha256(array),
            }
            for name, array in sorted(arrays.items())
        }
        metadata["semantic_digest"] = hashlib.sha256(_canonical_json(metadata)).hexdigest()
        arrays["metadata_utf8"] = np.frombuffer(_canonical_json(metadata), dtype=np.uint8)

        temporary: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            temporary = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            if path.exists() and not overwrite:
                raise FileExistsError(f"refusing to overwrite observation archive: {path}")
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def load_npz(
        path: str | Path,
        device: torch.device | str = "cpu",
        *,
        verify: bool = True,
        strict: bool = False,
    ) -> GaussianObservationField:
        """Load :meth:`save_npz` output without pickle or optional StructSplat imports."""
        if strict and not verify:
            raise ValueError("strict observation loading requires integrity verification")
        with np.load(Path(path), allow_pickle=False) as data:
            if "metadata_utf8" not in data.files:
                raise ValueError("observation archive is missing metadata_utf8")
            metadata_payload = np.asarray(data["metadata_utf8"], dtype=np.uint8).tobytes()
            metadata = (
                _strict_json_object(metadata_payload, label="observation metadata")
                if strict
                else json.loads(metadata_payload)
            )
            if not isinstance(metadata, dict):
                raise ValueError("observation metadata must be a JSON object")
            if strict:
                _require_exact_keys(metadata, _METADATA_KEYS, label="observation metadata")
            if metadata.get("schema") != _SCHEMA:
                raise ValueError(f"unsupported observation schema {metadata.get('schema')!r}")
            if metadata.get("coordinate_convention") != "half_integer_pixel_centers":
                raise ValueError("unsupported observation coordinate convention")
            if strict and metadata.get("kernel_semantics") != "structsplat_rs_rounded_aabb_v1":
                raise ValueError("unsupported observation kernel semantics")
            expected_arrays = metadata.get("arrays")
            if not isinstance(expected_arrays, dict):
                raise ValueError("observation metadata is missing array descriptors")
            actual_names = set(data.files) - {"metadata_utf8"}
            if actual_names != set(expected_arrays):
                raise ValueError("observation archive arrays do not match metadata")
            if strict:
                if not actual_names >= _REQUIRED_ARRAYS or not actual_names <= (
                    _REQUIRED_ARRAYS | _OPTIONAL_ARRAYS
                ):
                    raise ValueError("strict observation archive has unsupported array names")
                for name, descriptor in expected_arrays.items():
                    _require_exact_keys(
                        descriptor,
                        _ARRAY_DESCRIPTOR_KEYS,
                        label=f"observation array descriptor {name!r}",
                    )
            if verify:
                digest_metadata = dict(metadata)
                stored_digest = digest_metadata.pop("semantic_digest", None)
                actual_digest = hashlib.sha256(_canonical_json(digest_metadata)).hexdigest()
                if stored_digest != actual_digest:
                    raise ValueError("observation semantic metadata digest mismatch")
                for name, descriptor in expected_arrays.items():
                    array = np.asarray(data[name])
                    actual = {
                        "dtype": array.dtype.str,
                        "shape": list(array.shape),
                        "sha256": _array_sha256(array),
                    }
                    if actual != descriptor:
                        raise ValueError(f"observation array integrity mismatch: {name}")

            def tensor(name: str) -> torch.Tensor:
                return torch.from_numpy(np.asarray(data[name]).copy()).to(device)

            result = GaussianObservationField(
                width=int(metadata["width"]),
                height=int(metadata["height"]),
                means=tensor("means"),
                log_scales=tensor("log_scales"),
                rotations=tensor("rotations"),
                colors=tensor("colors"),
                amplitudes=tensor("amplitudes"),
                mean_residuals=(
                    tensor("mean_residuals") if "mean_residuals" in data.files else None
                ),
                color_grads=tensor("color_grads") if "color_grads" in data.files else None,
                filter_variance=(
                    tensor("filter_variance") if "filter_variance" in data.files else None
                ),
                blend_mode=str(metadata["blend_mode"]),
                epsilon=float(metadata["epsilon"]),
                sigma_cutoff=float(metadata["sigma_cutoff"]),
                support_fade_alpha=float(metadata["support_fade_alpha"]),
                aa_dilation=float(metadata["aa_dilation"]),
                view_id=metadata.get("view_id"),
                fit_window=tuple(int(value) for value in metadata["fit_window"]),
                n_init=(None if metadata.get("n_init") is None else int(metadata["n_init"])),
                provider=str(metadata["provider"]),
                producer_version=metadata.get("producer_version"),
                producer_source_digest=metadata.get("producer_source_digest"),
                fit_config_digest=metadata.get("fit_config_digest"),
            )
            if int(metadata.get("n_opt", -1)) != result.n:
                raise ValueError("observation n_opt metadata does not match array cardinality")
            return result

    def _validate_xy(self, xy: torch.Tensor) -> torch.Tensor:
        xy = torch.as_tensor(xy, dtype=self.dtype, device=self.device)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (S,2)")
        if not bool(torch.isfinite(xy).all()):
            raise ValueError("xy must be finite")
        return xy

    def _cross_values(
        self, xy: torch.Tensor, component_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self._cross_weights(xy, component_ids)
        dx = self._cross_displacements(xy, component_ids)

        colors = self.colors[component_ids][None, :, :].expand(xy.shape[0], -1, -1)
        if self.color_grads is not None:
            theta = self.rotations[component_ids]
            cos = torch.cos(theta)[None, :]
            sin = torch.sin(theta)[None, :]
            scales = self.color_scales()[component_ids]
            local_x = (cos * dx[..., 0] + sin * dx[..., 1]) / scales[None, :, 0].clamp_min(1e-6)
            local_y = (-sin * dx[..., 0] + cos * dx[..., 1]) / scales[None, :, 1].clamp_min(1e-6)
            grads = self.color_grads[component_ids]
            colors = (
                colors
                + local_x[..., None] * grads[None, :, 0, :]
                + local_y[..., None] * grads[None, :, 1, :]
            )
        return weights, colors

    def _cross_weights(self, xy: torch.Tensor, component_ids: torch.Tensor) -> torch.Tensor:
        dx = self._cross_displacements(xy, component_ids)
        conics = self.conics()[component_ids]
        q = (
            conics[None, :, 0] * dx[..., 0].square()
            + 2.0 * conics[None, :, 1] * dx[..., 0] * dx[..., 1]
            + conics[None, :, 2] * dx[..., 1].square()
        )
        weights = torch.exp(-0.5 * q)
        if self.support_fade_alpha > 0.0:
            floor = self.support_fade_alpha * math.exp(-0.5 * self.sigma_cutoff**2)
            weights = (weights - floor).clamp_min(0.0)
        support_centers = self.support_centers(component_ids)
        radii = self.radii()[component_ids]
        inside_support = (
            (xy[:, None, 0] >= support_centers[None, :, 0] - radii[None, :, 0])
            & (xy[:, None, 0] <= support_centers[None, :, 0] + radii[None, :, 0])
            & (xy[:, None, 1] >= support_centers[None, :, 1] - radii[None, :, 1])
            & (xy[:, None, 1] <= support_centers[None, :, 1] + radii[None, :, 1])
        )
        return (
            weights
            * inside_support
            * self.valid_domain(xy)[:, None]
            * self.amplitudes[component_ids][None, :]
        )

    def _paired_weights(self, xy: torch.Tensor, component_ids: torch.Tensor) -> torch.Tensor:
        """Weight-only paired evaluation, sharing exact support semantics with paired values.

        This is the weight half of :meth:`_paired_values`.  ``query_weight_sum`` uses it so the
        indexed denominator path never materializes color, while still streaming the identical
        ``(point, component)`` pairs the color path evaluates.
        """
        dx = self._paired_displacements(xy, component_ids)
        conics = self.conics()[component_ids]
        q = (
            conics[:, 0] * dx[:, 0].square()
            + 2.0 * conics[:, 1] * dx[:, 0] * dx[:, 1]
            + conics[:, 2] * dx[:, 1].square()
        )
        weights = torch.exp(-0.5 * q)
        if self.support_fade_alpha > 0.0:
            floor = self.support_fade_alpha * math.exp(-0.5 * self.sigma_cutoff**2)
            weights = (weights - floor).clamp_min(0.0)
        support_centers = self.support_centers(component_ids)
        radii = self.radii()[component_ids]
        inside_support = (
            (xy[:, 0] >= support_centers[:, 0] - radii[:, 0])
            & (xy[:, 0] <= support_centers[:, 0] + radii[:, 0])
            & (xy[:, 1] >= support_centers[:, 1] - radii[:, 1])
            & (xy[:, 1] <= support_centers[:, 1] + radii[:, 1])
        )
        return weights * inside_support * self.valid_domain(xy) * self.amplitudes[component_ids]

    def _paired_values(
        self, xy: torch.Tensor, component_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dx = self._paired_displacements(xy, component_ids)
        conics = self.conics()[component_ids]
        q = (
            conics[:, 0] * dx[:, 0].square()
            + 2.0 * conics[:, 1] * dx[:, 0] * dx[:, 1]
            + conics[:, 2] * dx[:, 1].square()
        )
        weights = torch.exp(-0.5 * q)
        if self.support_fade_alpha > 0.0:
            floor = self.support_fade_alpha * math.exp(-0.5 * self.sigma_cutoff**2)
            weights = (weights - floor).clamp_min(0.0)
        support_centers = self.support_centers(component_ids)
        radii = self.radii()[component_ids]
        inside_support = (
            (xy[:, 0] >= support_centers[:, 0] - radii[:, 0])
            & (xy[:, 0] <= support_centers[:, 0] + radii[:, 0])
            & (xy[:, 1] >= support_centers[:, 1] - radii[:, 1])
            & (xy[:, 1] <= support_centers[:, 1] + radii[:, 1])
        )
        weights = weights * inside_support * self.valid_domain(xy) * self.amplitudes[component_ids]

        colors = self.colors[component_ids]
        if self.color_grads is not None:
            theta = self.rotations[component_ids]
            cos = torch.cos(theta)
            sin = torch.sin(theta)
            scales = self.color_scales()[component_ids]
            local_x = (cos * dx[:, 0] + sin * dx[:, 1]) / scales[:, 0].clamp_min(1e-6)
            local_y = (-sin * dx[:, 0] + cos * dx[:, 1]) / scales[:, 1].clamp_min(1e-6)
            grads = self.color_grads[component_ids]
            colors = colors + local_x[:, None] * grads[:, 0, :] + local_y[:, None] * grads[:, 1, :]
        return weights, colors

    def _cross_displacements(
        self,
        xy: torch.Tensor,
        component_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-product displacements, preserving legacy arithmetic without residuals."""
        if self.mean_residuals is None:
            return xy[:, None, :] - self.means[component_ids][None, :, :]
        fit_x, fit_y, _, _ = self.fit_window
        origin = xy.new_tensor([fit_x + 0.5, fit_y + 0.5])
        local_xy = xy - origin
        return local_xy[:, None, :] - self.local_means(component_ids)[None, :, :]

    def _paired_displacements(
        self,
        xy: torch.Tensor,
        component_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Paired displacements, preserving legacy arithmetic without residuals."""
        if self.mean_residuals is None:
            return xy - self.means[component_ids]
        fit_x, fit_y, _, _ = self.fit_window
        origin = xy.new_tensor([fit_x + 0.5, fit_y + 0.5])
        return (xy - origin) - self.local_means(component_ids)


@dataclass(frozen=True)
class GaussianObservationIndexStats:
    """Bounded-state diagnostics for one exact tile-overlap index."""

    tile_size: int
    tiles_x: int
    tiles_y: int
    nonempty_tiles: int
    total_entries: int
    max_candidates: int
    retained_bytes: int = 0
    component_id_dtype: str = "int64"
    max_query_pairs: int | None = None


DEFAULT_MAX_QUERY_PAIRS = 1_048_576


class GaussianObservationIndex:
    """Exact CPU tile-overlap index for local point queries on a frozen field.

    Only non-empty tiles are stored, in three contiguous CSR arrays rather than one tensor per
    tile::

        tile_keys     int64 [T]      sorted linear tile IDs for non-empty tiles
        tile_offsets  int64 [T + 1]  CSR offsets into component_ids
        component_ids int32|int64 [E] component IDs, ascending within each tile row

    Index state therefore scales with component/tile overlap, not canvas pixels, and query work
    scales with local candidates rather than every component.  Queries stream a bounded
    ``(point, component)`` pair sequence in canonical (point-major, ascending-component) order and
    evaluate the exact paired field, so no per-tile Python loop or Cartesian product is
    materialized.  A future CUDA backend can implement this same
    ``query``/``query_weight_sum`` interface.
    """

    DEFAULT_MAX_ENTRIES = 16_000_000
    DEFAULT_MAX_CANDIDATES = 200_000
    DEFAULT_MAX_QUERY_PAIRS = DEFAULT_MAX_QUERY_PAIRS
    # Largest ``field.n - 1`` that is retained as int32; a test seam forces the int64 fallback
    # without allocating a giant real field.
    _int32_component_limit = 2**31 - 1

    def __init__(
        self,
        field: GaussianObservationField,
        tile_size: int = 16,
        *,
        max_entries: int | None = DEFAULT_MAX_ENTRIES,
        max_candidates: int | None = DEFAULT_MAX_CANDIDATES,
        max_query_pairs: int | None = DEFAULT_MAX_QUERY_PAIRS,
    ):
        if field.device.type != "cpu":
            raise ValueError("GaussianObservationIndex is the CPU reference backend")
        self._validate_limits(tile_size, max_entries, max_candidates, max_query_pairs)
        self.field = field
        self.tile_size = int(tile_size)
        self.max_query_pairs = None if max_query_pairs is None else int(max_query_pairs)
        self.tiles_x = math.ceil(field.width / self.tile_size)
        self.tiles_y = math.ceil(field.height / self.tile_size)

        ids, tile_x0, tile_x1, tile_y0, tile_y1 = self._component_tile_ranges(field, self.tile_size)
        widths = tile_x1 - tile_x0 + 1
        heights = tile_y1 - tile_y0 + 1
        counts = widths * heights
        self.estimated_entries = int(counts.sum())
        if max_entries is not None and self.estimated_entries > max_entries:
            raise ValueError(
                "Gaussian observation index entry cap exceeded before allocation: "
                f"estimated={self.estimated_entries}, max_entries={max_entries}"
            )

        component_dtype = torch.int32 if field.n - 1 <= self._int32_component_limit else torch.int64
        self.component_id_dtype = component_dtype
        (
            self.tile_keys,
            self.tile_offsets,
            self.component_ids,
            self.max_candidates,
        ) = self._build_csr(ids, tile_x0, tile_y0, widths, counts, component_dtype)
        self.n_entries = int(self.component_ids.numel())

        if max_candidates is not None and self.max_candidates > max_candidates:
            raise ValueError(
                "Gaussian observation index candidate cap exceeded: "
                f"max_candidates_observed={self.max_candidates}, "
                f"max_candidates={max_candidates}"
            )
        if self.n_entries != self.estimated_entries:
            raise RuntimeError("Gaussian observation index preflight/build entry mismatch")

        # Progress counters, updated in place by streamed queries.
        self.total_pairs_evaluated = 0
        self.total_query_points = 0
        self.peak_pair_chunk = 0
        self.last_pair_chunk = 0

    def _build_csr(
        self,
        ids: torch.Tensor,
        tile_x0: torch.Tensor,
        tile_y0: torch.Tensor,
        widths: torch.Tensor,
        counts: torch.Tensor,
        component_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Direct two-pass CSR build in canonical component-ID order without a tile dictionary."""
        if ids.numel() == 0:
            return (
                torch.empty(0, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
                torch.empty(0, dtype=component_dtype),
                0,
            )
        total = int(counts.sum())
        block_starts = torch.cumsum(counts, 0) - counts
        entry_owner = torch.repeat_interleave(torch.arange(ids.numel()), counts)
        within = torch.arange(total, dtype=torch.long) - block_starts[entry_owner]
        width_per_entry = widths[entry_owner]
        row = torch.div(within, width_per_entry, rounding_mode="floor")
        col = within - row * width_per_entry
        tile_key = (tile_y0[entry_owner] + row) * self.tiles_x + (tile_x0[entry_owner] + col)
        entry_component = ids[entry_owner]
        # Stable sort keeps ascending component order within each tile row, matching the frozen
        # grouped reference whose per-tile lists were appended in component order.
        order = torch.argsort(tile_key, stable=True)
        sorted_keys = tile_key[order]
        sorted_components = entry_component[order]
        unique_keys, tile_counts = torch.unique_consecutive(sorted_keys, return_counts=True)
        offsets = torch.zeros(unique_keys.numel() + 1, dtype=torch.long)
        torch.cumsum(tile_counts, 0, out=offsets[1:])
        return (
            unique_keys.contiguous(),
            offsets,
            sorted_components.to(component_dtype).contiguous(),
            int(tile_counts.max()),
        )

    @staticmethod
    def _validate_limits(
        tile_size: int,
        max_entries: int | None,
        max_candidates: int | None,
        max_query_pairs: int | None = None,
    ) -> None:
        if not isinstance(tile_size, int) or isinstance(tile_size, bool) or tile_size <= 0:
            raise ValueError("tile_size must be a positive integer")
        for name, value in (
            ("max_entries", max_entries),
            ("max_candidates", max_candidates),
            ("max_query_pairs", max_query_pairs),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer or None")

    @staticmethod
    def _tile_bounds(field: GaussianObservationField, tile_size: int):
        """Yield exact clipped inclusive tile bounds without allocating index state.

        This scalar Python generator is retained as the frozen membership oracle for the private
        grouped reference and its equivalence test; the production build uses the vectorized
        :meth:`_component_tile_ranges`, which is tested to reproduce it exactly.
        """
        tiles_x = math.ceil(field.width / tile_size)
        tiles_y = math.ceil(field.height / tile_size)
        fit_x, fit_y, fit_width, fit_height = field.fit_window
        fit_right = fit_x + fit_width
        fit_bottom = fit_y + fit_height
        radii = field.radii()
        support_centers = field.support_centers()
        for component_id in range(field.n):
            if float(field.amplitudes[component_id]) <= 0.0:
                continue
            center_x = float(support_centers[component_id, 0])
            center_y = float(support_centers[component_id, 1])
            radius_x = int(radii[component_id, 0])
            radius_y = int(radii[component_id, 1])
            left = max(center_x - radius_x, fit_x)
            right = min(center_x + radius_x, fit_right)
            top = max(center_y - radius_y, fit_y)
            bottom = min(center_y + radius_y, fit_bottom)
            if left >= fit_right or right < fit_x or top >= fit_bottom or bottom < fit_y:
                continue
            tile_x0 = max(0, math.floor(left / tile_size))
            tile_x1 = min(tiles_x - 1, math.floor(right / tile_size))
            tile_y0 = max(0, math.floor(top / tile_size))
            tile_y1 = min(tiles_y - 1, math.floor(bottom / tile_size))
            yield component_id, tile_x0, tile_x1, tile_y0, tile_y1

    @staticmethod
    def _component_tile_ranges(
        field: GaussianObservationField,
        tile_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized exact clipped inclusive tile ranges for every overlapping component.

        Returns ``(component_ids, tile_x0, tile_x1, tile_y0, tile_y1)`` in ascending component
        order.  The clipped-rectangle arithmetic mirrors :meth:`_tile_bounds` exactly, computed in
        float64 so the ``float(...)`` upcast in the scalar oracle is reproduced bit-for-bit.
        """
        tiles_x = math.ceil(field.width / tile_size)
        tiles_y = math.ceil(field.height / tile_size)
        fit_x, fit_y, fit_width, fit_height = field.fit_window
        fit_right = fit_x + fit_width
        fit_bottom = fit_y + fit_height
        centers = field.support_centers().to(torch.float64)
        radii = field.radii().to(torch.float64)
        left = torch.clamp(centers[:, 0] - radii[:, 0], min=float(fit_x))
        right = torch.clamp(centers[:, 0] + radii[:, 0], max=float(fit_right))
        top = torch.clamp(centers[:, 1] - radii[:, 1], min=float(fit_y))
        bottom = torch.clamp(centers[:, 1] + radii[:, 1], max=float(fit_bottom))
        overlap = (
            (field.amplitudes > 0.0)
            & (left < fit_right)
            & (right >= fit_x)
            & (top < fit_bottom)
            & (bottom >= fit_y)
        )
        ids = overlap.nonzero(as_tuple=True)[0]
        scale = float(tile_size)
        tile_x0 = torch.clamp(torch.floor(left[ids] / scale).long(), min=0)
        tile_x1 = torch.clamp(torch.floor(right[ids] / scale).long(), max=tiles_x - 1)
        tile_y0 = torch.clamp(torch.floor(top[ids] / scale).long(), min=0)
        tile_y1 = torch.clamp(torch.floor(bottom[ids] / scale).long(), max=tiles_y - 1)
        return ids, tile_x0, tile_x1, tile_y0, tile_y1

    @classmethod
    def estimate_entries(cls, field: GaussianObservationField, tile_size: int = 16) -> int:
        """Exactly count component--tile overlaps before any CSR allocation."""
        if field.device.type != "cpu":
            raise ValueError("GaussianObservationIndex is the CPU reference backend")
        cls._validate_limits(tile_size, None, None)
        _, tile_x0, tile_x1, tile_y0, tile_y1 = cls._component_tile_ranges(field, tile_size)
        return int(((tile_x1 - tile_x0 + 1) * (tile_y1 - tile_y0 + 1)).sum())

    @property
    def n_tiles(self) -> int:
        """Number of non-empty indexed tiles."""
        return int(self.tile_keys.numel())

    @property
    def payload_bytes(self) -> int:
        """Bytes retained by the three CSR arrays (excludes the shared field)."""
        return sum(
            tensor.element_size() * tensor.numel()
            for tensor in (self.tile_keys, self.tile_offsets, self.component_ids)
        )

    @property
    def stats(self) -> GaussianObservationIndexStats:
        """Immutable index size/candidate diagnostics for preflight and run records."""
        return GaussianObservationIndexStats(
            tile_size=self.tile_size,
            tiles_x=self.tiles_x,
            tiles_y=self.tiles_y,
            nonempty_tiles=self.n_tiles,
            total_entries=self.n_entries,
            max_candidates=self.max_candidates,
            retained_bytes=self.payload_bytes,
            component_id_dtype=str(self.component_id_dtype).removeprefix("torch."),
            max_query_pairs=self.max_query_pairs,
        )

    def _effective_pair_chunk(self, point_count: int, component_chunk: int) -> int:
        """Cap the transient pair chunk by both the hard cap and the caller-implied budget."""
        budget = max(1, point_count * component_chunk)
        if self.max_query_pairs is not None:
            budget = min(budget, self.max_query_pairs)
        return max(1, budget)

    def _iter_pair_chunks(self, xy: torch.Tensor, component_chunk: int):
        """Yield bounded ``(point_index, component_id)`` chunks in canonical order.

        Points keep their original order; components stream ascending within each point's CSR row.
        A single row may cross a chunk boundary — the reduction stays deterministic because chunks
        are contiguous slices of one global canonical pair sequence.
        """
        field = self.field
        count = xy.shape[0]
        valid_indices = field.valid_domain(xy).nonzero(as_tuple=True)[0]
        if not self.tile_keys.numel() or not valid_indices.numel():
            return
        valid_xy = xy[valid_indices]
        tile_x = torch.floor(valid_xy[:, 0] / self.tile_size).long()
        tile_y = torch.floor(valid_xy[:, 1] / self.tile_size).long()
        keys = tile_y * self.tiles_x + tile_x
        position = torch.searchsorted(self.tile_keys, keys)
        clamped = position.clamp_max(self.tile_keys.numel() - 1)
        found = (position < self.tile_keys.numel()) & (self.tile_keys[clamped] == keys)
        rows = position[found]
        point_index = valid_indices[found]
        if not rows.numel():
            return
        row_start = self.tile_offsets[rows]
        row_len = self.tile_offsets[rows + 1] - row_start
        self.total_query_points += int(point_index.numel())
        pair_offsets = torch.zeros(row_len.numel() + 1, dtype=torch.long)
        torch.cumsum(row_len, 0, out=pair_offsets[1:])
        total_pairs = int(pair_offsets[-1])
        effective = self._effective_pair_chunk(count, component_chunk)
        for start in range(0, total_pairs, effective):
            stop = min(start + effective, total_pairs)
            positions = torch.arange(start, stop, dtype=torch.long)
            owner = torch.searchsorted(pair_offsets, positions, right=True) - 1
            csr_index = row_start[owner] + (positions - pair_offsets[owner])
            chunk = int(positions.numel())
            self.last_pair_chunk = chunk
            self.peak_pair_chunk = max(self.peak_pair_chunk, chunk)
            self.total_pairs_evaluated += chunk
            yield point_index[owner], self.component_ids[csr_index].long()

    def _accumulate(self, buffer: torch.Tensor, point_index: torch.Tensor, values: torch.Tensor):
        if buffer.requires_grad or values.requires_grad:
            return buffer.index_add(0, point_index, values)
        buffer.index_add_(0, point_index, values)
        return buffer

    def query(self, xy: torch.Tensor, component_chunk: int = 4096) -> ObservationQuery:
        """Query colors through the exact CSR pair stream."""
        xy = self.field._validate_xy(xy)
        if component_chunk <= 0:
            raise ValueError("component_chunk must be positive")
        numerator = torch.zeros(xy.shape[0], 3, dtype=self.field.dtype)
        denominator = torch.zeros(xy.shape[0], dtype=self.field.dtype)
        for point_index, component_ids in self._iter_pair_chunks(xy, component_chunk):
            weights, colors = self.field._paired_values(xy[point_index], component_ids)
            denominator = self._accumulate(denominator, point_index, weights)
            numerator = self._accumulate(numerator, point_index, weights[:, None] * colors)
        if self.field.blend_mode == "normalized":
            color = numerator / (denominator[:, None] + self.field.epsilon)
        else:
            color = numerator
        return ObservationQuery(
            color=color,
            numerator=numerator,
            weight_sum=denominator,
            valid=self.field.valid_domain(xy),
        )

    def query_weight_sum(self, xy: torch.Tensor, component_chunk: int = 4096) -> torch.Tensor:
        """Query only the exact CSR denominator, sharing the pair stream but skipping color."""
        xy = self.field._validate_xy(xy)
        if component_chunk <= 0:
            raise ValueError("component_chunk must be positive")
        denominator = torch.zeros(xy.shape[0], dtype=self.field.dtype)
        for point_index, component_ids in self._iter_pair_chunks(xy, component_chunk):
            weights = self.field._paired_weights(xy[point_index], component_ids)
            denominator = self._accumulate(denominator, point_index, weights)
        return denominator


class _GroupedObservationIndexReference:
    """Frozen pre-CSR grouped tile index, retained only as a parity/benchmark oracle.

    This reproduces the original dictionary-of-per-tile-tensors build and per-tile
    ``_cross_values`` query.  It is deliberately private: the CSR
    :class:`GaussianObservationIndex` is the production default, and this class exists so tests and
    the placement benchmark can compare the accelerated path against the exact behavior it
    replaced.
    """

    def __init__(
        self,
        field: GaussianObservationField,
        tile_size: int = 16,
        *,
        max_entries: int | None = GaussianObservationIndex.DEFAULT_MAX_ENTRIES,
        max_candidates: int | None = GaussianObservationIndex.DEFAULT_MAX_CANDIDATES,
    ):
        if field.device.type != "cpu":
            raise ValueError("grouped observation index is the CPU reference backend")
        GaussianObservationIndex._validate_limits(tile_size, max_entries, max_candidates)
        self.field = field
        self.tile_size = int(tile_size)
        self.tiles_x = math.ceil(field.width / self.tile_size)
        self.tiles_y = math.ceil(field.height / self.tile_size)
        tile_lists: dict[int, list[int]] = {}
        for (
            component_id,
            tile_x0,
            tile_x1,
            tile_y0,
            tile_y1,
        ) in GaussianObservationIndex._tile_bounds(field, self.tile_size):
            for tile_y in range(tile_y0, tile_y1 + 1):
                for tile_x in range(tile_x0, tile_x1 + 1):
                    tile_lists.setdefault(tile_y * self.tiles_x + tile_x, []).append(component_id)
        self._tiles = {
            key: torch.tensor(component_ids, dtype=torch.long, device=field.device)
            for key, component_ids in tile_lists.items()
        }
        self.n_entries = sum(ids.numel() for ids in self._tiles.values())
        self.max_candidates = max((len(ids) for ids in tile_lists.values()), default=0)

    @property
    def n_tiles(self) -> int:
        return len(self._tiles)

    def query(self, xy: torch.Tensor, component_chunk: int = 4096) -> ObservationQuery:
        xy = self.field._validate_xy(xy)
        if component_chunk <= 0:
            raise ValueError("component_chunk must be positive")
        numerator = torch.zeros(xy.shape[0], 3, dtype=self.field.dtype)
        denominator = torch.zeros(xy.shape[0], dtype=self.field.dtype)
        for point_indices, component_ids in self._groups(xy):
            points = xy[point_indices]
            local_num = torch.zeros(points.shape[0], 3, dtype=self.field.dtype)
            local_den = torch.zeros(points.shape[0], dtype=self.field.dtype)
            for start in range(0, component_ids.numel(), component_chunk):
                ids = component_ids[start : start + component_chunk]
                weights, colors = self.field._cross_values(points, ids)
                local_den = local_den + weights.sum(dim=1)
                local_num = local_num + (weights[..., None] * colors).sum(dim=1)
            numerator[point_indices] = local_num
            denominator[point_indices] = local_den
        if self.field.blend_mode == "normalized":
            color = numerator / (denominator[:, None] + self.field.epsilon)
        else:
            color = numerator
        return ObservationQuery(
            color=color,
            numerator=numerator,
            weight_sum=denominator,
            valid=self.field.valid_domain(xy),
        )

    def query_weight_sum(self, xy: torch.Tensor, component_chunk: int = 4096) -> torch.Tensor:
        xy = self.field._validate_xy(xy)
        if component_chunk <= 0:
            raise ValueError("component_chunk must be positive")
        denominator = torch.zeros(xy.shape[0], dtype=self.field.dtype)
        for point_indices, component_ids in self._groups(xy):
            points = xy[point_indices]
            local = torch.zeros(points.shape[0], dtype=self.field.dtype)
            for start in range(0, component_ids.numel(), component_chunk):
                ids = component_ids[start : start + component_chunk]
                local = local + self.field._cross_weights(points, ids).sum(dim=1)
            denominator[point_indices] = local
        return denominator

    def _groups(self, xy: torch.Tensor):
        valid_indices = self.field.valid_domain(xy).nonzero(as_tuple=True)[0]
        if not valid_indices.numel():
            return
        valid_xy = xy[valid_indices]
        tile_x = torch.floor(valid_xy[:, 0] / self.tile_size).long()
        tile_y = torch.floor(valid_xy[:, 1] / self.tile_size).long()
        keys = tile_y * self.tiles_x + tile_x
        for key in keys.unique(sorted=True).tolist():
            component_ids = self._tiles.get(key)
            if component_ids is not None:
                yield valid_indices[keys == key], component_ids


class ObservationQueryBackend(Protocol):
    """Pluggable point-query surface shared by reference, indexed, and future GPU paths."""

    def query(self, xy: torch.Tensor, component_chunk: int = 4096) -> ObservationQuery: ...

    def query_weight_sum(self, xy: torch.Tensor, component_chunk: int = 4096) -> torch.Tensor: ...


def _half_open_uniform_xy(
    unit_xy: torch.Tensor,
    fit_window: tuple[int, int, int, int],
) -> torch.Tensor:
    """Map ``[0,1)`` draws into a dtype-safe half-open fitted window.

    A float draw below one can still round to the exclusive upper endpoint after a
    native-resolution multiply/add. Clamp that rare result to the predecessor representable in
    the draw dtype. This preserves the draw count and RNG stream; it never resamples or drops an
    attempt.
    """
    if unit_xy.ndim != 2 or unit_xy.shape[1] != 2 or not unit_xy.is_floating_point():
        raise ValueError("unit_xy must be a floating tensor with shape (N,2)")
    fit_x, fit_y, fit_width, fit_height = fit_window
    if fit_width <= 0 or fit_height <= 0:
        raise ValueError("fit_window must have positive width and height")
    lower = unit_xy.new_tensor([fit_x, fit_y])
    extent = unit_xy.new_tensor([fit_width, fit_height])
    upper = lower + extent
    upper_open = torch.nextafter(upper, lower)
    xy = unit_xy * extent + lower
    return torch.maximum(torch.minimum(xy, upper_open), lower)


class GaussianPointProposal:
    """O(N)-state continuous Gaussian-mixture proposal with a uniform coverage floor.

    Component ``i`` is selected in proportion to ``amplitude_i * 2*pi*sqrt(det(Sigma_i))``
    and then sampled from its normalized Gaussian. Draws are thinned against the exact frozen
    support/fade weight; rejected draws become explicit null attempts and are never resampled.
    Consequently the active field subdensity is ``weight_sum(x) / total_mass`` and is invariant
    to exact amplitude splits. The uniform branch bounds importance weights and covers
    background. Fixed-attempt averaging, including nulls with zero loss, is required.
    """

    risk_measure = "continuous_area"

    def __init__(
        self,
        field: GaussianObservationField,
        query_backend: ObservationQueryBackend | None = None,
    ):
        self.field = field
        self.query_backend = field if query_backend is None else query_backend
        if (
            isinstance(self.query_backend, GaussianObservationIndex)
            and self.query_backend.field is not field
        ):
            raise ValueError("query backend must index the proposal field")
        determinant_sqrt = field.effective_variances().prod(dim=1).sqrt()
        self.component_masses = field.amplitudes * (2.0 * math.pi) * determinant_sqrt
        self.total_mass = self.component_masses.sum()
        if not bool(torch.isfinite(self.total_mass)) or float(self.total_mass) <= 0.0:
            raise ValueError("field has no positive Gaussian proposal mass")

    def gaussian_density(self, xy: torch.Tensor) -> torch.Tensor:
        """Continuous active subdensity of the thinned Gaussian branch."""
        return self.query_backend.query_weight_sum(xy) / self.total_mass

    def sample(
        self,
        count: int,
        *,
        uniform_fraction: float,
        generator: torch.Generator,
    ) -> ObservationSamples:
        """Draw attempts from ``eta * uniform + (1-eta) * q_G`` with uniform target risk."""
        if count <= 0:
            raise ValueError("count must be positive")
        if not 0.0 < uniform_fraction <= 1.0:
            raise ValueError("uniform_fraction must be in (0,1]")
        use_uniform = (
            torch.rand(count, generator=generator, device=self.field.device) < uniform_fraction
        )
        xy = torch.empty(count, 2, dtype=self.field.dtype, device=self.field.device)
        joint_density = torch.zeros(count, dtype=self.field.dtype, device=self.field.device)
        active = use_uniform.clone()
        component_ids = torch.full((count,), -1, dtype=torch.long, device=self.field.device)

        uniform_indices = use_uniform.nonzero(as_tuple=True)[0]
        if uniform_indices.numel():
            fit_x, fit_y, fit_width, fit_height = self.field.fit_window
            uniform_xy = torch.rand(
                uniform_indices.numel(),
                2,
                generator=generator,
                device=self.field.device,
                dtype=self.field.dtype,
            )
            uniform_xy = _half_open_uniform_xy(
                uniform_xy,
                (fit_x, fit_y, fit_width, fit_height),
            )
            xy[uniform_indices] = uniform_xy
            joint_density[uniform_indices] = uniform_fraction / (fit_width * fit_height)

        gaussian_indices = (~use_uniform).nonzero(as_tuple=True)[0]
        if gaussian_indices.numel():
            drawn = torch.multinomial(
                self.component_masses,
                gaussian_indices.numel(),
                replacement=True,
                generator=generator,
            )
            component_ids[gaussian_indices] = drawn
            normal = torch.randn(
                gaussian_indices.numel(),
                2,
                generator=generator,
                device=self.field.device,
                dtype=self.field.dtype,
            )
            scales = self.field.effective_variances()[drawn].sqrt()
            theta = self.field.rotations[drawn]
            cos = torch.cos(theta)
            sin = torch.sin(theta)
            local = normal * scales
            offsets = torch.stack(
                [
                    cos * local[:, 0] - sin * local[:, 1],
                    sin * local[:, 0] + cos * local[:, 1],
                ],
                dim=-1,
            )
            if self.field.mean_residuals is None:
                # Preserve schema-v1 proposal arithmetic for existing archives.
                sampled_xy = self.field.means[drawn] + offsets
            else:
                fit_x, fit_y, _, _ = self.field.fit_window
                origin = offsets.new_tensor([fit_x + 0.5, fit_y + 0.5])
                sampled_xy = self.field.local_means(drawn) + offsets + origin
            xy[gaussian_indices] = sampled_xy
            base_weight = self.field.amplitudes[drawn] * torch.exp(
                -0.5 * normal.square().sum(dim=1)
            )
            exact_weight = self.field.component_weight(xy[gaussian_indices], drawn)
            acceptance = (
                exact_weight / base_weight.clamp_min(torch.finfo(self.field.dtype).tiny)
            ).clamp(0.0, 1.0)
            accepted = (
                torch.rand(
                    gaussian_indices.numel(),
                    generator=generator,
                    device=self.field.device,
                    dtype=self.field.dtype,
                )
                < acceptance
            )
            active[gaussian_indices] = accepted
            joint_density[gaussian_indices] = torch.where(
                accepted,
                (1.0 - uniform_fraction) * exact_weight / self.total_mass,
                torch.zeros_like(exact_weight),
            )

        valid = self.field.valid_domain(xy)
        fit_width, fit_height = self.field.fit_window[2:]
        uniform_density = 1.0 / (fit_width * fit_height)
        target_density = active.to(self.field.dtype) * uniform_density
        gaussian_density = self.gaussian_density(xy)
        proposal_density = (
            uniform_fraction * valid.to(self.field.dtype) * uniform_density
            + (1.0 - uniform_fraction) * gaussian_density
        )
        importance = torch.where(
            active,
            target_density / proposal_density.clamp_min(torch.finfo(self.field.dtype).tiny),
            torch.zeros_like(target_density),
        )
        return ObservationSamples(
            xy=xy,
            proposal_component_ids=component_ids,
            proposal_density=proposal_density,
            joint_density=joint_density,
            target_density=target_density,
            importance=importance,
            inside_fit_window=valid,
            active=active,
            risk_measure=self.risk_measure,
        )


class GaussianPixelProposal:
    """O(N)-state proposal for the finite set of fitted-window pixel centers.

    Component ``i`` receives envelope mass ``amplitude_i * area_i``, where ``area_i`` is the
    number of discrete pixel centers in its clipped rounded support rectangle. The Gaussian
    branch selects a component by that mass, samples one rectangle pixel uniformly, and accepts
    it with probability ``component_weight / amplitude``. Rejections remain explicit null
    attempts. Thus its active marginal probability at pixel ``p`` is
    ``weight_sum(p) / envelope_mass`` without storing a pixel image or component-pixel table.

    Mixing this sub-proposal with a positive uniform fraction yields an unbiased fixed-attempt
    estimator of the uniform discrete-pixel risk. It is intentionally separate from
    :class:`GaussianPointProposal`, whose target is continuous fitted-window area.
    """

    risk_measure = "discrete_pixels"

    def __init__(
        self,
        field: GaussianObservationField,
        query_backend: ObservationQueryBackend | None = None,
    ):
        self.field = field
        self.query_backend = field if query_backend is None else query_backend
        if (
            isinstance(self.query_backend, GaussianObservationIndex)
            and self.query_backend.field is not field
        ):
            raise ValueError("query backend must index the proposal field")

        fit_x, fit_y, fit_width, fit_height = field.fit_window
        fit_lower = torch.tensor([fit_x, fit_y], dtype=torch.long, device=field.device)
        fit_upper = fit_lower + torch.tensor(
            [fit_width - 1, fit_height - 1],
            dtype=torch.long,
            device=field.device,
        )
        support_pixels = field.support_pixels()
        lower = torch.maximum(support_pixels - field.radii(), fit_lower)
        upper = torch.minimum(support_pixels + field.radii(), fit_upper)
        sizes = (upper - lower + 1).clamp_min(0)
        areas = sizes.prod(dim=1)

        self.rectangle_lower = lower
        self.rectangle_sizes = sizes
        self.component_areas = areas
        self.component_masses = field.amplitudes * areas.to(field.dtype)
        self.envelope_mass = self.component_masses.sum()
        self.pixel_count = fit_width * fit_height
        if not bool(torch.isfinite(self.envelope_mass)) or float(self.envelope_mass) <= 0.0:
            raise ValueError("field has no positive discrete Gaussian proposal envelope mass")

    def gaussian_probability(self, xy: torch.Tensor) -> torch.Tensor:
        """Active pixel probability of the rejection-thinned Gaussian branch."""
        xy = self.field._validate_xy(xy)
        pixel_indices = xy - 0.5
        if not bool(torch.equal(pixel_indices, pixel_indices.round())):
            raise ValueError("discrete Gaussian probabilities require pixel-center coordinates")
        return self.query_backend.query_weight_sum(xy) / self.envelope_mass

    def sample(
        self,
        count: int,
        *,
        uniform_fraction: float,
        generator: torch.Generator,
    ) -> ObservationSamples:
        """Draw fixed attempts from a uniform/discrete-Gaussian mixture."""
        if count <= 0:
            raise ValueError("count must be positive")
        if not 0.0 < uniform_fraction <= 1.0:
            raise ValueError("uniform_fraction must be in (0,1]")

        use_uniform = (
            torch.rand(count, generator=generator, device=self.field.device) < uniform_fraction
        )
        xy = torch.empty(count, 2, dtype=self.field.dtype, device=self.field.device)
        joint_density = torch.zeros(count, dtype=self.field.dtype, device=self.field.device)
        active = use_uniform.clone()
        component_ids = torch.full((count,), -1, dtype=torch.long, device=self.field.device)

        fit_x, fit_y, fit_width, fit_height = self.field.fit_window
        uniform_indices = use_uniform.nonzero(as_tuple=True)[0]
        if uniform_indices.numel():
            uniform_x = torch.randint(
                fit_width,
                (uniform_indices.numel(),),
                generator=generator,
                device=self.field.device,
            )
            uniform_y = torch.randint(
                fit_height,
                (uniform_indices.numel(),),
                generator=generator,
                device=self.field.device,
            )
            uniform_pixels = torch.stack([uniform_x + fit_x, uniform_y + fit_y], dim=-1)
            xy[uniform_indices] = uniform_pixels.to(self.field.dtype) + 0.5
            joint_density[uniform_indices] = uniform_fraction / self.pixel_count

        gaussian_indices = (~use_uniform).nonzero(as_tuple=True)[0]
        if gaussian_indices.numel():
            drawn = torch.multinomial(
                self.component_masses,
                gaussian_indices.numel(),
                replacement=True,
                generator=generator,
            )
            component_ids[gaussian_indices] = drawn
            areas = self.component_areas[drawn]
            # Grouped integer draws are exactly uniform for each component's potentially
            # different rectangle area without storing any component-pixel list.
            flat_offsets = torch.empty_like(areas)
            for component_id in drawn.unique(sorted=True).tolist():
                selected = drawn == component_id
                area = int(self.component_areas[component_id])
                flat_offsets[selected] = torch.randint(
                    area,
                    (int(selected.sum()),),
                    generator=generator,
                    device=self.field.device,
                )
            widths = self.rectangle_sizes[drawn, 0]
            pixel_x = self.rectangle_lower[drawn, 0] + torch.remainder(flat_offsets, widths)
            pixel_y = self.rectangle_lower[drawn, 1] + torch.div(
                flat_offsets,
                widths,
                rounding_mode="floor",
            )
            xy[gaussian_indices] = (
                torch.stack([pixel_x, pixel_y], dim=-1).to(self.field.dtype) + 0.5
            )

            exact_weight = self.field.component_weight(xy[gaussian_indices], drawn)
            acceptance = (
                exact_weight
                / self.field.amplitudes[drawn].clamp_min(torch.finfo(self.field.dtype).tiny)
            ).clamp(0.0, 1.0)
            accepted = (
                torch.rand(
                    gaussian_indices.numel(),
                    generator=generator,
                    device=self.field.device,
                    dtype=self.field.dtype,
                )
                < acceptance
            )
            active[gaussian_indices] = accepted
            joint_density[gaussian_indices] = torch.where(
                accepted,
                (1.0 - uniform_fraction) * exact_weight / self.envelope_mass,
                torch.zeros_like(exact_weight),
            )

        valid = self.field.valid_domain(xy)
        if not bool(valid.all()):
            raise RuntimeError("discrete proposal produced a point outside the fitted window")
        target_probability = active.to(self.field.dtype) / self.pixel_count
        proposal_probability = uniform_fraction / self.pixel_count + (
            1.0 - uniform_fraction
        ) * self.gaussian_probability(xy)
        importance = torch.where(
            active,
            target_probability / proposal_probability.clamp_min(torch.finfo(self.field.dtype).tiny),
            torch.zeros_like(target_probability),
        )
        return ObservationSamples(
            xy=xy,
            proposal_component_ids=component_ids,
            proposal_density=proposal_probability,
            joint_density=joint_density,
            target_density=target_probability,
            importance=importance,
            inside_fit_window=valid,
            active=active,
            risk_measure=self.risk_measure,
        )
