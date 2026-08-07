"""GPS stereo depth from transient renders of fitted 2D Gaussian fields.

This opt-in initializer keeps calibration and serialized Gaussian observation fields as the only
per-scene reconstruction inputs.  It renders two bounded proxies by exact field queries, performs
pure-torch calibrated rectification, asks a pluggable symmetric stereo backend for inverse depth,
samples that geometry only at original fitted-component centers, lifts the original 2D
covariances, and reduces the candidates to one exact 3D Gaussian budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField, GaussianObservationIndex
from rtgs.core.sh import sh_to_rgb
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.depth.stereo import RectifiedStereoRequest, StereoDepthBackend, StereoDepthPrediction
from rtgs.lift.base import lift_covariance
from rtgs.lift.compact_carve import (
    CompactInitializationResult,
    CompactLineage,
    _center_and_extent,
)
from rtgs.lift.merge import merge_by_voxel


@dataclass(frozen=True)
class FieldProxyConfig:
    """Exact bounded query and rectification controls."""

    resolution: int = 1024
    row_batch: int = 32
    tile_size: int = 16
    support_threshold: float = 1e-6
    max_index_entries: int = 16_000_000
    max_candidates_per_tile: int = 200_000
    max_query_pairs: int = 1_048_576

    def __post_init__(self) -> None:
        for name in (
            "resolution",
            "row_batch",
            "tile_size",
            "max_index_entries",
            "max_candidates_per_tile",
            "max_query_pairs",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.support_threshold) or self.support_threshold < 0:
            raise ValueError("support_threshold must be finite and non-negative")


@dataclass(frozen=True)
class FieldProxy:
    """One unrectified CPU proxy plus its calibrated letterboxed camera."""

    rgb: torch.Tensor  # (3,S,S), float32 CPU [0,1]
    support: torch.Tensor  # (S,S), float32 CPU {0,1}
    camera: Camera
    scale: float
    pad_x: float
    pad_y: float
    receipt: dict[str, object]


@dataclass(frozen=True)
class RectifiedStereoGeometry:
    """Shared-orientation stereo cameras and exact proxy homographies."""

    left_camera: Camera
    right_camera: Camera
    rectified_left_camera: Camera
    rectified_right_camera: Camera
    rectified_rotation: torch.Tensor
    rectified_intrinsics: torch.Tensor
    rectified_to_left: torch.Tensor
    rectified_to_right: torch.Tensor
    left_to_rectified: torch.Tensor
    right_to_rectified: torch.Tensor
    baseline_world: float
    focal_pixels: float


@dataclass(frozen=True)
class GPSFieldInitializerConfig:
    """Frozen component sampling, covariance lift, and exact-count reduction controls."""

    n_init_3d: int = 3000
    left_view: str = "C0001"
    right_view: str = "C0022"
    proxy_right_view: str = "C0022"
    near: float = 0.05
    bounds_scale: float = 0.5
    minimum_confidence: float = 0.05
    maximum_cycle_error_px: float = 1.0
    confidence_decay_tau_px: float = 1.0
    disparity_noise_floor_px: float = 0.5
    minimum_axial_sigma_fraction: float = 0.005
    maximum_axial_sigma_fraction: float = 0.25
    minimum_valid_candidate_fraction: float = 0.5
    voxel_size_extent_fraction: float = 1.0 / 512.0
    color_bin_size: float = 0.1
    init_opacity: float = 0.1
    sh_degree: int = 0
    proxy: FieldProxyConfig = FieldProxyConfig()

    def __post_init__(self) -> None:
        if self.n_init_3d <= 0:
            raise ValueError("n_init_3d must be positive")
        if self.sh_degree != 0:
            raise ValueError("GPS field initialization is frozen to SH degree zero")
        positive = (
            "near",
            "bounds_scale",
            "maximum_cycle_error_px",
            "confidence_decay_tau_px",
            "disparity_noise_floor_px",
            "minimum_axial_sigma_fraction",
            "maximum_axial_sigma_fraction",
            "voxel_size_extent_fraction",
            "color_bin_size",
            "init_opacity",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("minimum_confidence must lie in [0,1]")
        if not 0 < self.minimum_valid_candidate_fraction <= 1:
            raise ValueError("minimum_valid_candidate_fraction must lie in (0,1]")
        if self.minimum_axial_sigma_fraction > self.maximum_axial_sigma_fraction:
            raise ValueError("axial sigma fraction bounds are reversed")
        if not 0 < self.init_opacity < 1:
            raise ValueError("init_opacity must lie in (0,1)")


@dataclass(frozen=True)
class GPSFieldArtifacts:
    """Detached dense-adapter evidence retained outside the returned compact model."""

    left_proxy_rgb: torch.Tensor
    right_proxy_rgb: torch.Tensor
    left_proxy_support: torch.Tensor
    right_proxy_support: torch.Tensor
    left_rectified_rgb: torch.Tensor
    right_rectified_rgb: torch.Tensor
    left_rectified_support: torch.Tensor
    right_rectified_support: torch.Tensor
    prediction: StereoDepthPrediction
    geometry: RectifiedStereoGeometry
    receipt: dict[str, object]


class GPSFieldInitializationError(RuntimeError):
    """Fail-closed initializer error carrying non-outcome diagnostic counts."""

    def __init__(self, message: str, diagnostics: dict[str, object]):
        super().__init__(message)
        self.diagnostics = diagnostics


def native_to_proxy_coordinates(
    field: GaussianObservationField,
    xy: torch.Tensor,
    resolution: int,
) -> torch.Tensor:
    """Map full-native half-integer coordinates into a centered-square proxy."""

    side = max(field.width, field.height)
    scale = resolution / side
    pad_x = 0.5 * (side - field.width)
    pad_y = 0.5 * (side - field.height)
    offset = xy.new_tensor([pad_x, pad_y])
    return scale * (xy + offset)


def proxy_camera(camera: Camera, width: int, height: int, resolution: int) -> Camera:
    """Apply the exact centered-square native-to-proxy intrinsic transform."""

    if camera.width != width or camera.height != height:
        raise ValueError("field canvas and camera dimensions differ")
    side = max(width, height)
    scale = resolution / side
    pad_x = 0.5 * (side - width)
    pad_y = 0.5 * (side - height)
    return Camera(
        fx=scale * camera.fx,
        fy=scale * camera.fy,
        cx=scale * (camera.cx + pad_x),
        cy=scale * (camera.cy + pad_y),
        width=resolution,
        height=resolution,
        R=camera.R.detach().cpu(),
        t=camera.t.detach().cpu(),
    )


def render_field_proxy(
    field: GaussianObservationField,
    camera: Camera,
    config: FieldProxyConfig | None = None,
) -> FieldProxy:
    """Render one CPU proxy by exact indexed field queries at every proxy center."""

    config = FieldProxyConfig() if config is None else config
    if field.device.type != "cpu" or camera.R.device.type != "cpu":
        raise ValueError("field proxy rendering requires CPU field and camera")
    index = GaussianObservationIndex(
        field,
        tile_size=config.tile_size,
        max_entries=config.max_index_entries,
        max_candidates=config.max_candidates_per_tile,
        max_query_pairs=config.max_query_pairs,
    )
    side = max(field.width, field.height)
    scale = config.resolution / side
    pad_x = 0.5 * (side - field.width)
    pad_y = 0.5 * (side - field.height)
    rgb = torch.zeros(3, config.resolution, config.resolution, dtype=torch.float32)
    support = torch.zeros(config.resolution, config.resolution, dtype=torch.float32)
    x_proxy = torch.arange(config.resolution, dtype=field.dtype) + 0.5
    for row_start in range(0, config.resolution, config.row_batch):
        row_end = min(config.resolution, row_start + config.row_batch)
        y_proxy = torch.arange(row_start, row_end, dtype=field.dtype) + 0.5
        yy, xx = torch.meshgrid(y_proxy, x_proxy, indexing="ij")
        native = torch.stack(
            (xx / scale - pad_x, yy / scale - pad_y),
            dim=-1,
        ).reshape(-1, 2)
        query = index.query(native, component_chunk=4096)
        in_canvas = (
            (native[:, 0] >= 0.5)
            & (native[:, 0] <= field.width - 0.5)
            & (native[:, 1] >= 0.5)
            & (native[:, 1] <= field.height - 0.5)
        )
        active = query.valid & in_canvas & (query.weight_sum >= config.support_threshold)
        color = torch.where(
            active[:, None],
            query.color.clamp(0, 1),
            torch.zeros_like(query.color),
        )
        rows = row_end - row_start
        rgb[:, row_start:row_end] = color.reshape(rows, config.resolution, 3).permute(2, 0, 1)
        support[row_start:row_end] = active.reshape(rows, config.resolution).float()
    return FieldProxy(
        rgb=rgb,
        support=support,
        camera=proxy_camera(camera, field.width, field.height, config.resolution),
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        receipt={
            "schema": "rtgs.field_proxy.v1",
            "view_id": field.view_id,
            "field_components": field.n,
            "canvas": [field.width, field.height],
            "resolution": config.resolution,
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "support_pixels": int(support.sum()),
            "support_fraction": float(support.mean()),
            "index_entries": index.n_entries,
            "index_payload_bytes": index.payload_bytes,
            "query_points": config.resolution**2,
            "evaluated_pairs": index.total_pairs_evaluated,
            "peak_pair_chunk": index.peak_pair_chunk,
        },
    )


def _normalized(value: torch.Tensor, *, label: str) -> torch.Tensor:
    norm = value.norm()
    if not bool(torch.isfinite(norm)) or float(norm) <= 1e-8:
        raise ValueError(f"cannot normalize degenerate {label}")
    return value / norm


def build_rectified_stereo_geometry(
    left: Camera,
    right: Camera,
) -> RectifiedStereoGeometry:
    """Build the frozen shared-orientation, baseline-aligned rectification."""

    if left.width != right.width or left.height != right.height:
        raise ValueError("rectified proxy cameras must share dimensions")
    dtype = torch.float64
    center_left = left.position.to(dtype=dtype, device="cpu")
    center_right = right.position.to(dtype=dtype, device="cpu")
    x_axis = _normalized(center_right - center_left, label="stereo baseline")
    mean_down = _normalized(
        left.R[1].double().cpu() + right.R[1].double().cpu(),
        label="mean camera down",
    )
    y_axis = _normalized(mean_down - torch.dot(mean_down, x_axis) * x_axis, label="rectified down")
    z_axis = torch.linalg.cross(x_axis, y_axis)
    mean_forward = _normalized(
        left.R[2].double().cpu() + right.R[2].double().cpu(),
        label="mean camera forward",
    )
    if float(torch.dot(z_axis, mean_forward)) < 0:
        y_axis = -y_axis
        z_axis = -z_axis
    rotation = torch.stack((x_axis, y_axis, z_axis), dim=0)
    if not torch.allclose(rotation @ rotation.T, torch.eye(3, dtype=dtype), atol=1e-10):
        raise RuntimeError("rectified stereo rotation is not orthonormal")
    focal = min(left.fx, left.fy, right.fx, right.fy)
    cx, cy = left.width / 2.0, left.height / 2.0
    intrinsics = torch.tensor(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )
    rect_left = Camera(
        fx=focal,
        fy=focal,
        cx=cx,
        cy=cy,
        width=left.width,
        height=left.height,
        R=rotation.float(),
        t=(-rotation @ center_left).float(),
    )
    rect_right = Camera(
        fx=focal,
        fy=focal,
        cx=cx,
        cy=cy,
        width=right.width,
        height=right.height,
        R=rotation.float(),
        t=(-rotation @ center_right).float(),
    )

    def homographies(camera: Camera) -> tuple[torch.Tensor, torch.Tensor]:
        k_original = camera.K.double().cpu()
        r_original = camera.R.double().cpu()
        rectified_to_original = k_original @ r_original @ rotation.T @ torch.linalg.inv(intrinsics)
        original_to_rectified = torch.linalg.inv(rectified_to_original)
        return rectified_to_original.float(), original_to_rectified.float()

    rect_to_left, left_to_rect = homographies(left)
    rect_to_right, right_to_rect = homographies(right)
    return RectifiedStereoGeometry(
        left_camera=left,
        right_camera=right,
        rectified_left_camera=rect_left,
        rectified_right_camera=rect_right,
        rectified_rotation=rotation.float(),
        rectified_intrinsics=intrinsics.float(),
        rectified_to_left=rect_to_left,
        rectified_to_right=rect_to_right,
        left_to_rectified=left_to_rect,
        right_to_rectified=right_to_rect,
        baseline_world=float((center_right - center_left).norm()),
        focal_pixels=float(focal),
    )


def _homography_grid(
    homography: torch.Tensor,
    height: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float32
    y = torch.arange(height, dtype=dtype, device=device) + 0.5
    x = torch.arange(width, dtype=dtype, device=device) + 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    points = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
    mapped = points @ homography.to(device=device, dtype=dtype).T
    positive = mapped[..., 2] > 1e-8
    uv = mapped[..., :2] / mapped[..., 2:].clamp_min(1e-8)
    in_bounds = (
        positive
        & (uv[..., 0] >= 0.5)
        & (uv[..., 0] <= width - 0.5)
        & (uv[..., 1] >= 0.5)
        & (uv[..., 1] <= height - 0.5)
    )
    grid = torch.stack((2 * uv[..., 0] / width - 1, 2 * uv[..., 1] / height - 1), dim=-1)
    return grid, in_bounds


def rectify_field_proxies(
    left: FieldProxy,
    right: FieldProxy,
    geometry: RectifiedStereoGeometry,
    device: str | torch.device,
) -> tuple[RectifiedStereoRequest, dict[str, torch.Tensor]]:
    """Transfer two CPU proxies once and rectify RGB/support on the requested device."""

    target = torch.device(device)
    if left.rgb.device.type != "cpu" or right.rgb.device.type != "cpu":
        raise ValueError("unrectified field proxies must remain CPU-resident")
    height, width = left.support.shape
    if (height, width) != right.support.shape or height != width:
        raise ValueError("field proxy shapes differ or are not square")

    def warp(proxy: FieldProxy, homography: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rgb = proxy.rgb.to(device=target, dtype=torch.float32)
        support = proxy.support.to(device=target, dtype=torch.float32)
        grid, valid = _homography_grid(homography, height, width, target)
        rectified_rgb = F.grid_sample(
            rgb[None],
            grid[None],
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[0]
        rectified_support = F.grid_sample(
            support[None, None],
            grid[None],
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[0, 0]
        rectified_rgb = torch.where(valid[None], rectified_rgb, torch.zeros_like(rectified_rgb))
        rectified_support = torch.where(
            valid,
            rectified_support,
            torch.zeros_like(rectified_support),
        ).clamp(0, 1)
        return rectified_rgb.clamp(0, 1), rectified_support

    left_rgb, left_support = warp(left, geometry.rectified_to_left)
    right_rgb, right_support = warp(right, geometry.rectified_to_right)
    request = RectifiedStereoRequest(
        left_image=(2 * left_rgb - 1) * left_support[None],
        right_image=(2 * right_rgb - 1) * right_support[None],
        left_support=left_support,
        right_support=right_support,
        focal_pixels=geometry.focal_pixels,
        baseline_world=geometry.baseline_world,
    )
    request.validate()
    return request, {
        "left_rgb": left_rgb,
        "right_rgb": right_rgb,
        "left_support": left_support,
        "right_support": right_support,
    }


def apply_homography(
    homography: torch.Tensor, uv: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map half-integer pixel coordinates and return mapped coordinates plus positive scale."""

    homogeneous = torch.cat((uv, torch.ones_like(uv[:, :1])), dim=-1)
    mapped = homogeneous @ homography.to(uv).T
    positive = (mapped[:, 2] > 1e-8) & torch.isfinite(mapped).all(dim=1)
    denominator = torch.where(positive, mapped[:, 2], torch.ones_like(mapped[:, 2]))
    result = mapped[:, :2] / denominator[:, None]
    finite = torch.isfinite(result).all(dim=1)
    valid = positive & finite
    return torch.where(valid[:, None], result, torch.zeros_like(result)), valid


def _sample_map(map_value: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    height, width = map_value.shape
    finite = torch.isfinite(uv).all(dim=1)
    safe_uv = torch.where(finite[:, None], uv, torch.zeros_like(uv))
    grid = torch.stack((2 * safe_uv[:, 0] / width - 1, 2 * safe_uv[:, 1] / height - 1), dim=-1)
    sampled = F.grid_sample(
        map_value[None, None],
        grid[None, :, None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )[0, 0, :, 0]
    return torch.where(finite, sampled, torch.zeros_like(sampled))


def _component_covariances(field: GaussianObservationField) -> torch.Tensor:
    variances = field.effective_variances()
    theta = field.rotations
    cos, sin = theta.cos(), theta.sin()
    cov00 = cos.square() * variances[:, 0] + sin.square() * variances[:, 1]
    cov01 = cos * sin * (variances[:, 0] - variances[:, 1])
    cov11 = sin.square() * variances[:, 0] + cos.square() * variances[:, 1]
    return torch.stack(
        (
            torch.stack((cov00, cov01), dim=-1),
            torch.stack((cov01, cov11), dim=-1),
        ),
        dim=-2,
    )


def _cpu_prediction(prediction: StereoDepthPrediction) -> StereoDepthPrediction:
    values: dict[str, Any] = {}
    for name in (
        "left_inverse_depth",
        "right_inverse_depth",
        "left_confidence",
        "right_confidence",
        "left_cycle_error_px",
        "right_cycle_error_px",
        "left_flow_px",
        "right_flow_px",
    ):
        values[name] = getattr(prediction, name).detach().float().cpu()
    values["left_valid"] = prediction.left_valid.detach().cpu()
    values["right_valid"] = prediction.right_valid.detach().cpu()
    values["diagnostics"] = dict(prediction.diagnostics)
    result = StereoDepthPrediction(**values)
    result.validate()
    return result


def _subinputs(inputs: ReconstructionInputs, names: tuple[str, ...]) -> ReconstructionInputs:
    lookup = {name: index for index, name in enumerate(inputs.view_names)}
    if len(lookup) != inputs.n_views or any(name not in lookup for name in names):
        raise ValueError("GPS frozen view is missing or duplicated")
    indices = [lookup[name] for name in names]
    return ReconstructionInputs(
        observations=[inputs.observations[index] for index in indices],
        cameras=[inputs.cameras[index] for index in indices],
        view_names=list(names),
        bounds_hint=inputs.bounds_hint,
        name=f"{inputs.name}-gps-pair",
    )


class GPSFieldProxyInitializer:
    """Lift fitted components using depth from a transient GPS field-proxy pair."""

    def __init__(
        self,
        backend: StereoDepthBackend,
        config: GPSFieldInitializerConfig | None = None,
        *,
        device: str = "cuda:0",
    ):
        self.backend = backend
        self.config = GPSFieldInitializerConfig() if config is None else config
        self.device = device
        self.last_artifacts: GPSFieldArtifacts | None = None

    def initialize(self, inputs: ReconstructionInputs) -> CompactInitializationResult:
        """Return one exact-budget GPS initialization or fail without backfilling."""

        result, artifacts = self.initialize_with_artifacts(inputs)
        self.last_artifacts = artifacts
        return result

    def initialize_with_artifacts(
        self,
        inputs: ReconstructionInputs,
    ) -> tuple[CompactInitializationResult, GPSFieldArtifacts]:
        inputs.validate()
        if any(field.device.type != "cpu" for field in inputs.observations):
            raise ValueError("GPS field initializer requires CPU compact observations")
        pair = _subinputs(inputs, (self.config.left_view, self.config.right_view))
        proxy_source = _subinputs(inputs, (self.config.left_view, self.config.proxy_right_view))
        left_field, right_source_field = proxy_source.observations
        left_camera = pair.cameras[0]
        right_camera = pair.cameras[1]
        if (
            right_source_field.width != pair.observations[1].width
            or right_source_field.height != pair.observations[1].height
        ):
            raise ValueError("shuffled right field canvas differs from retained right geometry")
        left_proxy = render_field_proxy(left_field, left_camera, self.config.proxy)
        right_proxy = render_field_proxy(right_source_field, right_camera, self.config.proxy)
        geometry = build_rectified_stereo_geometry(left_proxy.camera, right_proxy.camera)
        request, rectified = rectify_field_proxies(
            left_proxy,
            right_proxy,
            geometry,
            self.device,
        )
        request = RectifiedStereoRequest(
            left_image=request.left_image,
            right_image=request.right_image,
            left_support=request.left_support,
            right_support=request.right_support,
            focal_pixels=request.focal_pixels,
            baseline_world=request.baseline_world,
            maximum_cycle_error_px=self.config.maximum_cycle_error_px,
            confidence_decay_tau_px=self.config.confidence_decay_tau_px,
        )
        prediction_gpu = self.backend.predict_pair(request)
        prediction = _cpu_prediction(prediction_gpu)
        rectified_cpu = {key: value.detach().float().cpu() for key, value in rectified.items()}
        del prediction_gpu, request, rectified
        release = getattr(self.backend, "release", None)
        if callable(release):
            release()
        target_device = torch.device(self.device)
        if target_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(target_device)
            torch.cuda.empty_cache()

        center, extent = _center_and_extent(inputs, torch.float32)
        lower = center - self.config.bounds_scale * extent
        upper = center + self.config.bounds_scale * extent
        candidate_parts: list[dict[str, torch.Tensor]] = []
        maps = (
            (
                prediction.left_inverse_depth,
                prediction.left_cycle_error_px,
                prediction.left_confidence,
                prediction.left_valid,
                geometry.left_to_rectified,
            ),
            (
                prediction.right_inverse_depth,
                prediction.right_cycle_error_px,
                prediction.right_confidence,
                prediction.right_valid,
                geometry.right_to_rectified,
            ),
        )
        for source_view, (field, camera, map_values) in enumerate(
            zip(pair.observations, pair.cameras, maps, strict=True)
        ):
            inverse_depth, cycle_error, confidence, dense_valid, to_rectified = map_values
            native_uv = field.native_means(dtype=torch.float32)
            proxy_uv = native_to_proxy_coordinates(field, native_uv, self.config.proxy.resolution)
            rectified_uv, homography_positive = apply_homography(to_rectified, proxy_uv)
            in_rectified = (
                homography_positive
                & (rectified_uv[:, 0] >= 0.5)
                & (rectified_uv[:, 0] <= self.config.proxy.resolution - 0.5)
                & (rectified_uv[:, 1] >= 0.5)
                & (rectified_uv[:, 1] <= self.config.proxy.resolution - 0.5)
            )
            valid_float = dense_valid.float()
            sampled_valid = _sample_map(valid_float, rectified_uv)
            denominator = sampled_valid.clamp_min(1e-8)
            sampled_inverse = _sample_map(inverse_depth * valid_float, rectified_uv) / denominator
            sampled_error = _sample_map(cycle_error * valid_float, rectified_uv) / denominator
            sampled_confidence = _sample_map(confidence * valid_float, rectified_uv) / denominator

            d_orig = torch.stack(
                (
                    (native_uv[:, 0] - camera.cx) / camera.fx,
                    (native_uv[:, 1] - camera.cy) / camera.fy,
                    torch.ones(field.n, dtype=native_uv.dtype),
                ),
                dim=-1,
            )
            q = d_orig @ (geometry.rectified_rotation @ camera.R.T).T
            q_z = q[:, 2]
            sampled_geometry_valid = (
                in_rectified
                & (sampled_valid >= 0.5)
                & torch.isfinite(sampled_inverse)
                & (sampled_inverse > 0)
                & torch.isfinite(sampled_error)
                & (sampled_error <= self.config.maximum_cycle_error_px)
                & torch.isfinite(sampled_confidence)
                & (sampled_confidence >= self.config.minimum_confidence)
                & torch.isfinite(q_z)
                & (q_z > 0)
            )
            safe_inverse = torch.where(
                sampled_geometry_valid,
                sampled_inverse,
                torch.ones_like(sampled_inverse),
            )
            safe_error = torch.where(
                sampled_geometry_valid,
                sampled_error,
                torch.zeros_like(sampled_error),
            )
            rectified_depth = safe_inverse.reciprocal()
            safe_q_z = torch.where(sampled_geometry_valid, q_z, torch.ones_like(q_z))
            original_depth = rectified_depth / safe_q_z
            disparity = (safe_inverse * geometry.focal_pixels * geometry.baseline_world).abs()
            sigma_disparity = self.config.disparity_noise_floor_px + safe_error
            sigma_rectified = (
                geometry.focal_pixels
                * geometry.baseline_world
                * sigma_disparity
                / disparity.square().clamp_min(1e-12)
            )
            sigma_rectified = sigma_rectified.clamp(
                self.config.minimum_axial_sigma_fraction * rectified_depth,
                self.config.maximum_axial_sigma_fraction * rectified_depth,
            )
            sigma_original = sigma_rectified / safe_q_z
            ray_sigma = sigma_original * d_orig.norm(dim=-1)
            means = camera.unproject(native_uv, original_depth)
            valid = (
                sampled_geometry_valid
                & torch.isfinite(original_depth)
                & (original_depth > self.config.near)
                & torch.isfinite(ray_sigma)
                & (ray_sigma > 0)
                & torch.isfinite(means).all(dim=1)
                & (means >= lower).all(dim=1)
                & (means <= upper).all(dim=1)
            )
            lift_depth = torch.where(valid, original_depth, torch.ones_like(original_depth))
            lift_ray_sigma = torch.where(valid, ray_sigma, torch.ones_like(ray_sigma))
            means = camera.unproject(native_uv, lift_depth)
            component_ids = torch.arange(field.n, dtype=torch.long)
            colors = (
                field.component_color(native_uv.to(field.dtype), component_ids).float().clamp(0, 1)
            )
            covariances_2d = _component_covariances(field).float()
            covariances_3d = lift_covariance(
                camera,
                native_uv,
                covariances_2d,
                lift_depth,
                lift_ray_sigma,
            )
            mass = (
                field.amplitudes.float()
                * (2 * math.pi)
                * field.effective_variances().float().prod(dim=1).sqrt()
            )
            normalized_mass = mass / mass.sum().clamp_min(torch.finfo(mass.dtype).eps)
            score = sampled_confidence * normalized_mass
            candidate_parts.append(
                {
                    "source_view": torch.full((field.n,), source_view, dtype=torch.long),
                    "component_id": component_ids,
                    "native_uv": native_uv,
                    "means": means,
                    "covariances": covariances_3d,
                    "colors": colors,
                    "depth": lift_depth,
                    "depth_sigma": torch.where(
                        valid, sigma_original, torch.ones_like(sigma_original)
                    ),
                    "ray_sigma": lift_ray_sigma,
                    "score": score,
                    "confidence": sampled_confidence,
                    "cycle_error": sampled_error,
                    "sampled_valid": sampled_valid,
                    "valid": valid,
                }
            )

        combined = {
            name: torch.cat([part[name] for part in candidate_parts]) for name in candidate_parts[0]
        }
        candidate_count = int(combined["valid"].numel())
        valid_indices = combined["valid"].nonzero(as_tuple=True)[0]
        valid_count = int(valid_indices.numel())
        diagnostics: dict[str, object] = {
            "schema": "rtgs.gps_field_proxy_initializer.v1",
            "source_views": [self.config.left_view, self.config.right_view],
            "proxy_right_view": self.config.proxy_right_view,
            "candidate_count": candidate_count,
            "valid_candidate_count": valid_count,
            "valid_depth_candidate_fraction": valid_count / candidate_count,
            "minimum_valid_candidate_fraction": self.config.minimum_valid_candidate_fraction,
            "bounds_center": center.tolist(),
            "bounds_extent": extent,
            "search_aabb_lower": lower.tolist(),
            "search_aabb_upper": upper.tolist(),
            "left_proxy": left_proxy.receipt,
            "right_proxy": right_proxy.receipt,
            "stereo": prediction.diagnostics,
        }
        if valid_count / candidate_count < self.config.minimum_valid_candidate_fraction:
            raise GPSFieldInitializationError(
                "GPS valid candidate fraction is below the frozen gate",
                diagnostics,
            )
        if valid_count < self.config.n_init_3d:
            raise GPSFieldInitializationError(
                "GPS produced fewer valid candidates than the exact output count",
                diagnostics,
            )

        raw = Gaussians3D.from_means_covs(
            combined["means"][valid_indices],
            combined["covariances"][valid_indices],
            combined["colors"][valid_indices],
            torch.full((valid_count,), self.config.init_opacity, dtype=torch.float32),
            sh_degree=self.config.sh_degree,
        )
        voxel_size = extent * self.config.voxel_size_extent_fraction
        merged, group = merge_by_voxel(
            raw,
            voxel_size=voxel_size,
            opacity_mode="union",
            component_weights=combined["score"][valid_indices],
            color_bin_size=self.config.color_bin_size,
            return_group=True,
        )
        keys = torch.cat(
            (
                torch.floor(raw.means / voxel_size).long(),
                torch.floor(
                    sh_to_rgb(raw.sh[:, 0]).clamp(0, 1) / self.config.color_bin_size
                ).long(),
            ),
            dim=-1,
        )
        unique_keys, expected_group = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
        if not torch.equal(group, expected_group) or merged.n != unique_keys.shape[0]:
            raise RuntimeError("GPS fusion group correspondence differs from the frozen key")
        group_scores = torch.zeros(merged.n, dtype=torch.float32).index_add_(
            0,
            group,
            combined["score"][valid_indices].float(),
        )
        representative = torch.full((merged.n,), -1, dtype=torch.long)
        source_view = combined["source_view"][valid_indices]
        component_id = combined["component_id"][valid_indices]
        score = combined["score"][valid_indices]
        for row in sorted(
            range(valid_count),
            key=lambda item: (
                -float(score[item]),
                int(source_view[item]),
                int(component_id[item]),
            ),
        ):
            group_id = int(group[row])
            if int(representative[group_id]) < 0:
                representative[group_id] = row
        if bool((representative < 0).any()):
            raise RuntimeError("GPS fusion left a group without representative lineage")
        order = sorted(
            range(merged.n),
            key=lambda item: (
                -float(group_scores[item]),
                *[int(value) for value in unique_keys[item].tolist()],
                int(source_view[representative[item]]),
                int(component_id[representative[item]]),
            ),
        )
        diagnostics["merged_group_count"] = merged.n
        diagnostics["voxel_size"] = voxel_size
        diagnostics["color_bin_size"] = self.config.color_bin_size
        if merged.n < self.config.n_init_3d:
            raise GPSFieldInitializationError(
                "GPS fusion produced fewer groups than the exact output count",
                diagnostics,
            )
        selected_groups = torch.tensor(order[: self.config.n_init_3d], dtype=torch.long)
        selected_valid_rows = representative[selected_groups]
        selected_candidate_rows = valid_indices[selected_valid_rows]
        gaussians = merged.subset(selected_groups)
        if gaussians.n != self.config.n_init_3d:
            raise RuntimeError("GPS field initializer exact-count selection failed")
        diagnostics.update(
            {
                "n_init_3d": gaussians.n,
                "selection_policy": (
                    "group_score_desc,voxel_key_lexicographic,representative_lineage"
                ),
                "selected_group_score_min": float(group_scores[selected_groups].min()),
                "selected_group_score_median": float(group_scores[selected_groups].median()),
                "selected_left_right_consistency_median_px": float(
                    combined["cycle_error"][selected_candidate_rows].median()
                ),
                "selected_left_right_consistency_p90_px": float(
                    torch.quantile(combined["cycle_error"][selected_candidate_rows], 0.9)
                ),
            }
        )
        result = CompactInitializationResult(
            gaussians=gaussians,
            lineage=CompactLineage(
                source_view_indices=combined["source_view"][selected_candidate_rows],
                source_component_indices=combined["component_id"][selected_candidate_rows],
                source_xy=combined["native_uv"][selected_candidate_rows],
            ),
            depths=combined["depth"][selected_candidate_rows],
            depth_sigmas=combined["depth_sigma"][selected_candidate_rows],
            ray_sigmas=combined["ray_sigma"][selected_candidate_rows],
            scores=group_scores[selected_groups],
            diagnostics=diagnostics,
        )
        artifacts = GPSFieldArtifacts(
            left_proxy_rgb=left_proxy.rgb,
            right_proxy_rgb=right_proxy.rgb,
            left_proxy_support=left_proxy.support,
            right_proxy_support=right_proxy.support,
            left_rectified_rgb=rectified_cpu["left_rgb"],
            right_rectified_rgb=rectified_cpu["right_rgb"],
            left_rectified_support=rectified_cpu["left_support"],
            right_rectified_support=rectified_cpu["right_support"],
            prediction=prediction,
            geometry=geometry,
            receipt={
                "schema": "rtgs.gps_field_artifacts.v1",
                "source_views": [self.config.left_view, self.config.right_view],
                "proxy_right_view": self.config.proxy_right_view,
                "resolution": self.config.proxy.resolution,
            },
        )
        return result, artifacts


__all__ = [
    "FieldProxy",
    "FieldProxyConfig",
    "GPSFieldArtifacts",
    "GPSFieldInitializationError",
    "GPSFieldInitializerConfig",
    "GPSFieldProxyInitializer",
    "RectifiedStereoGeometry",
    "apply_homography",
    "build_rectified_stereo_geometry",
    "native_to_proxy_coordinates",
    "proxy_camera",
    "rectify_field_proxies",
    "render_field_proxy",
]
