"""CPU contract tests for the opt-in GPS field-proxy stereo initializer."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.depth.gps_gaussian import (
    GPSGaussianConfig,
    GPSGaussianStereoBackend,
    load_sanitized_gps_state,
)
from rtgs.depth.stereo import (
    RectifiedStereoRequest,
    StereoDepthPrediction,
    stereo_depth_from_bidirectional_flow,
)
from rtgs.lift.gps_field import (
    FieldProxyConfig,
    GPSFieldInitializationError,
    GPSFieldInitializerConfig,
    GPSFieldProxyInitializer,
    apply_homography,
    build_rectified_stereo_geometry,
    native_to_proxy_coordinates,
    proxy_camera,
    render_field_proxy,
)


def _parallel_camera(x: float, *, size: int = 16) -> Camera:
    position = torch.tensor([x, 0.0, 0.0])
    rotation = torch.eye(3)
    return Camera(
        fx=10.0,
        fy=10.0,
        cx=size / 2,
        cy=size / 2,
        width=size,
        height=size,
        R=rotation,
        t=-rotation @ position,
    )


def _field_for_targets(
    camera: Camera,
    targets: torch.Tensor,
    colors: torch.Tensor,
    view_id: str,
) -> GaussianObservationField:
    means, depth = camera.project(targets)
    assert bool((depth > 0).all())
    count = targets.shape[0]
    return GaussianObservationField(
        width=camera.width,
        height=camera.height,
        means=means,
        log_scales=torch.log(torch.full((count, 2), 0.55)),
        rotations=torch.linspace(-0.2, 0.2, count),
        colors=colors,
        amplitudes=torch.linspace(0.7, 1.0, count),
        view_id=view_id,
        n_init=count,
        provider="synthetic_fixture",
    )


def _synthetic_inputs(*, include_shuffle: bool = False) -> ReconstructionInputs:
    targets = torch.tensor(
        [
            [-0.31, -0.27, 3.0],
            [0.26, -0.24, 3.0],
            [-0.28, 0.29, 3.0],
            [0.33, 0.25, 3.0],
        ]
    )
    colors = torch.tensor(
        [
            [0.9, 0.1, 0.1],
            [0.1, 0.8, 0.2],
            [0.1, 0.2, 0.9],
            [0.8, 0.7, 0.1],
        ]
    )
    left = _parallel_camera(-0.5)
    right = _parallel_camera(0.5)
    observations = [
        _field_for_targets(left, targets, colors, "left"),
        _field_for_targets(right, targets, colors, "right"),
    ]
    cameras = [left, right]
    names = ["left", "right"]
    if include_shuffle:
        observations.append(_field_for_targets(right, targets, 1.0 - colors, "shuffle"))
        cameras.append(right)
        names.append("shuffle")
    return ReconstructionInputs(
        observations=observations,
        cameras=cameras,
        view_names=names,
        bounds_hint=(torch.tensor([0.0, 0.0, 3.0]), 4.0),
        name="gps-field-fixture",
    )


class _ConstantDepthBackend:
    def __init__(self, inverse_depth: float = 1.0 / 3.0, *, valid: bool = True):
        self.inverse_depth = inverse_depth
        self.valid = valid
        self.requests: list[RectifiedStereoRequest] = []

    def predict_pair(self, request: RectifiedStereoRequest) -> StereoDepthPrediction:
        request.validate()
        self.requests.append(request)
        shape = request.left_support.shape
        inverse = torch.full(shape, self.inverse_depth, device=request.left_image.device)
        confidence = torch.ones_like(inverse) if self.valid else torch.zeros_like(inverse)
        cycle = torch.zeros_like(inverse)
        valid = torch.full(shape, self.valid, dtype=torch.bool, device=inverse.device)
        flow_scale = request.focal_pixels * request.baseline_world * self.inverse_depth
        result = StereoDepthPrediction(
            left_inverse_depth=inverse if self.valid else torch.zeros_like(inverse),
            right_inverse_depth=inverse if self.valid else torch.zeros_like(inverse),
            left_confidence=confidence,
            right_confidence=confidence.clone(),
            left_cycle_error_px=cycle,
            right_cycle_error_px=cycle.clone(),
            left_valid=valid,
            right_valid=valid.clone(),
            left_flow_px=torch.full_like(inverse, -flow_scale),
            right_flow_px=torch.full_like(inverse, flow_scale),
            diagnostics={"backend": "constant-depth-fixture"},
        )
        result.validate(shape)
        return result


def _initializer_config(*, proxy_right_view: str = "right") -> GPSFieldInitializerConfig:
    return GPSFieldInitializerConfig(
        n_init_3d=4,
        left_view="left",
        right_view="right",
        proxy_right_view=proxy_right_view,
        minimum_valid_candidate_fraction=1.0,
        proxy=FieldProxyConfig(
            resolution=16,
            row_batch=4,
            tile_size=4,
            max_index_entries=10_000,
            max_candidates_per_tile=100,
            max_query_pairs=1_024,
        ),
    )


def test_bidirectional_flow_uses_gps_signs_and_unclamped_cycle_sampling():
    support = torch.ones(3, 6)
    request = RectifiedStereoRequest(
        left_image=torch.zeros(3, 3, 6),
        right_image=torch.zeros(3, 3, 6),
        left_support=support,
        right_support=support.clone(),
        focal_pixels=10.0,
        baseline_world=0.5,
    )
    prediction = stereo_depth_from_bidirectional_flow(
        request,
        left_flow_px=-torch.ones(3, 6),
        right_flow_px=torch.ones(3, 6),
    )

    assert torch.equal(prediction.left_valid[:, 0], torch.zeros(3, dtype=torch.bool))
    assert bool(prediction.left_valid[:, 1:].all())
    assert bool(prediction.right_valid[:, :-1].all())
    assert torch.equal(prediction.right_valid[:, -1], torch.zeros(3, dtype=torch.bool))
    assert torch.allclose(prediction.left_inverse_depth[:, 1:], torch.full((3, 5), 0.2))
    assert torch.allclose(prediction.right_inverse_depth[:, :-1], torch.full((3, 5), 0.2))
    assert torch.equal(prediction.left_cycle_error_px[:, 1:], torch.zeros(3, 5))
    assert torch.equal(prediction.left_confidence[:, 1:], torch.ones(3, 5))


def test_proxy_is_an_exact_direct_field_query_with_frozen_letterbox_coordinates():
    camera = Camera(
        fx=8.0,
        fy=7.0,
        cx=2.0,
        cy=1.0,
        width=4,
        height=2,
        R=torch.eye(3),
        t=torch.zeros(3),
    )
    field = GaussianObservationField(
        width=4,
        height=2,
        means=torch.tensor([[1.5, 0.5]]),
        log_scales=torch.log(torch.tensor([[0.7, 0.6]])),
        rotations=torch.tensor([0.1]),
        colors=torch.tensor([[1.2, 0.3, -0.2]]),
        amplitudes=torch.tensor([0.8]),
        view_id="letterbox",
        provider="synthetic_fixture",
    )
    config = FieldProxyConfig(
        resolution=4,
        row_batch=2,
        tile_size=2,
        max_index_entries=100,
        max_candidates_per_tile=10,
        max_query_pairs=32,
    )
    proxy = render_field_proxy(field, camera, config)
    centers = torch.stack(
        torch.meshgrid(torch.arange(4) + 0.5, torch.arange(4) + 0.5, indexing="ij"),
        dim=-1,
    )[..., [1, 0]]
    native = centers.reshape(-1, 2) - torch.tensor([0.0, 1.0])
    query = field.query(native)
    in_canvas = (
        (native[:, 0] >= 0.5)
        & (native[:, 0] <= 3.5)
        & (native[:, 1] >= 0.5)
        & (native[:, 1] <= 1.5)
    )
    active = query.valid & in_canvas & (query.weight_sum >= config.support_threshold)
    expected = torch.where(active[:, None], query.color.clamp(0, 1), 0.0)

    assert torch.equal(proxy.rgb.permute(1, 2, 0).reshape(-1, 3), expected)
    assert torch.equal(proxy.support.reshape(-1), active.float())
    assert torch.equal(
        native_to_proxy_coordinates(field, field.native_means(), 4),
        torch.tensor([[1.5, 1.5]]),
    )
    assert proxy.camera.fx == 8.0
    assert proxy.camera.fy == 7.0
    assert proxy.camera.cx == 2.0
    assert proxy.camera.cy == 2.0
    assert proxy.receipt["query_points"] == 16


def test_rectification_aligns_epipolar_rows_and_parallel_pair_is_identity():
    left = _parallel_camera(-0.5)
    right = _parallel_camera(0.5)
    geometry = build_rectified_stereo_geometry(left, right)
    points = torch.tensor([[-0.2, -0.1, 3.0], [0.3, 0.25, 4.0]])
    left_uv, _ = left.project(points)
    right_uv, _ = right.project(points)
    left_rectified, left_valid = apply_homography(geometry.left_to_rectified, left_uv)
    right_rectified, right_valid = apply_homography(geometry.right_to_rectified, right_uv)

    assert bool(left_valid.all() and right_valid.all())
    assert torch.allclose(left_rectified[:, 1], right_rectified[:, 1], atol=1e-6)
    assert torch.allclose(geometry.rectified_to_left, torch.eye(3), atol=1e-6)
    assert torch.allclose(geometry.rectified_to_right, torch.eye(3), atol=1e-6)
    assert geometry.baseline_world == pytest.approx(1.0)


def test_cpu_fake_backend_exercises_exact_count_lift_and_shuffled_tensor_only():
    inputs = _synthetic_inputs(include_shuffle=True)
    correct_backend = _ConstantDepthBackend()
    correct, correct_artifacts = GPSFieldProxyInitializer(
        correct_backend,
        _initializer_config(),
        device="cpu",
    ).initialize_with_artifacts(inputs)
    shuffled_backend = _ConstantDepthBackend()
    shuffled, shuffled_artifacts = GPSFieldProxyInitializer(
        shuffled_backend,
        _initializer_config(proxy_right_view="shuffle"),
        device="cpu",
    ).initialize_with_artifacts(inputs)

    assert correct.n_init_3d == shuffled.n_init_3d == 4
    assert bool(torch.isfinite(correct.gaussians.means).all())
    assert correct.diagnostics["candidate_count"] == 8
    assert correct.diagnostics["valid_candidate_count"] == 8
    assert correct_artifacts.receipt["proxy_right_view"] == "right"
    assert shuffled_artifacts.receipt["proxy_right_view"] == "shuffle"
    assert not torch.equal(
        correct_backend.requests[0].right_image,
        shuffled_backend.requests[0].right_image,
    )
    assert torch.equal(
        correct_artifacts.geometry.right_to_rectified,
        shuffled_artifacts.geometry.right_to_rectified,
    )
    assert set(shuffled.lineage.source_view_indices.tolist()) <= {0, 1}
    assert int(shuffled.lineage.source_component_indices.max()) < 4


def test_invalid_dense_prediction_fails_at_gate_without_entering_bad_covariance_math():
    inputs = _synthetic_inputs()
    with pytest.raises(GPSFieldInitializationError, match="valid candidate fraction") as caught:
        GPSFieldProxyInitializer(
            _ConstantDepthBackend(valid=False),
            replace(_initializer_config(), minimum_valid_candidate_fraction=0.5),
            device="cpu",
        ).initialize(inputs)
    assert caught.value.diagnostics["valid_candidate_count"] == 0


def _state_signature(state: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> str:
    rows = [
        {"key": key, "shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(state.items())
        if key.startswith(prefixes)
    ]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_sanitized_checkpoint_loader_binds_bytes_hash_and_used_tensor_signatures(tmp_path: Path):
    state = {
        "img_encoder.layer.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "raft_stereo.layer.bias": torch.tensor([1.0, 2.0], dtype=torch.float16),
        "gs_parm_regresser.unused": torch.ones(1),
    }
    checkpoint = tmp_path / "state.pt"
    torch.save(state, checkpoint)
    payload = checkpoint.read_bytes()
    config = GPSGaussianConfig(
        repository=tmp_path,
        checkpoint=checkpoint,
        checkpoint_bytes=len(payload),
        checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
        used_key_count=2,
        used_signature_sha256=_state_signature(state, ("img_encoder.", "raft_stereo.")),
        img_encoder_key_count=1,
        img_encoder_signature_sha256=_state_signature(state, ("img_encoder.",)),
        raft_stereo_key_count=1,
        raft_stereo_signature_sha256=_state_signature(state, ("raft_stereo.",)),
    )

    loaded = load_sanitized_gps_state(config)
    assert set(loaded) == set(state)
    with pytest.raises(RuntimeError, match="byte size"):
        load_sanitized_gps_state(replace(config, checkpoint_bytes=len(payload) + 1))
    with pytest.raises(RuntimeError, match="combined key/shape/dtype signature"):
        load_sanitized_gps_state(replace(config, used_key_count=3))


def test_gps_adapter_construction_is_lazy_and_import_has_no_external_side_effects(tmp_path: Path):
    backend = GPSGaussianStereoBackend(
        GPSGaussianConfig(repository=tmp_path / "missing", checkpoint=tmp_path / "missing.pt")
    )
    with pytest.raises(RuntimeError, match="has not been loaded"):
        _ = backend.receipt

    script = """
import sys
import rtgs.depth.gps_gaussian
assert 'scipy' not in sys.modules
assert not any(name == 'core' or name.startswith('core.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_proxy_camera_rejects_field_calibration_canvas_mismatch():
    with pytest.raises(ValueError, match="canvas"):
        proxy_camera(_parallel_camera(0.0), width=15, height=16, resolution=16)
