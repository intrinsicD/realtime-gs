"""CPU sentinels for exact inverse-projection-fiber geometry.

These tests deliberately use explicit development geometry, not the frozen experiment roots.
They validate the mechanism without constructing or executing an official synthetic scene.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch
from benchmarks import inverse_projection_fiber_iter1e as iter1e
from benchmarks import inverse_projection_fiber_protocol as protocol
from benchmarks import inverse_projection_fiber_transaction as tx
from benchmarks.inverse_projection_fiber_transaction import (
    DEVELOPMENT_ONLY,
    ReceiptDomainError,
)

from rtgs.core.camera import Camera
from rtgs.lift.inverse_projection_fiber import (
    FreeGaussianGeometry,
    InverseProjectionFiber,
    covariance_projection_design,
    hard_correspondence_loss,
    pairwise_conic_cost,
    pairwise_gaussian_geometry_cost,
)
from rtgs.render.projection import (
    EWA_DILATION,
    EWA_NEAR,
    project_covariances_ewa,
)
from rtgs.render.torch_ref import _DILATION, _NEAR


@pytest.fixture
def tmp_path(request, tmp_path_factory):
    """Module-local temp path without pytest's persistent ``*current`` symlink."""

    digest = hashlib.sha256(request.node.nodeid.encode("utf-8")).hexdigest()[:20]
    path = tmp_path_factory.getbasetemp() / f"case_{digest}"
    path.mkdir()
    return path


def _camera(
    *,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    fx: float = 71.0,
    fy: float = 67.0,
    cx: float = 40.0,
    cy: float = 36.0,
) -> Camera:
    return Camera(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=80,
        height=72,
        R=rotation,
        t=translation,
    )


def _off_axis_cameras() -> tuple[Camera, Camera]:
    # A signed-permutation rotation is exactly orthonormal even after Camera's float32
    # storage conversion. Its optical axis points along -world-x rather than world-z.
    source = _camera(
        rotation=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        ),
        translation=torch.tensor([0.15, -0.25, 0.4]),
    )
    other = _camera(
        rotation=torch.eye(3),
        translation=torch.tensor([1.5, -0.5, 3.0]),
        fx=73.0,
        fy=69.0,
    )
    return source, other


def _fiber() -> tuple[InverseProjectionFiber, Camera, Camera]:
    source, other = _off_axis_cameras()
    fiber = InverseProjectionFiber(
        cameras=(source, other),
        source_view_indices=torch.tensor([0], dtype=torch.long),
        source_component_indices=torch.tensor([3], dtype=torch.long),
        source_means2d=torch.tensor([[24.5, 47.5]], dtype=torch.float64),
        source_covariances2d=torch.tensor(
            [[[5.2, 0.85], [0.85, 3.1]]],
            dtype=torch.float64,
        ),
        initial_depths=torch.tensor([2.25], dtype=torch.float64),
        depth_lower=1.2,
        depth_upper=3.6,
    )
    return fiber, source, other


def _source_errors(fiber: InverseProjectionFiber) -> tuple[float, float]:
    means2d, covariances2d, _ = fiber.source_projection()
    center_error = (means2d - fiber.source_means2d).norm(dim=-1).max()
    covariance_error = (
        (covariances2d - fiber.source_covariances2d)
        .flatten(1)
        .norm(dim=-1)
        .div(fiber.source_covariances2d.flatten(1).norm(dim=-1))
        .max()
    )
    return float(center_error.detach()), float(covariance_error.detach())


def test_source_projection_is_exact_before_and_after_parameter_perturbation():
    fiber, source, _ = _fiber()
    promoted_rotation = source.R.to(torch.float64)
    assert torch.equal(
        promoted_rotation @ promoted_rotation.T,
        torch.eye(3, dtype=torch.float64),
    )
    assert not torch.equal(
        fiber.source_means2d[0],
        fiber.source_means2d.new_tensor([source.cx, source.cy]),
    )

    initial_center_error, initial_covariance_error = _source_errors(fiber)
    assert initial_center_error <= 1e-8
    assert initial_covariance_error <= 1e-8

    with torch.no_grad():
        fiber.depth_logits.add_(0.71)
        fiber.cross.copy_(torch.tensor([[0.43, -0.27]], dtype=torch.float64))
        fiber.log_ray_scale.add_(0.38)

    perturbed_center_error, perturbed_covariance_error = _source_errors(fiber)
    assert perturbed_center_error <= 1e-8
    assert perturbed_covariance_error <= 1e-8


def test_fiber_covariance_is_spd_after_null_coordinate_perturbation():
    fiber, _, _ = _fiber()
    with torch.no_grad():
        fiber.depth_logits.sub_(0.44)
        fiber.cross.copy_(torch.tensor([[1.2, -0.8]], dtype=torch.float64))
        fiber.log_ray_scale.add_(0.7)

    means, covariances = fiber.means_covariances()
    eigenvalues = torch.linalg.eigvalsh(covariances)
    assert torch.isfinite(means).all()
    assert torch.isfinite(covariances).all()
    assert torch.equal(covariances, covariances.transpose(-1, -2))
    assert bool((eigenvalues > 0).all())
    assert bool((fiber.depths() >= 1.2).all())
    assert bool((fiber.depths() <= 3.6).all())


def test_covariance_projection_design_has_rank_five_then_six():
    cameras = (
        _camera(rotation=torch.eye(3), translation=torch.zeros(3)),
        _camera(rotation=torch.eye(3), translation=torch.tensor([-1.0, 0.0, 0.0])),
        _camera(rotation=torch.eye(3), translation=torch.tensor([0.0, -1.0, 0.0])),
    )
    world_mean = torch.tensor([0.2, -0.15, 2.3], dtype=torch.float64)

    pair_design = covariance_projection_design(world_mean.repeat(2, 1), cameras[:2])
    pair_singular_values = torch.linalg.svdvals(pair_design)
    pair_tolerance = pair_singular_values[0] * 1e-10
    assert int((pair_singular_values > pair_tolerance).sum()) == 5
    assert float(pair_singular_values[4] / pair_singular_values[0]) >= 1e-8

    triple_design = covariance_projection_design(world_mean.repeat(3, 1), cameras)
    triple_singular_values = torch.linalg.svdvals(triple_design)
    triple_tolerance = triple_singular_values[0] * 1e-10
    assert int((triple_singular_values > triple_tolerance).sum()) == 6
    assert float(triple_singular_values[5] / triple_singular_values[0]) >= 1e-8


def test_fiber_coordinates_match_central_finite_difference_gradients():
    fiber, _, other = _fiber()
    with torch.no_grad():
        fiber.depth_logits.add_(0.31)
        fiber.cross.copy_(torch.tensor([[0.17, -0.11]], dtype=torch.float64))
        fiber.log_ray_scale.add_(0.23)

    target_mean = torch.tensor([[33.2, 38.1]], dtype=torch.float64)
    target_covariance = torch.tensor(
        [[[4.1, -0.32], [-0.32, 2.7]]],
        dtype=torch.float64,
    )

    # A finite difference must hold the stopped Mahalanobis metric fixed. Recomputing its
    # forward value at each perturbation would differentiate a quantity that the specified
    # stop_gradient deliberately excludes from autograd.
    with torch.no_grad():
        baseline_projection = fiber.project(other)
        fixed_center_metric = 0.5 * (baseline_projection.covariances2d + target_covariance)

    def loss() -> torch.Tensor:
        projected = fiber.project(other)
        delta = projected.means2d - target_mean
        solved = torch.linalg.solve(fixed_center_metric, delta.unsqueeze(-1))
        center = (delta.unsqueeze(-2) @ solved).squeeze()
        conic = pairwise_conic_cost(
            projected.covariances2d,
            target_covariance,
        )[0, 0]
        return center + 0.25 * conic

    fiber.zero_grad(set_to_none=True)
    loss().backward()
    epsilon = 1e-6
    coordinates = (
        ("depth_logit", fiber.depth_logits, (0,)),
        ("cross_0", fiber.cross, (0, 0)),
        ("cross_1", fiber.cross, (0, 1)),
        ("log_ray_scale", fiber.log_ray_scale, (0,)),
    )
    for name, parameter, index in coordinates:
        autograd_value = float(parameter.grad[index])
        original = float(parameter.detach()[index])
        with torch.no_grad():
            parameter[index] = original + epsilon
        positive = float(loss().detach())
        with torch.no_grad():
            parameter[index] = original - epsilon
        negative = float(loss().detach())
        with torch.no_grad():
            parameter[index] = original
        finite_difference = (positive - negative) / (2.0 * epsilon)
        relative_error = abs(finite_difference - autograd_value) / max(
            1e-8,
            abs(finite_difference),
            abs(autograd_value),
        )
        assert abs(finite_difference) > 1e-8, name
        assert relative_error <= 2e-4, (
            name,
            finite_difference,
            autograd_value,
            relative_error,
        )


def test_hard_min_is_bit_exact_under_colocated_duplicate_target():
    predicted_means = torch.tensor(
        [[10.25, 11.75], [20.5, 18.25]],
        dtype=torch.float64,
    )
    predicted_covariances = torch.tensor(
        [
            [[2.7, 0.2], [0.2, 1.8]],
            [[3.4, -0.35], [-0.35, 2.2]],
        ],
        dtype=torch.float64,
    )
    target_means = torch.tensor(
        [[10.0, 12.0], [20.25, 18.5], [31.0, 8.0]],
        dtype=torch.float64,
    )
    target_covariances = torch.tensor(
        [
            [[2.5, 0.1], [0.1, 1.7]],
            [[3.1, -0.2], [-0.2, 2.0]],
            [[1.9, 0.0], [0.0, 2.6]],
        ],
        dtype=torch.float64,
    )
    original_cost = pairwise_gaussian_geometry_cost(
        predicted_means,
        predicted_covariances,
        target_means,
        target_covariances,
        include_conic=True,
    )
    original_loss, original_assignment = hard_correspondence_loss(original_cost)

    duplicated_means = torch.cat([target_means, target_means[1:2]], dim=0)
    duplicated_covariances = torch.cat(
        [target_covariances, target_covariances[1:2]],
        dim=0,
    )
    duplicated_cost = pairwise_gaussian_geometry_cost(
        predicted_means,
        predicted_covariances,
        duplicated_means,
        duplicated_covariances,
        include_conic=True,
    )
    duplicated_loss, duplicated_assignment = hard_correspondence_loss(duplicated_cost)

    assert torch.equal(original_loss, duplicated_loss)
    assert torch.equal(original_assignment, duplicated_assignment)


def test_free_geometry_cholesky_parameterization_remains_spd():
    means = torch.tensor(
        [[0.1, -0.2, 2.0], [-0.4, 0.3, 2.5]],
        dtype=torch.float64,
    )
    factors = torch.tensor(
        [
            [[0.7, 0.0, 0.0], [0.2, 0.5, 0.0], [-0.1, 0.15, 0.4]],
            [[0.45, 0.0, 0.0], [-0.3, 0.8, 0.0], [0.25, -0.2, 0.6]],
        ],
        dtype=torch.float64,
    )
    initial_covariances = factors @ factors.transpose(-1, -2)
    geometry = FreeGaussianGeometry(means, initial_covariances)
    assert torch.allclose(
        geometry.covariances(),
        initial_covariances,
        atol=1e-14,
        rtol=1e-14,
    )

    with torch.no_grad():
        geometry.log_diagonal.add_(
            torch.tensor([[0.3, -0.2, 0.1], [-0.4, 0.25, -0.15]], dtype=torch.float64)
        )
        geometry.lower.add_(
            torch.tensor([[1.1, -0.9, 0.6], [-0.75, 1.3, -0.5]], dtype=torch.float64)
        )
    covariances = geometry.covariances()
    eigenvalues = torch.linalg.eigvalsh(covariances)
    assert torch.isfinite(covariances).all()
    assert bool((eigenvalues > 0).all())

    covariances.square().sum().backward()
    assert torch.isfinite(geometry.log_diagonal.grad).all()
    assert torch.isfinite(geometry.lower.grad).all()


def test_shared_ewa_projection_matches_reference_equations_including_near_clamp():
    assert EWA_DILATION == _DILATION
    assert EWA_NEAR == _NEAR
    camera = _camera(
        rotation=torch.eye(3),
        translation=torch.zeros(3),
        fx=53.0,
        fy=61.0,
        cx=32.0,
        cy=30.0,
    )
    means = torch.tensor(
        [[0.01, -0.02, 0.02], [0.2, 0.1, 1.7]],
        dtype=torch.float32,
    )
    covariances = torch.tensor(
        [
            [[0.08, 0.01, -0.015], [0.01, 0.05, 0.012], [-0.015, 0.012, 0.04]],
            [[0.12, -0.02, 0.01], [-0.02, 0.09, 0.015], [0.01, 0.015, 0.07]],
        ],
        dtype=torch.float32,
    )

    projected = project_covariances_ewa(means, covariances, camera)
    reference_means_cam = camera.world_to_cam(means)
    reference_depth = reference_means_cam[:, 2]
    reference_means2d, _ = camera.project(means)
    reference_covariances_cam = camera.R.to(means) @ covariances @ camera.R.to(means).T
    safe_depth = reference_depth.clamp_min(_NEAR)
    reference_jacobians = torch.zeros(2, 2, 3, dtype=means.dtype)
    reference_jacobians[:, 0, 0] = camera.fx / safe_depth
    reference_jacobians[:, 0, 2] = -camera.fx * reference_means_cam[:, 0] / safe_depth**2
    reference_jacobians[:, 1, 1] = camera.fy / safe_depth
    reference_jacobians[:, 1, 2] = -camera.fy * reference_means_cam[:, 1] / safe_depth**2
    reference_covariances2d = (
        reference_jacobians @ reference_covariances_cam @ reference_jacobians.transpose(-1, -2)
    )
    reference_covariances2d = reference_covariances2d + _DILATION * torch.eye(2)

    assert torch.equal(projected.means_cam, reference_means_cam)
    assert torch.equal(projected.depth, reference_depth)
    assert torch.equal(projected.means2d, reference_means2d)
    assert torch.equal(projected.jacobians, reference_jacobians)
    assert torch.equal(projected.covariances2d, reference_covariances2d)


def _development_initialization():
    scene = protocol._build_scene(91, 92)
    common = protocol._new_fiber(
        cameras=scene["cameras"],
        source_view_indices=scene["source_view_indices"],
        source_component_indices=scene["source_component_indices"],
        source_means2d=scene["source_means2d"],
        source_covariances2d=scene["source_covariances2d"],
        initial_depths=scene["initial_depths"],
    )
    with torch.no_grad():
        means, covariances = common.means_covariances()
    return scene, means, covariances


def _assert_development_receipt(kind: str, receipt: dict[str, object]) -> None:
    domain = iter1e.DEVELOPMENT_DOMAIN
    assert receipt["schema"] == domain.schema(kind)
    assert receipt["namespace"] == domain.namespace
    assert receipt["receipt_domain"] == "development"
    assert receipt["root_consumption_status"] == DEVELOPMENT_ONLY
    assert receipt["roots"] == []
    domain.validate_receipt(kind, receipt)
    serialized = json.dumps(receipt, allow_nan=False, sort_keys=True)
    for literal in domain.forbidden_literals:
        assert literal not in serialized
    for root in domain.forbidden_roots:
        assert str(root) not in serialized


def _assert_durable_descriptor(kind: str, descriptor: dict[str, object]) -> dict[str, object]:
    path = Path(str(descriptor["path"]))
    assert path.is_file()
    assert protocol._sha256_file(path) == descriptor["sha256"]
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == descriptor["receipt"]
    _assert_development_receipt(kind, stored)
    return stored


def test_development_raw_common_rank_and_initialization_receipts_are_domain_pure(
    tmp_path,
):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    scene = protocol._build_scene(91, 92)
    raw = protocol._write_scene_input_receipt(
        scene,
        artifacts,
        domain=iter1e.DEVELOPMENT_DOMAIN,
        root_consumption_status=DEVELOPMENT_ONLY,
    )
    common, means, covariances, common_descriptor = protocol._write_common_constructor_receipt(
        scene,
        raw,
        artifacts,
        iter1e.DEVELOPMENT_DOMAIN,
        DEVELOPMENT_ONLY,
    )
    assert common is not None
    assert means is not None
    assert covariances is not None
    rank_value = protocol._rank_sentinel(scene)
    rank_descriptor = protocol._write_rank_receipt(
        scene,
        rank_value,
        artifacts,
        iter1e.DEVELOPMENT_DOMAIN,
        DEVELOPMENT_ONLY,
    )
    model, initialization_descriptor = protocol._build_and_write_initialization_receipt(
        "fiber_conic",
        scene,
        means,
        covariances,
        artifacts,
        iter1e.DEVELOPMENT_DOMAIN,
        DEVELOPMENT_ONLY,
    )
    assert model is not None

    raw_receipt = _assert_durable_descriptor("raw_inputs", raw)
    common_receipt = _assert_durable_descriptor(
        "common_constructor",
        common_descriptor,
    )
    rank_receipt = _assert_durable_descriptor("rank_sentinel", rank_descriptor)
    initialization_receipt = _assert_durable_descriptor(
        "arm_initialization",
        initialization_descriptor,
    )
    assert raw_receipt["scene_root"] == 91
    assert raw_receipt["initial_depth_root"] == 92
    assert common_receipt["pass"] is True
    assert rank_receipt["rank"]["pass"] is True
    assert initialization_receipt["pass"] is True

    for path in artifacts.rglob("*"):
        if not path.is_file():
            continue
        serialized = path.read_text(encoding="utf-8")
        for literal in iter1e.DEVELOPMENT_DOMAIN.forbidden_literals:
            assert literal not in serialized
        for root in iter1e.DEVELOPMENT_DOMAIN.forbidden_roots:
            assert str(root) not in serialized


def test_free_and_fiber_realized_initializations_meet_frozen_tolerance():
    scene, means, covariances = _development_initialization()
    for arm in ("free_conic", "fiber_conic"):
        model = protocol._build_model(arm, scene, means, covariances)
        receipt = protocol._initialization_equivalence(
            model,
            scene,
            means,
            covariances,
        )
        assert receipt["pass"], (arm, receipt)
        assert receipt["constructor_input_means_sha256"] == protocol._tensor_sha256(means)
        assert receipt["constructor_input_covariances_sha256"] == protocol._tensor_sha256(
            covariances
        )
        if arm.startswith("fiber_"):
            assert receipt["contract"] == "fiber_byte_identity"
            assert receipt["realized_means_sha256"] == receipt["constructor_input_means_sha256"]
            assert (
                receipt["realized_covariances_sha256"]
                == receipt["constructor_input_covariances_sha256"]
            )
        else:
            assert receipt["contract"] == "free_numerical_equivalence"
            assert (
                receipt["realized_covariances_sha256"]
                != receipt["constructor_input_covariances_sha256"]
            )


def test_fiber_byte_identity_rejects_one_ulp_covariance_drift(monkeypatch):
    scene, means, covariances = _development_initialization()
    model = protocol._build_model("fiber_conic", scene, means, covariances)
    original_geometry_state = protocol._geometry_state

    def one_ulp_drift(candidate):
        realized_means, realized_covariances = original_geometry_state(candidate)
        realized_covariances = realized_covariances.clone()
        value = realized_covariances[0, 0, 0]
        realized_covariances[0, 0, 0] = torch.nextafter(
            value,
            torch.full_like(value, torch.inf),
        )
        return realized_means, realized_covariances

    monkeypatch.setattr(protocol, "_geometry_state", one_ulp_drift)
    receipt = protocol._initialization_equivalence(model, scene, means, covariances)
    assert receipt["checks"]["realized_covariance_relative_maximum_le_1e-14"]
    assert not receipt["checks"]["fiber_realized_covariances_byte_identical"]
    assert not receipt["pass"]


def test_training_uses_durable_passing_initialization_and_writes_development_receipt(
    tmp_path,
    monkeypatch,
):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    scene, means, covariances = _development_initialization()
    (artifacts / "scene_91").mkdir()
    model, initialization_descriptor = protocol._build_and_write_initialization_receipt(
        "free_conic",
        scene,
        means,
        covariances,
        artifacts,
        iter1e.DEVELOPMENT_DOMAIN,
        DEVELOPMENT_ONLY,
    )
    assert model is not None
    initialization = _assert_durable_descriptor(
        "arm_initialization",
        initialization_descriptor,
    )
    assert initialization["pass"] is True

    monkeypatch.setattr(protocol, "UPDATES", 1)
    monkeypatch.setattr(protocol, "CHECKPOINT_INTERVAL", 1)
    arm_directory = Path(str(initialization_descriptor["path"])).parent
    record = protocol._train_arm(
        model,
        "free_conic",
        scene,
        arm_directory,
        "development-config",
        "development-provenance",
        initialization_descriptor,
        domain=iter1e.DEVELOPMENT_DOMAIN,
        root_consumption_status=DEVELOPMENT_ONLY,
    )
    stored = json.loads((arm_directory / "result.json").read_text(encoding="utf-8"))
    _assert_development_receipt("arm_result", stored)
    assert record["schema"] == iter1e.DEVELOPMENT_DOMAIN.schema("arm_result")
    assert record["root_consumption_status"] == DEVELOPMENT_ONLY
    assert stored["initialization_receipt"]["sha256"] == initialization_descriptor["sha256"]
    assert stored["updates"] == 1
    assert stored["objective_evaluations"] == 2


@pytest.mark.parametrize("mutation", ["absent", "same_content_new_inode"])
def test_training_rejects_missing_or_replaced_initialization_before_optimizer(
    tmp_path,
    monkeypatch,
    mutation,
):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    scene, means, covariances = _development_initialization()
    (artifacts / "scene_91").mkdir()
    model, initialization_descriptor = protocol._build_and_write_initialization_receipt(
        "free_conic",
        scene,
        means,
        covariances,
        artifacts,
        iter1e.DEVELOPMENT_DOMAIN,
        DEVELOPMENT_ONLY,
    )
    assert model is not None
    initialization_path = Path(str(initialization_descriptor["path"]))
    if mutation == "absent":
        initialization_path.unlink()
    else:
        replacement = initialization_path.with_name("unowned-initialization.json")
        replacement.write_bytes(initialization_path.read_bytes())
        replacement.replace(initialization_path)

    optimizer_calls: list[object] = []

    def optimizer_spy(*args, **kwargs):
        optimizer_calls.append((args, kwargs))
        raise AssertionError("optimizer reached after invalid durable initialization")

    monkeypatch.setattr(protocol.torch.optim, "Adam", optimizer_spy)
    with pytest.raises(
        (FileNotFoundError, RuntimeError),
        match="initialization|receipt entries",
    ):
        protocol._train_arm(
            model,
            "free_conic",
            scene,
            initialization_path.parent,
            "development-config",
            "development-provenance",
            initialization_descriptor,
            domain=iter1e.DEVELOPMENT_DOMAIN,
            root_consumption_status=DEVELOPMENT_ONLY,
        )
    assert optimizer_calls == []


def test_failing_initialization_is_durable_before_optimizer_and_training(
    tmp_path,
    monkeypatch,
):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    scene, means, covariances = _development_initialization()
    (artifacts / "scene_91").mkdir()
    original_equivalence = protocol._initialization_equivalence

    def failing_equivalence(model, candidate_scene, candidate_means, candidate_covariances):
        receipt = original_equivalence(
            model,
            candidate_scene,
            candidate_means,
            candidate_covariances,
        )
        return {
            **receipt,
            "checks": {**receipt["checks"], "injected_development_failure": False},
            "pass": False,
        }

    monkeypatch.setattr(protocol, "_initialization_equivalence", failing_equivalence)
    model, initialization_descriptor = protocol._build_and_write_initialization_receipt(
        "fiber_conic",
        scene,
        means,
        covariances,
        artifacts,
        iter1e.DEVELOPMENT_DOMAIN,
        DEVELOPMENT_ONLY,
    )
    assert model is not None
    stored = _assert_durable_descriptor(
        "arm_initialization",
        initialization_descriptor,
    )
    assert stored["pass"] is False
    optimizer_calls: list[object] = []

    def optimizer_spy(*args, **kwargs):
        optimizer_calls.append((args, kwargs))
        raise AssertionError("optimizer was constructed after failed initialization")

    monkeypatch.setattr(protocol.torch.optim, "Adam", optimizer_spy)
    with pytest.raises(RuntimeError, match="forbidden after failed initialization"):
        protocol._train_arm(
            model,
            "fiber_conic",
            scene,
            Path(str(initialization_descriptor["path"])).parent,
            "development-config",
            "development-provenance",
            initialization_descriptor,
            domain=iter1e.DEVELOPMENT_DOMAIN,
            root_consumption_status=DEVELOPMENT_ONLY,
        )
    assert optimizer_calls == []
    assert not (Path(str(initialization_descriptor["path"])).parent / "gaussians_init.ply").exists()


def test_combined_sentinel_requires_literal_true():
    for value in (False, None, 0, "PASS"):
        with pytest.raises(RuntimeError, match="combined validity sentinel failed"):
            protocol._require_combined_sentinels({"pass": value})
    protocol._require_combined_sentinels({"pass": True})


def test_combined_sentinel_includes_every_frozen_family():
    sentinels = {
        "construction": {"pass": True},
        "finite_difference": {"pass": True},
        "duplicate_hard_min": {"pass": True},
        "rank": [{"pass": True}],
        "checkpoint_count": 1,
        "checkpoint_expected_count": 1,
        "initialization_equivalence_pass": True,
        "checkpoint_sentinels_pass": True,
    }
    assert protocol._combined_sentinels_pass(sentinels)
    for key in (
        "construction",
        "finite_difference",
        "duplicate_hard_min",
        "rank",
        "initialization_equivalence_pass",
        "checkpoint_sentinels_pass",
    ):
        mutated = copy.deepcopy(sentinels)
        if key in {"construction", "finite_difference", "duplicate_hard_min"}:
            mutated[key]["pass"] = False
        elif key == "rank":
            mutated[key][0]["pass"] = False
        else:
            mutated[key] = False
        assert not protocol._combined_sentinels_pass(mutated), key

    mismatched_count = copy.deepcopy(sentinels)
    mismatched_count["checkpoint_count"] = 0
    assert not protocol._combined_sentinels_pass(mismatched_count)


def test_protocol_spec_and_receipt_domains_are_immutable_and_disjoint_in_memory():
    spec = iter1e.SPEC
    assert spec.official_domain is iter1e.OFFICIAL_DOMAIN
    assert spec.development_domain is iter1e.DEVELOPMENT_DOMAIN
    assert spec.official_domain.label == "official"
    assert spec.development_domain.label == "development"
    assert spec.official_domain.namespace != spec.development_domain.namespace
    assert spec.official_domain.schema_family != spec.development_domain.schema_family
    assert not set(spec.scene_roots + spec.initial_depth_roots) & {91, 92}

    with pytest.raises(FrozenInstanceError):
        spec.protocol_label = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="exactly three paired roots"):
        replace(spec, scene_roots=(1, 2))
    with pytest.raises(ValueError, match="pairwise distinct"):
        replace(spec, scene_roots=(1, 2, 3), initial_depth_roots=(3, 4, 5))
    with pytest.raises(ValueError, match="development label"):
        replace(spec, development_domain=spec.official_domain)
    with pytest.raises(ReceiptDomainError, match="cannot permit roots"):
        replace(spec.development_domain, permitted_roots=(91,))
    with pytest.raises(ReceiptDomainError, match="only DEVELOPMENT_ONLY"):
        replace(
            spec.development_domain,
            permitted_root_consumption_statuses=("DEVELOPMENT_TEST",),
        )


def _synthetic_protocol_spec(tmp_path: Path) -> protocol.ProtocolSpec:
    label = "synthetic-lifecycle-protocol"
    # Long synthetic sentinels avoid accidental substring collisions with ordinary hexadecimal
    # digests while retaining the development-domain leak check exercised by these tests.
    scene_roots = (8_201_000_001, 8_202_000_001, 8_203_000_001)
    depth_roots = (8_211_000_001, 8_212_000_001, 8_213_000_001)
    roots = scene_roots + depth_roots
    statuses = ("SYNTHETIC_IDLE", "SYNTHETIC_TRANSITION", "SYNTHETIC_CONSUMED")
    official = tx.ReceiptDomain(
        protocol_label=label,
        label="official",
        namespace="synthetic.inverse-projection.iter1e.v1",
        schema_family="synthetic_inverse_projection_iter1e",
        permitted_root_consumption_statuses=statuses,
        permitted_roots=roots,
        official_phases_permitted=True,
        commit_states_permitted=True,
    )
    development = tx.ReceiptDomain(
        protocol_label=label,
        label="development",
        namespace="synthetic.inverse-projection.development.iter1e.v1",
        schema_family="synthetic_inverse_projection_development_iter1e",
        permitted_root_consumption_statuses=(DEVELOPMENT_ONLY,),
        forbidden_roots=roots,
        forbidden_literals=(official.namespace, official.schema_family, *statuses),
    )
    return replace(
        iter1e.SPEC,
        protocol_label=label,
        official_domain=official,
        development_domain=development,
        scene_roots=scene_roots,
        initial_depth_roots=depth_roots,
        preregistration=protocol.ROOT / "CLAUDE.md",
        preregistration_review=protocol.ROOT / "README.md",
        verification_receipt=protocol.ROOT / "docs/RESEARCH_LOOP.md",
        implementation_review=protocol.ROOT / "docs/ARCHITECTURE.md",
        bound_historical_sha256=(),
        declared_source_paths=(Path("benchmarks/inverse_projection_fiber_protocol.py"),),
        verification_base=tmp_path / "verification",
        official_out=tmp_path / "result.json",
        official_artifacts_dir=tmp_path / "artifacts",
    )


def _synthetic_invalid_payload() -> dict[str, object]:
    observation = {"hashes": {"development.py": "0" * 64}, "errors": {}}
    return {
        "status": "INVALID",
        "phase": "synthetic_failure",
        "source_observation_start": observation,
        "first_source_observation": observation,
    }


def test_preflight_rejects_nonfrozen_official_paths_before_writing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen one-shot"):
        protocol._preflight_protocol(
            iter1e.SPEC,
            tmp_path / "arbitrary-result.json",
            tmp_path / "arbitrary-artifacts",
        )
    assert not list(tmp_path.iterdir())


def test_implementation_review_is_bound_to_exact_current_source_hashes(tmp_path: Path) -> None:
    synthetic_review = protocol.ROOT / "benchmarks/results/synthetic_iter1e_review.json"
    verification_source = iter1e.PREREGISTRATION_REVIEW
    spec = replace(
        _synthetic_protocol_spec(tmp_path),
        verification_receipt=verification_source,
        implementation_review=synthetic_review,
        declared_source_paths=(
            Path("benchmarks/inverse_projection_fiber_protocol.py"),
            verification_source.relative_to(protocol.ROOT),
            synthetic_review.relative_to(protocol.ROOT),
        ),
    )
    manifest = protocol._reviewed_source_manifest(spec)
    review = spec.development_domain.make_receipt(
        "implementation_review",
        {
            "status": "PASS",
            "recommendation": "PASS",
            "independent_review": True,
            "reviewed_protocol_label": spec.protocol_label,
            "reviewed_source_manifest": manifest,
            "reviewed_source_count": len(manifest),
            "reviewed_source_manifest_sha256": protocol._source_manifest_sha256(manifest),
            "verification_receipt_sha256": protocol._sha256_file(verification_source),
        },
        root_consumption_status=DEVELOPMENT_ONLY,
    )
    protocol._validate_implementation_review(spec, review)

    tampered_manifest = copy.deepcopy(manifest)
    tampered_manifest[0]["file_sha256"] = "0" * 64
    tampered = {**review, "reviewed_source_manifest": tampered_manifest}
    with pytest.raises(RuntimeError, match="current exact-source"):
        protocol._validate_implementation_review(spec, tampered)


def test_prepared_inode_hardlink_collider_never_authorizes_followup_mutation(
    tmp_path: Path,
) -> None:
    spec = _synthetic_protocol_spec(tmp_path)
    domain = spec.development_domain
    descriptor = tx.open_directory(tmp_path)
    try:
        receipt = domain.make_receipt(
            "hardlink_fixture",
            {"status": "DEVELOPMENT_FIXTURE"},
            root_consumption_status=DEVELOPMENT_ONLY,
        )
        prepared = tx.prepare_json(
            descriptor,
            "result.json",
            domain,
            "hardlink_fixture",
            receipt,
            nonce="1" * 32,
        )

        def inject_hardlink(event, _context):
            if event == "before_exclusive_link":
                os.link(
                    prepared.recovery_name,
                    prepared.target_name,
                    src_dir_fd=descriptor,
                    dst_dir_fd=descriptor,
                    follow_symlinks=False,
                )

        with pytest.raises(tx.OwnedMutationError) as caught:
            tx.publish_exclusive(descriptor, prepared, event_hook=inject_hardlink)
        safe_ownership, mutation_allowed = protocol._safe_public_state_after_error(
            caught.value,
            None,
        )
        assert safe_ownership is None
        assert mutation_allowed is False
        public = tx.capture_entry(descriptor, prepared.target_name)
        assert public.ownership is not None
        assert public.ownership.same_identity_and_hash(prepared.ownership)
    finally:
        os.close(descriptor)


def test_invalid_publication_never_mutates_a_forbidden_uncertain_name(tmp_path: Path) -> None:
    spec = _synthetic_protocol_spec(tmp_path)
    artifacts = spec.official_artifacts_dir
    artifacts.mkdir()
    terminal = artifacts / "terminal.json"
    lifecycle = artifacts / "lifecycle.json"
    publication = protocol._publish_invalid_protocol(
        spec,
        out=spec.official_out,
        terminal_path=terminal,
        lifecycle_path=lifecycle,
        payload=_synthetic_invalid_payload(),
        root_consumption_status=spec.official_domain.permitted_root_consumption_statuses[-1],
        result_ownership=None,
        terminal_ownership=None,
        lifecycle_ownership=None,
        result_mutation_allowed=False,
        terminal_mutation_allowed=True,
        lifecycle_mutation_allowed=False,
    )
    assert not spec.official_out.exists()
    assert terminal.is_file()
    assert not lifecycle.exists()
    assert "contested or uncertain" in publication["publication_failures"]["aggregate"]
    assert list(artifacts.glob("invalid-publication-fallback-*.json"))


def test_invalid_publication_isolates_one_parent_open_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec = _synthetic_protocol_spec(tmp_path)
    result_parent = tmp_path / "result-parent"
    result_parent.mkdir()
    out = result_parent / "result.json"
    artifacts = spec.official_artifacts_dir
    artifacts.mkdir()
    terminal = artifacts / "terminal.json"
    lifecycle = artifacts / "lifecycle.json"
    original_open_directory = protocol.open_directory

    def fail_result_parent(path):
        if Path(path).resolve() == result_parent.resolve():
            raise OSError("injected aggregate-parent failure")
        return original_open_directory(path)

    monkeypatch.setattr(protocol, "open_directory", fail_result_parent)
    publication = protocol._publish_invalid_protocol(
        spec,
        out=out,
        terminal_path=terminal,
        lifecycle_path=lifecycle,
        payload=_synthetic_invalid_payload(),
        root_consumption_status=spec.official_domain.permitted_root_consumption_statuses[-1],
        result_ownership=None,
        terminal_ownership=None,
        lifecycle_ownership=None,
        result_mutation_allowed=True,
        terminal_mutation_allowed=True,
        lifecycle_mutation_allowed=False,
    )
    assert "aggregate-parent failure" in publication["publication_failures"]["aggregate"]
    assert terminal.is_file()
    assert list(artifacts.glob("invalid-publication-fallback-*.json"))


def _committed_synthetic_claim(
    spec: protocol.ProtocolSpec,
    *,
    root_status: str | None = None,
    roots: tuple[int, ...] | None = None,
    source_drift: bool = False,
):
    domain = spec.official_domain
    status_value = (
        domain.permitted_root_consumption_statuses[-1] if root_status is None else root_status
    )
    root_values = spec.scene_roots + spec.initial_depth_roots if roots is None else roots
    source_start = {"hashes": {"synthetic.py": "1" * 64}, "errors": {}}
    source_end = (
        {"hashes": {"synthetic.py": "2" * 64}, "errors": {}} if source_drift else source_start
    )
    artifacts = spec.official_artifacts_dir
    artifacts.mkdir()
    terminal_path = artifacts / "terminal.json"
    lifecycle_path = artifacts / "lifecycle.json"
    result = domain.make_receipt(
        "aggregate",
        {
            "status": "PASS",
            "source_observation_start": source_start,
            "source_observation_end": source_end,
        },
        root_consumption_status=status_value,
        roots=root_values,
        official_phase="final_preparation",
        commit_state="UNCOMMITTED",
    )
    result_fd = tx.open_directory(spec.official_out.parent)
    artifacts_fd = tx.open_directory(artifacts)
    try:
        result_prepared = tx.prepare_json(
            result_fd,
            spec.official_out.name,
            domain,
            "aggregate",
            result,
        )
        result_report = tx.publish_exclusive(result_fd, result_prepared)
        terminal = domain.make_receipt(
            "terminal",
            {
                "status": "PASS",
                "phase": "complete",
                "aggregate_sha256": result_prepared.payload_sha256,
                "source_observation_start": source_start,
                "source_observation_end": source_end,
            },
            root_consumption_status=status_value,
            roots=root_values,
            official_phase="complete",
            commit_state="UNCOMMITTED",
        )
        terminal_prepared = tx.prepare_json(
            artifacts_fd,
            terminal_path.name,
            domain,
            "terminal",
            terminal,
        )
        terminal_report = tx.publish_exclusive(artifacts_fd, terminal_prepared)
        pending_lifecycle = domain.make_receipt(
            "lifecycle",
            {"status": "PENDING", "phase": "synthetic_precommit"},
            root_consumption_status=status_value,
            roots=root_values,
            official_phase="synthetic_precommit",
            commit_state="UNCOMMITTED",
        )
        pending_prepared = tx.prepare_json(
            artifacts_fd,
            lifecycle_path.name,
            domain,
            "lifecycle",
            pending_lifecycle,
        )
        pending_report = tx.publish_exclusive(artifacts_fd, pending_prepared)
        assert pending_report.public_entry is not None
        assert pending_report.public_entry.ownership is not None
        lifecycle = domain.make_receipt(
            "lifecycle",
            {
                "status": "PASS",
                "phase": "complete",
                "aggregate_sha256": result_prepared.payload_sha256,
                "terminal_sha256": terminal_prepared.payload_sha256,
                "source_observation_start": source_start,
                "source_observation_end": source_end,
            },
            root_consumption_status=status_value,
            roots=root_values,
            official_phase="complete",
            commit_state="COMMITTED",
        )
        lifecycle_prepared = tx.prepare_json(
            artifacts_fd,
            lifecycle_path.name,
            domain,
            "lifecycle",
            lifecycle,
        )
        lifecycle_report = tx.exchange_owned(
            artifacts_fd,
            lifecycle_prepared,
            pending_report.public_entry.ownership,
        )
    finally:
        tx.os.close(result_fd)
        tx.os.close(artifacts_fd)
    return {
        "terminal_path": terminal_path,
        "lifecycle_path": lifecycle_path,
        "result": result,
        "terminal": terminal,
        "lifecycle": lifecycle,
        "prepared": (result_prepared, terminal_prepared, lifecycle_prepared),
        "reports": (result_report, terminal_report, lifecycle_report),
    }


def test_claim_validator_accepts_only_complete_cross_hashed_lifecycle_last_commit(
    tmp_path: Path,
) -> None:
    spec = _synthetic_protocol_spec(tmp_path)
    fixture = _committed_synthetic_claim(spec)
    validation = protocol._validate_committed_protocol(
        spec,
        out=spec.official_out,
        terminal_path=fixture["terminal_path"],
        lifecycle_path=fixture["lifecycle_path"],
        result=fixture["result"],
        terminal=fixture["terminal"],
        lifecycle=fixture["lifecycle"],
        result_prepared=fixture["prepared"][0],
        terminal_prepared=fixture["prepared"][1],
        lifecycle_prepared=fixture["prepared"][2],
        reports=fixture["reports"],
    )
    assert validation["status"] == "PASS"
    assert validation["commit_state"] == "COMMITTED"


@pytest.mark.parametrize("mutation", ["root_status", "root_subset", "source_drift"])
def test_claim_validator_rejects_incomplete_root_or_source_claims(
    tmp_path: Path,
    mutation: str,
) -> None:
    spec = _synthetic_protocol_spec(tmp_path)
    kwargs = {}
    if mutation == "root_status":
        kwargs["root_status"] = spec.official_domain.permitted_root_consumption_statuses[1]
    elif mutation == "root_subset":
        kwargs["roots"] = (spec.scene_roots[0],)
    else:
        kwargs["source_drift"] = True
    fixture = _committed_synthetic_claim(spec, **kwargs)
    with pytest.raises(RuntimeError, match="consumed|six|source"):
        protocol._validate_committed_protocol(
            spec,
            out=spec.official_out,
            terminal_path=fixture["terminal_path"],
            lifecycle_path=fixture["lifecycle_path"],
            result=fixture["result"],
            terminal=fixture["terminal"],
            lifecycle=fixture["lifecycle"],
            result_prepared=fixture["prepared"][0],
            terminal_prepared=fixture["prepared"][1],
            lifecycle_prepared=fixture["prepared"][2],
            reports=fixture["reports"],
        )


def test_run_protocol_prepares_before_one_closing_observation_and_commits_lifecycle_last(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec = _synthetic_protocol_spec(tmp_path)
    domain = spec.official_domain
    roots = spec.scene_roots + spec.initial_depth_roots
    idle, _transition, consumed = domain.permitted_root_consumption_statuses
    events: list[tuple[str, str, str | None, str | None]] = []
    prepared_metadata: dict[int, tuple[str, str | None, str | None]] = {}
    required_observation_paths = {
        spec.preregistration.relative_to(protocol.ROOT).as_posix(),
        spec.preregistration_review.relative_to(protocol.ROOT).as_posix(),
        spec.verification_receipt.relative_to(protocol.ROOT).as_posix(),
        spec.implementation_review.relative_to(protocol.ROOT).as_posix(),
    }
    observation = {
        "hashes": {path: "3" * 64 for path in required_observation_paths},
        "errors": {},
    }

    def preflight(candidate_spec, out, artifacts):
        assert candidate_spec is spec
        assert Path(out).resolve() == spec.official_out.resolve()
        assert Path(artifacts).resolve() == spec.official_artifacts_dir.resolve()
        return Path(out).resolve(), Path(artifacts).resolve(), observation

    def provenance(candidate_spec, candidate_domain, *, root_consumption_status):
        assert candidate_spec is spec
        return candidate_domain.make_receipt(
            "provenance",
            {"source_observation": observation, "status": "SYNTHETIC_START"},
            root_consumption_status=root_consumption_status,
            roots=roots,
        )

    def closing_observation(candidate_spec):
        assert candidate_spec is spec
        events.append(("source_observation", "closure", None, None))
        return observation

    def build_scene(scene_root, depth_root):
        events.append(("build_scene", str(scene_root), None, None))
        return {
            "scene_root": scene_root,
            "initial_depth_root": depth_root,
            "initial_depth_sha256": f"{scene_root:064x}"[-64:],
        }

    descriptor_counter = 0

    def descriptor(kind, payload, artifacts_dir):
        nonlocal descriptor_counter
        descriptor_counter += 1
        receipt = domain.make_receipt(
            kind,
            payload,
            root_consumption_status=consumed,
            roots=roots,
        )
        return {
            "receipt": receipt,
            "path": str(artifacts_dir / f"synthetic-{descriptor_counter}.json"),
            "recovery_path": str(artifacts_dir / f"synthetic-{descriptor_counter}.recovery"),
            "sha256": f"{descriptor_counter:064x}",
        }

    def raw_writer(scene, artifacts_dir, **_kwargs):
        return descriptor(
            "raw_inputs",
            {
                "scene_root": scene["scene_root"],
                "initial_depth_root": scene["initial_depth_root"],
            },
            artifacts_dir,
        )

    means = torch.zeros(protocol.N_HYPOTHESES, 3, dtype=torch.float64)
    covariances = torch.eye(3, dtype=torch.float64).repeat(protocol.N_HYPOTHESES, 1, 1)

    def common_writer(scene, _raw, artifacts_dir, *_args):
        return (
            object(),
            means,
            covariances,
            descriptor("common_constructor", {"pass": True}, artifacts_dir),
        )

    def rank_writer(_scene, rank, artifacts_dir, *_args):
        return descriptor("rank_sentinel", {"rank": rank, "pass": True}, artifacts_dir)

    def initialization_writer(arm, _scene, _means, _covariances, artifacts_dir, *_args):
        return object(), descriptor(
            "arm_initialization",
            {"arm": arm, "pass": True},
            artifacts_dir,
        )

    def train_writer(_model, arm, scene, *_args, **_kwargs):
        return {
            "scene_root": scene["scene_root"],
            "arm": arm,
            "checkpoints": [{"pass": True}],
        }

    original_prepare = protocol.prepare_json
    original_publish = protocol.publish_exclusive
    original_exchange = protocol.exchange_owned

    def prepare_spy(dir_fd, target_name, receipt_domain, kind, value, **kwargs):
        prepared = original_prepare(
            dir_fd,
            target_name,
            receipt_domain,
            kind,
            value,
            **kwargs,
        )
        metadata = (kind, value.get("official_phase"), value.get("commit_state"))
        prepared_metadata[id(prepared)] = metadata
        events.append(("prepare", target_name, metadata[1], metadata[2]))
        return prepared

    def publish_spy(dir_fd, prepared, **kwargs):
        kind, phase, commit_state = prepared_metadata[id(prepared)]
        events.append(("publish", kind, phase, commit_state))
        return original_publish(dir_fd, prepared, **kwargs)

    def exchange_spy(dir_fd, prepared, expected, **kwargs):
        kind, phase, commit_state = prepared_metadata[id(prepared)]
        events.append(("exchange", kind, phase, commit_state))
        return original_exchange(dir_fd, prepared, expected, **kwargs)

    monkeypatch.setattr(protocol, "_preflight_protocol", preflight)
    monkeypatch.setattr(protocol, "_provenance", provenance)
    monkeypatch.setattr(protocol, "_source_observation", closing_observation)
    monkeypatch.setattr(protocol, "_construction_sentinel", lambda: {"pass": True})
    monkeypatch.setattr(
        protocol,
        "_gradient_and_duplicate_sentinels",
        lambda: ({"pass": True}, {"pass": True}),
    )
    monkeypatch.setattr(protocol, "_build_scene", build_scene)
    monkeypatch.setattr(protocol, "_write_scene_input_receipt", raw_writer)
    monkeypatch.setattr(protocol, "_write_common_constructor_receipt", common_writer)
    monkeypatch.setattr(protocol, "_rank_sentinel", lambda _scene: {"pass": True})
    monkeypatch.setattr(protocol, "_write_rank_receipt", rank_writer)
    monkeypatch.setattr(protocol, "_build_and_write_initialization_receipt", initialization_writer)
    monkeypatch.setattr(protocol, "_train_arm", train_writer)
    monkeypatch.setattr(protocol, "_aggregate", lambda records, _roots: {"count": len(records)})
    monkeypatch.setattr(
        protocol,
        "_scientific_gates",
        lambda _records, _sentinels, _roots: {"overall_status": "PASS"},
    )
    monkeypatch.setattr(protocol, "UPDATES", 0)
    monkeypatch.setattr(protocol, "CHECKPOINT_INTERVAL", 1)
    monkeypatch.setattr(protocol, "prepare_json", prepare_spy)
    monkeypatch.setattr(protocol, "publish_exclusive", publish_spy)
    monkeypatch.setattr(protocol, "exchange_owned", exchange_spy)

    result = protocol.run_protocol(spec, spec.official_out, spec.official_artifacts_dir)
    assert result["status"] == "PASS"
    assert result["root_consumption_status"] == consumed
    assert spec.official_out.is_file()
    assert (spec.official_artifacts_dir / "commit_validation.json").is_file()

    def event_index(
        action: str,
        kind: str,
        phase: str | None = None,
        commit_state: str | None = None,
    ) -> int:
        for index, event in enumerate(events):
            if (
                event[0] == action
                and event[1] == kind
                and (phase is None or event[2] == phase)
                and (commit_state is None or event[3] == commit_state)
            ):
                return index
        raise AssertionError((action, kind, phase, commit_state, events))

    source_indices = [
        index for index, event in enumerate(events) if event[0] == "source_observation"
    ]
    assert len(source_indices) == 1
    source_index = source_indices[0]
    final_prepare_indices = (
        event_index("prepare", "result.json", "final_preparation", "UNCOMMITTED"),
        event_index("prepare", "terminal.json", "complete", "UNCOMMITTED"),
        event_index("prepare", "lifecycle.json", "complete", "COMMITTED"),
    )
    assert max(final_prepare_indices) < source_index
    result_publish = event_index("publish", "aggregate", "final_preparation", "UNCOMMITTED")
    terminal_publish = event_index("publish", "terminal", "complete", "UNCOMMITTED")
    lifecycle_commit = event_index("exchange", "lifecycle", "complete", "COMMITTED")
    assert source_index < result_publish < terminal_publish < lifecycle_commit
    generator_transition = event_index(
        "exchange",
        "lifecycle",
        "official_generators_consumed",
        "UNCOMMITTED",
    )
    first_scene = event_index("build_scene", str(spec.scene_roots[0]))
    assert generator_transition < first_scene
    assert idle != consumed
