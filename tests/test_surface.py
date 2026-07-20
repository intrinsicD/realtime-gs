"""Calibrated oriented-point inputs, target validation, and surface losses."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.lift.surface import (
    CanonicalOrientedPointMap,
    OrientedPointBackend,
    OrientedPointPrediction,
    OrientedPointProvenance,
    OrientedPointTargets,
    canonicalize_oriented_point_prediction,
    estimate_registered_depth_normals,
    local_plane_loss,
    shortest_axis_normal_loss,
    validate_oriented_point_targets,
)


def _camera(
    *,
    height: int = 3,
    width: int = 4,
    fx: float = 2.0,
    fy: float = 4.0,
    cx: float = 0.0,
    cy: float = 0.0,
    R: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
) -> Camera:
    return Camera(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        width=width,
        height=height,
        R=torch.eye(3) if R is None else R,
        t=torch.zeros(3) if t is None else t,
    )


def _provenance(view_id: str = "view-0", config_id: str = "config-sha256"):
    return OrientedPointProvenance(
        view_id=view_id,
        backend_name="fixture-rgbd",
        backend_version="1",
        config_id=config_id,
    )


def _prediction(
    camera: Camera,
    *,
    geometry: torch.Tensor | None = None,
    normals: torch.Tensor | None = None,
    geometry_kind: str = "camera_z_depth",
    normal_frame: str = "camera",
    valid: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    provenance: OrientedPointProvenance | None = None,
) -> OrientedPointPrediction:
    height, width = camera.height, camera.width
    if geometry is None:
        geometry = torch.full((height, width), 2.0)
    if normals is None:
        normals = torch.tensor([0.0, 0.0, -2.0]).expand(height, width, -1).clone()
    return OrientedPointPrediction(
        geometry=geometry,
        normals=normals,
        geometry_kind=geometry_kind,  # type: ignore[arg-type]
        normal_frame=normal_frame,  # type: ignore[arg-type]
        provenance=_provenance() if provenance is None else provenance,
        valid=valid,
        confidence=confidence,
    )


def _canonical(
    prediction: OrientedPointPrediction,
    camera: Camera,
    *,
    dtype: torch.dtype = torch.float32,
) -> CanonicalOrientedPointMap:
    return canonicalize_oriented_point_prediction(
        prediction,
        camera,
        (camera.height, camera.width),
        expected_view_id="view-0",
        expected_config_id="config-sha256",
        device="cpu",
        dtype=dtype,
    )


def test_oriented_backend_protocol_uses_stable_view_id_and_image_shape():
    camera = _camera()

    class FixtureBackend:
        def predict(
            self,
            image_shape: tuple[int, int],
            camera: Camera,
            *,
            view_id: str,
        ) -> OrientedPointPrediction:
            assert image_shape == (3, 4)
            assert view_id == "view-0"
            return _prediction(camera)

    backend: OrientedPointBackend = FixtureBackend()
    prediction = backend.predict((3, 4), camera, view_id="view-0")
    assert prediction.provenance.view_id == "view-0"


def test_camera_z_canonicalization_uses_half_pixel_depth_not_ray_range_and_clones():
    camera = _camera(height=2, width=2)
    depth = torch.full((2, 2), 2.0, dtype=torch.float64)
    normals = torch.tensor([0.0, 0.0, -4.0], dtype=torch.float64).expand(2, 2, -1).clone()
    confidence = torch.full((2, 2), 0.75, dtype=torch.float64)
    prediction = _prediction(
        camera,
        geometry=depth,
        normals=normals,
        confidence=confidence,
    )
    result = _canonical(prediction, camera)

    # Array (0,0) is repository pixel (0.5,0.5), and z is optical-axis depth.
    assert torch.allclose(result.points_world[0, 0], torch.tensor([0.5, 0.25, 2.0]))
    assert result.points_world[0, 0].norm() > 2.0
    assert torch.equal(result.normals_world[0, 0], torch.tensor([0.0, 0.0, -1.0]))
    assert result.points_world.dtype == torch.float32
    assert torch.equal(result.confidence, torch.full((2, 2), 0.75))
    assert result.valid.all()
    assert not result.points_world.requires_grad

    depth.zero_()
    normals.zero_()
    confidence.zero_()
    assert result.points_world[0, 0, 2] == 2
    assert result.normals_world[0, 0, 2] == -1
    assert result.confidence[0, 0] == 0.75


def test_camera_points_and_camera_normals_use_nonidentity_world_transform():
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    camera = _camera(
        height=1,
        width=1,
        R=rotation,
        t=torch.tensor([0.25, -0.5, 1.0]),
    )
    point_camera = torch.tensor([[[1.0, 2.0, 3.0]]])
    normal_camera = torch.tensor([[[2.0, 0.0, 0.0]]])
    result = _canonical(
        _prediction(
            camera,
            geometry=point_camera,
            normals=normal_camera,
            geometry_kind="camera_points",
        ),
        camera,
    )

    expected_point = camera.cam_to_world(point_camera.reshape(-1, 3))[0]
    assert torch.allclose(result.points_world[0, 0], expected_point)
    assert torch.equal(result.normals_world[0, 0], torch.tensor([0.0, -1.0, 0.0]))
    projected, projected_depth = camera.project(result.points_world.reshape(-1, 3))
    assert projected_depth[0] == 3
    assert torch.isfinite(projected).all()


def test_world_geometry_and_normals_pass_through_without_frame_relabeling():
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    camera = _camera(height=1, width=1, R=rotation, t=torch.tensor([0.0, 0.0, 1.0]))
    point_camera = torch.tensor([[[-1.0, 0.5, 2.0]]])
    point_world = camera.cam_to_world(point_camera.reshape(-1, 3)).reshape(1, 1, 3)
    normal_world = torch.tensor([[[0.0, 3.0, 0.0]]])
    result = _canonical(
        _prediction(
            camera,
            geometry=point_world,
            normals=normal_world,
            geometry_kind="world_points",
            normal_frame="world",
        ),
        camera,
    )
    assert torch.allclose(result.points_world, point_world)
    assert torch.equal(result.normals_world, torch.tensor([[[0.0, 1.0, 0.0]]]))


def test_canonicalization_infers_invalid_sentinels_and_materializes_safe_zeros():
    camera = _camera(height=2, width=2)
    depth = torch.tensor([[2.0, float("nan")], [0.0, 3.0]])
    normals = torch.tensor([0.0, 0.0, -1.0]).expand(2, 2, -1).clone()
    normals[1, 1] = 0
    confidence = torch.tensor([[0.8, float("nan")], [2.0, float("inf")]])
    result = _canonical(
        _prediction(
            camera,
            geometry=depth,
            normals=normals,
            confidence=confidence,
        ),
        camera,
    )
    assert torch.equal(result.valid, torch.tensor([[True, False], [False, False]]))
    assert torch.isfinite(result.points_world).all()
    assert torch.isfinite(result.normals_world).all()
    assert torch.isfinite(result.confidence).all()
    assert result.points_world[~result.valid].count_nonzero() == 0
    assert result.normals_world[~result.valid].count_nonzero() == 0
    assert result.confidence[~result.valid].count_nonzero() == 0


def test_explicit_invalid_pixels_may_hold_nonfinite_sentinels():
    camera = _camera(height=1, width=2)
    prediction = _prediction(
        camera,
        geometry=torch.tensor([[2.0, float("nan")]]),
        normals=torch.tensor([[[0.0, 0.0, -1.0], [float("nan"), 0.0, 0.0]]]),
        valid=torch.tensor([[True, False]]),
        confidence=torch.tensor([[0.5, float("nan")]]),
    )
    result = _canonical(prediction, camera)
    assert torch.equal(result.valid, torch.tensor([[True, False]]))
    assert result.points_world[0, 1].count_nonzero() == 0
    assert result.normals_world[0, 1].count_nonzero() == 0
    assert result.confidence[0, 1] == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("geometry_kind", "range", "geometry_kind"),
        ("normal_frame", "model", "normal_frame"),
        ("confidence", torch.full((3, 4), 1.1), "confidence"),
        ("normals", torch.zeros(3, 4, 3), "normals"),
        ("geometry", torch.zeros(3, 4), "geometry"),
    ],
)
def test_canonicalization_rejects_invalid_enums_and_valid_values(field, value, message):
    camera = _camera()
    kwargs = {field: value, "valid": torch.ones(3, 4, dtype=torch.bool)}
    with pytest.raises(ValueError, match=message):
        _canonical(_prediction(camera, **kwargs), camera)


def test_canonicalization_rejects_attached_shape_device_and_dtype_contracts():
    camera = _camera()
    attached = _prediction(camera, geometry=torch.ones(3, 4, requires_grad=True))
    with pytest.raises(ValueError, match="geometry must be detached"):
        _canonical(attached, camera)
    with pytest.raises(ValueError, match="geometry must have shape"):
        _canonical(_prediction(camera, geometry=torch.ones(2, 4)), camera)
    with pytest.raises(ValueError, match="floating dtype"):
        _canonical(_prediction(camera, geometry=torch.ones(3, 4, dtype=torch.long)), camera)
    with pytest.raises(ValueError, match="camera resolution"):
        canonicalize_oriented_point_prediction(
            _prediction(camera),
            camera,
            (2, 4),
            expected_view_id="view-0",
            expected_config_id="config-sha256",
            device="cpu",
            dtype=torch.float32,
        )


@pytest.mark.parametrize(
    ("expected_view", "expected_config", "message"),
    [
        ("other", "config-sha256", "view_id mismatch"),
        ("view-0", "other", "config_id mismatch"),
    ],
)
def test_canonicalization_rejects_provenance_mismatch(expected_view, expected_config, message):
    camera = _camera()
    with pytest.raises(ValueError, match=message):
        canonicalize_oriented_point_prediction(
            _prediction(camera),
            camera,
            (3, 4),
            expected_view_id=expected_view,
            expected_config_id=expected_config,
            device="cpu",
            dtype=torch.float32,
        )


def test_provenance_and_prediction_records_are_frozen():
    camera = _camera()
    provenance = _provenance()
    prediction = _prediction(camera, provenance=provenance)
    with pytest.raises(FrozenInstanceError):
        provenance.view_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        prediction.geometry_kind = "world_points"  # type: ignore[misc]


@pytest.mark.parametrize("overflow_field", ["geometry", "normals"])
def test_canonicalization_rejects_postcast_overflow(overflow_field):
    camera = _camera(height=1, width=1)
    geometry = torch.ones(1, 1, dtype=torch.float64)
    normals = torch.ones(1, 1, 3, dtype=torch.float64)
    if overflow_field == "geometry":
        geometry[0, 0] = 1e40
    else:
        normals[0, 0, 0] = 1e40
    with pytest.raises(ValueError, match="remain finite"):
        _canonical(
            _prediction(
                camera,
                geometry=geometry,
                normals=normals,
                valid=torch.ones(1, 1, dtype=torch.bool),
            ),
            camera,
        )


@pytest.mark.parametrize("underflow_field", ["geometry", "normals"])
def test_canonicalization_rejects_valid_values_that_underflow_to_zero(underflow_field):
    camera = _camera(height=1, width=1)
    geometry = torch.ones(1, 1, dtype=torch.float64)
    normals = torch.ones(1, 1, 3, dtype=torch.float64)
    if underflow_field == "geometry":
        geometry[0, 0] = 1e-50
    else:
        normals[0, 0] = torch.tensor([1e-50, 0.0, 0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="remain finite|remain finite and positive"):
        _canonical(
            _prediction(
                camera,
                geometry=geometry,
                normals=normals,
                valid=torch.ones(1, 1, dtype=torch.bool),
            ),
            camera,
        )


def test_registered_depth_normals_flat_plane_match_five_point_orientation():
    camera = _camera(height=9, width=9, fx=525.0, fy=525.0, cx=4.5, cy=4.5)
    depth = torch.full((9, 9), 2.0)
    normals, valid = estimate_registered_depth_normals(depth, camera)

    assert int(valid.sum()) == 25
    assert not valid[:2].any() and not valid[-2:].any()
    assert not valid[:, :2].any() and not valid[:, -2:].any()
    assert torch.allclose(
        normals[valid], torch.tensor([0.0, 0.0, -1.0]).expand(int(valid.sum()), -1)
    )
    pixels = torch.tensor([[4.5, 4.5], [2.5, 2.5]])
    _, rays = camera.pixel_rays(pixels)
    rays = torch.nn.functional.normalize(rays, dim=-1)
    assert torch.dot(normals[4, 4], rays[0]) < 0

    away, away_valid = estimate_registered_depth_normals(depth, camera, orient_toward_camera=False)
    assert torch.equal(away_valid, valid)
    assert torch.allclose(
        away[away_valid], torch.tensor([0.0, 0.0, 1.0]).expand(int(valid.sum()), -1)
    )


def test_registered_depth_normals_match_a_slanted_metric_plane():
    camera = _camera(height=9, width=9, fx=20.0, fy=20.0, cx=4.5, cy=4.5)
    rows, columns = torch.meshgrid(
        torch.arange(9, dtype=torch.float64) + 0.5,
        torch.arange(9, dtype=torch.float64) + 0.5,
        indexing="ij",
    )
    plane_normal = torch.tensor([0.1, -0.05, -1.0], dtype=torch.float64)
    plane_normal = plane_normal / plane_normal.norm()
    rays = torch.stack(
        [
            (columns - camera.cx) / camera.fx,
            (rows - camera.cy) / camera.fy,
            torch.ones_like(rows),
        ],
        dim=-1,
    )
    depth = -2.0 / (rays * plane_normal).sum(dim=-1)
    normals, valid = estimate_registered_depth_normals(depth, camera)
    cosine = (normals[valid] * plane_normal).sum(dim=-1).abs()
    assert valid.any()
    assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-10, rtol=0)


def test_registered_depth_normals_reject_invalid_or_discontinuous_neighborhoods_safely():
    camera = _camera(height=9, width=9, fx=100.0, fy=100.0, cx=4.5, cy=4.5)
    depth = torch.full((9, 9), 2.0)
    depth[4, 6] = float("nan")
    depth[6, 4] = 2.2
    normals, valid = estimate_registered_depth_normals(depth, camera)
    assert not valid[4, 4]
    assert torch.isfinite(normals).all()
    assert normals[~valid].count_nonzero() == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pixel_offset": 0}, "pixel_offset"),
        ({"min_depth": 5.0, "max_depth": 0.3}, "depth range"),
        ({"max_abs_depth_delta": -1.0}, "max_abs_depth_delta"),
        ({"max_relative_depth_delta": float("nan")}, "max_relative_depth_delta"),
        ({"min_cross_norm": 0.0}, "min_cross_norm"),
        ({"min_abs_incidence": 1.1}, "min_abs_incidence"),
    ],
)
def test_registered_depth_normals_reject_invalid_configuration(kwargs, message):
    camera = _camera(height=9, width=9)
    with pytest.raises(ValueError, match=message):
        estimate_registered_depth_normals(torch.full((9, 9), 2.0), camera, **kwargs)


def _validated(
    *,
    indices: torch.Tensor | None = None,
    points: torch.Tensor | None = None,
    plane_normals: torch.Tensor | None = None,
    alignment_normals: torch.Tensor | None = None,
    n_retained: int = 3,
) -> OrientedPointTargets:
    indices = torch.tensor([0, 2], dtype=torch.long) if indices is None else indices
    count = indices.numel()
    points = torch.zeros(count, 3) if points is None else points
    plane_normals = (
        torch.tensor([[0.0, 0.0, 2.0]]).expand(count, -1).clone()
        if plane_normals is None
        else plane_normals
    )
    return validate_oriented_point_targets(
        OrientedPointTargets(indices, points, plane_normals, alignment_normals),
        n_retained,
        device="cpu",
        dtype=torch.float32,
    )


def test_local_plane_loss_matches_formula_extent_and_tangent_gradient():
    means = torch.tensor(
        [[4.0, -3.0, 2.0], [7.0, 8.0, 9.0], [-2.0, 5.0, -1.0]],
        requires_grad=True,
    )
    targets = _validated(
        points=torch.tensor([[1.0, 2.0, 0.0], [-3.0, 4.0, 2.0]]),
        plane_normals=torch.tensor([[0.0, 0.0, 4.0], [0.0, 3.0, 0.0]]),
    )

    loss = local_plane_loss(means, targets, scene_extent=2.0)
    # Absolute distances are 2 and 1, averaged and divided by extent 2.
    assert torch.allclose(loss, torch.tensor(0.75))
    loss.backward()
    assert means.grad is not None
    assert means.grad[0, 0] == 0
    assert means.grad[0, 1] == 0
    assert means.grad[1].count_nonzero() == 0
    assert means.grad[2, 0] == 0
    assert means.grad[2, 2] == 0

    translated_scaled_means = means.detach() * 7.0 + torch.tensor([3.0, -2.0, 1.0])
    translated_scaled_points = targets.points * 7.0 + torch.tensor([3.0, -2.0, 1.0])
    scaled_targets = _validated(
        points=translated_scaled_points,
        plane_normals=targets.plane_normals,
    )
    scaled_loss = local_plane_loss(translated_scaled_means, scaled_targets, scene_extent=14.0)
    assert torch.equal(loss.detach(), scaled_loss)


def test_shortest_axis_loss_uses_requested_rotation_columns_per_target():
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3)
    targets = _validated(
        plane_normals=torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        alignment_normals=torch.tensor([[0.0, 0.0, 5.0], [-2.0, 0.0, 0.0]]),
    )
    axis_indices = torch.tensor([2, 0], dtype=torch.long)
    assert shortest_axis_normal_loss(quats, axis_indices, targets) == 0

    reordered = torch.tensor([0, 2], dtype=torch.long)
    assert shortest_axis_normal_loss(quats, reordered, targets) == 1


def test_shortest_axis_loss_uses_columns_for_nonidentity_rotation():
    root_half = 2.0**-0.5
    quat = torch.tensor([[root_half, 0.5, 0.5, 0.0]])
    first_column = torch.tensor([[0.5, 0.5, -root_half]])
    target = _validated(
        indices=torch.tensor([0]),
        plane_normals=torch.tensor([[0.0, 0.0, 1.0]]),
        alignment_normals=first_column,
        n_retained=1,
    )

    loss = shortest_axis_normal_loss(quat, torch.tensor([0]), target)
    assert torch.allclose(loss, torch.zeros(()), atol=1e-6)


def test_shortest_axis_loss_is_normal_and_quaternion_sign_invariant():
    quat = torch.tensor([[0.9238795, 0.0, 0.3826834, 0.0]])
    normal = torch.tensor([[0.25, 0.0, 1.0]])
    target = _validated(
        indices=torch.tensor([0]),
        plane_normals=torch.tensor([[0.0, 1.0, 0.0]]),
        alignment_normals=normal,
        n_retained=1,
    )
    negated_normal_target = _validated(
        indices=torch.tensor([0]),
        plane_normals=torch.tensor([[0.0, -1.0, 0.0]]),
        alignment_normals=-normal,
        n_retained=1,
    )
    axes = torch.tensor([2], dtype=torch.long)

    reference = shortest_axis_normal_loss(quat, axes, target)
    assert torch.equal(reference, shortest_axis_normal_loss(-quat, axes, target))
    assert torch.equal(reference, shortest_axis_normal_loss(quat, axes, negated_normal_target))


def test_shortest_axis_loss_has_finite_nonzero_quaternion_gradient():
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)
    target = _validated(
        indices=torch.tensor([0]),
        plane_normals=torch.tensor([[0.0, 0.0, 1.0]]),
        alignment_normals=torch.tensor([[1.0, 0.0, 1.0]]),
        n_retained=1,
    )
    loss = shortest_axis_normal_loss(quat, torch.tensor([2]), target)
    loss.backward()

    assert quat.grad is not None
    assert torch.isfinite(quat.grad).all()
    assert quat.grad.count_nonzero() > 0


def test_alignment_only_control_changes_normal_loss_but_not_plane_loss():
    means = torch.tensor([[0.0, 0.0, 1.0]])
    correct = _validated(
        indices=torch.tensor([0]),
        points=torch.zeros(1, 3),
        plane_normals=torch.tensor([[0.0, 0.0, 1.0]]),
        alignment_normals=torch.tensor([[1.0, 0.0, 0.0]]),
        n_retained=1,
    )
    shuffled = _validated(
        indices=torch.tensor([0]),
        points=torch.zeros(1, 3),
        plane_normals=torch.tensor([[0.0, 0.0, 1.0]]),
        alignment_normals=torch.tensor([[0.0, 1.0, 0.0]]),
        n_retained=1,
    )
    assert torch.equal(
        local_plane_loss(means, correct, 2.0), local_plane_loss(means, shuffled, 2.0)
    )
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    axis_indices = torch.tensor([0], dtype=torch.long)
    assert shortest_axis_normal_loss(quats, axis_indices, correct) == 0
    assert shortest_axis_normal_loss(quats, axis_indices, shuffled) == 1


def test_validation_normalizes_detaches_clones_and_converts_dtype():
    source = OrientedPointTargets(
        indices=torch.tensor([0, 2]),
        points=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64),
        plane_normals=torch.tensor([[0.0, 0.0, 4.0], [0.0, 3.0, 0.0]], dtype=torch.float64),
        alignment_normals=torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, -5.0]]),
    )
    result = validate_oriented_point_targets(source, 3, device="cpu", dtype=torch.float32)

    assert result.indices.data_ptr() != source.indices.data_ptr()
    assert result.points.data_ptr() != source.points.data_ptr()
    assert result.plane_normals.data_ptr() != source.plane_normals.data_ptr()
    assert result.alignment_normals.data_ptr() != source.alignment_normals.data_ptr()
    assert result.points.dtype == torch.float32
    assert torch.equal(result.plane_normals.norm(dim=-1), torch.ones(2))
    assert torch.equal(result.alignment_normals.norm(dim=-1), torch.ones(2))
    assert not result.points.requires_grad
    source.plane_normals.zero_()
    source.alignment_normals.zero_()
    assert torch.equal(result.plane_normals.norm(dim=-1), torch.ones(2))
    assert torch.equal(result.alignment_normals.norm(dim=-1), torch.ones(2))


@pytest.mark.parametrize(
    ("targets", "n_retained", "message"),
    [
        (
            OrientedPointTargets(torch.tensor([0.0]), torch.zeros(1, 3), torch.ones(1, 3)),
            1,
            "int64",
        ),
        (
            OrientedPointTargets(torch.tensor([1, 0]), torch.zeros(2, 3), torch.ones(2, 3)),
            2,
            "increasing and unique",
        ),
        (
            OrientedPointTargets(torch.tensor([0, 0]), torch.zeros(2, 3), torch.ones(2, 3)),
            2,
            "increasing and unique",
        ),
        (
            OrientedPointTargets(torch.tensor([2]), torch.zeros(1, 3), torch.ones(1, 3)),
            2,
            "out-of-range",
        ),
        (
            OrientedPointTargets(torch.tensor([0]), torch.zeros(1, 2), torch.ones(1, 3)),
            1,
            "points must have shape",
        ),
        (
            OrientedPointTargets(
                torch.tensor([0]), torch.tensor([[float("nan"), 0.0, 0.0]]), torch.ones(1, 3)
            ),
            1,
            "finite",
        ),
        (
            OrientedPointTargets(torch.tensor([0]), torch.zeros(1, 3), torch.zeros(1, 3)),
            1,
            "nonzero",
        ),
        (
            OrientedPointTargets(
                torch.tensor([0]),
                torch.zeros(1, 3),
                torch.ones(1, 3),
                torch.zeros(1, 3),
            ),
            1,
            "nonzero",
        ),
    ],
)
def test_validation_rejects_invalid_targets(targets, n_retained, message):
    with pytest.raises(ValueError, match=message):
        validate_oriented_point_targets(targets, n_retained, device="cpu", dtype=torch.float32)


def test_validation_rejects_attached_values_and_bad_retained_count():
    attached = OrientedPointTargets(
        torch.tensor([0]), torch.zeros(1, 3, requires_grad=True), torch.ones(1, 3)
    )
    with pytest.raises(ValueError, match="points must be detached"):
        validate_oriented_point_targets(attached, 1, device="cpu", dtype=torch.float32)
    with pytest.raises(ValueError, match="positive integer"):
        validate_oriented_point_targets(
            OrientedPointTargets(torch.tensor([0]), torch.zeros(1, 3), torch.ones(1, 3)),
            0,
            device="cpu",
            dtype=torch.float32,
        )


@pytest.mark.parametrize("overflow_field", ["points", "plane_normals"])
def test_validation_rejects_values_that_overflow_requested_dtype(overflow_field):
    values = {
        "points": torch.zeros(1, 3, dtype=torch.float64),
        "plane_normals": torch.ones(1, 3, dtype=torch.float64),
    }
    values[overflow_field][0, 0] = 1e300
    targets = OrientedPointTargets(torch.tensor([0]), **values)

    with pytest.raises(ValueError, match="remain finite"):
        validate_oriented_point_targets(targets, 1, device="cpu", dtype=torch.float32)


@pytest.mark.parametrize(
    "axis_indices",
    [torch.tensor([0.0, 1.0]), torch.tensor([0]), torch.tensor([0, 3])],
)
def test_shortest_axis_loss_rejects_invalid_axis_indices(axis_indices):
    targets = _validated()
    with pytest.raises((ValueError, TypeError), match="axis_indices"):
        shortest_axis_normal_loss(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3), axis_indices, targets)
