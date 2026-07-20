"""Tests for the hard RGB-free boundary after compact per-view fitting."""

import hashlib
import json
from dataclasses import asdict, fields, replace

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.reconstruction_inputs import BundleLoadLimits, ReconstructionInputs
from rtgs.data.scene import SceneData


def _camera(width: int = 8, height: int = 6, offset: float = 0.0) -> Camera:
    return Camera.look_at(
        eye=torch.tensor([offset, 0.0, -3.0]),
        target=torch.zeros(3),
        width=width,
        height=height,
    )


def _observation(view_id: str, width: int = 8, height: int = 6) -> GaussianObservationField:
    return GaussianObservationField(
        width=width,
        height=height,
        means=torch.tensor([[3.5, 2.5], [5.5, 3.5]]),
        log_scales=torch.log(torch.tensor([[1.2, 0.8], [0.7, 1.1]])),
        rotations=torch.tensor([0.2, -0.4]),
        colors=torch.tensor([[1.2, -0.1, 0.4], [0.2, 0.8, 1.1]]),
        amplitudes=torch.tensor([0.7, 0.5]),
        view_id=view_id,
        n_init=1,
    )


def _inputs() -> ReconstructionInputs:
    return ReconstructionInputs(
        observations=[_observation("left"), _observation("right")],
        cameras=[_camera(offset=-0.2), _camera(offset=0.2)],
        view_names=["left", "right"],
        points=torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.1, 0.3]]),
        point_visibility=[torch.tensor([0, 1]), torch.tensor([1])],
        bounds_hint=(torch.tensor([0.0, 0.0, 0.1]), 2.5),
        name="compact-scene",
    )


def _rewrite_manifest(path, mutate) -> None:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    mutate(manifest)
    manifest.pop("semantic_digest", None)

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    manifest["semantic_digest"] = hashlib.sha256(canonical(manifest)).hexdigest()
    manifest_path.write_bytes(canonical(manifest))


def test_type_and_manifest_have_no_rgb_or_image_members(tmp_path, monkeypatch):
    from PIL import Image as PILImage

    forbidden = {"images", "image", "rgb", "masks", "image_paths", "rgb_paths"}
    assert forbidden.isdisjoint({item.name for item in fields(ReconstructionInputs)})
    inputs = _inputs()
    path = tmp_path / "bundle"
    inputs.save(path)
    manifest_text = (path / "manifest.json").read_text()
    assert forbidden.isdisjoint(json.loads(manifest_text))
    assert "rgb" not in manifest_text.lower()
    assert "image_path" not in manifest_text.lower()

    def forbidden_rgb_open(*_args, **_kwargs):
        raise AssertionError("RGB loading is forbidden for ReconstructionInputs.load")

    monkeypatch.setattr(PILImage, "open", forbidden_rgb_open)
    loaded = ReconstructionInputs.load(path)
    assert loaded.view_names == inputs.view_names
    assert loaded.n_init_2d == [1, 1]
    assert loaded.n_opt_2d == [2, 2]
    assert torch.equal(loaded.points, inputs.points)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(loaded.point_visibility, inputs.point_visibility, strict=True)
    )
    assert torch.equal(loaded.bounds_hint[0], inputs.bounds_hint[0])
    assert loaded.bounds_hint[1] == inputs.bounds_hint[1]


def test_camera_projection_and_teacher_query_survive_bundle_round_trip(tmp_path):
    inputs = _inputs()
    path = tmp_path / "bundle"
    inputs.save(path)
    loaded = ReconstructionInputs.load(path)
    world = torch.tensor([[0.1, -0.2, 0.3], [-0.3, 0.2, 0.5]])
    for expected_camera, actual_camera, expected_field, actual_field in zip(
        inputs.cameras,
        loaded.cameras,
        inputs.observations,
        loaded.observations,
        strict=True,
    ):
        expected_uv, expected_depth = expected_camera.project(world)
        actual_uv, actual_depth = actual_camera.project(world)
        assert torch.allclose(actual_uv, expected_uv, atol=1e-6)
        assert torch.allclose(actual_depth, expected_depth, atol=1e-6)
        points = expected_field.pixel_centers()
        assert torch.equal(actual_field.query(points).color, expected_field.query(points).color)


def test_strict_bundle_round_trip_accepts_integrity_checked_mean_residuals(tmp_path):
    inputs = _inputs()
    inputs = replace(
        inputs,
        observations=[
            replace(
                inputs.observations[0],
                mean_residuals=torch.tensor(
                    [[1.0e-5, -2.0e-5], [3.0e-5, -4.0e-5]],
                    dtype=torch.float32,
                ),
            ),
            inputs.observations[1],
        ],
    )
    path = tmp_path / "residual-bundle"
    inputs.save(path)

    loaded = ReconstructionInputs.load(path, strict=True)

    assert torch.equal(
        loaded.observations[0].mean_residuals,
        inputs.observations[0].mean_residuals,
    )
    assert loaded.observations[1].mean_residuals is None


def test_strict_bundle_preflight_stats_survive_device_transfer(tmp_path):
    path = tmp_path / "bundle"
    _inputs().save(path)
    loaded = ReconstructionInputs.load(path, strict=True)
    assert loaded.archive_stats is not None
    assert loaded.archive_stats.manifest_bytes == (path / "manifest.json").stat().st_size
    assert len(loaded.archive_stats.teacher_archives) == 2
    assert loaded.archive_stats.geometry_archive is not None
    assert loaded.archive_stats.total_compressed_bytes == sum(
        item.compressed_bytes for item in loaded.archive_stats.archives
    )
    assert loaded.archive_stats.total_uncompressed_bytes > 0
    assert loaded.to("cpu").archive_stats is loaded.archive_stats


@pytest.mark.parametrize(
    ("limit_name", "message"),
    [
        ("max_manifest_bytes", "manifest byte cap exceeded"),
        ("max_teacher_archives", "teacher archive count cap exceeded"),
        ("max_archive_compressed_bytes", "archive compressed byte cap exceeded"),
        ("max_total_compressed_bytes", "bundle aggregate compressed byte cap exceeded"),
        ("max_zip_members", "ZIP member cap exceeded"),
        ("max_member_uncompressed_bytes", "member uncompressed byte cap exceeded"),
        ("max_archive_uncompressed_bytes", "archive uncompressed byte cap exceeded"),
    ],
)
def test_every_strict_bundle_cap_fails_before_numpy_load(
    tmp_path, monkeypatch, limit_name, message
):
    path = tmp_path / "bundle"
    _inputs().save(path)

    def forbidden_numpy_load(*_args, **_kwargs):
        raise AssertionError("np.load must not run before strict archive preflight passes")

    monkeypatch.setattr("rtgs.data.reconstruction_inputs.np.load", forbidden_numpy_load)
    with pytest.raises(ValueError, match=message):
        ReconstructionInputs.load(
            path,
            strict=True,
            limits=BundleLoadLimits(**{limit_name: 1}),
        )


def test_strict_bundle_limit_defaults_are_preregistered_literals():
    assert asdict(BundleLoadLimits()) == {
        "max_manifest_bytes": 8_388_608,
        "max_teacher_archives": 64,
        "max_archive_compressed_bytes": 268_435_456,
        "max_total_compressed_bytes": 2_147_483_648,
        "max_zip_members": 64,
        "max_member_uncompressed_bytes": 268_435_456,
        "max_archive_uncompressed_bytes": 1_073_741_824,
    }


def test_strict_bundle_rejects_symlinks_and_resolved_escape(tmp_path):
    path = tmp_path / "bundle"
    _inputs().save(path)
    teacher = path / "teachers" / "0000.teacher.npz"
    outside = tmp_path / "outside.teacher.npz"
    teacher.rename(outside)
    teacher.symlink_to(outside)
    with pytest.raises(ValueError, match="must not contain symlinks"):
        ReconstructionInputs.load(path, strict=True)

    teacher.unlink()
    outside.rename(teacher)

    def escape(manifest):
        manifest["views"][0]["teacher"] = "../outside.teacher.npz"

    _rewrite_manifest(path, escape)
    with pytest.raises(ValueError, match="teacher path must be"):
        ReconstructionInputs.load(path, strict=True)


def test_strict_bundle_rejects_root_and_intermediate_directory_symlinks(tmp_path):
    real_root = tmp_path / "real-root"
    _inputs().save(real_root)
    root_link = tmp_path / "root-link"
    root_link.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(ValueError, match="root must be an ordinary directory"):
        ReconstructionInputs.load(root_link, strict=True)

    bundle = tmp_path / "intermediate-bundle"
    _inputs().save(bundle)
    teachers = bundle / "teachers"
    outside = tmp_path / "outside-teachers"
    teachers.rename(outside)
    teachers.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="must not contain symlinks"):
        ReconstructionInputs.load(bundle, strict=True)


def test_strict_manifest_exact_keys_and_identifier_are_opt_in(tmp_path):
    compatible = tmp_path / "compatible"
    _inputs().save(compatible)

    def add_extension(manifest):
        manifest["legacy_extension"] = {"accepted": True}

    _rewrite_manifest(compatible, add_extension)
    assert ReconstructionInputs.load(compatible).n_views == 2
    with pytest.raises(ValueError, match="manifest keys are not exact"):
        ReconstructionInputs.load(compatible, strict=True)

    invalid_name = tmp_path / "invalid-name"
    _inputs().save(invalid_name)

    def change_name(manifest):
        manifest["name"] = "not a strict identifier"

    _rewrite_manifest(invalid_name, change_name)
    assert ReconstructionInputs.load(invalid_name).name == "not a strict identifier"
    with pytest.raises(ValueError, match="manifest name must match"):
        ReconstructionInputs.load(invalid_name, strict=True)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("view", "view record 0 keys are not exact"),
        ("camera", "view record 0 camera keys are not exact"),
        ("geometry", "geometry record keys are not exact"),
    ],
)
def test_strict_manifest_nested_key_sets_are_exact(tmp_path, target, message):
    path = tmp_path / f"nested-{target}"
    _inputs().save(path)

    def add_extra(manifest):
        if target == "view":
            manifest["views"][0]["extra"] = True
        elif target == "camera":
            manifest["views"][0]["camera"]["extra"] = True
        else:
            manifest["geometry"]["extra"] = True

    _rewrite_manifest(path, add_extra)
    with pytest.raises(ValueError, match=message):
        ReconstructionInputs.load(path, strict=True)


def test_strict_identifier_grammar_and_length_boundaries():
    import rtgs.data.reconstruction_inputs as module

    assert module._IDENTIFIER.pattern == r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z"
    for value in ("A", "a" * 128, "A_1.name-with-dashes"):
        assert module._require_identifier(value, label="test") == value
    for value in ("", "-leading", "has/slash", "has space", "ümlaut", "a" * 129):
        with pytest.raises(ValueError, match="must match"):
            module._require_identifier(value, label="test")


def test_bundle_rejects_teacher_tampering_and_overwrite(tmp_path):
    path = tmp_path / "bundle"
    inputs = _inputs()
    inputs.save(path)
    with pytest.raises(FileExistsError):
        inputs.save(path)
    teacher = path / "teachers" / "0000.teacher.npz"
    teacher.write_bytes(teacher.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="teacher archive digest mismatch"):
        ReconstructionInputs.load(path)


def test_from_scene_keeps_training_calibration_but_drops_rgb_references():
    images = [torch.rand(6, 8, 3) for _ in range(3)]
    scene = SceneData(
        images=images,
        cameras=[_camera(offset=-0.2), _camera(), _camera(offset=0.2)],
        view_names=["a", "b", "held-out"],
        train_indices=[0, 1],
        test_indices=[2],
        name="source-scene",
    )
    observations = [_observation("a"), _observation("b"), _observation("held-out")]
    compact = ReconstructionInputs.from_scene(scene, observations)
    assert compact.view_names == ["a", "b"]
    assert compact.n_views == 2
    assert not hasattr(compact, "images")
    with pytest.raises(AttributeError):
        compact.images = images
