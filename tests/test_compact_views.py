"""CPU-only tests for strict compact 2D-Gaussian view containers."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest
import torch

import rtgs.data.compact_views as compact_views
from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.compact_views import (
    COMPACT_VIEW_BYTE_CAP,
    CompactDataset,
    CompactView,
    CompactViewTooLarge,
    save_compact_view,
    write_compact_dataset_manifest,
)

_CALIBRATION_SHA256 = hashlib.sha256(b"tiny calibrated capture").hexdigest()
_RGB_SHA256 = hashlib.sha256(b"source rgb").hexdigest()
_MASK_SHA256 = hashlib.sha256(b"source mask").hexdigest()


def _camera(*, offset: float = 0.0, width: int = 8, height: int = 6) -> Camera:
    return Camera.look_at(
        eye=torch.tensor([offset, 0.0, -3.0]),
        target=torch.zeros(3),
        width=width,
        height=height,
    )


def _observation(
    view_id: str,
    *,
    fit_window: tuple[int, int, int, int] = (1, 1, 3, 2),
) -> GaussianObservationField:
    return GaussianObservationField(
        width=8,
        height=6,
        means=torch.tensor([[1.5, 1.5], [2.5, 1.5], [3.5, 2.5]]),
        log_scales=torch.log(torch.tensor([[0.8, 0.6], [0.7, 0.9], [0.6, 0.6]])),
        rotations=torch.tensor([0.2, -0.4, 0.1]),
        colors=torch.tensor([[1.2, -0.1, 0.4], [0.2, 0.8, 1.1], [0.4, 0.3, 0.2]]),
        amplitudes=torch.tensor([0.7, 0.5, 0.8]),
        mean_residuals=torch.tensor(
            [[1.0e-5, -2.0e-5], [0.0, 0.0], [-3.0e-5, 4.0e-5]],
            dtype=torch.float32,
        ),
        fit_window=fit_window,
        view_id=view_id,
        n_init=3,
    )


def _save_view(
    path: Path,
    view_id: str,
    *,
    masked: bool,
    offset: float = 0.0,
) -> tuple[GaussianObservationField, Camera, torch.Tensor | None]:
    fit_window = (1, 1, 3, 2) if masked else (0, 0, 8, 6)
    observation = _observation(view_id, fit_window=fit_window)
    camera = _camera(offset=offset)
    alpha = torch.tensor([[True, False, True], [False, True, True]]) if masked else None
    save_compact_view(
        path,
        observation,
        camera,
        calibration_sha256=_CALIBRATION_SHA256,
        source_rgb_name=f"{view_id}.jpg",
        source_rgb_sha256=_RGB_SHA256,
        alpha_crop=alpha,
        source_mask_name=(f"mask_{view_id}.png" if masked else None),
        source_mask_sha256=(_MASK_SHA256 if masked else None),
    )
    return observation, camera, alpha


def _zip_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {item.filename: archive.read(item) for item in archive.infolist()}


def _write_zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    path.unlink()
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members:
            archive.writestr(name, payload)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _rebind_metadata(metadata: dict) -> bytes:
    metadata.pop("semantic_digest", None)
    metadata["semantic_digest"] = hashlib.sha256(_canonical_json(metadata)).hexdigest()
    return _canonical_json(metadata)


def _assert_field_equal(
    actual: GaussianObservationField,
    expected: GaussianObservationField,
) -> None:
    assert actual.width == expected.width
    assert actual.height == expected.height
    assert actual.fit_window == expected.fit_window
    assert actual.view_id == expected.view_id
    assert actual.n_init == expected.n_init
    for name in (
        "means",
        "log_scales",
        "rotations",
        "colors",
        "amplitudes",
        "mean_residuals",
    ):
        assert torch.equal(getattr(actual, name), getattr(expected, name))
    query_points = expected.pixel_centers()
    assert torch.equal(actual.query(query_points).color, expected.query(query_points).color)


def test_masked_view_round_trip_preserves_teacher_camera_and_lossless_alpha(
    tmp_path,
) -> None:
    path = tmp_path / "masked.rtgsv"
    observation, camera, alpha = _save_view(path, "masked", masked=True)
    expected_teacher = tmp_path / "expected.teacher.npz"
    observation.save_npz(expected_teacher)

    loaded = CompactView.load(path)

    assert loaded.bytes == path.stat().st_size <= COMPACT_VIEW_BYTE_CAP
    assert loaded.sha256 == compact_views.file_sha256(path)
    assert loaded.calibration_sha256 == _CALIBRATION_SHA256
    assert loaded.source == {
        "rgb": {"name": "masked.jpg", "sha256": _RGB_SHA256},
        "mask": {"name": "mask_masked.png", "sha256": _MASK_SHA256},
    }
    _assert_field_equal(loaded.observation, observation)
    assert loaded.camera.fx == camera.fx
    assert loaded.camera.fy == camera.fy
    assert loaded.camera.cx == camera.cx
    assert loaded.camera.cy == camera.cy
    assert torch.equal(loaded.camera.R, camera.R)
    assert torch.equal(loaded.camera.t, camera.t)
    assert loaded.alpha is not None
    assert torch.equal(loaded.alpha.crop_mask(), alpha)
    expected_full = torch.zeros(6, 8, dtype=torch.bool)
    expected_full[1:3, 1:4] = alpha
    assert torch.equal(loaded.alpha.full_mask((6, 8)), expected_full)

    members = _zip_members(path)
    assert set(members) == {"metadata.json", "teacher.npz", "alpha.packbits"}
    assert members["teacher.npz"] == expected_teacher.read_bytes()


def test_unmasked_view_round_trip_has_no_alpha_member(tmp_path) -> None:
    path = tmp_path / "unmasked.rtgsv"
    observation, camera, alpha = _save_view(path, "unmasked", masked=False)

    loaded = CompactView.load(path)

    assert alpha is None
    assert loaded.alpha is None
    assert loaded.source["mask"] is None
    assert set(_zip_members(path)) == {"metadata.json", "teacher.npz"}
    _assert_field_equal(loaded.observation, observation)
    world = torch.tensor([[0.1, -0.2, 0.3], [-0.3, 0.2, 0.5]])
    expected_uv, expected_depth = camera.project(world)
    actual_uv, actual_depth = loaded.camera.project(world)
    assert torch.equal(actual_uv, expected_uv)
    assert torch.equal(actual_depth, expected_depth)


def test_total_byte_cap_accepts_exact_boundary(tmp_path, monkeypatch) -> None:
    path = tmp_path / "boundary.rtgsv"
    _save_view(path, "boundary", masked=False)
    real_lstat = compact_views.os.lstat

    def exact_boundary_lstat(candidate, *args, **kwargs):
        result = real_lstat(candidate, *args, **kwargs)
        if not args and not kwargs and Path(candidate) == path:
            values = list(result)
            values[6] = COMPACT_VIEW_BYTE_CAP
            return os.stat_result(values)
        return result

    monkeypatch.setattr(compact_views.os, "lstat", exact_boundary_lstat)

    loaded = CompactView.load(path)

    assert loaded.bytes == COMPACT_VIEW_BYTE_CAP


def test_total_byte_cap_rejects_before_opening_zip(tmp_path, monkeypatch) -> None:
    path = tmp_path / "oversized.rtgsv"
    _save_view(path, "oversized", masked=False)
    real_lstat = compact_views.os.lstat

    def oversized_lstat(candidate, *args, **kwargs):
        result = real_lstat(candidate, *args, **kwargs)
        if not args and not kwargs and Path(candidate) == path:
            values = list(result)
            values[6] = COMPACT_VIEW_BYTE_CAP + 1
            return os.stat_result(values)
        return result

    def zip_open_forbidden(*_args, **_kwargs):
        raise AssertionError("oversized compact view reached ZIP parsing")

    monkeypatch.setattr(compact_views.os, "lstat", oversized_lstat)
    monkeypatch.setattr(compact_views.zipfile, "ZipFile", zip_open_forbidden)

    with pytest.raises(CompactViewTooLarge) as error:
        CompactView.load(path)
    assert error.value.actual_bytes == COMPACT_VIEW_BYTE_CAP + 1
    assert error.value.byte_cap == COMPACT_VIEW_BYTE_CAP


def test_view_rejects_corrupt_unknown_and_duplicate_members(tmp_path) -> None:
    corrupt = tmp_path / "corrupt.rtgsv"
    _save_view(corrupt, "corrupt", masked=False)
    members = _zip_members(corrupt)
    teacher = bytearray(members["teacher.npz"])
    teacher[-1] ^= 0x01
    _write_zip(
        corrupt,
        [
            ("metadata.json", members["metadata.json"]),
            ("teacher.npz", bytes(teacher)),
        ],
    )
    with pytest.raises(ValueError, match="teacher digest mismatch"):
        CompactView.load(corrupt)

    unknown = tmp_path / "unknown.rtgsv"
    _save_view(unknown, "unknown", masked=False)
    members = _zip_members(unknown)
    _write_zip(
        unknown,
        [
            ("metadata.json", members["metadata.json"]),
            ("teacher.npz", members["teacher.npz"]),
            ("unknown.bin", b"unexpected"),
        ],
    )
    with pytest.raises(ValueError, match="ZIP member set is not exact"):
        CompactView.load(unknown)

    duplicate = tmp_path / "duplicate.rtgsv"
    _save_view(duplicate, "duplicate", masked=False)
    members = _zip_members(duplicate)
    with pytest.warns(UserWarning, match="Duplicate name"):
        _write_zip(
            duplicate,
            [
                ("metadata.json", members["metadata.json"]),
                ("teacher.npz", members["teacher.npz"]),
                ("teacher.npz", members["teacher.npz"]),
            ],
        )
    with pytest.raises(ValueError, match="duplicate ZIP members"):
        CompactView.load(duplicate)


def test_view_rejects_nonzero_alpha_padding_bits_with_valid_digests(tmp_path) -> None:
    path = tmp_path / "padding.rtgsv"
    _save_view(path, "padding", masked=True)
    members = _zip_members(path)
    alpha = bytearray(members["alpha.packbits"])
    assert len(alpha) == 1
    alpha[-1] |= 0b1000_0000
    metadata = json.loads(members["metadata.json"])
    metadata["alpha"]["sha256"] = hashlib.sha256(alpha).hexdigest()
    _write_zip(
        path,
        [
            ("metadata.json", _rebind_metadata(metadata)),
            ("teacher.npz", members["teacher.npz"]),
            ("alpha.packbits", bytes(alpha)),
        ],
    )

    with pytest.raises(ValueError, match="non-zero padding bits"):
        CompactView.load(path)


def test_dataset_manifest_round_trip_and_reconstruction_inputs(tmp_path) -> None:
    frame = tmp_path / "frame"
    directory = frame / "gaussians2d"
    directory.mkdir(parents=True)
    right_path = directory / "right.rtgsv"
    left_path = directory / "left.rtgsv"
    right_observation, right_camera, _ = _save_view(
        right_path,
        "right",
        masked=False,
        offset=0.2,
    )
    left_observation, left_camera, left_alpha = _save_view(
        left_path,
        "left",
        masked=True,
        offset=-0.2,
    )
    bounds = (torch.tensor([0.1, -0.2, 0.3]), 2.5)
    write_compact_dataset_manifest(
        directory,
        name="tiny-frame",
        calibration_sha256=_CALIBRATION_SHA256,
        view_paths=[right_path, left_path],
        bounds_hint=bounds,
    )

    dataset = CompactDataset.load(frame)
    inputs = dataset.to_reconstruction_inputs()

    assert dataset.path == directory.resolve()
    assert dataset.name == "tiny-frame"
    assert dataset.n_views == 2
    assert [view.view_id for view in dataset.views] == ["right", "left"]
    assert dataset.alphas[0] is None
    assert dataset.alphas[1] is not None
    assert torch.equal(dataset.alphas[1].crop_mask(), left_alpha)
    assert torch.equal(dataset.bounds_hint[0], bounds[0])
    assert dataset.bounds_hint[1] == bounds[1]
    assert inputs.view_names == ["right", "left"]
    assert inputs.name == "tiny-frame"
    assert inputs.points is None
    assert inputs.point_visibility is None
    _assert_field_equal(inputs.observations[0], right_observation)
    _assert_field_equal(inputs.observations[1], left_observation)
    world = torch.tensor([[0.0, 0.0, 0.0], [0.2, -0.1, 0.4]])
    for expected, actual in zip(
        [right_camera, left_camera],
        inputs.cameras,
        strict=True,
    ):
        expected_uv, expected_depth = expected.project(world)
        actual_uv, actual_depth = actual.project(world)
        assert torch.equal(actual_uv, expected_uv)
        assert torch.equal(actual_depth, expected_depth)

    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["views"][0]["sha256"] = "0" * 64
    manifest_path.write_bytes(_rebind_metadata(manifest))
    with pytest.raises(ValueError, match="compact dataset view digest mismatch"):
        CompactDataset.load(directory)
