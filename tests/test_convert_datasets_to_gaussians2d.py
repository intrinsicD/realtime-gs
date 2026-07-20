"""CPU-only tests for dataset discovery and transactional source removal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from scripts import convert_datasets_to_gaussians2d as converter

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.compact_views import (
    CompactDataset,
    save_compact_view,
    write_compact_dataset_manifest,
)


def _calibration_record(view_id: str) -> dict:
    return {
        "camera_id": view_id,
        "extrinsics": {
            "view_matrix": [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        },
        "intrinsics": {
            "camera_matrix": [
                6.0,
                0.0,
                3.5,
                0.0,
                6.0,
                2.5,
                0.0,
                0.0,
                1.0,
            ],
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
            "resolution": [8, 6],
        },
    }


def _source_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset_root = tmp_path / "dataset"
    capture = dataset_root / "capture"
    frame = capture / "frame_00001"
    rgb = frame / "rgb"
    mask = frame / "mask"
    rgb.mkdir(parents=True)
    mask.mkdir()

    calibration = capture / "calibration_dome.json"
    calibration.write_text(
        json.dumps(
            {
                "cameras": [
                    _calibration_record("C0004"),
                    _calibration_record("C0012"),
                ]
            }
        ),
        encoding="utf-8",
    )
    (rgb / "rgb_4.jpeg").write_bytes(b"camera four RGB")
    (rgb / "C0012.jpg").write_bytes(b"camera twelve RGB")
    (rgb / "mask_C0004.jpg").write_bytes(b"stray mask, not RGB")
    (rgb / "notes.png").write_bytes(b"uncalibrated non-camera file")
    (mask / "mask_C0004.jpg").write_bytes(b"lossy mask")
    (mask / "mask_C0004.png").write_bytes(b"authoritative lossless mask")
    return dataset_root, frame, calibration


def _observation(
    view_id: str,
    *,
    masked: bool,
    fit_config_digest: str | None = None,
) -> GaussianObservationField:
    return GaussianObservationField(
        width=8,
        height=6,
        means=torch.tensor([[1.5, 1.5], [2.5, 1.5]]),
        log_scales=torch.log(torch.tensor([[0.8, 0.6], [0.7, 0.9]])),
        rotations=torch.tensor([0.2, -0.4]),
        colors=torch.tensor([[0.8, 0.1, 0.4], [0.2, 0.7, 0.9]]),
        amplitudes=torch.tensor([0.7, 0.5]),
        mean_residuals=torch.zeros(2, 2),
        fit_window=(1, 1, 3, 2) if masked else (0, 0, 8, 6),
        view_id=view_id,
        n_init=2,
        fit_config_digest=fit_config_digest,
    )


def _save_converted_view(
    source: converter.SourceView,
    *,
    fit_config_digest: str | None = None,
    camera: Camera | None = None,
) -> Path:
    masked = source.mask_path is not None
    path = source.frame / "gaussians2d" / f"{source.view_id}.rtgsv"
    alpha = torch.tensor([[True, False, True], [False, True, True]]) if masked else None
    save_compact_view(
        path,
        _observation(
            source.view_id,
            masked=masked,
            fit_config_digest=fit_config_digest,
        ),
        converter._camera_from_source(source)[0] if camera is None else camera,
        calibration_sha256=source.calibration_sha256,
        source_rgb_name=source.rgb_path.name,
        source_rgb_sha256=source.rgb_sha256,
        alpha_crop=alpha,
        source_mask_name=None if source.mask_path is None else source.mask_path.name,
        source_mask_sha256=source.mask_sha256,
    )
    return path


def _write_manifest(frame: converter.SourceFrame, paths: list[Path]) -> None:
    write_compact_dataset_manifest(
        frame.path / "gaussians2d",
        name=frame.path.name,
        calibration_sha256=frame.calibration_sha256,
        view_paths=paths,
        bounds_hint=(torch.zeros(3), 1.0),
    )


def test_discovery_canonicalizes_ids_prefers_png_and_excludes_stray_mask(
    tmp_path: Path,
) -> None:
    dataset_root, frame_path, calibration = _source_tree(tmp_path)

    frames = converter.discover_source_frames(dataset_root)

    assert len(frames) == 1
    frame = frames[0]
    assert frame.path == frame_path.resolve()
    assert frame.calibration_path == calibration.resolve()
    assert [view.view_id for view in frame.views] == ["C0004", "C0012"]
    assert [view.rgb_path.name for view in frame.views] == ["rgb_4.jpeg", "C0012.jpg"]
    assert frame.views[0].mask_path == (frame_path / "mask/mask_C0004.png").resolve()
    assert frame.views[1].mask_path is None
    assert all(view.rgb_path.name != "mask_C0004.jpg" for view in frame.views)
    assert frame.source_directories == (
        (frame_path / "rgb").resolve(),
        (frame_path / "mask").resolve(),
    )


def test_incomplete_conversion_refuses_source_removal_without_deleting(
    tmp_path: Path,
) -> None:
    dataset_root, frame_path, calibration = _source_tree(tmp_path)
    frame = converter.discover_source_frames(dataset_root)[0]
    first_path = _save_converted_view(frame.views[0])
    _write_manifest(frame, [first_path])

    with pytest.raises(RuntimeError, match="compact view inventory mismatch"):
        converter.remove_verified_sources([frame], dataset_root=dataset_root.resolve())

    assert (frame_path / "rgb").is_dir()
    assert (frame_path / "mask").is_dir()
    assert (frame_path / "rgb/rgb_4.jpeg").read_bytes() == b"camera four RGB"
    assert (frame_path / "mask/mask_C0004.png").read_bytes() == (b"authoritative lossless mask")
    assert calibration.is_file()
    assert first_path.is_file()
    assert (frame_path / "gaussians2d/manifest.json").is_file()


def test_complete_verified_conversion_removes_only_sources(
    tmp_path: Path,
) -> None:
    dataset_root, frame_path, calibration = _source_tree(tmp_path)
    frame = converter.discover_source_frames(dataset_root)[0]
    view_paths = [_save_converted_view(source) for source in frame.views]
    _write_manifest(frame, view_paths)

    converter.remove_verified_sources([frame], dataset_root=dataset_root.resolve())

    assert not (frame_path / "rgb").exists()
    assert not (frame_path / "mask").exists()
    assert calibration.is_file()
    assert frame_path.is_dir()
    assert all(path.is_file() for path in view_paths)
    dataset = CompactDataset.load(frame_path / "gaussians2d")
    assert [view.view_id for view in dataset.views] == ["C0004", "C0012"]
    assert dataset.views[0].alpha is not None
    assert dataset.views[1].alpha is None


def test_production_contract_accepts_expected_digest_and_rejects_other_iterations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root, _, _ = _source_tree(tmp_path)
    source = converter.discover_source_frames(dataset_root)[0].views[0]
    expected_digest = "a" * 64
    path = _save_converted_view(source, fit_config_digest=expected_digest)
    calls: list[tuple[int, int, int, int]] = []

    def expected_fit_digest(
        source_arg: converter.SourceView,
        *,
        height: int,
        width: int,
        n_gaussians: int,
        iterations: int,
    ) -> str:
        assert source_arg is source
        calls.append((height, width, n_gaussians, iterations))
        return expected_digest if iterations == 1000 else "b" * 64

    monkeypatch.setattr(converter, "_expected_fit_digest", expected_fit_digest)

    assert converter._bundle_matches_source(
        path,
        source,
        n_gaussians=2,
        iterations=1000,
    )
    assert not converter._bundle_matches_source(
        path,
        source,
        n_gaussians=2,
        iterations=999,
    )
    assert calls == [(2, 3, 2, 1000), (2, 3, 2, 999)]


def test_bundle_with_wrong_embedded_camera_is_rejected(tmp_path: Path) -> None:
    dataset_root, _, _ = _source_tree(tmp_path)
    source = converter.discover_source_frames(dataset_root)[0].views[0]
    wrong_camera = Camera.look_at(
        eye=torch.tensor([0.0, 0.0, -3.0]),
        target=torch.zeros(3),
        width=8,
        height=6,
    )
    path = _save_converted_view(source, camera=wrong_camera)

    assert not converter._bundle_matches_source(path, source)


def test_source_mutation_after_inventory_prevents_deletion(
    tmp_path: Path,
) -> None:
    dataset_root, frame_path, _ = _source_tree(tmp_path)
    frame = converter.discover_source_frames(dataset_root)[0]
    view_paths = [_save_converted_view(source) for source in frame.views]
    _write_manifest(frame, view_paths)
    frame.views[0].rgb_path.write_bytes(b"mutated after inventory")

    with pytest.raises(RuntimeError, match="source file changed during conversion"):
        converter.remove_verified_sources([frame], dataset_root=dataset_root.resolve())

    assert (frame_path / "rgb").is_dir()
    assert (frame_path / "mask").is_dir()
    assert all(path.is_file() for path in view_paths)
    assert (frame_path / "gaussians2d/manifest.json").is_file()
