"""Outcome-free tests for the sealed point-rasterizer experiment harness."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest
import torch
from benchmarks import point_rasterizer_parity as experiment
from PIL import Image as PILImage


def _temporary_calibration(path: Path) -> None:
    payload = {
        "cameras": [
            {
                "camera_id": "C0001",
                "intrinsics": {
                    "resolution": [160, 96],
                    "camera_matrix": [120.0, 0.0, 79.5, 0.0, 118.0, 47.5, 0.0, 0.0, 1.0],
                },
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
                        2.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ]
                },
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_camera_only_loader_does_not_decode_or_enumerate_images(tmp_path, monkeypatch):
    calibration = tmp_path / "calibration_dome.json"
    _temporary_calibration(calibration)

    def forbidden(*args, **kwargs):
        raise AssertionError("camera-only loading touched an image path")

    monkeypatch.setattr(PILImage, "open", forbidden)
    monkeypatch.setattr(Path, "iterdir", forbidden)
    import rtgs.data.calibrated as calibrated

    monkeypatch.setattr(calibrated, "load_calibrated_scene", forbidden)
    camera = experiment.load_calibrated_camera_only(calibration, downscale=16)
    assert camera.width == 10
    assert camera.height == 6
    assert camera.fx == pytest.approx(7.5)
    assert camera.fy == pytest.approx(7.375)
    assert camera.cx == pytest.approx(5.0)
    assert camera.cy == pytest.approx(3.0)
    assert torch.equal(camera.R, torch.eye(3))
    assert torch.equal(camera.t, torch.tensor([0.0, 0.0, 2.0]))


def test_harness_import_is_outcome_free_and_constructors_are_explicit():
    source = inspect.getsource(experiment)
    fixture_source = inspect.getsource(experiment.official_fixture)
    teacher_source = inspect.getsource(experiment.official_teacher)
    assert "official_fixture(seed)" not in source.split("def execute_phase_a", maxsplit=1)[0]
    assert 'torch.Generator(device="cpu").manual_seed(seed)' in fixture_source
    assert "seed not in OFFICIAL_SEEDS" in fixture_source
    assert "55 / 96" not in teacher_source
    assert experiment.OFFICIAL_SEEDS == (91301, 91302, 91303)
    assert experiment.POINT_CHUNKS == (1, 7, 4096)
    assert experiment.GAUSSIAN_CHUNKS == (1, 3, 4096)


def test_atomic_json_creation_refuses_overwrite(tmp_path):
    path = tmp_path / "artifact.json"
    digest = experiment._exclusive_json(path, {"status": "PASS"})
    assert digest == experiment.sha256_file(path)
    with pytest.raises(FileExistsError):
        experiment._exclusive_json(path, {"status": "FAIL"})
    assert experiment.strict_json_load(path) == {"status": "PASS"}


@pytest.mark.parametrize(
    "verdict_text",
    (
        "Verdict: `PASS`",
        "## Verdict\n\n**QUALIFIED** — narrow mechanism claim only.",
    ),
)
def test_audit_verdict_requires_current_artifact_bindings(verdict_text):
    text = "\n".join((verdict_text, "prereg-123", "seal-456", "result-789"))
    verdict = experiment.validate_audit_text(
        text,
        preregistration_sha256="prereg-123",
        seal_sha256="seal-456",
        result_sha256="result-789",
    )
    assert verdict in {"PASS", "QUALIFIED"}
    with pytest.raises(RuntimeError, match="missing current artifact bindings"):
        experiment.validate_audit_text(
            verdict_text,
            preregistration_sha256="prereg-123",
            seal_sha256="seal-456",
            result_sha256="result-789",
        )


def test_calibrated_sampling_honors_small_and_large_image_branches():
    small, small_mode, small_seed = experiment.calibrated_sample_indices(64, 64)
    assert torch.equal(small, torch.arange(4096))
    assert small_mode == "all_pixel_centers"
    assert small_seed is None

    first, large_mode, large_seed = experiment.calibrated_sample_indices(65, 64)
    second, _, _ = experiment.calibrated_sample_indices(65, 64)
    assert first.shape == (4096,)
    assert torch.equal(first, second)
    assert int(first.min()) >= 0 and int(first.max()) < 65 * 64
    assert large_mode == "uniform_with_replacement"
    assert large_seed == 93001


def test_exact_command_rejects_extra_arguments(monkeypatch):
    monkeypatch.setattr(sys, "executable", str(experiment.ROOT / ".venv/bin/python"))
    monkeypatch.setattr(sys, "argv", ["point_rasterizer_parity.py", "run"])
    experiment.assert_exact_command("run")
    monkeypatch.setattr(sys, "argv", ["point_rasterizer_parity.py", "run", "--output", "x"])
    with pytest.raises(RuntimeError, match="without extra args"):
        experiment.assert_exact_command("run")


def test_preregistration_and_historical_dense_anchor_hashes_are_still_bound():
    assert experiment.sha256_file(experiment.ROOT / experiment.PREREGISTRATION) == (
        experiment.PREREGISTRATION_SHA256
    )
    # These are pre-implementation chronology anchors, not a permanent ban on later
    # repository development.  Once the one-shot experiment is sealed, its immutable
    # seal is the evidence that the official run used the frozen versions.
    seal = experiment.strict_json_load(experiment.SEAL)
    for relative, expected in experiment.FROZEN_ANCHOR_HASHES.items():
        assert seal["source_hashes"][relative] == expected


def test_calibrated_path_has_no_rgb_loader_or_pil_dependency():
    source = inspect.getsource(experiment.run_calibrated)
    loader_source = inspect.getsource(experiment.load_calibrated_camera_only)
    combined = source + loader_source
    assert "load_calibrated_scene" not in combined
    assert "SceneData" not in combined
    assert "PIL" not in combined
    assert "rgb" not in loader_source.lower()
    assert "mask" not in loader_source.lower()
