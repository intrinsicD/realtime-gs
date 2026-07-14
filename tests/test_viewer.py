"""CPU-only tests for interactive-viewer data preparation and CLI wiring."""

from __future__ import annotations

import numpy as np
import torch

from rtgs.cli import main as cli_main
from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D, quat_to_rotmat
from rtgs.core.sh import rgb_to_sh
from rtgs.viewer import (
    _image_uint8,
    camera_to_viewer_pose,
    prepare_viewer_data,
    selected_gaussians,
)


def _gaussians() -> Gaussians3D:
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.75, 0.25], [0.0, 0.0, 1.0]])
    return Gaussians3D(
        means=torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [float("nan"), 0.0, 0.0]]),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1),
        log_scales=torch.tensor([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]]).log(),
        opacity=torch.tensor([0.5, 0.8, 1.0]),
        sh=rgb_to_sh(colors)[:, None],
    )


def test_prepare_viewer_data_filters_ranks_and_preserves_covariance():
    data = prepare_viewer_data(_gaussians())
    assert data.n == 2
    assert data.source_indices.tolist() == [1, 0]
    assert data.centers.dtype == np.float32
    assert np.allclose(data.centers[0], [1.0, 2.0, 3.0])
    assert np.allclose(data.covariances[0], np.eye(3) * 0.2**2, atol=1e-6)
    assert np.allclose(data.rgbs[0], [0.0, 0.75, 0.25], atol=1e-6)
    assert np.allclose(data.opacities[:, 0], [0.8, 0.5])


def test_selected_gaussians_matches_controls_and_cap():
    gaussians = _gaussians()
    data = prepare_viewer_data(gaussians, max_gaussians=1)
    selected = selected_gaussians(gaussians, data, count=50, opacity_scale=2.0)
    assert selected.n == 1
    assert torch.equal(selected.means[0], gaussians.means[1])
    assert selected.opacity.item() == 1.0


def test_camera_to_viewer_pose_uses_camera_to_world_rotation():
    camera = Camera.look_at(
        eye=torch.tensor([2.0, -1.0, 3.0]),
        target=torch.tensor([0.0, 0.0, 0.0]),
        width=80,
        height=40,
    )
    wxyz, position = camera_to_viewer_pose(camera)
    recovered = quat_to_rotmat(torch.from_numpy(wxyz).float()[None])[0]
    assert torch.allclose(recovered, camera.R.T, atol=1e-5)
    assert np.allclose(position, camera.position.numpy())


def test_viewer_image_conversion_resizes_on_cpu():
    image = torch.linspace(0.0, 1.0, 40 * 20 * 3).reshape(40, 20, 3)
    converted = _image_uint8(image, max_side=10)
    assert converted.shape == (10, 5, 3)
    assert converted.dtype == np.uint8


def test_cli_view_auto_detects_initial_without_importing_viser(tmp_path, monkeypatch):
    gaussians = _gaussians().subset(torch.tensor([0, 1]))
    gaussians.save_ply(tmp_path / "gaussians.ply")
    gaussians.subset(torch.tensor([0])).save_ply(tmp_path / "gaussians_init.ply")
    called = {}

    def fake_launch(models, **kwargs):
        called["models"] = models
        called["kwargs"] = kwargs

    monkeypatch.setattr("rtgs.viewer.launch_viewer", fake_launch)
    result = cli_main(
        [
            "view",
            "--gaussians",
            str(tmp_path / "gaussians.ply"),
            "--device",
            "cpu",
            "--no-open",
        ]
    )
    assert result == 0
    assert list(called["models"]) == ["final", "initial"]
    assert called["kwargs"]["scene"] is None
    assert called["kwargs"]["device"] == torch.device("cpu")
    assert called["kwargs"]["open_browser"] is False
