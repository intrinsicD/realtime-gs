"""Calibrated object-capture loading and external 2D-Gaussian adapters."""

import json

import numpy as np
import torch
from PIL import Image

from rtgs.data.calibrated import load_calibrated_scene
from rtgs.image2gs.adapters import load_gaussians2d


def test_load_calibrated_scene_resizes_masks_and_intrinsics(tmp_path):
    frame = tmp_path / "frame_00001"
    (frame / "rgb").mkdir(parents=True)
    (frame / "mask").mkdir()
    cameras = []
    for index in range(2):
        camera_id = f"C{index:04d}"
        rgb = np.zeros((6, 8, 3), dtype=np.uint8)
        rgb[1:5, 2:6] = (200, 100, 50)
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[1:5, 2:6] = 255
        Image.fromarray(rgb).save(frame / "rgb" / f"{camera_id}.jpg")
        Image.fromarray(mask).save(frame / "mask" / f"mask_{camera_id}.png")
        view = np.eye(4)
        view[0, 3] = float(index)
        cameras.append(
            {
                "camera_id": camera_id,
                "extrinsics": {"view_matrix": view.reshape(-1).tolist()},
                "intrinsics": {
                    "camera_matrix": [6.0, 0.0, 3.0, 0.0, 6.0, 2.0, 0.0, 0.0, 1.0],
                    "distortion_coefficients": [0.0] * 5,
                    "resolution": [8, 6],
                },
            }
        )
    (tmp_path / "calibration_dome.json").write_text(json.dumps({"cameras": cameras}))

    scene = load_calibrated_scene(frame, downscale=2, test_every=2)
    assert scene.n_views == 2
    assert scene.images[0].shape == (3, 4, 3)
    assert scene.masks is not None and scene.masks[0].shape == (3, 4)
    assert scene.cameras[0].fx == 3.0
    assert scene.cameras[0].cx == 1.75  # scaled OpenCV principal point + half-pixel shift
    assert scene.train_indices == [0] and scene.test_indices == [1]
    center, extent = scene.center_and_extent()
    assert torch.isfinite(center).all() and extent > 0


def test_structsplat_adapter_converts_rs_and_pixel_centers(tmp_path):
    path = tmp_path / "field.npz"
    np.savez(
        path,
        means=np.array([[2.0, 3.0]], np.float32),
        log_scales=np.log(np.array([[2.0, 1.0]], np.float32)),
        rotations=np.array([np.pi / 4], np.float32),
        colors=np.array([[1.2, 0.4, -0.1]], np.float32),
        opacities=np.array([0.0], np.float32),
    )
    gaussian = load_gaussians2d(path)
    assert torch.allclose(gaussian.xy, torch.tensor([[2.5, 3.5]]))
    assert torch.allclose(gaussian.color, torch.tensor([[1.0, 0.4, 0.0]]))
    assert torch.allclose(gaussian.weight, torch.tensor([0.5]))
    expected = torch.tensor([[[2.5, 1.5], [1.5, 2.5]]])
    assert torch.allclose(gaussian.covariance(), expected, atol=1e-5)
