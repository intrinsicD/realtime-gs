"""COLMAP parsing: text fixtures and binary round-trip."""

import struct

import numpy as np
import torch

from rtgs.data.colmap import (
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
    read_model,
    read_points3d_text,
)

CAMERAS_TXT = """# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
1 PINHOLE 640 480 500.0 505.0 320.0 240.0
2 SIMPLE_PINHOLE 320 240 260.0 160.0 120.0
"""

IMAGES_TXT = """# Image list with two lines of data per image:
1 0.9998477 0.0 0.0174524 0.0 0.1 -0.2 2.5 1 frame_000.png
100.0 200.0 -1 300.5 100.25 42
2 1.0 0.0 0.0 0.0 -0.3 0.0 2.0 2 frame_001.png

"""

POINTS_TXT = """# 3D point list
7 0.5 -0.25 1.5 200 100 50 0.75 1 0 2 1
9 -1.0 2.0 0.5 10 20 30 1.5 1 1
"""


def test_read_cameras_text(tmp_path):
    p = tmp_path / "cameras.txt"
    p.write_text(CAMERAS_TXT)
    cams = read_cameras_text(p)
    assert cams[1]["model"] == "PINHOLE"
    assert cams[1]["params"] == [500.0, 505.0, 320.0, 240.0]
    assert cams[2]["model"] == "SIMPLE_PINHOLE"
    assert cams[2]["width"] == 320


def test_read_images_text(tmp_path):
    p = tmp_path / "images.txt"
    p.write_text(IMAGES_TXT)
    images = read_images_text(p)
    assert len(images) == 2
    assert images[1]["name"] == "frame_000.png"
    assert images[1]["camera_id"] == 1
    assert images[1]["point3d_ids"] == [42]
    assert np.allclose(images[2]["tvec"], [-0.3, 0.0, 2.0])


def test_read_points_text(tmp_path):
    p = tmp_path / "points3D.txt"
    p.write_text(POINTS_TXT)
    pts = read_points3d_text(p)
    assert pts.shape == (2, 3)
    assert torch.allclose(pts[0], torch.tensor([0.5, -0.25, 1.5]))


def _write_binary_model(d):
    with open(d / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 1, 640, 480))  # id=1, PINHOLE
        f.write(struct.pack("<4d", 500.0, 505.0, 320.0, 240.0))
    with open(d / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<4d", 1.0, 0.0, 0.0, 0.0))
        f.write(struct.pack("<3d", 0.1, -0.2, 2.5))
        f.write(struct.pack("<I", 1))
        f.write(b"frame_000.png\x00")
        f.write(struct.pack("<Q", 2))  # two 2D points
        f.write(struct.pack("<ddq", 100.0, 200.0, -1))
        f.write(struct.pack("<ddq", 300.5, 100.25, 42))
    with open(d / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Q", 7))
        f.write(struct.pack("<3d", 0.5, -0.25, 1.5))
        f.write(struct.pack("<3B", 200, 100, 50))
        f.write(struct.pack("<d", 0.75))
        f.write(struct.pack("<Q", 2))
        f.write(struct.pack("<ii", 1, 0))
        f.write(struct.pack("<ii", 2, 1))


def test_read_binary_model(tmp_path):
    _write_binary_model(tmp_path)
    cams = read_cameras_binary(tmp_path / "cameras.bin")
    assert cams[1]["model"] == "PINHOLE"
    assert cams[1]["params"] == [500.0, 505.0, 320.0, 240.0]
    images = read_images_binary(tmp_path / "images.bin")
    assert images[1]["name"] == "frame_000.png"
    assert images[1]["point3d_ids"] == [42]
    assert np.allclose(images[1]["tvec"], [0.1, -0.2, 2.5])
    model = read_model(tmp_path)
    assert model.points.shape == (1, 3)
    assert torch.allclose(model.points[0], torch.tensor([0.5, -0.25, 1.5]))


def test_binary_preferred_over_text(tmp_path):
    _write_binary_model(tmp_path)
    (tmp_path / "cameras.txt").write_text(CAMERAS_TXT)
    (tmp_path / "images.txt").write_text(IMAGES_TXT)
    (tmp_path / "points3D.txt").write_text(POINTS_TXT)
    model = read_model(tmp_path)
    assert len(model.images) == 1  # came from binary
