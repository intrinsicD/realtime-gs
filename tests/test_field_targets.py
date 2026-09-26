"""Dense photometric targets decoded from compact 2D-Gaussian views (RTGS-025)."""

from __future__ import annotations

import hashlib

import pytest
import torch

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.compact_views import CompactView, save_compact_view
from rtgs.data.field_targets import (
    decode_compact_view,
    decode_field_grid,
    downscale_pinhole,
    quadrature_sites,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")

_SHA = hashlib.sha256(b"field target fixture").hexdigest()


def _field(fit_window=(2, 2, 6, 4)) -> GaussianObservationField:
    return GaussianObservationField(
        width=8,
        height=6,
        means=torch.tensor([[3.5, 3.5], [5.0, 3.0], [6.5, 4.5]]),
        log_scales=torch.log(torch.tensor([[0.9, 0.7], [0.8, 1.0], [0.7, 0.7]])),
        rotations=torch.tensor([0.2, -0.4, 0.1]),
        colors=torch.tensor([[1.2, -0.1, 0.4], [0.2, 0.8, 1.1], [0.4, 0.3, 0.2]]),
        amplitudes=torch.tensor([0.7, 0.5, 0.8]),
        fit_window=fit_window,
        view_id="C0001",
        n_init=3,
    )


def _camera() -> Camera:
    return Camera.look_at(
        eye=torch.tensor([0.0, 0.0, -3.0]), target=torch.zeros(3), width=8, height=6
    )


def test_quadrature_sites_are_edge_origin_box_samples() -> None:
    sites = quadrature_sites(width=2, height=1, factor=4, supersample=2)
    assert sites.shape == (2, 4, 2)
    expected_first = torch.tensor([[1.0, 1.0], [3.0, 1.0], [1.0, 3.0], [3.0, 3.0]])
    assert torch.equal(sites[0], expected_first)
    assert torch.equal(sites[1], expected_first + torch.tensor([4.0, 0.0]))
    centers = quadrature_sites(width=2, height=1, factor=4, supersample=1)
    assert torch.equal(centers[:, 0], torch.tensor([[2.0, 2.0], [6.0, 2.0]]))


def test_downscale_pinhole_preserves_projection_in_scaled_pixels() -> None:
    camera = _camera()
    small = downscale_pinhole(camera, 2)
    assert (small.width, small.height) == (4, 3)
    points = torch.tensor([[0.1, -0.2, 0.3], [-0.3, 0.2, -0.1]])
    full_uv, _ = camera.project(points)
    small_uv, _ = small.project(points)
    assert torch.allclose(small_uv * 2, full_uv, atol=1e-6)
    with pytest.raises(ValueError, match="divide"):
        downscale_pinhole(camera, 5)


@pytest.mark.parametrize("supersample", [1, 2])
def test_index_decode_matches_reference_scan_and_zeroes_outside_window(supersample) -> None:
    field = _field()
    color, coverage, name = decode_field_grid(
        field, downscale=2, supersample=supersample, backend="index"
    )
    assert name == "index"
    assert color.shape == (3, 4, 3) and coverage.shape == (3, 4)
    reference, reference_coverage, _ = decode_field_grid(
        field, downscale=2, supersample=supersample, backend="reference"
    )
    assert torch.allclose(color, reference, atol=1e-6)
    assert torch.equal(coverage, reference_coverage)
    # Downscaled column 0 and row 0 lie entirely left of / above the (2, 2) window origin.
    assert torch.equal(color[:, 0], torch.zeros(3, 3))
    assert torch.equal(color[0], torch.zeros(4, 3))
    assert torch.equal(coverage[1:, 1:], torch.ones(2, 3))
    assert float(color.min()) >= 0.0 and float(color.max()) <= 1.0
    # A direct box average of clamped reference queries reproduces the decoded target.
    sites = quadrature_sites(4, 3, 2, supersample).reshape(-1, 2)
    query = field.query(sites)
    direct = (query.color * query.valid[:, None]).clamp(0, 1)
    direct = direct.reshape(3, 4, supersample * supersample, 3).mean(2)
    assert torch.allclose(color, direct, atol=1e-6)


def test_compact_view_decode_includes_alpha_and_cache_round_trip(tmp_path) -> None:
    field = _field(fit_window=(2, 2, 4, 2))
    alpha = torch.tensor([[True, True, False, False], [True, False, False, True]])
    path = tmp_path / "C0001.rtgsv"
    save_compact_view(
        path,
        field,
        _camera(),
        calibration_sha256=_SHA,
        source_rgb_name="C0001.jpg",
        source_rgb_sha256=_SHA,
        alpha_crop=alpha,
        source_mask_name="mask_C0001.png",
        source_mask_sha256=_SHA,
    )
    view = CompactView.load(path)
    decoded = decode_compact_view(view, downscale=2, backend="index", cache_dir=tmp_path / "c")
    assert decoded.camera.width == 4 and decoded.camera.height == 3
    assert decoded.source_sha256 == view.sha256
    expected_alpha = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [0.0, 0.75, 0.25, 0.0], [0.0, 0.0, 0.0, 0.0]]
    )
    assert torch.equal(decoded.alpha, expected_alpha)
    cached = list((tmp_path / "c").glob("C0001_*.pt"))
    assert len(cached) == 1
    again = decode_compact_view(view, downscale=2, backend="reference", cache_dir=tmp_path / "c")
    assert again.backend == "index"  # served from the cache, not recomputed
    assert torch.equal(again.color, decoded.color)
    assert torch.equal(again.alpha, decoded.alpha)


def test_decode_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        decode_field_grid(_field(), downscale=2, backend="gpu")


@requires_cuda
def test_cuda_decode_matches_cpu_index() -> None:
    field = _field()
    cpu, _, _ = decode_field_grid(field, downscale=2, backend="index")
    gpu, _, name = decode_field_grid(field, downscale=2, backend="cuda")
    assert name == "cuda"
    assert torch.allclose(cpu, gpu, atol=2e-5)
