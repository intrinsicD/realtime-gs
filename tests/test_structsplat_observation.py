"""Parity checks between the optional StructSplat renderer and the frozen teacher contract."""

import pytest
import torch

from rtgs.image2gs.structsplat_backend import field_to_observation

structsplat = pytest.importorskip("structsplat")


def _native_field():
    from structsplat.gaussians import GaussianField

    return GaussianField(
        means=torch.tensor([[-0.2, 1.1], [2.4, 2.2], [4.1, 0.3]], dtype=torch.float64),
        log_scales=torch.log(
            torch.tensor([[1.2, 0.7], [0.8, 1.4], [1.1, 0.9]], dtype=torch.float64)
        ),
        rotations=torch.tensor([0.3, -0.7, 1.1], dtype=torch.float64),
        colors=torch.tensor(
            [[1.3, -0.1, 0.2], [0.2, 0.8, 1.1], [-0.3, 0.4, 0.7]],
            dtype=torch.float64,
        ),
        opacities=torch.tensor([1.2, -0.4, 0.8], dtype=torch.float64),
        color_grads=torch.tensor(
            [
                [[0.1, 0.0, -0.1], [0.0, 0.2, 0.0]],
                [[-0.1, 0.1, 0.0], [0.2, 0.0, 0.1]],
                [[0.0, -0.2, 0.1], [0.1, 0.1, -0.1]],
            ],
            dtype=torch.float64,
        ),
        filter_variance=torch.tensor([0.25, 0.0, 0.4], dtype=torch.float64),
    )


@pytest.mark.parametrize("blend_mode", ["normalized", "additive"])
def test_full_grid_matches_independent_structsplat_renderer(blend_mode):
    from structsplat.render import render, render_additive

    field = _native_field()
    height, width = 4, 5
    sigma_cutoff = 2.5
    fade_alpha = 0.35
    aa_dilation = 0.2
    render_fn = render if blend_mode == "normalized" else render_additive
    expected = render_fn(
        field.means,
        field.conics(aa_dilation),
        field.colors,
        field.radii(sigma_cutoff, aa_dilation),
        height,
        width,
        opacities=field.opacity_values(),
        support_fade=True,
        sigma_cutoff=sigma_cutoff,
        color_grads=field.color_grads,
        scales=field.effective_scales(0.0),
        rotations=field.rotations,
        support_fade_alpha=fade_alpha,
    )
    observation = field_to_observation(
        field,
        canvas_size=(height, width),
        blend_mode=blend_mode,
        sigma_cutoff=sigma_cutoff,
        support_fade_alpha=fade_alpha,
        aa_dilation=aa_dilation,
    )
    actual = observation.query(observation.pixel_centers(), component_chunk=1).color.reshape(
        height, width, 3
    )
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_crop_translation_preserves_native_render_and_clipping():
    from structsplat.render import render

    field = _native_field()
    crop_height, crop_width = 4, 5
    fit_window = (2, 3, crop_width, crop_height)
    expected = render(
        field.means,
        field.conics(),
        field.colors,
        field.radii(3.0),
        crop_height,
        crop_width,
        opacities=field.opacity_values(),
        color_grads=field.color_grads,
        scales=field.effective_scales(),
        rotations=field.rotations,
    )
    observation = field_to_observation(
        field,
        canvas_size=(9, 10),
        fit_window=fit_window,
        view_id="crop-view",
        n_init=2,
    )
    actual = observation.query(observation.pixel_centers(), component_chunk=2).color.reshape(
        crop_height, crop_width, 3
    )
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    assert torch.equal(
        observation.means,
        field.means + field.means.new_tensor([fit_window[0] + 0.5, fit_window[1] + 0.5]),
    )
    outside = observation.query(torch.tensor([[1.5, 3.5]], dtype=torch.float64))
    assert not bool(outside.valid.item())
    assert torch.equal(outside.color, torch.zeros_like(outside.color))


def test_odd_crop_translation_preserves_half_tie_support_rounding():
    from structsplat.gaussians import GaussianField
    from structsplat.render import render

    field = GaussianField(
        means=torch.tensor([[0.5, 0.5], [1.5, 2.5]], dtype=torch.float64),
        log_scales=torch.log(torch.tensor([[0.45, 0.55], [0.6, 0.4]], dtype=torch.float64)),
        rotations=torch.tensor([0.0, 0.4], dtype=torch.float64),
        colors=torch.tensor([[0.9, 0.2, 0.1], [0.1, 0.4, 0.8]], dtype=torch.float64),
        opacities=torch.tensor([1.1, 0.7], dtype=torch.float64),
    )
    crop_height, crop_width = 4, 5
    fit_window = (1, 2, crop_width, crop_height)
    expected = render(
        field.means,
        field.conics(),
        field.colors,
        field.radii(3.0),
        crop_height,
        crop_width,
        opacities=field.opacity_values(),
    )
    observation = field_to_observation(
        field,
        canvas_size=(9, 10),
        fit_window=fit_window,
    )
    actual = observation.query(
        observation.pixel_centers(),
        component_chunk=1,
    ).color.reshape(crop_height, crop_width, 3)

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
    expected_pixels = torch.round(field.means).long() + torch.tensor(fit_window[:2])
    assert torch.equal(observation.support_pixels(), expected_pixels)
    assert not torch.equal(
        torch.round(observation.means - 0.5).long(),
        expected_pixels,
    )


def test_float32_native_crop_residual_preserves_provider_query_and_support_boundary():
    from structsplat.gaussians import GaussianField
    from structsplat.render import render

    half = torch.tensor(0.5)
    field = GaussianField(
        means=torch.stack(
            [
                torch.stack([torch.nextafter(half, torch.tensor(torch.inf)), half]),
                torch.tensor([2.2, 1.4]),
            ]
        ),
        log_scales=torch.log(torch.tensor([[0.55, 0.7], [0.8, 0.6]])),
        rotations=torch.tensor([0.2, -0.4]),
        colors=torch.tensor([[0.9, 0.2, 0.1], [0.1, 0.4, 0.8]]),
        opacities=torch.tensor([1.1, 0.7]),
    )
    crop_height, crop_width = 3, 4
    fit_window = (1282, 1936, crop_width, crop_height)
    expected = render(
        field.means,
        field.conics(),
        field.colors,
        field.radii(3.0),
        crop_height,
        crop_width,
        opacities=field.opacity_values(),
    )
    observation = field_to_observation(
        field,
        canvas_size=(4608, 5328),
        fit_window=fit_window,
    )
    actual = observation.query(
        observation.pixel_centers(),
        component_chunk=1,
    ).color.reshape(crop_height, crop_width, 3)

    assert observation.mean_residuals is not None
    assert bool((observation.mean_residuals != 0).any())
    assert torch.equal(observation.local_means(), field.means)
    assert observation.support_pixels()[0, 0].item() == fit_window[0] + 1
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)

    reconstructed = GaussianField(
        observation.local_means(),
        observation.log_scales,
        observation.rotations,
        observation.colors,
        opacities=field.opacities,
    )
    reconstructed_render = render(
        reconstructed.means,
        reconstructed.conics(),
        reconstructed.colors,
        reconstructed.radii(3.0),
        crop_height,
        crop_width,
        opacities=reconstructed.opacity_values(),
    )
    assert torch.equal(reconstructed_render, expected)


def test_production_fit_callback_exports_reloadable_rgb_free_teacher(tmp_path, monkeypatch):
    from PIL import Image as PILImage

    from rtgs.core.observation2d import GaussianObservationField
    from rtgs.image2gs.fit import FitConfig, fit_image

    image = torch.rand(16, 16, 3, generator=torch.Generator().manual_seed(3))
    observations = []
    _, history = fit_image(
        image,
        FitConfig(
            n_gaussians=16,
            max_gaussians=16,
            iterations=1,
            backend="structsplat",
            adaptive_density=False,
            structsplat_renderer="normalized",
        ),
        seed=5,
        observation_callback=observations.append,
        observation_view_id="smoke-view",
    )
    assert history["observation_exported"]
    assert len(observations) == 1
    path = tmp_path / "smoke.teacher.npz"
    observations[0].save_npz(path)

    def forbidden_rgb_open(*_args, **_kwargs):
        raise AssertionError("RGB loading is forbidden after teacher export")

    monkeypatch.setattr(PILImage, "open", forbidden_rgb_open)
    loaded = GaussianObservationField.load_npz(path)
    query = loaded.query(loaded.pixel_centers())
    assert loaded.view_id == "smoke-view"
    assert loaded.n_init == 16
    assert loaded.n == 16
    assert loaded.producer_source_digest is not None
    assert loaded.fit_config_digest is not None
    assert torch.isfinite(query.color).all()
