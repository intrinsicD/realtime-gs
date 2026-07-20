"""Correctness tests for compact RGB-free 2D Gaussian observations."""

import json

import numpy as np
import pytest
import torch

import rtgs.core.observation2d as observation2d
from rtgs.core.observation2d import (
    GaussianObservationField,
    GaussianObservationIndex,
    GaussianPixelProposal,
    GaussianPointProposal,
    fixed_attempt_mean,
)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize(
    "fit_window",
    ((0, 0, 5328, 4608), (1282, 1936, 3599, 2225)),
)
def test_continuous_uniform_affine_is_strictly_half_open_at_native_scale(dtype, fit_window):
    zero = torch.tensor(0.0, dtype=dtype)
    one = torch.tensor(1.0, dtype=dtype)
    largest_below_one = torch.nextafter(one, zero)
    unit = torch.stack(
        [
            torch.stack([zero, zero]),
            torch.stack([largest_below_one, largest_below_one]),
            torch.stack([largest_below_one, zero]),
            torch.stack([zero, largest_below_one]),
        ]
    )
    xy = observation2d._half_open_uniform_xy(unit, fit_window)
    lower = torch.tensor(fit_window[:2], dtype=dtype)
    upper = lower + torch.tensor(fit_window[2:], dtype=dtype)
    assert bool(torch.isfinite(xy).all())
    assert bool((xy >= lower).all())
    assert bool((xy < upper).all())
    assert torch.equal(
        xy[1],
        torch.nextafter(upper, lower),
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_mixed_continuous_proposal_keeps_endpoint_uniform_draws_bounded(dtype, monkeypatch):
    fit_window = (1282, 1936, 3599, 2225)
    field = GaussianObservationField(
        width=5328,
        height=4608,
        means=torch.tensor([[3000.5, 3000.5]], dtype=dtype),
        log_scales=torch.zeros(1, 2, dtype=dtype),
        rotations=torch.zeros(1, dtype=dtype),
        colors=torch.ones(1, 3, dtype=dtype),
        amplitudes=torch.ones(1, dtype=dtype),
        fit_window=fit_window,
    )
    original_rand = torch.rand
    calls = 0

    def forced_rand(*shape, **kwargs):
        nonlocal calls
        calls += 1
        target_dtype = kwargs.get("dtype", torch.get_default_dtype())
        target_device = kwargs.get("device", "cpu")
        if calls == 1:
            return torch.tensor(
                [0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9],
                dtype=target_dtype,
                device=target_device,
            )
        if calls == 2:
            one = torch.tensor(1.0, dtype=target_dtype, device=target_device)
            zero = torch.tensor(0.0, dtype=target_dtype, device=target_device)
            return torch.nextafter(one, zero).expand(*shape).clone()
        if calls == 3:
            return torch.zeros(*shape, dtype=target_dtype, device=target_device)
        return original_rand(*shape, **kwargs)

    monkeypatch.setattr(torch, "rand", forced_rand)
    eta = 0.25
    samples = GaussianPointProposal(field).sample(
        8,
        uniform_fraction=eta,
        generator=torch.Generator().manual_seed(991601),
    )
    uniform = samples.proposal_component_ids == -1
    upper = torch.tensor(
        [fit_window[0] + fit_window[2], fit_window[1] + fit_window[3]],
        dtype=dtype,
    )
    assert int(uniform.sum()) == 4
    assert bool(samples.active[uniform].all())
    assert bool(samples.inside_fit_window[uniform].all())
    assert bool((samples.xy[uniform] < upper).all())
    assert bool(torch.isfinite(samples.proposal_density).all())
    assert bool(torch.isfinite(samples.importance).all())
    assert float(samples.importance.max()) <= 1.0 / eta + 1e-6


def _field(
    *,
    means: torch.Tensor | None = None,
    colors: torch.Tensor | None = None,
    amplitudes: torch.Tensor | None = None,
    log_scales: torch.Tensor | None = None,
    rotations: torch.Tensor | None = None,
    mean_residuals: torch.Tensor | None = None,
    color_grads: torch.Tensor | None = None,
    filter_variance: torch.Tensor | None = None,
    epsilon: float = 1e-8,
    support_fade_alpha: float = 0.0,
    aa_dilation: float = 0.0,
    fit_window: tuple[int, int, int, int] | None = None,
    n_init: int | None = None,
    provider: str = "structsplat",
    producer_version: str | None = None,
    producer_source_digest: str | None = None,
    fit_config_digest: str | None = None,
) -> GaussianObservationField:
    means = torch.tensor([[3.5, 3.5], [5.5, 3.5]]) if means is None else means
    n = means.shape[0]
    colors = torch.tensor([[1.4, -0.2, 0.3], [0.1, 0.8, 1.2]]) if colors is None else colors
    amplitudes = torch.tensor([0.7, 0.4]) if amplitudes is None else amplitudes
    return GaussianObservationField(
        width=9,
        height=7,
        means=means,
        log_scales=(
            torch.log(torch.tensor([[1.5, 1.0]]).repeat(n, 1)) if log_scales is None else log_scales
        ),
        rotations=torch.tensor([0.2]).repeat(n) if rotations is None else rotations,
        colors=colors,
        amplitudes=amplitudes,
        mean_residuals=mean_residuals,
        color_grads=color_grads,
        filter_variance=filter_variance,
        epsilon=epsilon,
        sigma_cutoff=3.0,
        support_fade_alpha=support_fade_alpha,
        aa_dilation=aa_dilation,
        view_id="view-A",
        fit_window=fit_window,
        n_init=n_init,
        provider=provider,
        producer_version=producer_version,
        producer_source_digest=producer_source_digest,
        fit_config_digest=fit_config_digest,
    )


def test_normalized_query_keeps_unbounded_colors_and_returns_density():
    field = _field(
        means=torch.tensor([[3.5, 3.5]]),
        colors=torch.tensor([[1.4, -0.2, 0.3]]),
        amplitudes=torch.tensor([0.7]),
    )
    query = field.query(torch.tensor([[3.5, 3.5]]))
    assert torch.allclose(query.weight_sum, torch.tensor([0.7]))
    assert torch.allclose(query.numerator, torch.tensor([[0.98, -0.14, 0.21]]))
    assert torch.allclose(query.color, field.colors, atol=1e-6)
    assert query.color[0, 0] > 1.0
    assert query.color[0, 1] < 0.0


@pytest.mark.parametrize(
    ("log_scale", "message"),
    [(100.0, "derived observation tensors must be finite"), (-1000.0, "scales must be positive")],
)
def test_derived_scale_overflow_and_underflow_fail_closed(log_scale, message):
    with pytest.raises(ValueError, match=message):
        _field(
            means=torch.tensor([[3.5, 3.5]]),
            colors=torch.ones(1, 3),
            amplitudes=torch.ones(1),
            log_scales=torch.full((1, 2), log_scale),
        )


def test_query_matches_manual_normalized_gaussian_equation():
    field = _field(epsilon=0.25)
    xy = torch.tensor([[4.5, 3.5]])
    dx = xy[:, None, :] - field.means[None, :, :]
    conics = field.conics()
    q = (
        conics[None, :, 0] * dx[..., 0].square()
        + 2 * conics[None, :, 1] * dx[..., 0] * dx[..., 1]
        + conics[None, :, 2] * dx[..., 1].square()
    )
    weights = torch.exp(-0.5 * q) * field.amplitudes
    expected_num = (weights[..., None] * field.colors).sum(dim=1)
    expected = expected_num / (weights.sum(dim=1, keepdim=True) + field.epsilon)
    actual = field.query(xy)
    assert torch.allclose(actual.numerator, expected_num)
    assert torch.allclose(actual.color, expected)


def test_query_uses_clipped_support_and_zero_outside_image():
    field = _field(
        means=torch.tensor([[-0.25, 1.5]]),
        colors=torch.ones(1, 3),
        amplitudes=torch.ones(1),
    )
    inside = field.query(torch.tensor([[0.5, 1.5]]))
    outside = field.query(torch.tensor([[-0.5, 1.5], [9.5, 1.5]]))
    assert inside.weight_sum.item() > 0
    assert torch.equal(outside.weight_sum, torch.zeros(2))
    assert torch.equal(outside.color, torch.zeros(2, 3))


def test_crop_domain_and_patch_queries_share_the_point_equation():
    field = _field(fit_window=(2, 1, 5, 4))
    patch = field.query_patch((1.5, 0.5), (3, 4), component_chunk=1)
    y, x = torch.meshgrid(
        torch.arange(3, dtype=field.dtype) + 0.5,
        torch.arange(4, dtype=field.dtype) + 1.5,
        indexing="ij",
    )
    points = torch.stack([x, y], dim=-1).reshape(-1, 2)
    direct = field.query(points, component_chunk=2)
    assert torch.equal(patch.color.reshape(-1, 3), direct.color)
    assert torch.equal(patch.weight_sum.reshape(-1), direct.weight_sum)
    assert torch.equal(patch.valid.reshape(-1), direct.valid)
    assert not bool(patch.valid[0, 0])
    assert bool(patch.valid[1, 1])


def test_query_is_chunk_and_component_order_invariant_and_xy_differentiable():
    field = _field()
    xy = torch.tensor([[3.25, 3.1], [4.75, 3.8]], requires_grad=True)
    reference = field.query(xy, component_chunk=1)
    permuted = GaussianObservationField(
        width=field.width,
        height=field.height,
        means=field.means.flip(0),
        log_scales=field.log_scales.flip(0),
        rotations=field.rotations.flip(0),
        colors=field.colors.flip(0),
        amplitudes=field.amplitudes.flip(0),
    )
    actual = permuted.query(xy, component_chunk=8)
    assert torch.allclose(actual.color, reference.color, atol=1e-7)
    assert torch.allclose(actual.weight_sum, reference.weight_sum, atol=1e-7)
    gradient = torch.autograd.grad(reference.color.square().sum(), xy)[0]
    assert torch.isfinite(gradient).all()


def test_sparse_tile_index_matches_reference_query_and_proposal_density():
    field = _field(
        support_fade_alpha=0.4,
        color_grads=torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) / 20.0,
        filter_variance=torch.tensor([0.0, 0.3]),
        fit_window=(1, 1, 7, 5),
    )
    generator = torch.Generator().manual_seed(29)
    points = torch.rand(80, 2, generator=generator)
    points[:, 0] = points[:, 0] * 11.0 - 1.0
    points[:, 1] = points[:, 1] * 9.0 - 1.0
    reference = field.query(points, component_chunk=1)
    for tile_size in (1, 2, 16):
        index = GaussianObservationIndex(field, tile_size=tile_size)
        actual = index.query(points, component_chunk=1)
        assert torch.allclose(actual.color, reference.color, atol=1e-7)
        assert torch.allclose(actual.numerator, reference.numerator, atol=1e-7)
        assert torch.allclose(actual.weight_sum, reference.weight_sum, atol=1e-7)
        assert torch.equal(index.query_weight_sum(points), reference.weight_sum)
        assert index.n_tiles <= index.n_entries
        indexed_proposal = GaussianPointProposal(field, query_backend=index)
        dense_proposal = GaussianPointProposal(field)
        assert torch.equal(
            indexed_proposal.gaussian_density(points),
            dense_proposal.gaussian_density(points),
        )


def test_tile_index_exact_preflight_caps_and_stats():
    field = _field(fit_window=(1, 1, 7, 5))
    estimated = GaussianObservationIndex.estimate_entries(field, tile_size=2)
    assert estimated > 0

    index = GaussianObservationIndex(field, tile_size=2)
    assert index.estimated_entries == estimated == index.n_entries
    assert index.stats.total_entries == estimated
    assert index.stats.nonempty_tiles == index.n_tiles
    assert index.stats.max_candidates == index.max_candidates
    assert index.stats.tiles_x == 5
    assert index.stats.tiles_y == 4

    with pytest.raises(ValueError, match="entry cap exceeded before allocation"):
        GaussianObservationIndex(field, tile_size=2, max_entries=estimated - 1)
    with pytest.raises(ValueError, match="candidate cap exceeded"):
        GaussianObservationIndex(
            field,
            tile_size=2,
            max_entries=estimated,
            max_candidates=index.max_candidates - 1,
        )


def test_affine_component_color_uses_structsplat_effective_local_axes():
    field = _field(
        means=torch.tensor([[3.5, 3.5]]),
        colors=torch.zeros(1, 3),
        amplitudes=torch.ones(1),
        rotations=torch.zeros(1),
        color_grads=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]),
    )
    colors = field.component_color(torch.tensor([[5.0, 4.0]]), torch.tensor([0], dtype=torch.long))
    assert torch.allclose(colors, torch.tensor([[1.0, 1.0, 0.0]]))

    filtered = _field(
        means=torch.tensor([[3.5, 3.5]]),
        colors=torch.zeros(1, 3),
        amplitudes=torch.ones(1),
        rotations=torch.zeros(1),
        color_grads=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]),
        filter_variance=torch.tensor([1.75]),
    )
    filtered_color = filtered.component_color(
        torch.tensor([[5.5, 3.5]]), torch.tensor([0], dtype=torch.long)
    )
    assert torch.allclose(filtered_color, torch.tensor([[1.0, 0.0, 0.0]]))


def test_exact_opacity_mass_split_preserves_field_and_proposal():
    parent = _field(
        means=torch.tensor([[3.5, 3.5]]),
        colors=torch.tensor([[0.2, 0.6, 1.1]]),
        amplitudes=torch.tensor([0.8]),
        support_fade_alpha=1.0,
    )
    children = GaussianObservationField(
        width=parent.width,
        height=parent.height,
        means=parent.means.repeat(2, 1),
        log_scales=parent.log_scales.repeat(2, 1),
        rotations=parent.rotations.repeat(2),
        colors=parent.colors.repeat(2, 1),
        amplitudes=torch.tensor([0.2, 0.6]),
        epsilon=parent.epsilon,
        sigma_cutoff=parent.sigma_cutoff,
        support_fade_alpha=parent.support_fade_alpha,
    )
    pixels = parent.pixel_centers()
    parent_query = parent.query(pixels, component_chunk=1)
    child_query = children.query(pixels, component_chunk=1)
    assert torch.allclose(child_query.weight_sum, parent_query.weight_sum, atol=1e-7)
    assert torch.allclose(child_query.numerator, parent_query.numerator, atol=1e-7)
    assert torch.allclose(child_query.color, parent_query.color, atol=1e-7)

    parent_proposal = GaussianPointProposal(parent)
    child_proposal = GaussianPointProposal(children)
    assert torch.allclose(child_proposal.total_mass, parent_proposal.total_mass, atol=1e-7)
    assert torch.allclose(
        child_proposal.gaussian_density(pixels),
        parent_proposal.gaussian_density(pixels),
        atol=1e-7,
    )


def test_unscaled_duplication_is_a_negative_control_when_epsilon_is_nonzero():
    parent = _field(
        means=torch.tensor([[3.5, 3.5]]),
        colors=torch.tensor([[0.4, 0.2, 0.8]]),
        amplitudes=torch.tensor([0.5]),
        epsilon=0.25,
    )
    duplicate = GaussianObservationField(
        width=parent.width,
        height=parent.height,
        means=parent.means.repeat(2, 1),
        log_scales=parent.log_scales.repeat(2, 1),
        rotations=parent.rotations.repeat(2),
        colors=parent.colors.repeat(2, 1),
        amplitudes=parent.amplitudes.repeat(2),
        epsilon=parent.epsilon,
    )
    xy = torch.tensor([[3.5, 3.5]])
    assert not torch.allclose(parent.query(xy).color, duplicate.query(xy).color)


def test_mixture_sampling_is_seeded_and_has_bounded_importance():
    proposal = GaussianPointProposal(_field())
    first = proposal.sample(256, uniform_fraction=0.25, generator=torch.Generator().manual_seed(17))
    second = proposal.sample(
        256, uniform_fraction=0.25, generator=torch.Generator().manual_seed(17)
    )
    assert torch.equal(first.xy, second.xy)
    assert torch.equal(first.proposal_component_ids, second.proposal_component_ids)
    assert torch.equal(first.proposal_density, second.proposal_density)
    assert torch.equal(first.joint_density, second.joint_density)
    assert torch.equal(first.inside_fit_window, second.inside_fit_window)
    assert torch.equal(first.active, second.active)
    assert bool((first.proposal_component_ids == -1).any())
    assert bool((first.proposal_component_ids >= 0).any())
    assert torch.isfinite(first.importance).all()
    assert float(first.importance.max()) <= 4.0 + 1e-5

    eta = 0.25
    fit_width, fit_height = proposal.field.fit_window[2:]
    uniform = first.proposal_component_ids == -1
    assert torch.allclose(
        first.joint_density[uniform],
        torch.full_like(first.joint_density[uniform], eta / (fit_width * fit_height)),
    )
    gaussian = ~uniform
    ids = first.proposal_component_ids[gaussian]
    exact_weight = proposal.field.component_weight(first.xy[gaussian], ids)
    expected_joint = torch.where(
        first.active[gaussian],
        (1.0 - eta) * exact_weight / proposal.total_mass,
        torch.zeros_like(exact_weight),
    )
    assert torch.allclose(first.joint_density[gaussian], expected_joint, atol=1e-7)
    uniform_at_xy = first.inside_fit_window.to(first.xy.dtype) / (fit_width * fit_height)
    expected_marginal = eta * uniform_at_xy + (1.0 - eta) * proposal.gaussian_density(first.xy)
    assert torch.allclose(first.proposal_density, expected_marginal, atol=1e-7)
    assert first.risk_measure == "continuous_area"
    losses = torch.arange(first.xy.shape[0], dtype=first.xy.dtype)
    expected_fixed_attempt = (losses * first.importance).sum() / first.xy.shape[0]
    assert torch.equal(fixed_attempt_mean(losses, first), expected_fixed_attempt)


def test_continuous_gaussian_proposal_has_analytic_density_and_o_n_state():
    field = _field(
        means=torch.tensor([[3.5, 3.5]]),
        colors=torch.ones(1, 3),
        amplitudes=torch.tensor([0.7]),
        filter_variance=torch.tensor([0.4]),
        aa_dilation=0.2,
    )
    proposal = GaussianPointProposal(field)
    determinant_sqrt = field.effective_variances().prod(1).sqrt()
    expected = 1.0 / (2.0 * torch.pi * determinant_sqrt)
    assert torch.allclose(proposal.gaussian_density(field.means), expected)
    assert proposal.component_masses.numel() == field.n
    assert not any(
        tensor.ndim >= 2 and tensor.shape[:2] == (field.height, field.width)
        for tensor in vars(proposal).values()
        if isinstance(tensor, torch.Tensor)
    )


def test_discrete_gaussian_proposal_has_clipped_o_n_rectangle_state():
    field = _field(fit_window=(1, 1, 7, 5))
    proposal = GaussianPixelProposal(field)

    support_pixels = field.support_pixels()
    fit_lower = torch.tensor([1, 1])
    fit_upper = torch.tensor([7, 5])
    expected_lower = torch.maximum(support_pixels - field.radii(), fit_lower)
    expected_upper = torch.minimum(support_pixels + field.radii(), fit_upper)
    expected_sizes = (expected_upper - expected_lower + 1).clamp_min(0)
    expected_areas = expected_sizes.prod(dim=1)

    assert torch.equal(proposal.rectangle_lower, expected_lower)
    assert torch.equal(proposal.rectangle_sizes, expected_sizes)
    assert torch.equal(proposal.component_areas, expected_areas)
    assert torch.equal(
        proposal.envelope_mass,
        (field.amplitudes * expected_areas.to(field.dtype)).sum(),
    )
    assert proposal.pixel_count == 35
    assert all(
        tensor.numel() <= 2 * field.n
        for tensor in vars(proposal).values()
        if isinstance(tensor, torch.Tensor) and tensor.ndim > 0
    )


def test_support_rounding_occurs_in_translated_fit_coordinates():
    fit_window = (1, 2, 4, 3)
    local_means = torch.tensor([[0.5, 0.5], [1.5, 1.5], [2.5, 0.5]])
    origin = torch.tensor([fit_window[0] + 0.5, fit_window[1] + 0.5])
    field = _field(
        means=local_means + origin,
        colors=torch.ones(3, 3),
        amplitudes=torch.ones(3),
        fit_window=fit_window,
    )

    expected_pixels = torch.round(local_means).long() + torch.tensor(fit_window[:2])
    assert torch.equal(field.support_pixels(), expected_pixels)
    assert torch.equal(
        field.support_centers(),
        expected_pixels.to(field.dtype) + 0.5,
    )

    # Rounding after translation is wrong for two of these half ties because torch.round uses
    # ties-to-even and the odd x crop offset changes which integer is even.
    legacy_pixels = torch.round(field.means - 0.5).long()
    assert not torch.equal(legacy_pixels, expected_pixels)


def test_mean_residual_recovers_exact_local_float32_and_near_half_support():
    fit_window = (1282, 1936, 4, 3)
    origin = torch.tensor([fit_window[0] + 0.5, fit_window[1] + 0.5])
    half = torch.tensor(0.5)
    local = torch.stack(
        [
            torch.stack([torch.nextafter(half, torch.tensor(torch.inf)), half]),
            torch.tensor([1.125123, 1.875321]),
        ]
    )
    native = local + origin
    residuals = local - (native - origin)
    residuals_source = residuals.clone()
    field = GaussianObservationField(
        width=5328,
        height=4608,
        means=native,
        log_scales=torch.zeros(2, 2),
        rotations=torch.zeros(2),
        colors=torch.ones(2, 3),
        amplitudes=torch.ones(2),
        mean_residuals=residuals_source,
        fit_window=fit_window,
    )
    residuals_source.zero_()

    assert torch.equal(field.local_means(), local)
    assert torch.equal(field.native_means(), native)
    assert torch.equal(
        field.native_means(dtype=torch.float64),
        local.double() + origin.double(),
    )
    assert field.support_pixels()[0, 0].item() == fit_window[0] + 1
    assert torch.round(native[0, 0] - origin[0]).item() == 0
    assert bool((field.mean_residuals != 0).any())
    assert field.to("cpu").mean_residuals.data_ptr() != field.mean_residuals.data_ptr()

    with pytest.raises(TypeError, match="mean_residuals must use float32"):
        GaussianObservationField(
            width=5328,
            height=4608,
            means=native,
            log_scales=torch.zeros(2, 2),
            rotations=torch.zeros(2),
            colors=torch.ones(2, 3),
            amplitudes=torch.ones(2),
            mean_residuals=residuals.double(),
            fit_window=fit_window,
        )
    with pytest.raises(ValueError, match=r"mean_residuals must have shape \(N,2\)"):
        GaussianObservationField(
            width=5328,
            height=4608,
            means=native,
            log_scales=torch.zeros(2, 2),
            rotations=torch.zeros(2),
            colors=torch.ones(2, 3),
            amplitudes=torch.ones(2),
            mean_residuals=torch.zeros(2, 1),
            fit_window=fit_window,
        )


def test_residual_corrected_crop_geometry_matches_local_query_index_and_pixel_proposal():
    fit_window = (1282, 1936, 4, 3)
    local = torch.tensor(
        [
            [torch.nextafter(torch.tensor(0.5), torch.tensor(torch.inf)), 0.5],
            [2.2, 1.4],
        ]
    )
    native_origin = torch.tensor([fit_window[0] + 0.5, fit_window[1] + 0.5])
    native = local + native_origin
    residuals = local - (native - native_origin)
    kwargs = {
        "log_scales": torch.log(torch.tensor([[0.55, 0.7], [0.8, 0.6]])),
        "rotations": torch.tensor([0.2, -0.4]),
        "colors": torch.tensor([[0.9, 0.2, 0.1], [0.1, 0.4, 0.8]]),
        "amplitudes": torch.tensor([0.8, 0.6]),
    }
    cropped = GaussianObservationField(
        width=5328,
        height=4608,
        means=native,
        mean_residuals=residuals,
        fit_window=fit_window,
        **kwargs,
    )
    local_native = local + 0.5
    local_field = GaussianObservationField(
        width=fit_window[2],
        height=fit_window[3],
        means=local_native,
        mean_residuals=local - (local_native - 0.5),
        **kwargs,
    )
    native_pixels = cropped.pixel_centers()
    local_pixels = local_field.pixel_centers()
    expected = local_field.query(local_pixels, component_chunk=1)
    actual = cropped.query(native_pixels, component_chunk=1)
    torch.testing.assert_close(actual.color, expected.color, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(actual.weight_sum, expected.weight_sum, atol=1e-7, rtol=1e-7)
    assert torch.equal(
        cropped.support_pixels() - torch.tensor(fit_window[:2]),
        local_field.support_pixels(),
    )

    indexed = GaussianObservationIndex(cropped, tile_size=2)
    torch.testing.assert_close(
        indexed.query(native_pixels, component_chunk=1).color,
        expected.color,
        atol=1e-7,
        rtol=1e-7,
    )
    cropped_samples = GaussianPixelProposal(cropped).sample(
        256,
        uniform_fraction=0.25,
        generator=torch.Generator().manual_seed(404),
    )
    local_samples = GaussianPixelProposal(local_field).sample(
        256,
        uniform_fraction=0.25,
        generator=torch.Generator().manual_seed(404),
    )
    assert torch.equal(
        cropped_samples.xy - torch.tensor(fit_window[:2]),
        local_samples.xy,
    )
    assert torch.equal(cropped_samples.proposal_density, local_samples.proposal_density)
    assert torch.equal(cropped_samples.active, local_samples.active)


def test_discrete_gaussian_sampling_matches_exact_marginal_and_keeps_null_attempts():
    field = _field(support_fade_alpha=0.4, fit_window=(1, 1, 7, 5))
    proposal = GaussianPixelProposal(field)
    eta = 0.25
    first = proposal.sample(
        4096,
        uniform_fraction=eta,
        generator=torch.Generator().manual_seed(424201),
    )
    second = proposal.sample(
        4096,
        uniform_fraction=eta,
        generator=torch.Generator().manual_seed(424201),
    )

    assert torch.equal(first.xy, second.xy)
    assert torch.equal(first.proposal_component_ids, second.proposal_component_ids)
    assert torch.equal(first.active, second.active)
    assert torch.equal(first.proposal_density, second.proposal_density)
    assert torch.equal(first.joint_density, second.joint_density)
    assert torch.equal(first.xy - 0.5, (first.xy - 0.5).round())
    assert bool(first.inside_fit_window.all())
    assert first.risk_measure == "discrete_pixels"

    uniform = first.proposal_component_ids == -1
    gaussian = ~uniform
    rejected = gaussian & ~first.active
    assert bool(uniform.any())
    assert bool(gaussian.any())
    assert bool(rejected.any())
    assert bool((gaussian & first.active).any())

    expected_marginal = eta / proposal.pixel_count + (
        (1.0 - eta) * field.query_weight_sum(first.xy) / proposal.envelope_mass
    )
    assert torch.allclose(first.proposal_density, expected_marginal, atol=1e-7)
    assert torch.allclose(
        first.joint_density[uniform],
        torch.full_like(first.joint_density[uniform], eta / proposal.pixel_count),
    )
    component_ids = first.proposal_component_ids[gaussian]
    exact_weight = field.component_weight(first.xy[gaussian], component_ids)
    expected_joint = torch.where(
        first.active[gaussian],
        (1.0 - eta) * exact_weight / proposal.envelope_mass,
        torch.zeros_like(exact_weight),
    )
    assert torch.allclose(first.joint_density[gaussian], expected_joint, atol=1e-7)
    assert torch.equal(
        first.target_density,
        first.active.to(field.dtype) / proposal.pixel_count,
    )
    assert torch.equal(
        first.importance[~first.active], torch.zeros_like(first.importance[~first.active])
    )
    assert float(first.importance.max()) <= 1.0 / eta + 1e-6

    losses = torch.linspace(0.0, 1.0, first.xy.shape[0], dtype=field.dtype)
    expected = (losses * first.importance).sum() / first.xy.shape[0]
    assert torch.equal(fixed_attempt_mean(losses, first), expected)


def test_discrete_gaussian_proposal_exact_finite_risk_identity():
    field = _field(support_fade_alpha=0.6, fit_window=(1, 1, 7, 5))
    proposal = GaussianPixelProposal(field)
    pixels = field.pixel_centers()
    eta = 0.2
    target_probability = torch.full(
        (pixels.shape[0],),
        1.0 / proposal.pixel_count,
        dtype=field.dtype,
    )
    proposal_probability = eta / proposal.pixel_count + (
        (1.0 - eta) * proposal.gaussian_probability(pixels)
    )
    losses = torch.linspace(0.05, 1.25, pixels.shape[0], dtype=field.dtype).square()

    exact = losses.mean()
    enumerated_expectation = (
        proposal_probability * target_probability / proposal_probability * losses
    ).sum()
    assert torch.allclose(enumerated_expectation, exact, atol=1e-7)
    assert float(proposal_probability.max() - proposal_probability.min()) > 0.0
    assert float(1.0 - proposal_probability.sum()) > 0.0


def test_discrete_gaussian_proposal_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="no positive discrete"):
        GaussianPixelProposal(_field(amplitudes=torch.zeros(2)))

    proposal = GaussianPixelProposal(_field())
    generator = torch.Generator().manual_seed(424202)
    with pytest.raises(ValueError, match="count must be positive"):
        proposal.sample(0, uniform_fraction=0.2, generator=generator)
    with pytest.raises(ValueError, match="uniform_fraction"):
        proposal.sample(1, uniform_fraction=0.0, generator=generator)
    with pytest.raises(ValueError, match="pixel-center"):
        proposal.gaussian_probability(torch.tensor([[3.25, 3.5]]))


def test_virtual_background_label_gradient_matches_exact_target_with_epsilon():
    field = _field(
        means=torch.tensor([[3.5, 3.5], [3.5, 3.5]]),
        colors=torch.tensor([[1.0, 0.0, 0.25], [0.0, 1.0, 0.75]]),
        amplitudes=torch.tensor([0.25, 0.75]),
        epsilon=0.25,
    )
    xy = torch.tensor([[3.5, 3.5]])
    weights = field.component_weight(xy.repeat(field.n, 1), torch.arange(field.n))
    denominator = weights.sum() + field.epsilon
    target = field.query(xy).color
    prediction = torch.tensor([[0.2, 0.3, 0.4]], requires_grad=True)
    expected_loss = sum(
        (weights[index] / denominator) * (prediction - field.colors[index]).square().sum()
        for index in range(field.n)
    )
    expected_loss = expected_loss + (field.epsilon / denominator) * prediction.square().sum()
    expected_grad = torch.autograd.grad(expected_loss, prediction, retain_graph=True)[0]
    exact_grad = torch.autograd.grad((prediction - target).square().sum(), prediction)[0]
    assert torch.allclose(expected_grad, exact_grad, atol=1e-7)

    uncorrected_loss = sum(
        (weights[index] / weights.sum()) * (prediction - field.colors[index]).square().sum()
        for index in range(field.n)
    )
    uncorrected_grad = torch.autograd.grad(uncorrected_loss, prediction)[0]
    assert not torch.allclose(uncorrected_grad, exact_grad)


def test_lossless_npz_round_trip_preserves_semantics_and_contains_no_rgb_path(tmp_path):
    fit_window = (1, 1, 7, 5)
    local_means = torch.tensor([[2.125123, 2.875321], [4.5, 2.5]])
    origin = torch.tensor([fit_window[0] + 0.5, fit_window[1] + 0.5])
    native_means = local_means + origin
    field = _field(
        means=native_means,
        mean_residuals=local_means - (native_means - origin),
        support_fade_alpha=0.4,
        color_grads=torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
        filter_variance=torch.tensor([0.0, 0.25]),
        aa_dilation=0.125,
        fit_window=fit_window,
        n_init=1,
    )
    path = tmp_path / "observation.npz"
    field.save_npz(path)
    with pytest.raises(FileExistsError):
        field.save_npz(path)
    loaded = GaussianObservationField.load_npz(path)
    for name in (
        "means",
        "log_scales",
        "rotations",
        "colors",
        "amplitudes",
        "mean_residuals",
        "color_grads",
        "filter_variance",
    ):
        assert torch.equal(getattr(loaded, name), getattr(field, name))
    assert loaded.width == field.width
    assert loaded.height == field.height
    assert loaded.view_id == field.view_id
    assert loaded.support_fade_alpha == field.support_fade_alpha
    assert loaded.aa_dilation == field.aa_dilation
    assert loaded.fit_window == field.fit_window
    assert loaded.n_init == field.n_init
    assert torch.equal(
        loaded.query(field.pixel_centers()).color,
        field.query(field.pixel_centers()).color,
    )

    with np.load(path, allow_pickle=False) as archive:
        assert not {"rgb", "image", "image_path", "source_path"} & set(archive.files)
        metadata = json.loads(archive["metadata_utf8"].tobytes())
        assert not {"rgb", "image", "image_path", "source_path"} & set(metadata)
        assert metadata["arrays"]["mean_residuals"]["dtype"] == "<f4"
        assert len(metadata["arrays"]["mean_residuals"]["sha256"]) == 64


def test_residual_free_schema_v1_archive_loads_with_identical_legacy_semantics(tmp_path):
    field = _field(fit_window=(1, 1, 7, 5))
    expected = field.query(field.pixel_centers()).color
    path = tmp_path / "legacy-v1.teacher.npz"
    field.save_npz(path)

    with np.load(path, allow_pickle=False) as archive:
        assert "mean_residuals" not in archive.files
    loaded = GaussianObservationField.load_npz(path, strict=True)
    assert loaded.mean_residuals is None
    assert torch.equal(loaded.means, field.means)
    assert torch.equal(loaded.local_means(), field.means - torch.tensor([1.5, 1.5]))
    assert torch.equal(loaded.native_means(), field.means)
    assert torch.equal(loaded.query(field.pixel_centers()).color, expected)


def test_synthetic_provider_and_provenance_strictly_round_trip(tmp_path):
    field = _field(
        provider="synthetic_fixture",
        producer_version="official-fixture-v1",
        producer_source_digest="a" * 64,
        fit_config_digest="b" * 64,
    )
    path = tmp_path / "synthetic.teacher.npz"
    field.save_npz(path)
    loaded = GaussianObservationField.load_npz(path, strict=True)
    assert loaded.provider == "synthetic_fixture"
    assert loaded.producer_version == field.producer_version
    assert loaded.producer_source_digest == field.producer_source_digest
    assert loaded.fit_config_digest == field.fit_config_digest
    assert torch.equal(
        loaded.query(field.pixel_centers()).color, field.query(field.pixel_centers()).color
    )

    with pytest.raises(ValueError, match="provider must be exactly"):
        _field(provider="unregistered")


def test_strict_observation_metadata_rejects_unknown_keys(tmp_path):
    source = tmp_path / "source.npz"
    modified = tmp_path / "modified.npz"
    _field().save_npz(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    metadata = json.loads(arrays["metadata_utf8"].tobytes())
    metadata["unexpected"] = "rejected-before-semantic-digest"
    arrays["metadata_utf8"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(), dtype=np.uint8
    )
    np.savez_compressed(modified, **arrays)

    with pytest.raises(ValueError, match="metadata keys are not exact"):
        GaussianObservationField.load_npz(modified, strict=True)


def test_observation_archive_rejects_array_tampering(tmp_path):
    source = tmp_path / "source.npz"
    tampered = tmp_path / "tampered.npz"
    _field().save_npz(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    arrays["colors"][0, 0] += 0.125
    np.savez_compressed(tampered, **arrays)
    with pytest.raises(ValueError, match="array integrity mismatch"):
        GaussianObservationField.load_npz(tampered)


def test_observation_archive_integrity_covers_optional_mean_residuals(tmp_path):
    source = tmp_path / "residual-source.npz"
    tampered = tmp_path / "residual-tampered.npz"
    field = _field(mean_residuals=torch.full((2, 2), 1e-5))
    field.save_npz(source)
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    arrays["mean_residuals"][0, 0] += 1e-5
    np.savez_compressed(tampered, **arrays)

    with pytest.raises(ValueError, match="array integrity mismatch: mean_residuals"):
        GaussianObservationField.load_npz(tampered)
