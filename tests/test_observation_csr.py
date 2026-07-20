"""Phase 1 correctness/determinism tests for the flattened CSR observation index.

These tests pin the accelerated CSR :class:`GaussianObservationIndex` to the frozen behavior it
replaced: the all-component :meth:`GaussianObservationField.query` reference and the private
grouped :class:`_GroupedObservationIndexReference` oracle. Everything here is CPU-only and
deterministic.
"""

from __future__ import annotations

import pytest
import torch

from rtgs.core.observation2d import (
    GaussianObservationField,
    GaussianObservationIndex,
    ObservationQuery,
    _GroupedObservationIndexReference,
)

# The established float32 numerical contract for indexed-vs-reference agreement.
_ATOL = 1e-6
_RTOL = 2e-6


def _make_field(
    *,
    n: int = 40,
    width: int = 96,
    height: int = 72,
    seed: int = 0,
    blend_mode: str = "normalized",
    support_fade_alpha: float = 0.0,
    aa_dilation: float = 0.0,
    with_color_grads: bool = False,
    with_filter_variance: bool = False,
    fit_window: tuple[int, int, int, int] | None = None,
    mean_residuals: torch.Tensor | None = None,
    scale_low: float = 0.5,
    scale_high: float = 3.0,
    zero_amplitude_fraction: float = 0.0,
    dtype: torch.dtype = torch.float32,
) -> GaussianObservationField:
    generator = torch.Generator().manual_seed(seed)
    means = (
        torch.rand(n, 2, generator=generator, dtype=dtype)
        * torch.tensor([width - 2.0, height - 2.0], dtype=dtype)
        + 1.0
    )
    log_scales = torch.log(
        torch.rand(n, 2, generator=generator, dtype=dtype) * (scale_high - scale_low) + scale_low
    )
    amplitudes = torch.rand(n, generator=generator, dtype=dtype) * 0.9 + 0.1
    if zero_amplitude_fraction > 0.0:
        zeroed = torch.rand(n, generator=generator, dtype=dtype) < zero_amplitude_fraction
        amplitudes = torch.where(zeroed, torch.zeros_like(amplitudes), amplitudes)
    return GaussianObservationField(
        width=width,
        height=height,
        means=means,
        log_scales=log_scales,
        rotations=(torch.rand(n, generator=generator, dtype=dtype) - 0.5) * 2.0,
        colors=torch.rand(n, 3, generator=generator, dtype=dtype) * 1.4 - 0.2,
        amplitudes=amplitudes,
        color_grads=(
            torch.randn(n, 2, 3, generator=generator, dtype=dtype) / 8.0
            if with_color_grads
            else None
        ),
        filter_variance=(
            torch.rand(n, generator=generator, dtype=dtype) * 0.4 if with_filter_variance else None
        ),
        mean_residuals=mean_residuals,
        blend_mode=blend_mode,
        support_fade_alpha=support_fade_alpha,
        aa_dilation=aa_dilation,
        fit_window=fit_window,
    )


def _sample_points(field: GaussianObservationField, count: int, seed: int) -> torch.Tensor:
    """Points spanning beyond the canvas so support/window edges and empty tiles are exercised."""
    generator = torch.Generator().manual_seed(seed)
    xy = torch.rand(count, 2, generator=generator, dtype=field.dtype)
    xy[:, 0] = xy[:, 0] * (field.width + 4.0) - 2.0
    xy[:, 1] = xy[:, 1] * (field.height + 4.0) - 2.0
    return xy


def _grouped_rows(reference: _GroupedObservationIndexReference) -> dict[int, list[int]]:
    return {key: ids.tolist() for key, ids in reference._tiles.items()}


# --------------------------------------------------------------------------------------------- #
# Construction: membership, ordering, caps, dtype selection.
# --------------------------------------------------------------------------------------------- #


@pytest.mark.parametrize("tile_size", (1, 2, 16))
@pytest.mark.parametrize("n", (1, 5, 40, 200))
def test_csr_construction_matches_grouped_reference_membership_and_ordering(tile_size, n):
    field = _make_field(n=n, seed=n, width=98, height=72, fit_window=(1, 1, 96, 70))
    index = GaussianObservationIndex(field, tile_size=tile_size)
    reference = _GroupedObservationIndexReference(field, tile_size=tile_size)
    rows = _grouped_rows(reference)

    # Non-empty tile count, entry count, and max candidates match the frozen oracle.
    assert index.n_entries == GaussianObservationIndex.estimate_entries(field, tile_size=tile_size)
    assert index.n_entries == reference.n_entries
    assert index.n_tiles == len(rows) == index.tile_keys.numel()
    assert index.max_candidates == reference.max_candidates
    assert index.tile_offsets.numel() == index.tile_keys.numel() + 1
    assert int(index.tile_offsets[-1]) == index.n_entries

    # Exact tile membership with ascending (canonical) component order in every CSR row.
    assert index.tile_keys.tolist() == sorted(rows)
    for position, key in enumerate(index.tile_keys.tolist()):
        lo, hi = int(index.tile_offsets[position]), int(index.tile_offsets[position + 1])
        row = index.component_ids[lo:hi].tolist()
        assert row == rows[key]
        assert row == sorted(row)


@pytest.mark.parametrize("tile_size", (1, 2, 16))
def test_vectorized_ranges_match_scalar_tile_bounds_oracle(tile_size):
    for seed in range(6):
        field = _make_field(n=64, seed=seed, fit_window=(2, 3, 88, 60), width=96, height=72)
        ids, tx0, tx1, ty0, ty1 = GaussianObservationIndex._component_tile_ranges(field, tile_size)
        expected = list(GaussianObservationIndex._tile_bounds(field, tile_size))
        got = list(
            zip(
                ids.tolist(),
                tx0.tolist(),
                tx1.tolist(),
                ty0.tolist(),
                ty1.tolist(),
                strict=True,
            )
        )
        assert got == expected


def test_entry_and_candidate_caps_fail_before_allocation():
    field = _make_field(n=60, seed=3, fit_window=(1, 1, 94, 70), width=96, height=72)
    estimated = GaussianObservationIndex.estimate_entries(field, tile_size=2)
    built = GaussianObservationIndex(field, tile_size=2)
    with pytest.raises(ValueError, match="entry cap exceeded before allocation"):
        GaussianObservationIndex(field, tile_size=2, max_entries=estimated - 1)
    with pytest.raises(ValueError, match="candidate cap exceeded"):
        GaussianObservationIndex(
            field,
            tile_size=2,
            max_entries=estimated,
            max_candidates=built.max_candidates - 1,
        )
    with pytest.raises(ValueError, match="max_query_pairs must be a positive integer"):
        GaussianObservationIndex(field, tile_size=2, max_query_pairs=0)


def test_signed_int32_boundary_and_forced_int64_component_ids(monkeypatch):
    field = _make_field(n=12, seed=5)
    # Default small field retains int32 component IDs.
    default = GaussianObservationIndex(field)
    assert default.component_id_dtype == torch.int32
    assert default.component_ids.dtype == torch.int32
    assert default.stats.component_id_dtype == "int32"

    # Exact signed-int32 boundary: limit == n - 1 keeps int32; one lower forces int64 without
    # allocating a giant real field.
    monkeypatch.setattr(GaussianObservationIndex, "_int32_component_limit", field.n - 1)
    assert GaussianObservationIndex(field).component_id_dtype == torch.int32
    monkeypatch.setattr(GaussianObservationIndex, "_int32_component_limit", field.n - 2)
    forced = GaussianObservationIndex(field)
    assert forced.component_id_dtype == torch.int64
    assert forced.component_ids.dtype == torch.int64

    # The int64 fallback stays exact.
    points = _sample_points(field, 128, seed=9)
    reference = field.query(points, component_chunk=1)
    torch.testing.assert_close(forced.query(points).color, reference.color, atol=_ATOL, rtol=_RTOL)


def test_payload_bytes_and_stats_expose_retained_csr_state():
    field = _make_field(n=80, seed=7)
    index = GaussianObservationIndex(field, tile_size=16, max_query_pairs=4096)
    expected = sum(
        tensor.element_size() * tensor.numel()
        for tensor in (index.tile_keys, index.tile_offsets, index.component_ids)
    )
    assert index.payload_bytes == expected
    stats = index.stats
    assert stats.retained_bytes == expected
    assert stats.max_query_pairs == 4096
    assert stats.total_entries == index.n_entries
    assert stats.nonempty_tiles == index.n_tiles


def test_empty_index_is_well_formed_and_returns_zeros():
    field = _make_field(n=4, seed=1, zero_amplitude_fraction=1.0)  # all amplitudes forced to zero
    index = GaussianObservationIndex(field)
    assert index.n_entries == 0
    assert index.n_tiles == 0
    assert index.tile_offsets.tolist() == [0]
    points = _sample_points(field, 32, seed=2)
    query = index.query(points)
    assert torch.equal(query.weight_sum, torch.zeros(points.shape[0]))
    assert torch.equal(query.color, torch.zeros(points.shape[0], 3))
    assert torch.equal(index.query_weight_sum(points), torch.zeros(points.shape[0]))


# --------------------------------------------------------------------------------------------- #
# Query parity across feature combinations, edges, and blending modes.
# --------------------------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="plain"),
        pytest.param({"support_fade_alpha": 0.5}, id="support-fade"),
        pytest.param({"aa_dilation": 0.3, "with_filter_variance": True}, id="aa-filter-variance"),
        pytest.param({"with_color_grads": True}, id="affine-color"),
        pytest.param({"blend_mode": "additive"}, id="additive"),
        pytest.param(
            {"blend_mode": "additive", "with_color_grads": True, "support_fade_alpha": 0.4},
            id="additive-affine-fade",
        ),
        pytest.param({"scale_low": 0.3, "scale_high": 6.0}, id="mixed-radii"),
        pytest.param({"zero_amplitude_fraction": 0.4}, id="some-zero-amplitudes"),
    ],
)
def test_csr_query_matches_field_and_grouped_reference(kwargs):
    field = _make_field(n=120, seed=11, fit_window=(1, 1, 94, 70), width=96, height=72, **kwargs)
    points = _sample_points(field, 512, seed=17)
    reference = field.query(points, component_chunk=1)
    grouped = _GroupedObservationIndexReference(field, tile_size=16)
    for tile_size in (1, 2, 16):
        index = GaussianObservationIndex(field, tile_size=tile_size)
        actual = index.query(points)
        torch.testing.assert_close(actual.color, reference.color, atol=_ATOL, rtol=_RTOL)
        torch.testing.assert_close(actual.numerator, reference.numerator, atol=_ATOL, rtol=_RTOL)
        torch.testing.assert_close(actual.weight_sum, reference.weight_sum, atol=_ATOL, rtol=_RTOL)
        torch.testing.assert_close(
            index.query_weight_sum(points), reference.weight_sum, atol=_ATOL, rtol=_RTOL
        )
        assert torch.equal(actual.valid, reference.valid)
    grouped_query = grouped.query(points)
    torch.testing.assert_close(
        GaussianObservationIndex(field).query(points).color,
        grouped_query.color,
        atol=_ATOL,
        rtol=_RTOL,
    )


def test_csr_query_handles_repeated_tile_keys_window_edges_and_empty_tiles():
    field = _make_field(n=60, seed=13, fit_window=(2, 2, 90, 66), width=96, height=72)
    index = GaussianObservationIndex(field, tile_size=16)

    fit_x, fit_y, fit_w, fit_h = field.fit_window
    # Many points in one tile (repeated tile keys), points on the fitted-window edges, points in
    # a definitely-empty far tile, and out-of-window points.
    repeated = torch.full((50, 2), fit_x + 5.5, dtype=field.dtype)
    edges = torch.tensor(
        [
            [fit_x + 0.5, fit_y + 0.5],
            [fit_x + fit_w - 0.5, fit_y + fit_h - 0.5],
            [fit_x - 0.5, fit_y + 1.5],  # just outside the window
            [fit_x + fit_w + 0.5, fit_y + 1.5],
        ],
        dtype=field.dtype,
    )
    empty = torch.tensor([[fit_x + fit_w - 0.5, fit_y + 0.5]], dtype=field.dtype)
    points = torch.cat([repeated, edges, empty])
    reference = field.query(points, component_chunk=1)
    actual = index.query(points)
    torch.testing.assert_close(actual.color, reference.color, atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(actual.weight_sum, reference.weight_sum, atol=_ATOL, rtol=_RTOL)
    # Out-of-window points contribute nothing.
    assert torch.equal(actual.valid, field.valid_domain(points))
    assert float(actual.weight_sum[52]) == 0.0
    assert float(actual.weight_sum[53]) == 0.0


def test_csr_query_matches_reference_with_odd_crop_offset_mean_residuals():
    fit_window = (1283, 1937, 40, 30)
    local = torch.tensor(
        [
            [torch.nextafter(torch.tensor(0.5), torch.tensor(torch.inf)), 0.5],
            [5.2, 4.4],
            [12.6, 8.1],
            [20.5, 15.5],
        ]
    )
    origin = torch.tensor([fit_window[0] + 0.5, fit_window[1] + 0.5])
    native = local + origin
    residuals = (local - (native - origin)).to(torch.float32)
    field = GaussianObservationField(
        width=5328,
        height=4608,
        means=native,
        log_scales=torch.log(torch.tensor([[1.4, 0.9], [0.8, 1.1], [1.6, 1.6], [0.7, 2.2]])),
        rotations=torch.tensor([0.2, -0.4, 0.9, -1.1]),
        colors=torch.tensor([[0.9, 0.2, 0.1], [0.1, 0.4, 0.8], [0.3, 0.7, 0.2], [0.6, 0.1, 0.5]]),
        amplitudes=torch.tensor([0.8, 0.6, 0.9, 0.5]),
        mean_residuals=residuals,
        support_fade_alpha=0.4,
        fit_window=fit_window,
    )
    points = field.pixel_centers()
    reference = field.query(points, component_chunk=1)
    index = GaussianObservationIndex(field, tile_size=16)
    torch.testing.assert_close(index.query(points).color, reference.color, atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(
        index.query_weight_sum(points), reference.weight_sum, atol=_ATOL, rtol=_RTOL
    )


def test_csr_query_coordinate_gradients_match_reference():
    field = _make_field(n=48, seed=19, with_color_grads=True, support_fade_alpha=0.3)
    # Points near component centers so gradients are well-defined and non-zero.
    base = field.means[torch.arange(0, field.n, 3)] + 0.15
    reference_xy = base.detach().clone().requires_grad_(True)
    reference = field.query(reference_xy, component_chunk=1)
    reference_grad = torch.autograd.grad(reference.color.square().sum(), reference_xy)[0]

    index = GaussianObservationIndex(field, tile_size=16)
    indexed_xy = base.detach().clone().requires_grad_(True)
    indexed = index.query(indexed_xy)
    indexed_grad = torch.autograd.grad(indexed.color.square().sum(), indexed_xy)[0]

    assert torch.isfinite(indexed_grad).all()
    torch.testing.assert_close(indexed_grad, reference_grad, atol=1e-5, rtol=1e-5)


# --------------------------------------------------------------------------------------------- #
# Bounded pair streaming and determinism.
# --------------------------------------------------------------------------------------------- #


def test_every_paired_call_respects_max_query_pairs_including_split_rows(monkeypatch):
    # A single 16x16 tile holds every component, so one query point yields one long CSR row.
    field = _make_field(n=60, seed=23, width=16, height=16, scale_low=1.5, scale_high=2.5)
    index = GaussianObservationIndex(field, tile_size=16)
    assert index.n_tiles == 1  # heavily populated single tile
    row_length = int(index.tile_offsets[-1])
    assert row_length == field.n

    observed: list[int] = []
    original = GaussianObservationField._paired_values

    def tracked(self, xy, component_ids):
        observed.append(int(component_ids.numel()))
        return original(self, xy, component_ids)

    monkeypatch.setattr(GaussianObservationField, "_paired_values", tracked)

    cap = 7
    capped = GaussianObservationIndex(field, tile_size=16, max_query_pairs=cap)
    point = torch.tensor([[7.5, 7.5]], dtype=field.dtype)
    result = capped.query(point, component_chunk=4096)
    assert observed  # paired evaluation actually ran
    assert max(observed) <= cap
    assert len(observed) >= (row_length + cap - 1) // cap  # the long row was split
    assert capped.peak_pair_chunk <= cap

    reference = field.query(point, component_chunk=1)
    torch.testing.assert_close(result.color, reference.color, atol=_ATOL, rtol=_RTOL)


def test_row_crossing_a_stream_boundary_stays_exact():
    # Three points share one heavily populated tile; a tiny cap forces rows to straddle chunks.
    field = _make_field(n=50, seed=29, width=16, height=16, scale_low=1.5, scale_high=2.5)
    points = torch.tensor([[6.5, 6.5], [7.5, 7.5], [8.5, 8.5]], dtype=field.dtype)
    reference = field.query(points, component_chunk=1)
    for cap in (5, 13, 51, field.n * 3):
        index = GaussianObservationIndex(field, tile_size=16, max_query_pairs=cap)
        actual = index.query(points, component_chunk=4096)
        torch.testing.assert_close(actual.color, reference.color, atol=_ATOL, rtol=_RTOL)
        torch.testing.assert_close(
            index.query_weight_sum(points, component_chunk=4096),
            reference.weight_sum,
            atol=_ATOL,
            rtol=_RTOL,
        )
        assert index.peak_pair_chunk <= cap


def test_query_counters_accumulate_and_track_peak_chunk():
    field = _make_field(n=90, seed=31)
    index = GaussianObservationIndex(field, tile_size=16, max_query_pairs=64)
    points = _sample_points(field, 300, seed=32)
    index.query(points, component_chunk=4096)
    first_pairs = index.total_pairs_evaluated
    assert first_pairs > 0
    assert 0 < index.peak_pair_chunk <= 64
    assert index.total_query_points > 0
    index.query_weight_sum(points, component_chunk=4096)
    assert index.total_pairs_evaluated > first_pairs  # cumulative across calls


def test_different_legal_pair_budgets_agree_within_contract():
    field = _make_field(n=140, seed=37, with_color_grads=True, support_fade_alpha=0.25)
    points = _sample_points(field, 400, seed=41)
    baseline = GaussianObservationIndex(field, tile_size=16, max_query_pairs=1 << 20).query(points)
    for cap in (3, 16, 129, 1024):
        candidate = GaussianObservationIndex(field, tile_size=16, max_query_pairs=cap).query(points)
        torch.testing.assert_close(candidate.color, baseline.color, atol=_ATOL, rtol=_RTOL)
        torch.testing.assert_close(
            candidate.weight_sum, baseline.weight_sum, atol=_ATOL, rtol=_RTOL
        )


# --------------------------------------------------------------------------------------------- #
# Backend substitutability.
# --------------------------------------------------------------------------------------------- #


def test_grouped_reference_backend_is_query_substitutable():
    field = _make_field(n=64, seed=43, with_color_grads=True)
    points = _sample_points(field, 256, seed=44)
    grouped = _GroupedObservationIndexReference(field, tile_size=16)
    csr = GaussianObservationIndex(field, tile_size=16)
    torch.testing.assert_close(
        csr.query(points).color, grouped.query(points).color, atol=_ATOL, rtol=_RTOL
    )
    torch.testing.assert_close(
        csr.query_weight_sum(points), grouped.query_weight_sum(points), atol=_ATOL, rtol=_RTOL
    )


def test_custom_third_party_backend_still_satisfies_the_protocol():
    field = _make_field(n=32, seed=47)
    points = _sample_points(field, 100, seed=48)

    class _ScaledFieldBackend:
        """A minimal third-party ObservationQueryBackend that delegates to the field."""

        def __init__(self, field: GaussianObservationField):
            self.field = field

        def query(self, xy: torch.Tensor, component_chunk: int = 4096) -> ObservationQuery:
            return self.field.query(xy, component_chunk=component_chunk)

        def query_weight_sum(self, xy: torch.Tensor, component_chunk: int = 4096) -> torch.Tensor:
            return self.field.query_weight_sum(xy, component_chunk=component_chunk)

    backend = _ScaledFieldBackend(field)
    reference = field.query(points, component_chunk=1)
    torch.testing.assert_close(backend.query(points).color, reference.color, atol=_ATOL, rtol=_RTOL)
    torch.testing.assert_close(
        backend.query_weight_sum(points), reference.weight_sum, atol=_ATOL, rtol=_RTOL
    )
