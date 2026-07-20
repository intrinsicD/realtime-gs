"""Focused CPU checks for the analytic-field product-kernel microbenchmark."""

from __future__ import annotations

import math

import pytest

from rtgs import bench


def test_field_benchmark_presets_are_bounded() -> None:
    quick = bench.BenchConfig.quick()
    full = bench.BenchConfig.full()
    smoke = bench.BenchConfig.smoke()

    assert (quick.field_components, quick.field_repeats, quick.field_chunk_size) == (96, 3, 64)
    assert (full.field_components, full.field_repeats, full.field_chunk_size) == (256, 10, 128)
    assert (smoke.field_components, smoke.field_repeats, smoke.field_chunk_size) == (16, 1, 16)


def test_field_product_kernel_benchmark_is_deterministic_and_reports_workload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((10.0, 12.0, 20.0, 24.0))
    monkeypatch.setattr(bench.time, "perf_counter", lambda: next(ticks))

    first = bench.run_field_product_kernel_benchmark(
        components=3,
        repeats=2,
        chunk_size=2,
    )
    second = bench.run_field_product_kernel_benchmark(
        components=3,
        repeats=2,
        chunk_size=2,
    )

    assert first["components_per_field"] == 3
    assert first["field_l2_evaluations"] == 2
    assert first["component_pair_terms"] == 6 * 3 * 3 * 2
    assert first["seconds"] == 2.0
    assert first["evaluations_per_s"] == 1.0
    assert math.isfinite(first["l2_total"])
    assert first["l2_total"] > 0.0
    assert second["l2_total"] == first["l2_total"]
    assert second["seconds"] == 4.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"components": 0, "repeats": 1, "chunk_size": 1}, "components"),
        ({"components": 1, "repeats": -1, "chunk_size": 1}, "repeats"),
        ({"components": 1, "repeats": 1, "chunk_size": True}, "chunk_size"),
    ],
)
def test_field_product_kernel_benchmark_rejects_invalid_sizes(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        bench.run_field_product_kernel_benchmark(**kwargs)
