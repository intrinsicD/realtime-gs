"""CPU-only contract tests for the exploratory native StructSplat capacity sweep."""

from __future__ import annotations

import dataclasses
import inspect

import pytest
import torch
from benchmarks import stage1_capacity_sweep as sweep

from rtgs.core.observation2d import GaussianObservationField


def test_import_is_cpu_first_and_provider_imports_are_lazy() -> None:
    source = inspect.getsource(sweep)
    prefix = source.split("def structsplat_source_binding", maxsplit=1)[0]
    assert "import structsplat" not in prefix
    assert "torch.cuda.is_available()" not in prefix
    assert sweep.DEFAULT_VIEWS == ("C0014",)
    assert sweep.HELDOUT_VIEW not in sweep.TRAIN_VIEWS
    assert sweep.LOCAL_LIBSTDCXX_PRELOAD.is_absolute()


def test_view_resolution_is_closed_over_training_views_and_rejects_heldout() -> None:
    assert sweep.resolve_views(None) == ("C0014",)
    assert sweep.resolve_views(["all"]) == sweep.TRAIN_VIEWS
    assert sweep.resolve_views(["c0014", "C0031"]) == ("C0014", "C0031")
    with pytest.raises(ValueError, match="held out"):
        sweep.resolve_views([sweep.HELDOUT_VIEW])
    with pytest.raises(ValueError, match="outside the frozen training set"):
        sweep.resolve_views(["C9999"])
    with pytest.raises(ValueError, match="cannot be combined"):
        sweep.resolve_views(["all", "C0014"])
    with pytest.raises(ValueError, match="duplicate"):
        sweep.resolve_views(["C0014", "c0014"])
    with pytest.raises(ValueError, match="not in the frozen training set"):
        sweep.view_paths(sweep.HELDOUT_VIEW)


def test_core_profile_contains_required_scale_initializer_and_recipe_cells() -> None:
    core = sweep.resolve_arms("core", None)
    assert [
        (arm.initializer, arm.n_init_2d, arm.iterations, arm.scale_cap_mode) for arm in core
    ] == [
        ("aniso_onedge", 640, 100, "feature"),
        ("aniso_onedge", 2000, 100, "feature"),
        ("aniso_onedge", 2000, 500, "feature"),
        ("quadtree_wse", 2000, 500, "feature"),
        ("quadtree_wse", 5000, 1000, "feature"),
        ("quadtree_wse", 5000, 1000, "none"),
    ]
    extended = sweep.resolve_arms("extended", None)
    assert any(arm.n_init_2d == 10000 and arm.iterations == 2000 for arm in extended)
    adaptive = extended[-1]
    assert adaptive.adaptive
    assert (adaptive.n_init_2d, adaptive.n_max_2d, adaptive.iterations) == (2000, 10000, 2000)


def test_preregistered_causal_contrasts_change_exactly_one_axis() -> None:
    sweep.validate_causal_contrasts()
    for contrast in sweep.CAUSAL_CONTRASTS:
        left = sweep.ARM_BY_NAME[contrast["left"]].causal_axes()
        right = sweep.ARM_BY_NAME[contrast["right"]].causal_axes()
        assert {key for key in left if left[key] != right[key]} == {contrast["changed_axis"]}

    broken = dataclasses.replace(
        sweep.ARM_BY_NAME["count_aniso_n2000_i100"],
        initializer="quadtree_wse",
    )
    left = sweep.ARM_BY_NAME["current_aniso_n640_i100"].causal_axes()
    right = broken.causal_axes()
    assert {key for key in left if left[key] != right[key]} == {
        "initializer",
        "component_budget",
    }


def test_arm_validation_couples_adaptive_flag_and_capacity() -> None:
    with pytest.raises(ValueError, match="adaptive must be true"):
        sweep.Arm("bad", "quadtree_wse", 20, 10, max_2d=40)
    with pytest.raises(ValueError, match="adaptive must be true"):
        sweep.Arm("bad", "quadtree_wse", 20, 10, adaptive=True)
    with pytest.raises(ValueError, match="unsupported initializer"):
        sweep.Arm("bad", "random", 20, 10)
    with pytest.raises(ValueError, match="unsupported scale cap"):
        sweep.Arm("bad", "quadtree_wse", 20, 10, scale_cap_mode="hard")


def test_weighted_error_metrics_report_exact_pixel_and_scalar_counts() -> None:
    target = torch.zeros(3, 4, 3)
    prediction = target.clone()
    prediction[1, 2] = 1.0
    weights = torch.zeros(3, 4)
    weights[1, 2] = 1.0
    result = sweep.weighted_error_metrics(prediction, target, weights)
    assert result["weighted_pixel_count"] == 1
    assert result["weighted_scalar_count"] == 3
    assert result["raw"]["mse"] == pytest.approx(1.0)
    assert result["raw"]["psnr_db"] == pytest.approx(0.0)


def test_weighted_ssim_identity_is_one_and_foreground_corruption_is_detected() -> None:
    generator = torch.Generator().manual_seed(771)
    target = torch.rand(23, 19, 3, generator=generator)
    weights = torch.zeros(23, 19)
    weights[5:18, 4:15] = 1.0
    identity_small_tiles = sweep.weighted_ssim(target, target, weights, tile_rows=4)
    identity_large_tile = sweep.weighted_ssim(target, target, weights, tile_rows=100)
    corrupted = target.clone()
    corrupted[9:14, 8:12] = 1.0 - corrupted[9:14, 8:12]
    corrupted_score = sweep.weighted_ssim(corrupted, target, weights, tile_rows=5)
    assert identity_small_tiles == pytest.approx(1.0, abs=2e-6)
    assert identity_large_tile == pytest.approx(identity_small_tiles, abs=2e-6)
    assert corrupted_score < identity_small_tiles


def test_weighted_ssim_validates_shapes_window_and_positive_support() -> None:
    image = torch.zeros(12, 13, 3)
    with pytest.raises(ValueError, match="matching"):
        sweep.weighted_ssim(image, image[:11], torch.ones(12, 13))
    with pytest.raises(ValueError, match="weights must match"):
        sweep.weighted_ssim(image, image, torch.ones(12, 12))
    with pytest.raises(ValueError, match="positive pixels"):
        sweep.weighted_ssim(image, image, torch.zeros(12, 13))
    with pytest.raises(ValueError, match="positive odd"):
        sweep.weighted_ssim(image, image, torch.ones(12, 13), window=4)


def test_runtime_estimates_are_explicitly_conservative_and_monotonic() -> None:
    rows = sweep.estimated_runtime_table()
    values = {row["arm"]: row["conservative_seconds"] for row in rows}
    assert values["current_aniso_n640_i100"] == pytest.approx(16.693338783981744)
    assert values["budget_aniso_n2000_i500"] > values["count_aniso_n2000_i100"]
    assert values["extended_quadtree_n10000_i2000"] > values["scale_quadtree_n5000_i1000"]
    assert all("scheduling estimates" not in row["model"] for row in rows)


def test_fixed_and_adaptive_fit_outcomes_fail_closed() -> None:
    fixed = sweep.ARM_BY_NAME["current_aniso_n640_i100"]
    sweep.validate_fit_outcome(fixed, m_opt_2d=640, iterations_run=100)
    with pytest.raises(RuntimeError, match="returned 639"):
        sweep.validate_fit_outcome(fixed, m_opt_2d=639, iterations_run=100)
    with pytest.raises(RuntimeError, match="ran 99"):
        sweep.validate_fit_outcome(fixed, m_opt_2d=640, iterations_run=99)

    adaptive = sweep.ARM_BY_NAME["extended_adaptive_quadtree_n2000_to10000_i2000"]
    sweep.validate_fit_outcome(adaptive, m_opt_2d=5000, iterations_run=1200)
    with pytest.raises(RuntimeError, match="outside"):
        sweep.validate_fit_outcome(adaptive, m_opt_2d=10001, iterations_run=1200)


def test_semantic_parity_gates_nonfinite_and_excess_error() -> None:
    sweep.validate_semantic_parity(torch.zeros(8), torch.full((3,), 1e-6))
    with pytest.raises(RuntimeError, match="exceeds"):
        sweep.validate_semantic_parity(torch.tensor([6e-4]), torch.zeros(1))
    with pytest.raises(RuntimeError, match="mean_abs"):
        sweep.validate_semantic_parity(torch.full((8,), 2e-6), torch.zeros(1))
    with pytest.raises(RuntimeError, match="mean_abs"):
        sweep.validate_semantic_parity(torch.zeros(1), torch.full((8,), 2e-6))
    with pytest.raises(RuntimeError, match="finite"):
        sweep.validate_semantic_parity(torch.zeros(1), torch.tensor([float("nan")]))


def test_query_parity_sampling_includes_component_support_and_uniform_pixels() -> None:
    field = GaussianObservationField(
        width=12,
        height=10,
        means=torch.tensor([[3.5, 4.5], [5.5, 5.5], [6.5, 4.5]]),
        log_scales=torch.zeros(3, 2),
        rotations=torch.zeros(3),
        colors=torch.full((3, 3), 0.5),
        amplitudes=torch.ones(3),
        fit_window=(2, 3, 6, 4),
        view_id="fixture",
        n_init=3,
        provider="synthetic_fixture",
        producer_version="test",
        producer_source_digest="a" * 64,
        fit_config_digest="b" * 64,
    )
    xy, sample_x, sample_y, counts = sweep.query_parity_samples(
        field,
        sample_count=8,
        generator=torch.Generator().manual_seed(4),
    )
    assert counts == {"component_support_samples": 3, "uniform_crop_samples": 5}
    assert xy.shape == (8, 2)
    assert torch.equal(xy[:3], field.support_centers())
    assert torch.equal(sample_x[:3], torch.tensor([1, 3, 4]))
    assert torch.equal(sample_y[:3], torch.tensor([1, 2, 1]))
    assert bool((sample_x >= 0).all() and (sample_x < 6).all())
    assert bool((sample_y >= 0).all() and (sample_y < 4).all())


def test_assemble_refuses_existing_outputs_before_mutating(tmp_path) -> None:
    (tmp_path / "plan.json").write_text(
        '{"selected_views":["C0014"],"arms":[{"name":"current_aniso_n640_i100"}]}',
        encoding="utf-8",
    )
    sentinel = tmp_path / "result.json"
    sentinel.write_text("do not replace", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        sweep.assemble(tmp_path)

    assert sentinel.read_text(encoding="utf-8") == "do not replace"
    assert not (tmp_path / "C0014_capacity_contact_sheet.png").exists()
