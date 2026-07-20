"""CPU-only tests for the sealed GaussianImage++ provider parity harness."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from benchmarks import gaussianimage_plus_provider_parity as experiment


def _native_like_output(
    field: experiment.DirectCovarianceField,
) -> dict[str, np.ndarray | dict[str, object]]:
    raw, clamped, projection = experiment.render_cpu(field)
    flattened: list[int] = []
    bins = []
    for candidates in projection.tile_candidates:
        lower = len(flattened)
        flattened.extend(candidates)
        bins.append([lower, len(flattened)])
    return {
        "means": field.means.numpy(),
        "covariances": field.covariances.numpy(),
        "colors": field.colors.numpy(),
        "opacities": field.opacities.numpy(),
        "background": field.background.numpy(),
        "raw": raw.numpy(),
        "clamped": clamped.numpy(),
        "xys": projection.xys.numpy(),
        "conics": projection.conics.numpy(),
        "radii": projection.radii.numpy(),
        "hits": projection.hits.numpy(),
        "gaussian_ids_sorted": np.asarray(flattened, dtype=np.int32),
        "tile_bins": np.asarray(bins, dtype=np.int32),
        "metadata": {},
    }


def test_harness_import_is_cpu_only_and_outcome_free():
    source = inspect.getsource(experiment)
    prefix = source.split("def execute_official_run", maxsplit=1)[0]
    assert "import gsplat" not in source
    assert "PIL" not in source
    assert "imageio" not in source
    assert "execute_official_run(" not in prefix
    assert experiment.MAX_TILE_POPULATION == 256
    assert pytest.approx(1.0 / 255.0) == experiment.ALPHA_CUTOFF


def test_field_validation_is_strict_and_cpu_normalized():
    field = experiment.DirectCovarianceField(
        means=np.asarray([[1.0, 2.0]], dtype=np.float64),
        covariances=[[2.0, 0.0, 3.0]],
        colors=[[0.2, 0.3, 0.4]],
        opacities=[[1.0]],
        background=[1.0, 1.0, 1.0],
        height=8,
        width=9,
    )
    assert field.means.device.type == "cpu"
    assert field.means.dtype == torch.float32
    assert field.tile_bounds == (1, 1)
    with pytest.raises(ValueError, match="means must have shape"):
        experiment.DirectCovarianceField(
            means=[[1.0, 2.0, 3.0]],
            covariances=[[1.0, 0.0, 1.0]],
            colors=[[1.0, 1.0, 1.0]],
            opacities=[[1.0]],
            background=[1.0, 1.0, 1.0],
            height=8,
            width=8,
        )
    with pytest.raises(ValueError, match="non-finite"):
        experiment.DirectCovarianceField(
            means=[[float("nan"), 2.0]],
            covariances=[[1.0, 0.0, 1.0]],
            colors=[[1.0, 1.0, 1.0]],
            opacities=[[1.0]],
            background=[1.0, 1.0, 1.0],
            height=8,
            width=8,
        )
    with pytest.raises(ValueError, match="16x16"):
        experiment.DirectCovarianceField(
            means=[[1.0, 2.0]],
            covariances=[[1.0, 0.0, 1.0]],
            colors=[[1.0, 1.0, 1.0]],
            opacities=[[1.0]],
            background=[1.0, 1.0, 1.0],
            height=8,
            width=8,
            block_size=8,
        )


def test_direct_covariance_projection_uses_inverse_and_integer_pixel_coordinates():
    field = experiment.synthetic_fields()["overlap_upper_clamp"]
    projection = experiment.project_cpu(field)
    assert torch.equal(projection.xys, field.means)
    assert torch.allclose(projection.conics[0], torch.tensor([0.25, 0.0, 1.0 / 9.0]), atol=1e-7)
    assert torch.equal(projection.radii, torch.tensor([9, 9, 6], dtype=torch.int32))
    assert torch.equal(projection.hits, torch.ones(3, dtype=torch.int32))

    raw, _, _ = experiment.render_cpu(field, projection)
    assert raw[5, 4].tolist() == pytest.approx([1.4, 0.1, 0.2], abs=2e-7)


def test_additive_clamp_cutoff_and_background_semantics():
    fields = experiment.synthetic_fields()
    overlap_raw, overlap_clamped, overlap_projection = experiment.render_cpu(
        fields["overlap_upper_clamp"]
    )
    experiment.validate_synthetic_semantics(
        "overlap_upper_clamp", overlap_raw, overlap_clamped, overlap_projection
    )
    assert float(overlap_raw[5, 4, 0]) > 1.0
    assert float(overlap_clamped[5, 4, 0]) == 1.0

    fractional_raw, fractional_clamped, fractional_projection = experiment.render_cpu(
        fields["fractional_rotated_lower_clamp"]
    )
    experiment.validate_synthetic_semantics(
        "fractional_rotated_lower_clamp",
        fractional_raw,
        fractional_clamped,
        fractional_projection,
    )
    assert float(fractional_raw.min()) < 0.0
    assert float(fractional_clamped.min()) == 0.0

    cutoff_raw, cutoff_clamped, cutoff_projection = experiment.render_cpu(fields["cutoff"])
    experiment.validate_synthetic_semantics("cutoff", cutoff_raw, cutoff_clamped, cutoff_projection)
    assert float(cutoff_raw[8, 11].sum()) > 0.0
    assert float(cutoff_raw[8, 12].sum()) == 0.0

    culled_raw, culled_clamped, culled_projection = experiment.render_cpu(
        fields["all_culled_background"]
    )
    experiment.validate_synthetic_semantics(
        "all_culled_background", culled_raw, culled_clamped, culled_projection
    )
    assert torch.equal(culled_raw[0, 0], torch.tensor([0.7, 0.8, 0.9]))

    hit_raw, hit_clamped, hit_projection = experiment.render_cpu(
        fields["one_hit_background_ignored"]
    )
    experiment.validate_synthetic_semantics(
        "one_hit_background_ignored", hit_raw, hit_clamped, hit_projection
    )
    assert torch.equal(hit_raw[0, 0], torch.zeros(3))


def test_fractional_center_uses_integer_lattice_not_half_pixel_lattice():
    source = experiment.synthetic_fields()["fractional_rotated_lower_clamp"]
    field = experiment.DirectCovarianceField(
        means=source.means[:1],
        covariances=source.covariances[:1],
        colors=source.colors[:1],
        opacities=source.opacities[:1],
        background=source.background,
        height=source.height,
        width=source.width,
    )
    raw, _, projection = experiment.render_cpu(field)
    y, x = 9, 7
    gaussian_id = 0
    delta_x = field.means[gaussian_id, 0] - float(x)
    delta_y = field.means[gaussian_id, 1] - float(y)
    conic = projection.conics[gaussian_id]
    sigma = (
        0.5 * (conic[0] * delta_x.square() + conic[2] * delta_y.square())
        + conic[1] * delta_x * delta_y
    )
    alpha = field.opacities[gaussian_id, 0] * torch.exp(-sigma)
    integer_contribution = alpha * field.colors[gaussian_id]
    assert torch.allclose(raw[y, x] - integer_contribution, torch.zeros(3), atol=1e-7)

    half_delta_x = field.means[gaussian_id, 0] - (x + 0.5)
    half_delta_y = field.means[gaussian_id, 1] - (y + 0.5)
    half_sigma = (
        0.5 * (conic[0] * half_delta_x.square() + conic[2] * half_delta_y.square())
        + conic[1] * half_delta_x * half_delta_y
    )
    half_contribution = (
        field.opacities[gaussian_id, 0] * torch.exp(-half_sigma) * field.colors[gaussian_id]
    )
    assert not torch.allclose(integer_contribution, half_contribution, atol=1e-4)


def test_radius_clip_and_out_of_frame_tile_intersection():
    fields = experiment.synthetic_fields()
    clipped_raw, clipped_clamped, clipped_projection = experiment.render_cpu(fields["radius_clip"])
    experiment.validate_synthetic_semantics(
        "radius_clip", clipped_raw, clipped_clamped, clipped_projection
    )
    assert int(clipped_projection.hits.sum()) == 0
    assert torch.equal(clipped_raw[0, 0], torch.tensor([0.7, 0.8, 0.9]))

    outside_raw, outside_clamped, outside_projection = experiment.render_cpu(
        fields["out_of_frame_intersection"]
    )
    experiment.validate_synthetic_semantics(
        "out_of_frame_intersection",
        outside_raw,
        outside_clamped,
        outside_projection,
    )
    assert int(outside_projection.hits[0]) == 1
    assert float(outside_raw[:, 0].abs().sum()) > 0.0


def test_tile_population_cap_is_a_hard_guard():
    field = experiment.synthetic_fields()["tile_cap_sentinel"]
    projection = experiment.project_cpu(field)
    assert projection.max_tile_population == 257
    with pytest.raises(experiment.TilePopulationError, match="exceeds the frozen cap 256"):
        experiment.render_cpu(field, projection)


def test_spd_filter_is_deterministic_preserves_order_and_does_not_repair():
    field = experiment.DirectCovarianceField(
        means=[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
        covariances=[
            [1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0],
            [-1.0, 0.0, -1.0],
            [2.0, 0.25, 3.0],
        ],
        colors=torch.arange(12, dtype=torch.float32).reshape(4, 3),
        opacities=torch.ones(4, 1),
        background=torch.ones(3),
        height=16,
        width=16,
    )
    mask = field.spd_mask()
    assert torch.equal(mask, torch.tensor([True, False, False, True]))
    first, first_mask = field.filtered_spd()
    second, second_mask = field.filtered_spd()
    assert torch.equal(first_mask, second_mask)
    assert torch.equal(first.means, field.means[[0, 3]])
    assert first.content_hash() == second.content_hash()
    assert torch.equal(first.covariances, field.covariances[[0, 3]])


@pytest.mark.parametrize("color_norm", (False, True))
def test_checkpoint_adapter_adds_slv_bound_and_applies_only_declared_color_activation(
    tmp_path: Path, color_norm: bool
):
    path = tmp_path / "checkpoint.pth.tar"
    state = {
        "_xyz": torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
        "_cov2d": torch.tensor([[1.0, 0.25, 2.0], [2.0, -0.5, 3.0]]),
        "_features_dc": torch.tensor([[-2.0, 0.0, 2.0], [0.2, 0.3, 0.4]]),
        "_opacity": torch.ones(2, 1),
        "background": torch.ones(3),
    }
    torch.save(
        {
            "gs": state,
            "slv_bound": torch.tensor([[0.5, 0.0, 0.5], [1.0, 0.0, 1.0]]),
            "num_gs": 2,
            "psnr": 21.0,
            "ms-ssim": float("nan"),
        },
        path,
    )
    adapted = experiment.load_checkpoint_cpu(
        path,
        expected_sha256=experiment.sha256_file(path),
        height=16,
        width=16,
        color_norm=color_norm,
    )
    assert torch.equal(
        adapted.field.covariances,
        torch.tensor([[1.5, 0.25, 2.5], [3.0, -0.5, 4.0]]),
    )
    expected_colors = torch.sigmoid(state["_features_dc"]) if color_norm else state["_features_dc"]
    assert torch.equal(adapted.field.colors, expected_colors)
    assert adapted.checkpoint_num_gaussians == 2
    assert adapted.reported_psnr == 21.0
    assert adapted.reported_ms_ssim is None


def test_checkpoint_adapter_rejects_count_mismatch(tmp_path: Path):
    path = tmp_path / "checkpoint.pth.tar"
    torch.save(
        {
            "gs": {
                "_xyz": torch.zeros(1, 2),
                "_cov2d": torch.ones(1, 3),
                "_features_dc": torch.zeros(1, 3),
                "_opacity": torch.ones(1, 1),
                "background": torch.ones(3),
            },
            "slv_bound": torch.zeros(1, 3),
            "num_gs": 2,
        },
        path,
    )
    with pytest.raises(ValueError, match="num_gs"):
        experiment.load_checkpoint_cpu(
            path,
            expected_sha256=experiment.sha256_file(path),
            height=16,
            width=16,
            color_norm=False,
        )


def test_checkpoint_adapter_rejects_wrong_hash_before_decode(tmp_path: Path, monkeypatch):
    path = tmp_path / "checkpoint.pth.tar"
    path.write_bytes(b"not a checkpoint")
    decoded = False

    def fail_if_decoded(*args, **kwargs):
        nonlocal decoded
        decoded = True
        raise AssertionError("torch.load must not run for bytes with the wrong hash")

    monkeypatch.setattr(torch, "load", fail_if_decoded)
    with pytest.raises(RuntimeError, match="checkpoint bytes differ"):
        experiment.load_checkpoint_cpu(
            path,
            expected_sha256="0" * 64,
            height=16,
            width=16,
            color_norm=False,
        )
    assert decoded is False


def test_field_npz_roundtrip_is_pickle_free_and_binds_content(tmp_path: Path):
    field = experiment.synthetic_fields()["overlap_upper_clamp"]
    path = tmp_path / "field.npz"
    digest = experiment.save_field_npz(path, field)
    assert digest == experiment.sha256_file(path)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        assert metadata["content_hash"] == field.content_hash()
        assert np.array_equal(archive["means"], field.means.numpy())
    with pytest.raises(FileExistsError):
        experiment.save_field_npz(path, field)


def test_worker_comparison_checks_projection_candidates_and_images():
    field = experiment.synthetic_fields()["overlap_upper_clamp"]
    native = _native_like_output(field)
    comparison = experiment.compare_worker_output(field, native)
    assert comparison["passed"] is True
    assert comparison["candidate_sets_exact"] is True
    assert comparison["radii_exact"] is True
    assert comparison["raw_max_abs"] == 0.0

    native["raw"] = np.asarray(native["raw"]).copy()
    native["raw"][0, 0, 0] += 0.1
    failed = experiment.compare_worker_output(field, native)
    assert failed["passed"] is False
    assert failed["raw_allclose"] is False


def test_atomic_json_creation_refuses_overwrite(tmp_path: Path):
    path = tmp_path / "artifact.json"
    digest = experiment._exclusive_json(path, {"status": "PASS"})
    assert digest == experiment.sha256_file(path)
    with pytest.raises(FileExistsError):
        experiment._exclusive_json(path, {"status": "FAIL"})
    assert experiment.strict_json_load(path) == {"status": "PASS"}


def test_nonfinite_native_metric_writes_terminal_failure_after_attempt(tmp_path: Path):
    attempt_path = tmp_path / "attempt.json"
    result_path = tmp_path / "result.json"
    attempt_sha256 = experiment._exclusive_json(attempt_path, {"status": "STARTED"})
    field = experiment.synthetic_fields()["overlap_upper_clamp"]
    native = _native_like_output(field)
    native["raw"] = np.asarray(native["raw"]).copy()
    native["raw"][0, 0, 0] = float("nan")
    comparison = experiment.compare_worker_output(field, native)
    assert np.isnan(comparison["raw_max_abs"])

    digest, written = experiment.write_terminal_result(
        result_path,
        {
            "artifact_type": "gaussianimage_plus_provider_parity_result",
            "status": "FAIL",
            "results": {"injected_native_nonfinite": comparison},
        },
        attempt_sha256=attempt_sha256,
        seal_file_sha256="1" * 64,
    )
    assert attempt_path.is_file()
    assert result_path.is_file()
    assert digest == experiment.sha256_file(result_path)
    assert written["status"] == "FAIL"
    assert written["error_type"] == "TerminalPayloadValidationError"
    assert experiment.strict_json_load(result_path) == written


def test_exact_command_rejects_extra_arguments(monkeypatch):
    monkeypatch.setattr(sys, "executable", str(experiment.ROOT / ".venv/bin/python"))
    monkeypatch.setattr(
        sys,
        "argv",
        [str(experiment.ROOT / experiment.HARNESS), "seal"],
    )
    experiment.assert_exact_command("seal")
    monkeypatch.setattr(
        sys,
        "argv",
        [str(experiment.ROOT / experiment.HARNESS), "seal", "--extra"],
    )
    with pytest.raises(RuntimeError, match="exact command"):
        experiment.assert_exact_command("seal")


def test_worker_environment_is_exact_and_drops_foreign_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/tmp/foreign")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/foreign.so")
    environment = experiment.worker_environment()
    assert environment["PYTHONPATH"] == str(experiment.EXTERNAL_REPO / "gsplat")
    assert environment["LD_PRELOAD"] == str(experiment.SYSTEM_LIBSTDCXX)
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"


def test_worker_source_is_render_only_and_has_explicit_binary_guard():
    source = (experiment.ROOT / experiment.WORKER).read_text(encoding="utf-8")
    assert "import gsplat.csrc as csrc" in source
    assert "expected_csrc_sha256" in source
    assert "expected_input_sha256" in source
    assert "torch.load" in source
    assert "rasterize_gaussians_plus" in source
    forbidden = (
        "import train",
        "SimpleTrainer2d",
        "loss.backward",
        "optimizer.step",
        "PIL",
        "Image.open",
        "cv2",
        "imageio",
    )
    assert not any(token in source for token in forbidden)


def test_checkpoint_hashing_and_decode_are_strictly_post_attempt_and_bound():
    command_source = inspect.getsource(experiment.command_run)
    assert command_source.index("_exclusive_json(ATTEMPT") < command_source.index(
        "execute_official_run(seal)"
    )

    pre_attempt_sources = (
        inspect.getsource(experiment.load_and_verify_seal),
        inspect.getsource(experiment.verify_external_bindings_pre_attempt),
        inspect.getsource(experiment.external_static_bindings),
    )
    for source in pre_attempt_sources:
        assert "sha256_file(REAL_CHECKPOINT)" not in source
        assert "REAL_CHECKPOINT.read_bytes" not in source
        assert "torch.load" not in source

    adapter_source = inspect.getsource(experiment.load_checkpoint_cpu)
    assert adapter_source.index("payload = path.read_bytes()") < adapter_source.index("torch.load")
    assert adapter_source.index("actual_sha256 != expected_sha256") < adapter_source.index(
        "torch.load"
    )

    worker_source = (experiment.ROOT / experiment.WORKER).read_text(encoding="utf-8")
    main_source = worker_source.split("def main(", maxsplit=1)[1]
    assert main_source.index("input_payload = args.input.read_bytes()") < main_source.index(
        "load_checkpoint("
    )
    assert main_source.index("input_sha256 != args.expected_input_sha256") < main_source.index(
        "load_checkpoint("
    )


def test_seal_scope_contains_only_new_experiment_files():
    assert experiment.SEALED_PATHS == (
        experiment.HARNESS,
        experiment.WORKER,
        experiment.TEST,
        experiment.PREREGISTRATION,
        experiment.IMPLEMENTATION_REVIEW,
        experiment.IMPLEMENTATION_REVIEW_ADDENDUM,
    )
    assert not any(str(path).startswith("src/") for path in experiment.SEALED_PATHS)
    assert not any("compact_occupancy" in str(path) for path in experiment.SEALED_PATHS)


def test_frozen_external_identifiers_are_literal_without_accessing_external_state():
    assert experiment.EXTERNAL_COMMIT == "549cfaab2b400248f685c12782a180f3cfc038b0"
    assert experiment.EXTERNAL_CSRC_SHA256 == (
        "9b57b7e0531a50d87c529d3541fbf370f9d85455836ac0cf5414c01ce48ac222"
    )
    assert experiment.REAL_CHECKPOINT_SHA256 == (
        "ad611facd72e813dece1b95c3268dbfd82f8af01cdb5ad67e1c7675cc670794b"
    )
    assert experiment.REAL_HEIGHT == 120
    assert experiment.REAL_WIDTH == 160
