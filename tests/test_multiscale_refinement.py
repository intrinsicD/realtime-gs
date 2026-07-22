"""Outcome-free CPU checks for the preregistered multiscale refinement seam."""

from __future__ import annotations

import copy
import json
import math
import statistics
from pathlib import Path

import benchmarks.multiscale_refinement_ablation as multiscale
import pytest
import torch

import rtgs.optim.trainer as trainer_module
from rtgs.core.camera import Camera
from rtgs.data.synthetic import make_synthetic_scene
from rtgs.optim.trainer import (
    TrainConfig,
    Trainer,
    TrainStepControl,
    area_downsample_2x,
    downscale_camera,
)


def _tiny_config(*, iterations: int = 4, seed: int = 17, densify: bool = False) -> TrainConfig:
    return TrainConfig(
        iterations=iterations,
        rasterizer="torch",
        device="cpu",
        densify=densify,
        target_sh_degree=0,
        sh_degree_interval=max(iterations, 1),
        ssim_lambda=0.0,
        use_masks=False,
        random_background=False,
        eval_every=max(iterations // 2, 1),
        quaternion_update_policy="current",
        seed=seed,
    )


def _assert_gaussians_bit_exact(left, right) -> None:
    for field in multiscale.GAUSSIAN_FIELDS:
        assert torch.equal(getattr(left, field), getattr(right, field)), field


def _established_history(history: dict) -> dict:
    return {
        key: value
        for key, value in history.items()
        if key not in {"elapsed", "step_control_metadata", "checkpoint_callback_seconds"}
    }


def test_none_and_all_unit_controls_preserve_established_training_bit_exactly():
    scene = make_synthetic_scene(n_gaussians=5, n_cameras=3, image_size=12, seed=17)
    initialization = scene.gt_gaussians.detach()
    initialization.means += 0.02
    config = _tiny_config()

    omitted_final, omitted_history = Trainer(config).train(scene, initialization.detach())
    none_final, none_history = Trainer(config).train(
        scene, initialization.detach(), step_controls=None
    )
    unit_controls = tuple(TrainStepControl(1, 1) for _ in range(config.iterations))
    unit_final, unit_history = Trainer(config).train(
        scene, initialization.detach(), step_controls=unit_controls
    )

    _assert_gaussians_bit_exact(omitted_final, none_final)
    _assert_gaussians_bit_exact(omitted_final, unit_final)
    assert _established_history(omitted_history) == _established_history(none_history)
    assert _established_history(omitted_history) == _established_history(unit_history)
    for key in (
        "loss",
        "loss_terms",
        "sampled_train_views",
        "psnr",
        "n_gaussians",
        "active_sh_degree",
    ):
        assert unit_history[key] == omitted_history[key]
    assert "step_control_metadata" not in omitted_history
    assert unit_history["step_control_metadata"]["render_pixels"] == 4 * 12 * 12
    assert unit_history["step_control_metadata"]["loss_pixels"] == 4 * 12 * 12


def test_area_pool_and_camera_downscale_match_frozen_math():
    image = torch.arange(8 * 6 * 3, dtype=torch.float32).reshape(8, 6, 3)
    pooled = area_downsample_2x(image)
    direct64 = image.double().reshape(4, 2, 3, 2, 3).mean(dim=(1, 3))
    assert pooled.shape == (4, 3, 3)
    assert pooled.dtype == image.dtype
    assert pooled.device == image.device
    assert torch.allclose(pooled.double(), direct64, atol=1e-7, rtol=1e-6)
    plane = area_downsample_2x(image[..., 0])
    assert torch.equal(plane, pooled[..., 0])

    camera = Camera.look_at(torch.tensor([2.0, 0.4, 1.2]), torch.zeros(3), width=16, height=12)
    half = downscale_camera(camera, 2)
    points = torch.tensor([[0.0, 0.0, 0.0], [0.2, -0.1, 0.1]], dtype=torch.float32)
    full_uv, full_depth = camera.project(points)
    half_uv, half_depth = half.project(points)
    assert torch.allclose(half_uv, full_uv / 2, atol=1e-6, rtol=1e-6)
    assert torch.equal(half_depth, full_depth)
    low_pixels = torch.tensor([[0.5, 0.5], [3.5, 4.5]], dtype=torch.float32)
    low_origin, low_rays = half.pixel_rays(low_pixels)
    full_origin, full_rays = camera.pixel_rays(2 * low_pixels)
    assert torch.allclose(low_origin, full_origin, atol=1e-6, rtol=1e-6)
    assert torch.allclose(low_rays, full_rays, atol=1e-6, rtol=1e-6)
    assert (half.width, half.height) == (8, 6)
    assert (half.fx, half.fy, half.cx, half.cy) == (
        camera.fx / 2,
        camera.fy / 2,
        camera.cx / 2,
        camera.cy / 2,
    )
    assert torch.equal(half.R, camera.R)
    assert torch.equal(half.t, camera.t)
    assert torch.equal(half.position, camera.position)


def test_frozen_schedules_exposures_and_view_rng_are_exact():
    schedules = multiscale.frozen_schedules()
    assert schedules["orders"] == {
        "3": list(multiscale.ARM_ORDERS[3]),
        "4": list(multiscale.ARM_ORDERS[4]),
        "5": list(multiscale.ARM_ORDERS[5]),
    }
    expected = {
        "full": (276480, 276480),
        "camera_blocked": (172800, 172800),
        "pyramid_blocked": (276480, 172800),
        "camera_interleaved": (172800, 172800),
    }
    for arm, counts in expected.items():
        record = schedules["arms"][arm]
        assert (record["exposure"]["render_pixels"], record["exposure"]["loss_pixels"]) == counts
        assert record["last_step"] == [1, 1]
        assert record["sha256"] == multiscale.canonical_json_hash(record["sequence"])
    assert schedules["arms"]["camera_blocked"]["render_scale_counts"] == {"1": 60, "2": 60}
    assert schedules["arms"]["camera_interleaved"]["render_scale_counts"] == {
        "1": 60,
        "2": 60,
    }
    expected_views = multiscale.probe_view_schedule(17, n_views=4, iterations=8)
    generator = torch.Generator().manual_seed(17)
    literal = [int(torch.randint(0, 4, (1,), generator=generator)) for _ in range(8)]
    assert expected_views == literal


def test_runtime_nine_view_pyramid_invariants_are_read_only():
    scene = make_synthetic_scene(n_gaussians=2, n_cameras=9, image_size=48, seed=26)
    scene.gt_depths = None
    scene.gt_gaussians = None
    before = multiscale.scene_hashes(scene)
    record = multiscale.verify_training_pyramid_invariants(scene)
    assert len(record["views"]) == 9
    assert record["test_point_count"] == 32
    assert record["maximum_projection_error"] <= 1e-6
    assert record["maximum_ray_error"] <= 1e-6
    assert record["direct_float64_pool_parity"] is True
    assert record["maximum_direct_float64_pool_error"] <= 1e-6
    assert all(item["direct_float64_pool_parity"] is True for item in record["views"])
    assert all(item["maximum_direct_float64_pool_error"] <= 1e-6 for item in record["views"])
    assert multiscale.scene_hashes(scene) == before


def test_all_four_toy_arms_keep_the_same_sampled_view_schedule():
    scene = make_synthetic_scene(n_gaussians=4, n_cameras=4, image_size=12, seed=18)
    initialization = scene.gt_gaussians.detach()
    initialization.means += 0.015
    config = _tiny_config(seed=19)
    full = TrainStepControl(1, 1)
    camera = TrainStepControl(2, 2)
    pyramid = TrainStepControl(1, 2)
    controls = {
        "full": None,
        "camera_blocked": (camera, camera, full, full),
        "pyramid_blocked": (pyramid, pyramid, full, full),
        "camera_interleaved": (camera, full, camera, full),
    }
    schedules = {}
    for arm, sequence in controls.items():
        _, history = Trainer(config).train(scene, initialization.detach(), step_controls=sequence)
        schedules[arm] = history["sampled_train_views"]
    assert all(schedule == schedules["full"] for schedule in schedules.values())


def test_resolution_transition_keeps_optimizer_state_and_full_resolution_observers(
    monkeypatch,
):
    scene = make_synthetic_scene(n_gaussians=4, n_cameras=3, image_size=12, seed=20)
    initialization = scene.gt_gaussians.detach()
    initialization.means += 0.01
    config = _tiny_config(seed=21)
    controls = (
        TrainStepControl(2, 2),
        TrainStepControl(2, 2),
        TrainStepControl(1, 1),
        TrainStepControl(1, 1),
    )
    baseline_final, baseline_history = Trainer(config).train(
        scene, initialization.detach(), step_controls=controls
    )

    real_adam = torch.optim.Adam
    optimizers = []

    class TrackingAdam(real_adam):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.parameter_ids = []
            optimizers.append(self)

        def step(self, closure=None):
            self.parameter_ids.append(id(self.param_groups[0]["params"][0]))
            return super().step(closure)

    monkeypatch.setattr(torch.optim, "Adam", TrackingAdam)
    native_dimensions = []
    original_evaluate = Trainer.evaluate

    def observe_native(scene_arg, gaussians, renderer=None, indices=None):
        native_dimensions.append([(camera.width, camera.height) for camera in scene_arg.cameras])
        return original_evaluate(scene_arg, gaussians, renderer, indices)

    monkeypatch.setattr(Trainer, "evaluate", staticmethod(observe_native))
    clock = {"now": 0.0}
    monkeypatch.setattr(trainer_module.time, "perf_counter", lambda: clock["now"])
    callback_steps = []

    def delayed_mutating_callback(snapshot, step):
        callback_steps.append(step)
        snapshot.means.fill_(float("nan"))
        clock["now"] += 100.0

    observed_final, observed_history = Trainer(config).train(
        scene,
        initialization.detach(),
        step_controls=controls,
        checkpoint_callback=delayed_mutating_callback,
    )
    _assert_gaussians_bit_exact(baseline_final, observed_final)
    assert _established_history(baseline_history) == _established_history(observed_history)
    assert baseline_history["checkpoint_callback_seconds"] == 0.0
    assert observed_history["checkpoint_callback_seconds"] == 200.0
    assert callback_steps == [2, 4]
    assert observed_history["elapsed"] == [(2, 0.0), (4, 0.0)]
    assert len(optimizers) == 6
    assert all(len(optimizer.parameter_ids) == 4 for optimizer in optimizers)
    assert all(len(set(optimizer.parameter_ids)) == 1 for optimizer in optimizers)
    assert native_dimensions == [[(12, 12)] * 3, [(12, 12)] * 3]
    assert observed_final.n == initialization.n
    assert observed_final.sh_degree == 0


@pytest.mark.parametrize(
    ("scene_size", "densify", "controls", "message"),
    [
        (12, False, (TrainStepControl(1, 1),), "length"),
        (12, False, (TrainStepControl(3, 3),) * 2, "render_downscale"),
        (12, False, (TrainStepControl(2, 1),) * 2, "multiple"),
        (15, False, (TrainStepControl(2, 2),) * 2, "even"),
        (12, True, (TrainStepControl(2, 2),) * 2, "density control"),
    ],
)
def test_invalid_step_controls_fail_before_optimization(scene_size, densify, controls, message):
    scene = make_synthetic_scene(n_gaussians=3, n_cameras=2, image_size=scene_size, seed=23)
    config = _tiny_config(iterations=2, seed=24, densify=densify)
    with pytest.raises((ValueError, TypeError), match=message):
        Trainer(config).train(scene, scene.gt_gaussians.detach(), step_controls=controls)


def test_full_resolution_heldout_observer_records_exact_raw_evidence():
    scene = make_synthetic_scene(n_gaussians=3, n_cameras=12, image_size=48, seed=25)
    truths = multiscale.construct_truth(scene)
    _, extent = scene.center_and_extent()
    record = multiscale.evaluate_checkpoint(
        scene.gt_gaussians.detach(), truths, extent=extent, step=0
    )
    multiscale.validate_checkpoint_record(record)
    assert [item["view"] for item in record["per_view"]] == list(multiscale.TEST_INDICES)
    assert all(
        item["full_rgb_value_count"] == 48 * 48 * 3
        for item in [view["evidence"] for view in record["per_view"]]
    )
    assert record["active_sh_degree"] == 0
    assert record["primitive_count"] == scene.gt_gaussians.n


def test_metric_evidence_uses_raw_unclamped_color_values():
    target = torch.zeros(48, 48, 3, dtype=torch.float32)
    predicted = torch.full_like(target, 1.5)
    alpha = torch.ones(48, 48, dtype=torch.float32)
    depth = torch.full((48, 48), 2.0, dtype=torch.float32)
    support = torch.ones(48, 48, dtype=torch.bool)
    truth = multiscale.TruthView(
        view=3,
        target=target,
        camera=Camera.look_at(torch.tensor([0.0, 0.0, 2.0]), torch.zeros(3), width=48, height=48),
        color=target,
        alpha=alpha,
        depth=depth,
        support=support,
        record={"aggregate_sha256": "toy"},
    )
    evidence, metrics = multiscale._metric_evidence(predicted, alpha, depth, truth, extent=2.0)
    expected_sse = 48 * 48 * 3 * 1.5**2
    assert evidence["full_rgb_squared_error_sum"] == expected_sse
    assert evidence["foreground_rgb_squared_error_sum"] == expected_sse
    assert metrics["psnr_full"] == pytest.approx(-10 * math.log10(1.5**2))


def _toy_checkpoint(step: int, mse: float = 0.01) -> dict:
    evidence = {
        "foreground_rgb_squared_error_sum": mse * 3,
        "foreground_rgb_value_count": 3,
        "full_rgb_squared_error_sum": mse * 48 * 48 * 3,
        "full_rgb_value_count": 48 * 48 * 3,
        "crop_rgb_squared_error_sum": mse * 3,
        "crop_rgb_value_count": 3,
        "depth_squared_error_sum": 0.01,
        "depth_intersection_pixel_count": 1,
        "alpha_intersection_pixel_count": 1,
        "alpha_union_pixel_count": 1,
        "truth_support_pixel_count": 1,
        "extent": 1.0,
        "ssim_crop": 0.9,
    }
    metrics = multiscale.recompute_metrics_from_evidence(evidence)
    views = []
    for view in multiscale.TEST_INDICES:
        item = {
            "view": view,
            "step": step,
            "primitive_count": 1,
            "crop_bounds": [0, 1, 0, 1],
            "evidence": copy.deepcopy(evidence),
            "metrics": dict(metrics),
        }
        item["record_sha256"] = multiscale.canonical_json_hash(item)
        views.append(item)
    checkpoint = {
        "step": step,
        "per_view": views,
        "mean": {
            metric: statistics.fmean(float(item["metrics"][metric]) for item in views)
            for metric in multiscale.METRICS
        },
        "primitive_count": 1,
        "active_sh_degree": 0,
    }
    checkpoint["record_sha256"] = multiscale.canonical_json_hash(checkpoint)
    return checkpoint


def test_runtime_exposure_accounting_separates_every_render_scope():
    step_zero = _toy_checkpoint(0)
    sampled_views = [step % len(multiscale.TRAIN_INDICES) for step in range(120)]
    arms = []
    for arm in multiscale.ARMS:
        controls = multiscale.control_sequence(arm)
        history = {
            "sampled_train_views": list(sampled_views),
            "psnr": [(step, 0.0) for step in (30, 60, 90, 120)],
        }
        if arm != "full":
            exposure = multiscale.exposure_record(arm)
            history["step_control_metadata"] = {
                "sequence_sha256": multiscale.schedule_record(arm)["sha256"],
                "render_pixels": exposure["render_pixels"],
                "loss_pixels": exposure["loss_pixels"],
                "per_view_scale_counts": multiscale.per_view_scale_counts(controls, sampled_views),
            }
        arms.append(
            {
                "arm": arm,
                "history": history,
                "checkpoints": [
                    step_zero,
                    *[_toy_checkpoint(step) for step in (30, 60, 90, 120)],
                ],
            }
        )

    record = multiscale.runtime_exposure_accounting(arms, step_zero)
    expected_optimization = {
        "full": (276480, 276480, 1.0),
        "camera_blocked": (172800, 172800, 0.625),
        "pyramid_blocked": (276480, 172800, 1.0),
        "camera_interleaved": (172800, 172800, 0.625),
    }
    for arm, expected in expected_optimization.items():
        arm_record = record["by_arm"][arm]
        optimization = arm_record["optimization"]
        assert (
            optimization["render_pixels"],
            optimization["loss_pixels"],
            optimization["render_ratio"],
        ) == expected
        assert optimization["included_in_optimization_ratio"] is True
        native = arm_record["native_training_evaluation"]
        heldout = arm_record["held_out_checkpoint_callbacks"]
        assert native["render_pixels"] == 82944
        assert heldout["render_pixels"] == 27648
        assert native["included_in_optimization_ratio"] is False
        assert heldout["included_in_optimization_ratio"] is False
    assert record["shared_manual_step_zero"]["render_pixels"] == 6912
    assert record["shared_manual_step_zero"]["included_in_optimization_ratio"] is False

    tampered = copy.deepcopy(arms)
    tampered[1]["history"]["step_control_metadata"]["render_pixels"] += 1
    with pytest.raises(RuntimeError, match="Trainer render exposure"):
        multiscale.runtime_exposure_accounting(tampered, step_zero)


def test_paired_delta_boundary_uses_mean_of_seed_differences():
    baseline = [0.2, 0.7, 0.3]
    candidate = [0.17999999999999997, 0.6799999999999999, 0.2800000000000001]
    paired_mean = statistics.fmean(
        candidate_value - baseline_value
        for candidate_value, baseline_value in zip(candidate, baseline, strict=True)
    )
    separate_means = statistics.fmean(candidate) - statistics.fmean(baseline)
    assert paired_mean >= -0.02
    assert separate_means < -0.02

    by_seed = {}
    for index, seed in enumerate(multiscale.SEEDS):
        by_seed[str(seed)] = {
            "full": {
                "final": {
                    "psnr_fg": 20.0,
                    "ssim_crop": 0.9,
                    "depth_rmse_over_extent": 0.1,
                    "alpha_iou": baseline[index],
                    "foreground_coverage": baseline[index],
                },
                "foreground_psnr_auc_db": 20.0,
            },
            "camera_blocked": {
                "final": {
                    "psnr_fg": 20.0,
                    "ssim_crop": 0.9,
                    "depth_rmse_over_extent": 0.1,
                    "alpha_iou": candidate[index],
                    "foreground_coverage": candidate[index],
                },
                "foreground_psnr_auc_db": 20.0,
            },
        }
    decision = multiscale._candidate_decision({"by_seed": by_seed}, "camera_blocked")
    expected_deltas = [
        candidate_value - baseline_value
        for candidate_value, baseline_value in zip(candidate, baseline, strict=True)
    ]
    assert decision["final_alpha_iou_deltas"] == expected_deltas
    assert decision["final_foreground_coverage_deltas"] == expected_deltas
    assert decision["mean_final_alpha_iou_delta"] == paired_mean
    assert decision["mean_final_foreground_coverage_delta"] == paired_mean
    criteria = decision["quality_noninferiority_criteria"]
    assert criteria["mean_final_alpha_iou_delta_at_least_minus_0_02"] is True
    assert criteria["mean_final_coverage_delta_at_least_minus_0_02"] is True


def test_raw_numerator_auc_and_decision_tampering_are_rejected():
    checkpoint = _toy_checkpoint(0)
    multiscale.validate_checkpoint_record(checkpoint)
    tampered = copy.deepcopy(checkpoint)
    tampered["per_view"][0]["evidence"]["foreground_rgb_squared_error_sum"] *= 2
    item = tampered["per_view"][0]
    item["record_sha256"] = multiscale.canonical_json_hash(
        {key: value for key, value in item.items() if key != "record_sha256"}
    )
    tampered["record_sha256"] = multiscale.canonical_json_hash(
        {key: value for key, value in tampered.items() if key != "record_sha256"}
    )
    with pytest.raises(ValueError, match="raw evidence"):
        multiscale.validate_checkpoint_record(tampered)

    checkpoints = [
        _toy_checkpoint(step, mse=0.01 + index * 0.001)
        for index, step in enumerate(multiscale.CHECKPOINT_STEPS)
    ]
    auc = multiscale.normalized_auc(checkpoints, "psnr_fg")
    multiscale.validate_reported_auc(checkpoints, "psnr_fg", auc)
    with pytest.raises(ValueError, match="AUC"):
        multiscale.validate_reported_auc(checkpoints, "psnr_fg", auc + 0.01)

    by_seed = {}
    for seed in multiscale.SEEDS:
        by_seed[str(seed)] = {}
        for arm in multiscale.ARMS:
            exposure = multiscale.exposure_record(arm)
            by_seed[str(seed)][arm] = {
                "final": {
                    "psnr_fg": 20.0,
                    "ssim_crop": 0.9,
                    "depth_rmse_over_extent": 0.1,
                    "alpha_iou": 0.8,
                    "foreground_coverage": 0.85,
                },
                "foreground_psnr_auc_db": 20.0,
                "render_pixels": exposure["render_pixels"],
                "loss_pixels": exposure["loss_pixels"],
                "render_ratio": exposure["render_ratio"],
            }
    summary = {"by_seed": by_seed}
    decisions = multiscale.frozen_decisions(summary)
    multiscale.validate_reported_decisions(summary, decisions)
    decisions["candidates"]["camera_blocked"]["quality_improvement"] = True
    with pytest.raises(ValueError, match="decisions"):
        multiscale.validate_reported_decisions(summary, decisions)


def _passing_verification_record() -> dict:
    commands = []
    for command in multiscale.verification_commands():
        stdout = "passed\n"
        stderr = ""
        commands.append(
            {
                "command": list(command),
                "returncode": 0,
                "seconds": 0.0,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_sha256": multiscale.sha256_bytes(stdout.encode()),
                "stderr_sha256": multiscale.sha256_bytes(stderr.encode()),
            }
        )
    return {"passed": True, "commands": commands}


def _patch_temporary_seal_context(monkeypatch, tmp_path):
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    seal_path = tmp_path / "fixed-seal.json"
    monkeypatch.setattr(multiscale, "ROOT", tmp_path)
    monkeypatch.setattr(multiscale, "SEALED_PATHS", (Path("source.py"),))
    monkeypatch.setattr(multiscale, "DEFAULT_SEAL", seal_path)
    monkeypatch.setattr(
        multiscale,
        "verify_preregistration",
        lambda: {"path": "prereg.md", "sha256": "prereg"},
    )
    monkeypatch.setattr(
        multiscale,
        "verify_implementation_review",
        lambda: {"path": "review.md", "sha256": "review"},
    )
    monkeypatch.setattr(multiscale, "verify_default_seam", lambda: {"default": "frozen"})
    monkeypatch.setattr(multiscale, "environment_metadata", lambda: {"environment": "cpu"})
    monkeypatch.setattr(multiscale, "assert_official_environment", lambda _value: None)
    monkeypatch.setattr(
        multiscale,
        "environment_fingerprint",
        lambda value: multiscale.canonical_json_hash(value),
    )
    monkeypatch.setattr(multiscale, "git_metadata", lambda: {"commit": "test"})
    return source, seal_path


def test_seal_refuses_source_drift_during_mocked_verification(monkeypatch, tmp_path):
    source, _ = _patch_temporary_seal_context(monkeypatch, tmp_path)

    def mutating_verification():
        source.write_text("VALUE = 2\n", encoding="utf-8")
        return _passing_verification_record()

    monkeypatch.setattr(multiscale, "run_verification", mutating_verification)
    with pytest.raises(RuntimeError, match="snapshot drifted.*source_hashes"):
        multiscale.create_seal()


def test_seal_source_verification_and_exact_path_tampering_fail_closed(monkeypatch, tmp_path):
    _, seal_path = _patch_temporary_seal_context(monkeypatch, tmp_path)
    monkeypatch.setattr(multiscale, "run_verification", lambda: _passing_verification_record())
    payload = multiscale.create_seal()
    seal_path.write_text(json.dumps(payload), encoding="utf-8")
    binding = multiscale.load_and_verify_seal(seal_path)
    assert binding["path"] == str(seal_path.resolve())

    copied_path = tmp_path / "copied-seal.json"
    copied_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed preregistered seal path"):
        multiscale.load_and_verify_seal(copied_path)

    source_tampered = copy.deepcopy(payload)
    source_tampered["source_hashes"]["source.py"] = "0" * 64
    seal_path.write_text(json.dumps(source_tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source-hashes binding"):
        multiscale.load_and_verify_seal(seal_path)

    verification_tampered = copy.deepcopy(payload)
    verification_tampered["verification"]["commands"][0]["stdout"] = "tampered\n"
    seal_path.write_text(json.dumps(verification_tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stdout hash differs"):
        multiscale.load_and_verify_seal(seal_path)


def test_schedule_source_manifest_and_once_only_marker_tampering_fail_closed(tmp_path):
    payload = {
        "artifact_type": multiscale.ARTIFACT_TYPE,
        "schedules": multiscale.frozen_schedules(),
    }
    payload["schedules"]["arms"]["camera_blocked"]["sequence"][0] = [1, 1]
    with pytest.raises(ValueError, match="schedules"):
        multiscale.validate_result_payload(payload)

    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    paths = (Path("source.py"),)
    hashes, aggregate = multiscale.source_hashes(paths, root=tmp_path)
    multiscale.verify_source_manifest(paths, hashes, aggregate, root=tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source manifest"):
        multiscale.verify_source_manifest(paths, hashes, aggregate, root=tmp_path)

    marker = tmp_path / "attempt.json"
    seal = {"sha256": "seal", "source_aggregate": "aggregate"}
    output = tmp_path / "result.json"
    binding = multiscale.claim_attempt(marker, output=output, seal=seal)
    assert multiscale.verify_attempt_marker(marker, binding) == binding
    first = marker.read_bytes()
    with pytest.raises(RuntimeError, match="already claimed"):
        multiscale.claim_attempt(marker, output=output, seal=seal)
    assert marker.read_bytes() == first
    marker_payload = multiscale.strict_json_load(marker)
    marker_payload["output"] = str(tmp_path / "tampered.json")
    marker.write_text(json.dumps(marker_payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="payload or digest changed"):
        multiscale.verify_attempt_marker(marker, binding)
