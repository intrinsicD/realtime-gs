#!/usr/bin/env python3
"""Independent artifact audit for the Janelle Beam Fusion covariance-refit screen."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.compact_views import CompactDataset
from rtgs.data.reconstruction_inputs import ReconstructionInputs
from rtgs.lift.beam_fusion import (
    BeamFusionConfig,
    _component_covariances_2d,
    fuse_gaussian_beams,
)
from rtgs.lift.splat_sfm import _spd_project, _triangulate_covariances

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d"
PROTOCOL = ROOT / "benchmarks/results/20260723_beam_covariance_refit_PREREG.md"
HARNESS = ROOT / "benchmarks/beam_covariance_refit.py"
SYNTHETIC_TEST = ROOT / "tests/test_beam_covariance_refit.py"
IMPORTED_HELPER = ROOT / "benchmarks/beam_convergence_dynamics.py"
POSTRUN_FIX = ROOT / "benchmarks/results/20260723_beam_covariance_refit_POSTRUN_FIX.patch"
DEFAULT_PRIMARY = ROOT / "runs/beam_covariance_refit_20260723"
DEFAULT_REPEAT = ROOT / "runs/beam_covariance_refit_repeat_20260723"
DEFAULT_OUTPUT = ROOT / "benchmarks/results/20260723_beam_covariance_refit_AUDIT.json"

ARMS = ("ci", "track-lsq", "track-robust")
SELECTED_VIEWS = (0, 3, 6, 9, 12, 15, 18, 21)
EXPECTED_STEPS = tuple(range(25, 1_001, 25))
EXPECTED_REVISION = "c2a7e120a5cafdcf22d4bff6f5b9868b860eb1df"
EXPECTED_PROTOCOL_SHA256 = "9287b25650d9d4a4f42442ef24dce6e0b2c658f04b47430eca2975ca31835c22"
EXPECTED_HARNESS_SHA256 = "ebae0f5c6697690d3c8c92bc3657dffe1e6206a32970e5894e86fb603bc93336"
CURRENT_HARNESS_SHA256 = "8eb11a50fa9055578139985350ece981861c717e81317445863c9233c890995e"
EXPECTED_TEST_SHA256 = "75db5bf9f56ebb19db6a664e43bac9e8d4b8fb0c475fe652c48d6427c23efb51"
POST_RUN_HELPER_SHA256 = "6521af11d0af8513cd6963de260786e37c9791506a0782619b0561045fe2ffa9"
POSTRUN_FIX_SHA256 = "2c6a82a169fd2925d7b208a304f20724f2f942a44058c8e10d8c9539f7c8bf03"
EXPECTED_MANIFEST_SHA256 = "b1c8e256d73e2c05f3cb4797a615bdbb2639a637f12908a5c96a2a9a9f912847"


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, condition: bool, detail: Any) -> None:
        self.checks.append({"name": name, "passed": bool(condition), "detail": detail})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reconstructed_executed_harness_sha256() -> str:
    current = HARNESS.read_bytes()
    fixed = b'f"../../runs/{out.name}/{name}/gaussians_'
    executed = b'f"../../../runs/{out.name}/{name}/gaussians_'
    if current.count(fixed) != 2:
        raise RuntimeError("cannot reconstruct the two-line post-run viewer-path correction")
    return hashlib.sha256(current.replace(fixed, executed)).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _strict_json(path: Path) -> bool:
    def reject(value: str) -> None:
        raise ValueError(value)

    try:
        json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    except ValueError:
        return False
    return True


def _without_timings(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_timings(item) for key, item in value.items() if key != "elapsed_seconds"
        }
    if isinstance(value, list):
        return [_without_timings(item) for item in value]
    return value


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().to(torch.float64)
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def _close_dict(first: dict[str, float], second: dict[str, float]) -> bool:
    return all(
        math.isclose(
            float(first[key]),
            float(second[key]),
            rel_tol=2e-5,
            abs_tol=2e-6,
        )
        for key in first
    )


def _selected_inputs(dataset: CompactDataset) -> ReconstructionInputs:
    complete = dataset.to_reconstruction_inputs()
    return ReconstructionInputs(
        observations=[complete.observations[index] for index in SELECTED_VIEWS],
        cameras=[complete.cameras[index] for index in SELECTED_VIEWS],
        view_names=[complete.view_names[index] for index in SELECTED_VIEWS],
        bounds_hint=complete.bounds_hint,
        name=f"{complete.name}-covariance-audit",
    )


def _beam_result(dataset: CompactDataset):
    config = BeamFusionConfig(
        min_views=3,
        transverse_gate_sigma=3.0,
        max_color_distance=0.35,
        color_sigma=0.25,
        fold_in_gate_sigma=3.0,
        nms_voxel_size=float(dataset.bounds_hint[1]) / 100.0,
        init_opacity=0.10,
        source_chunk=256,
        max_components=800,
        seed_budget_multiplier=4,
    )
    inputs = _selected_inputs(dataset)
    return inputs, fuse_gaussian_beams(inputs, config)


def _member_data(result, inputs) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros(result.n_components, inputs.n_views, dtype=torch.bool)
    covariances = torch.zeros(
        result.n_components,
        inputs.n_views,
        2,
        2,
        dtype=torch.float64,
    )
    per_view = [_component_covariances_2d(observation) for observation in inputs.observations]
    offsets = result.component_offsets.tolist()
    for component, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        views = result.contributor_view_indices[start:end]
        splats = result.contributor_component_indices[start:end]
        if views.unique().numel() != views.numel():
            raise RuntimeError(f"duplicate contributor view in component {component}")
        mask[component, views] = True
        for view, splat in zip(views.tolist(), splats.tolist(), strict=True):
            covariances[component, view] = per_view[view][splat]
    return mask, covariances


def _jacobians(points_world: torch.Tensor, cameras) -> torch.Tensor:
    points = points_world.to(torch.float64)
    result = torch.zeros(points.shape[0], len(cameras), 2, 3, dtype=torch.float64)
    for view_index, camera in enumerate(cameras):
        camera_points = camera.world_to_cam(points)
        z = camera_points[:, 2].clamp_min(1e-8)
        camera_jacobian = torch.zeros(points.shape[0], 2, 3, dtype=torch.float64)
        camera_jacobian[:, 0, 0] = camera.fx / z
        camera_jacobian[:, 0, 2] = -camera.fx * camera_points[:, 0] / z.square()
        camera_jacobian[:, 1, 1] = camera.fy / z
        camera_jacobian[:, 1, 2] = -camera.fy * camera_points[:, 1] / z.square()
        result[:, view_index] = camera_jacobian @ camera.R.to(torch.float64)
    return result


def _covariance_residuals(
    jacobians: torch.Tensor,
    covariances: torch.Tensor,
    mask: torch.Tensor,
    observed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted = jacobians @ covariances[:, None].to(torch.float64) @ jacobians.transpose(-1, -2)
    relative = (
        (predicted - observed).norm(dim=(-2, -1)) / observed.norm(dim=(-2, -1)).clamp_min(1e-12)
    )[mask]
    eigenvalues, eigenvectors = torch.linalg.eigh(observed)
    inverse_sqrt = (
        eigenvectors
        @ torch.diag_embed(eigenvalues.clamp_min(1e-12).rsqrt())
        @ eigenvectors.transpose(-1, -2)
    )
    normalized = inverse_sqrt @ predicted @ inverse_sqrt
    identity = torch.eye(2, dtype=torch.float64)
    whitened = (normalized - identity).square().mean(dim=(-2, -1)).sqrt()[mask]
    return relative, whitened


def _finite(model: Gaussians3D) -> bool:
    return all(
        bool(torch.isfinite(value).all())
        for value in (model.means, model.quats, model.log_scales, model.opacity, model.sh)
    )


def _trajectory(directory: Path, arm: str) -> tuple[dict[str, Any], list[tuple[int, float]]]:
    record = _read(directory / arm / "dynamics.json")
    points = [(0, float(record["init_metrics"]["psnr_fg"]))]
    points.extend((int(row["step"]), float(row["metric_psnr_fg"])) for row in record["curve"])
    return record, points


def _auc(points: list[tuple[int, float]]) -> float:
    return sum(
        (right_step - left_step) * (left_value + right_value) / 2.0
        for (left_step, left_value), (right_step, right_value) in zip(
            points[:-1], points[1:], strict=True
        )
    )


def _artifact_manifest(directory: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def _audit_dataset(audit: Audit) -> dict[str, Any]:
    manifest_path = DATASET / "manifest.json"
    manifest = _read(manifest_path)
    audit.check(
        "compact manifest hash",
        _sha256(manifest_path) == EXPECTED_MANIFEST_SHA256,
        _sha256(manifest_path),
    )
    files = []
    for entry in manifest["views"]:
        path = DATASET / entry["path"]
        actual = _sha256(path)
        audit.check(
            f"compact payload hash: {entry['view_id']}",
            actual == entry["sha256"] and path.stat().st_size == entry["bytes"],
            {"sha256": actual, "bytes": path.stat().st_size},
        )
        files.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": actual,
            }
        )
    return {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": _sha256(manifest_path),
        "semantic_digest": manifest["semantic_digest"],
        "calibration_sha256": manifest["calibration_sha256"],
        "files": files,
    }


def build(primary: Path, repeat: Path) -> dict[str, Any]:
    audit = Audit()
    summary = _read(primary / "summary.json")
    repeat_summary = _read(repeat / "summary.json")
    audit.check("primary summary strict JSON", _strict_json(primary / "summary.json"), None)
    audit.check("repeat summary strict JSON", _strict_json(repeat / "summary.json"), None)
    audit.check(
        "summary schema and status",
        summary.get("schema") == "rtgs.beam_covariance_refit.v1"
        and summary.get("status") == "complete",
        {"schema": summary.get("schema"), "status": summary.get("status")},
    )
    expected_parameters = {
        "selected_global_views": list(SELECTED_VIEWS),
        "evaluation_local_views": [0, 2, 4, 6],
        "downscale": 32,
        "n_init": 800,
        "iterations": 1_000,
        "fixed_topology": True,
        "seed": 0,
    }
    actual_parameters = {key: summary.get(key) for key in expected_parameters}
    audit.check(
        "frozen common parameters",
        actual_parameters == expected_parameters,
        actual_parameters,
    )
    audit.check(
        "arm order and completeness",
        tuple(summary["arms"]) == ARMS,
        list(summary["arms"]),
    )
    audit.check(
        "protocol hash bound in primary summary",
        summary["protocol"]["sha256"] == EXPECTED_PROTOCOL_SHA256
        and _sha256(PROTOCOL) == EXPECTED_PROTOCOL_SHA256,
        {
            "summary": summary["protocol"]["sha256"],
            "actual": _sha256(PROTOCOL),
        },
    )
    audit.check(
        "post-run viewer-only harness correction bound",
        _sha256(HARNESS) == CURRENT_HARNESS_SHA256 and _sha256(POSTRUN_FIX) == POSTRUN_FIX_SHA256,
        {
            "current_harness": _sha256(HARNESS),
            "postrun_fix": _sha256(POSTRUN_FIX),
        },
    )
    audit.check(
        "exact executed harness reconstructed from current source and reverse patch",
        _reconstructed_executed_harness_sha256() == EXPECTED_HARNESS_SHA256,
        _reconstructed_executed_harness_sha256(),
    )
    audit.check(
        "preregistered synthetic-test hash preserved",
        _sha256(SYNTHETIC_TEST) == EXPECTED_TEST_SHA256,
        _sha256(SYNTHETIC_TEST),
    )
    audit.check(
        "post-run imported-helper hash preserved",
        _sha256(IMPORTED_HELPER) == POST_RUN_HELPER_SHA256,
        _sha256(IMPORTED_HELPER),
    )

    input_binding = _audit_dataset(audit)
    dataset = CompactDataset.load(DATASET, device="cpu")
    inputs, beam = _beam_result(dataset)
    audit.check(
        "deterministic Beam Fusion replay count",
        beam.n_components == 800,
        beam.n_components,
    )
    member_mask, observed_covariances = _member_data(beam, inputs)
    audit.check(
        "contributor-link count replay",
        int(member_mask.sum()) == int(summary["covariance"]["n_observation_links"]),
        {
            "recomputed": int(member_mask.sum()),
            "summary": int(summary["covariance"]["n_observation_links"]),
        },
    )
    unique_links = set(
        zip(
            beam.contributor_view_indices.tolist(),
            beam.contributor_component_indices.tolist(),
            strict=True,
        )
    )
    unique_per_view = [
        len({component for view, component in unique_links if view == view_index})
        for view_index in range(inputs.n_views)
    ]

    dynamics: dict[str, dict[str, Any]] = {}
    trajectories: dict[str, list[tuple[int, float]]] = {}
    primary_models: dict[str, Gaussians3D] = {}
    repeat_models: dict[str, Gaussians3D] = {}
    for arm in ARMS:
        record, points = _trajectory(primary, arm)
        repeat_record, repeat_points = _trajectory(repeat, arm)
        dynamics[arm], trajectories[arm] = record, points
        audit.check(
            f"{arm}: complete primary checkpoint grid",
            tuple(int(row["step"]) for row in record["curve"]) == EXPECTED_STEPS,
            [row["step"] for row in record["curve"]],
        )
        audit.check(
            f"{arm}: complete repeat checkpoint grid",
            tuple(int(row["step"]) for row in repeat_record["curve"]) == EXPECTED_STEPS,
            [row["step"] for row in repeat_record["curve"]],
        )
        audit.check(
            f"{arm}: deterministic scientific dynamics repeat",
            _canonical_digest(_without_timings(record))
            == _canonical_digest(_without_timings(repeat_record)),
            {
                "primary": _canonical_digest(_without_timings(record)),
                "repeat": _canonical_digest(_without_timings(repeat_record)),
            },
        )
        audit.check(
            f"{arm}: repeated PSNR trajectory",
            points == repeat_points,
            {"primary_endpoint": points[-1], "repeat_endpoint": repeat_points[-1]},
        )
        primary_models[arm] = Gaussians3D.load_ply(primary / arm / "gaussians_init.ply")
        repeat_models[arm] = Gaussians3D.load_ply(repeat / arm / "gaussians_init.ply")
        for state in ("gaussians_init.ply", "gaussians_final.ply"):
            first_path = primary / arm / state
            second_path = repeat / arm / state
            first = Gaussians3D.load_ply(first_path)
            second = Gaussians3D.load_ply(second_path)
            audit.check(
                f"{arm}/{state}: finite fixed-count primary artifact",
                first.n == 800
                and _finite(first)
                and bool(((first.opacity >= 0) & (first.opacity <= 1)).all()),
                {"count": first.n},
            )
            audit.check(
                f"{arm}/{state}: exact repeat PLY",
                second.n == 800 and _finite(second) and _sha256(first_path) == _sha256(second_path),
                {"primary": _sha256(first_path), "repeat": _sha256(second_path)},
            )

    audit.check(
        "timing-free primary and repeat summaries are exact",
        _canonical_digest(_without_timings(summary))
        == _canonical_digest(_without_timings(repeat_summary)),
        {
            "primary": _canonical_digest(_without_timings(summary)),
            "repeat": _canonical_digest(_without_timings(repeat_summary)),
        },
    )
    replay_fields = {
        field: torch.equal(
            getattr(primary_models["ci"], field),
            getattr(beam.gaussians, field),
        )
        for field in ("means", "quats", "log_scales", "opacity", "sh")
    }
    audit.check(
        "saved CI initialization equals deterministic Beam replay",
        all(replay_fields.values()),
        replay_fields,
    )
    frozen_fields = {
        arm: {
            field: torch.equal(
                getattr(primary_models["ci"], field),
                getattr(primary_models[arm], field),
            )
            for field in ("means", "opacity", "sh")
        }
        for arm in ARMS
    }
    audit.check(
        "all non-covariance initialization fields bit-exact",
        all(all(fields.values()) for fields in frozen_fields.values()),
        frozen_fields,
    )

    jacobians = _jacobians(primary_models["ci"].means, inputs.cameras)
    covariance_diagnostics: dict[str, Any] = {}
    for arm in ARMS:
        covariance = primary_models[arm].covariance().to(torch.float64)
        relative, whitened = _covariance_residuals(
            jacobians,
            covariance,
            member_mask,
            observed_covariances,
        )
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
        recomputed = {
            "relative_frobenius_residual": _quantiles(relative),
            "whitened_rms_residual": _quantiles(whitened),
            "sigma_all_axes": _quantiles(eigenvalues.sqrt().reshape(-1)),
        }
        recorded = summary["covariance"]["arms"][arm]
        audit.check(
            f"{arm}: independently recomputed relative residual",
            _close_dict(
                recomputed["relative_frobenius_residual"],
                recorded["relative_frobenius_residual"],
            ),
            {
                "recomputed": recomputed["relative_frobenius_residual"],
                "recorded": recorded["relative_frobenius_residual"],
            },
        )
        audit.check(
            f"{arm}: independently recomputed whitened residual",
            _close_dict(
                recomputed["whitened_rms_residual"],
                recorded["whitened_rms_residual"],
            ),
            {
                "recomputed": recomputed["whitened_rms_residual"],
                "recorded": recorded["whitened_rms_residual"],
            },
        )
        covariance_diagnostics[arm] = recomputed

    raw_lsq, linear_residual = _triangulate_covariances(
        primary_models["ci"].means,
        member_mask,
        observed_covariances,
        inputs.cameras,
    )
    raw_eigenvalues = torch.linalg.eigvalsh(raw_lsq)
    raw_non_spd = int((raw_eigenvalues[:, 0] <= 0).sum())
    audit.check(
        "raw LS non-SPD count replay",
        raw_non_spd == int(summary["covariance"]["lsq"]["raw_non_spd_count"]),
        {
            "recomputed": raw_non_spd,
            "recorded": summary["covariance"]["lsq"]["raw_non_spd_count"],
        },
    )
    audit.check(
        "raw LS linear residual replay",
        _close_dict(
            _quantiles(linear_residual),
            summary["covariance"]["lsq"]["linear_system_residual"],
        ),
        {
            "recomputed": _quantiles(linear_residual),
            "recorded": summary["covariance"]["lsq"]["linear_system_residual"],
        },
    )
    projected_lsq = _spd_project(
        raw_lsq,
        min_sigma=1e-4,
        max_sigma=0.5 * float(dataset.bounds_hint[1]),
    )
    audit.check(
        "saved track-LSQ covariance equals bounded raw LS replay",
        torch.allclose(
            primary_models["track-lsq"].covariance().to(torch.float64),
            projected_lsq,
            rtol=2e-5,
            atol=2e-7,
        ),
        float(
            (primary_models["track-lsq"].covariance().to(torch.float64) - projected_lsq).abs().max()
        ),
    )

    outcomes: dict[str, Any] = {}
    ci_final_psnr = float(summary["arms"]["ci"]["final_metrics"]["psnr_fg"])
    for arm in ARMS:
        init = summary["arms"][arm]["init_metrics"]
        final = summary["arms"][arm]["final_metrics"]
        area = _auc(trajectories[arm])
        reaches = [step for step, psnr in trajectories[arm] if psnr >= ci_final_psnr]
        outcomes[arm] = {
            "init_psnr_fg": float(init["psnr_fg"]),
            "init_alpha_iou": float(init["alpha_iou"]),
            "init_alpha_inside": float(init["alpha_inside"]),
            "init_alpha_outside": float(init["alpha_outside"]),
            "final_psnr_fg": float(final["psnr_fg"]),
            "final_alpha_iou": float(final["alpha_iou"]),
            "final_alpha_inside": float(final["alpha_inside"]),
            "final_alpha_outside": float(final["alpha_outside"]),
            "psnr_fg_auc_db_steps": area,
            "psnr_fg_auc_mean_db": area / 1_000.0,
            "first_step_reaching_ci_final_psnr": min(reaches) if reaches else None,
        }

    ci = outcomes["ci"]
    ci_residual = covariance_diagnostics["ci"]["whitened_rms_residual"]["median"]
    gates: dict[str, Any] = {}
    for arm in ("track-lsq", "track-robust"):
        outcome = outcomes[arm]
        residual = covariance_diagnostics[arm]["whitened_rms_residual"]["median"]
        residual_reduction = 1.0 - residual / ci_residual
        init_iou_gain = outcome["init_alpha_iou"] / ci["init_alpha_iou"] - 1.0
        init_psnr_delta = outcome["init_psnr_fg"] - ci["init_psnr_fg"]
        auc_gain = outcome["psnr_fg_auc_db_steps"] / ci["psnr_fg_auc_db_steps"] - 1.0
        final_psnr_delta = outcome["final_psnr_fg"] - ci["final_psnr_fg"]
        final_iou_delta = outcome["final_alpha_iou"] - ci["final_alpha_iou"]
        rule_1 = residual_reduction >= 0.20
        rule_2 = init_iou_gain >= 0.10 and init_psnr_delta >= -0.25
        rule_3 = (auc_gain >= 0.01 or final_psnr_delta >= 0.10) and final_iou_delta >= -0.01
        gates[arm] = {
            "median_whitened_residual_reduction_vs_ci": residual_reduction,
            "initial_alpha_iou_relative_gain_vs_ci": init_iou_gain,
            "initial_psnr_fg_delta_db_vs_ci": init_psnr_delta,
            "psnr_fg_auc_relative_gain_vs_ci": auc_gain,
            "final_psnr_fg_delta_db_vs_ci": final_psnr_delta,
            "final_alpha_iou_delta_vs_ci": final_iou_delta,
            "rule_1_direct_covariance": rule_1,
            "rule_2_initial_coverage_and_quality": rule_2,
            "rule_3_pipeline_utility": rule_3,
            "all_preregistered_rules": rule_1 and rule_2 and rule_3,
        }
    robust_preference = {
        "median_whitened_residual_at_least_5pct_better_than_lsq": (
            covariance_diagnostics["track-robust"]["whitened_rms_residual"]["median"]
            <= 0.95 * covariance_diagnostics["track-lsq"]["whitened_rms_residual"]["median"]
        ),
        "better_auc_than_lsq": (
            outcomes["track-robust"]["psnr_fg_auc_db_steps"]
            > outcomes["track-lsq"]["psnr_fg_auc_db_steps"]
        ),
        "better_final_psnr_than_lsq": (
            outcomes["track-robust"]["final_psnr_fg"] > outcomes["track-lsq"]["final_psnr_fg"]
        ),
    }
    robust_preference["preferred"] = robust_preference[
        "median_whitened_residual_at_least_5pct_better_than_lsq"
    ] and (
        robust_preference["better_auc_than_lsq"] or robust_preference["better_final_psnr_than_lsq"]
    )

    return {
        "schema": "rtgs.beam_covariance_refit.audit.v1",
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "execution_binding": {
            "git_revision": EXPECTED_REVISION,
            "protocol_sha256": _sha256(PROTOCOL),
            "executed_harness_sha256": _reconstructed_executed_harness_sha256(),
            "current_harness_sha256": _sha256(HARNESS),
            "postrun_viewer_fix_sha256": _sha256(POSTRUN_FIX),
            "synthetic_test_sha256": _sha256(SYNTHETIC_TEST),
            "imported_helper_post_run_sha256": _sha256(IMPORTED_HELPER),
            "source_binding_limitation": (
                "beam_convergence_dynamics.py was imported by the official harness while dirty "
                "but was not separately hashed in the preregistration; its unchanged current "
                "hash is preserved post-run, so this is usable development evidence but not a "
                "fully pre-bound replay package"
            ),
            "postrun_integration_fix": (
                "The generated viewer manifest path was one parent too high. The adjacent reverse-"
                "applicable patch binds the two-line path-only correction and reconstructs the "
                "exact executed harness hash; no numerical path changed."
            ),
            "official_command": (
                "PYTHONUNBUFFERED=1 .venv/bin/python "
                "benchmarks/beam_covariance_refit.py "
                "--protocol benchmarks/results/20260723_beam_covariance_refit_PREREG.md "
                "--out runs/beam_covariance_refit_20260723"
            ),
            "repeat_command": (
                "CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 .venv/bin/python "
                "benchmarks/beam_covariance_refit.py "
                "--protocol benchmarks/results/20260723_beam_covariance_refit_PREREG.md "
                "--out runs/beam_covariance_refit_repeat_20260723"
            ),
            "python": sys.version,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "input_binding": input_binding,
        "primary": {
            "directory": str(primary.relative_to(ROOT)),
            "summary_sha256": _sha256(primary / "summary.json"),
            "artifact_manifest": _artifact_manifest(primary),
        },
        "repeat": {
            "directory": str(repeat.relative_to(ROOT)),
            "summary_sha256": _sha256(repeat / "summary.json"),
            "artifact_manifest": _artifact_manifest(repeat),
        },
        "lineage": {
            "components": beam.n_components,
            "contributor_links": int(member_mask.sum()),
            "unique_2d_contributors": len(unique_links),
            "unique_2d_contributors_per_view": unique_per_view,
            "unique_input_fraction": len(unique_links)
            / sum(observation.n for observation in inputs.observations),
        },
        "raw_lsq": {
            "non_spd_count": raw_non_spd,
            "non_spd_fraction": raw_non_spd / beam.n_components,
            "linear_residual": _quantiles(linear_residual),
            "minimum_raw_eigenvalue": float(raw_eigenvalues.min()),
        },
        "covariance_diagnostics_recomputed_from_saved_initial_ply": covariance_diagnostics,
        "outcomes": outcomes,
        "preregistered_gates": gates,
        "robust_preference": robust_preference,
        "claim_disposition": {
            "beam_fusion_exposes_pixel_or_gaussian_partial_correspondences": (
                "confirm_as_gaussian_to_gaussian_CSR_lineage"
            ),
            "track_lsq_is_a_better_correspondence_consistent_covariance_estimator": (
                "retire_failed_direct_covariance_gate_after_SPD_projection"
            ),
            "track_lsq_improves_fitted_view_coverage_and_convergence": (
                "confirm_narrowly_for_this_fixed_topology_development_run"
            ),
            "track_robust_improves_over_ci_or_lsq": "retire",
            "track_lsq_effect_is_evidence_for_physical_3d_covariance": (
                "retire_effect_is_anisotropic_scale_inflation_not_reprojection_fidelity"
            ),
            "production_default_change": "not_authorized",
            "held_out_multi_scene_or_cuda_behavior": "not_tested",
        },
        "protocol_findings": [
            (
                "All selected and evaluated cameras were fitted; no held-out or generalization "
                "claim is available."
            ),
            (
                "635/800 raw LS covariance solutions are non-SPD. Eigenvalue clamping turns the "
                "treatment into a highly anisotropic scale heuristic and destroys its raw linear "
                "fit interpretation."
            ),
            (
                "The preregistration omitted a hash for the dirty imported convergence helper. "
                "The exact same-environment repeat and post-run hash narrow but do not erase that "
                "source-binding defect."
            ),
            (
                "A post-result two-line viewer-manifest relative-path correction is isolated in "
                "20260723_beam_covariance_refit_POSTRUN_FIX.patch; reversing it reconstructs the "
                "executed harness hash."
            ),
            "CPU elapsed times are non-decisional and support no performance claim.",
        ],
        "checks": audit.checks,
        "check_summary": {
            "total": len(audit.checks),
            "passed": sum(bool(check["passed"]) for check in audit.checks),
            "failed": sum(not bool(check["passed"]) for check in audit.checks),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--repeat", type=Path, default=DEFAULT_REPEAT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.primary.resolve(strict=True), args.repeat.resolve(strict=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["check_summary"], indent=2))
    print(f"wrote {args.out}")
    return 1 if result["check_summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
