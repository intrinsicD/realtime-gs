#!/usr/bin/env python3
"""Independent artifact and gate audit for masked Beam density partitions."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
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
from rtgs.lift.beam_partition import (
    MaskedPartitionConfig,
    refit_beam_covariances_from_masked_partitions,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d"
PROTOCOL = ROOT / "benchmarks/results/20260723_beam_partition_covariance_PREREG.md"
DEFAULT_PRIMARY = ROOT / "runs/beam_partition_covariance_20260723"
DEFAULT_REPEAT = ROOT / "runs/beam_partition_covariance_20260723_repeat"
DEFAULT_OUTPUT = ROOT / "benchmarks/results/20260723_beam_partition_covariance_AUDIT.json"

ARMS = ("ci", "pou-area", "pou-full")
SELECTED_VIEWS = (0, 3, 6, 9, 12, 15, 18, 21)
EXPECTED_STEPS = tuple(range(25, 1_001, 25))
EXPECTED_REVISION = "c2a7e120a5cafdcf22d4bff6f5b9868b860eb1df"
EXPECTED_PROTOCOL_SHA256 = "550abb9b931fb60644c0851c5ac488de969e1bbe1f0d6ed598cae353d8290562"
EXPECTED_MANIFEST_SHA256 = "b1c8e256d73e2c05f3cb4797a615bdbb2639a637f12908a5c96a2a9a9f912847"
EXPECTED_SOURCE_HASHES = {
    "src/rtgs/lift/beam_partition.py": (
        "088956887dea77bfeb720714ee937a58ac83ff1166b2c8080fd15c516c0ad196"
    ),
    "src/rtgs/lift/beam_fusion.py": (
        "575c12fdb59ad7a430178ed5899eb9d546cddc965f50617eeed0b40fe9ca2e12"
    ),
    "benchmarks/beam_partition_covariance.py": (
        "5ac7a63d4b8105b0b0b5f392fc541cf4dbbb7ca660f4fa6f93839b1fc07f817e"
    ),
    "tests/test_beam_partition.py": (
        "656ab4f6b9d0129bd8e395f990ee67de8c5899b96abd77ce8a46407bf52a6986"
    ),
    "benchmarks/beam_covariance_refit.py": (
        "8eb11a50fa9055578139985350ece981861c717e81317445863c9233c890995e"
    ),
    "benchmarks/beam_convergence_dynamics.py": (
        "6521af11d0af8513cd6963de260786e37c9791506a0782619b0561045fe2ffa9"
    ),
}


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


def _tensor_digest(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(contiguous.dtype).encode())
        digest.update(json.dumps(list(contiguous.shape)).encode())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(value)

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


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
    return hashlib.sha256(payload.encode()).hexdigest()


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


def _selected_inputs(dataset: CompactDataset) -> ReconstructionInputs:
    complete = dataset.to_reconstruction_inputs()
    return ReconstructionInputs(
        observations=[complete.observations[index] for index in SELECTED_VIEWS],
        cameras=[complete.cameras[index] for index in SELECTED_VIEWS],
        view_names=[complete.view_names[index] for index in SELECTED_VIEWS],
        bounds_hint=complete.bounds_hint,
        name=f"{complete.name}-partition-audit",
    )


def _beam_config(dataset: CompactDataset) -> BeamFusionConfig:
    return BeamFusionConfig(
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


def _partition_config() -> MaskedPartitionConfig:
    return MaskedPartitionConfig(
        quadrature_order=5,
        assignment_chunk=8_192,
        min_partition_mass=1e-12,
        min_variance_px=1e-6,
    )


def _finite_spd(model: Gaussians3D) -> bool:
    values = (model.means, model.quats, model.log_scales, model.opacity, model.sh)
    return (
        all(bool(torch.isfinite(value).all()) for value in values)
        and bool((torch.linalg.eigvalsh(model.covariance()) > 0).all())
        and bool(((model.opacity >= 0) & (model.opacity <= 1)).all())
    )


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
    return first.keys() == second.keys() and all(
        math.isclose(float(first[key]), float(second[key]), rel_tol=2e-5, abs_tol=2e-6)
        for key in first
    )


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


def _member_covariances(
    beam,
    tables: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros(beam.n_components, len(tables), dtype=torch.bool)
    covariances = torch.zeros(
        beam.n_components,
        len(tables),
        2,
        2,
        dtype=torch.float64,
    )
    offsets = beam.component_offsets.tolist()
    for component, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        views = beam.contributor_view_indices[start:stop]
        splats = beam.contributor_component_indices[start:stop]
        if views.unique().numel() != views.numel():
            raise RuntimeError(f"duplicate contributor view in component {component}")
        mask[component, views] = True
        for view, splat in zip(views.tolist(), splats.tolist(), strict=True):
            covariances[component, view] = tables[view][splat]
    return mask, covariances


def _covariance_diagnostics(
    jacobians: torch.Tensor,
    covariance: torch.Tensor,
    mask: torch.Tensor,
    observed: torch.Tensor,
) -> dict[str, dict[str, float]]:
    covariance = covariance.to(torch.float64)
    predicted = jacobians @ covariance[:, None] @ jacobians.transpose(-1, -2)
    relative = (
        (predicted - observed).norm(dim=(-2, -1)) / observed.norm(dim=(-2, -1)).clamp_min(1e-12)
    )[mask]
    eigenvalues, eigenvectors = torch.linalg.eigh(observed)
    inverse_sqrt = (
        eigenvectors
        @ torch.diag_embed(eigenvalues.clamp_min(1e-12).rsqrt())
        @ eigenvectors.transpose(-1, -2)
    )
    identity = torch.eye(2, dtype=torch.float64)
    whitened = (
        (inverse_sqrt @ predicted @ inverse_sqrt - identity)
        .square()
        .mean(dim=(-2, -1))
        .sqrt()[mask]
    )
    values = torch.linalg.eigvalsh(covariance).clamp_min(0)
    sigmas = values.sqrt()
    return {
        "relative_frobenius_residual": _quantiles(relative),
        "whitened_rms_residual": _quantiles(whitened),
        "sigma_all_axes": _quantiles(sigmas.reshape(-1)),
        "sigma_min": _quantiles(sigmas[:, 0]),
        "sigma_max": _quantiles(sigmas[:, -1]),
        "condition_number": _quantiles(values[:, -1] / values[:, 0].clamp_min(1e-24)),
    }


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


def _audit_dataset(audit: Audit) -> dict[str, Any]:
    manifest_path = DATASET / "manifest.json"
    manifest = _read(manifest_path)
    actual_manifest_hash = _sha256(manifest_path)
    audit.check(
        "compact manifest hash",
        actual_manifest_hash == EXPECTED_MANIFEST_SHA256,
        actual_manifest_hash,
    )
    files = []
    for entry in manifest["views"]:
        path = DATASET / entry["path"]
        actual_hash = _sha256(path)
        audit.check(
            f"compact payload hash: {entry['view_id']}",
            actual_hash == entry["sha256"] and path.stat().st_size == entry["bytes"],
            {"sha256": actual_hash, "bytes": path.stat().st_size},
        )
        files.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": actual_hash,
            }
        )
    return {
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": actual_manifest_hash,
        "semantic_digest": manifest["semantic_digest"],
        "calibration_sha256": manifest["calibration_sha256"],
        "files": files,
    }


def _replay_initializations(audit: Audit, summary: dict[str, Any]) -> dict[str, Any]:
    dataset = CompactDataset.load(DATASET, device="cpu")
    inputs = _selected_inputs(dataset)
    beam_config = _beam_config(dataset)
    torch.manual_seed(0)
    beam = fuse_gaussian_beams(inputs, beam_config)
    alphas = [dataset.views[index].alpha for index in SELECTED_VIEWS]
    partition = refit_beam_covariances_from_masked_partitions(
        inputs,
        alphas,
        beam,
        beam_config,
        _partition_config(),
    )
    initializations = {
        "ci": beam.gaussians,
        "pou-area": partition.area_matched,
        "pou-full": partition.full_moment,
    }
    audit.check(
        "replayed Beam component and link counts",
        beam.n_components == 800
        and int(beam.contributor_view_indices.numel())
        == summary["covariance"]["partition"]["n_contributor_links"],
        {
            "components": beam.n_components,
            "links": int(beam.contributor_view_indices.numel()),
        },
    )
    audit.check(
        "replayed deterministic partition diagnostics",
        partition.diagnostics == summary["covariance"]["partition"],
        {
            "replay_digest": _canonical_digest(partition.diagnostics),
            "recorded_digest": _canonical_digest(summary["covariance"]["partition"]),
        },
    )
    replay_ply_hashes = {}
    with tempfile.TemporaryDirectory(prefix="rtgs-beam-partition-audit-") as temporary:
        temporary_path = Path(temporary)
        for arm, initialization in initializations.items():
            path = temporary_path / f"{arm}.ply"
            initialization.save_ply(path)
            replay_ply_hashes[arm] = _sha256(path)

    native_tables = []
    area_tables = []
    full_tables = []
    for field, view_partition in zip(inputs.observations, partition.partitions, strict=True):
        native = _component_covariances_2d(field)
        area = native.clone()
        full = native.clone()
        anchors = view_partition.anchor_component_indices
        area[anchors] = view_partition.area_matched_covariances2d
        full[anchors] = view_partition.covariances2d
        native_tables.append(native)
        area_tables.append(area)
        full_tables.append(full)
    mask, native_members = _member_covariances(beam, native_tables)
    area_mask, area_members = _member_covariances(beam, area_tables)
    full_mask, full_members = _member_covariances(beam, full_tables)
    audit.check(
        "replayed member masks are identical",
        torch.equal(mask, area_mask) and torch.equal(mask, full_mask),
        int(mask.sum()),
    )
    return {
        "dataset": dataset,
        "inputs": inputs,
        "beam": beam,
        "partition": partition,
        "initializations": initializations,
        "replay_ply_hashes": replay_ply_hashes,
        "member_mask": mask,
        "targets": {
            "ci": native_members,
            "pou-area": area_members,
            "pou-full": full_members,
        },
    }


def build(primary: Path, repeat: Path) -> dict[str, Any]:
    audit = Audit()
    summary = _read(primary / "summary.json")
    repeat_summary = _read(repeat / "summary.json")
    audit.check(
        "summary schema and completeness",
        summary.get("schema") == "rtgs.beam_partition_covariance.v1"
        and summary.get("status") == "complete"
        and tuple(summary.get("arms", {})) == ARMS,
        {
            "schema": summary.get("schema"),
            "status": summary.get("status"),
            "arms": list(summary.get("arms", {})),
        },
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
    actual_protocol_hash = _sha256(PROTOCOL)
    audit.check(
        "protocol source and summary binding",
        actual_protocol_hash == EXPECTED_PROTOCOL_SHA256
        and summary["protocol"]["sha256"] == EXPECTED_PROTOCOL_SHA256
        and repeat_summary["protocol"]["sha256"] == EXPECTED_PROTOCOL_SHA256,
        {
            "actual": actual_protocol_hash,
            "primary": summary["protocol"]["sha256"],
            "repeat": repeat_summary["protocol"]["sha256"],
        },
    )
    source_hashes = {path: _sha256(ROOT / path) for path in EXPECTED_SOURCE_HASHES}
    audit.check(
        "all preregistered source hashes preserved",
        source_hashes == EXPECTED_SOURCE_HASHES,
        source_hashes,
    )
    git_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    audit.check(
        "base git revision",
        git_revision == EXPECTED_REVISION,
        git_revision,
    )
    first_dynamics_mtime = min((primary / arm / "dynamics.json").stat().st_mtime_ns for arm in ARMS)
    audit.check(
        "protocol predates measured dynamics artifacts",
        PROTOCOL.stat().st_mtime_ns < first_dynamics_mtime,
        {
            "protocol_mtime_ns": PROTOCOL.stat().st_mtime_ns,
            "first_dynamics_mtime_ns": first_dynamics_mtime,
        },
    )
    input_binding = _audit_dataset(audit)
    replay = _replay_initializations(audit, summary)

    dynamics: dict[str, dict[str, Any]] = {}
    trajectories: dict[str, list[tuple[int, float]]] = {}
    primary_initializations: dict[str, Gaussians3D] = {}
    for arm in ARMS:
        record, points = _trajectory(primary, arm)
        repeat_record, repeat_points = _trajectory(repeat, arm)
        dynamics[arm], trajectories[arm] = record, points
        audit.check(
            f"{arm}: complete primary and repeat checkpoint grids",
            tuple(int(row["step"]) for row in record["curve"]) == EXPECTED_STEPS
            and tuple(int(row["step"]) for row in repeat_record["curve"]) == EXPECTED_STEPS,
            {
                "primary_count": len(record["curve"]),
                "repeat_count": len(repeat_record["curve"]),
            },
        )
        audit.check(
            f"{arm}: deterministic timing-free dynamics repeat",
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
        primary_initializations[arm] = Gaussians3D.load_ply(primary / arm / "gaussians_init.ply")
        for state in ("gaussians_init.ply", "gaussians_final.ply"):
            first_path = primary / arm / state
            second_path = repeat / arm / state
            first = Gaussians3D.load_ply(first_path)
            second = Gaussians3D.load_ply(second_path)
            audit.check(
                f"{arm}/{state}: finite SPD fixed-count artifact",
                first.n == 800 and second.n == 800 and _finite_spd(first) and _finite_spd(second),
                {"primary_count": first.n, "repeat_count": second.n},
            )
            audit.check(
                f"{arm}/{state}: byte-exact repeat PLY",
                _sha256(first_path) == _sha256(second_path),
                {"primary": _sha256(first_path), "repeat": _sha256(second_path)},
            )
        audit.check(
            f"{arm}: source replay exactly reproduces initial PLY",
            replay["replay_ply_hashes"][arm] == _sha256(primary / arm / "gaussians_init.ply"),
            {
                "replay": replay["replay_ply_hashes"][arm],
                "artifact": _sha256(primary / arm / "gaussians_init.ply"),
            },
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
    frozen_fields = {
        arm: {
            field: torch.equal(
                getattr(primary_initializations["ci"], field),
                getattr(primary_initializations[arm], field),
            )
            for field in ("means", "opacity", "sh")
        }
        for arm in ARMS
    }
    audit.check(
        "all saved non-covariance initialization fields bit-exact",
        all(all(fields.values()) for fields in frozen_fields.values()),
        frozen_fields,
    )
    prior_ci = ROOT / "runs/beam_covariance_refit_20260723/ci/gaussians_init.ply"
    audit.check(
        "CI control unchanged from preceding covariance experiment",
        prior_ci.is_file() and _sha256(prior_ci) == _sha256(primary / "ci/gaussians_init.ply"),
        {
            "preceding": _sha256(prior_ci) if prior_ci.is_file() else None,
            "current": _sha256(primary / "ci/gaussians_init.ply"),
        },
    )

    partition_views = summary["covariance"]["partition"]["views"]
    validity = {
        "all_views_have_anchors_and_mass": all(
            view["unique_anchors"] > 0
            and view["positive_masked_samples"] > 0
            and view["partition_mass_min"] >= summary["partition_config"]["min_partition_mass"]
            for view in partition_views
        ),
        "partition_of_unity_error_at_most_1e_12": all(
            view["partition_of_unity_relative_error"] <= 1e-12 for view in partition_views
        ),
        "native_ci_roundtrip_error_at_most_1e_4": (
            summary["covariance"]["partition"]["native_ci_roundtrip_relative_error"]["max"] <= 1e-4
        ),
        "frozen_fields_and_count": (
            all(all(fields.values()) for fields in frozen_fields.values())
            and all(
                model.n == 800 and _finite_spd(model) for model in primary_initializations.values()
            )
        ),
    }
    validity["all"] = all(validity.values())
    audit.check("all preregistered mechanism-validity gates", validity["all"], validity)

    jacobians = _jacobians(primary_initializations["ci"].means, replay["inputs"].cameras)
    covariance_diagnostics: dict[str, Any] = {}
    for arm in ARMS:
        covariance = primary_initializations[arm].covariance()
        own = _covariance_diagnostics(
            jacobians,
            covariance,
            replay["member_mask"],
            replay["targets"][arm],
        )
        native = _covariance_diagnostics(
            jacobians,
            covariance,
            replay["member_mask"],
            replay["targets"]["ci"],
        )
        recorded_own = summary["covariance"]["arms_against_own_partition_targets"][arm]
        recorded_native = summary["covariance"]["arms_against_native_contributor_covariances"][arm]
        audit.check(
            f"{arm}: recomputed own-target covariance diagnostics",
            all(_close_dict(own[key], recorded_own[key]) for key in own),
            {
                "recomputed_median": own["whitened_rms_residual"]["median"],
                "recorded_median": recorded_own["whitened_rms_residual"]["median"],
            },
        )
        audit.check(
            f"{arm}: recomputed native-target covariance diagnostics",
            all(_close_dict(native[key], recorded_native[key]) for key in native),
            {
                "recomputed_median": native["whitened_rms_residual"]["median"],
                "recorded_median": recorded_native["whitened_rms_residual"]["median"],
            },
        )
        covariance_diagnostics[arm] = {
            "own_partition_target": own,
            "native_contributor_target": native,
        }

    outcomes: dict[str, Any] = {}
    ci_final_psnr = float(summary["arms"]["ci"]["final_metrics"]["psnr_fg"])
    for arm in ARMS:
        init = summary["arms"][arm]["init_metrics"]
        final = summary["arms"][arm]["final_metrics"]
        area = _auc(trajectories[arm])
        reached = [step for step, value in trajectories[arm] if value >= ci_final_psnr]
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
            "first_step_reaching_ci_final_psnr": min(reached) if reached else None,
        }

    ci = outcomes["ci"]
    gates: dict[str, Any] = {}
    for arm in ("pou-area", "pou-full"):
        outcome = outcomes[arm]
        inside_gain = outcome["init_alpha_inside"] / ci["init_alpha_inside"] - 1.0
        iou_gain = outcome["init_alpha_iou"] / ci["init_alpha_iou"] - 1.0
        init_psnr_delta = outcome["init_psnr_fg"] - ci["init_psnr_fg"]
        auc_gain = outcome["psnr_fg_auc_db_steps"] / ci["psnr_fg_auc_db_steps"] - 1.0
        final_psnr_delta = outcome["final_psnr_fg"] - ci["final_psnr_fg"]
        final_iou_delta = outcome["final_alpha_iou"] - ci["final_alpha_iou"]
        coverage = (
            inside_gain >= 0.25
            and iou_gain >= 0.10
            and init_psnr_delta >= -0.25
            and outcome["init_alpha_outside"] <= 0.005
        )
        optimization = final_iou_delta >= -0.01 and (auc_gain >= 0.01 or final_psnr_delta >= 0.10)
        gates[arm] = {
            "initial_alpha_inside_relative_gain_vs_ci": inside_gain,
            "initial_alpha_iou_relative_gain_vs_ci": iou_gain,
            "initial_psnr_fg_delta_db_vs_ci": init_psnr_delta,
            "initial_alpha_outside": outcome["init_alpha_outside"],
            "psnr_fg_auc_relative_gain_vs_ci": auc_gain,
            "final_psnr_fg_delta_db_vs_ci": final_psnr_delta,
            "final_alpha_iou_delta_vs_ci": final_iou_delta,
            "coverage_mechanism": coverage,
            "optimization_outcome": optimization,
            "both_required_gates": coverage and optimization,
        }
    area = outcomes["pou-area"]
    full = outcomes["pou-full"]
    shape_init = (
        full["init_alpha_iou"] / area["init_alpha_iou"] - 1.0 >= 0.02
        and full["init_psnr_fg"] - area["init_psnr_fg"] >= -0.10
        and full["init_alpha_outside"] - area["init_alpha_outside"] <= 0.001
    )
    shape_pipeline = full["final_alpha_iou"] - area["final_alpha_iou"] >= -0.01 and (
        full["psnr_fg_auc_db_steps"] / area["psnr_fg_auc_db_steps"] - 1.0 >= 0.01
        or full["final_psnr_fg"] - area["final_psnr_fg"] >= 0.10
    )
    shape_gate = {
        "initial_shape_clause": shape_init,
        "pipeline_shape_clause": shape_pipeline,
        "pou_full_passes_coverage_and_optimization": gates["pou-full"]["both_required_gates"],
        "full_partition_shape_adds_value": (
            gates["pou-full"]["both_required_gates"] and (shape_init or shape_pipeline)
        ),
        "auc_relative_gain_full_vs_area": (
            full["psnr_fg_auc_db_steps"] / area["psnr_fg_auc_db_steps"] - 1.0
        ),
        "final_psnr_delta_full_vs_area": (full["final_psnr_fg"] - area["final_psnr_fg"]),
    }

    return {
        "schema": "rtgs.beam_partition_covariance.audit.v1",
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "execution_binding": {
            "git_revision": git_revision,
            "protocol_sha256": actual_protocol_hash,
            "source_hashes": source_hashes,
            "official_command": (
                ".venv/bin/python benchmarks/beam_partition_covariance.py "
                "--protocol benchmarks/results/"
                "20260723_beam_partition_covariance_PREREG.md "
                "--out runs/beam_partition_covariance_20260723"
            ),
            "repeat_command": (
                ".venv/bin/python benchmarks/beam_partition_covariance.py "
                "--protocol benchmarks/results/"
                "20260723_beam_partition_covariance_PREREG.md "
                "--out runs/beam_partition_covariance_20260723_repeat"
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
        "lineage_replay": {
            "components": replay["beam"].n_components,
            "contributor_links": int(replay["beam"].contributor_view_indices.numel()),
            "unique_view_component_anchors": len(
                set(
                    zip(
                        replay["beam"].contributor_view_indices.tolist(),
                        replay["beam"].contributor_component_indices.tolist(),
                        strict=True,
                    )
                )
            ),
            "csr_and_depth_sha256": _tensor_digest(
                replay["beam"].component_offsets,
                replay["beam"].contributor_view_indices,
                replay["beam"].contributor_component_indices,
                replay["beam"].contributor_depths,
            ),
        },
        "mechanism_validity": validity,
        "covariance_diagnostics_recomputed_from_saved_initial_ply": (covariance_diagnostics),
        "outcomes": outcomes,
        "preregistered_gates": gates,
        "full_shape_gate": shape_gate,
        "claim_disposition": {
            "exact_native_2d_contributors_are_used_as_anchors": (
                "confirm_from_CSR_source_replay_without_3d_projection_matching"
            ),
            "masked_density_is_partitioned": ("confirm_at_frozen_order_5_quadrature_resolution"),
            "partition_covariance_increases_initial_visible_coverage": "retire",
            "partition_covariance_improves_fixed_topology_fitted_view_optimization": (
                "confirm_narrowly_for_this_single_scene_CPU_development_run"
            ),
            "full_partition_shape_is_better_than_area_scaling": (
                "narrow_to_1.87pct_AUC_gain_but_overall_preregistered_shape_gate_failed"
            ),
            "partition_moments_recover_physical_3d_covariance": "not_established",
            "production_default_change": "not_authorized",
            "held_out_multi_scene_or_cuda_behavior": "not_tested",
        },
        "protocol_findings": [
            (
                "All construction, fitting, and evaluation cameras are fitted views; no held-out "
                "or generalization claim is available."
            ),
            (
                "The mask integral is a deterministic 25-sample Gaussian quadrature "
                "approximation, not an exact continuous or full-pixel integral."
            ),
            (
                "Most determinant-matching scalar covariance multipliers are below one while a "
                "few are extreme (maximum recorded 21290x); the frozen 3D Beam sigma bound "
                "limits their output effect, but the distribution is not uniform upscaling."
            ),
            (
                "Raw per-anchor partitions were not persisted in the official run directories. "
                "The audit regenerated them exactly from bound source and data, and both full "
                "runs regenerated byte-identical initial PLYs."
            ),
            ("CPU elapsed times are non-decisional and support no performance or real-time claim."),
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
