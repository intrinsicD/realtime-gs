#!/usr/bin/env python3
"""Independent audit of the Janelle beam-convergence-dynamics mechanism screen."""

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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/beam_convergence_dynamics_replication_20260723"
DEFAULT_REPEAT = ROOT / "runs/beam_convergence_dynamics_repeat2_20260723"
DEFAULT_OUTPUT = (
    ROOT / "benchmarks/results/20260723_beam_convergence_dynamics_REPLICATION_AUDIT.json"
)
DATASET = ROOT / "dataset/2025_03_07_stage_with_fabric/frame_00008/gaussians2d"
HARNESS = ROOT / "benchmarks/beam_convergence_dynamics.py"

ARMS = ("beam-adc", "random-adc", "beam-fixed", "random-fixed")
ADC_ARMS = ("beam-adc", "random-adc")
EXPECTED_STEPS = tuple(range(25, 1_001, 25))
EXPECTED_RESETS = (100, 200, 300, 400, 500)
EXECUTED_REVISION = "c2a7e120a5cafdcf22d4bff6f5b9868b860eb1df"
EXECUTED_HARNESS_SHA256 = "bbfe4172958af8f1188999f0eb1d4c41dccef2299b40ff93909f65e8dcf17991"
EXPECTED_DATASET_SET_SHA256 = "5811b08c5d37d6d4e797e9e2aab18d9a6f420266041bb9b874ec380a43c507f2"


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def require(self, name: str, condition: bool, detail: Any) -> None:
        self.checks.append({"name": name, "passed": bool(condition), "detail": detail})
        if not condition:
            raise RuntimeError(f"audit check failed: {name}: {detail}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _strict_json(path: Path) -> bool:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    try:
        json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except ValueError:
        return False
    return True


def _normalize_nonfinite(
    value: Any,
    *,
    path: str = "root",
    found: list[str] | None = None,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize_nonfinite(item, path=f"{path}.{key}", found=found)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _normalize_nonfinite(item, path=f"{path}[{index}]", found=found)
            for index, item in enumerate(value)
        ]
    if isinstance(value, float) and not math.isfinite(value):
        if found is not None:
            found.append(path)
        return None
    return value


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dataset_set_digest() -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    files = []
    for path in sorted(DATASET.iterdir()):
        if not path.is_file():
            continue
        file_hash = _sha256(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_hash))
        files.append(
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": file_hash}
        )
    return digest.hexdigest(), files


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


def _finite_model(model: Gaussians3D) -> bool:
    return all(
        bool(torch.isfinite(tensor).all())
        for tensor in (model.means, model.quats, model.log_scales, model.opacity, model.sh)
    )


def _expected_ply_count(path: Path, record: dict[str, Any]) -> int:
    if path.name == "gaussians_init.ply":
        return 800
    if path.name == "gaussians_final.ply":
        return int(record["final_n"])
    step = int(path.stem.rsplit("_", 1)[1])
    by_step = {int(row["step"]): int(row["n"]) for row in record["curve"]}
    return by_step[step]


def _audit_plys(
    audit: Audit,
    directory: Path,
    records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    loaded = 0
    total_bytes = 0
    minimum = math.inf
    maximum = 0
    for arm, record in records.items():
        for path in sorted((directory / arm).glob("*.ply")):
            model = Gaussians3D.load_ply(path)
            expected = _expected_ply_count(path, record)
            valid = (
                model.n == expected
                and _finite_model(model)
                and bool(((model.opacity >= 0.0) & (model.opacity <= 1.0)).all())
                and bool((torch.linalg.vector_norm(model.quats, dim=-1) > 0.0).all())
            )
            audit.require(
                f"{directory.name}/{arm}/{path.name}: finite PLY and bound count",
                valid,
                {"actual": model.n, "expected": expected},
            )
            loaded += 1
            total_bytes += path.stat().st_size
            minimum = min(minimum, model.n)
            maximum = max(maximum, model.n)
    return {
        "artifacts_loaded": loaded,
        "bytes_read": total_bytes,
        "minimum_gaussians": int(minimum),
        "maximum_gaussians": int(maximum),
        "all_finite": True,
        "all_counts_match_trajectory": True,
    }


def _scientific_record(record: dict[str, Any]) -> dict[str, Any]:
    return _normalize_nonfinite(
        {key: value for key, value in record.items() if key != "elapsed_seconds"}
    )


def _audit_run(
    audit: Audit,
    directory: Path,
    expected_arms: tuple[str, ...],
) -> dict[str, Any]:
    summary_path = directory / "summary.json"
    summary = _read(summary_path)
    audit.require(
        f"{directory.name}: schema",
        summary.get("schema") == "rtgs.beam_convergence_dynamics.v1",
        summary.get("schema"),
    )
    audit.require(
        f"{directory.name}: arm set",
        tuple(summary["arms"]) == expected_arms,
        list(summary["arms"]),
    )
    expected_parameters = {
        "selected_global_views": [0, 3, 6, 9, 12, 15, 18, 21],
        "downscale": 32,
        "n_init": 800,
        "iterations": 1_000,
        "density": {
            "start": 20,
            "stop": 500,
            "every": 4,
            "grad_threshold": 0.003,
            "absgrad": False,
            "opacity_reset_every": 100,
            "max_gaussians": 8_000,
        },
        "seed": 0,
    }
    actual_parameters = {key: summary[key] for key in expected_parameters}
    audit.require(
        f"{directory.name}: frozen parameters",
        actual_parameters == expected_parameters,
        actual_parameters,
    )
    audit.require(
        f"{directory.name}: summary is strict JSON",
        _strict_json(summary_path),
        str(summary_path),
    )

    records = {}
    nonfinite = {}
    for arm in expected_arms:
        path = directory / arm / "dynamics.json"
        record = _read(path)
        records[arm] = record
        found: list[str] = []
        normalized = _normalize_nonfinite(record, path=arm, found=found)
        nonfinite[arm] = found
        curve = record["curve"]
        audit.require(
            f"{directory.name}/{arm}: complete checkpoint grid",
            tuple(int(row["step"]) for row in curve) == EXPECTED_STEPS,
            [row["step"] for row in curve],
        )
        audit.require(
            f"{directory.name}/{arm}: summary endpoint binding",
            int(curve[-1]["n"]) == int(record["final_n"]) == int(summary["arms"][arm]["final_n"])
            and math.isclose(
                float(curve[-1]["metric_psnr_fg"]),
                float(summary["arms"][arm]["final_psnr_fg"]),
                abs_tol=0.0,
            )
            and math.isclose(
                float(curve[-1]["metric_alpha_iou"]),
                float(summary["arms"][arm]["final_alpha_iou"]),
                abs_tol=0.0,
            ),
            summary["arms"][arm],
        )
        expected_nonfinite = []
        for index, row in enumerate(curve):
            if int(row["n_confident"]) == 0:
                expected_nonfinite.extend(
                    [
                        f"{arm}.curve[{index}].chamfer_current_to_surface_mean",
                        f"{arm}.curve[{index}].chamfer_current_to_surface_p90",
                        f"{arm}.curve[{index}].chamfer_surface_to_current_mean",
                    ]
                )
        audit.require(
            f"{directory.name}/{arm}: non-finite values only mark empty chamfer sets",
            sorted(found) == sorted(expected_nonfinite),
            found,
        )
        audit.require(
            f"{directory.name}/{arm}: normalized trajectory is strict JSON",
            bool(json.dumps(normalized, allow_nan=False)),
            len(found),
        )

    for kind in ("beam", "random"):
        adc = records.get(f"{kind}-adc")
        fixed = records.get(f"{kind}-fixed")
        if adc is not None and fixed is not None:
            audit.require(
                f"{directory.name}/{kind}: identical step-0 metrics across density modes",
                adc["init_metrics"] == fixed["init_metrics"],
                adc["init_metrics"],
            )

    for arm in ADC_ARMS:
        if arm not in records:
            continue
        curve_by_step = {int(row["step"]): row for row in records[arm]["curve"]}
        resets = sorted(
            {
                int(event["iteration"])
                for event in records[arm]["density_events"]
                if event["opacity_reset"]
            }
        )
        audit.require(
            f"{directory.name}/{arm}: reset schedule",
            tuple(resets) == EXPECTED_RESETS,
            resets,
        )
        for step in resets:
            row = curve_by_step[step]
            audit.require(
                f"{directory.name}/{arm}/{step}: reset collapses confident alpha",
                int(row["n_confident"]) == 0
                and float(row["frac_opacity_lt_02"]) == 1.0
                and float(row["metric_alpha_iou"]) <= 0.001,
                {
                    "n_confident": row["n_confident"],
                    "frac_opacity_lt_02": row["frac_opacity_lt_02"],
                    "alpha_iou": row["metric_alpha_iou"],
                },
            )

    return {
        "directory": str(directory.relative_to(ROOT)),
        "summary_sha256": _sha256(summary_path),
        "summary": summary,
        "records": {arm: _normalize_nonfinite(record) for arm, record in records.items()},
        "raw_dynamics_strict_json": {
            arm: _strict_json(directory / arm / "dynamics.json") for arm in records
        },
        "nonfinite_paths_normalized_to_null": nonfinite,
        "scientific_record_sha256": {
            arm: _json_digest(_scientific_record(record)) for arm, record in records.items()
        },
        "ply_audit": _audit_plys(audit, directory, records),
        "artifact_manifest": _artifact_manifest(directory),
    }


def _birth_accounting(record: dict[str, Any]) -> dict[str, Any]:
    events = record["density_events"]
    cloned = sum(int(event["cloned"]) for event in events)
    split = sum(int(event["split"]) for event in events)
    pruned = sum(int(event["pruned"]) for event in events)
    cumulative_newborn_rows = cloned + 2 * split
    cumulative_removed_rows = pruned + split
    final = int(record["final_n"])
    return {
        "cloned_rows": cloned,
        "split_parents": split,
        "pruned_rows_excluding_split_parents": pruned,
        "cumulative_newborn_rows": cumulative_newborn_rows,
        "cumulative_removed_rows": cumulative_removed_rows,
        "accounting_final": 800 + cumulative_newborn_rows - cumulative_removed_rows,
        "recorded_final": final,
    }


def _outcome(record: dict[str, Any]) -> dict[str, Any]:
    final = record["curve"][-1]
    return {
        "init_psnr_fg": float(record["init_metrics"]["psnr_fg"]),
        "init_alpha_iou": float(record["init_metrics"]["alpha_iou"]),
        "init_alpha_inside": float(record["init_metrics"]["alpha_inside"]),
        "final_psnr_fg": float(final["metric_psnr_fg"]),
        "final_alpha_iou": float(final["metric_alpha_iou"]),
        "final_n": int(final["n"]),
        "final_survivors": int(final["survivors"]),
        "final_original_fraction": float(final["survivors"]) / float(final["n"]),
        "final_displacement_mean": final["displacement_mean"],
        "final_displacement_p90": final["displacement_p90"],
        "final_chamfer_current_to_surface_mean": final["chamfer_current_to_surface_mean"],
        "final_chamfer_surface_to_current_mean": final["chamfer_surface_to_current_mean"],
    }


def build(primary: Path, repeat: Path) -> dict[str, Any]:
    audit = Audit()
    dataset_digest, dataset_files = _dataset_set_digest()
    audit.require(
        "dataset file-set digest",
        dataset_digest == EXPECTED_DATASET_SET_SHA256,
        dataset_digest,
    )
    primary_result = _audit_run(audit, primary, ARMS)
    repeat_result = _audit_run(audit, repeat, ADC_ARMS)

    for arm in ADC_ARMS:
        first = primary_result["scientific_record_sha256"][arm]
        second = repeat_result["scientific_record_sha256"][arm]
        audit.require(
            f"{arm}: exact same-environment scientific repeat",
            first == second,
            {"first": first, "second": second},
        )
        primary_final = primary / arm / "gaussians_final.ply"
        repeat_final = repeat / arm / "gaussians_final.ply"
        audit.require(
            f"{arm}: exact repeated final PLY",
            _sha256(primary_final) == _sha256(repeat_final),
            {"first": _sha256(primary_final), "second": _sha256(repeat_final)},
        )

    records = primary_result["records"]
    outcomes = {arm: _outcome(record) for arm, record in records.items()}
    birth_accounting = {arm: _birth_accounting(records[arm]) for arm in ADC_ARMS}
    for arm, accounting in birth_accounting.items():
        audit.require(
            f"{arm}: density surgery accounting",
            accounting["accounting_final"] == accounting["recorded_final"],
            accounting,
        )

    beam_init = Gaussians3D.load_ply(primary / "beam-fixed/gaussians_init.ply")
    random_init = Gaussians3D.load_ply(primary / "random-fixed/gaussians_init.ply")
    initializer_field_equality = {
        "means": bool(torch.equal(beam_init.means, random_init.means)),
        "quats": bool(torch.equal(beam_init.quats, random_init.quats)),
        "log_scales": bool(torch.equal(beam_init.log_scales, random_init.log_scales)),
        "opacity": bool(torch.equal(beam_init.opacity, random_init.opacity)),
        "sh": bool(torch.equal(beam_init.sh, random_init.sh)),
    }
    audit.require(
        "fixed-topology contrast is count-matched but not placement-only",
        initializer_field_equality["opacity"]
        and not initializer_field_equality["means"]
        and not initializer_field_equality["log_scales"]
        and not initializer_field_equality["sh"],
        initializer_field_equality,
    )

    fixed_fg_delta = (
        outcomes["beam-fixed"]["final_psnr_fg"] - outcomes["random-fixed"]["final_psnr_fg"]
    )
    fixed_iou_delta = (
        outcomes["beam-fixed"]["final_alpha_iou"] - outcomes["random-fixed"]["final_alpha_iou"]
    )
    adc_fg_delta = outcomes["beam-adc"]["final_psnr_fg"] - outcomes["random-adc"]["final_psnr_fg"]
    adc_iou_delta = (
        outcomes["beam-adc"]["final_alpha_iou"] - outcomes["random-adc"]["final_alpha_iou"]
    )

    original_doc = {
        "beam-adc": {
            "final_psnr_fg_rounded": 26.92,
            "final_n": 4_390,
            "final_survivors": 737,
        },
        "random-adc": {
            "final_psnr_fg_rounded": 24.78,
            "final_alpha_iou_rounded": 0.407,
            "final_n": 1_288,
            "final_survivors": 443,
        },
    }
    original_doc_deltas = {
        "beam-adc": {
            "psnr_fg_vs_rounded_doc": outcomes["beam-adc"]["final_psnr_fg"] - 26.92,
            "n": outcomes["beam-adc"]["final_n"] - 4_390,
            "survivors": outcomes["beam-adc"]["final_survivors"] - 737,
        },
        "random-adc": {
            "psnr_fg_vs_rounded_doc": outcomes["random-adc"]["final_psnr_fg"] - 24.78,
            "alpha_iou_vs_rounded_doc": outcomes["random-adc"]["final_alpha_iou"] - 0.407,
            "n": outcomes["random-adc"]["final_n"] - 1_288,
            "survivors": outcomes["random-adc"]["final_survivors"] - 443,
        },
    }

    return {
        "schema": "rtgs.beam_convergence_dynamics.replication_audit.v1",
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "execution_binding": {
            "git_revision_observed_clean_at_launch": EXECUTED_REVISION,
            "executed_harness_sha256": EXECUTED_HARNESS_SHA256,
            "current_harness_sha256": _sha256(HARNESS),
            "command_primary": (
                "CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 .venv/bin/python "
                "benchmarks/beam_convergence_dynamics.py "
                "--out runs/beam_convergence_dynamics_replication_20260723"
            ),
            "command_repeat": (
                "CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 .venv/bin/python "
                "benchmarks/beam_convergence_dynamics.py "
                "--out runs/beam_convergence_dynamics_repeat2_20260723 "
                "--arms beam-adc random-adc"
            ),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "platform": platform.platform(),
        },
        "input_binding": {
            "dataset": str(DATASET.relative_to(ROOT)),
            "file_count": len(dataset_files),
            "file_set_sha256": dataset_digest,
            "files": dataset_files,
        },
        "primary": primary_result,
        "same_environment_adc_repeat": repeat_result,
        "outcomes": outcomes,
        "derived": {
            "fixed_topology_beam_minus_random_psnr_fg": fixed_fg_delta,
            "fixed_topology_beam_minus_random_alpha_iou": fixed_iou_delta,
            "adc_beam_minus_random_psnr_fg": adc_fg_delta,
            "adc_beam_minus_random_alpha_iou": adc_iou_delta,
            "birth_accounting": birth_accounting,
            "initializer_field_equality": initializer_field_equality,
            "documented_original_adc_values": original_doc,
            "replication_minus_documented_original": original_doc_deltas,
        },
        "claim_disposition": {
            "opacity_reset_sawtooth": "confirm",
            "beam_original_positions_are_mostly_preserved": "confirm_narrowly",
            "beam_originals_become_a_final_population_minority": "confirm",
            "beam_is_better_than_random_at_fixed_count": "confirm_for_full_initializer_package",
            "fixed_topology_delta_is_caused_by_position_alone": "retire_not_isolated",
            "exact_original_adc_endpoint_counts": "retire_not_reproduced",
            "production_gsplat_or_held_out_convergence": "not_tested",
        },
        "checks": audit.checks,
        "check_summary": {
            "total": len(audit.checks),
            "passed": sum(bool(check["passed"]) for check in audit.checks),
            "failed": sum(not bool(check["passed"]) for check in audit.checks),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--repeat", type=Path, default=DEFAULT_REPEAT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build(args.run.resolve(strict=True), args.repeat.resolve(strict=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["check_summary"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
