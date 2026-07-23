#!/usr/bin/env python3
"""Artifact audit for the post-hoc Beam partition opacity probe."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIMARY = ROOT / "runs/beam_partition_covariance_20260723/opacity_probe.json"
DEFAULT_REPEAT = ROOT / "runs/beam_partition_covariance_20260723_repeat/opacity_probe.json"
DEFAULT_OUT = ROOT / "benchmarks/results/20260723_beam_partition_opacity_probe_AUDIT.json"
ARMS = ("ci", "pou-area", "pou-full")
EXPECTED_SOURCE_SHA256 = "315a9d8cb4ada8a2d24f6ce066cccbd0a222b3fed93546a665cffa3153635018"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(value)

    value = json.loads(path.read_text(), parse_constant=reject)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _scientific_arm(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "initial_ply"}


def build(primary_path: Path, repeat_path: Path) -> dict[str, Any]:
    primary = _read(primary_path)
    repeat = _read(repeat_path)
    checks = []

    def check(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check(
        "schemas and exploratory scope",
        primary.get("schema") == "rtgs.beam_partition_opacity_probe.v1"
        and repeat.get("schema") == primary.get("schema")
        and primary.get("status") == "complete_exploratory_posthoc"
        and primary.get("all_evaluation_views_were_fitted") is True
        and primary.get("no_optimization") is True,
        {
            "schema": primary.get("schema"),
            "status": primary.get("status"),
            "all_views_fitted": primary.get("all_evaluation_views_were_fitted"),
        },
    )
    check(
        "bound probe source",
        primary["source_sha256"] == EXPECTED_SOURCE_SHA256
        and repeat["source_sha256"] == EXPECTED_SOURCE_SHA256
        and _sha256(ROOT / "benchmarks/beam_partition_opacity_probe.py") == EXPECTED_SOURCE_SHA256,
        {
            "primary": primary["source_sha256"],
            "repeat": repeat["source_sha256"],
            "current": _sha256(ROOT / "benchmarks/beam_partition_opacity_probe.py"),
        },
    )
    check(
        "frozen factors, thresholds, and alpha-IoU definition",
        primary["opacity_factors"] == [0.5, 1.0, 2.0, 4.0, 8.0]
        and primary["alpha_thresholds"] == [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        and primary["alpha_iou_threshold"] == 0.5,
        {
            "opacity_factors": primary["opacity_factors"],
            "alpha_thresholds": primary["alpha_thresholds"],
        },
    )
    primary_summary = _read(ROOT / "runs/beam_partition_covariance_20260723/summary.json")
    repeat_summary = _read(ROOT / "runs/beam_partition_covariance_20260723_repeat/summary.json")
    for arm in ARMS:
        first = primary["arms"][arm]
        second = repeat["arms"][arm]
        check(
            f"{arm}: exact repeated scientific probe",
            _scientific_arm(first) == _scientific_arm(second),
            {
                "primary_ply_sha256": first["initial_ply_sha256"],
                "repeat_ply_sha256": second["initial_ply_sha256"],
            },
        )
        check(
            f"{arm}: fixed 800-Gaussian PLY",
            first["n_gaussians"] == 800
            and second["n_gaussians"] == 800
            and first["initial_ply_sha256"] == second["initial_ply_sha256"],
            {
                "primary_count": first["n_gaussians"],
                "repeat_count": second["n_gaussians"],
            },
        )
        baseline = first["opacity_factors"]["1"]
        recorded = primary_summary["arms"][arm]["init_metrics"]
        repeat_recorded = repeat_summary["arms"][arm]["init_metrics"]
        keys = {
            "psnr_fg": "psnr_fg",
            "alpha_iou_at_0_5": "alpha_iou",
            "alpha_inside": "alpha_inside",
            "alpha_outside": "alpha_outside",
        }
        check(
            f"{arm}: factor-one metrics reproduce both base runs",
            all(
                baseline[probe_key] == recorded[summary_key]
                and baseline[probe_key] == repeat_recorded[summary_key]
                for probe_key, summary_key in keys.items()
            ),
            {key: baseline[key] for key in keys},
        )
        inside = [
            first["opacity_factors"][f"{factor:g}"]["alpha_inside"]
            for factor in primary["opacity_factors"]
        ]
        check(
            f"{arm}: alpha-inside responds monotonically to opacity only",
            all(right > left for left, right in zip(inside[:-1], inside[1:], strict=True)),
            inside,
        )

    full = primary["arms"]["pou-full"]
    ci = primary["arms"]["ci"]
    headline = {
        "pou_full_baseline_alpha_0_01": full["baseline_threshold_curve"]["0.01"],
        "pou_full_baseline_alpha_0_02": full["baseline_threshold_curve"]["0.02"],
        "pou_full_opacity_x8": full["opacity_factors"]["8"],
        "ci_opacity_x8": ci["opacity_factors"]["8"],
    }
    return {
        "schema": "rtgs.beam_partition_opacity_probe.audit.v1",
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "primary": {
            "path": str(primary_path.relative_to(ROOT)),
            "sha256": _sha256(primary_path),
        },
        "repeat": {
            "path": str(repeat_path.relative_to(ROOT)),
            "sha256": _sha256(repeat_path),
        },
        "headline": headline,
        "claim_disposition": {
            "pou_full_has_broad_low_alpha_support_on_fitted_views": (
                "confirm_at_posthoc_thresholds_only"
            ),
            "fixed_opacity_0_1_is_a_major_step0_bottleneck": (
                "support_as_exploratory_sensitivity_not_causal_selection"
            ),
            "opacity_x8_is_a_valid_initializer_rule": "not_tested_and_not_authorized",
            "opacity_is_the_only_missing_factor": "not_established",
            "held_out_or_default_claim": "not_authorized",
        },
        "findings": [
            (
                "Alpha IoU uses a hard 0.5 prediction threshold. At base opacity 0.1, pou-full "
                "already has 0.944 foreground recall at alpha>0.01 and 0.910 at alpha>0.02, but "
                "only 0.00886 IoU at alpha>0.5."
            ),
            (
                "Changing only uniform opacity from 0.1 to 0.8 raises pou-full alpha IoU to "
                "0.7223 and foreground PSNR to 16.472 dB, with outside alpha rising to 0.0150."
            ),
            (
                "The sweep is post-hoc, fitted-view-only, and globally uniform. It identifies "
                "optical thickness as a likely bottleneck but cannot select factor eight or rule "
                "out residual geometry, count, visibility, or topology bottlenecks."
            ),
        ],
        "checks": checks,
        "check_summary": {
            "total": len(checks),
            "passed": sum(bool(item["passed"]) for item in checks),
            "failed": sum(not bool(item["passed"]) for item in checks),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--repeat", type=Path, default=DEFAULT_REPEAT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = build(args.primary.resolve(strict=True), args.repeat.resolve(strict=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["check_summary"], indent=2))
    return 1 if result["check_summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
