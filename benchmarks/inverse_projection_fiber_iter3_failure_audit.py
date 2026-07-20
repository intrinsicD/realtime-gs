#!/usr/bin/env python3
"""Read-only audit of the consumed Iteration 3 synthetic attempt.

This script never imports the experiment generator and never constructs an official root. It
recomputes the completed root-0 association, dust, geometry-validity, and held-out metrics directly
from the sealed NPZ, checks them against ROOT_RESULT.json, and reports the incomplete transaction.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter3_SYNTHETIC_ATTEMPT.json"
RESULT = ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter3_SYNTHETIC_RESULT.json"
ARTIFACTS = ROOT / "runs/inverse_projection_fiber_iter3_synthetic_20260717"
ROOT0_JSON = ARTIFACTS / "root_0/ROOT_RESULT.json"
ROOT0_NPZ = ARTIFACTS / "root_0/evidence.npz"
ROOT1_INITIAL = ARTIFACTS / "root_1/gaussians_init.ply"
REAL_ATTEMPT = ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter3_REAL_ATTEMPT.json"
REAL_RESULT = ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter3_REAL.json"
REAL_ARTIFACTS = ROOT / "runs/inverse_projection_fiber_iter3_real_20260717"

ARMS = ("hardmin", "row", "uot_uniform", "uot_area", "oracle", "shuffled_view")
FIT_VIEWS = 5
PARENTS = 8
SUPPORT_FRACTION = 0.20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _close(observed: float | None, expected: float | None, *, atol: float = 1e-10) -> bool:
    if observed is None or expected is None:
        return observed is expected
    return math.isclose(observed, expected, rel_tol=1e-10, abs_tol=atol)


def _arm_metrics(data: Any, arm: str) -> dict[str, Any]:
    source_labels = data["source_parent_labels"]
    source_views = data["source_view_indices"]
    source_inlier = source_labels >= 0
    view_purity: list[float] = []
    track_outlier: list[float] = []
    track_inlier: list[float] = []
    observation_outlier: list[float] = []
    observation_inlier: list[float] = []
    correct_support = np.zeros((PARENTS, FIT_VIEWS), dtype=np.float64)
    transport_views: list[dict[str, Any]] = []

    for view in range(FIT_VIEWS):
        prefix = f"{arm}_view_{view}"
        real = data[f"{prefix}_real_mass"]
        track_dust = data[f"{prefix}_track_dustbin_mass"]
        label_view = (view + 1) % FIT_VIEWS if arm == "shuffled_view" else view
        target_labels = data[f"fit_view_{label_view}_parent_labels"]
        active = source_views != view
        same = (
            (source_labels[:, None] == target_labels[None, :])
            & source_inlier[:, None]
            & (target_labels >= 0)[None, :]
            & active[:, None]
        )
        denominator = float(real[active & source_inlier].sum())
        view_purity.append(float(real[same].sum()) / denominator if denominator else 0.0)

        row_target_key = f"{prefix}_augmented_target_row_marginal"
        column_target_key = f"{prefix}_augmented_target_column_marginal"
        is_uot = arm in {"uot_uniform", "uot_area", "shuffled_view"}
        if is_uot:
            declared_track = data[row_target_key][:-1]
            declared_observation = data[column_target_key][:-1]
        else:
            declared_track = data[f"{prefix}_track_capacities"]
            declared_observation = data.get(f"{prefix}_observation_capacities", None)

        outlier_tracks = active & ~source_inlier
        inlier_tracks = active & source_inlier
        track_outlier_denominator = float(declared_track[outlier_tracks].sum())
        track_inlier_denominator = float(declared_track[inlier_tracks].sum())
        track_outlier.append(
            float(track_dust[outlier_tracks].sum()) / track_outlier_denominator
            if track_outlier_denominator
            else 0.0
        )
        track_inlier.append(
            float(track_dust[inlier_tracks].sum()) / track_inlier_denominator
            if track_inlier_denominator
            else 0.0
        )
        for parent in range(PARENTS):
            rows = active & (source_labels == parent)
            columns = target_labels == parent
            parent_denominator = float(declared_track[rows].sum())
            if parent_denominator and columns.any():
                correct_support[parent, view] = (
                    float(real[rows][:, columns].sum()) / parent_denominator
                )

        observation_dust_key = f"{prefix}_observation_dustbin_mass"
        if observation_dust_key not in data:
            continue
        observation_dust = data[observation_dust_key]
        target_outlier = target_labels < 0
        outlier_denominator = float(declared_observation[target_outlier].sum())
        inlier_denominator = float(declared_observation[~target_outlier].sum())
        observation_outlier.append(
            float(observation_dust[target_outlier].sum()) / outlier_denominator
            if outlier_denominator
            else 0.0
        )
        observation_inlier.append(
            float(observation_dust[~target_outlier].sum()) / inlier_denominator
            if inlier_denominator
            else 0.0
        )

        if not is_uot:
            continue
        dust_dust = float(data[f"{prefix}_dustbin_dustbin_mass"][0])
        augmented = np.block(
            [[real, track_dust[:, None]], [observation_dust[None, :], np.array([[dust_dust]])]]
        )
        row_target = data[row_target_key]
        column_target = data[column_target_key]
        realized_row = augmented.sum(axis=1)
        realized_column = augmented.sum(axis=0)
        ratios = {
            "track_row": float(realized_row[:-1].sum() / row_target[:-1].sum()),
            "observation_column": float(realized_column[:-1].sum() / column_target[:-1].sum()),
            "augmented": float(augmented.sum() / row_target.sum()),
        }
        row_positive = row_target > 0
        column_positive = column_target > 0
        max_relative = float(
            np.concatenate(
                [
                    np.abs(realized_row[row_positive] - row_target[row_positive])
                    / row_target[row_positive],
                    np.abs(realized_column[column_positive] - column_target[column_positive])
                    / column_target[column_positive],
                ]
            ).max()
        )
        residual = float(data[f"{prefix}_fixed_point_residual"][0])
        transport_views.append(
            {
                "view": view,
                "ratios": ratios,
                "max_marginal_relative_error": max_relative,
                "fixed_point_residual": residual,
                "pass": (
                    all(0.20 <= value <= 5.0 for value in ratios.values())
                    and max_relative <= 4.0
                    and residual <= 0.05
                ),
            }
        )

    inlier = source_labels >= 0
    means = data[f"{arm}_means"]
    covariances = data[f"{arm}_covariances"]
    gt_means = data["gt_means"][source_labels[inlier]]
    center_errors = np.linalg.norm(means[inlier] - gt_means, axis=-1)
    source_means = data[f"{arm}_source_means"]
    source_covariances = data[f"{arm}_source_covariances"]
    source_covariance_denominator = np.linalg.norm(
        data["source_covariances2d"].reshape(len(source_labels), -1), axis=1
    )
    source_covariance_error = (
        np.linalg.norm(
            (source_covariances - data["source_covariances2d"]).reshape(len(source_labels), -1),
            axis=1,
        )
        / source_covariance_denominator
    )

    heldout_correct = 0
    heldout_denominator = 0
    for view in range(2):
        evaluable = data[f"{arm}_heldout_view_{view}_evaluable"]
        assignment = data[f"{arm}_heldout_view_{view}_assignment"]
        target_labels = data[f"heldout_view_{view}_parent_labels"]
        heldout_correct += int(
            (target_labels[assignment[evaluable]] == source_labels[evaluable]).sum()
        )
        heldout_denominator += int(evaluable.sum())

    return {
        "parent_purity": float(np.mean(view_purity)),
        "parent_purity_by_view": view_purity,
        "parent_completeness": float(np.all(correct_support >= SUPPORT_FRACTION, axis=1).mean()),
        "correct_support_by_parent_view": correct_support.tolist(),
        "outlier_track_dust_recall": float(np.mean(track_outlier)),
        "inlier_track_dust_false_positive_rate": float(np.mean(track_inlier)),
        "outlier_observation_dust_recall": (
            float(np.mean(observation_outlier)) if observation_outlier else None
        ),
        "inlier_observation_dust_false_positive_rate": (
            float(np.mean(observation_inlier)) if observation_inlier else None
        ),
        "center_error_p90": float(np.quantile(center_errors, 0.9)),
        "source_center_max_px": float(
            np.linalg.norm(source_means - data["source_means2d"], axis=-1).max()
        ),
        "source_covariance_relative_max": float(source_covariance_error.max()),
        "finite_spd": bool(
            np.isfinite(means).all()
            and np.isfinite(covariances).all()
            and (np.linalg.eigvalsh(covariances) > 0).all()
        ),
        "depth_bound_incidence": int(
            ((data[f"{arm}_source_depths"] <= 1.2) | (data[f"{arm}_source_depths"] >= 3.6)).sum()
        ),
        "heldout_parent_assignment_accuracy": (
            heldout_correct / heldout_denominator if heldout_denominator else 0.0
        ),
        "heldout_assignment_denominator": heldout_denominator,
        "transport_mass_diagnostics": {
            "applicable": bool(transport_views),
            "views": transport_views,
            "pass": all(item["pass"] for item in transport_views),
        },
    }


def main() -> int:
    if RESULT.exists():
        raise RuntimeError("failure audit requires the official synthetic RESULT to be absent")
    if any(path.exists() for path in (REAL_ATTEMPT, REAL_RESULT, REAL_ARTIFACTS)):
        raise RuntimeError("failure audit requires the real namespace to remain untouched")
    attempt = json.loads(ATTEMPT.read_bytes())
    summary = json.loads(ROOT0_JSON.read_bytes())
    with np.load(ROOT0_NPZ, allow_pickle=False) as data:
        recomputed = {arm: _arm_metrics(data, arm) for arm in ARMS}
        fragment_offsets: list[float] = []
        for view in range(FIT_VIEWS):
            labels = data[f"fit_view_{view}_parent_labels"]
            means = data[f"fit_view_{view}_means"]
            for parent in range(PARENTS):
                children = means[labels == parent]
                parent_mean = children.mean(axis=0)
                fragment_offsets.extend(np.linalg.norm(children - parent_mean, axis=-1).tolist())

    comparisons: dict[str, Any] = {}
    for arm, metrics in recomputed.items():
        reported = summary["arms"][arm]
        checks = {
            "parent_purity": _close(
                metrics["parent_purity"], reported["association"]["parent_purity"]
            ),
            "parent_completeness": _close(
                metrics["parent_completeness"], reported["association"]["parent_completeness"]
            ),
            "track_outlier_recall": _close(
                metrics["outlier_track_dust_recall"],
                reported["association"]["outlier_track_dust_recall"],
            ),
            "track_inlier_fpr": _close(
                metrics["inlier_track_dust_false_positive_rate"],
                reported["association"]["inlier_track_dust_false_positive_rate"],
            ),
            "observation_outlier_recall": _close(
                metrics["outlier_observation_dust_recall"],
                reported["association"]["outlier_observation_dust_recall"],
            ),
            "observation_inlier_fpr": _close(
                metrics["inlier_observation_dust_false_positive_rate"],
                reported["association"]["inlier_observation_dust_false_positive_rate"],
            ),
            "center_error_p90": _close(
                metrics["center_error_p90"], reported["geometry"]["center_error_p90"]
            ),
            "source_center": _close(
                metrics["source_center_max_px"], reported["geometry"]["source_center_max_px"]
            ),
            "source_covariance": _close(
                metrics["source_covariance_relative_max"],
                reported["geometry"]["source_covariance_relative_max"],
            ),
            "finite_spd": metrics["finite_spd"] is reported["geometry"]["finite_spd"],
            "depth_bounds": (
                metrics["depth_bound_incidence"] == reported["geometry"]["depth_bound_incidence"]
            ),
            "heldout_accuracy": _close(
                metrics["heldout_parent_assignment_accuracy"],
                reported["heldout"]["heldout_parent_assignment_accuracy"],
            ),
            "heldout_denominator": (
                metrics["heldout_assignment_denominator"]
                == reported["heldout"]["heldout_assignment_denominator"]
            ),
            "transport_mass_pass": (
                metrics["transport_mass_diagnostics"]["pass"]
                is reported["transport_mass_diagnostics"]["pass"]
            ),
        }
        comparisons[arm] = {"checks": checks, "pass": all(checks.values())}

    def accepted(arm: str) -> bool:
        item = recomputed[arm]
        return bool(
            item["parent_purity"] >= 0.90
            and item["parent_completeness"] >= 0.90
            and item["outlier_track_dust_recall"] >= 0.80
            and item["outlier_observation_dust_recall"] >= 0.80
            and item["inlier_track_dust_false_positive_rate"] <= 0.20
            and item["inlier_observation_dust_false_positive_rate"] <= 0.20
            and item["transport_mass_diagnostics"]["pass"]
        )

    area = recomputed["uot_area"]
    payload = {
        "namespace": "rtgs.inverse-projection-fiber.iter3.failure-audit.v1",
        "status": "OFFICIAL_EXECUTION_FAILURE",
        "attempt": {
            "path": str(ATTEMPT),
            "sha256": _sha256(ATTEMPT),
            "receipt_status": attempt["status"],
            "roots": attempt["root_tuples"],
            "source_hashes": attempt["source_hashes"],
        },
        "transaction": {
            "official_result_absent": not RESULT.exists(),
            "completed_root_indices": [0],
            "partial_root_indices": [1],
            "unstarted_root_indices": [2],
            "real_namespace_untouched": not any(
                path.exists() for path in (REAL_ATTEMPT, REAL_RESULT, REAL_ARTIFACTS)
            ),
        },
        "artifacts": {
            "root_0_result": {"path": str(ROOT0_JSON), "sha256": _sha256(ROOT0_JSON)},
            "root_0_evidence": {"path": str(ROOT0_NPZ), "sha256": _sha256(ROOT0_NPZ)},
            "root_1_initial": {"path": str(ROOT1_INITIAL), "sha256": _sha256(ROOT1_INITIAL)},
        },
        "root_0_recomputed": recomputed,
        "root_0_reported_metric_checks": comparisons,
        "root_0_all_reported_metrics_match": all(item["pass"] for item in comparisons.values()),
        "root_0_transport_acceptance": {
            "uot_uniform": accepted("uot_uniform"),
            "uot_area": accepted("uot_area"),
            "future_roots_cannot_restore_real_release": not accepted("uot_uniform")
            and not accepted("uot_area"),
        },
        "root_0_effects": {
            "area_minus_hard_purity": area["parent_purity"]
            - recomputed["hardmin"]["parent_purity"],
            "area_minus_row_purity": area["parent_purity"] - recomputed["row"]["parent_purity"],
            "area_minus_uniform_purity": area["parent_purity"]
            - recomputed["uot_uniform"]["parent_purity"],
            "area_minus_shuffled_purity": area["parent_purity"]
            - recomputed["shuffled_view"]["parent_purity"],
        },
        "root_0_decomposition_granularity": {
            "definition": (
                "distance from each inlier split-child center to its view/parent moment mean"
            ),
            "count": len(fragment_offsets),
            "nonzero_fraction": float(np.mean(np.asarray(fragment_offsets) > 1e-12)),
            "median_px": float(np.median(fragment_offsets)),
            "p90_px": float(np.quantile(fragment_offsets, 0.9)),
            "maximum_px": float(np.max(fragment_offsets)),
        },
        "scope": (
            "Root 0 is valid partial evidence; the three-root synthetic result is incomplete and "
            "cannot pass. No official-root replay was performed."
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
