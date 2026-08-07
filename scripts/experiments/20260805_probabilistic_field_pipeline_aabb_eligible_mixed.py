#!/usr/bin/env python3
"""Run the support-fallback matrix with forward-AABB-eligible fixed anchors.

This successor deliberately reuses the frozen support-fallback producer.  The reused driver is
bound by an exact byte digest before import; this wrapper then changes only the task/run identity
and requires the new, source-level fixed-anchor eligibility policy in the reviewed task.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "20260805_probabilistic_field_pipeline_aabb_eligible_mixed"
TASK_RELATIVE = Path("experiments/tasks") / f"{TASK_ID}.json"
RUN_RELATIVE = Path("runs") / TASK_ID
DRIVER_RELATIVE = Path("scripts/experiments") / f"{TASK_ID}.py"
BASE_DRIVER_RELATIVE = Path(
    "scripts/experiments/20260805_probabilistic_field_pipeline_support_fallback_mixed.py"
)
BASE_DRIVER_SHA256 = "9d453b967b09005b63d3bef6aac48b817ac841ea2b7faa3593d4f980e4310169"
ANCHOR_ELIGIBILITY_CONTRACT = {
    "policy": "forward_search_aabb_intersection_v1",
    "selection_order": "train_only_bounds_then_filter_then_capacity_balance_then_seeded_mass_pool",
    "failure_policy": "fail_if_global_eligible_count_below_n_init_3d",
    "arm_scope": "identical_native_and_candidate_preprocessing",
    "outcome_inputs": [],
    "required_diagnostics": [
        "anchor_candidate_count",
        "anchor_forward_aabb_eligible_count",
        "anchor_forward_aabb_rejected_count",
        "anchor_forward_aabb_eligible_counts_per_view",
        "anchor_forward_aabb_rejected_counts_per_view",
    ],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_base_path = ROOT / BASE_DRIVER_RELATIVE
if _sha256(_base_path) != BASE_DRIVER_SHA256:
    raise RuntimeError("pinned support-fallback base driver digest does not match")
_spec = importlib.util.spec_from_file_location("rtgs_support_fallback_driver_base", _base_path)
if _spec is None or _spec.loader is None:  # pragma: no cover - importlib defensive boundary
    raise ImportError(f"cannot load pinned base driver: {_base_path}")
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)

# Every function in the reused producer resolves these globals dynamically.  Updating all four
# identities, including ``__file__`` for spawned workers, keeps every subprocess and receipt under
# the successor's single canonical task/run root.
_base.__file__ = str(Path(__file__).resolve())
_base.TASK_ID = TASK_ID
_base.TASK_RELATIVE = TASK_RELATIVE
_base.RUN_RELATIVE = RUN_RELATIVE
_base.DRIVER_RELATIVE = DRIVER_RELATIVE

_base_assert_task_contract = _base._assert_task_contract
_base_enforce_result_invariants = _base._enforce_result_invariants


def _assert_task_contract(task: Mapping[str, Any]) -> None:
    """Require both the reused-driver byte binding and the new geometry policy."""

    _base_assert_task_contract(task)
    configuration = _base._task_configuration(task)
    if configuration.get("base_driver_binding") != {
        "path": BASE_DRIVER_RELATIVE.as_posix(),
        "algorithm": "sha256-bytes-v1",
        "sha256": BASE_DRIVER_SHA256,
    }:
        raise ValueError("task base-driver binding does not match this successor")
    if configuration.get("fixed_anchor_eligibility") != ANCHOR_ELIGIBILITY_CONTRACT:
        raise ValueError("task fixed-anchor eligibility contract does not match this successor")


_base._assert_task_contract = _assert_task_contract


def _validate_anchor_eligibility_diagnostics(result: Any) -> None:
    """Fail closed unless one fit carries an internally consistent eligibility receipt."""

    diagnostics = result.diagnostics.get("placement")
    if not isinstance(diagnostics, dict):
        raise RuntimeError("fixed-anchor eligibility diagnostics are missing")
    if diagnostics.get("anchor_eligibility_policy") != ANCHOR_ELIGIBILITY_CONTRACT["policy"]:
        raise RuntimeError("fixed-anchor eligibility policy diagnostic does not match")
    scalar_keys = (
        "n_init_3d",
        "anchor_candidate_count",
        "anchor_forward_aabb_eligible_count",
        "anchor_forward_aabb_rejected_count",
    )
    if any(type(diagnostics.get(key)) is not int for key in scalar_keys):
        raise RuntimeError("fixed-anchor eligibility scalar diagnostics have wrong JSON types")
    if diagnostics["n_init_3d"] <= 0:
        raise RuntimeError("fixed-anchor requested budget must be positive")
    eligible_per_view = diagnostics.get("anchor_forward_aabb_eligible_counts_per_view")
    rejected_per_view = diagnostics.get("anchor_forward_aabb_rejected_counts_per_view")
    component_counts = result.diagnostics.get("target_component_counts_used")
    optimized_views = result.optimized_view_indices
    if (
        not isinstance(eligible_per_view, list)
        or not isinstance(rejected_per_view, list)
        or len(eligible_per_view) != len(optimized_views)
        or len(rejected_per_view) != len(optimized_views)
        or any(type(value) is not int or value < 0 for value in eligible_per_view)
        or any(type(value) is not int or value < 0 for value in rejected_per_view)
    ):
        raise RuntimeError("fixed-anchor per-view eligibility diagnostics are invalid")
    if (
        not isinstance(component_counts, list)
        or any(type(value) is not int or value <= 0 for value in component_counts)
        or any(
            type(index) is not int or index < 0 or index >= len(component_counts)
            for index in optimized_views
        )
        or len(set(optimized_views)) != len(optimized_views)
    ):
        raise RuntimeError("fixed-anchor candidate-capacity diagnostics are invalid")
    optimized_component_counts = [component_counts[index] for index in optimized_views]
    if any(
        eligible + rejected != capacity
        for eligible, rejected, capacity in zip(
            eligible_per_view,
            rejected_per_view,
            optimized_component_counts,
            strict=True,
        )
    ):
        raise RuntimeError("fixed-anchor per-view eligibility exceeds candidate capacity")
    candidate_count = diagnostics["anchor_candidate_count"]
    eligible_count = diagnostics["anchor_forward_aabb_eligible_count"]
    rejected_count = diagnostics["anchor_forward_aabb_rejected_count"]
    if (
        candidate_count < 0
        or eligible_count < diagnostics["n_init_3d"]
        or rejected_count < 0
        or eligible_count != sum(eligible_per_view)
        or rejected_count != sum(rejected_per_view)
        or candidate_count != eligible_count + rejected_count
        or candidate_count != sum(optimized_component_counts)
    ):
        raise RuntimeError("fixed-anchor eligibility counts are inconsistent")


def _enforce_result_invariants(
    task: Mapping[str, Any],
    result: Any,
    *,
    association_required: bool,
) -> dict[str, object]:
    _validate_anchor_eligibility_diagnostics(result)
    return _base_enforce_result_invariants(
        task,
        result,
        association_required=association_required,
    )


_base._enforce_result_invariants = _enforce_result_invariants
compile_cell_plan = _base.compile_cell_plan
plan_payload = _base.plan_payload
main = _base.main


def __getattr__(name: str) -> object:
    return getattr(_base, name)


if __name__ == "__main__":
    main()
