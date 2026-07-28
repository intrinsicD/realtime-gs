#!/usr/bin/env python3
"""Validate task-first experiments and render their canonical results page.

New result-bearing experiments use one immutable task id:

    YYYYMMDD_<task_slug>_<data_slug>

The task is registered under ``experiments/tasks/`` before a run starts. ``init-run`` binds a
run directory to the exact task, data seal, command, and source state. ``render`` consumes the
common ``metrics.json`` schema and writes the only supported experiment ``index.html`` template.

Typical use:

    python scripts/experiment_contract.py validate
    python scripts/experiment_contract.py validate-data experiments/tasks/<task_id>.json
    python scripts/experiment_contract.py init-run experiments/tasks/<task_id>.json
    python scripts/experiment_contract.py render runs/<task_id>
    python scripts/experiment_contract.py check-run runs/<task_id>
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import math
import os
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TASK_SCHEMA_VERSION = 1
PROGRAM_SCHEMA_VERSION = 1
DATA_SEAL_SCHEMA_VERSION = 1
REPORT_TEMPLATE_VERSION = 1

ARMS = ("direct_compact", "beam_fusion", "rgb_3dgs")
TASK_STATUSES = ("draft", "ready", "blocked")
EVIDENCE_PHASES = ("development", "confirmatory")
REQUIRED_CHARTS = ("quality", "resources", "stage_runtime")
REQUIRED_MODEL_ARTIFACTS = (
    "gaussians_init.ply",
    "gaussians.ply",
    "training_history.json",
    "gaussians.config.json",
    "input_boundary_receipt.json",
    "resource_receipt.json",
)
EVIDENCE_SUFFIXES = ("RESULT.md", "RESULT.json", "AUDIT.md", "AUDIT.json")
COMPACT_ALLOWED = {"calibration", "gaussians2d"}
COMPACT_FORBIDDEN = {
    "PIL",
    "SceneData",
    "mask",
    "rgb",
    "rtgs.data.calibrated",
    "rtgs.optim.trainer",
}
DIRECT_COMPACT_FORBIDDEN = {
    "rtgs.carrier_pipeline",
    "rtgs.lift.beam_fusion",
    "rtgs.lift.carrier_refinement",
    "rtgs.optim.carrier_schedule",
}
SLUG_RE = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*\Z")
VIEW_ID_RE = re.compile(r"C\d{4}\Z")


class DuplicateKeyError(ValueError):
    """Raised when JSON repeats a key and would otherwise silently overwrite it."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError, DuplicateKeyError) as error:
        raise ValueError(f"{path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _is_safe_relative(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _strings(value: object, *, nonempty: bool = True) -> bool:
    return (
        isinstance(value, list)
        and (bool(value) or not nonempty)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def _task_id(task: dict[str, Any]) -> str:
    return f"{task.get('date', '')}_{task.get('task_slug', '')}_{task.get('data_slug', '')}"


def validate_task(task: dict[str, Any], path: Path, *, root: Path = ROOT) -> list[str]:
    """Return all structural and policy violations for one task."""

    errors: list[str] = []
    required = {
        "schema_version",
        "task_id",
        "date",
        "task_slug",
        "data_slug",
        "arm",
        "status",
        "owner",
        "depends_on",
        "title",
        "question",
        "hypothesis",
        "claim_boundary",
        "evidence_phase",
        "datasets",
        "splits",
        "seeds",
        "input_policy",
        "execution_guards",
        "stages",
        "comparators",
        "primary_metrics",
        "required_charts",
        "resource_protocol",
        "data_seal",
        "run_command",
        "blockers",
    }
    missing = sorted(required - set(task))
    if missing:
        errors.append(f"missing keys: {', '.join(missing)}")
        return errors

    if task["schema_version"] != TASK_SCHEMA_VERSION:
        errors.append(f"schema_version must be {TASK_SCHEMA_VERSION}")
    task_id = task["task_id"]
    if not isinstance(task_id, str) or task_id != _task_id(task):
        errors.append("task_id must equal <date>_<task_slug>_<data_slug>")
    if isinstance(task_id, str) and path.name != f"{task_id}.json":
        errors.append(f"task filename must be {task_id}.json")
    try:
        dt.datetime.strptime(str(task["date"]), "%Y%m%d")
    except ValueError:
        errors.append("date must be a valid YYYYMMDD value")
    for key in ("task_slug", "data_slug"):
        if not isinstance(task[key], str) or SLUG_RE.fullmatch(task[key]) is None:
            errors.append(f"{key} must be lowercase snake_case")
    if task["arm"] not in ARMS:
        errors.append(f"arm must be one of: {', '.join(ARMS)}")
    if task["status"] not in TASK_STATUSES:
        errors.append(f"status must be one of: {', '.join(TASK_STATUSES)}")
    if task["owner"] is not None and (
        not isinstance(task["owner"], str) or not task["owner"].strip()
    ):
        errors.append("owner must be null or a non-empty agent/human identifier")
    if task["status"] == "ready" and task["owner"] is None:
        errors.append("ready tasks require a frozen owner")
    if not isinstance(task["depends_on"], list) or not all(
        isinstance(item, str) and item.strip() for item in task["depends_on"]
    ):
        errors.append("depends_on must be a list of task ids")
    elif len(task["depends_on"]) != len(set(task["depends_on"])):
        errors.append("depends_on must not contain duplicates")
    elif task_id in task["depends_on"]:
        errors.append("a task cannot depend on itself")
    if task["evidence_phase"] not in EVIDENCE_PHASES:
        errors.append(f"evidence_phase must be one of: {', '.join(EVIDENCE_PHASES)}")
    for key in ("title", "question", "hypothesis", "claim_boundary"):
        if not isinstance(task[key], str) or not task[key].strip():
            errors.append(f"{key} must be a non-empty string")

    datasets = task["datasets"]
    dataset_ids: list[str] = []
    if not isinstance(datasets, list) or not datasets:
        errors.append("datasets must be a non-empty list")
    else:
        for index, dataset in enumerate(datasets):
            label = f"datasets[{index}]"
            if not isinstance(dataset, dict):
                errors.append(f"{label} must be an object")
                continue
            needed = {
                "id",
                "role",
                "frame_path",
                "compact_manifest",
                "calibration",
                "rgb_pattern",
                "mask_pattern",
            }
            absent = sorted(needed - set(dataset))
            if absent:
                errors.append(f"{label} missing: {', '.join(absent)}")
                continue
            dataset_id = dataset["id"]
            if not isinstance(dataset_id, str) or SLUG_RE.fullmatch(dataset_id) is None:
                errors.append(f"{label}.id must be lowercase snake_case")
            else:
                dataset_ids.append(dataset_id)
            if dataset["role"] not in {"development", "replication", "confirmation"}:
                errors.append(f"{label}.role is invalid")
            for key in ("frame_path", "compact_manifest", "calibration"):
                if not _is_safe_relative(dataset[key]):
                    errors.append(f"{label}.{key} must be a repository-relative path")
            if dataset["rgb_pattern"] != "rgb/C*.jpg":
                errors.append(f"{label}.rgb_pattern must select only canonical RGB JPEGs")
            if dataset["mask_pattern"] != "mask/mask_C*.png":
                errors.append(f"{label}.mask_pattern must select only lossless PNG masks")
        if len(dataset_ids) != len(set(dataset_ids)):
            errors.append("dataset ids must be unique")

    splits = task["splits"]
    if not isinstance(splits, dict):
        errors.append("splits must be an object keyed by dataset id")
    else:
        if set(splits) != set(dataset_ids):
            errors.append("splits keys must exactly match dataset ids")
        for dataset_id, split in splits.items():
            if not isinstance(split, dict) or set(split) != {"train", "heldout"}:
                errors.append(f"splits.{dataset_id} must contain exactly train and heldout")
                continue
            if not _strings(split["train"]) or not _strings(split["heldout"]):
                errors.append(f"splits.{dataset_id} train/heldout must be non-empty string lists")
                continue
            all_views = split["train"] + split["heldout"]
            if any(VIEW_ID_RE.fullmatch(view_id) is None for view_id in all_views):
                errors.append(f"splits.{dataset_id} contains a non-canonical camera id")
            if len(all_views) != len(set(all_views)):
                errors.append(f"splits.{dataset_id} train and heldout must be disjoint")

    seeds = task["seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) < 3
        or not all(isinstance(seed, int) and seed >= 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        errors.append("seeds must contain at least three unique non-negative integers")

    policy = task["input_policy"]
    policy_keys = {"reconstruction_allowed", "reconstruction_forbidden", "evaluation_allowed"}
    if not isinstance(policy, dict) or set(policy) != policy_keys:
        errors.append(
            "input_policy must contain reconstruction_allowed, reconstruction_forbidden, "
            "and evaluation_allowed"
        )
    elif not all(_strings(policy[key]) for key in policy_keys):
        errors.append("every input_policy value must be a non-empty string list")
    elif task["arm"] in {"direct_compact", "beam_fusion"}:
        if set(policy["reconstruction_allowed"]) != COMPACT_ALLOWED:
            errors.append("compact arms may reconstruct from calibration and gaussians2d only")
        missing_forbidden = sorted(COMPACT_FORBIDDEN - set(policy["reconstruction_forbidden"]))
        if missing_forbidden:
            errors.append(
                "compact arm reconstruction_forbidden is missing: " + ", ".join(missing_forbidden)
            )
        if set(policy["evaluation_allowed"]) != COMPACT_ALLOWED:
            errors.append("compact-arm evaluation must also remain image-free")
        if task["arm"] == "direct_compact":
            missing_direct_forbidden = sorted(
                DIRECT_COMPACT_FORBIDDEN - set(policy["reconstruction_forbidden"])
            )
            if missing_direct_forbidden:
                errors.append(
                    "direct_compact reconstruction_forbidden is missing: "
                    + ", ".join(missing_direct_forbidden)
                )
    elif task["arm"] == "rgb_3dgs":
        required_rgb = {"calibration", "gaussians3d_initialization", "mask", "rgb"}
        if not required_rgb <= set(policy["reconstruction_allowed"]):
            errors.append("rgb_3dgs reconstruction_allowed is missing RGB baseline modalities")

    guards = task["execution_guards"]
    if not _strings(guards):
        errors.append("execution_guards must be a non-empty string list")
    elif task["arm"] in {"direct_compact", "beam_fusion"}:
        compact_guards = {
            "deny_image_capable_imports",
            "deny_image_suffix_open",
            "forbidden_modules_absent_at_exit",
            "negative_controls",
        }
        missing_guards = sorted(compact_guards - set(guards))
        if missing_guards:
            errors.append("compact execution_guards is missing: " + ", ".join(missing_guards))
        if task["arm"] == "direct_compact" and "deny_beam_imports" not in guards:
            errors.append("direct_compact execution_guards is missing: deny_beam_imports")
    elif task["arm"] == "rgb_3dgs":
        rgb_guards = {"deny_heldout_training_access", "split_hash_at_exit"}
        missing_guards = sorted(rgb_guards - set(guards))
        if missing_guards:
            errors.append("rgb_3dgs execution_guards is missing: " + ", ".join(missing_guards))

    stages = task["stages"]
    if not isinstance(stages, list) or not stages:
        errors.append("stages must be a non-empty list")
    else:
        stage_ids: list[str] = []
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict) or set(stage) != {"id", "label", "purpose"}:
                errors.append(f"stages[{index}] must contain exactly id, label, and purpose")
                continue
            if not isinstance(stage["id"], str) or SLUG_RE.fullmatch(stage["id"]) is None:
                errors.append(f"stages[{index}].id must be lowercase snake_case")
            else:
                stage_ids.append(stage["id"])
            if not all(
                isinstance(stage[key], str) and stage[key].strip() for key in ("label", "purpose")
            ):
                errors.append(f"stages[{index}] label/purpose must be non-empty")
        if len(stage_ids) != len(set(stage_ids)):
            errors.append("stage ids must be unique")
        if task["arm"] == "direct_compact":
            stage_text = json.dumps(stages, sort_keys=True).lower()
            if "beam" in stage_text or "carrier" in stage_text:
                errors.append("direct_compact stages must not contain Beam/carrier mechanisms")

    if not isinstance(task["comparators"], list) or not task["comparators"]:
        errors.append("comparators must be a non-empty list")
    elif not all(
        isinstance(item, dict)
        and set(item) == {"id", "label", "purpose"}
        and isinstance(item["id"], str)
        and SLUG_RE.fullmatch(item["id"]) is not None
        and isinstance(item["label"], str)
        and bool(item["label"].strip())
        and isinstance(item["purpose"], str)
        and bool(item["purpose"].strip())
        for item in task["comparators"]
    ):
        errors.append("every comparator must contain canonical id, label, and purpose")

    metrics = task["primary_metrics"]
    required_metric_keys = {"id", "label", "unit", "direction", "aggregation"}
    if not isinstance(metrics, list) or not metrics:
        errors.append("primary_metrics must be a non-empty list")
    else:
        metric_ids: list[str] = []
        for index, metric in enumerate(metrics):
            if not isinstance(metric, dict) or set(metric) != required_metric_keys:
                errors.append(f"primary_metrics[{index}] has the wrong keys")
                continue
            if not isinstance(metric["id"], str) or SLUG_RE.fullmatch(metric["id"]) is None:
                errors.append(f"primary_metrics[{index}].id must be lowercase snake_case")
            else:
                metric_ids.append(metric["id"])
            if metric["direction"] not in {"lower", "higher", "descriptive"}:
                errors.append(f"primary_metrics[{index}].direction is invalid")
            for key in ("label", "unit", "aggregation"):
                if not isinstance(metric[key], str) or not metric[key].strip():
                    errors.append(f"primary_metrics[{index}].{key} must be non-empty")
        if len(metric_ids) != len(set(metric_ids)):
            errors.append("primary metric ids must be unique")

    if task["required_charts"] != list(REQUIRED_CHARTS):
        errors.append(f"required_charts must be exactly: {', '.join(REQUIRED_CHARTS)}")

    resources = task["resource_protocol"]
    resource_keys = {
        "scope",
        "warmup_runs",
        "measured_runs",
        "aggregation",
        "cuda_metrics",
        "host_metrics",
    }
    if not isinstance(resources, dict) or set(resources) != resource_keys:
        errors.append("resource_protocol has the wrong keys")
    else:
        if not isinstance(resources["warmup_runs"], int) or resources["warmup_runs"] < 1:
            errors.append("resource_protocol.warmup_runs must be at least one")
        if not isinstance(resources["measured_runs"], int) or resources["measured_runs"] < 3:
            errors.append("resource_protocol.measured_runs must be at least three")
        for key in ("scope", "aggregation"):
            if not isinstance(resources[key], str) or not resources[key].strip():
                errors.append(f"resource_protocol.{key} must be non-empty")
        for key in ("cuda_metrics", "host_metrics"):
            if not _strings(resources[key]):
                errors.append(f"resource_protocol.{key} must be a non-empty string list")

    if not _is_safe_relative(task["data_seal"]):
        errors.append("data_seal must be a repository-relative path")
    else:
        seal_path = root / task["data_seal"]
        if not seal_path.is_file():
            errors.append(f"data_seal does not exist: {task['data_seal']}")
        else:
            try:
                seal = _load_json(seal_path)
            except ValueError as error:
                errors.append(str(error))
            else:
                expected_profile = (
                    "compact" if task["arm"] in {"direct_compact", "beam_fusion"} else "rgb"
                )
                if seal.get("input_profile") != expected_profile:
                    errors.append(
                        f"data_seal input_profile must be {expected_profile!r} for this arm"
                    )
                sealed_paths = [
                    item.get("path", "") for item in seal.get("files", []) if isinstance(item, dict)
                ]
                if expected_profile == "compact" and any(
                    "/rgb/" in value or "/mask/" in value for value in sealed_paths
                ):
                    errors.append("compact data_seal must not bind RGB or mask files")
                if expected_profile == "rgb" and (
                    not any("/rgb/" in value for value in sealed_paths)
                    or not any("/mask/" in value for value in sealed_paths)
                ):
                    errors.append("rgb data_seal must bind both RGB and mask files")

    command = task["run_command"]
    if command is not None and not _strings(command):
        errors.append("run_command must be null or a non-empty argv list")
    if task["status"] == "ready" and not _strings(command):
        errors.append(f"{task['status']} tasks require a frozen run_command")
    elif task["status"] == "ready":
        expected_driver = f"scripts/experiments/{task_id}.py"
        if expected_driver not in command:
            errors.append(f"ready task run_command must name {expected_driver}")
        if not (root / expected_driver).is_file():
            errors.append(f"ready task driver does not exist: {expected_driver}")
    if not isinstance(task["blockers"], list) or not all(
        isinstance(item, str) and item.strip() for item in task["blockers"]
    ):
        errors.append("blockers must be a list of non-empty strings")
    if task["status"] == "ready" and task["blockers"]:
        errors.append(f"{task['status']} tasks cannot retain blockers")
    return errors


def validate_program(
    program: dict[str, Any],
    path: Path,
    tasks: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "program_id",
        "date",
        "data_slug",
        "title",
        "report_template_version",
        "arms",
    }
    missing = sorted(required - set(program))
    if missing:
        return [f"missing keys: {', '.join(missing)}"]
    if program["schema_version"] != PROGRAM_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PROGRAM_SCHEMA_VERSION}")
    expected_id = f"{program['date']}_three_claim_arms_{program['data_slug']}"
    if program["program_id"] != expected_id:
        errors.append("program_id must equal <date>_three_claim_arms_<data_slug>")
    if path.name != f"{program['program_id']}.json":
        errors.append(f"program filename must be {program['program_id']}.json")
    if program["report_template_version"] != REPORT_TEMPLATE_VERSION:
        errors.append(f"report_template_version must be {REPORT_TEMPLATE_VERSION}")
    arms = program["arms"]
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        errors.append(f"arms must contain exactly: {', '.join(ARMS)}")
    else:
        if len(set(arms.values())) != len(ARMS):
            errors.append("program arms must reference three distinct task ids")
        for arm, task_id in arms.items():
            task = tasks.get(task_id)
            if task is None:
                errors.append(f"{arm} references unknown task {task_id!r}")
            elif task.get("arm") != arm:
                errors.append(f"{task_id} declares arm {task.get('arm')!r}, expected {arm!r}")
            elif (
                task.get("date") != program["date"] or task.get("data_slug") != program["data_slug"]
            ):
                errors.append(f"{task_id} does not share the program date/data slug")
    return errors


def validate_repository(*, root: Path = ROOT) -> list[str]:
    """Validate every registered task and program without touching large local data."""

    errors: list[str] = []
    task_dir = root / "experiments" / "tasks"
    program_dir = root / "experiments" / "programs"
    tasks: dict[str, dict[str, Any]] = {}
    if not task_dir.is_dir():
        return ["missing experiments/tasks"]
    for path in sorted(task_dir.glob("*.json")):
        try:
            task = _load_json(path)
        except ValueError as error:
            errors.append(str(error))
            continue
        task_id = task.get("task_id")
        if isinstance(task_id, str):
            if task_id in tasks:
                errors.append(f"duplicate task_id: {task_id}")
            tasks[task_id] = task
        errors.extend(
            f"{path.relative_to(root)}: {item}" for item in validate_task(task, path, root=root)
        )
    if not tasks:
        errors.append("no experiment tasks registered")
    for task_id, task in tasks.items():
        for dependency in task.get("depends_on", []):
            if dependency not in tasks:
                errors.append(f"{task_id}: depends_on references unknown task {dependency!r}")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            errors.append(f"task dependency cycle includes {task_id}")
            return
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in tasks[task_id].get("depends_on", []):
            if dependency in tasks:
                visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in tasks:
        visit(task_id)

    if not program_dir.is_dir():
        errors.append("missing experiments/programs")
    else:
        programs = sorted(program_dir.glob("*.json"))
        if not programs:
            errors.append("no experiment programs registered")
        for path in programs:
            try:
                program = _load_json(path)
            except ValueError as error:
                errors.append(str(error))
                continue
            errors.extend(
                f"{path.relative_to(root)}: {item}"
                for item in validate_program(program, path, tasks)
            )
    return errors


def _dataset_files(task: dict[str, Any], *, root: Path) -> tuple[dict[str, Any], list[Path]]:
    """Resolve only the files permitted by the task arm's reconstruction boundary."""

    datasets_payload: list[dict[str, Any]] = []
    files: list[Path] = []
    compact_profile = task["arm"] in {"direct_compact", "beam_fusion"}
    for dataset in task["datasets"]:
        frame = root / dataset["frame_path"]
        manifest_path = root / dataset["compact_manifest"]
        manifest = _load_json(manifest_path)
        views = manifest.get("views")
        if not isinstance(views, list) or not views:
            raise ValueError(f"{manifest_path}: compact manifest has no views")
        view_ids = [view.get("view_id") for view in views if isinstance(view, dict)]
        if len(view_ids) != len(views) or any(
            not isinstance(view_id, str) or VIEW_ID_RE.fullmatch(view_id) is None
            for view_id in view_ids
        ):
            raise ValueError(f"{manifest_path}: invalid compact view ids")

        split = task["splits"][dataset["id"]]
        if set(split["train"] + split["heldout"]) != set(view_ids):
            raise ValueError(
                f"{task['task_id']}: split for {dataset['id']} must cover every compact view"
            )

        files.extend([root / dataset["calibration"], manifest_path])
        for view in views:
            view_id = view["view_id"]
            if compact_profile:
                files.append(manifest_path.parent / view["path"])
            else:
                files.extend(
                    [
                        frame / f"rgb/{view_id}.jpg",
                        frame / f"mask/mask_{view_id}.png",
                    ]
                )
        datasets_payload.append(
            {
                "id": dataset["id"],
                "role": dataset["role"],
                "frame_path": dataset["frame_path"],
                "view_ids": view_ids,
                "selected_modalities": (
                    ["calibration", "gaussians2d"]
                    if compact_profile
                    else ["calibration", "rgb", "mask"]
                ),
                "canonical_rgb_pattern": (None if compact_profile else dataset["rgb_pattern"]),
                "canonical_mask_pattern": (None if compact_profile else dataset["mask_pattern"]),
            }
        )

    unique: dict[str, Path] = {}
    for path in files:
        try:
            relative = path.resolve(strict=True).relative_to(root.resolve())
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(f"missing or out-of-repository data file: {path}") from error
        unique[relative.as_posix()] = path
    return {"datasets": datasets_payload}, [unique[key] for key in sorted(unique)]


def build_data_seal(task: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    dataset_payload, files = _dataset_files(task, root=root)
    records = []
    for path in files:
        records.append(
            {
                "path": path.resolve().relative_to(root.resolve()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "schema_version": DATA_SEAL_SCHEMA_VERSION,
        "data_slug": task["data_slug"],
        "input_profile": "compact" if task["arm"] in {"direct_compact", "beam_fusion"} else "rgb",
        **dataset_payload,
        "files": records,
    }


def verify_data_seal(task: dict[str, Any], *, root: Path = ROOT) -> list[str]:
    path = root / task["data_seal"]
    try:
        stored = _load_json(path)
        current = build_data_seal(task, root=root)
    except ValueError as error:
        return [str(error)]
    if stored != current:
        return [
            f"{task['data_seal']} does not match the selected data bytes; "
            "do not run until the task/data seal is deliberately refreshed"
        ]
    return []


def _git_output(root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout


def init_run(task_path: Path, *, root: Path = ROOT, development: bool = False) -> Path:
    """Create a non-overwriting run root and lock its exact task/data/source inputs."""

    task_path = task_path.resolve(strict=True)
    task = _load_json(task_path)
    errors = validate_task(task, task_path, root=root)
    if errors:
        raise ValueError("invalid task:\n- " + "\n- ".join(errors))
    if task["status"] != "ready":
        raise ValueError("init-run requires task status 'ready'")
    seal_errors = verify_data_seal(task, root=root)
    if seal_errors:
        raise ValueError("\n".join(seal_errors))
    for dependency in task["depends_on"]:
        dependency_errors = validate_run(root / "runs" / dependency, root=root)
        if dependency_errors:
            raise ValueError(
                f"dependency {dependency} is not a complete canonical run:\n- "
                + "\n- ".join(dependency_errors)
            )

    dirty = bool(_git_output(root, "status", "--porcelain", "--untracked-files=all").strip())
    if dirty and not development:
        raise ValueError(
            "official runs require a clean tracked worktree (use --development only for dev)"
        )
    run = root / "runs" / task["task_id"]
    if run.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {run}")

    task_relative = task_path.relative_to(root.resolve()).as_posix()
    seal_path = root / task["data_seal"]
    diff = _git_output(root, "diff", "--binary", "HEAD").encode("utf-8")
    lock = {
        "schema_version": 1,
        "task_id": task["task_id"],
        "task_path": task_relative,
        "task_sha256": _sha256_file(task_path),
        "data_seal_path": task["data_seal"],
        "data_seal_sha256": _sha256_file(seal_path),
        "source_commit": _git_output(root, "rev-parse", "HEAD").strip(),
        "source_dirty": dirty,
        "source_diff_sha256": _sha256_bytes(diff),
        "development": development,
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": task["run_command"],
        "report_template_version": REPORT_TEMPLATE_VERSION,
    }
    run.mkdir(parents=True)
    _write_json(run / "task.lock.json", lock)
    return run


def _metric_errors(payload: dict[str, Any], task: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "report_template_version",
        "task_id",
        "summary",
        "decision",
        "claim_boundary",
        "metrics",
        "metric_metadata",
        "charts",
        "artifacts",
        "evidence",
        "viewer_command",
        "notes",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"metrics.json missing: {', '.join(missing)}"]
    if payload["schema_version"] != 1:
        errors.append("metrics.json schema_version must be 1")
    if payload["report_template_version"] != REPORT_TEMPLATE_VERSION:
        errors.append(f"report_template_version must be {REPORT_TEMPLATE_VERSION}")
    if payload["task_id"] != task["task_id"]:
        errors.append("metrics.json task_id does not match the locked task")
    for key in ("summary", "decision", "claim_boundary"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            errors.append(f"metrics.json {key} must be non-empty")
    if payload["claim_boundary"] != task["claim_boundary"]:
        errors.append("metrics.json claim_boundary must exactly match the frozen task")

    metrics = payload["metrics"]
    metadata = payload["metric_metadata"]
    if not isinstance(metrics, dict) or not metrics:
        errors.append("metrics must be a non-empty object")
    elif not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in metrics.values()
    ):
        errors.append("every metric must be a finite number")
    if not isinstance(metadata, dict) or set(metadata) != set(metrics):
        errors.append("metric_metadata keys must exactly match metrics")
    else:
        meta_keys = {"label", "unit", "group", "direction"}
        for metric_id, item in metadata.items():
            if (
                not isinstance(item, dict)
                or set(item) != meta_keys
                or item.get("direction") not in {"lower", "higher", "descriptive"}
                or not all(
                    isinstance(item.get(key), str) and bool(item[key].strip())
                    for key in ("label", "unit", "group")
                )
            ):
                errors.append(f"metric_metadata.{metric_id} is invalid")
    task_metrics = {item["id"]: item for item in task["primary_metrics"]}
    missing_task_metrics = sorted(set(task_metrics) - set(metrics))
    if missing_task_metrics:
        errors.append(
            "metrics is missing frozen primary metrics: " + ", ".join(missing_task_metrics)
        )
    elif isinstance(metadata, dict):
        for metric_id, task_metric in task_metrics.items():
            report_metric = metadata.get(metric_id)
            if (
                isinstance(report_metric, dict)
                and report_metric.get("direction") != task_metric["direction"]
            ):
                errors.append(f"metric_metadata.{metric_id}.direction changed from the frozen task")

    charts = payload["charts"]
    if not isinstance(charts, list):
        errors.append("charts must be a list")
    else:
        chart_ids = [chart.get("id") for chart in charts if isinstance(chart, dict)]
        if chart_ids != task["required_charts"]:
            errors.append("charts must appear once in the frozen required_charts order")
        for index, chart in enumerate(charts):
            if not isinstance(chart, dict) or set(chart) != {"id", "title", "unit", "values"}:
                errors.append(f"charts[{index}] has the wrong keys")
                continue
            values = chart["values"]
            if not isinstance(values, list) or not values:
                errors.append(f"charts[{index}].values must be non-empty")
                continue
            for value_index, value in enumerate(values):
                if (
                    not isinstance(value, dict)
                    or set(value) != {"label", "value"}
                    or not isinstance(value["label"], str)
                    or not value["label"].strip()
                    or not isinstance(value["value"], (int, float))
                    or isinstance(value["value"], bool)
                    or not math.isfinite(float(value["value"]))
                ):
                    errors.append(f"charts[{index}].values[{value_index}] is invalid")

    artifacts = payload["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("artifacts must be a non-empty list")
    else:
        paths: list[str] = []
        for index, artifact in enumerate(artifacts):
            if (
                not isinstance(artifact, dict)
                or set(artifact) != {"label", "path"}
                or not isinstance(artifact["label"], str)
                or not artifact["label"].strip()
                or not _is_safe_relative(artifact["path"])
            ):
                errors.append(f"artifacts[{index}] is invalid")
                continue
            paths.append(artifact["path"])
        missing_artifacts = sorted(set(REQUIRED_MODEL_ARTIFACTS) - set(paths))
        if missing_artifacts:
            errors.append("artifacts is missing: " + ", ".join(missing_artifacts))
    evidence = payload["evidence"]
    expected_evidence = [
        f"benchmarks/results/{task['task_id']}_{suffix}" for suffix in EVIDENCE_SUFFIXES
    ]
    if not isinstance(evidence, list):
        errors.append("evidence must be a list")
    else:
        evidence_paths: list[str] = []
        for index, item in enumerate(evidence):
            if (
                not isinstance(item, dict)
                or set(item) != {"label", "path"}
                or not isinstance(item["label"], str)
                or not item["label"].strip()
                or not _is_safe_relative(item["path"])
            ):
                errors.append(f"evidence[{index}] is invalid")
                continue
            evidence_paths.append(item["path"])
        if evidence_paths != expected_evidence:
            errors.append(
                "evidence paths must be the canonical RESULT/AUDIT markdown and JSON files"
            )
    if not _strings(payload["viewer_command"]):
        errors.append("viewer_command must be a non-empty argv list")
    if not isinstance(payload["notes"], list) or not all(
        isinstance(note, str) and note.strip() for note in payload["notes"]
    ):
        errors.append("notes must be a list of non-empty strings")
    return errors


def _locked_task(run: Path, *, root: Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    errors: list[str] = []
    lock_path = run / "task.lock.json"
    try:
        lock = _load_json(lock_path)
    except ValueError as error:
        return {}, {}, [str(error)]
    task_id = lock.get("task_id")
    task_path_value = lock.get("task_path")
    if not isinstance(task_id, str) or not _is_safe_relative(task_path_value):
        return {}, lock, ["task.lock.json has invalid task_id/task_path"]
    if run.name != task_id:
        errors.append("run directory name must exactly equal the locked task_id")
    task_path = root / task_path_value
    try:
        task = _load_json(task_path)
    except ValueError as error:
        return {}, lock, errors + [str(error)]
    if task.get("task_id") != task_id:
        errors.append("locked task_id does not match task source")
    if lock.get("task_sha256") != _sha256_file(task_path):
        errors.append("task source changed after the run was initialized")
    seal_path_value = lock.get("data_seal_path")
    if not _is_safe_relative(seal_path_value):
        errors.append("task.lock.json data_seal_path is invalid")
    else:
        seal_path = root / seal_path_value
        if not seal_path.is_file() or lock.get("data_seal_sha256") != _sha256_file(seal_path):
            errors.append("data seal changed or disappeared after run initialization")
    if lock.get("report_template_version") != REPORT_TEMPLATE_VERSION:
        errors.append("task lock uses an unsupported report template")
    return task, lock, errors


def validate_run(run: Path, *, root: Path = ROOT, require_index: bool = True) -> list[str]:
    run = run.resolve()
    if not run.is_dir():
        return [f"{run} is not a directory"]
    try:
        relative_run = run.relative_to((root / "runs").resolve())
    except ValueError:
        return [f"{run} is outside the canonical runs/ directory"]
    if len(relative_run.parts) != 1:
        return ["official experiment run must be one exact top-level runs/<task_id> directory"]
    task, _lock, errors = _locked_task(run, root=root)
    if not task:
        return errors
    metrics_path = run / "metrics.json"
    try:
        metrics = _load_json(metrics_path)
    except ValueError as error:
        return errors + [str(error)]
    errors.extend(_metric_errors(metrics, task))
    for artifact in metrics.get("artifacts", []):
        if isinstance(artifact, dict) and _is_safe_relative(artifact.get("path")):
            target = run / artifact["path"]
            if not target.is_file() or target.stat().st_size == 0:
                errors.append(f"missing or empty artifact: {artifact['path']}")
    for evidence in metrics.get("evidence", []):
        if isinstance(evidence, dict) and _is_safe_relative(evidence.get("path")):
            target = root / evidence["path"]
            if not target.is_file() or target.stat().st_size == 0:
                errors.append(f"missing or empty evidence: {evidence['path']}")
    if require_index:
        index = run / "index.html"
        if not index.is_file():
            errors.append("missing canonical index.html; run the render command")
        elif f'content="{REPORT_TEMPLATE_VERSION}"' not in index.read_text(
            encoding="utf-8", errors="replace"
        ):
            errors.append("index.html was not generated by the current canonical template")
    return errors


def _format_number(value: int | float) -> str:
    absolute = abs(float(value))
    if absolute >= 1_000_000_000:
        return f"{float(value) / 1_000_000_000:.3f}B"
    if absolute >= 1_000_000:
        return f"{float(value) / 1_000_000:.3f}M"
    if absolute >= 1_000:
        return f"{float(value):,.2f}"
    if absolute and absolute < 0.001:
        return f"{float(value):.3e}"
    return f"{float(value):.6g}"


def _render_chart(chart: dict[str, Any]) -> str:
    values = chart["values"]
    numeric = [float(item["value"]) for item in values]
    lower = min(0.0, min(numeric))
    upper = max(0.0, max(numeric))
    span = upper - lower or 1.0
    zero = 100.0 * (0.0 - lower) / span
    rows = []
    for item in values:
        endpoint = 100.0 * (float(item["value"]) - lower) / span
        left = min(zero, endpoint)
        width = abs(endpoint - zero)
        rows.append(
            "<div class='bar-row'>"
            f"<div class='bar-label'>{html.escape(item['label'])}</div>"
            "<div class='bar-track'>"
            f"<div class='bar-zero' style='left:{zero:.4f}%'></div>"
            f"<div class='bar-fill' style='left:{left:.4f}%;width:{width:.4f}%'></div>"
            "</div>"
            f"<div class='bar-value'>{html.escape(_format_number(item['value']))}</div>"
            "</div>"
        )
    return (
        "<section class='panel chart'>"
        f"<h3>{html.escape(chart['title'])}</h3>"
        f"<p class='unit'>{html.escape(chart['unit'])}</p>" + "".join(rows) + "</section>"
    )


def render_run(run: Path, *, root: Path = ROOT) -> Path:
    """Write the canonical report page after validating all source data."""

    run = run.resolve()
    errors = validate_run(run, root=root, require_index=False)
    if errors:
        raise ValueError("cannot render invalid run:\n- " + "\n- ".join(errors))
    task, lock, lock_errors = _locked_task(run, root=root)
    if lock_errors:
        raise ValueError("\n".join(lock_errors))
    metrics = _load_json(run / "metrics.json")

    grouped: dict[str, list[tuple[str, int | float, dict[str, str]]]] = defaultdict(list)
    for metric_id, value in metrics["metrics"].items():
        metadata = metrics["metric_metadata"][metric_id]
        grouped[metadata["group"]].append((metric_id, value, metadata))
    metric_sections = []
    for group, rows in grouped.items():
        body = "".join(
            "<tr>"
            f"<td>{html.escape(metadata['label'])}</td>"
            f"<td>{html.escape(_format_number(value))}</td>"
            f"<td>{html.escape(metadata['unit'])}</td>"
            f"<td>{html.escape(metadata['direction'])}</td>"
            "</tr>"
            for _metric_id, value, metadata in rows
        )
        metric_sections.append(
            "<section class='panel'><h3>"
            + html.escape(group)
            + "</h3><div class='table-wrap'><table><thead><tr>"
            "<th>Metric</th><th>Value</th><th>Unit</th><th>Better</th>"
            "</tr></thead><tbody>" + body + "</tbody></table></div></section>"
        )

    stages = "".join(
        "<li><strong>"
        + html.escape(stage["label"])
        + "</strong><span>"
        + html.escape(stage["purpose"])
        + "</span></li>"
        for stage in task["stages"]
    )
    charts = "".join(_render_chart(chart) for chart in metrics["charts"])
    artifacts = "".join(
        f"<li><a href='{html.escape(artifact['path'], quote=True)}'>"
        f"{html.escape(artifact['label'])}</a></li>"
        for artifact in metrics["artifacts"]
    )
    evidence = "".join(
        f"<li><a href='{html.escape(os.path.relpath(root / item['path'], run), quote=True)}'>"
        f"{html.escape(item['label'])}</a></li>"
        for item in metrics["evidence"]
    )
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in metrics["notes"])
    allowed = ", ".join(task["input_policy"]["reconstruction_allowed"])
    forbidden = ", ".join(task["input_policy"]["reconstruction_forbidden"])
    evaluation = ", ".join(task["input_policy"]["evaluation_allowed"])
    task_link = os.path.relpath(root / lock["task_path"], run)
    seal_link = os.path.relpath(root / lock["data_seal_path"], run)
    viewer = shlex.join(metrics["viewer_command"])
    seeds = ", ".join(str(seed) for seed in task["seeds"])
    datasets = ", ".join(f"{dataset['id']} ({dataset['role']})" for dataset in task["datasets"])

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="rtgs-experiment-report-template" content="{REPORT_TEMPLATE_VERSION}">
<title>{html.escape(task["title"])}</title>
<style>
:root {{ color-scheme: light; --ink:#172033; --muted:#5c667a; --line:#d8deea;
--paper:#f5f7fb; --panel:#fff; --accent:#3659d9; --accent2:#7b93ec; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--paper); color:var(--ink);
font:15px/1.5 system-ui,-apple-system,sans-serif; }}
main {{ max-width:1120px; margin:auto; padding:36px 22px 70px; }}
h1 {{ font-size:clamp(28px,4vw,46px); line-height:1.08; margin:.2rem 0 .7rem; }}
h2 {{ margin:2.2rem 0 .8rem; }} h3 {{ margin:.1rem 0 .7rem; }}
.eyebrow,.unit {{ color:var(--muted); text-transform:uppercase; letter-spacing:.08em;
font-size:12px; font-weight:700; }} .summary {{ font-size:19px; max-width:850px; }}
.chips {{ display:flex; gap:8px; flex-wrap:wrap; margin:18px 0; }}
.chip {{ background:#e8edff; color:#2543b4; border-radius:999px; padding:5px 10px;
font-weight:650; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(270px,1fr));
gap:14px; }} .panel {{ background:var(--panel); border:1px solid var(--line);
border-radius:14px; padding:18px; box-shadow:0 2px 8px #1520400a; }}
.boundary strong {{ display:block; margin-bottom:5px; }} .boundary p {{ margin:.2rem 0 1rem; }}
.pipeline {{ display:flex; list-style:none; gap:10px; padding:0; overflow-x:auto; }}
.pipeline li {{ position:relative; min-width:170px; flex:1; background:var(--panel);
border:1px solid var(--line); border-top:4px solid var(--accent); border-radius:10px;
padding:13px; }} .pipeline li:not(:last-child)::after {{ content:"→"; position:absolute;
right:-10px; top:40%; color:var(--accent); font-weight:800; z-index:2; }}
.pipeline span {{ display:block; color:var(--muted); font-size:13px; margin-top:5px; }}
.table-wrap {{ overflow:auto; }} table {{ width:100%; border-collapse:collapse; }}
th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); }}
th {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.05em; }}
.chart {{ min-width:0; }} .bar-row {{ display:grid; grid-template-columns:minmax(110px,1fr)
minmax(110px,2fr) 80px; gap:9px; align-items:center; margin:9px 0; }}
.bar-label {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.bar-track {{ position:relative; height:14px; background:#edf0f6; border-radius:8px;
overflow:hidden; }} .bar-zero {{ position:absolute; top:0; bottom:0; width:1px;
background:#596277; z-index:2; }}
.bar-fill {{ position:absolute; top:0; height:100%;
background:linear-gradient(90deg,var(--accent),var(--accent2)); }}
.bar-value {{ text-align:right; font-variant-numeric:tabular-nums; }}
code,pre {{ background:#eef1f7; border-radius:7px; }} code {{ padding:2px 5px; }}
pre {{ padding:13px; overflow:auto; }} a {{ color:#254fc4; }}
.claim {{ border-left:5px solid #e0902f; }} footer {{ color:var(--muted); margin-top:28px; }}
@media (max-width:620px) {{ .bar-row {{ grid-template-columns:1fr 65px; }}
.bar-track {{ grid-column:1 / -1; grid-row:2; }} }}
</style>
</head>
<body><main>
<p class="eyebrow">realtime-gs · canonical experiment report v{REPORT_TEMPLATE_VERSION}</p>
<h1>{html.escape(task["title"])}</h1>
<div class="chips">
<span class="chip">{html.escape(task["arm"])}</span>
<span class="chip">{html.escape(task["evidence_phase"])}</span>
<span class="chip">decision: {html.escape(metrics["decision"])}</span>
</div>
<p class="summary">{html.escape(metrics["summary"])}</p>

<section class="panel claim"><h2>Claim boundary</h2>
<p>{html.escape(metrics["claim_boundary"])}</p></section>

<h2>Input boundary</h2>
<div class="grid boundary">
<section class="panel"><strong>Reconstruction may read</strong>
<p>{html.escape(allowed)}</p></section>
<section class="panel"><strong>Reconstruction must reject</strong>
<p>{html.escape(forbidden)}</p></section>
<section class="panel"><strong>Evaluation may read</strong>
<p>{html.escape(evaluation)}</p></section>
</div>

<h2>Pipeline</h2><ol class="pipeline">{stages}</ol>

<h2>Metrics</h2><div class="grid">{"".join(metric_sections)}</div>

<h2>Diagrams</h2><div class="grid">{charts}</div>

<h2>Protocol and provenance</h2>
<section class="panel">
<p><strong>Task:</strong>
<a href="{html.escape(task_link, quote=True)}">{html.escape(task["task_id"])}</a></p>
<p><strong>Data seal:</strong>
<a href="{html.escape(seal_link, quote=True)}">
{html.escape(lock["data_seal_path"])}</a></p>
<p><strong>Source commit:</strong> <code>{html.escape(lock["source_commit"])}</code>
 · dirty: <code>{str(bool(lock["source_dirty"])).lower()}</code></p>
<p><strong>Datasets:</strong> {html.escape(datasets)}</p>
<p><strong>Seeds:</strong> {html.escape(seeds)}</p>
<p><a href="metrics.json">metrics.json</a> · <a href="task.lock.json">task.lock.json</a></p>
</section>

<h2>Artifacts</h2><section class="panel"><ul>{artifacts}</ul></section>

<h2>Result and audit</h2><section class="panel"><ul>{evidence}</ul></section>

<h2>Viewer</h2><pre>{html.escape(viewer)}</pre>

<h2>Notes</h2><section class="panel"><ul>{notes}</ul></section>
<footer>This page was generated from the frozen task and metrics.json. Do not hand-edit it.</footer>
</main></body></html>
"""
    path = run / "index.html"
    temporary = run / ".index.html.tmp"
    temporary.write_text(page, encoding="utf-8")
    temporary.replace(path)
    return path


def _print_errors(label: str, errors: list[str]) -> int:
    if not errors:
        print(f"{label}: OK")
        return 0
    print(f"{label}: {len(errors)} problem(s):", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="validate all registered tasks and programs")

    validate_data = subparsers.add_parser("validate-data", help="rehash and verify a task's data")
    validate_data.add_argument("task", type=Path)

    seal_data = subparsers.add_parser("seal-data", help="write the selected data-byte seal")
    seal_data.add_argument("task", type=Path)
    seal_data.add_argument("--out", type=Path, default=None)

    init = subparsers.add_parser("init-run", help="create a task-locked, non-overwriting run")
    init.add_argument("task", type=Path)
    init.add_argument(
        "--development",
        action="store_true",
        help="allow a dirty tracked worktree and mark the run non-official",
    )

    render = subparsers.add_parser("render", help="render canonical index.html from metrics.json")
    render.add_argument("run", type=Path)

    check = subparsers.add_parser("check-run", help="validate task lock, metrics, page, and links")
    check.add_argument("run", type=Path)
    args = parser.parse_args(argv)

    if args.command == "validate":
        return _print_errors("experiment_contract", validate_repository())
    if args.command == "validate-data":
        task = _load_json(args.task)
        return _print_errors("experiment_data", verify_data_seal(task))
    if args.command == "seal-data":
        task = _load_json(args.task)
        out = args.out or (ROOT / task["data_seal"])
        _write_json(out, build_data_seal(task))
        print(f"data seal: {out}")
        return 0
    if args.command == "init-run":
        print(f"run initialized: {init_run(args.task, development=args.development)}")
        return 0
    if args.command == "render":
        print(f"results page: {render_run(args.run)}")
        return 0
    if args.command == "check-run":
        return _print_errors("experiment_run", validate_run(args.run))
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
