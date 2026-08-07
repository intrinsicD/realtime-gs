#!/usr/bin/env python3
"""Validate task-first experiments and render their canonical results page.

New result-bearing experiments use one immutable task id:

    YYYYMMDD_<task_slug>_<data_slug>

The task is registered under ``experiments/tasks/`` before a run starts. A distinct prospective
reviewer approves the exact protocol digest without consuming outcomes. ``init-run`` then binds a
run directory to that review, the task, data seal, command, and source state. ``render`` consumes
the common metrics/history/config/environment/receipt schemas and writes the generated v2
``index.html``, ``README.md``, and checksummed ``manifest.json``. Frozen v1 tasks retain their
historical single-page renderer.

Typical use:

    python scripts/experiment_contract.py validate
    python scripts/experiment_contract.py validate-data experiments/tasks/<task_id>.json
    python scripts/experiment_contract.py review-digest experiments/tasks/<task_id>.json
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
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TASK_SCHEMA_VERSION = 2
TASK_LOCK_SCHEMA_VERSION = 2
PROGRAM_SCHEMA_VERSION = 1
DATA_SEAL_SCHEMA_VERSION = 1
REPORT_TEMPLATE_VERSION = 2
SUPPORTED_REPORT_TEMPLATE_VERSIONS = (1, 2)
LEGACY_V1_TASK_IDS = frozenset(
    {
        "20260728_beam_fusion_claim_stage_frames00008_00009",
        "20260728_rgb_3dgs_comparison_stage_frames00008_00009",
        "20260728_vram_claim_stage_frames00008_00009",
        "20260729_field_sweep_placement_stage_frames00008_00009",
        "20260730_field_sweep_placement_f64_stage_frames00008_00009",
    }
)

ARMS = ("direct_compact", "beam_fusion", "rgb_3dgs")
TASK_STATUSES = ("draft", "ready", "blocked")
PROTOCOL_REVIEW_VERDICTS = ("pending", "approved", "rejected")
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
REQUIRED_V2_ARTIFACTS = REQUIRED_MODEL_ARTIFACTS + (
    "run_receipt.json",
    "environment.json",
)
REQUIRED_V2_FAILURE_ARTIFACTS = (
    "training_history.json",
    "gaussians.config.json",
    "input_boundary_receipt.json",
    "resource_receipt.json",
    "run_receipt.json",
    "environment.json",
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
REVIEW_FIELD_RE = re.compile(
    r"^- (Task ID|Protocol SHA-256|Reviewer|Verdict|Outcome Access): "
    r"`([^`\n]+)`[^\S\n]*$",
    re.MULTILINE,
)
REVIEW_SECTIONS = (
    "## Scope",
    "## Checks",
    "## Findings",
    "## Protected Actions Not Taken",
)
TASK_LOCK_KEYS = frozenset(
    {
        "schema_version",
        "task_id",
        "task_path",
        "task_sha256",
        "protocol_sha256",
        "protocol_review",
        "protocol_review_artifact_sha256",
        "data_seal_path",
        "data_seal_sha256",
        "source_commit",
        "source_dirty",
        "source_diff_sha256",
        "development",
        "started_at_utc",
        "command",
        "report_template_version",
    }
)


class DuplicateKeyError(ValueError):
    """Raised when JSON repeats a key and would otherwise silently overwrite it."""


class NonFiniteJsonError(ValueError):
    """Raised when JSON contains NaN or infinity."""


class LinkCollector(HTMLParser):
    """Collect concrete href/src targets from a generated report."""

    def __init__(self) -> None:
        super().__init__()
        self.links: set[str] = set()

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.links.add(value)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise NonFiniteJsonError(f"non-finite JSON value: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, json.JSONDecodeError, DuplicateKeyError, NonFiniteJsonError) as error:
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


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return _sha256_bytes(encoded)


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


def _task_report_version(task: dict[str, Any]) -> int:
    """Return the frozen report version, defaulting only named historical tasks to v1."""

    task_id = task.get("task_id")
    legacy = isinstance(task_id, str) and task_id in LEGACY_V1_TASK_IDS
    default = 1 if legacy else -1
    value = task.get("report_template_version", default)
    return value if isinstance(value, int) and not isinstance(value, bool) else -1


def protocol_sha256(task: dict[str, Any]) -> str:
    """Hash the protocol while excluding review metadata and administrative status."""

    protocol = {
        key: value for key, value in task.items() if key not in {"protocol_review", "status"}
    }
    encoded = json.dumps(
        protocol,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _review_artifact_errors(
    body: str,
    *,
    task_id: str,
    reviewer: str,
    verdict: str,
    digest: str,
) -> list[str]:
    """Validate the machine-readable header and required review narrative."""

    errors: list[str] = []
    if not body.startswith("# Prospective Protocol Review\n"):
        errors.append("protocol review artifact requires '# Prospective Protocol Review'")

    pairs = REVIEW_FIELD_RE.findall(body)
    fields: dict[str, str] = {}
    duplicates: set[str] = set()
    for name, value in pairs:
        if name in fields:
            duplicates.add(name)
        fields[name] = value
    for name in sorted(duplicates):
        errors.append(f"protocol review artifact repeats {name}")

    expected = {
        "Task ID": task_id,
        "Protocol SHA-256": digest,
        "Reviewer": reviewer,
        "Verdict": verdict,
        "Outcome Access": "none",
    }
    if set(fields) != set(expected):
        errors.append(
            "protocol review artifact must contain exactly the five canonical header fields"
        )
    else:
        for name, value in expected.items():
            if fields[name] != value:
                errors.append(f"protocol review artifact {name} must equal {value!r}")

    section_matches: list[tuple[str, re.Match[str]]] = []
    for heading in REVIEW_SECTIONS:
        match = re.search(rf"^{re.escape(heading)}[^\S\n]*$", body, re.MULTILINE)
        if match is None:
            errors.append(f"protocol review artifact is missing {heading}")
        else:
            section_matches.append((heading, match))
    section_matches.sort(key=lambda item: item[1].start())
    for index, (heading, match) in enumerate(section_matches):
        end = (
            section_matches[index + 1][1].start() if index + 1 < len(section_matches) else len(body)
        )
        if not body[match.end() : end].strip():
            errors.append(f"protocol review artifact has no content below {heading}")
    return errors


def _validate_protocol_review(
    task: dict[str, Any],
    *,
    root: Path,
) -> list[str]:
    """Validate distinct prospective review and its exact protocol binding."""

    errors: list[str] = []
    review = task.get("protocol_review")
    keys = {"reviewer", "verdict", "protocol_sha256", "artifact"}
    if not isinstance(review, dict) or set(review) != keys:
        return [f"protocol_review must contain exactly: {', '.join(sorted(keys))}"]

    verdict = review["verdict"]
    if verdict not in PROTOCOL_REVIEW_VERDICTS:
        errors.append(
            "protocol_review.verdict must be one of: " + ", ".join(PROTOCOL_REVIEW_VERDICTS)
        )
        return errors

    if verdict == "pending":
        for key in ("reviewer", "protocol_sha256", "artifact"):
            if review[key] is not None:
                errors.append(f"pending protocol_review requires {key} to be null")
        if task.get("status") == "ready":
            errors.append("ready tasks require an approved prospective protocol review")
        return errors

    reviewer = review["reviewer"]
    digest = review["protocol_sha256"]
    artifact = review["artifact"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        errors.append("completed protocol_review requires a reviewer")
    owner = task.get("owner")
    if not isinstance(owner, str) or not owner.strip():
        errors.append("completed protocol_review requires a frozen task owner")
    if (
        isinstance(reviewer, str)
        and reviewer.strip()
        and isinstance(owner, str)
        and reviewer.strip().casefold() == owner.strip().casefold()
    ):
        errors.append("prospective protocol reviewer must differ from the task owner")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        errors.append("protocol_review.protocol_sha256 must be a lowercase SHA-256")
    elif digest != protocol_sha256(task):
        errors.append("protocol_review digest does not match the current protocol")

    artifact_path = Path(artifact) if isinstance(artifact, str) else None
    safe_artifact = (
        artifact_path is not None
        and _is_safe_relative(artifact)
        and artifact_path.parts
        == (
            "experiments",
            "reviews",
            f"{task.get('task_id')}_PROTOCOL_REVIEW.md",
        )
    )
    if not safe_artifact:
        errors.append(
            "protocol_review.artifact must be experiments/reviews/<task_id>_PROTOCOL_REVIEW.md"
        )
    else:
        target = root / artifact_path
        if not target.is_file():
            errors.append(f"protocol review artifact does not exist: {artifact}")
        else:
            body = target.read_text(encoding="utf-8")
            if all(
                isinstance(value, str) for value in (task.get("task_id"), reviewer, verdict, digest)
            ):
                errors.extend(
                    _review_artifact_errors(
                        body,
                        task_id=task["task_id"],
                        reviewer=reviewer,
                        verdict=verdict,
                        digest=digest,
                    )
                )

    if verdict == "approved" and task.get("status") != "ready":
        errors.append("approved protocol_review requires task status 'ready'")
    if verdict == "rejected" and task.get("status") != "blocked":
        errors.append("rejected protocol_review requires task status 'blocked'")
    if task.get("status") == "ready" and verdict != "approved":
        errors.append("ready tasks require protocol_review verdict 'approved'")
    return errors


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
        "protocol_review",
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
    report_version = _task_report_version(task)
    if report_version not in SUPPORTED_REPORT_TEMPLATE_VERSIONS:
        errors.append(
            f"new tasks must explicitly set report_template_version to {REPORT_TEMPLATE_VERSION}"
        )
    elif report_version == 1 and not (
        isinstance(task.get("task_id"), str) and task["task_id"] in LEGACY_V1_TASK_IDS
    ):
        errors.append("report_template_version 1 is restricted to grandfathered task ids")
    task_id = task["task_id"]
    if not isinstance(task_id, str) or task_id != _task_id(task):
        errors.append("task_id must equal <date>_<task_slug>_<data_slug>")
    if isinstance(task_id, str) and path.name != f"{task_id}.json":
        errors.append(f"task filename must be {task_id}.json")
    try:
        relative_task_path = path.resolve().relative_to(root.resolve())
    except ValueError:
        errors.append("task file must live inside the repository")
    else:
        if len(relative_task_path.parts) != 3 or relative_task_path.parts[:2] != (
            "experiments",
            "tasks",
        ):
            errors.append("task file must live directly under experiments/tasks/")
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
    errors.extend(_validate_protocol_review(task, root=root))
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
            production_manifest = dataset.get("production_manifest")
            if production_manifest is not None:
                if not _is_safe_relative(production_manifest):
                    errors.append(f"{label}.production_manifest must be a repository-relative path")
                elif Path(production_manifest).parent != Path(dataset["compact_manifest"]).parent:
                    errors.append(f"{label}.production_manifest must sit beside compact_manifest")
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
                if (
                    expected_profile == "rgb"
                    and "gaussians2d" in policy["reconstruction_allowed"]
                    and not any("/gaussians2d" in value for value in sealed_paths)
                ):
                    errors.append("hybrid RGB/Gaussian2D data_seal must bind compact view files")

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
    errors.extend(verify_source_binding(task, root=root))
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
    if program["report_template_version"] not in SUPPORTED_REPORT_TEMPLATE_VERSIONS:
        errors.append(
            "report_template_version must be one of: "
            + ", ".join(str(item) for item in SUPPORTED_REPORT_TEMPLATE_VERSIONS)
        )
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
            elif _task_report_version(task) != program["report_template_version"]:
                errors.append(
                    f"{task_id} report template does not match the program report template"
                )
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
    bind_compact_inputs = compact_profile or (
        "gaussians2d" in task["input_policy"]["reconstruction_allowed"]
    )
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
        production_manifest = dataset.get("production_manifest")
        if production_manifest is not None:
            files.append(root / production_manifest)
        for view in views:
            view_id = view["view_id"]
            if bind_compact_inputs:
                files.append(manifest_path.parent / view["path"])
            if not compact_profile:
                files.extend(
                    [
                        frame / f"rgb/{view_id}.jpg",
                        frame / f"mask/mask_{view_id}.png",
                    ]
                )
        dataset_record = {
            "id": dataset["id"],
            "role": dataset["role"],
            "frame_path": dataset["frame_path"],
            "view_ids": view_ids,
            "selected_modalities": (
                ["calibration", "gaussians2d"]
                if compact_profile
                else (
                    ["calibration", "gaussians2d", "rgb", "mask"]
                    if bind_compact_inputs
                    else ["calibration", "rgb", "mask"]
                )
            ),
            "canonical_rgb_pattern": (None if compact_profile else dataset["rgb_pattern"]),
            "canonical_mask_pattern": (None if compact_profile else dataset["mask_pattern"]),
        }
        if production_manifest is not None:
            dataset_record["production_manifest"] = production_manifest
        datasets_payload.append(dataset_record)

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


def build_source_binding(binding: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    patterns = binding.get("patterns")
    if not _strings(patterns):
        raise ValueError("frozen_configuration.source_binding.patterns is invalid")
    paths: dict[str, Path] = {}
    for pattern in patterns:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError(f"source binding pattern is unsafe: {pattern}")
        for path in root.glob(pattern):
            if path.is_file():
                paths[path.relative_to(root).as_posix()] = path
    records = [
        {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for relative, path in sorted(paths.items())
    ]
    encoded = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return {
        "patterns": list(patterns),
        "file_count": len(records),
        "aggregate_sha256": _sha256_bytes(encoded),
    }


def verify_source_binding(task: dict[str, Any], *, root: Path = ROOT) -> list[str]:
    frozen = task.get("frozen_configuration")
    binding = frozen.get("source_binding") if isinstance(frozen, dict) else None
    if binding is None or (isinstance(binding, dict) and "patterns" not in binding):
        return []
    if not isinstance(binding, dict) or set(binding) != {
        "patterns",
        "file_count",
        "aggregate_sha256",
    }:
        return ["frozen_configuration.source_binding has the wrong keys"]
    try:
        current = build_source_binding(binding, root=root)
    except ValueError as error:
        return [str(error)]
    if binding != current:
        return ["behavior-bearing source differs from the frozen prospective source binding"]
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


def _development_source_state(root: Path) -> bytes:
    """Return a stable dirty-source record including untracked file contents.

    ``git diff HEAD`` deliberately omits untracked files. Development experiment locks must bind
    them as well because task drivers and opt-in research modules commonly begin untracked. The
    appended manifest records each untracked path, mode, byte count, and content digest.
    """

    tracked_diff = _git_output(root, "diff", "--binary", "HEAD").encode("utf-8")
    untracked = sorted(
        item
        for item in _git_output(
            root,
            "ls-files",
            "--others",
            "--exclude-standard",
        ).splitlines()
        if item
    )
    records = []
    for relative in untracked:
        path = (root / relative).resolve(strict=True)
        try:
            normalized = path.relative_to(root.resolve()).as_posix()
        except ValueError as error:
            raise ValueError(f"untracked source escapes repository: {relative}") from error
        if not path.is_file():
            continue
        records.append(
            {
                "path": normalized,
                "mode": path.stat().st_mode & 0o777,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    manifest = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return tracked_diff + b"\nRTGS-UNTRACKED-SOURCE-MANIFEST\0" + manifest


def init_run(task_path: Path, *, root: Path = ROOT, development: bool = False) -> Path:
    """Create a run root and lock the review, task, data, command, and source."""

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
    review = task["protocol_review"]
    review_path = root / review["artifact"]
    diff = _development_source_state(root)
    lock = {
        "schema_version": TASK_LOCK_SCHEMA_VERSION,
        "task_id": task["task_id"],
        "task_path": task_relative,
        "task_sha256": _sha256_file(task_path),
        "protocol_sha256": protocol_sha256(task),
        "protocol_review": review,
        "protocol_review_artifact_sha256": _sha256_file(review_path),
        "data_seal_path": task["data_seal"],
        "data_seal_sha256": _sha256_file(seal_path),
        "source_commit": _git_output(root, "rev-parse", "HEAD").strip(),
        "source_dirty": dirty,
        "source_diff_sha256": _sha256_bytes(diff),
        "development": development,
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": task["run_command"],
        "report_template_version": _task_report_version(task),
    }
    run.mkdir(parents=True)
    _write_json(run / "task.lock.json", lock)
    return run


def _metric_errors_v1(payload: dict[str, Any], task: dict[str, Any]) -> list[str]:
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
    if payload["report_template_version"] != 1:
        errors.append("report_template_version must be 1")
    if payload["task_id"] != task["task_id"]:
        errors.append("metrics.json task_id does not match the locked task")
    for key in ("summary", "decision", "claim_boundary"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            errors.append(f"metrics.json {key} must be non-empty")
    if payload["claim_boundary"] != task["claim_boundary"]:
        errors.append("metrics.json claim_boundary must exactly match the frozen task")

    metrics = payload["metrics"]
    metadata = payload["metric_metadata"]
    metric_ids = set(metrics) if isinstance(metrics, dict) else set()
    if not isinstance(metrics, dict) or not metrics:
        errors.append("metrics must be a non-empty object")
    elif not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in metrics.values()
    ):
        errors.append("every metric must be a finite number")
    if not isinstance(metadata, dict) or set(metadata) != metric_ids:
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
    missing_task_metrics = sorted(set(task_metrics) - metric_ids)
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


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _utc_datetime(value: object) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.utcoffset() != dt.timedelta(0):
        return None
    return parsed


def _run_receipt_errors(
    receipt: dict[str, Any], task: dict[str, Any], lock: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "task_id",
        "status",
        "started_at_utc",
        "finished_at_utc",
        "exit_code",
        "failure_phase",
        "message",
    }
    if set(receipt) != required:
        errors.append("run_receipt.json has the wrong keys")
        return errors
    if receipt["schema_version"] != 1:
        errors.append("run_receipt.json schema_version must be 1")
    if receipt["task_id"] != task["task_id"]:
        errors.append("run_receipt.json task_id does not match the locked task")
    if not isinstance(receipt["status"], str) or receipt["status"] not in {
        "completed",
        "failed",
    }:
        errors.append("run_receipt.json status must be completed or failed")
    started = _utc_datetime(receipt["started_at_utc"])
    finished = _utc_datetime(receipt["finished_at_utc"])
    locked_started = _utc_datetime(lock.get("started_at_utc"))
    if started is None or finished is None:
        errors.append("run_receipt.json timestamps must be ISO-8601 UTC strings")
    else:
        if locked_started is not None and started != locked_started:
            errors.append("run_receipt.json started_at_utc must match task.lock.json")
        if finished < started:
            errors.append("run_receipt.json finished_at_utc precedes its start")
    exit_code = receipt["exit_code"]
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        errors.append("run_receipt.json exit_code must be an integer")
    elif receipt["status"] == "completed" and exit_code != 0:
        errors.append("a completed run_receipt.json requires exit_code 0")
    elif receipt["status"] == "failed" and exit_code == 0:
        errors.append("a failed run_receipt.json requires a non-zero exit_code")
    phase = receipt["failure_phase"]
    if receipt["status"] == "completed" and phase is not None:
        errors.append("a completed run_receipt.json requires failure_phase null")
    if receipt["status"] == "failed" and (not isinstance(phase, str) or not phase.strip()):
        errors.append("a failed run_receipt.json requires a failure_phase")
    if not isinstance(receipt["message"], str) or not receipt["message"].strip():
        errors.append("run_receipt.json message must be non-empty")
    return errors


def _environment_errors(environment: dict[str, Any]) -> list[str]:
    required = {"schema_version", "python", "platform", "packages", "device"}
    if set(environment) != required:
        return ["environment.json has the wrong keys"]
    errors: list[str] = []
    if environment["schema_version"] != 1:
        errors.append("environment.json schema_version must be 1")
    for key in ("python", "platform"):
        if not isinstance(environment[key], str) or not environment[key].strip():
            errors.append(f"environment.json {key} must be non-empty")
    packages = environment["packages"]
    if (
        not isinstance(packages, dict)
        or not packages
        or not all(
            isinstance(key, str)
            and bool(key.strip())
            and isinstance(value, str)
            and bool(value.strip())
            for key, value in packages.items()
        )
    ):
        errors.append("environment.json packages must map package names to versions")
    device = environment["device"]
    if not isinstance(device, dict) or set(device) != {"type", "name", "cuda"}:
        errors.append("environment.json device has the wrong keys")
    else:
        for key in ("type", "name"):
            if not isinstance(device[key], str) or not device[key].strip():
                errors.append(f"environment.json device.{key} must be non-empty")
        if device["cuda"] is not None and (
            not isinstance(device["cuda"], str) or not device["cuda"].strip()
        ):
            errors.append("environment.json device.cuda must be null or a version string")
    return errors


def _history_errors(history: dict[str, Any], task: dict[str, Any], *, completed: bool) -> list[str]:
    required = {"schema_version", "records", "metric_metadata", "stage_markers"}
    if set(history) != required:
        return ["training_history.json has the wrong keys"]
    errors: list[str] = []
    if history["schema_version"] != 2:
        errors.append("training_history.json schema_version must be 2")
    records = history["records"]
    if not isinstance(records, list):
        return errors + ["training_history.json records must be a list"]
    if completed and not records:
        errors.append("a completed v2 run requires fitting-history records")
    dataset_ids = {item["id"] for item in task["datasets"]}
    stage_order = [item["id"] for item in task["stages"]]
    stage_labels = {item["id"]: item["label"] for item in task["stages"]}
    stage_ids = set(stage_order)
    seeds = set(task["seeds"])
    record_keys = {
        "step",
        "wall_seconds",
        "stage",
        "dataset_id",
        "arm_id",
        "seed",
        "split",
        "metric_id",
        "value",
    }
    seen: set[str] = set()
    observed_metrics: set[str] = set()
    observed_series: set[tuple[str, str, int]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != record_keys:
            errors.append(f"training_history.json records[{index}] has the wrong keys")
            continue
        if (
            not isinstance(record["step"], int)
            or isinstance(record["step"], bool)
            or record["step"] < 0
        ):
            errors.append(f"training_history.json records[{index}].step is invalid")
        if not _finite_number(record["wall_seconds"]) or record["wall_seconds"] < 0:
            errors.append(f"training_history.json records[{index}].wall_seconds is invalid")
        if not isinstance(record["stage"], str) or record["stage"] not in stage_ids:
            errors.append(f"training_history.json records[{index}].stage is not frozen")
        if not isinstance(record["dataset_id"], str) or record["dataset_id"] not in dataset_ids:
            errors.append(f"training_history.json records[{index}].dataset_id is not frozen")
        if not isinstance(record["arm_id"], str) or SLUG_RE.fullmatch(record["arm_id"]) is None:
            errors.append(f"training_history.json records[{index}].arm_id is invalid")
        if (
            not isinstance(record["seed"], int)
            or isinstance(record["seed"], bool)
            or record["seed"] not in seeds
        ):
            errors.append(f"training_history.json records[{index}].seed is not frozen")
        if not isinstance(record["split"], str) or record["split"] not in {
            "train",
            "validation",
            "diagnostic",
        }:
            errors.append(
                f"training_history.json records[{index}].split must not expose heldout/test data"
            )
        metric_id = record["metric_id"]
        if not isinstance(metric_id, str) or SLUG_RE.fullmatch(metric_id) is None:
            errors.append(f"training_history.json records[{index}].metric_id is invalid")
        else:
            observed_metrics.add(metric_id)
        if not _finite_number(record["value"]):
            errors.append(f"training_history.json records[{index}].value is invalid")
        if (
            isinstance(record["dataset_id"], str)
            and record["dataset_id"] in dataset_ids
            and isinstance(record["arm_id"], str)
            and SLUG_RE.fullmatch(record["arm_id"]) is not None
            and isinstance(record["seed"], int)
            and not isinstance(record["seed"], bool)
            and record["seed"] in seeds
        ):
            observed_series.add((record["dataset_id"], record["arm_id"], record["seed"]))
        identity = json.dumps(
            [
                record.get("step"),
                record.get("stage"),
                record.get("dataset_id"),
                record.get("arm_id"),
                record.get("seed"),
                record.get("split"),
                record.get("metric_id"),
            ],
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        if identity in seen:
            errors.append(f"training_history.json records[{index}] duplicates a data point")
        seen.add(identity)

    metadata = history["metric_metadata"]
    meta_keys = {"label", "unit", "group", "direction"}
    if not isinstance(metadata, dict) or set(metadata) != observed_metrics:
        errors.append("training_history.json metric_metadata must match recorded metric ids")
    else:
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
                errors.append(f"training_history.json metric_metadata.{metric_id} is invalid")
    markers = history["stage_markers"]
    if not isinstance(markers, list):
        errors.append("training_history.json stage_markers must be a list")
    else:
        if completed and not markers:
            errors.append(
                "a completed v2 run requires start/end boundaries for every fitting stage"
            )
        marker_keys = {
            "step",
            "wall_seconds",
            "stage",
            "dataset_id",
            "arm_id",
            "seed",
            "boundary",
            "label",
        }
        marker_identities: set[tuple[str, str, int, str, str]] = set()
        marker_groups: dict[tuple[str, str, int], list[tuple[int, float, str, str]]] = defaultdict(
            list
        )
        valid_markers: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
        for index, marker in enumerate(markers):
            if not isinstance(marker, dict) or set(marker) != marker_keys:
                errors.append(f"training_history.json stage_markers[{index}] is invalid")
                continue
            step = marker["step"]
            wall_seconds = marker["wall_seconds"]
            stage = marker["stage"]
            dataset_id = marker["dataset_id"]
            arm_id = marker["arm_id"]
            seed = marker["seed"]
            boundary = marker["boundary"]
            label = marker["label"]
            if (
                not isinstance(step, int)
                or isinstance(step, bool)
                or step < 0
                or not _finite_number(wall_seconds)
                or wall_seconds < 0
                or not isinstance(stage, str)
                or stage not in stage_ids
                or not isinstance(dataset_id, str)
                or dataset_id not in dataset_ids
                or not isinstance(arm_id, str)
                or SLUG_RE.fullmatch(arm_id) is None
                or not isinstance(seed, int)
                or isinstance(seed, bool)
                or seed not in seeds
                or not isinstance(boundary, str)
                or boundary not in {"start", "end"}
                or not isinstance(label, str)
                or not label.strip()
                or label != stage_labels[stage]
            ):
                errors.append(f"training_history.json stage_markers[{index}] is invalid")
                continue
            series_key = (dataset_id, arm_id, seed)
            identity = (*series_key, stage, boundary)
            if identity in marker_identities:
                errors.append(f"training_history.json stage_markers[{index}] is duplicated")
            marker_identities.add(identity)
            marker_groups[series_key].append((step, float(wall_seconds), stage, boundary))
            valid_markers[identity] = marker

        if set(marker_groups) - observed_series:
            errors.append(
                "training_history.json stage_markers contain a series with no history records"
            )
        expected_boundaries = [
            (stage, boundary) for stage in stage_order for boundary in ("start", "end")
        ]
        for series_key in sorted(observed_series):
            series_markers = marker_groups.get(series_key, [])
            observed_boundaries = [
                (stage, boundary) for _step, _wall_seconds, stage, boundary in series_markers
            ]
            expected = (
                expected_boundaries
                if completed
                else expected_boundaries[: len(observed_boundaries)]
            )
            if observed_boundaries != expected:
                errors.append(
                    "training_history.json stage_markers must contain ordered start/end "
                    f"boundaries for every frozen stage in series {series_key}"
                )
            marker_steps = [item[0] for item in series_markers]
            marker_times = [item[1] for item in series_markers]
            if marker_steps != sorted(marker_steps):
                errors.append(
                    "training_history.json stage_markers must be ordered by step within "
                    f"series {series_key}"
                )
            if marker_times != sorted(marker_times):
                errors.append(
                    "training_history.json stage_markers must be ordered by wall_seconds "
                    f"within series {series_key}"
                )

        for index, record in enumerate(records):
            if not isinstance(record, dict) or set(record) != record_keys:
                continue
            dataset_id = record["dataset_id"]
            arm_id = record["arm_id"]
            seed = record["seed"]
            stage = record["stage"]
            if not (
                isinstance(record["step"], int)
                and not isinstance(record["step"], bool)
                and _finite_number(record["wall_seconds"])
                and isinstance(dataset_id, str)
                and isinstance(arm_id, str)
                and isinstance(seed, int)
                and not isinstance(seed, bool)
                and isinstance(stage, str)
            ):
                continue
            series_key = (dataset_id, arm_id, seed)
            if series_key not in observed_series or stage not in stage_ids:
                continue
            start = valid_markers.get((*series_key, stage, "start"))
            end = valid_markers.get((*series_key, stage, "end"))
            if start is not None and (
                record["step"] < start["step"] or record["wall_seconds"] < start["wall_seconds"]
            ):
                errors.append(
                    f"training_history.json records[{index}] precedes its stage start boundary"
                )
            if end is not None and (
                record["step"] > end["step"] or record["wall_seconds"] > end["wall_seconds"]
            ):
                errors.append(
                    f"training_history.json records[{index}] follows its stage end boundary"
                )
    return errors


def _v2_commands_errors(commands: object, lock: dict[str, Any], *, completed: bool) -> list[str]:
    if not isinstance(commands, dict) or set(commands) != {
        "reproduce",
        "serve_report",
        "viewer",
    }:
        return ["metrics.json commands must contain reproduce, serve_report, and viewer"]
    errors: list[str] = []
    if commands["reproduce"] != lock.get("command"):
        errors.append("metrics.json reproduce command must exactly match task.lock.json")
    expected_server = [
        ".venv/bin/python",
        "-m",
        "http.server",
        "8765",
        "--directory",
        f"runs/{lock.get('task_id', '')}",
    ]
    if commands["serve_report"] != expected_server:
        errors.append(
            "metrics.json serve_report must use the canonical repository-root HTTP command"
        )
    viewer = commands["viewer"]
    if completed and not _strings(viewer):
        errors.append("a completed v2 run requires an exact viewer argv command")
    elif not completed and viewer is not None and not _strings(viewer):
        errors.append("a failed v2 run viewer command must be null or a non-empty argv list")
    comparison_source = (
        isinstance(viewer, list)
        and "--comparison-manifest" in viewer
        and any(item.endswith("viewer_comparison.json") for item in viewer)
    )
    model_source = isinstance(viewer, list) and any("gaussians.ply" in item for item in viewer)
    if _strings(viewer) and ("view" not in viewer or not (model_source or comparison_source)):
        errors.append(
            "metrics.json viewer must invoke view on gaussians.ply or viewer_comparison.json"
        )
    return errors


def _metric_errors_v2(
    payload: dict[str, Any],
    task: dict[str, Any],
    lock: dict[str, Any],
    *,
    completed: bool,
) -> list[str]:
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
        "commands",
        "notes",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"metrics.json missing: {', '.join(missing)}"]
    errors: list[str] = []
    if payload["schema_version"] != 2:
        errors.append("metrics.json schema_version must be 2")
    if payload["report_template_version"] != 2:
        errors.append("metrics.json report_template_version must be 2")
    errors.extend(_v2_commands_errors(payload["commands"], lock, completed=completed))
    errors.extend(_dataset_summary_errors(payload.get("dataset_summaries"), task, completed))

    if completed:
        legacy = dict(payload)
        legacy["schema_version"] = 1
        legacy["report_template_version"] = 1
        legacy["viewer_command"] = (
            payload["commands"].get("viewer", []) if isinstance(payload["commands"], dict) else []
        )
        errors.extend(_metric_errors_v1(legacy, task))
        paths = (
            {
                item["path"]
                for item in payload["artifacts"]
                if isinstance(item, dict) and _is_safe_relative(item.get("path"))
            }
            if isinstance(payload["artifacts"], list)
            else set()
        )
        missing_artifacts = sorted(set(REQUIRED_V2_ARTIFACTS) - paths)
        if missing_artifacts:
            errors.append("v2 artifacts is missing: " + ", ".join(missing_artifacts))
        return errors

    if payload["task_id"] != task["task_id"]:
        errors.append("metrics.json task_id does not match the locked task")
    for key in ("summary", "decision", "claim_boundary"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            errors.append(f"metrics.json {key} must be non-empty")
    if payload["claim_boundary"] != task["claim_boundary"]:
        errors.append("metrics.json claim_boundary must exactly match the frozen task")
    metrics = payload["metrics"]
    if not isinstance(metrics, dict) or not all(
        _finite_number(value) for value in metrics.values()
    ):
        errors.append("failed-run metrics must be an object of finite numbers")
    metadata = payload["metric_metadata"]
    if (
        not isinstance(metadata, dict)
        or not isinstance(metrics, dict)
        or set(metadata) != set(metrics)
    ):
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
    if payload["charts"] != []:
        errors.append(
            "failed-run charts must be empty; partial progress belongs in fitting history"
        )
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, list):
        errors.append("artifacts must be a list")
    else:
        paths: set[object] = set()
        for index, artifact in enumerate(artifacts):
            if (
                not isinstance(artifact, dict)
                or set(artifact) != {"label", "path"}
                or not isinstance(artifact["label"], str)
                or not artifact["label"].strip()
                or not _is_safe_relative(artifact["path"])
            ):
                errors.append(f"artifacts[{index}] is invalid")
            else:
                paths.add(artifact["path"])
        missing_artifacts = sorted(set(REQUIRED_V2_FAILURE_ARTIFACTS) - paths)
        if missing_artifacts:
            errors.append("failed v2 artifacts is missing: " + ", ".join(missing_artifacts))
    evidence = payload["evidence"]
    if not isinstance(evidence, list):
        errors.append("evidence must be a list")
    else:
        evidence_paths: set[str] = set()
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
            if item["path"] in evidence_paths:
                errors.append(f"evidence[{index}] repeats a path")
            evidence_paths.add(item["path"])
    if not isinstance(payload["notes"], list) or not all(
        isinstance(note, str) and note.strip() for note in payload["notes"]
    ):
        errors.append("notes must be a list of non-empty strings")
    return errors


def _dataset_summary_errors(
    summaries: object,
    task: dict[str, Any],
    completed: bool,
) -> list[str]:
    """Validate optional canonical per-dataset report inputs."""

    if summaries is None:
        return []
    if not isinstance(summaries, dict):
        return ["metrics.json dataset_summaries must be an object"]
    expected = {item["id"] for item in task["datasets"]}
    if completed and set(summaries) != expected:
        return ["dataset_summaries must exactly cover every frozen dataset"]
    errors: list[str] = []
    summary_keys = {
        "title",
        "summary",
        "metrics",
        "metric_metadata",
        "charts",
        "curves",
        "artifacts",
        "commands",
        "notes",
    }
    metadata_keys = {"label", "unit", "group", "direction"}
    for dataset_id, value in summaries.items():
        if dataset_id not in expected or not isinstance(value, dict) or set(value) != summary_keys:
            errors.append(f"dataset_summaries.{dataset_id} has the wrong keys or id")
            continue
        if not all(
            isinstance(value[key], str) and value[key].strip() for key in ("title", "summary")
        ):
            errors.append(f"dataset_summaries.{dataset_id} title/summary is invalid")
        final_metrics = value["metrics"]
        metadata = value["metric_metadata"]
        if (
            not isinstance(final_metrics, dict)
            or not final_metrics
            or not all(_finite_number(item) for item in final_metrics.values())
        ):
            errors.append(f"dataset_summaries.{dataset_id}.metrics is invalid")
        if (
            not isinstance(metadata, dict)
            or not isinstance(final_metrics, dict)
            or set(metadata) != set(final_metrics)
        ):
            errors.append(f"dataset_summaries.{dataset_id}.metric_metadata must match metrics")
        else:
            for metric_id, item in metadata.items():
                if (
                    not isinstance(item, dict)
                    or set(item) != metadata_keys
                    or item.get("direction") not in {"lower", "higher", "descriptive"}
                    or not all(
                        isinstance(item.get(key), str) and item[key].strip()
                        for key in ("label", "unit", "group")
                    )
                ):
                    errors.append(
                        f"dataset_summaries.{dataset_id}.metric_metadata.{metric_id} is invalid"
                    )
        charts = value["charts"]
        if not isinstance(charts, list):
            errors.append(f"dataset_summaries.{dataset_id}.charts must be a list")
        else:
            for index, chart in enumerate(charts):
                if not isinstance(chart, dict) or set(chart) != {"id", "title", "unit", "values"}:
                    errors.append(f"dataset_summaries.{dataset_id}.charts[{index}] is invalid")
                    continue
                values = chart["values"]
                if (
                    not isinstance(values, list)
                    or not values
                    or not all(
                        isinstance(item, dict)
                        and set(item) == {"label", "value"}
                        and isinstance(item["label"], str)
                        and item["label"].strip()
                        and _finite_number(item["value"])
                        for item in values
                    )
                ):
                    errors.append(
                        f"dataset_summaries.{dataset_id}.charts[{index}].values is invalid"
                    )
        curves = value["curves"]
        if not isinstance(curves, list) or not curves:
            errors.append(f"dataset_summaries.{dataset_id}.curves must be non-empty")
        else:
            for index, curve in enumerate(curves):
                if not isinstance(curve, dict) or set(curve) != {
                    "id",
                    "title",
                    "x_label",
                    "unit",
                    "direction",
                    "series",
                }:
                    errors.append(f"dataset_summaries.{dataset_id}.curves[{index}] is invalid")
                    continue
                if curve.get("direction") not in {"lower", "higher", "descriptive"}:
                    errors.append(
                        f"dataset_summaries.{dataset_id}.curves[{index}].direction is invalid"
                    )
                series = curve["series"]
                if not isinstance(series, list) or not series:
                    errors.append(
                        f"dataset_summaries.{dataset_id}.curves[{index}].series is invalid"
                    )
                    continue
                for series_index, item in enumerate(series):
                    if (
                        not isinstance(item, dict)
                        or set(item) != {"label", "points"}
                        or not isinstance(item["label"], str)
                        or not item["label"].strip()
                        or not isinstance(item["points"], list)
                        or not item["points"]
                        or not all(
                            isinstance(point, dict)
                            and set(point) == {"x", "value"}
                            and _finite_number(point["x"])
                            and _finite_number(point["value"])
                            for point in item["points"]
                        )
                    ):
                        errors.append(
                            f"dataset_summaries.{dataset_id}.curves[{index}]."
                            f"series[{series_index}] is invalid"
                        )
        artifacts = value["artifacts"]
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or not all(
                isinstance(item, dict)
                and set(item) == {"label", "path"}
                and isinstance(item["label"], str)
                and item["label"].strip()
                and _is_safe_relative(item["path"])
                for item in artifacts
            )
        ):
            errors.append(f"dataset_summaries.{dataset_id}.artifacts is invalid")
        commands = value["commands"]
        if (
            not isinstance(commands, dict)
            or set(commands) != {"viewer"}
            or not isinstance(commands["viewer"], list)
            or not commands["viewer"]
            or not all(isinstance(item, str) and item for item in commands["viewer"])
        ):
            errors.append(f"dataset_summaries.{dataset_id}.commands is invalid")
        if not isinstance(value["notes"], list) or not all(
            isinstance(item, str) and item.strip() for item in value["notes"]
        ):
            errors.append(f"dataset_summaries.{dataset_id}.notes is invalid")
    return errors


def _v2_source_errors(
    run: Path, task: dict[str, Any], lock: dict[str, Any]
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        receipt = _load_json(run / "run_receipt.json")
    except ValueError as error:
        return False, [str(error)]
    errors.extend(_run_receipt_errors(receipt, task, lock))
    completed = receipt.get("status") == "completed"
    try:
        environment = _load_json(run / "environment.json")
    except ValueError as error:
        errors.append(str(error))
    else:
        errors.extend(_environment_errors(environment))
    try:
        history = _load_json(run / "training_history.json")
    except ValueError as error:
        errors.append(str(error))
    else:
        errors.extend(_history_errors(history, task, completed=completed))
    for name in ("gaussians.config.json", "input_boundary_receipt.json", "resource_receipt.json"):
        try:
            value = _load_json(run / name)
        except ValueError as error:
            errors.append(str(error))
        else:
            if not value:
                errors.append(f"{name} must be a non-empty JSON object")
    return completed, errors


def _locked_task(run: Path, *, root: Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    errors: list[str] = []
    lock_path = run / "task.lock.json"
    try:
        lock = _load_json(lock_path)
    except ValueError as error:
        return {}, {}, [str(error)]
    if set(lock) != TASK_LOCK_KEYS:
        missing = sorted(TASK_LOCK_KEYS - set(lock))
        extra = sorted(set(lock) - TASK_LOCK_KEYS)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unexpected " + ", ".join(extra))
        errors.append("task.lock.json has the wrong keys: " + "; ".join(detail))
    if lock.get("schema_version") != TASK_LOCK_SCHEMA_VERSION:
        errors.append(f"task.lock.json schema_version must be {TASK_LOCK_SCHEMA_VERSION}")
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
    errors.extend(validate_task(task, task_path, root=root))
    if task.get("status") != "ready":
        errors.append("locked task must remain ready")
    try:
        digest = protocol_sha256(task)
    except (TypeError, ValueError) as error:
        errors.append(f"locked task protocol cannot be hashed: {error}")
    else:
        if lock.get("protocol_sha256") != digest:
            errors.append("task lock protocol digest does not match the task")
    review = task.get("protocol_review")
    if lock.get("protocol_review") != review:
        errors.append("task lock prospective review does not match the task")
    if isinstance(review, dict) and _is_safe_relative(review.get("artifact")):
        review_path = root / review["artifact"]
        if not review_path.is_file() or lock.get("protocol_review_artifact_sha256") != _sha256_file(
            review_path
        ):
            errors.append(
                "prospective review artifact changed or disappeared after run initialization"
            )
    else:
        errors.append("locked task prospective review artifact is invalid")
    seal_path_value = lock.get("data_seal_path")
    if not _is_safe_relative(seal_path_value):
        errors.append("task.lock.json data_seal_path is invalid")
    else:
        if seal_path_value != task.get("data_seal"):
            errors.append("task lock data seal path does not match the task")
        seal_path = root / seal_path_value
        if not seal_path.is_file() or lock.get("data_seal_sha256") != _sha256_file(seal_path):
            errors.append("data seal changed or disappeared after run initialization")
    if lock.get("command") != task.get("run_command"):
        errors.append("task lock command does not match the task")
    source_commit = lock.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", source_commit) is None
    ):
        errors.append("task lock source_commit must be a Git object id")
    if not isinstance(lock.get("source_dirty"), bool):
        errors.append("task lock source_dirty must be boolean")
    if not isinstance(lock.get("development"), bool):
        errors.append("task lock development must be boolean")
    if lock.get("development") is False and lock.get("source_dirty") is not False:
        errors.append("official task lock cannot record a dirty source state")
    source_diff = lock.get("source_diff_sha256")
    if not isinstance(source_diff, str) or re.fullmatch(r"[0-9a-f]{64}", source_diff) is None:
        errors.append("task lock source_diff_sha256 must be a lowercase SHA-256")
    started = lock.get("started_at_utc")
    if not isinstance(started, str):
        errors.append("task lock started_at_utc must be an ISO-8601 string")
    else:
        try:
            started_at = dt.datetime.fromisoformat(started)
        except ValueError:
            errors.append("task lock started_at_utc must be an ISO-8601 string")
        else:
            if started_at.utcoffset() != dt.timedelta(0):
                errors.append("task lock started_at_utc must include a UTC offset")
    report_version = lock.get("report_template_version")
    if report_version not in SUPPORTED_REPORT_TEMPLATE_VERSIONS:
        errors.append("task lock uses an unsupported report template")
    elif report_version != _task_report_version(task):
        errors.append("task lock report template does not match the frozen task")
    frozen = task.get("frozen_configuration")
    integrity = frozen.get("live_integrity_policy") if isinstance(frozen, dict) else None
    if isinstance(integrity, dict):
        if integrity.get("verify_source_at_coordinator_worker_and_bundle") is True:
            errors.extend(verify_source_binding(task, root=root))
        if integrity.get("verify_full_data_seal_at_coordinator_entry_exit_and_bundle") is True:
            errors.extend(verify_data_seal(task, root=root))
    return task, lock, errors


def _manifest_errors(
    run: Path,
    root: Path,
    task: dict[str, Any],
    metrics: dict[str, Any],
) -> list[str]:
    try:
        manifest = _load_json(run / "manifest.json")
    except ValueError as error:
        return [str(error)]
    required = {"schema_version", "task_id", "report_template_version", "entries"}
    if set(manifest) != required:
        return ["manifest.json has the wrong keys"]
    errors: list[str] = []
    if manifest["schema_version"] != 1:
        errors.append("manifest.json schema_version must be 1")
    if manifest["task_id"] != task["task_id"]:
        errors.append("manifest.json task_id does not match the locked task")
    if manifest["report_template_version"] != 2:
        errors.append("manifest.json report_template_version must be 2")
    entries = manifest["entries"]
    if not isinstance(entries, list):
        return errors + ["manifest.json entries must be a list"]
    entry_keys = {"label", "path", "scope", "role", "media_type", "size_bytes", "sha256"}
    seen: set[tuple[str, str]] = set()
    run_paths: set[str] = set()
    repository_paths: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != entry_keys:
            errors.append(f"manifest.json entries[{index}] has the wrong keys")
            continue
        if not isinstance(entry["label"], str) or not entry["label"].strip():
            errors.append(f"manifest.json entries[{index}].label must be non-empty")
        path = entry["path"]
        scope = entry["scope"]
        if (
            not _is_safe_relative(path)
            or not isinstance(scope, str)
            or scope not in {"run", "repository"}
        ):
            errors.append(f"manifest.json entries[{index}] has an invalid path/scope")
            continue
        identity = (scope, path)
        if identity in seen:
            errors.append(f"manifest.json repeats {scope} path: {path}")
            continue
        seen.add(identity)
        if not isinstance(entry["role"], str) or not entry["role"].strip():
            errors.append(f"manifest.json entries[{index}].role must be non-empty")
        if not isinstance(entry["media_type"], str) or "/" not in entry["media_type"]:
            errors.append(f"manifest.json entries[{index}].media_type is invalid")
        if (
            not isinstance(entry["size_bytes"], int)
            or isinstance(entry["size_bytes"], bool)
            or entry["size_bytes"] < 0
        ):
            errors.append(f"manifest.json entries[{index}].size_bytes is invalid")
        if (
            not isinstance(entry["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
        ):
            errors.append(f"manifest.json entries[{index}].sha256 is invalid")
        base = run if scope == "run" else root
        target = base / path
        try:
            target.resolve(strict=True).relative_to(base.resolve())
        except (FileNotFoundError, ValueError):
            errors.append(f"manifest target is missing or escapes {scope}: {path}")
            continue
        if target.is_symlink() or not target.is_file():
            errors.append(f"manifest target must be a regular non-symlink file: {path}")
            continue
        if target.stat().st_size != entry["size_bytes"]:
            errors.append(f"manifest size mismatch: {path}")
        if _sha256_file(target) != entry["sha256"]:
            errors.append(f"manifest SHA-256 mismatch: {path}")
        if scope == "run":
            run_paths.add(path)
        else:
            repository_paths.add(path)

    expected_run_paths = {
        path.relative_to(run).as_posix()
        for path in run.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and not (path.name.startswith(".") and path.name.endswith(".tmp"))
    }
    missing_run = sorted(expected_run_paths - run_paths)
    extra_run = sorted(run_paths - expected_run_paths)
    if missing_run:
        errors.append("manifest omits run files: " + ", ".join(missing_run))
    if extra_run:
        errors.append("manifest names unexpected run files: " + ", ".join(extra_run))
    expected_repository = {
        item["path"]
        for item in metrics.get("evidence", [])
        if isinstance(item, dict) and _is_safe_relative(item.get("path"))
    }
    if repository_paths != expected_repository:
        errors.append("manifest repository entries must exactly match metrics.json evidence")

    for name in ("index.html", "README.md"):
        if name not in run_paths:
            errors.append(f"manifest omits generated {name}")
    index_path = run / "index.html"
    readme_path = run / "README.md"
    if index_path.is_file():
        body = index_path.read_text(encoding="utf-8", errors="replace")
        collector = LinkCollector()
        collector.feed(body)
        if "manifest.json" not in collector.links or "README.md" not in collector.links:
            errors.append("index.html must link README.md and manifest.json")
        for entry in entries:
            if not isinstance(entry, dict) or not _is_safe_relative(entry.get("path")):
                continue
            if entry.get("scope") == "repository":
                link = os.path.relpath(root / entry["path"], run)
            else:
                link = entry["path"]
            if link != "index.html" and link not in collector.links:
                errors.append(f"index.html does not link manifest entry: {entry['path']}")
    if readme_path.is_file():
        body = readme_path.read_text(encoding="utf-8", errors="replace")
        markdown_links = set(re.findall(r"\]\(([^)]+)\)", body))
        if "manifest.json" not in markdown_links or "## Commands" not in body:
            errors.append("README.md must link manifest.json and include Commands")
        for entry in entries:
            if not isinstance(entry, dict) or not _is_safe_relative(entry.get("path")):
                continue
            if entry.get("scope") == "repository":
                link = os.path.relpath(root / entry["path"], run)
            else:
                link = entry["path"]
            if link != "README.md" and link not in markdown_links:
                errors.append(f"README.md does not link manifest entry: {entry['path']}")
    return errors


def _cell_bundle_errors(
    run: Path,
    root: Path,
    task: dict[str, Any],
    lock: dict[str, Any],
) -> list[str]:
    frozen = task.get("frozen_configuration")
    policy = frozen.get("cell_receipt_policy") if isinstance(frozen, dict) else None
    if policy is None:
        return []
    if not isinstance(policy, dict) or not _is_safe_relative(policy.get("bundle_path")):
        return ["frozen cell_receipt_policy is invalid"]
    required_policy = {
        "schema",
        "bundle_path",
        "warmup_cells",
        "measured_cells",
        "validate_before_resume_and_aggregation",
        "hash_every_required_artifact",
        "strict_semantic_bundle_replay",
        "warmup_artifacts",
        "measured_artifacts",
        "effective_sha256",
    }
    if set(policy) != required_policy or any(
        policy.get(name) is not True
        for name in (
            "validate_before_resume_and_aggregation",
            "hash_every_required_artifact",
            "strict_semantic_bundle_replay",
        )
    ):
        return ["frozen cell_receipt_policy lacks the strict semantic replay contract"]
    try:
        bundle = _load_json(run / policy["bundle_path"])
    except ValueError as error:
        return [str(error)]
    required_bundle = {
        "schema",
        "task_id",
        "protocol_sha256",
        "task_lock_sha256",
        "data_seal_sha256",
        "source_binding_sha256",
        "warmup_cell_count",
        "measured_cell_count",
        "entries",
    }
    errors: list[str] = []
    if set(bundle) != required_bundle:
        return ["cell bundle receipt has the wrong keys"]
    source_binding = frozen.get("source_binding", {})
    expected_header = {
        "schema": "rtgs.janelle_gaussian2d_image_cell_bundle.v1",
        "task_id": task["task_id"],
        "protocol_sha256": lock["protocol_sha256"],
        "task_lock_sha256": _sha256_file(run / "task.lock.json"),
        "data_seal_sha256": lock["data_seal_sha256"],
        "source_binding_sha256": source_binding.get("aggregate_sha256"),
        "warmup_cell_count": policy.get("warmup_cells"),
        "measured_cell_count": policy.get("measured_cells"),
    }
    for key, expected in expected_header.items():
        if bundle.get(key) != expected or (
            key in {"warmup_cell_count", "measured_cell_count"}
            and (not isinstance(bundle.get(key), int) or isinstance(bundle.get(key), bool))
        ):
            errors.append(f"cell bundle receipt {key} differs from the lock/task")
    warmup = frozen.get("warmup", {})
    expected_identities = [
        (
            warmup.get("dataset_id"),
            warmup.get("arm_id"),
            warmup.get("seed"),
            "warmup",
        ),
        *[
            (dataset["id"], comparator["id"], seed, "measured")
            for dataset in task["datasets"]
            for seed in task["seeds"]
            for comparator in task["comparators"]
        ],
    ]
    entries = bundle.get("entries")
    if not isinstance(entries, list) or len(entries) != len(expected_identities):
        return errors + ["cell bundle receipt entry count differs from the frozen matrix"]
    observed_identities = []
    observed_paths: set[str] = set()
    lock_sha256 = _sha256_file(run / "task.lock.json")
    required_receipt = {
        "schema",
        "task_id",
        "protocol_sha256",
        "task_lock_sha256",
        "data_seal_sha256",
        "source_binding_sha256",
        "dataset_id",
        "arm",
        "seed",
        "mode",
        "iterations",
        "output_path",
        "partition_sha256",
        "effective_sha256",
        "input_binding",
        "artifacts",
    }
    expected_effective = policy["effective_sha256"]
    expected_artifacts = {
        "warmup": policy["warmup_artifacts"],
        "measured": policy["measured_artifacts"],
    }
    if not all(
        isinstance(value, list)
        and bool(value)
        and all(_is_safe_relative(item) for item in value)
        and len(value) == len(set(value))
        for value in expected_artifacts.values()
    ):
        return errors + ["frozen cell artifact inventories are invalid"]
    run_relative = run.relative_to(root).as_posix()
    frozen_iterations = frozen.get("rgb_refinement", {}).get("iterations")
    warmup_iterations = warmup.get("iterations")
    optimizer_views = frozen.get("optimizer_views")
    validation_views = frozen.get("validation_views")
    if not (
        isinstance(frozen_iterations, int)
        and not isinstance(frozen_iterations, bool)
        and isinstance(warmup_iterations, int)
        and not isinstance(warmup_iterations, bool)
        and isinstance(optimizer_views, list)
        and isinstance(validation_views, list)
    ):
        return errors + ["frozen cell iteration/partition policy is invalid"]
    summary_metric_ids = [item.get("id") for item in task.get("primary_metrics", [])]
    if not summary_metric_ids or not all(isinstance(item, str) for item in summary_metric_ids):
        return errors + ["frozen primary metric inventory is invalid"]

    for index, entry in enumerate(entries):
        required_entry = {
            "dataset_id",
            "arm",
            "seed",
            "mode",
            "receipt_path",
            "receipt_bytes",
            "receipt_sha256",
        }
        if not isinstance(entry, dict) or set(entry) != required_entry:
            errors.append(f"cell bundle entries[{index}] has the wrong keys")
            continue
        if (
            not isinstance(entry["dataset_id"], str)
            or not isinstance(entry["arm"], str)
            or not isinstance(entry["seed"], int)
            or isinstance(entry["seed"], bool)
            or not isinstance(entry["mode"], str)
            or not isinstance(entry["receipt_bytes"], int)
            or isinstance(entry["receipt_bytes"], bool)
            or entry["receipt_bytes"] <= 0
            or not isinstance(entry["receipt_sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", entry["receipt_sha256"]) is None
        ):
            errors.append(f"cell bundle entries[{index}] has invalid typed values")
            continue
        identity = (entry["dataset_id"], entry["arm"], entry["seed"], entry["mode"])
        observed_identities.append(identity)
        dataset_id, arm, seed, mode = identity
        if mode == "warmup":
            cell_relative = Path("warmup") / dataset_id / arm
            iterations = warmup_iterations
        elif mode == "measured":
            cell_relative = Path("cells") / dataset_id / f"seed_{seed}" / arm
            iterations = frozen_iterations
        else:
            errors.append(f"cell bundle entries[{index}] mode is invalid")
            continue
        expected_receipt_relative = (cell_relative / "cell_receipt.json").as_posix()
        relative = entry["receipt_path"]
        if (
            not _is_safe_relative(relative)
            or relative != expected_receipt_relative
            or relative in observed_paths
        ):
            errors.append(f"cell bundle entries[{index}] receipt_path is invalid or duplicated")
            continue
        observed_paths.add(relative)
        receipt_path = run / relative
        if (
            not receipt_path.is_file()
            or receipt_path.stat().st_size != entry["receipt_bytes"]
            or _sha256_file(receipt_path) != entry["receipt_sha256"]
        ):
            errors.append(f"cell receipt changed or disappeared: {relative}")
            continue
        try:
            receipt = _load_json(receipt_path)
        except ValueError as error:
            errors.append(str(error))
            continue
        if set(receipt) != required_receipt:
            errors.append(f"cell receipt has the wrong keys: {relative}")
            continue
        receipt_identity = (
            receipt.get("dataset_id"),
            receipt.get("arm"),
            receipt.get("seed"),
            receipt.get("mode"),
        )
        if receipt_identity != identity:
            errors.append(f"cell receipt identity differs from bundle entry: {relative}")
        if (
            not isinstance(receipt.get("seed"), int)
            or isinstance(receipt.get("seed"), bool)
            or not isinstance(receipt.get("iterations"), int)
            or isinstance(receipt.get("iterations"), bool)
            or receipt.get("iterations") != iterations
            or receipt.get("schema") != policy["schema"]
        ):
            errors.append(f"cell receipt typed identity/configuration is invalid: {relative}")
        for key, expected in (
            ("task_id", task["task_id"]),
            ("protocol_sha256", lock["protocol_sha256"]),
            ("task_lock_sha256", lock_sha256),
            ("data_seal_sha256", lock["data_seal_sha256"]),
            ("source_binding_sha256", source_binding.get("aggregate_sha256")),
        ):
            if receipt.get(key) != expected:
                errors.append(f"cell receipt {key} differs: {relative}")
        output_value = receipt.get("output_path")
        expected_output = f"{run_relative}/{cell_relative.as_posix()}"
        if not _is_safe_relative(output_value) or output_value != expected_output:
            errors.append(f"cell receipt output_path is invalid: {relative}")
            continue
        output = root / output_value
        if output.resolve() != receipt_path.parent.resolve():
            errors.append(f"cell receipt output_path does not name its directory: {relative}")
            continue
        split = task.get("splits", {}).get(dataset_id)
        if not isinstance(split, dict) or not isinstance(split.get("heldout"), list):
            errors.append(f"cell receipt partition cannot be reconstructed: {relative}")
            continue
        partition_sha256 = _canonical_sha256(
            {
                "optimizer": optimizer_views,
                "validation": validation_views,
                "heldout": split["heldout"],
            }
        )
        if receipt.get("partition_sha256") != partition_sha256:
            errors.append(f"cell receipt partition digest differs: {relative}")

        try:
            expected_effective_sha256 = expected_effective[mode][arm][str(seed)]
        except (KeyError, TypeError):
            errors.append(f"cell receipt effective policy is absent: {relative}")
            expected_effective_sha256 = None
        if receipt.get("effective_sha256") != expected_effective_sha256:
            errors.append(f"cell receipt effective configuration differs: {relative}")

        artifacts = receipt.get("artifacts")
        artifact_names = expected_artifacts[mode]
        if (
            not isinstance(artifacts, list)
            or [item.get("path") if isinstance(item, dict) else None for item in artifacts]
            != artifact_names
        ):
            errors.append(f"cell receipt artifact inventory differs: {relative}")
            continue
        artifact_paths: set[str] = set()
        for artifact in artifacts:
            if (
                not isinstance(artifact, dict)
                or set(artifact) != {"path", "bytes", "sha256"}
                or not _is_safe_relative(artifact.get("path"))
                or artifact["path"] in artifact_paths
                or not isinstance(artifact.get("bytes"), int)
                or isinstance(artifact.get("bytes"), bool)
                or artifact["bytes"] <= 0
                or not isinstance(artifact.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]) is None
            ):
                errors.append(f"cell receipt artifact record is invalid: {relative}")
                continue
            artifact_paths.add(artifact["path"])
            target = output / artifact["path"]
            if (
                not target.is_file()
                or target.stat().st_size != artifact["bytes"]
                or _sha256_file(target) != artifact["sha256"]
            ):
                errors.append(
                    f"cell artifact changed or disappeared: {output_value}/{artifact['path']}"
                )

        try:
            summary = _load_json(output / "summary.json")
            field = _load_json(output / "field_lift.json")
        except ValueError as error:
            errors.append(str(error))
            continue
        if (
            summary.get("status") != "completed"
            or summary.get("task_id") != task["task_id"]
            or summary.get("dataset_id") != dataset_id
            or summary.get("arm") != arm
            or not isinstance(summary.get("seed"), int)
            or isinstance(summary.get("seed"), bool)
            or summary.get("seed") != seed
            or summary.get("warmup") is not (mode == "warmup")
        ):
            errors.append(f"cell summary identity differs: {relative}")
        effective = summary.get("effective")
        if not isinstance(effective, dict) or _canonical_sha256(effective) != receipt.get(
            "effective_sha256"
        ):
            errors.append(f"cell summary effective configuration differs: {relative}")
        if mode == "warmup":
            if (
                summary.get("heldout_outcome_access") is not False
                or (output / "heldout_metrics.json").exists()
            ):
                errors.append(f"warmup cell contains held-out outcome access: {relative}")
        else:
            metrics = summary.get("metrics")
            if (
                summary.get("heldout_opened_after_endpoint_saved") is not True
                or summary.get("measurement_endpoint_before_heldout") is not True
                or not isinstance(metrics, dict)
                or set(metrics) != set(summary_metric_ids)
                or any(not _finite_number(value) for value in metrics.values())
            ):
                errors.append(f"measured cell endpoint/metric semantics differ: {relative}")

        input_binding = receipt.get("input_binding")
        expected_input_keys = {
            "manifest_sha256",
            "compact_optimizer_sha256",
            "camera_records_sha256",
            "optimizer_validation_image_sha256",
        }
        if (
            not isinstance(input_binding, dict)
            or set(input_binding) != expected_input_keys
            or any(
                not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in input_binding.values()
            )
        ):
            errors.append(f"cell receipt input binding is invalid: {relative}")
            continue
        camera_alignment = field.get("camera_alignment")
        image_input = field.get("optimizer_validation_image_input")
        summary_input = summary.get("input_binding")
        expected_summary_input_keys = {
            "camera_records_sha256",
            "optimizer_validation_image_sha256",
        }
        if mode == "measured":
            expected_summary_input_keys.add("heldout_image_sha256")
        expected_input = {
            "manifest_sha256": field.get("manifest_sha256"),
            "compact_optimizer_sha256": field.get("loaded_optimizer_compact_sha256"),
            "camera_records_sha256": (
                camera_alignment.get("records_sha256")
                if isinstance(camera_alignment, dict)
                else None
            ),
            "optimizer_validation_image_sha256": (
                image_input.get("records_sha256") if isinstance(image_input, dict) else None
            ),
        }
        if input_binding != expected_input:
            errors.append(f"cell receipt input binding differs from cell evidence: {relative}")
        if (
            not isinstance(summary_input, dict)
            or set(summary_input) != expected_summary_input_keys
            or summary_input.get("camera_records_sha256") != input_binding["camera_records_sha256"]
            or summary_input.get("optimizer_validation_image_sha256")
            != input_binding["optimizer_validation_image_sha256"]
            or (
                mode == "measured"
                and (
                    not isinstance(summary_input.get("heldout_image_sha256"), str)
                    or re.fullmatch(r"[0-9a-f]{64}", summary_input["heldout_image_sha256"]) is None
                )
            )
        ):
            errors.append(f"cell summary input binding differs: {relative}")
    if observed_identities != expected_identities:
        errors.append("cell bundle identities/order differ from the frozen matrix")
    return errors


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
    task, lock, errors = _locked_task(run, root=root)
    if not task:
        return errors
    report_version = lock.get("report_template_version")
    completed = True
    if report_version == 2:
        completed, source_errors = _v2_source_errors(run, task, lock)
        errors.extend(source_errors)
        if completed:
            errors.extend(_cell_bundle_errors(run, root, task, lock))
    metrics_path = run / "metrics.json"
    try:
        metrics = _load_json(metrics_path)
    except ValueError as error:
        return errors + [str(error)]
    if report_version == 1:
        errors.extend(_metric_errors_v1(metrics, task))
    elif report_version == 2:
        errors.extend(_metric_errors_v2(metrics, task, lock, completed=completed))
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
        elif f'content="{report_version}"' not in index.read_text(
            encoding="utf-8", errors="replace"
        ):
            errors.append("index.html does not match the locked canonical template")
        if report_version == 2:
            if not (run / "README.md").is_file():
                errors.append("missing generated README.md; run the render command")
            if not (run / "manifest.json").is_file():
                errors.append("missing generated manifest.json; run the render command")
            elif index.is_file() and (run / "README.md").is_file():
                errors.extend(_manifest_errors(run, root, task, metrics))
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


def _render_run_v1(run: Path, *, root: Path = ROOT) -> Path:
    """Render a grandfathered v1 bundle without changing its historical contract."""

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
    review = task["protocol_review"]
    review_link = os.path.relpath(root / review["artifact"], run)
    viewer = shlex.join(metrics["viewer_command"])
    seeds = ", ".join(str(seed) for seed in task["seeds"])
    datasets = ", ".join(f"{dataset['id']} ({dataset['role']})" for dataset in task["datasets"])

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="rtgs-experiment-report-template" content="1">
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
<p class="eyebrow">realtime-gs · canonical experiment report v1</p>
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
<p><strong>Prospective review:</strong>
<a href="{html.escape(review_link, quote=True)}">{html.escape(review["reviewer"])}</a>
 · verdict: <code>{html.escape(review["verdict"])}</code>
 · protocol: <code>{html.escape(review["protocol_sha256"])}</code></p>
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


def _flatten_parameters(value: object, prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key in sorted(value):
            nested = f"{prefix}.{key}" if prefix else key
            rows.extend(_flatten_parameters(value[key], nested))
    elif isinstance(value, list):
        rows.append((prefix, json.dumps(value, ensure_ascii=True, allow_nan=False)))
    elif value is None:
        rows.append((prefix, "null"))
    elif isinstance(value, bool):
        rows.append((prefix, str(value).lower()))
    else:
        rows.append((prefix, str(value)))
    return rows


def _history_svg(
    metric_id: str,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
    stage_markers: list[dict[str, Any]],
) -> str:
    width, height = 820, 350
    left, right, top, bottom = 62, 18, 46, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    elapsed = [float(item["wall_seconds"]) for item in records]
    elapsed.extend(float(item["wall_seconds"]) for item in stage_markers)
    values = [float(item["value"]) for item in records]
    x_min, x_max = min(elapsed), max(elapsed)
    y_min, y_max = min(values), max(values)
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        padding = abs(y_min) * 0.05 or 0.5
        y_min -= padding
        y_max += padding
    else:
        padding = (y_max - y_min) * 0.08
        y_min -= padding
        y_max += padding

    def x_position(wall_seconds: float) -> float:
        return left + (wall_seconds - x_min) * plot_width / (x_max - x_min)

    def y_position(value: float) -> float:
        return top + (y_max - value) * plot_height / (y_max - y_min)

    grid: list[str] = []
    for index in range(5):
        fraction = index / 4
        y = top + fraction * plot_height
        label = y_max - fraction * (y_max - y_min)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" '
            'stroke="#d8deea" stroke-width="1"/>'
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="11" fill="#5c667a">{html.escape(_format_number(label))}</text>'
        )
    series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        series[record["split"]].append(record)
    colors = ("#3659d9", "#d95f36", "#17876d", "#8b4cc2", "#b17a13", "#326f9f")
    lines: list[str] = []
    legend: list[str] = []
    for index, (key, points) in enumerate(sorted(series.items())):
        color = colors[index % len(colors)]
        ordered = sorted(points, key=lambda item: (item["wall_seconds"], item["step"]))
        coordinates = " ".join(
            f"{x_position(float(item['wall_seconds'])):.2f},{y_position(float(item['value'])):.2f}"
            for item in ordered
        )
        lines.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            'stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        legend.append(f'<span><i style="background:{color}"></i>{html.escape(key)}</span>')
    marker_lookup = {(marker["stage"], marker["boundary"]): marker for marker in stage_markers}
    stage_order = list(dict.fromkeys(marker["stage"] for marker in stage_markers))
    stage_fills = ("#dce5ff", "#e1f3ed", "#f9e9da", "#eee2f8")
    bands: list[str] = []
    boundaries: list[str] = []
    stage_legend: list[str] = []
    for index, stage in enumerate(stage_order):
        start = marker_lookup.get((stage, "start"))
        end = marker_lookup.get((stage, "end"))
        if start is None:
            continue
        start_seconds = float(start["wall_seconds"])
        start_x = x_position(start_seconds)
        end_seconds = float(end["wall_seconds"]) if end is not None else x_max
        end_x = x_position(end_seconds)
        fill = stage_fills[index % len(stage_fills)]
        bands.append(
            f'<rect data-stage="{html.escape(stage, quote=True)}" '
            f'x="{start_x:.2f}" y="{top}" width="{max(0.0, end_x - start_x):.2f}" '
            f'height="{plot_height}" fill="{fill}" fill-opacity="0.52"/>'
        )
        if end_x - start_x >= 54:
            bands.append(
                f'<text x="{(start_x + end_x) / 2:.2f}" y="{top + 14}" '
                'text-anchor="middle" font-size="10" font-weight="650" fill="#465168">'
                f"{html.escape(start['label'])}</text>"
            )
        for boundary, marker in (("start", start), ("end", end)):
            if marker is None:
                continue
            marker_x = x_position(float(marker["wall_seconds"]))
            dash = "" if boundary == "start" else ' stroke-dasharray="5 3"'
            boundaries.append(
                f'<line class="stage-boundary" '
                f'data-stage="{html.escape(stage, quote=True)}" '
                f'data-boundary="{boundary}" x1="{marker_x:.2f}" y1="{top}" '
                f'x2="{marker_x:.2f}" y2="{top + plot_height}" stroke="#4c566d" '
                f'stroke-width="1.6"{dash}><title>'
                f"{html.escape(marker['label'])} · {boundary} · "
                f"{html.escape(_format_number(float(marker['wall_seconds'])))} s"
                "</title></line>"
            )
        end_text = (
            f"{_format_number(float(end['wall_seconds']))} s" if end is not None else "incomplete"
        )
        stage_legend.append(
            '<span class="stage-interval">'
            f'<i style="background:{fill}"></i><strong>{html.escape(start["label"])}</strong>: '
            f"start {_format_number(start_seconds)} s → end {html.escape(end_text)}</span>"
        )
    title = metadata[metric_id]["label"]
    unit = metadata[metric_id]["unit"]
    first = records[0]
    series_label = f"{first['dataset_id']} · {first['arm_id']} · seed {first['seed']}"
    return (
        "<section class='panel history-chart'>"
        f"<h3>{html.escape(title)}</h3><p class='series-context'>{html.escape(series_label)}</p>"
        f"<p class='unit'>{html.escape(unit)} over worker cell wall time</p>"
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title, quote=True)} over worker cell wall time, with '
        'stage start and end boundaries">'
        + "".join(bands)
        + "".join(grid)
        + f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" '
        f'y2="{top + plot_height}" stroke="#596277"/>'
        + "".join(lines)
        + "".join(boundaries)
        + f'<text x="{left}" y="{height - 38}" font-size="11" fill="#5c667a">'
        f"{html.escape(_format_number(x_min))}</text>"
        + f'<text x="{width - right}" y="{height - 38}" text-anchor="end" '
        f'font-size="11" fill="#5c667a">{html.escape(_format_number(x_max))}</text>'
        + f'<text x="{width / 2:.1f}" y="{height - 18}" text-anchor="middle" '
        'font-size="12" fill="#5c667a">worker cell wall time from worker start (s)</text></svg>'
        + "<div class='legend'>"
        + "".join(legend)
        + "</div><div class='stage-legend' aria-label='Stage intervals'>"
        + "".join(stage_legend)
        + "</div></section>"
    )


def _history_charts(history: dict[str, Any]) -> str:
    records = history["records"]
    if not records:
        return (
            "<section class='panel failure'><h3>No fitting history</h3>"
            "<p>The run failed before it produced a fitting record. See the failure receipt.</p>"
            "</section>"
        )
    by_metric_and_series: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["metric_id"],
            record["dataset_id"],
            record["arm_id"],
            record["seed"],
        )
        by_metric_and_series[key].append(record)
    markers_by_series: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for marker in history["stage_markers"]:
        key = (marker["dataset_id"], marker["arm_id"], marker["seed"])
        markers_by_series[key].append(marker)
    return "".join(
        _history_svg(
            key[0],
            by_metric_and_series[key],
            history["metric_metadata"],
            markers_by_series[key[1:]],
        )
        for key in sorted(by_metric_and_series)
    )


def _inventory_role(path: str, *, repository: bool) -> str:
    if repository:
        return "evidence"
    if path in {"index.html", "README.md"}:
        return "report"
    if path == "metrics.json":
        return "summary"
    if path == "training_history.json":
        return "fitting_history"
    if path == "gaussians.config.json":
        return "effective_parameters"
    if path == "task.lock.json":
        return "provenance"
    if path.endswith("receipt.json") or path.endswith("receipt.md") or path == "viewer_smoke.json":
        return "receipt"
    if path.endswith(".ply"):
        return "model"
    if Path(path).suffix.lower() in {".png", ".gif", ".jpg", ".jpeg", ".webp"}:
        return "preview"
    return "artifact"


def _media_type(path: str) -> str:
    suffixes = {
        ".csv": "text/csv",
        ".gif": "image/gif",
        ".html": "text/html",
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".json": "application/json",
        ".md": "text/markdown",
        ".npy": "application/octet-stream",
        ".npz": "application/octet-stream",
        ".ply": "application/octet-stream",
        ".png": "image/png",
        ".txt": "text/plain",
        ".webp": "image/webp",
    }
    return suffixes.get(Path(path).suffix.lower(), "application/octet-stream")


def _inventory_descriptors(run: Path, root: Path, metrics: dict[str, Any]) -> list[dict[str, str]]:
    labels = {
        item["path"]: item["label"]
        for item in metrics["artifacts"]
        if isinstance(item, dict) and _is_safe_relative(item.get("path"))
    }
    descriptors: list[dict[str, str]] = []
    excluded = {"index.html", "README.md", "manifest.json"}
    for target in sorted(run.rglob("*")):
        if target.is_symlink():
            raise ValueError(f"run bundles may not contain symlinks: {target.relative_to(run)}")
        if not target.is_file():
            continue
        path = target.relative_to(run).as_posix()
        if path in excluded or (target.name.startswith(".") and target.name.endswith(".tmp")):
            continue
        descriptors.append(
            {
                "label": labels.get(path, path.replace("_", " ")),
                "path": path,
                "scope": "run",
                "role": _inventory_role(path, repository=False),
            }
        )
    for name, label in (("index.html", "Interactive results report"), ("README.md", "Run note")):
        descriptors.append({"label": label, "path": name, "scope": "run", "role": "report"})
    for item in metrics["evidence"]:
        descriptors.append(
            {
                "label": item["label"],
                "path": item["path"],
                "scope": "repository",
                "role": "evidence",
            }
        )
    return sorted(descriptors, key=lambda item: (item["scope"], item["path"]))


def _descriptor_link(descriptor: dict[str, str], run: Path, root: Path) -> str:
    if descriptor["scope"] == "repository":
        return os.path.relpath(root / descriptor["path"], run)
    return descriptor["path"]


def _parameter_rows_html(parameters: list[tuple[str, str]]) -> str:
    return "".join(
        "<tr>"
        f"<td><code>{html.escape(name)}</code></td>"
        f"<td><code>{html.escape(value)}</code></td>"
        "</tr>"
        for name, value in parameters
    )


def _metrics_html(metrics: dict[str, Any]) -> str:
    if not metrics["metrics"]:
        return "<section class='panel failure'><p>No final metrics were produced.</p></section>"
    grouped: dict[str, list[tuple[int | float, dict[str, str]]]] = defaultdict(list)
    for metric_id, value in metrics["metrics"].items():
        metadata = metrics["metric_metadata"][metric_id]
        grouped[metadata["group"]].append((value, metadata))
    sections: list[str] = []
    for group, rows in grouped.items():
        body = "".join(
            "<tr>"
            f"<td>{html.escape(metadata['label'])}</td>"
            f"<td>{html.escape(_format_number(value))}</td>"
            f"<td>{html.escape(metadata['unit'])}</td>"
            f"<td>{html.escape(metadata['direction'])}</td></tr>"
            for value, metadata in rows
        )
        sections.append(
            f"<section class='panel'><h3>{html.escape(group)}</h3>"
            "<div class='table-wrap'><table><thead><tr><th>Metric</th><th>Value</th>"
            f"<th>Unit</th><th>Better</th></tr></thead><tbody>{body}</tbody></table></div>"
            "</section>"
        )
    return "".join(sections)


def _commands_html(commands: dict[str, Any]) -> str:
    viewer = commands["viewer"]
    viewer_text = shlex.join(viewer) if viewer else "Unavailable: the run did not complete."
    return (
        "<section class='panel commands'><h3>Reproduce</h3>"
        f"<pre>{html.escape(shlex.join(commands['reproduce']))}</pre>"
        "<h3>Serve this report</h3>"
        f"<pre>{html.escape(shlex.join(commands['serve_report']))}</pre>"
        "<p>Open <code>http://localhost:8765/index.html</code> from the served directory.</p>"
        "<h3>Start the orbit viewer</h3>"
        f"<pre>{html.escape(viewer_text)}</pre></section>"
    )


def _render_v2_readme(
    task: dict[str, Any],
    lock: dict[str, Any],
    metrics: dict[str, Any],
    receipt: dict[str, Any],
    parameters: list[tuple[str, str]],
    descriptors: list[dict[str, str]],
    run: Path,
    root: Path,
) -> str:
    status = receipt["status"]
    lines = [
        f"# {task['title']}",
        "",
        "> Generated by `scripts/experiment_contract.py`; do not hand-edit.",
        "",
        f"- Task: `{task['task_id']}`",
        f"- Status: **{status}**",
        f"- Decision: `{metrics['decision']}`",
        f"- Evidence phase: `{task['evidence_phase']}`",
        f"- Source commit: `{lock['source_commit']}`",
        "",
        "## Summary",
        "",
        metrics["summary"],
        "",
        "## Claim boundary",
        "",
        metrics["claim_boundary"],
        "",
    ]
    if status == "failed":
        lines.extend(
            [
                "## Failure",
                "",
                f"- Phase: `{receipt['failure_phase']}`",
                f"- Exit code: `{receipt['exit_code']}`",
                f"- Message: {receipt['message']}",
                "",
            ]
        )
    policy = task["input_policy"]
    lines.extend(
        [
            "## Input boundary",
            "",
            "- Reconstruction may read: "
            + ", ".join(f"`{item}`" for item in policy["reconstruction_allowed"]),
            "- Reconstruction must reject: "
            + ", ".join(f"`{item}`" for item in policy["reconstruction_forbidden"]),
            "- Evaluation may read: "
            + ", ".join(f"`{item}`" for item in policy["evaluation_allowed"]),
            "",
            "## Pipeline",
            "",
        ]
    )
    lines.extend(
        f"{index}. **{stage['label']}** — {stage['purpose']}"
        for index, stage in enumerate(task["stages"], start=1)
    )
    lines.append("")
    lines.extend(["## Effective parameters", "", "| Parameter | Value |", "|---|---|"])
    lines.extend(
        f"| <code>{html.escape(name)}</code> | "
        f"<code>{html.escape(value).replace('|', '&#124;')}</code> |"
        for name, value in parameters
    )
    lines.extend(["", "## Commands", "", "Run these from the repository root.", ""])
    command_sections = (
        ("Reproduce", metrics["commands"]["reproduce"]),
        ("Serve the report", metrics["commands"]["serve_report"]),
        ("Start the orbit viewer", metrics["commands"]["viewer"]),
    )
    for label, command in command_sections:
        lines.extend([f"### {label}", ""])
        if command:
            lines.extend(["```sh", shlex.join(command), "```", ""])
        else:
            lines.extend(["Unavailable because the run did not complete.", ""])
    lines.extend(["## Final metrics", "", "| Metric | Value | Unit |", "|---|---:|---|"])
    if metrics["metrics"]:
        for metric_id, value in metrics["metrics"].items():
            metadata = metrics["metric_metadata"][metric_id]
            label = html.escape(metadata["label"]).replace("|", "&#124;")
            unit = html.escape(metadata["unit"]).replace("|", "&#124;")
            lines.append(f"| {label} | {_format_number(value)} | {unit} |")
    else:
        lines.append("| No final metrics produced | — | — |")
    lines.extend(["", "## Artifacts and evidence", "", "- [Checksum manifest](manifest.json)"])
    for descriptor in descriptors:
        if descriptor["path"] == "README.md":
            continue
        link = _descriptor_link(descriptor, run, root)
        label = descriptor["label"].replace("[", "\\[").replace("]", "\\]")
        lines.append(f"- [{label}]({link}) — `{descriptor['role']}`")
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in metrics["notes"])
    lines.append("")
    return "\n".join(lines)


def _manifest_entries(
    descriptors: list[dict[str, str]], run: Path, root: Path
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for descriptor in descriptors:
        base = run if descriptor["scope"] == "run" else root
        target = base / descriptor["path"]
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"cannot inventory missing/non-regular file: {descriptor['path']}")
        entries.append(
            {
                **descriptor,
                "media_type": _media_type(descriptor["path"]),
                "size_bytes": target.stat().st_size,
                "sha256": _sha256_file(target),
            }
        )
    return entries


def _dataset_curve_svg(curve: dict[str, Any]) -> str:
    """Render one compact multi-series curve for a per-dataset report."""

    width, height = 820, 330
    left, right, top, bottom = 66, 20, 42, 58
    plot_width = width - left - right
    plot_height = height - top - bottom
    points = [point for series in curve["series"] for point in series["points"]]
    xs = [float(point["x"]) for point in points]
    ys = [float(point["value"]) for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        padding = abs(y_min) * 0.05 or 0.5
        y_min -= padding
        y_max += padding
    else:
        padding = 0.08 * (y_max - y_min)
        y_min -= padding
        y_max += padding

    def x_position(value: float) -> float:
        return left + (value - x_min) * plot_width / (x_max - x_min)

    def y_position(value: float) -> float:
        return top + (y_max - value) * plot_height / (y_max - y_min)

    grid = []
    for index in range(5):
        fraction = index / 4
        y = top + fraction * plot_height
        label = y_max - fraction * (y_max - y_min)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" '
            'stroke="#d8deea"/>'
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="11" fill="#5c667a">{html.escape(_format_number(label))}</text>'
        )
    colors = ("#3659d9", "#d95f36", "#17876d", "#8b4cc2", "#b17a13", "#326f9f")
    lines: list[str] = []
    legend: list[str] = []
    for index, series in enumerate(curve["series"]):
        color = colors[index % len(colors)]
        ordered = sorted(series["points"], key=lambda item: float(item["x"]))
        coordinates = " ".join(
            f"{x_position(float(point['x'])):.2f},{y_position(float(point['value'])):.2f}"
            for point in ordered
        )
        lines.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
            'stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        lines.extend(
            f'<circle cx="{x_position(float(point["x"])):.2f}" '
            f'cy="{y_position(float(point["value"])):.2f}" r="3.2" fill="{color}">'
            f"<title>{html.escape(series['label'])} · x={_format_number(point['x'])} · "
            f"{_format_number(point['value'])}</title></circle>"
            for point in ordered
        )
        legend.append(
            f'<span><i style="background:{color}"></i>{html.escape(series["label"])}</span>'
        )
    return (
        "<section class='panel history-chart'>"
        f"<h3>{html.escape(curve['title'])}</h3>"
        f"<p class='unit'>{html.escape(curve['unit'])} · better: "
        f"{html.escape(curve['direction'])}</p>"
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(curve["title"], quote=True)}">'
        + "".join(grid)
        + f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" '
        f'y2="{top + plot_height}" stroke="#596277"/>'
        + "".join(lines)
        + f'<text x="{width / 2:.1f}" y="{height - 17}" text-anchor="middle" '
        f'font-size="12" fill="#5c667a">{html.escape(curve["x_label"])}</text></svg>'
        + "<div class='legend'>"
        + "".join(legend)
        + "</div></section>"
    )


def _render_dataset_pages(
    run: Path,
    task: dict[str, Any],
    metrics: dict[str, Any],
    history: dict[str, Any],
) -> dict[str, str]:
    """Render canonical child reports requested by a metrics dataset_summaries map."""

    summaries = metrics.get("dataset_summaries")
    if not summaries:
        return {}
    labels = {item["id"]: item["role"] for item in task["datasets"]}
    links: dict[str, str] = {}
    shared_style = """
:root {
  color-scheme: light;
  --ink: #172033;
  --muted: #5c667a;
  --line: #d8deea;
  --paper: #f5f7fb;
  --panel: #fff;
  --accent: #3659d9;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, sans-serif;
}
main { max-width: 1180px; margin: auto; padding: 32px 22px 70px; }
h1 { font-size: clamp(27px, 4vw, 43px); line-height: 1.08; margin: .2rem 0 .7rem; }
h2 { margin: 2.1rem 0 .8rem; }
h3 { margin: .1rem 0 .7rem; }
.eyebrow, .unit {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .08em;
  font-size: 12px;
  font-weight: 700;
}
.summary { font-size: 18px; max-width: 900px; }
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
  gap: 14px;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 18px;
  box-shadow: 0 2px 8px #1520400a;
  min-width: 0;
}
.table-wrap { overflow: auto; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-size: 12px; text-transform: uppercase; }
.bar-row {
  display: grid;
  grid-template-columns: minmax(110px, 1fr) minmax(110px, 2fr) 80px;
  gap: 9px;
  align-items: center;
  margin: 9px 0;
}
.bar-track {
  position: relative;
  height: 14px;
  background: #edf0f6;
  border-radius: 8px;
  overflow: hidden;
}
.bar-zero { position: absolute; top: 0; bottom: 0; width: 1px; background: #596277; }
.bar-fill {
  position: absolute;
  top: 0;
  height: 100%;
  background: linear-gradient(90deg, var(--accent), #7b93ec);
}
.bar-value { text-align: right; font-variant-numeric: tabular-nums; }
svg { width: 100%; height: auto; }
.legend, .stage-legend { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 12px; }
.legend span { display: flex; align-items: center; gap: 5px; }
.legend i { width: 16px; height: 3px; }
.series-context { color: var(--muted); margin: -.35rem 0 .45rem; }
.stage-legend { margin-top: 9px; padding-top: 9px; border-top: 1px solid var(--line); }
.stage-interval { display: flex; align-items: center; gap: 5px; }
.stage-interval i { width: 14px; height: 14px; border: 1px solid #aab3c4; }
code, pre { background: #eef1f7; border-radius: 7px; }
code { padding: 2px 5px; }
pre { padding: 13px; overflow: auto; }
a { color: #254fc4; }
footer { color: var(--muted); margin-top: 28px; }
"""
    for dataset_id, summary in summaries.items():
        directory = run / "datasets" / dataset_id
        directory.mkdir(parents=True, exist_ok=True)
        child_history = {
            **history,
            "records": [item for item in history["records"] if item["dataset_id"] == dataset_id],
            "stage_markers": [
                item for item in history["stage_markers"] if item["dataset_id"] == dataset_id
            ],
        }
        artifact_rows = []
        for item in summary["artifacts"]:
            link = os.path.relpath(run / item["path"], directory)
            artifact_rows.append(
                f'<li><a href="{html.escape(link, quote=True)}">'
                f"{html.escape(item['label'])}</a></li>"
            )
        viewer = shlex.join(summary["commands"]["viewer"])
        notes = "".join(f"<li>{html.escape(item)}</li>" for item in summary["notes"])
        curve_html = "".join(_dataset_curve_svg(item) for item in summary["curves"])
        history_html = _history_charts(child_history)
        metrics_html = _metrics_html(summary)
        chart_html = "".join(_render_chart(item) for item in summary["charts"])
        artifact_html = "".join(artifact_rows)
        page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="rtgs-experiment-report-template" content="2">
<title>{html.escape(summary["title"])}</title>
<style>{shared_style}</style></head><body><main>
<p class="eyebrow">realtime-gs · canonical per-dataset experiment report v2</p>
<p><a href="../../index.html">← All datasets</a></p>
<h1>{html.escape(summary["title"])}</h1>
<p><code>{html.escape(dataset_id)}</code> · {html.escape(labels[dataset_id])}</p>
<p class="summary">{html.escape(summary["summary"])}</p>
<h2>All final metrics across measured seeds</h2><div class="grid">{curve_html}</div>
<h2>Optimizer and stage curves</h2><div class="grid">{history_html}</div>
<h2>Final metric table</h2><div class="grid">{metrics_html}</div>
<h2>Arm comparisons</h2><div class="grid">{chart_html}</div>
<h2>Orbit viewer</h2><section class="panel"><pre>{html.escape(viewer)}</pre></section>
<h2>Artifacts</h2><section class="panel"><ul>{artifact_html}</ul></section>
<h2>Notes</h2><section class="panel"><ul>{notes}</ul></section>
<footer>Generated from metrics.json and training_history.json. Do not hand-edit this page.</footer>
</main></body></html>"""
        target = directory / "index.html"
        temporary = directory / ".index.html.tmp"
        temporary.write_text(page, encoding="utf-8")
        temporary.replace(target)
        links[dataset_id] = target.relative_to(run).as_posix()
    return links


def _render_run_v2(run: Path, *, root: Path = ROOT) -> Path:
    errors = validate_run(run, root=root, require_index=False)
    if errors:
        raise ValueError("cannot render invalid run:\n- " + "\n- ".join(errors))
    task, lock, lock_errors = _locked_task(run, root=root)
    if lock_errors:
        raise ValueError("\n".join(lock_errors))
    metrics = _load_json(run / "metrics.json")
    history = _load_json(run / "training_history.json")
    config = _load_json(run / "gaussians.config.json")
    receipt = _load_json(run / "run_receipt.json")
    environment = _load_json(run / "environment.json")
    parameters = _flatten_parameters(config)
    dataset_report_links = _render_dataset_pages(run, task, metrics, history)
    descriptors = _inventory_descriptors(run, root, metrics)
    status = receipt["status"]
    status_detail = (
        f"Completed successfully (exit {receipt['exit_code']})."
        if status == "completed"
        else f"Failed during {receipt['failure_phase']} (exit {receipt['exit_code']}): "
        + receipt["message"]
    )
    stages = "".join(
        "<li><strong>"
        + html.escape(stage["label"])
        + "</strong><span>"
        + html.escape(stage["purpose"])
        + "</span></li>"
        for stage in task["stages"]
    )
    final_charts = "".join(_render_chart(chart) for chart in metrics["charts"])
    inventory_rows = []
    for descriptor in descriptors:
        link = _descriptor_link(descriptor, run, root)
        if descriptor["path"] == "index.html":
            label = html.escape(descriptor["label"])
        else:
            label = (
                f'<a href="{html.escape(link, quote=True)}">{html.escape(descriptor["label"])}</a>'
            )
        inventory_rows.append(
            f"<tr><td>{label}</td><td><code>{html.escape(descriptor['path'])}</code></td>"
            f"<td>{html.escape(descriptor['role'])}</td>"
            f"<td>{html.escape(descriptor['scope'])}</td></tr>"
        )
    task_link = os.path.relpath(root / lock["task_path"], run)
    seal_link = os.path.relpath(root / lock["data_seal_path"], run)
    review = task["protocol_review"]
    review_link = os.path.relpath(root / review["artifact"], run)
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in metrics["notes"])
    datasets = ", ".join(f"{item['id']} ({item['role']})" for item in task["datasets"])
    environment_rows = [
        ("Python", environment["python"]),
        ("Platform", environment["platform"]),
        ("Device", f"{environment['device']['type']} · {environment['device']['name']}"),
        ("CUDA", environment["device"]["cuda"] or "none"),
    ]
    environment_html = "".join(
        f"<tr><td>{html.escape(label)}</td><td><code>{html.escape(value)}</code></td></tr>"
        for label, value in environment_rows
    )
    allowed = ", ".join(task["input_policy"]["reconstruction_allowed"])
    forbidden = ", ".join(task["input_policy"]["reconstruction_forbidden"])
    evaluation = ", ".join(task["input_policy"]["evaluation_allowed"])
    dataset_reports = "".join(
        "<section class='panel'><h3>"
        + html.escape(dataset_id)
        + "</h3><p><a href='"
        + html.escape(path, quote=True)
        + "'>Open metrics, curves, artifacts, and orbit command</a></p></section>"
        for dataset_id, path in dataset_report_links.items()
    )
    page = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="rtgs-experiment-report-template" content="2">
<title>{html.escape(task["title"])}</title>
<style>
:root {{ color-scheme:light; --ink:#172033; --muted:#5c667a; --line:#d8deea;
--paper:#f5f7fb; --panel:#fff; --accent:#3659d9; --danger:#a73535; --ok:#17705a; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--paper); color:var(--ink);
font:15px/1.5 system-ui,-apple-system,sans-serif; }}
main {{ max-width:1180px; margin:auto; padding:36px 22px 72px; }}
h1 {{ font-size:clamp(28px,4vw,46px); line-height:1.08; margin:.2rem 0 .7rem; }}
h2 {{ margin:2.2rem 0 .8rem; }} h3 {{ margin:.1rem 0 .7rem; }}
.eyebrow,.unit {{ color:var(--muted); text-transform:uppercase; letter-spacing:.08em;
font-size:12px; font-weight:700; }} .summary {{ font-size:19px; max-width:900px; }}
.chips {{ display:flex; gap:8px; flex-wrap:wrap; margin:18px 0; }}
.chip {{ background:#e8edff; color:#2543b4; border-radius:999px; padding:5px 10px;
font-weight:650; }} .status {{ border-left:6px solid var(--ok); }}
.status.failed,.failure {{ border-left:6px solid var(--danger); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); gap:14px; }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:14px;
padding:18px; box-shadow:0 2px 8px #1520400a; min-width:0; }}
.claim {{ border-left:5px solid #e0902f; }} .pipeline {{ display:flex; list-style:none;
gap:10px; padding:0; overflow-x:auto; }} .pipeline li {{ min-width:180px; flex:1;
background:var(--panel); border:1px solid var(--line); border-top:4px solid var(--accent);
border-radius:10px; padding:13px; }} .pipeline span {{ display:block; color:var(--muted);
font-size:13px; margin-top:5px; }} .table-wrap {{ overflow:auto; }}
table {{ width:100%; border-collapse:collapse; }} th,td {{ text-align:left; padding:8px 10px;
border-bottom:1px solid var(--line); vertical-align:top; }} th {{ color:var(--muted);
font-size:12px; text-transform:uppercase; letter-spacing:.05em; }}
.bar-row {{ display:grid; grid-template-columns:minmax(110px,1fr) minmax(110px,2fr) 80px;
gap:9px; align-items:center; margin:9px 0; }} .bar-track {{ position:relative; height:14px;
background:#edf0f6; border-radius:8px; overflow:hidden; }} .bar-zero {{ position:absolute;
top:0; bottom:0; width:1px; background:#596277; z-index:2; }} .bar-fill {{ position:absolute;
top:0; height:100%; background:linear-gradient(90deg,var(--accent),#7b93ec); }}
.bar-value {{ text-align:right; font-variant-numeric:tabular-nums; }} svg {{ width:100%;
height:auto; }} .legend,.stage-legend {{ display:flex; flex-wrap:wrap; gap:6px 14px;
font-size:12px; }}
.legend span {{ display:flex; align-items:center; gap:5px; }}
.legend i {{ width:16px; height:3px; }} .series-context {{ color:var(--muted);
margin:-.35rem 0 .45rem; }} .stage-legend {{ margin-top:9px; padding-top:9px;
border-top:1px solid var(--line); }} .stage-interval {{ display:flex; align-items:center;
gap:5px; }} .stage-interval i {{ width:14px; height:14px; border:1px solid #aab3c4;
flex:0 0 auto; }}
code,pre {{ background:#eef1f7; border-radius:7px; }} code {{ padding:2px 5px; }}
pre {{ padding:13px; overflow:auto; }} a {{ color:#254fc4; }} footer {{ color:var(--muted);
margin-top:28px; }}
</style></head><body><main>
<p class="eyebrow">realtime-gs · canonical experiment bundle v2</p>
<h1>{html.escape(task["title"])}</h1>
<div class="chips"><span class="chip">{html.escape(task["task_id"])}</span>
<span class="chip">{html.escape(task["evidence_phase"])}</span>
<span class="chip">decision: {html.escape(metrics["decision"])}</span></div>
<section class="panel status {html.escape(status)}"><h2>Run status: {html.escape(status)}</h2>
<p>{html.escape(status_detail)}</p></section>
<p class="summary">{html.escape(metrics["summary"])}</p>
<section class="panel claim"><h2>Claim boundary</h2>
<p>{html.escape(metrics["claim_boundary"])}</p></section>

<h2>Input boundary</h2><div class="grid">
<section class="panel"><h3>Reconstruction may read</h3><p>{html.escape(allowed)}</p></section>
<section class="panel"><h3>Reconstruction must reject</h3><p>{html.escape(forbidden)}</p></section>
<section class="panel"><h3>Evaluation may read</h3><p>{html.escape(evaluation)}</p></section></div>

<h2>Pipeline</h2><ol class="pipeline">{stages}</ol>
<h2>Per-dataset reports</h2><div class="grid">{dataset_reports}</div>
<h2>Fitting process</h2><div class="grid">{_history_charts(history)}</div>
<h2>Final metrics</h2><div class="grid">{_metrics_html(metrics)}</div>
<h2>Required diagrams</h2><div class="grid">{final_charts}</div>

<h2>Effective parameters</h2><section class="panel table-wrap"><table><thead><tr>
<th>Parameter</th><th>Value</th></tr></thead><tbody>{_parameter_rows_html(parameters)}</tbody>
</table></section>

<h2>Commands</h2>{_commands_html(metrics["commands"])}

<h2>Protocol, environment, and provenance</h2><div class="grid">
<section class="panel"><h3>Frozen protocol</h3>
<p><strong>Task:</strong> <a href="{html.escape(task_link, quote=True)}">
{html.escape(task["task_id"])}</a></p>
<p><strong>Data seal:</strong> <a href="{html.escape(seal_link, quote=True)}">
{html.escape(lock["data_seal_path"])}</a></p>
<p><strong>Prospective review:</strong> <a href="{html.escape(review_link, quote=True)}">
{html.escape(review["reviewer"])}</a></p>
<p><strong>Protocol digest:</strong> <code>{html.escape(review["protocol_sha256"])}</code></p>
<p><strong>Source commit:</strong> <code>{html.escape(lock["source_commit"])}</code></p>
<p><strong>Datasets:</strong> {html.escape(datasets)}</p></section>
<section class="panel"><h3>Execution environment</h3>
<table><tbody>{environment_html}</tbody></table>
<p><a href="environment.json">Full environment record</a> ·
<a href="run_receipt.json">Run receipt</a></p>
</section></div>

<h2>Artifact inventory</h2><section class="panel table-wrap"><p>
<a href="README.md">Run note</a> · <a href="manifest.json">Checksummed manifest</a></p>
<table><thead><tr><th>Artifact</th><th>Path</th><th>Role</th><th>Scope</th></tr></thead>
<tbody>{"".join(inventory_rows)}</tbody></table></section>
<h2>Notes</h2><section class="panel"><ul>{notes}</ul></section>
<footer>Generated from frozen machine records. Do not hand-edit index.html, README.md,
or manifest.json.</footer>
</main></body></html>
"""
    readme = _render_v2_readme(task, lock, metrics, receipt, parameters, descriptors, run, root)
    for name, body in (("index.html", page), ("README.md", readme)):
        path = run / name
        temporary = run / f".{name}.tmp"
        temporary.write_text(body, encoding="utf-8")
        temporary.replace(path)
    manifest = {
        "schema_version": 1,
        "task_id": task["task_id"],
        "report_template_version": 2,
        "entries": _manifest_entries(descriptors, run, root),
    }
    _write_json(run / "manifest.json", manifest)
    output_errors = validate_run(run, root=root)
    if output_errors:
        raise ValueError("rendered bundle failed validation:\n- " + "\n- ".join(output_errors))
    return run / "index.html"


def render_run(run: Path, *, root: Path = ROOT) -> Path:
    """Render the report version frozen into the task and run lock."""

    run = run.resolve()
    task, lock, errors = _locked_task(run, root=root)
    if not task or errors:
        raise ValueError("cannot render invalid run lock:\n- " + "\n- ".join(errors))
    if lock["report_template_version"] == 1:
        return _render_run_v1(run, root=root)
    if lock["report_template_version"] == 2:
        return _render_run_v2(run, root=root)
    raise ValueError("unsupported report template version")


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

    review_digest = subparsers.add_parser(
        "review-digest",
        help="print the digest a prospective protocol review must bind",
    )
    review_digest.add_argument("task", type=Path)

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

    render = subparsers.add_parser("render", help="render the frozen canonical report bundle")
    render.add_argument("run", type=Path)

    check = subparsers.add_parser(
        "check-run", help="validate the task lock, producer records, generated report, and manifest"
    )
    check.add_argument("run", type=Path)
    args = parser.parse_args(argv)

    if args.command == "validate":
        return _print_errors("experiment_contract", validate_repository())
    if args.command == "validate-data":
        task = _load_json(args.task)
        return _print_errors("experiment_data", verify_data_seal(task))
    if args.command == "review-digest":
        task = _load_json(args.task)
        print(f"{task.get('task_id', '<missing-task-id>')} {protocol_sha256(task)}")
        return 0
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
