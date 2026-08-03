"""Strict realtime-gs receipt export for StructSplat BENCH-019.

This module is deliberately passive.  It never runs a reconstruction and never imports
StructSplat.  It validates one frozen cross-repository protocol cell, extracts metrics from
sealed JSON sources without transformations, binds the downstream factor actually declared by
the run receipt, and emits the exact ``structsplat.bench019.cell.v1`` row consumed upstream.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROTOCOL_SCHEMA = "structsplat.bench019.protocol.v1"
ROW_SCHEMA = "structsplat.bench019.cell.v1"
SOURCE_SCHEMA = "rtgs.structsplat_bench019.source.v1"
RUN_BINDING_SCHEMA = "rtgs.structsplat_bench019.run_binding.v1"
FACTOR_SCHEMA = "rtgs.structsplat_bench019.factor.v1"
EXPORT_RECEIPT_SCHEMA = "rtgs.structsplat_bench019.export_receipt.v1"
ASSEMBLY_RECEIPT_SCHEMA = "rtgs.structsplat_bench019.assembly_receipt.v1"
REQUIRED_CELL_ARTIFACTS = (
    "field",
    "history",
    "config",
    "target",
    "reconstruction",
    "error",
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_ARTIFACT_KEYS = frozenset({"path", "sha256", "bytes"})
_REPOSITORY_KEYS = frozenset(
    {"name", "root", "commit", "branch", "dirty", "status_sha256", "environment"}
)
_REVIEW_PROTOCOL_KEYS = frozenset(
    {
        "schema",
        "task_id",
        "state",
        "driver",
        "claim_scope",
        "repositories",
        "captures",
        "downstream",
        "predictors",
        "responses",
        "analysis",
        "aa_replay",
        "design_sha256",
    }
)
_FROZEN_PROTOCOL_KEYS = _REVIEW_PROTOCOL_KEYS | {"review", "protocol_sha256"}
_REVIEW_KEYS = frozenset({"driver", "reviewer", "verdict", "design_sha256", "artifact"})
_DOWNSTREAM_KEYS = frozenset(
    {
        "task_manifest",
        "dataset_manifest",
        "environment",
        "schedule_config",
        "command",
        "working_directory",
        "outcome_root",
        "seeds",
        "initializers",
        "result_schema",
    }
)
_SEMANTICS_KEYS = frozenset(
    {
        "provider",
        "equation",
        "blend_mode",
        "alpha_policy",
        "coordinate_convention",
        "semantic_digest",
    }
)
_ROW_KEYS = frozenset(
    {
        "schema",
        "status",
        "error",
        "capture_id",
        "frame_id",
        "family_id",
        "seed",
        "initializer",
        "replicate_id",
        "field_manifest_sha256",
        "field_semantic_digest",
        "downstream_factor_digest",
        "stage1",
        "downstream",
        "artifacts",
    }
)
_CELL_KEYS = frozenset(
    {"capture_id", "frame_id", "family_id", "seed", "initializer", "replicate_id"}
)
_RUN_BINDING_KEYS = frozenset(
    {
        "schema",
        "capture_id",
        "frame_id",
        "family_id",
        "seed",
        "initializer",
        "replicate_id",
        "field_manifest_sha256",
        "field_semantic_digest",
        "downstream_factor_digest",
    }
)
_SOURCE_KEYS = frozenset(
    {
        "schema",
        "protocol_digest",
        "status",
        "error",
        "cell",
        "sources",
        "metric_bindings",
        "artifacts",
    }
)
_EXPORT_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "diagnostic",
        "protocol",
        "protocol_digest",
        "source_manifest",
        "source_artifacts",
        "factor",
        "downstream_factor_digest",
        "row",
        "row_canonical_sha256",
    }
)
_ANALYSIS_KEYS = frozenset(
    {
        "bootstrap_replicates",
        "bootstrap_seed",
        "minimum_capture_groups",
        "minimum_frames",
        "minimum_family_count",
        "minimum_spearman",
        "minimum_bootstrap_lower",
        "minimum_lofo_top1_agreement",
        "selection_priority",
        "missing_policy",
    }
)
_AA_KEYS = frozenset(
    {
        "frame_id",
        "family_id",
        "seed",
        "initializer",
        "primary_replicate",
        "replay_replicate",
        "metric_abs_tolerance",
    }
)


class ExportError(ValueError):
    """Raised when a BENCH-019 export cannot preserve its declared evidence boundary."""


def canonical_json(value: object) -> bytes:
    """Return the canonical JSON representation shared with StructSplat BENCH-019."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    """Hash one ordinary file without loading it fully into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _strict_json_value(path: Path, *, label: str) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ExportError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ExportError(f"{label} contains non-finite JSON token {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExportError(f"could not read strict JSON from {label}: {error}") from error


def load_json_object(path: str | Path, *, label: str = "JSON") -> dict[str, Any]:
    """Load a duplicate-key- and non-finite-safe JSON object."""
    value = _strict_json_value(Path(path), label=label)
    if not isinstance(value, dict):
        raise ExportError(f"{label} must contain a JSON object")
    return value


def _exact_mapping(value: object, keys: frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = set(value) if isinstance(value, dict) else set()
        raise ExportError(
            f"{label} keys are not exact "
            f"(missing={sorted(keys - actual)}, extra={sorted(actual - keys)})"
        )
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ExportError(f"{label} is not a valid identifier")
    return value


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ExportError(f"{label} is not a lowercase SHA-256")
    return value


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExportError(f"{label} must be an integer")
    return value


def _finite(value: object, *, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExportError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ExportError(f"{label} must be finite")
    return value


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExportError(f"{label} must be a non-empty string")
    return value


def _artifact_descriptor(record: object, *, label: str) -> dict[str, Any]:
    value = _exact_mapping(record, _ARTIFACT_KEYS, label=label)
    if not isinstance(value["path"], str) or not value["path"]:
        raise ExportError(f"{label}.path must be a non-empty string")
    _sha256(value["sha256"], label=f"{label}.sha256")
    size = _integer(value["bytes"], label=f"{label}.bytes")
    if size < 0:
        raise ExportError(f"{label}.bytes must be non-negative")
    return value


def _artifact_path(record: object, base: Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    value = _artifact_descriptor(record, label=label)
    raw_path = value["path"]
    if not isinstance(raw_path, str) or not raw_path:
        raise ExportError(f"{label}.path must be a non-empty string")
    path = Path(raw_path)
    if not path.is_absolute():
        if ".." in path.parts:
            raise ExportError(f"{label}.path may not escape its manifest directory")
        path = base / path
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError as error:
        raise ExportError(f"{label} does not exist: {path}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ExportError(f"{label} must be an ordinary non-symlink file")
    resolved = path.resolve(strict=True)
    expected_bytes = _integer(value["bytes"], label=f"{label}.bytes")
    if expected_bytes < 0 or resolved.stat().st_size != expected_bytes:
        raise ExportError(f"{label} byte length differs from its descriptor")
    expected_hash = _sha256(value["sha256"], label=f"{label}.sha256")
    actual_hash = sha256_file(resolved)
    if actual_hash != expected_hash:
        raise ExportError(f"{label} SHA-256 differs from its descriptor")
    return resolved, {
        "path": str(resolved),
        "sha256": actual_hash,
        "bytes": expected_bytes,
    }


def describe_artifact(path: str | Path) -> dict[str, Any]:
    """Create an absolute descriptor for one ordinary non-symlink file."""
    raw = Path(path)
    try:
        mode = os.lstat(raw).st_mode
    except FileNotFoundError as error:
        raise ExportError(f"artifact does not exist: {raw}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ExportError("artifact must be an ordinary non-symlink file")
    resolved = raw.resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": int(resolved.stat().st_size),
    }


def _design_digest(protocol: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(protocol))
    for key in ("state", "design_sha256", "protocol_sha256", "review"):
        payload.pop(key, None)
    return _digest(payload)


def protocol_identity(
    protocol: Mapping[str, Any],
    *,
    protocol_base: str | Path = ".",
    allow_review: bool = False,
) -> str:
    """Validate and return the frozen protocol digest or review-state design digest."""
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ExportError(f"protocol schema must be {PROTOCOL_SCHEMA}")
    state = protocol.get("state")
    expected_keys = _REVIEW_PROTOCOL_KEYS if state == "review" else _FROZEN_PROTOCOL_KEYS
    _exact_mapping(dict(protocol), expected_keys, label="BENCH-019 protocol")
    _identifier(protocol.get("task_id"), label="protocol task_id")
    driver = _identifier(protocol.get("driver"), label="protocol driver")
    if protocol.get("claim_scope") not in {"general", "workload_specific"}:
        raise ExportError("protocol claim_scope must be general or workload_specific")
    recorded_design = _sha256(protocol.get("design_sha256"), label="design_sha256")
    if recorded_design != _design_digest(protocol):
        raise ExportError("protocol design digest does not match its contents")
    _protocol_cells(protocol)
    _metric_names(protocol, "predictors")
    _metric_names(protocol, "responses")
    if state == "review":
        if not allow_review:
            raise ExportError("formal export requires a frozen BENCH-019 protocol")
        return recorded_design
    if state != "frozen":
        raise ExportError("protocol state must be review or frozen")
    recorded = _sha256(protocol.get("protocol_sha256"), label="protocol_sha256")
    payload = copy.deepcopy(dict(protocol))
    payload.pop("protocol_sha256", None)
    if recorded != _digest(payload):
        raise ExportError("protocol digest does not match its contents")
    review = _exact_mapping(protocol.get("review"), _REVIEW_KEYS, label="protocol review")
    reviewer = _identifier(review.get("reviewer"), label="protocol reviewer")
    if (
        review.get("driver") != driver
        or reviewer.casefold() == driver.casefold()
        or review.get("verdict") != "approved"
        or review.get("design_sha256") != recorded_design
    ):
        raise ExportError("frozen protocol does not carry a distinct approval for its design")
    _artifact_path(
        review.get("artifact"),
        Path(protocol_base).resolve(),
        label="protocol review artifact",
    )
    _validate_formal_protocol(protocol, protocol_base=Path(protocol_base).resolve())
    return recorded


def _validate_formal_protocol(protocol: Mapping[str, Any], *, protocol_base: Path) -> None:
    """Mirror StructSplat's portable v1 invariants for formal export.

    Review-state diagnostics intentionally remain usable with small synthetic protocols. A frozen
    protocol, however, must satisfy the complete upstream shape before this repository can emit a
    formal row.
    """
    repositories = protocol.get("repositories")
    if not isinstance(repositories, list) or len(repositories) < 2:
        raise ExportError("BENCH-019 must bind both StructSplat and realtime-gs repositories")
    repository_names: list[str] = []
    clean_status_digest = hashlib.sha256(b"").hexdigest()
    for index, raw_repository in enumerate(repositories):
        repository = _exact_mapping(
            raw_repository,
            _REPOSITORY_KEYS,
            label=f"protocol repositories[{index}]",
        )
        repository_names.append(
            _identifier(repository.get("name"), label=f"repositories[{index}].name")
        )
        _nonempty_string(repository.get("root"), label=f"repositories[{index}].root")
        commit = repository.get("commit")
        if not isinstance(commit, str) or _HEX40.fullmatch(commit) is None:
            raise ExportError(f"repositories[{index}].commit must be a Git SHA")
        _nonempty_string(repository.get("branch"), label=f"repositories[{index}].branch")
        if repository.get("dirty") is not False:
            raise ExportError(f"repositories[{index}] must be clean")
        status_digest = _sha256(
            repository.get("status_sha256"),
            label=f"repositories[{index}].status_sha256",
        )
        if status_digest != clean_status_digest:
            raise ExportError(f"repositories[{index}] does not carry the clean status digest")
        _artifact_path(
            repository.get("environment"),
            protocol_base,
            label=f"repositories[{index}].environment",
        )
    if len(repository_names) != len(set(repository_names)):
        raise ExportError("protocol repository names must be unique")

    downstream = _exact_mapping(
        protocol.get("downstream"), _DOWNSTREAM_KEYS, label="protocol downstream"
    )
    command = downstream.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        raise ExportError("protocol downstream.command must be a non-empty argv list")
    working_directory = Path(
        _nonempty_string(downstream.get("working_directory"), label="downstream.working_directory")
    )
    outcome_root = Path(
        _nonempty_string(downstream.get("outcome_root"), label="downstream.outcome_root")
    )
    if not working_directory.is_absolute() or not outcome_root.is_absolute():
        raise ExportError("downstream working_directory and outcome_root must be absolute")
    if not working_directory.is_dir():
        raise ExportError("downstream.working_directory must exist as a directory")
    seeds = downstream.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ExportError("downstream.seeds must contain at least three unique integers")

    predictor_names = set(_metric_names(protocol, "predictors"))
    response_specs = protocol.get("responses")
    response_names = set(_metric_names(protocol, "responses"))
    if predictor_names & response_names:
        raise ExportError("predictor and response metric names must be disjoint")
    if not isinstance(response_specs, list):
        raise ExportError("protocol responses must be a list")
    primary_response = next(spec["name"] for spec in response_specs if spec["primary"])

    analysis = _exact_mapping(protocol.get("analysis"), _ANALYSIS_KEYS, label="protocol analysis")
    for name in (
        "bootstrap_replicates",
        "minimum_capture_groups",
        "minimum_frames",
        "minimum_family_count",
    ):
        value = analysis.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ExportError(f"analysis.{name} must be a positive integer")
    if isinstance(analysis.get("bootstrap_seed"), bool) or not isinstance(
        analysis.get("bootstrap_seed"), int
    ):
        raise ExportError("analysis.bootstrap_seed must be an integer")
    for name in ("minimum_spearman", "minimum_bootstrap_lower"):
        value = float(_finite(analysis.get(name), label=f"analysis.{name}"))
        if not -1.0 <= value <= 1.0:
            raise ExportError(f"analysis.{name} must lie in [-1,1]")
    agreement = float(
        _finite(
            analysis.get("minimum_lofo_top1_agreement"),
            label="analysis.minimum_lofo_top1_agreement",
        )
    )
    if not 0.0 <= agreement <= 1.0:
        raise ExportError("analysis.minimum_lofo_top1_agreement must lie in [0,1]")
    priority = analysis.get("selection_priority")
    if (
        not isinstance(priority, list)
        or any(not isinstance(name, str) for name in priority)
        or len(priority) != len(set(priority))
        or set(priority) != predictor_names
    ):
        raise ExportError("analysis.selection_priority must list every predictor exactly once")
    if analysis.get("missing_policy") != "fail_closed":
        raise ExportError("BENCH-019 supports only the fail_closed missing policy")
    if protocol.get("claim_scope") == "general" and (
        analysis["minimum_capture_groups"] < 3 or analysis["minimum_frames"] < 2
    ):
        raise ExportError("a general claim requires at least three groups and two frames")

    aa = _exact_mapping(protocol.get("aa_replay"), _AA_KEYS, label="protocol aa_replay")
    _, families = _protocol_cells(protocol)
    frame_id = _identifier(aa.get("frame_id"), label="aa_replay.frame_id")
    family_id = _identifier(aa.get("family_id"), label="aa_replay.family_id")
    if (frame_id, family_id) not in families:
        raise ExportError("aa_replay references an unknown frame/family")
    if aa.get("seed") not in seeds:
        raise ExportError("aa_replay.seed is not a frozen downstream seed")
    initializers = downstream.get("initializers")
    if not isinstance(initializers, list) or aa.get("initializer") not in initializers:
        raise ExportError("aa_replay.initializer is not frozen")
    primary_replicate = _identifier(
        aa.get("primary_replicate"), label="aa_replay.primary_replicate"
    )
    replay_replicate = _identifier(aa.get("replay_replicate"), label="aa_replay.replay_replicate")
    if primary_replicate == replay_replicate:
        raise ExportError("A/A replay labels must be distinct")
    tolerances = aa.get("metric_abs_tolerance")
    known_metrics = predictor_names | response_names
    if (
        not isinstance(tolerances, dict)
        or not tolerances
        or not set(tolerances) <= known_metrics
        or primary_response not in tolerances
        or not (set(tolerances) & predictor_names)
    ):
        raise ExportError(
            "A/A tolerances must contain the primary response and a Stage-1 predictor"
        )
    for name, tolerance in tolerances.items():
        if float(_finite(tolerance, label=f"aa_replay tolerance {name}")) < 0.0:
            raise ExportError("A/A tolerances must be non-negative")


def _metric_names(protocol: Mapping[str, Any], name: str) -> list[str]:
    raw = protocol.get(name)
    if not isinstance(raw, list) or not raw:
        raise ExportError(f"protocol {name} must be a non-empty list")
    result: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ExportError(f"protocol {name}[{index}] must be an object")
        required = {"name", "direction"}
        if name == "responses":
            required.add("primary")
        if set(item) != required:
            raise ExportError(f"protocol {name}[{index}] has unexpected fields")
        metric = _identifier(item.get("name"), label=f"protocol {name}[{index}].name")
        if metric in result:
            raise ExportError(f"protocol {name} contains duplicate metric {metric}")
        if item.get("direction") not in {"higher", "lower"}:
            raise ExportError(f"protocol {name}[{index}].direction must be higher or lower")
        if name == "responses" and not isinstance(item.get("primary"), bool):
            raise ExportError(f"protocol {name}[{index}].primary must be boolean")
        result.append(metric)
    if name == "responses" and sum(bool(item["primary"]) for item in raw) != 1:
        raise ExportError("protocol responses must declare exactly one primary metric")
    return result


def _protocol_cells(
    protocol: Mapping[str, Any],
) -> tuple[
    list[tuple[str, str, int, str, str]],
    dict[tuple[str, str], tuple[str, Mapping[str, Any]]],
]:
    downstream = _exact_mapping(
        protocol.get("downstream"), _DOWNSTREAM_KEYS, label="protocol downstream"
    )
    if downstream.get("result_schema") != ROW_SCHEMA:
        raise ExportError(f"protocol downstream.result_schema must be {ROW_SCHEMA}")
    for name in ("task_manifest", "dataset_manifest", "environment", "schedule_config"):
        _artifact_descriptor(downstream.get(name), label=f"protocol downstream.{name}")
    seeds = downstream.get("seeds")
    initializers = downstream.get("initializers")
    if not isinstance(seeds, list) or not isinstance(initializers, list):
        raise ExportError("protocol downstream seeds/initializers must be lists")
    checked_seeds = [_integer(seed, label="downstream seed") for seed in seeds]
    checked_initializers = [
        _identifier(value, label="downstream initializer") for value in initializers
    ]
    if (
        not checked_seeds
        or len(checked_seeds) != len(set(checked_seeds))
        or not checked_initializers
        or len(checked_initializers) != len(set(checked_initializers))
    ):
        raise ExportError("protocol downstream seeds/initializers must be non-empty and unique")
    aa = protocol.get("aa_replay")
    if not isinstance(aa, dict):
        raise ExportError("protocol aa_replay must be an object")
    primary = _identifier(aa.get("primary_replicate"), label="primary replicate")
    replay = _identifier(aa.get("replay_replicate"), label="A/A replay replicate")
    captures = protocol.get("captures")
    if not isinstance(captures, list) or not captures:
        raise ExportError("protocol captures must be a non-empty list")
    keys: list[tuple[str, str, int, str, str]] = []
    families: dict[tuple[str, str], tuple[str, Mapping[str, Any]]] = {}
    capture_ids: set[str] = set()
    frame_ids: set[str] = set()
    canonical_families: set[str] | None = None
    for capture in captures:
        capture = _exact_mapping(capture, frozenset({"id", "frames"}), label="protocol capture")
        capture_id = _identifier(capture.get("id"), label="capture id")
        if capture_id in capture_ids:
            raise ExportError(f"duplicate protocol capture {capture_id}")
        capture_ids.add(capture_id)
        frames = capture.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ExportError(f"capture {capture_id} frames must be a non-empty list")
        for frame in frames:
            frame = _exact_mapping(
                frame,
                frozenset({"id", "pixels", "masks", "cameras", "split", "families"}),
                label="protocol frame",
            )
            frame_id = _identifier(frame.get("id"), label="frame id")
            if frame_id in frame_ids:
                raise ExportError(f"duplicate protocol frame {frame_id}")
            frame_ids.add(frame_id)
            for name in ("pixels", "masks", "cameras"):
                _artifact_descriptor(frame.get(name), label=f"protocol frame {frame_id}.{name}")
            split = _exact_mapping(
                frame.get("split"),
                frozenset({"train", "heldout"}),
                label=f"protocol frame {frame_id} split",
            )
            train = split.get("train")
            heldout = split.get("heldout")
            if not isinstance(train, list) or not isinstance(heldout, list):
                raise ExportError(f"protocol frame {frame_id} split values must be lists")
            train_ids = [_identifier(value, label="train view") for value in train]
            heldout_ids = [_identifier(value, label="heldout view") for value in heldout]
            if (
                not train_ids
                or not heldout_ids
                or len(train_ids) != len(set(train_ids))
                or len(heldout_ids) != len(set(heldout_ids))
                or set(train_ids) & set(heldout_ids)
            ):
                raise ExportError(f"protocol frame {frame_id} split must be non-empty and disjoint")
            frame_families = frame.get("families")
            if not isinstance(frame_families, list) or not frame_families:
                raise ExportError(f"frame {frame_id} families must be a non-empty list")
            frame_family_ids: set[str] = set()
            for family in frame_families:
                family = _exact_mapping(
                    family,
                    frozenset({"id", "field_manifest", "stage1_metrics", "semantics"}),
                    label="protocol family",
                )
                family_id = _identifier(family.get("id"), label="family id")
                if family_id in frame_family_ids:
                    raise ExportError(f"frame {frame_id} has duplicate family {family_id}")
                frame_family_ids.add(family_id)
                _artifact_descriptor(
                    family.get("field_manifest"), label=f"family {family_id}.field_manifest"
                )
                _artifact_descriptor(
                    family.get("stage1_metrics"), label=f"family {family_id}.stage1_metrics"
                )
                semantics = _exact_mapping(
                    family.get("semantics"), _SEMANTICS_KEYS, label=f"family {family_id}.semantics"
                )
                equation = semantics.get("equation")
                expected_blend = {
                    "additive_sum": "additive",
                    "normalized_weighted_sum": "normalized",
                }.get(equation)
                if expected_blend is None or semantics.get("blend_mode") != expected_blend:
                    raise ExportError(f"family {family_id} equation/blend semantics disagree")
                _identifier(semantics.get("provider"), label=f"family {family_id} provider")
                _identifier(semantics.get("alpha_policy"), label=f"family {family_id} alpha policy")
                if not isinstance(semantics.get("coordinate_convention"), str) or not semantics.get(
                    "coordinate_convention"
                ):
                    raise ExportError(f"family {family_id} coordinate convention is invalid")
                _sha256(
                    semantics.get("semantic_digest"),
                    label=f"family {family_id} semantic digest",
                )
                pair = (frame_id, family_id)
                if pair in families:
                    raise ExportError(f"duplicate protocol frame/family {pair}")
                families[pair] = (capture_id, family)
                for seed in checked_seeds:
                    for initializer in checked_initializers:
                        keys.append((frame_id, family_id, seed, initializer, primary))
            if canonical_families is None:
                canonical_families = frame_family_ids
            elif frame_family_ids != canonical_families:
                raise ExportError("every protocol frame must contain the same field families")
    aa_key = (
        _identifier(aa.get("frame_id"), label="A/A frame"),
        _identifier(aa.get("family_id"), label="A/A family"),
        _integer(aa.get("seed"), label="A/A seed"),
        _identifier(aa.get("initializer"), label="A/A initializer"),
        replay,
    )
    base_key = (*aa_key[:4], primary)
    if base_key not in keys:
        raise ExportError("A/A replay does not identify a declared primary cell")
    keys.append(aa_key)
    if len(keys) != len(set(keys)):
        raise ExportError("protocol declares duplicate stable cell keys")
    return keys, families


def _cell_key(cell: Mapping[str, Any]) -> tuple[str, str, int, str, str]:
    return (
        _identifier(cell.get("frame_id"), label="cell.frame_id"),
        _identifier(cell.get("family_id"), label="cell.family_id"),
        _integer(cell.get("seed"), label="cell.seed"),
        _identifier(cell.get("initializer"), label="cell.initializer"),
        _identifier(cell.get("replicate_id"), label="cell.replicate_id"),
    )


def downstream_factor_record(
    protocol: Mapping[str, Any],
    *,
    frame_id: str,
    seed: int,
    initializer: str,
    protocol_base: str | Path = ".",
    allow_review: bool = False,
) -> dict[str, Any]:
    """Derive the family- and replicate-invariant downstream factor record.

    The factor is derived from verified global artifacts, not merely from digest strings copied
    into the protocol.  ``protocol_base`` resolves any relative descriptor paths.
    """
    identity = protocol_identity(
        protocol,
        protocol_base=protocol_base,
        allow_review=allow_review,
    )
    downstream = protocol.get("downstream")
    if not isinstance(downstream, dict):
        raise ExportError("protocol downstream must be an object")
    checked_frame_id = _identifier(frame_id, label="factor.frame_id")
    checked_seed = _integer(seed, label="factor.seed")
    checked_initializer = _identifier(initializer, label="factor.initializer")
    _, families = _protocol_cells(protocol)
    if not any(candidate_frame == checked_frame_id for candidate_frame, _ in families):
        raise ExportError("factor.frame_id is not declared by the protocol")
    if checked_seed not in downstream.get("seeds", []):
        raise ExportError("factor.seed is not declared by the protocol")
    if checked_initializer not in downstream.get("initializers", []):
        raise ExportError("factor.initializer is not declared by the protocol")
    bindings: dict[str, str] = {}
    base = Path(protocol_base).resolve()
    for name in ("task_manifest", "dataset_manifest", "environment", "schedule_config"):
        record = downstream.get(name)
        _, descriptor = _artifact_path(
            record,
            base,
            label=f"protocol downstream.{name}",
        )
        bindings[f"{name}_sha256"] = descriptor["sha256"]
    command = downstream.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(value, str) or not value for value in command)
    ):
        raise ExportError("protocol downstream.command must be a non-empty argv list")
    return {
        "schema": FACTOR_SCHEMA,
        "protocol_digest": identity,
        "frame_id": checked_frame_id,
        "seed": checked_seed,
        "initializer": checked_initializer,
        "bindings": bindings,
        "command": list(command),
        "result_schema": ROW_SCHEMA,
    }


def downstream_factor_digest(
    protocol: Mapping[str, Any],
    *,
    frame_id: str,
    seed: int,
    initializer: str,
    protocol_base: str | Path = ".",
    allow_review: bool = False,
) -> str:
    """Hash the only factor payload permitted to vary outside field-family identity."""
    return _digest(
        downstream_factor_record(
            protocol,
            frame_id=frame_id,
            seed=seed,
            initializer=initializer,
            protocol_base=protocol_base,
            allow_review=allow_review,
        )
    )


def _json_pointer(value: object, pointer: object, *, label: str) -> object:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ExportError(f"{label} must be a non-root RFC 6901 JSON pointer")
    current = value
    for raw_token in pointer[1:].split("/"):
        if re.search(r"~(?:[^01]|$)", raw_token):
            raise ExportError(f"{label} contains an invalid RFC 6901 escape")
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if token not in current:
                raise ExportError(f"{label} does not resolve at token {token!r}")
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise ExportError(f"{label} contains invalid list index {token!r}")
            index = int(token)
            if index >= len(current):
                raise ExportError(f"{label} list index is out of range")
            current = current[index]
        else:
            raise ExportError(f"{label} traverses through a scalar")
    return current


def _load_metric_sources(
    protocol: Mapping[str, Any],
    family: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    protocol_base: Path,
    source_base: Path,
) -> tuple[dict[str, object], dict[str, dict[str, Any]]]:
    stage1_path, stage1_descriptor = _artifact_path(
        family.get("stage1_metrics"), protocol_base, label="protocol family stage1_metrics"
    )
    documents = {"stage1_metrics": _strict_json_value(stage1_path, label="stage1_metrics")}
    descriptors = {"stage1_metrics": stage1_descriptor}
    raw_sources = source.get("sources")
    if not isinstance(raw_sources, dict):
        raise ExportError("source.sources must be an object")
    for raw_name, raw_record in raw_sources.items():
        name = _identifier(raw_name, label="source artifact name")
        if name == "stage1_metrics":
            raise ExportError("source manifest may not override protocol stage1_metrics")
        path, descriptor = _artifact_path(raw_record, source_base, label=f"source artifact {name}")
        documents[name] = _strict_json_value(path, label=f"source artifact {name}")
        descriptors[name] = descriptor
    return documents, descriptors


def _extract_metrics(
    protocol: Mapping[str, Any],
    source: Mapping[str, Any],
    documents: Mapping[str, object],
) -> tuple[dict[str, int | float], dict[str, int | float]]:
    bindings = source.get("metric_bindings")
    if not isinstance(bindings, dict) or set(bindings) != {"stage1", "downstream"}:
        raise ExportError("metric_bindings must contain exactly stage1 and downstream")
    output: dict[str, dict[str, int | float]] = {}
    for kind, protocol_name in (("stage1", "predictors"), ("downstream", "responses")):
        names = _metric_names(protocol, protocol_name)
        raw_kind = bindings.get(kind)
        if not isinstance(raw_kind, dict) or set(raw_kind) != set(names):
            raise ExportError(f"metric_bindings.{kind} differs from the frozen metric names")
        values: dict[str, int | float] = {}
        for name in names:
            binding = raw_kind[name]
            if not isinstance(binding, dict) or set(binding) != {"source", "pointer"}:
                raise ExportError(f"metric binding {kind}.{name} must contain source/pointer")
            source_name = _identifier(binding["source"], label=f"binding {kind}.{name}.source")
            if kind == "stage1" and source_name != "stage1_metrics":
                raise ExportError("Stage-1 predictors must come from the frozen Stage-1 artifact")
            if kind == "downstream" and source_name == "stage1_metrics":
                raise ExportError("downstream responses may not come from Stage-1 metrics")
            if source_name not in documents:
                raise ExportError(f"metric binding names unknown source {source_name}")
            raw_value = _json_pointer(
                documents[source_name],
                binding["pointer"],
                label=f"metric binding {kind}.{name}.pointer",
            )
            values[name] = _finite(raw_value, label=f"metric {kind}.{name}")
        output[kind] = values
    return output["stage1"], output["downstream"]


def _validate_metric_source_set(source: Mapping[str, Any], documents: Mapping[str, object]) -> None:
    """Reject receipt sources that are not load-bearing for metrics or the run binding."""
    bindings = source.get("metric_bindings")
    if not isinstance(bindings, dict):
        raise ExportError("metric_bindings must be an object")
    expected = {"stage1_metrics", "run_receipt"}
    for kind in ("stage1", "downstream"):
        raw_kind = bindings.get(kind)
        if not isinstance(raw_kind, dict):
            raise ExportError(f"metric_bindings.{kind} must be an object")
        for binding in raw_kind.values():
            if not isinstance(binding, dict):
                raise ExportError(f"metric_bindings.{kind} contains a non-object binding")
            source_name = binding.get("source")
            if isinstance(source_name, str):
                expected.add(source_name)
    if set(documents) != expected:
        raise ExportError("source manifest contains missing or unreferenced source artifacts")


def _validate_run_binding(
    document: object,
    *,
    cell: Mapping[str, Any],
    field_manifest_sha256: str,
    field_semantic_digest: str,
    factor_digest: str,
) -> None:
    if not isinstance(document, dict):
        raise ExportError("run_receipt must contain a JSON object")
    binding = _exact_mapping(document.get("bench019"), _RUN_BINDING_KEYS, label="run binding")
    expected = {
        "schema": RUN_BINDING_SCHEMA,
        **{name: cell[name] for name in _CELL_KEYS},
        "field_manifest_sha256": field_manifest_sha256,
        "field_semantic_digest": field_semantic_digest,
        "downstream_factor_digest": factor_digest,
    }
    if binding != expected:
        raise ExportError("run_receipt BENCH-019 binding differs from the frozen cell")


def _write_new_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise ExportError(f"refusing to overwrite output: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        raise


def _write_new_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(
        json.dumps(row, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n" for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise ExportError(f"refusing to overwrite output: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        raise


def export_cell(
    protocol_path: str | Path,
    source_path: str | Path,
    output_path: str | Path,
    receipt_path: str | Path,
    *,
    allow_review_protocol: bool = False,
) -> dict[str, Any]:
    """Export one exact BENCH-019 row and a separate provenance receipt."""
    protocol_file = Path(protocol_path).resolve(strict=True)
    source_file = Path(source_path).resolve(strict=True)
    output_file = Path(output_path)
    receipt_file = Path(receipt_path)
    if output_file.exists() or receipt_file.exists():
        raise ExportError("cell output and export receipt must both be new files")
    protocol = load_json_object(protocol_file, label="BENCH-019 protocol")
    identity = protocol_identity(
        protocol,
        protocol_base=protocol_file.parent,
        allow_review=allow_review_protocol,
    )
    source = load_json_object(source_file, label="BENCH-019 source manifest")
    _exact_mapping(source, _SOURCE_KEYS, label="source manifest")
    if source.get("schema") != SOURCE_SCHEMA:
        raise ExportError(f"source manifest schema must be {SOURCE_SCHEMA}")
    if source.get("protocol_digest") != identity:
        raise ExportError("source manifest binds a different protocol digest")
    cell = _exact_mapping(source.get("cell"), _CELL_KEYS, label="source cell")
    key = _cell_key(cell)
    expected_keys, families = _protocol_cells(protocol)
    if key not in expected_keys:
        raise ExportError(f"source cell {key} is not declared by the protocol")
    family_pair = (key[0], key[1])
    capture_id, family = families[family_pair]
    if cell.get("capture_id") != capture_id:
        raise ExportError("source capture_id differs from the frozen frame")
    field_manifest_path, field_manifest = _artifact_path(
        family.get("field_manifest"),
        protocol_file.parent,
        label="protocol family field_manifest",
    )
    del field_manifest_path
    semantics = family.get("semantics")
    if not isinstance(semantics, dict):
        raise ExportError("protocol family semantics must be an object")
    semantic_digest = _sha256(
        semantics.get("semantic_digest"), label="protocol field semantic digest"
    )
    factor_record = downstream_factor_record(
        protocol,
        frame_id=key[0],
        seed=key[2],
        initializer=key[3],
        protocol_base=protocol_file.parent,
        allow_review=allow_review_protocol,
    )
    factor_digest = _digest(factor_record)
    status = source.get("status")
    error = source.get("error")
    if status not in {"ok", "error"} or not isinstance(error, str):
        raise ExportError("source status/error must be an ok|error label and string")
    row: dict[str, Any] = {
        "schema": ROW_SCHEMA,
        "status": status,
        "error": error,
        **cell,
        "field_manifest_sha256": field_manifest["sha256"],
        "field_semantic_digest": semantic_digest,
        "downstream_factor_digest": factor_digest,
        "stage1": {},
        "downstream": {},
        "artifacts": {},
    }
    source_descriptors: dict[str, dict[str, Any]] = {}
    if status == "error":
        if not error.strip():
            raise ExportError("error cells require a non-empty diagnostic")
        if source.get("sources") != {} or source.get("artifacts") != {}:
            raise ExportError("error source manifests require empty source/artifact objects")
        bindings = source.get("metric_bindings")
        if bindings != {"stage1": {}, "downstream": {}}:
            raise ExportError("error source manifests must carry empty metric bindings")
    else:
        if error:
            raise ExportError("successful source manifests require an empty error string")
        documents, source_descriptors = _load_metric_sources(
            protocol,
            family,
            source,
            protocol_base=protocol_file.parent,
            source_base=source_file.parent,
        )
        if "run_receipt" not in documents:
            raise ExportError("successful export requires a sealed run_receipt source")
        _validate_run_binding(
            documents["run_receipt"],
            cell=cell,
            field_manifest_sha256=field_manifest["sha256"],
            field_semantic_digest=semantic_digest,
            factor_digest=factor_digest,
        )
        stage1, downstream = _extract_metrics(protocol, source, documents)
        _validate_metric_source_set(source, documents)
        raw_artifacts = source.get("artifacts")
        if not isinstance(raw_artifacts, dict) or set(raw_artifacts) != set(
            REQUIRED_CELL_ARTIFACTS
        ):
            raise ExportError("successful export must bind exactly the six BENCH-019 artifacts")
        artifacts = {}
        for name in REQUIRED_CELL_ARTIFACTS:
            _, artifacts[name] = _artifact_path(
                raw_artifacts[name], source_file.parent, label=f"cell artifact {name}"
            )
        row["stage1"] = stage1
        row["downstream"] = downstream
        row["artifacts"] = artifacts
    _write_new_json(output_file, row)
    output_descriptor = describe_artifact(output_file)
    receipt = {
        "schema": EXPORT_RECEIPT_SCHEMA,
        "diagnostic": bool(allow_review_protocol),
        "protocol": describe_artifact(protocol_file),
        "protocol_digest": identity,
        "source_manifest": describe_artifact(source_file),
        "source_artifacts": source_descriptors,
        "factor": factor_record,
        "downstream_factor_digest": factor_digest,
        "row": output_descriptor,
        "row_canonical_sha256": _digest(row),
    }
    try:
        _write_new_json(receipt_file, receipt)
    except Exception:
        output_file.unlink(missing_ok=True)
        raise
    return row


def validate_row(
    protocol: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    protocol_base: str | Path = ".",
    row_base: str | Path = ".",
    allow_review_protocol: bool = False,
) -> tuple[str, str, int, str, str]:
    """Validate one exported row against the local strict subset of BENCH-019."""
    protocol_identity(
        protocol,
        protocol_base=protocol_base,
        allow_review=allow_review_protocol,
    )
    value = _exact_mapping(dict(row), _ROW_KEYS, label="exported row")
    if value.get("schema") != ROW_SCHEMA:
        raise ExportError(f"row schema must be {ROW_SCHEMA}")
    cell = {name: value[name] for name in _CELL_KEYS}
    key = _cell_key(cell)
    expected_keys, families = _protocol_cells(protocol)
    if key not in expected_keys:
        raise ExportError(f"row cell {key} is not declared by the protocol")
    capture_id, family = families[(key[0], key[1])]
    if value.get("capture_id") != capture_id:
        raise ExportError("row capture differs from the frozen frame")
    manifest = family.get("field_manifest")
    semantics = family.get("semantics")
    if not isinstance(manifest, dict) or not isinstance(semantics, dict):
        raise ExportError("protocol family manifest/semantics are invalid")
    if value.get("field_manifest_sha256") != manifest.get("sha256"):
        raise ExportError("row field manifest digest differs")
    _artifact_path(
        manifest,
        Path(protocol_base).resolve(),
        label="protocol family field_manifest",
    )
    if value.get("field_semantic_digest") != semantics.get("semantic_digest"):
        raise ExportError("row field semantic digest differs")
    expected_factor = downstream_factor_digest(
        protocol,
        frame_id=key[0],
        seed=key[2],
        initializer=key[3],
        protocol_base=protocol_base,
        allow_review=allow_review_protocol,
    )
    if value.get("downstream_factor_digest") != expected_factor:
        raise ExportError("row downstream factor digest differs")
    status = value.get("status")
    if status == "error":
        if not isinstance(value.get("error"), str) or not value["error"].strip():
            raise ExportError("error row has no diagnostic")
        if value.get("stage1") or value.get("downstream") or value.get("artifacts"):
            raise ExportError("error row carries successful metrics or artifacts")
        return key
    if status != "ok" or value.get("error") != "":
        raise ExportError("successful row has invalid status/error")
    for name, protocol_name in (("stage1", "predictors"), ("downstream", "responses")):
        metrics = value.get(name)
        expected_names = _metric_names(protocol, protocol_name)
        if not isinstance(metrics, dict) or set(metrics) != set(expected_names):
            raise ExportError(f"row {name} metrics differ from the frozen protocol")
        for metric_name in expected_names:
            _finite(metrics[metric_name], label=f"row {name}.{metric_name}")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(REQUIRED_CELL_ARTIFACTS):
        raise ExportError("successful row does not bind the six required artifacts")
    base = Path(row_base).resolve()
    for name in REQUIRED_CELL_ARTIFACTS:
        _artifact_path(artifacts[name], base, label=f"row artifact {name}")
    return key


def _verified_receipt_artifact(record: object, *, label: str) -> dict[str, Any]:
    _, descriptor = _artifact_path(record, Path("/"), label=label)
    if record != descriptor:
        raise ExportError(f"{label} must use its resolved absolute path")
    return descriptor


def _replay_receipt_source(
    protocol: Mapping[str, Any],
    *,
    protocol_file: Path,
    protocol_digest: str,
    row: Mapping[str, Any],
    receipt: Mapping[str, Any],
    factor_digest: str,
) -> None:
    """Reproduce a row from the receipt-bound source manifest during assembly."""
    source_descriptor = _verified_receipt_artifact(
        receipt.get("source_manifest"), label="export receipt source manifest"
    )
    source_file = Path(source_descriptor["path"])
    source = load_json_object(source_file, label="receipt-bound source manifest")
    _exact_mapping(source, _SOURCE_KEYS, label="receipt-bound source manifest")
    if source.get("schema") != SOURCE_SCHEMA:
        raise ExportError(f"receipt source manifest schema must be {SOURCE_SCHEMA}")
    if source.get("protocol_digest") != protocol_digest:
        raise ExportError("receipt source manifest binds a different protocol digest")
    cell = _exact_mapping(source.get("cell"), _CELL_KEYS, label="receipt source cell")
    if cell != {name: row[name] for name in _CELL_KEYS}:
        raise ExportError("receipt source cell differs from the assembled row")
    if source.get("status") != row.get("status") or source.get("error") != row.get("error"):
        raise ExportError("receipt source status/error differs from the assembled row")

    _, families = _protocol_cells(protocol)
    capture_id, family = families[(row["frame_id"], row["family_id"])]
    if cell.get("capture_id") != capture_id:
        raise ExportError("receipt source capture differs from the frozen frame")
    source_artifacts = receipt.get("source_artifacts")
    if not isinstance(source_artifacts, dict):
        raise ExportError("export receipt source_artifacts must be an object")
    if row.get("status") == "error":
        if (
            source.get("sources") != {}
            or source.get("artifacts") != {}
            or source.get("metric_bindings") != {"stage1": {}, "downstream": {}}
            or source_artifacts != {}
        ):
            raise ExportError("error receipt/source chain carries a successful payload")
        return

    documents, reproduced_descriptors = _load_metric_sources(
        protocol,
        family,
        source,
        protocol_base=protocol_file.parent,
        source_base=source_file.parent,
    )
    if source_artifacts != reproduced_descriptors:
        raise ExportError("export receipt source_artifacts differ from its sealed source manifest")
    if "run_receipt" not in documents:
        raise ExportError("receipt source manifest has no sealed run_receipt")
    _validate_run_binding(
        documents["run_receipt"],
        cell=cell,
        field_manifest_sha256=row["field_manifest_sha256"],
        field_semantic_digest=row["field_semantic_digest"],
        factor_digest=factor_digest,
    )
    stage1, downstream = _extract_metrics(protocol, source, documents)
    _validate_metric_source_set(source, documents)
    if stage1 != row.get("stage1") or downstream != row.get("downstream"):
        raise ExportError("receipt source metrics do not reproduce the assembled row")

    raw_artifacts = source.get("artifacts")
    if not isinstance(raw_artifacts, dict) or set(raw_artifacts) != set(REQUIRED_CELL_ARTIFACTS):
        raise ExportError("receipt source does not bind exactly the six cell artifacts")
    reproduced_artifacts: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_CELL_ARTIFACTS:
        _, reproduced_artifacts[name] = _artifact_path(
            raw_artifacts[name], source_file.parent, label=f"receipt source cell artifact {name}"
        )
    if reproduced_artifacts != row.get("artifacts"):
        raise ExportError("receipt source cell artifacts do not reproduce the assembled row")


def _validate_export_receipt(
    protocol: Mapping[str, Any],
    *,
    protocol_file: Path,
    protocol_digest: str,
    row: Mapping[str, Any],
    row_file: Path,
    receipt_file: Path,
    allow_review_protocol: bool,
) -> dict[str, Any]:
    receipt = load_json_object(receipt_file, label=f"export receipt {receipt_file}")
    _exact_mapping(receipt, _EXPORT_RECEIPT_KEYS, label="export receipt")
    if receipt.get("schema") != EXPORT_RECEIPT_SCHEMA:
        raise ExportError(f"export receipt schema must be {EXPORT_RECEIPT_SCHEMA}")
    if receipt.get("diagnostic") is not bool(allow_review_protocol):
        raise ExportError("export receipt diagnostic state differs from assembly mode")
    if receipt.get("protocol_digest") != protocol_digest:
        raise ExportError("export receipt binds a different protocol digest")
    if _verified_receipt_artifact(
        receipt.get("protocol"), label="export receipt protocol"
    ) != describe_artifact(protocol_file):
        raise ExportError("export receipt binds a different protocol file")
    if _verified_receipt_artifact(
        receipt.get("row"), label="export receipt row"
    ) != describe_artifact(row_file):
        raise ExportError("export receipt binds a different row file")
    if receipt.get("row_canonical_sha256") != _digest(row):
        raise ExportError("export receipt canonical row digest differs")
    source_artifacts = receipt.get("source_artifacts")
    if not isinstance(source_artifacts, dict):
        raise ExportError("export receipt source_artifacts must be an object")
    for name, descriptor in source_artifacts.items():
        _identifier(name, label="export receipt source artifact name")
        _verified_receipt_artifact(
            descriptor,
            label=f"export receipt source artifact {name}",
        )
    key = _cell_key(row)
    expected_factor = downstream_factor_record(
        protocol,
        frame_id=key[0],
        seed=key[2],
        initializer=key[3],
        protocol_base=protocol_file.parent,
        allow_review=allow_review_protocol,
    )
    if receipt.get("factor") != expected_factor:
        raise ExportError("export receipt downstream factor record differs")
    if receipt.get("downstream_factor_digest") != _digest(expected_factor):
        raise ExportError("export receipt downstream factor digest differs")
    _replay_receipt_source(
        protocol,
        protocol_file=protocol_file,
        protocol_digest=protocol_digest,
        row=row,
        receipt=receipt,
        factor_digest=_digest(expected_factor),
    )
    return receipt


def assemble_rows(
    protocol_path: str | Path,
    row_paths: Sequence[str | Path],
    output_path: str | Path,
    receipt_path: str | Path,
    *,
    export_receipt_paths: Sequence[str | Path] | None = None,
    allow_review_protocol: bool = False,
    allow_incomplete: bool = False,
) -> list[dict[str, Any]]:
    """Order exact per-cell exports into one append-only JSONL source for StructSplat."""
    protocol_file = Path(protocol_path).resolve(strict=True)
    protocol = load_json_object(protocol_file, label="BENCH-019 protocol")
    identity = protocol_identity(
        protocol,
        protocol_base=protocol_file.parent,
        allow_review=allow_review_protocol,
    )
    expected, _ = _protocol_cells(protocol)
    if export_receipt_paths is None or len(export_receipt_paths) != len(row_paths):
        raise ExportError("assembly requires exactly one export receipt per cell row")
    indexed: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
    source_records: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
    for raw_path, raw_receipt_path in zip(row_paths, export_receipt_paths, strict=True):
        path = Path(raw_path).resolve(strict=True)
        export_receipt_path = Path(raw_receipt_path).resolve(strict=True)
        row = load_json_object(path, label=f"cell row {path}")
        key = validate_row(
            protocol,
            row,
            protocol_base=protocol_file.parent,
            row_base=path.parent,
            allow_review_protocol=allow_review_protocol,
        )
        if key in indexed:
            raise ExportError(f"duplicate exported cell {key}")
        _validate_export_receipt(
            protocol,
            protocol_file=protocol_file,
            protocol_digest=identity,
            row=row,
            row_file=path,
            receipt_file=export_receipt_path,
            allow_review_protocol=allow_review_protocol,
        )
        indexed[key] = row
        source_records[key] = {
            "row": describe_artifact(path),
            "export_receipt": describe_artifact(export_receipt_path),
        }
    missing = [key for key in expected if key not in indexed]
    if missing and not allow_incomplete:
        raise ExportError(f"assembly is missing {len(missing)} frozen cells")
    ordered = [indexed[key] for key in expected if key in indexed]
    output_file = Path(output_path)
    receipt_file = Path(receipt_path)
    if output_file.exists() or receipt_file.exists():
        raise ExportError("assembly output and receipt must both be new files")
    _write_new_jsonl(output_file, ordered)
    receipt = {
        "schema": ASSEMBLY_RECEIPT_SCHEMA,
        "diagnostic": bool(allow_review_protocol or allow_incomplete),
        "protocol": describe_artifact(protocol_file),
        "protocol_digest": identity,
        "row_sources": [source_records[key] for key in expected if key in source_records],
        "rows": describe_artifact(output_file),
        "expected_cell_count": len(expected),
        "exported_cell_count": len(ordered),
        "missing_cells": [list(key) for key in missing],
        "status_counts": {
            "ok": sum(row["status"] == "ok" for row in ordered),
            "error": sum(row["status"] == "error" for row in ordered),
        },
    }
    try:
        _write_new_json(receipt_file, receipt)
    except Exception:
        output_file.unlink(missing_ok=True)
        raise
    return ordered


__all__ = [
    "ASSEMBLY_RECEIPT_SCHEMA",
    "EXPORT_RECEIPT_SCHEMA",
    "ExportError",
    "FACTOR_SCHEMA",
    "PROTOCOL_SCHEMA",
    "REQUIRED_CELL_ARTIFACTS",
    "ROW_SCHEMA",
    "RUN_BINDING_SCHEMA",
    "SOURCE_SCHEMA",
    "assemble_rows",
    "canonical_json",
    "describe_artifact",
    "downstream_factor_digest",
    "downstream_factor_record",
    "export_cell",
    "load_json_object",
    "protocol_identity",
    "sha256_file",
    "validate_row",
]
