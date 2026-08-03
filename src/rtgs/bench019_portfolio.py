"""Source-bound capture portfolio contract for StructSplat BENCH-019.

The portfolio is deliberately pre-outcome.  It binds acquired source material, records the
development/confirmation split, and makes missing Stage-1 production visible.  It is not a
StructSplat protocol and cannot authorize opening confirmation outcomes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rtgs.bench019 import ExportError, canonical_json, describe_artifact

PORTFOLIO_SCHEMA = "rtgs.structsplat_bench019.capture_portfolio.v1"
REQUIRED_FIELD_FAMILIES = (
    "gaussianimage_additive",
    "structsplat_normalized_no_boundary",
    "structsplat_normalized_mask_contained",
)
TUM_URL_BASE = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1"
PINNED_TUM_ARCHIVES = {
    "tum_fr1_xyz": (
        "rgbd_dataset_freiburg1_xyz.tgz",
        448204271,
        "a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98",
    ),
    "tum_fr1_rpy": (
        "rgbd_dataset_freiburg1_rpy.tgz",
        410268381,
        "78103722a25873dbbb4de027eaa8c810a6382100691f02b6cf95c3adc91c4ac1",
    ),
    "tum_fr1_desk": (
        "rgbd_dataset_freiburg1_desk.tgz",
        344011403,
        "e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c",
    ),
    "tum_fr1_desk2": (
        "rgbd_dataset_freiburg1_desk2.tgz",
        349445005,
        "a569e4cb453a3cd9285bc985fcb109e65f055c75b33a4b155acd9a68d96b77d2",
    ),
}

_TOP_KEYS = frozenset(
    {
        "schema",
        "state",
        "inventory_date",
        "outcome_access",
        "development_capture_ids",
        "confirmation_capture_ids",
        "captures",
        "gates",
    }
)
_CAPTURE_KEYS = frozenset(
    {
        "id",
        "role",
        "source_kind",
        "origin",
        "frame_id",
        "frame_plan_state",
        "view_ids",
        "mask_policy_state",
        "source_artifacts",
        "source_digest",
        "field_families",
        "blockers",
    }
)
_SOURCE_KEYS = frozenset({"id", "artifact"})
_ARTIFACT_KEYS = frozenset({"path", "sha256", "bytes"})
_FAMILY_KEYS = frozenset(
    {
        "id",
        "provider",
        "equation",
        "blend_mode",
        "state",
        "observed_views",
        "required_views",
        "evidence",
    }
)
_GATE_KEYS = frozenset(
    {
        "source_groups_bound",
        "field_families_complete",
        "confirmation_outcomes_opened",
        "formal_protocol_frozen",
    }
)
_FAMILY_SEMANTICS = {
    "gaussianimage_additive": ("gaussianimage", "additive_sum", "additive"),
    "structsplat_normalized_no_boundary": (
        "structsplat",
        "normalized_weighted_sum",
        "normalized",
    ),
    "structsplat_normalized_mask_contained": (
        "structsplat",
        "normalized_weighted_sum",
        "normalized",
    ),
}
_FAMILY_STATES = {
    "complete_stage1_unbound",
    "incomplete_live_unbound",
    "not_produced",
}


def _exact(value: object, keys: frozenset[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = set(value) if isinstance(value, dict) else set()
        raise ExportError(
            f"{label} keys are not exact "
            f"(missing={sorted(keys - actual)}, extra={sorted(actual - keys)})"
        )
    return value


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or any(char.isspace() for char in value):
        raise ExportError(f"{label} must be a non-empty whitespace-free identifier")
    return value


def _string_list(value: object, *, label: str, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ExportError(f"{label} must be {'a' if allow_empty else 'a non-empty'} list")
    result = [_identifier(item, label=f"{label} item") for item in value]
    if len(result) != len(set(result)):
        raise ExportError(f"{label} contains duplicates")
    return result


def source_digest(sources: object) -> str:
    """Return the canonical digest of an ordered source-artifact inventory."""
    if not isinstance(sources, list):
        raise ExportError("source_artifacts must be a list")
    return hashlib.sha256(canonical_json(sources)).hexdigest()


def _validate_descriptor(
    value: object,
    *,
    label: str,
    verify_files: bool,
) -> dict[str, Any]:
    descriptor = _exact(value, _ARTIFACT_KEYS, label=label)
    if not isinstance(descriptor["path"], str) or not Path(descriptor["path"]).is_absolute():
        raise ExportError(f"{label}.path must be absolute")
    digest = descriptor["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ExportError(f"{label}.sha256 must be a lowercase SHA-256")
    size = descriptor["bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ExportError(f"{label}.bytes must be a non-negative integer")
    if verify_files and describe_artifact(descriptor["path"]) != descriptor:
        raise ExportError(f"{label} differs from the acquired source file")
    return descriptor


def validate_capture_portfolio(
    value: Mapping[str, Any],
    *,
    verify_files: bool = False,
) -> dict[str, int]:
    """Validate a pre-outcome six-group portfolio and optionally rehash every source file."""
    portfolio = _exact(dict(value), _TOP_KEYS, label="capture portfolio")
    if portfolio["schema"] != PORTFOLIO_SCHEMA:
        raise ExportError(f"portfolio schema must be {PORTFOLIO_SCHEMA}")
    if portfolio["state"] != "source_bound_preproduction":
        raise ExportError("portfolio state must remain source_bound_preproduction")
    if portfolio["outcome_access"] != "none":
        raise ExportError("the source portfolio may not record downstream outcome access")
    if not isinstance(portfolio["inventory_date"], str) or not portfolio["inventory_date"]:
        raise ExportError("inventory_date must be a non-empty string")

    development = _string_list(
        portfolio["development_capture_ids"], label="development_capture_ids"
    )
    confirmation = _string_list(
        portfolio["confirmation_capture_ids"], label="confirmation_capture_ids"
    )
    if len(development) != 3 or len(confirmation) != 3:
        raise ExportError(
            "the portfolio must bind exactly three development and three confirmation groups"
        )
    if set(development) & set(confirmation):
        raise ExportError("development and confirmation capture groups must be disjoint")

    captures = portfolio["captures"]
    if not isinstance(captures, list) or len(captures) != 6:
        raise ExportError("captures must contain the six declared groups")
    indexed: dict[str, dict[str, Any]] = {}
    source_file_count = 0
    for index, raw_capture in enumerate(captures):
        capture = _exact(raw_capture, _CAPTURE_KEYS, label=f"captures[{index}]")
        capture_id = _identifier(capture["id"], label=f"captures[{index}].id")
        if capture_id in indexed:
            raise ExportError(f"duplicate capture ID {capture_id}")
        role = capture["role"]
        if role not in {"development", "confirmation"}:
            raise ExportError(f"capture {capture_id} has invalid role")
        expected_ids = development if role == "development" else confirmation
        if capture_id not in expected_ids:
            raise ExportError(f"capture {capture_id} role differs from its declared split")
        for name in ("source_kind", "origin", "frame_id", "frame_plan_state", "mask_policy_state"):
            if not isinstance(capture[name], str) or not capture[name]:
                raise ExportError(f"capture {capture_id}.{name} must be a non-empty string")
        _string_list(capture["view_ids"], label=f"capture {capture_id}.view_ids", allow_empty=True)

        sources = capture["source_artifacts"]
        if not isinstance(sources, list) or not sources:
            raise ExportError(f"capture {capture_id} has no source artifacts")
        source_ids: set[str] = set()
        for source_index, raw_source in enumerate(sources):
            source = _exact(
                raw_source,
                _SOURCE_KEYS,
                label=f"capture {capture_id}.source_artifacts[{source_index}]",
            )
            source_id = _identifier(source["id"], label=f"capture {capture_id} source id")
            if source_id in source_ids:
                raise ExportError(f"capture {capture_id} has duplicate source artifact {source_id}")
            source_ids.add(source_id)
            _validate_descriptor(
                source["artifact"],
                label=f"capture {capture_id} source {source_id}",
                verify_files=verify_files,
            )
        if capture_id in PINNED_TUM_ARCHIVES:
            archive_name, expected_bytes, expected_sha256 = PINNED_TUM_ARCHIVES[capture_id]
            if capture["source_kind"] != "tum_rgbd_archive":
                raise ExportError(f"capture {capture_id} must remain a TUM RGB-D archive")
            if capture["origin"] != f"{TUM_URL_BASE}/{archive_name}":
                raise ExportError(f"capture {capture_id} origin differs from the official pin")
            if len(sources) != 1 or sources[0]["id"] != "official_archive":
                raise ExportError(
                    f"capture {capture_id} must bind exactly its official_archive source"
                )
            archive = sources[0]["artifact"]
            if archive["bytes"] != expected_bytes or archive["sha256"] != expected_sha256:
                raise ExportError(f"capture {capture_id} differs from the pinned official archive")
        if capture["source_digest"] != source_digest(sources):
            raise ExportError(f"capture {capture_id} source digest differs")
        source_file_count += len(sources)

        families = capture["field_families"]
        if not isinstance(families, list) or len(families) != len(REQUIRED_FIELD_FAMILIES):
            raise ExportError(f"capture {capture_id} must record all required field families")
        family_ids: list[str] = []
        for family_index, raw_family in enumerate(families):
            family = _exact(
                raw_family,
                _FAMILY_KEYS,
                label=f"capture {capture_id}.field_families[{family_index}]",
            )
            family_id = _identifier(family["id"], label=f"capture {capture_id} family id")
            family_ids.append(family_id)
            if family_id not in _FAMILY_SEMANTICS:
                raise ExportError(f"capture {capture_id} has unknown family {family_id}")
            expected_semantics = _FAMILY_SEMANTICS[family_id]
            actual_semantics = (
                family["provider"],
                family["equation"],
                family["blend_mode"],
            )
            if actual_semantics != expected_semantics:
                raise ExportError(f"capture {capture_id} family {family_id} changes semantics")
            if family["state"] not in _FAMILY_STATES:
                raise ExportError(f"capture {capture_id} family {family_id} has invalid state")
            for name in ("observed_views", "required_views"):
                count = family[name]
                if count is not None and (
                    isinstance(count, bool) or not isinstance(count, int) or count < 0
                ):
                    raise ExportError(f"capture {capture_id} family {family_id}.{name} is invalid")
            evidence = family["evidence"]
            if evidence is not None:
                _validate_descriptor(
                    evidence,
                    label=f"capture {capture_id} family {family_id} evidence",
                    verify_files=verify_files,
                )
            state = family["state"]
            observed = family["observed_views"]
            required = family["required_views"]
            if state == "complete_stage1_unbound" and (
                evidence is None or observed is None or required is None or observed != required
            ):
                raise ExportError(
                    f"capture {capture_id} family {family_id} is not evidence-complete"
                )
            if state == "incomplete_live_unbound" and evidence is not None:
                raise ExportError(
                    f"capture {capture_id} family {family_id} incomplete state carries a seal"
                )
            if state == "not_produced" and (observed != 0 or evidence is not None):
                raise ExportError(
                    f"capture {capture_id} family {family_id} not_produced state is inconsistent"
                )
            if role == "confirmation" and (state != "not_produced" or evidence is not None):
                raise ExportError("confirmation field outcomes must remain unopened and unproduced")
        if family_ids != list(REQUIRED_FIELD_FAMILIES):
            raise ExportError(f"capture {capture_id} field-family order differs from the contract")
        blockers = capture["blockers"]
        if (
            not isinstance(blockers, list)
            or not blockers
            or any(not isinstance(blocker, str) or not blocker for blocker in blockers)
        ):
            raise ExportError(f"capture {capture_id} must expose its production blockers")
        indexed[capture_id] = capture

    if set(indexed) != set(development) | set(confirmation):
        raise ExportError("capture records differ from the declared split IDs")
    gates = _exact(portfolio["gates"], _GATE_KEYS, label="portfolio gates")
    expected_gates = {
        "source_groups_bound": True,
        "field_families_complete": False,
        "confirmation_outcomes_opened": False,
        "formal_protocol_frozen": False,
    }
    if gates != expected_gates:
        raise ExportError("preproduction portfolio gates must remain closed except source binding")
    return {
        "development_capture_groups": len(development),
        "confirmation_capture_groups": len(confirmation),
        "source_files": source_file_count,
    }


__all__ = [
    "PINNED_TUM_ARCHIVES",
    "PORTFOLIO_SCHEMA",
    "REQUIRED_FIELD_FAMILIES",
    "TUM_URL_BASE",
    "source_digest",
    "validate_capture_portfolio",
]
