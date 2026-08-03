#!/usr/bin/env python3
"""Create or verify the source-only six-group StructSplat BENCH-019 portfolio."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

from rtgs.bench019 import ExportError, describe_artifact, load_json_object
from rtgs.bench019_portfolio import (
    PORTFOLIO_SCHEMA,
    REQUIRED_FIELD_FAMILIES,
    source_digest,
    validate_capture_portfolio,
)

STAGE_VIEWS = (
    "C0001",
    "C0004",
    "C0005",
    "C0006",
    "C0008",
    "C0009",
    "C0012",
    "C0014",
    "C0018",
    "C0019",
    "C0020",
    "C0021",
    "C0022",
    "C0025",
    "C0026",
    "C0028",
    "C0029",
    "C0030",
    "C0031",
    "C0034",
    "C0037",
    "C0039",
    "C1000",
    "C1001",
    "C1002",
    "C1004",
)

_TUM_ARCHIVES = {
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
_TUM_URL_BASE = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1"


def _source(source_id: str, path: Path) -> dict:
    return {"id": source_id, "artifact": describe_artifact(path)}


def _family(
    family_id: str,
    *,
    state: str,
    observed: int | None,
    required: int | None,
    evidence: Path | None = None,
) -> dict:
    if family_id == "gaussianimage_additive":
        provider, equation, blend = "gaussianimage", "additive_sum", "additive"
    else:
        provider, equation, blend = "structsplat", "normalized_weighted_sum", "normalized"
    return {
        "id": family_id,
        "provider": provider,
        "equation": equation,
        "blend_mode": blend,
        "state": state,
        "observed_views": observed,
        "required_views": required,
        "evidence": describe_artifact(evidence) if evidence is not None else None,
    }


def _unproduced_families() -> list[dict]:
    return [
        _family(family_id, state="not_produced", observed=0, required=None)
        for family_id in REQUIRED_FIELD_FAMILIES
    ]


def _janelle_capture(
    *,
    capture_id: str,
    role: str,
    root: Path,
    frame_id: str,
    filename_by_view: dict[str, str],
    mask_filename_by_view: dict[str, str] | None,
    field_families: list[dict],
    blockers: list[str],
) -> dict:
    views = sorted(filename_by_view)
    sources = [_source("calibration", root / "calibration_dome.json")]
    sources.extend(
        _source(f"rgb_{view_id}", root / frame_id / "rgb" / filename_by_view[view_id])
        for view_id in views
    )
    if mask_filename_by_view is not None:
        if set(mask_filename_by_view) != set(views):
            raise ExportError(f"capture {capture_id} RGB/mask view sets differ")
        sources.extend(
            _source(f"mask_{view_id}", root / frame_id / "mask" / mask_filename_by_view[view_id])
            for view_id in views
        )
    return {
        "id": capture_id,
        "role": role,
        "source_kind": "calibrated_multiview",
        "origin": str(root.resolve(strict=True)),
        "frame_id": frame_id,
        "frame_plan_state": "selected_source_frame",
        "view_ids": views,
        "mask_policy_state": (
            "source_binary_masks_bound" if mask_filename_by_view is not None else "missing_unfrozen"
        ),
        "source_artifacts": sources,
        "source_digest": source_digest(sources),
        "field_families": field_families,
        "blockers": blockers,
    }


def _tum_capture(capture_id: str, role: str, tum_root: Path) -> dict:
    archive_name, expected_bytes, expected_sha256 = _TUM_ARCHIVES[capture_id]
    sources = [_source("official_archive", tum_root / archive_name)]
    descriptor = sources[0]["artifact"]
    if descriptor["bytes"] != expected_bytes or descriptor["sha256"] != expected_sha256:
        raise ExportError(f"{capture_id} archive differs from the pinned official acquisition")
    return {
        "id": capture_id,
        "role": role,
        "source_kind": "tum_rgbd_archive",
        "origin": f"{_TUM_URL_BASE}/{archive_name}",
        "frame_id": "pending_keyframe_selection",
        "frame_plan_state": "source_bound_adapter_pending",
        "view_ids": [],
        "mask_policy_state": "registered_depth_validity_unfrozen",
        "source_artifacts": sources,
        "source_digest": source_digest(sources),
        "field_families": _unproduced_families(),
        "blockers": [
            "keyframe, camera, train/heldout, and mask policy adapter is not frozen",
            "the three matched Stage-1 field families have not been produced",
        ],
    }


def build_portfolio(stage_root: Path, karate_root: Path, tum_root: Path) -> dict:
    """Build the fixed source portfolio without reading any downstream outcome."""
    stage_frame = stage_root / "frame_00008"
    stage_counts = {}
    for family_id, directory in (
        ("gaussianimage_additive", "gaussians2d_gaussianimage_fullres"),
        ("structsplat_normalized_no_boundary", "gaussians2d_structsplat_no_boundary_fullres"),
        (
            "structsplat_normalized_mask_contained",
            "gaussians2d_structsplat_mask_contained_fullres",
        ),
    ):
        stage_counts[family_id] = len(list((stage_frame / directory).glob("C*.rtgsv")))
    gaussian_manifest = stage_frame / "gaussians2d_gaussianimage_fullres/production_manifest.json"
    no_boundary_manifest = (
        stage_frame / "gaussians2d_structsplat_no_boundary_fullres/production_manifest.json"
    )
    mask_contained_manifest = (
        stage_frame / "gaussians2d_structsplat_mask_contained_fullres/production_manifest.json"
    )
    mask_contained_complete = (
        stage_counts["structsplat_normalized_mask_contained"] == len(STAGE_VIEWS)
        and mask_contained_manifest.is_file()
    )
    stage_families = [
        _family(
            "gaussianimage_additive",
            state="complete_stage1_unbound",
            observed=stage_counts["gaussianimage_additive"],
            required=len(STAGE_VIEWS),
            evidence=gaussian_manifest,
        ),
        _family(
            "structsplat_normalized_no_boundary",
            state="complete_stage1_unbound",
            observed=stage_counts["structsplat_normalized_no_boundary"],
            required=len(STAGE_VIEWS),
            evidence=no_boundary_manifest,
        ),
        _family(
            "structsplat_normalized_mask_contained",
            state=(
                "complete_stage1_unbound" if mask_contained_complete else "incomplete_live_unbound"
            ),
            observed=stage_counts["structsplat_normalized_mask_contained"],
            required=len(STAGE_VIEWS),
            evidence=mask_contained_manifest if mask_contained_complete else None,
        ),
    ]
    stage_rgb = {view: f"{view}.jpg" for view in STAGE_VIEWS}
    stage_masks = {view: f"mask_{view}.png" for view in STAGE_VIEWS}
    stage_capture = _janelle_capture(
        capture_id="janelle_stage_fabric",
        role="development",
        root=stage_root,
        frame_id="frame_00008",
        filename_by_view=stage_rgb,
        mask_filename_by_view=stage_masks,
        field_families=stage_families,
        blockers=[
            "the mask-contained normalized family is an incomplete live production "
            "and is not bound",
            "no BENCH-019 realtime-gs cells or formal protocol are frozen",
        ],
    )

    karate_files = sorted((karate_root / "frame_00005/rgb").glob("rgb_*.jpeg"))
    if len(karate_files) != 30:
        raise ExportError("janelle_karate frame_00005 must contain exactly 30 source RGB views")
    karate_rgb = {f"C{int(path.stem.removeprefix('rgb_')):04d}": path.name for path in karate_files}
    karate_capture = _janelle_capture(
        capture_id="janelle_karate",
        role="confirmation",
        root=karate_root,
        frame_id="frame_00005",
        filename_by_view=karate_rgb,
        mask_filename_by_view=None,
        field_families=_unproduced_families(),
        blockers=[
            "mask policy and train/heldout camera split are not frozen",
            "the three matched Stage-1 field families have not been produced",
            "confirmation outcomes remain unopened",
        ],
    )

    development = ["janelle_stage_fabric", "tum_fr1_xyz", "tum_fr1_rpy"]
    confirmation = ["janelle_karate", "tum_fr1_desk", "tum_fr1_desk2"]
    portfolio = {
        "schema": PORTFOLIO_SCHEMA,
        "state": "source_bound_preproduction",
        "inventory_date": "2026-08-03",
        "outcome_access": "none",
        "development_capture_ids": development,
        "confirmation_capture_ids": confirmation,
        "captures": [
            stage_capture,
            _tum_capture("tum_fr1_xyz", "development", tum_root),
            _tum_capture("tum_fr1_rpy", "development", tum_root),
            karate_capture,
            _tum_capture("tum_fr1_desk", "confirmation", tum_root),
            _tum_capture("tum_fr1_desk2", "confirmation", tum_root),
        ],
        "gates": {
            "source_groups_bound": True,
            "field_families_complete": False,
            "confirmation_outcomes_opened": False,
            "formal_protocol_frozen": False,
        },
    }
    validate_capture_portfolio(portfolio, verify_files=True)
    return portfolio


def _write_new(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--stage-root", type=Path, required=True)
    create.add_argument("--karate-root", type=Path, required=True)
    create.add_argument("--tum-root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--portfolio", type=Path, required=True)
    verify.add_argument("--verify-files", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            portfolio = build_portfolio(args.stage_root, args.karate_root, args.tum_root)
            _write_new(args.output, portfolio)
            summary = validate_capture_portfolio(portfolio, verify_files=True)
        else:
            portfolio = load_json_object(args.portfolio, label="BENCH-019 capture portfolio")
            summary = validate_capture_portfolio(portfolio, verify_files=args.verify_files)
        print(json.dumps(summary, sort_keys=True))
        return 0
    except (ExportError, FileExistsError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
