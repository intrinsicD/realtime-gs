#!/usr/bin/env python3
"""Recover the completed masked bundle after a false-positive manifest-value check.

This verifier never fits, saves, or overwrites a teacher or ReconstructionInputs bundle. It
strict-loads the plan-bound outputs, verifies their hashes and compact semantics, performs the
intended structured forbidden-field check, and writes one exclusive recovery result.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
from pathlib import Path
from typing import Any

from compact_masked_bundle_acquisition import (
    HELDOUT_VIEW,
    REQUIRED_TEACHER_MEMBERS,
    ROOT,
    SEEDS,
    VIEWS,
    canonical_hash,
    observation_summary,
    sha256_file,
    structsplat_source_binding,
    teacher_members,
    write_json_exclusive,
)
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.reconstruction_inputs import ReconstructionInputs

DEFAULT_OUT = ROOT / "runs/compact_masked_bundle_640_20260717"
FORBIDDEN_KEYS = frozenset(
    {
        "image",
        "images",
        "image_path",
        "image_paths",
        "mask",
        "masks",
        "mask_path",
        "mask_paths",
        "rgb",
        "rgb_path",
        "rgb_paths",
        "source_image",
        "source_path",
        "source_rgb",
    }
)


def nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {str(key) for key in value} | set().union(
            *(nested_keys(item) for item in value.values())
        )
    if isinstance(value, list):
        return set().union(*(nested_keys(item) for item in value))
    return set()


def verify_plan_sources(plan: dict[str, Any]) -> dict[str, Any]:
    recorded = plan["repository"]
    current_hashes = {relative: sha256_file(ROOT / relative) for relative in recorded["files"]}
    if current_hashes != recorded["files"]:
        changed = sorted(
            relative
            for relative, digest in current_hashes.items()
            if digest != recorded["files"][relative]
        )
        raise RuntimeError(f"plan-bound realtime-gs sources changed: {changed}")
    if canonical_hash(current_hashes) != recorded["aggregate"]:
        raise RuntimeError("plan-bound source aggregate does not recompute")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if revision != recorded["git_revision"]:
        raise RuntimeError("git revision changed after acquisition")
    harness = "benchmarks/compact_masked_bundle_acquisition.py"
    return {
        "git_revision": revision,
        "aggregate": recorded["aggregate"],
        "original_harness_path": harness,
        "original_harness_sha256": current_hashes[harness],
        "original_harness_plan_sha256": recorded["files"][harness],
        "all_plan_bound_sources_match": True,
    }


def verify(out: Path) -> dict[str, Any]:
    plan_path = out / "plan.json"
    acquisition_path = out / "teacher_acquisition.json"
    failure_path = out / "failure.json"
    bundle_path = out / "reconstruction_inputs"
    recovery_path = out / "recovery_result.json"
    if recovery_path.exists():
        raise FileExistsError(f"refusing to overwrite recovery result: {recovery_path}")
    for path in (plan_path, acquisition_path, failure_path, bundle_path):
        if not path.exists():
            raise FileNotFoundError(f"required recovery input is missing: {path}")

    plan_sha256 = sha256_file(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["training_views"] != list(VIEWS) or plan["seeds"] != list(SEEDS):
        raise RuntimeError("plan view/seed order differs from frozen acquisition")
    if plan["heldout_view_excluded"] != HELDOUT_VIEW:
        raise RuntimeError("plan does not explicitly exclude the held-out view")
    source_verification = verify_plan_sources(plan)
    current_external = structsplat_source_binding()
    if current_external != plan["external_structsplat"]:
        raise RuntimeError("StructSplat source changed after acquisition")

    aggregate = json.loads(acquisition_path.read_text(encoding="utf-8"))
    if aggregate.get("status") != "PASS" or aggregate["plan_sha256"] != plan_sha256:
        raise RuntimeError("aggregate acquisition record is not a PASS for this plan")
    records: list[dict[str, Any]] = []
    acquisition_fields: list[GaussianObservationField] = []
    extension_hashes: set[str] = set()
    for index, (view_id, seed) in enumerate(zip(VIEWS, SEEDS, strict=True)):
        record_path = out / "acquisition" / f"{index:04d}.json"
        teacher_path = out / "acquisition" / f"{index:04d}.teacher.npz"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if not (
            record.get("status") == "PASS"
            and record["plan_sha256"] == plan_sha256
            and record["index"] == index
            and record["view_id"] == view_id
            and record["seed"] == seed
        ):
            raise RuntimeError(f"invalid acquisition record for {view_id}")
        if record != aggregate["views"][index]:
            raise RuntimeError(f"aggregate acquisition record differs for {view_id}")
        if sha256_file(teacher_path) != record["teacher_sha256"]:
            raise RuntimeError(f"acquisition archive hash changed for {view_id}")
        if frozenset(teacher_members(teacher_path)) != REQUIRED_TEACHER_MEMBERS:
            raise RuntimeError(f"acquisition archive members changed for {view_id}")
        field = GaussianObservationField.load_npz(teacher_path, strict=True)
        if observation_summary(field) != record["teacher"]:
            raise RuntimeError(f"acquisition archive semantics changed for {view_id}")
        if field.n_init != 640 or field.n != 640 or field.view_id != view_id:
            raise RuntimeError(f"acquisition archive cardinality/view changed for {view_id}")
        extension_hashes.add(canonical_hash(record["loaded_cuda_extension"]))
        records.append(record)
        acquisition_fields.append(field)
    if len(extension_hashes) != 1:
        raise RuntimeError("workers did not use one identical CUDA extension")

    loaded = ReconstructionInputs.load(bundle_path, device="cpu", strict=True)
    if loaded.view_names != list(VIEWS):
        raise RuntimeError("strict bundle view order differs")
    if loaded.n_init_2d != [640] * len(VIEWS) or loaded.n_opt_2d != [640] * len(VIEWS):
        raise RuntimeError("strict bundle cardinalities differ")
    if any(
        value is not None for value in (loaded.points, loaded.point_visibility, loaded.bounds_hint)
    ):
        raise RuntimeError("strict bundle unexpectedly contains geometry")
    if [observation_summary(field) for field in loaded.observations] != [
        observation_summary(field) for field in acquisition_fields
    ]:
        raise RuntimeError("bundle teacher semantics differ from acquisition archives")

    manifest_path = bundle_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema"] != "rtgs.reconstruction_inputs.v1":
        raise RuntimeError("unexpected reconstruction bundle schema")
    if manifest["name"] != "compact_masked_bundle_640_20260717":
        raise RuntimeError("unexpected reconstruction bundle name")
    if manifest["geometry"] is not None:
        raise RuntimeError("bundle manifest contains geometry")
    if any(record["view_id"] == HELDOUT_VIEW for record in manifest["views"]):
        raise RuntimeError("bundle manifest contains held-out view")
    forbidden_present = sorted(FORBIDDEN_KEYS & {key.lower() for key in nested_keys(manifest)})
    if forbidden_present:
        raise RuntimeError(f"bundle manifest contains forbidden field keys: {forbidden_present}")
    expected_paths = [f"teachers/{index:04d}.teacher.npz" for index in range(len(VIEWS))]
    if [record["teacher"] for record in manifest["views"]] != expected_paths:
        raise RuntimeError("bundle teacher paths differ from strict schema order")

    bundle_files = sorted(path for path in bundle_path.rglob("*") if path.is_file())
    bundle_hashes = {
        path.relative_to(bundle_path).as_posix(): sha256_file(path) for path in bundle_files
    }
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if not (
        failure.get("status") == "FAIL"
        and failure.get("error") == "bundle manifest contains a forbidden raster/path field"
    ):
        raise RuntimeError("original failure is not the expected false-positive postcondition")

    recovery_source = Path(__file__).resolve()
    result = {
        "artifact_type": "compact_masked_bundle_acquisition_recovery_v1",
        "status": "PASS",
        "decision_bearing": False,
        "recovery_scope": (
            "post-acquisition verification only; no fit, teacher write, bundle write, lift, or "
            "viewer action"
        ),
        "original_attempt_status": "FAIL_POSTCONDITION_FALSE_POSITIVE",
        "original_failure_sha256": sha256_file(failure_path),
        "original_failure": failure["error"],
        "repair": (
            "replace manifest-wide substring search with structured forbidden-key inspection; "
            "the required bundle name may contain 'masked'"
        ),
        "plan_sha256": plan_sha256,
        "source_verification": source_verification,
        "recovery_verifier": {
            "path": recovery_source.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(recovery_source),
        },
        "external_structsplat": current_external,
        "teacher_acquisition_sha256": sha256_file(acquisition_path),
        "training_views": list(VIEWS),
        "seeds": list(SEEDS),
        "heldout_view_excluded": HELDOUT_VIEW,
        "n_views": loaded.n_views,
        "n_init_2d": loaded.n_init_2d,
        "n_opt_2d": loaded.n_opt_2d,
        "total_components": sum(loaded.n_opt_2d),
        "bundle": {
            "path": bundle_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "semantic_digest": manifest["semantic_digest"],
            "calibration_digest": manifest["calibration_digest"],
            "files": bundle_hashes,
            "aggregate_sha256": canonical_hash(bundle_hashes),
            "archive_stats": dataclasses.asdict(loaded.archive_stats),
            "contains_geometry": False,
            "contains_dense_rgb_mask_or_source_path_fields": False,
            "structured_manifest_keys": sorted(nested_keys(manifest)),
        },
        "views": [
            {
                "view_id": record["view_id"],
                "seed": record["seed"],
                "fit_window": record["fit_window"],
                "teacher_sha256": record["teacher_sha256"],
                "foreground_psnr_db": record["metrics"]["foreground_rgb"]["psnr_db"],
                "foreground_hole_fraction": record["metrics"]["foreground_hole_fraction"],
                "archive_query_cuda_raster_max_abs_error": record["metrics"][
                    "archive_query_cuda_raster_parity"
                ]["max_abs_error"],
            }
            for record in records
        ],
        "mean_foreground_psnr_db": sum(
            record["metrics"]["foreground_rgb"]["psnr_db"] for record in records
        )
        / len(records),
        "viewer": {
            "status": "NOT_APPLICABLE_STAGE1_ONLY",
            "reason": "the requested acquisition stops at compact 2D teachers, not a 3D PLY",
        },
    }
    write_json_exclusive(recovery_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    result = verify(parse_args().out.expanduser().resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
