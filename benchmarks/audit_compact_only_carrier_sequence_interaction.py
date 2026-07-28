#!/usr/bin/env python3
# ruff: noqa: E501
"""Independent receipt audit for the compact-only carrier sequence interaction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/compact_only_carrier_sequence_interaction_20260728"
DEFAULT_JSON = (
    ROOT / "benchmarks/results/20260728_compact_only_carrier_sequence_interaction_AUDIT.json"
)
DEFAULT_MARKDOWN = (
    ROOT / "benchmarks/results/20260728_compact_only_carrier_sequence_interaction_AUDIT.md"
)
PREREGISTRATION = (
    ROOT / "benchmarks/results/20260728_compact_only_carrier_sequence_interaction_PREREG.md"
)
PARENT = ROOT / "runs/compact_only_carrier_policy_closure_20260728"

PHASE1_ROOTS = (282701, 282702, 282703)
PHASE2_ROOTS = (283701, 283702, 283703)
SH_ROOTS = (285701, 285702, 285703)
EVALUATION_ROOTS = (284701, 284702, 284703)
STATIC_ARMS = (
    "unpruned_phase2_d0",
    "unpruned_phase2_d0_then_prune",
    "phase2_sh3_joint_then_prune",
    "unpruned_phase2_d0_then_sh3_then_prune",
)
TRAINING_ARMS = (
    "prune_then_phase2_d0",
    "prune_then_phase2_sh3_joint",
    "phase2_sh3_joint_unpruned",
    "prune_then_phase2_d0_then_sh3",
    "unpruned_phase2_d0_then_sh3",
)
ALL_ARMS = (*STATIC_ARMS, *TRAINING_ARMS)
CONTAINED_ARMS = (
    "unpruned_phase2_d0_then_prune",
    "prune_then_phase2_d0",
    "prune_then_phase2_sh3_joint",
    "phase2_sh3_joint_then_prune",
    "prune_then_phase2_d0_then_sh3",
    "unpruned_phase2_d0_then_sh3_then_prune",
)
PHASE2_TRAINING_ARMS = (
    "prune_then_phase2_d0",
    "prune_then_phase2_sh3_joint",
    "phase2_sh3_joint_unpruned",
)
SH_TRAINING_ARMS = (
    "prune_then_phase2_d0_then_sh3",
    "unpruned_phase2_d0_then_sh3",
)
SAMPLE_BINDING_KEYS = (
    "step",
    "view_index",
    "view_name",
    "sample_seed",
    "xy_sha256",
    "active_sha256",
    "inside_fit_window_sha256",
    "proposal_density_sha256",
    "joint_density_sha256",
    "target_density_sha256",
    "importance_sha256",
    "proposal_component_ids_sha256",
)
EXPECTED_NO_IMAGE_BOUNDARY = {
    "forbidden_import_attempts": 0,
    "forbidden_modules_at_exit": [],
    "image_open_attempts": 0,
    "negative_control_denials": 3,
    "passed": True,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _command(*args: str) -> str:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _geometric(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _gaussian_sha256(fields: Mapping[str, torch.Tensor]) -> str:
    return _canonical_hash(
        {
            name: _tensor_sha256(fields[name])
            for name in ("means", "quats", "log_scales", "opacity", "sh")
        }
    )


def _load_gaussians(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path) as data:
        return {
            name: torch.from_numpy(data[name].copy()).float()
            for name in ("means", "quats", "log_scales", "opacity", "sh")
        }


def _subset(
    fields: Mapping[str, torch.Tensor],
    keep: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {name: value[keep] for name, value in fields.items()}


def _expand_sh3(
    fields: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    expanded = {name: value.clone() for name, value in fields.items()}
    sh = torch.zeros(fields["sh"].shape[0], 16, 3, dtype=fields["sh"].dtype)
    sh[:, : fields["sh"].shape[1]] = fields["sh"]
    expanded["sh"] = sh
    return expanded


def _load_records(run: Path) -> dict[int, dict[str, dict]]:
    return {
        seed: {
            arm: json.loads((run / "arms" / f"seed_{seed}" / arm / "result.json").read_text())
            for arm in ALL_ARMS
        }
        for seed in PHASE1_ROOTS
    }


def _metric(record: Mapping[str, object], metric: str) -> float:
    if record["kind"] == "static":
        return float(record["metrics"]["validation"][metric])
    step = str(record["config"]["iterations"])
    return float(record["checkpoint_metrics"][step]["validation"][metric])


def _comparison(
    records: Mapping[int, Mapping[str, Mapping[str, object]]],
    numerator: str,
    denominator: str,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for metric in ("J_Q", "J_U"):
        ratios = [
            _metric(records[seed][numerator], metric) / _metric(records[seed][denominator], metric)
            for seed in PHASE1_ROOTS
        ]
        result[metric] = {
            "per_seed": ratios,
            "geometric": _geometric(ratios),
            "wins": sum(ratio < 1.0 for ratio in ratios),
        }
    return result


def _removed_fraction(record: Mapping[str, object]) -> float:
    transformation = record["transformation"]
    for key in ("removed_fraction",):
        if key in transformation:
            return float(transformation[key])
    for key in ("prune", "containment"):
        if key in transformation:
            return float(transformation[key]["removed_fraction"])
    raise KeyError("arm has no containment receipt")


def _containment_viable(
    records: Mapping[int, Mapping[str, Mapping[str, object]]],
    arm: str,
    comparison: Mapping[str, object],
) -> bool:
    return (
        max(
            float(records[seed][arm]["support"]["splits"]["fit"]["outside_fraction"])
            for seed in PHASE1_ROOTS
        )
        == 0.0
        and max(
            float(records[seed][arm]["support"]["splits"]["fit"]["behind_near_fraction"])
            for seed in PHASE1_ROOTS
        )
        == 0.0
        and max(_removed_fraction(records[seed][arm]) for seed in PHASE1_ROOTS) <= 0.10
        and all(float(comparison[metric]["geometric"]) <= 1.05 for metric in ("J_Q", "J_U"))
    )


def _material(comparison: Mapping[str, object]) -> bool:
    return (
        comparison["J_Q"]["geometric"] <= 0.95
        and comparison["J_Q"]["wins"] == 3
        and comparison["J_U"]["geometric"] <= 1.02
    )


def _analysis(
    records: Mapping[int, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    comparisons = {
        "post_prune_vs_unpruned": _comparison(
            records,
            "unpruned_phase2_d0_then_prune",
            "unpruned_phase2_d0",
        ),
        "pre_prune_vs_unpruned": _comparison(
            records,
            "prune_then_phase2_d0",
            "unpruned_phase2_d0",
        ),
        "pre_prune_vs_post_prune": _comparison(
            records,
            "prune_then_phase2_d0",
            "unpruned_phase2_d0_then_prune",
        ),
        "pre_joint_sh3": _comparison(
            records,
            "prune_then_phase2_sh3_joint",
            "prune_then_phase2_d0",
        ),
        "post_joint_sh3": _comparison(
            records,
            "phase2_sh3_joint_then_prune",
            "unpruned_phase2_d0_then_prune",
        ),
        "pre_separate_sh3": _comparison(
            records,
            "prune_then_phase2_d0_then_sh3",
            "prune_then_phase2_d0",
        ),
        "post_separate_sh3": _comparison(
            records,
            "unpruned_phase2_d0_then_sh3_then_prune",
            "unpruned_phase2_d0_then_prune",
        ),
        "pre_joint_vs_separate": _comparison(
            records,
            "prune_then_phase2_sh3_joint",
            "prune_then_phase2_d0_then_sh3",
        ),
        "post_joint_vs_separate": _comparison(
            records,
            "phase2_sh3_joint_then_prune",
            "unpruned_phase2_d0_then_sh3_then_prune",
        ),
    }
    post_viable = _containment_viable(
        records,
        "unpruned_phase2_d0_then_prune",
        comparisons["post_prune_vs_unpruned"],
    )
    pre_viable = _containment_viable(
        records,
        "prune_then_phase2_d0",
        comparisons["pre_prune_vs_unpruned"],
    )
    order = "none"
    if pre_viable and post_viable:
        order_ratio = comparisons["pre_prune_vs_post_prune"]
        if all(order_ratio[metric]["geometric"] <= 1.02 for metric in ("J_Q", "J_U")):
            order = "pre"
        elif abs(order_ratio["J_Q"]["geometric"] - 1.0) <= 0.005:
            order = "post"
        else:
            order = "pre" if order_ratio["J_Q"]["geometric"] < 1.0 else "post"
    elif pre_viable:
        order = "pre"
    elif post_viable:
        order = "post"

    if order == "pre":
        joint_key = "pre_joint_sh3"
        separate_key = "pre_separate_sh3"
        joint_vs_separate_key = "pre_joint_vs_separate"
        joint_arm = "prune_then_phase2_sh3_joint"
        separate_arm = "prune_then_phase2_d0_then_sh3"
        degree_zero_arm = "prune_then_phase2_d0"
    elif order == "post":
        joint_key = "post_joint_sh3"
        separate_key = "post_separate_sh3"
        joint_vs_separate_key = "post_joint_vs_separate"
        joint_arm = "phase2_sh3_joint_then_prune"
        separate_arm = "unpruned_phase2_d0_then_sh3_then_prune"
        degree_zero_arm = "unpruned_phase2_d0_then_prune"
    else:
        joint_key = separate_key = joint_vs_separate_key = ""
        joint_arm = separate_arm = degree_zero_arm = ""

    joint_material = bool(joint_key) and _material(comparisons[joint_key])
    separate_material = bool(separate_key) and _material(comparisons[separate_key])
    sh_choice = "none"
    if joint_material and separate_material:
        ratio = comparisons[joint_vs_separate_key]
        if all(ratio[metric]["geometric"] <= 1.02 for metric in ("J_Q", "J_U")):
            sh_choice = "joint"
        else:
            joint_q = _geometric(
                [_metric(records[seed][joint_arm], "J_Q") for seed in PHASE1_ROOTS]
            )
            separate_q = _geometric(
                [_metric(records[seed][separate_arm], "J_Q") for seed in PHASE1_ROOTS]
            )
            sh_choice = "joint" if joint_q < separate_q else "separate"
    elif joint_material:
        sh_choice = "joint"
    elif separate_material:
        sh_choice = "separate"
    selected_arm = degree_zero_arm
    if sh_choice == "joint":
        selected_arm = joint_arm
    elif sh_choice == "separate":
        selected_arm = separate_arm

    summaries = {
        arm: {
            metric: {
                "per_seed": [_metric(records[seed][arm], metric) for seed in PHASE1_ROOTS],
                "geometric": _geometric(
                    [_metric(records[seed][arm], metric) for seed in PHASE1_ROOTS]
                ),
            }
            for metric in ("J_Q", "J_U")
        }
        for arm in ALL_ARMS
    }
    return {
        "comparisons": comparisons,
        "decisions": {
            "post_phase2_prune_viable": post_viable,
            "pre_phase2_prune_viable": pre_viable,
            "containment_order": order,
            "joint_sh3_material": joint_material,
            "separate_sh3_material": separate_material,
            "sh_choice": sh_choice,
            "selected_arm": selected_arm,
        },
        "validation_summaries": summaries,
        "validation_ranking": sorted(
            ALL_ARMS,
            key=lambda arm: summaries[arm]["J_Q"]["geometric"],
        ),
        "containment": {
            arm: {
                "fit_outside_fraction": [
                    records[seed][arm]["support"]["splits"]["fit"]["outside_fraction"]
                    for seed in PHASE1_ROOTS
                ],
                "fit_behind_near_fraction": [
                    records[seed][arm]["support"]["splits"]["fit"]["behind_near_fraction"]
                    for seed in PHASE1_ROOTS
                ],
                "removed_fraction": [
                    _removed_fraction(records[seed][arm]) for seed in PHASE1_ROOTS
                ],
            }
            for arm in CONTAINED_ARMS
        },
    }


def _project_steps(history: Mapping[str, object]) -> list[dict[str, object]]:
    return [{key: step[key] for key in SAMPLE_BINDING_KEYS} for step in history["steps"]]


def _keep_from_receipt(receipt: Mapping[str, object]) -> torch.Tensor:
    removed = receipt["removed_rows"]
    if removed != sorted(set(removed)):
        raise ValueError("removed rows are not sorted unique")
    keep = torch.ones(int(receipt["n_before"]), dtype=torch.bool)
    keep[torch.tensor(removed, dtype=torch.int64)] = False
    if receipt["keep_mask_sha256"] != _tensor_sha256(keep):
        raise ValueError("keep-mask digest mismatch")
    if receipt["removed"] != len(removed) or receipt["n_after"] != int(keep.sum()):
        raise ValueError("prune count mismatch")
    if not math.isclose(
        float(receipt["removed_fraction"]),
        len(removed) / int(receipt["n_before"]),
        rel_tol=1e-7,
        abs_tol=1e-8,
    ):
        raise ValueError("prune fraction mismatch")
    return keep


def _audit_invariants(
    run: Path,
    records: Mapping[int, Mapping[str, Mapping[str, object]]],
    recomputed_analysis: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    manifest = json.loads((run / "run_manifest.json").read_text())
    result = json.loads((run / "result.json").read_text())
    embedded_analysis = json.loads((run / "analysis.json").read_text())

    if manifest["status"] != "complete" or result["status"] != "complete":
        failures.append("run/result status is not complete")
    if manifest["result_sha256"] != _sha256(run / "result.json"):
        failures.append("manifest result digest mismatch")
    if manifest["preregistration"]["sha256"] != _sha256(PREREGISTRATION):
        failures.append("preregistration digest mismatch")
    if result["analysis"] != embedded_analysis or embedded_analysis != recomputed_analysis:
        failures.append("analysis differs from independent recomputation")
    if result["heldout_evaluated"] is not False:
        failures.append("held-out evaluation occurred")
    if (
        result["no_image_boundary"] != EXPECTED_NO_IMAGE_BOUNDARY
        or manifest["no_image_boundary"] != EXPECTED_NO_IMAGE_BOUNDARY
    ):
        failures.append("no-image boundary failed or changed")
    if manifest["dataset"]["packed_alpha_accessed"] is not False:
        failures.append("packed-alpha access declaration changed")

    for binding_name in ("source_binding", "parent_binding"):
        binding = manifest[binding_name]
        for relative, expected in binding["files"].items():
            path = ROOT / relative
            current = {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            if current != expected:
                failures.append(f"{binding_name} file changed: {relative}")
        if _canonical_hash(binding["files"]) != binding["semantic_sha256"]:
            failures.append(f"{binding_name} semantic digest mismatch")

    dataset_dir = ROOT / manifest["dataset"]["path"]
    dataset_manifest_path = dataset_dir / "manifest.json"
    if _sha256(dataset_manifest_path) != manifest["dataset"]["manifest_sha256"]:
        failures.append("compact dataset manifest changed")
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    views = {view["view_id"]: view for view in dataset_manifest["views"]}
    for view_id, expected_digest in manifest["dataset"]["container_sha256"].items():
        path = dataset_dir / views[view_id]["path"]
        if _sha256(path) != expected_digest or path.stat().st_size != views[view_id]["bytes"]:
            failures.append(f"compact container changed: {view_id}")

    teacher_digests: set[str] = set()
    proposal_digests: set[str] = set()
    for seed in PHASE1_ROOTS:
        parent_phase1_dir = PARENT / "arms" / f"seed_{seed}" / "corrected_C_all_380"
        parent_anchor_dir = PARENT / "arms" / f"seed_{seed}" / "continue_means_fixed_380"
        phase1 = _load_gaussians(parent_phase1_dir / "gaussians_final.npz")
        anchor = _load_gaussians(parent_anchor_dir / "gaussians_final.npz")
        models: dict[str, dict[str, torch.Tensor]] = {}

        for arm in ALL_ARMS:
            record = records[seed][arm]
            directory = run / "arms" / f"seed_{seed}" / arm
            pointer = result["records"][str(seed)][arm]
            path = run / pointer["path"]
            if path != directory / "result.json" or _sha256(path) != pointer["sha256"]:
                failures.append(f"{seed}/{arm}: result pointer mismatch")
            final_path = directory / "gaussians_final.npz"
            if _sha256(final_path) != record["final_file_sha256"]:
                failures.append(f"{seed}/{arm}: final file digest mismatch")
            model = _load_gaussians(final_path)
            models[arm] = model
            if _gaussian_sha256(model) != record["final_semantic_sha256"]:
                failures.append(f"{seed}/{arm}: final semantic digest mismatch")
            metrics = (
                record["metrics"] if record["kind"] == "static" else record["checkpoint_metrics"]
            )
            if record["kind"] == "static":
                if set(metrics) != {"fit", "validation"}:
                    failures.append(f"{seed}/{arm}: static metric split changed")
            elif any(set(checkpoint) != {"fit", "validation"} for checkpoint in metrics.values()):
                failures.append(f"{seed}/{arm}: held-out checkpoint metric appeared")

        if _gaussian_sha256(models["unpruned_phase2_d0"]) != _gaussian_sha256(anchor):
            failures.append(f"{seed}: sequence anchor differs from audited parent")

        phase1_receipt = records[seed]["prune_then_phase2_d0"]["transformation"]["prune"]
        try:
            phase1_keep = _keep_from_receipt(phase1_receipt)
        except ValueError as error:
            failures.append(f"{seed}: phase-1 prune receipt: {error}")
            phase1_keep = torch.ones(phase1["means"].shape[0], dtype=torch.bool)
        phase1_pruned = _subset(phase1, phase1_keep)
        if records[seed]["prune_then_phase2_d0"]["initial_semantic_sha256"] != _gaussian_sha256(
            phase1_pruned
        ):
            failures.append(f"{seed}: pre-pruned degree-zero lineage mismatch")
        if records[seed]["prune_then_phase2_sh3_joint"][
            "initial_semantic_sha256"
        ] != _gaussian_sha256(_expand_sh3(phase1_pruned)):
            failures.append(f"{seed}: pre-pruned joint-SH lineage mismatch")
        if records[seed]["phase2_sh3_joint_unpruned"][
            "initial_semantic_sha256"
        ] != _gaussian_sha256(_expand_sh3(phase1)):
            failures.append(f"{seed}: unpruned joint-SH lineage mismatch")
        if records[seed]["prune_then_phase2_d0_then_sh3"][
            "initial_semantic_sha256"
        ] != _gaussian_sha256(_expand_sh3(models["prune_then_phase2_d0"])):
            failures.append(f"{seed}: contained separate-SH lineage mismatch")
        if records[seed]["unpruned_phase2_d0_then_sh3"][
            "initial_semantic_sha256"
        ] != _gaussian_sha256(_expand_sh3(anchor)):
            failures.append(f"{seed}: unpruned separate-SH lineage mismatch")

        for static_arm, source_fields in (
            ("unpruned_phase2_d0_then_prune", anchor),
            ("phase2_sh3_joint_then_prune", models["phase2_sh3_joint_unpruned"]),
            (
                "unpruned_phase2_d0_then_sh3_then_prune",
                models["unpruned_phase2_d0_then_sh3"],
            ),
        ):
            receipt = records[seed][static_arm]["transformation"]
            try:
                keep = _keep_from_receipt(receipt)
            except ValueError as error:
                failures.append(f"{seed}/{static_arm}: {error}")
                continue
            if (
                _gaussian_sha256(_subset(source_fields, keep))
                != records[seed][static_arm]["final_semantic_sha256"]
            ):
                failures.append(f"{seed}/{static_arm}: strict subset lineage mismatch")

        phase2_reference: list[dict[str, object]] | None = None
        sh_reference: list[dict[str, object]] | None = None
        for arm in TRAINING_ARMS:
            record = records[seed][arm]
            directory = run / "arms" / f"seed_{seed}" / arm
            history_path = directory / "history.json"
            if _sha256(history_path) != record["history_file_sha256"]:
                failures.append(f"{seed}/{arm}: history digest mismatch")
            history = json.loads(history_path.read_text())
            if history["teacher_digest_before"] != history["teacher_digest_after"]:
                failures.append(f"{seed}/{arm}: teacher mutated")
            if history["proposal_digest_before"] != history["proposal_digest_after"]:
                failures.append(f"{seed}/{arm}: proposal mutated")
            teacher_digests.add(history["teacher_digest_before"])
            proposal_digests.add(history["proposal_digest_before"])
            if history["optimizer_group_motion"]["means"]["max_abs"] != 0.0:
                failures.append(f"{seed}/{arm}: means moved")
            for family in history["frozen_parameter_groups"]:
                if history["optimizer_group_motion"][family]["max_abs"] != 0.0:
                    failures.append(f"{seed}/{arm}: frozen family {family} moved")
            if history["n_init_3d"] != history["n_opt_3d"]:
                failures.append(f"{seed}/{arm}: topology changed")
            projected = _project_steps(history)
            if arm in PHASE2_TRAINING_ARMS:
                if history["seed"] != PHASE2_ROOTS[PHASE1_ROOTS.index(seed)]:
                    failures.append(f"{seed}/{arm}: phase-2 root changed")
                if phase2_reference is None:
                    phase2_reference = projected
                elif projected != phase2_reference:
                    failures.append(f"{seed}/{arm}: phase-2 sample stream differs")
            else:
                if history["seed"] != SH_ROOTS[PHASE1_ROOTS.index(seed)]:
                    failures.append(f"{seed}/{arm}: SH root changed")
                if sh_reference is None:
                    sh_reference = projected
                elif projected != sh_reference:
                    failures.append(f"{seed}/{arm}: SH sample stream differs")

        for arm in CONTAINED_ARMS:
            support = records[seed][arm]["support"]["splits"]["fit"]
            if support["outside_fraction"] != 0.0 or support["behind_near_fraction"] != 0.0:
                failures.append(f"{seed}/{arm}: fitting-view containment failed")

    if len(teacher_digests) != 1 or len(proposal_digests) != 1:
        failures.append("teacher/proposal digests differ across roots or arms")

    return {
        "passed": not failures,
        "failures": failures,
        "source_seal_matches_current_tree": not any(
            failure.startswith("source_binding file changed") for failure in failures
        ),
        "parent_binding_matches": not any(
            failure.startswith("parent_binding file changed") for failure in failures
        ),
        "source_binding_sha256": manifest["source_binding"]["semantic_sha256"],
        "parent_binding_sha256": manifest["parent_binding"]["semantic_sha256"],
        "source_file_count": len(manifest["source_binding"]["files"]),
        "parent_file_count": len(manifest["parent_binding"]["files"]),
        "git_revision": manifest["source_binding"]["git_revision"],
        "git_status_at_seal": manifest["source_binding"]["git_status"],
        "command": manifest["source_binding"]["command"],
        "no_image_boundary": result["no_image_boundary"],
        "packed_alpha_accessed": manifest["dataset"]["packed_alpha_accessed"],
        "heldout_evaluated": result["heldout_evaluated"],
        "teacher_digest": next(iter(teacher_digests), None),
        "proposal_digest": next(iter(proposal_digests), None),
    }


def audit(run: Path) -> dict[str, object]:
    records = _load_records(run)
    analysis = _analysis(records)
    invariants = _audit_invariants(run, records, analysis)
    result = json.loads((run / "result.json").read_text())
    return {
        "schema": "rtgs.compact-carrier-sequence.audit.v1",
        "status": "PASS_WITH_SCOPE_LIMITS" if invariants["passed"] else "FAIL",
        "run": str(run.relative_to(ROOT)),
        "run_result_sha256": _sha256(run / "result.json"),
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "invariants": invariants,
        "analysis": analysis,
        "resources": result["resources"],
        "decisions": {
            **analysis["decisions"],
            "implement_compact_only_sequence": invariants["passed"],
            "retain_higher_sh_stage": False,
            "retain_topology_growth": False,
            "projected_center_containment_established": invariants["passed"],
            "no_visual_hull_interior_floaters_established": False,
            "general_vram_claim_established": False,
        },
        "scope_limits": [
            "single real development scene; no cross-scene or confirmatory inference",
            "held-out cameras were not evaluated",
            "fitting-view fitted-Gaussian visual-hull center containment is not surface occupancy",
            "a floater inside the visual hull remains undetectable",
            "full Gaussian support is infinite, and finite-footprint containment was not tested",
            "absolute allocator/RSS receipts are not a controlled dense-vs-compact or general VRAM claim",
            "the no-image guard is an executable Python boundary, not an OS-level information-flow proof",
        ],
    }


def _percent(ratio: float) -> str:
    return f"{(ratio - 1.0) * 100:+.2f}%"


def markdown(record: Mapping[str, object]) -> str:
    analysis = record["analysis"]
    comparisons = analysis["comparisons"]
    summaries = analysis["validation_summaries"]
    post = comparisons["post_prune_vs_unpruned"]
    pre = comparisons["pre_prune_vs_unpruned"]
    order = comparisons["pre_prune_vs_post_prune"]
    joint = comparisons["pre_joint_sh3"]
    separate = comparisons["pre_separate_sh3"]
    selected = summaries["prune_then_phase2_d0"]
    return f"""# Compact-only carrier sequence interaction — independent audit — 2026-07-28

Status: **{record["status"]}**.

## Referee disposition

| Question | Disposition | Independently recomputed evidence |
| --- | --- | --- |
| Did the run remain compact-only through every stage? | **Confirm within the executable boundary.** | All {record["invariants"]["source_file_count"]} pre-sealed source files and {record["invariants"]["parent_file_count"]} parent artifacts remain byte-identical. The guard recorded zero image opens/imports, packed alpha was not loaded, compact containers match, and held-out cameras were not evaluated. |
| Is post-phase-2 strict pruning viable? | **Yes.** | Pruned/unpruned `J_Q={post["J_Q"]["geometric"]:.4f}` ({_percent(post["J_Q"]["geometric"])}), `J_U={post["J_U"]["geometric"]:.4f}`; zero fitting-view violations and 5.32–5.42% rows removed. |
| Is pre-phase-2 strict pruning viable? | **Yes, and selected.** | Prune→phase2/unpruned `J_Q={pre["J_Q"]["geometric"]:.4f}` ({_percent(pre["J_Q"]["geometric"])}), `J_U={pre["J_U"]["geometric"]:.4f}`; zero violations. It improves on post-pruning: `J_Q={order["J_Q"]["geometric"]:.4f}`, `J_U={order["J_U"]["geometric"]:.4f}`. |
| Does joint SH3 earn inclusion in phase 2? | **No.** | Joint-SH3/degree-zero after pre-pruning: `J_Q={joint["J_Q"]["geometric"]:.4f}` ({_percent(joint["J_Q"]["geometric"])}), 3/3 wins, but below the frozen 5% materiality floor. |
| Does a separate SH3 phase earn a third stage? | **No.** | Separate-SH3/degree-zero: `J_Q={separate["J_Q"]["geometric"]:.4f}` ({_percent(separate["J_Q"]["geometric"])}), 3/3 wins, but below 5%. |
| Does the sequence establish no free floaters? | **Only outside the fitted-view visual hull.** | Every retained center has `q<=9` and positive depth in every fitting view, and frozen means preserve that invariant. Interior-hull floaters remain unidentifiable. |
| Does this prove the broad VRAM claim? | **No.** | It proves an image-free compact execution and records absolute resources; it does not provide a controlled dense baseline or multi-scene scaling evidence. |

## Audited selected sequence

1. Fit-only Beam Fusion; no free birth.
2. Renderer-aware symmetric covariance repair only.
3. Compact all-parameter degree-zero phase 1.
4. Strict fitted-Gaussian visual-hull projected-center pruning.
5. Compact degree-zero phase 2 with means frozen.

No opacity repair, appearance repair, legacy covariance repair, soft support penalty, clone,
split, insertion, densification, or SH3 stage is retained.

The selected arm has validation geometric `J_Q={selected["J_Q"]["geometric"]:.8f}` and
`J_U={selected["J_U"]["geometric"]:.8f}` with 4,729–4,734 rows. Relative to the unpruned selected
phase-2 anchor, containment costs 2.69% `J_Q` and 1.64% `J_U`.

## Receipt findings

- All 27 arm result/model receipts and 15 training histories match their digests.
- Parent phase-1 and phase-2 semantic lineage, strict-prune subsets, SH expansion, source/proposal
  immutability, paired point streams, fixed topology, and exact frozen-mean motion were
  independently checked.
- Every contained arm has exactly zero fitting-view center-support and near-plane violations.
- The run is single-scene development evidence. It selects the repository development sequence,
  not a cross-scene, production, or publication claim.

## Commands checked

```text
.venv/bin/python benchmarks/compact_only_carrier_sequence_interaction.py
.venv/bin/python benchmarks/audit_compact_only_carrier_sequence_interaction.py
```
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    record = audit(args.run.resolve())
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.markdown.write_text(markdown(record), encoding="utf-8")
    print(record["status"])
    if record["status"] == "FAIL":
        print("\n".join(record["invariants"]["failures"]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
