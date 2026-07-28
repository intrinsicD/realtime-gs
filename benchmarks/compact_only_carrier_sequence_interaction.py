#!/usr/bin/env python3
"""Resolve compact carrier containment order and higher-SH stage interactions."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

if __package__:
    from benchmarks import compact_only_carrier_policy_closure as closure
    from benchmarks import compact_only_carrier_stage_ablation as stage1
else:
    import compact_only_carrier_policy_closure as closure
    import compact_only_carrier_stage_ablation as stage1

from rtgs.core.gaussians3d import Gaussians3D
from rtgs.data.compact_views import CompactDataset

ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    ROOT / "benchmarks/results/20260728_compact_only_carrier_sequence_interaction_PREREG.md"
)
DEFAULT_OUT = ROOT / "runs/compact_only_carrier_sequence_interaction_20260728"
PARENT = ROOT / "runs/compact_only_carrier_policy_closure_20260728"
PARENT_AUDIT_JSON = (
    ROOT / "benchmarks/results/20260728_compact_only_carrier_policy_closure_AUDIT.json"
)
PARENT_AUDIT_MARKDOWN = (
    ROOT / "benchmarks/results/20260728_compact_only_carrier_policy_closure_AUDIT.md"
)

PHASE1_ROOTS = closure.PHASE1_ROOTS
PHASE2_ROOTS = closure.LATER_ROOTS
SH_ROOTS = (285701, 285702, 285703)
EVALUATION_ROOTS = closure.EVALUATION_ROOTS

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


def _command(*args: str) -> str:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _source_binding() -> dict[str, object]:
    paths = {
        "benchmarks/compact_only_carrier_sequence_interaction.py",
        "benchmarks/compact_only_carrier_policy_closure.py",
        "benchmarks/compact_only_carrier_stage_ablation.py",
    }
    paths.update(str(path.relative_to(ROOT)) for path in (ROOT / "src/rtgs").rglob("*.py"))
    files = {
        relative: {
            "sha256": stage1._file_sha256(ROOT / relative),
            "bytes": (ROOT / relative).stat().st_size,
        }
        for relative in sorted(paths)
    }
    return {
        "git_revision": _command("git", "rev-parse", "HEAD").strip(),
        "git_status": _command("git", "status", "--short").splitlines(),
        "command": [sys.executable, *sys.argv],
        "files": files,
        "semantic_sha256": stage1._canonical_hash(files),
    }


def _parent_binding() -> dict[str, object]:
    paths = {
        PARENT / "result.json",
        PARENT / "analysis.json",
        PARENT / "run_manifest.json",
        PARENT_AUDIT_JSON,
        PARENT_AUDIT_MARKDOWN,
    }
    for seed in PHASE1_ROOTS:
        for arm in ("corrected_C_all_380", "continue_means_fixed_380"):
            directory = PARENT / "arms" / f"seed_{seed}" / arm
            paths.add(directory / "result.json")
            paths.add(directory / "gaussians_final.npz")
    for root in EVALUATION_ROOTS:
        paths.add(PARENT / "banks" / f"evaluation_{root}.npz")
        paths.add(PARENT / "banks" / f"evaluation_{root}.json")
    files = {
        str(path.relative_to(ROOT)): {
            "sha256": stage1._file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    }
    return {
        "files": files,
        "semantic_sha256": stage1._canonical_hash(files),
    }


def _metric(
    record: Mapping[str, object],
    metric: str,
) -> float:
    return closure._metric(record, metric)


def _geometric(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


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
    if "removed_fraction" in transformation:
        return float(transformation["removed_fraction"])
    if "prune" in transformation:
        return float(transformation["prune"]["removed_fraction"])
    if "containment" in transformation:
        return float(transformation["containment"]["removed_fraction"])
    raise KeyError("arm does not carry a containment receipt")


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


def _material(
    comparison: Mapping[str, object],
) -> bool:
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
        joint_ratio = comparisons[joint_vs_separate_key]
        if all(joint_ratio[metric]["geometric"] <= 1.02 for metric in ("J_Q", "J_U")):
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
    decisions = {
        "post_phase2_prune_viable": post_viable,
        "pre_phase2_prune_viable": pre_viable,
        "containment_order": order,
        "joint_sh3_material": joint_material,
        "separate_sh3_material": separate_material,
        "sh_choice": sh_choice,
        "selected_arm": selected_arm,
    }
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
        "decisions": decisions,
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


def _assert_contained(
    record: Mapping[str, object],
    *,
    seed: int,
    arm: str,
) -> None:
    support = record["support"]["splits"]["fit"]
    if support["outside_fraction"] != 0.0 or support["behind_near_fraction"] != 0.0:
        raise RuntimeError(f"{seed}/{arm}: projected-center containment failed")


def _load_parent(
    seed: int,
    arm: str,
) -> tuple[Gaussians3D, dict[str, object]]:
    directory = PARENT / "arms" / f"seed_{seed}" / arm
    result = json.loads((directory / "result.json").read_text())
    if result["status"] != "complete":
        raise RuntimeError(f"parent arm is incomplete: {seed}/{arm}")
    model_path = directory / "gaussians_final.npz"
    if stage1._file_sha256(model_path) != result["final_file_sha256"]:
        raise RuntimeError(f"parent model digest changed: {seed}/{arm}")
    model = Gaussians3D.load_npz(model_path)
    if stage1._gaussians_sha256(model) != result["final_semantic_sha256"]:
        raise RuntimeError(f"parent semantic digest changed: {seed}/{arm}")
    return model, result


def run(out: Path) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("the frozen sequence-interaction run requires cuda:0")
    out.mkdir(parents=True, exist_ok=True)
    source_binding = _source_binding()
    parent_binding = _parent_binding()
    manifest_path = out / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["source_binding"]["semantic_sha256"] != source_binding["semantic_sha256"]:
            raise RuntimeError("refusing to resume under a different source binding")
        if manifest["parent_binding"]["semantic_sha256"] != parent_binding["semantic_sha256"]:
            raise RuntimeError("refusing to resume under a different parent binding")
    else:
        manifest = {
            "schema": "rtgs.compact-carrier-sequence.run.v1",
            "status": "running",
            "preregistration": {
                "path": str(PREREGISTRATION.relative_to(ROOT)),
                "sha256": stage1._file_sha256(PREREGISTRATION),
            },
            "source_binding": source_binding,
            "parent_binding": parent_binding,
            "roots": {
                "phase1": list(PHASE1_ROOTS),
                "phase2": list(PHASE2_ROOTS),
                "sh": list(SH_ROOTS),
                "evaluation": list(EVALUATION_ROOTS),
            },
            "global_splits": {
                "fit": list(closure.FIT_GLOBAL),
                "validation": list(closure.VALIDATION_GLOBAL),
                "heldout_not_evaluated": list(closure.HELDOUT_GLOBAL),
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
            },
        }
        stage1._atomic_json(manifest_path, manifest)

    guard = stage1.NoImageGuard()
    started = time.perf_counter()
    with guard:
        dataset = CompactDataset.load(stage1.DATASET, device="cpu", load_alpha=False)
        development_cpu = stage1._inputs(
            dataset,
            closure.DEVELOPMENT_GLOBAL,
            name="frame00008-sequence-development",
        )
        fit_cpu = stage1._subset(
            development_cpu,
            closure.LOCAL_SPLITS["fit"],
            name="frame00008-sequence-fit",
        )
        extent = float(dataset.bounds_hint[1])
        manifest["dataset"] = {
            "path": str(stage1.DATASET.relative_to(ROOT)),
            "manifest_sha256": stage1._file_sha256(stage1.DATASET / "manifest.json"),
            "container_sha256": {view.view_id: view.sha256 for view in dataset.views},
            "container_bytes": sum(view.bytes for view in dataset.views),
            "packed_alpha_accessed": False,
            "development_view_ids": list(development_cpu.view_names),
            "fit_view_ids": list(fit_cpu.view_names),
            "heldout_view_ids_not_evaluated": [
                dataset.views[index].view_id for index in closure.HELDOUT_GLOBAL
            ],
        }
        stage1._atomic_json(manifest_path, manifest)

        development_gpu = development_cpu.to("cuda:0")
        fit_gpu = stage1._subset(
            development_gpu,
            closure.LOCAL_SPLITS["fit"],
            name="frame00008-sequence-fit-gpu",
        )
        banks: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
        for root in EVALUATION_ROOTS:
            path, bank = stage1._prepare_bank(PARENT, development_cpu, root)
            banks[root] = bank
            print(f"[bank:{root}] {path.relative_to(PARENT)}", flush=True)

        records: dict[int, dict[str, dict[str, object]]] = {}
        for seed, phase2_seed, sh_seed, evaluation_root in zip(
            PHASE1_ROOTS,
            PHASE2_ROOTS,
            SH_ROOTS,
            EVALUATION_ROOTS,
            strict=True,
        ):
            records[seed] = {}
            bank = banks[evaluation_root]
            phase1, phase1_parent = _load_parent(seed, "corrected_C_all_380")
            anchor, anchor_parent = _load_parent(seed, "continue_means_fixed_380")
            if anchor_parent["initial_semantic_sha256"] != phase1_parent["final_semantic_sha256"]:
                raise RuntimeError(f"{seed}: parent phase-2 lineage changed")

            records[seed]["unpruned_phase2_d0"] = closure._run_static_arm(
                out,
                seed=seed,
                arm="unpruned_phase2_d0",
                model=anchor,
                development_inputs=development_gpu,
                banks=bank,
                transformation={
                    "kind": "audited_parent_anchor",
                    "parent_result_sha256": stage1._file_sha256(
                        PARENT
                        / "arms"
                        / f"seed_{seed}"
                        / "continue_means_fixed_380"
                        / "result.json"
                    ),
                },
            )

            post_pruned, post_receipt = closure.strict_prune(anchor, fit_gpu)
            records[seed]["unpruned_phase2_d0_then_prune"] = closure._run_static_arm(
                out,
                seed=seed,
                arm="unpruned_phase2_d0_then_prune",
                model=post_pruned,
                development_inputs=development_gpu,
                banks=bank,
                transformation=post_receipt,
            )

            pre_pruned, pre_receipt = closure.strict_prune(phase1, fit_gpu)
            phase2_d0, record = closure._run_training_arm(
                out,
                seed=seed,
                arm="prune_then_phase2_d0",
                initialization=pre_pruned,
                fit_inputs=fit_gpu,
                development_inputs=development_gpu,
                banks=bank,
                config=closure._train_config(
                    iterations=380,
                    checkpoints=(0, 190, 380),
                    seed=phase2_seed,
                    extent=extent,
                    trainable=("quats", "scales", "opacities", "sh0"),
                ),
                transformation={"kind": "strict_prune_then_phase2", "prune": pre_receipt},
            )
            records[seed]["prune_then_phase2_d0"] = record

            pre_joint, record = closure._run_training_arm(
                out,
                seed=seed,
                arm="prune_then_phase2_sh3_joint",
                initialization=pre_pruned.with_sh_degree(3),
                fit_inputs=fit_gpu,
                development_inputs=development_gpu,
                banks=bank,
                config=closure._train_config(
                    iterations=380,
                    checkpoints=(0, 190, 380),
                    seed=phase2_seed,
                    extent=extent,
                    trainable=("quats", "scales", "opacities", "sh0", "shN"),
                    sh_degree=3,
                ),
                transformation={
                    "kind": "strict_prune_then_joint_sh3_phase2",
                    "prune": pre_receipt,
                },
            )
            records[seed]["prune_then_phase2_sh3_joint"] = record

            unpruned_joint, record = closure._run_training_arm(
                out,
                seed=seed,
                arm="phase2_sh3_joint_unpruned",
                initialization=phase1.with_sh_degree(3),
                fit_inputs=fit_gpu,
                development_inputs=development_gpu,
                banks=bank,
                config=closure._train_config(
                    iterations=380,
                    checkpoints=(0, 190, 380),
                    seed=phase2_seed,
                    extent=extent,
                    trainable=("quats", "scales", "opacities", "sh0", "shN"),
                    sh_degree=3,
                ),
                transformation={"kind": "joint_sh3_phase2_unpruned"},
            )
            records[seed]["phase2_sh3_joint_unpruned"] = record
            joint_post_pruned, joint_post_receipt = closure.strict_prune(
                unpruned_joint,
                fit_gpu,
            )
            records[seed]["phase2_sh3_joint_then_prune"] = closure._run_static_arm(
                out,
                seed=seed,
                arm="phase2_sh3_joint_then_prune",
                model=joint_post_pruned,
                development_inputs=development_gpu,
                banks=bank,
                transformation=joint_post_receipt,
            )

            pre_separate, record = closure._run_training_arm(
                out,
                seed=seed,
                arm="prune_then_phase2_d0_then_sh3",
                initialization=phase2_d0.with_sh_degree(3),
                fit_inputs=fit_gpu,
                development_inputs=development_gpu,
                banks=bank,
                config=closure._train_config(
                    iterations=380,
                    checkpoints=(0, 190, 380),
                    seed=sh_seed,
                    extent=extent,
                    trainable=("sh0", "shN"),
                    sh_degree=3,
                ),
                transformation={
                    "kind": "contained_separate_sh3",
                    "containment": pre_receipt,
                },
            )
            records[seed]["prune_then_phase2_d0_then_sh3"] = record

            unpruned_separate, record = closure._run_training_arm(
                out,
                seed=seed,
                arm="unpruned_phase2_d0_then_sh3",
                initialization=anchor.with_sh_degree(3),
                fit_inputs=fit_gpu,
                development_inputs=development_gpu,
                banks=bank,
                config=closure._train_config(
                    iterations=380,
                    checkpoints=(0, 190, 380),
                    seed=sh_seed,
                    extent=extent,
                    trainable=("sh0", "shN"),
                    sh_degree=3,
                ),
                transformation={"kind": "unpruned_separate_sh3"},
            )
            records[seed]["unpruned_phase2_d0_then_sh3"] = record
            separate_post_pruned, separate_post_receipt = closure.strict_prune(
                unpruned_separate,
                fit_gpu,
            )
            records[seed]["unpruned_phase2_d0_then_sh3_then_prune"] = closure._run_static_arm(
                out,
                seed=seed,
                arm="unpruned_phase2_d0_then_sh3_then_prune",
                model=separate_post_pruned,
                development_inputs=development_gpu,
                banks=bank,
                transformation=separate_post_receipt,
            )

            for arm in CONTAINED_ARMS:
                _assert_contained(records[seed][arm], seed=seed, arm=arm)
            print(f"[root:{seed}] complete", flush=True)

        analysis = _analysis(records)
        stage1._atomic_json(out / "analysis.json", analysis)
        guard_record = guard.record()
        if not guard_record["passed"]:
            raise RuntimeError(f"no-image boundary failed: {guard_record}")
        result = {
            "schema": "rtgs.compact-carrier-sequence.result.v1",
            "status": "complete",
            "scope": "single-scene compact-only development",
            "records": {
                str(seed): {
                    arm: {
                        "path": str(
                            (out / "arms" / f"seed_{seed}" / arm / "result.json").relative_to(out)
                        ),
                        "sha256": stage1._file_sha256(
                            out / "arms" / f"seed_{seed}" / arm / "result.json"
                        ),
                    }
                    for arm in ALL_ARMS
                }
                for seed in PHASE1_ROOTS
            },
            "analysis": analysis,
            "heldout_evaluated": False,
            "no_image_boundary": guard_record,
            "resources": {
                "total_seconds": time.perf_counter() - started,
                "peak_rss_bytes": stage1._peak_rss_bytes(),
                "compact_container_bytes": sum(view.bytes for view in dataset.views),
                "development_teacher_tensor_bytes": stage1._teacher_bytes(development_gpu),
            },
        }
        stage1._atomic_json(out / "result.json", result)
        manifest["status"] = "complete"
        manifest["result_sha256"] = stage1._file_sha256(out / "result.json")
        manifest["no_image_boundary"] = guard_record
        manifest["total_seconds"] = result["resources"]["total_seconds"]
        stage1._atomic_json(manifest_path, manifest)
    print(
        f"[complete] selected={analysis['decisions']['selected_arm']} "
        f"elapsed={time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    return run(args.out.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
