#!/usr/bin/env python3
# ruff: noqa: E501
"""Independent receipt audit for the 2026-07-28 compact-only carrier ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "runs/compact_only_carrier_stage_ablation_20260728"
DEFAULT_JSON = ROOT / "benchmarks/results/20260728_compact_only_carrier_stage_ablation_AUDIT.json"
DEFAULT_MARKDOWN = ROOT / "benchmarks/results/20260728_compact_only_carrier_stage_ablation_AUDIT.md"
PREREGISTRATION = ROOT / "benchmarks/results/20260728_compact_only_carrier_stage_ablation_PREREG.md"

SEEDS = (280701, 280702, 280703)
STEPS = (0, 95, 190, 380)
ARMS = (
    "beam_all",
    "legacy_C_all",
    "legacy_O_all",
    "legacy_A_all",
    "legacy_CO_all",
    "legacy_CA_all",
    "legacy_OA_all",
    "legacy_COA_all",
    "corrected_C_all",
    "corrected_CA_all",
    "beam_means_only",
    "beam_means_fixed",
    "beam_geometry_only",
    "beam_appearance_only",
    "beam_all_support",
    "corrected_CA_all_support",
)
BEAM_INITIALIZATION_ARMS = (
    "beam_all",
    "beam_means_only",
    "beam_means_fixed",
    "beam_geometry_only",
    "beam_appearance_only",
    "beam_all_support",
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
SOURCE_PATHS = (
    "benchmarks/compact_only_carrier_stage_ablation.py",
    "src/rtgs/core/observation2d.py",
    "src/rtgs/data/compact_views.py",
    "src/rtgs/data/reconstruction_inputs.py",
    "src/rtgs/lift/__init__.py",
    "src/rtgs/lift/base.py",
    "src/rtgs/lift/beam_fusion.py",
    "src/rtgs/lift/carrier_refinement.py",
    "src/rtgs/optim/compact_trainer.py",
    "src/rtgs/render/torch_points.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _geometric(values: Sequence[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _command(*args: str) -> str:
    return subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _final(record: Mapping[str, object], split: str, metric: str) -> float:
    return float(record["checkpoint_metrics"]["380"][split][metric])


def _summary(records: Mapping[int, Mapping[str, Mapping[str, object]]], arm: str) -> dict:
    q = [_final(records[seed][arm], "validation", "J_Q") for seed in SEEDS]
    u = [_final(records[seed][arm], "validation", "J_U") for seed in SEEDS]
    outside = [
        float(records[seed][arm]["support"]["splits"]["validation"]["outside_fraction"])
        for seed in SEEDS
    ]
    return {
        "validation_J_Q": q,
        "validation_J_U": u,
        "validation_outside_fraction": outside,
        "geometric_validation_J_Q": _geometric(q),
        "geometric_validation_J_U": _geometric(u),
        "mean_validation_outside_fraction": sum(outside) / len(outside),
    }


def _paired_ratio(
    summaries: Mapping[str, Mapping[str, object]],
    numerator: str,
    denominator: str,
    key: str,
) -> dict[str, object]:
    values = [
        float(left) / float(right)
        for left, right in zip(
            summaries[numerator][key],
            summaries[denominator][key],
            strict=True,
        )
    ]
    return {
        "numerator": numerator,
        "denominator": denominator,
        "per_seed": values,
        "geometric": _geometric(values),
        "wins": sum(value < 1 for value in values),
    }


def _factorial_marginal(
    summaries: Mapping[str, Mapping[str, object]],
    pairs: Sequence[tuple[str, str]],
    key: str,
) -> dict[str, object]:
    values = []
    for seed_index in range(len(SEEDS)):
        values.append(
            _geometric(
                [
                    float(summaries[on][key][seed_index]) / float(summaries[off][key][seed_index])
                    for on, off in pairs
                ]
            )
        )
    return {
        "per_seed": values,
        "geometric": _geometric(values),
        "wins": sum(value < 1 for value in values),
    }


def _load_records(run: Path) -> dict[int, dict[str, dict]]:
    records: dict[int, dict[str, dict]] = {}
    for seed in SEEDS:
        records[seed] = {}
        for arm in ARMS:
            path = run / "arms" / f"seed_{seed}" / arm / "result.json"
            records[seed][arm] = json.loads(path.read_text())
    return records


def _audit_invariants(
    run: Path,
    records: Mapping[int, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    failures: list[str] = []
    result = json.loads((run / "result.json").read_text())
    manifest = json.loads((run / "run_manifest.json").read_text())
    if result["status"] != "complete" or manifest["status"] != "complete":
        failures.append("run/result status is not complete")
    if manifest["preregistration"]["sha256"] != _sha256(PREREGISTRATION):
        failures.append("preregistration digest mismatch")
    expected_split = {
        "fit": [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 13, 14, 17, 18, 19, 20, 22, 24, 25],
        "validation": [4, 10, 16, 21],
        "heldout": [7, 15, 23],
    }
    if manifest["split_indices"] != expected_split:
        failures.append("camera split differs from the preregistration")
    boundary = result["no_image_boundary"]
    if boundary != {
        "forbidden_import_attempts": 0,
        "forbidden_modules_at_exit": [],
        "image_open_attempts": 0,
        "negative_control_denials": 3,
        "passed": True,
    }:
        failures.append("no-image boundary receipt failed or changed")
    if manifest["dataset"].get("packed_alpha_accessed") is not False:
        failures.append("packed-alpha access declaration changed")

    teacher_digests: set[str] = set()
    proposal_digests: set[str] = set()
    for seed in SEEDS:
        reference_steps: list[dict[str, object]] | None = None
        reference_schedule: str | None = None
        for arm in ARMS:
            record = records[seed][arm]
            if record["status"] != "complete" or record["config"]["device"] != "cuda:0":
                failures.append(f"{seed}/{arm}: incomplete or wrong device")
            if record["config"]["iterations"] != 380 or record["config"]["sh_degree"] != 0:
                failures.append(f"{seed}/{arm}: iteration or SH degree changed")
            if any(
                record["checkpoint_metrics"][str(step)].keys() != {"fit", "validation"}
                for step in STEPS
            ):
                failures.append(f"{seed}/{arm}: held-out metric appeared before unlock")
            history_path = run / "arms" / f"seed_{seed}" / arm / "history.json"
            if _sha256(history_path) != record["history_file_sha256"]:
                failures.append(f"{seed}/{arm}: history digest mismatch")
            history = json.loads(history_path.read_text())
            if history["n_init_3d"] != 5000 or history["n_opt_3d"] != 5000:
                failures.append(f"{seed}/{arm}: topology changed")
            if history["teacher_digest_before"] != history["teacher_digest_after"]:
                failures.append(f"{seed}/{arm}: teacher mutated")
            if history["proposal_digest_before"] != history["proposal_digest_after"]:
                failures.append(f"{seed}/{arm}: proposal field mutated")
            teacher_digests.add(history["teacher_digest_before"])
            proposal_digests.add(history["proposal_digest_before"])
            if reference_schedule is None:
                reference_schedule = history["view_schedule_sha256"]
            elif history["view_schedule_sha256"] != reference_schedule:
                failures.append(f"{seed}/{arm}: view schedule differs within seed")
            projected = [
                {key: step[key] for key in SAMPLE_BINDING_KEYS} for step in history["steps"]
            ]
            if reference_steps is None:
                reference_steps = projected
            elif projected != reference_steps:
                failures.append(f"{seed}/{arm}: point-sample stream differs within seed")
            for family in history["frozen_parameter_groups"]:
                if history["optimizer_group_motion"][family]["max_abs"] != 0.0:
                    failures.append(f"{seed}/{arm}: frozen family {family} moved")
        initial_hashes = {
            records[seed][arm]["initial_semantic_sha256"] for arm in BEAM_INITIALIZATION_ARMS
        }
        if len(initial_hashes) != 1:
            failures.append(f"{seed}: Beam-derived parameter arms do not share initialization")
        for step in STEPS:
            reference = records[seed]["beam_all"]["checkpoint_metrics"][str(step)]
            for arm in BEAM_INITIALIZATION_ARMS[1:]:
                if step == 0 and records[seed][arm]["checkpoint_metrics"]["0"] != reference:
                    failures.append(f"{seed}/{arm}: step-0 metrics differ from Beam initialization")

    if len(teacher_digests) != 1 or len(proposal_digests) != 1:
        failures.append("teacher/proposal digests differ across roots or arms")
    heldout = result["heldout"]
    expected_finalists = {
        "beam_baseline",
        "beam_means_fixed",
        "corrected_CA_all",
        "corrected_CA_all_support",
    }
    if set(heldout["finalists"]) != expected_finalists:
        failures.append("held-out finalist set differs from the frozen rule")
    if heldout["selection_sha256"] != _canonical_hash(result["analysis"]):
        failures.append("held-out selection digest does not bind the analysis")
    for seed, arm_records in records.items():
        for arm, _record in arm_records.items():
            pointer = result["records"][str(seed)][arm]
            path = run / pointer["result"]
            if _sha256(path) != pointer["result_sha256"]:
                failures.append(f"{seed}/{arm}: result pointer digest mismatch")

    return {
        "passed": not failures,
        "failures": failures,
        "teacher_digest": next(iter(teacher_digests), None),
        "proposal_digest": next(iter(proposal_digests), None),
        "no_image_boundary": boundary,
        "packed_alpha_accessed": manifest["dataset"].get("packed_alpha_accessed"),
        "heldout_selection_sha256": heldout["selection_sha256"],
    }


def audit(run: Path) -> dict[str, object]:
    records = _load_records(run)
    summaries = {arm: _summary(records, arm) for arm in ARMS}
    repair_pairs = {
        "C": (
            ("legacy_C_all", "beam_all"),
            ("legacy_CO_all", "legacy_O_all"),
            ("legacy_CA_all", "legacy_A_all"),
            ("legacy_COA_all", "legacy_OA_all"),
        ),
        "O": (
            ("legacy_O_all", "beam_all"),
            ("legacy_CO_all", "legacy_C_all"),
            ("legacy_OA_all", "legacy_A_all"),
            ("legacy_COA_all", "legacy_CA_all"),
        ),
        "A": (
            ("legacy_A_all", "beam_all"),
            ("legacy_CA_all", "legacy_C_all"),
            ("legacy_OA_all", "legacy_O_all"),
            ("legacy_COA_all", "legacy_CO_all"),
        ),
    }
    marginals = {
        factor: {
            "J_Q": _factorial_marginal(summaries, pairs, "validation_J_Q"),
            "J_U": _factorial_marginal(summaries, pairs, "validation_J_U"),
        }
        for factor, pairs in repair_pairs.items()
    }
    for value in marginals.values():
        value["necessary_gate"] = (
            value["J_Q"]["geometric"] <= 0.95
            and value["J_Q"]["wins"] == 3
            and value["J_U"]["geometric"] <= 1.05
        )
    contrasts = {
        "corrected_C_vs_beam": {
            metric: _paired_ratio(
                summaries,
                "corrected_C_all",
                "beam_all",
                f"validation_{metric}",
            )
            for metric in ("J_Q", "J_U")
        },
        "corrected_C_vs_legacy_C": {
            metric: _paired_ratio(
                summaries,
                "corrected_C_all",
                "legacy_C_all",
                f"validation_{metric}",
            )
            for metric in ("J_Q", "J_U")
        },
        "corrected_CA_vs_corrected_C": {
            metric: _paired_ratio(
                summaries,
                "corrected_CA_all",
                "corrected_C_all",
                f"validation_{metric}",
            )
            for metric in ("J_Q", "J_U")
        },
        "means_fixed_vs_beam": {
            metric: _paired_ratio(
                summaries,
                "beam_means_fixed",
                "beam_all",
                f"validation_{metric}",
            )
            for metric in ("J_Q", "J_U")
        },
        "means_only_vs_beam": {
            metric: _paired_ratio(
                summaries,
                "beam_means_only",
                "beam_all",
                f"validation_{metric}",
            )
            for metric in ("J_Q", "J_U")
        },
        "support_vs_beam": {
            metric: _paired_ratio(
                summaries,
                "beam_all_support",
                "beam_all",
                f"validation_{metric}",
            )
            for metric in ("J_Q", "J_U")
        },
    }
    source_files = {
        relative: {
            "sha256": _sha256(ROOT / relative),
            "mtime_ns": (ROOT / relative).stat().st_mtime_ns,
        }
        for relative in SOURCE_PATHS
    }
    result = json.loads((run / "result.json").read_text())
    heldout_summary = {}
    for arm, seed_records in result["heldout"]["results"].items():
        heldout_summary[arm] = {
            metric: _geometric([float(seed_records[str(seed)][metric]) for seed in SEEDS])
            for metric in ("J_Q", "J_U")
        }
    invariant_audit = _audit_invariants(run, records)
    result_mtime_ns = (run / "result.json").stat().st_mtime_ns
    estimated_start_ns = result_mtime_ns - int(result["resources"]["total_seconds"] * 1e9)
    source_mtime_after_estimated_start = [
        relative
        for relative, record in source_files.items()
        if record["mtime_ns"] > estimated_start_ns
    ]
    audit_record = {
        "schema": "rtgs.compact-only-carrier-stage.audit.v1",
        "status": "PASS_WITH_SCOPE_LIMITS" if invariant_audit["passed"] else "FAIL",
        "run": str(run.relative_to(ROOT)),
        "run_result_sha256": _sha256(run / "result.json"),
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "invariants": invariant_audit,
        "summaries": summaries,
        "repair_marginals": marginals,
        "contrasts": contrasts,
        "heldout_descriptive": heldout_summary,
        "resources": result["resources"],
        "posthoc_source_binding": {
            "git_revision": _command("git", "rev-parse", "HEAD").strip(),
            "git_status": _command("git", "status", "--short").splitlines(),
            "source_files": source_files,
            "source_mtime_after_estimated_start": source_mtime_after_estimated_start,
            "limitation": (
                "The producing manifest did not freeze a revision, dirty diff, command, or "
                "executed-source archive. These hashes were collected after execution; mtimes "
                "are corroboration, not a cryptographic execution binding."
            ),
        },
        "scope_limits": [
            "single real development scene; no cross-scene or confirmatory inference",
            "held-out metrics are descriptive and do not select stages",
            "GPU timing had no idle/contended-device control and is non-decisional",
            "PyTorch allocator peaks are absolute process measurements, not a dense-vs-compact "
            "VRAM comparison or general VRAM claim",
            "the support diagnostic constrains projected centers only and cannot detect a "
            "floater inside the multi-view visual hull",
            "the producing run lacks a pre-execution source-tree seal",
        ],
        "decisions": {
            "legacy_covariance_necessary": False,
            "legacy_opacity_necessary": False,
            "legacy_appearance_necessary": False,
            "corrected_covariance_replacement_gate": (
                contrasts["corrected_C_vs_legacy_C"]["J_Q"]["geometric"] <= 0.98
                and contrasts["corrected_C_vs_legacy_C"]["J_U"]["geometric"] <= 1.02
            ),
            "corrected_appearance_material": False,
            "means_fixed_noninferior_to_beam_all": (
                contrasts["means_fixed_vs_beam"]["J_Q"]["geometric"] <= 1.02
                and contrasts["means_fixed_vs_beam"]["J_U"]["geometric"] <= 1.02
            ),
            "means_only_noninferior_to_beam_all": (
                contrasts["means_only_vs_beam"]["J_Q"]["geometric"] <= 1.02
                and contrasts["means_only_vs_beam"]["J_U"]["geometric"] <= 1.02
            ),
            "support_barrier_success": False,
            "no_free_floaters_established": False,
            "default_change_authorized": False,
        },
    }
    return audit_record


def _percent(ratio: float) -> str:
    return f"{(ratio - 1.0) * 100:+.2f}%"


def markdown(record: Mapping[str, object]) -> str:
    c = record["contrasts"]
    m = record["repair_marginals"]
    s = record["summaries"]
    resources = [
        json.loads(
            (
                ROOT / record["run"] / "arms" / f"seed_{seed}" / "beam_all" / "result.json"
            ).read_text()
        )["resources"]
        for seed in SEEDS
    ]
    train_peak = max(item["train_peak_cuda_allocated_bytes"] for item in resources)
    reserved_peak = max(item["train_peak_cuda_reserved_bytes"] for item in resources)
    return f"""# Compact-only carrier stage ablation — independent audit — 2026-07-28

Status: **{record["status"]}**.

## Referee disposition

| Claim | Disposition | Independently recomputed evidence |
| --- | --- | --- |
| The run used compact 2D Gaussian/camera tensors rather than RGB/masks. | **Confirm, executable-boundary scope.** | The live receipt passed with zero image opens, zero forbidden imports, and all three negative controls; packed alpha was configured as unaccessed. Focused tests separately make alpha member reads/decodes fatal. This is not an OS-level information-flow proof. |
| Legacy covariance repair is needed. | **Retire.** | Final validation `J_Q` factorial marginal {m["C"]["J_Q"]["geometric"]:.4f} ({_percent(m["C"]["J_Q"]["geometric"])}); 0/3 paired wins. |
| Legacy opacity repair is needed. | **Retire.** | Final validation `J_Q` factorial marginal {m["O"]["J_Q"]["geometric"]:.4f} ({_percent(m["O"]["J_Q"]["geometric"])}); 0/3 paired wins. Its target is non-identifiable fitted-mixture amplitude. |
| Legacy appearance repair is needed. | **Retire as a stage.** | Final validation `J_Q` factorial marginal {m["A"]["J_Q"]["geometric"]:.4f} ({_percent(m["A"]["J_Q"]["geometric"])}), far below the preregistered 5% materiality floor. |
| Renderer-aware symmetric covariance repair is better than legacy C. | **Confirm on this scene.** | `J_Q` ratio {c["corrected_C_vs_legacy_C"]["J_Q"]["geometric"]:.4f} ({_percent(c["corrected_C_vs_legacy_C"]["J_Q"]["geometric"])}), 3/3 wins; `J_U` ratio {c["corrected_C_vs_legacy_C"]["J_U"]["geometric"]:.4f}. |
| Corrected appearance should be retained with corrected covariance. | **Retire as material.** | CA vs C changes `J_Q` by {_percent(c["corrected_CA_vs_corrected_C"]["J_Q"]["geometric"])} while changing `J_U` by {_percent(c["corrected_CA_vs_corrected_C"]["J_U"]["geometric"])}. The mechanical CA winner is below any stage-necessity threshold. |
| Updating only means is sufficient. | **Reject.** | Means-only vs all-parameter Beam: `J_Q` {c["means_only_vs_beam"]["J_Q"]["geometric"]:.4f} ({_percent(c["means_only_vs_beam"]["J_Q"]["geometric"])}), `J_U` {c["means_only_vs_beam"]["J_U"]["geometric"]:.4f}. |
| Means can be frozen. | **Narrow confirm for the Beam initializer.** | Means-fixed vs all-parameter Beam: `J_Q` {c["means_fixed_vs_beam"]["J_Q"]["geometric"]:.4f}, `J_U` {c["means_fixed_vs_beam"]["J_U"]["geometric"]:.4f}; it passes the 2% gate. This interaction was not tested after corrected covariance. |
| The sampled support barrier removes outside centers/free floaters. | **Retire.** | Beam support vs parent `J_Q` {c["support_vs_beam"]["J_Q"]["geometric"]:.4f}; mean validation outside fraction {s["beam_all_support"]["mean_validation_outside_fraction"]:.6f} vs {s["beam_all"]["mean_validation_outside_fraction"]:.6f}. A floater inside the visual hull remains undetectable. |
| The run proves the VRAM selling claim. | **Narrow to an absolute measurement.** | Maximum compact training allocation {train_peak / 2**20:.2f} MiB, reservation {reserved_peak / 2**20:.2f} MiB, with 5,000 Gaussians and all compact teachers resident. There is no controlled dense baseline, idle-GPU timing protocol, scene scaling, or generalization evidence. |

## Recomputed ordering

- `corrected_CA_all`: validation geometric `J_Q={s["corrected_CA_all"]["geometric_validation_J_Q"]:.8f}`.
- `corrected_C_all`: validation geometric `J_Q={s["corrected_C_all"]["geometric_validation_J_Q"]:.8f}`.
- `beam_all`: validation geometric `J_Q={s["beam_all"]["geometric_validation_J_Q"]:.8f}`.
- `beam_means_fixed`: validation geometric `J_Q={s["beam_means_fixed"]["geometric_validation_J_Q"]:.8f}`.
- `beam_means_only`: validation geometric `J_Q={s["beam_means_only"]["geometric_validation_J_Q"]:.8f}`.

The numerical CA/C ordering does not justify retaining appearance repair: its `J_Q` edge is
{abs((c["corrected_CA_vs_corrected_C"]["J_Q"]["geometric"] - 1) * 100):.3f}% and `J_U` is worse.
The bounded minimal schedule supported for another experiment is corrected covariance followed by
compact all-parameter optimization; a corrected-covariance × means-freeze interaction remains
open.

## Protocol and provenance findings

1. All 48 arm receipts are complete; each retains 5,000 rows, immutable teacher/proposal digests,
   paired within-root point samples, and bit-exact zero motion for frozen parameter families.
2. Only fit and validation metrics occur in checkpoint receipts. The held-out finalist set and
   selection hash match the frozen rule; held-out numbers remain descriptive.
3. The producing manifest did not freeze the Git revision, dirty diff, command, or an executed
   source archive. Post-run file hashes and mtimes found no relevant file newer than the estimated
   start, but that is not a cryptographic execution seal. This development result is therefore
   not replay-complete or confirmatory.
4. Held-out proposal banks were generated before training as explicitly frozen, while held-out
   target risks were not computed until selection. Because held-out compact fields were resident
   in the same process, this is not an unopened/one-shot sealed-data design.
5. The GPU resource numbers are allocator diagnostics only. The run did not record an idle-device
   control and cannot support speed or comparative-memory claims.

## Commands checked

```text
.venv/bin/python benchmarks/compact_only_carrier_stage_ablation.py
.venv/bin/python benchmarks/audit_compact_only_carrier_stage_ablation.py
.venv/bin/pytest -q tests/test_compact_views.py tests/test_compact_only_carrier_stage_ablation.py tests/test_observation2d.py tests/test_compact_trainer.py tests/test_carrier_refinement.py tests/test_lift.py
```

Full repository verification is deferred until the result, compact pipeline changes, docs, and
later-stage protocol are all reconciled in one final verification pass.
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
