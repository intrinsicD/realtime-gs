#!/usr/bin/env python3
"""Resume-safe operator for the full-frame compact-initializer convergence suite.

Each arm uses :mod:`benchmarks.full_compact_reconstruction` for the scientific work. This
wrapper fixes arm order, refuses to overwrite partial artifacts, and advances a successful arm
through the same 30k parent and 10k fixed-topology continuation phases until the frozen joint
plateau rule fires or the 70k ceiling is reached. It never opens source RGB.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "benchmarks/full_compact_reconstruction.py"
DEFAULT_PROTOCOL = ROOT / "benchmarks/results/20260721_all_initializers_frame00008_PREREG.md"
DEFAULT_OUT = ROOT / "runs/all_initializers_frame00008_20260721"
ARMS = ("topk", "dense-merge", "easy-only", "splat-sfm", "field", "random")
CONTINUATIONS = (
    ("polish", "polish_30000_40000"),
    ("tail", "tail_40000_50000"),
    ("cooldown", "cooldown_50000_60000"),
    ("settle", "settle_60000_70000"),
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_status(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _completed(directory: Path) -> bool:
    return (directory / "fit_complete.json").is_file()


def _run(command: list[str], *, dry_run: bool) -> None:
    print("[suite-command]", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def _parent_command(
    python: str,
    arm: str,
    out: Path,
    protocol: Path,
) -> list[str]:
    return [
        python,
        str(HARNESS),
        "--phase",
        "fit",
        "--initializer",
        arm,
        "--out",
        str(out),
        "--preregistration",
        str(protocol),
        "--fit-mode",
        "all",
        "--max-tracks",
        "5000",
        "--depth-samples",
        "32",
        "--min-views",
        "2",
        "--robust-view-fraction",
        "0.60",
        "--min-placement-score",
        "0.01",
        "--init-opacity",
        "0.10",
        "--dense-merge-voxel",
        "0.06",
        "--sfm-min-views",
        "2",
        "--sfm-source-chunk",
        "256",
        "--field-max-tracks",
        "128",
        "--field-max-train-views",
        "26",
        "--iterations",
        "30000",
        "--eval-every",
        "1000",
        "--density-strategy",
        "gsplat-default",
        "--densify-start",
        "500",
        "--densify-stop",
        "15000",
        "--densify-every",
        "100",
        "--max-gaussians",
        "100000",
        "--prune-opacity",
        "0.005",
        "--prune-scale-frac",
        "0.1",
        "--seed",
        "0",
    ]


def _continuation_command(
    python: str,
    phase: str,
    parent: Path,
    out: Path,
) -> list[str]:
    return [
        python,
        str(HARNESS),
        "--phase",
        phase,
        "--parent-out",
        str(parent),
        "--out",
        str(out),
    ]


def _record_directory(directory: Path) -> dict:
    fit = _read_json(directory / "fit_complete.json")
    result = {
        "directory": str(directory),
        "fit_complete_sha256": _sha256(directory / "fit_complete.json"),
        "n_final_gaussians": fit["n_final_gaussians"],
        "final_ply_sha256": fit["final_ply_sha256"],
    }
    selection_path = directory / "model_selection.json"
    if selection_path.is_file():
        selection = _read_json(selection_path)
        result["model_selection_sha256"] = _sha256(selection_path)
        result["selected_global_step"] = selection["selected"]["global_step"]
        result["joint_status"] = selection["convergence"]["joint_status"]
    return result


def run(args: argparse.Namespace) -> int:
    protocol = args.protocol.resolve(strict=True)
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    arms = tuple(args.arms or ARMS)
    status_path = out_root / "suite_status.json"
    status = {
        "schema": "rtgs.compact_initializer_suite.operator.v1",
        "created_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "protocol": {"path": str(protocol), "sha256": _sha256(protocol)},
        "harness": {"path": str(HARNESS), "sha256": _sha256(HARNESS)},
        "arm_order": list(arms),
        "source_rgb_opened": False,
        "arms": {},
    }
    if status_path.is_file():
        previous = _read_json(status_path)
        if previous.get("protocol", {}).get("sha256") != status["protocol"]["sha256"]:
            raise RuntimeError("existing suite status is bound to a different protocol")
        status["created_utc"] = previous.get("created_utc", status["created_utc"])
        status["arms"] = previous.get("arms", {})
    _write_status(status_path, status)

    for arm in arms:
        arm_root = out_root / arm
        arm_root.mkdir(exist_ok=True)
        parent = arm_root / "fit_0_30000"
        arm_status = status["arms"].setdefault(arm, {"state": "starting", "phases": []})
        try:
            if not _completed(parent):
                if parent.exists():
                    raise RuntimeError(f"partial parent exists; recover explicitly: {parent}")
                _run(_parent_command(args.python, arm, parent, protocol), dry_run=args.dry_run)
            if args.dry_run:
                continue
            arm_status["phases"] = [_record_directory(parent)]
            current = parent
            for phase, name in CONTINUATIONS:
                continuation = arm_root / name
                if not _completed(continuation):
                    if continuation.exists():
                        raise RuntimeError(
                            f"partial continuation exists; inspect before resuming: {continuation}"
                        )
                    _run(
                        _continuation_command(args.python, phase, current, continuation),
                        dry_run=False,
                    )
                record = _record_directory(continuation)
                arm_status["phases"].append(record)
                current = continuation
                status["updated_utc"] = _utc_now()
                _write_status(status_path, status)
                if record["joint_status"] == "plateau":
                    break
                if record["joint_status"] != "still_improving":
                    raise RuntimeError(
                        f"unexpected convergence status for {arm}/{phase}: "
                        f"{record['joint_status']!r}"
                    )
            arm_status.update(
                {
                    "state": "complete",
                    "completed_utc": _utc_now(),
                    "terminal_directory": str(current),
                    "terminal": arm_status["phases"][-1],
                }
            )
        except Exception as error:
            arm_status.update(
                {
                    "state": "failed",
                    "failed_utc": _utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            status["updated_utc"] = _utc_now()
            _write_status(status_path, status)
            print(f"[suite] {arm} failed: {error}", file=sys.stderr, flush=True)
            if not args.keep_going:
                raise
        status["updated_utc"] = _utc_now()
        _write_status(status_path, status)

    if not args.dry_run:
        status["completed_utc"] = _utc_now()
        status["all_requested_arms_terminal"] = all(
            status["arms"].get(arm, {}).get("state") in {"complete", "failed"} for arm in arms
        )
        _write_status(status_path, status)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--arms", nargs="*", choices=ARMS)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
