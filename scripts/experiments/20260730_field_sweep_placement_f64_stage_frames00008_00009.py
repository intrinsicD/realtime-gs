#!/usr/bin/env python3
"""Run the float64-repaired successor to the fixed-anchor field-sweep experiment.

The consumed 20260729 driver remains unchanged. This successor reuses its frozen orchestration
and treatment logic, changes only the common field-lift compute dtype to float64, and adds a
structured worker-failure receipt. The new task id, digest, source lock, and prospective review
keep the two official attempts forensically separate.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TASK_ID = "20260730_field_sweep_placement_f64_stage_frames00008_00009"
ROOT = Path(__file__).resolve().parents[2]
TASK_RELATIVE = Path("experiments/tasks") / f"{TASK_ID}.json"
RUN_RELATIVE = Path("runs") / TASK_ID
_BASE_DRIVER = (
    ROOT / "scripts/experiments/20260729_field_sweep_placement_stage_frames00008_00009.py"
)
_SPEC = importlib.util.spec_from_file_location("_rtgs_field_sweep_20260729", _BASE_DRIVER)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load the frozen base driver: {_BASE_DRIVER}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

# The base module reads these globals at call time. Point worker subprocesses back to this wrapper
# so every successor process reinstalls the repaired task identity and compute dtype.
_IMPL.__file__ = __file__
_IMPL.TASK_ID = TASK_ID
_IMPL.TASK_RELATIVE = TASK_RELATIVE
_IMPL.RUN_RELATIVE = RUN_RELATIVE
_IMPL.FIELD_CONFIG = {**_IMPL.FIELD_CONFIG, "compute_dtype": "float64"}
_BASE_WORKER = _IMPL._worker

ARM_MODES = _IMPL.ARM_MODES
ARM_ORDER = _IMPL.ARM_ORDER
COMPACT_LOADING = _IMPL.COMPACT_LOADING
FIELD_CONFIG = _IMPL.FIELD_CONFIG
REFIT_CONFIG = _IMPL.REFIT_CONFIG
NoImageGuard = _IMPL.NoImageGuard
_protocol_sha256 = _IMPL._protocol_sha256
_sha256_file = _IMPL._sha256_file
_validate_run_binding = _IMPL._validate_run_binding


def _failure_root(
    run: Path,
    *,
    dataset_id: str,
    seed: int,
    arm: str,
    warmup: bool,
) -> Path:
    output = run / _IMPL._cell_relative(dataset_id, seed, arm, warmup)
    if output.is_dir():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        output.parent.glob(f".{output.name}.worker-*"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if candidates:
        return candidates[-1]
    return Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.worker-failed-{os.getpid()}.",
            dir=output.parent,
        )
    )


def _worker(*, task: dict[str, Any], run: Path, **cell: Any) -> None:
    try:
        _BASE_WORKER(task=task, run=run, **cell)
    except Exception as error:
        root = _failure_root(run, **cell)
        receipt = root / "failure.json"
        if not receipt.exists():
            _IMPL._write_json_new(
                receipt,
                {
                    "schema_version": 1,
                    "task_id": TASK_ID,
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset_id": cell["dataset_id"],
                    "seed": cell["seed"],
                    "arm": cell["arm"],
                    "warmup": cell["warmup"],
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        raise


_IMPL._worker = _worker


def main() -> None:
    _IMPL.main()


if __name__ == "__main__":
    main()
