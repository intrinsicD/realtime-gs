#!/usr/bin/env python3
"""Build post-hoc no-preview handoffs for the compact carrier metric experiments.

The scientific results and audits predate these reader artifacts.  This script derives one
representative initialization/final PLY pair and its displayed metrics from the already frozen
NPZ/JSON outputs, then exercises both the results page and ``rtgs view``.  It does not modify the
audited result, analysis, manifest, arm, bank, or model files.
"""

from __future__ import annotations

import hashlib
import html
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from rtgs.core.gaussians3d import Gaussians3D

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class HandoffSpec:
    run: str
    title: str
    initialization: str
    final: str
    arm_result: str
    arm_history: str
    result_note: str
    audit_note: str
    page_port: int
    viewer_port: int


SPECS = (
    HandoffSpec(
        run="runs/compact_only_carrier_stage_ablation_20260728",
        title="Compact-only carrier stage ablation",
        initialization="repairs/corrected_C.npz",
        final="arms/seed_280701/corrected_C_all/gaussians_final.npz",
        arm_result="arms/seed_280701/corrected_C_all/result.json",
        arm_history="arms/seed_280701/corrected_C_all/history.json",
        result_note=("benchmarks/results/20260728_compact_only_carrier_stage_ablation_RESULT.md"),
        audit_note=("benchmarks/results/20260728_compact_only_carrier_stage_ablation_AUDIT.md"),
        page_port=8891,
        viewer_port=8991,
    ),
    HandoffSpec(
        run="runs/compact_only_carrier_policy_closure_20260728",
        title="Compact-only carrier policy closure",
        initialization="corrected_C_initialization.npz",
        final="arms/seed_282701/continue_means_fixed_380/gaussians_final.npz",
        arm_result="arms/seed_282701/continue_means_fixed_380/result.json",
        arm_history="arms/seed_282701/continue_means_fixed_380/history.json",
        result_note=("benchmarks/results/20260728_compact_only_carrier_policy_closure_RESULT.md"),
        audit_note=("benchmarks/results/20260728_compact_only_carrier_policy_closure_AUDIT.md"),
        page_port=8892,
        viewer_port=8992,
    ),
    HandoffSpec(
        run="runs/compact_only_carrier_sequence_interaction_20260728",
        title="Compact-only carrier sequence interaction",
        initialization=(
            "../compact_only_carrier_policy_closure_20260728/"
            "arms/seed_282701/corrected_C_all_380/gaussians_final.npz"
        ),
        final="arms/seed_282701/prune_then_phase2_d0/gaussians_final.npz",
        arm_result="arms/seed_282701/prune_then_phase2_d0/result.json",
        arm_history="arms/seed_282701/prune_then_phase2_d0/history.json",
        result_note=(
            "benchmarks/results/20260728_compact_only_carrier_sequence_interaction_RESULT.md"
        ),
        audit_note=(
            "benchmarks/results/20260728_compact_only_carrier_sequence_interaction_AUDIT.md"
        ),
        page_port=8893,
        viewer_port=8993,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _relative_from_run(run: Path, target: Path) -> str:
    return target.resolve().relative_to(ROOT).as_posix()


def _wait_http(url: str, process: subprocess.Popen[str], timeout: float = 15.0) -> int:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"server exited before {url}: stdout={stdout!r}, stderr={stderr!r}")
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                response.read(256)
                return int(response.status)
        except Exception as error:  # noqa: BLE001 - retain the final network failure
            last_error = error
            time.sleep(0.1)
    raise RuntimeError(f"timed out waiting for {url}: {last_error}")


def _stop(process: subprocess.Popen[str]) -> tuple[int, str, str]:
    process.terminate()
    try:
        stdout, stderr = process.communicate(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
    return int(process.returncode), stdout, stderr


def _write_handoff(spec: HandoffSpec) -> None:
    run = ROOT / spec.run
    if not run.is_dir():
        raise FileNotFoundError(run)
    initialization_path = (run / spec.initialization).resolve()
    final_path = (run / spec.final).resolve()
    arm_result_path = run / spec.arm_result
    arm_history_path = run / spec.arm_history
    for path in (initialization_path, final_path, arm_result_path, arm_history_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    arm = _load_json(arm_result_path)
    iterations = int(arm["config"]["iterations"])
    final_metrics = arm["checkpoint_metrics"][str(iterations)]["validation"]
    initialization = Gaussians3D.load_npz(initialization_path)
    final = Gaussians3D.load_npz(final_path)
    initialization.save_ply(run / "gaussians_init.ply")
    final.save_ply(run / "gaussians.ply")

    metrics = {
        "metrics": {
            "representative_validation_J_Q": float(final_metrics["J_Q"]),
            "representative_validation_J_U": float(final_metrics["J_U"]),
            "representative_initial_gaussians": initialization.n,
            "representative_final_gaussians": final.n,
        },
        "scope": (
            "Post-hoc reader handoff for one preregistered representative root; the audited "
            "aggregate result.json and RESULT/AUDIT notes remain authoritative."
        ),
        "representative_arm": arm["arm"],
        "representative_seed": int(arm["seed"]),
        "source_arm_result": spec.arm_result,
        "source_final_npz": spec.final,
    }
    (run / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(arm_history_path, run / "training_history.json")
    config = {
        "training": {"packed": False, "antialiased": False},
        "handoff": {
            "post_hoc": True,
            "no_previews": True,
            "scientific_artifacts_unchanged": True,
            "representative_arm": arm["arm"],
            "representative_seed": int(arm["seed"]),
            "source_initialization": _relative_from_run(run, initialization_path),
            "source_final": _relative_from_run(run, final_path),
            "source_arm_result": spec.arm_result,
        },
    }
    (run / "gaussians.config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    result_link = Path("../..") / spec.result_note
    audit_link = Path("../..") / spec.audit_note
    metric_values = metrics["metrics"]
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>{html.escape(spec.title)}</title>
<style>
body {{ font: 16px/1.5 system-ui, sans-serif; max-width: 880px; margin: 3rem auto;
padding: 0 1rem; }}
code {{ background: #eee; padding: .1rem .25rem; }}
table {{ border-collapse: collapse; }} th, td {{ border: 1px solid #bbb; padding: .4rem .7rem; }}
</style>
<h1>{html.escape(spec.title)}</h1>
<p><strong>Post-hoc no-preview reader handoff.</strong> The scientific run, result, and audit
precede this page. This page selects one frozen root for model inspection and does not alter the
audited aggregate decision.</p>
<table>
<tr><th>Representative arm</th><td>{html.escape(str(arm["arm"]))}</td></tr>
<tr><th>Seed</th><td>{int(arm["seed"])}</td></tr>
<tr><th>Validation J_Q</th><td>{metric_values["representative_validation_J_Q"]:.8f}</td></tr>
<tr><th>Validation J_U</th><td>{metric_values["representative_validation_J_U"]:.8f}</td></tr>
<tr><th>Initial / final rows</th><td>{initialization.n} / {final.n}</td></tr>
</table>
<p>
<a href="gaussians_init.ply">initial PLY</a> ·
<a href="gaussians.ply">final PLY</a> ·
<a href="metrics.json">metrics.json</a> ·
<a href="training_history.json">training history</a> ·
<a href="gaussians.config.json">handoff config</a> ·
<a href="{result_link.as_posix()}">result note</a> ·
<a href="{audit_link.as_posix()}">independent audit</a>
</p>
<p>Claim boundary: single-scene compact-field development evidence; no general quality, VRAM,
runtime, physical no-floater, or production claim.</p>
</html>
"""
    (run / "index.html").write_text(page, encoding="utf-8")

    page_command = [
        sys.executable,
        "-m",
        "http.server",
        str(spec.page_port),
        "--bind",
        "127.0.0.1",
        "--directory",
        str(run),
    ]
    page_process = subprocess.Popen(
        page_command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    page_url = f"http://127.0.0.1:{spec.page_port}/index.html"
    page_status = _wait_http(page_url, page_process)
    page_returncode, page_stdout, page_stderr = _stop(page_process)

    viewer_command = [
        str(ROOT / ".venv/bin/rtgs"),
        "view",
        "--gaussians",
        str(run / "gaussians.ply"),
        "--initial",
        str(run / "gaussians_init.ply"),
        "--max-viewer-gaussians",
        "5000",
        "--host",
        "127.0.0.1",
        "--port",
        str(spec.viewer_port),
        "--no-open",
    ]
    viewer_process = subprocess.Popen(
        viewer_command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    viewer_url = f"http://127.0.0.1:{spec.viewer_port}/"
    viewer_status = _wait_http(viewer_url, viewer_process)
    viewer_returncode, viewer_stdout, viewer_stderr = _stop(viewer_process)

    receipt = {
        "schema": "rtgs.compact-carrier-posthoc-handoff.v1",
        "post_hoc": True,
        "scientific_artifacts_unchanged": True,
        "preview_policy": "no-previews",
        "index.html": {
            "command": " ".join(page_command),
            "url": page_url,
            "status": page_status,
            "server_returncode_after_terminate": page_returncode,
            "stdout": page_stdout,
            "stderr": page_stderr,
            "sha256": _sha256(run / "index.html"),
        },
        "rtgs view": {
            "command": " ".join(viewer_command),
            "url": viewer_url,
            "status": viewer_status,
            "server_returncode_after_terminate": viewer_returncode,
            "stdout": viewer_stdout,
            "stderr": viewer_stderr,
        },
        "artifacts": {
            name: {"bytes": (run / name).stat().st_size, "sha256": _sha256(run / name)}
            for name in (
                "gaussians_init.ply",
                "gaussians.ply",
                "metrics.json",
                "training_history.json",
                "gaussians.config.json",
                "index.html",
            )
        },
    }
    (run / "smoke_receipt.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"prepared and smoked {run.relative_to(ROOT)}")


def main() -> int:
    for spec in SPECS:
        _write_handoff(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
