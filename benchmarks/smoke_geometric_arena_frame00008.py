#!/usr/bin/env python3
"""Operational HTTP smokes for the geometric-arena result viewer and index."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import quote, urldefrag, urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/geometric_arena_frame00008_20260724"
INDEX = RUN / "index.html"
MANIFEST = ROOT / "benchmarks/results/20260724_geometric_arena_frame00008_VIEWER.json"
VIEWER_RECEIPT = ROOT / "benchmarks/results/20260724_geometric_arena_frame00008_VIEWER_RECEIPT.json"
INDEX_RECEIPT = ROOT / "benchmarks/results/20260724_geometric_arena_frame00008_INDEX_RECEIPT.json"
SCENE = Path("/home/alex/Dropbox/Work/Janelle/2025_03_07_stage_with_fabric/frame_00008")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _fetch(url: str, *, timeout: float = 5.0) -> tuple[int, bytes]:
    with urlopen(url, timeout=timeout) as response:  # noqa: S310 - loopback-only smoke
        return int(response.status), response.read()


def _wait_for_http(url: str, process: subprocess.Popen[str]) -> tuple[int, bytes]:
    deadline = time.monotonic() + 60.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited before HTTP readiness with code {process.returncode}"
            )
        try:
            return _fetch(url)
        except (OSError, URLError) as error:
            last_error = error
            time.sleep(0.2)
    raise TimeoutError(f"server did not become ready: {last_error}")


def _pid_owns_port(pid: int, port: int) -> bool:
    try:
        output = subprocess.check_output(
            ["ss", "-ltnp", f"sport = :{port}"],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return f"pid={pid}," in output


def _port_closed(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(1.0)
        return client.connect_ex(("127.0.0.1", port)) != 0


def _stop(process: subprocess.Popen[str], port: int) -> tuple[bool, bool]:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    return process.poll() is not None, _port_closed(port)


def _compute_processes() -> list[str]:
    command = (
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    )
    try:
        output = subprocess.check_output(command, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line for line in output.splitlines() if line.strip()]


def _rss_kib(pid: int) -> int | None:
    status = Path(f"/proc/{pid}/status")
    if not status.is_file():
        return None
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None


def _viewer(port: int) -> int:
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    methods = manifest.get("methods")
    if manifest.get("schema") != "rtgs.viewer-comparison.v1" or not isinstance(methods, list):
        raise RuntimeError("viewer manifest schema changed")
    models = []
    for record in methods:
        initial = (MANIFEST.parent / record["initial"]).resolve()
        final = (MANIFEST.parent / record["final"]).resolve()
        models.append(
            {
                "name": record["name"],
                "initial_sha256": _sha256(initial),
                "final_sha256": _sha256(final),
            }
        )

    command = [
        str(ROOT / ".venv-cuda/bin/rtgs"),
        "view",
        "--comparison-manifest",
        str(MANIFEST.relative_to(ROOT)),
        "--scene",
        str(SCENE),
        "--downscale",
        "16",
        "--device",
        "cpu",
        "--max-viewer-gaussians",
        "20000",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-open",
    ]
    display_command = "CUDA_VISIBLE_DEVICES='' " + " ".join(
        [f".venv-cuda/bin/{Path(command[0]).name}", *command[1:]]
    )
    log_path = RUN / "viewer_smoke.log"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        url = f"http://127.0.0.1:{port}/"
        try:
            status, payload = _wait_for_http(url, process)
            owned = _pid_owns_port(process.pid, port)
            rss = _rss_kib(process.pid)
            compute_processes = _compute_processes()
        finally:
            stopped, closed = _stop(process, port)
    receipt = {
        "schema": "rtgs.geometric_arena_frame00008.viewer_receipt.v1",
        "checked_utc": dt.datetime.now(dt.UTC).isoformat(),
        "command": display_command,
        "manifest": {
            "path": str(MANIFEST.relative_to(ROOT)),
            "sha256": _sha256(MANIFEST),
            "method_count": len(methods),
            "model_count": len(methods) * 2,
        },
        "models": models,
        "server": {
            "pid": process.pid,
            "host": "127.0.0.1",
            "port": port,
            "pid_owned_listening_socket": owned,
            "http_status": status,
            "response_bytes": len(payload),
            "response_sha256": hashlib.sha256(payload).hexdigest(),
            "rss_kib_at_check": rss,
            "cuda_visible_devices": "",
            "nvidia_compute_processes_at_check": compute_processes,
            "pid_stopped_after_stop": stopped,
            "port_closed_after_stop": closed,
        },
        "log": {
            "path": str(log_path.relative_to(ROOT)),
            "sha256": _sha256(log_path),
            "bytes": log_path.stat().st_size,
        },
        "scope": "CPU HTTP/model-load smoke only; no quality or performance claim",
    }
    passed = (
        status == 200
        and len(payload) > 0
        and owned
        and not compute_processes
        and stopped
        and closed
        and len(methods) == 15
    )
    receipt["passed"] = passed
    _write_exclusive(VIEWER_RECEIPT, receipt)
    print(json.dumps({"passed": passed, "receipt": str(VIEWER_RECEIPT)}, indent=2))
    return 0 if passed else 1


class _Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        for key in ("href", "src"):
            if values.get(key):
                self.references.append(str(values[key]))


def _index(port: int) -> int:
    if not INDEX.is_file():
        raise FileNotFoundError(INDEX)
    parser = _Links()
    parser.feed(INDEX.read_text(encoding="utf-8"))
    targets: set[Path] = set()
    for reference in parser.references:
        clean, _fragment = urldefrag(reference)
        parsed = urlsplit(clean)
        if not clean or parsed.scheme or parsed.netloc:
            continue
        target = (INDEX.parent / parsed.path).resolve()
        if not target.is_relative_to(ROOT):
            raise RuntimeError(f"index target escapes repository: {target}")
        targets.add(target)

    command = [
        sys.executable,
        "-m",
        "http.server",
        str(port),
        "--bind",
        "127.0.0.1",
        "--directory",
        ".",
    ]
    log_path = RUN / "index_smoke.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        index_url = f"http://127.0.0.1:{port}/" + quote(str(INDEX.relative_to(ROOT)), safe="/")
        try:
            index_status, _index_payload = _wait_for_http(index_url, process)
            owned = _pid_owns_port(process.pid, port)
            rss = _rss_kib(process.pid)
            target_records = []
            for target in sorted(targets):
                url = f"http://127.0.0.1:{port}/" + quote(
                    str(target.relative_to(ROOT)),
                    safe="/",
                )
                status, payload = _fetch(url)
                target_records.append(
                    {
                        "path": str(target.relative_to(ROOT)),
                        "http_status": status,
                        "response_bytes": len(payload),
                        "response_sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
        finally:
            stopped, closed = _stop(process, port)

    status_counts: dict[str, int] = {}
    for record in target_records:
        key = str(record["http_status"])
        status_counts[key] = status_counts.get(key, 0) + 1
    passed = (
        index_status == 200
        and owned
        and stopped
        and closed
        and bool(target_records)
        and all(
            item["http_status"] == 200 and item["response_bytes"] > 0 for item in target_records
        )
    )
    receipt = {
        "schema": "rtgs.geometric_arena_frame00008.index_receipt.v1",
        "checked_utc": dt.datetime.now(dt.UTC).isoformat(),
        "serve_command": " ".join([".venv/bin/python", *command[1:]]),
        "url": index_url,
        "index": {
            "path": str(INDEX.relative_to(ROOT)),
            "sha256": _sha256(INDEX),
            "bytes": INDEX.stat().st_size,
            "http_status": index_status,
        },
        "links": {
            "references": len(parser.references),
            "unique_local_targets": len(targets),
            "status_counts": status_counts,
            "targets_with_nonpositive_content_length": sum(
                item["response_bytes"] <= 0 for item in target_records
            ),
            "total_target_bytes": sum(item["response_bytes"] for item in target_records),
            "targets": target_records,
        },
        "server": {
            "pid": process.pid,
            "host": "127.0.0.1",
            "port": port,
            "pid_owned_listening_socket": owned,
            "rss_kib_at_check": rss,
            "pid_stopped_after_stop": stopped,
            "port_closed_after_stop": closed,
        },
        "scope": "HTTP and completed local-link smoke only; no visual-quality or performance claim",
        "passed": passed,
    }
    _write_exclusive(INDEX_RECEIPT, receipt)
    print(json.dumps({"passed": passed, "receipt": str(INDEX_RECEIPT)}, indent=2))
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("viewer", "index"))
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.target == "viewer":
        return _viewer(args.port or 8787)
    return _index(args.port or 8791)


if __name__ == "__main__":
    raise SystemExit(main())
