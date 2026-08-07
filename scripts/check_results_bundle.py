#!/usr/bin/env python3
"""Verify that a results-bearing run directory satisfies CLAUDE.md Hard Rule 7.

Rule 7 requires every results-bearing run to save ``--out`` artifacts/previews and a generated
experiment handoff. V2 adds a summary-bound relative-link ``index.html``, accompanying
``README.md``, full SHA-256 manifest, and a structured browser-side smoke receipt for every
reported page and exact ``rtgs view`` command. This script is the final results-bearing gate;
frozen v1 bundles retain their historical checks.

It validates the *bundle*, not the science: it cannot tell you whether a number is right, only
whether the artifact a reader needs in order to check it is present and reachable. Promoting a
claim still goes through the ``realtime-gs-results-audit`` skill.

Usage:
    python scripts/check_results_bundle.py runs/<name>
    python scripts/check_results_bundle.py runs/<name> --no-previews   # metrics-only run
    python scripts/check_results_bundle.py runs/<name> --json          # machine-readable

Exit status is 0 when the bundle is complete, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

# Written by `rtgs run --out` (see rtgs/cli.py).
REQUIRED_ARTIFACTS = (
    "gaussians_init.ply",
    "gaussians.ply",
    "metrics.json",
    "training_history.json",
    "gaussians.config.json",
)

# Written by `rtgs run --out --preview` (see rtgs.visualize.save_reconstruction_artifacts).
REQUIRED_PREVIEWS = (
    "reconstruction_contact_sheet.png",
    "reconstruction.gif",
    "novel_orbit.gif",
    "novel_elevation.gif",
)

RESULTS_PAGE = "index.html"
V2_CORE_FILES = (
    "task.lock.json",
    "metrics.json",
    "training_history.json",
    "gaussians.config.json",
    "input_boundary_receipt.json",
    "resource_receipt.json",
    "run_receipt.json",
    "environment.json",
    "index.html",
    "README.md",
    "manifest.json",
)

# Frozen v1 bundles accept their historical free-form receipts. V2 uses the structured
# `viewer_smoke.json` validator below.
RECEIPT_NAMES = ("smoke_receipt.json", "smoke_receipt.md", "viewer_smoke.json", "AUDIT.md")


class LinkCollector(HTMLParser):
    """Collect href/src targets from the results page."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.links.append(value)

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def check_bundle(run_dir: Path, *, previews: bool) -> list[str]:
    problems: list[str] = []

    if not run_dir.is_dir():
        return [f"{run_dir} is not a directory"]

    if _is_v2_bundle(run_dir):
        return _check_v2_bundle(run_dir, previews=previews)

    for name in REQUIRED_ARTIFACTS:
        path = run_dir / name
        if not path.is_file():
            problems.append(f"missing required artifact: {name}")
        elif path.stat().st_size == 0:
            problems.append(f"required artifact is empty: {name}")

    if previews:
        for name in REQUIRED_PREVIEWS:
            if not (run_dir / name).is_file():
                problems.append(
                    f"missing preview: {name} (run with --preview, or pass --no-previews "
                    "if this run legitimately has none)"
                )

    problems.extend(_check_metrics(run_dir))
    problems.extend(_check_results_page(run_dir))
    problems.extend(_check_receipts(run_dir))
    return problems


def _is_v2_bundle(run_dir: Path) -> bool:
    if (run_dir / "manifest.json").is_file():
        return True
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics = None
        if isinstance(metrics, dict) and metrics.get("report_template_version") == 2:
            return True
    index_path = run_dir / "index.html"
    return index_path.is_file() and (
        'name="rtgs-experiment-report-template" content="2"'
        in index_path.read_text(encoding="utf-8", errors="replace")
    )


def _check_v2_bundle(run_dir: Path, *, previews: bool) -> list[str]:
    problems: list[str] = []
    for name in V2_CORE_FILES:
        path = run_dir / name
        if not path.is_file():
            problems.append(f"missing required v2 file: {name}")
        elif path.stat().st_size == 0:
            problems.append(f"required v2 file is empty: {name}")

    status = _v2_status(run_dir, problems)
    completed = status == "completed"
    if completed:
        for name in ("gaussians_init.ply", "gaussians.ply"):
            path = run_dir / name
            if not path.is_file():
                problems.append(f"missing required artifact: {name}")
            elif path.stat().st_size == 0:
                problems.append(f"required artifact is empty: {name}")
        if previews:
            for name in REQUIRED_PREVIEWS:
                if not (run_dir / name).is_file():
                    problems.append(
                        f"missing preview: {name} (run with --preview, or pass --no-previews "
                        "if this run legitimately has none)"
                    )
        problems.extend(_check_metrics(run_dir))
        expected_viewers = _v2_viewer_targets(run_dir)
        if expected_viewers is None:
            problems.append("metrics.json viewer commands are missing or invalid")
        else:
            problems.extend(_check_v2_viewer_smoke(run_dir, expected_viewers))
    elif status == "failed":
        problems.append(
            "run_receipt.json records a failed run; this report is inspectable but is not a "
            "results-bearing bundle"
        )
    problems.extend(_check_results_page(run_dir, require_model=completed))
    problems.extend(_check_v2_manifest(run_dir))
    return problems


def _v2_status(run_dir: Path, problems: list[str]) -> str | None:
    path = run_dir / "run_receipt.json"
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        problems.append(f"run_receipt.json is not valid JSON: {error}")
        return None
    status = receipt.get("status") if isinstance(receipt, dict) else None
    if not isinstance(status, str) or status not in {"completed", "failed"}:
        problems.append("run_receipt.json status must be completed or failed")
        return None
    return status


def _v2_viewer_targets(run_dir: Path) -> list[dict[str, object]] | None:
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    summaries = metrics.get("dataset_summaries") if isinstance(metrics, dict) else None
    if isinstance(summaries, dict) and len(summaries) > 1:
        result = []
        for dataset_id, summary in summaries.items():
            commands = summary.get("commands") if isinstance(summary, dict) else None
            viewer = commands.get("viewer") if isinstance(commands, dict) else None
            if (
                not isinstance(dataset_id, str)
                or not dataset_id
                or not isinstance(viewer, list)
                or not viewer
                or not all(isinstance(item, str) and item for item in viewer)
            ):
                return None
            result.append(
                {
                    "dataset_id": dataset_id,
                    "viewer_command": viewer,
                    "report_target": f"datasets/{dataset_id}/index.html",
                }
            )
        return result
    commands = metrics.get("commands") if isinstance(metrics, dict) else None
    viewer = commands.get("viewer") if isinstance(commands, dict) else None
    if (
        not isinstance(viewer, list)
        or not viewer
        or not all(isinstance(item, str) and item for item in viewer)
    ):
        return None
    return [{"dataset_id": None, "viewer_command": viewer, "report_target": RESULTS_PAGE}]


def _viewer_smoke_entry_errors(
    entry: object,
    *,
    expected_viewer: list[str],
    expected_target: str,
    prefix: str,
) -> list[str]:
    if not isinstance(entry, dict) or set(entry) != {
        "viewer_command",
        "report",
        "browser",
        "checks",
    }:
        return [f"{prefix} has the wrong shape"]
    problems: list[str] = []
    if entry["viewer_command"] != expected_viewer:
        problems.append(f"{prefix} viewer_command must exactly match commands.viewer")

    report = entry["report"]
    if not isinstance(report, dict) or set(report) != {
        "target",
        "http_status",
        "local_targets_ok",
    }:
        problems.append(f"{prefix} report has the wrong shape")
    else:
        if report["target"] != expected_target or report["http_status"] != 200:
            problems.append(f"{prefix} must record {expected_target} HTTP 200")
        if report["local_targets_ok"] is not True:
            problems.append(f"{prefix} must confirm every local report target loaded")

    browser = entry["browser"]
    if not isinstance(browser, dict) or set(browser) != {
        "name",
        "version",
        "user_agent",
        "webgl2",
        "renderer",
    }:
        problems.append(f"{prefix} browser has the wrong shape")
    else:
        for key in ("name", "version", "user_agent"):
            if not isinstance(browser[key], str) or not browser[key].strip():
                problems.append(f"{prefix} browser.{key} must be non-empty")
        if browser["webgl2"] is not True:
            problems.append(f"{prefix} must confirm WebGL2 availability")
        if browser["renderer"] is not None and (
            not isinstance(browser["renderer"], str) or not browser["renderer"].strip()
        ):
            problems.append(f"{prefix} browser.renderer must be null or non-empty")

    checks = entry["checks"]
    if not isinstance(checks, dict) or set(checks) != {
        "viewer_ready",
        "canvas_count",
        "rendered_content_visible",
        "framebuffer_nonbackground_pixels",
        "orbit_camera_changed",
        "client_errors",
        "client_warnings",
    }:
        problems.append(f"{prefix} checks has the wrong shape")
    else:
        if checks["viewer_ready"] is not True:
            problems.append(f"{prefix} must confirm the viewer reached ready state")
        if (
            not isinstance(checks["canvas_count"], int)
            or isinstance(checks["canvas_count"], bool)
            or checks["canvas_count"] < 1
        ):
            problems.append(f"{prefix} canvas_count must be at least one")
        if checks["rendered_content_visible"] is not True:
            problems.append(f"{prefix} must confirm visible rendered scene content")
        if (
            not isinstance(checks["framebuffer_nonbackground_pixels"], int)
            or isinstance(checks["framebuffer_nonbackground_pixels"], bool)
            or checks["framebuffer_nonbackground_pixels"] < 1
        ):
            problems.append(f"{prefix} framebuffer_nonbackground_pixels must be at least one")
        if checks["orbit_camera_changed"] is not True:
            problems.append(f"{prefix} must confirm an orbit changed the camera")
        if checks["client_errors"] != []:
            problems.append(f"{prefix} client_errors must be an empty list")
        warnings = checks["client_warnings"]
        if not isinstance(warnings, list) or any(
            not isinstance(item, str) or not item.strip() for item in warnings
        ):
            problems.append(f"{prefix} client_warnings must be a list of non-empty strings")
    return problems


def _check_v2_viewer_smoke(run_dir: Path, expected_viewers: list[dict[str, object]]) -> list[str]:
    """Validate the structured attestation that a browser rendered and orbited the viewer."""

    path = run_dir / "viewer_smoke.json"
    if not path.is_file():
        return ["missing viewer_smoke.json (v2 requires a browser-side WebGL/orbit smoke receipt)"]
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return [f"viewer_smoke.json is not valid JSON: {error}"]
    if len(expected_viewers) == 1:
        if not isinstance(receipt, dict) or set(receipt) != {
            "schema_version",
            "status",
            "viewer_command",
            "report",
            "browser",
            "checks",
        }:
            return ["viewer_smoke.json has the wrong top-level shape"]
        problems: list[str] = []
        if receipt["schema_version"] != 1:
            problems.append("viewer_smoke.json schema_version must be 1")
        if receipt["status"] != "passed":
            problems.append("viewer_smoke.json status must be passed")
        expected = expected_viewers[0]
        problems.extend(
            _viewer_smoke_entry_errors(
                {key: receipt[key] for key in ("viewer_command", "report", "browser", "checks")},
                expected_viewer=expected["viewer_command"],
                expected_target=expected["report_target"],
                prefix="viewer_smoke.json",
            )
        )
        return problems

    if not isinstance(receipt, dict) or set(receipt) != {
        "schema_version",
        "status",
        "entries",
    }:
        return ["multi-viewer viewer_smoke.json has the wrong top-level shape"]
    problems = []
    if receipt["schema_version"] != 2:
        problems.append("multi-viewer viewer_smoke.json schema_version must be 2")
    if receipt["status"] != "passed":
        problems.append("multi-viewer viewer_smoke.json status must be passed")
    entries = receipt["entries"]
    if not isinstance(entries, list) or len(entries) != len(expected_viewers):
        return problems + ["viewer_smoke.json entry count differs from dataset summaries"]
    for index, (entry, expected) in enumerate(zip(entries, expected_viewers, strict=True)):
        if not isinstance(entry, dict) or set(entry) != {
            "dataset_id",
            "viewer_command",
            "report",
            "browser",
            "checks",
        }:
            problems.append(f"viewer_smoke.json entries[{index}] has the wrong shape")
            continue
        if entry["dataset_id"] != expected["dataset_id"]:
            problems.append(f"viewer_smoke.json entries[{index}] dataset identity differs")
        problems.extend(
            _viewer_smoke_entry_errors(
                {key: entry[key] for key in ("viewer_command", "report", "browser", "checks")},
                expected_viewer=expected["viewer_command"],
                expected_target=expected["report_target"],
                prefix=f"viewer_smoke.json entries[{index}]",
            )
        )
    return problems


def _safe_relative(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_v2_manifest(run_dir: Path) -> list[str]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        return []
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return [f"manifest.json is not valid JSON: {error}"]
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "task_id",
        "report_template_version",
        "entries",
    }:
        return ["manifest.json has the wrong top-level shape"]
    problems: list[str] = []
    if manifest["schema_version"] != 1 or manifest["report_template_version"] != 2:
        problems.append("manifest.json has unsupported schema/report versions")
    entries = manifest["entries"]
    if not isinstance(entries, list):
        return problems + ["manifest.json entries must be a list"]
    repository_root = run_dir.parent.parent if run_dir.parent.name == "runs" else run_dir.parent
    entry_keys = {"label", "path", "scope", "role", "media_type", "size_bytes", "sha256"}
    seen: set[tuple[str, str]] = set()
    run_paths: set[str] = set()
    links: list[tuple[str, str]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != entry_keys:
            problems.append(f"manifest entry {index} has the wrong keys")
            continue
        scope, target_path = entry["scope"], entry["path"]
        if (
            not isinstance(scope, str)
            or scope not in {"run", "repository"}
            or not _safe_relative(target_path)
        ):
            problems.append(f"manifest entry {index} has an invalid path/scope")
            continue
        identity = (scope, target_path)
        if identity in seen:
            problems.append(f"manifest repeats {scope} path: {target_path}")
            continue
        seen.add(identity)
        if not isinstance(entry["label"], str) or not entry["label"].strip():
            problems.append(f"manifest entry {index} has an invalid label")
        if not isinstance(entry["role"], str) or not entry["role"].strip():
            problems.append(f"manifest entry {index} has an invalid role")
        if not isinstance(entry["media_type"], str) or "/" not in entry["media_type"]:
            problems.append(f"manifest entry {index} has an invalid media type")
        if (
            not isinstance(entry["size_bytes"], int)
            or isinstance(entry["size_bytes"], bool)
            or entry["size_bytes"] < 0
        ):
            problems.append(f"manifest entry {index} has an invalid byte size")
        if (
            not isinstance(entry["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
        ):
            problems.append(f"manifest entry {index} has an invalid SHA-256")
        base = run_dir if scope == "run" else repository_root
        target = base / target_path
        try:
            target.resolve(strict=True).relative_to(base.resolve())
        except (FileNotFoundError, ValueError):
            problems.append(f"manifest target is missing or escapes {scope}: {target_path}")
            continue
        if target.is_symlink() or not target.is_file():
            problems.append(f"manifest target is not a regular file: {target_path}")
            continue
        if target.stat().st_size != entry["size_bytes"]:
            problems.append(f"manifest size mismatch: {target_path}")
        if _sha256(target) != entry["sha256"]:
            problems.append(f"manifest SHA-256 mismatch: {target_path}")
        if scope == "run":
            run_paths.add(target_path)
            link = target_path
        else:
            link = os.path.relpath(target, run_dir)
        links.append((target_path, link))

    expected_run_paths = {
        item.relative_to(run_dir).as_posix()
        for item in run_dir.rglob("*")
        if item.is_file()
        and item.name != "manifest.json"
        and not (item.name.startswith(".") and item.name.endswith(".tmp"))
    }
    if run_paths != expected_run_paths:
        missing = sorted(expected_run_paths - run_paths)
        extra = sorted(run_paths - expected_run_paths)
        if missing:
            problems.append("manifest omits run files: " + ", ".join(missing))
        if extra:
            problems.append("manifest names unexpected run files: " + ", ".join(extra))

    index = (
        (run_dir / "index.html").read_text(encoding="utf-8", errors="replace")
        if (run_dir / "index.html").is_file()
        else ""
    )
    readme = (
        (run_dir / "README.md").read_text(encoding="utf-8", errors="replace")
        if (run_dir / "README.md").is_file()
        else ""
    )
    collector = LinkCollector()
    collector.feed(index)
    markdown_links = set(re.findall(r"\]\(([^)]+)\)", readme))
    if "manifest.json" not in collector.links or "manifest.json" not in markdown_links:
        problems.append("index.html and README.md must both link manifest.json")
    if "## Commands" not in readme:
        problems.append("README.md does not contain the exact command handoff")
    for target_path, link in links:
        if target_path != "index.html" and link not in collector.links:
            problems.append(f"index.html does not link manifest entry: {target_path}")
        if target_path != "README.md" and link not in markdown_links:
            problems.append(f"README.md does not link manifest entry: {target_path}")
    return problems


def _check_metrics(run_dir: Path) -> list[str]:
    path = run_dir / "metrics.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"metrics.json is not valid JSON: {exc}"]
    if not isinstance(payload, dict) or "metrics" not in payload:
        return ["metrics.json has no top-level 'metrics' object"]
    metrics = payload["metrics"]
    if not isinstance(metrics, dict) or not metrics:
        return ["metrics.json 'metrics' is empty"]
    non_finite = [
        key
        for key, value in metrics.items()
        if isinstance(value, float) and (value != value or value in {float("inf"), -float("inf")})
    ]
    if non_finite:
        return [f"metrics.json has non-finite values for: {', '.join(sorted(non_finite))}"]
    return []


def _check_results_page(run_dir: Path, *, require_model: bool = True) -> list[str]:
    """The page must exist, use relative links, resolve every local target, and cite metrics."""
    page = run_dir / RESULTS_PAGE
    if not page.is_file():
        return [
            f"missing {RESULTS_PAGE} (Rule 7 requires a summary-bound relative-link results page)"
        ]

    html = page.read_text(encoding="utf-8", errors="replace")
    collector = LinkCollector()
    collector.feed(html)

    problems: list[str] = []
    local_targets = 0
    for link in collector.links:
        parsed = urlparse(link)
        if parsed.scheme in {"http", "https", "mailto", "data"}:
            continue
        if not parsed.path or link.startswith("#"):
            continue
        if parsed.path.startswith("/"):
            problems.append(f"{RESULTS_PAGE} uses an absolute link '{link}' (must be relative)")
            continue
        local_targets += 1
        target = (run_dir / unquote(parsed.path)).resolve()
        if not target.exists():
            problems.append(f"{RESULTS_PAGE} links to '{parsed.path}' which does not exist")

    if local_targets == 0:
        problems.append(
            f"{RESULTS_PAGE} links to no local artifact (the page must bind the saved models, "
            "previews, and metrics it summarizes)"
        )

    required_references = ("gaussians.ply", "metrics.json") if require_model else ("metrics.json",)
    for name in required_references:
        if name not in html:
            problems.append(f"{RESULTS_PAGE} does not reference {name}")

    # "Summary-bound" means the page shows the numbers, not just links to the JSON.
    visible_text = " ".join(collector.text)
    if not re.search(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", visible_text, re.IGNORECASE):
        problems.append(
            f"{RESULTS_PAGE} contains no numeric value (it must carry the summary metrics, "
            "not only a link to metrics.json)"
        )
    return problems


def _check_receipts(run_dir: Path) -> list[str]:
    """Rule 7 wants evidence the page and an `rtgs view` command were actually exercised."""
    receipts = [name for name in RECEIPT_NAMES if (run_dir / name).is_file()]
    if not receipts:
        return [
            "missing smoke-test receipt (expected one of: " + ", ".join(RECEIPT_NAMES) + ")",
            "missing recorded 'rtgs view' command (no receipt file to carry it)",
        ]

    blob = "\n".join(
        (run_dir / name).read_text(encoding="utf-8", errors="replace") for name in receipts
    )
    problems: list[str] = []
    if "rtgs view" not in blob:
        problems.append(
            f"no receipt in {', '.join(receipts)} records an 'rtgs view' command "
            "(Rule 7 requires the exact viewer command)"
        )
    if RESULTS_PAGE not in blob:
        problems.append(
            f"no receipt in {', '.join(receipts)} records a smoke test of {RESULTS_PAGE}"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="results-bearing run directory to validate")
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="skip the preview-artifact requirement (metrics-only run)",
    )
    parser.add_argument("--json", action="store_true", help="emit a machine-readable verdict")
    args = parser.parse_args(argv)

    problems = check_bundle(args.run_dir, previews=not args.no_previews)

    if args.json:
        print(
            json.dumps(
                {
                    "run_dir": str(args.run_dir),
                    "complete": not problems,
                    "problems": problems,
                },
                indent=2,
            )
        )
    elif problems:
        print(
            f"check_results_bundle: {len(problems)} problem(s) in {args.run_dir}:", file=sys.stderr
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
    else:
        print(f"check_results_bundle: OK ({args.run_dir})")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
