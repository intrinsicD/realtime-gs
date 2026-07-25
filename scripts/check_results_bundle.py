#!/usr/bin/env python3
"""Verify that a results-bearing run directory satisfies CLAUDE.md Hard Rule 7.

Rule 7 requires every results-bearing run to save ``--out`` artifacts and previews, a
summary-bound relative-link ``index.html`` results page, and smoke-test receipts for both that
page and an ``rtgs view`` command. That was prose with no gate, so a run could be reported as
complete while missing the viewer handoff entirely. This script is the gate.

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
import json
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

# A receipt proves the page and the viewer command were actually exercised. Any one of these
# names satisfies it; the content must name the checked target.
RECEIPT_NAMES = ("smoke_receipt.json", "smoke_receipt.md", "viewer_smoke.json", "AUDIT.md")


class LinkCollector(HTMLParser):
    """Collect href/src targets from the results page."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.links.append(value)


def check_bundle(run_dir: Path, *, previews: bool) -> list[str]:
    problems: list[str] = []

    if not run_dir.is_dir():
        return [f"{run_dir} is not a directory"]

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


def _check_results_page(run_dir: Path) -> list[str]:
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

    for name in ("gaussians.ply", "metrics.json"):
        if name not in html:
            problems.append(f"{RESULTS_PAGE} does not reference {name}")

    # "Summary-bound" means the page shows the numbers, not just links to the JSON.
    if not re.search(r"\d+\.\d+", html):
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
