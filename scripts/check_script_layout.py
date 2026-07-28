#!/usr/bin/env python3
"""Keep ``scripts/`` durable and one-off experiment scripts in ``scripts/experiments/``.

Without this gate, experiment-specific drivers accumulate at the top level of ``scripts/``
until an agent cannot tell repository tooling from a spent protocol runner. The rule is an
allowlist: every top-level file in ``scripts/`` must be declared durable here, with a reason.
Anything else belongs in ``scripts/experiments/`` (see that directory's README).

Run: python scripts/check_script_layout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

# name -> why it is durable repository tooling rather than an experiment driver.
DURABLE_SCRIPTS: dict[str, str] = {
    "verify.sh": "the repository verification gate; CI runs the same sequence",
    "docs_sync.py": "docs<->code structural checker, part of verify.sh",
    "check_ara.py": "ara/ claim-ledger structural checker, part of verify.sh",
    "check_script_layout.py": "this checker, part of verify.sh",
    "check_results_bundle.py": "Hard Rule 7 results-bundle gate, run per results-bearing run",
    "experiment_contract.py": "task/run contracts and the canonical experiment report renderer",
    "convert_datasets_to_gaussians2d.py": "resumable dataset migration reused across experiments",
    "render_compact_structsplat_gallery.py": "reusable gallery renderer for previews and figures",
    # Grandfathered: both paths are bound by source hash in DECLARED_SOURCE_PATHS in
    # benchmarks/inverse_projection_fiber_iter1e.py and cited by sealed notes under
    # benchmarks/results/. Moving them would break replay integrity of committed evidence.
    "verify_iter1e_development_tree.py": "PINNED: source-hash bound by sealed iter1e evidence",
    "write_iter1e_verification_receipt.py": "PINNED: source-hash bound by sealed iter1e evidence",
}


def main() -> int:
    if not SCRIPTS.is_dir():
        print("check_script_layout: missing scripts/ directory", file=sys.stderr)
        return 1

    errors: list[str] = []

    for path in sorted(SCRIPTS.iterdir()):
        if path.is_dir() or path.name.startswith("."):
            continue
        if path.name not in DURABLE_SCRIPTS:
            errors.append(
                f"scripts/{path.name} is not declared durable. Move it to scripts/experiments/ "
                "(see scripts/experiments/README.md), or add it to DURABLE_SCRIPTS in "
                "scripts/check_script_layout.py with a reason."
            )

    for name in sorted(DURABLE_SCRIPTS):
        if not (SCRIPTS / name).is_file():
            errors.append(
                f"DURABLE_SCRIPTS lists scripts/{name} but that file does not exist "
                "(remove the stale allowlist entry)"
            )

    experiments = SCRIPTS / "experiments"
    if not experiments.is_dir():
        errors.append("missing scripts/experiments/ directory")
    elif not (experiments / "README.md").is_file():
        errors.append("missing scripts/experiments/README.md (the layout policy lives there)")

    if errors:
        print(f"check_script_layout: {len(errors)} problem(s):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print(f"check_script_layout: OK ({len(DURABLE_SCRIPTS)} durable scripts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
