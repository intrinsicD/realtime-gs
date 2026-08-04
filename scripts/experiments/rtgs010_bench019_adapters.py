#!/usr/bin/env python3
"""Plan, verify, and materialize development-only BENCH-019 source adapters.

Examples::

    .venv/bin/python scripts/experiments/rtgs010_bench019_adapters.py \
      plan-stage --portfolio /path/portfolio.json --output /new/stage.adapter.json
    .venv/bin/python scripts/experiments/rtgs010_bench019_adapters.py \
      plan-tum --portfolio /path/portfolio.json --capture-id tum_fr1_xyz \
      --output /new/tum_fr1_xyz.adapter.json
    .venv/bin/python scripts/experiments/rtgs010_bench019_adapters.py \
      verify-adapter --adapter /path/adapter.json --verify-sources
    .venv/bin/python scripts/experiments/rtgs010_bench019_adapters.py \
      materialize-tum --adapter /path/adapter.json --output /new/materialized

Planning reads only development sources.  Confirmation adapters and materialization fail closed;
all writes are exclusive-new.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rtgs.bench019 import ExportError, load_json_object
from rtgs.bench019_adapters import (
    build_calibrated_adapter,
    build_tum_adapter,
    materialize_tum_adapter,
    validate_materialization,
    validate_source_adapter,
    write_source_adapter,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    stage = commands.add_parser("plan-stage", help="bind a calibrated development capture")
    stage.add_argument("--portfolio", type=Path, required=True)
    stage.add_argument("--capture-id", default="janelle_stage_fabric")
    stage.add_argument("--output", type=Path, required=True)

    tum = commands.add_parser("plan-tum", help="bind a development TUM RGB-D archive")
    tum.add_argument("--portfolio", type=Path, required=True)
    tum.add_argument("--capture-id", required=True)
    tum.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify-adapter", help="validate an adapter manifest")
    verify.add_argument("--adapter", type=Path, required=True)
    verify.add_argument("--verify-sources", action="store_true")

    materialize = commands.add_parser(
        "materialize-tum", help="create a calibrated RGB/mask development directory"
    )
    materialize.add_argument("--adapter", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)

    verify_materialized = commands.add_parser(
        "verify-materialization", help="validate a TUM materialization receipt"
    )
    verify_materialized.add_argument("--receipt", type=Path, required=True)
    verify_materialized.add_argument("--verify-files", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded RTGS-010 adapter command surface."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan-stage":
            adapter = build_calibrated_adapter(
                args.portfolio,
                capture_id=args.capture_id,
            )
            summary = write_source_adapter(adapter, args.output)
        elif args.command == "plan-tum":
            adapter = build_tum_adapter(
                args.portfolio,
                capture_id=args.capture_id,
            )
            summary = write_source_adapter(adapter, args.output)
        elif args.command == "verify-adapter":
            adapter = load_json_object(args.adapter, label="BENCH-019 source adapter")
            summary = validate_source_adapter(
                adapter,
                verify_sources=args.verify_sources,
            )
        elif args.command == "materialize-tum":
            summary = materialize_tum_adapter(args.adapter, args.output)
        else:
            summary = validate_materialization(
                args.receipt,
                verify_files=args.verify_files,
            )
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
        return 0
    except (ExportError, FileExistsError, OSError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
