#!/usr/bin/env python3
"""Export source-bound realtime-gs cells for StructSplat BENCH-019.

Diagnostic review-state example::

    .venv/bin/python scripts/experiments/rtgs009_structsplat_bench019_exporter.py \
      export --protocol /path/to/bench019.protocol.json \
      --source /path/to/cell.source.json --output /new/path/cell.json \
      --receipt /new/path/cell.export.json --allow-review-protocol

Formal use omits ``--allow-review-protocol`` and therefore requires a frozen protocol.  Assemble
all independently exported rows with ``assemble``; the default refuses missing cells.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rtgs.bench019 import (
    ExportError,
    assemble_rows,
    downstream_factor_record,
    export_cell,
    load_json_object,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    factor = subparsers.add_parser("factor", help="print the deterministic downstream factor")
    factor.add_argument("--protocol", type=Path, required=True)
    factor.add_argument("--frame-id", required=True)
    factor.add_argument("--seed", type=int, required=True)
    factor.add_argument("--initializer", required=True)
    factor.add_argument("--allow-review-protocol", action="store_true")

    export = subparsers.add_parser("export", help="export one exact BENCH-019 cell row")
    export.add_argument("--protocol", type=Path, required=True)
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--receipt", type=Path, required=True)
    export.add_argument("--allow-review-protocol", action="store_true")

    assemble = subparsers.add_parser("assemble", help="assemble rows in frozen protocol order")
    assemble.add_argument("--protocol", type=Path, required=True)
    assemble.add_argument("--cell", type=Path, action="append", required=True)
    assemble.add_argument("--export-receipt", type=Path, action="append", required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.add_argument("--receipt", type=Path, required=True)
    assemble.add_argument("--allow-review-protocol", action="store_true")
    assemble.add_argument("--allow-incomplete", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded exporter command surface."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "factor":
            protocol_path = args.protocol.resolve(strict=True)
            protocol = load_json_object(protocol_path, label="BENCH-019 protocol")
            value = downstream_factor_record(
                protocol,
                frame_id=args.frame_id,
                seed=args.seed,
                initializer=args.initializer,
                protocol_base=protocol_path.parent,
                allow_review=args.allow_review_protocol,
            )
            print(json.dumps(value, indent=2, sort_keys=True))
            return 0
        if args.command == "export":
            row = export_cell(
                args.protocol,
                args.source,
                args.output,
                args.receipt,
                allow_review_protocol=args.allow_review_protocol,
            )
            print(json.dumps(row, indent=2, sort_keys=True))
            return 0
        rows = assemble_rows(
            args.protocol,
            args.cell,
            args.output,
            args.receipt,
            export_receipt_paths=args.export_receipt,
            allow_review_protocol=args.allow_review_protocol,
            allow_incomplete=args.allow_incomplete,
        )
        print(json.dumps({"rows": len(rows), "output": str(args.output)}, sort_keys=True))
        return 0
    except ExportError as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
