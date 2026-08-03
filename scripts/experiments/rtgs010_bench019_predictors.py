#!/usr/bin/env python3
"""Collect or replay development-only BENCH-019 Stage-1 predictors.

Examples::

    .venv/bin/python scripts/experiments/rtgs010_bench019_predictors.py collect \
      --adapter /path/stage.adapter.json --compact-field /path/gaussians2d \
      --family-id gaussianimage_additive --output /new/stage1.predictors.json
    .venv/bin/python scripts/experiments/rtgs010_bench019_predictors.py verify \
      --predictors /path/stage1.predictors.json --verify-files

Collection requires the adapter-bound portfolio to mark the requested family evidence-complete and
pin an adjacent production receipt for the exact compact manifest/views. Publication always
performs a full deterministic source/field replay; ``verify --verify-files`` repeats that replay.
Unsupported requests such as alpha agreement, MS-SSIM, LPIPS, track yield, or conditioning fail
before input access.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rtgs.bench019 import ExportError, load_json_object
from rtgs.bench019_predictors import (
    FIELD_FAMILIES,
    PredictorConfig,
    build_stage1_predictors,
    validate_stage1_predictors,
    write_stage1_predictors,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    collect = commands.add_parser("collect", help="collect one source-bound predictor artifact")
    collect.add_argument("--adapter", type=Path, required=True)
    collect.add_argument("--compact-field", type=Path, required=True)
    collect.add_argument("--family-id", choices=FIELD_FAMILIES, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--predictor", action="append")
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--sample-cap-per-stratum", type=int, default=4096)
    collect.add_argument("--boundary-radius-px", type=int, default=3)
    collect.add_argument("--component-chunk", type=int, default=256)
    collect.add_argument("--tile-size", type=int, default=16)
    collect.add_argument("--max-index-entries", type=int, default=16_000_000)
    collect.add_argument("--max-index-candidates", type=int, default=200_000)
    collect.add_argument("--max-query-pairs", type=int, default=1_048_576)
    collect.add_argument("--view-byte-cap", type=int, default=8_388_608)
    collect.add_argument("--psnr-mse-floor", type=float, default=1e-12)

    verify = commands.add_parser("verify", help="validate and optionally replay an artifact")
    verify.add_argument("--predictors", type=Path, required=True)
    verify.add_argument("--verify-files", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded RTGS-010 predictor command surface."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "collect":
            if args.output.exists():
                raise FileExistsError(f"refusing to overwrite predictor artifact: {args.output}")
            config = PredictorConfig(
                seed=args.seed,
                sample_cap_per_stratum=args.sample_cap_per_stratum,
                boundary_radius_px=args.boundary_radius_px,
                component_chunk=args.component_chunk,
                tile_size=args.tile_size,
                max_index_entries=args.max_index_entries,
                max_index_candidates=args.max_index_candidates,
                max_query_pairs=args.max_query_pairs,
                view_byte_cap=args.view_byte_cap,
                psnr_mse_floor=args.psnr_mse_floor,
            )
            value = build_stage1_predictors(
                args.adapter,
                args.compact_field,
                family_id=args.family_id,
                config=config,
                requested_predictors=args.predictor,
            )
            summary = write_stage1_predictors(value, args.output)
        else:
            value = load_json_object(args.predictors, label="BENCH-019 Stage-1 predictors")
            summary = validate_stage1_predictors(value, verify_files=args.verify_files)
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
        return 0
    except (ExportError, FileExistsError, OSError, ValueError) as error:
        print(f"error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
