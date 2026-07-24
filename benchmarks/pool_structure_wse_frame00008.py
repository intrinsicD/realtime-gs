#!/usr/bin/env python3
"""Compare pooled gradient, pooled structure/no-WSE, and pooled structure/WSE on Janelle.

This prospective development experiment reuses the artifact-complete calibrated runner from
``benchmarks.new_variants_frame00008`` while replacing only its arm definitions, output namespace,
viewer manifest, source binding, and JSON schema. All three arms use the same fixed-capacity pool.
The two structure-tensor arms differ only in final candidate subset selection: the matched density
control keeps the first N points from the shared oversampled candidate stream, while WSE performs
anisotropic crowding elimination down to N.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from benchmarks import new_variants_frame00008 as runner
except ModuleNotFoundError:
    # Direct ``python benchmarks/<script>.py`` execution places ``benchmarks/`` rather than the
    # repository root on sys.path. Package imports use the first branch; the official CLI uses
    # this same-file fallback.
    import new_variants_frame00008 as runner

from rtgs.image2gs.fit import FitConfig

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks/results/20260724_pool_structure_wse_frame00008_PREREG_V3.md"
OUT = ROOT / "runs/pool_structure_wse_frame00008_20260724"
VIEWER = ROOT / "benchmarks/results/20260724_pool_structure_wse_frame00008_VIEWER.json"

ARMS = (
    "pool-gradient",
    "pool-structure-density",
    "pool-structure-wse",
)


def _pool_config(**overrides: Any) -> FitConfig:
    common: dict[str, Any] = {
        "n_gaussians": 640,
        "max_gaussians": None,
        "iterations": 300,
        "backend": "native",
        "native_renderer": "cuda",
        "lr": 1e-2,
        "grad_init_mix": 0.7,
        "row_chunk": 64,
        "log_every": 50,
        "convergence_patience": 0,
        "appearance_parameterization": "weight_color_9p",
        "pool": True,
        "pool_capacity": 1_280,
        "pool_triage_every": 50,
        "pool_prune_count": 32,
        "pool_spawn_count": 32,
        "pool_min_live": 1,
    }
    common.update(overrides)
    return FitConfig(**common)


def _fit_configs() -> dict[str, FitConfig]:
    return {
        "pool-gradient": _pool_config(),
        "pool-structure-density": _pool_config(
            init_strategy="structure_tensor",
            structure_sampling="density",
        ),
        "pool-structure-wse": _pool_config(
            init_strategy="structure_tensor",
            structure_sampling="wse",
        ),
    }


def _schema_writer(path: Path, value: Any) -> None:
    if isinstance(value, dict):
        value = dict(value)
        schema = value.get("schema")
        if isinstance(schema, str) and schema.startswith("rtgs.new_variants_frame00008."):
            value["schema"] = schema.replace(
                "rtgs.new_variants_frame00008.",
                "rtgs.pool_structure_wse_frame00008.",
                1,
            )
    _ORIGINAL_WRITE_JSON(path, value)


_ORIGINAL_WRITE_JSON = runner._write_json


def _configure_runner() -> None:
    runner.DEFAULT_PROTOCOL = PROTOCOL
    runner.DEFAULT_OUT = OUT
    runner.VIEWER_MANIFEST = VIEWER
    runner.STAGE1_ARMS = ARMS
    runner.REPORT_ARMS = ARMS
    runner.SOURCE_FILES = (
        *runner.SOURCE_FILES,
        "benchmarks/pool_structure_wse_frame00008.py",
    )
    runner._fit_configs = _fit_configs
    runner._write_json = _schema_writer


def main() -> int:
    _configure_runner()
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
