#!/usr/bin/env python3
"""Frozen Iteration 1e wrapper for inverse-projection-fiber fitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks import inverse_projection_fiber_protocol as protocol
from benchmarks.inverse_projection_fiber_transaction import (
    DEVELOPMENT_ONLY,
    ReceiptDomain,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_LABEL = "inverse-projection-fiber-iteration-1e"
OFFICIAL_NAMESPACE = "rtgs.inverse-projection-fiber.iter1e.v1"
DEVELOPMENT_NAMESPACE = "rtgs.inverse-projection-fiber.development.iter1e.v1"
OFFICIAL_SCHEMA_FAMILY = "inverse_projection_fiber_iter1e"
DEVELOPMENT_SCHEMA_FAMILY = "inverse_projection_fiber_development_iter1e"
SCENE_ROOTS = (17_687_011, 17_687_012, 17_687_013)
INITIAL_DEPTH_ROOTS = (17_687_111, 17_687_112, 17_687_113)
OFFICIAL_ROOTS = SCENE_ROOTS + INITIAL_DEPTH_ROOTS
DEVELOPMENT_ROOTS = (91, 92)
OFFICIAL_ROOT_STATUSES = (
    "ROOTS_NOT_STARTED",
    "ROOT_TRANSITION_ATTEMPTED",
    "OFFICIAL_GENERATORS_CONSUMED",
)
VERIFICATION_BASE = Path("/tmp/rtgs_ipf_iter1e_verify_20260717_001")
OFFICIAL_OUT = ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter1e_RESULT.json"
OFFICIAL_ARTIFACTS_DIR = ROOT / "runs/inverse_projection_fiber_iter1e_official_20260717"

PREREGISTRATION = ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter1e_PREREG.md"
PREREGISTRATION_REVIEW = (
    ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter1e_PREREG_REVIEW.md"
)
VERIFICATION_RECEIPT = (
    ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter1e_VERIFICATION.json"
)
IMPLEMENTATION_REVIEW = (
    ROOT / "benchmarks/results/20260717_inverse_projection_fiber_iter1e_IMPLEMENTATION_REVIEW.json"
)

BOUND_HISTORICAL_SHA256 = (
    (
        Path("benchmarks/results/20260717_inverse_projection_fiber_iter1d_PREREG.md"),
        "91d3e1c601c6eeb41f4c828e1f600c5dd7a1f52c818754745548130d9b35fe9c",
    ),
    (
        Path("benchmarks/results/20260717_inverse_projection_fiber_iter1d_PREREG_REVIEW_FAIL.md"),
        "e9ecb31f6ed4738a902b38aa2b87dca53ec4ccbcc725c003a82954bee7791598",
    ),
    (
        Path("benchmarks/results/20260717_inverse_projection_fiber_iter1e_PREREG.md"),
        "7b2f52631355e15f5ef1c2098309af4bcfa6f91a6250813d009a01fc83737a06",
    ),
    (
        Path("benchmarks/results/20260717_inverse_projection_fiber_iter1e_PREREG_REVIEW.md"),
        "fda1e4dd700d87888c3c0965e11e3a988720f76801cc8d0805d5f347c4b7bef5",
    ),
)

OFFICIAL_DOMAIN = ReceiptDomain(
    protocol_label=PROTOCOL_LABEL,
    label="official",
    namespace=OFFICIAL_NAMESPACE,
    schema_family=OFFICIAL_SCHEMA_FAMILY,
    permitted_root_consumption_statuses=OFFICIAL_ROOT_STATUSES,
    permitted_roots=OFFICIAL_ROOTS,
    official_phases_permitted=True,
    commit_states_permitted=True,
)
DEVELOPMENT_DOMAIN = ReceiptDomain(
    protocol_label=PROTOCOL_LABEL,
    label="development",
    namespace=DEVELOPMENT_NAMESPACE,
    schema_family=DEVELOPMENT_SCHEMA_FAMILY,
    permitted_root_consumption_statuses=(DEVELOPMENT_ONLY,),
    forbidden_roots=OFFICIAL_ROOTS,
    forbidden_literals=(
        OFFICIAL_NAMESPACE,
        OFFICIAL_SCHEMA_FAMILY,
        *OFFICIAL_ROOT_STATUSES,
    ),
)

DECLARED_SOURCE_PATHS = (
    Path("benchmarks/inverse_projection_fiber_iter1e.py"),
    Path("benchmarks/inverse_projection_fiber_protocol.py"),
    Path("benchmarks/inverse_projection_fiber_transaction.py"),
    Path("scripts/verify_iter1e_development_tree.py"),
    Path("scripts/write_iter1e_verification_receipt.py"),
    *tuple(path for path, _digest in BOUND_HISTORICAL_SHA256),
    VERIFICATION_RECEIPT.relative_to(ROOT),
    IMPLEMENTATION_REVIEW.relative_to(ROOT),
    Path("src/rtgs/lift/inverse_projection_fiber.py"),
    Path("src/rtgs/render/projection.py"),
    Path("src/rtgs/render/torch_ref.py"),
    Path("src/rtgs/data/synthetic.py"),
    Path("src/rtgs/__init__.py"),
    Path("src/rtgs/core/__init__.py"),
    Path("src/rtgs/data/__init__.py"),
    Path("src/rtgs/lift/__init__.py"),
    Path("src/rtgs/render/__init__.py"),
    Path("src/rtgs/core/camera.py"),
    Path("src/rtgs/core/gaussians3d.py"),
    Path("src/rtgs/core/sh.py"),
    Path("tests/test_inverse_projection_fiber.py"),
    Path("tests/test_inverse_projection_transaction.py"),
    Path("tests/test_iter1e_development_tree_scan.py"),
)

SPEC = protocol.ProtocolSpec(
    protocol_label=PROTOCOL_LABEL,
    official_domain=OFFICIAL_DOMAIN,
    development_domain=DEVELOPMENT_DOMAIN,
    preregistration=PREREGISTRATION,
    preregistration_review=PREREGISTRATION_REVIEW,
    verification_receipt=VERIFICATION_RECEIPT,
    implementation_review=IMPLEMENTATION_REVIEW,
    bound_historical_sha256=BOUND_HISTORICAL_SHA256,
    declared_source_paths=DECLARED_SOURCE_PATHS,
    scene_roots=SCENE_ROOTS,
    initial_depth_roots=INITIAL_DEPTH_ROOTS,
    verification_base=VERIFICATION_BASE,
    official_out=OFFICIAL_OUT,
    official_artifacts_dir=OFFICIAL_ARTIFACTS_DIR,
    development_roots=DEVELOPMENT_ROOTS,
    verification_test_paths=(
        Path("tests/test_inverse_projection_fiber.py"),
        Path("tests/test_inverse_projection_transaction.py"),
        Path("tests/test_iter1e_development_tree_scan.py"),
    ),
    verification_scanner=Path("scripts/verify_iter1e_development_tree.py"),
)


def run(out: Path, artifacts_dir: Path) -> dict[str, Any]:
    return protocol.run_protocol(SPEC, out, artifacts_dir)


def main() -> int:
    args = protocol.parse_args()
    result = run(args.out, args.artifacts_dir)
    print(
        json.dumps(
            {
                "namespace": result["namespace"],
                "status": result["status"],
                "out": str(args.out),
                "artifacts_dir": str(args.artifacts_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
