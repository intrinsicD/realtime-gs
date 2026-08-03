"""Contract tests for the source-only BENCH-019 capture portfolio."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from rtgs.bench019 import ExportError, describe_artifact
from rtgs.bench019_portfolio import (
    PORTFOLIO_SCHEMA,
    REQUIRED_FIELD_FAMILIES,
    source_digest,
    validate_capture_portfolio,
)


def _families(*, confirmation: bool) -> list[dict]:
    records = []
    for family_id in REQUIRED_FIELD_FAMILIES:
        additive = family_id == "gaussianimage_additive"
        records.append(
            {
                "id": family_id,
                "provider": "gaussianimage" if additive else "structsplat",
                "equation": "additive_sum" if additive else "normalized_weighted_sum",
                "blend_mode": "additive" if additive else "normalized",
                "state": "not_produced" if confirmation else "incomplete_live_unbound",
                "observed_views": 0,
                "required_views": None,
                "evidence": None,
            }
        )
    return records


def _portfolio(tmp_path: Path) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    development = ["dev_a", "dev_b", "dev_c"]
    confirmation = ["confirm_a", "confirm_b", "confirm_c"]
    captures = []
    for capture_id in [*development, *confirmation]:
        source_path = tmp_path / f"{capture_id}.bin"
        source_path.write_bytes(capture_id.encode())
        sources = [{"id": "source", "artifact": describe_artifact(source_path)}]
        is_confirmation = capture_id in confirmation
        captures.append(
            {
                "id": capture_id,
                "role": "confirmation" if is_confirmation else "development",
                "source_kind": "test",
                "origin": "local-test",
                "frame_id": "frame",
                "frame_plan_state": "selected",
                "view_ids": [],
                "mask_policy_state": "test",
                "source_artifacts": sources,
                "source_digest": source_digest(sources),
                "field_families": _families(confirmation=is_confirmation),
                "blockers": ["test fixture is not production"],
            }
        )
    return {
        "schema": PORTFOLIO_SCHEMA,
        "state": "source_bound_preproduction",
        "inventory_date": "2026-08-03",
        "outcome_access": "none",
        "development_capture_ids": development,
        "confirmation_capture_ids": confirmation,
        "captures": captures,
        "gates": {
            "source_groups_bound": True,
            "field_families_complete": False,
            "confirmation_outcomes_opened": False,
            "formal_protocol_frozen": False,
        },
    }


def test_portfolio_requires_disjoint_three_plus_three_and_exact_sources(tmp_path: Path) -> None:
    portfolio = _portfolio(tmp_path)
    assert validate_capture_portfolio(portfolio, verify_files=True) == {
        "development_capture_groups": 3,
        "confirmation_capture_groups": 3,
        "source_files": 6,
    }

    overlap = copy.deepcopy(portfolio)
    overlap["confirmation_capture_ids"][0] = overlap["development_capture_ids"][0]
    with pytest.raises(ExportError, match="disjoint"):
        validate_capture_portfolio(overlap)


def test_portfolio_detects_source_tamper_and_confirmation_outcome(tmp_path: Path) -> None:
    portfolio = _portfolio(tmp_path)
    source_path = Path(portfolio["captures"][0]["source_artifacts"][0]["artifact"]["path"])
    source_path.write_bytes(b"tampered")
    with pytest.raises(ExportError, match="differs from the acquired source"):
        validate_capture_portfolio(portfolio, verify_files=True)

    portfolio = _portfolio(tmp_path / "fresh")
    confirmation_family = portfolio["captures"][3]["field_families"][0]
    confirmation_family["state"] = "incomplete_live_unbound"
    with pytest.raises(ExportError, match="confirmation field outcomes"):
        validate_capture_portfolio(portfolio)


def test_committed_portfolio_is_structurally_pre_outcome() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "experiments/data/structsplat_bench019_capture_portfolio.json"
    if not path.exists():
        pytest.skip("committed portfolio is created after the generator contract")
    import json

    portfolio = json.loads(path.read_text(encoding="utf-8"))
    summary = validate_capture_portfolio(portfolio)
    assert summary["development_capture_groups"] == 3
    assert summary["confirmation_capture_groups"] == 3
