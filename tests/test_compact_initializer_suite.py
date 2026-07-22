"""CPU-only contracts for the full-frame compact-initializer suite operator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def suite_operator():
    path = Path(__file__).resolve().parents[1] / "benchmarks/run_compact_initializer_suite.py"
    spec = importlib.util.spec_from_file_location("rtgs_compact_initializer_suite_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def suite_summarizer():
    path = Path(__file__).resolve().parents[1] / "benchmarks/summarize_compact_initializer_suite.py"
    spec = importlib.util.spec_from_file_location("rtgs_compact_initializer_summary_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_suite_inventory_and_order_are_frozen(suite_operator):
    assert suite_operator.ARMS == (
        "topk",
        "dense-merge",
        "easy-only",
        "splat-sfm",
        "field",
        "random",
    )


@pytest.mark.parametrize(
    "arm",
    ("topk", "dense-merge", "easy-only", "splat-sfm", "field", "random"),
)
def test_parent_commands_bind_common_schedule_and_never_evaluate(
    suite_operator,
    tmp_path,
    arm,
):
    command = suite_operator._parent_command(
        "python",
        arm,
        tmp_path / arm,
        tmp_path / "protocol.md",
    )

    assert command[command.index("--initializer") + 1] == arm
    assert command[command.index("--iterations") + 1] == "30000"
    assert command[command.index("--densify-stop") + 1] == "15000"
    assert command[command.index("--max-gaussians") + 1] == "100000"
    assert command[command.index("--field-max-tracks") + 1] == "128"
    assert "evaluate" not in command


@pytest.mark.parametrize(
    ("phase", "expected_name"),
    (
        ("polish", "polish_30000_40000"),
        ("tail", "tail_40000_50000"),
        ("cooldown", "cooldown_50000_60000"),
        ("settle", "settle_60000_70000"),
    ),
)
def test_continuation_order_is_frozen(suite_operator, phase, expected_name):
    assert (phase, expected_name) in suite_operator.CONTINUATIONS
    command = suite_operator._continuation_command(
        "python",
        phase,
        Path("parent"),
        Path(expected_name),
    )
    assert command[command.index("--phase") + 1] == phase
    assert command[command.index("--parent-out") + 1] == "parent"
    assert "evaluate" not in command


def test_summary_decision_applies_both_materiality_gates(suite_summarizer):
    def arm(psnr: float, objective: float) -> dict:
        return {
            "status": "complete",
            "initial_metrics_all_26_fitted_views": {"psnr_fg": psnr - 20.0},
            "downstream": {
                "selected_metrics_all_26_fitted_views": {"psnr_fg": psnr},
                "selected_objective": objective,
            },
        }

    material = suite_summarizer._decision(
        {"winner": arm(38.20, 0.00240), "runner": arm(38.00, 0.00250)}
    )
    psnr_only = suite_summarizer._decision(
        {"winner": arm(38.20, 0.002497), "runner": arm(38.00, 0.00250)}
    )

    assert material["material_quality_winner"] is True
    assert psnr_only["material_quality_winner"] is False


def test_summary_inventory_disposes_every_repository_initializer(suite_summarizer):
    applicability = suite_summarizer.APPLICABILITY

    assert set(applicability["prospective_compact_arms"]) == {
        "topk",
        "dense-merge",
        "easy-only",
        "splat-sfm",
        "field",
        "random",
    }
    assert set(applicability["historical_compact_anchor"]) == {"beam-fusion"}
    assert set(applicability["inapplicable_to_compact_only_bundle"]) == {
        "gradient",
        "legacy-carve",
        "depth",
        "hybrid",
        "classic-sfm",
    }
    assert set(applicability["not_public_arms"]) == {"field-placement-fallback"}
