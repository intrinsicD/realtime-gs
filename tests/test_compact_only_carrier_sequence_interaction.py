"""CPU-only tests for the final compact carrier sequence interaction."""

from __future__ import annotations

import subprocess
import sys

from benchmarks import compact_only_carrier_sequence_interaction as sequence


def _record(
    arm: str,
    q: float,
    u: float,
) -> dict[str, object]:
    metrics = {"validation": {"J_Q": q, "J_U": u}}
    contained = arm in sequence.CONTAINED_ARMS
    transformation: dict[str, object] = {"kind": "identity"}
    receipt = {"removed_fraction": 0.05}
    if arm.startswith("prune_then_"):
        key = "containment" if arm.endswith("then_sh3") else "prune"
        transformation[key] = receipt
    elif contained:
        transformation = receipt
    support = {
        "splits": {
            "fit": {
                "outside_fraction": 0.0 if contained else 0.01,
                "behind_near_fraction": 0.0,
            }
        }
    }
    if arm in sequence.STATIC_ARMS:
        return {
            "kind": "static",
            "metrics": metrics,
            "transformation": transformation,
            "support": support,
        }
    return {
        "kind": "training",
        "config": {"iterations": 380},
        "checkpoint_metrics": {"380": metrics},
        "transformation": transformation,
        "support": support,
    }


def test_inventory_contains_no_growth_or_dense_arm() -> None:
    assert set(sequence.ALL_ARMS) == {
        "unpruned_phase2_d0",
        "unpruned_phase2_d0_then_prune",
        "prune_then_phase2_d0",
        "prune_then_phase2_sh3_joint",
        "phase2_sh3_joint_unpruned",
        "phase2_sh3_joint_then_prune",
        "prune_then_phase2_d0_then_sh3",
        "unpruned_phase2_d0_then_sh3",
        "unpruned_phase2_d0_then_sh3_then_prune",
    }
    assert all("clone" not in arm and "split" not in arm for arm in sequence.ALL_ARMS)
    assert set(sequence.closure.DEVELOPMENT_GLOBAL).isdisjoint(sequence.closure.HELDOUT_GLOBAL)


def test_frozen_analysis_prefers_viable_pre_prune_and_material_separate_sh() -> None:
    values = {
        "unpruned_phase2_d0": (1.00, 1.00),
        "unpruned_phase2_d0_then_prune": (1.02, 1.02),
        "prune_then_phase2_d0": (1.00, 1.00),
        "prune_then_phase2_sh3_joint": (0.91, 0.99),
        "phase2_sh3_joint_unpruned": (0.90, 0.98),
        "phase2_sh3_joint_then_prune": (0.92, 1.00),
        "prune_then_phase2_d0_then_sh3": (0.88, 0.98),
        "unpruned_phase2_d0_then_sh3": (0.87, 0.97),
        "unpruned_phase2_d0_then_sh3_then_prune": (0.90, 0.99),
    }
    records = {
        seed: {arm: _record(arm, *values[arm]) for arm in sequence.ALL_ARMS}
        for seed in sequence.PHASE1_ROOTS
    }

    analysis = sequence._analysis(records)

    assert analysis["decisions"] == {
        "post_phase2_prune_viable": True,
        "pre_phase2_prune_viable": True,
        "containment_order": "pre",
        "joint_sh3_material": True,
        "separate_sh3_material": True,
        "sh_choice": "separate",
        "selected_arm": "prune_then_phase2_d0_then_sh3",
    }


def test_import_keeps_dense_image_modules_out() -> None:
    code = """
import importlib
import sys
module = importlib.import_module('benchmarks.compact_only_carrier_sequence_interaction')
assert 'rtgs.data.scene' not in sys.modules
assert 'rtgs.data.calibrated' not in sys.modules
assert 'rtgs.optim.trainer' not in sys.modules
guard = module.stage1.NoImageGuard()
with guard:
    assert guard.record()['passed']
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
