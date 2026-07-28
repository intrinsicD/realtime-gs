"""CPU-only protocol tests for the compact-only carrier stage ablation."""

from __future__ import annotations

import subprocess
import sys

import pytest
from benchmarks import compact_only_carrier_stage_ablation as ablation


def test_camera_split_and_arm_inventory_are_frozen() -> None:
    split = ablation._split_indices(26)
    assert split["heldout"] == (7, 15, 23)
    assert split["validation"] == (4, 10, 16, 21)
    assert len(split["fit"]) == 19
    assert set(split["fit"]).isdisjoint(split["validation"])
    assert set(split["fit"]).isdisjoint(split["heldout"])
    assert set().union(*map(set, split.values())) == set(range(26))
    assert len(ablation.ARMS) == 16
    assert {
        name for name in ablation.ARMS if name.startswith("legacy_") and name.endswith("_all")
    } == {
        "legacy_C_all",
        "legacy_O_all",
        "legacy_A_all",
        "legacy_CO_all",
        "legacy_CA_all",
        "legacy_OA_all",
        "legacy_COA_all",
    }


def test_parameter_ablation_rates_and_support_contract() -> None:
    means_only = ablation._train_config(
        ablation.ARMS["beam_means_only"],
        ablation.TRAIN_ROOTS[0],
        extent=2.0,
    )
    means_fixed = ablation._train_config(
        ablation.ARMS["beam_means_fixed"],
        ablation.TRAIN_ROOTS[0],
        extent=2.0,
    )
    supported = ablation._train_config(
        ablation.ARMS["beam_all_support"],
        ablation.TRAIN_ROOTS[0],
        extent=2.0,
    )
    assert means_only.lr_means > 0
    assert means_only.lr_quats == means_only.lr_scales == 0
    assert means_only.lr_opacity == means_only.lr_sh == 0
    assert means_fixed.lr_means == 0
    assert means_fixed.lr_quats > 0 and means_fixed.lr_scales > 0
    assert means_fixed.lr_opacity > 0 and means_fixed.lr_sh > 0
    assert supported.support_mode == "teacher_ellipse_center"
    assert supported.support_loss_weight == pytest.approx(0.01)
    assert supported.support_samples_per_step == 256


def test_fraction_ratio_handles_exact_zero_without_nonfinite_json_values() -> None:
    summaries = {
        "on": {"outside": [0.0, 0.0, 0.2]},
        "off": {"outside": [0.0, 0.4, 0.0]},
    }
    result = ablation._fraction_ratio(summaries, "on", "off", "outside")
    assert result["per_seed"] == [1.0, 0.0, None]
    assert result["geometric"] is None
    assert result["denominator_zero_count"] == 2


def test_import_and_live_guard_keep_image_capable_modules_out() -> None:
    code = """
import builtins
import importlib
import sys
from benchmarks.compact_only_carrier_stage_ablation import NoImageGuard

assert 'rtgs.data.scene' not in sys.modules
assert 'rtgs.data.calibrated' not in sys.modules
assert 'rtgs.optim.trainer' not in sys.modules
guard = NoImageGuard()
with guard:
    importlib.import_module('rtgs.lift.beam_fusion')
    assert guard.record()['passed']
assert guard.record()['passed']
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
