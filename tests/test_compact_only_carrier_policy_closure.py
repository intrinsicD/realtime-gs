"""CPU-only tests for the compact-only carrier policy-closure mechanisms."""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from benchmarks import compact_only_carrier_policy_closure as closure

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D
from rtgs.core.observation2d import GaussianObservationField
from rtgs.core.sh import rgb_to_sh
from rtgs.data.reconstruction_inputs import ReconstructionInputs


def _gaussians() -> Gaussians3D:
    return Gaussians3D(
        means=torch.tensor([[0.0, 0.0, 0.0], [0.2, -0.1, 0.1]]),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        log_scales=torch.log(torch.tensor([[0.1, 0.3, 0.2], [0.4, 0.1, 0.2]])),
        opacity=torch.tensor([0.2, 0.7]),
        sh=rgb_to_sh(torch.tensor([[0.3, 0.4, 0.5], [0.7, 0.2, 0.1]]))[:, None],
    )


def test_train_configs_freeze_exact_families_and_activate_higher_sh() -> None:
    fixed = closure._train_config(
        iterations=380,
        checkpoints=(0, 190, 380),
        seed=1,
        extent=2.0,
        trainable=("quats", "scales", "opacities", "sh0"),
    )
    higher = closure._train_config(
        iterations=380,
        checkpoints=(0, 190, 380),
        seed=2,
        extent=2.0,
        trainable=("sh0", "shN"),
        sh_degree=3,
    )
    assert fixed.lr_means == 0
    assert fixed.lr_quats > 0 and fixed.lr_scales > 0
    assert fixed.lr_opacity > 0 and fixed.lr_sh > 0
    assert fixed.lr_sh_rest == 0
    assert higher.sh_degree == 3
    assert higher.lr_means == higher.lr_quats == higher.lr_scales == 0
    assert higher.lr_opacity == 0
    assert higher.lr_sh > 0 and higher.lr_sh_rest > 0


def test_matched_clone_operators_share_positions_but_only_mass_variant_preserves_invariants() -> (
    None
):
    base = _gaussians()
    legacy, legacy_receipt = closure.clone_all(base, "legacy_copy")
    mass, mass_receipt = closure.clone_all(base, "mass_tangent")

    assert legacy.n == mass.n == 2 * base.n
    assert torch.equal(legacy.means, mass.means)
    assert torch.equal(legacy.quats, mass.quats)
    assert torch.equal(legacy.sh, mass.sh)
    assert torch.equal(legacy.means[: base.n], base.means)
    assert legacy_receipt["combined_opacity_error_max_abs"] > 0
    assert mass_receipt["combined_opacity_error_max_abs"] < 1e-7
    assert mass_receipt["axis_second_moment_error_max_abs"] < 1e-8
    expected_opacity = 1.0 - torch.sqrt(1.0 - base.opacity)
    assert torch.allclose(mass.opacity[: base.n], expected_opacity)
    assert torch.allclose(mass.opacity[base.n :], expected_opacity)


def test_strict_prune_removes_center_outside_teacher_union() -> None:
    camera = Camera.look_at(
        eye=torch.tensor([0.0, 0.0, -3.0]),
        target=torch.zeros(3),
        width=32,
        height=32,
    )
    inside = torch.tensor([[0.0, 0.0, 0.0]])
    teacher_xy, _depth = camera.project(inside)
    field = GaussianObservationField(
        width=32,
        height=32,
        means=teacher_xy,
        log_scales=torch.zeros(1, 2),
        rotations=torch.zeros(1),
        colors=torch.tensor([[0.5, 0.5, 0.5]]),
        amplitudes=torch.ones(1),
        fit_window=(0, 0, 32, 32),
        view_id="v0",
    )
    inputs = ReconstructionInputs(
        observations=[field],
        cameras=[camera],
        view_names=["v0"],
        bounds_hint=(torch.zeros(3), 20.0),
        name="strict-prune-test",
    )
    base = _gaussians()
    base.means[0] = inside[0]
    base.means[1] = torch.tensor([10.0, 0.0, 0.0])

    pruned, receipt = closure.strict_prune(base, inputs)

    assert pruned.n == 1
    assert torch.equal(pruned.means, inside)
    assert receipt["removed_rows"] == [1]
    assert receipt["removed_fraction"] == pytest.approx(0.5)


def test_policy_inventory_excludes_free_birth_and_heldout_evaluation() -> None:
    assert set(closure.PHASE1_ARMS) == {
        "corrected_C_all_380",
        "corrected_C_means_fixed_380",
        "corrected_C_means_only_380",
    }
    assert "strict_prune" in closure.STATIC_ARMS
    assert "legacy_clone_all_recover_190" in closure.LATER_TRAINING_ARMS
    assert "mass_tangent_clone_all_recover_190" in closure.LATER_TRAINING_ARMS
    assert set(closure.DEVELOPMENT_GLOBAL).isdisjoint(closure.HELDOUT_GLOBAL)
    assert len(closure.DEVELOPMENT_GLOBAL) == 23


def test_import_keeps_dense_image_modules_out() -> None:
    code = """
import importlib
import sys
module = importlib.import_module('benchmarks.compact_only_carrier_policy_closure')
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
