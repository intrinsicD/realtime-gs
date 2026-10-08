"""Evaluation controls for the jet2 analytic experiment (plan P3, deterministic part).

Each control builds ideal surfels on the analytic surface, applies one known corruption and
asserts that the evaluation moves the metric meant to catch it. Signatures and tolerances were
fixed before the first run of this file; a failure is reported, not tuned away.

First run (2026-10-08): the exact-surfel coverage prediction (> 0.95) failed at 0.905 on both
surfaces. Cause: surfels and coverage points were drawn with the same seed, so 4000 of the 5000
points sat on centres (0.8 + 0.2 x 0.53). For randomly placed centres the expected coverage at
radius r is the Poisson value 1 - exp(-N pi r^2 / A) (~0.53 here), so the > 0.95 prediction was
wrong in principle: coverage is a comparison between arms, not an absolute quality score. Fixed
by separate seeds and a Poisson-consistency assertion.
"""

import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rtgs.core.gaussians3d import Gaussians3D, rotmat_to_quat
from rtgs.optim.jet_prior import mean_neighbour_distance

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
driver = importlib.import_module("20261008_jet2_prior_2dgs_analytic")
SURFACES = {"sphere": ((1.0, 1.0, 1.0), 4 * math.pi), "ellipsoid": ((1.0, 0.75, 0.5), 6.97148)}
N = 4000
TASK_PRIOR = {"jet2": {"radius": 3.0, "min_mass": 0.5, "min_conditioning": 0.02}}


def _ideal(surface: str, tilt_deg: float = 0.0, shift: float = 0.0, seed: int = 11):
    axes, area = SURFACES[surface]
    x = driver.surface_samples(axes, N, seed=seed)
    n, _ = driver.surface_frame(x, axes)
    helper = np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    t1 = np.cross(n, helper)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    if tilt_deg:
        phi = np.random.default_rng(seed + 1).uniform(0, 2 * np.pi, N)[:, None]
        axis = np.cos(phi) * t1 + np.sin(phi) * t2
        a = math.radians(tilt_deg)
        n = math.cos(a) * n + math.sin(a) * np.cross(axis, n)
        t1 = np.cross(n, helper)
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
        t2 = np.cross(n, t1)
    centres = x - shift * driver.surface_frame(x, axes)[0]
    rot = torch.tensor(np.stack([t1, t2, n], 2), dtype=torch.float32)
    sigma = 0.75 * math.sqrt(area / N)
    log_scales = torch.tensor([math.log(sigma)] * 2 + [-math.inf]).expand(N, 3).clone()
    model = Gaussians3D(
        torch.tensor(centres, dtype=torch.float32),
        rotmat_to_quat(rot),
        log_scales,
        torch.full((N,), 0.9),
        torch.zeros(N, 1, 3),
    )
    return model, axes


def _evaluate(model, axes, cohort=None):
    frozen = {
        "h0": mean_neighbour_distance(model.means),
        "cohort": torch.ones(model.n, dtype=torch.bool) if cohort is None else cohort,
    }
    task = {
        "datasets": [{"surface_axes": list(axes)}],
        "jet2_prior_configs": TASK_PRIOR,
        "coverage": {"mesh_points": 5000},
    }
    return driver.analytic_geometry(model, frozen, task, seed=0)


@pytest.mark.parametrize("surface", list(SURFACES))
def test_exact_surfels_score_as_exact(surface):
    g = _evaluate(*_ideal(surface))
    area, r = SURFACES[surface][1], g["coverage_radius"]
    assert g["normal_angle_median"] < 0.5 and abs(g["signed_distance_mean"]) < 1e-5
    assert abs(g["coverage"] - (1 - math.exp(-N * math.pi * r * r / area))) < 0.03
    assert g["estimator_masked_fraction"] < 0.2
    assert g["curvature_rel_error_median"] < 0.15
    assert 0.9 < g["curvature_trace_ratio_median"] < 1.1


@pytest.mark.parametrize("surface", list(SURFACES))
def test_tilted_normals_are_measured_and_degrade_curvature(surface):
    exact = _evaluate(*_ideal(surface))
    tilted = _evaluate(*_ideal(surface, tilt_deg=5.0))
    assert 4.5 < tilted["normal_angle_median"] < 5.5
    assert tilted["curvature_rel_error_median"] > 2 * exact["curvature_rel_error_median"]


@pytest.mark.parametrize("surface", list(SURFACES))
def test_inward_shift_is_measured_in_absolute_units(surface):
    g = _evaluate(*_ideal(surface, shift=0.005))
    assert -0.0055 < g["signed_distance_mean"] < -0.0045 and g["inward_fraction"] > 0.99


def test_clones_trip_the_collapse_gate():
    model, _ = _ideal("sphere")
    h0 = mean_neighbour_distance(model.means)
    clean = driver.v21.collapse_fraction(model.means, h0)
    cloned = torch.cat([model.means] * 8)
    collapsed = driver.v21.collapse_fraction(cloned, torch.cat([h0] * 8))
    assert clean < 0.01 and collapsed > 0.99


def test_opacity_selection_is_visible_in_counts_and_cohort():
    # 10 % of splats tilted by 30 deg and made transparent: the opacity subset hides them; the
    # subset count and the frozen cohort's tail statistic do not (its median does not either).
    model, axes = _ideal("sphere")
    bad_model, _ = _ideal("sphere", tilt_deg=30.0)
    bad = torch.arange(N) % 10 == 0
    model.quats[bad] = bad_model.quats[bad]
    model.opacity[bad] = 0.1
    g = _evaluate(model, axes)
    assert g["n_subset"] == N - int(bad.sum()) and g["n_cohort"] == N
    assert g["normal_angle_median"] < 0.5 and g["normal_fraction_gt_10deg"] == 0.0
    assert abs(g["cohort_normal_fraction_gt_10deg"] - 0.1) < 0.01


def test_coverage_drops_with_a_hole():
    # removing every splat above z = 0.5 (a cap of a quarter of the sphere's area) must lower
    # coverage by about that fraction at the same frozen radius
    model, axes = _ideal("sphere")
    full = _evaluate(model, axes)
    hole = model.means[:, 2] > 0.5
    model.opacity[hole] = 0.1
    frozen_h0 = mean_neighbour_distance(model.means)  # same centres, same radius as `full`
    g = driver.analytic_geometry(
        model,
        {"h0": frozen_h0, "cohort": torch.ones(N, dtype=torch.bool)},
        {
            "datasets": [{"surface_axes": list(axes)}],
            "jet2_prior_configs": TASK_PRIOR,
            "coverage": {"mesh_points": 5000},
        },
        seed=0,
    )
    assert g["coverage_radius"] == full["coverage_radius"]
    assert abs(g["coverage"] / full["coverage"] - 0.75) < 0.05
