"""Analytic-surface geometry of the jet2 driver: closest point, normal, Weingarten map."""

import importlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/experiments"))
driver = importlib.import_module("20261008_jet2_prior_2dgs_analytic")
ELLIPSOID = (1.0, 0.75, 0.5)


def test_sphere_closest_point_normal_and_shape_operator():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(500, 3)) * rng.uniform(0.3, 2.0, size=(500, 1))
    x = driver.closest_point(p, (1.0, 1.0, 1.0))
    expected = p / np.linalg.norm(p, axis=1, keepdims=True)
    assert np.abs(x - expected).max() < 1e-9
    n, w = driver.surface_frame(x, (1.0, 1.0, 1.0))
    assert np.abs(n - x).max() < 1e-12  # outward
    proj = np.eye(3) - x[:, :, None] * x[:, None, :]
    assert np.abs(w - proj).max() < 1e-12  # S = +I/R on the tangent plane


def test_ellipsoid_closest_point_is_a_foot_point():
    rng = np.random.default_rng(1)
    p = rng.normal(size=(500, 3)) * rng.uniform(0.2, 2.0, size=(500, 1))
    x = driver.closest_point(p, ELLIPSOID)
    n, _ = driver.surface_frame(x, ELLIPSOID)
    d = p - x
    tangential = d - (d * n).sum(1, keepdims=True) * n
    assert np.abs(tangential).max() < 1e-8  # p - x is normal to the surface at x
    # it is the global minimiser: no dense surface sample is closer
    samples = driver.surface_samples(ELLIPSOID, 20000, seed=0)
    brute = np.linalg.norm(p[:50, None] - samples[None], axis=2).min(1)
    assert (np.linalg.norm(d[:50], axis=1) <= brute + 1e-12).all()


def test_ellipsoid_principal_curvatures_at_vertices():
    # at the vertex (a, 0, 0) the principal curvatures are a/b^2 and a/c^2
    x = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.5]])
    n, w = driver.surface_frame(x, ELLIPSOID)
    assert np.allclose(n, [[1, 0, 0], [0, 0, 1]])
    k = np.sort(np.linalg.eigvalsh(w), 1)[:, 1:]
    assert np.allclose(k[0], sorted([1.0 / 0.75**2, 1.0 / 0.5**2]))
    assert np.allclose(k[1], sorted([0.5 / 1.0**2, 0.5 / 0.75**2]))


def test_ellipsoid_interior_points_near_the_short_axis_plane():
    # no regular root at (0.1, 0, 0); cancellation at (0.1, 0, 1e-8): the singular branch
    p = np.array([[0.1, 0.0, 0.0], [0.1, 0.0, 1e-8], [0.0, 0.2, -1e-12], [0.3, 0.1, 0.0]])
    x = driver.closest_point(p, ELLIPSOID)
    assert np.abs((x**2 / np.square(ELLIPSOID)).sum(1) - 1).max() < 1e-9
    samples = driver.surface_samples(ELLIPSOID, 20000, seed=0)
    brute = np.linalg.norm(p[:, None] - samples[None], axis=2).min(1)
    assert (np.linalg.norm(p - x, axis=1) <= brute + 1e-12).all()


def test_fully_masked_curvature_reports_unavailable_not_crash():
    from rtgs.core.gaussians3d import Gaussians3D

    rng = np.random.default_rng(2)
    u = rng.normal(size=(40, 3))
    means = torch.tensor(u / np.linalg.norm(u, axis=1, keepdims=True), dtype=torch.float32)
    model = Gaussians3D.from_means_covs(
        means,
        torch.eye(3).expand(40, 3, 3) * 1e-6,
        torch.full((40, 3), 0.5),
        torch.full((40,), 0.9),
    )
    frozen = {"h0": torch.ones(40), "cohort": torch.ones(40, dtype=torch.bool)}
    task = {
        "datasets": [{"surface_axes": [1.0, 1.0, 1.0]}],
        "jet2_prior_configs": {"jet2": {"radius": 3.0, "min_mass": 0.5, "min_conditioning": 0.02}},
        "coverage": {"mesh_points": 500},
    }
    geometry = driver.analytic_geometry(model, frozen, task, seed=0)
    assert geometry["n_curvature"] == 0 and geometry["curvature_rel_error_median"] is None
    assert geometry["estimator_masked_fraction"] == 1.0 and geometry["n_subset"] == 40


def test_effective_config_records_the_executed_prior_spec():
    import json

    task_path = Path(__file__).resolve().parents[1] / (
        "experiments/tasks/20261008_jet2_prior_2dgs_analytic_sphere.json"
    )
    task = json.loads(task_path.read_text())
    driver.v21.SMOKE = {"mode": "smoke", "iterations": 600}
    try:
        _, _, prior, h0_step = driver.cell_setup(task, "jet2", 9561)
    finally:
        driver.v21.SMOKE = None
    recorded = task["field_prior_configs"]["jet2"]
    assert h0_step == 300 and prior.start == 300 and recorded["start"] == 300
    assert recorded["log_every"] == 100 and recorded["weight"] == 0.1
