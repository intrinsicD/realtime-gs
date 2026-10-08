"""Analytic-surface geometry of the jet2 driver: closest point, normal, Weingarten map."""

import importlib
import sys
from pathlib import Path

import numpy as np

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
