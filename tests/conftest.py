"""Shared fixtures: deterministic seeding and tiny ground-truthed scenes.

Scene fixtures are session-scoped (building them renders images) — tests must not
mutate them in place.
"""

from __future__ import annotations

import pytest
import torch

from rtgs.data.synthetic import make_synthetic_scene
from rtgs.image2gs.fit import FitConfig, fit_views


@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(0)
    yield


@pytest.fixture(scope="session")
def tiny_scene():
    """8 views, 32x32, 25 GT gaussians — the standard test scene."""
    return make_synthetic_scene(n_gaussians=25, n_cameras=8, image_size=32, seed=0)


@pytest.fixture(scope="session")
def tiny_fits(tiny_scene):
    """Stage-1 fits for tiny_scene (shared across lifting tests)."""
    cfg = FitConfig(n_gaussians=120, iterations=120, log_every=60)
    g2ds, hists = fit_views(tiny_scene.images, cfg, seed=0)
    return g2ds, hists
