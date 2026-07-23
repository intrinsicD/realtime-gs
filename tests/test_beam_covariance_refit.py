from __future__ import annotations

import torch
from benchmarks.beam_covariance_refit import (
    _replace_covariances,
    relative_reprojection_residuals,
    robust_refit_from_jacobians,
    whitened_reprojection_residuals,
)

from rtgs.core.gaussians3d import Gaussians3D


def _synthetic_covariance_problem():
    jacobians = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.8, 0.2, 0.1], [-0.1, 0.7, 0.4]],
            ],
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.7, -0.2, 0.3], [0.1, 0.6, 0.5]],
            ],
        ],
        dtype=torch.float64,
    )
    true_covariances = torch.tensor(
        [
            [[0.08, 0.018, -0.009], [0.018, 0.045, 0.011], [-0.009, 0.011, 0.025]],
            [[0.035, -0.006, 0.004], [-0.006, 0.07, 0.015], [0.004, 0.015, 0.05]],
        ],
        dtype=torch.float64,
    )
    observed = jacobians @ true_covariances[:, None] @ jacobians.transpose(-1, -2)
    mask = torch.ones(2, 4, dtype=torch.bool)
    initial = true_covariances.clone()
    initial[:, 0, 0] *= 1.7
    initial[:, 1, 1] *= 0.65
    initial[:, 2, 2] *= 1.35
    initial = 0.5 * (initial + initial.transpose(-1, -2))
    return jacobians, true_covariances, observed, mask, initial


def test_robust_covariance_refit_recovers_consistent_synthetic_projections():
    jacobians, truth, observed, mask, initial = _synthetic_covariance_problem()
    before = whitened_reprojection_residuals(jacobians, initial, mask, observed).mean()
    fitted, history = robust_refit_from_jacobians(
        jacobians,
        mask,
        observed,
        initial,
        initial,
        steps=220,
        learning_rate=0.025,
        huber_delta=0.25,
        prior_weight=0.0,
        min_sigma=1e-4,
        max_sigma=1.0,
    )
    after = whitened_reprojection_residuals(jacobians, fitted, mask, observed).mean()
    relative = relative_reprojection_residuals(jacobians, fitted, mask, observed)
    assert history[0]["step"] == 1
    assert history[-1]["step"] == 220
    assert after < before * 0.02
    assert float(relative.max()) < 2e-3
    assert torch.allclose(fitted, truth, atol=2e-4, rtol=2e-3)
    assert bool((torch.linalg.eigvalsh(fitted) > 0).all())


def test_covariance_replacement_preserves_all_non_covariance_fields_bit_exact():
    means = torch.tensor([[0.1, -0.2, 1.0], [0.3, 0.1, 1.2]])
    colors = torch.tensor([[0.2, 0.4, 0.8], [0.7, 0.3, 0.1]])
    opacity = torch.tensor([0.1, 0.2])
    base_covariance = torch.eye(3).expand(2, 3, 3).clone() * 0.02
    base = Gaussians3D.from_means_covs(means, base_covariance, colors, opacity)
    replacement = torch.tensor(
        [
            [[0.04, 0.01, 0.00], [0.01, 0.03, 0.002], [0.00, 0.002, 0.02]],
            [[0.02, 0.00, 0.003], [0.00, 0.05, 0.004], [0.003, 0.004, 0.03]],
        ]
    )
    changed = _replace_covariances(base, replacement)
    assert torch.equal(changed.means, base.means)
    assert torch.equal(changed.opacity, base.opacity)
    assert torch.equal(changed.sh, base.sh)
    assert torch.allclose(changed.covariance(), replacement, atol=1e-6, rtol=1e-5)
