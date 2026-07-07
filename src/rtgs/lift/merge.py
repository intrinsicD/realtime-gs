"""Moment-matched gaussian merging (mixture reduction).

Gaussians lifted from different views that describe the same surface patch are fused by
matching the first two moments of the weighted mixture (Runnalls 2007; used for splats
by Hierarchical 3DGS with opacity-times-size weights):

  w   = sum_i w_i,  mu = sum_i w_i mu_i / w
  Cov = sum_i w_i (Cov_i + (mu_i - mu)(mu_i - mu)^T) / w

Weights are w_i = opacity_i * prod(scales_i) (opacity times volume); merged opacity is
1 - prod(1 - opacity_i), colors/SH merge with the same weights. Grouping is a voxel hash
over the means — cheap and adequate for initialization purposes.
"""

from __future__ import annotations

import torch

from rtgs.core.gaussians3d import Gaussians3D


def merge_by_voxel(g: Gaussians3D, voxel_size: float) -> Gaussians3D:
    """Merge all gaussians that fall into the same voxel cell; singletons pass through."""
    if g.n == 0:
        return g
    keys = torch.floor(g.means / voxel_size).long()
    _, group = torch.unique(keys, dim=0, return_inverse=True)
    n_groups = int(group.max()) + 1

    w = (g.opacity * g.scales.prod(dim=-1)).clamp_min(1e-12)  # (N,)
    w_sum = torch.zeros(n_groups).index_add_(0, group, w)

    means_w = torch.zeros(n_groups, 3).index_add_(0, group, g.means * w[:, None])
    mu = means_w / w_sum[:, None]

    covs = g.covariance()  # (N,3,3)
    diff = g.means - mu[group]  # (N,3)
    second = covs + diff[:, :, None] * diff[:, None, :]
    cov_w = torch.zeros(n_groups, 9).index_add_(0, group, second.reshape(-1, 9) * w[:, None])
    cov_merged = (cov_w / w_sum[:, None]).reshape(n_groups, 3, 3)
    # Symmetrize against accumulation noise.
    cov_merged = 0.5 * (cov_merged + cov_merged.transpose(-1, -2))

    k = g.sh.shape[1]
    sh_w = torch.zeros(n_groups, k * 3).index_add_(0, group, g.sh.reshape(-1, k * 3) * w[:, None])
    sh_merged = (sh_w / w_sum[:, None]).reshape(n_groups, k, 3)

    log_1m = torch.log1p(-g.opacity.clamp(0.0, 0.995))
    log_1m_sum = torch.zeros(n_groups).index_add_(0, group, log_1m)
    opacity = (1.0 - log_1m_sum.exp()).clamp(0.01, 0.995)

    evals, evecs = torch.linalg.eigh(cov_merged)
    det = torch.linalg.det(evecs)
    evecs = evecs.clone()
    evecs[det < 0, :, 2] *= -1.0
    from rtgs.core.gaussians3d import rotmat_to_quat

    return Gaussians3D(
        means=mu,
        quats=rotmat_to_quat(evecs),
        log_scales=evals.clamp_min(1e-12).sqrt().log(),
        opacity=opacity,
        sh=sh_merged,
    )
