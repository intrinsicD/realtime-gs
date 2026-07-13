"""Container for a 3D gaussian splatting scene (3DGS parametrization).

Covariance is factored as ``Sigma = R S S^T R^T`` with unit quaternion rotation and
log-scales (3DGS). Colors are spherical-harmonics coefficients; opacity is stored as its
actual [0,1] value (optimizers keep their own raw/logit parameters). PLY export follows the
INRIA 3DGS attribute layout so standard viewers can open our scenes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rtgs.core.sh import num_sh_bases, sh_degree_from_bases


def quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """Convert (N,4) quaternions (w, x, y, z) to (N,3,3) rotation matrices."""
    q = torch.nn.functional.normalize(quats, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        dim=-2,
    )


def rotmat_to_quat(rotmats: torch.Tensor) -> torch.Tensor:
    """Convert (N,3,3) rotation matrices to (N,4) quaternions (w, x, y, z).

    Uses the numerically robust branch-per-largest-diagonal method, vectorized.
    """
    m = rotmats
    n = m.shape[0]
    q = torch.zeros(n, 4, dtype=m.dtype, device=m.device)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    # Branch 0: trace positive
    c0 = trace > 0
    s = torch.sqrt((trace.clamp_min(-0.999999) + 1.0).clamp_min(1e-12)) * 2  # 4w
    q0 = torch.stack(
        [
            0.25 * s,
            (m[:, 2, 1] - m[:, 1, 2]) / s,
            (m[:, 0, 2] - m[:, 2, 0]) / s,
            (m[:, 1, 0] - m[:, 0, 1]) / s,
        ],
        -1,
    )
    # Branch 1: m00 largest
    s1 = torch.sqrt((1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2]).clamp_min(1e-12)) * 2
    q1 = torch.stack(
        [
            (m[:, 2, 1] - m[:, 1, 2]) / s1,
            0.25 * s1,
            (m[:, 0, 1] + m[:, 1, 0]) / s1,
            (m[:, 0, 2] + m[:, 2, 0]) / s1,
        ],
        -1,
    )
    # Branch 2: m11 largest
    s2 = torch.sqrt((1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2]).clamp_min(1e-12)) * 2
    q2 = torch.stack(
        [
            (m[:, 0, 2] - m[:, 2, 0]) / s2,
            (m[:, 0, 1] + m[:, 1, 0]) / s2,
            0.25 * s2,
            (m[:, 1, 2] + m[:, 2, 1]) / s2,
        ],
        -1,
    )
    # Branch 3: m22 largest
    s3 = torch.sqrt((1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1]).clamp_min(1e-12)) * 2
    q3 = torch.stack(
        [
            (m[:, 1, 0] - m[:, 0, 1]) / s3,
            (m[:, 0, 2] + m[:, 2, 0]) / s3,
            (m[:, 1, 2] + m[:, 2, 1]) / s3,
            0.25 * s3,
        ],
        -1,
    )
    c1 = (~c0) & (m[:, 0, 0] >= m[:, 1, 1]) & (m[:, 0, 0] >= m[:, 2, 2])
    c2 = (~c0) & (~c1) & (m[:, 1, 1] >= m[:, 2, 2])
    q = torch.where(c0[:, None], q0, torch.where(c1[:, None], q1, torch.where(c2[:, None], q2, q3)))
    return torch.nn.functional.normalize(q, dim=-1)


@dataclass
class Gaussians3D:
    """N 3D gaussians with SH colors."""

    means: torch.Tensor  # (N, 3)
    quats: torch.Tensor  # (N, 4) unit quaternions (w, x, y, z)
    log_scales: torch.Tensor  # (N, 3)
    opacity: torch.Tensor  # (N,) in [0, 1]
    sh: torch.Tensor  # (N, K, 3), K = (degree+1)^2

    def __post_init__(self) -> None:
        n = self.means.shape[0]
        if (
            self.quats.shape != (n, 4)
            or self.log_scales.shape != (n, 3)
            or self.opacity.shape != (n,)
            or self.sh.ndim != 3
            or self.sh.shape[0] != n
            or self.sh.shape[2] != 3
        ):
            raise ValueError("inconsistent Gaussians3D field shapes")
        sh_degree_from_bases(self.sh.shape[1])  # validate K

    @property
    def n(self) -> int:
        """Number of gaussians."""
        return self.means.shape[0]

    @property
    def sh_degree(self) -> int:
        """SH degree implied by the coefficient count."""
        return sh_degree_from_bases(self.sh.shape[1])

    @property
    def scales(self) -> torch.Tensor:
        """(N,3) actual scales (exp of log-scales)."""
        return self.log_scales.exp()

    def covariance(self) -> torch.Tensor:
        """(N,3,3) world-space covariances R S S^T R^T."""
        rot = quat_to_rotmat(self.quats)
        s = self.scales
        rs = rot * s[:, None, :]
        return rs @ rs.transpose(-1, -2)

    @staticmethod
    def from_means_covs(
        means: torch.Tensor,
        covs: torch.Tensor,
        colors: torch.Tensor,
        opacity: torch.Tensor,
        sh_degree: int = 0,
        min_scale: float = 1e-6,
    ) -> Gaussians3D:
        """Build from explicit (N,3,3) covariances via eigendecomposition.

        ``colors`` are RGB in [0,1]; they land in the SH degree-0 band, higher bands zero.
        """
        from rtgs.core.sh import rgb_to_sh

        evals, evecs = torch.linalg.eigh(covs)  # ascending eigenvalues
        # Ensure right-handed rotation (det +1) by flipping the last eigenvector if needed.
        det = torch.linalg.det(evecs)
        evecs = evecs.clone()
        evecs[det < 0, :, 2] *= -1.0
        scales = evals.clamp_min(min_scale**2).sqrt()
        quats = rotmat_to_quat(evecs)
        n = means.shape[0]
        k = num_sh_bases(sh_degree)
        sh = torch.zeros(n, k, 3, dtype=means.dtype, device=means.device)
        sh[:, 0] = rgb_to_sh(colors)
        return Gaussians3D(
            means=means,
            quats=quats,
            log_scales=scales.log(),
            opacity=opacity,
            sh=sh,
        )

    def subset(self, mask_or_idx: torch.Tensor) -> Gaussians3D:
        """Select a subset by boolean mask or index tensor."""
        return Gaussians3D(
            self.means[mask_or_idx],
            self.quats[mask_or_idx],
            self.log_scales[mask_or_idx],
            self.opacity[mask_or_idx],
            self.sh[mask_or_idx],
        )

    @staticmethod
    def cat(parts: list[Gaussians3D]) -> Gaussians3D:
        """Concatenate several gaussian sets (SH degrees must match)."""
        return Gaussians3D(
            torch.cat([p.means for p in parts]),
            torch.cat([p.quats for p in parts]),
            torch.cat([p.log_scales for p in parts]),
            torch.cat([p.opacity for p in parts]),
            torch.cat([p.sh for p in parts]),
        )

    def detach(self) -> Gaussians3D:
        """Detached copy (no autograd history)."""
        return Gaussians3D(
            self.means.detach().clone(),
            self.quats.detach().clone(),
            self.log_scales.detach().clone(),
            self.opacity.detach().clone(),
            self.sh.detach().clone(),
        )

    def to(self, device: torch.device | str) -> Gaussians3D:
        """Return a copy on ``device``."""
        return Gaussians3D(
            self.means.to(device),
            self.quats.to(device),
            self.log_scales.to(device),
            self.opacity.to(device),
            self.sh.to(device),
        )

    def with_sh_degree(self, degree: int) -> Gaussians3D:
        """Return a copy padded or truncated to ``degree`` spherical harmonics."""
        k = num_sh_bases(degree)
        if k == self.sh.shape[1]:
            return self.detach()
        sh = self.sh.new_zeros(self.n, k, 3)
        shared = min(k, self.sh.shape[1])
        sh[:, :shared] = self.sh[:, :shared]
        return Gaussians3D(
            self.means.detach().clone(),
            self.quats.detach().clone(),
            self.log_scales.detach().clone(),
            self.opacity.detach().clone(),
            sh,
        )

    def save_npz(self, path: str | Path) -> None:
        """Save to a compressed .npz archive."""
        np.savez_compressed(
            Path(path),
            means=self.means.detach().cpu().numpy(),
            quats=self.quats.detach().cpu().numpy(),
            log_scales=self.log_scales.detach().cpu().numpy(),
            opacity=self.opacity.detach().cpu().numpy(),
            sh=self.sh.detach().cpu().numpy(),
        )

    @staticmethod
    def load_npz(path: str | Path) -> Gaussians3D:
        """Load from :meth:`save_npz` output."""
        data = np.load(Path(path))
        return Gaussians3D(
            means=torch.from_numpy(data["means"]).float(),
            quats=torch.from_numpy(data["quats"]).float(),
            log_scales=torch.from_numpy(data["log_scales"]).float(),
            opacity=torch.from_numpy(data["opacity"]).float(),
            sh=torch.from_numpy(data["sh"]).float(),
        )

    def save_ply(self, path: str | Path) -> None:
        """Export in the standard INRIA-3DGS PLY layout (opens in common 3DGS viewers)."""
        n = self.n
        k = self.sh.shape[1]
        sh_np = self.sh.detach().cpu().numpy()
        f_dc = sh_np[:, 0, :]  # (N,3)
        f_rest = sh_np[:, 1:, :].transpose(0, 2, 1).reshape(n, -1)  # channel-major like 3DGS
        opacity_logit = torch.logit(self.opacity.clamp(1e-6, 1 - 1e-6)).detach().cpu().numpy()

        props = ["x", "y", "z", "nx", "ny", "nz"]
        props += [f"f_dc_{i}" for i in range(3)]
        props += [f"f_rest_{i}" for i in range(3 * (k - 1))]
        props += ["opacity"]
        props += [f"scale_{i}" for i in range(3)]
        props += [f"rot_{i}" for i in range(4)]

        data = np.concatenate(
            [
                self.means.detach().cpu().numpy(),
                np.zeros((n, 3), dtype=np.float32),
                f_dc,
                f_rest,
                opacity_logit[:, None],
                self.log_scales.detach().cpu().numpy(),
                self.quats.detach().cpu().numpy(),
            ],
            axis=1,
        ).astype("<f4")

        header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
        header += [f"property float {p}" for p in props]
        header += ["end_header"]
        with open(Path(path), "wb") as f:
            f.write(("\n".join(header) + "\n").encode("ascii"))
            f.write(data.tobytes())

    @staticmethod
    def load_ply(path: str | Path) -> Gaussians3D:
        """Load a PLY written by :meth:`save_ply` (or any standard 3DGS PLY)."""
        with open(Path(path), "rb") as f:
            props: list[str] = []
            n = 0
            while True:
                line = f.readline().decode("ascii").strip()
                if line.startswith("element vertex"):
                    n = int(line.split()[-1])
                elif line.startswith("property float"):
                    props.append(line.split()[-1])
                elif line == "end_header":
                    break
            data = np.frombuffer(f.read(4 * n * len(props)), dtype="<f4").reshape(n, len(props))
        col = {p: i for i, p in enumerate(props)}
        means = data[:, [col["x"], col["y"], col["z"]]]
        n_rest = sum(1 for p in props if p.startswith("f_rest_"))
        k = 1 + n_rest // 3
        sh = np.zeros((n, k, 3), dtype=np.float32)
        sh[:, 0, :] = data[:, [col["f_dc_0"], col["f_dc_1"], col["f_dc_2"]]]
        if n_rest:
            rest = data[:, [col[f"f_rest_{i}"] for i in range(n_rest)]]
            sh[:, 1:, :] = rest.reshape(n, 3, k - 1).transpose(0, 2, 1)
        log_scales = data[:, [col["scale_0"], col["scale_1"], col["scale_2"]]]
        quats = data[:, [col["rot_0"], col["rot_1"], col["rot_2"], col["rot_3"]]]
        opacity = 1.0 / (1.0 + np.exp(-data[:, col["opacity"]]))
        return Gaussians3D(
            means=torch.from_numpy(means.copy()),
            quats=torch.from_numpy(quats.copy()),
            log_scales=torch.from_numpy(log_scales.copy()),
            opacity=torch.from_numpy(opacity.copy()),
            sh=torch.from_numpy(sh),
        )
