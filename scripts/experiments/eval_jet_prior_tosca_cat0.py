"""Geometry evaluation for task 20261007_jet_consistency_prior_tosca_cat0.

    .venv/bin/python scripts/experiments/eval_jet_prior_tosca_cat0.py <gaussians.npz> [--out DIR]

Models are trained in mesh units (the frame of ``cat0.obj``). For splats with opacity > 0.3:

* normal angle: smallest scale axis vs the face normal at the closest mesh point (trimesh),
  unoriented, degrees; centre distance to that point (mesh units, / bbox diagonal, / h_i with
  h_i the mean distance to the 6 nearest subset centres);
* export in the SplatDiffuseLBO splat frame (``data/cat_align.npz``: x_mesh = s R x_splat + t,
  so x_splat = R^T (x_mesh - t) / s; quaternions rotated by R^T; log scales - log s);
* Zhou-Laehner point-cloud Laplacian (robust_laplacian defaults, shift-invert eigsh as in the
  companion's case_e) on the exported centres, ``max_{k<=10} |lambda_k / lambda_k^P2 - 1|``
  against ``data/cat_reference.npz`` (splat units). robust_laplacian lives in the companion's
  Python environment, so this step runs there in a subprocess.

The companion's own operator (``splat_lbo_cat.py``) is not run here: it reads a hard-coded PLY,
and its v3 assembly needs > 11 GB host RAM and ~35 min per model (see the task record).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
COMPANION = Path.home() / "Documents/SplatDiffuseLBO"
COMPANION_PYTHON = Path.home() / "miniconda3/bin/python3.12"
MESH = ROOT / "dataset/external/tosca_cat0/source/cat0.obj"
OPACITY_MIN = 0.3

ZL_SNIPPET = r"""
import sys, json, numpy as np, robust_laplacian
from scipy.sparse.linalg import eigsh, splu, LinearOperator
c = np.load(sys.argv[1])["c"]; lam_p2 = np.load(sys.argv[2])["lam_p2"]
L, M = robust_laplacian.point_cloud_laplacian(c)
L, M = L.tocsc(), M.tocsc()
s = -1e-8 * L.diagonal().max()
lu = splu((L - s * M).tocsc(), permc_spec="MMD_AT_PLUS_A")
lam = np.sort(eigsh(L, 32, M=M, sigma=s, which="LM", OPinv=LinearOperator(L.shape, lu.solve))[0])
err = lam[1:31] / lam_p2[1:31] - 1
print(json.dumps({"lam": lam.tolist(), "err": err.tolist(),
                  "max_err_k10": float(np.abs(err[:10]).max())}))
"""


def subset(model):
    keep = model.opacity.detach() > OPACITY_MIN
    return model.subset(keep), int(keep.sum())


def normals(model) -> np.ndarray:
    from rtgs.optim.jet_prior import splat_frames

    return splat_frames(model.quats, model.log_scales)[:, 2].detach().cpu().double().numpy()


def mean_knn_distance(points: np.ndarray, k: int = 6) -> np.ndarray:
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(points).query(points, k + 1)
    return distances[:, 1:].mean(1)


def geometry(model) -> dict:
    import trimesh

    mesh = trimesh.load(MESH, process=False)
    diagonal = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    kept, count = subset(model)
    centres = kept.means.detach().cpu().double().numpy()
    closest, distance, face = trimesh.proximity.closest_point(mesh, centres)
    cosine = np.abs((normals(kept) * mesh.face_normals[face]).sum(1)).clip(0, 1)
    angle = np.degrees(np.arccos(cosine))
    h = mean_knn_distance(centres)
    ratio = distance / h
    return {
        "n_total": model.n,
        "n_subset": count,
        "normal_angle_median": float(np.median(angle)),
        "normal_angle_p90": float(np.percentile(angle, 90)),
        "normal_frac_gt30": float((angle > 30).mean()),
        "centre_distance_median_mesh_units": float(np.median(distance)),
        "centre_distance_p90_mesh_units": float(np.percentile(distance, 90)),
        "centre_distance_median": float(np.median(distance) / diagonal),
        "centre_distance_p90": float(np.percentile(distance, 90) / diagonal),
        "centre_distance_over_h_median": float(np.median(ratio)),
        "h_median_mesh_units": float(np.median(h)),
        "bbox_diagonal": diagonal,
    }


def to_companion_frame(model):
    """Return the model in the SplatDiffuseLBO splat frame."""
    from rtgs.core.gaussians3d import Gaussians3D, quat_to_rotmat, rotmat_to_quat

    align = np.load(COMPANION / "data/cat_align.npz")
    scale = float(align["scale_splat_to_mesh"])
    rotation = torch.tensor(align["R"], dtype=torch.float64)
    shift = torch.tensor(align["t"], dtype=torch.float64)
    means = ((model.means.detach().double() - shift) @ rotation) / scale  # R^T (x - t) / s
    rot = rotation.T @ quat_to_rotmat(model.quats.detach().double())
    return Gaussians3D(
        means=means.float(),
        quats=rotmat_to_quat(rot).float(),
        log_scales=model.log_scales.detach() - float(np.log(scale)),
        opacity=model.opacity.detach(),
        sh=model.sh.detach(),
    )


def zhou_laehner(centres_splat: np.ndarray, scratch: Path) -> dict:
    path = scratch / "zl_centres.npz"
    np.savez(path, c=centres_splat.astype(np.float64))
    result = subprocess.run(
        [
            str(COMPANION_PYTHON),
            "-c",
            ZL_SNIPPET,
            str(path),
            str(COMPANION / "data/cat_reference.npz"),
        ],
        capture_output=True,
        text=True,
        check=True,
        env={"OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "PATH": "/usr/bin:/bin"},
        timeout=1800,
    )
    path.unlink()
    return json.loads(result.stdout.strip().splitlines()[-1])


def evaluate_model(model, out: Path) -> dict:
    """Geometry in mesh units, export in the companion frame, ZL spectrum on its centres."""
    model = model.to("cpu")
    record = geometry(model)
    companion = to_companion_frame(model)
    companion.save_ply(out / "gaussians_splat_frame.ply")
    kept, _ = subset(companion)
    zl = zhou_laehner(kept.means.double().numpy(), out)
    record["zl_err_k10"] = zl["max_err_k10"]
    record["zl_lambda"] = zl["lam"]
    record["splat_frame_ply"] = "gaussians_splat_frame.ply"
    record["lbo_err_k10"] = None  # companion operator deferred (host-RAM limit), see task record
    return record


def main() -> None:
    from rtgs.core.gaussians3d import Gaussians3D

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.model.parent
    out.mkdir(parents=True, exist_ok=True)
    load = Gaussians3D.load_npz if args.model.suffix == ".npz" else Gaussians3D.load_ply
    print(json.dumps(evaluate_model(load(args.model), out), indent=1))


if __name__ == "__main__":
    main()
