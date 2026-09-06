#!/usr/bin/env python3
"""Reproduce RTGS-016's published crop in the original full-SH PLY frame.

This is a capture-specific import receipt replay, not a general Polycam decoder.
The separately stored similarity was verified against 20,000 appearance/shape
matches and every published center. Nothing here enters reconstruction workers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def main() -> None:
    import numpy as np
    import torch
    from scipy.spatial import cKDTree

    from rtgs.core.gaussians3d import Gaussians3D

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads((args.source / "import_alignment.json").read_text())
    for name, digest in receipt["source_files"].items():
        if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"capture bytes changed: {name}")
    payload = (args.source / "published.splat").read_bytes()
    if payload[:6] != b"splat\x02":
        raise ValueError("not the sealed Polycam degree-2 capture payload")
    count = struct.unpack_from("<I", payload, 6)[0]
    if len(payload) != 64 + 56 * count or count != receipt["published_count"]:
        raise ValueError("sealed published payload length/count changed")
    # First block: standard 32-byte splats; remaining SH bytes are intentionally
    # never interpreted. Full-precision SH comes from the original PLY instead.
    rows = np.ndarray(
        (count,),
        dtype=[("position", "<f4", (3,)), ("rest", "u1", (20,))],
        buffer=payload,
        offset=64,
    )
    original = Gaussians3D.load_ply(args.source / "reference.ply")
    rotation = np.asarray(receipt["rotation"])
    mapped = original.means.numpy() @ rotation.T * receipt["scale"] + receipt["translation"]
    distance, indices = cKDTree(mapped).query(rows["position"])
    if not np.isfinite(distance).all() or distance.max() >= 2e-6:
        raise ValueError("published centers do not match the original PLY")
    if len(np.unique(indices)) != count:
        raise ValueError("published crop is not a one-to-one original-row selection")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    selected = Gaussians3D(
        **{
            name: getattr(original, name)[torch.from_numpy(indices)].clone()
            for name in ("means", "quats", "log_scales", "opacity", "sh")
        }
    )
    selected.save_ply(args.output)
    if hashlib.sha256(args.output.read_bytes()).hexdigest() != receipt["output"]["sha256"]:
        raise ValueError("replayed full-SH crop differs from the acquisition receipt")
    print(f"Reproduced {count} original degree-{selected.sh_degree} Gaussians: {args.output}")


if __name__ == "__main__":
    main()
