"""Workflow tooling: CLI end-to-end and docs-sync checker."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from rtgs.cli import main as cli_main

REPO = Path(__file__).resolve().parent.parent
TINY_SCENE = "synthetic:n_gaussians=10,n_cameras=4,image_size=24"


def test_cli_run_end_to_end(tmp_path, capsys):
    rc = cli_main(
        [
            "run",
            "--scene",
            TINY_SCENE,
            "--n-gaussians",
            "40",
            "--fit-iterations",
            "20",
            "--refine-iters",
            "5",
            "--lifter",
            "depth",
            "--rasterizer",
            "torch",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert "init_psnr" in metrics["metrics"]
    assert (tmp_path / "out" / "gaussians_init.ply").exists()
    assert (tmp_path / "out" / "gaussians.ply").exists()
    assert (tmp_path / "out" / "reconstruction_contact_sheet.png").exists()
    assert (tmp_path / "out" / "reconstruction.gif").exists()
    printed = json.loads(capsys.readouterr().out.split("saved")[0])
    assert printed["metrics"]["init_n_gaussians"] > 0


def test_cli_fit_and_render(tmp_path):
    from PIL import Image as PILImage

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    gen = torch.Generator().manual_seed(0)
    for i in range(2):
        arr = (torch.rand(16, 16, 3, generator=gen).numpy() * 255).astype(np.uint8)
        PILImage.fromarray(arr).save(img_dir / f"im_{i}.png")
    rc = cli_main(
        [
            "fit-images",
            "--images",
            str(img_dir),
            "--out",
            str(tmp_path / "fits"),
            "--n-gaussians",
            "20",
            "--iterations",
            "10",
        ]
    )
    assert rc == 0
    assert len(list((tmp_path / "fits").glob("*.npz"))) == 2

    # Render a saved gaussian set from synthetic cameras.
    from rtgs.data.synthetic import make_gt_gaussians

    g = make_gt_gaussians(n=8, seed=0)
    g.save_ply(tmp_path / "g.ply")
    rc = cli_main(
        [
            "render",
            "--scene",
            TINY_SCENE,
            "--gaussians",
            str(tmp_path / "g.ply"),
            "--rasterizer",
            "torch",
            "--out",
            str(tmp_path / "renders"),
        ]
    )
    assert rc == 0
    assert len(list((tmp_path / "renders").glob("*.png"))) == 4


def test_docs_sync_passes():
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "docs_sync.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
