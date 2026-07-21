"""Tests for the igsv live-viewer bridge (skipped when igsv is not installed)."""

from __future__ import annotations

import asyncio
import struct

import numpy as np
import pytest
import torch

igsv = pytest.importorskip("igsv")

from rtgs.core.gaussians3d import Gaussians3D  # noqa: E402
from rtgs.live import LiveViewer, gaussians3d_to_frame  # noqa: E402


def _tiny_gaussians(n: int = 24, sh_bases: int = 4) -> Gaussians3D:
    g = torch.Generator().manual_seed(11)
    quats = torch.randn(n, 4, generator=g)
    return Gaussians3D(
        means=torch.randn(n, 3, generator=g),
        quats=quats / quats.norm(dim=-1, keepdim=True),
        log_scales=torch.randn(n, 3, generator=g) * 0.3 - 3.0,
        opacity=torch.rand(n, generator=g),
        sh=torch.randn(n, sh_bases, 3, generator=g) * 0.3,
    )


def test_frame_conversion_matches_fields():
    g3d = _tiny_gaussians()
    frame = gaussians3d_to_frame(g3d, name="conv-test")

    assert frame.n == g3d.n
    assert frame.name == "conv-test"
    np.testing.assert_allclose(frame.positions, g3d.means.numpy(), rtol=1e-6)
    np.testing.assert_allclose(frame.quats_wxyz, g3d.quats.numpy(), rtol=1e-6)
    np.testing.assert_allclose(frame.scales, g3d.log_scales.exp().numpy(), rtol=1e-6)
    np.testing.assert_allclose(frame.opacities, g3d.opacity.numpy(), rtol=1e-6)
    np.testing.assert_allclose(frame.sh0, g3d.sh[:, 0, :].numpy(), rtol=1e-6)
    assert frame.sh_rest is not None
    assert frame.sh_rest.shape == (g3d.n, 3, 3)  # bases 1..3 of degree-1 SH
    assert frame.sh_degree == 1


def test_frame_conversion_degree0_has_no_rest():
    frame = gaussians3d_to_frame(_tiny_gaussians(sh_bases=1))
    assert frame.sh_rest is None
    assert frame.sh_degree == 0


def test_frame_is_detached_copy():
    g3d = _tiny_gaussians(n=8)
    g3d.means.requires_grad_(True)
    frame = gaussians3d_to_frame(g3d)
    frame.positions[0, 0] = 123.0
    assert float(g3d.means.detach()[0, 0]) != 123.0


def test_live_viewer_streams_checkpoint_callback():
    """checkpoint_callback -> igsv server -> real WebSocket keyframe + stats."""
    g3d = _tiny_gaussians(n=16)
    viewer = LiveViewer(port=0, name="cb-test")
    viewer.start()
    try:
        viewer.checkpoint_callback(g3d, 7)  # trainer-shaped call from the main thread
        port = viewer.server.bound_port

        async def probe() -> tuple[int, int]:
            import aiohttp
            from igsv import protocol

            total = None
            async with (
                aiohttp.ClientSession() as session,
                session.ws_connect(f"http://127.0.0.1:{port}/ws") as ws,
            ):
                for _ in range(60):  # publish() is async server-side; poll messages
                    msg = protocol.decode_message(await asyncio.wait_for(ws.receive_bytes(), 5))
                    if msg.type == protocol.MessageType.KEYFRAME_END:
                        total = struct.unpack_from("<II", msg.payload)[1]
                    elif msg.type == protocol.MessageType.STATS and total is not None:
                        step = struct.unpack_from("<I", msg.payload)[0]
                        return total, step
            raise AssertionError("no keyframe/stats received")

        total, step = asyncio.run(asyncio.wait_for(probe(), timeout=10))
        assert total == 16
        assert step == 7
    finally:
        viewer.stop()
