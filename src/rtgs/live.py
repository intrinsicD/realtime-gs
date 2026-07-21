"""Live-training bridge to the igsv browser viewer (interactive-gs-viewer repo).

Streams :class:`~rtgs.core.gaussians3d.Gaussians3D` snapshots over igsv's binary
WebSocket protocol so a WebGPU browser client can watch stage-3 refinement live
(``rtgs refine --live``); the same server keeps serving the final state for
inspection afterwards. ``igsv`` is an optional dependency imported lazily inside
functions — importing this module needs torch only, and the CPU test suite skips
these tests when igsv is absent. The browser view is a diagnostic (CLAUDE.md
rule 7); quantitative decisions use the exact rasterizer.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from rtgs.core.gaussians3d import Gaussians3D

if TYPE_CHECKING:  # pragma: no cover - typing only
    import igsv


def _require_igsv() -> Any:
    try:
        import igsv
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "live viewing requires the optional igsv package from the "
            "interactive-gs-viewer repository: pip install -e <path>/interactive-gs-viewer/server"
        ) from exc
    return igsv


def gaussians3d_to_frame(gaussians: Gaussians3D, name: str = "scene") -> igsv.SplatFrame:
    """Convert a Gaussians3D set to an igsv SplatFrame (detached CPU copy).

    Field mapping is 1:1: both sides use (w, x, y, z) quaternions, linear scales,
    actual [0, 1] opacity, and SH DC coefficients with the C0 convention.
    """
    igsv_mod = _require_igsv()
    g = gaussians

    def to_np(t):  # detach -> CPU -> numpy copy; safe to call from a callback
        return t.detach().to("cpu").numpy().copy()

    sh_rest = to_np(g.sh[:, 1:, :]) if g.sh.shape[1] > 1 else None
    return igsv_mod.SplatFrame(
        positions=to_np(g.means),
        quats_wxyz=to_np(g.quats),
        scales=to_np(g.scales),
        opacities=to_np(g.opacity),
        sh0=to_np(g.sh[:, 0, :]),
        sh_rest=sh_rest,
        name=name,
    )


class LiveViewer:
    """igsv server wrapper with a trainer-checkpoint-shaped publishing API.

    Usage inside a training script::

        viewer = LiveViewer(port=8890)
        viewer.start()
        trainer.train(scene, init, checkpoint_callback=viewer.checkpoint_callback)
        viewer.stop()

    ``publish`` may be called from any thread; snapshots are coalesced by the
    igsv server, so a slow browser can never stall training.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8890,
        *,
        token: str | None = None,
        name: str = "rtgs",
    ) -> None:
        igsv_mod = _require_igsv()
        self._igsv = igsv_mod
        self.name = name
        self._server = igsv_mod.ViewerServer(host=host, port=port, token=token)
        self._t0 = time.perf_counter()
        self._last: tuple[int, float] | None = None  # (step, wall) for iters/sec

    @property
    def server(self) -> Any:
        """The underlying igsv ViewerServer (e.g. for set_control_handler)."""
        return self._server

    @property
    def url(self) -> str:
        return self._server.url

    def start(self) -> None:
        self._server.start_background()

    def stop(self) -> None:
        self._server.stop()

    def publish(
        self,
        gaussians: Gaussians3D,
        step: int = 0,
        *,
        loss: float = float("nan"),
        psnr: float = float("nan"),
    ) -> None:
        now = time.perf_counter()
        iters_per_sec = float("nan")
        if self._last is not None and now > self._last[1] and step > self._last[0]:
            iters_per_sec = (step - self._last[0]) / (now - self._last[1])
        self._last = (step, now)
        frame = gaussians3d_to_frame(gaussians, name=self.name)
        stats = self._igsv.TrainStats(
            loss=loss,
            iters_per_sec=iters_per_sec,
            psnr=psnr,
            wall_seconds=now - self._t0,
        )
        self._server.publish(frame, step=step, stats=stats)

    @property
    def checkpoint_callback(self) -> Callable[[Gaussians3D, int], None]:
        """Adapter matching ``Trainer.train(checkpoint_callback=...)``."""

        def callback(snapshot: Gaussians3D, completed_step: int) -> None:
            self.publish(snapshot, step=completed_step)

        return callback

    def __enter__(self) -> LiveViewer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
