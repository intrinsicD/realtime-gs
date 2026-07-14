"""Interactive browser viewer for saved 3D Gaussian reconstructions.

The live, orbitable view uses Viser's WebGL Gaussian renderer.  Viser is an optional
dependency and is imported only when :func:`create_viewer` is called, so importing rtgs and
running the CPU test suite never requires it.  Calibrated-camera snapshots use the regular
``Rasterizer`` abstraction and can therefore be rendered exactly with gsplat on CUDA.
"""

from __future__ import annotations

import math
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rtgs.core.camera import Camera
from rtgs.core.gaussians3d import Gaussians3D, rotmat_to_quat
from rtgs.core.sh import sh_to_rgb


@dataclass(frozen=True)
class ViewerSplatData:
    """CPU arrays ordered from most to least visually significant."""

    centers: np.ndarray
    covariances: np.ndarray
    rgbs: np.ndarray
    opacities: np.ndarray
    source_indices: torch.Tensor

    @property
    def n(self) -> int:
        """Number of finite splats available to the viewer."""
        return int(self.centers.shape[0])


@dataclass
class ViewerApp:
    """Running viewer handles, primarily useful for embedding and smoke tests."""

    server: Any
    _gaussian_handle: list[Any]
    frustum_handles: list[Any]

    @property
    def gaussian_handle(self) -> Any:
        """Current splat handle (it is replaced when model/count controls change)."""
        return self._gaussian_handle[0]

    def stop(self) -> None:
        """Stop the HTTP/WebSocket server."""
        self.server.stop()


def prepare_viewer_data(
    gaussians: Gaussians3D, max_gaussians: int | None = None
) -> ViewerSplatData:
    """Convert a Gaussian set to Viser arrays and compute a useful prefix ordering.

    Prefixes are ranked by opacity times ellipsoid cross-sectional area.  This makes the
    splat-count slider useful: lowering the count keeps high-contribution splats instead of
    relying on arbitrary PLY row order.  Non-finite primitives are omitted from the preview.
    """
    if max_gaussians is not None and max_gaussians < 1:
        raise ValueError("max_gaussians must be positive")

    with torch.no_grad():
        g = gaussians.to("cpu")
        covariances = g.covariance()
        rgbs = sh_to_rgb(g.sh[:, 0])
        opacities = g.opacity.clamp(0.0, 1.0)
        finite = (
            torch.isfinite(g.means).all(dim=-1)
            & torch.isfinite(covariances).all(dim=(-2, -1))
            & torch.isfinite(rgbs).all(dim=-1)
            & torch.isfinite(opacities)
        )
        source_indices = finite.nonzero(as_tuple=True)[0]
        if source_indices.numel() == 0:
            raise ValueError("gaussian set has no finite primitives to display")

        scales = g.scales[source_indices]
        cross_section = (
            scales[:, 0] * scales[:, 1] + scales[:, 0] * scales[:, 2] + scales[:, 1] * scales[:, 2]
        )
        importance = opacities[source_indices] * cross_section
        order = torch.argsort(importance, descending=True, stable=True)
        source_indices = source_indices[order]
        if max_gaussians is not None:
            source_indices = source_indices[:max_gaussians]

        return ViewerSplatData(
            centers=np.ascontiguousarray(g.means[source_indices].numpy(), dtype=np.float32),
            covariances=np.ascontiguousarray(covariances[source_indices].numpy(), dtype=np.float32),
            rgbs=np.ascontiguousarray(rgbs[source_indices].numpy(), dtype=np.float32),
            opacities=np.ascontiguousarray(
                opacities[source_indices, None].numpy(), dtype=np.float32
            ),
            source_indices=source_indices,
        )


def selected_gaussians(
    gaussians: Gaussians3D,
    data: ViewerSplatData,
    count: int,
    opacity_scale: float,
) -> Gaussians3D:
    """Apply the viewer's count/opacity controls to an exact-render Gaussian set."""
    count = min(max(int(count), 1), data.n)
    selected = gaussians.subset(data.source_indices[:count].to(gaussians.means.device))
    return Gaussians3D(
        means=selected.means,
        quats=selected.quats,
        log_scales=selected.log_scales,
        opacity=(selected.opacity * opacity_scale).clamp(0.0, 1.0),
        sh=selected.sh,
    )


def camera_to_viewer_pose(camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Return Viser/OpenCV camera-to-world ``(wxyz, position)`` arrays."""
    rotation_c2w = camera.R.detach().cpu().T.contiguous()
    wxyz = rotmat_to_quat(rotation_c2w[None])[0].numpy().astype(np.float64)
    position = camera.position.detach().cpu().numpy().astype(np.float64)
    return wxyz, position


def _camera_vertical_fov(camera: Camera) -> float:
    return 2.0 * math.atan(0.5 * camera.height / camera.fy)


def _image_uint8(image: torch.Tensor, max_side: int | None = None) -> np.ndarray:
    image = image.detach().cpu().clamp(0.0, 1.0)
    if max_side is not None and max(image.shape[:2]) > max_side:
        scale = max_side / max(image.shape[:2])
        height = max(1, round(image.shape[0] * scale))
        width = max(1, round(image.shape[1] * scale))
        image = torch.nn.functional.interpolate(
            image.permute(2, 0, 1)[None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0)
    return np.ascontiguousarray((image.numpy() * 255.0).round().astype(np.uint8))


def _require_viser():
    try:
        import viser
    except ImportError as exc:
        raise RuntimeError(
            "the interactive viewer is optional; install it with `pip install -e '.[viewer]'`"
        ) from exc
    return viser


def create_viewer(
    models: dict[str, Gaussians3D],
    *,
    scene=None,
    device: torch.device | str = "cpu",
    snapshot_rasterizer: str = "auto",
    snapshot_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    max_viewer_gaussians: int | None = None,
) -> ViewerApp:
    """Create and populate a non-blocking Viser server.

    Args:
        models: Named Gaussian sets. The first entry is selected initially.
        scene: Optional ``SceneData``; enables calibrated cameras, RGB references, and exact
            snapshots through the rasterizer abstraction.
        device: Device used only for exact snapshots.
        snapshot_rasterizer: ``auto``, ``torch``, or ``gsplat``.
        snapshot_dir: Optional directory in which exact snapshot PNGs are also saved.
        host/port: Viser server binding.
        max_viewer_gaussians: Optional transfer/display cap, independent of fitted counts.
    """
    if not models:
        raise ValueError("at least one Gaussian model is required")
    viser = _require_viser()
    prepared = {
        name: prepare_viewer_data(model, max_gaussians=max_viewer_gaussians)
        for name, model in models.items()
    }
    model_names = tuple(models)
    current_name = model_names[0]
    current_data = prepared[current_name]

    server = viser.ViserServer(host=host, port=port, label="realtime-gs")
    server.scene.set_up_direction("+z")
    gaussian_handle = server.scene.add_gaussian_splats(
        "/reconstruction",
        centers=current_data.centers,
        covariances=current_data.covariances,
        rgbs=current_data.rgbs,
        opacities=current_data.opacities,
    )
    gaussian_handle_box = [gaussian_handle]

    if scene is not None:
        viewer_center, viewer_extent = scene.center_and_extent()
        viewer_center = viewer_center.detach().cpu().numpy().astype(np.float64)
    else:
        viewer_center = np.median(current_data.centers, axis=0).astype(np.float64)
        radii = np.linalg.norm(current_data.centers - viewer_center[None], axis=-1)
        viewer_extent = max(2.2 * float(np.quantile(radii, 0.99)), 1e-3)

    frustum_handles: list[Any] = []
    camera_labels: list[str] = []
    if scene is not None:
        names = scene.view_names or [f"view_{index:04d}" for index in range(scene.n_views)]
        test_indices = set(scene.testing_views)
        _, extent = scene.center_and_extent()
        for index, (camera, image, name) in enumerate(
            zip(scene.cameras, scene.images, names, strict=True)
        ):
            split = "test" if index in test_indices else "train"
            label = f"{index:03d} · {name} · {split}"
            camera_labels.append(label)
            wxyz, position = camera_to_viewer_pose(camera)
            handle = server.scene.add_camera_frustum(
                f"/cameras/{index:04d}_{name}",
                fov=_camera_vertical_fov(camera),
                aspect=camera.width / camera.height,
                scale=max(extent * 0.06, 1e-4),
                color=(230, 120, 40) if split == "test" else (50, 140, 230),
                image=_image_uint8(image, max_side=320),
                format="jpeg",
                jpeg_quality=80,
                wxyz=wxyz,
                position=position,
            )
            frustum_handles.append(handle)

    server.gui.add_markdown(
        "**realtime-gs viewer**\n\nOrbit in the viewport. The WebGL preview uses degree-0 "
        "color; exact snapshots use every active SH band."
    )
    model_control = server.gui.add_dropdown(
        "Gaussian set", options=model_names, initial_value=current_name
    )
    count_control = server.gui.add_slider(
        "Displayed splats",
        min=1,
        max=current_data.n,
        step=max(1, current_data.n // 500),
        initial_value=current_data.n,
    )
    opacity_control = server.gui.add_slider(
        "Opacity multiplier", min=0.0, max=2.0, step=0.05, initial_value=1.0
    )
    frame_button = server.gui.add_button("Frame reconstruction")

    show_cameras = None
    split_control = None
    camera_control = None
    reference_handle = None
    snapshot_handle = None
    snapshot_status = None
    jump_button = None
    snapshot_button = None
    if scene is not None:
        show_cameras = server.gui.add_checkbox("Show cameras", initial_value=True)
        split_options = ["all", "train"]
        if scene.testing_views:
            split_options.append("test")
        split_control = server.gui.add_dropdown(
            "Camera split", options=split_options, initial_value="all"
        )
        camera_control = server.gui.add_dropdown(
            "Selected camera", options=camera_labels, initial_value=camera_labels[0]
        )
        jump_button = server.gui.add_button("Jump to camera")
        reference_handle = server.gui.add_image(
            _image_uint8(scene.images[0]), label="Reference RGB", format="jpeg"
        )
        snapshot_handle = server.gui.add_image(
            np.zeros((32, 32, 3), dtype=np.uint8), label="Exact rasterizer snapshot"
        )
        snapshot_button = server.gui.add_button("Render exact snapshot", color="green")
        snapshot_status = server.gui.add_markdown(
            f"Snapshot backend: `{snapshot_rasterizer}` on `{device}`."
        )

    lock = threading.Lock()

    def selected_camera_index() -> int:
        assert camera_control is not None
        return camera_labels.index(camera_control.value)

    def set_client_camera(client: Any, index: int) -> None:
        assert scene is not None
        camera = scene.cameras[index]
        wxyz, position = camera_to_viewer_pose(camera)
        client.camera.position = position
        client.camera.wxyz = wxyz
        client.camera.fov = _camera_vertical_fov(camera)

    def frame_client_camera(client: Any) -> None:
        offset = viewer_extent * np.array([0.8, -0.8, 0.6], dtype=np.float64)
        client.camera.position = viewer_center + offset
        client.camera.look_at = viewer_center
        client.camera.up_direction = np.array([0.0, 0.0, 1.0])

    def replace_splats() -> None:
        nonlocal current_data
        with lock:
            current_data = prepared[model_control.value]
            count_control.max = current_data.n
            count_control.step = max(1, current_data.n // 500)
            count_control.value = min(int(count_control.value), current_data.n)
            count = int(count_control.value)
            opacity = np.clip(
                current_data.opacities[:count] * float(opacity_control.value), 0.0, 1.0
            )
            gaussian_handle_box[0].remove()
            gaussian_handle_box[0] = server.scene.add_gaussian_splats(
                "/reconstruction",
                centers=current_data.centers[:count],
                covariances=current_data.covariances[:count],
                rgbs=current_data.rgbs[:count],
                opacities=opacity,
            )

    @model_control.on_update
    def _(_: Any) -> None:
        new_count = prepared[model_control.value].n
        count_control.max = new_count
        count_control.step = max(1, new_count // 500)
        if int(count_control.value) == new_count:
            replace_splats()
        else:
            # Assignment invokes the count callback synchronously.
            count_control.value = new_count

    @count_control.on_update
    def _(_: Any) -> None:
        replace_splats()

    @opacity_control.on_update
    def _(_: Any) -> None:
        with lock:
            count = int(count_control.value)
            gaussian_handle_box[0].opacities = np.clip(
                current_data.opacities[:count] * float(opacity_control.value), 0.0, 1.0
            )

    @frame_button.on_click
    def _(event: Any) -> None:
        if event.client is not None:
            frame_client_camera(event.client)

    if scene is not None:
        assert show_cameras is not None
        assert split_control is not None
        assert camera_control is not None
        assert reference_handle is not None
        assert snapshot_handle is not None
        assert snapshot_status is not None
        assert jump_button is not None
        assert snapshot_button is not None

        def update_frustum_visibility() -> None:
            test_indices = set(scene.testing_views)
            for index, handle in enumerate(frustum_handles):
                matching_split = (
                    split_control.value == "all"
                    or (split_control.value == "test" and index in test_indices)
                    or (split_control.value == "train" and index not in test_indices)
                )
                handle.visible = bool(show_cameras.value and matching_split)

        @show_cameras.on_update
        def _(_: Any) -> None:
            update_frustum_visibility()

        @split_control.on_update
        def _(_: Any) -> None:
            update_frustum_visibility()

        @camera_control.on_update
        def _(_: Any) -> None:
            reference_handle.image = _image_uint8(scene.images[selected_camera_index()])

        @jump_button.on_click
        def _(event: Any) -> None:
            if event.client is not None:
                set_client_camera(event.client, selected_camera_index())

        for index, handle in enumerate(frustum_handles):

            @handle.on_click
            def _(event: Any, camera_index: int = index) -> None:
                camera_control.value = camera_labels[camera_index]
                if event.client is not None:
                    set_client_camera(event.client, camera_index)

        @snapshot_button.on_click
        def _(_: Any) -> None:
            snapshot_button.disabled = True
            index = selected_camera_index()
            name = model_control.value
            try:
                from PIL import Image as PILImage

                from rtgs.render.base import get_rasterizer

                snapshot_status.content = (
                    f"Rendering `{name}` camera {index} with `{snapshot_rasterizer}`…"
                )
                data = prepared[name]
                selected = selected_gaussians(
                    models[name], data, int(count_control.value), float(opacity_control.value)
                ).to(device)
                camera = scene.cameras[index].to(device)
                renderer = get_rasterizer(snapshot_rasterizer, device=device)
                with torch.no_grad():
                    output = renderer.render(selected, camera).color.clamp(0.0, 1.0)
                image = _image_uint8(output)
                snapshot_handle.image = image
                saved = ""
                if snapshot_dir is not None:
                    snapshot_dir.mkdir(parents=True, exist_ok=True)
                    path = snapshot_dir / f"{name}_camera_{index:04d}.png"
                    PILImage.fromarray(image).save(path)
                    saved = f" Saved `{path}`."
                snapshot_status.content = (
                    f"Rendered `{name}` camera {index}: {selected.n:,} splats with "
                    f"`{snapshot_rasterizer}` on `{device}`.{saved}"
                )
            except Exception as exc:  # keep the viewer alive and expose actionable errors
                snapshot_status.content = f"Snapshot failed: `{type(exc).__name__}: {exc}`"
            finally:
                snapshot_button.disabled = False

        @server.on_client_connect
        def _(client: Any) -> None:
            set_client_camera(client, selected_camera_index())

    else:

        @server.on_client_connect
        def _(client: Any) -> None:
            frame_client_camera(client)

    return ViewerApp(
        server=server,
        _gaussian_handle=gaussian_handle_box,
        frustum_handles=frustum_handles,
    )


def launch_viewer(
    models: dict[str, Gaussians3D],
    *,
    open_browser: bool = True,
    **kwargs: Any,
) -> None:
    """Run :func:`create_viewer` until interrupted with Ctrl-C."""
    app = create_viewer(models, **kwargs)
    server = app.server
    browser_host = server.get_host()
    if browser_host in {"0.0.0.0", "::"}:
        browser_host = "127.0.0.1"
    url = f"http://{browser_host}:{server.get_port()}"
    print(f"interactive viewer -> {url}  (Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("stopping viewer")
    finally:
        app.stop()
