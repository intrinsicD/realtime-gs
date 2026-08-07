"""Safe lazy adapter for the frozen GPS-Gaussian stereo submodules.

The official repository is an external, checksum-bound research dependency.  Importing this
module does not import that repository, SciPy, or a checkpoint.  A backend instance validates the
clean Git tree and tensor-only state payload immediately before constructing only the image
encoder and symmetric RAFT stereo network used by the preregistered experiment.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from rtgs.depth.stereo import (
    RectifiedStereoRequest,
    StereoDepthPrediction,
    stereo_depth_from_bidirectional_flow,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", os.fspath(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _state_rows(state: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> list[dict]:
    return [
        {"key": key, "shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(state.items())
        if key.startswith(prefixes)
    ]


def _rows_sha256(rows: list[dict]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class GPSGaussianConfig:
    """Frozen external source, state, and execution bindings."""

    repository: Path
    checkpoint: Path
    device: str = "cuda:0"
    repository_commit: str = "0024776deee4824f270d4bb534a17ffd85f63cf2"
    repository_tree: str = "ad9815910afe3cd441458a1e79dd1f56bef3ab7e"
    checkpoint_bytes: int = 20_680_271
    checkpoint_sha256: str = "6699a109af8f4cee0664fdbf9d581a9bbec74650b68e2acb90e4f040f6c5ba90"
    used_key_count: int = 132
    used_signature_sha256: str = "79fe3c02d85b60afc3b46a1dae7bdb57d9787e903e1e6e8887cf17a2510fe553"
    img_encoder_key_count: int = 64
    img_encoder_signature_sha256: str = (
        "5431406313bce1bf156e4dceb8f9b37e130c0302c3503409fb5ed491cf1b8fd2"
    )
    raft_stereo_key_count: int = 68
    raft_stereo_signature_sha256: str = (
        "735e4a2afdae68ae819c2e62b072cecb14de18c57060a0b621c3d9a67394654d"
    )
    iterations: int = 3
    mixed_precision: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "repository", Path(self.repository))
        object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if not self.mixed_precision:
            raise ValueError("the frozen GPS adapter requires mixed_precision=True")


def _validate_external_repository(config: GPSGaussianConfig) -> dict[str, object]:
    if config.repository.is_symlink():
        raise ValueError("GPS repository must not be a symlink")
    repo = config.repository.resolve(strict=True)
    if not repo.is_dir():
        raise ValueError("GPS repository must be an ordinary directory")
    head = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    status = _git(repo, "status", "--porcelain=v1")
    if head != config.repository_commit or tree != config.repository_tree or status:
        raise RuntimeError("GPS repository differs from the frozen clean commit/tree")
    return {
        "repository": os.fspath(repo),
        "repository_commit": head,
        "repository_tree": tree,
        "repository_clean": True,
    }


def load_sanitized_gps_state(config: GPSGaussianConfig) -> dict[str, torch.Tensor]:
    """Load and strictly validate the tensor-only state without deserializing arbitrary pickle."""

    if config.checkpoint.is_symlink():
        raise ValueError("GPS checkpoint must not be a symlink")
    path = config.checkpoint.resolve(strict=True)
    if not path.is_file():
        raise ValueError("GPS checkpoint must be an ordinary non-symlink file")
    if path.stat().st_size != config.checkpoint_bytes:
        raise RuntimeError("sanitized GPS checkpoint byte size differs from the frozen value")
    if _sha256_file(path) != config.checkpoint_sha256:
        raise RuntimeError("sanitized GPS checkpoint SHA-256 differs from the frozen value")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not payload:
        raise TypeError("sanitized GPS checkpoint must contain one non-empty state mapping")
    if any(not isinstance(key, str) for key in payload):
        raise TypeError("sanitized GPS state keys must be strings")
    if any(not isinstance(value, torch.Tensor) for value in payload.values()):
        raise TypeError("sanitized GPS state values must all be tensors")
    state: dict[str, torch.Tensor] = dict(payload)
    used = _state_rows(state, ("img_encoder.", "raft_stereo."))
    encoder = _state_rows(state, ("img_encoder.",))
    raft = _state_rows(state, ("raft_stereo.",))
    expected = (
        (used, config.used_key_count, config.used_signature_sha256, "combined"),
        (
            encoder,
            config.img_encoder_key_count,
            config.img_encoder_signature_sha256,
            "img_encoder",
        ),
        (
            raft,
            config.raft_stereo_key_count,
            config.raft_stereo_signature_sha256,
            "raft_stereo",
        ),
    )
    for rows, count, digest, label in expected:
        if len(rows) != count or _rows_sha256(rows) != digest:
            raise RuntimeError(f"sanitized GPS {label} key/shape/dtype signature mismatch")
    return state


def _module_origins(repo: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    repo = repo.resolve()
    for name, module in sorted(sys.modules.items()):
        if name != "core" and not name.startswith("core."):
            continue
        paths: list[Path] = []
        file_value = getattr(module, "__file__", None)
        if file_value is not None:
            paths.append(Path(file_value).resolve())
        namespace = getattr(module, "__path__", None)
        if namespace is not None:
            paths.extend(Path(item).resolve() for item in namespace)
        if not paths:
            raise RuntimeError(f"external module {name} has no verifiable origin")
        for path in paths:
            try:
                path.relative_to(repo)
            except ValueError as error:
                raise RuntimeError(
                    f"external module {name} escaped the frozen repository"
                ) from error
        source = next((path for path in paths if path.suffix == ".py" and path.is_file()), None)
        records.append(
            {
                "module": name,
                "origins": [os.fspath(path) for path in paths],
                "source_sha256": None if source is None else _sha256_file(source),
                "source_bytes": None if source is None else source.stat().st_size,
            }
        )
    return records


def _import_external_classes(config: GPSGaussianConfig):
    receipt = _validate_external_repository(config)
    repo = config.repository.resolve()
    existing = [name for name in sys.modules if name == "core" or name.startswith("core.")]
    if existing:
        _module_origins(repo)
    sys.path.insert(0, os.fspath(repo))
    try:
        extractor_module = importlib.import_module("core.extractor")
        raft_module = importlib.import_module("core.raft_stereo_human")
    finally:
        with suppress(ValueError):
            sys.path.remove(os.fspath(repo))
    receipt["imported_sources"] = _module_origins(repo)
    after = _git(repo, "status", "--porcelain=v1")
    if after:
        raise RuntimeError("GPS repository became dirty during validated imports")
    return extractor_module.UnetExtractor, raft_module.RAFTStereoHuman, receipt


class GPSGaussianStereoBackend:
    """Frozen GPS image encoder + symmetric RAFT stereo inference adapter."""

    def __init__(self, config: GPSGaussianConfig):
        self.config = config
        self._img_encoder = None
        self._raft_stereo = None
        self._receipt: dict[str, object] | None = None

    @property
    def receipt(self) -> dict[str, object]:
        """Return the validated external-source/model receipt after first use."""

        if self._receipt is None:
            raise RuntimeError("GPS backend has not been loaded")
        return dict(self._receipt)

    def _load(self) -> None:
        if self._img_encoder is not None:
            return
        device = torch.device(self.config.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("the frozen GPS backend requires an available CUDA device")
        UnetExtractor, RAFTStereoHuman, receipt = _import_external_classes(self.config)
        args = SimpleNamespace(
            mixed_precision=True,
            corr_implementation="reg",
            corr_levels=4,
            corr_radius=4,
            n_downsample=3,
            n_gru_layers=1,
            slow_fast_gru=None,
            encoder_dims=[32, 48, 96],
            hidden_dims=[96, 96, 96],
        )
        encoder = UnetExtractor(in_channel=3, encoder_dim=args.encoder_dims)
        raft = RAFTStereoHuman(args)
        state = load_sanitized_gps_state(self.config)
        encoder_state = {
            key.removeprefix("img_encoder."): value
            for key, value in state.items()
            if key.startswith("img_encoder.")
        }
        raft_state = {
            key.removeprefix("raft_stereo."): value
            for key, value in state.items()
            if key.startswith("raft_stereo.")
        }
        encoder.load_state_dict(encoder_state, strict=True)
        raft.load_state_dict(raft_state, strict=True)
        del state, encoder_state, raft_state
        encoder.requires_grad_(False).eval().to(device)
        raft.requires_grad_(False).eval().to(device)
        if any(
            parameter.requires_grad for parameter in (*encoder.parameters(), *raft.parameters())
        ):
            raise RuntimeError("GPS parameter freezing failed")
        receipt.update(
            {
                "schema": "rtgs.gps_gaussian_external_binding.v1",
                "checkpoint": os.fspath(self.config.checkpoint.resolve()),
                "checkpoint_bytes": self.config.checkpoint_bytes,
                "checkpoint_sha256": self.config.checkpoint_sha256,
                "device": str(device),
                "iterations": self.config.iterations,
                "mixed_precision": True,
                "autocast_dtype": "torch.float16",
                "parameter_gradients": False,
                "training_mode": False,
            }
        )
        self._img_encoder = encoder
        self._raft_stereo = raft
        self._receipt = receipt

    def predict_pair(self, request: RectifiedStereoRequest) -> StereoDepthPrediction:
        """Run the exact ``concat -> encoder[2] -> symmetric RAFT split`` dataflow."""

        request.validate()
        self._load()
        device = torch.device(self.config.device)
        if request.left_image.device != device or request.left_image.dtype != torch.float32:
            raise ValueError("GPS request must be float32 on the configured CUDA device")
        image = torch.stack((request.left_image, request.right_image), dim=0)
        if image.shape != (2, 3, 1024, 1024):
            raise ValueError("frozen GPS inference requires exact (2,3,1024,1024) input")
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.config.mixed_precision,
            ),
        ):
            pyramid = self._img_encoder(image)
            if len(pyramid) != 3 or pyramid[2].shape != (2, 96, 128, 128):
                raise RuntimeError("GPS image encoder produced an unexpected pyramid")
            symmetric = self._raft_stereo(
                pyramid[2],
                iters=self.config.iterations,
                test_mode=True,
            )
        if not isinstance(symmetric, tuple) or len(symmetric) != 2:
            raise RuntimeError("GPS symmetric RAFT did not return left/right splits")
        left_flow, right_flow = symmetric
        if left_flow.shape != (1, 1, 1024, 1024) or right_flow.shape != left_flow.shape:
            raise RuntimeError("GPS symmetric RAFT returned unexpected flow shapes")
        return stereo_depth_from_bidirectional_flow(
            request,
            left_flow[0, 0].float(),
            right_flow[0, 0].float(),
        )

    def release(self) -> None:
        """Release dense adapter CUDA state without resetting complete-boundary peaks."""

        self._img_encoder = None
        self._raft_stereo = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "GPSGaussianConfig",
    "GPSGaussianStereoBackend",
    "load_sanitized_gps_state",
]
