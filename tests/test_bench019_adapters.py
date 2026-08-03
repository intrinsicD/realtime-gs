"""CPU-only contracts for deterministic BENCH-019 source adapters."""

from __future__ import annotations

import copy
import json
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import rtgs.bench019_adapters as A
from rtgs.bench019 import ExportError, canonical_json, describe_artifact
from rtgs.bench019_portfolio import (
    PORTFOLIO_SCHEMA,
    REQUIRED_FIELD_FAMILIES,
    source_digest,
)
from rtgs.data.calibrated import load_calibrated_scene


def _families(*, confirmation: bool) -> list[dict]:
    result = []
    for family_id in REQUIRED_FIELD_FAMILIES:
        additive = family_id == "gaussianimage_additive"
        result.append(
            {
                "id": family_id,
                "provider": "gaussianimage" if additive else "structsplat",
                "equation": "additive_sum" if additive else "normalized_weighted_sum",
                "blend_mode": "additive" if additive else "normalized",
                "state": "not_produced" if confirmation else "incomplete_live_unbound",
                "observed_views": 0,
                "required_views": None,
                "evidence": None,
            }
        )
    return result


def _png(array: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def _add_tar_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    member.mtime = 0
    archive.addfile(member, BytesIO(payload))


def _tum_archive(path: Path, *, count: int = A.VIEW_COUNT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb_lines = ["# timestamp path"]
    depth_lines = ["# timestamp path"]
    pose_lines = ["# timestamp tx ty tz qx qy qz qw"]
    rgb_payloads: list[tuple[str, bytes]] = []
    depth_payloads: list[tuple[str, bytes]] = []
    for index in range(count):
        rgb_timestamp = f"{1 + index / 10:.3f}"
        depth_timestamp = f"{1.01 + index / 10:.3f}"
        rgb_name = f"rgb/{index:04d}.png"
        depth_name = f"depth/{index:04d}.png"
        rgb_lines.append(f"{rgb_timestamp} {rgb_name}")
        depth_lines.append(f"{depth_timestamp} {depth_name}")
        pose_lines.append(f"{depth_timestamp} {0.1 * index:.6f} 0 0 0 0 0 1")
        rgb = np.full((A.TUM_HEIGHT, A.TUM_WIDTH, 3), index % 255, dtype=np.uint8)
        depth = np.full((A.TUM_HEIGHT, A.TUM_WIDTH), 10_000, dtype=np.uint16)
        depth[0, 0] = 0
        depth[0, 1] = 1_500
        depth[0, 2] = 25_000
        depth[0, 3] = 25_001
        rgb_payloads.append((rgb_name, _png(rgb)))
        depth_payloads.append((depth_name, _png(depth)))
    with tarfile.open(path, mode="w:gz") as archive:
        prefix = "rgbd_dataset_test"
        _add_tar_bytes(archive, f"{prefix}/rgb.txt", ("\n".join(rgb_lines) + "\n").encode())
        _add_tar_bytes(archive, f"{prefix}/depth.txt", ("\n".join(depth_lines) + "\n").encode())
        _add_tar_bytes(
            archive,
            f"{prefix}/groundtruth.txt",
            ("\n".join(pose_lines) + "\n").encode(),
        )
        for name, payload in [*rgb_payloads, *depth_payloads]:
            _add_tar_bytes(archive, f"{prefix}/{name}", payload)
    return path


def _stage_sources(tmp_path: Path) -> tuple[list[dict], list[str]]:
    root = tmp_path / "stage"
    rgb_dir = root / "frame_00008/rgb"
    mask_dir = root / "frame_00008/mask"
    rgb_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    views = [f"C{index:04d}" for index in range(A.VIEW_COUNT)]
    cameras = []
    sources = []
    for index, view_id in enumerate(views):
        rgb_path = rgb_dir / f"{view_id}.png"
        mask_path = mask_dir / f"mask_{view_id}.png"
        Image.fromarray(np.full((6, 8, 3), index, dtype=np.uint8)).save(rgb_path)
        Image.fromarray(np.full((6, 8), 255, dtype=np.uint8)).save(mask_path)
        sources.extend(
            [
                {"id": f"rgb_{view_id}", "artifact": describe_artifact(rgb_path)},
                {"id": f"mask_{view_id}", "artifact": describe_artifact(mask_path)},
            ]
        )
        cameras.append(
            {
                "camera_id": view_id,
                "extrinsics": {
                    "view_matrix": [
                        1.0,
                        0.0,
                        0.0,
                        0.1 * index,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ]
                },
                "intrinsics": {
                    "camera_matrix": [6.0, 0.0, 3.0, 0.0, 6.0, 2.0, 0.0, 0.0, 1.0],
                    "distortion_coefficients": [0.0] * 5,
                    "resolution": [8, 6],
                },
            }
        )
    calibration = root / "calibration_dome.json"
    calibration.write_text(json.dumps({"cameras": cameras}), encoding="utf-8")
    sources.insert(0, {"id": "calibration", "artifact": describe_artifact(calibration)})
    return sources, views


def _capture(
    capture_id: str,
    *,
    role: str,
    source_kind: str,
    sources: list[dict],
    view_ids: list[str] | None = None,
    mask_policy_state: str = "test",
) -> dict:
    confirmation = role == "confirmation"
    return {
        "id": capture_id,
        "role": role,
        "source_kind": source_kind,
        "origin": "local-test",
        "frame_id": "frame",
        "frame_plan_state": "selected",
        "view_ids": [] if view_ids is None else view_ids,
        "mask_policy_state": mask_policy_state,
        "source_artifacts": sources,
        "source_digest": source_digest(sources),
        "field_families": _families(confirmation=confirmation),
        "blockers": ["synthetic adapter fixture"],
    }


def _single_source(tmp_path: Path, name: str) -> list[dict]:
    path = tmp_path / f"{name}.bin"
    path.write_bytes(name.encode())
    return [{"id": "source", "artifact": describe_artifact(path)}]


def _portfolio(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    stage_sources, views = _stage_sources(tmp_path)
    tum_path = _tum_archive(tmp_path / "tum_dev.tgz")
    tum_sources = [{"id": "official_archive", "artifact": describe_artifact(tum_path)}]
    karate_sources = _single_source(tmp_path, "karate_rgb")
    development = ["janelle_stage_fabric", "tum_dev", "dev_c"]
    confirmation = ["janelle_karate", "tum_confirmation", "confirm_c"]
    portfolio = {
        "schema": PORTFOLIO_SCHEMA,
        "state": "source_bound_preproduction",
        "inventory_date": "2026-08-03",
        "outcome_access": "none",
        "development_capture_ids": development,
        "confirmation_capture_ids": confirmation,
        "captures": [
            _capture(
                "janelle_stage_fabric",
                role="development",
                source_kind="calibrated_multiview",
                sources=stage_sources,
                view_ids=views,
                mask_policy_state="source_binary_masks_bound",
            ),
            _capture(
                "tum_dev",
                role="development",
                source_kind="tum_rgbd_archive",
                sources=tum_sources,
            ),
            _capture(
                "dev_c",
                role="development",
                source_kind="test",
                sources=_single_source(tmp_path, "dev_c"),
            ),
            _capture(
                "janelle_karate",
                role="confirmation",
                source_kind="calibrated_multiview",
                sources=karate_sources,
                mask_policy_state="missing_unfrozen",
            ),
            _capture(
                "tum_confirmation",
                role="confirmation",
                source_kind="tum_rgbd_archive",
                sources=tum_sources,
            ),
            _capture(
                "confirm_c",
                role="confirmation",
                source_kind="test",
                sources=_single_source(tmp_path, "confirm_c"),
            ),
        ],
        "gates": {
            "source_groups_bound": True,
            "field_families_complete": False,
            "confirmation_outcomes_opened": False,
            "formal_protocol_frozen": False,
        },
    }
    path = tmp_path / "portfolio.json"
    path.write_bytes(canonical_json(portfolio) + b"\n")
    return path


def _refresh_adapter_digest(adapter: dict) -> None:
    adapter.pop("semantic_digest", None)
    adapter["semantic_digest"] = A._canonical_digest(adapter)


def test_calibrated_adapter_binds_exact_views_cameras_masks_and_split(tmp_path: Path) -> None:
    adapter = A.build_calibrated_adapter(_portfolio(tmp_path))

    assert A.validate_source_adapter(adapter, verify_sources=True) == {
        "views": 26,
        "train_views": 23,
        "heldout_views": 3,
    }
    assert [view["id"] for view in adapter["views"] if view["split"] == "heldout"] == [
        "C0007",
        "C0015",
        "C0023",
    ]
    assert adapter["views"][0]["camera"]["cx"] == 3.5
    assert adapter["views"][0]["camera"]["cy"] == 2.5
    assert adapter["mask_policy"]["kind"] == "source_binary_masks"
    output = tmp_path / "published.adapter.json"
    assert A.write_source_adapter(adapter, output)["views"] == 26
    with pytest.raises(FileExistsError):
        A.write_source_adapter(adapter, output)


def test_calibrated_adapter_replays_sources_and_rejects_camera_drift(tmp_path: Path) -> None:
    adapter = A.build_calibrated_adapter(_portfolio(tmp_path))
    adapter["views"][0]["camera"]["fx"] += 1.0
    _refresh_adapter_digest(adapter)
    with pytest.raises(ExportError, match="deterministic source replay"):
        A.validate_source_adapter(adapter, verify_sources=True)


def test_karate_and_tum_confirmation_fail_before_payload_access(tmp_path: Path) -> None:
    portfolio_path = _portfolio(tmp_path)
    with pytest.raises(ExportError, match="no source-backed mask policy"):
        A.build_calibrated_adapter(portfolio_path, capture_id="janelle_karate")
    with pytest.raises(ExportError, match="confirmation-sealed"):
        A.build_tum_adapter(portfolio_path, capture_id="tum_confirmation")


def test_tum_association_is_strict_and_pose_interpolation_is_bounded() -> None:
    first = [A.TimedPath(0, "0", "rgb.png")]
    accepted = [A.TimedPath(A.TUM_ASSOCIATION_MAX_NS - 1, "0.019999999", "depth.png")]
    rejected = [A.TimedPath(A.TUM_ASSOCIATION_MAX_NS, "0.020", "depth.png")]
    assert len(A._associate_paths(first, accepted)) == 1
    assert A._associate_paths(first, rejected) == []

    poses = [
        A.TimedPose(0, "0", np.zeros(3), np.asarray([0.0, 0.0, 0.0, 1.0])),
        A.TimedPose(
            A.TUM_POSE_INTERPOLATION_MAX_NS,
            "0.020",
            np.asarray([0.02, 0.0, 0.0]),
            np.asarray([0.0, 0.0, 0.0, 1.0]),
        ),
    ]
    interpolated = A._interpolate_pose(poses, A.TUM_POSE_INTERPOLATION_MAX_NS // 2)
    assert interpolated is not None
    assert np.allclose(interpolated.center, [0.01, 0.0, 0.0])
    too_wide = [poses[0], copy.deepcopy(poses[1])]
    object.__setattr__(too_wide[1], "timestamp_ns", A.TUM_POSE_INTERPOLATION_MAX_NS + 1)
    assert A._interpolate_pose(too_wide, A.TUM_POSE_INTERPOLATION_MAX_NS // 2) is None


def test_half_up_selection_preserves_endpoints_and_rejects_small_population() -> None:
    indices = A._half_up_uniform_indices(77, 26)
    assert indices == [
        0,
        3,
        6,
        9,
        12,
        15,
        18,
        21,
        24,
        27,
        30,
        33,
        36,
        40,
        43,
        46,
        49,
        52,
        55,
        58,
        61,
        64,
        67,
        70,
        73,
        76,
    ]
    with pytest.raises(ExportError, match="cannot select"):
        A._half_up_uniform_indices(25, 26)


def test_tum_adapter_is_deterministic_and_preserves_camera_convention(tmp_path: Path) -> None:
    portfolio_path = _portfolio(tmp_path)
    first = A.build_tum_adapter(portfolio_path, capture_id="tum_dev")
    second = A.build_tum_adapter(portfolio_path, capture_id="tum_dev")

    assert first == second
    assert A.validate_source_adapter(first, verify_sources=True)["heldout_views"] == 3
    assert first["selection_policy"]["pose_keyframes"] == 26
    assert first["selection_policy"]["uniform_source_indices"] == list(range(26))
    assert first["views"][0]["camera"]["fx"] == 525.0
    assert first["views"][0]["camera"]["cx"] == 320.0
    assert first["views"][0]["mask_source"]["min_depth_m"] == 0.3


def test_adapter_validation_rejects_digest_consistent_policy_drift(tmp_path: Path) -> None:
    portfolio_path = _portfolio(tmp_path)
    stage = A.build_calibrated_adapter(portfolio_path)
    stage["views"][0]["preprocessing"]["rgb"] = "identity"
    _refresh_adapter_digest(stage)
    with pytest.raises(ExportError, match="preprocessing differs"):
        A.validate_source_adapter(stage)

    baseline = A.build_tum_adapter(portfolio_path, capture_id="tum_dev")
    mutations = [
        (
            lambda value: value["selection_policy"].__setitem__("pose_keyframes", "26"),
            "pose_keyframes",
        ),
        (
            lambda value: value["views"][0]["source_metadata"].__setitem__(
                "rgb_timestamp_token", "9.0"
            ),
            "rgb_timestamp_token differs",
        ),
        (
            lambda value: value["views"][0]["camera"]["t"].__setitem__(0, 1.0),
            "camera differs from its TUM pose",
        ),
        (
            lambda value: value["views"][0]["preprocessing"].__setitem__(
                "mask", "field_weight_as_alpha"
            ),
            "TUM preprocessing differs",
        ),
    ]
    for mutate, message in mutations:
        candidate = copy.deepcopy(baseline)
        mutate(candidate)
        _refresh_adapter_digest(candidate)
        with pytest.raises(ExportError, match=message):
            A.validate_source_adapter(candidate)


def test_tum_materialization_is_exclusive_and_receipt_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = A.build_tum_adapter(_portfolio(tmp_path), capture_id="tum_dev")
    adapter_path = tmp_path / "adapter.json"
    adapter_path.write_bytes(canonical_json(adapter) + b"\n")
    output = tmp_path / "materialized"

    failed_output = tmp_path / "failed-materialization"
    with monkeypatch.context() as local_patch:
        local_patch.setattr(
            A.tempfile,
            "mkdtemp",
            lambda **_kwargs: (_ for _ in ()).throw(OSError("synthetic temp failure")),
        )
        with pytest.raises(OSError, match="synthetic temp failure"):
            A.materialize_tum_adapter(adapter_path, failed_output)
    assert not failed_output.exists()

    assert A.materialize_tum_adapter(adapter_path, output) == {
        "capture_id": "tum_dev",
        "role": "development",
        "outputs": 54,
    }
    scene = load_calibrated_scene(
        output,
        view_ids=["C0000"],
        test_every=0,
        load_masks=True,
    )
    assert scene.cameras[0].fx == 525.0
    assert scene.cameras[0].cx == 320.0
    assert scene.masks is not None
    assert int(scene.masks[0].sum().item()) == A.TUM_WIDTH * A.TUM_HEIGHT - 2
    with pytest.raises(FileExistsError, match="existing path"):
        A.materialize_tum_adapter(adapter_path, output)

    rgb = output / "rgb/C0000.png"
    rgb.write_bytes(b"tampered")
    with pytest.raises(ExportError, match="differs from its bound file"):
        A.validate_materialization(output / "materialization_receipt.json", verify_files=True)


def test_safe_tum_archive_rejects_links(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.tgz"
    with tarfile.open(path, mode="w:gz") as archive:
        for name in ("rgb.txt", "depth.txt", "groundtruth.txt"):
            _add_tar_bytes(archive, f"sequence/{name}", b"0 payload\n")
        link = tarfile.TarInfo("sequence/rgb/link.png")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        archive.addfile(link)
    with pytest.raises(ExportError, match="links are forbidden"), A.SafeTumArchive(path):
        pass


def test_safe_tum_archive_rejects_special_members(tmp_path: Path) -> None:
    path = tmp_path / "unsafe-special.tgz"
    with tarfile.open(path, mode="w:gz") as archive:
        for name in ("rgb.txt", "depth.txt", "groundtruth.txt"):
            _add_tar_bytes(archive, f"sequence/{name}", b"0 payload\n")
        fifo = tarfile.TarInfo("sequence/fifo")
        fifo.type = tarfile.FIFOTYPE
        archive.addfile(fifo)
    with pytest.raises(ExportError, match="special members are forbidden"), A.SafeTumArchive(path):
        pass
