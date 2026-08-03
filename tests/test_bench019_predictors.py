"""CPU-only contracts for the BENCH-019 Stage-1 predictor collector."""

from __future__ import annotations

import copy
import hashlib
import json
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import rtgs.bench019_predictors as P
from rtgs.bench019 import ExportError, canonical_json, describe_artifact
from rtgs.bench019_adapters import (
    TUM_HEIGHT,
    TUM_WIDTH,
    VIEW_COUNT,
    build_calibrated_adapter,
    build_tum_adapter,
    materialize_tum_adapter,
    write_source_adapter,
)
from rtgs.bench019_portfolio import (
    PORTFOLIO_SCHEMA,
    REQUIRED_FIELD_FAMILIES,
    source_digest,
)
from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField
from rtgs.data.calibrated import _undistort
from rtgs.data.compact_views import (
    COMPACT_VIEW_BYTE_CAP,
    save_compact_view,
    write_compact_dataset_manifest,
)


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


def _capture(
    capture_id: str,
    *,
    role: str,
    source_kind: str,
    sources: list[dict],
    view_ids: list[str] | None = None,
    mask_policy_state: str = "test",
) -> dict:
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
        "field_families": _families(confirmation=role == "confirmation"),
        "blockers": ["synthetic predictor fixture"],
    }


def _single_source(tmp_path: Path, name: str) -> list[dict]:
    path = tmp_path / f"{name}.bin"
    path.write_bytes(name.encode())
    return [{"id": "source", "artifact": describe_artifact(path)}]


def _png(array: np.ndarray) -> bytes:
    stream = BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def _add_tar_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    member.mtime = 0
    archive.addfile(member, BytesIO(payload))


def _tum_archive(path: Path) -> Path:
    rgb_lines = ["# timestamp path"]
    depth_lines = ["# timestamp path"]
    pose_lines = ["# timestamp tx ty tz qx qy qz qw"]
    payloads: list[tuple[str, bytes]] = []
    for index in range(VIEW_COUNT):
        rgb_timestamp = f"{1 + index / 10:.3f}"
        depth_timestamp = f"{1.01 + index / 10:.3f}"
        rgb_name = f"rgb/{index:04d}.png"
        depth_name = f"depth/{index:04d}.png"
        rgb_lines.append(f"{rgb_timestamp} {rgb_name}")
        depth_lines.append(f"{depth_timestamp} {depth_name}")
        pose_lines.append(f"{depth_timestamp} {0.1 * index:.6f} 0 0 0 0 0 1")
        rgb = np.full((TUM_HEIGHT, TUM_WIDTH, 3), 30 + index, dtype=np.uint8)
        depth = np.full((TUM_HEIGHT, TUM_WIDTH), 5_000, dtype=np.uint16)
        depth[:20, :20] = 0
        payloads.extend([(rgb_name, _png(rgb)), (depth_name, _png(depth))])
    with tarfile.open(path, mode="w:gz") as archive:
        prefix = "rgbd_dataset_test"
        for name, lines in (
            ("rgb.txt", rgb_lines),
            ("depth.txt", depth_lines),
            ("groundtruth.txt", pose_lines),
        ):
            _add_tar_bytes(archive, f"{prefix}/{name}", ("\n".join(lines) + "\n").encode())
        for name, payload in payloads:
            _add_tar_bytes(archive, f"{prefix}/{name}", payload)
    return path


def _stage_portfolio(tmp_path: Path) -> tuple[Path, list[str], Path]:
    root = tmp_path / "stage"
    frame = root / "frame_00008"
    rgb_dir = frame / "rgb"
    mask_dir = frame / "mask"
    rgb_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    views = [f"C{index:04d}" for index in range(VIEW_COUNT)]
    sources: list[dict] = []
    cameras: list[dict] = []
    for index, view_id in enumerate(views):
        rgb = np.zeros((6, 8, 3), dtype=np.uint8)
        rgb[..., 0] = 20 + index
        rgb[..., 1] = np.arange(8, dtype=np.uint8)[None] * 20
        rgb[..., 2] = np.arange(6, dtype=np.uint8)[:, None] * 25
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[1:5, 2:7] = 255
        rgb_path = rgb_dir / f"{view_id}.png"
        mask_path = mask_dir / f"mask_{view_id}.png"
        Image.fromarray(rgb).save(rgb_path)
        Image.fromarray(mask).save(mask_path)
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
                        index * 0.01,
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

    dummy = _single_source(tmp_path, "dummy")
    development = ["janelle_stage_fabric", "dev_b", "dev_c"]
    confirmation = ["confirm_a", "confirm_b", "confirm_c"]
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
                sources=sources,
                view_ids=views,
                mask_policy_state="source_binary_masks_bound",
            ),
            _capture("dev_b", role="development", source_kind="test", sources=dummy),
            _capture("dev_c", role="development", source_kind="test", sources=dummy),
            _capture("confirm_a", role="confirmation", source_kind="test", sources=dummy),
            _capture("confirm_b", role="confirmation", source_kind="test", sources=dummy),
            _capture("confirm_c", role="confirmation", source_kind="test", sources=dummy),
        ],
        "gates": {
            "source_groups_bound": True,
            "field_families_complete": False,
            "confirmation_outcomes_opened": False,
            "formal_protocol_frozen": False,
        },
    }
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_bytes(canonical_json(portfolio) + b"\n")
    return portfolio_path, views, frame


def _camera(record: dict, *, drift: float = 0.0) -> Camera:
    translation = list(record["t"])
    translation[0] += drift
    return Camera(
        fx=record["fx"],
        fy=record["fy"],
        cx=record["cx"],
        cy=record["cy"],
        width=record["width"],
        height=record["height"],
        R=torch.tensor(record["R"], dtype=torch.float32).reshape(3, 3),
        t=torch.tensor(translation, dtype=torch.float32),
    )


def _observation(view_id: str, *, blend_mode: str) -> GaussianObservationField:
    return GaussianObservationField(
        width=8,
        height=6,
        means=torch.tensor([[3.5, 2.5], [5.5, 3.5]], dtype=torch.float32),
        log_scales=torch.log(torch.tensor([[1.8, 1.4], [1.2, 1.5]], dtype=torch.float32)),
        rotations=torch.tensor([0.0, 0.2], dtype=torch.float32),
        colors=torch.tensor([[0.2, 0.5, 0.3], [0.4, 0.3, 0.6]], dtype=torch.float32),
        amplitudes=torch.tensor([0.8, 0.5], dtype=torch.float32),
        blend_mode=blend_mode,
        fit_window=(0, 0, 8, 6),
        view_id=view_id,
        n_init=2,
        provider="native" if blend_mode == "additive" else "structsplat",
    )


def _fixture(
    tmp_path: Path,
    *,
    blend_mode: str = "additive",
    alpha_drift: bool = False,
    camera_drift: float = 0.0,
    source_drift: bool = False,
) -> tuple[Path, Path]:
    portfolio_path, views, frame = _stage_portfolio(tmp_path)
    adapter = build_calibrated_adapter(portfolio_path)
    adapter_path = tmp_path / "adapter.json"
    write_source_adapter(adapter, adapter_path)
    directory = tmp_path / "gaussians2d"
    directory.mkdir()
    paths = []
    calibration_sha256 = adapter["source_artifacts"][0]["artifact"]["sha256"]
    sources = {item["id"]: item["artifact"] for item in adapter["source_artifacts"]}
    for adapter_view, view_id in zip(adapter["views"], views, strict=True):
        mask = np.array(Image.open(frame / "mask" / f"mask_{view_id}.png"), dtype=np.uint8) > 127
        if alpha_drift and view_id == views[0]:
            mask[0, 0] = True
        path = directory / f"{view_id}.rtgsv"
        save_compact_view(
            path,
            _observation(view_id, blend_mode=blend_mode),
            _camera(adapter_view["camera"], drift=camera_drift if view_id == views[0] else 0.0),
            calibration_sha256=calibration_sha256,
            source_rgb_name=Path(sources[f"rgb_{view_id}"]["path"]).name,
            source_rgb_sha256=(
                "0" * 64
                if source_drift and view_id == views[0]
                else sources[f"rgb_{view_id}"]["sha256"]
            ),
            alpha_crop=mask,
            source_mask_name=Path(sources[f"mask_{view_id}"]["path"]).name,
            source_mask_sha256=sources[f"mask_{view_id}"]["sha256"],
        )
        paths.append(path)
    write_compact_dataset_manifest(
        directory,
        name="tiny_bench019_field",
        calibration_sha256=calibration_sha256,
        view_paths=paths,
        bounds_hint=None,
    )
    return adapter_path, directory


def _config() -> P.PredictorConfig:
    return P.PredictorConfig(
        seed=17,
        sample_cap_per_stratum=12,
        boundary_radius_px=1,
        component_chunk=2,
        tile_size=4,
        max_index_entries=10_000,
        max_index_candidates=100,
        max_query_pairs=1_000,
        view_byte_cap=COMPACT_VIEW_BYTE_CAP,
    )


def _refresh_digest(value: dict) -> None:
    value.pop("semantic_digest", None)
    value["semantic_digest"] = hashlib.sha256(canonical_json(value)).hexdigest()


def _tum_fixture(tmp_path: Path) -> tuple[Path, Path]:
    portfolio_path, _views, _frame = _stage_portfolio(tmp_path)
    portfolio = json.loads(portfolio_path.read_text())
    archive = _tum_archive(tmp_path / "tum.tgz")
    sources = [{"id": "official_archive", "artifact": describe_artifact(archive)}]
    replacement = _capture(
        "dev_b",
        role="development",
        source_kind="tum_rgbd_archive",
        sources=sources,
    )
    portfolio["captures"][1] = replacement
    portfolio_path.write_bytes(canonical_json(portfolio) + b"\n")
    adapter = build_tum_adapter(portfolio_path, capture_id="dev_b")
    adapter_path = tmp_path / "tum.adapter.json"
    write_source_adapter(adapter, adapter_path)
    materialized = tmp_path / "materialized"
    materialize_tum_adapter(adapter_path, materialized)

    directory = tmp_path / "tum-gaussians2d"
    directory.mkdir()
    calibration_sha256 = describe_artifact(materialized / "calibration_dome.json")["sha256"]
    paths = []
    for view in adapter["views"]:
        view_id = view["id"]
        rgb_path = materialized / "rgb" / f"{view_id}.png"
        mask_path = materialized / "mask" / f"mask_{view_id}.png"
        mask = np.array(Image.open(mask_path), dtype=np.uint8) > 127
        field = GaussianObservationField(
            width=TUM_WIDTH,
            height=TUM_HEIGHT,
            means=torch.tensor([[320.5, 240.5]], dtype=torch.float32),
            log_scales=torch.log(torch.tensor([[80.0, 70.0]], dtype=torch.float32)),
            rotations=torch.zeros(1),
            colors=torch.full((1, 3), 0.3),
            amplitudes=torch.ones(1),
            blend_mode="normalized",
            fit_window=(0, 0, TUM_WIDTH, TUM_HEIGHT),
            view_id=view_id,
            n_init=1,
            provider="structsplat",
        )
        path = directory / f"{view_id}.rtgsv"
        save_compact_view(
            path,
            field,
            _camera(view["camera"]),
            calibration_sha256=calibration_sha256,
            source_rgb_name=rgb_path.name,
            source_rgb_sha256=describe_artifact(rgb_path)["sha256"],
            alpha_crop=mask,
            source_mask_name=mask_path.name,
            source_mask_sha256=describe_artifact(mask_path)["sha256"],
        )
        paths.append(path)
    write_compact_dataset_manifest(
        directory,
        name="tiny_tum_bench019_field",
        calibration_sha256=calibration_sha256,
        view_paths=paths,
        bounds_hint=None,
    )
    return adapter_path, directory


def test_additive_predictors_are_deterministic_aggregated_and_replayable(tmp_path: Path) -> None:
    adapter_path, directory = _fixture(tmp_path)
    first = P.build_stage1_predictors(
        adapter_path,
        directory,
        family_id="gaussianimage_additive",
        config=_config(),
    )
    second = P.build_stage1_predictors(
        adapter_path,
        directory,
        family_id="gaussianimage_additive",
        config=_config(),
    )

    assert first == second
    assert P.validate_stage1_predictors(first, verify_files=True) == {
        "capture_id": "janelle_stage_fabric",
        "field_family": "gaussianimage_additive",
        "views": 26,
        "predictors": len(P.SUPPORTED_PREDICTORS),
    }
    assert first["field_family"]["equation"] == "additive_sum"
    assert first["aggregates"]["train"]["n_views"] == 23
    assert first["aggregates"]["heldout"]["n_views"] == 3
    assert first["predictors"]["all_total_rows"] == 52
    assert (
        first["predictors"]["all_complete_field_bytes"]
        == sum(item["artifact"]["bytes"] for item in first["compact_field"]["files"])
        + first["compact_field"]["manifest"]["bytes"]
    )
    assert first["support_definition"] == P.SUPPORT_DEFINITION
    assert "alpha_agreement" not in first["predictors"]

    output = tmp_path / "predictors.json"
    assert P.write_stage1_predictors(first, output)["views"] == 26
    with pytest.raises(FileExistsError):
        P.write_stage1_predictors(first, output)


def test_normalized_family_preserves_equation_and_rejects_relabelling(tmp_path: Path) -> None:
    adapter_path, directory = _fixture(tmp_path, blend_mode="normalized")
    result = P.build_stage1_predictors(
        adapter_path,
        directory,
        family_id="structsplat_normalized_no_boundary",
        config=_config(),
    )

    assert result["field_family"]["equation"] == "normalized_weighted_sum"
    assert result["field_family"]["blend_mode"] == "normalized"
    with pytest.raises(ExportError, match="semantics differ"):
        P.build_stage1_predictors(
            adapter_path,
            directory,
            family_id="gaussianimage_additive",
            config=_config(),
        )

    relabelled = copy.deepcopy(result)
    relabelled["field_family"] = {
        "id": "gaussianimage_additive",
        "provider": "native",
        "equation": "additive_sum",
        "blend_mode": "additive",
    }
    _refresh_digest(relabelled)
    with pytest.raises(ExportError, match="semantics differ"):
        P.validate_stage1_predictors(relabelled, verify_files=True)


@pytest.mark.parametrize("name", sorted(P.UNSUPPORTED_PREDICTORS))
def test_unavailable_predictors_fail_before_any_input_access(tmp_path: Path, name: str) -> None:
    with pytest.raises(ExportError, match=name):
        P.build_stage1_predictors(
            tmp_path / "missing-adapter.json",
            tmp_path / "missing-field",
            family_id="gaussianimage_additive",
            config=_config(),
            requested_predictors=[name],
        )


def test_collector_rejects_camera_and_alpha_drift(tmp_path: Path) -> None:
    camera_adapter, camera_field = _fixture(tmp_path / "camera", camera_drift=0.1)
    with pytest.raises(ExportError, match="source/camera binding differs"):
        P.build_stage1_predictors(
            camera_adapter,
            camera_field,
            family_id="gaussianimage_additive",
            config=_config(),
        )

    alpha_adapter, alpha_field = _fixture(tmp_path / "alpha", alpha_drift=True)
    with pytest.raises(ExportError, match="alpha differs"):
        P.build_stage1_predictors(
            alpha_adapter,
            alpha_field,
            family_id="gaussianimage_additive",
            config=_config(),
        )

    source_adapter, source_field = _fixture(tmp_path / "source", source_drift=True)
    with pytest.raises(ExportError, match="source hash/name binding differs"):
        P.build_stage1_predictors(
            source_adapter,
            source_field,
            family_id="gaussianimage_additive",
            config=_config(),
        )


def test_predictor_validator_rejects_digest_consistent_metric_drift(tmp_path: Path) -> None:
    adapter_path, directory = _fixture(tmp_path)
    result = P.build_stage1_predictors(
        adapter_path,
        directory,
        family_id="gaussianimage_additive",
        config=_config(),
    )
    result["views"][0]["metrics"]["sampled_query_rgb_mae"] += 0.01
    _refresh_digest(result)
    with pytest.raises(ExportError, match="sufficient statistics"):
        P.validate_stage1_predictors(result)


def test_tum_materialization_and_derived_masks_feed_normalized_collector(tmp_path: Path) -> None:
    adapter_path, directory = _tum_fixture(tmp_path)
    config = P.PredictorConfig(
        seed=3,
        sample_cap_per_stratum=8,
        boundary_radius_px=1,
        component_chunk=1,
        tile_size=16,
        max_index_entries=10_000,
        max_index_candidates=10,
        max_query_pairs=256,
        view_byte_cap=COMPACT_VIEW_BYTE_CAP,
    )
    result = P.build_stage1_predictors(
        adapter_path,
        directory,
        family_id="structsplat_normalized_no_boundary",
        config=config,
    )

    assert result["capture_id"] == "dev_b"
    assert result["aggregates"]["all"]["n_views"] == 26
    assert result["field_family"]["equation"] == "normalized_weighted_sum"
    assert P.validate_stage1_predictors(result, verify_files=True)["views"] == 26


def test_sparse_source_sampling_matches_calibrated_dense_undistortion() -> None:
    height, width = 11, 13
    rgb = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[2:9, 3:11] = 255
    camera = {
        "fx": 9.0,
        "fy": 8.5,
        "cx": 6.2,
        "cy": 5.4,
        "width": width,
        "height": height,
    }
    distortion = [0.08, -0.03, 0.004, -0.006, 0.01]
    dense_rgb = _undistort(
        torch.from_numpy(rgb.copy()).to(torch.float32) / 255.0,
        camera["fx"],
        camera["fy"],
        camera["cx"],
        camera["cy"],
        distortion,
    )
    dense_mask = (
        _undistort(
            torch.from_numpy(mask.copy()).to(torch.float32) / 255.0,
            camera["fx"],
            camera["fy"],
            camera["cx"],
            camera["cy"],
            distortion,
            mask=True,
        )
        > 0.5
    )
    expected_alpha = P._derive_alpha_crop(
        mask,
        camera=camera,
        distortion=distortion,
        fit_window=(0, 0, width, height),
        row_chunk=3,
    )
    assert torch.equal(expected_alpha, dense_mask)

    pixels = torch.tensor([[0, 0], [2, 3], [6, 5], [11, 9], [12, 10]], dtype=torch.long)
    xy = pixels.to(torch.float32) + 0.5
    sampled = P._sample_source_rgb(rgb, xy, camera=camera, distortion=distortion)
    expected = dense_rgb[pixels[:, 1], pixels[:, 0]]
    assert torch.allclose(sampled, expected, atol=2e-7, rtol=0.0)
