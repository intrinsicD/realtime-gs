"""CPU-only checks for the post-result compact-factorial visualizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks import (
    visualize_compact_occupancy_refinement_factorial as visualizer,
)
from PIL import Image


def _write_png(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_strict_json_rejects_duplicate_and_nonfinite_values(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"status":"PASS","status":"FAIL"}', encoding="utf-8")
    with pytest.raises(visualizer.ProtocolInvalid, match="duplicate JSON key"):
        visualizer.strict_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(visualizer.ProtocolInvalid, match="non-finite JSON"):
        visualizer.strict_json(nonfinite)


def test_result_linked_seed_plys_are_exact_and_fail_closed(tmp_path: Path) -> None:
    workers = {}
    for arm in visualizer.ARMS:
        path = tmp_path / "runs" / f"arm_{arm}" / "gaussians_final.ply"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"ply-{arm}".encode())
        workers[arm] = {
            "status": "PASS",
            "arm": arm,
            "training_seed": visualizer.REQUIRED_SEED,
            "n_opt_3d": 835,
            "final_ply": {
                "path": path.relative_to(tmp_path).as_posix(),
                "sha256": visualizer.sha256_file(path),
                "bytes": path.stat().st_size,
                "n_gaussians": 835,
            },
        }
    result = {"workers": {str(visualizer.REQUIRED_SEED): workers}}
    records = visualizer.validate_result_ply_bindings(result, root=tmp_path)
    assert tuple(records) == visualizer.ARMS

    target = tmp_path / records["C"]["path"]
    target.write_bytes(b"changed")
    with pytest.raises(visualizer.ProtocolInvalid, match="changed for arm C"):
        visualizer.validate_result_ply_bindings(result, root=tmp_path)

    target.write_bytes(b"ply-C")
    result["workers"][str(visualizer.REQUIRED_SEED)]["D"]["final_ply"]["path"] = "../x.ply"
    with pytest.raises(visualizer.ProtocolInvalid, match="canonical repository-relative"):
        visualizer.validate_result_ply_bindings(result, root=tmp_path)


def test_result_validation_requires_pass_and_exact_deferred_contract(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text('{"compact":true}', encoding="utf-8")
    workers = {}
    for arm in visualizer.ARMS:
        ply = tmp_path / "runs" / f"arm_{arm}" / "final.ply"
        ply.parent.mkdir(parents=True)
        ply.write_bytes(f"ply-{arm}".encode())
        workers[arm] = {
            "status": "PASS",
            "arm": arm,
            "training_seed": visualizer.REQUIRED_SEED,
            "n_opt_3d": 835,
            "final_ply": {
                "path": ply.relative_to(tmp_path).as_posix(),
                "sha256": visualizer.sha256_file(ply),
                "bytes": ply.stat().st_size,
                "n_gaussians": 835,
            },
        }
    payload = {
        "artifact_type": "compact_occupancy_refinement_factorial_iter3_result_v1",
        "status": "PASS",
        "runtime": {
            "preload": str(visualizer.PRELOAD),
            "preload_sha256": visualizer.EXPECTED_PRELOAD_SHA256,
        },
        "inputs": {"teacher_bundle": visualizer.directory_binding(bundle, root=tmp_path)},
        "workers": {str(visualizer.REQUIRED_SEED): workers},
        "visualization": {
            "status": "DEFERRED_POST_RESULT",
            "required_seed": visualizer.REQUIRED_SEED,
            "required_arms": list(visualizer.ARMS),
            "required_backend": "gsplat",
            "required_native_resolution": [
                visualizer.NATIVE_WIDTH,
                visualizer.NATIVE_HEIGHT,
            ],
        },
    }
    result_path = tmp_path / "RESULT.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded, bindings = visualizer.validate_result(
        result_path=result_path,
        root=tmp_path,
        bundle_path=bundle,
    )
    assert loaded["status"] == "PASS"
    assert bindings["result"]["sha256"] == visualizer.sha256_file(result_path)

    payload["status"] = "FAIL"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(visualizer.ProtocolInvalid, match="not an immutable PASS"):
        visualizer.validate_result(
            result_path=result_path,
            root=tmp_path,
            bundle_path=bundle,
        )

    payload["status"] = "PASS"
    payload["visualization"]["required_seed"] += 1
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(visualizer.ProtocolInvalid, match="contract differs"):
        visualizer.validate_result(
            result_path=result_path,
            root=tmp_path,
            bundle_path=bundle,
        )


def test_image_access_guard_allows_only_output_images_and_denies_dataset(tmp_path: Path) -> None:
    output = tmp_path / "output"
    dataset = tmp_path / "dataset"
    output.mkdir()
    dataset.mkdir()
    source = dataset / "source.png"
    source.write_bytes(b"source")
    elsewhere = tmp_path / "elsewhere.png"
    elsewhere.write_bytes(b"elsewhere")

    guard = visualizer.ImageAccessGuard(output, dataset_root=dataset)
    with guard, (output / "render.png").open("wb") as stream:
        stream.write(b"render")
    assert guard.record()["passed"] is True
    assert guard.record()["allowed_output_image_open_attempts"] >= 1

    denied = visualizer.ImageAccessGuard(output, dataset_root=dataset)
    with denied:
        with pytest.raises(visualizer.ProtocolInvalid, match="forbidden"):
            source.read_bytes()
        with pytest.raises(visualizer.ProtocolInvalid, match="forbidden"):
            elsewhere.read_bytes()
    assert denied.record()["source_rgb_or_dataset_open_attempts"] == 2
    assert denied.record()["passed"] is False


def test_contact_sheet_uses_all_arm_view_pngs_without_source_rgb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(visualizer, "ARMS", ("A", "B"))
    monkeypatch.setattr(visualizer, "VIEWS", ("V0", "V1"))
    monkeypatch.setattr(visualizer, "NATIVE_WIDTH", 4)
    monkeypatch.setattr(visualizer, "NATIVE_HEIGHT", 3)
    workers = {}
    for arm_index, arm in enumerate(visualizer.ARMS):
        images = []
        for view_index, view in enumerate(visualizer.VIEWS):
            path = tmp_path / "output" / f"arm_{arm}" / f"{view}.png"
            _write_png(path, (4, 3), (arm_index * 80, view_index * 80, 40))
            images.append(
                {
                    "view_id": view,
                    "path": path.relative_to(tmp_path).as_posix(),
                    "sha256": visualizer.sha256_file(path),
                }
            )
        workers[arm] = {"images": images}

    contact_path = tmp_path / "output" / "CONTACT.png"
    guard = visualizer.ImageAccessGuard(tmp_path / "output", dataset_root=tmp_path / "dataset")
    with guard:
        record = visualizer.build_contact_sheet(
            contact_path,
            workers,
            root=tmp_path,
            thumbnail_width=32,
        )
    assert guard.record()["passed"] is True
    assert record["columns"] == ["A", "B"]
    assert record["rows"] == ["V0", "V1"]
    assert record["source_panels_are_native_resolution"] is True
    assert record["sha256"] == visualizer.sha256_file(contact_path)
    with Image.open(contact_path) as image:
        assert list(image.size) == record["dimensions"]


def test_worker_payload_cpu_validation_binds_png_extension_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(visualizer, "ARMS", ("A",))
    monkeypatch.setattr(visualizer, "VIEWS", ("V0",))
    monkeypatch.setattr(visualizer, "NATIVE_WIDTH", 4)
    monkeypatch.setattr(visualizer, "NATIVE_HEIGHT", 3)
    plan_path = tmp_path / "PLAN.json"
    plan_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(visualizer, "PLAN", plan_path)
    image_path = tmp_path / "renders" / "arm_A" / "V0.png"
    _write_png(image_path, (4, 3), (20, 30, 40))
    extension = tmp_path / "gsplat_ext.so"
    extension.write_bytes(b"binary")
    plan = {
        "result": {"sha256": "1" * 64},
        "sources": {"aggregate_sha256": "2" * 64},
        "plys": {"A": {"path": "arm_A.ply", "sha256": "3" * 64}},
        "cameras": {"semantic_sha256": "4" * 64},
        "outputs": {"directory": "renders"},
        "preload": {
            "requested_path": "/tmp/libstdc++.so.6",
            "resolved_path": "/tmp/libstdc++.so.6.0.test",
            "sha256": "6" * 64,
        },
    }
    runtime = {
        "python": "test",
        "process_id": 1,
        "ld_preload_at_startup": plan["preload"]["requested_path"],
        "preload": plan["preload"],
    }
    runtime["binding_sha256"] = visualizer.canonical_hash(runtime)
    payload = {
        "artifact_type": ("compact_occupancy_refinement_factorial_iter3_visualization_worker_v1"),
        "status": "PASS",
        "arm": "A",
        "training_seed": visualizer.REQUIRED_SEED,
        "plan_sha256": visualizer.sha256_file(plan_path),
        "result_sha256": "1" * 64,
        "source_aggregate_sha256": "2" * 64,
        "ply": plan["plys"]["A"],
        "camera_semantic_sha256": "4" * 64,
        "images": [
            {
                "view_id": "V0",
                "path": image_path.relative_to(tmp_path).as_posix(),
                "sha256": visualizer.sha256_file(image_path),
                "bytes": image_path.stat().st_size,
                "dimensions": [4, 3],
                "color_tensor_sha256": "5" * 64,
                "backend": visualizer.EXPECTED_BACKEND,
                "device": "cuda:0",
            }
        ],
        "render_contract": {
            "backend": visualizer.EXPECTED_BACKEND,
            "device": "cuda",
            "rasterizer": "gsplat",
            "packed": False,
            "antialiased": False,
            "native_resolution": [4, 3],
        },
        "rgb_denial": {"passed": True, "source_rgb_or_dataset_open_attempts": 0},
        "gsplat_package": {
            "version": "test",
            "module_file": str(tmp_path / "gsplat" / "__init__.py"),
            "module_root": str(tmp_path / "gsplat"),
            "source_file_count": 1,
            "source_manifest_sha256": "7" * 64,
        },
        "loaded_gsplat_extension": {
            "path": str(extension),
            "sha256": visualizer.sha256_file(extension),
            "bytes": extension.stat().st_size,
        },
        "runtime": runtime,
    }
    assert visualizer.validate_worker_payload(payload, plan=plan, arm="A", root=tmp_path)

    image_path.write_bytes(b"tampered")
    with pytest.raises(visualizer.ProtocolInvalid, match="PNG binding differs"):
        visualizer.validate_worker_payload(payload, plan=plan, arm="A", root=tmp_path)


def test_public_run_arguments_do_not_accept_worker_overrides() -> None:
    args = visualizer.parse_args([])
    assert args.operation == "run"
    with pytest.raises(SystemExit):
        visualizer.parse_args(["run", "--arm", "A"])
    with pytest.raises(SystemExit):
        visualizer.parse_args(["_worker", "--arm", "A"])


def test_failure_payload_is_finite_and_bounded() -> None:
    payload = visualizer._bounded_failure(RuntimeError("x" * 5000), stage="test", arm="A")
    encoded = visualizer.canonical_bytes(payload)
    assert json.loads(encoded)["status"] == "FAIL"
    assert len(payload["failure"]["message"]) == 2000
