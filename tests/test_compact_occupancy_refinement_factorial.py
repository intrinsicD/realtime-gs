from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import math
import os
import stat
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from benchmarks import compact_occupancy_refinement_factorial as factorial

from rtgs.core.camera import Camera
from rtgs.core.observation2d import GaussianObservationField, GaussianPointProposal
from rtgs.data.reconstruction_inputs import ReconstructionInputs


def _skip_without_sealed_torch_binding() -> None:
    """Skip when the installed torch is not the sealed reproduction build.

    The runtime-path binding is pinned to a specific torch build via
    ``TORCH_INSTANTIATOR_SHA256`` (the SHA-256 of ``torch.distributed.nn.jit.instantiator``).
    On any other torch install the binding raises ``ProtocolInvalid``; a clean CI checkout
    uses a different (CPU) torch than the sealed build, so these binding tests skip rather
    than hard-fail — matching the existing CUDA precondition-skip in this module.
    """
    try:
        instantiator = importlib.import_module("torch.distributed.nn.jit.instantiator")
        path = Path(instantiator.__file__).resolve(strict=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception as exc:  # pragma: no cover - defensive: torch layout differs
        pytest.skip(f"torch instantiator source unavailable for the sealed binding: {exc}")
    if digest != factorial.TORCH_INSTANTIATOR_SHA256:
        pytest.skip("torch instantiator source binding differs from the sealed reproduction build")


@contextlib.contextmanager
def _without_preloaded_rgb_modules():
    saved = {
        name: module
        for name, module in tuple(sys.modules.items())
        if factorial.RGBAccessGuard._forbidden_module(name)
    }
    for name in saved:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        sys.modules.update(saved)


def _run_local_source_failure_probe(kind: str) -> dict:
    script = r"""
import json
import os
import sys
import types
from benchmarks import compact_occupancy_refinement_factorial as factorial

kind = sys.argv[1]
if kind == "shadowed":
    module = types.ModuleType("rtgs.core.camera")
    module.__file__ = str(factorial.ROOT / "tests/test_compact_occupancy_refinement_factorial.py")
    sys.modules[module.__name__] = module
elif kind == "unbound":
    module = types.ModuleType("factorial_unbound_local_probe")
    module.__file__ = str(factorial.ROOT / "src/rtgs/cli.py")
    sys.modules[module.__name__] = module
else:
    raise ValueError(kind)

def rejected(call):
    try:
        call()
    except factorial.ProtocolInvalid as error:
        return str(error)
    raise AssertionError("binding unexpectedly accepted injected module")

source_error = rejected(factorial.source_hashes)
os.environ["LD_PRELOAD"] = str(factorial.PRELOAD)
factorial.torch.cuda.is_available = lambda: True
factorial._torch_runtime_import_path_binding = lambda: {"normalized_sys_path": list(sys.path)}
runtime_error = rejected(factorial.runtime_binding)
print(json.dumps({
    "module_origin_violations": factorial.module_origin_violations(),
    "unbound_local_sources": factorial.unbound_loaded_local_sources(),
    "source_error": source_error,
    "runtime_error": runtime_error,
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, kind],
        cwd=factorial.ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    return json.loads(completed.stdout)


def _field(view_name: str, *, proxy: bool = False) -> GaussianObservationField:
    return GaussianObservationField(
        width=8,
        height=6,
        means=torch.tensor([[2.5, 2.5], [5.5, 3.5]]),
        log_scales=torch.log(torch.tensor([[0.8, 1.1], [1.0, 0.7]])),
        rotations=torch.tensor([0.2, -0.3]),
        colors=(torch.ones(2, 3) if proxy else torch.tensor([[0.2, 0.4, 0.6], [0.7, 0.3, 0.1]])),
        amplitudes=torch.tensor([0.8, 0.4]) if not proxy else torch.tensor([0.9, 0.6]),
        filter_variance=torch.tensor([0.1, 0.2]),
        blend_mode="normalized",
        epsilon=1e-8,
        sigma_cutoff=3.0,
        support_fade_alpha=0.0,
        aa_dilation=0.0,
        view_id=view_name,
        fit_window=(0, 0, 8, 6),
        n_init=2,
        provider="synthetic_fixture",
    )


def _camera() -> Camera:
    return Camera(
        fx=7.0,
        fy=7.0,
        cx=4.0,
        cy=3.0,
        width=8,
        height=6,
        R=torch.eye(3),
        t=torch.tensor([0.0, 0.0, 2.0]),
    )


def _inputs(*, proxy: bool = False) -> ReconstructionInputs:
    fields = [_field(name, proxy=proxy) for name in factorial.EXPECTED_VIEWS]
    return ReconstructionInputs(
        observations=fields,
        cameras=[_camera() for _ in fields],
        view_names=list(factorial.EXPECTED_VIEWS),
        name="fixture",
    )


def _test_bank_sample():
    field = _field(factorial.EXPECTED_VIEWS[0])
    evaluation_seed = factorial.TEST_EVALUATION_SEEDS[0]
    generator_seed = factorial.evaluation_bank_seed(
        evaluation_seed,
        factorial.EXPECTED_VIEWS[0],
        "uniform",
    )
    samples = GaussianPointProposal(field, field).sample(
        factorial.BANK_ATTEMPTS,
        uniform_fraction=1.0,
        generator=torch.Generator(device="cpu").manual_seed(generator_seed),
    )
    color = field.query(samples.xy, component_chunk=640).color
    return field, evaluation_seed, generator_seed, samples, color


def _validate_test_bank(field, evaluation_seed, generator_seed, samples, color):
    factorial._validate_bank_invariants(
        evaluation_seed=evaluation_seed,
        seed_domain="focused_test",
        view_index=0,
        view_name=factorial.EXPECTED_VIEWS[0],
        kind="uniform",
        generator_seed=generator_seed,
        product=field,
        samples=samples,
        color=color,
        require_color=color is not None,
    )


def _decision_records(d_over_b: float = 0.8, u_over_b: float = 1.0):
    result = {}
    for seed in factorial.TEST_TRAIN_SEEDS:
        arms = {}
        for arm in factorial.ARMS:
            scale_q = d_over_b if arm == "D" else 1.0
            scale_u = u_over_b if arm == "D" else 1.0
            arms[arm] = {
                "checkpoint_metrics": {
                    str(step): {
                        "J_Q": base * scale_q,
                        "J_U": base * scale_u,
                    }
                    for step, base in zip(
                        factorial.CHECKPOINTS,
                        (1.0, 0.8, 0.6, 0.5),
                        strict=True,
                    )
                }
            }
        result[seed] = arms
    return result


def test_evaluation_bank_seed_is_stable_and_view_kind_isolated():
    first, second = factorial.TEST_EVALUATION_SEEDS
    value = factorial.evaluation_bank_seed(first, "C0001", "uniform")
    assert value == factorial.evaluation_bank_seed(first, "C0001", "uniform")
    assert 0 <= value < 2**63
    assert value != factorial.evaluation_bank_seed(first, "C0008", "uniform")
    assert value != factorial.evaluation_bank_seed(first, "C0001", "proposal")
    assert value != factorial.evaluation_bank_seed(second, "C0001", "uniform")


def test_log_auc_and_geometric_mean_use_frozen_math():
    assert factorial.log_auc([2.0] * 4) == pytest.approx(math.log(2.0))
    assert factorial.geometric_mean([0.5, 2.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        factorial.log_auc([1.0, 2.0])
    with pytest.raises(ValueError):
        factorial.geometric_mean([0.0])


def test_decision_passes_only_when_every_gate_passes():
    passed = factorial.compute_decision(
        _decision_records(), [0.99] * 21, seeds=factorial.TEST_TRAIN_SEEDS
    )
    assert passed["decision"] == "AUTHORIZE_DENSITY_FOLLOWUP"
    assert all(passed["gates"].values())

    active_failure = factorial.compute_decision(
        _decision_records(),
        [0.94] + [0.99] * 20,
        seeds=factorial.TEST_TRAIN_SEEDS,
    )
    assert active_failure["decision"] == "NO_REFINEMENT_TARGET_PROMOTION"
    assert not active_failure["gates"]["active_fraction_each_ge_0_95"]

    uniform_failure = factorial.compute_decision(
        _decision_records(d_over_b=0.8, u_over_b=1.11),
        [0.99] * 21,
        seeds=factorial.TEST_TRAIN_SEEDS,
    )
    assert uniform_failure["decision"] == "NO_REFINEMENT_TARGET_PROMOTION"
    assert not uniform_failure["gates"]["final_J_U_each_ratio_le_1_10"]


def test_raw_proxy_alignment_and_product_amplitudes_are_exact():
    teachers = _inputs()
    proxies = _inputs(proxy=True)
    products, records = factorial.build_product_fields(teachers, proxies)
    assert [record["m_opt_2d"] for record in records] == [2] * 7
    for teacher, proxy, product in zip(
        teachers.observations, proxies.observations, products, strict=True
    ):
        assert torch.equal(product.amplitudes, teacher.amplitudes * proxy.amplitudes)
        assert torch.equal(product.colors, torch.ones_like(product.colors))
        assert product.color_grads is None


def test_product_construction_retains_variable_per_view_m_opt_list():
    teachers = _inputs()
    proxies = _inputs(proxy=True)
    teacher = teachers.observations[0]
    proxy = proxies.observations[0]
    teacher_three = replace(
        teacher,
        means=torch.cat([teacher.means, teacher.means[:1] + torch.tensor([[0.2, 0.1]])]),
        log_scales=torch.cat([teacher.log_scales, teacher.log_scales[:1]]),
        rotations=torch.cat([teacher.rotations, teacher.rotations[:1]]),
        colors=torch.cat([teacher.colors, teacher.colors[:1]]),
        amplitudes=torch.cat([teacher.amplitudes, torch.tensor([0.2])]),
        filter_variance=torch.cat([teacher.filter_variance, teacher.filter_variance[:1]]),
    )
    proxy_three = replace(
        proxy,
        means=teacher_three.means,
        log_scales=teacher_three.log_scales,
        rotations=teacher_three.rotations,
        colors=torch.ones(3, 3),
        amplitudes=torch.tensor([0.9, 0.6, 0.5]),
        filter_variance=teacher_three.filter_variance,
    )
    variable_teachers = replace(teachers, observations=[teacher_three, *teachers.observations[1:]])
    variable_proxies = replace(proxies, observations=[proxy_three, *proxies.observations[1:]])
    products, records = factorial.build_product_fields(variable_teachers, variable_proxies)
    assert [record["m_opt_2d"] for record in records] == [3, 2, 2, 2, 2, 2, 2]
    assert [field.n for field in products] == [3, 2, 2, 2, 2, 2, 2]


def test_raw_proxy_alignment_rejects_geometry_scalar_and_camera_mismatch():
    teachers = _inputs()
    proxies = _inputs(proxy=True)
    bad_geometry = replace(proxies.observations[0], means=proxies.observations[0].means.flip(0))
    with pytest.raises(factorial.ProtocolInvalid, match="positionally misaligned"):
        factorial.validate_proxy_alignment(
            teachers,
            replace(proxies, observations=[bad_geometry, *proxies.observations[1:]]),
        )

    bad_scalar = replace(proxies.observations[0], amplitudes=torch.tensor([1.1, 0.6]))
    with pytest.raises(factorial.ProtocolInvalid, match=r"outside \[0,1\]"):
        factorial.validate_proxy_alignment(
            teachers,
            replace(proxies, observations=[bad_scalar, *proxies.observations[1:]]),
        )

    camera = replace(proxies.cameras[0], fx=8.0)
    with pytest.raises(factorial.ProtocolInvalid, match="camera differs"):
        factorial.validate_proxy_alignment(
            teachers,
            replace(proxies, cameras=[camera, *proxies.cameras[1:]]),
        )

    bad_blend = replace(proxies.observations[0], blend_mode="additive")
    with pytest.raises(factorial.ProtocolInvalid, match="support/canvas semantics differ"):
        factorial.validate_proxy_alignment(
            teachers,
            replace(proxies, observations=[bad_blend, *proxies.observations[1:]]),
        )

    bad_n_init = replace(proxies.observations[0], n_init=3)
    with pytest.raises(factorial.ProtocolInvalid, match="support/canvas semantics differ"):
        factorial.validate_proxy_alignment(
            teachers,
            replace(proxies, observations=[bad_n_init, *proxies.observations[1:]]),
        )


def test_bank_archive_roundtrip_hashes_every_tensor(tmp_path):
    teachers = _inputs()
    products, _ = factorial.build_product_fields(teachers, _inputs(proxy=True))
    path = tmp_path / "banks.npz"
    seed = factorial.TEST_EVALUATION_SEEDS[0]
    record = factorial.generate_test_bank_archive(seed, teachers, products, path)
    loaded, metadata = factorial.load_bank_archive(
        path,
        seed,
        expected_seed_domain="focused_test",
    )
    assert record["sha256"] == factorial.sha256_file(path)
    assert len(loaded) == 7
    assert metadata["evaluation_seed"] == seed
    assert metadata["seed_domain"] == "focused_test"
    for view in loaded:
        assert view["uniform"]["xy"].shape == (4096, 2)
        assert view["uniform"]["active"].all()
        assert view["proposal"]["active"].shape == (4096,)

    with path.open("r+b") as stream:
        stream.seek(-32, 2)
        byte = stream.read(1)
        stream.seek(-1, 1)
        stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises((factorial.ProtocolInvalid, zipfile.BadZipFile)):
        factorial.load_bank_archive(
            path,
            seed,
            expected_seed_domain="focused_test",
        )


def test_archive_load_invokes_seed_firewall_before_file_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    missing = tmp_path / "must_not_be_opened.npz"
    sentinel = factorial.TEST_EVALUATION_SEEDS[0]
    observed = []

    def reject(*seeds: int) -> None:
        observed.extend(seeds)
        raise factorial.ProtocolInvalid("focused seed-firewall sentinel")

    monkeypatch.setattr(factorial, "reject_official_seed_in_focused_test", reject)
    with pytest.raises(factorial.ProtocolInvalid, match="seed-firewall sentinel"):
        factorial.load_bank_archive(
            missing,
            sentinel,
            expected_seed_domain="focused_test",
        )
    assert observed == [sentinel]
    assert not missing.exists()


def test_bank_invariant_diagnostic_retains_nonfinite_sample_context():
    field, evaluation_seed, generator_seed, samples, color = _test_bank_sample()
    proposal_density = samples.proposal_density.clone()
    proposal_density[11] = torch.inf
    bad_samples = replace(samples, proposal_density=proposal_density)
    with pytest.raises(factorial.BankInvariantError) as captured:
        _validate_test_bank(field, evaluation_seed, generator_seed, bad_samples, color)
    diagnostic = captured.value.diagnostic
    assert diagnostic["evaluation_seed"] == factorial.TEST_EVALUATION_SEEDS[0]
    assert diagnostic["seed_domain"] == "focused_test"
    assert diagnostic["view_name"] == factorial.EXPECTED_VIEWS[0]
    assert diagnostic["kind"] == "uniform"
    assert diagnostic["generator_seed"] == generator_seed
    assert diagnostic["first_failing_index"] == 11
    assert diagnostic["first_failing_xy"] == pytest.approx(samples.xy[11].tolist())
    assert diagnostic["fit_window"] == list(field.fit_window)
    assert diagnostic["predicate_counts"]["nonfinite_proposal_density_count"] == 1
    assert diagnostic["tensor_sha256"]["proposal_density"] == factorial.tensor_hash(
        proposal_density
    )


def test_bank_invariant_diagnostic_retains_nonfinite_xy_context():
    field, evaluation_seed, generator_seed, samples, color = _test_bank_sample()
    xy = samples.xy.clone()
    xy[13, 0] = torch.inf
    bad_samples = replace(samples, xy=xy)
    with pytest.raises(factorial.BankInvariantError) as captured:
        _validate_test_bank(field, evaluation_seed, generator_seed, bad_samples, color)
    diagnostic = captured.value.diagnostic
    assert diagnostic["first_failing_index"] == 13
    assert diagnostic["first_failing_xy"] == [None, float(xy[13, 1])]
    assert diagnostic["predicate_counts"]["nonfinite_xy_count"] == 1
    assert diagnostic["predicate_counts"]["uniform_coordinate_outside_half_open_count"] == 1
    assert diagnostic["tensor_sha256"]["xy"] == factorial.tensor_hash(xy)


def test_bank_invariant_diagnostic_retains_nonfinite_color_in_terminal_failure():
    field, evaluation_seed, generator_seed, samples, color = _test_bank_sample()
    color = color.clone()
    color[17, 2] = torch.nan
    with pytest.raises(factorial.BankInvariantError) as captured:
        _validate_test_bank(field, evaluation_seed, generator_seed, samples, color)
    diagnostic = captured.value.diagnostic
    assert diagnostic["first_failing_index"] == 17
    assert diagnostic["first_failing_xy"] == pytest.approx(samples.xy[17].tolist())
    assert diagnostic["predicate_counts"]["nonfinite_color_count"] == 1
    assert diagnostic["tensor_sha256"]["color"] == factorial.tensor_hash(color)
    terminal = factorial.bounded_failure(captured.value, stage="test_bank_generation")
    assert terminal["scientific_decision"] == "UNAVAILABLE"
    assert terminal["promotion_authorized"] is False
    assert terminal["failure"]["bank_invariant"] == diagnostic


def test_bank_invariant_diagnostic_structures_wrong_sample_shape():
    field, evaluation_seed, generator_seed, samples, _color = _test_bank_sample()
    bad_samples = replace(samples, xy=samples.xy[:-1])
    with pytest.raises(factorial.BankInvariantError) as captured:
        _validate_test_bank(field, evaluation_seed, generator_seed, bad_samples, None)
    diagnostic = captured.value.diagnostic
    assert diagnostic["first_failing_index"] == factorial.BANK_ATTEMPTS - 1
    assert diagnostic["first_failing_xy"] is None
    assert diagnostic["predicate_counts"]["shape_mismatch_count"] == 1
    assert diagnostic["shape_mismatches"]["xy"] == {
        "expected": [factorial.BANK_ATTEMPTS, 2],
        "actual": [factorial.BANK_ATTEMPTS - 1, 2],
    }
    assert diagnostic["tensor_sha256"]["xy"] == factorial.tensor_hash(bad_samples.xy)


def test_rgb_guard_has_negative_controls_and_no_false_positive(tmp_path):
    guard = factorial.RGBAccessGuard()
    with _without_preloaded_rgb_modules(), guard:
        assert json.loads('{"compact": true}')["compact"] is True
        (tmp_path / "allowed.json").write_text("{}", encoding="utf-8")
    assert guard.record() == {
        "source_rgb_open_attempts": 0,
        "forbidden_import_attempts": 0,
        "negative_control_denials": 3,
        "forbidden_modules_at_entry": [],
        "forbidden_modules_at_exit": [],
        "passed": True,
    }


def test_rgb_guard_counts_real_forbidden_attempt(tmp_path):
    guard = factorial.RGBAccessGuard()
    with _without_preloaded_rgb_modules(), guard, pytest.raises(PermissionError):
        (tmp_path / "forbidden.png").read_bytes()
    assert guard.record()["source_rgb_open_attempts"] == 1
    assert guard.record()["passed"] is False


def test_rgb_guard_denies_importlib_bypass_and_counts_attempt():
    guard = factorial.RGBAccessGuard()
    with _without_preloaded_rgb_modules():
        with guard, pytest.raises(ImportError):
            importlib.import_module("rtgs.data.calibrated")
        assert guard.record()["forbidden_import_attempts"] == 1
        assert "rtgs.data.calibrated" not in sys.modules


def test_rgb_guard_fails_if_forbidden_module_crosses_entry_or_exit():
    sys.modules["PIL.Image"] = object()
    try:
        with pytest.raises(factorial.ProtocolInvalid, match="before denial boundary"):
            factorial.RGBAccessGuard().__enter__()
    finally:
        sys.modules.pop("PIL.Image", None)

    with _without_preloaded_rgb_modules():
        guard = factorial.RGBAccessGuard()
        try:
            with pytest.raises(factorial.ProtocolInvalid, match="crossed denial boundary"), guard:
                sys.modules["PIL.Image"] = object()
        finally:
            sys.modules.pop("PIL.Image", None)


def test_paired_worker_validation_checks_sampling_and_target_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(factorial, "ROOT", tmp_path)
    base = {key: f"same-{key}" for key in factorial.PAIRED_SAMPLE_KEYS if key not in {"view_index"}}
    base["view_index"] = 0
    left_steps = []
    right_steps = []
    for step in range(140):
        left_steps.append(
            {
                **base,
                "step": step + 1,
                "target_density_sha256": f"left-target-{step}",
                "importance_sha256": f"left-importance-{step}",
            }
        )
        right_steps.append(
            {
                **base,
                "step": step + 1,
                "target_density_sha256": f"right-target-{step}",
                "importance_sha256": f"right-importance-{step}",
            }
        )
    factorial.exclusive_json(tmp_path / "left.json", {"steps": left_steps})
    factorial.exclusive_json(tmp_path / "right.json", {"steps": right_steps})
    left = {"arm": "A", "history": {"path": "left.json"}}
    right = {"arm": "C", "history": {"path": "right.json"}}
    result = factorial.validate_paired_workers(left, right)
    assert result["steps"] == 140
    assert result["target_hash_difference_counts"] == {
        "target_density_sha256": 140,
        "importance_sha256": 140,
    }
    right_steps[10]["xy_sha256"] = "different"
    (tmp_path / "right2.json").write_text(json.dumps({"steps": right_steps}), encoding="utf-8")
    with pytest.raises(factorial.ProtocolInvalid, match="xy_sha256"):
        factorial.validate_paired_workers(left, {"arm": "C", "history": {"path": "right2.json"}})

    partially_equal = [dict(step) for step in right_steps]
    partially_equal[10]["xy_sha256"] = left_steps[10]["xy_sha256"]
    partially_equal[10]["target_density_sha256"] = left_steps[10]["target_density_sha256"]
    (tmp_path / "right3.json").write_text(json.dumps({"steps": partially_equal}), encoding="utf-8")
    with pytest.raises(factorial.ProtocolInvalid, match="at every step"):
        factorial.validate_paired_workers(left, {"arm": "C", "history": {"path": "right3.json"}})


def test_step_zero_requires_exact_snapshot_and_metrics():
    records = {
        arm: {
            "snapshots": [{"semantic_sha256": "same"}],
            "checkpoint_metrics": {"0": {"J_U": 1.0, "J_Q": 2.0}},
        }
        for arm in factorial.ARMS
    }
    assert factorial.validate_step_zero(records)["all_four_arms_exact"] is True
    records["D"]["checkpoint_metrics"]["0"]["J_Q"] = 2.1
    with pytest.raises(factorial.ProtocolInvalid, match="step-zero"):
        factorial.validate_step_zero(records)


def test_terminal_failure_payload_is_finite_and_bounded():
    payload = factorial.bounded_failure(
        RuntimeError("x" * 5000), stage="test", attempt_sha256="a" * 64
    )
    encoded = factorial.canonical_bytes(payload)
    assert b'"status":"FAIL"' in encoded
    assert payload["decision"] == "UNAVAILABLE"
    assert payload["scientific_decision"] == "UNAVAILABLE"
    assert payload["promotion_authorized"] is False
    assert len(payload["failure"]["message"]) == 2000
    with pytest.raises(ValueError):
        factorial.canonical_bytes({"bad": float("nan")})

    bank_error = factorial.BankInvariantError(
        "boundary",
        {
            "seed_domain": "focused_test",
            "view_name": "C0001",
            "first_failing_index": 7,
            "first_failing_xy": [8.0, 2.0],
        },
    )
    bank_payload = factorial.bounded_failure(bank_error, stage="test_bank")
    assert bank_payload["failure"]["bank_invariant"] == bank_error.diagnostic

    worker_error = factorial.WorkerProcessError(
        "worker failed",
        {
            "seed": factorial.TEST_TRAIN_SEEDS[0],
            "arm": "A",
            "process": {"stdout_tail": "child out", "stderr_tail": "child err"},
            "worker_artifact": {"sha256": "f" * 64},
        },
    )
    worker_payload = factorial.bounded_failure(worker_error, stage="test_worker")
    assert worker_payload["failure"]["worker_process"] == worker_error.diagnostic


def test_run_entry_reports_terminal_serialization_fallback_as_failure(tmp_path, monkeypatch):
    seal = tmp_path / "seal.json"
    attempt = tmp_path / "attempt.json"
    result = tmp_path / "result.json"
    factorial.exclusive_json(seal, {"status": "SEALED"})
    monkeypatch.setattr(factorial, "SEAL", seal)
    monkeypatch.setattr(factorial, "ATTEMPT", attempt)
    monkeypatch.setattr(factorial, "RESULT", result)
    monkeypatch.setattr(factorial, "_namespace_clean_for_run", lambda: True)
    monkeypatch.setattr(
        factorial,
        "verify_seal",
        lambda: {"preregistration_sha256": "p" * 64, "config": {}},
    )
    monkeypatch.setattr(
        factorial,
        "execute_attempt",
        lambda **_kwargs: {"status": "PASS", "nonfinite": float("nan")},
    )
    assert factorial.run_entry() == 2
    written = factorial.strict_json(result)
    assert written["status"] == "FAIL"
    assert written["failure"]["stage"] == "terminal_result_serialization"


def test_run_entry_rechecks_namespace_before_attempt_publication(tmp_path, monkeypatch):
    attempt = tmp_path / "attempt.json"
    monkeypatch.setattr(factorial, "ATTEMPT", attempt)
    checks = iter((True, False))
    monkeypatch.setattr(factorial, "_namespace_clean_for_run", lambda: next(checks))
    monkeypatch.setattr(
        factorial,
        "verify_seal",
        lambda: {"preregistration_sha256": "p" * 64, "config": {}},
    )
    monkeypatch.setattr(factorial, "SEAL", tmp_path / "seal.json")
    (tmp_path / "seal.json").write_text("{}", encoding="utf-8")
    with pytest.raises(factorial.ProtocolInvalid, match="before attempt publication"):
        factorial.run_entry()
    assert not attempt.exists()


def test_fresh_harness_import_does_not_preload_rgb_capable_modules():
    script = """
import json
import sys
from benchmarks import compact_occupancy_refinement_factorial as factorial
forbidden = [
    name for name in sys.modules
    if name == 'PIL' or name.startswith('PIL.')
    or name in {'rtgs.data.calibrated', 'rtgs.data.scene'}
]
print(json.dumps({
    'forbidden': sorted(forbidden),
    'unbound_local_sources': factorial.unbound_loaded_local_sources(),
    'module_origin_violations': factorial.module_origin_violations(),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=factorial.ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    assert json.loads(completed.stdout) == {
        "forbidden": [],
        "module_origin_violations": [],
        "unbound_local_sources": [],
    }


def test_shadowed_rtgs_module_fails_source_and_runtime_bindings():
    record = _run_local_source_failure_probe("shadowed")
    assert len(record["module_origin_violations"]) == 1
    assert record["module_origin_violations"][0].startswith("shadowed:rtgs.core.camera=")
    assert record["unbound_local_sources"] == []
    assert "loaded rtgs module origin mismatch" in record["source_error"]
    assert "loaded rtgs module origin mismatch" in record["runtime_error"]


def test_unbound_repository_local_module_fails_source_and_runtime_bindings():
    assert Path("src/rtgs/cli.py") not in factorial.SOURCE_PATHS
    record = _run_local_source_failure_probe("unbound")
    assert record["module_origin_violations"] == []
    assert "src/rtgs/cli.py" in record["unbound_local_sources"]
    assert "loaded local source closure is not sealed" in record["source_error"]
    assert "loaded local source closure is not sealed" in record["runtime_error"]


def test_frozen_config_exposes_no_dense_checkpoint_evaluation():
    config = factorial.frozen_test_config("D", factorial.TEST_TRAIN_SEEDS[0])
    assert config.device == "cuda:0"
    assert config.schedule_mode == "balanced_cycle"
    assert config.target_mode == "proposal_attempt"
    assert config.evaluate_checkpoint_risks is False
    assert config.checkpoints == (0, 35, 70, 140)
    assert config.outer_microbatch == config.attempts_per_step == 128

    common = factorial.common_train_config_record(config)
    optimizer = factorial.optimizer_record(common)
    assert common["proposal_mode"] == "area_gaussian"
    assert common["lr_means"] == 1.6e-4
    assert common["sh_degree"] == 0
    assert common["sh_color_activation"] == "hard"
    assert common["kernel_support_mode"] == "hard"
    assert common["visibility_margin_sigma"] == 3.0
    assert optimizer["betas"] == [0.9, 0.999]
    assert optimizer["foreach"] is False
    assert not (
        set(factorial.TRAIN_SEEDS + factorial.EVALUATION_SEEDS)
        & set(factorial.TEST_TRAIN_SEEDS + factorial.TEST_EVALUATION_SEEDS)
    )
    assert not (
        set(
            factorial.TRAIN_SEEDS
            + factorial.EVALUATION_SEEDS
            + factorial.TEST_TRAIN_SEEDS
            + factorial.TEST_EVALUATION_SEEDS
        )
        & set(factorial.CONSUMED_SEEDS)
    )


def test_torch_runtime_path_binding_survives_exact_six_adam_groups_and_step():
    _skip_without_sealed_torch_binding()
    before = factorial._torch_runtime_import_path_binding()
    parameters = {
        name: torch.nn.Parameter(torch.tensor([float(index + 1)]))
        for index, name in enumerate(("means", "quats", "scales", "opacities", "sh0", "shN"))
    }
    learning_rates = {
        "means": 1.6e-4 * factorial.EXPLICIT_EXTENT,
        "quats": 1e-3,
        "scales": 5e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 1.25e-4,
    }
    optimizers = {
        name: torch.optim.Adam(
            [{"params": [parameter], "lr": learning_rates[name], "name": name}],
            betas=(0.9, 0.999),
            eps=1e-15,
            weight_decay=0.0,
            amsgrad=False,
            maximize=False,
            foreach=False,
            fused=False,
        )
        for name, parameter in parameters.items()
    }
    sum(parameter.square().sum() for parameter in parameters.values()).backward()
    for optimizer in optimizers.values():
        optimizer.step()
    after = factorial._torch_runtime_import_path_binding()
    assert after == before
    assert after["rng_state_unchanged"] is True
    assert after["generated_source"] == {
        "name": "_remote_module_non_scriptable.py",
        "bytes": 2355,
        "sha256": "8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8",
    }


def test_full_runtime_binding_normalizes_fresh_random_directory_names():
    if not torch.cuda.is_available():
        pytest.skip("official runtime binding requires CUDA")
    script = """
import json
from benchmarks import compact_occupancy_refinement_factorial as factorial
runtime = factorial.runtime_binding()
from torch.distributed.nn.jit import instantiator
record = {"raw": instantiator.INSTANTIATED_TEMPLATE_DIR_PATH, "runtime": runtime}
print(json.dumps(record, sort_keys=True))
"""
    records = []
    environment = dict(os.environ)
    environment["LD_PRELOAD"] = str(factorial.PRELOAD)
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=factorial.ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=True,
        )
        records.append(json.loads(completed.stdout))
    assert records[0]["raw"] != records[1]["raw"]
    assert records[0]["runtime"] == records[1]["runtime"]
    assert records[0]["runtime"]["sys_path"][-1] == factorial.TORCH_TEMP_SENTINEL


def test_torch_runtime_path_rejects_foreign_duplicate_reordered_and_symlink_entries(tmp_path):
    _skip_without_sealed_torch_binding()
    factorial._torch_runtime_import_path_binding()
    from torch.distributed.nn.jit import instantiator

    original = list(sys.path)
    raw_path = instantiator.INSTANTIATED_TEMPLATE_DIR_PATH
    alias = tmp_path / "instantiator_alias"
    alias.symlink_to(raw_path, target_is_directory=True)
    mutations = (
        [*original, str(tmp_path / "foreign")],
        [*original, raw_path],
        [raw_path, *original[:-1]],
        [*original[:-1], str(alias)],
    )
    try:
        for mutated in mutations:
            sys.path[:] = mutated
            with pytest.raises(factorial.ProtocolInvalid, match="unique final sys.path"):
                factorial._torch_runtime_import_path_binding()
    finally:
        sys.path[:] = original


def test_full_runtime_binding_exposes_foreign_path_before_valid_temp_entry(tmp_path: Path):
    if not torch.cuda.is_available():
        pytest.skip("official runtime binding requires CUDA")
    script = r"""
import json
import sys
from benchmarks import compact_occupancy_refinement_factorial as factorial

baseline = factorial.runtime_binding()
foreign = sys.argv[1]
sys.path.insert(-1, foreign)
changed = factorial.runtime_binding()
differences = factorial._key_differences(
    {"runtime": baseline},
    {"runtime": changed},
)
print(json.dumps({
    "changed": changed != baseline,
    "differences": differences,
    "foreign": foreign,
    "sentinel_still_final": changed["sys_path"][-1] == factorial.TORCH_TEMP_SENTINEL,
}, sort_keys=True))
"""
    foreign = str(tmp_path / "foreign_before_temp")
    environment = dict(os.environ)
    environment["LD_PRELOAD"] = str(factorial.PRELOAD)
    completed = subprocess.run(
        [sys.executable, "-c", script, foreign],
        cwd=factorial.ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    record = json.loads(completed.stdout)
    assert record["changed"] is True
    assert record["sentinel_still_final"] is True
    assert any(
        difference["actual"] == foreign and difference["path"].startswith("$.runtime.sys_path[")
        for difference in record["differences"]
    )


def test_torch_runtime_path_rejects_mode_owner_origin_source_and_member_tampering(
    monkeypatch, tmp_path
):
    _skip_without_sealed_torch_binding()
    factorial._torch_runtime_import_path_binding()
    from torch.distributed.nn.jit import instantiator

    directory = Path(instantiator.INSTANTIATED_TEMPLATE_DIR_PATH)
    source = directory / "_remote_module_non_scriptable.py"
    original_source = source.read_bytes()
    original_mode = stat.S_IMODE(directory.stat().st_mode)

    directory.chmod(0o755)
    try:
        with pytest.raises(factorial.ProtocolInvalid, match="ownership/mode"):
            factorial._torch_runtime_import_path_binding()
    finally:
        directory.chmod(original_mode)

    actual_uid = os.getuid()
    with monkeypatch.context() as context:
        context.setattr(factorial.os, "getuid", lambda: actual_uid + 1)
        with pytest.raises(factorial.ProtocolInvalid, match="ownership/mode"):
            factorial._torch_runtime_import_path_binding()

    fake_instantiator = tmp_path / "instantiator.py"
    fake_instantiator.write_text("# spoof\n", encoding="utf-8")
    with monkeypatch.context() as context:
        context.setattr(instantiator, "__file__", str(fake_instantiator))
        with pytest.raises(factorial.ProtocolInvalid, match="outside the bound package"):
            factorial._torch_runtime_import_path_binding()

    source.write_bytes(original_source + b"# tamper\n")
    try:
        with pytest.raises(factorial.ProtocolInvalid, match="generated remote-module source"):
            factorial._torch_runtime_import_path_binding()
    finally:
        source.write_bytes(original_source)

    extra = directory / "unexpected.py"
    extra.write_text("# unexpected\n", encoding="utf-8")
    try:
        with pytest.raises(factorial.ProtocolInvalid, match="member allowlist"):
            factorial._torch_runtime_import_path_binding()
    finally:
        extra.unlink()


def test_runtime_binding_keeps_literal_environment_and_numeric_flags():
    if not torch.cuda.is_available():
        pytest.skip("official runtime binding requires CUDA")
    script = """
import json
import os
import torch
from benchmarks import compact_occupancy_refinement_factorial as factorial
os.environ["PYTHONPATH"] = "literal-test-path"
baseline = factorial.runtime_binding()
before_precision = torch.backends.cuda.matmul.fp32_precision
before_deterministic = torch.are_deterministic_algorithms_enabled()
try:
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.use_deterministic_algorithms(not before_deterministic)
    changed = factorial.runtime_binding()
finally:
    torch.backends.cuda.matmul.fp32_precision = before_precision
    torch.use_deterministic_algorithms(before_deterministic)
print(json.dumps({
    "baseline_pythonpath": baseline["pythonpath"],
    "baseline_preload": baseline["effective_ld_preload"],
    "baseline_precision": baseline["cuda_matmul_fp32_precision"],
    "baseline_deterministic": baseline["deterministic_algorithms"],
    "changed_precision": changed["cuda_matmul_fp32_precision"],
    "changed_deterministic": changed["deterministic_algorithms"],
    "different": changed != baseline,
}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["LD_PRELOAD"] = str(factorial.PRELOAD)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=factorial.ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    record = json.loads(completed.stdout)
    assert record["baseline_pythonpath"] == "literal-test-path"
    assert record["baseline_preload"] == str(factorial.PRELOAD)
    assert record["changed_precision"] == "tf32"
    assert record["changed_deterministic"] is (not record["baseline_deterministic"])
    assert record["different"] is True


def test_runtime_binding_rejects_perturbed_ld_preload(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LD_PRELOAD", f"{factorial.PRELOAD}:/tmp/foreign-libstdc++.so")
    with pytest.raises(factorial.ProtocolInvalid, match="effective LD_PRELOAD differs"):
        factorial.runtime_binding()


def test_binding_diff_and_terminal_payload_retain_exact_paths():
    expected = {"runtime": {"sys_path": ["a", factorial.TORCH_TEMP_SENTINEL]}, "config": 1}
    actual = {"runtime": {"sys_path": ["a", "/tmp/foreign"]}, "config": 1}
    differences = factorial._key_differences(expected, actual)
    assert differences == [
        {
            "path": "$.runtime.sys_path[1]",
            "expected": factorial.TORCH_TEMP_SENTINEL,
            "actual": "/tmp/foreign",
        }
    ]
    error = factorial.BindingInvariantError(
        "binding changed",
        {"phase": "exit", "differences": differences},
    )
    payload = factorial.bounded_failure(error, stage="worker")
    assert payload["failure"]["binding_invariant"] == error.diagnostic


def test_worker_loader_requires_passing_identical_binding_receipts(tmp_path):
    path = tmp_path / "worker.json"
    path.write_text(
        json.dumps(
            {
                "artifact_type": "compact_occupancy_refinement_factorial_iter3_worker_v1",
                "status": "PASS",
                "binding_validation": {
                    "passed": True,
                    "differences": [],
                    "entry": {"semantic_sha256": "a" * 64},
                    "exit": {"semantic_sha256": "b" * 64},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(factorial.BindingInvariantError, match="binding-validation") as captured:
        factorial._load_worker_record(path)
    assert captured.value.diagnostic["reported_differences"] == []
    assert captured.value.diagnostic["receipt_differences"] == [
        {
            "path": "$.semantic_sha256",
            "expected": "a" * 64,
            "actual": "b" * 64,
        }
    ]


def test_rejected_return_zero_worker_retains_receipt_hash_diff_and_process_tails(tmp_path):
    path = tmp_path / "worker.json"
    path.write_text(
        json.dumps(
            {
                "artifact_type": "compact_occupancy_refinement_factorial_iter3_worker_v1",
                "status": "PASS",
                "binding_validation": {
                    "passed": True,
                    "differences": [],
                    "entry": {"semantic_sha256": "a" * 64},
                    "exit": {"semantic_sha256": "b" * 64},
                },
            }
        ),
        encoding="utf-8",
    )
    process = {
        "returncode": 0,
        "stdout_tail": "return-zero stdout evidence",
        "stderr_tail": "return-zero stderr evidence",
    }
    with pytest.raises(factorial.WorkerProcessError) as captured:
        factorial._load_worker_record_with_evidence(
            path,
            seed=factorial.TEST_TRAIN_SEEDS[0],
            arm="A",
            evaluation_seed=factorial.TEST_EVALUATION_SEEDS[0],
            subprocess_record=process,
        )
    terminal = factorial.bounded_failure(captured.value, stage="focused_parent_worker_load")
    diagnostic = terminal["failure"]["worker_process"]
    assert diagnostic["process"] == process
    assert diagnostic["worker_artifact"]["status"] == "PASS"
    assert diagnostic["worker_artifact"]["sha256"] == factorial.sha256_file(path)
    assert diagnostic["receipt_failure"]["binding_invariant"]["receipt_differences"] == [
        {
            "path": "$.semantic_sha256",
            "expected": "a" * 64,
            "actual": "b" * 64,
        }
    ]


def test_review_gate_requires_one_exact_standalone_pass_line(tmp_path, monkeypatch):
    review = tmp_path / "review.md"
    monkeypatch.setattr(factorial, "IMPLEMENTATION_REVIEW", review)
    monkeypatch.setattr(factorial, "reviewed_source_hashes", lambda: ({}, "a" * 64))
    aggregate = f"{factorial.REVIEW_AGGREGATE_PREFIX}{'a' * 64}"
    review.write_text(f"Verdict: FAIL\nThe string Verdict: PASS appears in prose.\n{aggregate}\n")
    assert factorial._review_passed() is False
    review.write_text(f"Verdict: PASS\n{aggregate}\n")
    assert factorial._review_passed() is True
    review.write_text(f"Verdict: PASS\nVerdict: PASS\n{aggregate}\n")
    assert factorial._review_passed() is False
    review.write_text(f"Verdict: PASS\n{factorial.REVIEW_AGGREGATE_PREFIX}{'b' * 64}\n")
    assert factorial._review_passed() is False
