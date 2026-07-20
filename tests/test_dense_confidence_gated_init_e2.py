"""Protocol and CPU-smoke tests for the dense-confidence E2 harness."""

from __future__ import annotations

import pytest
from benchmarks import dense_confidence_gated_init_e2 as e2


def test_e2_config_matches_frozen_schedule():
    config = e2.frozen_config(e2.MAIN_SEED)

    assert config.iterations == 300
    assert config.rasterizer == "gsplat"
    assert config.device == "cuda"
    assert config.density_strategy == "gsplat-default"
    assert config.density.max_gaussians == 2319
    assert config.density.start_iter == 25
    assert config.density.stop_iter == 275
    assert config.density.every == 25
    assert config.density.absgrad is True
    assert config.eval_every == 50
    assert config.sh_degree_interval == 75


def test_e2_split_keeps_validation_and_heldout_out_of_training():
    assert e2.VALIDATION_VIEW not in e2.TRAIN_VIEWS
    assert e2.HELDOUT_VIEW not in e2.TRAIN_VIEWS
    assert e2.VALIDATION_VIEW != e2.HELDOUT_VIEW
    assert [name for name, _, _ in e2.EXECUTIONS] == [
        "topk",
        "dense_all",
        "easy_only",
        "topk_repeat",
    ]


def _decision_record(psnr, count, seconds):
    return {
        "heldout_final_rgb": {"psnr_fg": psnr},
        "final_count": count,
        "native_optimization_seconds": seconds,
    }


def test_e2_decision_uses_tighter_control_repeat_envelope():
    executions = {
        "topk": _decision_record(20.00, 2000, 100.0),
        "topk_repeat": _decision_record(19.97, 2000, 101.0),
        "dense_all": _decision_record(20.20, 2319, 120.0),
        "easy_only": _decision_record(20.18, 2200, 110.0),
    }

    decision = e2.decide_e2(executions)

    assert decision["control_repeat_envelope_db"] == pytest.approx(0.03)
    assert decision["allowed_quality_deficit_db"] == pytest.approx(0.03)
    assert decision["best_competitor"] == "dense_all"
    assert decision["easy_only_wins"] is True


def test_e2_decision_rejects_quality_count_or_time_failure():
    base = {
        "topk": _decision_record(20.00, 2000, 100.0),
        "topk_repeat": _decision_record(19.80, 2000, 101.0),
        "dense_all": _decision_record(20.20, 2319, 120.0),
        "easy_only": _decision_record(20.09, 2200, 110.0),
    }
    assert e2.decide_e2(base)["easy_only_wins"] is False

    base["easy_only"] = _decision_record(20.20, 2400, 110.0)
    assert e2.decide_e2(base)["easy_only_wins"] is False

    base["easy_only"] = _decision_record(20.20, 2200, 121.0)
    assert e2.decide_e2(base)["easy_only_wins"] is False


def test_e2_cpu_reference_smoke(tmp_path):
    result = e2.run_cpu_smoke(tmp_path)

    assert set(result) == {"easy_only", "topk", "dense_all"}
    assert [result[name]["initial_count"] for name in ("easy_only", "topk", "dense_all")] == [
        2,
        4,
        6,
    ]
    assert all(record["final_count"] <= 6 for record in result.values())
    assert (tmp_path / "cpu_smoke.json").is_file()
