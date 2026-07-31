"""Contract tests for the additive native bundle-production coordinator."""

from __future__ import annotations

import copy

import pytest
from scripts.experiments import prepare_additive_native_bundles as production


def test_registered_bundle_production_freezes_full_resolution_native_additive_fit() -> None:
    task = production._task(production.DEFAULT_TASK)
    config = production._production_config(task)

    assert config["downscale"] == 1
    assert config["per_view_process_isolation"] is True
    assert config["field_semantics"] == {
        "provider": "native",
        "blend_mode": "additive",
        "sigma_cutoff": 12.0**0.5,
        "support_fade_alpha": 0.0,
        "aa_dilation": 0.0,
    }
    assert config["fit_config"]["n_gaussians"] == 640
    assert config["fit_config"]["iterations"] == 100
    assert config["fit_config"]["native_renderer"] == "cuda"
    assert len(production._view_ids(task, "frame_00008")) == 26
    assert production._view_seed(config, dataset_index=0, view_index=0) == 30073000
    assert production._view_seed(config, dataset_index=1, view_index=25) == 30073125


def test_bundle_production_rejects_semantic_or_effective_config_drift() -> None:
    task = production._task(production.DEFAULT_TASK)
    changed = copy.deepcopy(task)
    changed["frozen_configuration"][production.PRODUCTION_KEY]["field_semantics"]["blend_mode"] = (
        "normalized"
    )
    with pytest.raises(ValueError, match="field_semantics"):
        production._production_config(changed)

    changed = copy.deepcopy(task)
    del changed["frozen_configuration"][production.PRODUCTION_KEY]["fit_config"]["row_chunk"]
    with pytest.raises(ValueError, match="every effective"):
        production._production_config(changed)


def test_bundle_paths_are_bound_to_additive_directories_and_sidecars() -> None:
    task = production._task(production.DEFAULT_TASK)
    for dataset in production._datasets(task):
        compact_manifest = production._repository_path(
            dataset["compact_manifest"],
            label="compact_manifest",
        )
        production_manifest = production._repository_path(
            dataset["production_manifest"],
            label="production_manifest",
        )
        assert compact_manifest.parent.name == "gaussians2d_additive"
        assert production_manifest.parent == compact_manifest.parent
        assert production_manifest.name == "production_manifest.json"
