"""Scene containers and dataset loading (synthetic, COLMAP, calibrated captures).

The RGB-free reconstruction-input path must not import Pillow or calibrated scene loaders as a
package side effect.  Keep those legacy public attributes lazy so compact Stage-2/3 processes can
install their source-image denial boundary before any RGB-capable module is loaded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rtgs.data.reconstruction_inputs import (
    BundleArchiveStats,
    BundleLoadLimits,
    NpzArchiveStats,
    ReconstructionInputs,
)

if TYPE_CHECKING:
    from rtgs.data.calibrated import load_calibrated_scene
    from rtgs.data.compact_views import (
        CompactDataset,
        CompactView,
        CompactViewTooLarge,
        PackedAlpha,
    )
    from rtgs.data.field_inputs import SceneFits
    from rtgs.data.scene import SceneData

__all__ = [
    "BundleArchiveStats",
    "BundleLoadLimits",
    "COMPACT_VIEW_BYTE_CAP",
    "CompactDataset",
    "CompactView",
    "CompactViewTooLarge",
    "NpzArchiveStats",
    "PackedAlpha",
    "ReconstructionInputs",
    "SceneFits",
    "SceneData",
    "load_calibrated_scene",
    "save_compact_view",
    "write_compact_dataset_manifest",
]


def __getattr__(name: str) -> Any:
    if name in {
        "COMPACT_VIEW_BYTE_CAP",
        "CompactDataset",
        "CompactView",
        "CompactViewTooLarge",
        "PackedAlpha",
        "save_compact_view",
        "write_compact_dataset_manifest",
    }:
        from rtgs.data import compact_views

        return getattr(compact_views, name)
    if name == "SceneData":
        from rtgs.data.scene import SceneData

        return SceneData
    if name == "SceneFits":
        from rtgs.data.field_inputs import SceneFits

        return SceneFits
    if name == "load_calibrated_scene":
        from rtgs.data.calibrated import load_calibrated_scene

        return load_calibrated_scene
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
