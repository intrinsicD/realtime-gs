"""Scene containers and dataset loading (synthetic, COLMAP, calibrated captures)."""

from rtgs.data.calibrated import load_calibrated_scene
from rtgs.data.scene import SceneData

__all__ = ["SceneData", "load_calibrated_scene"]
