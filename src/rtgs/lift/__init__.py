"""Stage 2: lift per-view 2D gaussians into a world-space 3D gaussian set.

Variants (see docs/ARCHITECTURE.md): `gradient` (multi-view photometric descent),
`depth` (monocular depth backend), `carve` (voxel consistency carving), plus the
`sfm` and `random` baselines. Register new variants in ``_LIFTERS``.
"""

from rtgs.lift.base import Lifter
from rtgs.lift.baselines import RandomLifter, SfMLifter
from rtgs.lift.carve import CarveLifter
from rtgs.lift.cost_volume import CostVolumeLifter
from rtgs.lift.depth import DepthLifter
from rtgs.lift.gradient import GradientLifter

_LIFTERS = {
    "gradient": GradientLifter,
    "depth": DepthLifter,
    "carve": CarveLifter,
    "cost": CostVolumeLifter,
    "sfm": SfMLifter,
    "random": RandomLifter,
}


def get_lifter(name: str, **kwargs) -> Lifter:
    """Instantiate a registered lifter by name, forwarding keyword arguments."""
    if name not in _LIFTERS:
        raise ValueError(f"unknown lifter '{name}' (available: {sorted(_LIFTERS)})")
    return _LIFTERS[name](**kwargs)


def lifter_names() -> list[str]:
    """Names of all registered lifting variants."""
    return sorted(_LIFTERS)


__all__ = [
    "CarveLifter",
    "CostVolumeLifter",
    "DepthLifter",
    "GradientLifter",
    "Lifter",
    "RandomLifter",
    "SfMLifter",
    "get_lifter",
    "lifter_names",
]
