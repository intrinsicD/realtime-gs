"""Stage 3: dense-image and compact-field 3DGS optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rtgs.optim.compact_trainer import CompactTrainConfig, CompactTrainer

if TYPE_CHECKING:
    from rtgs.optim.trainer import TrainConfig, Trainer, TrainStepControl

__all__ = [
    "CompactTrainConfig",
    "CompactTrainer",
    "TrainConfig",
    "Trainer",
    "TrainStepControl",
]


def __getattr__(name: str) -> Any:
    if name in {"TrainConfig", "Trainer", "TrainStepControl"}:
        from rtgs.optim import trainer

        return getattr(trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
