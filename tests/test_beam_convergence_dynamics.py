"""Focused contracts for the convergence-dynamics research harness."""

import json

import torch
from benchmarks.beam_convergence_dynamics import _chamfer


def test_empty_chamfer_is_strict_json_safe() -> None:
    mean, p90 = _chamfer(torch.empty(0, 3), torch.zeros(1, 3))

    assert mean is None
    assert p90 is None
    assert json.dumps({"mean": mean, "p90": p90}, allow_nan=False)


def test_nonempty_chamfer_remains_finite() -> None:
    mean, p90 = _chamfer(
        torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
    )

    assert mean == 1.0
    assert p90 == 1.0
