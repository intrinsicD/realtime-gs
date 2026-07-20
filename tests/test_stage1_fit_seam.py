"""Focused, outcome-free tests for the native Stage-1 fit research seam."""

from __future__ import annotations

import builtins
from dataclasses import replace
from typing import Any

import pytest
import torch

from rtgs.core.gaussians2d import Gaussians2D
from rtgs.core.metrics import masked_psnr, psnr
from rtgs.image2gs.fit import (
    FitConfig,
    NativeFitDiagnostic,
    NonFiniteRawParameterError,
    _validate_raw_parameters,
    fit_image,
    fit_image_from_initialization,
    init_gaussians_2d,
)
from rtgs.image2gs.renderer2d import render_gaussians_2d

# The function below is a test-local transcription of the native fitter frozen at this hash.
_LEGACY_FIT_SHA256 = "2a9b76d41e83cc444fa98b3a0f3aa45eb8b6032806fa3d899377acfd98257e18"
_MIN_DIAG = 0.3


def _softplus_inv(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def _legacy_crop_to_mask(
    image: torch.Tensor, mask: torch.Tensor, margin_fraction: float = 0.05
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mask.shape != image.shape[:2]:
        raise ValueError("mask size does not match image")
    foreground = torch.nonzero(mask > 0.5)
    if foreground.numel() == 0:
        raise ValueError("cannot fit an empty foreground mask")
    height, width = image.shape[:2]
    margin = max(2, round(max(height, width) * margin_fraction))
    y0 = max(0, int(foreground[:, 0].min()) - margin)
    y1 = min(height, int(foreground[:, 0].max()) + 1 + margin)
    x0 = max(0, int(foreground[:, 1].min()) - margin)
    x1 = min(width, int(foreground[:, 1].max()) + 1 + margin)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1], image.new_tensor([x0, y0])


def _legacy_fit_image(
    image: torch.Tensor,
    config: FitConfig,
    seed: int,
    mask: torch.Tensor | None = None,
) -> tuple[Gaussians2D, dict[str, Any], dict[str, Any]]:
    """Frozen pre-change algorithm plus detached observations that do not affect arithmetic."""
    assert len(_LEGACY_FIT_SHA256) == 64
    xy_offset = image.new_zeros(2)
    if mask is not None:
        image, mask, xy_offset = _legacy_crop_to_mask(image, mask)
    h, w = image.shape[:2]
    target = image
    if mask is not None:
        if mask.shape != image.shape[:2]:
            raise ValueError("mask size does not match image")
        target = image * mask.to(image).clamp(0, 1)[..., None]
    gen = torch.Generator(device=image.device).manual_seed(seed)
    rng_before = gen.get_state().clone()
    g0 = init_gaussians_2d(target, config.n_gaussians, config.grad_init_mix, gen)
    rng_after = gen.get_state().clone()

    wh = image.new_tensor([w, h])
    xy_raw = torch.logit((g0.xy / wh).clamp(1e-4, 1 - 1e-4))
    diag_raw = _softplus_inv(g0.chol[:, [0, 2]] - _MIN_DIAG)
    off_raw = g0.chol[:, 1].clone()
    color_raw = torch.logit(g0.color.clamp(1e-3, 1 - 1e-3))
    weight_raw = torch.logit(g0.weight.clamp(1e-3, 1 - 1e-3))
    names = ("xy_raw", "diag_raw", "off_raw", "color_raw", "weight_raw")
    params = [xy_raw, diag_raw, off_raw, color_raw, weight_raw]
    for parameter in params:
        parameter.requires_grad_(True)

    def build() -> Gaussians2D:
        diag = torch.nn.functional.softplus(diag_raw) + _MIN_DIAG
        chol = torch.stack([diag[:, 0], off_raw, diag[:, 1]], dim=-1)
        return Gaussians2D(
            xy=torch.sigmoid(xy_raw) * wh,
            chol=chol,
            color=torch.sigmoid(color_raw),
            weight=torch.sigmoid(weight_raw),
        )

    opt = torch.optim.Adam(params, lr=config.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=config.iterations, eta_min=config.lr * 0.1
    )
    trace: dict[str, Any] = {
        "initial_gaussians": g0.detach(),
        "initializer_rng_state_before": rng_before,
        "initializer_rng_state_after": rng_after,
        "initial_raw": {
            name: parameter.detach().clone() for name, parameter in zip(names, params, strict=True)
        },
        "optimizer_param_names": names,
        "lr_used": [],
        "loss": [],
    }
    history: dict[str, Any] = {"psnr": [], "stopped_iter": config.iterations - 1}
    best_psnr = -float("inf")
    stale_checks = 0
    for it in range(config.iterations):
        opt.zero_grad()
        rendered = render_gaussians_2d(build(), h, w, row_chunk=config.row_chunk)
        if mask is None:
            loss = torch.nn.functional.mse_loss(rendered, target)
        else:
            weights = 0.1 + 0.9 * mask.to(rendered).clamp(0, 1)
            loss = (((rendered - target) ** 2) * weights[..., None]).mean()
        loss.backward()
        trace["lr_used"].append(float(opt.param_groups[0]["lr"]))
        trace["loss"].append(loss.detach().clone())
        opt.step()
        sched.step()
        if it % config.log_every == 0 or it == config.iterations - 1:
            with torch.no_grad():
                history["psnr"].append((it, psnr(rendered.clamp(0, 1), target)))
        if config.convergence_patience and (it + 1) % config.convergence_check_every == 0:
            with torch.no_grad():
                cur = psnr(rendered.clamp(0, 1), target)
            if cur > best_psnr + config.convergence_tol:
                best_psnr = cur
                stale_checks = 0
            else:
                stale_checks += 1
                if stale_checks >= config.convergence_patience:
                    history["stopped_iter"] = it
                    break

    result = build().detach()
    with torch.no_grad():
        final = render_gaussians_2d(result, h, w, row_chunk=config.row_chunk)
        history["final_psnr_full"] = psnr(final.clamp(0, 1), target)
        history["final_psnr"] = (
            history["final_psnr_full"]
            if mask is None
            else masked_psnr(final.clamp(0, 1), image, mask)
        )
    result.xy += xy_offset
    return result, history, trace


def _assert_gaussians_bit_exact(left: Gaussians2D, right: Gaussians2D) -> None:
    for field in ("xy", "chol", "color", "weight"):
        assert torch.equal(getattr(left, field), getattr(right, field)), field


def _tiny_case(masked: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    image = torch.linspace(0.0, 1.0, 12 * 14 * 3, dtype=torch.float32).reshape(12, 14, 3)
    if not masked:
        return image, None
    mask = torch.zeros(12, 14)
    mask[3:9, 5:11] = 1.0
    return image, mask


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("explicit_current", [False, True])
def test_current_default_and_explicit_paths_are_bit_exact_to_legacy(
    masked: bool, explicit_current: bool
) -> None:
    image, mask = _tiny_case(masked)
    config = FitConfig(n_gaussians=7, iterations=3, lr=0.01, row_chunk=8, log_every=1)
    if explicit_current:
        config = replace(config, appearance_parameterization="weight_color_9p")
    legacy_g, legacy_history, legacy_trace = _legacy_fit_image(image, config, 193, mask)
    plain_g, plain_history = fit_image(image, config, seed=193, mask=mask)
    snapshots: list[NativeFitDiagnostic] = []
    current_g, current_history = fit_image(
        image,
        config,
        seed=193,
        mask=mask,
        diagnostic_callback=snapshots.append,
        diagnostic_steps=(0, 1, 3),
    )

    _assert_gaussians_bit_exact(plain_g, legacy_g)
    assert plain_history == legacy_history
    _assert_gaussians_bit_exact(current_g, legacy_g)
    assert current_history == legacy_history
    initial = snapshots[0]
    assert (initial.event, initial.step) == ("initial", 0)
    assert initial.initial_gaussians is not None
    _assert_gaussians_bit_exact(initial.initial_gaussians, legacy_trace["initial_gaussians"])
    assert torch.equal(
        initial.initializer_rng_state_before, legacy_trace["initializer_rng_state_before"]
    )
    assert torch.equal(
        initial.initializer_rng_state_after, legacy_trace["initializer_rng_state_after"]
    )
    assert initial.optimizer_param_names == legacy_trace["optimizer_param_names"]
    for name, expected in legacy_trace["initial_raw"].items():
        assert torch.equal(initial.raw_parameters[name], expected), name
    pre = [snapshot for snapshot in snapshots if snapshot.event == "pre_update"]
    assert [snapshot.lr_used for snapshot in pre] == legacy_trace["lr_used"]
    assert all(
        torch.equal(snapshot.loss, expected)
        for snapshot, expected in zip(pre, legacy_trace["loss"], strict=True)
    )
    assert [(s.event, s.step) for s in snapshots] == [
        ("initial", 0),
        ("pre_update", 0),
        ("post_update", 1),
        ("checkpoint", 1),
        ("pre_update", 1),
        ("post_update", 2),
        ("pre_update", 2),
        ("post_update", 3),
        ("checkpoint", 3),
    ]


def _probe_amplitude_gradient(snapshot: NativeFitDiagnostic) -> torch.Tensor:
    amplitude = (snapshot.gaussians.weight[:, None] * snapshot.gaussians.color).requires_grad_()
    probe = Gaussians2D(
        xy=snapshot.gaussians.xy,
        chol=snapshot.gaussians.chol,
        color=amplitude,
        weight=torch.ones_like(snapshot.gaussians.weight),
    )
    rendered = render_gaussians_2d(
        probe, snapshot.target.shape[0], snapshot.target.shape[1], row_chunk=8
    )
    loss = torch.nn.functional.mse_loss(rendered, snapshot.target)
    return torch.autograd.grad(loss, amplitude)[0]


def _assert_first_adam_update(
    snapshot_pre: NativeFitDiagnostic, snapshot_post: NativeFitDiagnostic
):
    beta1, beta2 = snapshot_post.optimizer_defaults["betas"]
    eps = snapshot_post.optimizer_defaults["eps"]
    assert snapshot_pre.lr_used is not None
    for name in snapshot_pre.optimizer_param_names:
        state = snapshot_post.optimizer_state[name]
        step = int(state["step"].item())
        assert step == 1
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        step_size = snapshot_pre.lr_used / (1.0 - beta1**step)
        denom = exp_avg_sq.sqrt() / (1.0 - beta2**step) ** 0.5 + eps
        expected = snapshot_pre.raw_parameters[name] - step_size * exp_avg / denom
        assert torch.allclose(snapshot_post.raw_parameters[name], expected, atol=2e-7, rtol=2e-5)


def test_candidate_common_forward_freeze_chain_rule_and_adam_state() -> None:
    image, _ = _tiny_case(False)
    generator = torch.Generator().manual_seed(41)
    g0 = init_gaussians_2d(image, 6, grad_mix=0.7, generator=generator)
    traces: dict[str, list[NativeFitDiagnostic]] = {}
    results: dict[str, Gaussians2D] = {}
    for arm in ("weight_color_9p", "unit_weight_bounded_8p"):
        snapshots: list[NativeFitDiagnostic] = []
        config = FitConfig(
            n_gaussians=6,
            iterations=2,
            row_chunk=8,
            log_every=1,
            appearance_parameterization=arm,
            freeze_geometry=True,
        )
        results[arm], _ = fit_image_from_initialization(
            image,
            g0,
            config,
            diagnostic_callback=snapshots.append,
            diagnostic_steps=(0, 1, 2),
        )
        traces[arm] = snapshots

    current_initial = traces["weight_color_9p"][0]
    candidate_initial = traces["unit_weight_bounded_8p"][0]
    for name in ("xy_raw", "diag_raw", "off_raw"):
        assert torch.equal(
            current_initial.raw_parameters[name], candidate_initial.raw_parameters[name]
        )
    assert torch.equal(current_initial.gaussians.xy, candidate_initial.gaussians.xy)
    assert torch.equal(current_initial.gaussians.chol, candidate_initial.gaussians.chol)
    expected_amplitude = torch.sigmoid(current_initial.raw_parameters["weight_raw"])[
        :, None
    ] * torch.sigmoid(current_initial.raw_parameters["color_raw"])
    assert torch.equal(
        candidate_initial.raw_parameters["amplitude_raw"], torch.logit(expected_amplitude)
    )
    assert torch.allclose(
        candidate_initial.gaussians.color, expected_amplitude, atol=1e-7, rtol=1e-6
    )
    assert candidate_initial.raw_parameters["amplitude_raw"].shape == (g0.n, 3)
    assert candidate_initial.raw_parameters["amplitude_raw"].numel() == 3 * g0.n

    for arm, snapshots in traces.items():
        appearance_names = (
            ("color_raw", "weight_raw") if arm == "weight_color_9p" else ("amplitude_raw",)
        )
        assert snapshots[0].optimizer_param_names == appearance_names
        initial_geometry = {
            name: snapshots[0].raw_parameters[name].clone()
            for name in ("xy_raw", "diag_raw", "off_raw")
        }
        initial_built_geometry = {
            "xy": snapshots[0].gaussians.xy.clone(),
            "chol": snapshots[0].gaussians.chol.clone(),
        }
        for snapshot in snapshots:
            if arm == "unit_weight_bounded_8p":
                assert torch.equal(
                    snapshot.gaussians.weight, torch.ones_like(snapshot.gaussians.weight)
                )
            for name, expected in initial_geometry.items():
                assert torch.equal(snapshot.raw_parameters[name], expected)
                assert snapshot.gradients[name] is None
            for name, expected in initial_built_geometry.items():
                assert torch.equal(getattr(snapshot.gaussians, name), expected)
            if arm != "weight_color_9p":
                assert "weight_raw" not in snapshot.optimizer_state
        _assert_first_adam_update(snapshots[1], snapshots[2])

    current_pre = traces["weight_color_9p"][1]
    grad_a = _probe_amplitude_gradient(current_pre)
    weight = current_pre.gaussians.weight[:, None]
    color = current_pre.gaussians.color
    expected_color_grad = grad_a * weight * color * (1.0 - color)
    expected_weight_grad = (grad_a * weight * (1.0 - weight) * color).sum(dim=1)
    assert torch.allclose(
        current_pre.gradients["color_raw"], expected_color_grad, atol=2e-6, rtol=2e-5
    )
    assert torch.allclose(
        current_pre.gradients["weight_raw"], expected_weight_grad, atol=2e-6, rtol=2e-5
    )

    candidate_pre = traces["unit_weight_bounded_8p"][1]
    grad_a = _probe_amplitude_gradient(candidate_pre)
    amplitude = candidate_pre.gaussians.color
    expected_amplitude_grad = grad_a * amplitude * (1.0 - amplitude)
    assert torch.allclose(
        candidate_pre.gradients["amplitude_raw"],
        expected_amplitude_grad,
        atol=2e-6,
        rtol=2e-5,
    )
    for field in ("xy", "chol"):
        assert torch.equal(
            getattr(results["weight_color_9p"], field),
            getattr(results["unit_weight_bounded_8p"], field),
        )
    assert torch.equal(
        results["unit_weight_bounded_8p"].weight,
        torch.ones_like(results["unit_weight_bounded_8p"].weight),
    )


def _mutate_nested(value: Any) -> None:
    if isinstance(value, torch.Tensor):
        value.fill_(123)
    elif isinstance(value, dict):
        for item in list(value.values()):
            _mutate_nested(item)
        value.clear()
    elif isinstance(value, (list, tuple)):
        for item in value:
            _mutate_nested(item)


def test_diagnostic_callback_isolated_from_fit_and_cpu_rng() -> None:
    image, _ = _tiny_case(False)
    config = FitConfig(n_gaussians=5, iterations=2, row_chunk=8, log_every=1)
    baseline, baseline_history = fit_image(image, config, seed=77)
    torch.manual_seed(991)
    rng_before = torch.random.get_rng_state().clone()

    def malicious(snapshot: NativeFitDiagnostic) -> None:
        if snapshot.initial_gaussians is not None:
            for field in ("xy", "chol", "color", "weight"):
                getattr(snapshot.initial_gaussians, field).fill_(17)
        if snapshot.initializer_rng_state_before is not None:
            snapshot.initializer_rng_state_before.zero_()
        if snapshot.initializer_rng_state_after is not None:
            snapshot.initializer_rng_state_after.zero_()
        for value in snapshot.raw_parameters.values():
            value.fill_(float("nan"))
        for value in snapshot.gradients.values():
            if value is not None:
                value.zero_()
        for field in ("xy", "chol", "color", "weight"):
            getattr(snapshot.gaussians, field).fill_(42)
        snapshot.target.zero_()
        if snapshot.rendered is not None:
            snapshot.rendered.zero_()
        if snapshot.loss is not None:
            snapshot.loss.zero_()
        _mutate_nested(snapshot.optimizer_state)
        _mutate_nested(snapshot.optimizer_param_groups)
        _mutate_nested(snapshot.optimizer_defaults)
        _mutate_nested(snapshot.scheduler_state)
        torch.manual_seed(1)
        torch.rand(100)

    observed, observed_history = fit_image(
        image,
        config,
        seed=77,
        diagnostic_callback=malicious,
        diagnostic_steps=(0, 1, 2),
    )
    _assert_gaussians_bit_exact(observed, baseline)
    assert observed_history == baseline_history
    assert torch.equal(torch.random.get_rng_state(), rng_before)


@pytest.mark.parametrize(
    ("field", "column"),
    [("xy", 0), ("chol", 1), ("color", 1), ("weight", None)],
)
@pytest.mark.parametrize(
    "nonfinite",
    [float("nan"), float("inf"), -float("inf")],
    ids=("nan", "positive_inf", "negative_inf"),
)
def test_nonfinite_initial_rows_fail_closed_before_transform_or_optimizer(
    field: str,
    column: int | None,
    nonfinite: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, _ = _tiny_case(False)
    g0 = init_gaussians_2d(image, 4, generator=torch.Generator().manual_seed(5))
    tensor = getattr(g0, field)
    if column is None:
        tensor[2] = nonfinite
    else:
        tensor[2, column] = nonfinite
    monkeypatch.setattr(
        torch,
        "logit",
        lambda *args, **kwargs: pytest.fail(
            "raw transform must not run for an invalid initializer"
        ),
    )
    monkeypatch.setattr(
        torch.optim,
        "Adam",
        lambda *args, **kwargs: pytest.fail("optimizer must not be constructed for invalid input"),
    )
    with pytest.raises(ValueError, match=rf"non-finite initial rows in {field}: \[2\]"):
        fit_image_from_initialization(
            image,
            g0,
            FitConfig(n_gaussians=4, iterations=1, row_chunk=8),
        )


@pytest.mark.parametrize(
    ("field", "column", "value", "message"),
    [
        ("xy", 0, -0.01, "xy rows outside image bounds"),
        ("xy", 0, 14.0, "xy rows outside image bounds"),
        ("chol", 0, 0.3, "Cholesky diagonal must exceed"),
        ("chol", 2, -0.1, "Cholesky diagonal must exceed"),
        ("color", 0, -0.01, r"color rows outside \[0,1\]"),
        ("color", 2, 1.01, r"color rows outside \[0,1\]"),
        ("weight", None, -0.01, r"weight rows outside \[0,1\]"),
        ("weight", None, 1.01, r"weight rows outside \[0,1\]"),
    ],
)
def test_invalid_initial_geometry_and_appearance_preconditions_fail_before_transform(
    field: str,
    column: int | None,
    value: float,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, _ = _tiny_case(False)
    g0 = init_gaussians_2d(image, 4, generator=torch.Generator().manual_seed(7))
    tensor = getattr(g0, field)
    if column is None:
        tensor[1] = value
    else:
        tensor[1, column] = value
    monkeypatch.setattr(
        torch,
        "logit",
        lambda *args, **kwargs: pytest.fail(
            "raw transform must not run for an invalid initializer"
        ),
    )
    with pytest.raises(ValueError, match=message):
        fit_image_from_initialization(
            image,
            g0,
            FitConfig(n_gaussians=4, iterations=1, row_chunk=8),
        )


def test_invalid_initial_xy_shape_fails_before_transform() -> None:
    image, _ = _tiny_case(False)
    g0 = init_gaussians_2d(image, 4, generator=torch.Generator().manual_seed(11))
    g0.xy = torch.cat([g0.xy, torch.zeros(4, 1)], dim=1)
    with pytest.raises(ValueError, match=r"xy shape must be \(4, 2\), got \(4, 3\)"):
        fit_image_from_initialization(
            image,
            g0,
            FitConfig(n_gaussians=4, iterations=1, row_chunk=8),
        )


def test_nonfinite_raw_parameter_error_preserves_structured_detached_evidence() -> None:
    raw = torch.tensor(
        [[0.0, 1.0, 2.0], [float("nan"), 3.0, 4.0], [5.0, float("inf"), -float("inf")]],
        requires_grad=True,
    )
    with pytest.raises(
        NonFiniteRawParameterError,
        match=r"non-finite raw rows in amplitude_raw: \[1, 2\]",
    ) as caught:
        _validate_raw_parameters({"amplitude_raw": raw})

    error = caught.value
    assert isinstance(error, ValueError)
    assert error.parameter_name == "amplitude_raw"
    assert error.row_indices == (1, 2)
    assert error.raw_tensor is not raw
    assert not error.raw_tensor.requires_grad
    assert error.raw_tensor.dtype == raw.dtype
    assert error.raw_tensor.device == raw.device
    assert torch.equal(torch.isnan(error.raw_tensor), torch.isnan(raw))
    assert torch.equal(torch.isposinf(error.raw_tensor), torch.isposinf(raw))
    assert torch.equal(torch.isneginf(error.raw_tensor), torch.isneginf(raw))
    with torch.no_grad():
        raw.zero_()
    assert bool(torch.isnan(error.raw_tensor[1, 0]))
    assert bool(torch.isposinf(error.raw_tensor[2, 1]))
    assert bool(torch.isneginf(error.raw_tensor[2, 2]))


def test_candidate_structsplat_rejected_before_optional_import(monkeypatch) -> None:
    attempted: list[str] = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "rtgs.image2gs.structsplat_backend":
            attempted.append(name)
            raise AssertionError("optional backend import must not occur")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError, match="native-only"):
        fit_image(
            torch.zeros(4, 4, 3),
            FitConfig(
                n_gaussians=2,
                iterations=1,
                backend="structsplat",
                appearance_parameterization="unit_weight_bounded_8p",
            ),
        )
    assert attempted == []
