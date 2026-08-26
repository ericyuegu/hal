"""Tests for the single-device Muon optimizer."""

import copy
import math

import pytest
import torch

from hal.training import muon


def _parameters() -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    generator = torch.Generator().manual_seed(7)
    muon_parameters = [
        torch.nn.Parameter(torch.randn(shape, generator=generator)) for shape in ((4, 4), (4, 4), (6, 3), (6, 3))
    ]
    adam_parameters = [torch.nn.Parameter(torch.randn(shape, generator=generator)) for shape in ((4,), (2, 3))]
    return muon_parameters, adam_parameters


def _optimizer(
    muon_parameters: list[torch.nn.Parameter],
    adam_parameters: list[torch.nn.Parameter],
) -> muon.SingleDeviceMuonWithAuxAdam:
    return muon.SingleDeviceMuonWithAuxAdam(
        [
            {
                "params": muon_parameters,
                "lr": 0.02,
                "momentum": 0.95,
                "weight_decay": 0.01,
                "use_muon": True,
            },
            {
                "params": adam_parameters,
                "lr": 3e-4,
                "betas": (0.9, 0.95),
                "eps": 1e-10,
                "weight_decay": 0.1,
                "use_muon": False,
            },
        ]
    )


@torch.no_grad()
def _reference_step(optimizer: muon.SingleDeviceMuonWithAuxAdam) -> None:
    """Run the scalar-loop implementation used by existing checkpoints."""
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            state = optimizer.state[parameter]
            if group["use_muon"]:
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                update = muon.muon_update(
                    parameter.grad,
                    state["momentum_buffer"],
                    beta=group["momentum"],
                )
            else:
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    state["step"] = 0
                state["step"] += 1
                update = muon.adam_update(
                    parameter.grad,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["step"],
                    group["betas"],
                    group["eps"],
                )
            parameter.mul_(1 - group["lr"] * group["weight_decay"])
            parameter.add_(update.reshape(parameter.shape), alpha=-group["lr"])


def test_batched_optimizer_matches_scalar_reference_and_state_schema() -> None:
    reference_parameters = _parameters()
    batched_parameters = tuple(
        [torch.nn.Parameter(parameter.detach().clone()) for parameter in group] for group in reference_parameters
    )
    reference = _optimizer(*reference_parameters)
    batched = _optimizer(*batched_parameters)
    generator = torch.Generator().manual_seed(11)

    for _ in range(3):
        for reference_parameter, batched_parameter in zip(
            (*reference_parameters[0], *reference_parameters[1]),
            (*batched_parameters[0], *batched_parameters[1]),
            strict=True,
        ):
            gradient = torch.randn(reference_parameter.shape, generator=generator)
            reference_parameter.grad = gradient.clone()
            batched_parameter.grad = gradient.clone()
        _reference_step(reference)
        batched.step()

    for reference_parameter, batched_parameter in zip(
        (*reference_parameters[0], *reference_parameters[1]),
        (*batched_parameters[0], *batched_parameters[1]),
        strict=True,
    ):
        torch.testing.assert_close(batched_parameter, reference_parameter, rtol=2e-4, atol=2e-4)
        reference_state = reference.state[reference_parameter]
        batched_state = batched.state[batched_parameter]
        assert reference_state.keys() == batched_state.keys()
        for name in reference_state:
            if isinstance(reference_state[name], torch.Tensor):
                torch.testing.assert_close(batched_state[name], reference_state[name])
            else:
                assert batched_state[name] == reference_state[name]


def test_batched_optimizer_state_dict_resumes_exactly() -> None:
    first_parameters = _parameters()
    first = _optimizer(*first_parameters)
    for parameter in (*first_parameters[0], *first_parameters[1]):
        parameter.grad = torch.ones_like(parameter)
    first.step()

    resumed_parameters = tuple(
        [torch.nn.Parameter(parameter.detach().clone()) for parameter in group] for group in first_parameters
    )
    resumed = _optimizer(*resumed_parameters)
    resumed.load_state_dict(copy.deepcopy(first.state_dict()))
    for first_parameter, resumed_parameter in zip(
        (*first_parameters[0], *first_parameters[1]),
        (*resumed_parameters[0], *resumed_parameters[1]),
        strict=True,
    ):
        gradient = torch.full_like(first_parameter, 0.25)
        first_parameter.grad = gradient.clone()
        resumed_parameter.grad = gradient.clone()

    first.step()
    resumed.step()

    for first_parameter, resumed_parameter in zip(
        (*first_parameters[0], *first_parameters[1]),
        (*resumed_parameters[0], *resumed_parameters[1]),
        strict=True,
    ):
        torch.testing.assert_close(resumed_parameter, first_parameter)


def test_stable_adamw_clips_each_tensor_from_the_updated_second_moment() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = muon.SingleDeviceMuonWithAuxAdam(
        [
            {
                "params": [parameter],
                "lr": 0.1,
                "betas": (0.9, 0.95),
                "eps": 1e-10,
                "weight_decay": 0.1,
                "update_clip_threshold": 1.0,
                "use_muon": False,
            }
        ]
    )
    optimizer.set_adam_diagnostic_parameters({"test_tensor": parameter})
    state = optimizer.state[parameter]
    state["exp_avg"] = torch.zeros_like(parameter)
    state["exp_avg_sq"] = torch.full_like(parameter, 1e-6)
    state["step"] = 100
    parameter.grad = torch.ones_like(parameter)

    before = parameter.detach().clone()
    optimizer.step()

    updated_second_moment = 0.95e-6 + 0.05
    updated_bias_correction = 1 - 0.95**101
    rms_ratio = math.sqrt(1 / (updated_second_moment / updated_bias_correction))
    clip_factor = 1 / rms_ratio
    first_moment = 0.1
    adam_direction = (first_moment / (1 - 0.9**101)) / (
        math.sqrt(updated_second_moment / updated_bias_correction) + 1e-10
    )
    expected = before - 0.1 * clip_factor * (adam_direction + 0.1 * before)
    diagnostics = optimizer.last_adam_diagnostics
    previous_rms_ratio = math.sqrt((1 - 0.95**100) / 1e-6)

    torch.testing.assert_close(parameter, expected)
    assert diagnostics["optimizer/test_tensor/rms_ratio_previous"] == pytest.approx(previous_rms_ratio)
    assert diagnostics["optimizer/test_tensor/rms_ratio_updated"] == pytest.approx(rms_ratio)
    assert diagnostics["optimizer/test_tensor/update_clip_factor"] == pytest.approx(clip_factor)
    assert diagnostics["optimizer/test_tensor/update_clip_active"] == 1


def test_loading_old_optimizer_state_retains_configured_update_clipping() -> None:
    source_parameters = _parameters()
    source = _optimizer(*source_parameters)
    state_dict = copy.deepcopy(source.state_dict())
    assert all("update_clip_threshold" not in group for group in state_dict["param_groups"])

    target_parameters = _parameters()
    target = muon.SingleDeviceMuonWithAuxAdam(
        [
            {
                "params": target_parameters[0],
                "lr": 0.02,
                "momentum": 0.95,
                "weight_decay": 0.01,
                "use_muon": True,
            },
            {
                "params": target_parameters[1],
                "lr": 3e-4,
                "betas": (0.9, 0.95),
                "eps": 1e-10,
                "weight_decay": 0.1,
                "update_clip_threshold": 1.0,
                "use_muon": False,
            },
        ]
    )
    target.load_state_dict(state_dict)

    adam_group = next(group for group in target.param_groups if not group["use_muon"])
    assert adam_group["update_clip_threshold"] == 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for batched Muon parity")
def test_batched_muon_matches_scalar_cuda_update() -> None:
    reference_muon, reference_adam = _parameters()
    reference_muon = [torch.nn.Parameter(parameter.detach().cuda()) for parameter in reference_muon]
    reference_adam = [torch.nn.Parameter(parameter.detach().cuda()) for parameter in reference_adam]
    batched_muon = [torch.nn.Parameter(parameter.detach().clone()) for parameter in reference_muon]
    batched_adam = [torch.nn.Parameter(parameter.detach().clone()) for parameter in reference_adam]
    reference = _optimizer(reference_muon, reference_adam)
    batched = _optimizer(batched_muon, batched_adam)
    generator = torch.Generator().manual_seed(23)

    for reference_parameter, batched_parameter in zip(
        (*reference_muon, *reference_adam),
        (*batched_muon, *batched_adam),
        strict=True,
    ):
        gradient = torch.randn(reference_parameter.shape, generator=generator).cuda()
        reference_parameter.grad = gradient.clone()
        batched_parameter.grad = gradient.clone()
    _reference_step(reference)
    batched.step()
    torch.cuda.synchronize()

    for reference_parameter, batched_parameter in zip(
        (*reference_muon, *reference_adam),
        (*batched_muon, *batched_adam),
        strict=True,
    ):
        torch.testing.assert_close(batched_parameter, reference_parameter, rtol=2e-4, atol=2e-4)
