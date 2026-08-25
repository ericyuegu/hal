"""Tests for the single-device Muon optimizer."""

import copy

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
