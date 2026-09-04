import math
from collections.abc import Mapping

import torch
import torch.distributed as dist
from torch import Tensor
from torch import nn


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_matrix_scale(
    d_out: int,
    d_in: int,
    *,
    muon_scale_clamp_min_one: bool = True,
) -> float:
    """Return the post-orthogonalization scale for one logical matrix.

    Historical experiments clamp ``d_out / d_in`` to one. Some parameterizations
    need the unclamped rectangular-matrix rule instead.
    """
    if d_out < 1 or d_in < 1:
        raise ValueError(f"matrix dimensions must be positive, got {(d_out, d_in)}")
    if not isinstance(muon_scale_clamp_min_one, bool):
        raise TypeError("muon_scale_clamp_min_one must be a bool")
    if muon_scale_clamp_min_one:
        return max(1, d_out / d_in) ** 0.5
    return (d_out / d_in) ** 0.5


def _orthogonalize_logical_matrices(
    matrices: Tensor,
    *,
    logical_splits: int,
    muon_scale_clamp_min_one: bool,
    ns_steps: int,
) -> Tensor:
    """Orthogonalize equal row-wise logical matrices, then restore fusion."""
    if not isinstance(logical_splits, int) or isinstance(logical_splits, bool) or logical_splits < 1:
        raise ValueError(f"logical_splits must be a positive integer, got {logical_splits!r}")
    d_out, d_in = matrices.shape[-2:]
    if d_out % logical_splits:
        raise ValueError(f"cannot split {d_out} output rows into {logical_splits} logical matrices")
    if logical_splits == 1:
        orthogonal = zeropower_via_newtonschulz5(matrices, steps=ns_steps)
        orthogonal *= muon_matrix_scale(
            d_out,
            d_in,
            muon_scale_clamp_min_one=muon_scale_clamp_min_one,
        )
        return orthogonal
    logical_d_out = d_out // logical_splits
    logical = matrices.reshape(*matrices.shape[:-2], logical_splits, logical_d_out, d_in)
    logical = zeropower_via_newtonschulz5(logical, steps=ns_steps)
    logical *= muon_matrix_scale(
        logical_d_out,
        d_in,
        muon_scale_clamp_min_one=muon_scale_clamp_min_one,
    )
    return logical.reshape_as(matrices)


def muon_update(
    grad,
    momentum,
    beta=0.95,
    ns_steps=5,
    nesterov=True,
    *,
    muon_scale_clamp_min_one: bool = True,
    logical_splits: int = 1,
):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    return _orthogonalize_logical_matrices(
        update,
        logical_splits=logical_splits,
        muon_scale_clamp_min_one=muon_scale_clamp_min_one,
        ns_steps=ns_steps,
    )


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """

    def __init__(
        self,
        params: list[nn.Parameter],
        lr: float = 0.02,
        weight_decay: float = 0,
        momentum: float = 0.95,
    ) -> None:
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert params and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (
                dist.get_world_size() - len(params) % dist.get_world_size()
            )
            for base_i in range(len(params))[:: dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(
                    params_pad[base_i : base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()]
                )

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """

    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


def _matrix_view(tensor: Tensor) -> Tensor:
    """View a Muon parameter or gradient as a matrix."""
    return tensor.view(len(tensor), -1) if tensor.ndim == 4 else tensor


def _batched_muon_step(
    parameters: list[nn.Parameter],
    momentum_buffers: list[Tensor],
    *,
    lr: float,
    momentum: float,
    weight_decay: float,
    muon_scale_clamp_min_one: bool = True,
    logical_splits: int = 1,
) -> None:
    """Apply one Muon update to a bucket of identically shaped matrices."""
    parameter_tensors: list[Tensor] = list(parameters)
    gradients = [parameter.grad for parameter in parameters]
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("Muon parameters must have gradients before the batched update")
    gradients = [gradient for gradient in gradients if gradient is not None]

    torch._foreach_lerp_(momentum_buffers, gradients, 1 - momentum)
    nesterov_updates = torch._foreach_lerp(gradients, momentum_buffers, momentum)
    matrices = torch.stack([_matrix_view(update) for update in nesterov_updates])
    matrices = _orthogonalize_logical_matrices(
        matrices,
        logical_splits=logical_splits,
        muon_scale_clamp_min_one=muon_scale_clamp_min_one,
        ns_steps=5,
    )
    updates = [update.reshape(parameter.shape) for update, parameter in zip(matrices, parameters, strict=True)]

    if weight_decay:
        torch._foreach_mul_(parameter_tensors, 1 - lr * weight_decay)
    torch._foreach_add_(parameter_tensors, updates, alpha=-lr)


def _foreach_adam_step(
    parameters: list[nn.Parameter],
    *,
    states: list[dict],
    step: int,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    update_clip_threshold: float | None,
    diagnostic_names: Mapping[int, str],
) -> dict[str, Tensor]:
    """Apply AdamW, optionally clipping each tensor's adaptive update."""
    parameter_tensors: list[Tensor] = list(parameters)
    gradients = [parameter.grad for parameter in parameters]
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("Adam parameters must have gradients before the foreach update")
    gradients = [gradient for gradient in gradients if gradient is not None]
    first_moments = [state["exp_avg"] for state in states]
    second_moments = [state["exp_avg_sq"] for state in states]

    previous_step = step - 1
    previous_bias_correction = 1 - betas[1] ** previous_step
    previous_rms_ratios = {
        id(parameter): _adam_rms_ratio(
            gradient,
            second_moment,
            bias_correction=previous_bias_correction,
            eps=eps,
        )
        for parameter, gradient, second_moment in zip(
            parameters,
            gradients,
            second_moments,
            strict=True,
        )
        if id(parameter) in diagnostic_names
    }

    torch._foreach_lerp_(first_moments, gradients, 1 - betas[0])
    squared_gradients = torch._foreach_mul(gradients, gradients)
    torch._foreach_lerp_(second_moments, squared_gradients, 1 - betas[1])
    updates = torch._foreach_div(first_moments, 1 - betas[0] ** step)
    denominators = torch._foreach_div(second_moments, 1 - betas[1] ** step)
    denominators = torch._foreach_sqrt(denominators)
    torch._foreach_add_(denominators, eps)
    updates = torch._foreach_div(updates, denominators)

    tracked = any(id(parameter) in diagnostic_names for parameter in parameters)
    if update_clip_threshold is None and not tracked:
        if weight_decay:
            torch._foreach_mul_(parameter_tensors, 1 - lr * weight_decay)
        torch._foreach_add_(parameter_tensors, updates, alpha=-lr)
        return {}

    second_moment_bias_correction = 1 - betas[1] ** step
    rms_ratios = [
        _adam_rms_ratio(
            gradient,
            second_moment,
            bias_correction=second_moment_bias_correction,
            eps=eps,
        )
        for gradient, second_moment in zip(gradients, second_moments, strict=True)
    ]
    if update_clip_threshold is None:
        clip_factors = [torch.ones((), device=parameter.device) for parameter in parameters]
    else:
        clip_factors = [(update_clip_threshold / rms_ratio).clamp(max=1.0) for rms_ratio in rms_ratios]

    diagnostics: dict[str, Tensor] = {}
    for parameter, gradient, update, rms_ratio, clip_factor in zip(
        parameters,
        gradients,
        updates,
        rms_ratios,
        clip_factors,
        strict=True,
    ):
        name = diagnostic_names.get(id(parameter))
        if name is None:
            continue
        parameter_rms = _tensor_rms(parameter)
        full_update = update + weight_decay * parameter
        prospective_unclipped_update_rms = lr * _tensor_rms(full_update)
        prospective_update_rms = clip_factor * prospective_unclipped_update_rms
        prefix = f"optimizer/{name}"
        diagnostics.update(
            {
                f"{prefix}/grad_rms": _tensor_rms(gradient),
                f"{prefix}/grad_abs_max": gradient.detach().float().abs().amax(),
                f"{prefix}/rms_ratio_previous": previous_rms_ratios[id(parameter)],
                f"{prefix}/rms_ratio_updated": rms_ratio,
                f"{prefix}/adam_direction_rms": _tensor_rms(update),
                f"{prefix}/parameter_rms": parameter_rms,
                f"{prefix}/prospective_unclipped_update_rms": prospective_unclipped_update_rms,
                f"{prefix}/prospective_update_rms": prospective_update_rms,
                f"{prefix}/update_parameter_rms_ratio": prospective_update_rms
                / parameter_rms.clamp_min(torch.finfo(torch.float32).tiny),
                f"{prefix}/update_clip_active": (clip_factor < 1).float(),
                f"{prefix}/update_clip_factor": clip_factor,
            }
        )

    if update_clip_threshold is None:
        if weight_decay:
            torch._foreach_mul_(parameter_tensors, 1 - lr * weight_decay)
        torch._foreach_add_(parameter_tensors, updates, alpha=-lr)
        return diagnostics

    if weight_decay:
        updates = torch._foreach_add(updates, parameter_tensors, alpha=weight_decay)
    clipped_updates = [update * clip_factor for update, clip_factor in zip(updates, clip_factors, strict=True)]
    torch._foreach_add_(parameter_tensors, clipped_updates, alpha=-lr)
    return diagnostics


def _tensor_rms(tensor: Tensor) -> Tensor:
    """Return a float32 root-mean-square scalar."""
    return tensor.detach().float().square().mean().sqrt()


def _adam_rms_ratio(
    gradient: Tensor,
    second_moment: Tensor,
    *,
    bias_correction: float,
    eps: float,
) -> Tensor:
    """Measure gradient scale against a bias-corrected second moment."""
    corrected = second_moment.detach().float()
    if bias_correction > 0:
        corrected = corrected / bias_correction
    denominator = corrected.clamp_min(eps**2)
    return (gradient.detach().float().square() / denominator).mean().sqrt()


def _validate_update_clip_threshold(group: dict) -> None:
    threshold = group.get("update_clip_threshold")
    if threshold is not None and (
        not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0
    ):
        raise ValueError(f"update_clip_threshold must be positive or None, got {threshold!r}")


def _validate_muon_scale_clamp(group: dict) -> None:
    clamp = group.get("muon_scale_clamp_min_one", True)
    if not isinstance(clamp, bool):
        raise TypeError("muon_scale_clamp_min_one must be a bool")


class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (
                    dist.get_world_size() - len(params) % dist.get_world_size()
                )
                for base_i in range(len(params))[:: dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(
                        params_pad[base_i : base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()]
                    )
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"], state["step"], group["betas"], group["eps"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                if "muon_scale_mode" in group:
                    raise ValueError(
                        "muon_scale_mode is a checkpoint-only compatibility field; use muon_scale_clamp_min_one"
                    )
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                _validate_muon_scale_clamp(group)
                logical_splits = group.get("logical_splits", 1)
                if not isinstance(logical_splits, int) or isinstance(logical_splits, bool) or logical_splits < 1:
                    raise ValueError(f"logical_splits must be a positive integer, got {logical_splits!r}")
                required = {
                    "params",
                    "lr",
                    "momentum",
                    "weight_decay",
                    "use_muon",
                }
                assert required <= group.keys() <= required | {"logical_splits", "muon_scale_clamp_min_one"}
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                _validate_update_clip_threshold(group)
                required = {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
                assert required <= group.keys() <= required | {"update_clip_threshold"}
        super().__init__(param_groups, dict())
        self._adam_diagnostic_names: dict[int, str] = {}
        self.last_adam_diagnostics: dict[str, Tensor] = {}

    def set_adam_diagnostic_parameters(self, parameters: Mapping[str, nn.Parameter]) -> None:
        """Select Adam tensors whose step diagnostics should be retained."""
        adam_parameter_ids = {
            id(parameter) for group in self.param_groups if not group["use_muon"] for parameter in group["params"]
        }
        unknown = {name for name, parameter in parameters.items() if id(parameter) not in adam_parameter_ids}
        if unknown:
            raise ValueError(f"Adam diagnostics requested for non-Adam parameters: {sorted(unknown)}")
        if len({id(parameter) for parameter in parameters.values()}) != len(parameters):
            raise ValueError("Adam diagnostic parameters must be unique")
        self._adam_diagnostic_names = {id(parameter): name for name, parameter in parameters.items()}

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state while retaining settings absent from old checkpoints."""
        clipping = [group.get("update_clip_threshold") for group in self.param_groups]
        clipping_missing = ["update_clip_threshold" not in group for group in state_dict["param_groups"]]
        muon_clamps = [group.get("muon_scale_clamp_min_one") for group in self.param_groups]
        muon_splits = [group.get("logical_splits") for group in self.param_groups]
        splits_missing = ["logical_splits" not in group for group in state_dict["param_groups"]]

        translated_groups: list[dict] = []
        clamp_missing: list[bool] = []
        for loaded_group in state_dict["param_groups"]:
            translated = dict(loaded_group)
            has_old_mode = "muon_scale_mode" in translated
            old_mode = translated.pop("muon_scale_mode", None)
            if has_old_mode:
                if "muon_scale_clamp_min_one" in translated:
                    raise ValueError("optimizer state contains both old and current Muon scale settings")
                if old_mode not in ("legacy", "o51"):
                    raise ValueError(f"unknown saved Muon scale mode {old_mode!r}")
                translated["muon_scale_clamp_min_one"] = old_mode == "legacy"
            clamp_missing.append("muon_scale_clamp_min_one" not in translated)
            translated_groups.append(translated)
        translated_state = dict(state_dict)
        translated_state["param_groups"] = translated_groups

        super().load_state_dict(translated_state)
        for group, threshold, missing, clamp, split, missing_clamp, missing_split in zip(
            self.param_groups,
            clipping,
            clipping_missing,
            muon_clamps,
            muon_splits,
            clamp_missing,
            splits_missing,
            strict=True,
        ):
            if missing and threshold is not None:
                group["update_clip_threshold"] = threshold
            if group["use_muon"] and missing_clamp and clamp is not None:
                group["muon_scale_clamp_min_one"] = clamp
            if group["use_muon"] and missing_split and split is not None:
                group["logical_splits"] = split

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.last_adam_diagnostics = {}
        for group in self.param_groups:
            if group["use_muon"]:
                buckets: dict[tuple[torch.device, torch.dtype, tuple[int, ...]], list[nn.Parameter]] = {}
                for parameter in group["params"]:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    state = self.state[parameter]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    matrix_shape = tuple(_matrix_view(parameter).shape)
                    key = (parameter.device, parameter.dtype, matrix_shape)
                    buckets.setdefault(key, []).append(parameter)
                for parameters in buckets.values():
                    _batched_muon_step(
                        parameters,
                        [self.state[parameter]["momentum_buffer"] for parameter in parameters],
                        lr=group["lr"],
                        momentum=group["momentum"],
                        weight_decay=group["weight_decay"],
                        muon_scale_clamp_min_one=group.get("muon_scale_clamp_min_one", True),
                        logical_splits=group.get("logical_splits", 1),
                    )
            else:
                parameters_by_step: dict[int, list[nn.Parameter]] = {}
                for parameter in group["params"]:
                    if parameter.grad is None:
                        parameter.grad = torch.zeros_like(parameter)
                    state = self.state[parameter]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                        state["step"] = 0
                    state["step"] += 1
                    parameters_by_step.setdefault(state["step"], []).append(parameter)
                for step, parameters in parameters_by_step.items():
                    self.last_adam_diagnostics.update(
                        _foreach_adam_step(
                            parameters,
                            states=[self.state[parameter] for parameter in parameters],
                            step=step,
                            lr=group["lr"],
                            betas=group["betas"],
                            eps=group["eps"],
                            weight_decay=group["weight_decay"],
                            update_clip_threshold=group.get("update_clip_threshold"),
                            diagnostic_names=self._adam_diagnostic_names,
                        )
                    )

        return loss
