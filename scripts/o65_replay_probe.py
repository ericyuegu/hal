"""Diagnose O64 replay precision against eager BF16 and FP32 controls."""

import copy
import dataclasses
import json
from pathlib import Path
from typing import Literal
from typing import TypedDict
from typing import cast

import torch
import tyro
from torch import Tensor

from hal.eval.self_play_checkpoint import load_evaluation_checkpoint
from hal.eval.self_play_checkpoint import sha256
from hal.models.action_sequence import NonlinearActionHead
from hal.models.attention import Trunk
from hal.models.replay_actor import ActorTargets
from hal.models.replay_actor import actor_loss
from hal.models.replay_actor import validate_actor_targets
from hal.models.self_play import SelfPlayTransformer
from hal.models.window_replay import ReplayOutput
from hal.models.window_replay import ReplaySlots
from hal.models.window_replay import replay_windows
from hal.models.window_replay import validate_replay_inputs
from hal.representation.player_identity import vocabulary_from_checkpoint_buffer
from scripts.o65_trunk_probe import errors


@dataclasses.dataclass(frozen=True, slots=True)
class Args:
    inputs: Path
    checkpoint: Path = Path("/o65-results/o64-inputs/step-0098304.pt")
    cohort: int = 0
    reference_bias_shift: float = 0.0
    objective: Literal["total", "policy", "kl"] = "total"
    advantages: Literal["recorded", "unit", "uncentered"] = "recorded"
    ratio: Literal["recorded", "one"] = "recorded"
    loss_reduction: Literal["fp32", "fp64"] = "fp32"
    shared_hidden_gradient: bool = False
    output: Path = Path("/cache/train-replay-probe.json")

    def __post_init__(self) -> None:
        if self.shared_hidden_gradient and (self.objective != "total" or self.loss_reduction != "fp32"):
            raise ValueError("shared history gradient requires the total objective and FP32 loss reduction")


class CapturedCohort(TypedDict):
    inputs: tuple[Tensor, ...]
    context: dict[str, Tensor]
    observed: Tensor
    behavior: Tensor
    advantages: Tensor
    selected: Tensor
    coefficients: Tensor


def load_cohort(path: Path, checkpoint_sha256: str, index: int) -> CapturedCohort:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or set(saved) != {"format", "provenance", "source_sha256", "training", "cohorts"}:
        raise ValueError("training probe schema mismatch")
    if saved["format"] != "hal.o65.training-probe.v1" or saved["provenance"]["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("training probe checkpoint or format mismatch")
    if saved["training"].get("clip_epsilon") != 0.2:
        raise ValueError("replay qualification requires clip_epsilon=0.2")
    if not 0 <= index < len(saved["cohorts"]):
        raise ValueError("training probe cohort is out of range")
    cohort = saved["cohorts"][index]
    if not isinstance(cohort, dict) or set(cohort) != set(CapturedCohort.__annotations__):
        raise ValueError("training probe cohort schema mismatch")
    if not isinstance(cohort["inputs"], tuple) or len(cohort["inputs"]) != 5 + len(dataclasses.fields(ReplaySlots)):
        raise ValueError("training probe replay inputs mismatch")
    if not isinstance(cohort["context"], dict) or any(
        not isinstance(name, str) or not isinstance(value, Tensor) for name, value in cohort["context"].items()
    ):
        raise ValueError("training probe context mismatch")
    if any(not isinstance(value, Tensor) for value in cohort["inputs"]) or any(
        not isinstance(cohort[name], Tensor)
        for name in ("observed", "behavior", "advantages", "selected", "coefficients")
    ):
        raise ValueError("training probe tensors mismatch")
    return cast(CapturedCohort, cohort)


class Comparison(TypedDict):
    eager: dict[str, float]
    compiled: dict[str, float]
    compiled_vs_eager: dict[str, float]
    repeat_vs_eager: dict[str, float]
    cosine: float
    eager_fp32_cosine: float
    compiled_fp32_cosine: float
    norms: dict[str, float]
    passed: bool


def compare(candidate: Tensor, eager: Tensor, high: Tensor, repeat: Tensor) -> Comparison:
    baseline = errors(eager, high)
    candidate_error = errors(candidate, high)
    direct = errors(candidate, eager)
    repeated = errors(repeat, eager)
    cosine = float(torch.nn.functional.cosine_similarity(candidate.float(), eager.float(), dim=0))
    passed = all(bool(torch.isfinite(value).all()) for value in (candidate, eager, high, repeat))
    return {
        "eager": baseline,
        "compiled": candidate_error,
        "compiled_vs_eager": direct,
        "repeat_vs_eager": repeated,
        "cosine": cosine,
        "eager_fp32_cosine": float(torch.nn.functional.cosine_similarity(eager.float(), high.float(), dim=0)),
        "compiled_fp32_cosine": float(torch.nn.functional.cosine_similarity(candidate.float(), high.float(), dim=0)),
        "norms": {
            name: float(value.float().norm())
            for name, value in (("eager", eager), ("compiled", candidate), ("fp32", high))
        },
        "passed": passed,
    }


def fp32_copy(model: SelfPlayTransformer) -> SelfPlayTransformer:
    vocabulary = vocabulary_from_checkpoint_buffer(cast(Tensor, model.player_code_bytes).cpu().numpy())
    high = (
        SelfPlayTransformer(
            model.cfg,
            model.ranges,
            vocabulary,
            max_opponent_queue=model.critic_residual.max_queue,
            critic_hidden_dim=model.critic_residual.queue_encoder.down.out_features,
            decoder_precision="fp32",
        )
        .to(next(model.parameters()).device)
        .eval()
    )
    high.load_state_dict(model.state_dict(), strict=True)
    return high


def history(trunk: Trunk, tokens: Tensor, padding: Tensor) -> Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        return trunk(tokens, padding)


def main(args: Args) -> None:
    torch.manual_seed(65)
    device = torch.device("cuda", torch.cuda.current_device())
    checkpoint = load_evaluation_checkpoint(args.checkpoint, device=device)
    model = checkpoint.model.requires_grad_(True).eval()
    model.percentile_head.requires_grad_(False)
    model.trunk.resolve_attention("cuda")
    reference = copy.deepcopy(model)
    with torch.no_grad():
        for module in reference.temporal.outputs.values():
            head = cast(NonlinearActionHead, module)
            assert head.down.bias is not None
            head.down.bias.add_(
                torch.linspace(
                    -args.reference_bias_shift, args.reference_bias_shift, head.down.out_features, device=device
                )
            )
    high, high_reference = fp32_copy(model), fp32_copy(reference)
    cohort = load_cohort(args.inputs, checkpoint.checkpoint_sha256, args.cohort)
    context = {name: value.to(device) for name, value in cohort["context"].items()}
    observed = cohort["observed"].to(device)
    slots = ReplaySlots(*(value.to(device) for value in cohort["inputs"][5:]))
    shape = slots.valid_WK.shape
    windows = shape[0]
    valid = cohort["selected"].to(device)
    coefficient = cohort["coefficients"].to(device)
    compiled_history = torch.compile(
        history,
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False, "emulate_precision_casts": True},
    )
    compiled_replay = torch.compile(
        replay_windows,
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False, "emulate_precision_casts": True},
    )

    def forward(current: SelfPlayTransformer, *, compiled: bool, fp32: bool) -> tuple[ReplayOutput, Tensor, Tensor]:
        with torch.enable_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not fp32, cache_enabled=False):
                tokens = current.context_tokens(context, observed)
            padding = torch.zeros(windows, dtype=torch.long, device=device)
            if fp32:
                hidden = current.trunk.forward_dense(tokens, padding, force_dense=True)
            elif compiled:
                hidden = compiled_history(current.trunk, tokens, padding)
            else:
                hidden = history(current.trunk, tokens, padding)
        operation = replay_windows if fp32 else compiled_replay
        if current is model and not compiled and not fp32:
            validate_replay_inputs(current, hidden, observed, slots)
        return operation(current, hidden, observed, slots, grad=True), hidden, tokens

    references = []
    for name, current, compiled, fp32 in (
        ("eager reference", reference, False, False),
        ("compiled reference", reference, True, False),
        ("FP32 reference", high_reference, False, True),
    ):
        print(name, flush=True)
        output, _, _ = forward(current, compiled=compiled, fp32=fp32)
        references.append(tuple(p.detach() for p in output.distributions))
        del output
    quantities = []
    gradient_groups: list[dict[str, Tensor]] = []
    ratios_by_mode: list[dict[str, float]] = []
    behavior = cohort["behavior"].to(device)
    advantages = cohort["advantages"].to(device)
    # Match the trainer's float64 moments and minimum sigma before the FP32 loss.
    selected_advantages = advantages[valid].double()
    mean = selected_advantages.mean()
    sigma = (selected_advantages.square().mean() - mean.square()).clamp_min(0).sqrt().clamp_min(1e-8)
    if args.advantages == "unit":
        advantages = torch.ones_like(advantages)
        mean, sigma = mean.new_zeros(()), sigma.new_ones(())
    elif args.advantages == "uncentered":
        mean = mean.new_zeros(())
    shared_upstream: Tensor | None = None
    if args.shared_hidden_gradient:
        output, hidden, _ = forward(high, compiled=False, fp32=True)
        targets = ActorTargets(
            output.log_probs_WKF.detach() if args.ratio == "one" else behavior,
            advantages,
            valid,
            coefficient,
            references[2],
            mean.float(),
            sigma.float(),
            valid.sum().float(),
        )
        loss = actor_loss(output, targets)
        shared_upstream = torch.autograd.grad(loss.loss, hidden)[0].detach()
        del output, hidden, loss
    loss_summaries = []
    for index, (name, current, compiled, fp32) in enumerate(
        (
            ("eager actor", model, False, False),
            ("eager repeat", model, False, False),
            ("compiled actor", model, True, False),
            ("FP32 actor", high, False, True),
        )
    ):
        print(name, flush=True)
        output, hidden, tokens = forward(current, compiled=compiled, fp32=fp32)
        targets = ActorTargets(
            output.log_probs_WKF.detach() if args.ratio == "one" else behavior,
            advantages,
            valid,
            coefficient,
            references[(0, 0, 1, 2)[index]],
            mean.float(),
            sigma.float(),
            valid.sum().float(),
        )
        validate_actor_targets(output, targets)
        loss = actor_loss(output, targets)
        features = output.features
        with torch.no_grad():
            values = current.self_play_value(
                features.trunk_WKd.flatten(0, 1),
                features.decoder_WKt.flatten(0, 1),
                features.condition_WK4.flatten(0, 1),
                features.opponent_condition_WK4.flatten(0, 1),
                features.opponent_queue_WKQ4.flatten(0, 1),
                features.opponent_lengths_WK.flatten(),
                validate=False,
            ).reshape(shape)
        loss_summaries.append({"policy": float(loss.policy_loss.detach()), "kl": float(loss.kl_loss.detach())})
        ratios = loss.joint_ratio_WK[valid].detach()
        ratios_by_mode.append(
            {
                "mean": float(ratios.mean()),
                "max_abs_gap_from_one": float((ratios - 1).abs().max()),
                "clip_fraction": float(((ratios - 1).abs() > 0.2).float().mean()),
            }
        )
        parameters = current.actor_parameters()
        objective = {"total": loss.loss, "policy": loss.policy_loss, "kl": loss.kl_loss}[args.objective]
        if args.loss_reduction == "fp64":
            normalized = (advantages - mean.float()) / sigma.float()
            ratio = loss.joint_ratio_WK
            surrogate = torch.minimum(ratio * normalized, ratio.clamp(0.8, 1.2) * normalized)
            policy = torch.where(valid, -surrogate, 0).double().sum() / valid.sum().double()
            kl_terms = loss.factor_kl_WKF.sum(-1) * torch.where(valid, coefficient / sigma.float(), 0)
            kl = kl_terms.double().sum() / valid.sum().double()
            objective = {"total": policy + kl, "policy": policy, "kl": kl}[args.objective]
        if shared_upstream is not None:
            objective = (hidden.float() * shared_upstream).sum()
        derivatives = torch.autograd.grad(objective, (*parameters, hidden, tokens), allow_unused=True)
        gradient = derivatives[:-2]
        names = {id(parameter): name for name, parameter in current.named_parameters()}
        grouped: dict[str, list[Tensor]] = {}
        for parameter, value in zip(parameters, gradient, strict=True):
            path = names[id(parameter)].split(".")
            name = ".".join(path[:3]) if path[0] in ("trunk", "temporal") else path[0]
            grouped.setdefault(name, []).append((torch.zeros_like(parameter) if value is None else value).flatten())
        gradient_groups.append({name: torch.cat(values).detach() for name, values in grouped.items()})
        quantities.append(
            {
                "log_probs": output.log_probs_WKF[output.factor_mask_WKF].detach(),
                "kl": loss.factor_kl_WKF[output.factor_mask_WKF].detach(),
                "values": values[valid].detach(),
                "hidden": hidden.detach().flatten(),
                "hidden_gradient": derivatives[-2].detach().flatten(),
                "token_gradient": derivatives[-1].detach().flatten(),
                "actor_gradients": torch.cat(
                    [
                        (torch.zeros_like(parameter) if value is None else value).flatten()
                        for parameter, value in zip(parameters, gradient, strict=True)
                    ]
                ).detach(),
            }
        )
        del output, loss, gradient
    report: dict[str, object] = {
        "format": "hal.o65.replay-diagnostics.v2",
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
        "inputs_sha256": sha256(args.inputs),
        "probe_sha256": sha256(Path(__file__)),
        "windows": windows,
        "cohort": args.cohort,
        "reference_bias_shift": args.reference_bias_shift,
        "objective": args.objective,
        "advantages": args.advantages,
        "ratio": args.ratio,
        "loss_reduction": args.loss_reduction,
        "shared_hidden_gradient": args.shared_hidden_gradient,
        "advantage_mean": float(mean),
        "advantage_sigma": float(sigma),
        "losses": dict(zip(("eager", "repeat", "compiled", "fp32"), loss_summaries, strict=True)),
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(device),
        "step_zero_ratios": dict(zip(("eager", "repeat", "compiled", "fp32"), ratios_by_mode, strict=True)),
    }
    passed = True
    report["gradient_groups"] = {
        name: compare(value, gradient_groups[0][name], gradient_groups[3][name], gradient_groups[1][name])
        for name, value in gradient_groups[2].items()
    }
    for name, value in quantities[2].items():
        comparison = compare(value, quantities[0][name], quantities[3][name], quantities[1][name])
        report[name] = comparison
        passed &= comparison["passed"]
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not passed:
        raise ValueError("replay precision diagnostic contains NaN or infinity")


if __name__ == "__main__":
    main(tyro.cli(Args))
