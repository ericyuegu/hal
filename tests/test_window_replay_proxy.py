"""Compare window replay against the recorded per-decision evaluator on the O63 proxy."""

import contextlib
import dataclasses
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import torch
from o65_reference import ROOT
from o65_reference import load_control
from torch import Tensor

from hal.inference.cuda_graph import CapturedCall
from hal.inference.graphed_actor import GraphedActorStep
from hal.models.attention import KVMemory
from hal.models.replay_actor import ActorTargets
from hal.models.replay_actor import actor_loss
from hal.models.replay_actor import validate_actor_targets
from hal.models.self_play import SelfPlayTransformer
from hal.models.window_replay import ReplayCriticFeatures
from hal.models.window_replay import ReplayOutput
from hal.models.window_replay import ReplaySlots
from hal.models.window_replay import replay_windows
from hal.models.window_replay import validate_replay_inputs

exp = load_control()


@dataclasses.dataclass(frozen=True, slots=True)
class ProxyBatch:
    features: dict[str, Tensor]
    observed: Tensor
    slots: ReplaySlots
    states: Any
    actions: Tensor
    source_indices: Tensor
    advantages_WK: Tensor
    value_targets_WK: Tensor
    selected_WK: Tensor
    kl_coefficient_WK: Tensor


def pack(policy: Any, states: Any, prepared: Any, keys: list[tuple[int, ...]], slots_per_window: int) -> ProxyBatch:
    decisions = prepared.decisions
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for row, meta in enumerate(states.meta):
        groups[exp._window_key(meta)].append(row)
    window_rows, source_indices, positions, valid = [], [], [], []
    for key in keys:
        rows = groups[key][:slots_per_window]
        window_rows.append(max(groups[key], key=lambda i: states.meta[i, exp.LENGTH]))
        source_indices.extend(rows)
        positions.append([int(states.meta[i, exp.LENGTH]) - 1 for i in rows] + [0] * (slots_per_window - len(rows)))
        valid.append([True] * len(rows) + [False] * (slots_per_window - len(rows)))
    chosen = states.take(source_indices)
    features, observed = policy._contexts(states.take(window_rows), length=256)
    device = policy.device
    shape = (len(keys), slots_per_window)
    occupied = torch.tensor(valid, device=device)
    source = torch.tensor(source_indices, device=device)
    integers = chosen.tensor(chosen.table.integers)
    conditions = chosen.tensor(chosen.table.conditions)

    def pad(value: Tensor) -> Tensor:
        target = value.new_zeros(*shape, *value.shape[1:])
        target[occupied] = value
        return target

    delay = pad(integers[:, exp.DELAY])
    replan = pad(integers[:, exp.REPLAN])
    condition = policy.model.ranges.normalize(conditions[:, 0], integers[:, exp.DELAY], integers[:, exp.REPLAN])
    opponent_condition = policy.model.ranges.normalize(
        conditions[:, 1], integers[:, exp.OPP_DELAY], integers[:, exp.OPP_REPLAN]
    )
    actions = chosen.tensor(chosen.table.actions)
    slots = ReplaySlots(
        torch.tensor(positions, device=device),
        occupied,
        torch.zeros(len(keys), 256, dtype=torch.long, device=device),
        pad(chosen.tensor(chosen.table.prefix)[:, :10]),
        delay,
        replan,
        pad(condition),
        pad(opponent_condition),
        pad(chosen.tensor(chosen.table.opponent)[:, : policy.model.critic_residual.max_queue]),
        pad(integers[:, exp.OPP_LENGTH]),
        pad(actions),
        pad(decisions.executed_factors_JF[source]),
    )
    return ProxyBatch(
        features,
        observed,
        slots,
        chosen,
        actions,
        source,
        pad(prepared.advantages[source]),
        pad(prepared.targets[source]),
        pad(prepared.selected[source]),
        pad(decisions.kl_coefficient_J[source]),
    )


def window_forward(
    model: SelfPlayTransformer,
    batch: ProxyBatch,
    *,
    grad: bool,
    fp32: bool = False,
    replay: Callable[..., ReplayOutput] = replay_windows,
) -> ReplayOutput:
    with (
        torch.set_grad_enabled(grad),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=not fp32, cache_enabled=False),
    ):
        hidden = model(
            batch.features, torch.zeros(len(batch.observed), dtype=torch.long, device="cuda"), batch.observed
        )
    return replay(model, hidden, batch.observed, batch.slots, grad=grad)


def old_forward(policy: Any, batch: ProxyBatch, *, grad: bool, fp32: bool = False) -> ReplayOutput:
    # _evaluate supplies the old per-decision packing; FP32 replaces only its
    # unconditional outer autocast, so every model computation stays FP32.
    with (
        torch.set_grad_enabled(grad),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=not fp32, cache_enabled=False),
    ):
        result = fp32_evaluate(policy, batch) if fp32 else policy._evaluate(batch.states, batch.actions)
    slots = batch.slots
    shape = tuple(slots.positions_WK.shape)

    def pad(value: Tensor) -> Tensor:
        target = value.new_zeros(*shape, *value.shape[1:])
        target[slots.valid_WK] = value
        return target

    # The timed control also includes this slot-layout conversion.
    features = ReplayCriticFeatures(
        *(pad(getattr(result.features, field.name)) for field in dataclasses.fields(result.features))
    )
    return ReplayOutput(
        pad(result.log_probs),
        tuple(pad(p) for p in result.distributions),
        slots.executed_WKF & slots.valid_WK[..., None],
        features,
    )


def fp32_evaluate(policy: Any, batch: ProxyBatch) -> Any:
    """Per-decision reference with FP32 dense attention; Flash requires BF16."""
    model = policy.model
    windows = batch.observed.shape[0]
    hidden = model.trunk.forward_dense(
        model.context_tokens(batch.features, batch.observed),
        torch.zeros(windows, dtype=torch.long, device="cuda"),
        force_dense=True,
    )
    key, value = model.temporal.history_attention.project_memory(hidden)
    rows = torch.arange(windows, device="cuda")[:, None].expand_as(batch.slots.valid_WK)[batch.slots.valid_WK]
    positions = batch.slots.positions_WK[batch.slots.valid_WK]
    outputs = []
    for start in range(0, len(rows), 128):
        window, position = rows[start : start + 128], positions[start : start + 128]
        states = batch.states[start : start + 128]
        outputs.append(
            policy._decode(
                states,
                hidden[window, position][:, None],
                batch.observed[window, position],
                torch.full((4, 4, len(window)), 0.5, device="cuda"),
                batch.actions[start : start + 128],
                history=KVMemory(
                    torch.stack((key[window], value[window])),
                    torch.arange(256, device="cuda")[None].expand(len(window), -1),
                    position,
                    256,
                ),
                count=4,
            )
        )
    return exp._concatenate_evaluations(outputs)


def values(model: SelfPlayTransformer, output: ReplayOutput) -> Tensor:
    features = output.features
    return model.self_play_value(
        features.trunk_WKd.flatten(0, 1),
        features.decoder_WKt.flatten(0, 1),
        features.condition_WK4.flatten(0, 1),
        features.opponent_condition_WK4.flatten(0, 1),
        features.opponent_queue_WKQ4.flatten(0, 1),
        features.opponent_lengths_WK.flatten(0, 1),
    ).reshape(output.log_probs_WKF.shape[:2])


def targets(output: ReplayOutput, reference: ReplayOutput, behavior: Tensor, batch: ProxyBatch) -> ActorTargets:
    selected = batch.selected_WK & output.factor_mask_WKF.any(-1)
    # The control's GAE targets stay fixed across all numerical paths.
    advantages = batch.advantages_WK
    return ActorTargets(
        behavior,
        advantages,
        selected,
        batch.kl_coefficient_WK,
        reference.distributions,
        advantages[selected].mean(),
        advantages[selected].std(unbiased=False),
        selected.sum().float(),
    )


def error(candidate: Tensor, reference: Tensor) -> dict[str, float]:
    delta = (candidate.detach().double() - reference.detach().double()).flatten()
    ref = reference.detach().double().flatten()
    return {
        "rmse": float(delta.square().mean().sqrt()),
        "max": float(delta.abs().max()),
        "relative_l2": float(delta.norm() / ref.norm().clamp_min(1e-30)),
    }


def flattened_gradients(loss: Tensor, parameters: tuple[torch.nn.Parameter, ...]) -> Tensor:
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return torch.cat(
        [
            (torch.zeros_like(p) if g is None else g).detach().flatten()
            for p, g in zip(parameters, gradients, strict=True)
        ]
    )


def quantities(
    policy: Any, reference: Any, batch: ProxyBatch, behavior: Tensor, *, window: bool, fp32: bool
) -> dict[str, Tensor]:
    compiled_replay = (
        torch.compile(
            replay_windows,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False, "emulate_precision_casts": True},
        )
        if window
        else replay_windows
    )
    forward = partial(window_forward, replay=compiled_replay) if window else old_forward
    output = forward(policy.model if window else policy, batch, grad=True, fp32=fp32)
    if window:
        inputs = replay_inputs(batch)
        score_forward = partial(
            forward_inputs, reference.model, tuple(batch.features), grad=False, replay=compiled_replay
        )
        score = capture_score(score_forward, inputs)
        ref_output = graph_score(score, inputs)
    else:
        ref_output = old_forward(reference, batch, grad=False, fp32=fp32)
    target = targets(output, ref_output, behavior, batch)
    validate_actor_targets(output, target)
    loss = actor_loss(output, target)
    functional_actor = flattened_gradients(loss.loss, policy.model.actor_parameters())
    if window:
        # Default-stream actor nodes must be released before side-stream capture.
        del output, loss
        actor_forward = partial(forward_inputs, policy.model, tuple(batch.features), replay=compiled_replay)
        graph = GraphedActorStep(actor_forward, inputs, target, policy.model.actor_parameters())
        graph.backward(inputs, target)
        output = graph.evaluation()
        actor_gradient = torch.cat([gradient.detach().flatten() for gradient in graph.gradients])
    else:
        actor_gradient = functional_actor
    diagnostic_kl = actor_loss(output, dataclasses.replace(target, selected_WK=output.factor_mask_WKF.any(-1)))
    mask = output.factor_mask_WKF
    selected = batch.slots.valid_WK
    critic_values = values(policy.model, output)
    critic_loss = (
        torch.where(selected, ((critic_values - batch.value_targets_WK) / 150).square(), 0).sum() / selected.sum()
    )
    result = {
        "log_probs": output.log_probs_WKF[mask].detach(),
        "kl": diagnostic_kl.factor_kl_WKF[mask].detach(),
        "values": critic_values[selected].detach(),
        "trunk_features": output.features.trunk_WKd[selected].detach(),
        "decoder_features": output.features.decoder_WKt[selected].detach(),
        "condition_features": torch.cat(
            (output.features.condition_WK4[selected], output.features.opponent_condition_WK4[selected]), -1
        ).detach(),
        "opponent_queue": output.features.opponent_queue_WKQ4[selected].float(),
        "opponent_lengths": output.features.opponent_lengths_WK[selected].float(),
        "actor_gradients": actor_gradient,
        "critic_gradients": flattened_gradients(critic_loss, policy.model.critic_parameters()),
    }
    if window:
        result["functional_actor_gradients"] = functional_actor
        # The returned tensors own their copies before graph pools are released.
        torch.cuda.synchronize()
    return result


def forward_inputs(
    model: SelfPlayTransformer,
    names: tuple[str, ...],
    tensors: tuple[Tensor, ...],
    *,
    grad: bool = True,
    replay: Callable[..., ReplayOutput] = replay_windows,
) -> ReplayOutput:
    count = len(names)
    observed = tensors[count]
    slots = ReplaySlots(*tensors[count + 1 :])
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
        hidden = model(
            dict(zip(names, tensors[:count], strict=True)),
            torch.zeros(observed.shape[0], dtype=torch.long, device="cuda"),
            observed,
        )
    return replay(model, hidden, observed, slots, grad=grad)


def graph_score(captured: CapturedCall[ReplayOutput], inputs: tuple[Tensor, ...]) -> ReplayOutput:
    output = captured(inputs)
    return ReplayOutput(
        output.log_probs_WKF.clone(),
        tuple(p.clone() for p in output.distributions),
        output.factor_mask_WKF.clone(),
        ReplayCriticFeatures(
            *(getattr(output.features, field.name).clone() for field in dataclasses.fields(output.features))
        ),
    )


def replay_inputs(batch: ProxyBatch) -> tuple[Tensor, ...]:
    return (
        *batch.features.values(),
        batch.observed,
        *(getattr(batch.slots, field.name) for field in dataclasses.fields(batch.slots)),
    )


def capture_score(
    forward: Callable[[tuple[Tensor, ...]], ReplayOutput], inputs: tuple[Tensor, ...]
) -> CapturedCall[ReplayOutput]:
    static = tuple(t.detach().clone() for t in inputs)
    return CapturedCall(partial(forward, static), static, device=inputs[0].device)


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(65)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(False)
    checkpoint = exp.load_evaluation_checkpoint(
        ROOT / "runs/proxy/final.pt", device=torch.device("cuda"), decoder_precision="bf16"
    )
    checkpoint.model.requires_grad_(True)
    checkpoint.model.percentile_head.requires_grad_(False)
    schedule = exp.development_config(checkpoint.model.head_offsets)
    policy = exp.TransformerPolicy(checkpoint.model, schedule, checkpoint.statistics, compiled=True)
    reference = policy.frozen_copy()
    # Nonzero KL and a nonzero residual critic exercise every precision path.
    with torch.no_grad():
        for head in policy.model.temporal.outputs.values():
            head.down.bias.add_(torch.linspace(-0.1, 0.1, head.down.out_features, device="cuda"))
        torch.nn.init.normal_(policy.model.critic_residual.residual.down.weight, std=0.005)
    source = Path(os.environ.get("HAL_MSL_SOURCE", "~/src/melee-sim-light")).expanduser()
    sampling = exp.read_native_sampler(ROOT / "experiments/o65/pilot-sampler.json")
    with contextlib.ExitStack() as resources:
        shards = []
        for index, cpu in enumerate(sorted(os.sched_getaffinity(0))[:4]):
            shard = exp.NativeSimulatorShard(
                source / "data", tuple(range(4 * index, 4 * index + 4)), cpu=cpu, device=policy.device, max_frame=28800
            )
            resources.callback(shard.close)
            shards.append(shard)
        engine = exp.RolloutEngine(
            (tuple(shards),),
            policy,
            exp.EpisodeSampler(sampling, schedule, exp.CounterRng(65)),
            {0: reference},
            fixed_anchor_lanes=True,
        )
        engine.initialize()
        rollout = engine.collect(0)
        states = rollout.records.states
        training = exp.TrainConfig(critic_warmup_updates=0, deterministic=False)
        prepared = exp.Trainer(engine, training).prepare(rollout)
        decisions = prepared.decisions
        groups: dict[tuple[int, ...], int] = {}
        for index, meta in enumerate(states.meta):
            if meta[exp.BLOCK] == 128:
                groups[exp._window_key(meta)] = index
        keys = []
        for cadence in (2, 4):
            candidates = [key for key, row in groups.items() if int(states.meta[row, exp.REPLAN]) == cadence]
            if len(candidates) < 2:
                raise ValueError("precision fixture needs two windows of each cadence")
            keys.extend(candidates[:2])
        batch = pack(policy, states, prepared, keys, 64)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = policy.model(
                batch.features, torch.zeros(len(keys), dtype=torch.long, device="cuda"), batch.observed
            )
        validate_replay_inputs(policy.model, hidden, batch.observed, batch.slots)
        behavior = torch.zeros(*batch.slots.positions_WK.shape, 16, device="cuda")
        behavior[batch.slots.valid_WK] = decisions.behavior_log_probs_JF[batch.source_indices]
        torch.save(
            {
                "features": batch.features,
                "observed": batch.observed,
                "slots": dataclasses.asdict(batch.slots),
                "behavior": behavior,
                "advantages": batch.advantages_WK,
                "value_targets": batch.value_targets_WK,
                "selected": batch.selected_WK,
                "kl_coefficient": batch.kl_coefficient_WK,
            },
            output_dir / "inputs.pt",
        )
        torch.save(
            {"student": policy.model.state_dict(), "reference": reference.model.state_dict()},
            output_dir / "weights.pt",
        )
        report: dict[str, Any] = {
            "format": "hal.o65.window-replay.v1",
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "checkpoint_sha256": checkpoint.checkpoint_sha256,
            "environment": exp._environment(),
            "seed": 65,
            "sampler_sha256": exp.sha256(ROOT / "experiments/o65/pilot-sampler.json"),
            "simulator_sha256": shards[0].shard.library_sha256,
            "adapter_sha256": exp.native_bridge.library_sha256(),
            "data_manifests": checkpoint.provenance["data_manifests"],
            "input_sha256": hashlib.sha256((output_dir / "inputs.pt").read_bytes()).hexdigest(),
            "weights_sha256": exp.sha256(output_dir / "weights.pt"),
            "source_sha256": {
                name: exp.sha256(ROOT / name)
                for name in (
                    "hal/models/action_sequence.py",
                    "hal/models/window_replay.py",
                    "hal/models/replay_actor.py",
                    "hal/inference/graphed_actor.py",
                    "experiments/065_selfplay_ppo.py",
                    "tests/test_window_replay_proxy.py",
                )
            },
            "student_state_sha256": exp._state_hash(policy.model.state_dict()),
            "reference_state_sha256": exp._state_hash(reference.model.state_dict()),
            "simulator_git_sha": subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip(),
            "simulator_data_manifest_sha256": exp.sha256(source / "data/raw/manifest.json"),
            "configuration": {
                "matches": 16,
                "schedule": dataclasses.asdict(schedule),
                "W": len(keys),
                "K": 64,
                "tf32": False,
                "compiled_scope": "replay_windows; trunk remains eager inside CUDA capture",
                "compiler_options": {"triton.cudagraphs": False, "emulate_precision_casts": True},
                "bias_shift": [-0.1, 0.1],
                "actor_shift_applied_before_rollout": True,
                "critic_residual_down_std": 0.005,
                "advantages": "control GAE from the native rollout, fixed across numerical paths",
                "critic_targets": "control GAE + V, fixed across numerical paths",
                "selection": "control counter sampling, 16 decisions per window",
                "training": dataclasses.asdict(training),
            },
        }
        default = quantities(policy, reference, batch, behavior, window=False, fp32=False)
        candidate = quantities(policy, reference, batch, behavior, window=True, fp32=False)
        report["actor_graph_vs_functional"] = error(
            candidate["actor_gradients"], candidate["functional_actor_gradients"]
        )
        high_model = exp.load_evaluation_checkpoint(
            ROOT / "runs/proxy/final.pt", device=torch.device("cuda"), decoder_precision="fp32"
        ).model
        high_model.load_state_dict(policy.model.state_dict())
        high_model.requires_grad_(True)
        high_model.percentile_head.requires_grad_(False)
        high_policy = exp.TransformerPolicy(high_model, schedule, checkpoint.statistics)
        high_ref_model = exp.load_evaluation_checkpoint(
            ROOT / "runs/proxy/final.pt", device=torch.device("cuda"), decoder_precision="fp32"
        ).model
        high_ref_model.load_state_dict(reference.model.state_dict())
        high_reference = exp.TransformerPolicy(high_ref_model, schedule, checkpoint.statistics)
        high = quantities(high_policy, high_reference, batch, behavior, window=False, fp32=True)
        precision = {}
        for name in high:
            control_error = error(default[name], high[name])
            candidate_error = error(candidate[name], high[name])
            # Absolute floors are in the quantity's units. Gradient elements
            # have their own scale; also gate their relative L2 error.
            floor = 1e-6 if "gradients" in name else 1e-5
            passed = (
                candidate_error["rmse"] <= 1.5 * control_error["rmse"] + floor
                and candidate_error["max"] <= 1.5 * control_error["max"] + floor
                and candidate_error["relative_l2"] <= 1.5 * control_error["relative_l2"] + 1e-5
            )
            precision[name] = {
                "default": control_error,
                "window": candidate_error,
                "absolute_floor": floor,
                "relative_l2_floor": 1e-5,
                "passed": passed,
            }
        report["precision"] = precision
        (output_dir / "measurements.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"precision": precision}), flush=True)
        assert all(row["passed"] for row in precision.values()), precision
        return report


@pytest.mark.integration
def test_proxy_window_replay_precision(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.fail("proxy window replay qualification requires CUDA")
    run(tmp_path)
