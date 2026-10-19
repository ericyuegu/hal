"""Proxy-scale numerical checks for batched rollout and training kernels."""

import contextlib
import dataclasses
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from test_065_columnar import ROOT
from test_065_columnar import exp
from torch import Tensor


@torch.no_grad()
def grouped_score(policy: Any, states: Any, actions: Tensor) -> Any:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(states.meta):
        entry = policy.caches.get(int(row[exp.STREAM]))
        seen = entry.history_length if entry is not None and entry.window == exp._window_key(row) else 0
        groups[int(row[exp.LENGTH]), seen].append(index)
    outputs = []
    order = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for (_, seen), indices in groups.items():
            batch = states.take(indices)
            hidden, observed, cache = policy._advance_rollout_cache(batch, seen)
            at = torch.tensor(indices, device=policy.device)
            count = 4 if batch.meta[0, exp.CLOCK] % 4 == 0 else 2
            outputs.append(
                policy._decode(
                    batch,
                    hidden[:, -1:],
                    observed,
                    policy._action_uniforms(batch, exp.CounterRng(65)),
                    actions[at],
                    history=cache.memory(),
                    count=count,
                )
            )
            order.extend(indices)
    return exp._take_evaluation(
        exp._concatenate_evaluations(outputs), torch.tensor(np.argsort(order), device=policy.device)
    )


class ComparisonPolicy(exp.TransformerPolicy):
    def __init__(self, model: Any, schedule: Any, statistics: Any) -> None:
        super().__init__(model, schedule, statistics, compiled=True)
        self.control = exp.TransformerPolicy(model, schedule, statistics, compiled=True)
        self.actual_outputs: list[Any] = []
        self.expected_outputs: list[Any] = []

    def clear_caches(self) -> None:
        super().clear_caches()
        self.control.clear_caches()

    def sample_tick(self, states: Any, rng: Any, layout: Any, ring: Any) -> Any:
        actual = super().sample_tick(states, rng, layout, ring)
        due = np.flatnonzero(states.meta[:, exp.CLOCK] % states.meta[:, exp.REPLAN] == 0)
        if len(due):
            selected = exp._take_evaluation(actual, torch.from_numpy(due).to(self.device))
            self.actual_outputs.append(selected)
            self.expected_outputs.append(grouped_score(self.control, states.take(due), selected.actions))
        torch.testing.assert_close(ring.controllers, self.device_wire_actions(ring.values), rtol=0, atol=0)
        return actual


def errors(actual: Tensor, expected: Tensor) -> dict[str, float]:
    difference = (actual.detach().float() - expected.detach().float()).abs().flatten()
    return {"max": float(difference.max()), "p99": float(difference.quantile(0.99))}


def executed_kl(policy: Any, reference: Any, mask: Tensor) -> Tensor:
    result = torch.zeros_like(policy.log_probs).reshape(-1, 4, 4)
    frames = mask.reshape(-1, 4, 4)
    for group, (p, q) in enumerate(zip(policy.distributions, reference.distributions, strict=True)):
        selected = frames[..., group]
        result[..., group][selected] = exp.categorical_kl(
            exp._log_distribution(p[selected]), exp._log_distribution(q[selected])
        )
    return result.flatten(1).sum(-1)


def gradient_comparison(actual: tuple[Tensor | None, ...], expected: tuple[Tensor | None, ...]) -> dict[str, float]:
    pairs = [(a, b) for a, b in zip(actual, expected, strict=True) if a is not None and b is not None]
    assert all((a is None) == (b is None) for a, b in zip(actual, expected, strict=True))
    left = torch.cat([a.float().flatten() for a, _ in pairs])
    right = torch.cat([b.float().flatten() for _, b in pairs])
    return {
        "cosine": float(torch.nn.functional.cosine_similarity(left, right, dim=0)),
        "relative_norm": float((left - right).norm() / right.norm().clamp_min(1e-12)),
        "norm_ratio": float(left.norm() / right.norm().clamp_min(1e-12)),
    }


def actor_gradients(run: Any, batch: Any) -> tuple[Tensor | None, ...]:
    # Sum the update's minibatch gradients at common weights. Each window keeps
    # its original step and normalization; no optimizer can obscure a loss error.
    parameters = run.policy.actor_parameters()
    run.actor_optimizer.zero_grad(set_to_none=True)
    for indices in batch.steps:
        indices = tuple(index for index in indices if bool(batch.selected[index]))
        at = torch.tensor(indices, device=run.policy.device)
        decisions = exp._take_decisions(batch.decisions, at)
        selected = batch.selected[at]
        statistics = exp.advantage_statistics(exp.advantage_moments(batch.advantages[at], selected))
        states = batch.rollout.records.states.take(indices)
        new = run.policy.evaluate(states, batch.old.actions[at])
        reference = exp._take_evaluation(batch.reference, at)
        objective = exp.ppo_loss(
            decisions,
            new.log_probs,
            batch.old.log_probs[at],
            batch.advantages[at],
            selected,
            statistics,
            clip_epsilon=run.config.clip_epsilon,
            importance_cap=run.config.importance_cap,
        )
        loss = objective.loss + exp.differentiable_kl_loss(
            decisions, exp.factor_kl(new, reference), selected, statistics
        )
        loss.backward()
    return tuple(None if p.grad is None else p.grad.detach().clone() for p in parameters)


def test_tick_rejects_unknown_and_duplicate_streams_before_compute() -> None:
    from test_065_loop import exp as loop_exp
    from test_065_loop import transformer_trainer

    run = transformer_trainer("cpu", development=True)
    state = run.engine._state(0, 0, 0, 0)
    rows = run.policy._columnar_states((state,))
    with pytest.raises(ValueError, match="unknown stream"):
        run.policy.sample_tick(
            rows, run.engine.sampler.rng, loop_exp.TickLayout((100, 101), (0, 0)), run.engine.rings[0]
        )
    with pytest.raises(ValueError, match="distinct"):
        run.policy.sample_tick(rows, run.engine.sampler.rng, loop_exp.TickLayout((0, 0), (0, 0)), run.engine.rings[0])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_wire_tables_keep_the_addresses_used_by_tick_graphs(device: str) -> None:
    from test_065_loop import transformer_trainer

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    policy = transformer_trainer(device).policy
    addresses = tuple(value.data_ptr() for value in policy.device_wire_tables)
    policy.clear_caches()
    assert tuple(value.data_ptr() for value in policy.device_wire_tables) == addresses


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_rotary_graph_owns_factors_after_the_eager_cache_changes(dtype: torch.dtype) -> None:
    from hal.inference.cuda_graph import CapturedCall
    from hal.models.attention import Rotary
    from hal.models.attention import apply_rotary_emb

    device = torch.device("cuda", torch.cuda.current_device())
    rotary = Rotary(32).to(device)
    inputs = torch.randn(2, 14, 2, 32, dtype=dtype, device=device)
    old_cos, old_sin = rotary.at(14, device, dtype)

    def rotate() -> Tensor:
        cosine, sine = rotary.at(14, device, dtype)
        return apply_rotary_emb(inputs, cosine, sine)

    expected = rotate().clone()
    captured = CapturedCall(rotate, (inputs,), device=device)
    rotary.at(37, device, dtype)
    # Simulate reuse of constants released when the eager cache changes length.
    old_cos.zero_()
    old_sin.zero_()
    torch.testing.assert_close(captured((inputs,)), expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tick_queues_and_resume_through_odd_resets_and_unsorted_streams(tmp_path: Path) -> None:
    from test_065_loop import PROVENANCE
    from test_065_loop import exp as loop_exp
    from test_065_loop import transformer_trainer

    run = transformer_trainer("cuda", development=True, compiled=True)
    shard = type(run.engine.groups[0][0])((3, 0), length=67, timeout=True)
    engine = loop_exp.RolloutEngine(((shard,),), run.policy, run.engine.sampler, {0: run.policy.frozen_copy()})
    engine.initialize()
    run = loop_exp.Trainer(engine, run.config)
    for _ in range(2):
        batch, _ = run.run_update()
        assert batch.rollout.resets > 0 and batch.rollout.timeouts > 0
        ring = engine.rings[0]
        torch.testing.assert_close(ring.controllers, run.policy.device_wire_actions(ring.values), rtol=0, atol=0)
    assert run.policy.tick_graphs
    loop_exp.verify_checkpoint_resume(run, tmp_path, PROVENANCE)
    assert run.update == 2
    assert not run.config.deterministic and not torch.are_deterministic_algorithms_enabled()
    assert not run.policy.tick_graphs
    run.run_update()
    ring = engine.rings[0]
    torch.testing.assert_close(ring.controllers, run.policy.device_wire_actions(ring.values), rtol=0, atol=0)


@pytest.mark.integration
def test_proxy_tick_numerics_and_update_gradients(tmp_path: Path) -> None:
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        pytest.fail("proxy tick qualification requires CUDA")
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    torch.manual_seed(65)
    try:
        # Preserve the FP32 control used to qualify the vectorization treatment.
        checkpoint = exp.load_evaluation_checkpoint(
            ROOT / "runs/proxy/final.pt", device=torch.device("cuda"), decoder_precision="fp32"
        )
        checkpoint.model.requires_grad_(True)
        checkpoint.model.percentile_head.requires_grad_(False)
        schedule = exp.development_config(checkpoint.model.head_offsets)
        policy = ComparisonPolicy(checkpoint.model, schedule, checkpoint.statistics)
        source = Path(os.environ.get("HAL_MSL_SOURCE", "~/src/melee-sim-light")).expanduser()
        sampling = exp.read_native_sampler(ROOT / "experiments/o65/pilot-sampler.json")
        with contextlib.ExitStack() as resources:
            shards = []
            for index, cpu in enumerate(sorted(os.sched_getaffinity(0))[:4]):
                shard = exp.NativeSimulatorShard(
                    source / "data",
                    tuple(range(4 * index, 4 * index + 4)),
                    cpu=cpu,
                    device=policy.device,
                    max_frame=28800,
                )
                resources.callback(shard.close)
                shards.append(shard)
            engine = exp.RolloutEngine(
                (tuple(shards),),
                policy,
                exp.EpisodeSampler(sampling, schedule, exp.CounterRng(65)),
                {0: policy.frozen_copy()},
                fixed_anchor_lanes=True,
            )
            engine.initialize()
            run = exp.Trainer(engine, exp.TrainConfig(critic_warmup_updates=0))
            rollout = engine.collect(0)
            actual = exp._concatenate_evaluations(policy.actual_outputs)
            expected = exp._concatenate_evaluations(policy.expected_outputs)
            batch = run.prepare(rollout)
            mask = batch.decisions.executed_factors_JF
            report: dict[str, Any] = {
                "log_prob": errors(actual.log_probs[mask], expected.log_probs[mask]),
                "value": errors(policy.values(actual.features), policy.values(expected.features)),
                "kl": errors(
                    executed_kl(actual, batch.reference, mask),
                    executed_kl(expected, batch.reference, mask),
                ),
                "control_cache_to_training_value": errors(policy.values(expected.features), batch.values),
                "tick_cache_to_training_value": errors(policy.values(actual.features), batch.values),
                "training_value_std": float(batch.values.detach().std()),
                "training_value_abs_max": float(batch.values.detach().abs().max()),
            }
            control = dataclasses.replace(
                batch,
                decisions=dataclasses.replace(
                    batch.decisions,
                    behavior_log_probs_JF=expected.log_probs,
                ),
            )
            report["actor_gradient"] = gradient_comparison(actor_gradients(run, batch), actor_gradients(run, control))
            gradients = []
            for features in (batch.old.features, control.old.features):
                loss = exp.critic_loss(
                    policy.values(features),
                    batch.targets,
                    torch.ones_like(batch.selected),
                    value_scale=run.config.value_scale,
                )
                gradients.append(torch.autograd.grad(loss, policy.critic_parameters()))
            report["critic_gradient"] = gradient_comparison(*gradients)
            (tmp_path / "tick-parity.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, sort_keys=True), flush=True)
            assert report["log_prob"]["max"] < 0.2
            assert report["log_prob"]["p99"] < 0.06
            # Cached values are diagnostic here; critic fitting still uses the
            # full-window pass. Limits are 3.3% and 1.3% of its 150-unit scale.
            assert report["value"]["max"] < 5
            assert report["value"]["p99"] < 2
            assert report["kl"]["max"] < 0.02
            assert report["kl"]["p99"] < 0.005
            for name in ("actor_gradient", "critic_gradient"):
                assert report[name]["cosine"] > 0.999
                assert report[name]["relative_norm"] < 0.03
    finally:
        torch.use_deterministic_algorithms(previous)


def test_fixed_anchor_lanes_keep_counts_and_ports_through_resets() -> None:
    from test_065_loop import FakeShard
    from test_065_loop import exp as loop_exp
    from test_065_loop import trainer

    base = trainer(groups=1, development=True)
    shard = FakeShard(tuple(range(16)), length=13, timeout=True)
    anchors = {index: base.policy.frozen_copy() for index in range(3)}
    engine = loop_exp.RolloutEngine(((shard,),), base.policy, base.engine.sampler, anchors, fixed_anchor_lanes=True)
    engine.initialize()
    assert engine.anchor_counts == (2,)
    rollout = engine.collect(0)
    assert rollout.resets > 0
    for (match, _), episode in rollout.records.states.table.episodes.items():
        assigned = [side for side, player in enumerate(episode.players) if player.anchor is not None]
        assert assigned == ([match % 2] if match < 2 else [])
    assert sum(player.anchor is not None for state in engine.matches.values() for player in state.episode.players) == 2


def test_snapshot_weights_reuse_storage_only_after_the_last_episode_releases_it() -> None:
    from test_065_loop import trainer

    run = trainer(groups=1, anchor_rate=1)
    run.config = dataclasses.replace(run.config, snapshot_interval=1, snapshot_capacity=1)
    run.run_update()
    first = run.engine.anchors[1]
    addresses = tuple(parameter.data_ptr() for parameter in first.model.parameters())
    weights = {name: value.detach().clone() for name, value in first.model.state_dict().items()}

    def reset_with(anchor: int) -> None:
        run.engine.anchor_candidates = (anchor,)
        for shard in run.engine.groups[0]:
            for lane, match in enumerate(shard.match_ids):
                episode = run.engine.matches[match].episode.number + 1
                run.engine._reset(shard, lane, match, run.engine.clocks[0], episode)

    reset_with(1)
    run.run_update()
    assert run.engine.anchors[1] is first
    assert run.engine.anchors[2] is not first
    for name, value in weights.items():
        torch.testing.assert_close(first.model.state_dict()[name], value, rtol=0, atol=0)
    reset_with(2)
    run.run_update()
    assert 1 not in run.engine.anchors
    assert run.engine.anchors[3] is first
    assert tuple(parameter.data_ptr() for parameter in first.model.parameters()) == addresses
    for name, value in run.policy.model.state_dict().items():
        torch.testing.assert_close(first.model.state_dict()[name], value, rtol=0, atol=0)


def test_resume_rejects_a_changed_opponent_layout(tmp_path: Path) -> None:
    from test_065_loop import PROVENANCE
    from test_065_loop import exp as loop_exp
    from test_065_loop import trainer

    run = trainer(groups=1, anchor_rate=0)
    run.run_update()
    manifest = loop_exp.save_checkpoint(run, tmp_path, PROVENANCE)
    run.engine.fixed_anchor_lanes = True
    with pytest.raises(ValueError, match="configuration mismatch"):
        loop_exp.restore_checkpoint(run, manifest, PROVENANCE)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tick_graphs_keep_each_engines_queue_storage_separate() -> None:
    from test_065_loop import exp as loop_exp
    from test_065_loop import transformer_trainer

    run = transformer_trainer("cuda", development=True, compiled=True)
    run.engine.collect(0)
    previous = run.engine.rings[0]
    saved = previous.values.clone()
    shard = type(run.engine.groups[0][0])((0,), length=67, timeout=True)
    engine = loop_exp.RolloutEngine(((shard,),), run.policy, run.engine.sampler, {0: run.policy.frozen_copy()})
    engine.initialize()
    engine.collect(0)
    torch.testing.assert_close(previous.values, saved, rtol=0, atol=0)
    current = engine.rings[0]
    torch.testing.assert_close(current.controllers, run.policy.device_wire_actions(current.values), rtol=0, atol=0)


def test_verifier_restores_execution_mode_if_graph_cleanup_fails(tmp_path: Path) -> None:
    from test_065_loop import PROVENANCE
    from test_065_loop import ToyPolicy
    from test_065_loop import exp as loop_exp
    from test_065_loop import trainer

    class FailOncePolicy(ToyPolicy):
        fail_clear = False

        def clear_caches(self) -> None:
            if self.fail_clear:
                self.fail_clear = False
                raise RuntimeError("graph cleanup failed")
            super().clear_caches()

    base = trainer(groups=1, anchor_rate=0)
    policy = FailOncePolicy("cpu")
    policy.model.load_state_dict(base.policy.model.state_dict())
    engine = loop_exp.RolloutEngine(base.engine.groups, policy, base.engine.sampler, {0: policy.frozen_copy()})
    engine.initialize()
    run = loop_exp.Trainer(engine, base.config)
    run.run_update()
    policy.fail_clear = True
    with pytest.raises(RuntimeError, match="graph cleanup failed"):
        loop_exp.verify_checkpoint_resume(run, tmp_path, PROVENANCE)
    assert not torch.are_deterministic_algorithms_enabled() and not run.config.deterministic
    assert run.update == 1
