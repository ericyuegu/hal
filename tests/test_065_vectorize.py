"""Dense policy graph ownership, anchors and resume invariants."""

import dataclasses
from pathlib import Path

import pytest
import torch
from torch import Tensor


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

    run = transformer_trainer("cuda", development=True)
    shard = type(run.engine.groups[0][0])((3, 0), length=67, timeout=True, device="cuda")
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
    for match, state in engine.matches.items():
        assigned = [side for side, player in enumerate(state.episode.players) if player.anchor is not None]
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

    run = transformer_trainer("cuda", development=True)
    run.engine.collect(0)
    previous = run.engine.rings[0]
    saved = previous.values.clone()
    shard = type(run.engine.groups[0][0])((0,), length=67, timeout=True, device="cuda")
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


def test_tick_layout_rejects_duplicate_streams() -> None:
    from test_065_loop import exp

    with pytest.raises(ValueError, match="distinct"):
        exp.TickLayout((0, 0), (0, 0))
