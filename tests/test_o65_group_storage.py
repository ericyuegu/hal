"""Rotating groups reuse captured storage without carrying another group's KV state."""

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import torch
from test_065_loop import PROVENANCE
from test_065_loop import FakeShard
from test_065_loop import exp
from test_065_loop import transformer_trainer


class PerGroupEngine(exp.RolloutEngine):
    """Control with an independent ring, table, cache and graph set for every group."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self.independent: dict[int, dict[tuple[int, ...], Any]] = {group: {} for group in range(len(self.groups))}

    def activate_rollout_storage(self) -> Any:
        self.rollout_workspaces = self.independent[self.cursor]
        return super().activate_rollout_storage()


def rotating_trainer(*, control: bool = False, device: str = "cpu", head_start: int = 0) -> Any:
    source = transformer_trainer(device, development=True, episode_length=197)
    groups = tuple(
        (FakeShard(tuple(range(4 * group, 4 * group + 4)), length=197, timeout=True, device=device),)
        for group in range(4)
    )
    cls = PerGroupEngine if control else exp.RolloutEngine
    engine = cls(groups, source.policy, source.engine.sampler, {0: source.reference})
    engine.initialize()
    config = dataclasses.replace(source.config, head_start_frames=head_start, snapshot_interval=64)
    return exp.Trainer(engine, config)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_four_groups_match_independent_caches_and_reuse_storage(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    control, run = rotating_trainer(control=True, device=device), rotating_trainer(device=device)
    storage = None
    for update in range(5):
        if update == 2:
            # Promotion truncates this slot in inactive groups, which reset on their next visit.
            for engine in (control.engine, run.engine):
                engine.truncate_slot(1)
                with torch.no_grad():
                    next(engine.anchors[1].model.parameters()).add_(0.005)
        expected = control.engine.collect(update)
        actual = run.engine.collect(update)
        assert actual.group == expected.group == update % 4
        for name in ("valid", "duration", "rewards", "bootstrap"):
            torch.testing.assert_close(getattr(actual.table, name), getattr(expected.table, name), atol=0, rtol=0)
        # Invalid rows may retain outputs from the last group that used the workspace.
        rows = actual.ids
        for name in ("integers", "actions", "behavior", "values"):
            torch.testing.assert_close(
                getattr(actual.table, name)[rows], getattr(expected.table, name)[rows], atol=0, rtol=0
            )
        assert exp._state_hash(run.engine.snapshot()) == exp._state_hash(control.engine.snapshot())
        current = actual.table, actual.table.store, run.engine.rings[actual.group]
        if storage is None:
            storage = current
        assert current == storage
        assert len(run.policy.tick_caches) == 1
        assert len(control.policy.tick_caches) == min(update + 1, 4)
        stacks = tuple(stack for stack in run.engine.anchor_stacks if stack is not None)
        if device == "cuda":
            assert len(stacks) == 4 and len(set(stacks)) == 1
            assert len(stacks[0].tick_caches) == 1
        for group, store in enumerate(run.engine.stores):
            if group != actual.group:
                assert len(store.records) <= 129
                assert run.engine.tables[group] is None


@pytest.mark.parametrize("group", range(4))
def test_collection_rejects_any_misaligned_group_clock_before_mutation(group: int) -> None:
    engine = rotating_trainer().engine
    engine.clocks[group] = 2
    before = exp._state_hash(engine.snapshot())
    with pytest.raises(ValueError, match="128-frame block boundary"):
        engine.collect(0)
    assert exp._state_hash(engine.snapshot()) == before
    assert not engine.rollout_workspaces


def test_four_group_head_start_and_resume_preserve_next_batch_and_update(tmp_path: Path) -> None:
    run = rotating_trainer(head_start=19)
    run.head_start()
    frames = run.engine.sampler.schedule.rollout_frames
    assert run.engine.clocks == [frames] * 4
    run.run_update()
    saved = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    expected, _ = run.run_update()
    restored = rotating_trainer(head_start=19)
    # Restore into an engine which has already used the shared workspace.
    restored.engine.collect(0)
    exp.restore_checkpoint(restored, saved, PROVENANCE)
    assert restored.engine.clocks == [2 * frames, frames, frames, frames]
    actual, _ = restored.run_update()
    assert exp._batch_hash(actual) == exp._batch_hash(expected)
    for value, reference in zip(restored.policy.model.parameters(), run.policy.model.parameters(), strict=True):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_four_cuda_groups_never_recapture_after_startup() -> None:
    run = rotating_trainer(device="cuda")
    run.capture_startup(rollout_only=True)
    graphs = dict(run.policy.tick_graphs)
    restarts = dict(run.policy.restart_graphs)
    caches = dict(run.policy.tick_caches)
    for update in range(5):
        run.engine.collect(update)
        assert run.policy.tick_graphs == graphs
        assert run.policy.restart_graphs == restarts
        assert run.policy.tick_caches == caches
    assert len(caches) == 1
    assert run.policy.post_startup_captures == 0
