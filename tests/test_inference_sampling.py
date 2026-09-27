import pytest
import torch

from hal.inference.sampling import StreamGroupRng


def _sampling_batch(stream_ids: list[int], generations: list[int] | None = None) -> tuple[list[int], list[int]]:
    return stream_ids, [0] * len(stream_ids) if generations is None else generations


def test_stream_rng_is_independent_of_batch_order() -> None:
    first = StreamGroupRng(7, ("a", "b"))
    second = StreamGroupRng(7, ("a", "b"))
    first.begin(*_sampling_batch([10, 20]))
    second.begin(*_sampling_batch([20, 10]))

    assert torch.equal(first.uniforms("a"), second.uniforms("a").flip(0))


def test_stream_reset_starts_a_new_generation() -> None:
    rng = StreamGroupRng(7, ("a",))
    rng.begin(*_sampling_batch([10]))
    first = rng.uniforms("a")
    rng.begin(*_sampling_batch([10], [1]))

    assert not torch.equal(rng.uniforms("a"), first)


def test_inactive_stream_does_not_advance_its_random_stream() -> None:
    mixed = StreamGroupRng(7, ("a",))
    mixed.begin(*_sampling_batch([10, 20]))
    mixed.uniforms("a", [True, False])
    second = mixed.uniforms("a", [True, True])

    slot_10 = StreamGroupRng(7, ("a",))
    slot_10.begin(*_sampling_batch([10]))
    slot_10.uniforms("a")
    slot_20 = StreamGroupRng(7, ("a",))
    slot_20.begin(*_sampling_batch([20]))

    assert second[0] == slot_10.uniforms("a")[0]
    assert second[1] == slot_20.uniforms("a")[0]


def test_stream_rng_rejects_unknown_groups() -> None:
    rng = StreamGroupRng(7, ("a",))
    rng.begin(*_sampling_batch([10]))
    with pytest.raises(ValueError, match="unknown group"):
        rng.uniforms("missing")


def test_stream_rng_preserves_the_experiment_random_stream() -> None:
    group_names = ("buttons", "main_stick", "c_stick", "triggers")
    rng = StreamGroupRng(0x123456789ABCDEF, group_names)
    rng.begin(*_sampling_batch([9, 2, 44]))

    values = {name: rng.uniforms(name).tolist() for name in group_names}

    assert values["buttons"] == pytest.approx([0.826551616191864, 0.2559979557991028, 0.3905690014362335])
    assert values["main_stick"] == pytest.approx([0.02666299045085907, 0.08389616012573242, 0.7409748435020447])
    assert values["c_stick"] == pytest.approx([0.580507755279541, 0.7949469685554504, 0.6630955338478088])
    assert values["triggers"] == pytest.approx([0.65818190574646, 0.10028249770402908, 0.5736757516860962])


def test_release_and_resets_discard_old_counters_without_repeating_draws() -> None:
    rng = StreamGroupRng(7, ("a", "b"))
    rng.begin([10, 20], [0, 0])
    first = rng.uniforms("a")
    rng.release(10)
    assert all(stream_id == 20 for stream_id, _, _, _ in rng.state())
    rng.begin([10, 20], [1, 0])
    second = rng.uniforms("a")
    assert second[0] != first[0]
    assert rng.generations[10] == 1
    for generation in range(2, 102):
        rng.begin([10], [generation])
        rng.uniforms("a")
    assert len(rng.state()) == 4
    assert rng.generations[20] == 0


def test_restore_sampling_continues_identical_draws_and_rejects_partial_state() -> None:
    control = StreamGroupRng(7, ("a", "b"))
    control.begin([1, 9], [0, 0])
    control.uniforms("a")
    restored = StreamGroupRng(7, ("a", "b"))
    restored.restore(generations=tuple(control.generations.items()), counters=control.state())
    restored.begin([9, 1], [0, 0])
    assert torch.equal(control.uniforms("a"), restored.uniforms("a").flip(0))
    previous = restored.state()
    with pytest.raises(ValueError, match="every group"):
        restored.restore(generations=((1, 0),), counters=((1, 0, "a", 3),))
    assert restored.state() == previous


def test_released_arbitrary_ids_leave_no_sampling_tombstones() -> None:
    rng = StreamGroupRng(7, ("a",))
    for stream_id in range(1000):
        rng.begin([stream_id], [3])
        rng.uniforms("a")
        rng.release(stream_id)
    assert rng.generations == {}
    assert rng.state() == ()
    assert rng.stream_ids == ()
    rng.begin([0], [4])
    repeated = StreamGroupRng(7, ("a",))
    repeated.begin([0], [4])
    assert torch.equal(rng.uniforms("a"), repeated.uniforms("a"))
