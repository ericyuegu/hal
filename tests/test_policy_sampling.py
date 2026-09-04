import pytest
import torch

from hal.eval.policy_sampling import SlotGroupRng
from hal.eval.policy_sampling import sample_categorical
from hal.training.features import Context


def _context(slot_ids: list[int], resets: list[bool] | None = None) -> Context:
    count = len(slot_ids)
    return Context(
        features={"dummy": torch.zeros(count, 1)},
        ctx_pad=torch.zeros(count, dtype=torch.int64),
        slot_ids=torch.tensor(slot_ids),
        reset=None if resets is None else torch.tensor(resets),
    )


def test_sample_categorical_accepts_fixed_uniforms() -> None:
    logits = torch.zeros(3, 4)
    uniform = torch.tensor([0.0, 0.3, 0.99])

    assert sample_categorical(logits, argmax=False, uniform=uniform).tolist() == [0, 1, 3]
    assert sample_categorical(logits, argmax=True).tolist() == [0, 0, 0]


def test_slot_rng_is_independent_of_batch_order() -> None:
    first = SlotGroupRng(7, ("a", "b"))
    second = SlotGroupRng(7, ("a", "b"))
    first.begin(_context([10, 20]))
    second.begin(_context([20, 10]))

    assert torch.equal(first.uniforms("a"), second.uniforms("a").flip(0))


def test_slot_reset_starts_a_new_generation() -> None:
    rng = SlotGroupRng(7, ("a",))
    rng.begin(_context([10]))
    first = rng.uniforms("a")
    rng.begin(_context([10], [True]))

    assert not torch.equal(rng.uniforms("a"), first)


def test_slot_rng_rejects_unknown_groups() -> None:
    rng = SlotGroupRng(7, ("a",))
    rng.begin(_context([10]))
    with pytest.raises(ValueError, match="unknown group"):
        rng.uniforms("missing")


def test_slot_rng_preserves_the_experiment_random_stream() -> None:
    group_names = ("buttons", "main_stick", "c_stick", "triggers")
    rng = SlotGroupRng(0x123456789ABCDEF, group_names)
    rng.begin(_context([9, 2, 44]))

    values = {name: rng.uniforms(name).tolist() for name in group_names}

    assert values["buttons"] == pytest.approx([0.826551616191864, 0.2559979557991028, 0.3905690014362335])
    assert values["main_stick"] == pytest.approx([0.02666299045085907, 0.08389616012573242, 0.7409748435020447])
    assert values["c_stick"] == pytest.approx([0.580507755279541, 0.7949469685554504, 0.6630955338478088])
    assert values["triggers"] == pytest.approx([0.65818190574646, 0.10028249770402908, 0.5736757516860962])
