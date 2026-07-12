"""EMAWeights: closed-form recurrence, copy_to, state-dict round-trip, buffer verbatim-copy."""

import pytest
import torch

from hal.training.ema import EMAWeights


def _module() -> torch.nn.Module:
    net = torch.nn.Linear(3, 2, bias=True)
    # A non-parameter buffer so copy_to / load_state_dict exercise the buffer paths
    # (verbatim-copied, never EMA-averaged) in the shared-helper tests, not just params.
    net.register_buffer("running", torch.arange(2, dtype=torch.float32))
    return net


def test_closed_form_recurrence() -> None:
    # init shadow at w0, then drive the live module to a constant w and update k times.
    # shadow_k == d^k * w0 + (1 - d^k) * w.
    decay = 0.9
    net = _module()
    w0 = {name: p.detach().clone() for name, p in net.named_parameters()}
    ema = EMAWeights(net, decay=decay)

    const = 5.0
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(const)

    k = 7
    for _ in range(k):
        ema.update(net)

    factor = decay**k
    for name, p0 in w0.items():
        expected = factor * p0 + (1.0 - factor) * torch.full_like(p0, const)
        assert torch.allclose(ema.state_dict()[name], expected, atol=1e-6), name


def test_copy_to_is_idempotent() -> None:
    net = _module()
    ema = EMAWeights(net, decay=0.99)

    const = 3.0
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(const)
    for _ in range(5):
        ema.update(net)

    ema.copy_to(net)
    after_first = {name: p.detach().clone() for name, p in net.named_parameters()}
    ema.copy_to(net)
    for name, p in net.named_parameters():
        assert torch.equal(p, after_first[name]), name
        assert torch.equal(p, ema.state_dict()[name]), name


def test_state_dict_round_trip() -> None:
    net = _module()
    ema = EMAWeights(net, decay=0.99)
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(2.0)
    for _ in range(3):
        ema.update(net)

    saved = {name: t.clone() for name, t in ema.state_dict().items()}

    # mutate the shadows via further updates, then restore.
    with torch.no_grad():
        for p in net.parameters():
            p.fill_(-1.0)
    for _ in range(10):
        ema.update(net)
    assert not torch.equal(ema.state_dict()[next(iter(saved))], saved[next(iter(saved))])

    ema.load_state_dict(saved)
    for name, t in saved.items():
        assert torch.equal(ema.state_dict()[name], t), name


def test_load_state_dict_fails_on_key_mismatch() -> None:
    net = _module()
    ema = EMAWeights(net, decay=0.99)
    good = {name: t.clone() for name, t in ema.state_dict().items()}

    missing = dict(good)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError):
        ema.load_state_dict(missing)

    extra = dict(good)
    extra["nonexistent.param"] = torch.zeros(1)
    with pytest.raises(ValueError):
        ema.load_state_dict(extra)


def test_buffers_copied_verbatim_not_averaged() -> None:
    net = torch.nn.Linear(3, 2)
    net.register_buffer("stat", torch.zeros(2))
    ema = EMAWeights(net, decay=0.5)

    with torch.no_grad():
        net.get_buffer("stat").fill_(9.0)
    ema.update(net)

    # buffer takes the latest value verbatim; a 0.5-averaged value would be 4.5.
    assert torch.equal(ema.state_dict()["stat"], torch.full((2,), 9.0))
