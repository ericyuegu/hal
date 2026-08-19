"""Contract tests for experiment 036's residual-readout-only treatment."""

import importlib.util
import sys
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

_PATH = Path(__file__).resolve().parents[2] / "experiments" / "036_residual_recursive_gru.py"
_SPEC = importlib.util.spec_from_file_location("test_exp036", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
exp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exp
_SPEC.loader.exec_module(exp)


def _decoder():
    torch.manual_seed(0)
    cfg = exp.TrainConfig()
    return exp.CausalTemporalDecoder(cfg, exp.StructuredControllerCodec(cfg.action_embed_dim)), cfg


def _inputs(cfg, batch=3, length=2):
    hidden = torch.randn(batch, length, cfg.d_model)
    observed = torch.stack([torch.randint(vocab, (batch, length)) for vocab in exp.GROUP_VOCABS], dim=-1)
    targets = torch.stack(
        [torch.randint(vocab, (batch, length, len(cfg.head_offsets))) for vocab in exp.GROUP_VOCABS], dim=-1
    )
    return hidden, observed, targets


def _assert_logits_close(left, right):
    assert len(left) == len(right)
    for left_step, right_step in zip(left, right, strict=True):
        for name in left_step:
            torch.testing.assert_close(left_step[name], right_step[name], equal_nan=True)


def test_identity_counts_and_frozen_defaults() -> None:
    cfg = exp.TrainConfig()
    model = exp.GPT(cfg)
    assert cfg.experiment_id == "036_residual_gru_v1"
    assert cfg.decoder_arch_version == 5
    assert cfg.batch_size == 512 and cfg.max_steps == 16_384 and cfg.seed == 0
    assert sum(parameter.numel() for parameter in model.temporal.cell.parameters()) == 99_072
    assert sum(parameter.numel() for parameter in model.parameters()) == 14_889_967


def test_same_seed_parameters_are_bitwise_identical_to_035() -> None:
    path035 = _PATH.with_name("035_recursive_gru.py")
    spec035 = importlib.util.spec_from_file_location("test_exp035_for_036_identity", path035)
    assert spec035 is not None and spec035.loader is not None
    exp035 = importlib.util.module_from_spec(spec035)
    saved_modules = {name: sys.modules.get(name) for name in (spec035.name, "hal_exp026_for_035")}
    try:
        sys.modules[spec035.name] = exp035
        spec035.loader.exec_module(exp035)
        torch.manual_seed(0)
        model035 = exp035.GPT(exp035.TrainConfig())
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    torch.manual_seed(0)
    model036 = exp.GPT(exp.TrainConfig())
    state035 = model035.state_dict()
    state036 = model036.state_dict()
    assert state035.keys() == state036.keys()
    for name in state035:
        assert state035[name].shape == state036[name].shape
        torch.testing.assert_close(state035[name], state036[name], rtol=0, atol=0)


def test_teacher_forcing_matches_stepwise_at_every_prefix() -> None:
    decoder, cfg = _decoder()
    hidden, observed, targets = _inputs(cfg)
    teacher = decoder.teacher_forced_logits(hidden, observed, targets)
    stepwise = decoder.forced_stepwise_logits(
        hidden.reshape(-1, 1, cfg.d_model),
        observed.reshape(-1, exp.N_GROUPS),
        targets.reshape(-1, len(cfg.head_offsets), exp.N_GROUPS),
    )
    reshaped = [{name: value.view(*hidden.shape[:2], -1) for name, value in step.items()} for step in stepwise]
    _assert_logits_close(teacher, reshaped)


def test_temporal_target_causality() -> None:
    decoder, cfg = _decoder()
    hidden, observed, targets = _inputs(cfg, batch=2, length=1)
    changed = targets.clone()
    changed[:, :, 5:] = torch.stack(
        [torch.randint(vocab, changed[:, :, 5:, group].shape) for group, vocab in enumerate(exp.GROUP_VOCABS)], dim=-1
    )
    before = decoder.teacher_forced_logits(hidden, observed, targets)
    after = decoder.teacher_forced_logits(hidden, observed, changed)
    _assert_logits_close(before[:5], after[:5])


def test_zeroed_cell_readout_is_exactly_rmsnorm_token() -> None:
    decoder, cfg = _decoder()
    for parameter in decoder.cell.parameters():
        nn.init.zeros_(parameter)
    hidden, observed, targets = _inputs(cfg, batch=2, length=1)
    previous = torch.cat((observed[:, :, None], targets[..., :-1, :]), dim=2)
    tokens = decoder._tokens(hidden, previous)
    states = decoder.teacher_forced_states(hidden, observed, targets)
    torch.testing.assert_close(states, exp.decoder_rmsnorm(tokens))


def test_only_raw_hidden_state_recurs() -> None:
    decoder, cfg = _decoder()

    class RecordingCell(nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, token, state):
            self.inputs.append(state.detach().clone())
            return state + 2.0

    recorder = RecordingCell()
    decoder.cell = recorder
    hidden, observed, targets = _inputs(cfg, batch=1, length=1)
    decoder.teacher_forced_states(hidden, observed, targets)
    torch.testing.assert_close(recorder.inputs[0], torch.zeros_like(recorder.inputs[0]))
    torch.testing.assert_close(recorder.inputs[1], torch.full_like(recorder.inputs[1], 2.0))


def test_fresh_state_and_batch_isolation() -> None:
    decoder, cfg = _decoder()
    hidden, observed, targets = _inputs(cfg, batch=2, length=1)
    first = decoder.forced_stepwise_logits(hidden, observed[:, -1], targets[:, -1])
    decoder.forced_stepwise_logits(hidden * 5, observed[:, -1], targets[:, -1])
    repeated = decoder.forced_stepwise_logits(hidden, observed[:, -1], targets[:, -1])
    _assert_logits_close(first, repeated)
    single = decoder.forced_stepwise_logits(hidden[:1], observed[:1, -1], targets[:1, -1])
    _assert_logits_close([{name: value[:1] for name, value in step.items()} for step in first], single)


@pytest.mark.parametrize("horizon", [4, 6])
def test_h4_h6_eager_compiled_and_legality(horizon: int) -> None:
    decoder, cfg = _decoder()
    hidden, observed, _ = _inputs(cfg, batch=2, length=1)
    uniforms = torch.rand(horizon, exp.N_GROUPS, hidden.shape[0])
    offsets = cfg.head_offsets[:horizon]
    eager = decoder.sample_indices(hidden, observed[:, -1], offsets, argmax=False, uniforms=uniforms)
    compiled = torch.compile(decoder.sample_indices, backend="eager")
    actual = compiled(hidden, observed[:, -1], offsets, argmax=False, uniforms=uniforms)
    torch.testing.assert_close(actual, eager)
    assert bool(decoder.codec.button_valid_for_trigger[actual[..., exp.TRIG_G], actual[..., exp.BUTTONS_G]].all())


def test_checkpoint_config_and_optimizer_contracts() -> None:
    cfg = exp.TrainConfig()
    assert exp.config_from_state(asdict(cfg)) == cfg
    for rejected in (exp._TrainConfig035(), exp.base035._TrainConfig026()):
        with pytest.raises(ValueError, match="experiment_id|not experiment 036"):
            exp.config_from_state(asdict(rejected))
    model = exp.GPT(cfg)
    optimizer = exp.make_optimizer(model, cfg)
    owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(owned) == sum(1 for _ in model.parameters())
    assert len({id(parameter) for parameter in owned}) == len(owned)


def test_production_and_smoke_config_isolation() -> None:
    exp.validate_config(exp.TrainConfig())
    exp.validate_config(replace(exp.TrainConfig(), max_steps=100, grad_accum_steps=8, push_to_r2=False))
    with pytest.raises(ValueError, match="production 036 config changed"):
        exp.validate_config(replace(exp.TrainConfig(), adam_lr=1e-3))
    with pytest.raises(ValueError, match="smoke 036 config changed"):
        exp.validate_config(replace(exp.TrainConfig(), max_steps=100, aux_loss_weight=0.5))
    wrong = asdict(exp.TrainConfig())
    wrong["experiment_id"] = "035_recursive_gru_v1"
    with pytest.raises(ValueError, match="experiment_id"):
        exp.config_from_state(wrong)
