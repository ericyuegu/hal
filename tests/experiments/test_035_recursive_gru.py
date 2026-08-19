"""Contract tests for experiment 035's mixer-only GRU intervention."""

import importlib.util
import sys
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parents[2] / "experiments" / "035_recursive_gru.py"
_SPEC = importlib.util.spec_from_file_location("test_exp035", _PATH)
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
        assert left_step.keys() == right_step.keys()
        for name in left_step:
            torch.testing.assert_close(left_step[name], right_step[name], equal_nan=True)


def test_defaults_are_exact_eligible_recipe_and_identity() -> None:
    cfg = exp.TrainConfig()
    reference = exp._TrainConfig026()
    assert cfg.batch_size == 512
    assert cfg.max_steps == 16_384
    assert cfg.seed == 0
    assert cfg.data_root == reference.data_root == "data/processed/ranked-anonymized-1/mds-policy-v7"
    assert cfg.head_offsets == (1, 2, 3, 4, 5, 6, 9, 12, 16, 20)
    assert cfg.experiment_id == "035_recursive_gru_v1"
    assert cfg.decoder_arch_version == 4
    assert cfg.temporal_layers == 1


def test_exact_cell_and_model_parameter_counts() -> None:
    model = exp.GPT(exp.TrainConfig())
    assert isinstance(model.temporal.cell, torch.nn.GRUCell)
    assert not hasattr(model.temporal, "blocks")
    assert sum(parameter.numel() for parameter in model.temporal.cell.parameters()) == 99_072
    assert sum(parameter.numel() for parameter in model.parameters()) == 14_889_967


def test_teacher_forcing_matches_stepwise_for_every_prefix() -> None:
    decoder, cfg = _decoder()
    hidden, observed, targets = _inputs(cfg)
    teacher = decoder.teacher_forced_logits(hidden, observed, targets)
    flat_hidden = hidden.reshape(-1, 1, cfg.d_model)
    flat_observed = observed.reshape(-1, exp.N_GROUPS)
    flat_targets = targets.reshape(-1, len(cfg.head_offsets), exp.N_GROUPS)
    stepwise = decoder.forced_stepwise_logits(flat_hidden, flat_observed, flat_targets)
    stepwise = [{name: logits.view(*hidden.shape[:2], -1) for name, logits in step.items()} for step in stepwise]
    _assert_logits_close(teacher, stepwise)


def test_target_temporal_causality() -> None:
    decoder, cfg = _decoder()
    hidden, observed, targets = _inputs(cfg, batch=2, length=1)
    changed = targets.clone()
    changed[:, :, 5:, :] = torch.stack(
        [torch.randint(vocab, changed[:, :, 5:, group].shape) for group, vocab in enumerate(exp.GROUP_VOCABS)], dim=-1
    )
    before = decoder.teacher_forced_logits(hidden, observed, targets)
    after = decoder.teacher_forced_logits(hidden, observed, changed)
    _assert_logits_close(before[:5], after[:5])


def test_each_decode_has_fresh_zero_state_and_batch_isolation() -> None:
    decoder, cfg = _decoder()
    hidden, observed, targets = _inputs(cfg, batch=2, length=1)
    first = decoder.forced_stepwise_logits(hidden, observed[:, -1], targets[:, -1])
    decoder.forced_stepwise_logits(hidden * 7, observed[:, -1], targets[:, -1])
    repeated = decoder.forced_stepwise_logits(hidden, observed[:, -1], targets[:, -1])
    _assert_logits_close(first, repeated)
    single = decoder.forced_stepwise_logits(hidden[:1], observed[:1, -1], targets[:1, -1])
    _assert_logits_close([{name: value[:1] for name, value in step.items()} for step in first], single)


@pytest.mark.parametrize("horizon", [4, 6])
def test_h4_h6_eager_and_compiled_parity(horizon: int) -> None:
    decoder, cfg = _decoder()
    hidden, observed, _ = _inputs(cfg, batch=2, length=1)
    uniforms = torch.rand(horizon, exp.N_GROUPS, hidden.shape[0])
    offsets = cfg.head_offsets[:horizon]
    eager = decoder.sample_indices(hidden, observed[:, -1], offsets, argmax=False, uniforms=uniforms)
    compiled = torch.compile(decoder.sample_indices, backend="eager")
    actual = compiled(hidden, observed[:, -1], offsets, argmax=False, uniforms=uniforms)
    torch.testing.assert_close(actual, eager)
    assert bool(decoder.codec.button_valid_for_trigger[actual[..., exp.TRIG_G], actual[..., exp.BUTTONS_G]].all())


def test_checkpoint_identity_round_trip_and_rejection() -> None:
    cfg = exp.TrainConfig()
    assert exp.config_from_state(asdict(cfg)) == cfg
    with pytest.raises(ValueError, match="not experiment 035"):
        exp.config_from_state(asdict(exp._TrainConfig026()))
    wrong = asdict(cfg)
    wrong["experiment_id"] = "034_rank_weighted_bc_v1"
    with pytest.raises(ValueError, match="experiment_id"):
        exp.config_from_state(wrong)


def test_optimizer_owns_every_parameter_once() -> None:
    cfg = exp.TrainConfig()
    model = exp.GPT(cfg)
    optimizer = exp.make_optimizer(model, cfg)
    owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(owned) == sum(1 for _ in model.parameters())
    assert len({id(parameter) for parameter in owned}) == len(owned)
    cell_ids = {id(parameter) for parameter in model.temporal.cell.parameters()}
    assert cell_ids <= {id(parameter) for parameter in owned}


def test_production_and_smoke_config_isolation() -> None:
    exp.validate_config(exp.TrainConfig())
    exp.validate_config(replace(exp.TrainConfig(), max_steps=100, grad_accum_steps=2, push_to_r2=False))
    with pytest.raises(ValueError, match="production 035 config changed"):
        exp.validate_config(replace(exp.TrainConfig(), adam_lr=1e-3))
    with pytest.raises(ValueError, match="smoke 035 config changed"):
        exp.validate_config(replace(exp.TrainConfig(), max_steps=100, aux_loss_weight=0.5))
    with pytest.raises(ValueError, match="exactly one GRUCell"):
        exp.validate_config(replace(exp.TrainConfig(), temporal_layers=2))
