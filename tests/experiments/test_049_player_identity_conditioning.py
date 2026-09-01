"""Contracts for O49's standalone O43 player-conditioning ablation."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
import torch

from hal.data.schema import Rank
from hal.training.features import A_DIM
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.training.player_identity import PlayerVocabulary


def _load() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "experiments" / "049_player_identity_conditioning.py"
    name = "test_exp049"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp = _load()


def _vocabulary() -> PlayerVocabulary:
    return PlayerVocabulary(("AA#1", "aa#1"))


def _cfg(**overrides) -> exp.TrainConfig:
    vocabulary = _vocabulary()
    values = dict(
        d_model=32,
        n_layers=1,
        n_heads=4,
        L_ctx=4,
        temporal_d_model=32,
        temporal_layers=1,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=64,
        batch_size=2,
        grad_accum_steps=1,
        reservoir_capacity=4,
        warmup_steps=1,
        max_steps=2,
        compile_trunk=False,
        compile_temporal=False,
        inference_mode="eager",
        num_workers=0,
        push_to_r2=False,
        player_vocab_size=vocabulary.size,
        player_vocab_sha256=vocabulary.sha256,
        player_sidecar_sha256="",
    )
    return exp.TrainConfig(**{**values, **overrides})


def _models(cfg: exp.TrainConfig) -> tuple[torch.nn.Module, exp.GPT]:
    torch.manual_seed(19)
    baseline = exp._BaseGPT(cfg).eval()
    torch.manual_seed(19)
    model = exp.GPT(cfg, vocabulary=_vocabulary()).eval()
    return baseline, model


def test_protocol_defaults_are_frozen_and_ego_only() -> None:
    cfg = exp.TrainConfig()
    assert cfg.player_embed_dim == 32
    assert cfg.player_vocab_size == 21_181
    assert len(cfg.train_source_names) == 44
    assert cfg.eval_player_rank == "MASTER"
    assert cfg.treatment == "conditioned"
    assert "ego_player_id" in exp.PLAYER_PROJECTION.columns
    assert "opp_player_id" not in exp.PLAYER_PROJECTION.columns
    assert "ego-id32-conditioned" in exp.model_tag(cfg)
    exp.validate_config(cfg, require_frozen_artifact=True)

    control = replace(cfg, treatment="control")
    differences = {name for name, value in asdict(cfg).items() if value != asdict(control)[name]}
    assert differences == {"treatment"}
    with pytest.raises(ValueError, match="frozen value"):
        exp.validate_config(replace(cfg, player_vocab_size=cfg.player_vocab_size + 1))
    with pytest.raises(ValueError, match="frozen O49 sidecar"):
        exp.validate_config(replace(cfg, player_sidecar_sha256="a" * 64), require_frozen_artifact=True)


@pytest.mark.parametrize("treatment", ["control", "conditioned"])
def test_zero_initialized_path_exactly_matches_o43(treatment: str) -> None:
    cfg = _cfg(treatment=treatment)
    baseline, model = _models(cfg)
    context = exp.synthetic_context(cfg, 2, torch.device("cpu"))
    context = exp._condition_context(context, _vocabulary().id_for_code("AA#1"))

    with torch.no_grad():
        expected = baseline.context_tokens(context.features)
        actual = model.context_tokens(context.features)

    assert torch.equal(actual, expected)
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    expected_delta = cfg.player_vocab_size * cfg.player_embed_dim + cfg.player_embed_dim * cfg.d_model
    assert model_parameters - baseline_parameters == expected_delta


def test_conditioned_ids_can_change_tokens_but_control_cannot() -> None:
    conditioned_cfg = _cfg(treatment="conditioned")
    baseline, conditioned = _models(conditioned_cfg)
    control = exp.GPT(replace(conditioned_cfg, treatment="control"), vocabulary=_vocabulary()).eval()
    control.load_state_dict(conditioned.state_dict())
    context = exp.synthetic_context(conditioned_cfg, 2, torch.device("cpu"))
    first = exp._condition_context(context, _vocabulary().id_for_code("AA#1"))
    second = exp._condition_context(context, _vocabulary().id_for_code("aa#1"))
    with torch.no_grad():
        conditioned.player_projection.weight.normal_()
        control.player_projection.weight.copy_(conditioned.player_projection.weight)
        first_tokens = conditioned.context_tokens(first.features)
        second_tokens = conditioned.context_tokens(second.features)
        control_first = control.context_tokens(first.features)
        control_second = control.context_tokens(second.features)
        baseline_tokens = baseline.context_tokens(context.features)

    assert not torch.equal(first_tokens, second_tokens)
    assert torch.equal(control_first, control_second)
    assert torch.equal(control_first, baseline_tokens)


def test_gradients_reach_only_a_conditioned_non_mask_identity() -> None:
    cfg = _cfg(treatment="conditioned")
    _, model = _models(cfg)
    with torch.no_grad():
        model.player_projection.weight.normal_()
    context = exp.synthetic_context(cfg, 2, torch.device("cpu"))
    context = exp._condition_context(context, _vocabulary().id_for_code("AA#1"))

    model.context_tokens(context.features).square().mean().backward()

    assert model.player_projection.weight.grad is not None
    assert torch.count_nonzero(model.player_projection.weight.grad)
    assert model.player_embedding.weight.grad is not None
    assert torch.count_nonzero(model.player_embedding.weight.grad[4])
    assert torch.count_nonzero(model.player_embedding.weight.grad[0]) == 0


def test_opponent_identity_is_rejected() -> None:
    cfg = _cfg()
    _, model = _models(cfg)
    context = exp.synthetic_context(cfg, 1, torch.device("cpu"))
    features = dict(context.features)
    features["opp_player_id"] = torch.zeros_like(features["ego_player_id"])
    with pytest.raises(ValueError, match="opponent"):
        model.context_tokens(features)
    with pytest.raises(ValueError, match="opponent"):
        exp._condition_context(Context(features, context.ctx_pad), 1)


def test_inference_accepts_exact_code_or_rank_and_rejects_oov() -> None:
    cfg = _cfg()
    _, model = _models(cfg)

    professional = exp.BF16Inference(model, cfg, player_code=" AA#1 ", compiled=False)
    ranked = exp.BF16Inference(model, cfg, player_rank=Rank.DIAMOND, compiled=False)
    default = exp.BF16Inference(model, cfg, compiled=False)

    assert professional.player_id == _vocabulary().id_for_code("AA#1")
    assert ranked.player_id == int(Rank.DIAMOND)
    assert default.player_id == int(Rank.MASTER)
    with pytest.raises(KeyError, match="absent"):
        exp.BF16Inference(model, cfg, player_code="Aa#1", compiled=False)
    with pytest.raises(ValueError, match="not both"):
        exp.BF16Inference(model, cfg, player_code="AA#1", player_rank=Rank.MASTER, compiled=False)


def test_checkpoint_carries_the_exact_ordered_vocabulary() -> None:
    cfg = _cfg()
    _, model = _models(cfg)

    restored = exp._vocabulary_from_state(model.state_dict())

    assert restored.codes == ("AA#1", "aa#1")
    assert restored.sha256 == cfg.player_vocab_sha256


def test_identity_validation_reports_each_ego_without_opponent_ids() -> None:
    cfg = _cfg()
    _, model = _models(cfg)
    context = exp.synthetic_context(cfg, 2, torch.device("cpu"))
    features = dict(context.features)
    features["ego_player_id"] = torch.tensor(
        [
            [_vocabulary().id_for_code("AA#1")] * cfg.L_ctx,
            [_vocabulary().id_for_code("aa#1")] * cfg.L_ctx,
        ]
    )
    batch = TrainBatch(
        Context(features, context.ctx_pad),
        torch.zeros(2, cfg.sample_chunk_length, A_DIM),
    )

    report = exp.identity_validation_metrics(model, [batch], cfg)

    assert report["total_replays"] == 1
    assert report["total_windows"] == 2
    assert report["opponent_identity_conditioned"] is False
    identities = report["identities"]
    assert [item["identity"] for item in identities] == ["AA#1", "aa#1"]
    assert all(item["windows"] == 1 for item in identities)
    assert all(item["valid_prefixes"] == cfg.L_ctx for item in identities)
