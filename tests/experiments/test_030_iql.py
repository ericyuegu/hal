"""Contracts for the faithful double-Q IQL experiment 030."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import Context
from hal.training.features import TrainBatch


def _load():
    path = Path(__file__).resolve().parents[2] / "experiments" / "030_iql.py"
    spec = importlib.util.spec_from_file_location("test_exp030", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp = _load()


def _cfg(**overrides):
    values = dict(
        d_model=32,
        n_layers=1,
        n_heads=4,
        L_ctx=4,
        temporal_d_model=32,
        temporal_layers=1,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=32,
        critic_d_model=32,
        critic_layers=1,
        critic_heads=4,
        critic_hidden_dim=32,
        batch_size=2,
        reservoir_capacity=4,
        max_steps=2,
        compile_trunk=False,
        compile_temporal=False,
        num_workers=0,
        push_to_r2=False,
    )
    return exp.TrainConfig(**{**values, **overrides})


def _actions(batch: int, length: int, generator: torch.Generator) -> torch.Tensor:
    actions = torch.empty(batch, length, A_DIM)
    actions[..., :4] = torch.rand(batch, length, 4, generator=generator) * 2 - 1
    actions[..., 4:6] = torch.rand(batch, length, 2, generator=generator)
    actions[..., 6:] = torch.randint(0, 2, (batch, length, A_DIM - 6), generator=generator).float()
    return actions


def _batch(cfg, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    context = exp.synthetic_context(cfg, cfg.batch_size, torch.device("cpu"))
    context = Context(features=context.features, ctx_pad=torch.tensor([0, 1][: cfg.batch_size]))
    target = _actions(cfg.batch_size, 4, generator)
    extended = {
        name: torch.cat((value, value[:, -1:].expand(-1, 4)), dim=1) for name, value in context.features.items()
    }
    for channel, values in zip(ACTION_CHANNELS, target.unbind(-1), strict=True):
        extended[f"ego_{channel}"][:, -4:] = values
    return exp.IQLBatch(
        batch=TrainBatch(context=context, target=target),
        extended_context=Context(features=extended, ctx_pad=context.ctx_pad),
        rewards=torch.randn(cfg.batch_size, 4, generator=generator),
        returns=torch.randn(cfg.batch_size, generator=generator),
        continuation=torch.tensor([1.0, 0.0][: cfg.batch_size]),
    )


def test_defaults_freeze_faithful_iql_treatment() -> None:
    cfg = exp.TrainConfig()
    assert cfg.head_offsets == (1, 2, 3, 4)
    assert cfg.sample_chunk_length == cfg.exec_horizon == 4
    assert cfg.batch_size == 512 and cfg.max_steps == 2**14
    assert cfg.q_objective == "scalar"
    assert (cfg.iql_expectile, cfg.iql_discount, cfg.iql_temperature, cfg.iql_weight_max) == (0.7, 0.99, 3.0, 100.0)
    assert (cfg.target_tau, cfg.learning_rate) == (0.005, 3e-4)
    assert (cfg.critic_d_model, cfg.critic_layers) == (256, 4)


def test_actor_q1_q2_and_value_have_no_shared_parameters() -> None:
    model = exp.TrainingModel(_cfg())
    identities = exp.parameter_id_sets(model)
    for index, left in enumerate(identities):
        for right in tuple(identities)[index + 1 :]:
            assert identities[left].isdisjoint(identities[right])
    assert not any(parameter.requires_grad for parameter in model.target_q1.parameters())
    assert not any(parameter.requires_grad for parameter in model.target_q2.parameters())
    for online, target in ((model.q1, model.target_q1), (model.q2, model.target_q2)):
        for expected, actual in zip(online.state_dict().values(), target.state_dict().values(), strict=True):
            torch.testing.assert_close(actual, expected)


def test_independently_initialized_q_heads_disagree() -> None:
    cfg = _cfg()
    model = exp.TrainingModel(cfg).eval()
    batch = _batch(cfg)
    chunk = exp.quantized_chunk(model.policy, batch.batch)
    q1 = model.q1(batch.batch.context, chunk)
    q2 = model.q2(batch.batch.context, chunk)
    assert not torch.equal(q1, q2)


def test_rolling_next_context_exactly_drops_four_frames_and_updates_padding() -> None:
    cfg = _cfg()
    batch = _batch(cfg)
    expected = exp.rolling_next_context(batch.extended_context, cfg.L_ctx)
    assert expected.ctx_pad.tolist() == [0, 0]
    for name, value in expected.features.items():
        torch.testing.assert_close(value, batch.extended_context.features[name][:, 4:])

    expired = {name: value.clone() for name, value in batch.extended_context.features.items()}
    for value in expired.values():
        value[:, :4] = 123 if value.is_floating_point() else 1
    changed = exp.rolling_next_context(Context(features=expired, ctx_pad=batch.extended_context.ctx_pad), cfg.L_ctx)
    for name in expected.features:
        torch.testing.assert_close(changed.features[name], expected.features[name])


def test_macro_target_uses_frame_discount_and_terminal_mask() -> None:
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    next_value = torch.tensor([5.0, 5.0], requires_grad=True)
    continuation = torch.tensor([1.0, 0.0])
    target = exp.chunk_td_target(rewards, next_value, continuation, macro_discount=0.5)
    gamma = 0.5**0.25
    immediate = 1 + gamma * 2 + gamma**2 * 3 + gamma**3 * 4
    torch.testing.assert_close(target, torch.tensor([immediate + 0.5 * 5, immediate]))
    assert not target.requires_grad


def test_replay_label_marks_only_fully_observed_terminal_chunk_noncontinuing() -> None:
    sample = {
        "p1_stock": np.full(10, 4, dtype=np.int32),
        "p2_stock": np.full(10, 4, dtype=np.int32),
        "p1_percent": np.zeros(10, dtype=np.float32),
        "p2_percent": np.zeros(10, dtype=np.float32),
    }
    labeled = exp.label_iql_replay(sample, discount=0.99, damage_shaping=0.01, win_reward=0.5)
    # A sampled chunk starts one frame after its state anchor. At the maximum
    # legal start (6), anchor 5 bootstraps into the terminal frame and is masked.
    assert labeled["p1_iql_continuation"][4] == 1
    assert labeled["p1_iql_continuation"][5] == 0


def test_aligned_labels_select_one_final_anchor_transition() -> None:
    window = {
        exp.EGO_REWARD_COLUMN: np.arange(8, dtype=np.float32),
        exp.EGO_RETURN_COLUMN: np.arange(8, dtype=np.float32) + 100,
        exp.EGO_CONTINUATION_COLUMN: np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32),
    }
    rewards, returns, continuation = exp.aligned_iql_labels([window], L_ctx=4)
    torch.testing.assert_close(rewards, torch.tensor([[4.0, 5.0, 6.0, 7.0]]))
    torch.testing.assert_close(returns, torch.tensor([104.0]))
    torch.testing.assert_close(continuation, torch.tensor([0.0]))


def test_hl_gauss_normalizes_absorbs_tails_and_decodes_each_head_before_min() -> None:
    bins = torch.linspace(-2, 2, 5)
    encoded = exp.encode_hl_gauss(torch.tensor([-100.0, 0.3, 100.0]), bins)
    torch.testing.assert_close(encoded.sum(-1), torch.ones(3), atol=1e-6, rtol=1e-6)
    assert encoded[0, 0] > 0.999 and encoded[-1, -1] > 0.999

    cfg = _cfg(q_objective="hl_gauss", iql_q_min=-2, iql_q_max=2, iql_q_bins=5)
    q1, q2 = exp.QNetwork(cfg), exp.QNetwork(cfg)
    logits1 = torch.tensor([[0.0, 0.0, 0.0, 0.0, 8.0]])
    logits2 = torch.tensor([[8.0, 0.0, 0.0, 0.0, 0.0]])
    conservative = torch.minimum(q1.decode(logits1), q2.decode(logits2))
    assert conservative.item() < -1.9


def test_actor_weight_is_paper_exponential_with_cap_and_no_normalization() -> None:
    advantage = torch.tensor([-1.0, 0.0, 1.0, 10.0])
    weight, clipped = exp.actor_weights(advantage, temperature=3.0, weight_max=100.0)
    torch.testing.assert_close(weight[:3], torch.exp(torch.tensor([-3.0, 0.0, 3.0])))
    assert weight[-1].item() == pytest.approx(100)
    assert clipped.tolist() == [False, False, False, True]
    assert weight.mean() != 1


def test_actor_loss_scores_only_final_prefix_and_four_by_four_chunk_decisions() -> None:
    cfg = _cfg()
    policy = exp.GPT(cfg)
    batch = _batch(cfg)
    chunk = exp.quantized_chunk(policy, batch.batch)
    history = policy.codec.quantize(exp.stack_actions(batch.batch.context.features))
    hidden = policy(batch.batch.context.features, batch.batch.context.ctx_pad, history)[:, -1:]
    dense = policy.temporal.teacher_forced_nll(hidden, history[:, -1:], chunk[:, None])
    assert dense.shape == (cfg.batch_size, 1, 4, 4)
    torch.testing.assert_close(exp.actor_nll(policy, batch.batch, chunk), dense[:, 0].mean(dim=(-2, -1)))


@pytest.mark.parametrize("objective", ["scalar", "hl_gauss"])
def test_one_cpu_update_runs_all_stages_and_keeps_targets_frozen(objective: str) -> None:
    cfg = _cfg(q_objective=objective, iql_q_bins=11)
    model = exp.TrainingModel(cfg)
    optimizers, _ = exp.make_optimizers(model, cfg)
    batch = _batch(cfg, seed=7)
    target_before = {name: value.clone() for name, value in model.target_q1.state_dict().items()}
    parts = exp.iql_update(model, batch, optimizers, cfg)
    assert all(
        math_value == math_value for math_value in (parts.value_loss, parts.actor_loss, parts.q1_loss, parts.q2_loss)
    )
    assert parts.weight.shape == (cfg.batch_size,)
    assert all(parameter.grad is None for parameter in model.target_q1.parameters())
    assert any(not torch.equal(value, target_before[name]) for name, value in model.target_q1.state_dict().items())


def test_polyak_update_is_exact_and_copies_buffers() -> None:
    cfg = _cfg()
    online, target = exp.QNetwork(cfg), exp.QNetwork(cfg)
    with torch.no_grad():
        for parameter in online.parameters():
            parameter.fill_(2)
        for parameter in target.parameters():
            parameter.zero_()
    exp.polyak_update(target, online, 0.25)
    for parameter in target.parameters():
        torch.testing.assert_close(parameter, torch.full_like(parameter, 0.5))
    torch.testing.assert_close(target.bin_values, online.bin_values)


def test_named_optimizer_and_actor_scheduler_state_round_trip() -> None:
    cfg = _cfg()
    first = exp.TrainingModel(cfg)
    optimizers, schedulers = exp.make_optimizers(first, cfg)
    exp.iql_update(first, _batch(cfg), optimizers, cfg)
    schedulers.actor.step()
    optimizer_state, scheduler_state = optimizers.state_dict(), schedulers.state_dict()
    assert set(optimizer_state) == {"actor", "q", "value"}
    assert set(scheduler_state) == {"actor"}

    second = exp.TrainingModel(cfg)
    restored_optimizers, restored_schedulers = exp.make_optimizers(second, cfg)
    restored_optimizers.load_state_dict(optimizer_state)
    restored_schedulers.load_state_dict(scheduler_state)
    assert restored_schedulers.actor.last_epoch == schedulers.actor.last_epoch
    assert restored_optimizers.actor.param_groups[0]["lr"] == optimizers.actor.param_groups[0]["lr"]


def test_policy_only_state_omits_every_critic() -> None:
    model = exp.TrainingModel(_cfg())
    policy_state = {
        name.removeprefix("policy."): value for name, value in model.state_dict().items() if name.startswith("policy.")
    }
    loaded = exp.GPT(model.policy.cfg)
    loaded.load_state_dict(policy_state, strict=True)
    assert not any(name.startswith(("q1.", "q2.", "value.", "target_q")) for name in policy_state)


def test_tiny_training_run_reaches_policy_only_eval_and_pinned_h2h(monkeypatch, tmp_path: Path) -> None:
    cfg = _cfg(
        max_steps=1,
        val_every=0,
        eval_every=0,
        ckpt_every=0,
        wandb_log_code=False,
        inference_mode="eager",
    )
    batch = _batch(cfg, seed=11)
    reference = tmp_path / "026-final.pt"
    monkeypatch.setattr(exp, "_make_loaders", lambda cfg, stats: ([batch], [batch]))
    monkeypatch.setattr(exp, "make_run_name", lambda *args: "tiny-030")
    monkeypatch.setattr(exp, "setup_run_dir", lambda name: (tmp_path, tmp_path / "replays"))
    monkeypatch.setattr(exp, "resolve_h2h_reference", lambda cfg, run_dir: reference)
    monkeypatch.setattr(exp, "save_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp, "_checkpoint_sha256", lambda path: "a" * 64)
    inference = object()
    evaluated: list[object] = []
    h2h: list[tuple[object, Path]] = []
    monkeypatch.setattr(exp, "BF16Inference", lambda model, cfg: inference)

    def fake_eval(model, stats, cfg, **kwargs):
        assert isinstance(model, exp.GPT)
        assert kwargs["inference"] is inference
        evaluated.append(model)
        return {}

    def fake_h2h(model, stats, cfg, *, reference, **kwargs):
        assert isinstance(model, exp.GPT)
        h2h.append((model, reference))
        return {}

    monkeypatch.setattr(exp, "eval_vs_cpu", fake_eval)
    monkeypatch.setattr(exp, "final_h2h", fake_h2h)

    class Run:
        id = "test"
        summary: dict[str, object] = {}

    monkeypatch.setattr(exp.wandb, "run", Run())
    monkeypatch.setattr(exp.wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(exp.wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp.wandb, "finish", lambda: None)
    exp.train(cfg, {}, comment="tiny")
    assert len(evaluated) == 1
    assert h2h == [(evaluated[0], reference)]
