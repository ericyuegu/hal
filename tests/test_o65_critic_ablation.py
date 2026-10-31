"""Critic arithmetic, architecture identity, and CPU learner updates."""

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from test_065_loop import PROVENANCE
from test_065_loop import FakeShard
from test_065_loop import exp
from test_065_loop import transformer_trainer
from test_o65_evaluation import small_model
from test_o65_evaluation import stats
from test_self_play_model import RANGES
from test_self_play_model import config
from test_window_replay_model import inputs

from hal.eval.self_play_checkpoint import load_evaluation_checkpoint
from hal.eval.self_play_checkpoint import save_evaluation_checkpoint
from hal.models.action_sequence import ActionSequenceTransformer
from hal.models.action_sequence import SwiGLU
from hal.models.action_sequence import decoder_rmsnorm
from hal.models.self_play import VALUE_BINS
from hal.models.self_play import CriticKind
from hal.models.self_play import QueueCritic
from hal.models.self_play import QueueResidualHead
from hal.models.self_play import SelfPlayTransformer
from hal.models.self_play import value_cross_entropy
from hal.models.window_replay import replay_windows
from hal.representation.player_identity import PlayerVocabulary
from hal.wire import ACTION_DIM


def test_scalar_values_and_loss_match_84fafaef() -> None:
    torch.manual_seed(65)
    policy = SelfPlayTransformer(
        config(), RANGES, max_opponent_queue=14, critic_hidden_dim=16, critic_kind="residual_scalar"
    )
    assert isinstance(policy.critic, QueueResidualHead)
    torch.nn.init.normal_(policy.critic.residual.down.weight)
    torch.nn.init.normal_(policy.critic.residual.down.bias)
    trunk = torch.randn(7, policy.cfg.d_model, requires_grad=True)
    decoder = torch.randn(7, policy.cfg.temporal_d_model, requires_grad=True)
    condition, opponent_condition = torch.randn(7, 4), torch.randn(7, 4)
    queue = policy.codec.quantize(torch.rand(7, 14, ACTION_DIM))
    lengths = torch.tensor([0, 1, 4, 7, 10, 13, 14])
    actual = policy.self_play_value(trunk, decoder, condition, opponent_condition, queue, lengths)
    features = decoder_rmsnorm(trunk.detach()).to(policy.value_head.up.weight.dtype)
    base = policy.value_head(features).squeeze(-1).float()
    embeddings = policy.codec.embed_frame(queue).detach().to(features.dtype)
    valid = (torch.arange(14)[None] < lengths[:, None])[..., None]
    frames = torch.cat((torch.where(valid, embeddings.detach(), 0), valid.to(embeddings.dtype)), -1)
    encoded = policy.critic.queue_encoder(frames.flatten(1))
    residual_inputs = torch.cat((decoder.detach().to(features.dtype), condition, opponent_condition, encoded), -1)
    expected = base + policy.critic.residual(residual_inputs).squeeze(-1).float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    target = torch.randn(7, requires_grad=True) * 170
    selected = torch.tensor([True, False, True, True, False, True, True])
    loss = exp.critic_loss(actual, target, selected, critic="residual_scalar")
    expected_loss = (((expected[selected] - target[selected].detach()) / 150.0).square()).mean()
    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
    parameters = policy.critic_parameters()
    actual_grad = torch.autograd.grad(loss, parameters, retain_graph=True)
    expected_grad = torch.autograd.grad(expected_loss, parameters, retain_graph=True)
    for left, right in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    loss.backward()
    assert trunk.grad is decoder.grad is None
    assert all(p.grad is None for p in policy.actor_parameters())
    assert all(p.requires_grad for p in policy.value_head.parameters())
    assert set(map(id, policy.actor_parameters())).isdisjoint(map(id, parameters))


def test_depth_zero_loads_v11_tensors_and_preserves_arithmetic() -> None:
    cfg, width, capacity = config(), 16, 14
    # The v11 module layout, independent of QueueCritic's constructor.
    legacy = torch.nn.Module()
    legacy.queue_encoder = SwiGLU(capacity * (4 * cfg.action_embed_dim + 1), width, width)
    legacy.head = SwiGLU(cfg.d_model + cfg.temporal_d_model + 8 + width, width, VALUE_BINS, output_bias=True)
    critic = QueueCritic(cfg, capacity, width, depth=0)
    critic.load_state_dict(legacy.state_dict(), strict=True)
    assert critic.state_dict().keys() == legacy.state_dict().keys()
    trunk, decoder, condition, opponent = torch.randn(4, 32), torch.randn(4, 32), torch.randn(4, 4), torch.randn(4, 4)
    queue, lengths = torch.randn(4, capacity, 4 * cfg.action_embed_dim), torch.tensor([0, 4, 8, 14])
    valid = (torch.arange(capacity)[None] < lengths[:, None])[..., None]
    frames = torch.cat((torch.where(valid, queue, 0), valid.to(queue.dtype)), -1)
    encoded = legacy.queue_encoder(frames.flatten(1))
    expected = legacy.head(torch.cat((trunk, decoder, condition, opponent, encoded), -1)).float()
    torch.testing.assert_close(critic(trunk, decoder, condition, opponent, queue, lengths), expected, rtol=0, atol=0)


def test_deep_critic_starts_uniform_and_learns() -> None:
    torch.manual_seed(65)
    cfg = config()
    critic = QueueCritic(cfg, 4, 32, depth=3)
    inputs = (
        torch.randn(12, 32) * 100,
        torch.randn(12, 32) * 0.01,
        torch.randn(12, 4),
        torch.randn(12, 4),
        torch.randn(12, 4, 4 * cfg.action_embed_dim),
        torch.tensor([0, 1, 2, 4] * 3),
    )
    initial = critic(*inputs)
    torch.testing.assert_close(initial, torch.zeros_like(initial), rtol=0, atol=0)
    torch.testing.assert_close(initial.softmax(-1), torch.full_like(initial, 1 / VALUE_BINS))
    targets = inputs[0][:, 0] * 0.5
    optimizer = torch.optim.Adam(critic.parameters(), lr=0.01)
    before = value_cross_entropy(initial, targets).mean().detach()
    for _ in range(8):
        optimizer.zero_grad()
        loss = value_cross_entropy(critic(*inputs), targets).mean()
        loss.backward()
        optimizer.step()
    assert value_cross_entropy(critic(*inputs), targets).mean() < before * 0.9
    assert len(critic.deep.blocks) == 3
    block = critic.deep.blocks[0]
    assert isinstance(block, torch.nn.Sequential)
    projection = block[1]
    assert isinstance(projection, SwiGLU)
    assert projection.up.out_features == 4 * critic.hidden_dim


@pytest.mark.parametrize(
    "kind,depth", [("hl_gauss", 0), ("hl_gauss", 3), ("residual_scalar", 0), ("residual_scalar", 3)]
)
def test_bc_weights_load_for_each_critic(kind: CriticKind, depth: int) -> None:
    cfg = config()
    bc = ActionSequenceTransformer(cfg)
    bc.percentile_head = SwiGLU(cfg.d_model, cfg.value_hidden_dim, 65, output_bias=True)
    model = SelfPlayTransformer(
        cfg, RANGES, max_opponent_queue=14, critic_hidden_dim=16, critic_kind=kind, critic_depth=depth
    )
    model.load_bc_state_dict(bc.state_dict())
    torch.testing.assert_close(model.value_head.up.weight, bc.value_head.up.weight, rtol=0, atol=0)


@pytest.mark.parametrize("coefficient", [0.1, -0.1, float("nan")])
def test_scalar_rejects_actor_value_loss(coefficient: float) -> None:
    with pytest.raises(ValueError, match="requires value_loss_coefficient 0"):
        exp.TrainConfig(critic="residual_scalar", value_loss_coefficient=coefficient)


def critic_trainer(kind: CriticKind, depth: int = 0) -> Any:
    base = transformer_trainer("cpu", decoder_precision="fp32")
    model = SelfPlayTransformer(
        base.policy.model.cfg,
        base.policy.model.ranges,
        PlayerVocabulary(()),
        max_opponent_queue=4,
        critic_hidden_dim=8,
        critic_kind=kind,
        critic_depth=depth,
        decoder_precision="fp32",
    ).eval()
    state = model.state_dict()
    state.update(
        {name: tensor for name, tensor in base.policy.model.state_dict().items() if not name.startswith("critic.")}
    )
    model.load_state_dict(state)
    policy = exp.TransformerPolicy(model, base.engine.sampler.schedule, base.policy.statistics)
    engine = exp.RolloutEngine(
        ((FakeShard((0,), length=67, timeout=True),),), policy, base.engine.sampler, {0: policy.frozen_copy()}
    )
    engine.initialize()
    cfg = dataclasses.replace(
        base.config, critic=kind, critic_hidden_dim=8, critic_depth=depth, value_loss_coefficient=0
    )
    return exp.Trainer(engine, cfg)


@pytest.mark.parametrize("kind,depth", [("hl_gauss", 3), ("residual_scalar", 0), ("residual_scalar", 3)])
def test_cpu_trainer_update_and_exact_resume(tmp_path: Path, kind: CriticKind, depth: int) -> None:
    run = critic_trainer(kind, depth)
    actor_before = tuple(p.detach().clone() for p in run.policy.actor_parameters())
    critic_before = tuple(p.detach().clone() for p in run.policy.critic_parameters())
    _, metrics = run.run_update()
    assert run.update == 1 and run.actor_steps > 0
    assert metrics["critic/loss"] > 0
    assert any(
        not torch.equal(old, new) for old, new in zip(critic_before, run.policy.critic_parameters(), strict=True)
    )
    assert any(not torch.equal(old, new) for old, new in zip(actor_before, run.policy.actor_parameters(), strict=True))
    if kind == "residual_scalar":
        assert metrics["policy/value_loss"] == 0
    checkpoint = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    restored = critic_trainer(kind, depth)
    exp.restore_checkpoint(restored, checkpoint, PROVENANCE)
    first, _ = run.run_update()
    second, _ = restored.run_update()
    assert exp._batch_hash(first) == exp._batch_hash(second)
    assert exp._state_hash(run.policy.model.state_dict()) == exp._state_hash(restored.policy.model.state_dict())
    assert exp._state_hash(run.critic_optimizer.state_dict()) == exp._state_hash(
        restored.critic_optimizer.state_dict()
    )


def test_resume_rejects_changed_critic_metadata(tmp_path: Path) -> None:
    run = critic_trainer("residual_scalar")
    path = exp.save_checkpoint(run, tmp_path, PROVENANCE)
    manifest = json.loads(path.read_text())
    shard = next(item for item in manifest["shards"] if item["role"] == "training")
    state = torch.load(tmp_path / shard["name"], weights_only=False)
    state["critic"]["kind"] = "hl_gauss"
    temporary = tmp_path / "changed.pt"
    torch.save(state, temporary)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    shard.update(name=f"{digest}.rank0.pt", sha256=digest)
    temporary.rename(tmp_path / shard["name"])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="critic metadata mismatch"):
        exp.restore_checkpoint(run, path, PROVENANCE)


@pytest.mark.parametrize("kind,depth", [("hl_gauss", 3), ("residual_scalar", 0), ("residual_scalar", 3)])
def test_evaluation_loads_each_critic_and_rejects_mismatches(tmp_path: Path, kind: CriticKind, depth: int) -> None:
    base = small_model()
    model = SelfPlayTransformer(
        base.cfg,
        base.ranges,
        PlayerVocabulary(()),
        max_opponent_queue=14,
        critic_hidden_dim=16,
        critic_kind=kind,
        critic_depth=depth,
        decoder_precision=base.decoder_precision,
    )
    path = tmp_path / "actor.pt"
    save_evaluation_checkpoint(
        path,
        model,
        stats(),
        provenance={
            "reference_sha256": "a" * 64,
            "git_sha": "b" * 40,
            "update": 0,
            "data_manifests": {"training": "c" * 64},
        },
    )
    loaded = load_evaluation_checkpoint(path, device=torch.device("cpu"))
    assert loaded.model.critic_kind == kind and loaded.model.critic.depth == depth
    other: CriticKind = "residual_scalar" if kind == "hl_gauss" else "hl_gauss"
    with pytest.raises(ValueError, match="critic differs"):
        load_evaluation_checkpoint(path, device=torch.device("cpu"), critic_kind=other)
    with pytest.raises(ValueError, match="critic differs"):
        load_evaluation_checkpoint(path, device=torch.device("cpu"), critic_hidden_dim=128)
    with pytest.raises(ValueError, match="critic differs"):
        load_evaluation_checkpoint(path, device=torch.device("cpu"), critic_depth=depth + 1)


@pytest.mark.parametrize(
    "kind,depth", [("hl_gauss", 0), ("hl_gauss", 3), ("residual_scalar", 0), ("residual_scalar", 3)]
)
def test_replay_keeps_decoder_state_detached(kind: CriticKind, depth: int) -> None:
    model = SelfPlayTransformer(
        config(),
        RANGES,
        max_opponent_queue=14,
        critic_hidden_dim=16,
        critic_kind=kind,
        critic_depth=depth,
        decoder_precision="fp32",
    )
    hidden, observed, slots = inputs(model)
    hidden.requires_grad_(True)
    output = replay_windows(model, hidden, observed, slots, grad=True)
    assert output.features.trunk_WKd.requires_grad
    assert not output.features.decoder_WKt.requires_grad


def test_default_critic_resumes_v11_checkpoint(tmp_path: Path) -> None:
    from test_065_loop import trainer

    run = trainer(groups=1)
    run.run_update()
    snapshot = exp.capture_checkpoint(run, PROVENANCE)
    state = dict(snapshot.state)
    state.pop("critic")
    state["version"] = 11
    manifest = {
        **snapshot.manifest,
        "format": "hal.o65.training.v11",
        "version": 11,
        "configuration_sha256": exp._configuration_digest(run, legacy_critic=True),
    }
    path = exp.write_checkpoint(dataclasses.replace(snapshot, state=state, manifest=manifest), tmp_path)
    restored = trainer(groups=1)
    exp.restore_checkpoint(restored, path, PROVENANCE)
    first, _ = run.run_update()
    second, _ = restored.run_update()
    assert exp._batch_hash(first) == exp._batch_hash(second)
    assert exp._state_hash(run.policy.model.state_dict()) == exp._state_hash(restored.policy.model.state_dict())


def test_v3_evaluation_checkpoint_keeps_its_policy_identity(tmp_path: Path) -> None:
    path = tmp_path / "actor.pt"
    save_evaluation_checkpoint(
        path,
        small_model(),
        stats(),
        provenance={
            "reference_sha256": "a" * 64,
            "git_sha": "b" * 40,
            "update": 0,
            "data_manifests": {"training": "c" * 64},
        },
    )
    expected = load_evaluation_checkpoint(path, device=torch.device("cpu"))
    raw = torch.load(path, weights_only=False)
    raw["format"] = "hal.o65.evaluation.v3"
    raw.pop("critic_kind")
    raw.pop("critic_depth")
    torch.save(raw, path)
    loaded = load_evaluation_checkpoint(path, device=torch.device("cpu"))
    assert loaded.policy_sha256 == expected.policy_sha256


@pytest.mark.parametrize(
    "field,value", [("critic_kind", "residual_scalar"), ("critic_depth", 2), ("critic_hidden_dim", 32)]
)
def test_actor_rejects_metadata_that_disagrees_with_tensors(tmp_path: Path, field: str, value: str | int) -> None:
    base = small_model()
    model = SelfPlayTransformer(
        base.cfg,
        base.ranges,
        PlayerVocabulary(()),
        max_opponent_queue=14,
        critic_hidden_dim=16,
        critic_depth=3,
        decoder_precision=base.decoder_precision,
    )
    path = tmp_path / "actor.pt"
    save_evaluation_checkpoint(
        path,
        model,
        stats(),
        provenance={
            "reference_sha256": "a" * 64,
            "git_sha": "b" * 40,
            "update": 0,
            "data_manifests": {"training": "c" * 64},
        },
    )
    raw = torch.load(path, weights_only=False)
    raw[field] = value
    torch.save(raw, path)
    with pytest.raises(ValueError):
        load_evaluation_checkpoint(path, device=torch.device("cpu"))


def test_deep_residual_starts_at_the_bc_baseline_and_fits_from_trunk_state() -> None:
    torch.manual_seed(65)
    policy = SelfPlayTransformer(
        config(), RANGES, max_opponent_queue=14, critic_hidden_dim=32, critic_kind="residual_scalar", critic_depth=3
    )
    assert isinstance(policy.critic, QueueResidualHead)
    trunk = torch.randn(64, policy.cfg.d_model)
    decoder = torch.randn(64, policy.cfg.temporal_d_model)
    condition, opponent_condition = torch.zeros(64, 4), torch.zeros(64, 4)
    queue = policy.codec.quantize(torch.rand(64, 14, ACTION_DIM))
    lengths = torch.zeros(64, dtype=torch.long)
    base = policy.value_head(decoder_rmsnorm(trunk).to(policy.value_head.up.weight.dtype)).squeeze(-1).float()
    values = policy.self_play_value(trunk, decoder, condition, opponent_condition, queue, lengths)
    torch.testing.assert_close(values, base.detach(), rtol=0, atol=0)
    # A target that depends only on the trunk state needs the deep head's trunk input.
    target = base.detach() + 40 * trunk[:, 0]
    optimizer = torch.optim.Adam(policy.critic.parameters(), lr=1e-3)
    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        values = policy.self_play_value(trunk, decoder, condition, opponent_condition, queue, lengths)
        loss = (values - target).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    assert losses[-1] < 0.5 * losses[0]
