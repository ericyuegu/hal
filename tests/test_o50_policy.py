import json
import subprocess
import sys
import threading
from dataclasses import asdict
from multiprocessing import Pipe
from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch
from torch import Tensor

import hal.inference.o50 as o50
from hal.controller import NEUTRAL_CONTROLLER_ACTION
from hal.controller import ControllerAction
from hal.data.feature_stats import FeatureStats
from hal.data.schema import Rank
from hal.inference.api import PolicyInput
from hal.inference.api import RuntimeConfig
from hal.inference.o50 import O50_REQUIRED_OBSERVATION_FIELDS
from hal.inference.o50 import O50Policy
from hal.inference.o50 import export_o50_policy
from hal.inference.o50 import load_o50_policy
from hal.inference.o50_model import CONTROLLER_GROUP_COUNT
from hal.inference.o50_model import O50Architecture
from hal.inference.o50_model import O50Config
from hal.inference.o50_model import O50Model
from hal.netplay_service.inference import ContinuousBatcher
from hal.netplay_service.inference import RemotePolicy
from hal.netplay_service.inference import ServingArena
from hal.training.physical_shard_loader import PhysicalRow
from hal.training.physical_shard_loader import RingSlotDescriptor
from hal.training.player_identity import FIRST_CONNECT_CODE_ID
from hal.training.player_identity import encode_player_codes


def _architecture() -> O50Architecture:
    return O50Architecture(
        d_model=8,
        n_layers=1,
        n_heads=1,
        attn_window=0,
        L_ctx=4,
        sample_chunk_length=4,
        head_offsets=(1, 2, 3, 4),
        temporal_d_model=8,
        temporal_layers=1,
        temporal_heads=1,
        temporal_ff_dim=8,
        group_head_dim=8,
        action_embed_dim=4,
        offset_embed_dim=2,
        action_vocab=1024,
        action_state_embed_dim=4,
        char_vocab=32,
        char_dim=2,
        stage_vocab=32,
        stage_dim=2,
        item_type_dim=2,
        item_state_dim=2,
        item_hidden_dim=4,
        item_dim=4,
        value_hidden_dim=4,
    )


def _stats() -> dict[str, FeatureStats]:
    return {name: FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0) for name in o50._O50_STATS_FIELDS}


def _resolved_stats() -> o50._ResolvedStats:
    names = tuple(f"source-{index:02d}" for index in range(44))
    return o50._ResolvedStats(
        stats=_stats(),
        source_names=names,
        source_replay_weights={name: index + 1 for index, name in enumerate(names)},
        source_manifest_sha256={name: "a" * 64 for name in names},
        source_stats_sha256={name: "b" * 64 for name in names},
        mds_schema_version=7,
    )


def _config() -> tuple[O50Config, tuple[str, ...], bytes]:
    codes = ("IBDW#0",)
    player_codes = encode_player_codes(codes)
    _stats_json, stats_sha256 = o50._encode_stats(_resolved_stats())
    return (
        O50Config(
            experiment_id="050_scaled_temporal_awr_v4",
            architecture=_architecture(),
            prediction_frames=4,
            player_vocab_size=FIRST_CONNECT_CODE_ID + len(codes),
            player_vocab_sha256=o50._sha256_bytes(player_codes),
            player_code_bytes=len(player_codes),
            amp_dtype="float32",
            depth_alpha=0.5,
            mds_schema_version=7,
            stats_sha256=stats_sha256,
        ),
        codes,
        player_codes,
    )


def _model() -> tuple[O50Model, O50Config, tuple[str, ...], bytes]:
    config, codes, player_codes = _config()
    model = O50Model(config)
    model.player_code_bytes.copy_(torch.from_numpy(np.frombuffer(player_codes, dtype=np.uint8).copy()))
    model.eval()
    return model, config, codes, player_codes


def _policy(*, seed: int = 7) -> O50Policy:
    model, config, codes, _player_codes = _model()
    return O50Policy(
        model,
        config,
        _stats(),
        codes,
        name="test O50",
        device=torch.device("cpu"),
        seed=seed,
        compiled=False,
    )


def _observation() -> dict[str, float]:
    return {name: 0.0 for name in O50_REQUIRED_OBSERVATION_FIELDS}


def _input(
    frame_id: int,
    delay: int,
    *,
    stream_id: int = 3,
    controlled_port: int = 1,
    player_identity: str = "IBDW#0",
    reset: bool = False,
) -> PolicyInput:
    pending = tuple(
        ControllerAction(
            main_x=0.2 * (index + 1), main_y=0.0, c_x=0.0, c_y=0.0, trigger_l=0.0, trigger_r=0.0, buttons=0
        )
        for index in range(delay)
    )
    return PolicyInput(
        stream_id=stream_id,
        frame_id=frame_id,
        controlled_port=controlled_port,
        observation=_observation(),
        applied_action=NEUTRAL_CONTROLLER_ACTION,
        pending_actions=pending,
        player_identity=player_identity,
        reset=reset,
    )


def test_export_uses_safe_checkpoint_types_and_loads_without_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, config, _codes, _player_codes = _model()
    checkpoint_config = {
        "experiment_id": config.experiment_id,
        "architecture": asdict(config.architecture),
        "prediction_frames": config.prediction_frames,
        "player_vocab_size": config.player_vocab_size,
        "player_vocab_sha256": config.player_vocab_sha256,
        "amp_dtype": config.amp_dtype,
        "depth_alpha": config.depth_alpha,
        "mds_schema_version": config.mds_schema_version,
        "source_names": tuple(source.name for source in o50.streams.POLICY_WORLD_V8_SOURCES),
    }
    row = PhysicalRow("test", 1, 2)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "cfg": checkpoint_config,
            "model": model.state_dict(),
            "loader": {"row": row, "slot": RingSlotDescriptor(0, row, 1, 2)},
        },
        checkpoint,
    )
    monkeypatch.setattr(o50, "_resolve_export_stats", lambda _config: _resolved_stats())
    bundle = tmp_path / "policy.halpolicy"
    manifest = export_o50_policy(checkpoint, bundle)
    assert manifest.backend == "hal.o50.temporal-awr"
    assert {member.name for member in manifest.members} == {
        "backend.json",
        "players.json",
        "stats.json",
        "weights.safetensors",
    }

    policy = load_o50_policy(bundle, device="cpu", seed=11, compiled=False)
    assert policy.spec.required_observation_fields == O50_REQUIRED_OBSERVATION_FIELDS


def test_stats_content_and_provenance_hashes_are_strict() -> None:
    config, _codes, _player_codes = _config()
    encoded, digest = o50._encode_stats(_resolved_stats())
    assert digest == config.stats_sha256
    assert o50._decode_stats(encoded, config) == _stats()
    raw = json.loads(encoded)
    raw["finalized"]["position_x"]["mean"] = 1.0
    with pytest.raises(ValueError, match="statistics SHA-256"):
        o50._decode_stats(o50._canonical_json(raw), config)


@pytest.mark.parametrize(("delay", "replan", "frames", "expected_plans"), [(2, 2, 4, 2), (3, 1, 4, 4)])
def test_runtime_forces_pending_prefix_and_caches_only_sampled_tail(
    delay: int,
    replan: int,
    frames: int,
    expected_plans: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    policy.prepare(RuntimeConfig(max_batch_size=2, transport_delays=(delay,)))
    forced_calls: list[tuple[Tensor, Tensor]] = []

    def fake_decode(
        _features: dict[str, Tensor],
        _padding: Tensor,
        forced: Tensor,
        force_mask: Tensor,
        _uniforms: Tensor,
    ) -> Tensor:
        forced_calls.append((forced.detach().cpu(), force_mask.detach().cpu()))
        output = torch.zeros(2, 4, 14)
        for index in range(delay, 4):
            output[:, index, 0] = 0.25 + 0.25 * (index - delay)
        return output

    monkeypatch.setattr(policy, "_decode", fake_decode)
    outputs = [policy.step([_input(frame, delay, reset=frame == 0)])[0] for frame in range(frames)]
    assert len(forced_calls) == expected_plans
    assert [output.action.main_x for output in outputs[:replan]] == [0.25 + 0.25 * index for index in range(replan)]
    pending = torch.from_numpy(np.stack([o50._action_vector(action) for action in _input(0, delay).pending_actions]))
    expected_forced = policy._model.codec.quantize(pending)
    assert torch.equal(forced_calls[0][0][0, :delay], expected_forced)
    assert forced_calls[0][1][0].tolist() == [index < delay for index in range(4)]


def test_runtime_batches_delay_two_and_three_without_crossing_stream_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    policy.prepare(RuntimeConfig(max_batch_size=2, transport_delays=(2, 3)))
    masks: list[Tensor] = []

    def fake_decode(
        _features: dict[str, Tensor],
        _padding: Tensor,
        _forced: Tensor,
        force_mask: Tensor,
        _uniforms: Tensor,
    ) -> Tensor:
        masks.append(force_mask.detach().cpu())
        output = torch.zeros(2, 4, 14)
        output[0, 2:, 0] = torch.tensor([0.25, 0.5])
        output[1, 3, 0] = 0.75
        return output

    monkeypatch.setattr(policy, "_decode", fake_decode)
    outputs = policy.step(
        [
            _input(0, 2, stream_id=2, reset=True),
            _input(0, 3, stream_id=3, reset=True),
        ]
    )

    assert [output.action.main_x for output in outputs] == [0.25, 0.75]
    assert masks[0].tolist() == [
        [True, True, False, False],
        [True, True, True, False],
    ]
    assert len(policy._states[2].queued) == 1
    assert len(policy._states[3].queued) == 0


def test_eager_runtime_decodes_only_active_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _policy()
    policy.prepare(RuntimeConfig(max_batch_size=2, transport_delays=(2, 3)))
    shapes: list[int] = []

    def fake_decode(
        features: dict[str, Tensor],
        padding: Tensor,
        forced: Tensor,
        force_mask: Tensor,
        uniforms: Tensor,
    ) -> Tensor:
        rows = features["stage"].shape[0]
        shapes.append(rows)
        assert padding.shape == (rows,)
        assert forced.shape[0] == rows
        assert force_mask.shape[0] == rows
        assert uniforms.shape[-1] == rows
        return torch.zeros(rows, 4, 14)

    monkeypatch.setattr(policy, "_decode", fake_decode)

    policy.step([_input(0, 3, reset=True)])

    assert shapes == [1]


def test_compiled_runtime_keeps_the_prepared_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _policy()
    policy.prepare(RuntimeConfig(max_batch_size=2, transport_delays=(2, 3)))
    policy._compiled = True
    shapes: list[int] = []

    def fake_decode(features: dict[str, Tensor], *_args: Tensor) -> Tensor:
        rows = features["stage"].shape[0]
        shapes.append(rows)
        return torch.zeros(rows, 4, 14)

    monkeypatch.setattr(policy, "_decode", fake_decode)

    policy.step([_input(0, 3, reset=True)])

    assert shapes == [2]


@pytest.mark.parametrize("delay", [2, 3])
def test_remote_policy_actions_exactly_match_the_direct_path(delay: int) -> None:
    runtime = RuntimeConfig(max_batch_size=1, transport_delays=(delay,))
    torch.manual_seed(17)
    direct = _policy(seed=11)
    torch.manual_seed(17)
    served = _policy(seed=11)
    direct.prepare(runtime)
    served.prepare(runtime)
    parent, child = Pipe()
    with ServingArena.create(1, delay, served.spec.required_observation_fields) as arena:
        stop = threading.Event()
        batcher = ContinuousBatcher(served, runtime, arena, {0: parent})
        server = threading.Thread(target=batcher.serve, args=(stop,), daemon=True)
        server.start()
        remote = RemotePolicy(served.spec, runtime, arena, child, 0)
        try:
            for frame_id in range(6):
                item = _input(frame_id, delay, stream_id=5, reset=frame_id == 0)
                assert remote.step((item,)) == tuple(direct.step((item,)))
        finally:
            stop.set()
            server.join(timeout=1.0)
            parent.close()
            child.close()


def test_stream_delay_can_change_only_with_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = _policy()
    policy.prepare(RuntimeConfig(max_batch_size=1, transport_delays=(2, 3)))
    monkeypatch.setattr(policy, "_decode", lambda *_args: torch.zeros(1, 4, 14))
    policy.step([_input(0, 2, reset=True)])
    with pytest.raises(ValueError, match="changed transport delay"):
        policy.step([_input(1, 3)])
    policy.step([_input(1, 3, reset=True)])


@pytest.mark.parametrize("delay", [2, 3])
def test_temporal_decoder_advances_through_every_forced_action(
    delay: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, config, _codes, _player_codes = _model()
    decoder = model.temporal
    seen: list[Tensor] = []

    def record_step(
        _self: object,
        previous: Tensor,
        _offset: int,
        state_bias: Tensor,
        caches: list[tuple[Tensor, Tensor] | None],
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor] | None]]:
        seen.append(previous.clone())
        return torch.zeros(previous.shape[0], config.architecture.temporal_d_model), caches

    monkeypatch.setattr(decoder, "_decode_step", MethodType(record_step, decoder))
    observed = torch.zeros(1, CONTROLLER_GROUP_COUNT, dtype=torch.long)
    forced = torch.zeros(1, 4, CONTROLLER_GROUP_COUNT, dtype=torch.long)
    forced[:, :delay] = torch.arange(1, delay + 1, dtype=torch.long)[None, :, None]
    force_mask = torch.arange(4)[None, :] < delay
    uniforms = torch.full((4, CONTROLLER_GROUP_COUNT, 1), 0.5)
    sampled = decoder.sample_conditioned(
        torch.zeros(1, config.architecture.L_ctx, config.architecture.d_model),
        observed,
        forced,
        force_mask,
        uniforms,
        argmax=True,
        sample_start=delay,
    )
    assert sampled.shape == (1, 4, CONTROLLER_GROUP_COUNT)
    assert torch.equal(seen[0], observed)
    for depth in range(1, delay + 1):
        assert torch.equal(seen[depth], forced[:, depth - 1])


def test_player_identity_resolves_ranks_and_exact_connect_codes() -> None:
    policy = _policy()
    policy.prepare(RuntimeConfig(max_batch_size=1, transport_delays=(3,)))
    with pytest.raises(KeyError, match="ibdw"):
        policy.step([_input(0, 3, player_identity="ibdw#0")])
    with pytest.raises(KeyError, match="IBDW#0 "):
        policy.step([_input(0, 3, player_identity="IBDW#0 ")])
    assert policy._player_id("PLATINUM") == int(Rank.PLATINUM)
    assert policy._player_id("DIAMOND") == int(Rank.DIAMOND)
    assert policy._player_id("MASTER") == int(Rank.MASTER)
    assert policy._player_id("IBDW#0") == FIRST_CONNECT_CODE_ID
    with pytest.raises(ValueError, match="Platinum, Diamond, or Master"):
        policy._player_id("PRO")


def test_port_relative_adapter_and_applied_action_alignment() -> None:
    policy = _policy()
    policy.prepare(RuntimeConfig(max_batch_size=1, transport_delays=(3,)))
    item = _input(0, 3, controlled_port=2, reset=True)
    observation = dict(item.observation)
    observation["p1_position_x"] = 1.0
    observation["p2_position_x"] = 2.0
    applied = ControllerAction(0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0)
    relative = o50._relative_observation(
        PolicyInput(
            stream_id=item.stream_id,
            frame_id=item.frame_id,
            controlled_port=item.controlled_port,
            observation=observation,
            applied_action=applied,
            pending_actions=item.pending_actions,
            player_identity=item.player_identity,
            reset=True,
        )
    )
    assert relative["ego_position_x"] == 2.0
    assert relative["opp_position_x"] == 1.0
    assert relative["ego_main_stick_x"] == 0.5
    assert relative["ego_main_stick_y"] == -0.5


def test_new_stream_requires_reset_and_frame_gap_clears_context() -> None:
    policy = _policy()
    policy.prepare(RuntimeConfig(max_batch_size=1, transport_delays=(3,)))
    with pytest.raises(ValueError, match="reset=True"):
        policy.step([_input(0, 3)])
    policy.step([_input(0, 3, reset=True)])
    policy.step([_input(2, 3)])
    assert len(policy._states[3].history) == 1


def test_integer_item_sentinel_is_masked_instead_of_clamped_to_unknown() -> None:
    policy = _policy()
    policy.prepare(RuntimeConfig(max_batch_size=1, transport_delays=(3,)))
    original = _input(0, 3, reset=True)
    observation = dict(original.observation)
    observation["item0_type"] = (1 << 31) - 1
    observation["item0_state"] = (1 << 31) - 1
    item = PolicyInput(
        stream_id=original.stream_id,
        frame_id=original.frame_id,
        controlled_port=original.controlled_port,
        observation=observation,
        applied_action=original.applied_action,
        pending_actions=original.pending_actions,
        player_identity=original.player_identity,
        reset=True,
    )
    state = policy._ingest(item)
    context = policy._context([(item, state)])
    assert context.features["item0_type"][0, -1].item() == 0
    assert context.features["item0_state"][0, -1].item() == 0


def test_model_import_does_not_load_melee_or_experiments() -> None:
    code = """
import sys
import hal.inference.o50_model
assert 'melee' not in sys.modules
assert not any(name == 'experiments' or name.startswith('experiments.') for name in sys.modules)
import hal.inference.o50
assert not any(name == 'experiments' or name.startswith('experiments.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parents[1])
