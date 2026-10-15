"""Focused contract tests for the bounded historical gameplay runner."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from hal import streams
from hal.controller import ControllerAction
from hal.data.feature_stats import FeatureStats
from hal.eval import cross_stage
from hal.inference import api as inference_api
from hal.inference import o50
from hal.inference import o50_model
from hal.training import ego_stats
from hal.training import features as training_features
from hal.training import player_identity
from hal.wire import ACTION_CHANNELS
from research.mechinterp_20260911 import historical_gameplay


def _small_config() -> o50_model.O50Config:
    architecture = o50_model.O50Architecture(
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
    codes = b'["IBDW#0"]'
    return o50_model.O50Config(
        experiment_id="050_scaled_temporal_awr_v4",
        architecture=architecture,
        prediction_frames=4,
        player_vocab_size=player_identity.FIRST_CONNECT_CODE_ID + 1,
        player_vocab_sha256=hashlib.sha256(codes).hexdigest(),
        player_code_bytes=len(codes),
        amp_dtype="float32",
        depth_alpha=0.5,
        mds_schema_version=7,
        stats_sha256="c" * 64,
    )


def test_masked_identity_features_is_exact_and_does_not_mutate_input() -> None:
    identity = torch.tensor([[3, 4], [5, 6]])
    observation = torch.tensor([[1.0, 2.0]])
    features = {"ego_player_id": identity, "ego_position_x": observation}

    masked = historical_gameplay.masked_identity_features(features)

    assert torch.equal(masked["ego_player_id"], torch.zeros_like(identity))
    assert masked["ego_position_x"] is observation
    assert features["ego_player_id"] is identity


def test_protocols_keep_historical_invariants_and_separate_labels() -> None:
    labels = set()
    for protocol in historical_gameplay.PROTOCOLS.values():
        record = historical_gameplay.protocol_record(protocol)
        assert protocol.transport_delay == 2
        assert protocol.replan_interval == 2
        assert protocol.context_length == 256
        assert protocol.cpu_level == 9
        assert protocol.seed == 0
        assert protocol.start_retries == 0
        assert len(cast(list[object], record["schedule"])) == protocol.boots
        labels.add(protocol.label)

    assert len(labels) == len(historical_gameplay.PROTOCOLS)
    assert historical_gameplay.PROTOCOLS["historical96"].max_frames == 7_200
    assert historical_gameplay.PROTOCOLS["pilot32"].max_frames == 7_200
    assert historical_gameplay.PROTOCOLS["short32"].max_frames == 1_800


def test_cuda_execution_requires_an_explicit_supported_mode() -> None:
    assert historical_gameplay.CudaExecution(target="local_cuda", compiled=False).compiled is False
    with pytest.raises(ValueError, match="unknown CUDA target"):
        historical_gameplay.CudaExecution(target="cpu", compiled=False)  # ty: ignore[invalid-argument-type]


def test_checkpoint_export_requires_matching_bytes_hash_and_boundary(tmp_path: Path) -> None:
    path = tmp_path / "p1fyyp1z-step-0024576.model.pt"
    config = {
        "experiment_id": "050_scaled_temporal_awr_v4",
        "mds_schema_version": 7,
    }
    raw = {
        "cfg": config,
        "checkpoint_step": 24_575,
        "data_protocol": "o50-replay-ring-v8",
        "model": {"weight": torch.ones(2, 2)},
        "source_checkpoint_sha256": "a" * 64,
        "source_manifest_sha256": {"source": "b" * 64},
        "source_r2_etag": '"etag"',
        "source_selection_sha256": historical_gameplay._EXPECTED_SOURCE_SELECTION_SHA256,
        "source_uri": "r2://hal/checkpoint.pt",
        "wandb_id": "p1fyyp1z",
    }
    torch.save(raw, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {
        "boundary_step": 24_576,
        "bytes": path.stat().st_size,
        "output_sha256": digest,
        "raw_cfg": config,
        "run_id": "p1fyyp1z",
        "source_checkpoint_sha256": "a" * 64,
        "source_r2_etag": '"etag"',
    }
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata))

    loaded = historical_gameplay.load_checkpoint_export(path, expected_boundary_step=24_576)

    assert loaded.boundary_step == 24_576
    assert torch.equal(loaded.model["weight"], raw["model"]["weight"])
    with pytest.raises(ValueError, match="stores step"):
        historical_gameplay.load_checkpoint_export(path, expected_boundary_step=57_344)


def test_replay_root_must_not_mix_with_existing_results(tmp_path: Path) -> None:
    root = tmp_path / "replays"
    historical_gameplay.prepare_replay_root(root)
    (root / "old.slp").write_bytes(b"old")

    with pytest.raises(ValueError, match="absent or empty"):
        historical_gameplay.prepare_replay_root(root)


def test_single_control_rejects_nonempty_output_before_initialization(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    output.write_text("existing\n")
    replay_root = tmp_path / "replays"

    with pytest.raises(ValueError, match="already contains data"):
        historical_gameplay.run_early_portable_control(
            tmp_path / "checkpoint.pt",
            output,
            replay_root,
            device=torch.device("cpu"),
        )

    assert not replay_root.exists()


def test_checkpoint_pair_rejects_non_public_source_manifest() -> None:
    source_names = tuple(source.name for source in streams.POLICY_WORLD_V8_SOURCES)
    config = {"source_names": source_names}
    model = {"weight": torch.ones(1)}

    def checkpoint(step: int, manifests: dict[str, str]) -> historical_gameplay.CheckpointExport:
        return historical_gameplay.CheckpointExport(
            path=Path(f"step-{step}.pt"),
            boundary_step=step,
            export_sha256="c" * 64,
            config=config,
            model=model,
            source_checkpoint_sha256="a" * 64,
            source_r2_etag='"etag"',
            source_manifest_sha256=manifests,
            source_selection_sha256=historical_gameplay._EXPECTED_SOURCE_SELECTION_SHA256,
            source_uri="r2://hal/checkpoint.pt",
            wandb_id="p1fyyp1z",
        )

    early = checkpoint(24_576, dict(streams.POLICY_WORLD_V8_TRAIN_MANIFEST_SHA256))
    wrong = dict(streams.POLICY_WORLD_V8_TRAIN_MANIFEST_SHA256)
    wrong[source_names[0]] = "0" * 64
    slump = checkpoint(57_344, wrong)

    with pytest.raises(ValueError, match="source manifests differ"):
        historical_gameplay.validate_checkpoint_pair(early, slump)


@pytest.mark.parametrize(
    "name",
    [
        "early_identity",
        "slump_identity",
        "interp_a0",
        "interp_a0.25",
        "interp_a0.5",
        "interp_a0.75",
        "interp_a1",
        "reverse_a0",
        "reverse_a0.25",
        "reverse_a0.5",
        "reverse_a0.75",
        "reverse_a1",
    ],
)
def test_gameplay_rejects_identity_and_interpolation_controls(name: str) -> None:
    with pytest.raises(ValueError, match="identity or interpolation"):
        historical_gameplay.treatment_spec(name)


@pytest.mark.parametrize("name", ["restore_attention_qk_norms", "qk_gauge_q2_khalf"])
def test_gameplay_accepts_named_attention_transforms(name: str) -> None:
    treatment = historical_gameplay.treatment_spec(name)

    assert treatment.name == name
    assert treatment.decoder_semantics == "historical_latent_prefix"
    assert treatment.weight_intervention is not None


def test_portable_committed_prefix_is_a_distinct_semantic_treatment() -> None:
    treatment = historical_gameplay.treatment_spec("portable_committed_prefix")

    assert treatment.decoder_semantics == "portable_committed_prefix"
    assert treatment.weight_intervention is None


def test_latent_prefix_decoder_matches_fixed_seed_frozen_formula() -> None:
    torch.manual_seed(20260911)
    config = _small_config()
    architecture = config.architecture
    model = o50_model.O50Model(config).eval()
    batch = 3
    hidden = torch.randn(batch, architecture.L_ctx, architecture.d_model)
    observed = torch.zeros(batch, o50_model.CONTROLLER_GROUP_COUNT, dtype=torch.long)
    forced = torch.zeros(batch, 4, o50_model.CONTROLLER_GROUP_COUNT, dtype=torch.long)
    portable_force_mask = torch.tensor([[True, True, False, False]]).expand(batch, -1)
    uniforms = torch.rand(4, o50_model.CONTROLLER_GROUP_COUNT, batch)

    reference = model.temporal.sample_conditioned(
        hidden,
        observed,
        forced,
        torch.zeros_like(portable_force_mask),
        uniforms,
        sample_start=0,
    )
    adapter = historical_gameplay.HistoricalLatentPrefixDecoder(model.temporal, latent_frames=2)
    actual = adapter.sample_conditioned(
        hidden,
        observed,
        forced,
        portable_force_mask,
        uniforms,
        sample_start=2,
    )

    assert torch.equal(actual[:, :2], reference[:, 2:4])
    assert torch.equal(actual[:, 2:4], reference[:, 0:2])


class _FakeCodec:
    def quantize(self, actions: Tensor) -> Tensor:
        return torch.zeros((*actions.shape[:-1], o50_model.CONTROLLER_GROUP_COUNT), dtype=torch.long)

    def dequantize(self, indices: Tensor) -> Tensor:
        actions = torch.zeros((*indices.shape[:-1], len(ACTION_CHANNELS)))
        actions[..., 0] = indices[..., 1].float() / 10.0
        return actions


class _FakeDecoder:
    def __init__(self) -> None:
        self.calls = 0
        self.arguments: list[tuple[Tensor, int]] = []

    def reset(self) -> None:
        self.calls = 0
        self.arguments.clear()

    def sample_conditioned(
        self,
        hidden: Tensor,
        observed: Tensor,
        forced: Tensor,
        force_mask: Tensor,
        uniforms: Tensor,
        *,
        argmax: bool = False,
        sample_start: int = 0,
    ) -> Tensor:
        del hidden, observed, uniforms, argmax
        self.arguments.append((force_mask.detach().clone(), sample_start))
        main_stick_indices = (2, 4, 6, 8) if self.calls % 2 == 0 else (1, 3, 5, 7)
        self.calls += 1
        output = torch.zeros_like(forced)
        output[:, :, 1] = torch.tensor(main_stick_indices)
        return output


class _FakeModel:
    def __init__(self) -> None:
        self.codec = _FakeCodec()
        self.temporal = _FakeDecoder()
        self.contexts: list[tuple[dict[str, Tensor], Tensor]] = []

    def forward_dense(
        self,
        features: dict[str, Tensor],
        context_padding: Tensor,
        action_indices: Tensor,
    ) -> Tensor:
        del action_indices
        self.contexts.append(
            (
                {name: value.detach().clone() for name, value in features.items()},
                context_padding.detach().clone(),
            )
        )
        rows, length = context_padding.shape[0], next(iter(features.values())).shape[1]
        return torch.zeros(rows, length, 1)


def _unit_stats() -> Mapping[str, FeatureStats]:
    stats: dict[str, FeatureStats] = {}
    for name in training_features.BASE_ITEMS_PROJECTION.columns:
        if training_features.feature_kind(name, training_features.ITEM_COLUMNS) == "float":
            stats[ego_stats.consolidate_key(name)] = FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0)
    return stats


def _observation() -> dict[str, int]:
    return {name: 0 for name in o50.O50_REQUIRED_OBSERVATION_FIELDS}


def _action(main_x: float) -> ControllerAction:
    return ControllerAction(
        main_x=main_x,
        main_y=0.0,
        c_x=0.0,
        c_y=0.0,
        trigger_l=0.0,
        trigger_r=0.0,
        buttons=0,
    )


def test_historical_policy_replans_suffix_and_preserves_multistep_context() -> None:
    config = _small_config()
    model = _FakeModel()
    assets = historical_gameplay.RuntimeAssets(
        config=config,
        stats=_unit_stats(),
        player_codes=("IBDW#0",),
    )
    policy = historical_gameplay.HistoricalLatentPrefixPolicy(
        cast(historical_gameplay.MaskedIdentityO50Model, model),
        assets,
        name="test-historical",
        device=torch.device("cpu"),
        seed=20260911,
        compiled=False,
    )
    runtime = inference_api.RuntimeConfig(
        max_batch_size=1,
        transport_delays=(2,),
        replan_interval_frames=2,
    )
    policy.prepare(runtime)
    model.contexts.clear()
    model.temporal.reset()
    pending = (_action(0.0), _action(0.0))

    outputs = []
    for frame_id, main_x in enumerate((0.1, 0.2, 0.3, 0.4)):
        output = policy.step(
            [
                inference_api.PolicyInput(
                    stream_id=7,
                    frame_id=frame_id,
                    controlled_port=1,
                    observation=_observation(),
                    applied_action=_action(main_x),
                    pending_actions=pending,
                    player_identity="IBDW#0",
                    reset=frame_id == 0,
                )
            ]
        )
        outputs.append(output[0].action.main_x)

    assert outputs == pytest.approx([0.6, 0.8, 0.5, 0.7])
    assert len(model.contexts) == 2
    first_features, first_padding = model.contexts[0]
    second_features, second_padding = model.contexts[1]
    assert first_padding.tolist() == [3]
    assert second_padding.tolist() == [1]
    assert first_features["ego_main_stick_x"][0, -1].item() == pytest.approx(0.1)
    assert second_features["ego_main_stick_x"][0, -3:].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert len(model.temporal.arguments) == 2
    assert all(not force_mask.any() and sample_start == 0 for force_mask, sample_start in model.temporal.arguments)


def _match_row(
    boot_index: int,
    active_frames: int,
    *,
    taken: int,
    lost: int,
    dealt: float,
    received: float,
) -> cross_stage.MatchRow:
    return cross_stage.MatchRow(
        ego_character=1,
        opp_character=2,
        stage=3,
        boot_index=boot_index,
        match_ordinal=0,
        active_frames=active_frames,
        total_frames=active_frames + cross_stage.PREGAME_FRAMES,
        damage_dealt=dealt,
        damage_taken=received,
        stocks_taken=taken,
        stocks_lost=lost,
    )


def test_blocked_net_bootstrap_preserves_joint_arithmetic_and_separate_estimands() -> None:
    treatment = [
        _match_row(0, 3_600, taken=101, lost=100, dealt=101.0, received=100.0),
        _match_row(1, 7_200, taken=100, lost=100, dealt=100.0, received=100.0),
    ]
    control = [
        _match_row(0, 3_600, taken=0, lost=0, dealt=0.0, received=0.0),
        _match_row(1, 7_200, taken=0, lost=0, dealt=0.0, received=0.0),
    ]

    result = historical_gameplay.blocked_net_deltas(
        treatment,
        control,
        expected_boots=2,
        bootstrap_resamples=2_000,
        seed=7,
    )

    coverage = cast(dict[str, int], result["coverage"])
    active_time = cast(dict[str, float | bool], result["active_time"])
    estimands = cast(dict[str, dict[str, dict[str, float]]], result["estimands"])
    assert coverage == {
        "excluded_not_jointly_active": 0,
        "matched_scheduled_boots": 2,
        "rated_boots": 2,
        "zero_active_a": 0,
        "zero_active_b": 0,
    }
    assert active_time["a_total_minutes"] == pytest.approx(3.0)
    assert active_time["b_total_minutes"] == pytest.approx(3.0)
    assert active_time["per_boot_minutes_equal"] is True
    for metric in ("net_stock_per_min", "net_damage_per_min"):
        assert estimands["equal_boot"][metric] == {
            "ci_hi": pytest.approx(1.0),
            "ci_lo": pytest.approx(0.0),
            "point": pytest.approx(0.5),
        }
        assert estimands["pooled_ratio"][metric]["point"] == pytest.approx(1.0 / 3.0)


def test_factorial_bootstrap_uses_one_joint_resample_for_all_four_cells() -> None:
    frames = (3_600, 7_200)

    def rows(taken_rate: int, lost_rate: int, dealt_rate: float, received_rate: float) -> list[cross_stage.MatchRow]:
        return [
            _match_row(
                boot_index,
                active_frames,
                taken=taken_rate * (boot_index + 1),
                lost=lost_rate * (boot_index + 1),
                dealt=dealt_rate * (boot_index + 1),
                received=received_rate * (boot_index + 1),
            )
            for boot_index, active_frames in enumerate(frames)
        ]

    result = historical_gameplay.blocked_factorial_interaction(
        rows(0, 0, 0.0, 0.0),
        rows(1, 0, 10.0, 0.0),
        rows(0, 0, 0.0, 0.0),
        rows(3, 1, 30.0, 5.0),
        expected_boots=2,
        bootstrap_resamples=2_000,
        seed=7,
    )

    coverage = cast(dict[str, object], result["coverage"])
    estimands = cast(dict[str, dict[str, dict[str, float]]], result["estimands"])
    assert coverage == {
        "excluded_not_jointly_active": 0,
        "matched_scheduled_boots": 2,
        "rated_boots": 2,
        "zero_active_by_cell": {
            "early_historical": 0,
            "early_portable": 0,
            "slump_historical": 0,
            "slump_portable": 0,
        },
    }
    expected = {
        "stocks_taken_per_min": 2.0,
        "stocks_lost_per_min": 1.0,
        "damage_dealt_per_min": 20.0,
        "damage_taken_per_min": 5.0,
        "net_stock_per_min": 1.0,
        "net_damage_per_min": 15.0,
    }
    for estimand in estimands.values():
        for metric, point in expected.items():
            assert estimand[metric] == {
                "ci_hi": pytest.approx(point),
                "ci_lo": pytest.approx(point),
                "point": pytest.approx(point),
            }
    for estimand in estimands.values():
        assert estimand["net_stock_per_min"]["point"] == pytest.approx(
            estimand["stocks_taken_per_min"]["point"] - estimand["stocks_lost_per_min"]["point"]
        )
        assert estimand["net_damage_per_min"]["point"] == pytest.approx(
            estimand["damage_dealt_per_min"]["point"] - estimand["damage_taken_per_min"]["point"]
        )
