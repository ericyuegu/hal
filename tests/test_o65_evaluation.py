"""Timing, path parity, paired statistics, checkpoints and periodic evaluation."""

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Literal
from typing import cast

import melee
import numpy as np
import pytest
import torch

from hal.data.feature_stats import FeatureStats
from hal.data.feature_stats import consolidate_key
from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.eval.batched_self_play import BatchedSelfPlayModel
from hal.eval.batched_self_play import BatchedSelfPlayPolicy
from hal.eval.batched_self_play import EvaluationStream
from hal.eval.conditioned_matchups import EvaluationMatch
from hal.eval.conditioned_matchups import EvaluationSet
from hal.eval.conditioned_matchups import EvaluationStart
from hal.eval.conditioned_matchups import compare_simulators
from hal.eval.conditioned_matchups import evaluate
from hal.eval.conditioned_matchups import native_observation_rows
from hal.eval.conditioned_matchups import pilot_protocol
from hal.eval.conditioned_matchups import score_match
from hal.eval.conditioned_matchups import summarize_matches
from hal.eval.periodic_self_play import PeriodicEvaluator
from hal.eval.self_play_checkpoint import load_evaluation_checkpoint
from hal.eval.self_play_checkpoint import save_evaluation_checkpoint
from hal.eval.self_play_policy import PlayerCondition
from hal.eval.self_play_policy import SelfPlayPlayer
from hal.eval.self_play_policy import SelfPlayPolicy
from hal.eval.self_play_policy import evaluation_uniforms
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.conditioning import ConditioningRanges
from hal.models.self_play import DecoderPrecision
from hal.models.self_play import SelfPlayTransformer
from hal.representation.player_identity import PlayerVocabulary
from hal.sim.native import OBSERVATION_FIELDS
from hal.sim.native import NativeFrameBatch
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import Slot
from hal.sim.trajectory import Trajectory
from hal.wire import ACTION_DIM


def small_model(precision: DecoderPrecision = "fp32") -> SelfPlayTransformer:
    vocabulary = PlayerVocabulary(())
    cfg = ActionSequenceConfig(
        d_model=16,
        n_layers=1,
        n_heads=2,
        temporal_d_model=16,
        temporal_layers=1,
        temporal_heads=2,
        temporal_ff_dim=32,
        group_head_dim=8,
        value_hidden_dim=8,
        return_embed_dim=8,
        player_vocab_size=vocabulary.size,
        player_vocab_sha256=vocabulary.sha256,
    )
    return SelfPlayTransformer(
        cfg,
        ConditioningRanges((math.log(0.01), math.log(100)), (8, 10), (2, 4)),
        vocabulary,
        max_opponent_queue=14,
        critic_hidden_dim=8,
        decoder_precision=precision,
    ).eval()


def stats() -> dict[str, FeatureStats]:
    return {
        consolidate_key(name): FeatureStats(0, 1, -100, 100)
        for name in OBSERVATION_FIELDS
        if np.dtype(MDS_PER_FRAME_DTYPES[name]).kind == "f"
    }


def row(at: int, *, reset: bool = False) -> ObservationRow:
    flat = {name: 0 for name in OBSERVATION_FIELDS}
    flat.update(stage=25, p1_character=1, p2_character=22, p1_stock=4, p2_stock=4)
    return ObservationRow(at - 123, flat, np.zeros(ACTION_DIM, dtype=np.float32), reset)


@pytest.mark.parametrize(("delay", "replan"), [(8, 4), (10, 2)])
def test_actions_execute_after_delay_and_reset_clears_queue(delay: int, replan: int) -> None:
    model = small_model()
    player = SelfPlayPlayer(
        model, stats(), PlayerCondition(delay=delay, replan=replan), port=1, seed=17, logical_player=0
    )
    outputs = []
    first_sample = None
    for at in range(delay + replan):
        outputs.append(player.step(row(at)))
        if at == 0:
            assert player.last_sampled is not None
            first_sample = model.codec.dequantize(player.last_sampled).detach().numpy()
    np.testing.assert_array_equal(outputs[:delay], np.zeros((delay, ACTION_DIM)))
    assert first_sample is not None
    np.testing.assert_array_equal(outputs[delay : delay + replan], first_sample)
    np.testing.assert_array_equal(player.step(row(0, reset=True)), np.zeros(ACTION_DIM))
    assert player.frame == 1


def test_dolphin_and_native_actions_match_across_cache_handoff() -> None:
    torch.manual_seed(71)
    model = small_model()

    def policy() -> SelfPlayPolicy:
        return SelfPlayPolicy(
            {
                port: SelfPlayPlayer(
                    model,
                    stats(),
                    PlayerCondition(delay=8 if port == 1 else 10, replan=4 if port == 1 else 2),
                    port=port,
                    seed=42,
                    logical_player=port - 1,
                )
                for port in (1, 2)
            }
        )

    native, dolphin = policy(), policy()
    for at in range(262):
        rows = {Slot(0, port): [row(at, reset=at == 253)] for port in (1, 2)}
        submitted = np.zeros((1, 2, ACTION_DIM), dtype=np.float32)
        submitted[0, :, :2] = [0.9875, 0.1625]
        flat = rows[Slot(0, 1)][0].flat
        frame = NativeFrameBatch(
            {name: np.array([value]) for name, value in flat.items()},
            np.array([at - 123], dtype=np.int32),
            submitted,
            np.zeros((1, 2, 7), dtype=np.int16),
            np.zeros((1, 2), dtype=np.float32),
            np.array([False]),
            np.array([False]),
            np.array([at == 253]),
        )
        native_rows = native_observation_rows(frame)
        for slot, observations in rows.items():
            processed = np.zeros(ACTION_DIM, dtype=np.float32)
            processed[:2] = [0.975, 0]
            rows[slot] = [replace(observations[0], action=processed)]
        # The broker may iterate ports in a different order; counters belong to roles.
        expected = native.plan_rows(native_rows)
        actual = dolphin.plan_rows(dict(reversed(tuple(rows.items()))))
        for slot in rows:
            np.testing.assert_array_equal(expected[slot], actual[slot])
    assert native.players[1].frame == 9


def test_observation_gap_and_unsupported_executed_offsets_fail() -> None:
    model = small_model()
    with pytest.raises(ValueError, match="unsupervised"):
        SelfPlayPlayer(model, stats(), PlayerCondition(delay=10, replan=4), port=1, seed=1, logical_player=0)
    player = SelfPlayPlayer(model, stats(), PlayerCondition(), port=1, seed=1, logical_player=0)
    player.step(row(0))
    with pytest.raises(ValueError, match="consecutive"):
        player.step(row(2))


def test_counter_rng_separates_roles_and_preserves_shared_offsets() -> None:
    a = evaluation_uniforms(17, 0, 128, 4)
    np.testing.assert_array_equal(a[:2], evaluation_uniforms(17, 0, 128, 2))
    assert not np.array_equal(a, evaluation_uniforms(17, 1, 128, 4))


def test_batched_actions_match_serial_through_handoffs_and_partial_resets() -> None:
    torch.manual_seed(71)
    model = small_model()
    streams = tuple(
        EvaluationStream(
            Slot(match, port),
            PlayerCondition(delay=8 if role == 0 else 10, replan=4 if role == 0 else 2),
            seed,
            role,
        )
        for match, (seed, focal_port) in enumerate(((17, 1), (17, 2), (29, 1), (29, 2)))
        for role, port in enumerate((focal_port, 3 - focal_port))
    )
    batched = BatchedSelfPlayPolicy((BatchedSelfPlayModel(model, stats(), streams, compiled=False),))
    serial = {
        stream.slot: SelfPlayPlayer(
            model,
            stats(),
            stream.condition,
            port=stream.slot.port,
            seed=stream.seed,
            logical_player=stream.logical_player,
        )
        for stream in streams
    }
    for at in range(270):
        rows = {stream.slot: [row(at, reset=at == 253 and stream.slot.match % 2 == 0)] for stream in streams}
        # Non-neutral observed inputs exercise the feature and decoder boundaries.
        for slot, values in rows.items():
            action = np.zeros(ACTION_DIM, dtype=np.float32)
            action[:2] = [0.975, -0.3]
            rows[slot] = [replace(values[0], action=action)]
        actual = batched.plan_rows(dict(reversed(tuple(rows.items()))))
        for slot, values in rows.items():
            expected = serial[slot].step(values[0])
            np.testing.assert_array_equal(actual[slot][0], expected)
            torch.testing.assert_close(batched.models[0].last_sampled[slot], serial[slot].last_sampled, rtol=0, atol=0)


def test_batched_stream_validation_and_missing_stream_return() -> None:
    model = small_model()
    slot = Slot(0, 1)
    stream = EvaluationStream(slot, PlayerCondition(), 17, 0)
    with pytest.raises(ValueError, match="unsupervised"):
        BatchedSelfPlayModel(
            model, stats(), (replace(stream, condition=PlayerCondition(delay=10, replan=4)),), compiled=False
        )
    batched = BatchedSelfPlayPolicy(
        (BatchedSelfPlayModel(model, stats(), (stream, replace(stream, slot=Slot(1, 1))), compiled=False),)
    )
    batched.plan_rows({slot: [row(0)], Slot(1, 1): [row(0)]})
    batched.plan_rows({slot: [row(1)]})
    with pytest.raises(ValueError, match="must reset"):
        batched.plan_rows({slot: [row(2)], Slot(1, 1): [row(1)]})
    batched.plan_rows({slot: [row(2)], Slot(1, 1): [row(0, reset=True)]})
    with pytest.raises(ValueError, match="consecutive"):
        batched.plan_rows({slot: [row(4)]})


def test_batched_policy_keeps_distinct_model_weights_and_cadences() -> None:
    torch.manual_seed(13)
    models = (small_model(), small_model())
    groups = []
    serial = {}
    for role, model in enumerate(models):
        streams = tuple(
            EvaluationStream(
                Slot(match, port),
                PlayerCondition(delay=8 if role == 0 else 10, replan=4 if role == 0 else 2),
                17,
                role,
            )
            for match, port in enumerate((1 + role, 2 - role))
        )
        groups.append(BatchedSelfPlayModel(model, stats(), streams, compiled=False))
        for stream in streams:
            serial[stream.slot] = SelfPlayPlayer(
                model,
                stats(),
                stream.condition,
                port=stream.slot.port,
                seed=stream.seed,
                logical_player=role,
            )
    batched = BatchedSelfPlayPolicy(tuple(groups))
    for at in range(24):
        rows = {slot: [row(at)] for slot in serial}
        actual = batched.plan_rows(rows)
        for slot, player in serial.items():
            np.testing.assert_array_equal(actual[slot][0], player.step(rows[slot][0]))


def test_batched_evaluation_rejects_sliding_history_and_invalid_workers(tmp_path: Path) -> None:
    missing = tmp_path / "absent.pt"
    conditions = (PlayerCondition(), PlayerCondition())
    with pytest.raises(ValueError, match="requires block history"):
        evaluate(
            missing,
            (missing,),
            tmp_path / "sliding",
            protocol=pilot_protocol(),
            split="selection",
            conditions=conditions,
            simulators=("dolphin",),
            history_mode="sliding",
        )
    with pytest.raises(ValueError, match="worker count"):
        evaluate(
            missing,
            (missing,),
            tmp_path / "workers",
            protocol=pilot_protocol(),
            split="selection",
            conditions=conditions,
            simulators=("dolphin",),
            dolphin_workers=0,
        )
    assert not (tmp_path / "sliding").exists()
    assert not (tmp_path / "workers").exists()


def test_selection_and_held_out_reject_seed_and_start_overlap() -> None:
    protocol = pilot_protocol()
    with pytest.raises(ValueError, match="seeds overlap"):
        replace(protocol, held_out=EvaluationSet(protocol.selection.seeds, protocol.held_out.starts))
    mirrored = replace(protocol.selection.starts[0], name="mirror", characters=(22, 1))
    with pytest.raises(ValueError, match="starts overlap"):
        replace(protocol, held_out=EvaluationSet((912,), (mirrored,)))


def match(pair: str, port: int, net: int, *, frames: int = 3600) -> EvaluationMatch:
    return EvaluationMatch(
        "msl", pair, 17, "fd", port, frames, net, net > 0, "terminal", {"timeout": 0, "ego_damage": 60}, 10, None, 17
    )


def test_paired_statistics_do_not_treat_frames_as_samples() -> None:
    matches = [match("a", 1, 2), match("a", 2, 0), match("b", 1, 4), match("b", 2, -2)]
    summary = summarize_matches(matches)
    assert summary["paired_seeds"] == 2
    assert summary["net_stocks_per_min"] == {"mean": 1, "low": 1, "high": 1}
    assert summary["net_stock_lcb"] == 1
    with pytest.raises(ValueError, match="orientations"):
        summarize_matches(matches[:-1])
    budget = [replace(item, won=None, stop_reason="frame_budget") for item in matches]
    assert summarize_matches(budget)["win_rate"] == {"mean": None, "low": None, "high": None}


def test_simulator_divergence_requires_same_complete_schedule() -> None:
    matches = [match("a", 1, 2), match("a", 2, 0), match("b", 1, 4), match("b", 2, -2)]
    other = [replace(item, simulator="dolphin", events={"timeout": 0, "ego_damage": 40}) for item in matches]
    result = compare_simulators(matches, other)
    events = cast(dict[str, object], result["events"])
    assert events["ego_damage"] == {"mean": 20, "low": 20, "high": 20}
    with pytest.raises(ValueError, match="seeds or starts"):
        compare_simulators(matches, [replace(item, seed=99) for item in other])


def test_event_counts_use_onsets_and_ignore_countdown() -> None:
    post = {}
    for port in (1, 2):
        post[port] = {
            "stock": np.array([4, 4, 4, 3, 3, 3]),
            "percent": np.array([0, 10, 20, 0, 10, 20]),
            "action": np.array([melee.Action.EDGE_CATCHING.value] * 3 + [melee.Action.NEUTRAL_TECH.value] * 3),
            "position_x": np.array([0, 100, 100, 0, 0, 0]),
        }
    trajectory = Trajectory(np.array([-1, 0, 1, 2, 3, 4]), post, np.ones(6, dtype=np.uint32))
    scored = score_match(
        trajectory,
        simulator="msl",
        pair="a",
        seed=1,
        start=EvaluationStart("fd", 25, (1, 22)),
        focal_port=1,
        stop_reason="frame_budget",
        wall_seconds=1,
    )
    assert scored.active_frames == 5
    assert scored.events["ego_ledge_grabs_per_min"] == 720
    assert scored.events["ego_techs_per_min"] == 720
    assert scored.events["ego_deaths_per_min"] == 720
    assert scored.events["ego_offstage_fraction"] == 0.4
    assert scored.won is None


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_evaluation_snapshot_roundtrip_and_schema_rejection(tmp_path: Path, precision: DecoderPrecision) -> None:
    model = small_model(precision)
    path = tmp_path / "actor.pt"
    provenance: dict[str, object] = {
        "reference_sha256": "a" * 64,
        "git_sha": "b" * 40,
        "update": 1,
        "data_manifests": {"ranked": "c" * 64},
    }
    save_evaluation_checkpoint(path, model, stats(), provenance=provenance)
    loaded = load_evaluation_checkpoint(path, device=torch.device("cpu"))
    assert loaded.provenance == provenance
    assert loaded.model.decoder_precision == precision
    assert (
        SelfPlayPlayer(
            loaded.model, stats(), PlayerCondition(), port=1, seed=0, logical_player=0
        ).decoder.model.decoder_precision
        == precision
    )
    with pytest.raises(ValueError, match="decoder precision differs"):
        load_evaluation_checkpoint(
            path, device=torch.device("cpu"), decoder_precision="fp32" if precision == "bf16" else "bf16"
        )
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(loaded.model.state_dict()[name], tensor, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        save_evaluation_checkpoint(path, model, stats(), provenance=provenance)
    raw = torch.load(path, weights_only=False)
    raw["decoder_precision"] = "fp16"
    torch.save(raw, path)
    with pytest.raises(ValueError, match="decoder precision"):
        load_evaluation_checkpoint(path, device=torch.device("cpu"))
    raw["decoder_precision"] = precision
    raw["format"] = "future"
    torch.save(raw, path)
    with pytest.raises(ValueError, match="unsupported checkpoint"):
        load_evaluation_checkpoint(path, device=torch.device("cpu"))


def test_periodic_job_is_nonblocking_and_bounded(tmp_path: Path) -> None:
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"actor")
    job = PeriodicEvaluator(tmp_path / "jobs", every=2, arguments=("--anchors", str(checkpoint)))
    assert not job.submit(checkpoint, update=1)
    assert job.submit(checkpoint, update=2)
    assert not job.submit(checkpoint, update=4)
    process = job.process
    assert process is not None
    # Invalid actor must fail visibly, with no successful result publication.
    process.wait(timeout=30)
    with pytest.raises(RuntimeError, match="evaluation exited"):
        job.poll()
    assert job.process is None and job.log is None
    job.close()


@pytest.mark.parametrize("arguments", [("--split", "held_out"), ("--checkpoint=x",), ("--output", "x")])
def test_periodic_job_cannot_consume_held_out_or_override_checkpoint(
    tmp_path: Path, arguments: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match="owns"):
        PeriodicEvaluator(tmp_path, every=1, arguments=arguments)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA cache type requires GPU")
def test_cuda_evaluation_cache_matches_model_type() -> None:
    model = small_model().cuda()
    player = SelfPlayPlayer(model, stats(), PlayerCondition(), port=1, seed=17, logical_player=0)
    result = player.step(row(0))
    assert result.shape == (ACTION_DIM,)
    assert player.last_log_probs is not None
    assert torch.isfinite(player.last_log_probs).all()


def test_held_out_set_can_only_be_used_once_per_result_root(tmp_path: Path) -> None:
    from hal.eval.conditioned_matchups import claim_held_out

    protocol = pilot_protocol()
    claim = claim_held_out(tmp_path / "final-a", protocol.held_out, "a" * 64)
    assert claim.is_file()
    with pytest.raises(FileExistsError):
        claim_held_out(tmp_path / "final-b", protocol.held_out, "b" * 64)


def test_protocol_schema_and_separate_start_sets(tmp_path: Path) -> None:
    from dataclasses import asdict

    from hal.eval.conditioned_matchups import read_protocol

    protocol = pilot_protocol(max_frames=720)
    path = tmp_path / "protocol.json"
    raw = {"format": "hal.o65.protocol.v1", **asdict(protocol)}
    path.write_text(json.dumps(raw))
    assert read_protocol(path) == protocol
    raw["ignored"] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="schema"):
        read_protocol(path)


@pytest.mark.parametrize("history_mode", ["block", "sliding"])
def test_cached_evaluation_matches_full_block_windows_and_queue(history_mode: Literal["block", "sliding"]) -> None:
    from hal.inference.decision_decoder import DecisionDecoder

    torch.manual_seed(94)
    model = small_model()
    player = SelfPlayPlayer(
        model, stats(), PlayerCondition(), port=1, seed=17, logical_player=0, history_mode=history_mode
    )
    decoder = DecisionDecoder(model, ((8, 4),), compiled=False)
    tokens = []
    with torch.inference_mode():
        for at in range(260):
            observation = row(at)
            # Construct the independent full-window path from the frame features.
            columns = {
                name: torch.tensor(
                    [[value]], dtype=torch.int64 if np.dtype(MDS_PER_FRAME_DTYPES[name]).kind == "i" else torch.float32
                )
                for name, value in observation.flat.items()
            }
            context = player.builder(
                columns,
                torch.zeros(1, 1, 2, ACTION_DIM),
                torch.zeros(1, 1, dtype=torch.long),
                torch.tensor([[at == 0]]),
                torch.zeros(1, dtype=torch.long),
            )
            tokens.append(model.context_tokens(context.features))
            if at in (0, 124, 128, 252, 256):
                queue = torch.stack(tuple(player.queue))[None] if at else player.neutral.repeat(1, 8, 1)
                start = max(0, (at // 128) * 128 - 128) if history_mode == "block" else max(0, at - 255)
                hidden = model.trunk.forward_dense(
                    torch.cat(tokens[start:], dim=1), torch.zeros(1, dtype=torch.long), force_dense=True
                )
                expected = decoder(
                    hidden,
                    player.neutral[None],
                    queue,
                    torch.zeros(1),
                    torch.tensor([8]),
                    torch.tensor([4]),
                    torch.from_numpy(evaluation_uniforms(17, 0, at, 4)),
                    sampled_offsets=4,
                )
            player.step(observation)
            if at in (0, 124, 128, 252, 256):
                assert player.last_sampled is not None and player.last_log_probs is not None
                torch.testing.assert_close(player.last_sampled, expected.actions[0], rtol=0, atol=0)
                torch.testing.assert_close(player.last_log_probs, expected.log_probs[0], rtol=2e-5, atol=2e-5)


def test_repeated_physical_start_is_not_an_independent_sample() -> None:
    start = EvaluationStart("fd", 25, (1, 22))
    with pytest.raises(ValueError, match="physical configurations"):
        EvaluationSet((17,), (start, replace(start, name="duplicate")))
    with pytest.raises(ValueError, match="physical configurations"):
        EvaluationSet((17,), (start, replace(start, name="mirror", characters=(22, 1))))
    with pytest.raises(ValueError):
        replace(start, name="../escape")


def test_checkpoint_rejects_tensor_casts_and_nonfinite_weights(tmp_path: Path) -> None:
    model = small_model()
    path = tmp_path / "actor.pt"
    provenance: dict[str, object] = {
        "reference_sha256": "a" * 64,
        "git_sha": "b" * 40,
        "update": 1,
        "data_manifests": {"ranked": "c" * 64},
    }
    save_evaluation_checkpoint(path, model, stats(), provenance=provenance)
    raw = torch.load(path, weights_only=False)
    name = "observation_encoder.weight"
    original = raw["model"][name]
    raw["model"][name] = original.double()
    torch.save(raw, path)
    with pytest.raises(ValueError, match="dtype"):
        load_evaluation_checkpoint(path, device=torch.device("cpu"))
    raw["model"][name] = original.clone()
    raw["model"][name][0, 0] = float("nan")
    torch.save(raw, path)
    with pytest.raises(ValueError, match="finite"):
        load_evaluation_checkpoint(path, device=torch.device("cpu"))


def test_critic_changes_do_not_make_distinct_policy_pairs(tmp_path: Path) -> None:
    from hal.eval.conditioned_matchups import evaluate

    model = small_model()
    provenance: dict[str, object] = {
        "reference_sha256": "a" * 64,
        "git_sha": "b" * 40,
        "update": 1,
        "data_manifests": {"ranked": "c" * 64},
    }
    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    save_evaluation_checkpoint(first, model, stats(), provenance=provenance)
    with torch.no_grad():
        model.value_head.down.weight.add_(1)
    save_evaluation_checkpoint(second, model, stats(), provenance=provenance)
    with pytest.raises(ValueError, match="distinct policies"):
        evaluate(
            first,
            (second,),
            tmp_path / "results",
            protocol=pilot_protocol(),
            split="selection",
            conditions=(PlayerCondition(), PlayerCondition()),
            data_dir=tmp_path,
            device="cpu",
            compare_events=True,
        )


def test_native_readback_matches_dolphin_for_deadzone_and_rim_angles() -> None:
    submitted = np.zeros((1, 2, ACTION_DIM), dtype=np.float32)
    submitted[0, 0, :2] = [0.9875, 0.1625]
    submitted[0, 1, :2] = [0.7125, 0.7125]
    frame = NativeFrameBatch(
        {},
        np.array([17], dtype=np.int32),
        submitted,
        np.zeros((1, 2, 7), dtype=np.int16),
        np.zeros((1, 2), dtype=np.float32),
        np.array([False]),
        np.array([False]),
        np.array([False]),
    )
    observed = native_observation_rows(frame)
    np.testing.assert_array_equal(observed[Slot(0, 1)][0].action[:2], np.array([0.975, 0], dtype=np.float32))
    np.testing.assert_array_equal(observed[Slot(0, 2)][0].action[:2], np.array([0.7, 0.7], dtype=np.float32))
    # Converting readback must not change the committed action stream itself.
    np.testing.assert_array_equal(submitted[0, 0, :2], np.array([0.9875, 0.1625], dtype=np.float32))


def test_stockout_and_timeout_have_distinct_outcomes() -> None:
    post = {
        port: {
            "stock": np.array([4, 4]) if port == 1 else np.array([1, 0]),
            "percent": np.zeros(2),
            "action": np.zeros(2),
            "position_x": np.zeros(2),
        }
        for port in (1, 2)
    }
    trajectory = Trajectory(np.array([0, 1]), post, np.array([17, 18], dtype=np.uint32))
    start = EvaluationStart("fd", 25, (1, 22))
    terminal = score_match(
        trajectory,
        simulator="msl",
        pair="a",
        seed=17,
        start=start,
        focal_port=1,
        stop_reason="terminal",
        wall_seconds=1,
    )
    assert terminal.won is True and terminal.net_stocks == 4
    assert terminal.events["timeout"] == 0
    timeout = score_match(
        trajectory,
        simulator="msl",
        pair="a",
        seed=17,
        start=start,
        focal_port=1,
        stop_reason="timeout",
        wall_seconds=1,
    )
    assert timeout.won is None and timeout.events["timeout"] == 1
    with pytest.raises(ValueError, match="exactly one surviving"):
        score_match(
            replace(trajectory, post={1: post[1], 2: post[1]}),
            simulator="msl",
            pair="a",
            seed=17,
            start=start,
            focal_port=1,
            stop_reason="terminal",
            wall_seconds=1,
        )


def test_feature_statistics_are_part_of_policy_identity(tmp_path: Path) -> None:
    model = small_model()
    provenance: dict[str, object] = {
        "reference_sha256": "a" * 64,
        "git_sha": "b" * 40,
        "update": 1,
        "data_manifests": {"ranked": "c" * 64},
    }
    before, after = tmp_path / "before.pt", tmp_path / "after.pt"
    statistics = stats()
    save_evaluation_checkpoint(before, model, statistics, provenance=provenance)
    name = next(iter(statistics))
    statistics[name] = replace(statistics[name], mean=1)
    save_evaluation_checkpoint(after, model, statistics, provenance=provenance)
    left = load_evaluation_checkpoint(before, device=torch.device("cpu"))
    right = load_evaluation_checkpoint(after, device=torch.device("cpu"))
    assert left.policy_sha256 != right.policy_sha256


def test_early_dolphin_exit_cannot_be_scored_as_a_budget_stop() -> None:
    from peppi_py.game import EndMethod

    from hal.eval.conditioned_matchups import dolphin_stop_reason

    assert dolphin_stop_reason(None, captured_frames=132, max_frames=132) == "frame_budget"
    assert dolphin_stop_reason(EndMethod.TIME, captured_frames=120, max_frames=132) == "timeout"
    assert dolphin_stop_reason(EndMethod.GAME, captured_frames=120, max_frames=132) == "terminal"
    with pytest.raises(RuntimeError, match="ended early"):
        dolphin_stop_reason(None, captured_frames=131, max_frames=132)
    with pytest.raises(ValueError, match="frame budget"):
        dolphin_stop_reason(None, captured_frames=133, max_frames=132)


def test_critic_architecture_does_not_change_policy_identity(tmp_path: Path) -> None:
    first = small_model()
    second = SelfPlayTransformer(
        replace(first.cfg, value_hidden_dim=12),
        first.ranges,
        PlayerVocabulary(()),
        max_opponent_queue=14,
        critic_hidden_dim=8,
        decoder_precision=first.decoder_precision,
    ).eval()
    tensors = second.state_dict()
    for name, tensor in first.state_dict().items():
        if not name.startswith(("value_head.", "percentile_head.", "critic_residual.")):
            tensors[name] = tensor
    second.load_state_dict(tensors)
    provenance: dict[str, object] = {
        "reference_sha256": "a" * 64,
        "git_sha": "b" * 40,
        "update": 1,
        "data_manifests": {"ranked": "c" * 64},
    }
    before, after = tmp_path / "before.pt", tmp_path / "after.pt"
    save_evaluation_checkpoint(before, first, stats(), provenance=provenance)
    save_evaluation_checkpoint(after, second, stats(), provenance=provenance)
    left = load_evaluation_checkpoint(before, device=torch.device("cpu"))
    right = load_evaluation_checkpoint(after, device=torch.device("cpu"))
    assert left.policy_sha256 == right.policy_sha256


def test_decoder_precision_changes_policy_identity_and_rejects_old_artifacts(tmp_path: Path) -> None:
    control, treatment = small_model("fp32"), small_model("bf16")
    treatment.load_state_dict(control.state_dict())
    provenance: dict[str, object] = {
        "reference_sha256": "a" * 64,
        "git_sha": "b" * 40,
        "update": 1,
        "data_manifests": {"ranked": "c" * 64},
    }
    paths = (tmp_path / "fp32.pt", tmp_path / "bf16.pt")
    for path, model in zip(paths, (control, treatment), strict=True):
        save_evaluation_checkpoint(path, model, stats(), provenance=provenance)
    loaded = [load_evaluation_checkpoint(path, device=torch.device("cpu")) for path in paths]
    assert loaded[0].policy_sha256 != loaded[1].policy_sha256
    raw = torch.load(paths[0], weights_only=False)
    del raw["decoder_precision"]
    raw["format"] = "hal.o65.evaluation.v1"
    torch.save(raw, paths[0])
    with pytest.raises(ValueError, match="unsupported checkpoint"):
        load_evaluation_checkpoint(paths[0], device=torch.device("cpu"))
