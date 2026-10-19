"""Compare FP32 batched evaluation with independent serial MSL matches."""

import copy
import dataclasses
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from hal.eval.conditioned_matchups import EvaluationStart
from hal.eval.conditioned_matchups import ScheduledMatch
from hal.eval.conditioned_matchups import build_batched_policy
from hal.eval.conditioned_matchups import build_policy
from hal.eval.conditioned_matchups import run_msl_matches
from hal.eval.self_play_checkpoint import load_evaluation_checkpoint
from hal.eval.self_play_policy import PlayerCondition
from hal.sim import native_build
from hal.sim.inputs import native_controller_readback
from hal.sim.native import NativeMatch
from hal.sim.native import NativePlayer
from hal.sim.native import NativeRolloutBatch
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import Slot

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("distinct_weights", [False, True], ids=["shared-weights", "two-weights"])
def test_proxy_batched_msl_actions_and_outcomes_through_resets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, distinct_weights: bool
) -> None:
    assert torch.cuda.is_available(), "the proxy parity test requires CUDA"
    root = Path(os.environ.get("HAL_MSL_SOURCE", "~/src/melee-sim-light")).expanduser().resolve()
    library = root / "build/melee_core/python/libmelee_core.so"
    assert library.is_file(), f"required MSL library is missing: {library}"
    assert (root / "data/raw/manifest.json").is_file(), "required MSL data is missing"
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.setenv("MSL_CORE_LIBRARY", str(library))
    monkeypatch.setenv("HAL_NATIVE_LIBRARY", str(native_build.build(root, tmp_path / "libhal_native.so")))
    checkpoint_path = Path("runs/proxy/final.pt")
    assert checkpoint_path.is_file(), "required O63 proxy is missing"
    checkpoint = load_evaluation_checkpoint(checkpoint_path, device=torch.device("cuda"), decoder_precision="fp32")
    opponent = checkpoint
    if distinct_weights:
        model = copy.deepcopy(checkpoint.model)
        with torch.no_grad():
            weight = dict(model.named_parameters())["temporal.outputs.buttons.down.weight"]
            weight.add_(torch.linspace(-0.001, 0.001, weight.numel(), device=weight.device).view_as(weight))
        digest = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        opponent = dataclasses.replace(checkpoint, model=model, policy_sha256=digest.hexdigest())
        assert opponent.policy_sha256 != checkpoint.policy_sha256
    checkpoints = (checkpoint, opponent)
    conditions = (PlayerCondition(delay=8, replan=4), PlayerCondition(delay=10, replan=2))
    schedule = tuple(
        ScheduledMatch(EvaluationStart("fd", 25, (1, 22)), seed, port, str(seed))
        for seed in (17, 29)
        for port in (1, 2)
    )
    batched = build_batched_policy(checkpoints, conditions, schedule)
    assert len(batched.models) == (2 if distinct_weights else 1)
    assert all(model.compiled for model in batched.models)
    model_by_slot = {stream.slot: model for model in batched.models for stream in model.streams}
    serial = [build_policy(checkpoints, conditions, seed=game.seed, focal_port=game.focal_port) for game in schedule]
    # One-hit stockouts exercise winners and asynchronous resets after handoff.
    configs = tuple(
        NativeMatch(
            game.start.stage,
            (
                NativePlayer(1, game.characters[0], start_percent=100),
                NativePlayer(2, game.characters[1], start_percent=100),
            ),
            game.seed,
            stocks=1,
            damage_ratio=100,
            max_frame=400 + lane * 4,
        )
        for lane, game in enumerate(schedule)
    )
    mask = np.ones(len(schedule), dtype=np.bool_)
    outcomes = []
    states_trace: list[np.ndarray] = []
    actions_trace: list[np.ndarray] = []
    resets_trace: list[np.ndarray] = []
    max_log_gap = 0.0
    with (
        NativeRolloutBatch(root / "data", len(schedule)) as reference,
        NativeRolloutBatch(root / "data", len(schedule)) as treatment,
    ):
        expected_frame = reference.reset(configs, mask)
        actual_frame = treatment.reset(configs, mask)
        for at in range(560):
            assert expected_frame.records is not None and actual_frame.records is not None
            assert actual_frame.records.tobytes() == expected_frame.records.tobytes(), f"MSL state at {at}"
            states_trace.append(actual_frame.records.copy())
            actions = batched.plan_native(actual_frame)
            observed = native_controller_readback(expected_frame.applied_action)
            expected_actions = np.zeros_like(actions)
            for lane, policy in enumerate(serial):
                flat = {name: value[lane].item() for name, value in expected_frame.columns.items()}
                for port, player in policy.players.items():
                    expected_actions[lane, port - 1] = player.step(
                        ObservationRow(
                            int(expected_frame.frame_id[lane]),
                            flat,
                            observed[lane, port - 1],
                            bool(expected_frame.reset[lane]),
                        )
                    )
                    slot = Slot(lane, port)
                    sampled = model_by_slot[slot].last_sampled[slot]
                    assert player.last_sampled is not None and player.last_log_probs is not None
                    torch.testing.assert_close(sampled, player.last_sampled, rtol=0, atol=0)
                    gap = model_by_slot[slot].last_log_probs[slot] - player.last_log_probs
                    max_log_gap = max(max_log_gap, float(gap.abs().max()))
            np.testing.assert_array_equal(actions, expected_actions)
            actions_trace.append(actions.copy())
            expected_frame = reference.step(expected_actions, mask)
            actual_frame = treatment.step(actions, mask)
            ended = expected_frame.terminated | expected_frame.truncated
            resets_trace.append(ended.copy())
            if ended.any():
                assert actual_frame.records is not None and expected_frame.records is not None
                outcomes.append(expected_frame.records[ended].copy())
                assert actual_frame.records[ended].tobytes() == expected_frame.records[ended].tobytes()
                expected_frame = reference.reset(configs, ended)
                actual_frame = treatment.reset(configs, ended)
        assert actual_frame.records is not None
        final_records = actual_frame.records.copy()
    assert outcomes, "the test must compare completed episodes"
    assert all(outcome["terminated"].all() for outcome in outcomes), "the fixture must complete stockouts"
    print(f"maximum log-probability gap: {max_log_gap:.9g} nats; completed resets: {sum(map(len, outcomes))}")
    # A serial simulator must give the actors the same observations as a lane.
    for lane, config in enumerate(configs):
        with NativeRolloutBatch(root / "data", 1) as simulator:
            one = np.ones(1, dtype=np.bool_)
            frame = simulator.reset((config,), one)
            for states, actions, resets in zip(states_trace, actions_trace, resets_trace, strict=True):
                assert frame.records is not None
                assert frame.records[0].tobytes() == states[lane].tobytes()
                frame = simulator.step(actions[lane : lane + 1], one)
                if resets[lane]:
                    frame = simulator.reset((config,), one)
            assert frame.records is not None
            assert frame.records[0].tobytes() == final_records[lane].tobytes()
    short_schedule = schedule[:3]
    short_policy = build_batched_policy(checkpoints, conditions, short_schedule)
    limits = (1, 4, 28800)
    short_configs = tuple(
        NativeMatch(
            25,
            (NativePlayer(1, game.characters[0]), NativePlayer(2, game.characters[1])),
            game.seed,
            max_frame=limit,
        )
        for game, limit in zip(short_schedule, limits, strict=True)
    )
    with NativeRolloutBatch(root / "data", 3) as simulator:
        results = run_msl_matches(simulator, short_policy, short_configs, max_frames=130)
    assert [reason for _, reason, _ in results] == ["timeout", "timeout", "frame_budget"]
    assert [len(trajectory) for trajectory, _, _ in results] == [125, 128, 130]
    assert [int(trajectory.frame_id[-1]) for trajectory, _, _ in results] == [1, 4, 6]
    assert len({seconds for _, _, seconds in results}) == 1
    assert results[0][0].post[1]["percent"].dtype == np.float64
