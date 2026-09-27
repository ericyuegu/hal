import json
import signal
import threading
import time
from dataclasses import replace
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import melee
import numpy as np
import pytest
import torch
from peppi_py.game import EndMethod

import hal.netplay_service.runner as runner
from hal.eval.qualification import RealtimeBudgetCheck
from hal.eval.replays import ReplayEnd
from hal.eval.results import NetplayProgress
from hal.eval.results import PlayResult
from hal.eval.results import ScheduleEvent
from hal.eval.scheduling import FrameTiming
from hal.eval.scheduling import PlanDecision
from hal.inference.api import PolicySpec
from hal.inference.api import PreparedInferenceProfile
from hal.inference.api import RuntimeConfig
from hal.inference.benchmark import LatencyMeasurement
from hal.netplay_service.domain import Job
from hal.netplay_service.domain import JobStatus
from hal.netplay_service.domain import MatchChoices
from hal.netplay_service.health import ChunkHealth
from hal.netplay_service.health import SlotState
from hal.netplay_service.health import SlotStatus
from hal.netplay_service.health import write_slot_status
from hal.netplay_service.queue import QueueStore
from hal.netplay_service.replays import ReplayMetadata
from hal.netplay_service.replays import UploadedReplay
from hal.sim.trajectory import Trajectory


def _job(*, stage: str | None = None) -> Job:
    return Job(
        id="reservation",
        player_code="CRYO#610",
        choices=MatchChoices("FOX", "IBDW#0", 2, stage),
        status=JobStatus.REMATCH_READY if stage else JobStatus.CONNECTING,
        queue_position=None,
        attempt=1,
        game_count=0,
        connect_code="HAL#1",
        actual_stage=None,
        last_result=None,
        error_code=None,
        connect_deadline=None,
        rematch_deadline=None,
        cancel_after_game=False,
        lease_owner="slot-0",
        lease_expires_at=None,
        created_at=0,
        updated_at=0,
    )


def _slot_config(tmp_path: Path) -> runner.SlotConfig:
    return runner.SlotConfig(
        slot=0,
        worker_id="slot-0",
        stream_id=0,
        database=tmp_path / "queue.sqlite3",
        user_json=tmp_path / "user.json",
        bot_connect_code="HAL#1",
        slippi_port=51441,
        iso_path=tmp_path / "game.ciso",
        dolphin_path=tmp_path / "Slippi.AppImage",
        replay_dir=tmp_path / "replays",
        status_path=tmp_path / "slot.json",
        policy_sha256="a" * 64,
        checkpoint_sha256="c" * 64,
        git_sha="b" * 40,
        recovery_cooldown_seconds=0.0,
    )


def _runner_config(tmp_path: Path) -> runner.RunnerConfig:
    return runner.RunnerConfig(
        database=tmp_path / "queue.sqlite3",
        policy=tmp_path / "policy.hal",
        user_jsons=(tmp_path / "account.json",),
        slippi_ports=(51441,),
        iso_path=tmp_path / "game.ciso",
        dolphin_path=tmp_path / "Slippi.AppImage",
        replay_dir=tmp_path / "replays",
        status_path=tmp_path / "status.json",
        git_sha="a" * 40,
    )


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True])
def test_runner_rejects_invalid_preparation_timeout(tmp_path: Path, value: float) -> None:
    with pytest.raises(ValueError, match="preparation timeout"):
        replace(_runner_config(tmp_path), preparation_timeout_seconds=value)


@pytest.mark.parametrize("recovery_deadline,expected", [(None, 1900.0), (220.0, 220.0), (99.0, 99.0)])
def test_generation_uses_cold_preparation_budget_and_preserves_recovery_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recovery_deadline: float | None,
    expected: float,
) -> None:
    from multiprocessing import Pipe

    process = Mock(pid=123, name="gpu")
    process.is_alive.return_value = False
    context = SimpleNamespace(Event=threading.Event, Pipe=Pipe, Process=Mock(return_value=process))
    ready = Mock(side_effect=TimeoutError("preparation deadline"))
    monkeypatch.setattr(runner.mp, "get_context", lambda _method: context)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(runner, "_await_engine_ready", ready)

    with pytest.raises(TimeoutError, match="preparation deadline"):
        runner._run_generation(
            _runner_config(tmp_path), ("BOT#1",), "b" * 64, runner._ShutdownFlag(), recovery_deadline=recovery_deadline
        )

    assert ready.call_args.kwargs["deadline"] == expected
    process.start.assert_called_once()


def test_recovery_keeps_120_seconds_after_a_long_cold_preparation_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(_runner_config(tmp_path), preparation_timeout_seconds=3600)
    generation = Mock(side_effect=[runner._EngineLost("hung GPU"), None])
    monkeypatch.setattr(runner, "_run_generation", generation)
    monkeypatch.setattr(runner, "_bot_connect_codes", lambda _paths: ("BOT#1",))
    monkeypatch.setattr(runner, "_sha256", lambda _path: "b" * 64)
    monkeypatch.setattr(runner.time, "monotonic", lambda: 100.0)

    runner.run(config)

    assert [call.kwargs["recovery_deadline"] for call in generation.call_args_list] == [None, 220.0]


def test_live_policy_settings_follow_job_revision(tmp_path: Path) -> None:
    store = QueueStore(tmp_path / "queue.sqlite3")
    credentials = store.create_job("CRYO#610", MatchChoices("FOX", "IBDW#0", 2))
    claimed = store.claim_next("slot-0")
    assert claimed is not None
    with runner._LivePolicySettings(store, claimed, "slot-0") as settings:
        assert settings.current() == (20.0, 1.0)
        store.update_policy(credentials.job.id, credentials.token, desired_return=None, temperature=0.9)
        deadline = time.monotonic() + 2.0
        while settings.current() != (None, 0.9) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert settings.current() == (None, 0.9)


def test_bot_account_code_is_read_without_fallback(tmp_path: Path) -> None:
    account = tmp_path / "user.json"
    account.write_text(json.dumps({"connectCode": "HAL#1", "playKey": "secret"}))
    assert runner._bot_connect_code(account) == "HAL#1"

    account.write_text(json.dumps({"connectCode": "hal#1"}))
    with pytest.raises(ValueError, match="exact uppercase"):
        runner._bot_connect_code(account)


def test_runner_rejects_duplicate_slippi_accounts(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"connectCode":"HAL#1"}')
    second.write_text('{"connectCode":"HAL#1"}')

    with pytest.raises(ValueError, match="distinct connect codes"):
        runner._bot_connect_codes((first, second))


def test_gpu_preparation_shares_one_model_across_distinct_netplay_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from multiprocessing import Pipe

    model = torch.nn.Linear(1, 1)
    model.cfg = SimpleNamespace(L_ctx=256)
    built: list[object] = []

    def build(artifact: object, *, device: torch.device, inference_dtype: torch.dtype) -> torch.nn.Module:
        assert device == torch.device("cpu") and inference_dtype == torch.float32
        built.append(artifact)
        return model

    artifact = SimpleNamespace(
        capability_version=2,
        spec=PolicySpec("059", "o59-history-decoder", (), (0, 2, 3)),
        statistics={},
        vocabulary=SimpleNamespace(codes=()),
        checkpoint_sha256="a" * 64,
        return_p90=19.976,
    )
    policies: list[object] = []

    class Policy:
        sampling_seed = 7

        def __init__(self, received_model: object, *_args: object, **_kwargs: object) -> None:
            assert received_model is model
            policies.append(self)

        def validate_prepared_profile(self, profile: PreparedInferenceProfile) -> None:
            assert profile.checkpoint_sha256 == "a" * 64
            engine_profiles.append(profile)

    engine_profiles: list[object] = []

    qualified: list[FrameTiming] = []

    def qualify(_policy: object, runtime: RuntimeConfig, _wait: float, *, shape: tuple[int, int]):
        timing = FrameTiming(runtime.require_single_delay(), 1, shape[1], 4, shape[0])
        qualified.append(timing)
        return RealtimeBudgetCheck((timing,), (LatencyMeasurement(shape[0], shape[1], (0.001,)),))

    monkeypatch.setattr(runner, "read_action_sequence_artifact", lambda _path: artifact)
    monkeypatch.setattr("hal.inference.engine.build_action_sequence_model", build)
    monkeypatch.setattr(runner, "ActionSequencePolicy", Policy)
    monkeypatch.setattr(runner, "check_realtime_budget", qualify)
    monkeypatch.setattr(runner, "freeze_inference_runtime", lambda: None)
    parent, child = Pipe()
    try:
        engine, ready = runner._prepare_netplay_engine(
            runner._InferenceProcessConfig(tmp_path / "policy.halpolicy", "cpu", 7, False, 2, 0.0005),
            {17: parent},
        )
    finally:
        parent.close()
        child.close()
    assert len(built) == 1
    assert len(policies) == 2
    assert len(engine_profiles) == 2
    assert tuple(qualified) == runner._NETPLAY_TIMINGS
    assert {profile.fixed_prefix_frames for profile in engine.profiles} == {3, 4}
    assert ready.profiles == tuple(engine.profiles)
    assert ready.checkpoint_sha256 == "a" * 64
    assert ready.context_frames == 256


def test_slot_process_inherits_ignored_terminal_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    process = Mock()
    child_connection = Mock()
    previous = Mock()
    set_signal = Mock(side_effect=(previous, None))
    monkeypatch.setattr(runner.signal, "signal", set_signal)

    runner._start_slot_process(process, child_connection)

    assert set_signal.call_args_list == [
        ((signal.SIGINT, signal.SIG_IGN),),
        ((signal.SIGINT, previous),),
    ]
    process.start.assert_called_once_with()
    child_connection.close.assert_called_once_with()


def test_generation_pipe_failure_closes_every_prior_endpoint() -> None:
    from multiprocessing import Pipe

    opened = []

    def pipe(duplex: bool = True):
        if len(opened) == 4:
            raise OSError("pipe allocation failed")
        pair = Pipe(duplex=duplex)
        opened.extend(pair)
        return pair

    with pytest.raises(OSError, match="pipe allocation failed"):
        runner._open_generation_pipes(SimpleNamespace(Pipe=pipe), 2)
    assert len(opened) == 4
    assert all(connection.closed for connection in opened)


def test_generation_cleanup_skips_unstarted_processes() -> None:
    process = Mock(pid=None)
    stop = Mock()
    runner._terminate_processes((process,), stop)
    stop.set.assert_called_once_with()
    process.join.assert_not_called()


def test_surviving_child_prevents_recovery_after_two_second_termination_budget() -> None:
    process = Mock(pid=123, name="stuck-gpu")
    process.is_alive.return_value = True
    stop = Mock()
    with pytest.raises(RuntimeError, match="termination exceeded two seconds"):
        runner._terminate_processes((process,), stop)
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    stop.set.assert_called_once()


def test_runner_does_not_admit_a_replacement_when_old_child_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = runner.RunnerConfig(
        database=tmp_path / "queue.sqlite3",
        policy=tmp_path / "policy.hal",
        user_jsons=(tmp_path / "account.json",),
        slippi_ports=(51441,),
        iso_path=tmp_path / "game.ciso",
        dolphin_path=tmp_path / "Slippi.AppImage",
        replay_dir=tmp_path / "replays",
        status_path=tmp_path / "status.json",
        git_sha="a" * 40,
    )
    generation = Mock(side_effect=RuntimeError("old child survived"))
    monkeypatch.setattr(runner, "_bot_connect_codes", lambda _paths: ("BOT#1",))
    monkeypatch.setattr(runner, "_sha256", lambda _path: "b" * 64)
    monkeypatch.setattr(runner, "_run_generation", generation)

    with pytest.raises(RuntimeError, match="old child survived"):
        runner.run(config)

    generation.assert_called_once()


def test_generation_failure_records_active_lease_even_if_gpu_child_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from multiprocessing import Pipe

    store = QueueStore(tmp_path / "queue.sqlite3")
    credentials = store.create_job("CRYO#610", MatchChoices("FOX", "IBDW#0", 2))
    worker_id = "generation-aaaaaaaaaaaaaaaa-slot-0"
    assert store.claim_next(worker_id) is not None
    store.mark_connecting(credentials.job.id, worker_id, "BOT#1")
    store.mark_playing(credentials.job.id, worker_id)
    config = runner.RunnerConfig(
        database=store.path,
        policy=tmp_path / "policy.hal",
        user_jsons=(tmp_path / "account.json",),
        slippi_ports=(51441,),
        iso_path=tmp_path / "game.ciso",
        dolphin_path=tmp_path / "Slippi.AppImage",
        replay_dir=tmp_path / "replays",
        status_path=tmp_path / "status.json",
        git_sha="a" * 40,
    )
    stuck = Mock(pid=123, name="stuck-gpu")
    stuck.is_alive.return_value = True
    context = SimpleNamespace(Event=threading.Event, Pipe=Pipe, Process=Mock(return_value=stuck))
    monkeypatch.setattr(runner.mp, "get_context", lambda _method: context)
    monkeypatch.setattr(runner.secrets, "token_hex", lambda _length: "a" * 16)
    monkeypatch.setattr(
        runner,
        "_await_engine_ready",
        Mock(side_effect=runner._EngineLost("inference process stopped")),
    )

    with pytest.raises(RuntimeError, match="termination exceeded two seconds"):
        runner._run_generation(config, ("BOT#1",), "b" * 64, runner._ShutdownFlag(), recovery_deadline=None)

    failed = store.get_job(credentials.job.id, credentials.token)
    assert (failed.status, failed.error_code, failed.last_result) == (
        JobStatus.FAILED,
        "service_failure_bot_forfeit",
        "win",
    )
    assert stuck.kill.call_count == 1


def test_runner_cli_disables_compilation_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    account = tmp_path / "user.json"
    account.write_text('{"connectCode":"HAL#1"}')
    policy = tmp_path / "policy.halpolicy"
    policy.touch()
    captured: list[runner.RunnerConfig] = []
    monkeypatch.setattr(runner, "resolve_checkpoint", lambda _source: policy)
    monkeypatch.setattr(runner, "run", captured.append)

    runner.main(
        [
            str(policy),
            "--user-jsons",
            str(account),
            "--slippi-ports",
            "51441",
            "--git-sha",
            "test-sha",
        ]
    )
    assert captured[0].compiled is False
    assert captured[0].batch_wait_seconds == 0.0005
    assert captured[0].preparation_timeout_seconds == 1800.0


def test_runner_cli_accepts_explicit_cold_preparation_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    account = tmp_path / "user.json"
    account.write_text('{"connectCode":"HAL#1"}')
    policy = tmp_path / "policy.hal"
    policy.touch()
    captured: list[runner.RunnerConfig] = []
    monkeypatch.setattr(runner, "resolve_checkpoint", lambda _source: policy)
    monkeypatch.setattr(runner, "run", captured.append)

    runner.main(
        [
            str(policy),
            "--user-jsons",
            str(account),
            "--slippi-ports",
            "51441",
            "--git-sha",
            "test-sha",
            "--preparation-timeout-seconds",
            "600",
        ]
    )

    assert captured[0].preparation_timeout_seconds == 600.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_runner_rejects_invalid_batch_wait(tmp_path: Path, value: float) -> None:
    with pytest.raises(ValueError, match="batch coalescing"):
        runner.RunnerConfig(
            database=tmp_path / "queue.sqlite3",
            policy=tmp_path / "policy.hal",
            user_jsons=(tmp_path / "account.json",),
            slippi_ports=(51441,),
            iso_path=tmp_path / "game.ciso",
            dolphin_path=tmp_path / "Slippi.AppImage",
            replay_dir=tmp_path / "replays",
            status_path=tmp_path / "status.json",
            git_sha="a" * 40,
            batch_wait_seconds=value,
        )


def test_runner_rejects_boolean_port_and_fractional_frame_limit(tmp_path: Path) -> None:
    config = runner.RunnerConfig(
        database=tmp_path / "queue.sqlite3",
        policy=tmp_path / "policy.hal",
        user_jsons=(tmp_path / "account.json",),
        slippi_ports=(51441,),
        iso_path=tmp_path / "game.ciso",
        dolphin_path=tmp_path / "Slippi.AppImage",
        replay_dir=tmp_path / "replays",
        status_path=tmp_path / "status.json",
        git_sha="a" * 40,
    )
    with pytest.raises(ValueError, match="ports"):
        replace(config, slippi_ports=(True,))
    with pytest.raises(ValueError, match="max_frames"):
        replace(config, max_frames=6.5)
    with pytest.raises(ValueError, match="replay publication"):
        replace(config, publish_replays="false")
    with pytest.raises(ValueError, match="cooldown"):
        replace(_slot_config(tmp_path), recovery_cooldown_seconds=float("nan"))


def test_runner_status_marks_a_stale_slot_for_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    status_path = tmp_path / "runner.json"
    slot_path = runner._slot_status_path(status_path, 0)
    write_slot_status(
        slot_path,
        SlotStatus(
            slot=0,
            state=SlotState.IDLE,
            game_fps=60.0,
            frame_interval_p95_ms=16.7,
            dolphin_step_p95_ms=5.0,
            policy_round_trip_p95_ms=10.0,
            reason=None,
            recoveries=0,
            updated_at=100.0,
        ),
    )
    monkeypatch.setattr(runner.time, "time", lambda: 104.0)

    status = runner._write_status(
        status_path,
        policy_sha256="a" * 64,
        slot_paths=(slot_path,),
        started_at=50.0,
        model_inference_p95_ms=8.0,
        batch_wait_p95_ms=0.5,
    )

    assert status.state.value == "recovering"
    assert status.healthy_slots == 0
    assert status.model_inference_p95_ms == 8.0


def test_runner_reports_a_bounded_slot_startup_grace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    status_path = tmp_path / "runner.json"
    slot_path = runner._slot_status_path(status_path, 0)
    monkeypatch.setattr(runner.time, "time", lambda: 110.0)

    status = runner._write_status(
        status_path,
        policy_sha256="a" * 64,
        slot_paths=(slot_path,),
        started_at=100.0,
        model_inference_p95_ms=None,
        batch_wait_p95_ms=None,
    )

    assert status.state.value == "recovering"


def test_frame_observations_do_not_write_status_synchronously(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes = Mock()
    monkeypatch.setattr(runner, "write_slot_status", writes)
    health = runner._SlotHealthReporter(0, tmp_path / "slot.json")

    with health:
        health.connecting(2)
        health.playing()
        for frame_id in range(600):
            health.observe_policy(0.001)
            health.observe_frame(frame_id, 0.001)

    assert writes.call_count == 3


def test_health_publisher_failure_terminates_the_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_SLOT_STATUS_INTERVAL_SECONDS", 0.001)
    writes = Mock(side_effect=[None, OSError("disk failed")])
    monkeypatch.setattr(runner, "write_slot_status", writes)

    with runner._SlotHealthReporter(0, tmp_path / "slot.json") as health:
        deadline = time.monotonic() + 1.0
        while True:
            try:
                health.status()
            except RuntimeError as error:
                assert str(error) == "slot health publisher failed"
                break
            assert time.monotonic() < deadline
            time.sleep(0.001)


def test_recoverable_failure_closes_dolphin_before_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Session:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> Session:
            events.append("enter")
            return self

        def __exit__(self, *_args: object) -> None:
            events.append("close")

    monkeypatch.setattr(runner, "NetplaySession", Session)
    monkeypatch.setattr(
        runner,
        "run_netplay_match",
        Mock(side_effect=runner._RecoverableRuntimeError("low_frame_rate")),
    )
    store = Mock()
    stop = Mock()
    stop.is_set.return_value = False

    with pytest.raises(runner._RecoverableRuntimeError):
        runner._run_reservation(
            _slot_config(tmp_path),
            store,
            Mock(),
            RuntimeConfig(1, (2, 3)),
            _job(),
            stop,
            Mock(),
            FrameTiming(2, 2, 4, 2, 8),
        )

    assert events == ["enter", "close"]


def test_recoverable_failure_resets_slot_and_retries_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "_run_reservation",
        Mock(side_effect=runner._RecoverableRuntimeError("frame_stutter")),
    )
    store = Mock()
    health = Mock()
    stop = Mock()

    runner._handle_reservation(
        _slot_config(tmp_path),
        store,
        Mock(),
        RuntimeConfig(1, (2, 3)),
        _job(),
        stop,
        health,
        FrameTiming(2, 2, 4, 2, 8),
    )

    health.recovering.assert_called_once_with("frame_stutter")
    store.fail.assert_called_once_with("reservation", "slot-0", "runtime_degraded", retryable=True)
    stop.wait.assert_called_once_with(0.0)


def test_no_contest_ends_reservation_without_a_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Session:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> Session:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    def play(*_args: object, **kwargs: object) -> object:
        on_live = kwargs["on_live"]
        assert callable(on_live)
        on_live()
        return object()

    replay = tmp_path / "replays" / "slot-0" / "game.slp"
    monkeypatch.setattr(runner, "NetplaySession", Session)
    monkeypatch.setattr(runner, "run_netplay_match", play)
    monkeypatch.setattr(
        runner,
        "read_new_replay_end",
        lambda *_args: ReplayEnd(replay, EndMethod.NO_CONTEST),
    )
    store = Mock()
    stop = Mock()
    stop.is_set.return_value = False
    health = Mock()

    runner._run_reservation(
        _slot_config(tmp_path),
        store,
        Mock(),
        RuntimeConfig(1, (2, 3)),
        _job(),
        stop,
        health,
        FrameTiming(2, 2, 4, 2, 8),
    )

    store.mark_no_contest.assert_called_once_with("reservation", "slot-0")
    store.finish_game.assert_not_called()
    health.playing.assert_called_once_with()
    health.configure_schedule.assert_called_once_with(FrameTiming(2, 2, 4, 2, 8))


def test_first_game_is_random_and_rematch_uses_requested_stage() -> None:
    first = runner._setup(_job(), rematch=False)
    rematch = runner._setup(_job(stage="YOSHIS_STORY"), rematch=True)
    assert first.character is melee.Character.FOX
    assert first.stage is melee.Stage.RANDOM_STAGE
    assert rematch.stage is melee.Stage.YOSHIS_STORY
    with pytest.raises(ValueError, match="no requested stage"):
        runner._setup(_job(), rematch=True)


def test_local_qualification_retains_replay_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Session:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> Session:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

    def play(*_args: object, **kwargs: object) -> object:
        kwargs["on_live"]()
        return SimpleNamespace(
            trajectory=[0],
            stage=31,
            wall_seconds=1.0,
            game_fps=60.0,
            frame_interval_p95_ms=16.0,
            dolphin_step_p95_ms=1.0,
            inference_p95_ms=5.0,
            transport_correction_frames=0,
        )

    replay = tmp_path / "replays" / "slot-0" / "game.slp"
    pending = Mock(return_value=replay.with_suffix(".slp.upload.json"))
    uploaded = Mock()
    monkeypatch.setattr(runner, "NetplaySession", Session)
    monkeypatch.setattr(runner, "run_netplay_match", play)
    monkeypatch.setattr(runner, "read_new_replay_end", lambda *_args: ReplayEnd(replay, EndMethod.GAME))
    monkeypatch.setattr(runner, "_human_result", lambda _result: "tie")
    monkeypatch.setattr(runner, "_stage_name", lambda _stage: "FINAL_DESTINATION")
    monkeypatch.setattr(runner, "_write_pending_upload", pending)
    monkeypatch.setattr(runner, "_complete_pending_upload", uploaded)
    store = Mock()
    store.finish_game.return_value = JobStatus.COMPLETE
    stop = Mock()
    stop.is_set.return_value = False

    runner._run_reservation(
        replace(_slot_config(tmp_path), publish_replays=False),
        store,
        Mock(),
        RuntimeConfig(1, (2, 3)),
        _job(),
        stop,
        Mock(),
        FrameTiming(2, 1, 3, 4, 8),
    )

    pending.assert_called_once()
    uploaded.assert_not_called()
    store.finish_game.assert_called_once()


def test_pending_replay_upload_records_before_local_delete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay = tmp_path / "game.slp"
    replay.write_bytes(b"replay")
    now = datetime.now(UTC)
    metadata = ReplayMetadata(
        reservation_id="reservation",
        player_code="CRYO#610",
        game_number=1,
        actual_stage="BATTLEFIELD",
        result="win",
        policy_sha256="a" * 64,
        git_sha="b" * 40,
        started_at=now,
        ended_at=now,
    )
    sidecar = runner._write_pending_upload(replay, metadata)
    events: list[str] = []
    uploaded = UploadedReplay("key", "metadata", "c" * 64, 6, "etag")

    def upload(path: Path, actual: ReplayMetadata) -> UploadedReplay:
        assert path == replay
        assert actual == metadata
        events.append("upload")
        return uploaded

    class Store:
        def record_replay(self, *_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
            assert replay.exists()
            events.append("record")

    monkeypatch.setattr(runner, "upload_replay", upload)
    runner._complete_pending_upload(sidecar, Store())  # type: ignore[arg-type]
    assert events == ["upload", "record"]
    assert not replay.exists()
    assert not sidecar.exists()


def test_pending_replay_retry_is_limited_to_once_per_minute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drain = Mock()
    times = iter((100.0, 110.0, 160.0))
    monkeypatch.setattr(runner, "_drain_pending_uploads", drain)
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(times))
    store = Mock()

    next_attempt = runner._retry_pending_uploads(tmp_path, store, 0.0)
    next_attempt = runner._retry_pending_uploads(tmp_path, store, next_attempt)
    next_attempt = runner._retry_pending_uploads(tmp_path, store, next_attempt)

    assert next_attempt == 220.0
    assert drain.call_args_list == [((tmp_path, store),), ((tmp_path, store),)]


@pytest.mark.parametrize(
    ("ego_port", "p1", "p2", "expected"),
    [(1, 4, 0, "loss"), (2, 4, 0, "win"), (1, 2, 2, "tie")],
)
def test_result_is_reported_from_the_human_side(
    ego_port: int,
    p1: int,
    p2: int,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "summarize_trajectory",
        lambda _trajectory: SimpleNamespace(p1_stocks_left=p1, p2_stocks_left=p2),
    )
    result = SimpleNamespace(trajectory=object(), ego_port=ego_port)
    assert runner._human_result(result) == expected  # type: ignore[arg-type]


def test_dolphin_connection_failure_does_not_report_inference_loss(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "_run_reservation", Mock(side_effect=runner.DolphinConnectionLost("closed")))
    store, health, stop = Mock(), Mock(), Mock()
    runner._handle_reservation(
        _slot_config(tmp_path),
        store,
        Mock(),
        RuntimeConfig(1, (2,)),
        _job(),
        stop,
        health,
        FrameTiming(2, 2, 4, 2, 8),
    )
    health.recovering.assert_called_once_with("dolphin_connection_lost")
    store.forfeit_service_failure.assert_called_once_with("reservation", "slot-0")
    store.fail.assert_not_called()


def test_inference_failure_stops_slot_after_forfeiting_current_reservation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "_run_reservation", Mock(side_effect=runner.InferenceUnavailable("timed out")))
    store, health, stop = Mock(), Mock(), Mock()
    with pytest.raises(runner.InferenceUnavailable, match="timed out"):
        runner._handle_reservation(
            _slot_config(tmp_path),
            store,
            Mock(),
            RuntimeConfig(1, (2,)),
            _job(),
            stop,
            health,
            FrameTiming(2, 2, 4, 2, 8),
        )
    health.recovering.assert_called_once_with("inference_engine_lost")
    store.forfeit_service_failure.assert_called_once_with("reservation", "slot-0")
    store.claim_next.assert_not_called()


def test_no_usable_plan_forfeits_match_without_stopping_healthy_slot(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "_run_reservation", Mock(side_effect=runner.NoUsableActionPlan("late plans")))
    store, health, stop = Mock(), Mock(), Mock()
    runner._handle_reservation(
        _slot_config(tmp_path),
        store,
        Mock(),
        RuntimeConfig(1, (2,)),
        _job(),
        stop,
        health,
        FrameTiming(2, 1, 3, 4, 8),
    )
    health.recovering.assert_called_once_with("no_usable_action_plan")
    store.forfeit_service_failure.assert_called_once_with("reservation", "slot-0")
    store.fail.assert_not_called()


def test_match_measurement_preserves_source_frame_and_schedule_counters(tmp_path: Path) -> None:
    config = replace(_slot_config(tmp_path), measurement_dir=tmp_path / "measurements")
    timing = FrameTiming(2, 1, 3, 4, 8)
    trajectory = Trajectory(np.array([0, 1]), {}, np.array([0, 0]))
    event = ScheduleEvent(0, 3, "gameplay", 0, 0, 0, 0, 1, 0, False)
    neutral_wire = (0, 0, 0, 0, 0, 0, 0)
    decision = PlanDecision(
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        4,
        (4, 5, 6, 7, 8),
        (neutral_wire,) * 3,
        (neutral_wire,) * 3,
        True,
    )
    result = PlayResult(
        trajectory,
        1,
        2,
        31,
        0.1,
        (0.005,),
        (0.016, 0.016),
        (0.001, 0.001),
        0,
        (0,),
        (event,),
        1.5,
        0.4,
        (decision,),
        generation=1,
    )
    health = Mock()
    health.status.return_value.chunk_health = ChunkHealth(timing, submission_gaps=1)
    started = datetime.now(UTC)

    path = runner._write_match_measurement(
        config, _job(), result, timing, health, started, started + timedelta(seconds=2)
    )

    assert path is not None
    payload = json.loads(path.read_text())
    assert payload["worker_id"] == "slot-0"
    assert payload["schema_version"] == 2
    assert payload["generation"] == 1
    assert payload["checkpoint_sha256"] == "c" * 64
    assert payload["inference_source_frames"] == [0]
    assert payload["inference_seconds"] == [0.005]
    assert payload["schedule"]["submission_gaps"] == 1
    assert payload["controller_submission_gaps"] == 1
    assert payload["schedule_events"][0]["target_frame"] == 3
    assert payload["plan_decisions"][0]["generated_target_frames"] == [4, 5, 6, 7, 8]
    assert payload["connection_countdown_seconds"] == pytest.approx(1.5)
    assert payload["match_end_seconds"] == pytest.approx(0.4)
    assert payload["total_elapsed_seconds"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="plan decisions"):
        runner._write_match_measurement(
            config, _job(), replace(result, plan_decisions=()), timing, health, started, started
        )


def test_failed_match_measurement_keeps_partial_frame_and_stream_identity(tmp_path: Path) -> None:
    config = replace(_slot_config(tmp_path), measurement_dir=tmp_path / "measurements")
    timing = FrameTiming(2, 1, 3, 4, 8)
    progress = NetplayProgress(0, 1, 3, 17, 13, (-2, -1, 0, 1), (0,), (0.005,), (0.016,), (0.003,), ())
    health = Mock()
    health.status.return_value.chunk_health = ChunkHealth(timing, exhausted_chunks=1, submission_gaps=0)

    runner._write_match_failure(config, _job(), timing, health, datetime.now(UTC), progress, RuntimeError("lost"))

    path = config.measurement_dir / "reservation-game-1-failure.json"
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
    assert payload["worker_id"] == "slot-0"
    assert payload["failure"] == "RuntimeError: lost"
    assert payload["progress"]["observed_frame_ids"] == [-2, -1, 0, 1]
    assert payload["progress"]["pending_sequence"] == 3
    assert payload["schedule"]["exhausted_chunks"] == 1
    assert payload["controller_submission_gaps"] == 0


def test_runner_cli_accepts_bounded_coalescing_wait(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    account = tmp_path / "user.json"
    account.write_text('{"connectCode":"HAL#1"}')
    policy = tmp_path / "policy.halpolicy"
    policy.touch()
    captured: list[runner.RunnerConfig] = []
    monkeypatch.setattr(runner, "resolve_checkpoint", lambda _source: policy)
    monkeypatch.setattr(runner, "run", captured.append)
    runner.main(
        [
            str(policy),
            "--user-jsons",
            str(account),
            "--slippi-ports",
            "51441",
            "--git-sha",
            "test-sha",
            "--batch-wait-ms",
            "0.25",
        ]
    )
    assert captured[0].batch_wait_seconds == 0.00025
    assert (FrameTiming(2, 1, 3, 4, 8), FrameTiming(3, 1, 4, 4, 8)) == runner._NETPLAY_TIMINGS
