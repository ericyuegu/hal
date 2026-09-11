import json
import signal
import time
from datetime import UTC
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import melee
import pytest
from peppi_py.game import EndMethod

import hal.netplay_service.runner as runner
from hal.eval.play import ReplayEnd
from hal.inference.api import RuntimeConfig
from hal.netplay_service.domain import Job
from hal.netplay_service.domain import JobStatus
from hal.netplay_service.domain import MatchChoices
from hal.netplay_service.health import SlotState
from hal.netplay_service.health import SlotStatus
from hal.netplay_service.health import write_slot_status
from hal.netplay_service.replays import ReplayMetadata
from hal.netplay_service.replays import UploadedReplay


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
        git_sha="b" * 40,
        recovery_cooldown_seconds=0.0,
    )


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
    )

    store.mark_no_contest.assert_called_once_with("reservation", "slot-0")
    store.finish_game.assert_not_called()
    health.playing.assert_called_once_with()


def test_first_game_is_random_and_rematch_uses_requested_stage() -> None:
    first = runner._setup(_job(), rematch=False)
    rematch = runner._setup(_job(stage="YOSHIS_STORY"), rematch=True)
    assert first.character is melee.Character.FOX
    assert first.stage is melee.Stage.RANDOM_STAGE
    assert rematch.stage is melee.Stage.YOSHIS_STORY
    with pytest.raises(ValueError, match="no requested stage"):
        runner._setup(_job(), rematch=True)


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
