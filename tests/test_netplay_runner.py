import json
import signal
from datetime import UTC
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import melee
import pytest

import hal.netplay_service.runner as runner
from hal.netplay_service.domain import Job
from hal.netplay_service.domain import JobStatus
from hal.netplay_service.domain import MatchChoices
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
    previous = Mock()
    set_signal = Mock(side_effect=(previous, None))
    monkeypatch.setattr(runner.signal, "signal", set_signal)

    runner._start_slot_process(process)

    assert set_signal.call_args_list == [
        ((signal.SIGINT, signal.SIG_IGN),),
        ((signal.SIGINT, previous),),
    ]
    process.start.assert_called_once_with()


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
