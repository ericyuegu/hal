import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import melee
import numpy as np
import pytest
from peppi_py.game import EndMethod

from hal.eval.netplay import DolphinConnectionLost
from hal.eval.replays import ReplayEnd
from hal.eval.results import PlayResult
from hal.eval.scheduling import FrameTiming
from hal.inference.api import RuntimeConfig
from hal.netplay_service import reservation
from hal.netplay_service.domain import EndReason
from hal.netplay_service.domain import FinishedGame
from hal.netplay_service.domain import GameResult
from hal.netplay_service.domain import Job
from hal.netplay_service.domain import JobStatus
from hal.netplay_service.domain import Observed
from hal.netplay_service.domain import Phase
from hal.netplay_service.domain import Settings
from hal.netplay_service.domain import WindDown
from hal.netplay_service.queue_client import QueueEndpoint
from hal.netplay_service.queue_contract import InvalidTransitionError
from hal.netplay_service.reservation import ReservationLink
from hal.netplay_service.runner import SlotConfig
from hal.sim.netplay import ConnectAbandoned
from hal.sim.netplay import PlayerDisconnected
from hal.sim.trajectory import Trajectory

_ENDPOINT = QueueEndpoint("http://127.0.0.1:8787", "runner-token")
_RUNTIME = RuntimeConfig(1, (2, 3))
_TIMING = FrameTiming(2, 2, 4, 2, 8)


def _job(**changes: object) -> Job:
    values: dict[str, object] = dict(
        id="job",
        player_code="CRYO#610",
        online_delay=2,
        status=JobStatus.ASSIGNED,
        end_reason=None,
        queue_position=None,
        attempt=1,
        settings=Settings(1, "FOX", "IBDW#0", None, 20.0, 1.0),
        phase=None,
        phase_deadline=None,
        games=(),
        wind_down=None,
        lock_requests=0,
    )
    values.update(changes)
    return Job(**values)  # type: ignore[arg-type]


class _Queue:
    def __init__(self, responses: list[Job | Exception]) -> None:
        self.responses = responses
        self.reports: list[Observed] = []
        self.ends: list[tuple[EndReason, bool]] = []
        self.lock = threading.Lock()
        self.second_report = threading.Event()

    def report(self, _job_id: str, _worker_id: str, observed: Observed) -> Job:
        with self.lock:
            self.reports.append(observed)
            if len(self.reports) >= 2:
                self.second_report.set()
            response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        return response

    def end(self, _job_id: str, _worker_id: str, reason: EndReason, *, retryable: bool) -> Job:
        self.ends.append((reason, retryable))
        return _job(status=JobStatus.ENDED, end_reason=reason)


def test_reports_carry_increasing_seq_and_full_games() -> None:
    queue = _Queue([_job()])
    with ReservationLink(queue, _job(), "s/slot-0", interval_seconds=0.01) as link:  # type: ignore[arg-type]
        link.set_phase(Phase.IN_GAME, None)
        link.add_game(FinishedGame(1, "BATTLEFIELD", GameResult.WIN))
        link.end(EndReason.PLAYER_DISCONNECTED)
    seqs = [report.seq for report in queue.reports]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert queue.reports[-1].finished_games == (FinishedGame(1, "BATTLEFIELD", GameResult.WIN),)
    assert queue.ends == [(EndReason.PLAYER_DISCONNECTED, False)]


def test_settings_and_wind_down_follow_the_response() -> None:
    newer = _job(
        settings=Settings(2, "FALCO", "MANG#0", "POKEMON_STADIUM", 30.0, 1.0),
        wind_down=WindDown.PLAYER,
    )
    queue = _Queue([newer])
    with ReservationLink(queue, _job(), "s/slot-0", interval_seconds=0.01) as link:  # type: ignore[arg-type]
        assert link.settings().character == "FALCO"
        assert link.settings_at(1).character == "FOX"
        assert link.wind_down() is WindDown.PLAYER
        assert link.should_abort()
        assert queue.second_report.wait(0.1)
        assert queue.reports[-1].seen_revision == 2


def test_a_refused_report_marks_the_link_released() -> None:
    queue = _Queue([InvalidTransitionError("worker does not own this job")])
    with ReservationLink(queue, _job(), "s/slot-0", interval_seconds=0.01) as link:  # type: ignore[arg-type]
        assert link.released()
        link.end(EndReason.NO_SHOW)
    assert queue.ends == []


def test_an_ended_job_marks_the_link_released() -> None:
    queue = _Queue([_job(status=JobStatus.ENDED, end_reason=EndReason.PLAYER_LEFT)])
    with ReservationLink(queue, _job(), "s/slot-0", interval_seconds=0.01) as link:  # type: ignore[arg-type]
        assert link.released()


class _Session:
    def __init__(self, *_args: object, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.ego_character: melee.Character | None = None

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def lock_selection(self) -> None:
        driver = self.kwargs["menu_driver"]
        selection = driver._selection()  # type: ignore[attr-defined]
        driver.locked = selection  # type: ignore[attr-defined]
        self.ego_character = selection.character


def _play(results: list[object]):
    def play(session: _Session, *_args: object, **kwargs: object) -> object:
        outcome = results.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        session.lock_selection()
        on_live = kwargs["on_live"]
        assert callable(on_live)
        on_live()
        return outcome

    return play


def _fake_result() -> PlayResult:
    trajectory = Trajectory(np.array([0]), {}, np.array([0]))
    return PlayResult(
        trajectory,
        1,
        2,
        melee.Stage.BATTLEFIELD.value,
        1.0,
        (),
        (1 / 60,),
        (0.001,),
        0,
        generation=1,
    )


def _slot_config(tmp_path: Path) -> SlotConfig:
    return SlotConfig(
        slot=0,
        worker_id="s/slot-0",
        stream_id=0,
        queue_endpoint=_ENDPOINT,
        session_id="s",
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
        publish_replays=False,
    )


def _never() -> Mock:
    stop = Mock()
    stop.is_set.return_value = False
    return stop


def test_disconnect_after_a_game_ends_as_player_disconnected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = _Queue([_job()])
    monkeypatch.setattr(reservation, "NetplaySession", _Session)
    monkeypatch.setattr(
        reservation,
        "run_netplay_match",
        _play([_fake_result(), PlayerDisconnected("left")]),
    )
    monkeypatch.setattr(
        reservation,
        "read_new_replay_end",
        lambda *_args: ReplayEnd(tmp_path / "g.slp", EndMethod.GAME),
    )
    monkeypatch.setattr(reservation, "_game_result", lambda *_args: GameResult.LOSS)

    reservation.run_reservation(
        _slot_config(tmp_path),
        queue,  # type: ignore[arg-type]
        Mock(),
        _RUNTIME,
        _job(),
        _never(),
        Mock(),
        _TIMING,
    )

    assert queue.ends == [(EndReason.PLAYER_DISCONNECTED, False)]
    assert queue.reports[-1].finished_games[0].result is GameResult.LOSS


def test_leave_at_character_select_ends_as_player_canceled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = _Queue([_job(wind_down=WindDown.PLAYER)])
    monkeypatch.setattr(reservation, "NetplaySession", _Session)
    monkeypatch.setattr(reservation, "run_netplay_match", _play([ConnectAbandoned("wind down")]))

    reservation.run_reservation(
        _slot_config(tmp_path),
        queue,  # type: ignore[arg-type]
        Mock(),
        _RUNTIME,
        _job(),
        _never(),
        Mock(),
        _TIMING,
    )

    assert queue.ends == [(EndReason.PLAYER_CANCELED, False)]


def test_no_contest_is_recorded_and_the_session_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = _Queue([_job()])
    monkeypatch.setattr(reservation, "NetplaySession", _Session)
    monkeypatch.setattr(
        reservation,
        "run_netplay_match",
        _play([_fake_result(), PlayerDisconnected("left")]),
    )
    monkeypatch.setattr(
        reservation,
        "read_new_replay_end",
        lambda *_args: ReplayEnd(tmp_path / "g.slp", EndMethod.NO_CONTEST),
    )

    reservation.run_reservation(
        _slot_config(tmp_path),
        queue,  # type: ignore[arg-type]
        Mock(),
        _RUNTIME,
        _job(),
        _never(),
        Mock(),
        _TIMING,
    )

    assert queue.reports[-1].finished_games == (FinishedGame(1, "BATTLEFIELD", GameResult.NO_CONTEST),)


def test_service_failure_is_retryable_and_records_no_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queue = _Queue([_job()])
    monkeypatch.setattr(reservation, "NetplaySession", _Session)
    monkeypatch.setattr(
        reservation,
        "run_netplay_match",
        _play([DolphinConnectionLost("gone")]),
    )

    reservation.run_reservation(
        replace(_slot_config(tmp_path), graphics_backend="OGL"),
        queue,  # type: ignore[arg-type]
        Mock(),
        _RUNTIME,
        _job(),
        _never(),
        Mock(),
        _TIMING,
    )

    assert queue.ends == [(EndReason.SERVICE_FAILURE, True)]
    assert queue.reports[-1].finished_games == ()
