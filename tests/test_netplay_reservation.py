import threading

from hal.netplay_service.domain import EndReason
from hal.netplay_service.domain import FinishedGame
from hal.netplay_service.domain import GameResult
from hal.netplay_service.domain import Job
from hal.netplay_service.domain import JobStatus
from hal.netplay_service.domain import Observed
from hal.netplay_service.domain import Phase
from hal.netplay_service.domain import Settings
from hal.netplay_service.domain import WindDown
from hal.netplay_service.queue_contract import InvalidTransitionError
from hal.netplay_service.reservation import ReservationLink


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
