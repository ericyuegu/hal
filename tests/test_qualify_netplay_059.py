"""The qualification harness must not leave service descendants running."""

import importlib.util
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from dataclasses import replace
from multiprocessing.connection import Connection
from pathlib import Path
from unittest.mock import Mock

import pytest

from hal.eval.scheduling import FrameTiming
from hal.netplay_service.domain import JobStatus
from hal.netplay_service.domain import MatchChoices
from hal.netplay_service.health import ChunkHealth
from hal.netplay_service.queue import QueueStore

_SPEC = importlib.util.spec_from_file_location(
    "hal_qualify_netplay_059", Path(__file__).parents[1] / "scripts" / "qualify_netplay_059.py"
)
assert _SPEC is not None and _SPEC.loader is not None
qualify_netplay_059 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = qualify_netplay_059
_SPEC.loader.exec_module(qualify_netplay_059)


def _spawn_idle_descendant(connection: Connection) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    connection.send(child.pid)
    connection.close()
    while True:
        time.sleep(1)


def test_shutdown_finds_and_terminates_runner_descendants() -> None:
    parent, child = mp.get_context("spawn").Pipe(duplex=False)
    process = mp.get_context("spawn").Process(target=_spawn_idle_descendant, args=(child,))
    process.start()
    child.close()
    descendant_pid = parent.recv()
    parent.close()
    try:
        shutdown = qualify_netplay_059._terminate(process, {})
        assert not process.is_alive()
        assert descendant_pid in shutdown.forced_descendants
        assert not shutdown.remaining_descendants
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        identity = qualify_netplay_059._process_identity(descendant_pid)
        if identity is not None and identity[0] != "Z":
            os.kill(descendant_pid, signal.SIGKILL)


def test_qualifier_requires_a_display_before_creating_run_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    output = tmp_path / "qualification"
    config = qualify_netplay_059.QualificationConfig(
        output=output,
        bundle=tmp_path / "bundle.hal",
        bot_account=tmp_path / "bot.json",
        peer_account=tmp_path / "peer.json",
        delay=2,
        desired_return=19.976,
        minimum_games=10,
        minimum_gameplay_seconds=1800,
        smoke_frames=300,
        bot_slippi_port=51451,
        peer_slippi_port=51452,
    )
    with pytest.raises(RuntimeError, match="xvfb-run"):
        qualify_netplay_059.qualify(config)
    assert not output.exists()


def test_source_manifest_covers_all_maintained_runtime_modules() -> None:
    hashes = qualify_netplay_059._source_hashes()
    assert "hal/eval/qualification.py" in hashes
    assert "hal/inference/observation_history.py" in hashes
    assert "scripts/qualify_netplay_059.py" in hashes


@pytest.mark.parametrize("status", [JobStatus.PLAYING, JobStatus.FAILED, JobStatus.QUEUED])
def test_smoke_requires_a_healthy_bot_reservation(tmp_path: Path, status: JobStatus) -> None:
    store = QueueStore(tmp_path / "queue.sqlite3")
    credentials = store.create_job("TEST#1", MatchChoices("FOX", "IBDW#0", 2))
    current = replace(
        credentials.job, status=status, error_code="engine_unavailable" if status is JobStatus.FAILED else None
    )
    observed_store = Mock(spec=QueueStore)
    observed_store.get_job.return_value = current
    session = Mock()
    session.step.return_value = ({"id": 600}, True)
    runner = Mock()
    runner.is_alive.return_value = True
    if status is JobStatus.PLAYING:
        result = qualify_netplay_059._play_peer_game(
            session, {"id": 0}, credentials, 1, 600, store=observed_store, runner=runner
        )
        assert result.last_frame == 600
    else:
        with pytest.raises(RuntimeError, match="bot reservation stopped"):
            qualify_netplay_059._play_peer_game(
                session, {"id": 0}, credentials, 1, 600, store=observed_store, runner=runner
            )


def test_live_reservation_rejects_an_exited_runner(tmp_path: Path) -> None:
    store = QueueStore(tmp_path / "queue.sqlite3")
    credentials = store.create_job("TEST#1", MatchChoices("FOX", "IBDW#0", 2))
    with pytest.raises(RuntimeError, match="service exited"):
        qualify_netplay_059._validate_live_reservation(
            replace(credentials.job, status=JobStatus.PLAYING), is_runner_alive=False
        )


def test_resource_sampler_finishes_its_inflight_gpu_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sampling = threading.Event()

    def slow_gpu_sample(_pids: tuple[int, ...]) -> None:
        sampling.set()
        time.sleep(2.1)

    monkeypatch.setattr(qualify_netplay_059, "_gpu_memory", slow_gpu_sample)
    with qualify_netplay_059._ResourceSampler(os.getpid(), tmp_path / "status.json") as sampler:
        assert sampling.wait(timeout=2)
    assert not sampler.thread.is_alive()
    assert len(sampler.samples) == 1


@pytest.fixture
def match_measurement(tmp_path: Path) -> Path:
    timing = FrameTiming(2, 1, 3, 4, 8)
    health = ChunkHealth(timing, neutral_fallback_frames=10)
    event = {
        "choice_frame": -1,
        "target_frame": 2,
        "phase": "countdown",
        "deadline_misses": 0,
        "prefix_mismatches": 0,
        "exhausted_chunks": 0,
        "neutral_fallback_frames": 10,
        "submission_gaps": 0,
        "transport_corrections": 0,
        "inference_failed": False,
    }
    payload = {
        "schema_version": 1,
        "reservation_id": "reservation",
        "game_number": 1,
        "policy_bundle_sha256": "b" * 64,
        "source_git_sha": "c" * 40,
        "desired_return": 19.976,
        "observation_mode": "first_seen_speculative",
        "timing": asdict(timing),
        "gameplay_seconds": 10,
        "frame_ids": [*range(600), 0],
        "frame_interval_seconds": [1 / 60] * 600,
        "inference_source_frames": list(range(-120, 600, 4)),
        "inference_seconds": [0.1] * 105 + [0.01] * 75,
        "schedule": asdict(health),
        "schedule_events": [event],
        "controller_submission_gaps": 0,
    }
    path = tmp_path / "reservation-game-1.json"
    path.write_text(json.dumps(payload))
    return path


def _assess(path: Path) -> qualify_netplay_059.MatchAssessment:
    return qualify_netplay_059._assess_match(
        path,
        reservation_id="reservation",
        game_number=1,
        delay=2,
        desired_return=19.976,
        bundle_sha256="b" * 64,
        source_git_sha="c" * 40,
    )


def test_match_assessment_excludes_startup_latency_and_menu_frame(match_measurement: Path) -> None:
    payload = json.loads(match_measurement.read_text())
    payload["frame_interval_seconds"][-1] = 100
    match_measurement.write_text(json.dumps(payload))
    assessment = _assess(match_measurement)
    assert assessment.failures == ()
    assert assessment.steady_inference_count == 75
    assert assessment.steady_inference_p95_ms == 10
    assert assessment.steady_inference_p99_ms == 10
    assert assessment.steady_game_fps == pytest.approx(60)
    assert assessment.steady_frame_count == 299
    assert assessment.startup_count_by_event["neutral_fallback_frames"] == 10
    assert assessment.steady_count_by_event["neutral_fallback_frames"] == 0


@pytest.mark.parametrize(
    ("seconds", "fps", "message"),
    [(0.01201, 60, "p95"), (1 / 60, 60, "p99"), (0.01, 59, "59.5 FPS")],
)
def test_match_assessment_fails_slow_delivery_or_gameplay(
    match_measurement: Path, seconds: float, fps: float, message: str
) -> None:
    payload = json.loads(match_measurement.read_text())
    payload["inference_seconds"] = [seconds] * len(payload["inference_source_frames"])
    payload["frame_interval_seconds"] = [1 / fps] * 600
    match_measurement.write_text(json.dumps(payload))
    assert any(message in failure for failure in _assess(match_measurement).failures)


@pytest.mark.parametrize(
    ("counter", "should_fail"),
    [
        ("submission_gaps", True),
        ("neutral_fallback_frames", True),
        ("exhausted_chunks", True),
        ("deadline_misses", False),
        ("prefix_mismatches", False),
    ],
)
def test_match_assessment_checks_scheduling_without_mislabeling_rejected_plans(
    match_measurement: Path, counter: str, should_fail: bool
) -> None:
    payload = json.loads(match_measurement.read_text())
    event = dict(payload["schedule_events"][-1], choice_frame=400, target_frame=403, phase="gameplay")
    event[counter] += 1
    payload["schedule_events"].append(event)
    payload["schedule"][counter] += 1
    payload["controller_submission_gaps"] = payload["schedule"]["submission_gaps"]
    match_measurement.write_text(json.dumps(payload))
    assessment = _assess(match_measurement)
    assert bool(assessment.failures) is should_fail
    assert assessment.steady_count_by_event[counter] == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("inference_seconds", [], "missing or repeated"),
        ("inference_seconds", [float("nan")], "finite"),
        ("schedule_events", [], "missing scheduling"),
        ("schedule", None, "chunk health"),
        ("policy_bundle_sha256", "other-bundle", "identity"),
        ("source_git_sha", "other-source", "identity"),
        ("game_number", 2, "identity"),
        ("frame_ids", [0, 2, 0], "consecutive"),
    ],
)
def test_match_assessment_rejects_missing_or_mismatched_evidence(
    match_measurement: Path, field: str, value: object, message: str
) -> None:
    payload = json.loads(match_measurement.read_text())
    payload[field] = value
    match_measurement.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        _assess(match_measurement)
