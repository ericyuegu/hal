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
from types import SimpleNamespace
from unittest.mock import MagicMock
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


def test_qualification_graphics_backend_defaults_and_validation(tmp_path: Path) -> None:
    config = qualify_netplay_059.QualificationConfig(
        output=tmp_path / "qualification",
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
    assert config.graphics_backend == "Vulkan"
    assert replace(config, graphics_backend="OGL").graphics_backend == "OGL"
    with pytest.raises(ValueError, match="unsupported Dolphin graphics backend"):
        replace(config, graphics_backend="Null")


def test_qualification_routes_opengl_to_bot_peer_and_run_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bot = tmp_path / "bot.json"
    peer = tmp_path / "peer.json"
    bot.write_text('{"connectCode":"BOT#1"}')
    peer.write_text('{"connectCode":"PEER#1"}')
    output = tmp_path / "qualification"
    peer_backends: list[str] = []

    class PeerSession:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            peer_backends.append(str(kwargs["graphics_backend"]))

        def __enter__(self) -> PeerSession:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def start_match(self, _setup: object) -> None:
            raise RuntimeError("stop after peer construction")

    process = Mock(pid=123, exitcode=0)
    make_process = Mock(return_value=process)
    sampler = MagicMock()
    sampler.owned_processes = {}
    sampler.samples = []
    sampler.__enter__.return_value = sampler
    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.setattr(qualify_netplay_059, "_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(qualify_netplay_059, "_source_hashes", lambda: {})
    monkeypatch.setattr(
        qualify_netplay_059.subprocess,
        "run",
        Mock(return_value=subprocess.CompletedProcess([], 0, "source-sha\n", "")),
    )
    monkeypatch.setattr(qualify_netplay_059.mp, "get_context", lambda _method: SimpleNamespace(Process=make_process))
    monkeypatch.setattr(qualify_netplay_059, "_ResourceSampler", lambda *_args: sampler)
    monkeypatch.setattr(qualify_netplay_059, "_wait_ready", lambda *_args: 0.0)
    monkeypatch.setattr(qualify_netplay_059, "NetplaySession", PeerSession)
    monkeypatch.setattr(
        qualify_netplay_059, "_terminate", lambda *_args: qualify_netplay_059.ShutdownResult(0, False, (), ())
    )
    monkeypatch.setattr(qualify_netplay_059, "_assess_engine_audit", lambda *_args, **_kwargs: None)

    config = qualify_netplay_059.QualificationConfig(
        output=output,
        bundle=tmp_path / "bundle.hal",
        bot_account=bot,
        peer_account=peer,
        delay=2,
        desired_return=19.976,
        minimum_games=10,
        minimum_gameplay_seconds=1800,
        smoke_frames=300,
        bot_slippi_port=51451,
        peer_slippi_port=51452,
        graphics_backend="OGL",
    )
    with pytest.raises(RuntimeError, match="stop after peer construction"):
        qualify_netplay_059.qualify(config)

    runner_config = make_process.call_args.kwargs["args"][0]
    assert runner_config.graphics_backend == "OGL"
    assert peer_backends == ["OGL"]
    assert json.loads((output / "manifest.json").read_text())["graphics_backend"] == "OGL"
    assert json.loads((output / "run-result.json").read_text())["graphics_backend"] == "OGL"


def test_qualification_cli_accepts_explicit_graphics_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    qualified = Mock(return_value={})
    monkeypatch.setattr(qualify_netplay_059, "qualify", qualified)
    args = [
        str(tmp_path / "bundle.hal"),
        "--bot-account",
        str(tmp_path / "bot.json"),
        "--peer-account",
        str(tmp_path / "peer.json"),
        "--output",
        str(tmp_path / "qualification"),
    ]
    qualify_netplay_059.main(args)
    assert qualified.call_args.args[0].graphics_backend == "Vulkan"
    qualify_netplay_059.main([*args, "--graphics-backend", "OGL"])
    assert qualified.call_args.args[0].graphics_backend == "OGL"


def test_source_manifest_covers_all_maintained_runtime_modules() -> None:
    hashes = qualify_netplay_059._source_hashes()
    assert "hal/eval/qualification.py" in hashes
    assert "hal/inference/observation_history.py" in hashes
    assert "scripts/qualify_netplay_059.py" in hashes


@pytest.mark.parametrize(
    ("output", "owned", "expected"),
    [
        ("", (12,), 0),
        ("12, 111\n34, 222\n", (12,), 111),
        ("12, 0\n", (12,), 0),
        ("34, 222\n", (12,), None),
        ("12, N/A\n", (12,), None),
        ("12, 111\n34, N/A\n", (12,), None),
        ("12, 111\ninvalid\n", (12,), None),
        ("No running processes found\n", (12,), None),
    ],
)
def test_gpu_memory_requires_valid_matching_processes(
    output: str, owned: tuple[int, ...], expected: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed = subprocess.CompletedProcess(args=("nvidia-smi",), returncode=0, stdout=output, stderr="")
    monkeypatch.setattr(qualify_netplay_059.subprocess, "run", Mock(return_value=completed))
    assert qualify_netplay_059._gpu_memory(owned) == expected


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


def _engine_audit_fixture(tmp_path: Path) -> tuple[Path, Path]:
    measurement_dir = tmp_path / "match-measurements"
    audit_dir = measurement_dir / "engine-audits"
    audit_dir.mkdir(parents=True)
    budget_path = tmp_path / "runner-status.budget.json"
    budget_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "git_sha": "c" * 40,
                "bundle_sha256": "b" * 64,
                "checkpoint_sha256": "a" * 64,
                "qualified_capacity": 1,
                "capability_version": 2,
                "compiled": True,
            }
        )
    )
    profiles = [
        {
            "profile": {
                "name": f"netplay-delay-{delay}",
                "checkpoint_sha256": "a" * 64,
                "execution_mode": "kv_cache",
                "prediction_horizon_frames": 8,
                "fixed_prefix_frames": delay + 1,
                "update_shapes": [1, 2, 4],
                "capacity": 1,
            },
            "capture_attempts": 5,
            "capture_completed": 5,
        }
        for delay in (2, 3)
    ]
    common = {
        "schema_version": 1,
        "engine_generation_id": "d" * 16,
        "source_git_sha": "c" * 40,
        "policy_bundle_sha256": "b" * 64,
    }
    started = dict(common, phase="started")
    ready = dict(
        common,
        phase="ready",
        checkpoint_sha256="a" * 64,
        capability_version=2,
        compiled=True,
        outcome="serving",
        compilation_starts=0,
        profiles=profiles,
    )
    final = dict(ready, phase="final", outcome="stopped")
    for phase, record in (("started", started), ("ready", ready), ("final", final)):
        (audit_dir / f"{'d' * 16}-{phase}.json").write_text(json.dumps(record))
    return measurement_dir, budget_path


def _assess_engine_audit(tmp_path: Path) -> qualify_netplay_059.EngineAuditAssessment:
    return qualify_netplay_059._assess_engine_audit(
        tmp_path / "match-measurements",
        tmp_path / "runner-status.budget.json",
        source_git_sha="c" * 40,
        bundle_sha256="b" * 64,
        capacity=1,
    )


def test_engine_audit_requires_one_unchanged_prepared_generation(tmp_path: Path) -> None:
    measurement_dir, _ = _engine_audit_fixture(tmp_path)
    result = _assess_engine_audit(tmp_path)
    assert result.generation_id == "d" * 16
    assert result.profile_names == ("netplay-delay-2", "netplay-delay-3")
    assert result.compilation_starts == result.capture_attempts_delta == result.capture_completed_delta == 0
    assert Path(result.ready_path).parent == measurement_dir / "engine-audits"
    assert len(result.ready_sha256) == len(result.final_sha256) == 64


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("restart", "exactly one inference engine generation"),
        ("missing_final", "missing, extra, or incomplete"),
        ("malformed_final", "Expecting property name"),
        ("compile", "occurred after preparation"),
        ("capture_attempt", "occurred after preparation"),
        ("capture_completed", "occurred after preparation"),
        ("profile", "profile identity"),
        ("checkpoint", "checkpoint, capability"),
        ("source", "audit identity"),
        ("error_outcome", "serving outcome"),
        ("extra_temp", "missing, extra, or incomplete"),
        ("unqualified_budget", "preparation budget identity"),
    ],
)
def test_engine_audit_rejects_missing_or_tampered_evidence(tmp_path: Path, tamper: str, message: str) -> None:
    measurement_dir, budget_path = _engine_audit_fixture(tmp_path)
    audit_dir = measurement_dir / "engine-audits"
    final_path = audit_dir / f"{'d' * 16}-final.json"
    if tamper == "restart":
        (audit_dir / f"{'e' * 16}-started.json").write_text("{}")
    elif tamper == "missing_final":
        final_path.unlink()
    elif tamper == "malformed_final":
        final_path.write_text("{bad json")
    elif tamper == "extra_temp":
        (audit_dir / "leftover.tmp").touch()
    elif tamper == "unqualified_budget":
        budget = json.loads(budget_path.read_text())
        budget["compiled"] = False
        budget_path.write_text(json.dumps(budget))
    else:
        record = json.loads(final_path.read_text())
        if tamper == "compile":
            record["compilation_starts"] += 1
        elif tamper == "capture_attempt":
            record["profiles"][0]["capture_attempts"] += 1
        elif tamper == "capture_completed":
            record["profiles"][0]["capture_attempts"] += 1
            record["profiles"][0]["capture_completed"] += 1
        elif tamper == "profile":
            record["profiles"][0]["profile"]["fixed_prefix_frames"] += 1
        elif tamper == "checkpoint":
            record["checkpoint_sha256"] = "e" * 64
        elif tamper == "source":
            record["source_git_sha"] = "e" * 40
        else:
            record["outcome"] = "error"
        final_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match=message):
        _assess_engine_audit(tmp_path)


def _plan_decision(source_frame: int, sequence: int, timing: FrameTiming) -> dict[str, object]:
    prefix = [[0] * 7 for _ in range(timing.fixed_prefix_frames)]
    return {
        "request_stream_id": 0,
        "request_generation": 1,
        "request_sequence": sequence,
        "request_source_frame": source_frame,
        "response_stream_id": 0,
        "response_generation": 1,
        "response_sequence": sequence,
        "response_source_frame": source_frame,
        "choice_frame": source_frame + 1,
        "first_submittable_target": source_frame + timing.physical_delay_frames + 2,
        "generated_target_frames": list(
            range(source_frame + timing.fixed_prefix_frames + 1, source_frame + timing.prediction_horizon_frames + 1)
        ),
        "request_prefix_wire": prefix,
        "pinned_prefix_wire": [[0] * 7 for _ in prefix],
        "accepted": True,
    }


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
    inference_frames = list(range(-120, 600, 4))
    payload = {
        "schema_version": 3,
        "reservation_id": "reservation",
        "game_number": 1,
        "stream_id": 0,
        "generation": 1,
        "policy_bundle_sha256": "b" * 64,
        "source_git_sha": "c" * 40,
        "graphics_backend": "Vulkan",
        "desired_return": 19.976,
        "observation_mode": "first_seen_speculative",
        "timing": asdict(timing),
        "gameplay_seconds": 10,
        "frame_ids": [*range(600), 0],
        "frame_interval_seconds": [1 / 60] * 600,
        "inference_source_frames": inference_frames,
        "inference_seconds": [0.1] * 105 + [0.01] * 75,
        "plan_decisions": [_plan_decision(frame, index, timing) for index, frame in enumerate(inference_frames)],
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
        graphics_backend="Vulkan",
    )


def test_match_assessment_excludes_startup_latency_and_menu_frame(match_measurement: Path) -> None:
    payload = json.loads(match_measurement.read_text())
    payload["frame_interval_seconds"][-1] = 100
    match_measurement.write_text(json.dumps(payload))
    assessment = _assess(match_measurement)
    assert assessment.failures == ()
    assert assessment.steady_inference_count == 75
    assert assessment.plan_decision_count == 180
    assert assessment.accepted_plan_count == 180
    assert assessment.steady_inference_p95_ms == 10
    assert assessment.steady_inference_p99_ms == 10
    assert assessment.steady_game_fps == pytest.approx(60)
    assert assessment.steady_frame_count == 299
    assert assessment.startup_count_by_event["neutral_fallback_frames"] == 10
    assert assessment.steady_count_by_event["neutral_fallback_frames"] == 0


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("missing", "plan_decisions"),
        ("short", "every inference response"),
        ("identity", "identity"),
        ("generation", "identity"),
        ("bound", "submitability bound"),
        ("target", "target frames"),
        ("late_accepted", "accepted plan"),
        ("prefix_accepted", "accepted plan"),
        ("rejected_valid", "accepted plan"),
        ("wire_shape", "seven controller wire values"),
        ("wire_range", "invalid controller wire values"),
        ("old_schema", "identity"),
    ],
)
def test_match_assessment_rejects_missing_or_tampered_plan_evidence(
    match_measurement: Path, tamper: str, message: str
) -> None:
    payload = json.loads(match_measurement.read_text())
    if tamper == "missing":
        del payload["plan_decisions"]
    elif tamper == "short":
        payload["plan_decisions"].pop()
    elif tamper == "identity":
        payload["plan_decisions"][0]["response_sequence"] += 1
    elif tamper == "generation":
        payload["plan_decisions"][0]["request_generation"] += 1
        payload["plan_decisions"][0]["response_generation"] += 1
    elif tamper == "bound":
        payload["plan_decisions"][0]["first_submittable_target"] += 1
    elif tamper == "target":
        payload["plan_decisions"][0]["generated_target_frames"][0] += 1
    elif tamper == "late_accepted":
        payload["plan_decisions"][0]["choice_frame"] += 1
        payload["plan_decisions"][0]["first_submittable_target"] += 1
    elif tamper == "prefix_accepted":
        payload["plan_decisions"][0]["pinned_prefix_wire"][0][0] += 1
    elif tamper == "rejected_valid":
        payload["plan_decisions"][0]["accepted"] = False
    elif tamper == "wire_shape":
        payload["plan_decisions"][0]["request_prefix_wire"][0].pop()
    elif tamper == "wire_range":
        payload["plan_decisions"][0]["request_prefix_wire"][0][0] = 81
        payload["plan_decisions"][0]["pinned_prefix_wire"][0][0] = 81
    else:
        payload["schema_version"] = 1
    match_measurement.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        _assess(match_measurement)


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
        ("graphics_backend", "OGL", "identity"),
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
