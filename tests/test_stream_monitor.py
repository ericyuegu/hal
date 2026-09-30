import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from hal.netplay_service.stream_monitor import GameProgress
from hal.netplay_service.stream_monitor import ObsSample
from hal.netplay_service.stream_monitor import StreamMonitor
from hal.netplay_service.stream_monitor import poll
from hal.netplay_service.stream_monitor import read_obs
from hal.netplay_service.stream_monitor import read_processes
from hal.netplay_service.stream_monitor import read_progress
from hal.netplay_service.stream_monitor import run_monitor


def obs(at: float = 100) -> ObsSample:
    return ObsSample(at, 60, 0.3, 2, 0, 0, 750_000, 1000, True, False, 0)


def test_monitor_separates_game_stalls_from_obs_drops() -> None:
    monitor = StreamMonitor()
    for second in range(6):
        health = monitor.observe(
            obs(100 + second),
            GameProgress(0, 1, 60 * second, second),
            now=second,
            wall_time=100 + second,
        )
    assert health.game_fps_5s == 60
    assert health.alerts == ()
    stalled = GameProgress(0, 1, 300, 5)
    for second in range(6, 11):
        health = monitor.observe(obs(100 + second), stalled, now=second, wall_time=100 + second)
    assert health.game_fps_5s == 0
    assert set(health.alerts) == {"game_fps_low", "game_progress_stale"}
    assert health.encoder_skips == health.network_skips == health.render_skips == 0


def test_rolling_fps_resets_for_menus_new_generations_and_backward_frames() -> None:
    monitor = StreamMonitor()
    for second in range(7):
        monitor.observe(obs(), GameProgress(0, 1, second * 30, second), now=second, wall_time=100)
    health = monitor.observe(obs(), None, now=7, wall_time=100)
    assert health.game_fps_5s is None
    assert not health.alerts
    monitor.observe(obs(), GameProgress(0, 1, 500, 8), now=8, wall_time=100)
    for game in (GameProgress(0, 2, 1000, 10), GameProgress(0, 2, 0, 11)):
        health = monitor.observe(obs(), game, now=game.received_at, wall_time=100)
        assert health.game_fps_5s is None


def test_drop_counters_are_interval_deltas_and_reset_at_obs_restart() -> None:
    monitor = StreamMonitor()
    monitor.observe(obs(), None, now=0, wall_time=100)
    current = replace(
        obs(101), render_skipped=5, encoder_skipped=2, network_skipped=4, output_bytes=1_500_000, duration_ms=2000
    )
    health = monitor.observe(current, None, now=1, wall_time=101)
    assert (health.render_skips, health.encoder_skips, health.network_skips) == (3, 2, 4)
    assert health.bitrate_mbps == 6
    assert set(health.alerts) == {"render_drops", "encoder_drops", "network_drops"}
    repeated = monitor.observe(current, None, now=2, wall_time=102)
    assert repeated.render_skips is None
    assert not repeated.alerts
    reset = monitor.observe(obs(103), None, now=3, wall_time=103)
    assert reset.obs_counter_reset
    assert reset.render_skips is None
    assert reset.bitrate_mbps is None
    assert not reset.alerts


def test_stale_inactive_reconnecting_and_congested_streams_are_visible() -> None:
    monitor = StreamMonitor()
    sample = replace(obs(), fps=40, active=False, reconnecting=True, congestion=0.3)
    health = monitor.observe(sample, None, now=0, wall_time=104)
    assert set(health.alerts) == {
        "obs_stats_stale",
        "stream_inactive",
        "stream_reconnecting",
        "network_congestion",
        "obs_fps_low",
    }


def test_future_game_clock_is_not_reported_as_healthy_fps() -> None:
    health = StreamMonitor().observe(obs(), GameProgress(0, 1, 120, 12), now=10, wall_time=100)
    assert health.game_fps_5s is None
    assert health.prediction_age_seconds is None
    assert "game_clock_invalid" in health.alerts


def test_history_window_stays_bounded_over_long_runs() -> None:
    monitor = StreamMonitor()
    for second in range(10_000):
        health = monitor.observe(
            obs(100 + second), GameProgress(0, 1, second * 60, second), now=second, wall_time=100 + second
        )
    assert health.game_fps_5s == 60
    assert len(monitor._frames) == 6


def write_inputs(path: Path) -> None:
    (path / "obs-stats.json").write_text(
        json.dumps(
            {
                "at": 100,
                "obs": {
                    "activeFps": 60,
                    "averageFrameRenderTime": 0.3,
                    "renderSkippedFrames": 2,
                    "outputSkippedFrames": 0,
                },
                "stream": {
                    "outputSkippedFrames": 0,
                    "outputBytes": 750_000,
                    "outputDuration": 1000,
                    "outputActive": True,
                    "outputReconnecting": False,
                    "outputCongestion": 0,
                },
            }
        )
    )
    (path / "value.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "playing": True,
                "sample": {"stream_id": 0, "generation": 1, "source_frame": 120, "received_at": 10},
            }
        )
    )


def test_reads_existing_formats_and_excludes_countdown_and_menus(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    assert read_obs(tmp_path / "obs-stats.json") == obs()
    path = tmp_path / "value.json"
    data = json.loads(path.read_text())
    assert read_progress(path) == GameProgress(0, 1, 120, 10)
    data["sample"]["source_frame"] = -1
    path.write_text(json.dumps(data))
    assert read_progress(path) is None
    data["sample"]["source_frame"] = 120
    data["playing"] = False
    path.write_text(json.dumps(data))
    assert read_progress(path) is None


@pytest.mark.parametrize("value", [2, True, "1"])
def test_progress_schema_is_strict(tmp_path: Path, value) -> None:
    path = tmp_path / "value.json"
    path.write_text(json.dumps({"schema_version": value, "playing": False, "sample": None}))
    with pytest.raises(ValueError):
        read_progress(path)


@pytest.mark.parametrize(
    "field,value", [("activeFps", float("nan")), ("activeFps", True), ("renderSkippedFrames", -1)]
)
def test_invalid_stats_are_rejected(tmp_path: Path, field, value) -> None:
    write_inputs(tmp_path)
    path = tmp_path / "obs-stats.json"
    data = json.loads(path.read_text())
    data["obs"][field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        read_obs(path)


def test_missing_or_invalid_input_keeps_monitor_alive_with_explicit_errors(tmp_path: Path) -> None:
    health = poll(tmp_path, StreamMonitor())
    assert len(health.alerts) == 2
    assert all("unavailable:FileNotFoundError" in error for error in health.alerts)
    (tmp_path / "value.json").write_text("invalid")
    health = poll(tmp_path, StreamMonitor())
    assert "game_progress_unavailable:JSONDecodeError" in health.alerts
    assert health.game_fps_5s is None


def test_process_samples_include_identity_and_memory_but_no_arguments(tmp_path: Path) -> None:
    process = tmp_path / "432100"
    process.mkdir()
    # Fields 3..22, including CPU ticks and process start time.
    fields = ["0"] * 20
    fields[11], fields[12], fields[19] = "200", "50", "12345"
    (process / "stat").write_text("432100 (a name) " + " ".join(fields))
    (process / "status").write_text("Name:\tDolphin\nVmRSS:\t1024 kB\nRssAnon:\t512 kB\nThreads:\t3\n")
    (process / "cmdline").write_text("account-secret HAL#647")
    result = read_processes(tmp_path)
    assert len(result) == 1
    assert (result[0].pid, result[0].start_ticks, result[0].rss_kib, result[0].anonymous_kib) == (
        432100,
        12345,
        1024,
        512,
    )
    assert result[0].threads == 3
    assert "secret" not in str(result)
    assert "#" not in str(result)


def test_monitor_only_writes_its_output_directory_and_can_stop_independently(tmp_path: Path) -> None:
    run_dir, output = tmp_path / "run", tmp_path / "monitor"
    run_dir.mkdir()
    write_inputs(run_dir)
    before = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    stop = threading.Event()
    thread = threading.Thread(target=run_monitor, args=(run_dir, output, stop))
    thread.start()
    try:
        for _ in range(100):
            if (output / "history.jsonl").exists() and (output / "history.jsonl").stat().st_size > 0:
                break
            stop.wait(0.01)
        health = json.loads((output / "status.json").read_text())
        assert health["schema_version"] == 1
        assert json.loads((output / "history.jsonl").read_text()) == health
        assert not (output / "status.partial").exists()
    finally:
        stop.set()
        thread.join(3)
    assert not thread.is_alive()
    assert before == {p.name: p.read_bytes() for p in run_dir.iterdir()}
