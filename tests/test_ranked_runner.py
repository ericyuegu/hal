from contextlib import nullcontext
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from peppi_py.game import EndMethod

from hal.eval import ranked
from hal.eval.replays import ReplayEnd
from hal.sim.netplay import CountdownEnded


def config(tmp_path: Path, *, max_games: int = 1) -> ranked.RankedConfig:
    bundle = tmp_path / "bundle"
    bundle.write_bytes(b"policy")
    account = tmp_path / "account.json"
    account.write_text('{"connectCode":"LOCAL#123"}')
    key = tmp_path / "twitch-key"
    key.write_text("private-stream-key")
    return ranked.RankedConfig(bundle, account, tmp_path / "runs", key, "f5d44f74", max_games=max_games)


def test_fixed_timing_and_overlay_do_not_expose_a_connect_code() -> None:
    timing = ranked.timing()
    assert (timing.physical_delay_frames, timing.inference_allowance_frames) == (2, 1)
    assert (timing.fixed_prefix_frames, timing.replan_interval_frames, timing.prediction_horizon_frames) == (3, 4, 8)
    assert ranked.overlay(120) == "HAL · Cody Fox · advantage 120 · Ranked"
    assert "#" not in ranked.overlay(120)


def test_obs_failure_restarts_only_the_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = config(tmp_path)
    output = tmp_path / "stream-run"
    output.mkdir()
    stream = ranked.RankedStream(cfg, output, {})
    stream.playing.set()
    first, second = Mock(), Mock()
    first.start.side_effect = RuntimeError("OBS failed")
    second.start.side_effect = lambda *_args, **_kwargs: stream._stop.set()
    factory = Mock(side_effect=[first, second])
    monkeypatch.setattr(ranked, "ObsStudio", factory)

    stream._run()

    assert stream.playing.is_set()
    assert factory.call_count == 2
    first.close.assert_called_once()
    second.close.assert_called_once()
    assert "OBS failed" in (output / "stream-error.json").read_text()


@pytest.mark.parametrize("ending", [EndMethod.GAME, EndMethod.NO_CONTEST])
@pytest.mark.parametrize("interruption", [None, CountdownEnded, ranked.DolphinConnectionLost])
def test_player_recovers_without_requalifying_and_records_quits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ending: EndMethod, interruption: type[Exception] | None
) -> None:
    cfg = config(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    fixture = tmp_path / "fixture"
    fixture.write_bytes(b"fixture")
    monkeypatch.setattr(ranked, "ensure", lambda _fixture: fixture)
    monkeypatch.setattr(ranked, "configure_inference_process", lambda: None)
    monkeypatch.setattr(ranked, "freeze_inference_runtime", lambda: None)
    monkeypatch.setattr(ranked.torch.cuda, "get_device_name", lambda: "RTX PRO 6000 Blackwell")
    monkeypatch.setattr(ranked.subprocess, "check_output", lambda *_args, **_kwargs: "580.178.04")
    monkeypatch.setattr(ranked.torch.compiler, "set_stance", lambda _stance: nullcontext())
    policy = Mock(checkpoint_sha256="a" * 64, sampling_seed=120647, prepared_update_shapes=(1, 2, 4))
    load = Mock(return_value=policy)
    monkeypatch.setattr(ranked, "load_action_sequence_policy", load)
    monkeypatch.setattr(ranked, "start_inference_worker", lambda *_args: nullcontext(Mock()))
    monkeypatch.setattr(ranked, "DisplayGroup", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(ranked, "PulseAudio", lambda *_args: nullcontext(SimpleNamespace(environment={})))
    stream = SimpleNamespace(playing=Event(), screen=None)
    monkeypatch.setattr(ranked, "RankedStream", lambda *_args: nullcontext(stream))
    sessions = []

    def session_factory(*_args, **_kwargs):
        session = Mock()
        sessions.append(session)
        return nullcontext(session)

    monkeypatch.setattr(ranked, "NetplaySession", session_factory)
    replay = output / "game.slp"
    replay.write_bytes(b"recorded game")
    monkeypatch.setattr(ranked, "read_new_replay_end", lambda *_args: ReplayEnd(replay, ending))
    attempts = []

    def play(_session, _setup, _client, _runtime, _timing, **kwargs):
        attempts.append(kwargs["rematch"])
        if len(attempts) == 1 and interruption is not None:
            raise interruption("peer left")
        kwargs["on_live"]()
        return SimpleNamespace(
            ego_port=1,
            opponent_port=2,
            stage=25,
            trajectory=[0, 1],
            game_fps=60,
            frame_interval_p95_ms=17,
            inference_p95_ms=6,
            transport_correction_frames=0,
            wall_seconds=1,
        )

    monkeypatch.setattr(ranked, "run_netplay_match", play)
    stop = Mock()
    stop.is_set.return_value = False
    ranked._run(cfg, output, stop)

    assert attempts == ([False] if interruption is None else [False, interruption is CountdownEnded])
    assert len(sessions) == (2 if interruption is ranked.DolphinConnectionLost else 1)
    policy.prepare_prediction.assert_called_once()
    policy.qualify_prediction.assert_not_called()
    assert load.call_count == 1
    result = ranked.json.loads((output / "game-0001.json").read_text())
    assert result["end_method"] == ending.name
    assert result["replay_sha256"] == ranked.hashlib.sha256(b"recorded game").hexdigest()
    manifest = ranked.json.loads((output / "manifest.json").read_text())
    assert manifest["identity"] == "IBDW#0"
    assert manifest["advantage"] == 120


def test_run_records_failure_and_restores_signal_handlers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = config(tmp_path)
    monkeypatch.setattr(ranked.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(ranked.torch.cuda, "get_device_name", lambda: "RTX PRO 6000 Blackwell")
    monkeypatch.setattr(ranked.subprocess, "run", Mock())
    failure = Mock(side_effect=ValueError("bad artifact"))
    monkeypatch.setattr(ranked, "_run", failure)
    original = {number: ranked.signal.getsignal(number) for number in (ranked.signal.SIGINT, ranked.signal.SIGTERM)}
    with pytest.raises(ValueError, match="bad artifact"):
        ranked.run(cfg)
    assert {number: ranked.signal.getsignal(number) for number in original} == original
    status = next(cfg.output.glob("*/status.json"))
    assert ranked.json.loads(status.read_text())["state"] == "failed"
