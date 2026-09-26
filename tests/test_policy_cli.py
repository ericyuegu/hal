from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import melee
import pytest

from hal.scripts import policy as cli


def test_play_cli_defaults_are_calibrated() -> None:
    args = cli._play_parser().parse_args(["policy.halpolicy", "HUMAN#1"])
    assert args.character is melee.Character.FOX
    assert args.imitate == "IBDW#0"
    assert args.online_delay == 2
    assert args.replay_dir == Path("replays/human-play")
    assert args.slippi_port == 51441
    assert args.compiled is False
    assert args.history_mode == "auto"
    assert args.prediction_horizon is None
    assert args.thinking_allowance is None
    assert args.replan_interval is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("platinum", "PLATINUM"), ("Diamond", "DIAMOND"), ("MASTER", "MASTER"), ("ZAIN#0", "ZAIN#0")],
)
def test_play_cli_accepts_rank_or_exact_player_identity(value: str, expected: str) -> None:
    args = cli._play_parser().parse_args(["policy.halpolicy", "HUMAN#1", "--imitate", value])
    assert args.imitate == expected


def test_user_json_environment_and_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    environment = tmp_path / "environment.json"
    override = tmp_path / "override.json"
    environment.write_text("{}")
    override.write_text("{}")
    monkeypatch.setenv("HAL_SLIPPI_USER_JSON", str(environment))
    assert cli._user_json(None) == environment
    assert cli._user_json(str(override)) == override


def test_play_prepares_before_dolphin_connects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    events = []
    user_json = tmp_path / "user.json"
    user_json.write_text("{}")
    bundle = tmp_path / "policy.halpolicy"
    bundle.write_bytes(b"policy")

    class Policy:
        spec = None
        sampling_seed = 0
        context_frames = 8
        supported_horizons = (8,)
        prediction_horizon = 8

        def prepare_prediction(self, runtime, _horizon, _prefix):
            assert runtime.transport_delays == (2,)
            assert runtime.replan_interval_frames == 4
            assert (_horizon, _prefix) == (8, 3)
            events.append("prepare")

        def reset_prediction(self):
            pass

        def predict(self, *_args):
            pass

    class Session:
        def __init__(self, *_args, **_kwargs) -> None:
            events.append("session")

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *_args) -> None:
            pass

    def run_match(*_args, **_kwargs):
        events.append("connect")
        return SimpleNamespace(
            trajectory=[1],
            game_fps=60.0,
            frame_interval_p95_ms=16.7,
            inference_p95_ms=1.0,
            transport_correction_frames=0,
        )

    monkeypatch.setattr(cli, "resolve_checkpoint", lambda _path: bundle)
    monkeypatch.setattr(cli, "load_policy", lambda *_args, **_kwargs: Policy())
    monkeypatch.setattr(cli, "NetplaySession", Session)

    def check_budget(policy, runtime, _wait, *, shape):
        assert shape == (8, 3)
        policy.prepare_prediction(runtime, 8, 3)
        return SimpleNamespace(timings=(cli.FrameTiming(2, 1, 4, 8),))

    monkeypatch.setattr(cli, "check_realtime_budget", check_budget)
    monkeypatch.setattr(cli, "start_inference_worker", lambda policy, *_args: nullcontext(policy))
    monkeypatch.setattr(cli, "run_netplay_match", run_match)
    monkeypatch.setattr(cli, "require_completed_replay", lambda *_args: tmp_path / "game.slp")
    monkeypatch.setattr(cli.torch.compiler, "set_stance", lambda _value: nullcontext())
    args = cli._play_parser().parse_args(
        [
            str(bundle),
            "HUMAN#1",
            "--user-json",
            str(user_json),
            "--replay-dir",
            str(tmp_path / "replays"),
            "--prediction-horizon",
            "8",
            "--thinking-allowance",
            "1",
            "--replan-interval",
            "4",
        ]
    )
    cli._play(args)
    assert events == ["prepare", "session", "enter", "connect"]


def test_play_cli_defaults_to_backend_history_and_exposes_controls() -> None:
    assert cli._play_parser().parse_args(["policy.halpolicy", "HUMAN#1"]).history_mode == "auto"
    args = cli._play_parser().parse_args(
        [
            "policy.halpolicy",
            "HUMAN#1",
            "--history-mode",
            "kv_cache",
            "--kv-update-frames",
            "4",
        ]
    )
    assert args.history_mode == "kv_cache" and args.kv_update_frames == 4


def test_cpu_evaluation_defaults_to_backend_history_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    class Loaded(Exception):
        pass

    monkeypatch.setattr(cli, "resolve_checkpoint", lambda _path: Path("policy.halpolicy"))

    def load(_path: object, **options: object) -> None:
        assert options["history_mode"] == "auto"
        raise Loaded

    monkeypatch.setattr(cli, "load_policy", load)
    args = cli._policy_parser().parse_args(["eval", "policy.halpolicy"])
    with pytest.raises(Loaded):
        cli._eval(args)


def test_cpu_evaluation_dispatches_explicit_prediction_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = []
    loaded = []

    class Policy:
        spec = None
        sampling_seed = 0
        context_frames = 16
        supported_horizons = (4, 8)
        prediction_horizon = 4

        def prepare_prediction(self, runtime, horizon, prefix):
            prepared.append((runtime, horizon, prefix))

        def reset_prediction(self):
            pass

        def predict(self, requests):
            return ()

    def load(_path, **options):
        loaded.append(options)
        return Policy()

    monkeypatch.setattr(cli, "resolve_checkpoint", lambda _path: Path("policy.halpolicy"))
    monkeypatch.setattr(cli, "load_policy", load)
    monkeypatch.setattr(cli, "PolicyBatchAdapter", lambda _policy, _runtime, timing, **_options: timing)
    monkeypatch.setattr(cli, "default_session_cfg", lambda _replay_dir: None)
    monkeypatch.setattr(cli, "run_matches_vec", lambda *_args, **_kwargs: [[object()]])
    args = cli._policy_parser().parse_args(
        [
            "eval",
            "policy.halpolicy",
            "--history-mode",
            "kv_cache",
            "--kv-update-frames",
            "1",
            "--prediction-horizon",
            "8",
            "--thinking-allowance",
            "1",
            "--replan-interval",
            "4",
        ]
    )
    cli._eval(args)
    assert loaded[0]["history_mode"] == "kv_cache"
    assert loaded[0]["kv_update_frames"] == 1
    runtime, horizon, prefix = prepared[0]
    assert (runtime.transport_delays, runtime.replan_interval_frames, horizon, prefix) == ((2,), 4, 8, 3)


@pytest.mark.parametrize("option", ("--prediction-horizon", "--thinking-allowance"))
def test_play_requires_horizon_and_thinking_override_together(option: str) -> None:
    with pytest.raises(ValueError, match="set together"):
        cli._play(cli._play_parser().parse_args(["policy.halpolicy", "HUMAN#1", option, "8"]))
