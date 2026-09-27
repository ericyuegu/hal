from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import melee
import pytest

from hal.scripts import policy as cli


def test_play_cli_uses_prepared_netplay_profile() -> None:
    args = cli._play_parser().parse_args(["policy.hal", "HUMAN#1"])
    assert args.character is melee.Character.FOX
    assert args.imitate == "IBDW#0"
    assert args.online_delay == 2
    assert args.replay_dir == Path("replays/human-play")
    assert args.slippi_port == 51441
    assert args.compiled is True


@pytest.mark.parametrize(
    ("value", "expected"),
    [("platinum", "PLATINUM"), ("Diamond", "DIAMOND"), ("MASTER", "MASTER"), ("ZAIN#0", "ZAIN#0")],
)
def test_play_cli_accepts_rank_or_exact_player_identity(value: str, expected: str) -> None:
    args = cli._play_parser().parse_args(["policy.hal", "HUMAN#1", "--imitate", value])
    assert args.imitate == expected


def test_user_json_environment_and_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    environment = tmp_path / "environment.json"
    override = tmp_path / "override.json"
    environment.write_text("{}")
    override.write_text("{}")
    monkeypatch.setenv("HAL_SLIPPI_USER_JSON", str(environment))
    assert cli._user_json(None) == environment
    assert cli._user_json(str(override)) == override


@pytest.mark.parametrize("delay", (2, 3))
def test_play_prepares_before_dolphin_connects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delay: int) -> None:
    events = []
    user_json = tmp_path / "user.json"
    user_json.write_text("{}")
    bundle = tmp_path / "policy.hal"
    bundle.write_bytes(b"policy")
    policy = SimpleNamespace(return_p90=19.75, checkpoint_sha256="a" * 64)

    class Session:
        def __init__(self, *_args, **_kwargs) -> None:
            events.append("session")

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, *_args) -> None:
            return None

    def load(_path, **options):
        assert options["history_mode"] == "kv_cache"
        assert options["kv_update_frames"] == 4
        return policy

    def run_match(*_args, policy_settings, **_kwargs):
        assert policy_settings() == (19.75, 1.0)
        events.append("connect")
        return SimpleNamespace(trajectory=[1], game_fps=60.0, frame_interval_p95_ms=16.7, inference_p95_ms=1.0)

    def check_budget(candidate, runtime, wait, *, shape):
        assert candidate is policy
        assert shape == (8, delay + 1)
        assert runtime.transport_delays == (delay,)
        assert runtime.replan_interval_frames == 4
        assert wait == 0.0005
        events.append("prepare")
        return SimpleNamespace(timings=(cli.FrameTiming(delay, 1, delay + 1, 4, 8),))

    def start_engine(candidate, profile, wait):
        assert candidate is policy
        assert profile.checkpoint_sha256 == policy.checkpoint_sha256
        assert profile.execution_mode == "kv_cache"
        assert profile.prediction_horizon_frames == 8
        assert profile.fixed_prefix_frames == delay + 1
        assert profile.update_shapes == (1, 2, 4)
        assert profile.capacity == 1
        assert wait == 0.0005
        return nullcontext(policy)

    monkeypatch.setattr(cli, "configure_inference_process", lambda: events.append("configure"))
    monkeypatch.setattr(cli, "freeze_inference_runtime", lambda: events.append("freeze"))
    monkeypatch.setattr(cli, "resolve_checkpoint", lambda _path: bundle)
    monkeypatch.setattr(cli, "load_action_sequence_policy", load)
    monkeypatch.setattr(cli, "NetplaySession", Session)
    monkeypatch.setattr(cli, "check_realtime_budget", check_budget)
    monkeypatch.setattr(cli, "start_inference_worker", start_engine)
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
            "--online-delay",
            str(delay),
        ]
    )
    cli._play(args)
    assert events == ["configure", "prepare", "freeze", "session", "enter", "connect"]


@pytest.mark.parametrize(
    ("profile", "mode", "prefix", "stride", "observed"),
    [
        ("local", "kv_cache", 0, 2, True),
        ("official-059", "window", 2, 2, False),
        ("local-stride-one", "kv_cache", 0, 1, True),
    ],
)
def test_local_profiles_keep_physical_delay_separate_from_fixed_prefix(
    monkeypatch: pytest.MonkeyPatch, profile: str, mode: str, prefix: int, stride: int, observed: bool
) -> None:
    prepared = []
    loaded = []
    adapters = []

    class Policy:
        return_p90 = 19.75

        def prepare_prediction(self, runtime, horizon, fixed_prefix):
            prepared.append((runtime, horizon, fixed_prefix))

    def load(_path, **options):
        loaded.append(options)
        return Policy()

    def adapter(_policy, _runtime, timing, **options):
        adapters.append((timing, options))
        return object()

    def matches(_session, _matches, factory, **_options):
        factory()
        return [[object()]]

    monkeypatch.setattr(cli, "configure_inference_process", lambda: None)
    monkeypatch.setattr(cli, "freeze_inference_runtime", lambda: None)
    monkeypatch.setattr(cli, "resolve_checkpoint", lambda _path: Path("policy.hal"))
    monkeypatch.setattr(cli, "load_action_sequence_policy", load)
    monkeypatch.setattr(cli, "PolicyBatchAdapter", adapter)
    monkeypatch.setattr(cli, "default_session_cfg", lambda _replay_dir: None)
    monkeypatch.setattr(cli, "run_matches_vec", matches)
    cli._eval(cli._policy_parser().parse_args(["eval", "policy.hal", "--profile", profile]))
    assert loaded[0]["history_mode"] == mode
    runtime, horizon, fixed_prefix = prepared[0]
    assert (runtime.transport_delays, runtime.replan_interval_frames, horizon, fixed_prefix) == (
        (0,),
        stride,
        4,
        prefix,
    )
    timing, options = adapters[0]
    assert timing == cli.FrameTiming(0, 0, prefix, stride, 4)
    assert options["observed_actions"] is observed
    assert options["desired_return"] == 19.75


def test_play_rejects_unqualified_delay() -> None:
    with pytest.raises(SystemExit):
        cli._play_parser().parse_args(["policy.hal", "HUMAN#1", "--online-delay", "1"])
