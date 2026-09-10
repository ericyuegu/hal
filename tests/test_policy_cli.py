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
        def prepare(self, runtime) -> None:
            assert runtime.transport_delays == (2,)
            events.append("prepare")

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
        return SimpleNamespace(trajectory=[1], inference_p95_ms=1.0, transport_correction_frames=0)

    monkeypatch.setattr(cli, "resolve_checkpoint", lambda _path: bundle)
    monkeypatch.setattr(cli, "load_policy", lambda *_args, **_kwargs: Policy())
    monkeypatch.setattr(cli, "NetplaySession", Session)
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
        ]
    )
    cli._play(args)
    assert events == ["prepare", "session", "enter", "connect"]
