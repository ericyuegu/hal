"""Replay completion checks independent of game-loop execution."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from peppi_py.game import EndMethod

from hal.eval.replays import read_new_replay_end
from hal.eval.replays import require_completed_replay


@pytest.mark.parametrize("method", [EndMethod.TIME, EndMethod.GAME, EndMethod.RESOLVED])
def test_completed_replay_accepts_definitive_game_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: EndMethod,
) -> None:
    replay = tmp_path / "game.slp"
    replay.touch()
    monkeypatch.setattr(
        "hal.eval.replays.peppi_py.read_slippi",
        lambda *_args, **_kwargs: SimpleNamespace(end=SimpleNamespace(method=method)),
    )
    assert require_completed_replay(tmp_path, ()) == replay


def test_completed_replay_rejects_no_contest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay = tmp_path / "game.slp"
    replay.touch()
    monkeypatch.setattr(
        "hal.eval.replays.peppi_py.read_slippi",
        lambda *_args, **_kwargs: SimpleNamespace(end=SimpleNamespace(method=EndMethod.NO_CONTEST)),
    )
    with pytest.raises(RuntimeError, match="NO_CONTEST"):
        require_completed_replay(tmp_path, ())


def test_replay_end_reports_no_contest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay = tmp_path / "game.slp"
    replay.touch()
    monkeypatch.setattr(
        "hal.eval.replays.peppi_py.read_slippi",
        lambda *_args, **_kwargs: SimpleNamespace(end=SimpleNamespace(method=EndMethod.NO_CONTEST)),
    )

    replay_end = read_new_replay_end(tmp_path, ())

    assert replay_end.path == replay
    assert replay_end.method is EndMethod.NO_CONTEST
    assert not replay_end.completed
