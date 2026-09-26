"""Read and validate completed Dolphin replay records."""

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import peppi_py
from peppi_py.game import EndMethod


@dataclass(frozen=True, slots=True)
class ReplayEnd:
    path: Path
    method: EndMethod

    @property
    def completed(self) -> bool:
        return self.method in (EndMethod.TIME, EndMethod.GAME, EndMethod.RESOLVED)


def read_new_replay_end(replay_dir: Path, previous: Collection[Path]) -> ReplayEnd:
    """Read the game-end record from this invocation's replay."""
    replays = sorted(set(replay_dir.rglob("*.slp")) - set(previous))
    if len(replays) != 1:
        raise RuntimeError(f"expected one new replay in {replay_dir}, found {len(replays)}")
    replay = replays[0]
    try:
        game = peppi_py.read_slippi(str(replay), skip_frames=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(f"cannot parse completed netplay replay {replay}: {error}") from error
    if game.end is None:
        raise RuntimeError(f"netplay replay has no game-end record: {replay}")
    try:
        method = EndMethod(int(game.end.method))
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"netplay replay has unknown game-end method: {replay}") from error
    return ReplayEnd(replay, method)


def require_completed_replay(replay_dir: Path, previous: Collection[Path]) -> Path:
    """Return this invocation's replay after validating its game-end record."""
    replay_end = read_new_replay_end(replay_dir, previous)
    if not replay_end.completed:
        raise RuntimeError(f"netplay replay ended via {replay_end.method.name}: {replay_end.path}")
    return replay_end.path
