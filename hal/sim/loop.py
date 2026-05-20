"""The single per-frame loop that powers every closed-loop composition."""

from collections.abc import Mapping

from hal.sim.session import Matchup
from hal.sim.session import Session
from hal.sim.sources import ControllerSource
from hal.sim.trajectory import Trajectory


def drive(
    session_: Session,
    matchup: Matchup,
    sources: Mapping[int, ControllerSource],
    max_frames: int,
) -> Trajectory:
    """Run a match for up to ``max_frames`` frames, capturing each gamestate.

    Per-port ``sources`` produce inputs each frame. A source that returns
    ``None`` (``InternalControllerSource`` for CPU/human ports) is skipped
    when applying inputs. The same loop powers round-trip replay, online eval
    against CPUs, self-play, RL rollouts, and human exhibitions.

    Stops early if the match leaves IN_GAME / SUDDEN_DEATH.
    """
    captured: list[dict] = [session_.start_match(matchup)]
    for t in range(max_frames - 1):
        last = captured[-1]
        per_port = {p: src(t, last) for p, src in sources.items()}
        per_port = {p: i for p, i in per_port.items() if i is not None}
        frame, in_game = session_.step(per_port)
        captured.append(frame)
        if not in_game:
            break
    ports = tuple(p.port for p in matchup.players)
    return Trajectory.from_capture(captured, ports)
