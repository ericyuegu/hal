"""Sim-aware, model-agnostic eval primitives.

The harness only knows about ``ControllerSource``: experiments pass in their
own model-specific source impl (which owns the model + preprocessing +
rolling-history state) and the harness drives one match. None of this layer
imports torch.

Note: ``run_match`` returns ``None`` on Session failure (e.g. Dolphin
startup race, peppi parse error) rather than raising — eval sweeps want
to log-and-continue across many stages, not abort on the first crash.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from hal.sim.loop import drive
from hal.sim.session import Matchup
from hal.sim.session import Session
from hal.sim.sources import ControllerSource
from hal.sim.trajectory import Trajectory


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Inputs to ``Session(...)`` that don't depend on the match itself."""

    iso_path: str | Path
    dolphin_path: str | Path
    use_exi_inputs: bool = True
    enable_ffw: bool = True
    emulation_speed: float = 0.0
    blocking_input: bool = True
    replay_dir: str | Path | None = None
    step_timeout_seconds: float = 30.0
    tmp_home_directory: bool = True


def run_match(
    session_cfg: SessionConfig,
    matchup: Matchup,
    sources: Mapping[int, ControllerSource],
    *,
    max_frames: int,
) -> Trajectory | None:
    """Drive one match end-to-end. Returns the trajectory, or None if the
    Session raised (logged at WARNING)."""
    try:
        with Session(
            iso_path=session_cfg.iso_path,
            dolphin_path=session_cfg.dolphin_path,
            blocking_input=session_cfg.blocking_input,
            tmp_home_directory=session_cfg.tmp_home_directory,
            replay_dir=session_cfg.replay_dir,
            step_timeout_seconds=session_cfg.step_timeout_seconds,
            use_exi_inputs=session_cfg.use_exi_inputs,
            enable_ffw=session_cfg.enable_ffw,
            emulation_speed=session_cfg.emulation_speed,
        ) as s:
            return drive(s, matchup, sources, max_frames=max_frames)
    except Exception as e:
        logger.warning(f"run_match: Session crashed: {e!r}")
        return None
