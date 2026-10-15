"""Play a person against a policy in a windowed Slippi Dolphin.

The person plays port 1 on a GameCube controller adapter. The policy plays
port 2 through a blocking pipe, so Dolphin waits for each policy action and
the policy plays without input delay. The policy picks its character, then
starts the match on the requested stage after the person picks a character.
"""

from pathlib import Path
from typing import Final

import melee

from hal.sim.netplay import tested_dolphin_version
from hal.sim.process_vec import drive_process_vec
from hal.sim.rollout import ChunkPolicy
from hal.sim.rollout import VecMatch
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.session import SessionOptions
from hal.sim.trajectory import Trajectory

PERSON_PORT: Final[int] = 1
POLICY_PORT: Final[int] = 2
# The person needs time to pick a character, and a paused game sends no frames.
MENU_TIMEOUT_SECONDS: Final[float] = 600.0
PAUSE_TIMEOUT_SECONDS: Final[float] = 600.0


def local_session_options(
    iso_path: str | Path, dolphin_path: str | Path, replay_dir: Path, slippi_port: int
) -> SessionOptions:
    return SessionOptions(
        iso_path=iso_path,
        dolphin_path=dolphin_path,
        slippi_port=slippi_port,
        step_timeout_seconds=PAUSE_TIMEOUT_SECONDS,
        start_timeout_seconds=MENU_TIMEOUT_SECONDS,
        tmp_home_directory=True,
        replay_dir=replay_dir,
        blocking_input=True,
        emulation_speed=1.0,
        use_exi_inputs=False,
        enable_ffw=False,
        disable_audio=False,
        polling_mode=True,
        instant_match_restart=False,
        dolphin_version=tested_dolphin_version(str(dolphin_path)),
        dolphin_batch=True,
    )


def local_matchup(character: melee.Character, stage: melee.Stage) -> VecMatch:
    # The person's character comes from the menu. The worker reads it from the first frame.
    person = PlayerSetup(PERSON_PORT, character, controller_type=melee.ControllerType.GCN_ADAPTER)
    return VecMatch(Matchup(stage, (person, PlayerSetup(POLICY_PORT, character))), model_ports=(POLICY_PORT,))


def run_local_match(
    options: SessionOptions, match: VecMatch, policy: ChunkPolicy, *, max_frames: int
) -> list[Trajectory]:
    (boot,) = drive_process_vec(
        [options],
        [match],
        policy,
        max_frames=max_frames,
        worker_timeout_seconds=PAUSE_TIMEOUT_SECONDS,
        startup_timeout_seconds=MENU_TIMEOUT_SECONDS,
    )
    return boot
