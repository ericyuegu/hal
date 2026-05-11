"""Closed-loop interface to Dolphin via libmelee.

The package separates three orthogonal concerns:

- ``Session`` (``session.py``) owns the Dolphin process and connected controllers.
  It takes a ``Matchup`` and exposes ``start_match`` / ``step`` — nothing about
  where inputs come from or what the gamestate is used for.

- ``ControllerSource`` (``controller_sources.py``) is a per-port input producer.
  Implementations pull from MDS rows, .slp files, scripted sequences, models,
  or ``InternalControllerSource`` (CPU/human, driven inside Melee itself).

- ``Trajectory`` (``trajectory.py``) is columnar per-frame data with
  ``from_slp`` / ``from_mds_rows`` / ``from_capture`` constructors. ``diff.py``
  compares any two trajectories of compatible shape.

``drive(session, matchup, sources, max_frames)`` is the single loop that powers
every composition: round-trip validation, online eval vs CPUs, self-play, RL
rollouts, human exhibitions.
"""

from hal.emulator.controller_io import ControllerInputs
from hal.emulator.controller_io import ControllerInputsValue
from hal.emulator.controller_io import MdsControllerView
from hal.emulator.controller_io import apply_inputs
from hal.emulator.controller_sources import ControllerSource
from hal.emulator.controller_sources import InternalControllerSource
from hal.emulator.controller_sources import MdsControllerSource
from hal.emulator.controller_sources import ScriptedControllerSource
from hal.emulator.diff import DiffReport
from hal.emulator.diff import diff
from hal.emulator.drive import drive
from hal.emulator.session import Matchup
from hal.emulator.session import PlayerSetup
from hal.emulator.session import Session
from hal.emulator.trajectory import Trajectory
