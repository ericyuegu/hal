"""Closed-loop interface to Dolphin via libmelee.

The package separates three orthogonal concerns:

- ``Session`` (``session.py``) owns the Dolphin process and connected controllers.
  It takes a ``Matchup`` and exposes ``start_match`` / ``step`` — nothing about
  where inputs come from or what the gamestate is used for.

- ``ControllerSource`` (``sources.py``) is a per-port input producer.
  Implementations pull from MDS rows, .slp files, scripted sequences, models,
  or ``InternalControllerSource`` (CPU/human, driven inside Melee itself).

- ``Trajectory`` (``trajectory.py``) is columnar per-frame data with
  ``from_slp`` / ``from_mds_rows`` / ``from_capture`` constructors. ``diff.py``
  compares any two trajectories of compatible shape.

``drive(session, matchup, sources, max_frames)`` is the single loop that powers
every composition: round-trip validation, online eval vs CPUs, self-play, RL
rollouts, human exhibitions.
"""

from hal.sim.diff import DiffReport
from hal.sim.diff import diff
from hal.sim.inputs import ControllerInputs
from hal.sim.inputs import ControllerInputsValue
from hal.sim.inputs import MdsControllerView
from hal.sim.inputs import apply_inputs
from hal.sim.loop import drive
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.session import Session
from hal.sim.sources import ControllerSource
from hal.sim.sources import InternalControllerSource
from hal.sim.sources import MdsControllerSource
from hal.sim.sources import ScriptedControllerSource
from hal.sim.trajectory import Trajectory
from hal.sim.vec import BatchPolicy
from hal.sim.vec import Slot
from hal.sim.vec import VecMatch
from hal.sim.vec import drive_vec
