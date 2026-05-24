"""Public API. Notebooks and CLI scripts import from here, not deep paths."""

from hal.data.extract import extract_replay
from hal.data.index import ReplayIndexEntry
from hal.data.index import read_jsonl
from hal.data.index import write_jsonl
from hal.fixtures import Fixture
from hal.fixtures import ensure
from hal.fixtures import ensure_all
from hal.sim.diff import DiffReport
from hal.sim.diff import diff
from hal.sim.inputs import ControllerInputs
from hal.sim.inputs import ControllerInputsValue
from hal.sim.inputs import apply_inputs
from hal.sim.loop import drive
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.session import ReplayMatchup
from hal.sim.session import Session
from hal.sim.session import session
from hal.sim.sources import ControllerSource
from hal.sim.sources import InternalControllerSource
from hal.sim.sources import MdsControllerSource
from hal.sim.sources import ScriptedControllerSource
from hal.sim.trajectory import Trajectory
from hal.sim.vec import BatchPolicy
from hal.sim.vec import Slot
from hal.sim.vec import VecMatch
from hal.sim.vec import drive_vec

__all__ = [
    "BatchPolicy",
    "ControllerInputs",
    "ControllerInputsValue",
    "ControllerSource",
    "DiffReport",
    "Fixture",
    "InternalControllerSource",
    "Matchup",
    "MdsControllerSource",
    "PlayerSetup",
    "ReplayIndexEntry",
    "ReplayMatchup",
    "ScriptedControllerSource",
    "Session",
    "Slot",
    "Trajectory",
    "VecMatch",
    "apply_inputs",
    "diff",
    "drive",
    "drive_vec",
    "ensure",
    "ensure_all",
    "extract_replay",
    "read_jsonl",
    "session",
    "write_jsonl",
]
