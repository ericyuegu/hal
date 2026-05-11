"""Dolphin/Console lifecycle, port configuration, and per-frame stepping.

A ``Session`` knows how to: boot Dolphin, attach controllers per a ``Matchup``,
drive the menus to the first in-game frame, then advance one frame at a time
given per-port inputs. It does not know where inputs come from or what the
captured gamestate is used for — those are ``ControllerSource`` and consumer
concerns.

Teardown always kills the Dolphin process, even when the caller raises mid-
match. Use as a context manager.
"""

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Literal
from typing import Self

import melee
from loguru import logger

from hal.data.manifest import ReplayIndexEntry
from hal.emulator.controller_io import ControllerInputs
from hal.emulator.controller_io import apply_inputs
from hal.wire import slp_character_to_libmelee
from hal.wire import slp_stage_to_libmelee

# Menu states that signal "match is live, drive() can take over."
LIVE_MENU_STATES: frozenset[melee.Menu] = frozenset({melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH})


@dataclass(frozen=True, slots=True)
class PlayerSetup:
    """One port's configuration for a match."""

    port: int  # libmelee port, 1..4
    character: melee.Character
    costume: int = 0
    controller_type: melee.ControllerType = melee.ControllerType.STANDARD
    cpu_level: int = 0  # 0 means human/STANDARD; 1..9 for CPU
    name: str = ""


@dataclass(frozen=True, slots=True)
class Matchup:
    """Stage + per-port player setup. Independent of how the match's data
    was sourced (synthetic config, RL spec, replay manifest)."""

    stage: melee.Stage
    players: tuple[PlayerSetup, ...]


@dataclass(frozen=True, slots=True)
class ReplayMatchup(Matchup):
    """Matchup whose inputs/gamestate come from an MDS row.

    Carries the libmelee-port → MDS-prefix mapping (``{1: "p1", 2: "p2"}``)
    so round-trip / diff tooling knows which columns belong to which
    port. Synthetic matchups (RL self-play, eval-vs-CPU, smoke tests)
    use the plain ``Matchup`` and never construct this.
    """

    port_to_mds_prefix: dict[int, Literal["p1", "p2"]] = field(default_factory=dict)

    @classmethod
    def from_replay(cls, entry: ReplayIndexEntry) -> ReplayMatchup:
        """Derive from a manifest entry. Each PlayerEntry is mapped to a
        STANDARD-controller PlayerSetup on its original port; the two
        lowest-port players are tagged with MDS prefixes p1 / p2.
        """
        sorted_entries = sorted(entry.players, key=lambda p: p.port)
        if len(sorted_entries) < 2:
            raise ValueError(f"replay {entry.path} has fewer than 2 players")
        port_to_mds_prefix = {sorted_entries[0].port: "p1", sorted_entries[1].port: "p2"}
        players = tuple(
            PlayerSetup(
                port=p.port,
                character=slp_character_to_libmelee(p.character),
                costume=p.costume,
                controller_type=melee.ControllerType.STANDARD,
                cpu_level=0,
                name=p.name or p.code or "",
            )
            for p in sorted_entries
        )
        return cls(
            stage=slp_stage_to_libmelee(entry.stage),
            players=players,
            port_to_mds_prefix=port_to_mds_prefix,
        )


class Session:
    """Drives a single Dolphin instance.

    Construct, enter the context, ``start_match(matchup)`` to land at the first
    in-game frame, then loop ``step(inputs)`` until the trajectory ends. Exit
    of the context kills Dolphin even on exception.
    """

    def __init__(
        self,
        iso_path: str | Path,
        *,
        dolphin_path: str | Path,
        slippi_port: int = 51441,
        step_timeout_seconds: float = 5.0,
        setup_gecko_codes: bool = True,
        frozen_stadium: bool = True,
        tmp_home_directory: bool = True,
        replay_dir: str | Path | None = None,
    ) -> None:
        self.iso_path = str(iso_path)
        self.dolphin_path = str(dolphin_path)
        self.slippi_port = slippi_port
        self.step_timeout_seconds = step_timeout_seconds
        # Whether libmelee writes its custom GALE01r2.ini Gecko-code file.
        # The codes shipped are all $Optional (off by default), so this
        # generally has no behavioral effect — but exposing the toggle is
        # useful for round-trip experiments that need to match an .slp
        # recorded without libmelee's setup.
        self.setup_gecko_codes = setup_gecko_codes
        # Whether to flip the in-game "Frozen Stadium" menu toggle during
        # stage select. Original tournament-style .slps were typically
        # recorded with frozen stadium OFF; libmelee's MenuHelper defaults
        # it ON. Mismatch here is the most plausible source of post-spawn
        # physics drift even on non-PS stages.
        self.frozen_stadium = frozen_stadium
        # Use a throwaway tmp Dolphin home dir (libmelee default). Set False
        # to persist the Slippi-written .slp from this session.
        self.tmp_home_directory = tmp_home_directory
        # Where Slippi-Ishiiruka writes recorded .slp files. None falls back
        # to libmelee's default (~/Slippi or the tmp home if applicable).
        self.replay_dir = str(replay_dir) if replay_dir is not None else None
        self._console: melee.Console | None = None
        self._controllers: dict[int, melee.Controller] = {}
        self._menu_helpers: dict[int, melee.MenuHelper] = {}
        self._matchup: Matchup | None = None

    def __enter__(self) -> Self:
        self._boot()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._teardown()

    def _boot(self) -> None:
        """Construct Console — Dolphin doesn't launch until ``start_match``."""
        self._console = melee.Console(
            path=self.dolphin_path,
            slippi_port=self.slippi_port,
            blocking_input=False,
            polling_mode=False,
            setup_gecko_codes=self.setup_gecko_codes,
            tmp_home_directory=self.tmp_home_directory,
            replay_dir=self.replay_dir,
        )

    def _teardown(self) -> None:
        if self._console is not None:
            # libmelee's Console.stop() SIGKILLs Dolphin immediately, which
            # leaves Slippi's recorded .slp truncated (no GameEnd footer →
            # peppi can't parse it). Send SIGTERM first and wait briefly so
            # the .slp file-write thread finalizes the file.
            proc = getattr(self._console, "_process", None)
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3.0)
                except Exception as e:  # noqa: BLE001 — teardown must not raise
                    logger.warning(f"Console SIGTERM wait failed: {e}")
            try:
                self._console.stop()
            except Exception as e:  # noqa: BLE001 — teardown must not raise
                logger.warning(f"Console.stop() raised on teardown: {e}")
            self._console = None
        self._controllers.clear()
        self._menu_helpers.clear()

    def start_match(self, matchup: Matchup) -> dict:
        """Start Dolphin, configure controllers, and drive menus to IN_GAME.

        Returns the canonical-frame dict for the first in-game frame.
        """
        if self._console is None:
            raise RuntimeError("Session must be used as a context manager")
        self._matchup = matchup

        for player in matchup.players:
            # fix_analog_inputs=False so apply_inputs can write the wire
            # format that round-trips raw int8 directly. See
            # hal/emulator/controller_io.py for the wire math.
            controller = melee.Controller(
                console=self._console,
                port=player.port,
                type=player.controller_type,
                fix_analog_inputs=False,
            )
            self._controllers[player.port] = controller

        self._console.run(iso_path=self.iso_path)
        if not self._console.connect():
            raise RuntimeError("failed to connect to Dolphin Slippi server")

        for port, controller in self._controllers.items():
            if not controller.connect():
                raise RuntimeError(f"failed to connect controller on port {port}")
            self._menu_helpers[port] = melee.MenuHelper()

        # Drive menus until match goes live.
        nav_steps = 0
        while True:
            gamestate = self._step_blocking()
            if gamestate.menu_state in LIVE_MENU_STATES:
                return gamestate.to_canonical_dict()
            nav_steps += 1
            if nav_steps % 2000 == 0:
                logger.warning(
                    f"start_match: {nav_steps} menu steps without entering LIVE (still on {gamestate.menu_state})"
                )
            self._drive_menus(gamestate)

    def step(self, inputs: dict[int, ControllerInputs]) -> tuple[dict, bool]:
        """Punch inputs, advance one frame, return (canonical, in_game).

        ``in_game=False`` signals match end (menu state left IN_GAME /
        SUDDEN_DEATH). The caller — typically ``drive`` — uses this to stop
        the playback loop early.
        """
        if self._console is None:
            raise RuntimeError("Session must be used as a context manager")
        for port, src in inputs.items():
            apply_inputs(self._controllers[port], src)
        gamestate = self._step_blocking()
        return gamestate.to_canonical_dict(), gamestate.menu_state in LIVE_MENU_STATES

    def _step_blocking(self) -> melee.GameState:
        """``console.step`` retry-poll, dropping intermediate Nones.

        Bounded by ``step_timeout_seconds`` per call. Raises ``TimeoutError``
        if Dolphin produces no frame within the budget — guards against
        emulator hangs and disconnected slippstream sessions that would
        otherwise spin forever.
        """
        assert self._console is not None
        deadline = time.monotonic() + self.step_timeout_seconds
        while True:
            gs = self._console.step()
            if gs is not None:
                return gs
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Dolphin produced no frame in {self.step_timeout_seconds:.1f}s; "
                    "emulator hung or slippstream disconnected"
                )

    def _drive_menus(self, gamestate: melee.GameState) -> None:
        assert self._matchup is not None
        # libmelee's MenuHelper.menu_helper_simple navigates one port at a
        # time; in stage-select, only ONE controller may autostart, otherwise
        # both fight to move the cursor and the menu hangs forever. We pick
        # the lowest-port player as the autostart driver.
        autostart_port = min(p.port for p in self._matchup.players)
        for player in self._matchup.players:
            self._menu_helpers[player.port].menu_helper_simple(
                gamestate=gamestate,
                controller=self._controllers[player.port],
                character_selected=player.character,
                stage_selected=self._matchup.stage,
                cpu_level=player.cpu_level,
                costume=player.costume,
                autostart=player.port == autostart_port,
                frozen_stadium=self.frozen_stadium,
            )


@contextmanager
def session(iso_path: str | Path, *, dolphin_path: str | Path, **kwargs: object) -> Iterator[Session]:
    """Convenience: ``with session(iso, dolphin_path=...) as s: ...``."""
    s = Session(iso_path, dolphin_path=dolphin_path, **kwargs)
    try:
        with s:
            yield s
    except KeyboardInterrupt:
        sys.stderr.write("Caught KeyboardInterrupt; tearing down session.\n")
        raise
