"""Dolphin/Console lifecycle, port configuration, and per-frame stepping.

A ``Session`` knows how to: boot Dolphin, attach controllers per a ``Matchup``,
drive the menus to the first in-game frame, then advance one frame at a time
given per-port inputs. It does not know where inputs come from or what the
captured gamestate is used for — those are ``ControllerSource`` and consumer
concerns.

Teardown always kills the Dolphin process, even when the caller raises mid-
match. Use as a context manager.
"""

import atexit
import ctypes
import signal as _signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextlib import suppress
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Literal
from typing import Self

import melee
from loguru import logger

from hal.data.index import ReplayIndexEntry
from hal.data.slp_finalize import finalize_replay_dir
from hal.sim.inputs import ControllerInputs
from hal.sim.inputs import apply_inputs
from hal.wire import slp_character_to_libmelee
from hal.wire import slp_stage_to_libmelee

# Linux-only PR_SET_PDEATHSIG: have the kernel SIGKILL the Dolphin child if
# this Python process dies before _teardown can run (e.g. parent SIGKILL'd,
# OOM, segfault). Belt-and-suspenders on top of __exit__/atexit cleanup —
# without it, an orphaned Dolphin keeps UDP 51441 bound and breaks the next
# Session boot until reboot or manual kill (see PID-575155 incident).
_PR_SET_PDEATHSIG = 1


def _set_pdeathsig_sigkill() -> None:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.prctl(_PR_SET_PDEATHSIG, _signal.SIGKILL, 0, 0, 0)


# The patch swaps a process-global (``subprocess.Popen``), so concurrent boots
# (drive_vec starts N Sessions on a thread pool) must not interleave their
# patch/restore — otherwise one thread's restore clobbers another's, leaking the
# wrapper or dropping the pdeathsig. The window guarded is just ``Console.run``'s
# launch, which only spawns (it doesn't wait), so serializing it is cheap.
_POPEN_PATCH_LOCK = threading.Lock()


@contextmanager
def _popen_with_pdeathsig() -> Iterator[None]:
    """Monkeypatch ``subprocess.Popen`` to inject ``PR_SET_PDEATHSIG`` into any
    child spawned inside the block. Restores on exit. Used to wrap libmelee's
    ``Console.run`` (the actual ``Popen(...)`` call lives inside the library
    and is otherwise out of our reach). Serialized across threads via
    ``_POPEN_PATCH_LOCK`` since it mutates a process-global."""
    with _POPEN_PATCH_LOCK:
        original = subprocess.Popen

        def _wrapped(*args, **kwargs):
            user_pre = kwargs.pop("preexec_fn", None)

            def _pre():
                _set_pdeathsig_sigkill()
                if user_pre is not None:
                    user_pre()

            kwargs["preexec_fn"] = _pre
            return original(*args, **kwargs)

        subprocess.Popen = _wrapped  # type: ignore[misc]
        try:
            yield
        finally:
            subprocess.Popen = original  # type: ignore[misc]


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
        port_to_mds_prefix: dict[int, Literal["p1", "p2"]] = {
            sorted_entries[0].port: "p1",
            sorted_entries[1].port: "p2",
        }
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
        start_timeout_seconds: float = 120.0,
        setup_gecko_codes: bool = True,
        frozen_stadium: bool = True,
        tmp_home_directory: bool = True,
        replay_dir: str | Path | None = None,
        blocking_input: bool = False,
        emulation_speed: float = 1.0,
        use_exi_inputs: bool = False,
        enable_ffw: bool = False,
        polling_mode: bool = False,
    ) -> None:
        self.iso_path = str(iso_path)
        self.dolphin_path = str(dolphin_path)
        self.slippi_port = slippi_port
        self.step_timeout_seconds = step_timeout_seconds
        # Wall-clock budget for driving the menus to the first in-game frame.
        # Menu frames stream fast under FFW, so step_timeout_seconds (a per-poll
        # budget) never catches a menu that simply never reaches IN_GAME — a
        # degenerate match or a menu-nav edge case would otherwise spin at 100%
        # CPU forever. This caps the whole navigation and raises instead.
        self.start_timeout_seconds = start_timeout_seconds
        # Block console.step until the controller pipe has been read.
        # Required for closed-loop control where the input punched for frame N
        # must land before frame N+1 is advanced; without it, model inputs can
        # race the emulator and the policy effectively drives stale state.
        # Replay-style consumers (round-trip diff, MDS playback) keep the
        # default False so they don't pay the per-frame block.
        self.blocking_input = blocking_input
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
        # Speed knob (0 = uncapped). FFW + use_exi_inputs require the exi-ai
        # Ishiiruka build (DOLPHIN_EXIAI); pipe-input pathway ignores them.
        self.emulation_speed = emulation_speed
        self.use_exi_inputs = use_exi_inputs
        self.enable_ffw = enable_ffw
        # Non-blocking slippstream reads, so ``_step_blocking`` polls and its
        # ``step_timeout_seconds`` deadline can actually fire. With the default
        # (False) libmelee blocks in ``recv`` forever if Dolphin stops streaming
        # frames (e.g. an agent that pauses the match), which would hang an
        # unattended eval. Replay-style consumers (round-trip, MDS playback) keep
        # False so frame delivery stays bit-for-bit and pays no poll spin.
        self.polling_mode = polling_mode
        self._console: melee.Console | None = None
        self._controllers: dict[int, melee.Controller] = {}
        self._menu_helpers: dict[int, melee.MenuHelper] = {}
        self._matchup: Matchup | None = None
        # Bound method so atexit.unregister can target it during _teardown.
        self._atexit_kill = self._kill_dolphin_only

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
            blocking_input=self.blocking_input,
            polling_mode=self.polling_mode,
            setup_gecko_codes=self.setup_gecko_codes,
            tmp_home_directory=self.tmp_home_directory,
            replay_dir=self.replay_dir,
            emulation_speed=self.emulation_speed,
            use_exi_inputs=self.use_exi_inputs,
            enable_ffw=self.enable_ffw,
        )
        # libmelee writes Dolphin.ini via Python configparser, which lowercases
        # option names ("slippireplaydir = ..."); Slippi-Ishiiruka's own parser
        # only honors the CamelCase ``SlippiReplayDir``, so without this fixup
        # ``replay_dir`` is silently ignored and .slps land in ~/Slippi.
        if self.replay_dir:
            self._fix_dolphin_ini_case()
        # Belt-and-suspenders cleanup: if Python exits between __enter__ and
        # __exit__ (e.g. unhandled exception in __enter__ caller, sys.exit
        # mid-test, hard interpreter shutdown), still SIGKILL Dolphin.
        atexit.register(self._atexit_kill)

    def _fix_dolphin_ini_case(self) -> None:
        """Replace libmelee's lowercased ``slippireplaydir = ...`` with the
        CamelCase ``SlippiReplayDir`` that Ishiiruka actually reads. Leaving
        both in place would also confuse libmelee's own
        ``setup_dolphin_controller``: it re-parses the ini via configparser,
        which treats the two cases as a duplicate option and raises."""
        if self._console is None:
            return
        try:
            ini_path = Path(self._console._get_dolphin_config_path()) / "Dolphin.ini"
        except AttributeError:
            return
        if not ini_path.is_file():
            return
        lines = ini_path.read_text().splitlines(keepends=True)
        target = f"SlippiReplayDir = {self.replay_dir}\n"
        out: list[str] = []
        replaced = False
        for line in lines:
            if line.lower().lstrip().startswith("slippireplaydir"):
                if not replaced:
                    out.append(target)
                    replaced = True
                # Drop any further duplicates.
                continue
            out.append(line)
        if not replaced:
            # libmelee didn't write the lowercased key (replay_dir was None at
            # that time?), so insert ours at the top of [Core].
            text = "".join(out)
            if "[Core]\n" in text:
                text = text.replace("[Core]\n", f"[Core]\n{target}", 1)
            else:
                text = text.rstrip() + f"\n\n[Core]\n{target}"
            ini_path.write_text(text)
            return
        ini_path.write_text("".join(out))

    def _kill_dolphin_only(self) -> None:
        """Hard-kill the Dolphin subprocess. Idempotent, swallow-all. Called
        from _teardown (graceful path) and as an atexit handler (last-ditch).
        Does NOT touch libmelee's internal state — that's _teardown's job."""
        if self._console is None:
            return
        proc = getattr(self._console, "_process", None)
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.kill()
            proc.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Dolphin SIGKILL failed: {e}")

    def _teardown(self) -> None:
        # Unhook the atexit handler first so a normal teardown doesn't fire it
        # again at interpreter shutdown (harmless but noisy).
        with suppress(Exception):
            atexit.unregister(self._atexit_kill)
        if self._console is None:
            self._controllers.clear()
            self._menu_helpers.clear()
            return
        proc = getattr(self._console, "_process", None)
        # 1. Graceful SIGTERM so Slippi can finish flushing its current frame and
        #    cleanly close a match that reached GAME_END (which finalizes the
        #    .slp with full metadata). A match abandoned mid-game — stopped at
        #    max_frames while still IN_GAME — never gets GAME_END, so SIGTERM
        #    can't finalize it; step 4 repairs those.
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=10.0)
            except (OSError, subprocess.TimeoutExpired, RuntimeError) as e:
                logger.warning(f"Console SIGTERM wait failed: {e}")
        # 2. Hard SIGKILL ourselves before delegating to libmelee — its
        #    Console.stop() can raise inside slippstream.shutdown() when the
        #    worker never started, leaving its own proc.kill() unreached and
        #    Dolphin orphaned (PID 575155 incident, 2026-05-21).
        self._kill_dolphin_only()
        # 3. Now let libmelee tear down its state (slippstream worker handle,
        #    temp Dolphin home). Errors here are non-fatal since the
        #    Dolphin process is already dead.
        try:
            self._console.stop()
        except (OSError, subprocess.TimeoutExpired, RuntimeError, AssertionError) as e:
            logger.warning(f"Console.stop() raised on teardown: {e}")
        # 4. Finalize any .slp Slippi left unclosed (rawLength == 0): a match
        #    that hit max_frames mid-game is otherwise unparseable by peppi /
        #    slippilab even though the frame data is intact. No-op for matches
        #    that ended cleanly (already finalized by Dolphin at GAME_END).
        if self.replay_dir is not None:
            repaired = finalize_replay_dir(self.replay_dir)
            if repaired:
                logger.info(f"finalized {len(repaired)} unclosed .slp in {self.replay_dir}")
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
            # hal/sim/inputs.py for the wire math.
            controller = melee.Controller(
                console=self._console,
                port=player.port,
                type=player.controller_type,
                fix_analog_inputs=False,
            )
            self._controllers[player.port] = controller

        with _popen_with_pdeathsig():
            self._console.run(iso_path=self.iso_path)
        if not self._console.connect():
            raise RuntimeError("failed to connect to Dolphin Slippi server")

        for port, controller in self._controllers.items():
            if not controller.connect():
                raise RuntimeError(f"failed to connect controller on port {port}")
            self._menu_helpers[port] = melee.MenuHelper()

        return self._navigate_to_live()

    def _navigate_to_live(self) -> dict:
        """Drive the menus until the match goes live, returning the first
        in-game canonical frame.

        Raises ``TimeoutError`` if ``start_timeout_seconds`` elapses without
        reaching IN_GAME / SUDDEN_DEATH. Menu frames stream fast under FFW, so
        the per-poll ``step_timeout_seconds`` can't catch a menu that never
        goes live (each ``_step_blocking`` returns promptly); this wall-clock
        cap turns that logical loop into a loud, clean failure that callers
        (``drive_vec`` / ``run_match``) already log-and-continue on.
        """
        deadline = time.monotonic() + self.start_timeout_seconds
        nav_steps = 0
        # Track the autostart port's stage-select cursor extent: when a stall does
        # surface (libmelee's bang-bang stage cursor flakily fails to settle under
        # FFW load), this says whether it got near the target or nowhere close.
        sss_box: list[float] | None = None  # [xmin, xmax, ymin, ymax]
        while True:
            gamestate = self._step_blocking()
            if gamestate.menu_state in LIVE_MENU_STATES:
                return gamestate.to_canonical_dict()
            if gamestate.menu_state == melee.Menu.STAGE_SELECT and self._matchup is not None:
                autostart_port = min(p.port for p in self._matchup.players)
                cur = getattr(gamestate.players.get(autostart_port), "cursor", None)
                if cur is not None:
                    sss_box = (
                        [cur.x, cur.x, cur.y, cur.y]
                        if sss_box is None
                        else [
                            min(sss_box[0], cur.x),
                            max(sss_box[1], cur.x),
                            min(sss_box[2], cur.y),
                            max(sss_box[3], cur.y),
                        ]
                    )
            if time.monotonic() > deadline:
                stage = f"; stage={self._matchup.stage.name}" if self._matchup is not None else ""
                box = (
                    f"; SSS cursor x∈[{sss_box[0]:.1f},{sss_box[1]:.1f}] y∈[{sss_box[2]:.1f},{sss_box[3]:.1f}]"
                    if sss_box is not None
                    else ""
                )
                raise TimeoutError(
                    f"start_match: did not reach IN_GAME within {self.start_timeout_seconds:.0f}s "
                    f"(stuck on {gamestate.menu_state} after {nav_steps} menu steps{stage}{box})"
                )
            nav_steps += 1
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
        # In CSS, hold autostart until every CPU port has been toggled from
        # HUMAN to CPU AND the level slider matches. Without this, port 1's
        # coin-down races into the stage select while port 2's slot is still
        # HUMAN, the match starts with no AI bound to port 2, and the lvl-9
        # opponent never engages (slp records port 2 type=HUMAN,
        # cpu_level=None). Outside CSS the per-port state has already been
        # reset (libmelee re-creates PlayerState() on entering STAGE_SELECT),
        # so the gate would deadlock there — the configuration is locked in
        # by then, allow autostart unconditionally.
        in_css = gamestate.menu_state in (melee.Menu.CHARACTER_SELECT, melee.Menu.SLIPPI_ONLINE_CSS)
        cpu_ready = not in_css or all(
            (s := gamestate.players.get(p.port)) is not None
            and s.controller_status == melee.ControllerStatus.CONTROLLER_CPU
            and s.cpu_level == p.cpu_level
            for p in self._matchup.players
            if p.cpu_level > 0
        )
        for player in self._matchup.players:
            self._menu_helpers[player.port].menu_helper_simple(
                gamestate=gamestate,
                controller=self._controllers[player.port],
                character_selected=player.character,
                stage_selected=self._matchup.stage,
                cpu_level=player.cpu_level,
                costume=player.costume,
                autostart=player.port == autostart_port and cpu_ready,
                frozen_stadium=self.frozen_stadium,
            )


@contextmanager
def session(
    iso_path: str | Path,
    *,
    dolphin_path: str | Path,
    slippi_port: int = 51441,
    step_timeout_seconds: float = 5.0,
    setup_gecko_codes: bool = True,
    frozen_stadium: bool = True,
    tmp_home_directory: bool = True,
    replay_dir: str | Path | None = None,
) -> Iterator[Session]:
    """Convenience: ``with session(iso, dolphin_path=...) as s: ...``."""
    s = Session(
        iso_path,
        dolphin_path=dolphin_path,
        slippi_port=slippi_port,
        step_timeout_seconds=step_timeout_seconds,
        setup_gecko_codes=setup_gecko_codes,
        frozen_stadium=frozen_stadium,
        tmp_home_directory=tmp_home_directory,
        replay_dir=replay_dir,
    )
    try:
        with s:
            yield s
    except KeyboardInterrupt:
        sys.stderr.write("Caught KeyboardInterrupt; tearing down session.\n")
        raise
