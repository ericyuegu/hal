"""Tests for the vectorized eval driver.

Fast unit tests drive ``drive_vec`` / ``run_matches_vec`` against fake Sessions,
so the orchestration (lockstep batching, shrinking live set, crash isolation,
wave chunking, per-wave policy reset, port assignment) is verified
deterministically and without Dolphin.

``test_parallel_eval_real_dolphin_*`` (``@pytest.mark.integration``) runs the
same path against real Dolphin instances with a torch-free neutral policy.

Self-play is a ``VecMatch`` whose ``model_ports`` carries both ports; a single
batched call then receives both ports' slots. Multi-wave is ``run_matches_vec``
with more matches than ``max_parallel``.
"""

import multiprocessing as mp

# libmelee's slippstream client spawns a child via mp.Process; on Python 3.14
# the default start method is "forkserver", which re-imports the worker module
# and is flaky alongside libmelee. Match the other integration tests: force
# plain fork. (No-op for the fake-Session unit tests, which spawn no process.)
if mp.get_start_method(allow_none=True) != "fork":
    mp.set_start_method("fork", force=True)

from collections.abc import Mapping
from pathlib import Path

import melee
import pytest

from hal.eval.harness import SessionConfig
from hal.eval.harness import run_matches_vec
from hal.paths import EMULATOR_PATH
from hal.paths import ISO_PATH as _ISO_PATH
from hal.sim.inputs import ControllerInputs
from hal.sim.inputs import ControllerInputsValue
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.vec import Slot
from hal.sim.vec import VecMatch
from hal.sim.vec import drive_vec

ISO_PATH = Path(_ISO_PATH)
DOLPHIN_PATH = Path(EMULATOR_PATH)

_NEUTRAL = ControllerInputsValue(main_x=0.0, main_y=0.0, c_x=0.0, c_y=0.0, trigger_l=0.0, trigger_r=0.0, buttons=0)


def _post() -> dict:
    return {
        "position": {"x": 1.0, "y": 2.0},
        "percent": 0.0,
        "shield": 60.0,
        "stock": 4,
        "direction": 1.0,
        "action": 14,
        "jumps_used": 0,
        "airborne": 0,
        "hurtbox_state": 0,
        "hitlag_left": 0.0,
    }


def _frame(i: int, ports: tuple[int, ...]) -> dict:
    return {"id": i, "start": {"random_seed": 0}, "ports": {p: {"leader": {"post": _post()}} for p in ports}}


class FakeSession:
    """Stands in for ``Session``: a context manager that yields a fixed number
    of in-game frames, then reports the match ended. ``crash_at`` raises from
    ``step`` to exercise per-match crash isolation."""

    def __init__(self, *, length: int, ports: tuple[int, ...], crash_at: int | None = None) -> None:
        self.length, self.ports, self.crash_at, self.t = length, ports, crash_at, 0

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def start_match(self, matchup: Matchup) -> dict:
        return _frame(0, self.ports)

    def step(self, inputs: Mapping[int, ControllerInputs]) -> tuple[dict, bool]:
        self.t += 1
        if self.crash_at is not None and self.t >= self.crash_at:
            raise RuntimeError("fake dolphin crash")
        return _frame(self.t, self.ports), self.t < self.length


class RecordingPolicy:
    """Records the slot-set seen each frame; returns neutral for every slot."""

    def __init__(self) -> None:
        self.frames: list[frozenset[Slot]] = []

    def __call__(self, frame_index: int, obs: Mapping[Slot, dict]) -> Mapping[Slot, ControllerInputs]:
        self.frames.append(frozenset(obs))
        return {slot: _NEUTRAL for slot in obs}


class RaisingPolicy:
    """Stands in for a model forward that blows up (e.g. CUDA OOM) on first call."""

    def __call__(self, frame_index: int, obs: Mapping[Slot, dict]) -> Mapping[Slot, ControllerInputs]:
        raise RuntimeError("model forward OOM")


def _matchup(ports: tuple[int, ...]) -> Matchup:
    return Matchup(
        stage=melee.Stage.FINAL_DESTINATION,
        players=tuple(PlayerSetup(port=p, character=melee.Character.FOX) for p in ports),
    )


def test_drive_vec_self_play_batches_both_ports_and_skips_internal() -> None:
    # match 0: self-play — both ports model-driven. match 1: vs-cpu — only port 1.
    matches = [
        VecMatch(matchup=_matchup((1, 2)), model_ports=(1, 2)),
        VecMatch(matchup=_matchup((1, 2)), model_ports=(1,)),
    ]
    sessions = [FakeSession(length=6, ports=(1, 2)), FakeSession(length=4, ports=(1, 2))]
    policy = RecordingPolicy()

    trajs = drive_vec(sessions, matches, policy, max_frames=10)

    # Frame 0: both self-play slots + the one vs-cpu slot are batched together.
    assert policy.frames[0] == {Slot(0, 1), Slot(0, 2), Slot(1, 1)}
    # The internal (CPU) port is never handed to the policy.
    assert all(Slot(1, 2) not in f for f in policy.frames)
    # match 1 ends first (len 4) → live set shrinks to just the self-play slots.
    assert policy.frames[-1] == {Slot(0, 1), Slot(0, 2)}

    # Trajectories: one per match, capturing every matchup port (model or not).
    assert [len(t) for t in trajs] == [7, 5]  # length + the start frame
    assert set(trajs[0].post) == {1, 2}
    assert set(trajs[1].post) == {1, 2}


def test_drive_vec_isolates_a_crashing_session() -> None:
    matches = [
        VecMatch(matchup=_matchup((1, 2)), model_ports=(1, 2)),
        VecMatch(matchup=_matchup((1, 2)), model_ports=(1,)),
        VecMatch(matchup=_matchup((1, 2)), model_ports=(1, 2)),
    ]
    sessions = [
        FakeSession(length=8, ports=(1, 2)),
        FakeSession(length=8, ports=(1, 2)),
        FakeSession(length=8, ports=(1, 2), crash_at=3),  # raises on its 3rd step
    ]
    policy = RecordingPolicy()

    trajs = drive_vec(sessions, matches, policy, max_frames=20)

    assert trajs[2] is None  # crashed match → None
    assert trajs[0] is not None and trajs[1] is not None  # survivors complete
    assert len(trajs[0]) == 9
    # Batch carries match 2's two slots until it crashes, then drops them.
    assert len(policy.frames[0]) == 5  # 2 + 1 + 2
    assert Slot(2, 1) not in policy.frames[-1]


def test_drive_vec_returns_none_when_start_fails() -> None:
    class StartFails(FakeSession):
        def start_match(self, matchup: Matchup) -> dict:
            raise RuntimeError("boot race")

    matches = [VecMatch(matchup=_matchup((1, 2)), model_ports=(1,)) for _ in range(2)]
    sessions = [FakeSession(length=5, ports=(1, 2)), StartFails(length=5, ports=(1, 2))]
    trajs = drive_vec(sessions, matches, RecordingPolicy(), max_frames=10)

    assert trajs[1] is None
    assert trajs[0] is not None


def test_run_matches_vec_multi_wave(monkeypatch: pytest.MonkeyPatch) -> None:
    # 5 matches, max_parallel=2 → waves of [2, 2, 1]. Mix self-play and vs-cpu.
    # Distinct lengths let us verify result alignment by trajectory length.
    specs = [
        (VecMatch(matchup=_matchup((1, 2)), model_ports=(1, 2)), 3),
        (VecMatch(matchup=_matchup((1, 2)), model_ports=(1,)), 5),
        (VecMatch(matchup=_matchup((1, 2)), model_ports=(1, 2)), 7),
        (VecMatch(matchup=_matchup((1, 2)), model_ports=(1,)), 9),
        (VecMatch(matchup=_matchup((1, 2)), model_ports=(1, 2)), 11),
    ]
    matches = [m for m, _ in specs]
    # _build_session is called once per match, in match order (wave by wave,
    # offset by offset) — hand back a fake whose length identifies the match.
    fakes = iter(FakeSession(length=n, ports=(1, 2)) for _, n in specs)
    assigned_ports: list[int] = []

    def fake_build(session_cfg, *, slippi_port, replay_dir):
        assigned_ports.append(slippi_port)
        return next(fakes)

    monkeypatch.setattr("hal.eval.harness._build_session", fake_build)

    waves_built = 0

    def policy_factory() -> RecordingPolicy:
        nonlocal waves_built
        waves_built += 1
        return RecordingPolicy()

    cfg = SessionConfig(iso_path="unused.iso", dolphin_path="unused")
    trajs = run_matches_vec(cfg, matches, policy_factory, max_frames=20, max_parallel=2, base_slippi_port=51441)

    # One fresh policy per wave; 5 matches / 2 = 3 waves.
    assert waves_built == 3
    # Ports restart at base each wave: [51441,51442 | 51441,51442 | 51441].
    assert assigned_ports == [51441, 51442, 51441, 51442, 51441]
    # Results aligned to input order (length = match length + start frame).
    assert [len(t) for t in trajs] == [4, 6, 8, 10, 12]


def test_run_matches_vec_isolates_a_policy_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # 4 matches, max_parallel=2 → waves [0,1], [2,3]. The shared batched policy
    # raises only in the 2nd wave; that wave's matches come back None while the
    # first wave's complete (a policy OOM can't be pinned to one match, so the
    # whole wave is logged-and-skipped rather than aborting the sweep).
    matches = [VecMatch(matchup=_matchup((1, 2)), model_ports=(1,)) for _ in range(4)]
    fakes = iter(FakeSession(length=5, ports=(1, 2)) for _ in range(4))
    monkeypatch.setattr("hal.eval.harness._build_session", lambda *a, **k: next(fakes))

    waves_built = 0

    def policy_factory() -> RecordingPolicy | RaisingPolicy:
        nonlocal waves_built
        waves_built += 1
        return RaisingPolicy() if waves_built == 2 else RecordingPolicy()

    cfg = SessionConfig(iso_path="unused.iso", dolphin_path="unused")
    trajs = run_matches_vec(cfg, matches, policy_factory, max_frames=20, max_parallel=2)

    assert trajs[0] is not None and trajs[1] is not None  # wave 0 survived
    assert trajs[2] is None and trajs[3] is None  # wave 1's policy raised → None


class _StartFails(FakeSession):
    """A Session whose ``start_match`` always trips the stage-select stall —
    stands in for libmelee's flaky menu navigation never reaching IN_GAME."""

    def start_match(self, matchup: Matchup) -> dict:
        raise TimeoutError("did not reach IN_GAME (stuck on Menu.STAGE_SELECT)")


def test_run_matches_vec_retries_failed_start(monkeypatch: pytest.MonkeyPatch) -> None:
    # The first start attempt stalls; the bounded retry rebuilds a fresh Session
    # on a new port and succeeds — the flaky stage-select stall clears per-attempt.
    matches = [VecMatch(matchup=_matchup((1, 2)), model_ports=(1,))]
    seq = iter([_StartFails(length=5, ports=(1, 2)), FakeSession(length=5, ports=(1, 2))])
    ports: list[int] = []

    def fake_build(session_cfg, *, slippi_port, replay_dir):
        ports.append(slippi_port)
        return next(seq)

    monkeypatch.setattr("hal.eval.harness._build_session", fake_build)
    cfg = SessionConfig(iso_path="unused.iso", dolphin_path="unused")
    trajs = run_matches_vec(cfg, matches, RecordingPolicy, max_frames=20, max_parallel=1, base_slippi_port=51441)

    assert trajs[0] is not None  # recovered on the retry
    assert ports == [51441, 51442]  # initial attempt, then a fresh port (base + max_parallel)


def test_run_matches_vec_gives_up_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every attempt stalls → the match stays None after start_retries are spent,
    # logged-and-skipped rather than hanging the sweep.
    matches = [VecMatch(matchup=_matchup((1, 2)), model_ports=(1,))]
    builds = 0

    def fake_build(session_cfg, *, slippi_port, replay_dir):
        nonlocal builds
        builds += 1
        return _StartFails(length=5, ports=(1, 2))

    monkeypatch.setattr("hal.eval.harness._build_session", fake_build)
    cfg = SessionConfig(iso_path="unused.iso", dolphin_path="unused")
    trajs = run_matches_vec(cfg, matches, RecordingPolicy, max_frames=20, max_parallel=1, start_retries=2)

    assert trajs[0] is None  # never recovered
    assert builds == 3  # initial attempt + 2 retries


# --- real-Dolphin integration ------------------------------------------------


def _self_play_matchup(stage: melee.Stage) -> Matchup:
    return Matchup(
        stage=stage,
        players=(
            PlayerSetup(port=1, character=melee.Character.FOX),
            PlayerSetup(port=2, character=melee.Character.FOX),
        ),
    )


def _vs_cpu_matchup(stage: melee.Stage, cpu_level: int = 9) -> Matchup:
    return Matchup(
        stage=stage,
        players=(
            PlayerSetup(port=1, character=melee.Character.FOX),
            PlayerSetup(port=2, character=melee.Character.FOX, cpu_level=cpu_level),
        ),
    )


class NeutralBatchPolicy:
    """Torch-free BatchPolicy: neutral inputs for every live slot, recording the
    per-frame batch size so the test can assert cross-match batching."""

    def __init__(self, batch_sizes: list[int]) -> None:
        self._sizes = batch_sizes

    def __call__(self, frame_index: int, obs: Mapping[Slot, dict]) -> Mapping[Slot, ControllerInputs]:
        self._sizes.append(len(obs))
        return {slot: _NEUTRAL for slot in obs}


def _check_prereqs() -> None:
    if not ISO_PATH.is_file():
        pytest.skip(f"ISO missing at {ISO_PATH}; run `python -m hal.scripts.fetch --name ssbm.ciso`")
    if not DOLPHIN_PATH.is_file():
        pytest.skip(f"Dolphin missing at {DOLPHIN_PATH}")


@pytest.mark.integration
def test_parallel_eval_real_dolphin_multi_wave_self_play() -> None:
    """Three real matches — two self-play (both ports model-driven) + one
    vs-CPU — across two waves (max_parallel=2). Verifies concurrent boot on
    distinct slippi_ports, the single batched policy call spanning multiple
    matches, and a clean per-match Trajectory from each real rollout."""
    _check_prereqs()
    session_cfg = SessionConfig(
        iso_path=str(ISO_PATH),
        dolphin_path=str(DOLPHIN_PATH),
        use_exi_inputs=True,
        enable_ffw=True,
        emulation_speed=0.0,
        blocking_input=True,
        step_timeout_seconds=30.0,
        tmp_home_directory=True,
    )
    matches = [
        VecMatch(matchup=_self_play_matchup(melee.Stage.FINAL_DESTINATION), model_ports=(1, 2)),
        VecMatch(matchup=_vs_cpu_matchup(melee.Stage.BATTLEFIELD), model_ports=(1,)),
        VecMatch(matchup=_self_play_matchup(melee.Stage.YOSHIS_STORY), model_ports=(1, 2)),
    ]
    batch_sizes: list[int] = []
    trajs = run_matches_vec(
        session_cfg,
        matches,
        lambda: NeutralBatchPolicy(batch_sizes),
        max_frames=300,
        max_parallel=2,
    )

    assert all(t is not None for t in trajs), "a real match crashed"
    assert all(len(t) > 50 for t in trajs), [None if t is None else len(t) for t in trajs]
    # Wave 0 steps match 0 (self-play → 2 slots) and match 1 (vs-cpu → 1 slot)
    # together, so the policy gets one batched call of 3 slots — cross-match
    # plus self-play batching, on real Dolphin.
    assert max(batch_sizes) == 3
