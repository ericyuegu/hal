"""Unit tests for hal.eval.behavior — synthetic rows + dev-archive replay."""

import math
from collections.abc import Sequence
from dataclasses import fields
from dataclasses import is_dataclass
from pathlib import Path

import numpy as np
import pytest
from melee import Action
from melee import Character
from melee import Stage

from hal.data.behavior import BLASTZONES
from hal.data.behavior import EDGE_X
from hal.data.behavior import STARTING_STOCKS
from hal.data.behavior import BehaviorFrames
from hal.data.behavior import PlayerBehaviorFrames
from hal.eval.behavior import BehaviorRow
from hal.eval.behavior import analyze_replay
from hal.eval.behavior import behavior_rows
from hal.eval.behavior import read_replay_tolerant
from hal.paths import DEV_ARCHIVE_PATH

_DEATH_AT: tuple[int, int] = (200, 400)
_FRAMES: int = 600


def _zeros(n: int) -> np.ndarray:
    """A fresh zero column; each one is its own array so nothing is shared."""
    return np.zeros(n, dtype=np.float32)


def make_player(
    port: int,
    action: Sequence[int],
    *,
    character: int = int(Character.FOX.value),
    is_cpu: bool = False,
    x: float = 0.0,
    percent: Sequence[float] | None = None,
    stock: Sequence[int] | None = None,
) -> PlayerBehaviorFrames:
    """One port of synthetic frames; every column not named is a benign constant."""
    n = len(action)
    zeros = lambda: np.zeros(n, dtype=np.float32)  # noqa: E731 - one column each, not shared
    return PlayerBehaviorFrames(
        port=port,
        character=character,
        is_cpu=is_cpu,
        cpu_level=9 if is_cpu else 0,
        action=np.asarray(action, dtype=np.int32),
        x=np.full(n, x, dtype=np.float32),
        y=zeros(),
        percent=zeros() if percent is None else np.asarray(percent, dtype=np.float32),
        stock=np.full(n, STARTING_STOCKS, dtype=np.int32) if stock is None else np.asarray(stock, dtype=np.int32),
        direction=np.ones(n, dtype=np.float32),
        airborne=np.zeros(n, dtype=np.int8),
        state_age=zeros(),
        last_attack_landed=np.full(n, -1, dtype=np.int32),
        buttons=np.zeros(n, dtype=np.int32),
        main_stick_x=zeros(),
        main_stick_y=zeros(),
        c_stick_x=zeros(),
        c_stick_y=zeros(),
        trigger_l=zeros(),
        trigger_r=zeros(),
    )


def make_match(*, loser_is_cpu: bool = False) -> BehaviorFrames:
    """Port 1 loses two stocks to the bottom blastzone with nobody near it; port 2
    stands still further from the center for the whole match."""
    action = [Action.STANDING.value] * _FRAMES
    stock = [STARTING_STOCKS] * _FRAMES
    for i, at in enumerate(_DEATH_AT):
        action[at : at + 30] = [Action.DEAD_DOWN.value] * 30
        stock[at:] = [STARTING_STOCKS - i - 1] * (_FRAMES - at)
    loser = make_player(1, action, x=40.0, stock=stock, is_cpu=loser_is_cpu)
    winner = make_player(2, [Action.STANDING.value] * _FRAMES, x=60.0)
    return BehaviorFrames(
        stage=Stage.FINAL_DESTINATION,
        edge_x=EDGE_X[Stage.FINAL_DESTINATION],
        blastzones=BLASTZONES[Stage.FINAL_DESTINATION],
        frame_id=np.arange(_FRAMES, dtype=np.int64),
        players=(loser, winner),
    )


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------


def test_rows_are_one_per_player_in_port_order() -> None:
    loser, winner = behavior_rows(make_match(), names={1: "alpha", 2: "beta"})
    assert (loser.port, winner.port) == (1, 2)
    assert (loser.model, loser.opp_model) == ("alpha", "beta")
    assert (winner.model, winner.opp_model) == ("beta", "alpha")
    assert (loser.character, loser.stage) == ("FOX", "FINAL_DESTINATION")


def test_outcome_mirrors_between_the_two_rows() -> None:
    loser, winner = behavior_rows(make_match())
    assert (loser.outcome.stocks_left, winner.outcome.stocks_left) == (2, 4)
    assert (loser.outcome.stocks_lost, loser.outcome.stocks_taken) == (2, 0)
    assert (winner.outcome.stocks_lost, winner.outcome.stocks_taken) == (0, 2)
    assert loser.outcome.stock_delta == -winner.outcome.stock_delta == -2
    assert (loser.outcome.won, winner.outcome.won) == (False, True)
    assert loser.outcome.frames_total == _FRAMES
    assert 0 < loser.outcome.frames_active < _FRAMES  # the death animations are not active
    assert loser.outcome.minutes == pytest.approx(loser.outcome.frames_active / 3600.0)


def test_deaths_account_for_every_lost_stock() -> None:
    for row in behavior_rows(make_match()):
        assert row.death_taxonomy.deaths == STARTING_STOCKS - row.outcome.stocks_left
    loser, _ = behavior_rows(make_match())
    assert loser.death_taxonomy.deaths_bottom == len(_DEATH_AT)
    # The slippi-parity pair reads the same stock losses from the punish machine.
    assert loser.death_taxonomy.deaths_down == len(_DEATH_AT)
    assert loser.death_taxonomy.sd_count == len(_DEATH_AT)
    assert loser.death_taxonomy.deaths_sd_like == len(_DEATH_AT)


def test_center_control_shares_sum_to_at_most_one() -> None:
    loser, winner = behavior_rows(make_match())
    assert loser.positioning.center_control_frac == pytest.approx(1.0)
    assert winner.positioning.center_control_frac == pytest.approx(0.0)
    assert loser.positioning.center_control_frac + winner.positioning.center_control_frac <= 1.0


def test_names_fall_back_to_the_port_and_to_cpu() -> None:
    loser, winner = behavior_rows(make_match(loser_is_cpu=True))
    assert (loser.model, loser.is_cpu, loser.cpu_level) == ("cpu", True, 9)
    assert (winner.model, winner.opp_model) == ("port2", "cpu")
    # A supplied name always wins over the replay's own typing.
    named, _ = behavior_rows(make_match(loser_is_cpu=True), names={1: "alpha"})
    assert named.model == "alpha"


def test_undefined_ratios_are_nan_not_zero() -> None:
    """Nobody landed a hit, so every rate over openings or kills is undefined."""
    loser, _ = behavior_rows(make_match())
    stats = loser.conversion_stats
    assert stats.conversions == 0
    assert math.isnan(stats.damage_per_opening)
    assert math.isnan(stats.openings_per_kill)
    assert math.isnan(stats.neutral_win_ratio)


# ---------------------------------------------------------------------------
# Flat dict
# ---------------------------------------------------------------------------


def _components(row: BehaviorRow) -> list[object]:
    return [row] + [getattr(row, f.name) for f in fields(row) if is_dataclass(getattr(row, f.name))]


def test_flat_dict_is_one_level_with_unique_keys() -> None:
    row, _ = behavior_rows(make_match(), names={1: "alpha", 2: "beta"})
    flat = row.as_flat_dict()
    expected = sum(len(fields(getattr(row, f.name))) if is_dataclass(getattr(row, f.name)) else 1 for f in fields(row))
    assert len(flat) == expected
    assert not any(is_dataclass(value) for value in flat.values())


def test_flat_dict_keeps_every_value_under_its_own_field_name() -> None:
    row, _ = behavior_rows(make_match(), names={1: "alpha", 2: "beta"})
    for key, value in row.as_flat_dict().items():
        # Exactly one component owns the key, so no column silently shadows another.
        (owned,) = [getattr(c, key) for c in _components(row) if hasattr(c, key)]
        assert owned == value or (isinstance(value, float) and math.isnan(value) and math.isnan(owned))


def test_flat_dict_starts_with_the_row_identity() -> None:
    row, _ = behavior_rows(make_match(), names={1: "alpha", 2: "beta"})
    assert list(row.as_flat_dict())[:4] == ["replay", "model", "opp_model", "port"]


# ---------------------------------------------------------------------------
# Fixture: a real replay end to end
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not Path(DEV_ARCHIVE_PATH).exists(),
    reason=f"dev archive missing at {DEV_ARCHIVE_PATH}; run `python -m hal.scripts.fetch --name dev.7z`",
)
def test_analyze_replay_on_fixture(tmp_path: Path) -> None:
    """Rows from a real .slp, plus the invariants that must hold on any match."""
    replay = _fixture_replay(tmp_path)
    rows = analyze_replay(replay)
    assert rows is not None
    assert [row.port for row in rows] == [1, 2]

    for row in rows:
        assert row.replay == str(replay)
        assert row.death_taxonomy.deaths == STARTING_STOCKS - row.outcome.stocks_left
        assert row.outcome.frames_active > 0
        assert row.outcome.damage_dealt >= 0.0 and row.outcome.damage_taken >= 0.0
        for value in (
            row.positioning.center_control_frac,
            row.positioning.frac_offstage,
            row.positioning.frac_near_ledge_onstage,
            row.movement.idle_frac,
            row.conversion_stats.neutral_win_ratio,
            row.conversion_stats.counter_hit_ratio,
        ):
            assert math.isnan(value) or 0.0 <= value <= 1.0
        assert row.conversion_stats.kills <= row.conversion_stats.conversions
        assert row.death_taxonomy.sd_count <= row.death_taxonomy.deaths

    first, second = rows
    assert first.outcome.stock_delta == -second.outcome.stock_delta
    assert first.outcome.damage_dealt == pytest.approx(second.outcome.damage_taken)
    control = [row.positioning.center_control_frac for row in rows]
    assert sum(control) <= 1.0 + 1e-6
    assert len(set(rows[0].as_flat_dict())) == len(rows[0].as_flat_dict())


@pytest.mark.skipif(
    not Path(DEV_ARCHIVE_PATH).exists(),
    reason=f"dev archive missing at {DEV_ARCHIVE_PATH}; run `python -m hal.scripts.fetch --name dev.7z`",
)
def test_read_replay_tolerant_repairs_a_torn_file(tmp_path: Path) -> None:
    """A replay cut off mid-frame reads only after the trim, and yields rows."""
    import peppi_py

    replay = _fixture_replay(tmp_path)
    torn = tmp_path / "torn.slp"
    torn.write_bytes(replay.read_bytes()[:-500])
    with pytest.raises(BaseException):  # noqa: B017 - peppi raises OSError or PanicException
        peppi_py.read_slippi(str(torn), skip_frames=False)

    assert read_replay_tolerant(torn) is not None
    assert torn.stat().st_size < replay.stat().st_size
    rows = analyze_replay(torn)
    assert rows is not None
    assert rows[0].outcome.frames_active > 0


@pytest.mark.skipif(
    not Path(DEV_ARCHIVE_PATH).exists(),
    reason=f"dev archive missing at {DEV_ARCHIVE_PATH}; run `python -m hal.scripts.fetch --name dev.7z`",
)
def test_analyze_replay_drops_a_match_below_the_frame_floor(tmp_path: Path) -> None:
    assert analyze_replay(_fixture_replay(tmp_path), min_active_frames=10**9) is None


def _fixture_replay(tmp_path: Path) -> Path:
    """The first .slp of the dev archive, extracted under ``tmp_path``."""
    import py7zr

    with py7zr.SevenZipFile(DEV_ARCHIVE_PATH, "r") as z:
        members = [m for m in z.getnames() if m.endswith(".slp")]
        assert members
        z.extract(path=tmp_path, targets=[members[0]])
    return tmp_path / members[0]
