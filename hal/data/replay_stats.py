"""Per-replay aggregate stats: damage / stocks / inputs from one frame pass.

A small set of obvious aggregates — naive percent-delta damage (gated on
stock-reset frames so post-death percent drops don't count), final stocks, and
a button-edge input count — computed once per replay when ``build_index`` runs
with ``--with-stats``. 1v1 only; returns ``None`` otherwise. Not slippi-js
parity (no conversions / L-cancels / openings); intended as filter fodder.
"""

import dataclasses
from dataclasses import dataclass
from dataclasses import fields
from typing import Any

import numpy as np
from peppi_py.game import Game

from hal.wire import BUTTON_BITS
from hal.wire import VALID_LIBMELEE_PORTS
from hal.wire import peppi_port_to_libmelee

_ALL_BUTTON_BITS: int = 0
for _bit in BUTTON_BITS.values():
    _ALL_BUTTON_BITS |= _bit

# Window for crediting an opponent's hit as the killing blow. Generous on
# purpose: knockback trajectories can run 100-200 frames with no new hits
# (meteor + slow fall, off-stage forward-smash), so a tight window
# false-flags those as SDs. Beyond ~3s of no new hits, the death is more
# plausibly self-inflicted.
_KILL_CREDIT_FRAMES: int = 180
# An SD is "early" if it happened this many list-indices after the previous
# death (or game start, for the first stock). 480 ≈ 8 seconds at 60fps —
# in the middle of "5-10s after spawning".
_EARLY_SD_FRAMES: int = 480


@dataclass(frozen=True, slots=True)
class PlayerStats:
    port: int  # libmelee 1..4
    damage_dealt: float
    damage_taken: float
    stocks_remaining: int
    inputs: int
    sds: int
    early_sds: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlayerStats:
        return cls(**data)


@dataclass(frozen=True, slots=True)
class ReplayStats:
    players: tuple[PlayerStats, ...]  # 1v1 only; sorted by ascending port

    def __post_init__(self) -> None:
        if len(self.players) != 2:
            raise ValueError(f"ReplayStats requires exactly 2 players (1v1); got {len(self.players)}")
        ports = [p.port for p in self.players]
        if ports != sorted(ports):
            raise ValueError(f"players must be sorted by port; got {ports}")
        if len(set(ports)) != len(ports):
            raise ValueError(f"duplicate ports: {ports}")
        if any(p not in VALID_LIBMELEE_PORTS for p in ports):
            raise ValueError(f"ports must be in {VALID_LIBMELEE_PORTS}; got {ports}")

    def to_dict(self) -> dict[str, Any]:
        return {"players": [p.to_dict() for p in self.players]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReplayStats:
        return cls(players=tuple(PlayerStats.from_dict(p) for p in data["players"]))


@dataclass(frozen=True, slots=True)
class PlayerStatsMins:
    """Per-player minimums. None = no filter on that field. Any player meeting
    the floor is enough to keep the replay. Requires an index built with
    `index --with-stats`.
    """

    damage_dealt: float | None = None
    damage_taken: float | None = None
    stocks_remaining: int | None = None
    inputs: int | None = None

    def any_set(self) -> bool:
        return any(getattr(self, f.name) is not None for f in fields(self))


@dataclass(frozen=True, slots=True)
class PlayerStatsMaxes:
    """Per-player maximums. None = no filter on that field. ALL players must
    be at or below the ceiling for the replay to pass. Requires an index built
    with `index --with-stats`.
    """

    sds: int | None = None
    early_sds: int | None = None

    def any_set(self) -> bool:
        return any(getattr(self, f.name) is not None for f in fields(self))


def _percent_gained(percent: Any, stock: Any) -> float:
    """Sum of positive percent deltas, dropping stock-change frames.

    Stock-change includes the death frame (percent resets to 0) and any
    None-bounded transition. Used for both `damage_dealt` (passing the
    opponent's columns) and `damage_taken` (passing self's columns).
    The two arrays are pyarrow Arrays or None.
    """
    if percent is None or stock is None or len(percent) < 2:
        return 0.0
    # None float -> NaN propagates through diff; None int -> -1 sentinel forces
    # delta_stk != 0 across the boundary, excluding those frames from the gate.
    pct = np.asarray(percent.to_pylist(), dtype=np.float64)
    stk_raw = stock.to_pylist()
    stk = np.fromiter((s if s is not None else -1 for s in stk_raw), dtype=np.int32, count=len(stk_raw))

    delta_pct = np.diff(pct)
    delta_stk = np.diff(stk)
    keep = (delta_stk == 0) & np.isfinite(delta_pct) & (delta_pct > 0)
    return float(delta_pct[keep].sum())


def _final_stocks(stock: Any) -> int:
    """Last non-None stock value in the column; 0 if none present."""
    if stock is None:
        return 0
    for s in reversed(stock.to_pylist()):
        if s is not None:
            return int(s)
    return 0


def _count_sds(
    percent: Any,
    stock: Any,
    last_hit_by: Any,
    self_port_idx: int,
) -> tuple[int, int]:
    """Return (total_sds, early_sds) for one player.

    A stock loss is an SD when the killing blow can't be credited to an
    opponent: `last_hit_by` points at us, or no positive percent delta
    landed on us within `_KILL_CREDIT_FRAMES` of the death frame.
    Percent-increase is used as the "got hit" signal because it's the only
    one populated on every slp version (older replays lack `hitlag_left`,
    and `last_attack_landed` persists across consecutive hits of the same
    move so its diff misses repeat hits). An SD is "early" if it happens
    within `_EARLY_SD_FRAMES` of the prior death (or frame 0 for stock 1).
    """
    if percent is None or stock is None or last_hit_by is None or len(stock) < 2:
        return 0, 0
    pct = np.asarray(percent.to_pylist(), dtype=np.float64)
    stk = np.fromiter((s if s is not None else -1 for s in stock.to_pylist()), dtype=np.int32, count=len(stock))
    lhb = np.fromiter(
        (h if h is not None else -1 for h in last_hit_by.to_pylist()),
        dtype=np.int32,
        count=len(last_hit_by),
    )

    death_idx = np.where(np.diff(stk) < 0)[0] + 1
    if death_idx.size == 0:
        return 0, 0

    # Hit-on-us = positive percent delta with no stock change on that frame
    # (mirrors the gate in _percent_gained). The hit is recorded at the "after"
    # frame; running max gives the most recent hit frame at every index.
    delta_pct = np.diff(pct)
    delta_stk = np.diff(stk)
    hit_mask = (delta_stk == 0) & np.isfinite(delta_pct) & (delta_pct > 0)
    full_hit_frames = np.full(len(pct), -(10**9), dtype=np.int64)
    full_hit_frames[1:][hit_mask] = np.arange(1, len(pct))[hit_mask]
    last_hit_at = np.maximum.accumulate(full_hit_frames)

    sds = 0
    early_sds = 0
    prev_d = 0
    for d in death_idx:
        d = int(d)
        is_sd = (lhb[d] == self_port_idx) or (d - int(last_hit_at[d]) > _KILL_CREDIT_FRAMES)
        if is_sd:
            sds += 1
            if (d - prev_d) < _EARLY_SD_FRAMES:
                early_sds += 1
        prev_d = d
    return sds, early_sds


def _count_input_edges(buttons_physical: Any) -> int:
    """Total rising edges across BUTTON_BITS on pre.buttons_physical."""
    if buttons_physical is None or len(buttons_physical) < 2:
        return 0
    raw = buttons_physical.to_pylist()
    bits = np.fromiter((v if v is not None else 0 for v in raw), dtype=np.int64, count=len(raw))
    bits &= _ALL_BUTTON_BITS
    edges = bits[1:] & ~bits[:-1]
    # Popcount via uint8 view + unpackbits — portable across numpy versions.
    return int(np.unpackbits(edges.view(np.uint8)).sum())


@dataclass(frozen=True, slots=True)
class _PlayerColumns:
    libmelee_port: int
    peppi_idx: int  # 0..3, matches values found in `last_hit_by`
    pre: Any
    post: Any


def _player_stats(self_cols: _PlayerColumns, opp_cols: _PlayerColumns) -> PlayerStats:
    sds, early_sds = _count_sds(
        self_cols.post.percent,
        self_cols.post.stock,
        self_cols.post.last_hit_by,
        self_cols.peppi_idx,
    )
    return PlayerStats(
        port=self_cols.libmelee_port,
        damage_dealt=_percent_gained(opp_cols.post.percent, opp_cols.post.stock),
        damage_taken=_percent_gained(self_cols.post.percent, self_cols.post.stock),
        stocks_remaining=_final_stocks(self_cols.post.stock),
        inputs=_count_input_edges(self_cols.pre.buttons_physical),
        sds=sds,
        early_sds=early_sds,
    )


def compute_replay_stats(g: Game) -> ReplayStats | None:
    """Compute aggregates from a peppi Game with frames loaded (skip_frames=False).

    Returns None for non-1v1 replays or missing frames. Asserts that
    `post.percent` / `post.stock` are present — peppi guarantees these
    columns post-Slippi-1.0; a missing column is a peppi regression, not
    a per-replay condition to swallow.
    """
    if g.frames is None:
        return None
    if len(g.start.players) != 2:
        return None

    by_port = sorted(
        (
            _PlayerColumns(
                libmelee_port=peppi_port_to_libmelee(pl.port),
                peppi_idx=int(pl.port.value),
                pre=g.frames.ports[i].leader.pre,
                post=g.frames.ports[i].leader.post,
            )
            for i, pl in enumerate(g.start.players)
        ),
        key=lambda c: c.libmelee_port,
    )
    for c in by_port:
        assert c.post.percent is not None, f"peppi returned no percent column for port {c.libmelee_port}"
        assert c.post.stock is not None, f"peppi returned no stock column for port {c.libmelee_port}"

    a, b = by_port
    return ReplayStats(players=(_player_stats(a, b), _player_stats(b, a)))
