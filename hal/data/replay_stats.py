"""Per-replay aggregate stats: damage / stocks / inputs / death percents.

Computed once per replay when ``build_index`` runs with ``--with-stats``. 1v1
only; returns ``None`` otherwise. Not slippi-js parity (no conversions /
L-cancels / openings); intended as filter fodder.

All frame-level computations operate on rollback-deduplicated columns —
peppi-py emits one row per recorded state including rollback corrections, so
the same `frame_id` can appear 2-3 times with different (or identical) values.
Without deduping, a single death registers as multiple stock decrements; a
single hit can be summed multiple times into damage_dealt.
"""

import dataclasses
import types
from dataclasses import dataclass
from dataclasses import fields
from typing import Any

import numpy as np
from peppi_py.game import Game

from hal.wire import BUTTON_BITS
from hal.wire import VALID_LIBMELEE_PORTS
from hal.wire import dedupe_keep_idx
from hal.wire import peppi_port_to_libmelee

_ALL_BUTTON_BITS: int = 0
for _bit in BUTTON_BITS.values():
    _ALL_BUTTON_BITS |= _bit


@dataclass(frozen=True, slots=True)
class PlayerStats:
    port: int  # libmelee 1..4
    damage_dealt: float
    damage_taken: float
    stocks_remaining: int
    inputs: int
    death_percents: tuple[float, ...]  # percent at the moment of each stock loss, in order

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlayerStats:
        d = dict(data)
        d["death_percents"] = tuple(d.get("death_percents", ()))
        return cls(**d)


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


def _death_percents(percent: Any, stock: Any) -> tuple[float, ...]:
    """Percent at the moment of each stock loss, in chronological order.

    Detects stock decrements only between valid (non-sentinel) values, so
    game-end ``None`` transitions don't register as deaths. The recorded
    percent is from the frame BEFORE the decrement (the decrement frame
    itself has the post-respawn reset value).
    """
    if percent is None or stock is None or len(stock) < 2:
        return ()
    pct = np.asarray(percent.to_pylist(), dtype=np.float64)
    stk_raw = stock.to_pylist()
    stk = np.fromiter((s if s is not None else -1 for s in stk_raw), dtype=np.int32, count=len(stk_raw))
    prev = stk[:-1]
    nxt = stk[1:]
    death_mask = (prev >= 0) & (nxt >= 0) & (nxt < prev)
    death_idx = np.where(death_mask)[0]  # index of frame BEFORE the decrement
    if death_idx.size == 0:
        return ()
    return tuple(float(p) for p in pct[death_idx] if np.isfinite(p))


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


class _MaskedColumn:
    """Pyarrow-shaped view of a column with a row-index mask preapplied."""

    __slots__ = ("_values",)

    def __init__(self, src: Any, keep_idx: np.ndarray) -> None:
        full = src.to_pylist()
        self._values: list[Any] = [full[i] for i in keep_idx]

    def __len__(self) -> int:
        return len(self._values)

    def to_pylist(self) -> list[Any]:
        return list(self._values)


def _mask(src: Any, keep_idx: np.ndarray) -> Any:
    return None if src is None else _MaskedColumn(src, keep_idx)


@dataclass(frozen=True, slots=True)
class _PlayerColumns:
    libmelee_port: int
    pre: Any  # SimpleNamespace with buttons_physical
    post: Any  # SimpleNamespace with percent, stock


def _player_stats(self_cols: _PlayerColumns, opp_cols: _PlayerColumns) -> PlayerStats:
    return PlayerStats(
        port=self_cols.libmelee_port,
        damage_dealt=_percent_gained(opp_cols.post.percent, opp_cols.post.stock),
        damage_taken=_percent_gained(self_cols.post.percent, self_cols.post.stock),
        stocks_remaining=_final_stocks(self_cols.post.stock),
        inputs=_count_input_edges(self_cols.pre.buttons_physical),
        death_percents=_death_percents(self_cols.post.percent, self_cols.post.stock),
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

    keep_idx = dedupe_keep_idx(g.frames.id.to_pylist())

    by_port: list[_PlayerColumns] = []
    for i, pl in enumerate(g.start.players):
        leader = g.frames.ports[i].leader
        assert leader.post.percent is not None, f"peppi returned no percent column for port {pl.port}"
        assert leader.post.stock is not None, f"peppi returned no stock column for port {pl.port}"
        by_port.append(
            _PlayerColumns(
                libmelee_port=peppi_port_to_libmelee(pl.port),
                pre=types.SimpleNamespace(buttons_physical=_mask(leader.pre.buttons_physical, keep_idx)),
                post=types.SimpleNamespace(
                    percent=_mask(leader.post.percent, keep_idx),
                    stock=_mask(leader.post.stock, keep_idx),
                ),
            )
        )
    by_port.sort(key=lambda c: c.libmelee_port)

    a, b = by_port
    return ReplayStats(players=(_player_stats(a, b), _player_stats(b, a)))
