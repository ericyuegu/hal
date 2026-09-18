"""Incremental model contexts shared by training evaluation and portable inference."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
from operator import itemgetter
from typing import Literal

import numpy as np
import torch

from hal.data.feature_stats import FeatureStats
from hal.training.ego_stats import consolidate_key
from hal.training.features import ACTION_CHANNELS
from hal.training.features import NO_EXTRA_COLUMNS
from hal.training.features import SPATIAL_COLUMNS
from hal.training.features import SPATIAL_GATE_COLUMN
from hal.training.features import SPATIAL_INPUT_COLUMNS
from hal.training.features import ExtraColumns
from hal.training.features import FeatureProjection
from hal.training.features import derive_spatial
from hal.training.features import feature_kind
from hal.training.features import float_feature_transform
from hal.training.features import mask_sentinel_positions

# A frame lands in two raw scratch rows, one per dtype. Which row a column takes is
# decided by the Python type of its value on the slot's first frame (int -> int32,
# anything else -> float32) — the rule that also picks its mask sentinel.
_RAW_DTYPES: tuple[np.dtype, ...] = (np.dtype(np.float32), np.dtype(np.int32))

# Column order inside a raw row: normalized floats grouped by transform, then the raw
# stick/trigger/button channels, then the categoricals. Grouping by transform is what
# makes every run a contiguous slice, hence ONE vectorized op per run.
_TRANSFORM_ORDER: tuple[str, ...] = ("standardize", "minmax", "zero", "raw", "cat")

# Derived spatial columns that are a finite difference. The window's FIRST position has
# no predecessor inside the window, so ``derive_spatial`` zeroes and flags it there. The
# ring stores the true delta against the real previous frame; the window read re-applies
# that rule, which is what keeps a cold start, an instant-restart seam and a saturated
# context on the same alignment.
_DPOS_COLUMNS: tuple[str, ...] = tuple(c for c in SPATIAL_COLUMNS if c.endswith(("_dpos_x", "_dpos_y")))
_DPOS_MASK: Literal["spatial_dpos_mask"] = "spatial_dpos_mask"


@dataclass(frozen=True, slots=True)
class _Run:
    """One contiguous column run sharing a raw source row and a transform."""

    src: int  # index into the (float32, int32) raw row pair
    src_at: slice
    dst_at: slice
    transform: str
    a: np.ndarray  # mean (standardize) / min (minmax); empty otherwise
    b: np.ndarray  # std (standardize) / max - min (minmax); empty otherwise


@dataclass(frozen=True, slots=True)
class _Layout:
    """One slot's resolution of ``preprocess``'s routing into vectorized column runs.

    ``value_runs`` and ``cat_runs`` produce the model's float32 and int64 columns;
    ``mask_runs`` produces the per-float validity sidecars. ``spatial_at`` is where
    the derived block lands inside a value row, or ``None`` when the observation
    carries no ``stage`` column (i.e. predates matchup conditioning).

    Column ORDER is a function of the routed names only, never of which port is the
    ego, so two slots on opposite ports of one match share every index here and one
    slot's layout can address the whole batch.
    """

    gamestate_getters: tuple[Callable[[Mapping[str, float | int]], tuple] | None, ...]
    gamestate_at: tuple[slice, ...]
    action_at: tuple[slice, ...]
    action_channels: tuple[np.ndarray, ...]  # ACTION_CHANNELS indices feeding each action_at
    action_is_button: tuple[bool, ...]
    raw_widths: tuple[int, ...]
    value_runs: tuple[_Run, ...]
    cat_runs: tuple[_Run, ...]
    mask_runs: tuple[_Run, ...]
    value_names: tuple[str, ...]
    cat_names: tuple[str, ...]
    mask_names: tuple[str, ...]
    spatial_at: slice | None
    spatial_sources: tuple[tuple[int, int], ...]  # (raw row, index in it) per SPATIAL_INPUT_COLUMNS entry
    dpos_rows: np.ndarray  # value rows the window read zeroes at position 0
    dpos_mask_row: int  # value row the window read flags at position 0; -1 = no spatial block
    zero_value: np.ndarray
    zero_cat: np.ndarray
    zero_mask: np.ndarray


# %%
# --- layout resolution --------------------------------------------------------


def _column_transform(name: str, kind: str, stats: dict[str, FeatureStats], extra: ExtraColumns) -> str:
    """Which of :data:`_TRANSFORM_ORDER` one routed column takes. Degenerate stats
    (zero spread) collapse to ``zero``, matching ``_standardize`` / ``_normalize``."""
    if kind == "cat":
        return "cat"
    if kind in ("button", "stick_trigger"):
        return "raw"
    s = stats[consolidate_key(name)]
    if float_feature_transform(name, extra) == "standardize":
        return "zero" if s.std == 0 else "standardize"
    return "zero" if s.max == s.min else "minmax"


def _run_constants(block: list[str], transform: str, stats: dict[str, FeatureStats]) -> tuple[np.ndarray, np.ndarray]:
    """``(a, b)`` constants for one run, held in float32 so the per-frame arithmetic
    is the same float32 arithmetic ``preprocess`` does on a float32 column."""
    if transform not in ("standardize", "minmax"):
        empty = np.empty(0, dtype=np.float32)
        return empty, empty
    entries = [stats[consolidate_key(name)] for name in block]
    if transform == "standardize":
        a = [s.mean for s in entries]
        b = [s.std for s in entries]
    else:
        a = [s.min for s in entries]
        b = [s.max - s.min for s in entries]
    return np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)


def _runs(
    columns: list[tuple[str, int, str]], stats: dict[str, FeatureStats], keep: tuple[str, ...]
) -> tuple[tuple[_Run, ...], tuple[str, ...]]:
    """Merge consecutive same-source, same-transform columns into one run each.

    ``columns`` is ``(name, raw row, transform)`` in raw-row order and ``keep`` selects
    the transforms this destination block carries; destination rows are numbered in the
    order the surviving columns appear.
    """
    row_index = _row_indices(columns)
    runs: list[_Run] = []
    names = [name for name, _, transform in columns if transform in keep]
    dst = 0
    at = 0
    while at < len(columns):
        _, src, transform = columns[at]
        if transform not in keep:
            at += 1
            continue
        stop = at
        while stop + 1 < len(columns) and columns[stop + 1][1] == src and columns[stop + 1][2] == transform:
            stop += 1
        width = stop - at + 1
        a, b = _run_constants([columns[i][0] for i in range(at, stop + 1)], transform, stats)
        runs.append(
            _Run(
                src=src,
                src_at=slice(row_index[at], row_index[at] + width),
                dst_at=slice(dst, dst + width),
                transform=transform,
                a=a,
                b=b,
            )
        )
        dst += width
        at = stop + 1
    return tuple(runs), tuple(names)


def _row_indices(columns: list[tuple[str, int, str]]) -> list[int]:
    """Each column's index inside its own raw row."""
    seen = [0] * len(_RAW_DTYPES)
    out: list[int] = []
    for _, src, _ in columns:
        out.append(seen[src])
        seen[src] += 1
    return out


def _model_name(key: str, ego_prefix: str) -> str:
    """Raw frame key → the ``ego_``/``opp_`` name the model reads (see ``relabel_ego``)."""
    opp_prefix = "p2" if ego_prefix == "p1" else "p1"
    if key.startswith(f"{ego_prefix}_"):
        return f"ego_{key[3:]}"
    if key.startswith(f"{opp_prefix}_"):
        return f"opp_{key[3:]}"
    return key


def _getter(names: list[str]) -> Callable[[Mapping[str, float | int]], tuple] | None:
    """``itemgetter`` over ``names`` that always answers with a tuple. A key that
    disappears mid-match raises ``KeyError`` here, as reading the column did before."""
    if not names:
        return None
    if len(names) == 1:
        single = itemgetter(names[0])
        return lambda frame: (single(frame),)
    return itemgetter(*names)


def _build_layout(
    flat: Mapping[str, float | int],
    ego_prefix: str,
    stats: dict[str, FeatureStats],
    extra: ExtraColumns | None,
    projection: FeatureProjection | None = None,
) -> _Layout:
    """Resolve one slot's routing from its first observed frame.

    Column dtypes and the surviving key set come from that frame, the same rule the
    window builder applied to the first row of its buffer.
    """
    routing = NO_EXTRA_COLUMNS if extra is None else extra
    raw_key: dict[str, str] = {}
    gamestate: list[tuple[str, int, str]] = []
    for key, value in flat.items():
        if key == "frame":
            continue
        name = _model_name(key, ego_prefix)
        if projection is not None and name not in projection.columns:
            continue
        kind = feature_kind(name, routing)
        if kind == "derived":
            raise ValueError(
                f"{name!r} arrived as an input column, but the spatial block is derived on the fly by "
                "derive_spatial; materializing it into the MDS needs a schema bump and this derivation removed"
            )
        if kind == "drop":
            continue
        raw_key[name] = key
        gamestate.append((name, 1 if isinstance(value, int) else 0, _column_transform(name, kind, stats, routing)))

    action: list[tuple[str, int, str]] = []
    channel_of: dict[str, int] = {}
    for i, channel in enumerate(ACTION_CHANNELS):
        name = f"ego_{channel}"
        kind = feature_kind(name, routing)
        if kind not in ("button", "stick_trigger"):
            raise ValueError(f"action channel {name!r} routes as {kind!r}; expected a raw controller channel")
        action.append((name, 1 if kind == "button" else 0, "raw"))
        channel_of[name] = i
    shadowed = sorted(set(channel_of) & set(raw_key))
    if shadowed:
        raise ValueError(f"observation columns {shadowed} collide with the ego action channels of the same name")

    # Sorting by (transform, name) keeps the order independent of which port is the ego.
    rank = {transform: i for i, transform in enumerate(_TRANSFORM_ORDER)}
    ordered = sorted(gamestate, key=lambda c: (rank[c[2]], c[0])) + sorted(action, key=lambda c: rank[c[2]])

    getters: list[Callable[[Mapping[str, float | int]], tuple] | None] = []
    gamestate_at: list[slice] = []
    action_at: list[slice] = []
    action_channels: list[np.ndarray] = []
    action_is_button: list[bool] = []
    raw_widths: list[int] = []
    for src in range(len(_RAW_DTYPES)):
        row = [name for name, source, _ in ordered if source == src]
        n_gamestate = sum(1 for name in row if name not in channel_of)
        raw_widths.append(len(row))
        getters.append(_getter([raw_key[name] for name in row[:n_gamestate]]))
        gamestate_at.append(slice(0, n_gamestate))
        action_at.append(slice(n_gamestate, len(row)))
        action_channels.append(np.array([channel_of[name] for name in row[n_gamestate:]], dtype=np.intp))
        action_is_button.append(all(name.startswith("ego_button_") for name in row[n_gamestate:]))

    value_runs, value_names = _runs(ordered, stats, ("standardize", "minmax", "zero", "raw"))
    cat_runs, cat_names = _runs(ordered, stats, ("cat",))
    mask_runs, float_names = _runs(ordered, stats, ("standardize", "minmax", "zero"))

    spatial_at: slice | None = None
    spatial_sources: tuple[tuple[int, int], ...] = ()
    dpos_rows = np.empty(0, dtype=np.intp)
    dpos_mask_row = -1
    if SPATIAL_GATE_COLUMN in flat and (projection is None or projection.derive_spatial):
        row_index = _row_indices(ordered)
        located = {name: (src, row_index[at]) for at, (name, src, _) in enumerate(ordered)}
        missing = [name for name in SPATIAL_INPUT_COLUMNS if name not in located]
        if missing:
            raise ValueError(f"derive_spatial needs raw columns {missing}, which the observation does not carry")
        spatial_sources = tuple(located[name] for name in SPATIAL_INPUT_COLUMNS)
        spatial_at = slice(len(value_names), len(value_names) + len(SPATIAL_COLUMNS))
        value_names = value_names + SPATIAL_COLUMNS
        dpos_rows = np.array([value_names.index(name) for name in _DPOS_COLUMNS], dtype=np.intp)
        dpos_mask_row = value_names.index(_DPOS_MASK)

    layout = _Layout(
        gamestate_getters=tuple(getters),
        gamestate_at=tuple(gamestate_at),
        action_at=tuple(action_at),
        action_channels=tuple(action_channels),
        action_is_button=tuple(action_is_button),
        raw_widths=tuple(raw_widths),
        value_runs=value_runs,
        cat_runs=cat_runs,
        mask_runs=mask_runs,
        value_names=value_names,
        cat_names=cat_names,
        mask_names=tuple(f"{name}_mask" for name in float_names),
        spatial_at=spatial_at,
        spatial_sources=spatial_sources,
        dpos_rows=dpos_rows,
        dpos_mask_row=dpos_mask_row,
        zero_value=np.zeros(len(value_names), dtype=np.float32),
        zero_cat=np.zeros(len(cat_names), dtype=np.int64),
        zero_mask=np.zeros(len(float_names), dtype=np.float32),
    )
    return replace(layout, **_zero_rows(layout))


def _zero_rows(layout: _Layout) -> dict[str, np.ndarray]:
    """The preprocessed all-zero row — what a not-yet-observed context position holds.

    It is NOT zeros: a standardized column maps raw 0 to ``-mean/std``, which is
    exactly what the left-padded window produced once it went through ``preprocess``.
    """
    raw = _empty_raw(layout)
    masks = tuple(mask_sentinel_positions(row) for row in raw)
    value = np.zeros(len(layout.value_names), dtype=np.float32)
    cat = np.zeros(len(layout.cat_names), dtype=np.int64)
    mask = np.zeros(len(layout.mask_names), dtype=np.float32)
    _write_value_row(layout, raw, masks, value)
    _write_cat_row(layout, raw, masks, cat)
    _write_mask_row(layout, masks, mask)
    if layout.spatial_at is not None:
        value[layout.spatial_at] = _spatial_block(layout, [(raw, raw)])[:, 0]
    return {"zero_value": value, "zero_cat": cat, "zero_mask": mask}


def _empty_raw(layout: _Layout) -> tuple[np.ndarray, ...]:
    return tuple(np.zeros(w, dtype=d) for w, d in zip(layout.raw_widths, _RAW_DTYPES, strict=True))


# %%
# --- per-frame row writers ----------------------------------------------------


def _write_value_row(
    layout: _Layout, raw: tuple[np.ndarray, ...], masks: tuple[np.ndarray, ...], out: np.ndarray
) -> None:
    for run in layout.value_runs:
        src = raw[run.src][run.src_at]
        if run.transform == "standardize":
            x = (src - run.a) / run.b
        elif run.transform == "minmax":
            x = 2.0 * (src - run.a) / run.b - 1.0
        elif run.transform == "zero":
            x = np.zeros(src.shape, dtype=np.float32)
        else:
            x = src
        out[run.dst_at] = np.where(masks[run.src][run.src_at], 0.0, x)


def _write_cat_row(
    layout: _Layout, raw: tuple[np.ndarray, ...], masks: tuple[np.ndarray, ...], out: np.ndarray
) -> None:
    for run in layout.cat_runs:
        out[run.dst_at] = np.where(masks[run.src][run.src_at], 0, raw[run.src][run.src_at]).astype(np.int64)


def _write_mask_row(layout: _Layout, masks: tuple[np.ndarray, ...], out: np.ndarray) -> None:
    for run in layout.mask_runs:
        out[run.dst_at] = masks[run.src][run.src_at].astype(np.float32)


def _spatial_block(layout: _Layout, pairs: list[tuple[tuple[np.ndarray, ...], ...]]) -> np.ndarray:
    """The derived block for this frame of every slot → ``[len(SPATIAL_COLUMNS), n]``.

    ``pairs`` gives each slot's ``(previous raw rows, current raw rows)``. The
    derivation runs on that two-frame batch so a finite difference reads its true
    predecessor. A slot with no predecessor — cold start, or the frame right after an
    instant restart — pairs against the all-zero row, whose ``Stage.NO_STAGE`` marks
    the delta invalid: the same rule the zero-filled window pad relied on.

    Batched over slots because ``derive_spatial``'s cost is per call, not per element.
    """
    batch: dict[str, np.ndarray] = {}
    for name, (src, index) in zip(SPATIAL_INPUT_COLUMNS, layout.spatial_sources, strict=True):
        column = np.empty((len(pairs), 2), dtype=_RAW_DTYPES[src])
        for j, (prev, cur) in enumerate(pairs):
            column[j, 0] = prev[src][index]
            column[j, 1] = cur[src][index]
        batch[name] = column
    derived = derive_spatial(batch)
    return np.stack([derived[name][:, 1] for name in SPATIAL_COLUMNS])


# %%
class ContextHistory:
    """One slot's preprocessed context rows, in 2x-capacity mirror ring buffers.

    Every row is written at ``head`` and at ``head + L``, so the window of the last
    ``n <= L`` rows is the single contiguous slice ``[head + L - n, head + L)`` — no
    wrap-around branch and no rebuild to read it. Positions not yet written since the
    last reset hold the preprocessed all-zero row, which IS the left pad the model
    then hides through ``ctx_pad``.
    """

    __slots__ = ("layout", "L", "values", "cats", "masks", "raw", "prev", "written", "_value", "_cat", "_mask")

    def __init__(self, layout: _Layout, L: int) -> None:
        self.layout = layout
        self.L = L
        self.values = np.repeat(layout.zero_value[:, None], 2 * L, axis=1)
        self.cats = np.repeat(layout.zero_cat[:, None], 2 * L, axis=1)
        self.masks = np.repeat(layout.zero_mask[:, None], 2 * L, axis=1)
        self.raw = _empty_raw(layout)
        self.prev = _empty_raw(layout)
        self.written = 0
        self._value = np.empty(len(layout.value_names), dtype=np.float32)
        self._cat = np.empty(len(layout.cat_names), dtype=np.int64)
        self._mask = np.empty(len(layout.mask_names), dtype=np.float32)

    @classmethod
    def from_frame(
        cls,
        flat: Mapping[str, float | int],
        ego_prefix: str,
        stats: dict[str, FeatureStats],
        length: int,
        extra: ExtraColumns | None = None,
        projection: FeatureProjection | None = None,
    ) -> ContextHistory:
        return cls(_build_layout(flat, ego_prefix, stats, extra, projection), length)

    @property
    def count(self) -> int:
        """Real frames currently in context; caps at ``L``."""
        return min(self.written, self.L)

    def gather(self, flat: Mapping[str, float | int], action: np.ndarray) -> None:
        """Read one frame plus the ego action that produced it into the raw scratch row."""
        self.prev, self.raw = self.raw, self.prev
        layout = self.layout
        for src, row in enumerate(self.raw):
            getter = layout.gamestate_getters[src]
            if getter is not None:
                row[layout.gamestate_at[src]] = getter(flat)
            channels = layout.action_channels[src]
            if channels.size:
                values = action[channels]
                row[layout.action_at[src]] = values > 0.5 if layout.action_is_button[src] else values

    def push(self, spatial: np.ndarray | None) -> None:
        """Preprocess the gathered raw row and write it into every ring, twice."""
        layout = self.layout
        masks = tuple(mask_sentinel_positions(row) for row in self.raw)
        _write_value_row(layout, self.raw, masks, self._value)
        _write_cat_row(layout, self.raw, masks, self._cat)
        _write_mask_row(layout, masks, self._mask)
        if layout.spatial_at is not None:
            if spatial is None:
                raise RuntimeError("this slot's layout carries a derived spatial block, but none was supplied")
            self._value[layout.spatial_at] = spatial
        at = self.written % self.L
        for ring, row in ((self.values, self._value), (self.cats, self._cat), (self.masks, self._mask)):
            ring[:, at] = row
            ring[:, at + self.L] = row
        self.written += 1

    def window(self, n: int) -> slice:
        """Column slice holding the last ``n`` rows, oldest first."""
        head = self.written % self.L
        return slice(head + self.L - n, head + self.L)


@dataclass(frozen=True, slots=True)
class ContextWindows:
    """One replan's stacked context, packed by dtype for a single host→device copy.

    ``floats`` is ``[n_value + n_mask, B, L]``: the model's float columns, then every
    per-float validity sidecar. ``emitted`` selects the sidecars the batch carries —
    ``preprocess`` emits ``{name}_mask`` only where a mask fires, and a model reads an
    absent sidecar as zeros, so the two must agree on which fired.
    """

    layout: _Layout
    floats: np.ndarray
    cats: np.ndarray
    emitted: np.ndarray

    def features(
        self,
        device: str | torch.device,
        float_dtype: torch.dtype = torch.float32,
        *,
        all_masks: bool = False,
    ) -> dict[str, torch.Tensor]:
        layout = self.layout
        n_value = len(layout.value_names)
        packed = torch.from_numpy(self.floats).to(device, float_dtype)
        feats: dict[str, torch.Tensor] = dict(zip(layout.value_names, packed[:n_value].unbind(0), strict=True))
        masks = range(len(layout.mask_names)) if all_masks else np.flatnonzero(self.emitted)
        feats.update({layout.mask_names[k]: packed[n_value + k] for k in masks})
        if layout.cat_names:
            cats = torch.from_numpy(self.cats).to(device)
            feats.update(zip(layout.cat_names, cats.unbind(0), strict=True))
        return feats


def push_context_rows(rings: Sequence[ContextHistory]) -> None:
    if rings and rings[0].layout.spatial_at is not None:
        spatial = _spatial_block(rings[0].layout, [(ring.prev, ring.raw) for ring in rings])
        for j, ring in enumerate(rings):
            ring.push(spatial[:, j])
    else:
        for ring in rings:
            ring.push(None)


def stack_context_windows(
    rings: Sequence[ContextHistory], length: int, *, truncate_left_edge: bool = True
) -> ContextWindows:
    layout = rings[0].layout
    n_value, n_mask = len(layout.value_names), len(layout.mask_names)
    floats = np.empty((n_value + n_mask, len(rings), length), dtype=np.float32)
    cats = np.empty((len(layout.cat_names), len(rings), length), dtype=np.int64)
    for j, ring in enumerate(rings):
        at = ring.window(length)
        floats[:n_value, j] = ring.values[:, at]
        floats[n_value:, j] = ring.masks[:, at]
        cats[:, j] = ring.cats[:, at]
    # A full window has lost the first row's predecessor.
    if truncate_left_edge and layout.dpos_mask_row >= 0:
        floats[layout.dpos_rows, :, 0] = 0.0
        floats[layout.dpos_mask_row, :, 0] = 1.0
    return ContextWindows(layout=layout, floats=floats, cats=cats, emitted=floats[n_value:].any(axis=(1, 2)))
