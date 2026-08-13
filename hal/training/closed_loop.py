"""Generic receding-horizon closed-loop policy.

``RecedingHorizon`` is the torch-side ``BatchPolicy`` (see ``hal.sim.vec``) that
adapts any action-chunk model to the vectorized eval driver. It owns every part
of closed-loop play that is *invariant* across model architectures:

* per-slot rolling context (observed gamestate + the ego's own intended actions),
  capped at ``L_ctx`` and cleared at each instant-restart match boundary so a
  slot's context never spans two matches;
* the cold-start left-pad + alignment that lets the policy act from frame 0 while
  the context fills with real gameplay (reported as ``ctx_pad`` so the model masks
  the not-yet-filled prefix from attention);
* a per-slot replan clock — replan each slot every ``s`` frames (the execution
  horizon) and execute the chunk's first ``s`` actions, where ``s == L_chunk`` is
  plain open-loop; an instant restart resets that slot's clock and pending chunk;
* the real-time-chunking commitment: when the inference delay ``d > 0``, each new
  chunk is conditioned on the ``d`` actions already committed for its first frames
  (the previous chunk's ``[s : s+d]``; ``None`` at bootstrap), so the handoff is
  continuous (constraint ``d <= L_chunk - s``);
* stacking every live slot into one batch → :class:`Context`, and scattering the
  predicted chunks back.

The single *variant* — how a chunk is produced from a :class:`Context` and the
committed prefix — is injected as ``predict_chunk``. That closure is the only
thing that touches the model, so this class never imports a specific architecture.

Context storage
---------------
Normalization is frame-local, so each incoming frame is flattened and preprocessed
EXACTLY ONCE and the resulting row is written into per-slot ring buffers
(:class:`_Rings`). Each ring holds 2x capacity and each row is written twice — at
``head`` and at ``head + L_ctx`` — so any window read is ONE contiguous slice: no
per-frame Python loop over features, no window rebuild, no wrap-around branch.

:class:`_Layout` resolves the column routing of
``hal.training.features.preprocess`` once per slot into vectorized column runs, so
a frame costs a fixed handful of numpy calls instead of one per column.
``tests/test_closed_loop_rings.py`` drives this builder and ``preprocess`` over the
same frame stream and pins them tensor for tensor — that test is the contract that
keeps the two in step.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from operator import itemgetter
from typing import Literal

import numpy as np
import torch

from hal.data.feature_stats import FeatureStats
from hal.sim.inputs import ControllerInputs
from hal.sim.inputs import action_vec_to_controller
from hal.sim.rollout import ObservationRow
from hal.sim.rollout import PolicyRuntimeSpec
from hal.sim.vec import Slot
from hal.training.canonical import flatten_canonical_frame
from hal.training.ego_stats import consolidate_key

# Column routing has one home. The ring builder resolves it per slot rather than per
# window, so it reads the same routing helpers ``preprocess`` uses instead of
# restating the rules here.
from hal.training.features import _NO_EXTRA
from hal.training.features import _SPATIAL_GATE
from hal.training.features import _SPATIAL_INPUTS
from hal.training.features import ACTION_CHANNELS
from hal.training.features import NEUTRAL_ACTION
from hal.training.features import SPATIAL_COLUMNS
from hal.training.features import Context
from hal.training.features import ExtraColumns
from hal.training.features import FeatureProjection
from hal.training.features import _classify
from hal.training.features import _float_transform
from hal.training.features import _is_masked
from hal.training.features import derive_spatial

# A bound model + integration scheme: (Context, committed-action prefix or None)
# → predicted action chunks ``[n_live, L_chunk, d_action]`` (numpy, for the
# rolling-buffer plumbing). ``committed`` is ``[n_live, d, d_action]`` — the
# already-locked actions the new chunk's prefix is conditioned on (``None`` when
# ``d == 0`` or at bootstrap).
PredictChunk = Callable[[Context, np.ndarray | None], np.ndarray]

_PORT_TO_PREFIX: dict[int, Literal["p1", "p2"]] = {1: "p1", 2: "p2"}

# SlippiVec exposes a complete schema-v5 MDS row, including the controller inputs
# recorded for both ports.  Closed-loop inference deliberately does not consume
# those columns: its action token is the policy action that produced the incoming
# post-frame observation (neutral at bootstrap), which is tracked independently
# below.  Dropping the recorded controller block therefore gives the flat native
# row the same feature surface and alignment as ``flatten_canonical_frame``.
_FLAT_CONTROLLER_COLUMNS = frozenset(
    f"{prefix}_{suffix}" for prefix in ("p1", "p2") for suffix in (*ACTION_CHANNELS, "button_start")
)

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
    spatial_sources: tuple[tuple[int, int], ...]  # (raw row, index in it) per _SPATIAL_INPUTS entry
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
    if _float_transform(name, extra) == "standardize":
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
    routing = _NO_EXTRA if extra is None else extra
    raw_key: dict[str, str] = {}
    gamestate: list[tuple[str, int, str]] = []
    for key, value in flat.items():
        if key == "frame":
            continue
        name = _model_name(key, ego_prefix)
        if projection is not None and name not in projection.columns:
            continue
        kind = _classify(name, routing)
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
        kind = _classify(name, routing)
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
    if _SPATIAL_GATE in flat and (projection is None or projection.derive_spatial):
        row_index = _row_indices(ordered)
        located = {name: (src, row_index[at]) for at, (name, src, _) in enumerate(ordered)}
        missing = [name for name in _SPATIAL_INPUTS if name not in located]
        if missing:
            raise ValueError(f"derive_spatial needs raw columns {missing}, which the observation does not carry")
        spatial_sources = tuple(located[name] for name in _SPATIAL_INPUTS)
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
    masks = tuple(_is_masked(row) for row in raw)
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
    for name, (src, index) in zip(_SPATIAL_INPUTS, layout.spatial_sources, strict=True):
        column = np.empty((len(pairs), 2), dtype=_RAW_DTYPES[src])
        for j, (prev, cur) in enumerate(pairs):
            column[j, 0] = prev[src][index]
            column[j, 1] = cur[src][index]
        batch[name] = column
    derived = derive_spatial(batch)
    return np.stack([derived[name][:, 1] for name in SPATIAL_COLUMNS])


# %%
class _Rings:
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
        masks = tuple(_is_masked(row) for row in self.raw)
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
class _Windows:
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


@dataclass
class _SlotState:
    """Per-slot ring context, the last committed action, and the latest chunk."""

    rings: _Rings | None = None
    pending: np.ndarray | None = None
    offset: int = 0
    last_id: int | None = None  # previous frame's canonical id; a drop = instant-restart boundary
    reset_pending: bool = True
    last_action: np.ndarray | None = None  # the action returned for the PREVIOUS frame


@dataclass
class RecedingHorizon:
    """``BatchPolicy`` for any action-chunk model across N slots.

    Slots that share a replan boundary are stacked into one ``[n_due, L_ctx, ...]``
    batch and run through one ``predict_chunk`` call. Each slot owns its clock,
    because instant-restart boundaries occur asynchronously across Dolphin boots.

    Under instant-restart one boot plays many matches back-to-back; at each match
    boundary — the slot's incoming frame id drops below its last, as Dolphin restarts
    in-place into a new match — that slot's rings are dropped so its context never
    spans two matches, and it re-warms from the boundary (``ctx_pad`` reflects the
    refilling prefix). Its old pending chunk is discarded and it replans immediately;
    unrelated boots keep their current chunks.

    Construct fresh per eval wave (rolling state must not leak across waves).
    """

    predict_chunk: PredictChunk
    stats: dict[str, FeatureStats]
    L_ctx: int
    L_chunk: int
    s: int  # execution horizon: replan + execute this many actions per chunk
    d: int  # inference delay: length of the committed action prefix (0 = open-loop)
    device: str = "cuda"
    # dtype the packed float context arrives in. fp16 halves the launch-bound decode matmuls; the
    # model's float parameters must be cast to match.
    float_dtype: torch.dtype = torch.float32
    # The model's column routing beyond the built-in feature tables (schema v6 and later).
    # Must be the SAME object the train loader collates with, or the closed-loop token
    # would differ from the trained one.
    extra: ExtraColumns | None = None
    projection: FeatureProjection | None = None
    # A compiled inference program needs a stable feature mapping.  Emitting every
    # mask sidecar (zero when no value is masked) is numerically identical to the
    # sparse mapping but prevents first-use recompilation when live rows contain a
    # different subset of missing fields than the synthetic prewarm context.
    emit_all_masks: bool = False
    _slots: dict[Slot, _SlotState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 < self.s <= self.L_chunk:
            raise ValueError(f"execution horizon s={self.s} must satisfy 0 < s <= L_chunk={self.L_chunk}")
        if not 0 <= self.d <= self.L_chunk - self.s:
            raise ValueError(f"inference delay d={self.d} must satisfy 0 <= d <= L_chunk - s={self.L_chunk - self.s}")

    @property
    def runtime_spec(self) -> PolicyRuntimeSpec:
        """Scheduling and shared-memory sizes required by this loaded policy."""
        return PolicyRuntimeSpec(
            context_frames=self.L_ctx,
            prediction_frames=self.L_chunk,
            execution_stride=self.s,
            committed_frames=self.d,
            action_dim=len(ACTION_CHANNELS),
        )

    def __call__(self, frame_index: int, obs: Mapping[Slot, dict]) -> Mapping[Slot, ControllerInputs]:
        live = list(obs)
        self._ingest(live, obs)
        # No neutral-hold warm-up: the policy acts from frame 0. The still-empty context
        # prefix is hidden from attention via ctx_pad (see _replan), so the model sees
        # only real frames and the context fills with REAL gameplay rather than frames
        # produced by an idling model.
        due = [sl for sl in live if self._slots[sl].pending is None or self._slots[sl].offset >= self.s]
        if self.d == 0:
            if due:
                self._replan(due, committed=None)
        else:
            # A bootstrap/reset has no prior commitment. Continuing slots do, so the
            # two cases need separate forwards when both occur on the same frame.
            bootstrap = [sl for sl in due if self._slots[sl].pending is None]
            continuing = [sl for sl in due if self._slots[sl].pending is not None]
            if bootstrap:
                self._replan(bootstrap, committed=None)
            if continuing:
                self._replan(continuing, committed=self._committed(continuing))
        actions: dict[Slot, np.ndarray] = {}
        for sl in live:
            st = self._slots[sl]
            if st.pending is None:
                raise RuntimeError(f"slot {sl} has no pending action chunk after replanning")
            a = st.pending[st.offset]
            actions[sl] = a
            self._push_ego(sl, a)
            st.offset += 1
        return {sl: action_vec_to_controller(a) for sl, a in actions.items()}

    def plan_rows(self, rows: Mapping[Slot, Sequence[ObservationRow]]) -> Mapping[Slot, np.ndarray]:
        """Ingest worker-published rows and return one new chunk per slot.

        The worker supplies the action that produced each observation. This
        keeps the training-time ``(post_i, pre_i)`` alignment without sending
        nested canonical-frame dictionaries across processes.
        """
        live = list(rows)
        if not live:
            return {}
        for slot in live:
            slot_rows = rows[slot]
            if not slot_rows:
                raise ValueError(f"slot {slot} requested a plan without observation rows")
            for row in slot_rows:
                self._ingest_row(slot, row)
        bootstrap = [slot for slot in live if self._slots[slot].pending is None]
        continuing = [slot for slot in live if self._slots[slot].pending is not None]
        if bootstrap:
            self._replan(bootstrap, committed=None)
        if continuing:
            committed = self._committed(continuing) if self.d else None
            self._replan(continuing, committed=committed)
        return {slot: np.asarray(self._slots[slot].pending, dtype=np.float32) for slot in live}

    @staticmethod
    def _reset_state(st: _SlotState) -> None:
        st.rings = None
        st.pending = None
        st.offset = 0
        st.last_action = None
        st.reset_pending = True

    def _ingest_row(self, slot: Slot, row: ObservationRow) -> None:
        st = self._slots.setdefault(slot, _SlotState())
        if row.reset or (st.last_id is not None and row.frame_id < st.last_id):
            self._reset_state(st)
        st.last_id = row.frame_id
        if st.rings is None:
            st.rings = _Rings(
                _build_layout(row.flat, _PORT_TO_PREFIX[slot.port], self.stats, self.extra, self.projection),
                self.L_ctx,
            )
        action = np.asarray(row.action, dtype=np.float32)
        st.rings.gather(row.flat, action)
        if st.rings.layout.spatial_at is not None:
            spatial = _spatial_block(st.rings.layout, [(st.rings.prev, st.rings.raw)])
            st.rings.push(spatial[:, 0])
        else:
            st.rings.push(None)
        st.last_action = action

    def _ingest(self, live: list[Slot], obs: Mapping[Slot, dict]) -> None:
        """Flatten + preprocess this frame once per live slot, into that slot's rings.

        Each context position pairs a gamestate with the ego action that PRODUCED it —
        the previous frame's action, neutral at a bootstrap or right after a reset. That
        is the real ``(post_i, pre_i)`` alignment, not padding.
        """
        gathered: list[_Rings] = []
        for slot in live:
            st = self._slots.setdefault(slot, _SlotState())
            observed = obs[slot]
            if "ports" in observed and "id" in observed:
                fid = int(observed["id"])
                flat = flatten_canonical_frame(observed)
            elif "frame" in observed:
                fid = int(observed["frame"])
                flat = {name: value for name, value in observed.items() if name not in _FLAT_CONTROLLER_COLUMNS}
            else:
                raise ValueError("observation must be a nested Dolphin frame (id/ports) or a flat MDS row (frame)")
            # Instant-restart boundary: Dolphin restarted in-place into a new match, so the
            # canonical frame id reset to the pre-game countdown (dropped below the last id).
            # Drop this slot's rings so its context never spans two matches (stale stage,
            # stocks back to 4, teleported positions — a window with zero training support).
            if st.last_id is not None and fid < st.last_id:
                self._reset_state(st)
            st.last_id = fid
            if st.rings is None:
                st.rings = _Rings(
                    _build_layout(flat, _PORT_TO_PREFIX[slot.port], self.stats, self.extra, self.projection),
                    self.L_ctx,
                )
            st.rings.gather(flat, NEUTRAL_ACTION if st.last_action is None else st.last_action)
            gathered.append(st.rings)
        if gathered and gathered[0].layout.spatial_at is not None:
            spatial = _spatial_block(gathered[0].layout, [(r.prev, r.raw) for r in gathered])
            for j, rings in enumerate(gathered):
                rings.push(spatial[:, j])
        else:
            for rings in gathered:
                rings.push(None)

    def _push_ego(self, slot: Slot, a: np.ndarray) -> None:
        self._slots[slot].last_action = np.asarray(a, dtype=np.float32)

    def _stack_windows(self, live: list[Slot], length: int, *, truncate_left_edge: bool = True) -> _Windows:
        """Stack every live slot's newest ``length`` context rows into one batch.

        Each slot contributes ONE contiguous ring slice per ring — that is what the
        mirrored write buys. The finite-difference columns are then re-zeroed at window
        position 0, which has no predecessor inside the window.
        """
        rings = []
        for sl in live:
            slot_rings = self._slots[sl].rings
            if slot_rings is None:
                raise RuntimeError(f"slot {sl} was replanned before it observed a frame")
            rings.append(slot_rings)
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
        return _Windows(layout=layout, floats=floats, cats=cats, emitted=floats[n_value:].any(axis=(1, 2)))

    def _context(self, live: list[Slot]) -> Context:
        """Stack ``live``'s newest context rows into one device-resident batch.

        The batch always carries the full ``L_ctx`` window. Building the batch consumes each slot's
        reset flag. The model sees a match boundary once, in the first context after it."""
        windows = self._stack_windows(live, self.L_ctx, truncate_left_edge=True)
        layout = windows.layout
        # One host→device transfer per dtype. Moving ~73 feature tensors independently
        # makes CUDA scheduling/allocator overhead dominate when a trainer shares the
        # device; the rings already hold the batch packed, so this copies contiguous
        # memory rather than gathering it.
        n_value = len(layout.value_names)
        packed = torch.from_numpy(windows.floats).to(self.device, self.float_dtype)
        feats: dict[str, torch.Tensor] = dict(zip(layout.value_names, packed[:n_value].unbind(0), strict=True))
        mask_indices = range(len(layout.mask_names)) if self.emit_all_masks else np.flatnonzero(windows.emitted)
        feats.update({layout.mask_names[k]: packed[n_value + k] for k in mask_indices})
        if layout.cat_names:
            cats = torch.from_numpy(windows.cats).to(self.device)
            feats.update(zip(layout.cat_names, cats.unbind(0), strict=True))
        # Hide each slot's still-empty context prefix from attention (frames 0..L_ctx
        # fill from empty); 0 once a slot's history reaches L_ctx.
        ctx_pad = torch.tensor(
            [max(0, self.L_ctx - self._count(sl)) for sl in live],
            dtype=torch.long,
            device=self.device,
        )
        ctx = Context(
            features=feats,
            ctx_pad=ctx_pad,
            slot_ids=torch.tensor([sl.match * 8 + sl.port for sl in live], dtype=torch.long, device=self.device),
            reset=torch.tensor([self._slots[sl].reset_pending for sl in live], dtype=torch.bool, device=self.device),
        )
        for sl in live:
            self._slots[sl].reset_pending = False
        return ctx

    def _replan(self, live: list[Slot], committed: np.ndarray | None) -> None:
        """One batched forward over every live slot. ``live`` order is fixed by
        the caller and reused to scatter the per-slot chunks back."""
        ctx = self._context(live)
        plans = self.predict_chunk(ctx, committed)
        expected = (len(live), self.L_chunk)
        if plans.ndim != 3 or plans.shape[:2] != expected:
            raise ValueError(
                f"predict_chunk returned shape {plans.shape}; expected [n_due, L_chunk, d_action] "
                f"with prefix {expected}"
            )
        for i, sl in enumerate(live):
            self._slots[sl].pending = plans[i]
            self._slots[sl].offset = 0

    def _count(self, slot: Slot) -> int:
        rings = self._slots[slot].rings
        return 0 if rings is None else rings.count

    def _committed(self, live: list[Slot]) -> np.ndarray | None:
        """The ``d`` already-committed actions each new chunk is conditioned on:
        the previous chunk's actions for the new chunk's prefix frames (its
        ``[s : s+d]``, since the new chunk is anchored ``s`` frames later). ``None``
        at bootstrap (no previous chunk) or when ``d == 0`` (open-loop)."""
        if self.d <= 0:
            return None
        committed: list[np.ndarray] = []
        for sl in live:
            pending = self._slots[sl].pending
            if pending is None:
                raise RuntimeError(f"cannot build a committed prefix for bootstrap slot {sl}")
            committed.append(pending[self.s : self.s + self.d].astype(np.float32))
        return np.stack(committed, axis=0)
