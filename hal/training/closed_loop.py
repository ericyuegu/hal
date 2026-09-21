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
  (the previous chunk's ``[s : s+d]``; optionally neutral at bootstrap), so the handoff is
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
(``ContextHistory``). Each ring holds 2x capacity and each row is written twice — at
``head`` and at ``head + L_ctx`` — so any window read is ONE contiguous slice: no
per-frame Python loop over features, no window rebuild, no wrap-around branch.

The shared context layout resolves the column routing of
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
from hal.training.context_history import ContextHistory
from hal.training.context_history import ContextWindows
from hal.training.context_history import push_context_rows
from hal.training.context_history import stack_context_windows
from hal.training.features import ACTION_CHANNELS
from hal.training.features import NEUTRAL_ACTION
from hal.training.features import Context
from hal.training.features import ExtraColumns
from hal.training.features import FeatureProjection

# A bound model + integration scheme: (Context, committed-action prefix or None)
# → predicted action chunks ``[n_live, L_chunk, d_action]`` (numpy, for the
# rolling-buffer plumbing). ``committed`` is ``[n_live, d, d_action]`` — the
# already-locked actions the new chunk's prefix is conditioned on. Historical
# bootstrap uses ``None``; neutral bootstrap always supplies an array, even at d=0.
PredictChunk = Callable[[Context, np.ndarray | None], np.ndarray]

_PORT_TO_PREFIX: dict[int, Literal["p1", "p2"]] = {1: "p1", 2: "p2"}


@dataclass
class _SlotState:
    """Per-slot ring context, the last committed action, and the latest chunk."""

    rings: ContextHistory | None = None
    pending: np.ndarray | None = None
    offset: int = 0
    last_id: int | None = None  # previous frame's canonical id; a drop = instant-restart boundary
    reset_pending: bool = True
    last_action: np.ndarray | None = None  # the action returned for the PREVIOUS frame


@dataclass(frozen=True, slots=True)
class _FaultInput:
    windows: ContextWindows
    slots: tuple[Slot, ...]
    pads: np.ndarray
    resets: np.ndarray


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

    ``bootstrap_committed="neutral"`` starts each slot with ``d`` neutral actions
    and preserves committed values exactly in returned chunks. The default keeps
    historical bootstrap, batching, and predictor output semantics.
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
    fault_metadata: Callable[[], Mapping[str, object]] | None = None
    bootstrap_committed: Literal["none", "neutral"] = field(default="none", kw_only=True)
    _slots: dict[Slot, _SlotState] = field(default_factory=dict)
    _last_fault_input: _FaultInput | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not 0 < self.s <= self.L_chunk:
            raise ValueError(f"execution horizon s={self.s} must satisfy 0 < s <= L_chunk={self.L_chunk}")
        if not 0 <= self.d <= self.L_chunk - self.s:
            raise ValueError(f"inference delay d={self.d} must satisfy 0 <= d <= L_chunk - s={self.L_chunk - self.s}")
        if self.bootstrap_committed not in ("none", "neutral"):
            raise ValueError(f"unsupported bootstrap commitment {self.bootstrap_committed!r}")

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
        due = [sl for sl in live if self._slots[sl].pending is None or self._slots[sl].offset >= self.s]
        self._plan(due, separate_bootstrap=self.d > 0)
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
        self._plan(live, separate_bootstrap=True)
        return {slot: np.asarray(self._slots[slot].pending, dtype=np.float32) for slot in live}

    def _plan(self, live: list[Slot], *, separate_bootstrap: bool) -> None:
        if not live:
            return
        if self.bootstrap_committed == "neutral":
            self._replan(live, committed=self._committed(live))
            return
        # Preserve each historical entry point's batching and random-draw order.
        if not separate_bootstrap:
            self._replan(live, committed=None)
            return
        bootstrap = [slot for slot in live if self._slots[slot].pending is None]
        continuing = [slot for slot in live if self._slots[slot].pending is not None]
        if bootstrap:
            self._replan(bootstrap, committed=None)
        if continuing:
            self._replan(continuing, committed=self._committed(continuing))

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
            st.rings = ContextHistory.from_frame(
                row.flat, _PORT_TO_PREFIX[slot.port], self.stats, self.L_ctx, self.extra, self.projection
            )
        action = np.asarray(row.action, dtype=np.float32)
        st.rings.gather(row.flat, action)
        push_context_rows([st.rings])
        st.last_action = action

    def _ingest(self, live: list[Slot], obs: Mapping[Slot, dict]) -> None:
        """Flatten + preprocess this frame once per live slot, into that slot's rings.

        Each context position pairs a gamestate with the ego action that PRODUCED it —
        the previous frame's action, neutral at a bootstrap or right after a reset. That
        is the real ``(post_i, pre_i)`` alignment, not padding.
        """
        gathered: list[ContextHistory] = []
        for slot in live:
            st = self._slots.setdefault(slot, _SlotState())
            fid = obs[slot]["id"]
            # Instant-restart boundary: Dolphin restarted in-place into a new match, so the
            # canonical frame id reset to the pre-game countdown (dropped below the last id).
            # Drop this slot's rings so its context never spans two matches (stale stage,
            # stocks back to 4, teleported positions — a window with zero training support).
            if st.last_id is not None and fid < st.last_id:
                self._reset_state(st)
            st.last_id = fid
            flat = flatten_canonical_frame(obs[slot])
            if st.rings is None:
                st.rings = ContextHistory.from_frame(
                    flat, _PORT_TO_PREFIX[slot.port], self.stats, self.L_ctx, self.extra, self.projection
                )
            st.rings.gather(flat, NEUTRAL_ACTION if st.last_action is None else st.last_action)
            gathered.append(st.rings)
        push_context_rows(gathered)

    def _push_ego(self, slot: Slot, a: np.ndarray) -> None:
        self._slots[slot].last_action = np.asarray(a, dtype=np.float32)

    def _stack_windows(self, live: list[Slot], length: int, *, truncate_left_edge: bool = True) -> ContextWindows:
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
        return stack_context_windows(rings, length, truncate_left_edge=truncate_left_edge)

    def _context(self, live: list[Slot]) -> Context:
        """Stack ``live``'s newest context rows into one device-resident batch.

        The batch always carries the full ``L_ctx`` window. Building the batch consumes each slot's
        reset flag. The model sees a match boundary once, in the first context after it."""
        windows = self._stack_windows(live, self.L_ctx, truncate_left_edge=True)
        # These arrays already exist for the H2D transfer. Retaining references costs
        # no copies in the hot path and lets the parent serialize the exact input if
        # asynchronous CUDA execution later reports a fault.
        pads = np.fromiter((max(0, self.L_ctx - self._count(sl)) for sl in live), dtype=np.int64)
        resets = np.fromiter((self._slots[sl].reset_pending for sl in live), dtype=np.bool_)
        self._last_fault_input = _FaultInput(windows, tuple(live), pads, resets)
        # One host→device transfer per dtype. Moving ~73 feature tensors independently
        # makes CUDA scheduling/allocator overhead dominate when a trainer shares the
        # device; the rings already hold the batch packed, so this copies contiguous
        # memory rather than gathering it.
        feats = windows.features(self.device, self.float_dtype)
        # Hide each slot's still-empty context prefix from attention (frames 0..L_ctx
        # fill from empty); 0 once a slot's history reaches L_ctx.
        ctx_pad = torch.tensor(pads, dtype=torch.long, device=self.device)
        ctx = Context(
            features=feats,
            ctx_pad=ctx_pad,
            slot_ids=torch.tensor([sl.match * 8 + sl.port for sl in live], dtype=torch.long, device=self.device),
            reset=torch.tensor(resets, dtype=torch.bool, device=self.device),
            observation_counts=torch.tensor([self._count(sl) for sl in live], dtype=torch.long, device=self.device),
        )
        for sl in live:
            self._slots[sl].reset_pending = False
        return ctx

    def fault_snapshot(self) -> tuple[dict[str, object], dict[str, np.ndarray]]:
        """Return the last already-packed host input without touching CUDA."""
        metadata: dict[str, object] = {}
        arrays: dict[str, np.ndarray] = {}
        if self._last_fault_input is not None:
            captured = self._last_fault_input
            windows = captured.windows
            layout = windows.layout
            metadata = {
                "slots": [{"match": sl.match, "port": sl.port} for sl in captured.slots],
                "ctx_pad": captured.pads.tolist(),
                "reset": captured.resets.tolist(),
                "value_names": list(layout.value_names),
                "mask_names": list(layout.mask_names),
                "cat_names": list(layout.cat_names),
                "emitted_masks": windows.emitted.tolist(),
            }
            arrays = {"floats": windows.floats, "cats": windows.cats}
        if self.fault_metadata is not None:
            metadata.update(self.fault_metadata())
        return metadata, arrays

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
            pending = plans[i].copy()
            if self.bootstrap_committed == "neutral" and committed is not None:
                pending[: self.d] = committed[i]
            self._slots[sl].pending = pending
            self._slots[sl].offset = 0

    def _count(self, slot: Slot) -> int:
        rings = self._slots[slot].rings
        return 0 if rings is None else rings.count

    def _committed(self, live: list[Slot]) -> np.ndarray | None:
        """The ``d`` already-committed actions each new chunk is conditioned on:
        the previous chunk's actions for the new chunk's prefix frames (its
        ``[s : s+d]``, since the new chunk is anchored ``s`` frames later).

        Neutral mode includes bootstrap slots and returns an empty prefix at d=0.
        Historical mode requires continuing slots and returns None at d=0.
        """
        if self.bootstrap_committed == "neutral":
            committed = np.broadcast_to(NEUTRAL_ACTION, (len(live), self.d, len(ACTION_CHANNELS))).copy()
            for row, sl in enumerate(live):
                pending = self._slots[sl].pending
                if pending is not None:
                    committed[row] = pending[self.s : self.s + self.d]
            return committed
        if self.d <= 0:
            return None
        prefixes: list[np.ndarray] = []
        for sl in live:
            pending = self._slots[sl].pending
            if pending is None:
                raise RuntimeError(f"cannot build a committed prefix for bootstrap slot {sl}")
            prefixes.append(pending[self.s : self.s + self.d].astype(np.float32))
        return np.stack(prefixes, axis=0)
