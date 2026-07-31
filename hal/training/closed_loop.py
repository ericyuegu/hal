"""Generic receding-horizon closed-loop policy.

``RecedingHorizon`` is the torch-side ``BatchPolicy`` (see ``hal.sim.vec``) that
adapts any action-chunk model to the vectorized eval driver. It owns every part
of closed-loop play that is *invariant* across model architectures:

* per-slot rolling buffers (observed gamestate + the ego's own intended actions),
  capped at ``L_ctx`` and cleared at each instant-restart match boundary so a
  slot's context never spans two matches;
* the cold-start left-pad + alignment that lets the policy act from frame 0 while
  the buffer fills with real gameplay (reported as ``ctx_pad`` so the model masks
  the not-yet-filled prefix from attention);
* a per-slot replan clock — replan each slot every ``s`` frames (the execution
  horizon) and execute the chunk's first ``s`` actions, where ``s == L_chunk`` is
  plain open-loop; an instant restart resets that slot's clock and pending chunk;
* the real-time-chunking commitment: when the inference delay ``d > 0``, each new
  chunk is conditioned on the ``d`` actions already committed for its first frames
  (the previous chunk's ``[s : s+d]``; ``None`` at bootstrap), so the handoff is
  continuous (constraint ``d <= L_chunk - s``);
* stacking every live slot into one batch, ``preprocess`` → :class:`Context`, and
  scattering the predicted chunks back.

The single *variant* — how a chunk is produced from a :class:`Context` and the
committed prefix — is injected as ``predict_chunk``. That closure is the only
thing that touches the model, so this class never imports a specific architecture.
"""

from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Literal

import numpy as np
import torch

from hal.data.stats import FeatureStats
from hal.sim.inputs import ControllerInputs
from hal.sim.vec import Slot
from hal.training.canonical import flatten_canonical_frame
from hal.training.dataloader import relabel_ego
from hal.training.features import ACTION_CHANNELS
from hal.training.features import NEUTRAL_ACTION
from hal.training.features import Context
from hal.training.features import action_vec_to_controller
from hal.training.features import preprocess

# A bound model + integration scheme: (Context, committed-action prefix or None)
# → predicted action chunks ``[n_live, L_chunk, d_action]`` (numpy, for the
# rolling-buffer plumbing). ``committed`` is ``[n_live, d, d_action]`` — the
# already-locked actions the new chunk's prefix is conditioned on (``None`` when
# ``d == 0`` or at bootstrap).
PredictChunk = Callable[[Context, np.ndarray | None], np.ndarray]

_PORT_TO_PREFIX: dict[int, Literal["p1", "p2"]] = {1: "p1", 2: "p2"}


@dataclass
class _SlotState:
    """Per-slot rolling buffers + the slot's latest predicted chunk."""

    flat_hist: list = field(default_factory=list)
    ego_inputs_hist: list = field(default_factory=list)
    pending: np.ndarray | None = None
    offset: int = 0
    last_id: int | None = None  # previous frame's canonical id; a drop = instant-restart boundary


def _live_batch_from_rolling(
    flat_history: list[dict],
    ego_inputs_hist: list[np.ndarray],
    ego_prefix: str,
    L_ctx: int,
) -> dict[str, np.ndarray]:
    """``[1, L_ctx]`` batch the model expects, built from one slot's rolling buffers.

    Before the buffers fill to ``L_ctx`` (the first ``L_ctx`` closed-loop frames)
    we LEFT-PAD with zeros. The padded prefix is hidden from attention via
    ``ctx_pad`` (``L_ctx - len(flat_history)``, computed by the policy), so its
    contents never reach the prediction — zero is just a finite filler. The
    policy acts from frame 0 and the buffer fills with REAL gameplay.

    At replan time ``ego_inputs_hist`` is one short of ``flat_history`` (the
    current frame's action hasn't been chosen yet) until both hit the ``L_ctx``
    cap, after which they're equal. Front-padding ego by ``len(flat) - len(ego)``
    neutrals aligns ``ego[i]`` with the gamestate it produced in both regimes —
    this is the real ``(post_i, pre_i)`` alignment, NOT padding, and must stay
    even though the leftmost ``pad_g`` positions are masked out.
    """
    pad_g = L_ctx - len(flat_history)
    out: dict[str, np.ndarray] = {}
    keys = flat_history[0].keys()
    for k in keys:
        sample = flat_history[0][k]
        dtype = np.int32 if isinstance(sample, int) else np.float32
        vals = [h[k] for h in flat_history]
        if pad_g > 0:
            vals = [0] * pad_g + vals
        out[k] = np.array(vals, dtype=dtype)
    # Ego controller history (intended actions, not whatever libmelee reads back).
    # Buttons stored as int 0/1 so the classifier routes via "button".
    ego_aligned = [NEUTRAL_ACTION] * (len(flat_history) - len(ego_inputs_hist)) + list(ego_inputs_hist)
    if pad_g > 0:
        ego_aligned = [NEUTRAL_ACTION] * pad_g + ego_aligned
    hist_arr = np.stack(ego_aligned)
    for i, ch in enumerate(ACTION_CHANNELS):
        col = hist_arr[:, i]
        if ch.startswith("button_"):
            out[f"{ego_prefix}_{ch}"] = (col > 0.5).astype(np.int32)
        else:
            out[f"{ego_prefix}_{ch}"] = col.astype(np.float32)
    out.pop("frame", None)
    relabeled = relabel_ego(out, ego_prefix)
    return {k: v[None, ...] for k, v in relabeled.items()}


@dataclass
class RecedingHorizon:
    """``BatchPolicy`` for any action-chunk model across N slots.

    Slots that share a replan boundary are stacked into one ``[n_due, L_ctx, ...]``
    batch and run through one ``predict_chunk`` call. Each slot owns its clock,
    because instant-restart boundaries occur asynchronously across Dolphin boots.

    Under instant-restart one boot plays many matches back-to-back; at each match
    boundary — the slot's incoming frame id drops below its last, as Dolphin
    restarts in-place into a new match — that slot's rolling buffers are cleared so
    its context never spans two matches, and it re-warms from the boundary
    (``ctx_pad`` reflects the refilling prefix). Its old pending chunk is discarded
    and it replans immediately; unrelated boots keep their current chunks.

    Construct fresh per eval wave (rolling state must not leak across waves).
    """

    predict_chunk: PredictChunk
    stats: dict[str, FeatureStats]
    L_ctx: int
    L_chunk: int
    s: int  # execution horizon: replan + execute this many actions per chunk
    d: int  # inference delay: length of the committed action prefix (0 = open-loop)
    device: str = "cuda"
    _slots: dict[Slot, _SlotState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0 < self.s <= self.L_chunk:
            raise ValueError(f"execution horizon s={self.s} must satisfy 0 < s <= L_chunk={self.L_chunk}")
        if not 0 <= self.d <= self.L_chunk - self.s:
            raise ValueError(f"inference delay d={self.d} must satisfy 0 <= d <= L_chunk - s={self.L_chunk - self.s}")

    def __call__(self, frame_index: int, obs: Mapping[Slot, dict]) -> Mapping[Slot, ControllerInputs]:
        live = list(obs)
        for slot in live:
            st = self._slots.setdefault(slot, _SlotState())
            fid = obs[slot]["id"]
            # Instant-restart boundary: Dolphin restarted in-place into a new match, so the
            # canonical frame id reset to the pre-game countdown (dropped below the last id).
            # Clear this slot's buffers so its context never spans two matches (stale stage,
            # stocks back to 4, teleported positions — a window with zero training support).
            if st.last_id is not None and fid < st.last_id:
                st.flat_hist.clear()
                st.ego_inputs_hist.clear()
                st.pending = None
                st.offset = 0
            st.last_id = fid
            st.flat_hist.append(flatten_canonical_frame(obs[slot]))
            if len(st.flat_hist) > self.L_ctx:
                st.flat_hist.pop(0)
        # No neutral-hold warm-up: the policy acts from frame 0. The still-empty
        # buffer prefix is hidden from attention via ctx_pad (see _replan), so the
        # model sees only real frames and the buffer fills with REAL gameplay
        # rather than frames produced by an idling model.
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

    def _push_ego(self, slot: Slot, a: np.ndarray) -> None:
        st = self._slots[slot]
        st.ego_inputs_hist.append(a.astype(np.float32))
        if len(st.ego_inputs_hist) > self.L_ctx:
            st.ego_inputs_hist.pop(0)

    def _replan(self, live: list[Slot], committed: np.ndarray | None) -> None:
        """One batched forward over every live slot. ``live`` order is fixed by
        the caller and reused to scatter the per-slot chunks back."""
        stacked = self._build_stacked_batch(live)
        feats = {k: v.to(self.device) for k, v in preprocess(stacked, self.stats).items()}
        # Hide each slot's still-empty buffer prefix from attention (frames
        # 0..L_ctx fill from empty); 0 once a slot's history reaches L_ctx.
        ctx_pad = torch.tensor(
            [max(0, self.L_ctx - len(self._slots[sl].flat_hist)) for sl in live],
            dtype=torch.long,
            device=self.device,
        )
        plans = self.predict_chunk(Context(features=feats, ctx_pad=ctx_pad), committed)
        expected = (len(live), self.L_chunk)
        if plans.ndim != 3 or plans.shape[:2] != expected:
            raise ValueError(
                f"predict_chunk returned shape {plans.shape}; expected [n_due, L_chunk, d_action] "
                f"with prefix {expected}"
            )
        for i, sl in enumerate(live):
            self._slots[sl].pending = plans[i]
            self._slots[sl].offset = 0

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

    def _build_stacked_batch(self, live: list[Slot]) -> dict[str, np.ndarray]:
        per_slot = [
            _live_batch_from_rolling(
                self._slots[sl].flat_hist,
                self._slots[sl].ego_inputs_hist,
                ego_prefix=_PORT_TO_PREFIX[sl.port],
                L_ctx=self.L_ctx,
            )
            for sl in live
        ]
        return {k: np.concatenate([d[k] for d in per_slot], axis=0) for k in per_slot[0]}
