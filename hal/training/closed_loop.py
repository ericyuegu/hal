"""Generic receding-horizon closed-loop policy.

``RecedingHorizon`` is the torch-side ``BatchPolicy`` (see ``hal.sim.vec``) that
adapts any action-chunk model to the vectorized eval driver. It owns every part
of closed-loop play that is *invariant* across model architectures:

* per-slot rolling buffers (observed gamestate + the ego's own intended actions),
  capped at ``L_ctx``;
* the cold-start left-pad + alignment that lets the policy act from frame 0 while
  the buffer fills with real gameplay (reported as ``ctx_pad`` so the model masks
  the not-yet-filled prefix from attention);
* the replan clock (``n_lat>0`` → replan every ``n_lat``; else open-loop, replan
  every ``L_chunk``) and bridge bootstrap (zeros at start, else the previous
  chunk's first ``n_lat`` actions);
* stacking every live slot into one batch, ``preprocess`` → :class:`Context`, and
  scattering the predicted chunks back.

The single *variant* — how a chunk is produced from a :class:`Context` — is
injected as ``predict_chunk``. That closure is the only thing that touches the
model, so this class never imports a specific architecture.
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
from hal.training.features import A_DIM
from hal.training.features import ACTION_CHANNELS
from hal.training.features import NEUTRAL_ACTION
from hal.training.features import Context
from hal.training.features import action_vec_to_controller
from hal.training.features import preprocess

# A bound model + integration scheme: Context → predicted action chunks
# ``[n_live, L_chunk, d_action]`` (numpy, for the rolling-buffer plumbing).
PredictChunk = Callable[[Context], np.ndarray]

_PORT_TO_PREFIX: dict[int, Literal["p1", "p2"]] = {1: "p1", 2: "p2"}


@dataclass
class _SlotState:
    """Per-slot rolling buffers + the slot's latest predicted chunk / bridge."""

    flat_hist: list = field(default_factory=list)
    ego_inputs_hist: list = field(default_factory=list)
    pending: np.ndarray | None = None
    current_bridge: np.ndarray | None = None


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

    Every slot appears at frame 0 and warms up in lockstep, so all live slots
    replan on the same frames: at each boundary their contexts are stacked into a
    single ``[n_live, L_ctx, ...]`` batch and run through one ``predict_chunk``
    call. Slots only drop out (matches end) — never appear mid-rollout — so the
    batch shrinks monotonically.

    Construct fresh per eval wave (rolling state must not leak across waves).
    """

    predict_chunk: PredictChunk
    stats: dict[str, FeatureStats]
    L_ctx: int
    L_chunk: int
    n_lat: int
    device: str = "cuda"
    _slots: dict[Slot, _SlotState] = field(default_factory=dict)
    _offset: int = 0
    _bootstrapped: bool = False

    def __call__(self, frame_index: int, obs: Mapping[Slot, dict]) -> Mapping[Slot, ControllerInputs]:
        live = list(obs)
        for slot in live:
            st = self._slots.setdefault(slot, _SlotState())
            st.flat_hist.append(flatten_canonical_frame(obs[slot]))
            if len(st.flat_hist) > self.L_ctx:
                st.flat_hist.pop(0)
        # No neutral-hold warm-up: the policy acts from frame 0. The still-empty
        # buffer prefix is hidden from attention via ctx_pad (see _replan), so the
        # model sees only real frames and the buffer fills with REAL gameplay
        # rather than frames produced by an idling model.
        replan_period = self.n_lat if self.n_lat > 0 else self.L_chunk
        if not self._bootstrapped or self._offset >= replan_period:
            self._replan(live)
            self._offset = 0
            self._bootstrapped = True
        actions: dict[Slot, np.ndarray] = {}
        for s in live:
            st = self._slots[s]
            a = st.current_bridge[self._offset] if self.n_lat > 0 else st.pending[self._offset]
            actions[s] = a
            self._push_ego(s, a)
        self._offset += 1
        return {s: action_vec_to_controller(a) for s, a in actions.items()}

    def _push_ego(self, slot: Slot, a: np.ndarray) -> None:
        st = self._slots[slot]
        st.ego_inputs_hist.append(a.astype(np.float32))
        if len(st.ego_inputs_hist) > self.L_ctx:
            st.ego_inputs_hist.pop(0)

    def _replan(self, live: list[Slot]) -> None:
        """One batched forward over every live slot. ``live`` order is fixed by
        the caller and reused to scatter the per-slot chunks back."""
        stacked = self._build_stacked_batch(live)
        feats = {k: v.to(self.device) for k, v in preprocess(stacked, self.stats).items()}
        # Hide each slot's still-empty buffer prefix from attention (frames
        # 0..L_ctx fill from empty); 0 once a slot's history reaches L_ctx.
        ctx_pad = torch.tensor(
            [max(0, self.L_ctx - len(self._slots[s].flat_hist)) for s in live],
            dtype=torch.long,
            device=self.device,
        )
        bridges = self._bridges(live)
        bridge = None if bridges is None else torch.from_numpy(np.stack(bridges, axis=0)).to(self.device)
        plans = self.predict_chunk(Context(features=feats, bridge=bridge, ctx_pad=ctx_pad))
        for i, s in enumerate(live):
            self._slots[s].pending = plans[i]
            if bridges is not None:
                self._slots[s].current_bridge = bridges[i]

    def _bridges(self, live: list[Slot]) -> list[np.ndarray] | None:
        """Per-slot bridge actions played while the next chunk is computed:
        zeros at bootstrap (no prev chunk), else each slot's prev chunk's first
        ``n_lat`` actions. ``None`` when latency is disabled (open-loop)."""
        if self.n_lat <= 0:
            return None
        if not self._bootstrapped:
            return [np.zeros((self.n_lat, A_DIM), dtype=np.float32) for _ in live]
        return [self._slots[s].pending[: self.n_lat].astype(np.float32) for s in live]

    def _build_stacked_batch(self, live: list[Slot]) -> dict[str, np.ndarray]:
        per_slot = [
            _live_batch_from_rolling(
                self._slots[s].flat_hist,
                self._slots[s].ego_inputs_hist,
                ego_prefix=_PORT_TO_PREFIX[s.port],
                L_ctx=self.L_ctx,
            )
            for s in live
        ]
        return {k: np.concatenate([d[k] for d in per_slot], axis=0) for k in per_slot[0]}
