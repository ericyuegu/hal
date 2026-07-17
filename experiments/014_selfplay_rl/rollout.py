"""Rollout accumulation, window assembly, and GAE plumbing for Melee PPO (CPU).

Three stages sit between the Dolphin collector (M4b) and the PPO learner:

* :class:`SlotStream` accumulates one model-driven port's rollout — the sampled
  action indices, behavior log-probs, shaped rewards, done flags, the dequantized
  ego action vectors, and the flat gamestate frames. ``T`` transitions store
  ``T + 1`` flat frames (every step's pre-obs plus the final boundary obs) so a
  value can be read at the bootstrap position. A stream created mid-episode (at an
  iteration finalize) also carries a context-only PREFIX — the collector's rolling
  frames/actions preceding frame 0 — so the recompute can burn in across the
  iteration boundary. The prefix is never scored, rewarded, or entered into GAE.
* :func:`build_windows` cuts a finalized stream into end-aligned ``L_ctx`` context
  windows at ``stride`` spacing, segmented at episode boundaries. Each window
  SCORES only its trailing ``stride`` frames (its scored span); everything before
  is burn-in that exists purely to reproduce the rolling ``L_ctx`` context the
  acting policy saw at collection. The scored spans partition the stream, so every
  transition is scored exactly once. Each window's feature rows are built through
  the SAME ``live_batch_from_rolling`` path the closed-loop driver uses, so a
  window's stacked arrays are byte-identical to what ``RecedingHorizon`` would
  feed the model for the same trailing frames — the invariant that keeps the PPO
  recompute faithful to collection (pinned by test).
* :func:`gae_inputs` turns per-stream value arrays (``T + 1`` positions) into
  advantages/returns via tianshou's own ``_gae``, and :func:`scatter_gae` writes
  them back onto window rows through the ``frame_pos`` index map.

Off-by-one contract (consistent with ``rewards.py``): reward/terminated/truncated
index the transition ``t`` (frames ``t, t+1``); ``terminated[t]`` masks the
next-state value ``v_s_[t]``; a ``truncated`` tail bootstraps from the value at
frame ``T``. Scored spans are end-aligned within a segment: the trailing span is a
full ``stride`` block ending at the last frame, the short *head* span begins at the
segment start, and a window shorter than ``L_ctx`` (nothing further back to burn in
on) is left-padded with ``ctx_pad``.
"""

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace

import numpy as np
import torch
from beartype import beartype
from jaxtyping import jaxtyped
from nets_melee import N_GROUPS
from tianshou.algorithm.algorithm_base import _gae
from torch import Tensor

from hal.data.stats import FeatureStats
from hal.training.closed_loop import PORT_TO_PREFIX
from hal.training.closed_loop import live_batch_from_rolling
from hal.training.features import A_DIM
from hal.training.features import Context
from hal.training.features import preprocess


class SlotStream:
    """Mutable per-slot rollout accumulator (one model-driven port).

    ``append_step`` records one transition: the pre-obs flat frame, the executed
    action (dequantized ``A_DIM`` vector + its ``N_GROUPS`` class indices), the
    behavior log-prob, the shaped reward, and the done flags. ``append_boundary``
    stores the final ``T + 1``-th flat frame (the obs after the last step) so the
    bootstrap value has a frame to condition on. ``finalize`` freezes the lists
    into a :class:`FinalizedStream` of numpy arrays."""

    def __init__(
        self,
        ego_port: int,
        *,
        matchup: dict | None = None,
        prefix_flat: tuple[dict, ...] = (),
        prefix_ego: np.ndarray | None = None,
    ) -> None:
        if ego_port not in PORT_TO_PREFIX:
            raise ValueError(f"ego_port must be 1 or 2, got {ego_port}")
        ego = _EMPTY_PREFIX_EGO if prefix_ego is None else np.asarray(prefix_ego, np.float32).reshape(-1, A_DIM)
        if len(prefix_flat) != ego.shape[0]:
            raise ValueError(f"prefix_flat has {len(prefix_flat)} frames but prefix_ego has {ego.shape[0]} actions")
        self.ego_port = ego_port
        self.matchup = matchup
        self.prefix_flat = tuple(prefix_flat)
        self.prefix_ego = ego
        self._flat: list[dict] = []
        self._ego_act: list[np.ndarray] = []
        self._act_idx: list[np.ndarray] = []
        self._logp: list[float] = []
        self._rew: list[float] = []
        self._terminated: list[bool] = []
        self._truncated: list[bool] = []
        self._closed = False

    def append_step(
        self,
        *,
        flat: dict,
        ego_act: np.ndarray,
        act_idx: np.ndarray,
        logp: float,
        rew: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if self._closed:
            raise RuntimeError("append_step after append_boundary")
        self._flat.append(flat)
        self._ego_act.append(np.asarray(ego_act, dtype=np.float32).reshape(A_DIM))
        self._act_idx.append(np.asarray(act_idx, dtype=np.int64).reshape(N_GROUPS))
        self._logp.append(float(logp))
        self._rew.append(float(rew))
        self._terminated.append(bool(terminated))
        self._truncated.append(bool(truncated))

    def append_boundary(self, flat: dict) -> None:
        """Store the final boundary obs (frame ``T``). Closes the stream to steps."""
        if self._closed:
            raise RuntimeError("append_boundary called twice")
        self._flat.append(flat)
        self._closed = True

    def clear_prefix(self) -> None:
        """Drop the carried context prefix. For a stream whose episode ended before any
        step was recorded: the prefix belongs to the match that just died, and frame 0
        restarts cold — exactly like the collection caches."""
        if self._flat:
            raise RuntimeError("clear_prefix after steps were recorded (the prefix is their live context)")
        self.prefix_flat = ()
        self.prefix_ego = _EMPTY_PREFIX_EGO

    @property
    def n_recorded(self) -> int:
        """Transitions appended so far (``T`` before ``append_boundary``)."""
        return len(self._logp)

    def terminate_last(self, bonus: float) -> None:
        """Credit a terminal transition: add the win/loss ``bonus`` to the last recorded
        step's reward and mark it ``terminated`` (the collector calls this at an
        instant-restart episode boundary — the transition into the last pre-reset frame
        is the terminal one, so the bonus lands there, not on the discarded post-reset
        action)."""
        if self._closed:
            raise RuntimeError("terminate_last after append_boundary")
        if not self._rew:
            raise ValueError("terminate_last on a stream with no recorded steps")
        if self._terminated[-1]:
            raise RuntimeError("terminate_last: last step already terminated (double episode boundary)")
        self._rew[-1] += float(bonus)
        self._terminated[-1] = True

    def truncate_last(self) -> None:
        """Mark the last recorded step ``truncated`` (the iteration-budget tail bootstraps
        from the boundary value). No-op on an empty stream; never overrides ``terminated``
        (a step that ended an episode is not a truncation)."""
        if self._closed:
            raise RuntimeError("truncate_last after append_boundary")
        if self._rew and not self._terminated[-1]:
            self._truncated[-1] = True

    def finalize(self) -> FinalizedStream:
        T = len(self._logp)
        if len(self._flat) != T + 1:
            raise ValueError(f"expected T+1={T + 1} flat frames, got {len(self._flat)}; call append_boundary once")
        return FinalizedStream(
            ego_port=self.ego_port,
            matchup=self.matchup,
            flat=tuple(self._flat),
            ego_act=np.stack(self._ego_act) if T else np.zeros((0, A_DIM), np.float32),
            act_idx=np.stack(self._act_idx) if T else np.zeros((0, N_GROUPS), np.int64),
            logp=np.asarray(self._logp, np.float32),
            rew=np.asarray(self._rew, np.float32),
            terminated=np.asarray(self._terminated, bool),
            truncated=np.asarray(self._truncated, bool),
            prefix_flat=self.prefix_flat,
            prefix_ego=self.prefix_ego,
        )


_EMPTY_PREFIX_EGO = np.zeros((0, A_DIM), np.float32)


@dataclass(frozen=True, slots=True)
class FinalizedStream:
    """One slot's finalized rollout: ``T`` transitions + ``T + 1`` flat frames.

    ``prefix_flat``/``prefix_ego`` carry the ``<= L_ctx - 1`` frames (and the actions
    executed AT those frames) immediately preceding frame 0, for a stream that starts
    mid-episode at an iteration boundary — burn-in context only, never scored. Empty
    when frame 0 begins an episode (collection cold-started there too)."""

    ego_port: int
    matchup: dict | None
    flat: tuple[dict, ...]
    ego_act: np.ndarray  # [T, A_DIM] float32
    act_idx: np.ndarray  # [T, N_GROUPS] int64
    logp: np.ndarray  # [T] float32
    rew: np.ndarray  # [T] float32
    terminated: np.ndarray  # [T] bool
    truncated: np.ndarray  # [T] bool
    prefix_flat: tuple[dict, ...] = ()
    prefix_ego: np.ndarray = field(default_factory=lambda: _EMPTY_PREFIX_EGO)  # [P, A_DIM] float32

    def __post_init__(self) -> None:
        if len(self.prefix_flat) != self.prefix_ego.shape[0]:
            raise ValueError(
                f"prefix_flat has {len(self.prefix_flat)} frames but prefix_ego has {self.prefix_ego.shape[0]} actions"
            )

    @property
    def n_transitions(self) -> int:
        return int(self.logp.shape[0])


@dataclass(frozen=True, slots=True)
class RolloutIteration:
    """Finalized streams + completed-episode stats for one collect() payload."""

    streams: tuple[FinalizedStream, ...]
    episode_returns: np.ndarray  # [n_completed] float32
    episode_lengths: np.ndarray  # [n_completed] int64
    n_transitions: int


def build_iteration(streams: list[SlotStream]) -> RolloutIteration:
    """Finalize every stream and reduce completed-episode returns/lengths.

    A completed episode is a run of transitions ending at a ``terminated`` flag;
    an unfinished tail (the budget cut mid-match) is excluded from the stats so
    reported returns aren't biased low by a partial match."""
    finalized = tuple(s.finalize() for s in streams)
    returns: list[float] = []
    lengths: list[int] = []
    for st in finalized:
        seg_ret = 0.0
        seg_len = 0
        for t in range(st.n_transitions):
            seg_ret += float(st.rew[t])
            seg_len += 1
            if st.terminated[t]:
                returns.append(seg_ret)
                lengths.append(seg_len)
                seg_ret = 0.0
                seg_len = 0
    return RolloutIteration(
        streams=finalized,
        episode_returns=np.asarray(returns, np.float32),
        episode_lengths=np.asarray(lengths, np.int64),
        n_transitions=sum(st.n_transitions for st in finalized),
    )


@dataclass(frozen=True, slots=True)
class Window:
    """One ``L_ctx``-row context block whose trailing ``stride`` rows are scored.

    ``feats`` holds the relabeled ego/opp gamestate + ego-action columns (each row
    a frame; left-padded rows zero-filled), exactly as ``live_batch_from_rolling``
    produces them. ``frame_pos`` maps each row back to its stream frame index
    (``-1`` for pad AND for prefix rows, which have no stream frame). ``scored``
    marks the rows of this window's scored span — the rows whose recompute this
    window is responsible for (every stream frame incl. the bootstrap frame ``T``
    is scored in exactly one window; earlier rows are burn-in context). ``valid``
    = scored rows that carry a real transition (excludes the value-only bootstrap
    frame ``T``) — the only rows that enter any loss. ``adv``/``returns`` are zero
    until :func:`scatter_gae` fills the valid rows."""

    feats: dict[str, np.ndarray]
    ctx_pad: int
    act_idx: np.ndarray  # [L_ctx, N_GROUPS] int64
    logp: np.ndarray  # [L_ctx] float32 (behavior log-prob)
    rew: np.ndarray  # [L_ctx] float32
    terminated: np.ndarray  # [L_ctx] bool
    truncated: np.ndarray  # [L_ctx] bool
    scored: np.ndarray  # [L_ctx] bool (this window's scored span, incl. the bootstrap frame)
    valid: np.ndarray  # [L_ctx] bool (scored & real transition)
    frame_pos: np.ndarray  # [L_ctx] int64
    stream_id: int
    adv: np.ndarray  # [L_ctx] float32 (0 until scatter_gae)
    returns: np.ndarray  # [L_ctx] float32 (0 until scatter_gae)


def _segment_ids(terminated: np.ndarray, T: int) -> np.ndarray:
    """Per-frame episode id over frames ``0..T``. Frame ``i`` (``i < T``) belongs to
    the episode with ``sum(terminated[:i])`` prior terminations; the bootstrap frame
    ``T`` is attached to the last real episode (its value is masked in GAE whenever
    that episode terminated, so it never needs a window of its own)."""
    ep = np.zeros(T + 1, dtype=np.int64)
    ep[1:] = np.cumsum(terminated.astype(np.int64))
    ep[T] = ep[T - 1]
    return ep


def _tile_end_aligned(lo: int, hi: int, L_ctx: int) -> list[tuple[int, int]]:
    """Non-overlapping frame blocks covering ``[lo, hi]``, aligned to the RIGHT so the
    trailing block is a full ``L_ctx`` ending at ``hi`` and the short *head* block
    (left-padded downstream) begins at ``lo``. Returned left-to-right."""
    blocks: list[tuple[int, int]] = []
    end = hi
    while end >= lo:
        start = max(lo, end - L_ctx + 1)
        blocks.append((start, end))
        end = start - 1
    blocks.reverse()
    return blocks


def build_windows(
    stream: FinalizedStream, L_ctx: int, *, stride: int | None = None, stream_id: int = 0
) -> list[Window]:
    """Split ``stream`` at episode boundaries and cover each segment with end-aligned
    ``L_ctx`` windows whose scored spans tile the segment in ``stride`` blocks (see
    module docstring). ``stride=None`` means ``L_ctx`` (edge-to-edge, no burn-in).
    Empty when the stream has no steps.

    Burn-in makes the recompute context match collection: a scored position sees at
    least ``L_ctx - stride + 1`` trailing frames (reaching into ``prefix_flat`` across
    the stream head, but never across an episode boundary — collection cold-started
    there, so a short window is exact). The residual mismatch vs the rolling collection
    buffer is at most the ``stride - 1`` OLDEST frames; ``ratio_dev_epoch0`` from
    ``melee_ppo_update`` is the standing diagnostic for how much this (plus KV drift
    between rebuilds) perturbs the epoch-0 ratios."""
    stride = L_ctx if stride is None else stride
    if not 1 <= stride <= L_ctx:
        raise ValueError(f"stride must be in [1, L_ctx={L_ctx}], got {stride}")
    T = stream.n_transitions
    if T == 0:
        return []
    ego_prefix = PORT_TO_PREFIX[stream.ego_port]
    P = len(stream.prefix_flat)
    flat_all = list(stream.prefix_flat) + list(stream.flat)  # combined index c = frame + P
    ego_all = np.concatenate([stream.prefix_ego, stream.ego_act])  # [P + T] actions AT frame c - P
    ep = _segment_ids(stream.terminated, T)
    changes = np.flatnonzero(ep[1:] != ep[:-1]) + 1
    seg_starts = np.concatenate([[0], changes])
    seg_ends = np.concatenate([changes - 1, [T]])  # inclusive frame index of each segment's last frame

    windows: list[Window] = []
    for lo, hi in zip(seg_starts, seg_ends, strict=True):
        lo, hi = int(lo), int(hi)
        ctx_lo = -P if lo == 0 else lo  # only the stream-head segment may burn into the prefix
        for s_lo, s_hi in _tile_end_aligned(lo, hi, stride):
            a = max(ctx_lo, s_hi - L_ctx + 1)
            windows.append(_build_window(stream, flat_all, ego_all, a, s_hi, s_lo, L_ctx, ego_prefix, stream_id))
    return windows


def _build_window(
    stream: FinalizedStream,
    flat_all: list[dict],
    ego_all: np.ndarray,
    a: int,
    b: int,
    s_lo: int,
    L_ctx: int,
    ego_prefix: str,
    stream_id: int,
) -> Window:
    """Window over combined frames ``a..b`` (``a`` may be negative — prefix reach),
    scoring frames ``s_lo..b``. ``flat_all``/``ego_all`` are the prefix+stream arrays
    (combined index = frame + P)."""
    T = stream.n_transitions
    P = len(stream.prefix_flat)
    n = b - a + 1
    # ego action list feeding live_batch_from_rolling (right-aligned; short lists front-pad
    # NEUTRAL). The action that produced the block's first frame is NEUTRAL at an episode
    # start (cold buffer) and when the rolling cap already evicted it (max(a-1, -P) clamps);
    # otherwise it is the real preceding action, so the block front-pads ego by 0.
    ego_lo = a if _at_episode_start(stream, a) else max(a - 1, -P)
    ego = list(ego_all[ego_lo + P : b + P])
    flat_block = flat_all[a + P : b + 1 + P]
    stacked = live_batch_from_rolling(flat_block, ego, ego_prefix, L_ctx)
    feats = {k: v[0] for k, v in stacked.items()}
    ctx_pad = L_ctx - n

    act_idx = np.zeros((L_ctx, N_GROUPS), np.int64)
    logp = np.zeros(L_ctx, np.float32)
    rew = np.zeros(L_ctx, np.float32)
    terminated = np.zeros(L_ctx, bool)
    truncated = np.zeros(L_ctx, bool)
    scored = np.zeros(L_ctx, bool)
    valid = np.zeros(L_ctx, bool)
    frame_pos = np.full(L_ctx, -1, np.int64)
    for j in range(n):
        row = ctx_pad + j
        frame = a + j
        if frame < 0:  # prefix row: burn-in context only, no stream frame behind it
            continue
        frame_pos[row] = frame
        scored[row] = frame >= s_lo  # frame <= b == the span end by construction
        if frame < T:  # real transition (frame T is the value-only bootstrap)
            act_idx[row] = stream.act_idx[frame]
            logp[row] = stream.logp[frame]
            rew[row] = stream.rew[frame]
            terminated[row] = stream.terminated[frame]
            truncated[row] = stream.truncated[frame]
            valid[row] = scored[row]
    return Window(
        feats=feats,
        ctx_pad=ctx_pad,
        act_idx=act_idx,
        logp=logp,
        rew=rew,
        terminated=terminated,
        truncated=truncated,
        scored=scored,
        valid=valid,
        frame_pos=frame_pos,
        stream_id=stream_id,
        adv=np.zeros(L_ctx, np.float32),
        returns=np.zeros(L_ctx, np.float32),
    )


def _at_episode_start(stream: FinalizedStream, frame: int) -> bool:
    """Frame 0 with no carried prefix, or the frame right after a terminated transition,
    begins an episode. Negative frames (prefix reach) are mid-episode by construction."""
    if frame <= 0:
        return frame == 0 and not stream.prefix_flat
    return bool(stream.terminated[frame - 1])


def gae_inputs(
    streams: list[FinalizedStream],
    values: list[np.ndarray],
    *,
    gamma: float,
    gae_lambda: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-stream ``(advantages, returns)`` via tianshou's ``_gae``.

    ``values[i]`` holds ``V`` at all ``T + 1`` frames of ``streams[i]`` (read from a
    windowed forward). Mirrors tianshou's ``compute_episodic_return``: ``v_s = v[:T]``,
    ``v_s_ = v[1:] * ~terminated`` (a terminated step bootstraps nothing), and GAE
    accumulation resets at every ``end_flag = terminated | truncated`` boundary — so
    a truncated tail correctly bootstraps from the frame-``T`` value while a terminated
    tail does not. ``_gae`` is imported private on purpose: the buffer cross-check test
    pins its semantics so an upstream change fails loud."""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for stream, v in zip(streams, values, strict=True):
        T = stream.n_transitions
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        if v.shape[0] != T + 1:
            raise ValueError(f"values must have T+1={T + 1} entries, got {v.shape[0]}")
        v_s = v[:T]
        v_s_ = v[1:] * (~stream.terminated)
        end_flag = stream.terminated | stream.truncated
        adv = _gae(v_s, v_s_, stream.rew.astype(np.float64), end_flag, gamma, gae_lambda)
        returns = adv + v_s
        out.append((adv.astype(np.float32), returns.astype(np.float32)))
    return out


def scatter_gae(window: Window, adv: np.ndarray, returns: np.ndarray) -> Window:
    """Write a stream's per-transition ``adv``/``returns`` (length ``T``) onto a
    window's valid rows via ``frame_pos``. Rows outside ``[0, T)`` stay zero."""
    new_adv = np.zeros_like(window.adv)
    new_ret = np.zeros_like(window.returns)
    m = window.valid
    fp = window.frame_pos[m]
    new_adv[m] = adv[fp]
    new_ret[m] = returns[fp]
    return replace(window, adv=new_adv, returns=new_ret)


@dataclass(frozen=True, slots=True)
class WindowBatch:
    """Collated minibatch of windows: a Context plus per-position PPO targets."""

    context: Context
    act_idx: Tensor  # [B_w, L, N_GROUPS] int64
    logp_old: Tensor  # [B_w, L] float32
    adv: Tensor  # [B_w, L] float32
    returns: Tensor  # [B_w, L] float32
    valid: Tensor  # [B_w, L] bool

    def to(self, device: str | torch.device) -> WindowBatch:
        return WindowBatch(
            context=self.context.to(device),
            act_idx=self.act_idx.to(device),
            logp_old=self.logp_old.to(device),
            adv=self.adv.to(device),
            returns=self.returns.to(device),
            valid=self.valid.to(device),
        )


@jaxtyped(typechecker=beartype)
def collate_windows(windows: list[Window], stats: dict[str, FeatureStats]) -> WindowBatch:
    """Stack windows → ``preprocess`` → Context + stacked PPO targets. Same
    stack-then-preprocess order as the train dataloader, so normalization is
    identical to training/inference."""
    stacked = {k: np.stack([w.feats[k] for w in windows]) for k in windows[0].feats}
    feats = preprocess(stacked, stats)
    ctx_pad = torch.tensor([w.ctx_pad for w in windows], dtype=torch.long)
    return WindowBatch(
        context=Context(features=feats, ctx_pad=ctx_pad),
        act_idx=torch.from_numpy(np.stack([w.act_idx for w in windows])),
        logp_old=torch.from_numpy(np.stack([w.logp for w in windows])),
        adv=torch.from_numpy(np.stack([w.adv for w in windows])),
        returns=torch.from_numpy(np.stack([w.returns for w in windows])),
        valid=torch.from_numpy(np.stack([w.valid for w in windows])),
    )
