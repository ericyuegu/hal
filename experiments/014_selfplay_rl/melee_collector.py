"""Live Melee self-play collector + the drive loop that feeds the PPO learner.

Where the pieces fit:

* :class:`ActingPolicy` is the *population seam*. It owns a net + its ``SlotCaches``
  + a sampling temperature, and turns a batch of live slots' per-frame tokens into
  ``(act_idx, logp, action_vec)``. Training runs a single handle (``"ema"``) whose
  net is the EMA behavior snapshot; eval will run two handles (learner-EMA vs frozen
  IL) and a population is just additional handles. One impl here: :class:`NetActingPolicy`.
* :class:`RLBatchPolicy` is the torch-side ``BatchPolicy`` (see ``hal.sim.vec``). Per
  lockstep frame it keeps each slot's rolling gamestate/action history, computes the
  shaped reward for the *previous* step (the reward off-by-one from ``rewards.py``),
  batches one incremental decode per handle, records behavior ``(act_idx, logp)``, and
  packages a :class:`RolloutIteration` every ``rollout_frames`` transitions.
* :func:`drive_rl` is the long-lived, iteration-oriented drive loop modeled on
  ``hal.sim.vec.drive_vec``: it owns the Dolphin Sessions + a stepping thread pool,
  calls ``RLBatchPolicy`` once per frame, hands finished iterations to a bounded queue,
  and (in sync mode) blocks on a gate until the learner has consumed + advanced the EMA.

Torch lives only in :class:`ActingPolicy` implementations; ``RLBatchPolicy`` and
``drive_rl`` stay numpy/threading so the sim layer never imports the model.

Off-by-one contracts (binding, from ``rewards.py`` / ``rollout.py``):

* the reward for step ``t`` reads frames ``(t, t+1)``, so a step is recorded one frame
  *after* its action was chosen (the pending step), when the next obs is in hand;
* at an instant-restart boundary (``id`` drops) the ended episode's terminal bonus is
  read from the last PRE-reset frame and credited to the last recorded step (the
  transition into that frame), which is marked ``terminated``;
* the ego action at a frame's token is the action executed the PREVIOUS frame (column
  ``k`` carries ``a_{k-1}``), matching ``live_batch_from_rolling`` — the invariant that
  keeps collection log-probs faithful to the PPO recompute.
"""

import queue
import threading
import time
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import field
from types import MappingProxyType
from typing import Protocol

import numpy as np
import torch
from loguru import logger
from nets_melee import FactoredCategorical
from nets_melee import PolicyValueNet
from nets_melee import dequantize_groups
from rewards import step_reward
from rewards import terminal_bonus
from rl_config import RewardConfig
from rollout import RolloutIteration
from rollout import SlotStream
from rollout import build_iteration
from torch import Tensor

from hal.data.stats import FeatureStats
from hal.sim.inputs import ControllerInputs
from hal.sim.session import Matchup
from hal.sim.session import Session
from hal.sim.vec import Slot
from hal.training.canonical import flatten_canonical_frame
from hal.training.closed_loop import PORT_TO_PREFIX
from hal.training.closed_loop import live_batch_from_rolling
from hal.training.dataloader import relabel_ego
from hal.training.features import ACTION_CHANNELS
from hal.training.features import NEUTRAL_ACTION
from hal.training.features import Context
from hal.training.features import action_vec_to_controller
from hal.training.features import preprocess


# --- acting policy (the population seam; torch lives here) --------------------
class ActingPolicy(Protocol):
    """A net + its rolling KV caches + sampling temperature, batched over slots.

    ``step`` forwards ONE new token per row (incremental decode) and samples; ``rebuild``
    re-seeds a subset of rows' caches from a full windowed forward (drift reset);
    ``reset_slot`` cold-starts one row at an episode/reboot boundary. Rows index this
    handle's own ``SlotCaches`` (``0..n_slots-1``); ``RLBatchPolicy`` owns the stable
    Slot→row map."""

    def step(self, rows: Tensor, ctx: Context) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``ctx`` is an ``L=1`` batch over ``rows`` (stepped order). Returns
        ``(act_idx [n, N_GROUPS] int64, logp [n] float32, action_vec [n, A_DIM] float32)``."""
        ...

    def rebuild(self, rows: Tensor, windows: Context) -> None:
        """Re-seed ``rows``' caches from a full forward over their trailing ``L_ctx`` windows."""
        ...

    def reset_slot(self, row: int) -> None:
        """Cold-start one row's cache (length/position back to 0)."""
        ...


class NetActingPolicy:
    """Default :class:`ActingPolicy`: incremental decode + periodic rebuild over a
    ``PolicyValueNet``. Holds the net used at COLLECTION time (the EMA snapshot for the
    learner handle).

    ``step``/``rebuild`` run under ``torch.inference_mode`` (not just ``no_grad``): the
    outputs never re-enter autograd (they are copied to numpy immediately, and the learner
    recomputes log-probs on FRESH ``forward_full`` tensors built from the numpy windows, so
    the infamous "infectious inference-tensor" hazard — inference tensors raising when
    reused in a grad-tracked op — cannot arise here; the KV ring buffers are read/written
    only inside these two inference-mode methods). Validated: KV-equivalence tests + the
    end-to-end smoke keep ``ratio_dev_epoch0`` in the window-approximation floor, so
    inference_mode does not perturb the collection-vs-recompute log-probs. If a real run's
    first-hour ``overlap_frac`` comes in under target, the next lever is running these acting
    forwards on a dedicated CUDA stream (overlap the collector's inference with the learner's
    backward), not reverting the grad mode."""

    def __init__(
        self,
        net: PolicyValueNet,
        *,
        n_slots: int,
        device: str | torch.device = "cpu",
        temp: float = 1.0,
        seed: int = 0,
        max_pos: int = 0,
    ) -> None:
        from kv_cache import SlotCaches  # local import: kv_cache imports nets_melee, keep module load light

        if temp <= 0.0:
            raise ValueError(f"temp must be > 0, got {temp}")
        self.net = net
        self.device = torch.device(device)
        self.temp = temp
        self.caches = SlotCaches(net, n_slots, device=device, max_pos=max_pos)
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(seed)

    @torch.inference_mode()
    def step(self, rows: Tensor, ctx: Context) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.caches.n_slots
        all_slots = rows.shape[0] == n and rows.tolist() == list(range(n))
        slot_ids = None if all_slots else rows.to(self.device)  # None -> the no-gather view path
        hidden = self.caches.step_incremental(self.net, slot_ids, ctx.to(self.device))
        return self._sample(hidden)

    @torch.inference_mode()
    def rebuild(self, rows: Tensor, windows: Context) -> None:
        self.caches.rebuild(self.net, rows.to(self.device), windows.to(self.device))

    def reset_slot(self, row: int) -> None:
        self.caches.reset_slot(row)

    def _sample(self, hidden: Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        logits = self.net.policy_logits(hidden)
        if self.temp != 1.0:
            logits = logits / self.temp
        dist = FactoredCategorical(logits)
        idx = dist.sample(generator=self._gen)  # [n, N_GROUPS]
        logp = dist.log_prob(idx)  # [n]
        vecs = dequantize_groups(self.net.main_centers, self.net.c_centers, self.net.trig_centers, idx)
        return idx.cpu().numpy(), logp.cpu().numpy(), vecs.cpu().numpy()


# --- per-slot collection state ------------------------------------------------
@dataclass(slots=True)
class _Pending:
    """The step whose action is chosen but whose reward (needs the next obs) isn't in
    yet. ``flat`` is the frame the action was chosen at; it becomes the step's stored
    pre-obs once the next frame lands."""

    flat: dict
    act_idx: np.ndarray
    logp: float
    ego_act: np.ndarray


@dataclass(slots=True, kw_only=True)
class _StepSlot:
    """Rolling per-slot state shared by collection and eval: the ``handle`` + cache ``row``
    this slot decodes on, its ``ego_port``, the trailing gamestate/ego-action windows (capped
    at ``L_ctx``), and ``last_id`` for instant-restart detection.

    ``last_id`` is ``None`` until the slot's first frame (sentinel: the first frame of a
    fresh/rebooted slot can never read as an instant-restart reset)."""

    handle: str
    row: int
    ego_port: int
    flat_hist: list[dict] = field(default_factory=list)
    ego_hist: list[np.ndarray] = field(default_factory=list)
    last_id: int | None = None


@dataclass(slots=True, kw_only=True)
class _SlotRL(_StepSlot):
    """One model-driven port's live collection state on top of the shared rolling window.

    ``episode_steps`` counts transitions recorded in the CURRENT episode across stream
    finalizations — it distinguishes a boot-adjacent boundary (nothing ever recorded:
    benign) from a finalize-adjacent one (the episode's steps live in an already-emitted
    stream, so its terminal bonus is genuinely lost)."""

    matchup: dict | None
    stream: SlotStream
    pending: _Pending | None = None
    episode_steps: int = 0


def _token_features(cur_flat: dict, prev_action: np.ndarray, ego_prefix: str) -> dict[str, np.ndarray]:
    """The single new token ``[1, 1, ...]`` for incremental decode — byte-identical to
    the LAST column of ``live_batch_from_rolling`` (ego channel = the previous action,
    column ``k`` carries ``a_{k-1}``), so the cached rollout log-prob matches the PPO
    recompute pre-eviction."""
    out: dict[str, np.ndarray] = {}
    for k, v in cur_flat.items():
        out[k] = np.array([v], dtype=np.int32 if isinstance(v, int) else np.float32)
    for i, ch in enumerate(ACTION_CHANNELS):
        col = prev_action[i]
        if ch.startswith("button_"):
            out[f"{ego_prefix}_{ch}"] = np.array([1 if col > 0.5 else 0], dtype=np.int32)
        else:
            out[f"{ego_prefix}_{ch}"] = np.array([col], dtype=np.float32)
    out.pop("frame", None)
    relabeled = relabel_ego(out, ego_prefix)
    return {k: v[None, ...] for k, v in relabeled.items()}  # [1, 1]


# --- shared stepping core -----------------------------------------------------
# The three points where the two policies diverge, each an optional hook (eval passes none).
OnReset = Callable[[Slot], None]  # instant-restart boundary, BEFORE the common history/cache reset
AfterFrame = Callable[[Slot, dict], None]  # frame flattened + capped, BEFORE its token is built
AfterAction = Callable[[Slot, np.ndarray, float, np.ndarray], None]  # (slot, act_idx, logp, action_vec) sampled


class _KVStepper:
    """The one KV-cached per-frame stepping core, shared by :class:`RLBatchPolicy`
    (collection) and ``EvalBatchPolicy`` (judging).

    Owns the handle -> :class:`ActingPolicy` map, the stable per-handle cache-row counter, and
    the drift-refresh clock, and runs the three-pass frame step over a batch of live slots
    grouped by handle: PASS 1 detects an instant-restart ``id`` drop (cold-starting the slot's
    history + cache), flattens the frame into the rolling window (capped at ``L_ctx``) and
    builds its single incremental-decode token; PASS 2 runs one batched incremental decode per
    handle and caps the ego-action window; PASS 3 periodically rebuilds each handle's caches
    from the trailing windows (RoPE-drift reset + ``max_pos`` bound). The variant bookkeeping —
    reward/stream recording for collection, nothing for eval — rides the three optional hooks;
    the log-prob-faithful token build, the history caps, and the reset/refresh clocks live here
    exactly once so the two policies cannot drift apart."""

    def __init__(
        self,
        *,
        handles: Mapping[str, ActingPolicy],
        stats: dict[str, FeatureStats],
        L_ctx: int,
        refresh_every: int,
    ) -> None:
        self.handles = dict(handles)
        self.stats = stats
        self.L_ctx = L_ctx
        self.refresh_every = refresh_every
        self._rows: dict[str, int] = {name: 0 for name in self.handles}
        self._global_frame = 0
        self._force_rebuild = False

    def request_rebuild(self) -> None:
        """Rebuild every handle's caches at the end of the NEXT ``step`` call, off the
        periodic clock. The caller's hook for acting-weight swaps (the EMA→act_net copy):
        cached K/V computed under the old weights would otherwise persist — a hybrid
        behavior state no snapshot reproduces — until the periodic rebuild, up to
        ``refresh_every - 1`` frames later."""
        self._force_rebuild = True

    def attach_row(self, handle: str) -> int:
        """Assign the next stable cache row within ``handle``'s ``SlotCaches`` (once per slot)."""
        if handle not in self.handles:
            raise KeyError(f"slot assigned to unknown handle {handle!r}")
        row = self._rows[handle]
        self._rows[handle] += 1
        return row

    def reset_row(self, st: _StepSlot) -> None:
        """Cold-start a slot: drop its rolling history and reset its cache row to length 0."""
        st.flat_hist.clear()
        st.ego_hist.clear()
        self.handles[st.handle].reset_slot(st.row)

    def step(
        self,
        live: Sequence[Slot],
        obs: Mapping[Slot, dict],
        state: Mapping[Slot, _StepSlot],
        *,
        on_reset: OnReset | None = None,
        after_frame: AfterFrame | None = None,
        after_action: AfterAction | None = None,
    ) -> dict[Slot, ControllerInputs]:
        by_handle: dict[str, list[Slot]] = {}
        for slot in live:
            by_handle.setdefault(state[slot].handle, []).append(slot)

        # PASS 1: reset detection, flatten, (variant) complete the previous step, build this frame's token.
        tokens: dict[str, list[dict[str, np.ndarray]]] = {h: [] for h in by_handle}
        for slot in live:
            st = state[slot]
            frame = obs[slot]
            if "id" not in frame:
                raise KeyError(f"obs frame for slot {slot} has no 'id' — cannot detect episode boundaries")
            fid = int(frame["id"])
            if st.last_id is not None and fid < st.last_id:  # id dropped -> instant-restart into a new match
                if on_reset is not None:
                    on_reset(slot)  # variant terminal bookkeeping, while flat_hist[-1] is still pre-reset
                self.reset_row(st)
            st.last_id = fid
            cur_flat = flatten_canonical_frame(frame)
            st.flat_hist.append(cur_flat)
            if len(st.flat_hist) > self.L_ctx:
                st.flat_hist.pop(0)
            if after_frame is not None:
                after_frame(slot, cur_flat)
            prev_action = st.ego_hist[-1] if st.ego_hist else NEUTRAL_ACTION
            tokens[st.handle].append(_token_features(cur_flat, prev_action, PORT_TO_PREFIX[st.ego_port]))

        # PASS 2: one batched incremental decode per handle; (variant) record the sampled action.
        actions: dict[Slot, ControllerInputs] = {}
        for handle, slots in by_handle.items():
            rows = torch.tensor([state[s].row for s in slots], dtype=torch.long)
            act_idx, logp, vecs = self.handles[handle].step(rows, self._collate_tokens(tokens[handle]))
            for j, slot in enumerate(slots):
                st = state[slot]
                vec = vecs[j]
                st.ego_hist.append(vec)
                if len(st.ego_hist) > self.L_ctx:
                    st.ego_hist.pop(0)
                if after_action is not None:
                    after_action(slot, act_idx[j], float(logp[j]), vec)
                actions[slot] = action_vec_to_controller(vec)

        # PASS 3: periodic rebuild (drift reset) — the window's last column reproduces this
        # frame's token (ego one short via ego_hist[:-1]); step_incremental already produced the
        # action, rebuild only reseeds the caches for future frames.
        if self._force_rebuild or (self._global_frame > 0 and self._global_frame % self.refresh_every == 0):
            for handle, slots in by_handle.items():
                rows = torch.tensor([state[s].row for s in slots], dtype=torch.long)
                self.handles[handle].rebuild(rows, self._collate_windows(slots, state))
            self._force_rebuild = False
        self._global_frame += 1
        return actions

    def _collate_tokens(self, tokens: list[dict[str, np.ndarray]]) -> Context:
        stacked = {k: np.concatenate([t[k] for t in tokens], axis=0) for k in tokens[0]}
        feats = preprocess(stacked, self.stats)
        return Context(features=feats, ctx_pad=torch.zeros(len(tokens), dtype=torch.long))

    def _collate_windows(self, slots: list[Slot], state: Mapping[Slot, _StepSlot]) -> Context:
        per_slot: list[dict[str, np.ndarray]] = []
        ctx_pad: list[int] = []
        for slot in slots:
            st = state[slot]
            per_slot.append(
                live_batch_from_rolling(st.flat_hist, st.ego_hist[:-1], PORT_TO_PREFIX[st.ego_port], self.L_ctx)
            )
            ctx_pad.append(max(0, self.L_ctx - len(st.flat_hist)))
        stacked = {k: np.concatenate([d[k] for d in per_slot], axis=0) for k in per_slot[0]}
        feats = preprocess(stacked, self.stats)
        return Context(features=feats, ctx_pad=torch.tensor(ctx_pad, dtype=torch.long))


class RLBatchPolicy:
    """``BatchPolicy`` that collects self-play PPO rollouts across N slots.

    Both ports of each match are slots of the SAME training handle (``"ema"``); each is
    its own ego stream (relabeled by ego port). ``assign`` maps a slot to its handle
    name (a future population routes different slots to different handles). Iterations
    are pulled via :meth:`take_iteration` — the drive loop relays them to the learner."""

    def __init__(
        self,
        *,
        handles: Mapping[str, ActingPolicy],
        assign: Callable[[Slot], str],
        slots: Sequence[Slot],
        ego_port_of: Callable[[Slot], int],
        matchup_of: Callable[[Slot], dict | None],
        reward_cfg: RewardConfig,
        stats: dict[str, FeatureStats],
        L_ctx: int,
        refresh_every: int,
        rollout_frames: int,
    ) -> None:
        self.stepper = _KVStepper(handles=handles, stats=stats, L_ctx=L_ctx, refresh_every=refresh_every)
        self.handles = self.stepper.handles  # exposed for the cache-state assertions in tests
        self.assign = assign
        self.ego_port_of = ego_port_of
        self.matchup_of = matchup_of
        self.reward_cfg = reward_cfg
        self.rollout_frames = rollout_frames

        self.state: dict[Slot, _SlotRL] = {}
        for slot in slots:
            self._init_slot(slot)
        self._pending_iters: list[RolloutIteration] = []
        # Streams flushed (truncated + boundary appended) by a wave reboot, carried into
        # the next finalized iteration so recorded transitions are never discarded.
        self._orphans: list[SlotStream] = []

    # -- slot lifecycle --------------------------------------------------------
    def _init_slot(self, slot: Slot) -> None:
        handle = self.assign(slot)
        row = self.stepper.attach_row(handle)  # stable Slot -> cache row, assigned once
        ego_port = self.ego_port_of(slot)
        self.state[slot] = _SlotRL(
            handle=handle,
            row=row,
            ego_port=ego_port,
            matchup=self.matchup_of(slot),
            stream=SlotStream(ego_port, matchup=self.matchup_of(slot)),
        )

    def reset_slots(self, slots: Sequence[Slot], *, matchups: Mapping[Slot, dict] | None = None) -> None:
        """Cold-restart the given slots after a wave reboot. Each slot's in-progress
        stream is FLUSHED, not discarded: its tail is marked truncated, the boundary obs
        appended, and the closed stream parked as an orphan that joins the next finalized
        iteration — so recorded transitions survive the reboot. The orphaned stream keeps
        the OLD matchup it was collected under; when ``matchups`` is given (the wave-reboot
        rotation) each slot's fresh stream picks up the NEW matchup for the next slice."""
        for slot in slots:
            st = self.state[slot]
            if st.stream.n_recorded > 0:
                st.stream.truncate_last()
                st.stream.append_boundary(st.flat_hist[-1])
                self._orphans.append(st.stream)
            if matchups is not None:
                st.matchup = matchups[slot]
            st.stream = SlotStream(st.ego_port, matchup=st.matchup)
            self.stepper.reset_row(st)  # drop rolling history + cold-start the cache row
            st.pending = None
            st.last_id = None
            st.episode_steps = 0

    def _on_reset(self, slot: Slot) -> None:
        """Instant-restart hook (runs BEFORE the stepper cold-starts the slot's history +
        cache): the frame in hand is the NEW match and ``flat_hist[-1]`` is still the last
        pre-reset frame. Credit its win/loss to the last recorded step and drop the
        post-reset pending."""
        st = self.state[slot]
        if st.stream.n_recorded > 0:
            bonus = terminal_bonus(st.flat_hist[-1], st.ego_port, self.reward_cfg, terminated=True)
            st.stream.terminate_last(bonus)
        elif st.episode_steps > 0:
            # The episode's steps were emitted (truncated) with the last iteration and its
            # fresh stream has none — the terminated flag + win/loss bonus cannot attach.
            logger.warning(
                f"RLBatchPolicy: terminal bonus LOST on ego_port {st.ego_port} — episode ended on the first "
                f"frame after an iteration boundary ({st.episode_steps} steps already finalized as truncated)"
            )
        else:
            logger.debug(f"RLBatchPolicy: boundary on ego_port {st.ego_port} before any recorded step (boot noise)")
        if st.stream.n_recorded == 0:
            # The stream's carried context prefix belongs to the match that just ended;
            # its frame 0 restarts cold, exactly like the collection caches.
            st.stream.clear_prefix()
        st.pending = None
        st.episode_steps = 0

    def _complete_pending(self, slot: Slot, cur_flat: dict) -> None:
        """Complete the pending step now its next obs (``cur_flat``) is in hand — the reward
        off-by-one from ``rewards.py`` reads frames ``(t, t+1)``."""
        st = self.state[slot]
        if st.pending is None:
            return
        rew = step_reward(st.pending.flat, cur_flat, st.ego_port, self.reward_cfg)
        st.stream.append_step(
            flat=st.pending.flat,
            ego_act=st.pending.ego_act,
            act_idx=st.pending.act_idx,
            logp=st.pending.logp,
            rew=rew,
            terminated=False,
            truncated=False,
        )
        st.episode_steps += 1

    def _record_pending(self, slot: Slot, act_idx: np.ndarray, logp: float, vec: np.ndarray) -> None:
        """Stash the just-sampled action as the pending step (its reward needs the next obs);
        ``flat_hist[-1]`` is this frame, the step's stored pre-obs."""
        st = self.state[slot]
        st.pending = _Pending(flat=st.flat_hist[-1], act_idx=act_idx, logp=logp, ego_act=vec)

    # -- per-frame step --------------------------------------------------------
    def __call__(self, frame_index: int, obs: Mapping[Slot, dict]) -> Mapping[Slot, ControllerInputs]:
        live = [s for s in self.state if s in obs]  # stable order = insertion order
        if not live:
            return {}
        actions = self.stepper.step(
            live,
            obs,
            self.state,
            on_reset=self._on_reset,
            after_frame=self._complete_pending,
            after_action=self._record_pending,
        )
        if self._total_transitions() >= self.rollout_frames:
            self._pending_iters.append(self._finalize_iteration())
        return actions

    def _total_transitions(self) -> int:
        live = sum(st.stream.n_recorded for st in self.state.values())
        return live + sum(s.n_recorded for s in self._orphans)

    def _finalize_iteration(self) -> RolloutIteration:
        """Truncate every live stream's tail, append its boundary obs (frame ``T``), and
        reduce — together with any reboot-orphaned streams — to a :class:`RolloutIteration`,
        then start fresh streams. Rolling history + caches + the pending step persist, so
        the next stream picks up mid-episode with no gap and the boundary frame is the
        shared bootstrap of both. The fresh stream carries the rolling window's frames
        before the boundary as its context-only PREFIX (``flat_hist[-1]`` is the pending
        step's pre-obs — the new stream's frame 0 — so the prefix is everything older),
        letting the PPO recompute burn in across the iteration boundary."""
        streams = list(self.state.values())
        finalized_inputs: list[SlotStream] = self._orphans  # already truncated + closed by reset_slots
        self._orphans = []
        for st in streams:
            st.stream.truncate_last()
            boundary = st.flat_hist[-1] if st.flat_hist else _EMPTY_FLAT
            st.stream.append_boundary(boundary)
            finalized_inputs.append(st.stream)
        iteration = build_iteration(finalized_inputs)
        for st in streams:
            prefix_ego = np.stack(st.ego_hist[:-1]) if len(st.ego_hist) > 1 else None
            st.stream = SlotStream(
                st.ego_port, matchup=st.matchup, prefix_flat=tuple(st.flat_hist[:-1]), prefix_ego=prefix_ego
            )
        return iteration

    def take_iteration(self) -> RolloutIteration | None:
        """Pop one finished iteration if ready (the drive loop relays it to the learner)."""
        return self._pending_iters.pop(0) if self._pending_iters else None


# A stream that recorded no steps this iteration still needs one boundary flat to finalize
# (T=0 -> build_windows yields nothing); this read-only placeholder is only ever the T+1
# frame of an empty stream and never enters a window or a reward.
_EMPTY_FLAT: Mapping[str, float] = MappingProxyType({})


# --- drive loop ---------------------------------------------------------------
# (boot_index, attempt) -> a fresh, un-entered Session. The factory owns port assignment
# and must give each (boot, attempt) a distinct slippi_port (mirroring run_matches_vec's
# fresh-ports-per-retry discipline so a stuck, not-yet-reaped Dolphin can't collide).
BuildBoot = Callable[[int, int], Session]

# wave index -> that wave's per-boot Matchups. Waves rotate through the training prior so a
# long run self-plays the full character matchup distribution, not just the first n_boots.
WaveMatchups = Callable[[int], Sequence[Matchup]]


def matchup_meta(m: Matchup) -> dict:
    """The per-slot matchup metadata dict (injected live stage + per-port character ids).
    One builder shared by boot-wave setup and wave-reboot matchup rotation, so a rotated
    wave's stream metadata and injected stage stay consistent with what it booted."""
    return {"stage": int(m.stage.value), "character": {pl.port: int(pl.character.value) for pl in m.players}}


@dataclass(frozen=True, slots=True)
class CollectStats:
    """Collector-side throughput for one iteration: lockstep frames stepped since the
    previous boundary and the frames/s over that stepping interval (collector wall-clock
    between boundaries — excludes the sync-gate wait, so it is honest in sync mode)."""

    frames: int
    lockstep_sps: float


IterationPayload = tuple[RolloutIteration, CollectStats]


def drive_rl(
    build_boot: BuildBoot,
    wave_matchups: WaveMatchups,
    model_ports: Sequence[tuple[int, ...]],
    policy: RLBatchPolicy,
    *,
    n_iterations: int,
    queue_out: queue.Queue[IterationPayload],
    stop: threading.Event,
    on_iteration: Callable[[], None],
    sync_gate: threading.Event | None = None,
    start_retries: int = 2,
    reboot_every_iters: int = 0,
    wave0: int = 0,
    max_frames: int = 10_000_000,
    progress_every: int = 600,
) -> None:
    """Persistent, iteration-oriented drive loop (the ``drive_vec`` sibling for RL).

    ``build_boot(i, attempt)`` constructs boot ``i``'s Session; ``drive_rl`` enters and
    starts every boot concurrently, re-driving start-flaked boots on fresh Sessions/ports
    up to ``start_retries`` times (the libmelee stage-select flake — same absorption as
    ``run_matches_vec``), then steps all boots on a thread pool one lockstep frame at a
    time, feeding ``policy`` and relaying each finished iteration (plus collector-side
    :class:`CollectStats`) to ``queue_out``. ``on_iteration`` runs at every boundary
    (after the sync gate) to copy the freshly-advanced EMA into the behavior net. In sync
    mode (``sync_gate`` set) the loop blocks on the gate after each put, so the collector
    never runs ahead of the learner (lag 0).

    Matchup rotation: the wave's matchups come from ``wave_matchups(wave_idx)`` (wave
    ``wave0`` at launch — a resumed run seeds this so rotation continues through the prior
    instead of restarting at slice 0). Every wave reboot advances ``wave_idx``, so the run
    cycles through the full
    training prior instead of self-playing one fixed slice for its whole duration. A reboot
    fires on attrition (fewer than half the boots alive) OR on a schedule
    (``reboot_every_iters`` learned iterations, ``0`` = attrition-only); either way the
    in-progress streams flush truncated via ``policy.reset_slots`` (their recorded
    transitions carry into the next iteration as orphans) and the fresh streams pick up the
    new slice's matchups. Exceptions propagate (the learner thread sets ``stop`` on its own
    failure; a drive-loop death surfaces via ``queue_out`` going empty), and every entered
    Session is torn down on the way out."""
    entered: list[Session] = []
    n = len(model_ports)
    attempt0 = 0  # advances per wave so reboot retries rotate onto fresh ports too
    wave_idx = wave0
    matchups: Sequence[Matchup] = wave_matchups(wave_idx)
    if len(matchups) != n:
        raise ValueError(f"wave_matchups({wave_idx}) returned {len(matchups)} matchups; expected {n}")
    with ThreadPoolExecutor(max_workers=32) as pool:
        try:
            sessions, done, last_frame = _boot_wave(
                build_boot, matchups, entered, pool, attempt0=attempt0, start_retries=start_retries
            )
            attempt0 += start_retries + 1

            def reboot_wave(reason: str) -> None:
                nonlocal sessions, done, last_frame, attempt0, wave_idx, matchups
                _teardown(entered)
                wave_idx += 1
                matchups = wave_matchups(wave_idx)
                if len(matchups) != n:
                    raise ValueError(f"wave_matchups({wave_idx}) returned {len(matchups)} matchups; expected {n}")
                new_meta = {Slot(i, p): matchup_meta(matchups[i]) for i in range(n) for p in model_ports[i]}
                logger.warning(f"drive_rl: rebooting wave ({reason}) -> matchup slice {wave_idx}")
                policy.reset_slots(list(policy.state), matchups=new_meta)  # flushes in-progress streams truncated
                sessions, done, last_frame = _boot_wave(
                    build_boot, matchups, entered, pool, attempt0=attempt0, start_retries=start_retries
                )
                attempt0 += start_retries + 1

            logger.info(f"drive_rl: {sum(not d for d in done)}/{n} boots up; target {n_iterations} iterations")
            on_iteration()  # prime the behavior net (EMA -> act_net) before the first collect

            emitted = 0
            pending_reboot = False
            t0 = time.monotonic()
            frames_stepped = 0
            frames_mark = 0
            t_mark = time.monotonic()
            for t in range(max_frames):
                if stop.is_set() or emitted >= n_iterations:
                    break
                live = [i for i in range(n) if not done[i]]
                attrition = len(live) < max(1, n // 2)
                if attrition or pending_reboot:
                    reboot_wave("attrition" if attrition else f"scheduled@{emitted}iters")
                    pending_reboot = False
                    continue
                if progress_every and t > 0 and t % progress_every == 0:
                    logger.info(f"drive_rl: frame {t} | live {len(live)}/{n} | {t / (time.monotonic() - t0):.0f} f/s")

                for i in live:  # refresh injected live stage (instant-restart randomizes per match)
                    meta = last_frame[i]["_matchup"]
                    meta["stage"] = last_frame[i].get("stage", meta["stage"])
                obs = {Slot(i, p): last_frame[i] for i in live for p in model_ports[i]}
                inputs = policy(t, obs)
                futs = {
                    i: pool.submit(sessions[i].step, {p: inputs[Slot(i, p)] for p in model_ports[i]}) for i in live
                }
                frames_stepped += 1
                for i, fut in futs.items():
                    try:
                        frame, in_game = fut.result()
                    except Exception as e:  # noqa: BLE001 — one bad emulator must not kill the others
                        logger.warning(f"drive_rl: boot {i} step crashed: {e!r}")
                        done[i] = True
                        continue
                    if not in_game:  # real drop to menu (rare under instant-restart) — retire the boot
                        logger.info(f"drive_rl: boot {i} left IN_GAME at frame {t}")
                        done[i] = True
                        continue
                    frame["_matchup"] = last_frame[i]["_matchup"]
                    last_frame[i] = frame

                iteration = policy.take_iteration()
                if iteration is not None:
                    step_s = time.monotonic() - t_mark
                    stats = CollectStats(
                        frames=frames_stepped - frames_mark,
                        lockstep_sps=(frames_stepped - frames_mark) / max(1e-9, step_s),
                    )
                    _put(queue_out, (iteration, stats), stop)
                    emitted += 1
                    if stop.is_set():
                        break
                    if sync_gate is not None:
                        while not sync_gate.wait(timeout=_POLL_SECONDS):
                            if stop.is_set():
                                return
                        sync_gate.clear()
                    on_iteration()
                    frames_mark = frames_stepped  # sps window restarts AFTER the gate/EMA copy
                    t_mark = time.monotonic()
                    # Schedule a wave reboot (matchup rotation) at the next frame boundary.
                    pending_reboot = reboot_every_iters > 0 and emitted % reboot_every_iters == 0
            logger.info(f"drive_rl: done after {emitted} iterations")
        finally:
            _teardown(entered)


_POLL_SECONDS = 0.05


def _put(q: queue.Queue[IterationPayload], item: IterationPayload, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            q.put(item, timeout=_POLL_SECONDS)
            return
        except queue.Full:
            continue


def _boot_wave(
    build_boot: BuildBoot,
    matchups: Sequence[Matchup],
    entered: list[Session],
    pool: ThreadPoolExecutor,
    *,
    attempt0: int,
    start_retries: int,
) -> tuple[list[Session | None], list[bool], list[dict]]:
    """Enter + concurrently start a fresh wave, retrying start-flaked boots on fresh
    Sessions/ports (``build_boot(i, attempt)``) up to ``start_retries`` times — the
    stage-select flake clears per-attempt (see ``run_matches_vec``). Successful sessions
    are appended to ``entered`` for guaranteed teardown; a failed attempt's session is
    torn down immediately. Boots still down after all retries are marked done."""
    n = len(matchups)
    sessions: list[Session | None] = [None] * n
    last_frame: list[dict] = [{} for _ in range(n)]
    meta = [matchup_meta(m) for m in matchups]
    pending = list(range(n))
    for attempt in range(start_retries + 1):
        # Enter each session as it is built and append to ``entered`` immediately, so a
        # mid-batch __enter__ failure still leaves every already-entered session tracked
        # for teardown (a build/enter that raises here propagates, and the finally in
        # drive_rl tears down whatever was entered).
        fresh: dict[int, Session] = {}
        for i in pending:
            s = build_boot(i, attempt0 + attempt)
            s.__enter__()
            entered.append(s)
            fresh[i] = s
        futs = {i: pool.submit(s.start_match, matchups[i]) for i, s in fresh.items()}
        still_down: list[int] = []
        for i, fut in futs.items():
            try:
                f0 = fut.result()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"drive_rl: boot {i} start failed (attempt {attempt + 1}/{start_retries + 1}): {e!r}")
                entered.remove(fresh[i])  # tear down the failed attempt now, drop from the tracked set
                _close_one(fresh[i])
                still_down.append(i)
                continue
            f0["_matchup"] = meta[i]
            last_frame[i] = f0
            sessions[i] = fresh[i]  # already in ``entered`` (appended at enter time)
        pending = still_down
        if not pending:
            break
        if attempt < start_retries:
            logger.warning(f"drive_rl: retrying {len(pending)} boot(s) on fresh Sessions/ports")
    done = [sessions[i] is None for i in range(n)]
    return sessions, done, last_frame


def _close_one(s: Session) -> None:
    try:
        s.__exit__(None, None, None)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"drive_rl: session teardown error: {e!r}")


def _teardown(entered: list[Session]) -> None:
    while entered:
        _close_one(entered.pop())
