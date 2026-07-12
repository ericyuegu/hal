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
  ``k`` carries ``a_{k-1}``), matching ``_live_batch_from_rolling`` — the invariant that
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
from hal.training.closed_loop import _PORT_TO_PREFIX
from hal.training.closed_loop import _live_batch_from_rolling
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
    learner handle); ``step``/``rebuild`` never touch autograd."""

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

    @torch.no_grad()
    def step(self, rows: Tensor, ctx: Context) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden = self.caches.step_incremental(self.net, rows.to(self.device), ctx.to(self.device))
        return self._sample(hidden)

    @torch.no_grad()
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


@dataclass(slots=True)
class _SlotRL:
    """One model-driven port's live collection state."""

    ego_port: int
    handle: str
    row: int
    matchup: dict | None
    stream: SlotStream
    flat_hist: list[dict] = field(default_factory=list)
    ego_hist: list[np.ndarray] = field(default_factory=list)
    pending: _Pending | None = None
    last_id: int = 0


def _token_features(cur_flat: dict, prev_action: np.ndarray, ego_prefix: str) -> dict[str, np.ndarray]:
    """The single new token ``[1, 1, ...]`` for incremental decode — byte-identical to
    the LAST column of ``_live_batch_from_rolling`` (ego channel = the previous action,
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
        self.handles = dict(handles)
        self.assign = assign
        self.ego_port_of = ego_port_of
        self.matchup_of = matchup_of
        self.reward_cfg = reward_cfg
        self.stats = stats
        self.L_ctx = L_ctx
        self.refresh_every = refresh_every
        self.rollout_frames = rollout_frames

        # Stable Slot -> row within its handle's caches, assigned once.
        self._rows: dict[str, int] = {name: 0 for name in self.handles}
        self.state: dict[Slot, _SlotRL] = {}
        for slot in slots:
            self._init_slot(slot)
        self._global_frame = 0
        self._pending_iters: list[RolloutIteration] = []

    # -- slot lifecycle --------------------------------------------------------
    def _init_slot(self, slot: Slot) -> None:
        handle = self.assign(slot)
        if handle not in self.handles:
            raise KeyError(f"slot {slot} assigned to unknown handle {handle!r}")
        row = self._rows[handle]
        self._rows[handle] += 1
        ego_port = self.ego_port_of(slot)
        self.state[slot] = _SlotRL(
            ego_port=ego_port,
            handle=handle,
            row=row,
            matchup=self.matchup_of(slot),
            stream=SlotStream(ego_port, matchup=self.matchup_of(slot)),
        )

    def reset_slots(self, slots: Sequence[Slot]) -> None:
        """Cold-restart the given slots (drop rolling history + partial stream, reset the
        cache row) — used after a wave reboot so a reused Slot index doesn't carry the
        dead match's context."""
        for slot in slots:
            st = self.state[slot]
            st.flat_hist.clear()
            st.ego_hist.clear()
            st.pending = None
            st.last_id = 0
            st.stream = SlotStream(st.ego_port, matchup=st.matchup)
            self.handles[st.handle].reset_slot(st.row)

    def _on_reset(self, st: _SlotRL) -> None:
        """Instant-restart boundary: the frame in hand is the NEW match; ``flat_hist[-1]``
        is the last pre-reset frame. Credit its win/loss to the last recorded step, drop
        the post-reset pending, and cold-start the slot's history + cache."""
        if st.stream.n_recorded > 0:
            bonus = terminal_bonus(st.flat_hist[-1], st.ego_port, self.reward_cfg, terminated=True)
            st.stream.terminate_last(bonus)
        else:
            logger.warning(f"RLBatchPolicy: episode boundary on {st.ego_port} with no recorded step; bonus dropped")
        st.flat_hist.clear()
        st.ego_hist.clear()
        st.pending = None
        self.handles[st.handle].reset_slot(st.row)

    # -- per-frame step --------------------------------------------------------
    def __call__(self, frame_index: int, obs: Mapping[Slot, dict]) -> Mapping[Slot, ControllerInputs]:
        live = [s for s in self.state if s in obs]  # stable order = insertion order
        if not live:
            return {}
        by_handle: dict[str, list[Slot]] = {}
        for slot in live:
            by_handle.setdefault(self.state[slot].handle, []).append(slot)

        # PASS 1: reset detection, flatten, complete the previous step, build this frame's token.
        tokens: dict[str, list[dict[str, np.ndarray]]] = {h: [] for h in by_handle}
        for slot in live:
            st = self.state[slot]
            fid = int(obs[slot].get("id", st.last_id + 1))
            if fid < st.last_id:  # id dropped -> instant-restart into a new match
                self._on_reset(st)
            st.last_id = fid
            cur_flat = flatten_canonical_frame(obs[slot])
            st.flat_hist.append(cur_flat)
            if len(st.flat_hist) > self.L_ctx:
                st.flat_hist.pop(0)
            if st.pending is not None:
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
            prev_action = st.ego_hist[-1] if st.ego_hist else NEUTRAL_ACTION
            tokens[st.handle].append(_token_features(cur_flat, prev_action, _PORT_TO_PREFIX[st.ego_port]))

        # PASS 2: one batched incremental decode per handle; record the pending step.
        actions: dict[Slot, ControllerInputs] = {}
        for handle, slots in by_handle.items():
            rows = torch.tensor([self.state[s].row for s in slots], dtype=torch.long)
            ctx = self._collate_tokens(tokens[handle])
            act_idx, logp, vecs = self.handles[handle].step(rows, ctx)
            for j, slot in enumerate(slots):
                st = self.state[slot]
                vec = vecs[j]
                st.ego_hist.append(vec)
                if len(st.ego_hist) > self.L_ctx:
                    st.ego_hist.pop(0)
                st.pending = _Pending(flat=st.flat_hist[-1], act_idx=act_idx[j], logp=float(logp[j]), ego_act=vec)
                actions[slot] = action_vec_to_controller(vec)

        # PASS 3: periodic rebuild (drift reset) — the window's last column reproduces this
        # frame's token (ego one short via ego_hist[:-1]); step_incremental already produced
        # the action, rebuild only reseeds the caches for future frames.
        if self._global_frame > 0 and self._global_frame % self.refresh_every == 0:
            for handle, slots in by_handle.items():
                rows = torch.tensor([self.state[s].row for s in slots], dtype=torch.long)
                self.handles[handle].rebuild(rows, self._collate_windows(slots))

        self._global_frame += 1
        if self._total_transitions() >= self.rollout_frames:
            self._pending_iters.append(self._finalize_iteration())
        return actions

    def _total_transitions(self) -> int:
        return sum(st.stream.n_recorded for st in self.state.values())

    def _collate_tokens(self, tokens: list[dict[str, np.ndarray]]) -> Context:
        stacked = {k: np.concatenate([t[k] for t in tokens], axis=0) for k in tokens[0]}
        feats = preprocess(stacked, self.stats)
        return Context(features=feats, ctx_pad=torch.zeros(len(tokens), dtype=torch.long))

    def _collate_windows(self, slots: list[Slot]) -> Context:
        per_slot: list[dict[str, np.ndarray]] = []
        ctx_pad: list[int] = []
        for slot in slots:
            st = self.state[slot]
            per_slot.append(
                _live_batch_from_rolling(st.flat_hist, st.ego_hist[:-1], _PORT_TO_PREFIX[st.ego_port], self.L_ctx)
            )
            ctx_pad.append(max(0, self.L_ctx - len(st.flat_hist)))
        stacked = {k: np.concatenate([d[k] for d in per_slot], axis=0) for k in per_slot[0]}
        feats = preprocess(stacked, self.stats)
        return Context(features=feats, ctx_pad=torch.tensor(ctx_pad, dtype=torch.long))

    def _finalize_iteration(self) -> RolloutIteration:
        """Truncate every stream's tail, append its boundary obs (frame ``T``), reduce to a
        :class:`RolloutIteration`, then start fresh streams. Rolling history + caches +
        the pending step persist, so the next stream picks up mid-episode with no gap and
        the boundary frame is the shared bootstrap of both."""
        streams = list(self.state.values())
        finalized_inputs: list[SlotStream] = []
        for st in streams:
            st.stream.truncate_last()
            boundary = st.flat_hist[-1] if st.flat_hist else _EMPTY_FLAT
            st.stream.append_boundary(boundary)
            finalized_inputs.append(st.stream)
        iteration = build_iteration(finalized_inputs)
        for st in streams:
            st.stream = SlotStream(st.ego_port, matchup=st.matchup)
        return iteration

    def take_iteration(self) -> RolloutIteration | None:
        """Pop one finished iteration if ready (the drive loop relays it to the learner)."""
        return self._pending_iters.pop(0) if self._pending_iters else None


# A stream that recorded no steps this iteration still needs one boundary flat to finalize
# (T=0 -> build_windows yields nothing); this placeholder is only ever the T+1 frame of an
# empty stream and never enters a window or a reward.
_EMPTY_FLAT: dict[str, float] = {}


# --- drive loop ---------------------------------------------------------------
def drive_rl(
    build_sessions: Callable[[], tuple[list[Session], list[Matchup], list[tuple[int, ...]]]],
    policy: RLBatchPolicy,
    *,
    n_iterations: int,
    queue_out: queue.Queue[RolloutIteration],
    stop: threading.Event,
    on_iteration: Callable[[], None],
    sync_gate: threading.Event | None = None,
    max_frames: int = 10_000_000,
    progress_every: int = 600,
) -> None:
    """Persistent, iteration-oriented drive loop (the ``drive_vec`` sibling for RL).

    ``build_sessions`` returns freshly-constructed (not entered) Sessions + their
    ``Matchup``s + per-boot model ports; ``drive_rl`` enters them, starts every match
    concurrently, then steps all boots on a thread pool one lockstep frame at a time,
    feeding ``policy`` and relaying each finished :class:`RolloutIteration` to
    ``queue_out``. ``on_iteration`` runs at every boundary (after the sync gate) to copy
    the freshly-advanced EMA into the behavior net. In sync mode (``sync_gate`` set) the
    loop blocks on the gate after each put, so the collector never runs ahead of the
    learner (lag 0). A boot that dies is dropped; if fewer than half remain the whole
    wave reboots (its slots reset). Exceptions propagate (the learner thread sets
    ``stop`` on its own failure; a drive-loop death surfaces via ``queue_out`` going
    empty), and every entered Session is torn down on the way out."""
    entered: list[Session] = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        try:
            sessions, matchups, ports_of, done, last_frame = _boot_wave(build_sessions, entered, pool)
            n0 = len(sessions)
            logger.info(f"drive_rl: {sum(not d for d in done)}/{n0} boots up; target {n_iterations} iterations")
            on_iteration()  # prime the behavior net (EMA -> act_net) before the first collect

            emitted = 0
            t0 = time.monotonic()
            for t in range(max_frames):
                if stop.is_set() or emitted >= n_iterations:
                    break
                live = [i for i in range(len(sessions)) if not done[i]]
                if len(live) < max(1, n0 // 2):
                    logger.warning(f"drive_rl: only {len(live)}/{n0} boots alive at frame {t}; rebooting wave")
                    _teardown(entered)
                    policy.reset_slots(list(policy.state))
                    sessions, matchups, ports_of, done, last_frame = _boot_wave(build_sessions, entered, pool)
                    n0 = len(sessions)
                    continue
                if progress_every and t > 0 and t % progress_every == 0:
                    logger.info(f"drive_rl: frame {t} | live {len(live)}/{n0} | {t / (time.monotonic() - t0):.0f} f/s")

                for i in live:  # refresh injected live stage (instant-restart randomizes per match)
                    meta = last_frame[i]["_matchup"]
                    meta["stage"] = last_frame[i].get("stage", meta["stage"])
                obs = {Slot(i, p): last_frame[i] for i in live for p in ports_of[i]}
                inputs = policy(t, obs)
                futs = {i: pool.submit(sessions[i].step, {p: inputs[Slot(i, p)] for p in ports_of[i]}) for i in live}
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
                    _put(queue_out, iteration, stop)
                    emitted += 1
                    if stop.is_set():
                        break
                    if sync_gate is not None:
                        while not sync_gate.wait(timeout=_POLL_SECONDS):
                            if stop.is_set():
                                return
                        sync_gate.clear()
                    on_iteration()
            logger.info(f"drive_rl: done after {emitted} iterations")
        finally:
            _teardown(entered)


_POLL_SECONDS = 0.05


def _put(q: queue.Queue[RolloutIteration], item: RolloutIteration, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            q.put(item, timeout=_POLL_SECONDS)
            return
        except queue.Full:
            continue


def _boot_wave(
    build_sessions: Callable[[], tuple[list[Session], list[Matchup], list[tuple[int, ...]]]],
    entered: list[Session],
    pool: ThreadPoolExecutor,
) -> tuple[list[Session], list[Matchup], list[tuple[int, ...]], list[bool], list[dict]]:
    """Enter + concurrently start a fresh wave of Sessions (appending each to ``entered``
    for guaranteed teardown). A boot that fails to reach IN_GAME is marked done; the
    survivors set the shared t=0."""
    sessions, matchups, ports_of = build_sessions()
    for s in sessions:
        s.__enter__()
        entered.append(s)
    done = [False] * len(sessions)
    last_frame: list[dict] = [{} for _ in sessions]
    meta = [
        {"stage": int(m.stage.value), "character": {pl.port: int(pl.character.value) for pl in m.players}}
        for m in matchups
    ]
    futs = {i: pool.submit(s.start_match, m) for i, (s, m) in enumerate(zip(sessions, matchups, strict=True))}
    for i, fut in futs.items():
        try:
            f0 = fut.result()
            f0["_matchup"] = meta[i]
            last_frame[i] = f0
        except Exception as e:  # noqa: BLE001
            logger.warning(f"drive_rl: boot {i} start failed: {e!r}")
            done[i] = True
    return sessions, matchups, ports_of, done, last_frame


def _teardown(entered: list[Session]) -> None:
    while entered:
        s = entered.pop()
        try:
            s.__exit__(None, None, None)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"drive_rl: session teardown error: {e!r}")
