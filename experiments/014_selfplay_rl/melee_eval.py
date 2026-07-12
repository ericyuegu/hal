"""Self-play RL judge: head-to-head (EMA vs frozen IL) + vs-CPU, the entry that decides G3.

Two modes over one shared acting path:

* **head-to-head** (default when ``--ckpt`` given) — the checkpoint's EMA policy plays the
  frozen warm-start IL policy, both ports model-driven (cpu_level 0), over prior matchups
  with instant-restart. Port assignment ALTERNATES by boot parity (ema on p1 in even boots,
  p2 in odd) so port advantage washes out; winner attribution flips with it. Reports ema
  win rate (ties excluded from the denominator, still counted), mean stock diff, damage/min,
  bootstrap 95% CIs, and the ``G3 H2H`` verdict.
* **vs-CPU** (``--vs-cpu``) — the evaluated policy (EMA with ``--ckpt``, raw IL with
  ``--il-only``) vs a level-9 CPU over the prior sweep, pooled into the repo's standard
  ``vs_cpu_metrics``. ``--il-only`` pins a baseline; ``--ckpt --baseline <json>`` reports
  deltas and the ``G3 VS-CPU`` no-regression verdict.

The acting is the collector's KV-cached stepping (``NetActingPolicy`` + ``_token_features``)
behind a handle seam — ``EvalBatchPolicy`` is that path stripped of reward/stream/iteration
bookkeeping. Two handles ("ema", "il") for head-to-head; one for vs-CPU.

    uv run experiments/014_selfplay_rl/melee_eval.py --ckpt runs/<run>/latest.pt --h2h-matches 50
    uv run experiments/014_selfplay_rl/melee_eval.py --ckpt runs/<run>/latest.pt --vs-cpu --baseline il.json
    uv run experiments/014_selfplay_rl/melee_eval.py --il-only --vs-cpu
"""

import dataclasses
import json
import subprocess
import time
import warnings
from collections import deque
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tyro
from loguru import logger
from melee import Character
from melee_collector import ActingPolicy
from melee_collector import NetActingPolicy
from melee_collector import _token_features
from nets_melee import PolicyValueNet
from nets_melee import load_il_policy
from rl_config import MeleeRLConfig

from hal.eval.cross_stage import FRAMES_PER_MINUTE
from hal.eval.cross_stage import PRIOR_SWEEP_SEED_STAGE
from hal.eval.cross_stage import STARTING_STOCKS
from hal.eval.cross_stage import sweep_vs_cpu_prior
from hal.eval.cross_stage import vs_cpu_metrics
from hal.eval.harness import SessionConfig
from hal.eval.harness import default_session_cfg
from hal.eval.harness import run_matches_vec
from hal.eval.matchups import matchups_for
from hal.eval.scoring import MatchSummary
from hal.eval.scoring import summarize_trajectory
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.trajectory import Trajectory
from hal.sim.vec import Slot
from hal.sim.vec import VecMatch
from hal.training.canonical import flatten_canonical_frame
from hal.training.closed_loop import PORT_TO_PREFIX
from hal.training.closed_loop import live_batch_from_rolling
from hal.training.features import NEUTRAL_ACTION
from hal.training.features import Context
from hal.training.features import action_vec_to_controller
from hal.training.features import preprocess
from hal.training.stats import load_consolidated_stats

_BOOTSTRAP_DRAWS = 10_000
_CI_PCTILES = (2.5, 97.5)


# =============================================================================
# Attribution + statistics (pure; no torch/Dolphin — unit-tested in test_rl_eval.py)
# =============================================================================
def ema_port_for_boot(boot: int) -> int:
    """Which port the EMA policy drives in this boot: p1 in even boots, p2 in odd. The
    single source shared by the acting handle assignment and the winner attribution, so the
    two can never disagree (the classic port-alternation attribution bug)."""
    return 1 if boot % 2 == 0 else 2


@dataclass(frozen=True, slots=True)
class H2HMatch:
    """One finished head-to-head match, attributed to ema/il (never p1/p2)."""

    ema_stocks_left: int
    il_stocks_left: int
    ema_damage_dealt: float
    il_damage_dealt: float
    frames: int


def attribute_h2h(summary: MatchSummary, ema_port: int) -> H2HMatch:
    """Map a port-keyed ``MatchSummary`` onto ema/il given which port ema drove. Damage
    dealt by a side is the damage the OTHER side received."""
    if ema_port == 1:
        return H2HMatch(
            ema_stocks_left=summary.p1_stocks_left,
            il_stocks_left=summary.p2_stocks_left,
            ema_damage_dealt=summary.p2_damage_taken,
            il_damage_dealt=summary.p1_damage_taken,
            frames=summary.frames,
        )
    return H2HMatch(
        ema_stocks_left=summary.p2_stocks_left,
        il_stocks_left=summary.p1_stocks_left,
        ema_damage_dealt=summary.p1_damage_taken,
        il_damage_dealt=summary.p2_damage_taken,
        frames=summary.frames,
    )


def readout_from_traj(traj: Trajectory, ema_port: int) -> H2HMatch | None:
    """Attribute one match trajectory, or ``None`` if it must be discarded: a segment with
    no finite stock reading on either port (the deciding frame is non-finite) or fewer than
    two frames can't be scored honestly, so it is dropped (and counted by the caller)."""
    if len(traj) < 2:
        return None
    if not (np.isfinite(traj.post[1]["stock"]).any() and np.isfinite(traj.post[2]["stock"]).any()):
        return None
    return attribute_h2h(summarize_trajectory(traj), ema_port)


def _bootstrap_percentiles(samples: np.ndarray) -> tuple[float, float]:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(finite, _CI_PCTILES)
    return (float(lo), float(hi))


def _resample_indices(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(_BOOTSTRAP_DRAWS, n))


def summarize_h2h(matches: Sequence[H2HMatch], *, n_discarded: int, seed: int) -> dict:
    """Reduce attributed matches to the head-to-head report + G3 verdict.

    Win = ema has more stocks left; ties (equal stocks) are counted separately and EXCLUDED
    from the win-rate denominator. Bootstrap (10k seeded draws, resampling whole matches):
    95% CI on the win rate (ties dropped per resample) and on the mean stock diff. ``G3 H2H``
    PASSes iff the win-rate CI excludes 0.5 AND the mean stock diff (ema - il) > 0."""
    if not matches:
        raise ValueError("summarize_h2h called with no matches")
    stock_diff = np.array([m.ema_stocks_left - m.il_stocks_left for m in matches], dtype=np.float64)
    win = np.where(stock_diff > 0, 1.0, np.where(stock_diff < 0, 0.0, np.nan))  # nan = tie
    decided = win[np.isfinite(win)]
    n_ties = int(np.isnan(win).sum())
    win_rate = float(decided.mean()) if decided.size else float("nan")
    mean_stock_diff = float(stock_diff.mean())

    idx = _resample_indices(len(matches), seed)
    with warnings.catch_warnings():  # all-tie resamples -> nan win rate (empty slice), filtered below
        warnings.simplefilter("ignore", category=RuntimeWarning)
        wr_samples = np.nanmean(win[idx], axis=1)
    sd_samples = stock_diff[idx].mean(axis=1)
    win_ci = _bootstrap_percentiles(wr_samples)
    stock_ci = _bootstrap_percentiles(sd_samples)

    total_min = sum(m.frames for m in matches) / FRAMES_PER_MINUTE
    ema_dmg_min = sum(m.ema_damage_dealt for m in matches) / total_min if total_min else 0.0
    il_dmg_min = sum(m.il_damage_dealt for m in matches) / total_min if total_min else 0.0

    # PASS requires a FINITE CI that excludes 0.5 — an all-ties run yields a nan CI, which
    # must read as "no evidence", not as "excludes 0.5" via a nan comparison quirk.
    ci_excludes_half = bool(np.isfinite(win_ci).all()) and not (win_ci[0] <= 0.5 <= win_ci[1])
    g3 = "PASS" if (ci_excludes_half and mean_stock_diff > 0) else "FAIL"
    return {
        "n_matches": len(matches),
        "n_ties": n_ties,
        "n_discarded": n_discarded,
        "ema_win_rate": win_rate,
        "win_rate_ci95": list(win_ci),
        "mean_stock_diff": mean_stock_diff,
        "stock_diff_ci95": list(stock_ci),
        "ema_damage_per_min": ema_dmg_min,
        "il_damage_per_min": il_dmg_min,
        "g3_h2h": g3,
    }


def vs_cpu_net_stock_rate(summaries: Sequence[MatchSummary]) -> float:
    """Pooled (frame-weighted) net stocks per game-minute for an ego-on-p1 vs-CPU sweep:
    ``(stocks_taken - stocks_lost) / minutes``. The single scalar the G3 no-regression gate
    is judged on."""
    total_min = sum(s.frames for s in summaries) / FRAMES_PER_MINUTE
    if total_min == 0:
        return 0.0
    taken = sum(STARTING_STOCKS - s.p2_stocks_left for s in summaries)
    lost = sum(STARTING_STOCKS - s.p1_stocks_left for s in summaries)
    return (taken - lost) / total_min


def _bootstrap_net_rate(summaries: Sequence[MatchSummary], seed: int) -> np.ndarray:
    """Bootstrap distribution of the pooled net-stock rate (ratio estimator resampled over
    whole matches)."""
    taken = np.array([STARTING_STOCKS - s.p2_stocks_left for s in summaries], dtype=np.float64)
    lost = np.array([STARTING_STOCKS - s.p1_stocks_left for s in summaries], dtype=np.float64)
    frames = np.array([s.frames for s in summaries], dtype=np.float64)
    idx = _resample_indices(len(summaries), seed)
    minutes = frames[idx].sum(axis=1) / FRAMES_PER_MINUTE
    net = (taken[idx].sum(axis=1) - lost[idx].sum(axis=1)) / np.where(minutes > 0, minutes, np.nan)
    return net


def g3_vs_cpu_pass(ckpt_summaries: Sequence[MatchSummary], baseline_net_rate: float, seed: int) -> bool:
    """No-regression criterion: PASS iff the ckpt's net-stock-rate bootstrap 95% CI upper
    bound is at least the baseline's point estimate — i.e. the ckpt is NOT significantly
    worse than the baseline (a regression only fails when the whole CI sits below baseline)."""
    if not ckpt_summaries:
        return False
    _, hi = _bootstrap_percentiles(_bootstrap_net_rate(ckpt_summaries, seed))
    return bool(hi >= baseline_net_rate)


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True))


# =============================================================================
# Acting policy (torch): the collector's KV path, stripped of training bookkeeping
# =============================================================================
@dataclass(slots=True)
class _EvalSlot:
    handle: str
    row: int
    ego_port: int
    flat_hist: list[dict] = field(default_factory=list)
    ego_hist: list[np.ndarray] = field(default_factory=list)
    last_id: int | None = None


class EvalBatchPolicy:
    """``BatchPolicy`` that acts with one or more ``ActingPolicy`` handles, batched over
    slots, with NO reward/logp/stream/iteration bookkeeping.

    Mirrors ``RLBatchPolicy``'s three-pass frame step (reset detect + token build, one
    batched incremental decode per handle, periodic drift-reset rebuild) but reuses the
    collector's ``NetActingPolicy`` for the KV/sampling math verbatim — only the collection
    plumbing is dropped. ``handle_of`` routes each slot to a handle ("ema"/"il" for
    head-to-head, a single handle for vs-CPU); rows within a handle's caches are assigned
    lazily in slot-appearance order. Construct fresh per wave (caches must not leak)."""

    def __init__(
        self,
        *,
        handles: Mapping[str, ActingPolicy],
        handle_of: Callable[[Slot], str],
        stats: dict,
        L_ctx: int,
        refresh_every: int,
    ) -> None:
        self.handles = dict(handles)
        self.handle_of = handle_of
        self.stats = stats
        self.L_ctx = L_ctx
        self.refresh_every = refresh_every
        self._rows: dict[str, int] = {name: 0 for name in self.handles}
        self.state: dict[Slot, _EvalSlot] = {}
        self._global_frame = 0

    def _slot(self, slot: Slot) -> _EvalSlot:
        st = self.state.get(slot)
        if st is None:
            handle = self.handle_of(slot)
            if handle not in self.handles:
                raise KeyError(f"slot {slot} routed to unknown handle {handle!r}")
            row = self._rows[handle]
            self._rows[handle] += 1
            st = _EvalSlot(handle=handle, row=row, ego_port=slot.port)
            self.state[slot] = st
        return st

    def _reset(self, st: _EvalSlot) -> None:
        st.flat_hist.clear()
        st.ego_hist.clear()
        self.handles[st.handle].reset_slot(st.row)

    def __call__(self, frame_index: int, obs: Mapping[Slot, dict]) -> Mapping[Slot, object]:
        live = list(obs)
        if not live:
            return {}
        by_handle: dict[str, list[Slot]] = {}
        for slot in live:
            st = self._slot(slot)
            by_handle.setdefault(st.handle, []).append(slot)

        # PASS 1: reset detection (id drop = instant-restart into a new match), flatten, token.
        tokens: dict[str, list[dict[str, np.ndarray]]] = {h: [] for h in by_handle}
        for slot in live:
            st = self.state[slot]
            frame = obs[slot]
            fid = int(frame["id"])
            if st.last_id is not None and fid < st.last_id:
                self._reset(st)
            st.last_id = fid
            cur_flat = flatten_canonical_frame(frame)
            st.flat_hist.append(cur_flat)
            if len(st.flat_hist) > self.L_ctx:
                st.flat_hist.pop(0)
            prev_action = st.ego_hist[-1] if st.ego_hist else NEUTRAL_ACTION
            tokens[st.handle].append(_token_features(cur_flat, prev_action, PORT_TO_PREFIX[st.ego_port]))

        # PASS 2: one batched incremental decode per handle; record only the ego action.
        actions: dict[Slot, object] = {}
        for handle, slots in by_handle.items():
            rows = torch.tensor([self.state[s].row for s in slots], dtype=torch.long)
            _, _, vecs = self.handles[handle].step(rows, self._collate_tokens(tokens[handle]))
            for j, slot in enumerate(slots):
                st = self.state[slot]
                vec = vecs[j]
                st.ego_hist.append(vec)
                if len(st.ego_hist) > self.L_ctx:
                    st.ego_hist.pop(0)
                actions[slot] = action_vec_to_controller(vec)

        # PASS 3: periodic rebuild (RoPE-drift reset + max_pos bound on long matches).
        if self._global_frame > 0 and self._global_frame % self.refresh_every == 0:
            for handle, slots in by_handle.items():
                rows = torch.tensor([self.state[s].row for s in slots], dtype=torch.long)
                self.handles[handle].rebuild(rows, self._collate_windows(slots))
        self._global_frame += 1
        return actions

    def _collate_tokens(self, tokens: list[dict[str, np.ndarray]]) -> Context:
        stacked = {k: np.concatenate([t[k] for t in tokens], axis=0) for k in tokens[0]}
        return Context(features=preprocess(stacked, self.stats), ctx_pad=torch.zeros(len(tokens), dtype=torch.long))

    def _collate_windows(self, slots: list[Slot]) -> Context:
        per_slot: list[dict[str, np.ndarray]] = []
        ctx_pad: list[int] = []
        for slot in slots:
            st = self.state[slot]
            per_slot.append(
                live_batch_from_rolling(st.flat_hist, st.ego_hist[:-1], PORT_TO_PREFIX[st.ego_port], self.L_ctx)
            )
            ctx_pad.append(max(0, self.L_ctx - len(st.flat_hist)))
        stacked = {k: np.concatenate([d[k] for d in per_slot], axis=0) for k in per_slot[0]}
        return Context(features=preprocess(stacked, self.stats), ctx_pad=torch.tensor(ctx_pad, dtype=torch.long))


# =============================================================================
# Net loading
# =============================================================================
@dataclass(frozen=True, slots=True)
class LoadedPolicy:
    """A ready-to-act net + the arch/data context needed to build handles and sweeps."""

    net: PolicyValueNet
    L_ctx: int
    refresh_every: int
    stats: dict
    warm_start: str


def _load(ckpt: Path | None, warm_start_name: str, refresh_every: int, device: str) -> LoadedPolicy:
    """Build the evaluated net: the warm-start IL net (``--il-only``) or the checkpoint's EMA
    weights loaded onto that architecture (``--ckpt``). Stats come from the warm-start's
    training data root (same source as ``melee_train``)."""
    warm_path = Path("runs") / warm_start_name / "final.pt"
    if not warm_path.is_file():
        raise FileNotFoundError(f"warm-start checkpoint not found: {warm_path}")
    data_root = torch.load(warm_path, map_location="cpu", weights_only=False)["cfg"]["data_root"]
    stats = load_consolidated_stats(Path(data_root) / "stats.json")
    net, cfg = load_il_policy(warm_path)
    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(state["ema"])  # EMA weights == full param+buffer state_dict
        logger.info(f"loaded EMA weights from {ckpt} (train iter {state.get('step')})")
    net = net.to(device).eval().requires_grad_(False)
    return LoadedPolicy(net=net, L_ctx=cfg.L_ctx, refresh_every=refresh_every, stats=stats, warm_start=warm_start_name)


def _acting(
    net: PolicyValueNet, *, n_slots: int, L_ctx: int, refresh_every: int, temp: float, seed: int, device: str
) -> NetActingPolicy:
    return NetActingPolicy(
        net, n_slots=n_slots, device=device, temp=temp, seed=seed, max_pos=L_ctx + refresh_every + 8
    )


# =============================================================================
# Head-to-head driver
# =============================================================================
# A failed boot (never reached IN_GAME) is re-queued this many times before being dropped.
_H2H_BOOT_RETRIES = 2

# run_matches_vec-compatible runner, injectable so the retry/parity plumbing is testable
# without Dolphin (see test_rl_eval.py).
H2HRunner = Callable[..., list[list[Trajectory]]]


def run_h2h(
    ema: LoadedPolicy,
    il: LoadedPolicy,
    *,
    session_cfg: SessionConfig,
    n_matches_target: int,
    n_boots: int,
    max_frames: int,
    temp: float,
    seed: int,
    device: str,
    base_slippi_port: int,
    runner: H2HRunner = run_matches_vec,
) -> dict:
    """EMA vs frozen IL over prior matchups until ``n_matches_target`` valid matches (or the
    matchup queue is exhausted).

    Parity discipline (retry-invariant by construction): each wave runs its whole chunk in ONE
    ``drive_vec`` pass — ``start_retries=0``, so ``run_matches_vec`` never re-drives a subset
    whose local ``Slot.match`` indices would restart at 0 and desync from the parity list. A
    boot that produced nothing is instead RE-QUEUED into a later wave under a fresh global boot
    index, with its parity re-derived from that new index on both sides: the factory's
    ``handle_of`` and the attribution below read the same per-wave ``ema_ports`` list, indexed
    by the same wave-local position. Ports rotate per wave (the rotation ``start_retries``
    used to provide) so a stuck, not-yet-reaped Dolphin can't collide with the next wave.

    Censoring: under instant-restart a boot's LAST segment is the match still in progress when
    the frame budget ran out (``drive_vec`` flushes it), so it is dropped as censored by
    construction — scoring it would count a partial as a finished match and bias the stats
    toward whoever happened to lead mid-match."""
    queue: deque[tuple[tuple[Character, Character], int]] = deque(
        (m, 0) for m in matchups_for(max(n_boots, n_matches_target))
    )
    readouts: list[H2HMatch] = []
    n_discarded = 0
    n_censored = 0
    n_dropped = 0  # boots whose matchup exhausted its requeues without ever producing a match
    boot0 = 0  # global boot counter: the parity identity, advancing over every boot launched
    wave = 0
    while len(readouts) < n_matches_target and queue:
        chunk = [queue.popleft() for _ in range(min(n_boots, len(queue)))]
        ema_ports = [ema_port_for_boot(boot0 + j) for j in range(len(chunk))]
        matches = [
            VecMatch(
                matchup=Matchup(
                    stage=PRIOR_SWEEP_SEED_STAGE,
                    players=(
                        PlayerSetup(port=1, character=ego_char, cpu_level=0),
                        PlayerSetup(port=2, character=opp_char, cpu_level=0),
                    ),
                ),
                model_ports=(1, 2),
            )
            for (ego_char, opp_char), _ in chunk
        ]

        def factory(ema_ports: list[int] = ema_ports, wave: int = wave) -> EvalBatchPolicy:
            ema_handle = _acting(
                ema.net,
                n_slots=len(ema_ports),
                L_ctx=ema.L_ctx,
                refresh_every=ema.refresh_every,
                temp=temp,
                seed=seed + wave,
                device=device,
            )
            il_handle = _acting(
                il.net,
                n_slots=len(ema_ports),
                L_ctx=il.L_ctx,
                refresh_every=il.refresh_every,
                temp=temp,
                seed=seed + wave,
                device=device,
            )
            return EvalBatchPolicy(
                handles={"ema": ema_handle, "il": il_handle},
                handle_of=lambda s: "ema" if s.port == ema_ports[s.match] else "il",
                stats=ema.stats,
                L_ctx=ema.L_ctx,
                refresh_every=ema.refresh_every,
            )

        boots = runner(
            session_cfg,
            matches,
            factory,
            max_frames=max_frames,
            max_parallel=len(chunk),
            base_slippi_port=base_slippi_port + (wave % 8) * n_boots,
            start_retries=0,  # internal subset retries would desync Slot.match from ema_ports
        )
        for j, boot in enumerate(boots):
            matchup, attempts = chunk[j]
            if not boot:
                if attempts < _H2H_BOOT_RETRIES:
                    queue.append((matchup, attempts + 1))
                    logger.warning(f"h2h: boot {boot0 + j} produced no match; re-queued (attempt {attempts + 1})")
                else:
                    n_dropped += 1
                    logger.warning(f"h2h: boot {boot0 + j} failed {attempts + 1} times; dropping its matchup")
                continue
            n_censored += 1  # the boot's final segment: in progress at the budget, never finished
            for traj in boot[:-1]:
                m = readout_from_traj(traj, ema_ports[j])
                if m is None:
                    n_discarded += 1
                else:
                    readouts.append(m)
        logger.info(
            f"h2h: wave {wave} (boots {boot0}..{boot0 + len(chunk) - 1}) -> "
            f"{len(readouts)} matches, {n_discarded} discarded, {n_censored} censored"
        )
        boot0 += len(chunk)
        wave += 1

    if not readouts:
        raise RuntimeError("head-to-head produced no valid matches — every segment was discarded or censored")
    if n_discarded:
        logger.warning(f"h2h: discarded {n_discarded} non-finite/degenerate match segment(s)")
    return {
        **summarize_h2h(readouts, n_discarded=n_discarded, seed=seed),
        "n_censored": n_censored,
        "n_dropped_boots": n_dropped,
    }


# =============================================================================
# vs-CPU driver
# =============================================================================
def run_vs_cpu(
    pol: LoadedPolicy,
    *,
    session_cfg: SessionConfig,
    n_matchups: int,
    max_parallel: int,
    max_frames: int,
    cpu_level: int,
    temp: float,
    seed: int,
    device: str,
    baseline: dict | None,
) -> dict:
    """Evaluated policy vs a level-``cpu_level`` CPU over the prior sweep (ego on p1), pooled
    into the repo's ``vs_cpu_metrics``. Reuses ``sweep_vs_cpu_prior`` directly — our
    ``EvalBatchPolicy`` is a drop-in ``BatchPolicy`` factory (single handle). With a baseline
    JSON, reports per-metric deltas + the ``G3 VS-CPU`` no-regression verdict."""
    wave = {"i": 0}

    def factory() -> EvalBatchPolicy:
        handle = _acting(
            pol.net,
            n_slots=max_parallel,
            L_ctx=pol.L_ctx,
            refresh_every=pol.refresh_every,
            temp=temp,
            seed=seed + wave["i"],
            device=device,
        )
        wave["i"] += 1
        return EvalBatchPolicy(
            handles={"pol": handle},
            handle_of=lambda s: "pol",
            stats=pol.stats,
            L_ctx=pol.L_ctx,
            refresh_every=pol.refresh_every,
        )

    result = sweep_vs_cpu_prior(
        factory,
        session_cfg=session_cfg,
        n_matchups=n_matchups,
        max_parallel=max_parallel,
        cpu_level=cpu_level,
        max_frames=max_frames,
    )
    summaries = [s for _, _, s in result if s is not None]
    metrics = vs_cpu_metrics(result)
    net_rate = vs_cpu_net_stock_rate(summaries) if summaries else float("nan")
    report: dict = {"vs_cpu_metrics": metrics, "net_stock_rate": net_rate, "cpu_level": cpu_level}
    if summaries:
        report["net_stock_rate_ci95"] = list(_bootstrap_percentiles(_bootstrap_net_rate(summaries, seed)))
    if baseline is not None:
        base_metrics = baseline["vs_cpu_metrics"]
        report["baseline"] = base_metrics
        report["deltas"] = {k: metrics[k] - base_metrics[k] for k in metrics if k in base_metrics}
        base_rate = float(baseline["net_stock_rate"])
        report["baseline_net_stock_rate"] = base_rate
        report["g3_vs_cpu"] = "PASS" if g3_vs_cpu_pass(summaries, base_rate, seed) else "FAIL"
        report["g3_vs_cpu_criterion"] = "net-stock-rate bootstrap 95% CI upper bound >= baseline net-stock-rate"
    return report


# =============================================================================
# Entry
# =============================================================================
@dataclass
class Args:
    ckpt: Path | None = None  # RL checkpoint; its EMA weights are the evaluated policy
    il_only: bool = False  # evaluate the raw warm-start IL policy instead (pins baselines)
    warm_start: str = MeleeRLConfig().warm_start  # 012 run name under runs/ (IL anchor / arch source)
    vs_cpu: bool = False  # run vs-CPU instead of head-to-head
    h2h_matches: int = 50  # target number of valid head-to-head matches
    n_boots: int = 4  # parallel Dolphins (head-to-head boots / vs-CPU wave width)
    sweep_matches: int = 40  # vs-CPU: number of prior matchups (boots) in the sweep
    cpu_level: int = 9  # vs-CPU opponent level
    max_frames: int = 30_000  # per-boot frame budget (spans a boot's instant-restart matches)
    refresh_every: int = 64  # KV rebuild period (drift reset + max_pos bound)
    temp: float = 1.0  # sampling temperature (1.0 == training collection)
    seed: int = 0
    baseline: Path | None = None  # vs-CPU baseline JSON (from a prior --il-only run) for deltas + G3
    out: Path | None = None  # JSON output path (default runs/<derived>/eval_<mode>_<ts>.json)
    device: str | None = None
    base_slippi_port: int = 55000
    wandb_run: str | None = None  # existing W&B run id to append eval metrics to
    global_step: int | None = None  # W&B global_step for the appended metrics (default: ckpt's transitions)


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"git sha unavailable: {e!r}")
        return "unknown"


def _log_wandb(run_id: str, report: dict, global_step: int) -> None:
    """Append eval metrics to an existing W&B run. Repo conventions: W&B's own step stays
    the timestamp; the training step rides as ``global_step`` data + ``define_metric`` (never
    ``step=``), so async evals back-date onto the training axis; keys one level deep."""
    import wandb

    metrics: dict[str, float] = {}
    if report["mode"] == "h2h":
        metrics["eval/h2h_win_rate"] = float(report["ema_win_rate"])
        metrics["eval/h2h_stock_diff"] = float(report["mean_stock_diff"])
        metrics["eval/h2h_n_matches"] = float(report["n_matches"])
    else:
        for k, v in report["vs_cpu_metrics"].items():
            metrics[f"eval/vs_cpu_{k}"] = float(v)
        metrics["eval/vs_cpu_net_stock_rate"] = float(report["net_stock_rate"])
    run = wandb.init(project="hal", id=run_id, resume="must")
    wandb.define_metric("global_step")
    wandb.define_metric("eval/*", step_metric="global_step")
    wandb.log({"global_step": global_step, **metrics})
    run.finish()
    logger.info(f"appended {len(metrics)} eval metric(s) to W&B run {run_id} at global_step {global_step}")


def _default_out(args: Args, mode: str, policy: str) -> Path:
    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    if args.ckpt is not None:
        base = args.ckpt.parent
    else:
        base = Path("runs") / f"{args.warm_start}_il_eval"
    return base / f"eval_{mode}_{policy}_{stamp}.json"


def main(args: Args) -> None:
    if (args.ckpt is None) == (not args.il_only):
        raise ValueError("pass exactly one of --ckpt <path> or --il-only")
    if not args.vs_cpu and args.il_only:
        raise ValueError("--il-only has no head-to-head meaning (IL vs IL); pair it with --vs-cpu")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = "il" if args.il_only else "ema"
    mode = "vs_cpu" if args.vs_cpu else "h2h"

    if args.wandb_run is not None and args.global_step is None and args.ckpt is None:
        raise ValueError("--wandb-run with --il-only requires an explicit --global-step (no ckpt transitions counter)")

    warm_start_name = args.warm_start
    ckpt_transitions: int | None = None
    if args.ckpt is not None:
        ckpt_state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        ckpt_transitions = int(ckpt_state["transitions"])
        ckpt_warm = ckpt_state["cfg"]["rl"]["warm_start"]
        if ckpt_warm != warm_start_name:
            logger.warning(
                f"using checkpoint's warm_start {ckpt_warm!r} (overriding --warm-start {warm_start_name!r})"
            )
            warm_start_name = ckpt_warm

    logger.info(f"eval mode={mode} policy={policy} device={device} temp={args.temp} seed={args.seed}")
    t0 = time.monotonic()
    session_cfg = default_session_cfg(instant_match_restart=True)

    if not args.vs_cpu:
        ema = _load(args.ckpt, warm_start_name, args.refresh_every, device)
        il = _load(None, warm_start_name, args.refresh_every, device)
        result = run_h2h(
            ema,
            il,
            session_cfg=session_cfg,
            n_matches_target=args.h2h_matches,
            n_boots=args.n_boots,
            max_frames=args.max_frames,
            temp=args.temp,
            seed=args.seed,
            device=device,
            base_slippi_port=args.base_slippi_port,
        )
        logger.info(f"G3 H2H: {result['g3_h2h']}")
    else:
        pol = _load(args.ckpt, warm_start_name, args.refresh_every, device)
        baseline = json.loads(args.baseline.read_text()) if args.baseline is not None else None
        result = run_vs_cpu(
            pol,
            session_cfg=session_cfg,
            n_matchups=args.sweep_matches,
            max_parallel=args.n_boots,
            max_frames=args.max_frames,
            cpu_level=args.cpu_level,
            temp=args.temp,
            seed=args.seed,
            device=device,
            baseline=baseline,
        )
        if "g3_vs_cpu" in result:
            logger.info(f"G3 VS-CPU: {result['g3_vs_cpu']} ({result['g3_vs_cpu_criterion']})")

    report = {
        "mode": mode,
        "policy": policy,
        "ckpt": str(args.ckpt) if args.ckpt is not None else None,
        "warm_start": warm_start_name,
        "temp": args.temp,
        "seed": args.seed,
        "git_sha": _git_sha(),
        "wall_clock_s": time.monotonic() - t0,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(args).items()},
        **result,
    }
    out = args.out or _default_out(args, mode, policy)
    write_report(out, report)
    logger.info(f"wrote {out} ({report['wall_clock_s']:.0f}s wall-clock)")
    for line in json.dumps({k: v for k, v in report.items() if k != "config"}, indent=2).splitlines():
        logger.info(line)
    if args.wandb_run is not None:
        step = args.global_step if args.global_step is not None else ckpt_transitions
        assert step is not None  # guarded at entry: --il-only + --wandb-run requires --global-step
        _log_wandb(args.wandb_run, report, step)


if __name__ == "__main__":
    main(tyro.cli(Args))
