"""Melee self-play PPO entry point (gate: sync-mode end-to-end).

Wires the pieces landed across M3/M4 into one live loop: a warm-started
``PolicyValueNet`` (the learner), a frozen IL anchor, an EMA behavior snapshot that
acts in the collector, ``drive_rl`` owning the Dolphin self-play sessions, and the
windowed Melee PPO update. Two phases: a value-head-only warm-up (the critic learns
``V(s)`` while the actor stays byte-identical IL), then full PPO with a KL-to-IL anchor.

Threading / pipeline choice — DIRECT WIRING, not ``run_pipeline``: ``run_pipeline``'s
``collect()`` is a one-shot call returning a payload, but ``drive_rl`` is a *persistent*
loop that owns Dolphin Sessions across every iteration (booting them per-collect would
throw away instant-restart and pay menu navigation each time) and needs boundary hooks
(EMA→act_net copy, the sync gate) that live *inside* the loop. So the collector runs in
its own thread, pushing ``RolloutIteration``s to a ``queue.Queue(maxsize=1)`` that the
main-thread learner drains — preserving ``run_pipeline``'s guarantees: bounded lag 1
(the queue), lag 0 under ``--pipeline.no-overlap`` (the sync gate), and fail-loud
bidirectional shutdown (a stop event both sides honor).

    uv run experiments/014_selfplay_rl/melee_train.py --smoke
    uv run experiments/014_selfplay_rl/melee_train.py --wandb --run-name my_run
"""

import copy
import dataclasses
import queue
import signal
import threading
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import numpy as np
import torch
import tyro
from loguru import logger
from melee_collector import BuildBoot
from melee_collector import CollectStats
from melee_collector import IterationPayload
from melee_collector import NetActingPolicy
from melee_collector import RLBatchPolicy
from melee_collector import drive_rl
from melee_collector import matchup_meta
from nets_melee import ArchConfig
from nets_melee import PolicyValueNet
from nets_melee import load_il_policy
from ppo import melee_ppo_update
from ppo import value_warmup_update
from rl_config import EMAConfig
from rl_config import MeleeRLConfig
from rl_config import PipelineConfig
from rl_config import PPOConfig
from rl_config import RewardConfig
from rollout import FinalizedStream
from rollout import RolloutIteration
from rollout import Window
from rollout import build_windows
from rollout import collate_windows
from rollout import gae_inputs
from rollout import scatter_gae

from hal.eval.cross_stage import PRIOR_SWEEP_SEED_STAGE
from hal.eval.harness import SessionConfig
from hal.eval.harness import build_session
from hal.eval.harness import default_session_cfg
from hal.eval.matchups import matchups_for
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.session import Session
from hal.sim.vec import Slot
from hal.training.checkpoints import save_checkpoint
from hal.training.ema import EMAWeights
from hal.training.runs import make_run_name
from hal.training.stats import load_consolidated_stats


def _melee_ppo() -> PPOConfig:
    """Melee PPO preset (overrides the shared ``PPOConfig`` defaults)."""
    return PPOConfig(
        lr=3e-5,
        clip=0.2,
        epochs=3,
        minibatch_size=16,  # WINDOWS per minibatch (each window is up to L_ctx transitions)
        gamma=0.997,
        gae_lambda=0.95,
        ent_coef=0.003,
        vf_coef=0.5,
        target_kl=0.015,
        max_grad_norm=0.5,
    )


@dataclass
class Args:
    rl: MeleeRLConfig = field(default_factory=MeleeRLConfig)
    ppo: PPOConfig = field(default_factory=_melee_ppo)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ema: EMAConfig = field(default_factory=lambda: EMAConfig(decay=0.995))
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    seed: int = 0
    temp: float = 1.0  # collection sampling temperature (1.0 keeps behavior == recompute policy)
    value_warmup_lr: float = 1e-3
    total_iterations: int = 2000
    ckpt_every_iters: int = 50
    base_slippi_port: int = 51461
    device: str | None = None  # default cuda if available
    wandb: bool = False
    run_name: str | None = None
    resume: str | None = None
    smoke: bool = False  # 2 boots, 5 iterations, sync, no wandb


def _apply_smoke(args: Args) -> Args:
    # rollout_frames=2048 -> ~512 transitions/stream (>= L_ctx=256), so burn-in windows get
    # full within-stream context. Expected: the FIRST PPO iteration (behavior == learner,
    # EMA just re-anchored) shows ratio_dev_epoch0 near the KV-drift floor (<0.01); later
    # iterations add genuine EMA policy lag on top.
    return dataclasses.replace(
        args,
        rl=dataclasses.replace(
            args.rl,
            n_boots=2,
            rollout_frames=2048,
            refresh_every=64,
            value_warmup_iters=2,
            reboot_every_iters=2,  # exercise wave-reboot matchup rotation within the 5-iter smoke
        ),
        pipeline=PipelineConfig(overlap=False),
        total_iterations=5,
        ckpt_every_iters=5,
        wandb=False,
    )


# --- learner-side rollout math ------------------------------------------------
def _all_windows(streams: tuple[FinalizedStream, ...], L_ctx: int, stride: int) -> list[Window]:
    windows: list[Window] = []
    for si, stream in enumerate(streams):
        windows.extend(build_windows(stream, L_ctx, stride=stride, stream_id=si))
    return windows


@torch.no_grad()
def _stream_values(
    net: PolicyValueNet, windows: list[Window], streams: tuple[FinalizedStream, ...], stats: dict, device: str
) -> list[np.ndarray]:
    """``V`` at every ``T+1`` frame of each stream, read from one batched forward over the
    windows. With burn-in a frame appears in several windows; its value is read from the
    one window that SCORES it (the scored spans partition each stream), so every frame is
    written exactly once and always with at least the scored-span context behind it."""
    batch = collate_windows(windows, stats).to(device)
    v = net.values(net.forward_full(batch.context)).cpu().numpy()  # [B_w, L]
    values = [np.zeros(s.n_transitions + 1, np.float64) for s in streams]
    for wi, w in enumerate(windows):
        take = w.scored
        values[w.stream_id][w.frame_pos[take]] = v[wi][take]
    return values


def _prepare_windows(
    net: PolicyValueNet, iteration: RolloutIteration, stats: dict, ppo: PPOConfig, device: str, *, stride: int
) -> tuple[list[Window], dict[str, float]]:
    """Value pass → GAE → scatter advantages/returns onto windows; return the scattered
    windows plus value diagnostics (``adv_std``, ``v_explained_var``)."""
    windows = _all_windows(iteration.streams, net.cfg.L_ctx, stride)
    if not windows:
        return [], {"adv_std": 0.0, "v_explained_var": 0.0}
    values = _stream_values(net, windows, iteration.streams, stats, device)
    gae = gae_inputs(list(iteration.streams), values, gamma=ppo.gamma, gae_lambda=ppo.gae_lambda)
    scattered = [scatter_gae(w, *gae[w.stream_id]) for w in windows]
    advs = np.concatenate([w.adv[w.valid] for w in scattered]) if scattered else np.zeros(0)
    rets = np.concatenate([w.returns[w.valid] for w in scattered]) if scattered else np.zeros(0)
    vals = np.concatenate([values[w.stream_id][w.frame_pos[w.valid]] for w in scattered])
    var_ret = float(np.var(rets)) if rets.size else 0.0
    v_ev = 1.0 - float(np.var(rets - vals)) / var_ret if var_ret > 0 else 0.0
    return scattered, {"adv_std": float(np.std(advs)) if advs.size else 0.0, "v_explained_var": v_ev}


def _rollout_stats(iteration: RolloutIteration) -> dict[str, float]:
    """Per-iteration rollout quality: mean step reward, completed-episode return/length,
    and damage-dealt / stocks-taken per game-minute (pooled over streams)."""
    all_rew = (
        np.concatenate([s.rew for s in iteration.streams if s.n_transitions])
        if iteration.n_transitions
        else np.zeros(0)
    )
    frames = sum(s.n_transitions for s in iteration.streams)
    dmg = stocks = 0.0
    for s in iteration.streams:
        opp = 2 if s.ego_port == 1 else 1
        for t in range(s.n_transitions):
            pa, cu = s.flat[t], s.flat[t + 1]
            op, oc = pa[f"p{opp}_percent"], cu[f"p{opp}_percent"]
            sp, sc = pa[f"p{opp}_stock"], cu[f"p{opp}_stock"]
            if np.isfinite(op) and np.isfinite(oc) and sc == sp and oc > op:
                dmg += oc - op
            if np.isfinite(sp) and np.isfinite(sc) and sc < sp:
                stocks += 1.0
    minutes = frames / 3600.0 if frames else float("nan")
    return {
        "reward_mean": float(all_rew.mean()) if all_rew.size else 0.0,
        "ep_return_mean": float(iteration.episode_returns.mean()) if iteration.episode_returns.size else float("nan"),
        "ep_len_mean": float(iteration.episode_lengths.mean()) if iteration.episode_lengths.size else float("nan"),
        "n_episodes": float(iteration.episode_returns.size),
        "dmg_dealt_per_min": dmg / minutes if minutes else 0.0,
        "stocks_taken_per_min": stocks / minutes if minutes else 0.0,
    }


# --- session wiring -----------------------------------------------------------
def wave_matchups(wave: int, n_boots: int) -> list[Matchup]:
    """The self-play ``Matchup``s for wave ``wave``: the ``wave``-th contiguous ``n_boots``
    slice of the training-prior matchups. ``matchups_for`` is prefix-stable, so
    ``matchups_for((wave + 1) * n_boots)[wave * n_boots:]`` is exactly this wave's slice —
    successive waves tile the prior with no overlap, so across wave reboots the run rotates
    through the FULL training matchup distribution rather than self-playing one fixed slice.
    Both ports model-driven (cpu_level 0), seeded on Battlefield (instant-restart randomizes
    the stage after)."""
    prior = matchups_for((wave + 1) * n_boots)[wave * n_boots :]
    return [
        Matchup(
            stage=PRIOR_SWEEP_SEED_STAGE,
            players=(
                PlayerSetup(port=1, character=ego_char, cpu_level=0),
                PlayerSetup(port=2, character=opp_char, cpu_level=0),
            ),
        )
        for ego_char, opp_char in prior
    ]


def _slot_matchups(matchups: list[Matchup], ports: tuple[int, ...]) -> dict[Slot, dict]:
    return {Slot(i, p): matchup_meta(m) for i, m in enumerate(matchups) for p in ports}


def _boot_builder(session_cfg: SessionConfig, n_boots: int, base_port: int, replay_root: Path | None) -> BuildBoot:
    """Per-``(boot, attempt)`` Session builder for ``drive_rl``. Each retry attempt
    rotates onto a fresh port block (``(attempt % 8) * n_boots`` — the same discipline as
    ``run_matches_vec``) so a stuck, not-yet-reaped Dolphin can't collide with the retry."""

    def build(i: int, attempt: int) -> Session:
        replay_dir = None
        if replay_root is not None:
            replay_dir = replay_root / f"boot_{i:03d}"
            replay_dir.mkdir(parents=True, exist_ok=True)
        port = base_port + (attempt % 8) * n_boots + i
        return build_session(session_cfg, slippi_port=port, replay_dir=replay_dir)

    return build


def _resnapshot_ema(ema: EMAWeights, module: torch.nn.Module) -> None:
    """Reset the EMA shadow to the module's current weights. Used at the warmup→PPO phase
    switch: the EMA is frozen through warmup (acting stays byte-identical IL), so it must
    re-anchor to the warmed learner before it starts trailing PPO updates."""
    ema.load_state_dict(
        {
            **{n: p.detach() for n, p in module.named_parameters()},
            **{n: b.detach() for n, b in module.named_buffers()},
        }
    )


def main(args: Args) -> None:
    if args.smoke:
        args = _apply_smoke(args)
    if args.temp != 1.0:
        raise ValueError(
            f"temp must be 1.0 for training, got {args.temp}: NetActingPolicy records temperature-scaled logp "
            "but melee_ppo_update recomputes with raw logits, so any other temp silently corrupts the PPO ratios"
        )
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ports = (1, 2)

    warm_start = Path("runs") / args.rl.warm_start / "final.pt"
    if not warm_start.is_file():
        raise FileNotFoundError(f"warm-start checkpoint not found: {warm_start}")
    data_root = torch.load(warm_start, map_location="cpu", weights_only=False)["cfg"]["data_root"]
    stats = load_consolidated_stats(Path(data_root) / "stats.json")

    learner, cfg = load_il_policy(warm_start)
    if not 1 <= args.rl.ppo_window_stride <= cfg.L_ctx:
        raise ValueError(f"--rl.ppo-window-stride must be in [1, L_ctx={cfg.L_ctx}], got {args.rl.ppo_window_stride}")
    learner = learner.to(device).train()
    il_net = copy.deepcopy(learner).to(device).eval().requires_grad_(False)
    act_net = copy.deepcopy(learner).to(device).eval().requires_grad_(False)
    ema = EMAWeights(learner, decay=args.ema.decay)

    opt_ppo = torch.optim.AdamW(learner.parameters(), lr=args.ppo.lr)
    opt_warm = torch.optim.AdamW(learner.value_head.parameters(), lr=args.value_warmup_lr)
    const_sched = torch.optim.lr_scheduler.LambdaLR(opt_ppo, lambda _: 1.0)  # save_checkpoint needs a scheduler
    gen = torch.Generator().manual_seed(args.seed)  # CPU: only drives torch.randperm (minibatch shuffle)

    run_name = args.run_name or make_run_name("014_selfplay_rl", f"d{cfg.d_model}-L{cfg.n_layers}", data_root, "rl")
    ckpt_dir = Path("runs") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    replay_root = ckpt_dir / "replays"

    counters = {"iter": 0, "transitions": 0, "empty_iters": 0}
    total_iters = args.total_iterations
    wandb_id = None
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        learner.load_state_dict(state["model"])
        opt_ppo.load_state_dict(state["opt"])
        opt_warm.load_state_dict(state["opt_warm"])
        ema.load_state_dict(state["ema"])
        counters["iter"] = state["step"]
        counters["transitions"] = state["transitions"]
        wandb_id = state.get("wandb_id")
        warmup_done = bool(state["value_warmup_done"])
        if warmup_done != (counters["iter"] >= args.rl.value_warmup_iters):
            raise ValueError(
                f"resume phase mismatch: checkpoint value_warmup_done={warmup_done} at iter={counters['iter']}, "
                f"but --rl.value-warmup-iters={args.rl.value_warmup_iters} implies the opposite — resume with the "
                "same warmup config the checkpoint was trained under"
            )
        logger.info(f"resumed from {args.resume} at iter={counters['iter']} transitions={counters['transitions']}")
        if args.smoke:  # a smoke resume just proves one more clean iteration off the checkpoint
            total_iters = counters["iter"] + 1
        if counters["iter"] >= total_iters:
            raise ValueError(f"resume iter {counters['iter']} >= total_iterations {total_iters}")
    ema.copy_to(act_net)  # behavior net starts at the (resumed) EMA weights
    # True once the warmup→PPO switch has re-anchored the EMA this process. A run resumed
    # PAST warmup skips re-anchoring: its checkpointed EMA is already the post-anchor truth.
    # A run resumed EXACTLY at the boundary (iter == value_warmup_iters) also starts True, so
    # it likewise skips the re-anchor — consequence-free here, because that checkpoint was
    # saved right after the switch already ran and re-snapshotted the EMA in the prior process
    # (value_warmup_done=True at that iter), so there is nothing left to re-anchor.
    phase = {"ppo_started": counters["iter"] >= args.rl.value_warmup_iters}
    transitions_at_start = counters["transitions"]  # keep transitions_per_s honest across --resume

    if args.wandb:
        import wandb

        run = wandb.init(
            project="hal",
            name=run_name,
            id=wandb_id,
            resume="allow" if wandb_id else None,
            tags=["014", "rl", "selfplay"],
            config={**dataclasses.asdict(args), "L_ctx": cfg.L_ctx, "resolved_run": run_name},
        )
        wandb_id = run.id
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

    logger.info(f"run={run_name} device={device} overlap={args.pipeline.overlap} n_boots={args.rl.n_boots}")
    logger.info(f"ppo={args.ppo}")

    n_slots = args.rl.n_boots * len(ports)
    handle = NetActingPolicy(
        act_net,
        n_slots=n_slots,
        device=device,
        temp=args.temp,
        seed=args.seed,
        max_pos=cfg.L_ctx + args.rl.refresh_every + 8,
    )
    # A resumed run continues matchup rotation where the schedule left off rather than
    # restarting at slice 0 (attrition reboots aren't counted — the seed is the scheduled
    # floor, which is enough to keep long runs from re-grinding the head of the prior).
    wave0 = counters["iter"] // args.rl.reboot_every_iters if args.rl.reboot_every_iters else 0
    matchups = wave_matchups(wave0, args.rl.n_boots)
    slot_matchup = _slot_matchups(matchups, ports)
    slots = list(slot_matchup)
    pol = RLBatchPolicy(
        handles={"ema": handle},
        assign=lambda s: "ema",
        slots=slots,
        ego_port_of=lambda s: s.port,
        matchup_of=lambda s: slot_matchup[s],
        reward_cfg=args.reward,
        stats=stats,
        L_ctx=cfg.L_ctx,
        refresh_every=args.rl.refresh_every,
        rollout_frames=args.rl.rollout_frames,
    )
    session_cfg = default_session_cfg(replay_root, instant_match_restart=True)
    build_boot = _boot_builder(session_cfg, args.rl.n_boots, args.base_slippi_port, replay_root)

    snapshot_lock = threading.Lock()
    q: queue.Queue[IterationPayload] = queue.Queue(maxsize=1)
    stop = threading.Event()
    sync_gate = threading.Event() if not args.pipeline.overlap else None

    def on_iteration() -> None:
        with snapshot_lock:
            ema.copy_to(act_net)
        # act_net just changed under the caches: K/V computed by the old weights would
        # otherwise serve up to refresh_every-1 more frames of hybrid behavior.
        pol.stepper.request_rebuild()

    t_start = time.monotonic()

    def learn(iteration: RolloutIteration, cstats: CollectStats, queue_wait_s: float) -> None:
        t_learn = time.monotonic()  # learner_s_per_iter includes the value pass below
        windows, vdiag = _prepare_windows(
            learner, iteration, stats, args.ppo, device, stride=args.rl.ppo_window_stride
        )
        if not windows:
            counters["empty_iters"] += 1
            logger.warning(
                f"learn: iteration produced NO windows ({iteration.n_transitions} transitions); skipping update — "
                f"empty_iters={counters['empty_iters']} (learner iter={counters['iter']} now trails drive's emitted "
                "count by this much; a persistent gap means the collector keeps yielding stepless iterations)"
            )
            return
        warmup = counters["iter"] < args.rl.value_warmup_iters
        if not warmup and not phase["ppo_started"]:
            # Warmup froze the EMA (acting stayed byte-identical IL); re-anchor it to the
            # warmed learner before it starts trailing PPO updates.
            with snapshot_lock:
                _resnapshot_ema(ema, learner)
            phase["ppo_started"] = True

        def advance_ema() -> None:
            with snapshot_lock:
                ema.update(learner)

        if warmup:
            # on_step=None: the EMA (and therefore act_net) must not move during warmup —
            # ema.update is not a fixed point even for unchanged params (~ulp drift).
            metrics = value_warmup_update(
                learner, opt_warm, windows, args.ppo, stats, device=device, generator=gen, on_step=None
            )
        else:
            metrics = melee_ppo_update(
                learner,
                il_net,
                opt_ppo,
                windows,
                args.ppo,
                args.rl.kl_il_coef,
                stats,
                device=device,
                generator=gen,
                on_step=advance_ema,
            )
        learn_s = time.monotonic() - t_learn
        counters["iter"] += 1
        counters["transitions"] += iteration.n_transitions
        rl_stats = _rollout_stats(iteration)
        elapsed = time.monotonic() - t_start
        thr = {
            "transitions_per_s": (counters["transitions"] - transitions_at_start) / elapsed if elapsed else 0.0,
            "learner_s_per_iter": learn_s,
            "queue_wait_s": queue_wait_s,  # real blocking time in _next_iteration
            "lockstep_sps": cstats.lockstep_sps,  # collector-measured stepping rate
        }
        common = (
            f"iter={counters['iter']} transitions={counters['transitions']} "
            f"reward_mean={rl_stats['reward_mean']:.4f} dmg/min={rl_stats['dmg_dealt_per_min']:.1f} "
            f"ep_return={rl_stats['ep_return_mean']:.2f} n_eps={int(rl_stats['n_episodes'])} "
            f"vf={metrics['vf_loss']:.3f} v_ev={vdiag['v_explained_var']:.3f} "
            f"sps={cstats.lockstep_sps:.0f} learn_s={learn_s:.1f}"
        )
        if warmup:  # value-head-only phase: policy metrics don't exist, so none are printed
            logger.info(f"{common} phase=warmup")
        else:
            logger.info(
                f"{common} phase=ppo clip={metrics['clip_loss']:.3f} "
                f"kl_upd={metrics['approx_kl_update']:.4f} kl_col={metrics['approx_kl_collect']:.4f} "
                f"epochs={int(metrics['epochs_used'])} ratio_dev0={metrics['ratio_dev_epoch0']:.4f}"
            )
        _check_finite(metrics)
        if args.wandb:
            import wandb

            wandb.log(
                {
                    "global_step": counters["transitions"],
                    **{f"rollout/{k}": v for k, v in rl_stats.items()},
                    # metrics carries only what this phase measured (warmup: vf_loss only)
                    **{f"train/{k}": v for k, v in metrics.items()},
                    **{f"train/{k}": v for k, v in vdiag.items()},
                    **{f"throughput/{k}": v for k, v in thr.items()},
                }
            )
        if counters["iter"] % args.ckpt_every_iters == 0:
            _save(ckpt_dir / "latest.pt", counters, learner, opt_ppo, opt_warm, const_sched, ema, args, cfg, wandb_id)

    # --- collector thread + main-thread learner loop (direct wiring) ----------
    drive_error: list[BaseException] = []
    remaining = total_iters - counters["iter"]

    def drive_target() -> None:
        try:
            drive_rl(
                build_boot,
                lambda w: wave_matchups(w, args.rl.n_boots),
                [ports] * args.rl.n_boots,
                pol,
                n_iterations=remaining,
                queue_out=q,
                stop=stop,
                on_iteration=on_iteration,
                sync_gate=sync_gate,
                reboot_every_iters=args.rl.reboot_every_iters,
                wave0=wave0,
                progress_every=600,
            )
        except BaseException as exc:  # noqa: BLE001 — surfaced on the main thread
            drive_error.append(exc)
            stop.set()

    collector = threading.Thread(target=drive_target, name="rl-drive", daemon=True)
    collector.start()
    interrupted = False
    try:
        for _ in range(remaining):
            payload, wait_s = _next_iteration(q, collector, stop)
            if payload is None:
                break
            iteration, cstats = payload
            learn(iteration, cstats, wait_s)
            if sync_gate is not None:
                sync_gate.set()  # release the collector for the next (lag-0) iteration
    except KeyboardInterrupt:
        interrupted = True
        # SIGINT goes to the whole process group (Dolphins included), and repeats/echoes
        # would blow through the shutdown path before the final save — ignore from here on.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        logger.warning("SIGINT: stopping collection; final checkpoint follows (further SIGINT ignored)")
    finally:
        stop.set()
        if sync_gate is not None:
            sync_gate.set()  # unblock a collector parked on the gate so it can wind down
        while collector.is_alive():
            try:
                collector.join()
            except KeyboardInterrupt:  # a repeat SIGINT racing the SIG_IGN swap above
                interrupted = True
                signal.signal(signal.SIGINT, signal.SIG_IGN)

    _save(ckpt_dir / "latest.pt", counters, learner, opt_ppo, opt_warm, const_sched, ema, args, cfg, wandb_id)
    if drive_error and not interrupted:
        raise drive_error[0]
    logger.info(f"done: {counters['iter']} iterations, {counters['transitions']} transitions")


_POLL = 0.1


def _next_iteration(
    q: queue.Queue[IterationPayload], collector: threading.Thread, stop: threading.Event
) -> tuple[IterationPayload | None, float]:
    """Block for the next payload; returns it plus the real time spent blocked here
    (``queue_wait_s`` — the honest learner-starvation signal). ``None`` when the
    collector died/stopped with nothing queued."""
    t0 = time.monotonic()
    while True:
        try:
            return q.get(timeout=_POLL), time.monotonic() - t0
        except queue.Empty:
            if collector.is_alive() and not stop.is_set():
                continue
            try:
                return q.get_nowait(), time.monotonic() - t0
            except queue.Empty:
                return None, time.monotonic() - t0


def _check_finite(metrics: dict[str, float]) -> None:
    bad = [k for k, v in metrics.items() if not np.isfinite(v)]
    if bad:
        raise ValueError(f"non-finite train metric(s): {bad} in {metrics}")


def _save(
    path: Path,
    counters: dict[str, int],
    learner: PolicyValueNet,
    opt_ppo: torch.optim.Optimizer,
    opt_warm: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    ema: EMAWeights,
    args: Args,
    cfg: ArchConfig,
    wandb_id: str | None,
) -> None:
    save_checkpoint(
        path,
        step=counters["iter"],
        model=learner,
        opt=opt_ppo,
        sched=sched,
        cfg={**dataclasses.asdict(args), "L_ctx": cfg.L_ctx},
        wandb_id=wandb_id,
        extra={
            "opt_warm": opt_warm.state_dict(),
            "ema": ema.state_dict(),
            "value_warmup_done": counters["iter"] >= args.rl.value_warmup_iters,
            "transitions": counters["transitions"],
        },
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
