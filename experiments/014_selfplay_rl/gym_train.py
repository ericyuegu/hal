"""Gym PPO entry point — gates G1 (CartPole) and G2 (Atari Pong).

Proves the shared machinery (``BehaviorLogpPPO`` + EMA behavior policy +
double-buffer ``run_pipeline``) on two envs before any of it touches the
expensive Melee emulator. The EMA snapshot acts in the collector; the learner
trains fast weights; PPO ratios are importance weights against the recorded
behavior logprobs.

``--task`` selects a CleanRL-comparable preset (nets, hyperparams, env config):
``CartPole-v1`` -> two MLPs, PPO defaults, gate G1 >= 475 (both pipeline modes);
``Pong-v5`` -> shared Nature-CNN, CleanRL ``ppo_atari`` hyperparams + envpool
Atari wrappers, gate G2 >= +18 game score. The Pong config is pinned explicitly
(never envpool defaults) and printed at startup for parity auditing.

    uv run experiments/014_selfplay_rl/gym_train.py --task CartPole-v1 --pipeline.no-overlap
    uv run experiments/014_selfplay_rl/gym_train.py --task Pong-v5 --total-frames 1_000_000 --seed 0
"""

import dataclasses
import threading
import time
from collections import deque
from collections.abc import Callable
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import envpool
import numpy as np
import torch
import tyro
from envpool.python.envpool import EnvPoolMixin
from loguru import logger
from nets_gym import MLPActor
from nets_gym import MLPCritic
from nets_gym import NatureCNN
from nets_gym import NatureCNNActor
from nets_gym import NatureCNNCritic
from pipeline import run_pipeline
from ppo import BehaviorLogpPPO
from ppo import build_batch_add
from ppo import update_with_kl_stop
from rl_config import GymConfig
from rl_config import PipelineConfig
from rl_config import PPOConfig
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import VectorReplayBuffer
from tianshou.utils.net.discrete import dist_fn_categorical_from_logits
from torch import nn

from hal.training.ema import EMAWeights

# Explicitly pinned envpool Atari wrapper config (NOT envpool defaults) for CleanRL
# ppo_atari parity. episodic_life is a no-op for Pong (no lives) — pinned anyway for
# parity; reward_clip clips to [-1,1] but Pong's raw rewards are already +/-1 so the
# clipped stream equals the true game score (asserted in the collector).
PONG_ENV_KWARGS: Mapping[str, object] = {
    "img_height": 84,
    "img_width": 84,
    "gray_scale": True,
    "stack_num": 4,
    "frame_skip": 4,
    "noop_max": 30,
    "episodic_life": True,
    "reward_clip": True,
    "repeat_action_probability": 0.0,
    "full_action_space": False,
}

# (learner_actor, critic, ema_behavior_actor)
NetBundle = tuple[nn.Module, nn.Module, nn.Module]


@dataclasses.dataclass(frozen=True, slots=True)
class Preset:
    """Everything ``main`` needs that varies by task; built by the preset fns."""

    gym: GymConfig
    ppo: PPOConfig
    ema_decay: float
    env_kwargs: Mapping[str, object]
    build_nets: Callable[[int, torch.device], NetBundle]
    gate: str
    eval_threshold: float
    anneal_lr: bool
    fresh_env_eval: bool  # True: roll a fresh env for the gate (CartPole); False: use rollout mean (Pong)
    reward_is_pm1: bool  # assert stepped rewards are in {-1,0,1} (game score == clipped stream)


def _build_cartpole_nets(n_act: int, device: torch.device) -> NetBundle:
    return (
        MLPActor(n_act=n_act).to(device),
        MLPCritic().to(device),
        MLPActor(n_act=n_act).to(device),  # EMA behavior snapshot
    )


def _build_pong_nets(n_act: int, device: torch.device) -> NetBundle:
    cnn = NatureCNN(n_act).to(device)  # shared trunk: actor + critic reference one CNN
    act_net = NatureCNNActor(NatureCNN(n_act).to(device))  # EMA snapshot has its own CNN
    return NatureCNNActor(cnn), NatureCNNCritic(cnn), act_net


def _cartpole_preset(total_frames: int | None) -> Preset:
    gym = GymConfig(task="CartPole-v1", num_envs=8, horizon=128, total_frames=total_frames or 500_000)
    return Preset(
        gym=gym,
        ppo=PPOConfig(),  # lr 3e-4 defaults
        ema_decay=0.99,  # CartPole is short-horizon; faster trail than the 0.995 default
        env_kwargs={},
        build_nets=_build_cartpole_nets,
        gate="G1",
        eval_threshold=475.0,
        anneal_lr=False,
        fresh_env_eval=True,
        reward_is_pm1=False,
    )


def _pong_preset(total_frames: int | None) -> Preset:
    gym = GymConfig(task="Pong-v5", num_envs=8, horizon=128, total_frames=total_frames or 10_000_000)
    ppo = PPOConfig(
        lr=2.5e-4,
        clip=0.1,
        epochs=4,
        minibatch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        target_kl=0.015,
        max_grad_norm=0.5,
    )
    return Preset(
        gym=gym,
        ppo=ppo,
        ema_decay=0.99,
        env_kwargs=PONG_ENV_KWARGS,
        build_nets=_build_pong_nets,
        gate="G2",
        eval_threshold=18.0,
        anneal_lr=True,
        fresh_env_eval=False,
        reward_is_pm1=True,
    )


_PRESETS: Mapping[str, Callable[[int | None], Preset]] = {
    "CartPole-v1": _cartpole_preset,
    "Pong-v5": _pong_preset,
}


@dataclasses.dataclass(frozen=True)
class Args:
    task: str = "CartPole-v1"
    total_frames: int | None = None  # None -> preset default (CartPole 500k, Pong 10M)
    seed: int = 0
    pipeline: PipelineConfig = dataclasses.field(default_factory=PipelineConfig)
    wandb: bool = False
    run_name: str | None = None
    checkpoint_every: int = 100  # iterations between runs/<run>/latest.pt saves
    resume: str | None = None  # path to a latest.pt to continue from
    eval_episodes: int = 100  # fresh-env gate episodes (CartPole)


def _sample(act_net: nn.Module, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample actions + behavior logprobs from the EMA policy for a batch of obs."""
    logits, _ = act_net(obs)
    dist = torch.distributions.Categorical(logits=logits)
    act = dist.sample()
    return act.cpu().numpy(), dist.log_prob(act).cpu().numpy()


def _make_env(task: str, num_envs: int, seed: int, env_kwargs: Mapping[str, object]) -> EnvPoolMixin:
    return envpool.make(task, env_type="gymnasium", num_envs=num_envs, seed=seed, **env_kwargs)


def _set_lr(algo: BehaviorLogpPPO, lr: float) -> None:
    """Set the LR on the algorithm's optimizer(s) directly.

    We drive our own collect/learn loop (not tianshou's Trainer), so the built-in
    ``LRSchedulerFactoryLinear`` — which steps once per ``algo.update`` call, i.e.
    once per PPO epoch (variable under KL early-stop) rather than per iteration —
    does not match CleanRL's per-iteration anneal. Setting the LR ourselves once
    per iteration reproduces CleanRL's ``frac * lr`` schedule exactly.
    """
    for opt in algo._optimizers:
        for group in opt._optim.param_groups:
            group["lr"] = lr


def evaluate(act_net: nn.Module, *, preset: Preset, seed: int, n_episodes: int) -> float:
    """Roll the (already EMA-loaded) policy on a fresh env until ``n_episodes`` finish."""
    env = _make_env(preset.gym.task, preset.gym.num_envs, seed, preset.env_kwargs)
    num_envs = preset.gym.num_envs
    ids = np.arange(num_envs)
    obs, _ = env.reset()
    ep_ret = np.zeros(num_envs, dtype=np.float64)
    returns: list[float] = []
    with torch.inference_mode():
        while len(returns) < n_episodes:
            act, _ = _sample(act_net, obs)
            obs, rew, term, trunc, _ = env.step(act, ids)
            ep_ret += rew
            done = term | trunc
            for i in np.where(done)[0]:
                returns.append(float(ep_ret[i]))
                ep_ret[i] = 0.0
            obs = obs.copy()
            if done.any():
                reset_obs, _ = env.reset(ids[done])
                obs[done] = reset_obs
    return float(np.mean(returns[:n_episodes]))


def main(args: Args) -> None:
    if args.task not in _PRESETS:
        raise ValueError(f"unknown task {args.task!r}; known: {sorted(_PRESETS)}")
    preset = _PRESETS[args.task](args.total_frames)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task_slug = args.task.split("-")[0].lower()
    run_name = args.run_name or f"014_{task_slug}_{datetime.now():%y%m%d-%H%M%S}"
    ckpt_dir = Path("runs") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"run={run_name} task={args.task} gate={preset.gate} device={device} seed={args.seed}")
    logger.info(f"overlap={args.pipeline.overlap} anneal_lr={preset.anneal_lr} ema_decay={preset.ema_decay}")
    logger.info(f"ppo={preset.ppo}")
    logger.info(f"env_kwargs={dict(preset.env_kwargs)}")  # resolved config, for parity auditing

    num_envs, horizon = preset.gym.num_envs, preset.gym.horizon
    ids = np.arange(num_envs)
    env = _make_env(args.task, num_envs, args.seed, preset.env_kwargs)
    action_space = env.action_space
    n_act = int(action_space.n)

    learner_actor, critic, act_net = preset.build_nets(n_act, device)
    ema = EMAWeights(learner_actor, decay=preset.ema_decay)

    policy = ProbabilisticActorPolicy(
        actor=learner_actor,
        dist_fn=dist_fn_categorical_from_logits,
        action_space=action_space,
        action_scaling=False,
        action_bound_method=None,
    )
    algo = BehaviorLogpPPO(
        policy=policy,
        critic=critic,
        optim=AdamOptimizerFactory(lr=preset.ppo.lr),
        eps_clip=preset.ppo.clip,
        vf_coef=preset.ppo.vf_coef,
        ent_coef=preset.ppo.ent_coef,
        max_grad_norm=preset.ppo.max_grad_norm,
        gae_lambda=preset.ppo.gae_lambda,
        gamma=preset.ppo.gamma,
        advantage_normalization=True,
    ).to(device)

    iterations = preset.gym.total_frames // (horizon * num_envs)
    counters = {"iter": 0, "frames": 0}
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        learner_actor.load_state_dict(ckpt["learner"])
        critic.load_state_dict(ckpt["critic"])
        ema.load_state_dict(ckpt["ema"])
        algo.load_state_dict(ckpt["algo"])
        counters["iter"], counters["frames"] = ckpt["iter"], ckpt["env_frames"]
        if counters["iter"] >= iterations:
            raise ValueError(
                f"resume checkpoint is already at iter={counters['iter']} >= {iterations} total iterations "
                f"({preset.gym.total_frames} frames); raise --total-frames to continue this run"
            )
        logger.info(f"resumed from {args.resume} at iter={counters['iter']} frames={counters['frames']}")

    if args.wandb:
        import wandb

        wandb_config = {
            **dataclasses.asdict(args),
            "ppo": dataclasses.asdict(preset.ppo),
            "ema_decay": preset.ema_decay,
            "num_envs": num_envs,
            "horizon": horizon,
            "resolved_total_frames": preset.gym.total_frames,
            "env_kwargs": dict(preset.env_kwargs),
        }
        wandb.init(project="hal", name=run_name, tags=["014", "rl", task_slug], config=wandb_config)
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")

    snapshot_lock = threading.Lock()
    # Only the learner thread mutates these deques (the collector reports finished
    # episodes inside its payload — no cross-thread mutation during np.mean).
    ep_returns: deque[float] = deque(maxlen=100)
    ep_lens: deque[float] = deque(maxlen=100)
    rollout = {
        "obs": env.reset()[0],
        "ep_ret": np.zeros(num_envs, dtype=np.float64),
        "ep_len": np.zeros(num_envs, dtype=np.int64),
    }
    start = time.time()
    # sps counts only frames stepped THIS process: on --resume, counters["frames"] restores the
    # lifetime total, and dividing that by post-resume elapsed would inflate sps by resume-point/elapsed.
    frames_at_start = counters["frames"]

    def collect() -> tuple[VectorReplayBuffer, list[float], list[float]]:
        with snapshot_lock:
            ema.copy_to(act_net)
        buf = VectorReplayBuffer(horizon * num_envs, buffer_num=num_envs)
        obs, ep_ret, ep_len = rollout["obs"], rollout["ep_ret"], rollout["ep_len"]
        ret_done: list[float] = []
        len_done: list[float] = []
        with torch.inference_mode():
            for _ in range(horizon):
                act, logp = _sample(act_net, obs)
                obs_next, rew, term, trunc, _ = env.step(act, ids)
                if preset.reward_is_pm1:  # game score == clipped stream (Pong raw rewards are +/-1)
                    assert np.all(np.isin(rew, (-1.0, 0.0, 1.0))), f"unexpected reward {np.unique(rew)}"
                buf.add(
                    build_batch_add(
                        obs=obs,
                        act=act,
                        rew=rew,
                        terminated=term,
                        truncated=trunc,
                        obs_next=obs_next,
                        logp_old=logp,
                    ),
                    buffer_ids=ids,
                )
                ep_ret += rew
                ep_len += 1
                done = term | trunc
                for i in np.where(done)[0]:
                    ret_done.append(float(ep_ret[i]))
                    len_done.append(float(ep_len[i]))
                    ep_ret[i] = 0.0
                    ep_len[i] = 0
                obs = obs_next.copy()
                if done.any():
                    reset_obs, _ = env.reset(ids[done])
                    obs[done] = reset_obs
        rollout["obs"] = obs
        return buf, ret_done, len_done

    def learn(payload: tuple[VectorReplayBuffer, list[float], list[float]]) -> None:
        buf, ret_done, len_done = payload
        ep_returns.extend(ret_done)
        ep_lens.extend(len_done)

        if preset.anneal_lr:  # CleanRL per-iteration linear decay to 0
            frac = 1.0 - counters["iter"] / iterations
            _set_lr(algo, frac * preset.ppo.lr)

        def advance_ema() -> None:
            with snapshot_lock:
                ema.update(learner_actor)

        metrics = update_with_kl_stop(algo, buf, preset.ppo, on_epoch_end=advance_ema)
        counters["iter"] += 1
        counters["frames"] += horizon * num_envs
        sps = (counters["frames"] - frames_at_start) / (time.time() - start)
        ret_mean = float(np.mean(ep_returns)) if ep_returns else float("nan")
        len_mean = float(np.mean(ep_lens)) if ep_lens else float("nan")
        logger.info(
            f"iter={counters['iter']} frames={counters['frames']} "
            f"ep_return_mean={ret_mean:.1f} ep_len_mean={len_mean:.0f} "
            f"approx_kl={metrics['approx_kl']:.4f} epochs_used={int(metrics['epochs_used'])} "
            f"loss={metrics['loss']:.3f} sps={sps:.0f}"
        )
        if args.wandb:
            import wandb

            wandb.log(
                {
                    "global_step": counters["frames"],
                    "rollout/ep_return_mean": ret_mean,
                    "rollout/ep_len_mean": len_mean,
                    "train/approx_kl": metrics["approx_kl"],
                    "train/epochs_used": metrics["epochs_used"],
                    "throughput/sps": sps,
                    "throughput/env_frames": counters["frames"],
                }
            )
        if counters["iter"] % args.checkpoint_every == 0:
            torch.save(
                {
                    "learner": learner_actor.state_dict(),
                    "critic": critic.state_dict(),
                    "ema": ema.state_dict(),
                    "algo": algo.state_dict(),  # opt_via_algo_state_dict
                    "iter": counters["iter"],
                    "env_frames": counters["frames"],
                },
                ckpt_dir / "latest.pt",
            )

    run_pipeline(
        collect=collect,
        learn=learn,
        iterations=iterations - counters["iter"],
        overlap=args.pipeline.overlap,
    )

    with snapshot_lock:
        ema.copy_to(act_net)
    if preset.fresh_env_eval:
        score = evaluate(act_net, preset=preset, seed=args.seed + 10_000, n_episodes=args.eval_episodes)
        detail = f"eval mean return {score:.1f} over {args.eval_episodes} fresh-env eps"
    else:  # Pong: the rollout ep-return IS the true game score (unclipped == clipped)
        score = float(np.mean(ep_returns)) if ep_returns else float("nan")
        detail = f"final rollout ep_return_mean {score:.1f} over last {len(ep_returns)} eps"
    verdict = "PASS" if score >= preset.eval_threshold else "FAIL"
    logger.info(f"{preset.gate} {verdict}: {detail} (threshold {preset.eval_threshold})")


if __name__ == "__main__":
    main(tyro.cli(Args))
