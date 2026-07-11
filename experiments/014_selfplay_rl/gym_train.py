"""CartPole PPO entry point — gate G1 for the self-play RL core.

Proves the shared machinery (``BehaviorLogpPPO`` + EMA behavior policy +
double-buffer ``run_pipeline``) reaches CartPole-v1 >= 475 in both sync
(``--pipeline.no-overlap``) and overlapped (``--pipeline.overlap``) modes before
any of it touches the expensive Melee emulator. The EMA snapshot acts in the
collector; the learner trains fast weights; PPO ratios are importance weights
against the recorded behavior logprobs.

    uv run experiments/014_selfplay_rl/gym_train.py --pipeline.no-overlap --seed 0
    uv run experiments/014_selfplay_rl/gym_train.py --pipeline.overlap --seed 0
"""

import dataclasses
import threading
import time
from collections import deque

import envpool
import numpy as np
import torch
import tyro
from loguru import logger
from nets_gym import MLPActor
from nets_gym import MLPCritic
from pipeline import run_pipeline
from ppo import BehaviorLogpPPO
from ppo import build_batch_add
from ppo import update_with_kl_stop
from rl_config import EMAConfig
from rl_config import GymConfig
from rl_config import PipelineConfig
from rl_config import PPOConfig
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import VectorReplayBuffer
from tianshou.utils.net.discrete import dist_fn_categorical_from_logits

from hal.training.ema import EMAWeights


@dataclasses.dataclass(frozen=True)
class Args:
    gym: GymConfig = dataclasses.field(default_factory=GymConfig)
    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    # Gym overrides the default EMA decay (0.995 -> 0.99); CartPole is short-horizon.
    ema: EMAConfig = dataclasses.field(default_factory=lambda: EMAConfig(decay=0.99))
    pipeline: PipelineConfig = dataclasses.field(default_factory=PipelineConfig)
    seed: int = 0
    eval_episodes: int = 100
    eval_threshold: float = 475.0


def _sample(act_net: MLPActor, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample actions + behavior logprobs from the EMA policy for a batch of obs."""
    logits, _ = act_net(obs)
    dist = torch.distributions.Categorical(logits=logits)
    act = dist.sample()
    return act.cpu().numpy(), dist.log_prob(act).cpu().numpy()


def _make_env(task: str, num_envs: int, seed: int):
    return envpool.make(task, env_type="gymnasium", num_envs=num_envs, seed=seed)


def evaluate(act_net: MLPActor, *, task: str, num_envs: int, seed: int, n_episodes: int) -> float:
    """Roll the (already EMA-loaded) policy until ``n_episodes`` finish; mean return."""
    env = _make_env(task, num_envs, seed)
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device} overlap={args.pipeline.overlap} seed={args.seed}")

    env = _make_env(args.gym.task, args.gym.num_envs, args.seed)
    action_space = env.action_space
    num_envs, horizon = args.gym.num_envs, args.gym.horizon
    ids = np.arange(num_envs)

    learner_actor = MLPActor().to(device)
    critic = MLPCritic().to(device)
    act_net = MLPActor().to(device)  # EMA behavior snapshot; only the collector reads it
    ema = EMAWeights(learner_actor, decay=args.ema.decay)

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
        optim=AdamOptimizerFactory(lr=args.ppo.lr),
        eps_clip=args.ppo.clip,
        vf_coef=args.ppo.vf_coef,
        ent_coef=args.ppo.ent_coef,
        max_grad_norm=args.ppo.max_grad_norm,
        gae_lambda=args.ppo.gae_lambda,
        gamma=args.ppo.gamma,
        advantage_normalization=True,
    ).to(device)

    snapshot_lock = threading.Lock()
    ep_returns: deque[float] = deque(maxlen=100)
    rollout = {"obs": env.reset()[0], "ep_ret": np.zeros(num_envs, dtype=np.float64)}
    counters = {"iter": 0, "frames": 0}
    start = time.time()

    def collect() -> VectorReplayBuffer:
        with snapshot_lock:
            ema.copy_to(act_net)
        buf = VectorReplayBuffer(horizon * num_envs, buffer_num=num_envs)
        obs, ep_ret = rollout["obs"], rollout["ep_ret"]
        with torch.inference_mode():
            for _ in range(horizon):
                act, logp = _sample(act_net, obs)
                obs_next, rew, term, trunc, _ = env.step(act, ids)
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
                done = term | trunc
                for i in np.where(done)[0]:
                    ep_returns.append(float(ep_ret[i]))
                    ep_ret[i] = 0.0
                obs = obs_next.copy()
                if done.any():
                    reset_obs, _ = env.reset(ids[done])
                    obs[done] = reset_obs
        rollout["obs"] = obs
        return buf

    def learn(buf: VectorReplayBuffer) -> None:
        def advance_ema() -> None:
            with snapshot_lock:
                ema.update(learner_actor)

        metrics = update_with_kl_stop(algo, buf, args.ppo, on_epoch_end=advance_ema)
        counters["iter"] += 1
        counters["frames"] += horizon * num_envs
        sps = counters["frames"] / (time.time() - start)
        mean_ret = float(np.mean(ep_returns)) if ep_returns else float("nan")
        logger.info(
            f"iter={counters['iter']} frames={counters['frames']} "
            f"ep_return_mean={mean_ret:.1f} approx_kl={metrics['approx_kl']:.4f} "
            f"epochs_used={int(metrics['epochs_used'])} loss={metrics['loss']:.3f} sps={sps:.0f}"
        )

    iterations = args.gym.total_frames // (horizon * num_envs)
    run_pipeline(collect=collect, learn=learn, iterations=iterations, overlap=args.pipeline.overlap)

    with snapshot_lock:
        ema.copy_to(act_net)
    mean_ret = evaluate(
        act_net,
        task=args.gym.task,
        num_envs=num_envs,
        seed=args.seed + 10_000,
        n_episodes=args.eval_episodes,
    )
    verdict = "PASS" if mean_ret >= args.eval_threshold else "FAIL"
    logger.info(
        f"G1 {verdict}: eval mean return {mean_ret:.1f} over {args.eval_episodes} eps (threshold {args.eval_threshold})"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
