"""BehaviorLogpPPO + double-buffer pipeline unit tests (pure CPU, fast).

Covers the load-bearing claims of the RL core: the PPO ratio is an importance
weight against the RECORDED behavior logprob (not a current-policy recompute);
on-policy epoch-0 ratios are 1; the KL early-stop fires; and the double-buffer
runner keeps the collector at most one iteration ahead, runs every iteration
exactly once in order, and propagates a collector exception.
"""

import time

import numpy as np
import pytest
import torch
from nets_gym import MLPActor
from nets_gym import MLPCritic
from pipeline import run_pipeline
from ppo import BehaviorLogpPPO
from ppo import build_batch_add
from ppo import update_with_kl_stop
from rl_config import PPOConfig
from tianshou.algorithm import PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import VectorReplayBuffer
from tianshou.utils.net.discrete import dist_fn_categorical_from_logits

NUM_ENVS = 2
HORIZON = 64


def _policy(actor: MLPActor) -> ProbabilisticActorPolicy:
    import gymnasium as gym

    return ProbabilisticActorPolicy(
        actor=actor,
        dist_fn=dist_fn_categorical_from_logits,
        action_space=gym.spaces.Discrete(2),
        action_scaling=False,
        action_bound_method=None,
    )


def _make_algo(cls, lr: float = 3e-4) -> tuple:
    torch.manual_seed(0)
    actor, critic = MLPActor(), MLPCritic()
    algo = cls(
        policy=_policy(actor),
        critic=critic,
        optim=AdamOptimizerFactory(lr=lr),
        eps_clip=0.2,
        gamma=0.99,
        gae_lambda=0.95,
    )
    return algo, actor, critic


def _collect(actor: MLPActor, *, seed: int = 0, logp_override: float | None = None) -> VectorReplayBuffer:
    """Roll a short CartPole batch into a VectorReplayBuffer via the real add path."""
    import envpool

    env = envpool.make("CartPole-v1", env_type="gymnasium", num_envs=NUM_ENVS, seed=seed)
    ids = np.arange(NUM_ENVS)
    obs, _ = env.reset()
    buf = VectorReplayBuffer(HORIZON * NUM_ENVS, buffer_num=NUM_ENVS)
    with torch.inference_mode():
        for _ in range(HORIZON):
            logits, _ = actor(obs)
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample()
            logp = dist.log_prob(act).cpu().numpy()
            if logp_override is not None:
                logp = np.full(NUM_ENVS, logp_override, dtype=np.float32)
            act_np = act.cpu().numpy()
            obs_next, rew, term, trunc, _ = env.step(act_np, ids)
            buf.add(
                build_batch_add(
                    obs=obs,
                    act=act_np,
                    rew=rew,
                    terminated=term,
                    truncated=trunc,
                    obs_next=obs_next,
                    logp_old=logp,
                ),
                buffer_ids=ids,
            )
            obs = obs_next.copy()
            done = term | trunc
            if done.any():
                reset_obs, _ = env.reset(ids[done])
                obs[done] = reset_obs
    return buf


def test_recorded_logp_is_used_not_recomputed() -> None:
    # Record a behavior logp that the current policy would never produce (constant -0.5).
    behavior_logp = -0.5
    algo, actor, _ = _make_algo(BehaviorLogpPPO)

    buf = _collect(actor, logp_override=behavior_logp)
    batch, indices = buf.sample(0)
    processed = algo._preprocess_batch(batch, buf, indices)

    # BehaviorLogpPPO must pass the recorded value straight through.
    assert torch.allclose(processed.logp_old, torch.full_like(processed.logp_old, behavior_logp), atol=1e-5)

    # Stock PPO recomputes with the current policy -> it must NOT equal the recorded constant.
    stock, stock_actor, _ = _make_algo(PPO)
    buf2 = _collect(stock_actor, logp_override=behavior_logp)
    b2, i2 = buf2.sample(0)
    stock_processed = stock._preprocess_batch(b2, buf2, i2)
    assert not torch.allclose(
        stock_processed.logp_old, torch.full_like(stock_processed.logp_old, behavior_logp), atol=1e-2
    )


def test_epoch0_ratios_are_one_on_policy() -> None:
    # act_net == learner (sync, lag 0): the recorded logp equals a fresh forward's logp,
    # so the epoch-0 PPO ratio exp(logp_current - logp_old) is 1 everywhere.
    algo, actor, _ = _make_algo(BehaviorLogpPPO)
    buf = _collect(actor)
    batch, _ = buf.sample(0)

    logp_old = torch.as_tensor(batch.policy.logp_old, dtype=torch.float32).flatten()
    with torch.no_grad():
        dist = algo.policy(batch).dist
        logp_current = dist.log_prob(torch.as_tensor(batch.act)).flatten()
    ratio = (logp_current - logp_old).exp()
    assert (ratio - 1.0).abs().mean() < 1e-5


def test_kl_early_stop_fires() -> None:
    # Absurd lr blows the policy past target_kl after the first epoch -> fewer epochs used.
    algo, actor, _ = _make_algo(BehaviorLogpPPO, lr=5.0)
    buf = _collect(actor)
    cfg = PPOConfig(epochs=8, minibatch_size=64, target_kl=0.015)
    metrics = update_with_kl_stop(algo, buf, cfg)
    assert metrics["epochs_used"] < cfg.epochs
    assert metrics["approx_kl"] > cfg.target_kl


def test_pipeline_overlap_lag_order_and_exception() -> None:
    n = 12
    state = {"collecting": -1}  # highest index the collector has begun
    learned: list[int] = []
    gaps: list[int] = []
    counter = {"c": 0}

    def collect() -> int:
        i = counter["c"]
        counter["c"] += 1
        state["collecting"] = i
        time.sleep(0.005)  # give the learner time to fall behind, exercising overlap
        return i

    def learn(i: int) -> None:
        gaps.append(state["collecting"] - i)  # collector index minus the one we're learning
        time.sleep(0.005)
        learned.append(i)

    run_pipeline(collect=collect, learn=learn, iterations=n, overlap=True)

    assert learned == list(range(n))  # exactly once, in order
    assert max(gaps) <= 1  # collector never more than one iteration ahead (double buffer)
    assert max(gaps) == 1  # overlap actually happened (collector did get ahead)


def test_pipeline_propagates_collector_exception() -> None:
    learned: list[int] = []
    counter = {"c": 0}

    def collect() -> int:
        i = counter["c"]
        counter["c"] += 1
        if i == 3:
            raise RuntimeError("collector boom")
        return i

    def learn(i: int) -> None:
        learned.append(i)

    with pytest.raises(RuntimeError, match="collector boom"):
        run_pipeline(collect=collect, learn=learn, iterations=10, overlap=True)
    # payloads produced before the failure were still learned in order
    assert learned == [0, 1, 2] or learned == [0, 1] or learned == [0]


def test_pipeline_sync_matches_inline() -> None:
    seen: list[int] = []
    counter = {"c": 0}

    def collect() -> int:
        i = counter["c"]
        counter["c"] += 1
        return i

    def learn(i: int) -> None:
        seen.append(i)

    run_pipeline(collect=collect, learn=learn, iterations=5, overlap=False)
    assert seen == [0, 1, 2, 3, 4]
