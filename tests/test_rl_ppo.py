"""BehaviorLogpPPO + double-buffer pipeline unit tests (pure CPU, fast).

Covers the load-bearing claims of the RL core: the PPO ratio is an importance
weight against the RECORDED behavior logprob (not a current-policy recompute);
on-policy epoch-0 ratios are 1; the KL early-stop fires; and the double-buffer
runner keeps the collector at most one iteration ahead, runs every iteration
exactly once in order, and propagates a collector exception.
"""

import threading
import time
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch
from nets_gym import MLPActor
from nets_gym import MLPCritic
from nets_melee import A_VOCAB
from nets_melee import ArchConfig
from nets_melee import FactoredCategorical
from nets_melee import PolicyValueNet
from pipeline import run_pipeline
from ppo import BehaviorLogpPPO
from ppo import build_batch_add
from ppo import melee_ppo_update
from ppo import update_with_kl_stop
from rl_config import PPOConfig
from rollout import FinalizedStream
from rollout import build_windows
from rollout import collate_windows
from rollout import gae_inputs
from rollout import scatter_gae
from tianshou.algorithm import PPO
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.data import Batch
from tianshou.data import VectorReplayBuffer
from tianshou.utils.net.discrete import dist_fn_categorical_from_logits
from tianshou.utils.torch_utils import policy_within_training_step

from hal.data.stats import FeatureStats
from hal.training.features import A_DIM
from hal.training.features import FLOAT_FEATURES
from hal.wire import POST_FIELD_SUFFIXES

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


def _make_algo(cls: type[PPO], lr: float = 3e-4) -> tuple[PPO, MLPActor, MLPCritic]:
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


def test_pipeline_learner_exception_no_deadlock() -> None:
    # Regression: learner raises while the collector is blocked put()-ing into the
    # full maxsize-1 queue (steady state). The pipeline must propagate the learner's
    # exception promptly (no infinite join) and leave no collector thread alive.
    counter = {"c": 0}

    def collect() -> int:
        i = counter["c"]
        counter["c"] += 1
        return i  # instant -> collector races ahead and parks in the blocking put

    def learn(i: int) -> None:
        if i == 1:
            time.sleep(0.05)  # let the collector reach the full-queue put
            raise ValueError("learner boom")

    start = time.monotonic()
    with pytest.raises(ValueError, match="learner boom"):
        run_pipeline(collect=collect, learn=learn, iterations=50, overlap=True)
    elapsed = time.monotonic() - start

    assert elapsed < 5.0  # bounded shutdown, not a hang on join()
    assert all(t.name != "rl-collector" or not t.is_alive() for t in threading.enumerate())


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


# --- Melee windowed PPO update: parity + sync-case ratio ----------------------
_TINY_CFG = ArchConfig(
    d_model=16, n_layers=1, n_heads=2, L_ctx=8, char_vocab=8, char_dim=4, stage_vocab=8, stage_dim=2
)
_STATS = {
    key: FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0)
    for f in FLOAT_FEATURES
    for key in (f, f"nana_{f}")  # consolidate_key strips ego_/opp_ but keeps the nana_ infix
}
_CAT_SUFFIXES = frozenset({"stock", "action", "jumps_used", "airborne", "hurtbox_state"})
_GAE = {"gamma": 0.99, "gae_lambda": 0.95}


def _flat_frame(rng: np.random.Generator) -> dict:
    out: dict[str, float] = {}
    for p in ("p1", "p2", "p1_nana", "p2_nana"):
        for suf in POST_FIELD_SUFFIXES:
            out[f"{p}_{suf}"] = float(rng.integers(0, 3)) if suf in _CAT_SUFFIXES else float(rng.standard_normal())
    out["stage"] = 3
    out["p1_character"] = 1
    out["p2_character"] = 2
    return out


def _full_stream(T: int, *, logp: np.ndarray | None = None, seed: int = 0) -> FinalizedStream:
    rng = np.random.default_rng(seed)
    return FinalizedStream(
        ego_port=1,
        matchup=None,
        flat=tuple(_flat_frame(rng) for _ in range(T + 1)),
        ego_act=rng.uniform(-1, 1, size=(T, A_DIM)).astype(np.float32),
        act_idx=rng.integers(0, 5, size=(T, 4)).astype(np.int64),
        logp=(rng.standard_normal(T).astype(np.float32) if logp is None else logp),
        rew=rng.standard_normal(T).astype(np.float32),
        terminated=np.array([t == T - 1 for t in range(T)]),  # single complete episode
        truncated=np.zeros(T, bool),
    )


def _forward_frame_arrays(net: PolicyValueNet, windows: list, T: int, d: int):
    """Run the net over ``windows`` once and gather per-frame hidden/value/behavior-logp
    by ``frame_pos`` (works for any window length; each real frame appears once)."""
    batch = collate_windows(windows, _STATS)
    with torch.no_grad():
        hidden = net.forward_full(batch.context)  # [B, L, d]
        values = net.values(hidden)  # [B, L]
        logp = FactoredCategorical(net.policy_logits(hidden)).log_prob(batch.act_idx)  # [B, L]
    h = np.zeros((T + 1, d), np.float32)
    v = np.zeros(T + 1, np.float32)
    lp = np.zeros(T + 1, np.float32)
    for r, w in enumerate(windows):
        for j, fr in enumerate(w.frame_pos.tolist()):
            if fr >= 0:
                h[fr] = hidden[r, j].numpy()
                v[fr] = float(values[r, j])
                lp[fr] = float(logp[r, j])
    return h, v, lp


class _HeadActor(torch.nn.Module):
    """Linear head over a precomputed trunk hidden (tianshou actor mirror of policy_head)."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.head = torch.nn.Linear(weight.shape[1], weight.shape[0])
        self.head.weight.data.copy_(weight)
        self.head.bias.data.copy_(bias)

    def forward(self, obs, state=None, info=None):  # noqa: ANN001, ANN201
        return self.head(torch.as_tensor(obs, dtype=torch.float32)), None


class _HeadCritic(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.head = torch.nn.Linear(weight.shape[1], weight.shape[0])
        self.head.weight.data.copy_(weight)
        self.head.bias.data.copy_(bias)

    def forward(self, obs):  # noqa: ANN001, ANN201
        return self.head(torch.as_tensor(obs, dtype=torch.float32))


class _FactoredTorchDist:
    """FactoredCategorical wrapped in the log_prob/entropy/sample surface tianshou's
    ProbabilisticActorPolicy needs — so both update paths share the exact same
    distribution and only the surrounding PPO loss math is under test."""

    def __init__(self, logits: torch.Tensor) -> None:
        self._fc = FactoredCategorical(logits)

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        return self._fc.log_prob(act.long())

    def entropy(self) -> torch.Tensor:
        return self._fc.entropy()

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:  # noqa: ANN001, B008
        return self._fc.sample()

    @property
    def mode(self) -> torch.Tensor:
        return torch.stack([lp.argmax(-1) for lp in self._fc._log_probs], dim=-1)


def _tianshou_one_step(net: PolicyValueNet, stream: FinalizedStream, h: np.ndarray, cfg: PPOConfig, lr: float):
    """One stock-tianshou PPO update on the same data/hidden/heads; returns head deltas."""
    T = stream.n_transitions
    ph0 = net.policy_head.weight.detach().clone()
    vh0 = net.value_head.weight.detach().clone()
    actor = _HeadActor(net.policy_head.weight.detach(), net.policy_head.bias.detach())
    critic = _HeadCritic(net.value_head.weight.detach(), net.value_head.bias.detach())
    policy = ProbabilisticActorPolicy(
        actor=actor,
        dist_fn=_FactoredTorchDist,
        action_space=gym.spaces.MultiDiscrete([A_VOCAB, A_VOCAB, A_VOCAB, A_VOCAB]),
        action_scaling=False,
        action_bound_method=None,
    )
    algo = PPO(
        policy=policy,
        critic=critic,
        optim=AdamOptimizerFactory(lr=lr),
        eps_clip=cfg.clip,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        advantage_normalization=True,  # tianshou default (matched)
        value_clip=False,  # tianshou default (matched)
    )
    buf = VectorReplayBuffer(T, buffer_num=1)
    ids = np.zeros(1, np.int64)
    for t in range(T):
        buf.add(
            Batch(
                obs=h[t : t + 1],
                act=stream.act_idx[t : t + 1],
                rew=stream.rew[t : t + 1],
                terminated=stream.terminated[t : t + 1],
                truncated=stream.truncated[t : t + 1],
                obs_next=h[t + 1 : t + 2],
            ),
            buffer_ids=ids,
        )
    with policy_within_training_step(algo.policy):
        algo.update(buf, batch_size=None, repeat=1)
    return actor.head.weight.detach() - ph0, critic.head.weight.detach() - vh0


def test_melee_ppo_matches_tianshou_one_step() -> None:
    # Same tiny net semantics, same data/hidden, kl_il_coef=0, one epoch, one minibatch,
    # grad-clip disabled (a global norm over different param sets would otherwise rescale
    # the shared head grads differently) -> head parameter deltas must match tianshou.
    torch.manual_seed(0)
    net = PolicyValueNet(_TINY_CFG)
    d = _TINY_CFG.d_model
    T = 6
    lr = 1e-3
    cfg = PPOConfig(lr=lr, clip=0.2, epochs=1, minibatch_size=10_000, ent_coef=0.01, vf_coef=0.5, max_grad_norm=1e9)

    stream0 = _full_stream(T, seed=1)
    w0 = build_windows(stream0, L_ctx=1)
    h, v, lp = _forward_frame_arrays(net, w0, T, d)

    # Behavior logp is on-policy (stock PPO recomputes it identically at epoch 0).
    stream = replace(stream0, logp=lp[:T].astype(np.float32))
    ((adv, returns),) = gae_inputs([stream], [v], **_GAE)
    windows = [scatter_gae(w, adv, returns) for w in build_windows(stream, L_ctx=1)]

    ph0 = net.policy_head.weight.detach().clone()
    vh0 = net.value_head.weight.detach().clone()
    d_actor, d_critic = _tianshou_one_step(net, stream, h, cfg, lr)  # heads read pre-update snapshot

    optim = torch.optim.Adam(net.parameters(), lr=lr)
    melee_ppo_update(net, None, optim, windows, cfg, 0.0, _STATS)
    d_ph = net.policy_head.weight.detach() - ph0
    d_vh = net.value_head.weight.detach() - vh0

    assert torch.allclose(d_ph, d_actor, atol=1e-6), f"policy-head delta mismatch: {(d_ph - d_actor).abs().max()}"
    assert torch.allclose(d_vh, d_critic, atol=1e-6), f"value-head delta mismatch: {(d_vh - d_critic).abs().max()}"


def test_ratio_dev_epoch0_is_zero_on_policy() -> None:
    # Windows built from the SAME net that produced the behavior logp -> the first
    # epoch-0 minibatch recompute equals the behavior logp, so mean|ratio-1| ~ 0.
    torch.manual_seed(1)
    net = PolicyValueNet(_TINY_CFG)
    d = _TINY_CFG.d_model
    T = 5  # single episode <= L_ctx: cold-start window context matches collection exactly
    cfg = PPOConfig(clip=0.2, epochs=1, minibatch_size=10_000, max_grad_norm=1e9)

    stream0 = _full_stream(T, seed=2)
    w0 = build_windows(stream0, L_ctx=_TINY_CFG.L_ctx)
    _, v, lp = _forward_frame_arrays(net, w0, T, d)
    stream = replace(stream0, logp=lp[:T].astype(np.float32))
    ((adv, returns),) = gae_inputs([stream], [v], **_GAE)
    windows = [scatter_gae(w, adv, returns) for w in build_windows(stream, L_ctx=_TINY_CFG.L_ctx)]

    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    metrics = melee_ppo_update(net, None, optim, windows, cfg, 0.0, _STATS)
    assert metrics["ratio_dev_epoch0"] < 1e-4, metrics["ratio_dev_epoch0"]


def _on_policy_windows(net: PolicyValueNet, T: int, *, seed: int) -> list:
    """GAE-scattered windows whose behavior logp equals the net's own recompute."""
    stream0 = _full_stream(T, seed=seed)
    w0 = build_windows(stream0, L_ctx=_TINY_CFG.L_ctx)
    _, v, lp = _forward_frame_arrays(net, w0, T, _TINY_CFG.d_model)
    stream = replace(stream0, logp=lp[:T].astype(np.float32))
    ((adv, returns),) = gae_inputs([stream], [v], **_GAE)
    return [scatter_gae(w, adv, returns) for w in build_windows(stream, L_ctx=_TINY_CFG.L_ctx)]


def test_approx_kl_nonnegative() -> None:
    # k3 estimator: even when probability mass moves TOWARD the reference actions
    # (which drove the old signed estimator negative), both approx KLs stay >= 0.
    torch.manual_seed(3)
    net = PolicyValueNet(_TINY_CFG)
    cfg = PPOConfig(clip=0.2, epochs=3, minibatch_size=1, target_kl=1e9, max_grad_norm=1e9)
    windows = _on_policy_windows(net, 6, seed=3)

    optim = torch.optim.Adam(net.parameters(), lr=1e-2)  # big steps: real drift every epoch
    metrics = melee_ppo_update(net, None, optim, windows, cfg, 0.0, _STATS)
    assert metrics["approx_kl_update"] >= 0.0, metrics["approx_kl_update"]
    assert metrics["approx_kl_collect"] >= 0.0, metrics["approx_kl_collect"]
    assert metrics["approx_kl_update"] > 0.0  # lr=1e-2 for 3 epochs must register as movement


def test_zero_lr_gives_zero_update_kl() -> None:
    # With lr=0 the net never moves, so current == the pre-update reference exactly:
    # the update KL is 0 and the early stop can never fire.
    torch.manual_seed(4)
    net = PolicyValueNet(_TINY_CFG)
    cfg = PPOConfig(clip=0.2, epochs=3, minibatch_size=1, target_kl=0.015, max_grad_norm=1e9)
    windows = _on_policy_windows(net, 6, seed=4)

    optim = torch.optim.Adam(net.parameters(), lr=0.0)
    metrics = melee_ppo_update(net, None, optim, windows, cfg, 0.0, _STATS)
    assert metrics["approx_kl_update"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["epochs_used"] == float(cfg.epochs)


def test_new_diagnostics_ranges() -> None:
    torch.manual_seed(5)
    net = PolicyValueNet(_TINY_CFG)
    cfg = PPOConfig(clip=0.2, epochs=2, minibatch_size=1, target_kl=1e9, max_grad_norm=1e9)
    windows = _on_policy_windows(net, 6, seed=5)

    optim = torch.optim.Adam(net.parameters(), lr=1e-3)
    metrics = melee_ppo_update(net, None, optim, windows, cfg, 0.0, _STATS)
    assert 0.0 <= metrics["clip_frac"] <= 1.0
    assert 0.0 < metrics["ratio_ess"] <= 1.0
    # On-policy: recompute == recorded logp, so the importance weights are all 1 -> ESS 1.
    assert metrics["ratio_ess"] == pytest.approx(1.0, abs=1e-6)
