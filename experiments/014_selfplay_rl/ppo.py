"""PPO whose ratio is an importance weight against a *recorded behavior* policy.

Rollouts are collected by an EMA snapshot of the learner (`act_net`), not by the
learner itself, so the learner's current policy is NOT the behavior policy. Stock
tianshou PPO recomputes ``logp_old`` with the *current* policy inside
``_preprocess_batch`` (correct only when collection is on-policy); doing that here
would make the PPO ratio ``exp(logp_current - logp_current) == 1`` at epoch 0 and
destroy the off-policy correction. ``BehaviorLogpPPO`` instead reads ``logp_old``
from the value recorded at rollout time (``batch.policy.logp_old``), so the ratio
``exp(logp_current - logp_behavior)`` is a genuine importance weight against the
behavior policy that actually generated the data.

``update_with_kl_stop`` runs the multi-epoch PPO update with an early stop once the
current policy has drifted ``target_kl`` away from the behavior policy, measured on
the whole rollout buffer between epochs.
"""

from collections.abc import Callable
from typing import cast

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float
from jaxtyping import jaxtyped
from rl_config import PPOConfig
from tianshou.algorithm import PPO
from tianshou.data import Batch
from tianshou.data import ReplayBuffer
from tianshou.data import to_torch_as
from tianshou.data.types import LogpOldProtocol
from tianshou.data.types import RolloutBatchProtocol
from tianshou.utils.torch_utils import policy_within_training_step


class BehaviorLogpPPO(PPO):
    """PPO that trusts the recorded behavior logprob instead of recomputing it.

    Only ``_preprocess_batch`` changes vs stock PPO: GAE (returns + advantages)
    is still computed with the learner's critic, but ``logp_old`` comes from the
    behavior policy's recorded ``batch.policy.logp_old`` rather than a fresh
    forward of the current policy. Everything downstream (the clipped surrogate
    in ``_update_with_batch``) is unchanged.
    """

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> LogpOldProtocol:
        if self.recompute_adv:
            self._buffer, self._indices = buffer, indices
        batch = self._add_returns_and_advantages(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)
        batch.logp_old = to_torch_as(batch.policy.logp_old, batch.v_s).flatten()
        return cast(LogpOldProtocol, batch)


@jaxtyped(typechecker=beartype)
def _behavior_kl(algo: BehaviorLogpPPO, buffer: ReplayBuffer) -> Float[torch.Tensor, ""]:
    """Mean ``logp_behavior - logp_current`` over the whole buffer (approx KL).

    A single no-grad forward of the current policy on the recorded (obs, act),
    compared against the recorded behavior logprobs. Positive once the learner
    has moved probability mass away from the behavior policy's actions.
    """
    batch, _ = buffer.sample(0)
    logp_behavior = torch.as_tensor(batch.policy.logp_old, dtype=torch.float32).flatten()
    with torch.no_grad():
        dist = algo.policy(batch).dist
        act = torch.as_tensor(batch.act, device=dist.logits.device)
        logp_current = dist.log_prob(act).flatten().cpu()
    return (logp_behavior - logp_current).mean()


def update_with_kl_stop(
    algo: BehaviorLogpPPO,
    buffer: ReplayBuffer,
    ppo_cfg: PPOConfig,
    on_epoch_end: Callable[[], None] | None = None,
) -> dict[str, float]:
    """Run up to ``ppo_cfg.epochs`` PPO epochs, stopping early on KL divergence.

    Each epoch is one ``algo.update(..., repeat=1)`` over the full buffer (which
    re-runs GAE with the current critic — tianshou's own recompute-advantage
    variant). After every epoch the approximate KL between the behavior policy
    and the current policy is measured on the whole buffer; the loop breaks once
    it exceeds ``target_kl``.

    ``on_epoch_end`` (if given) runs after each epoch's optimizer pass — the hook
    the caller uses to advance the EMA behavior weights (per-epoch EMA update).
    """
    epochs_used = 0
    approx_kl = 0.0
    last_stats: dict[str, float] = {}
    with policy_within_training_step(algo.policy):
        for _ in range(ppo_cfg.epochs):
            stats = algo.update(buffer, batch_size=ppo_cfg.minibatch_size, repeat=1)
            epochs_used += 1
            if on_epoch_end is not None:
                on_epoch_end()
            last_stats = {
                "loss": float(stats.loss.mean),
                "actor_loss": float(stats.actor_loss.mean),
                "vf_loss": float(stats.vf_loss.mean),
                "ent_loss": float(stats.ent_loss.mean),
            }
            approx_kl = float(_behavior_kl(algo, buffer))
            if approx_kl > ppo_cfg.target_kl:
                break
    return {**last_stats, "approx_kl": approx_kl, "epochs_used": float(epochs_used)}


def build_batch_add(
    *,
    obs: np.ndarray,
    act: np.ndarray,
    rew: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    obs_next: np.ndarray,
    logp_old: np.ndarray,
) -> Batch:
    """One vectorized rollout step, shaped for ``VectorReplayBuffer.add``.

    ``policy.logp_old`` carries the behavior logprob that ``BehaviorLogpPPO``
    consumes; values are deliberately NOT stored (the learner recomputes them
    from its own critic at update time).
    """
    return Batch(
        obs=obs,
        act=act,
        rew=rew,
        terminated=terminated,
        truncated=truncated,
        obs_next=obs_next,
        info=Batch(),
        policy=Batch(logp_old=logp_old),
    )
