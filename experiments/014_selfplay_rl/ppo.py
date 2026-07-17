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
from nets_melee import FactoredCategorical
from nets_melee import PolicyValueNet
from rl_config import PPOConfig
from rollout import Window
from rollout import collate_windows
from tianshou.algorithm import PPO
from tianshou.data import Batch
from tianshou.data import ReplayBuffer
from tianshou.data import to_torch_as
from tianshou.data.types import LogpOldProtocol
from tianshou.data.types import RolloutBatchProtocol
from tianshou.utils.torch_utils import policy_within_training_step

from hal.data.stats import FeatureStats
from hal.training.features import Context

# tianshou PPO defaults matched by ``melee_ppo_update`` (read from the installed
# ``PPO.__init__`` / ``_update_with_batch``): per-minibatch advantage normalization
# ON with eps == A2C._eps and torch's default unbiased (ddof=1) std; value clipping
# OFF (plain returns-vs-value MSE — see the vf_loss line).
_ADV_NORM = True
_ADV_NORM_EPS = 1e-8  # tianshou A2C._eps


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


# %%
# --- windowed Melee PPO update (hand-transcribed from tianshou PPO) -----------
@jaxtyped(typechecker=beartype)
def _masked_mean(x: Float[torch.Tensor, "B L"], valid: torch.Tensor) -> Float[torch.Tensor, ""]:
    """Mean of ``x`` over valid (scored) positions only. Pad rows and the value-only
    bootstrap frame carry ``valid=False`` and are excluded from every loss term."""
    m = valid.to(x.dtype)
    return (x * m).sum() / m.sum().clamp(min=1.0)


def _forward_logp(
    net: PolicyValueNet, batch_context: Context, act_idx: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, FactoredCategorical]:
    hidden = net.forward_full(batch_context)
    logits = net.policy_logits(hidden)
    dist = FactoredCategorical(logits)
    return net.values(hidden), dist.log_prob(act_idx), dist


def value_warmup_update(
    net: PolicyValueNet,
    optim: torch.optim.Optimizer,
    windows: list[Window],
    ppo_cfg: PPOConfig,
    stats: dict[str, FeatureStats],
    *,
    device: str = "cpu",
    generator: torch.Generator | None = None,
    on_step: Callable[[], None] | None = None,
) -> dict[str, float]:
    """Value-head-only warm-start: fit ``V(s)`` to the GAE returns while the actor stays
    byte-identical to the IL policy.

    The trunk forward runs under ``no_grad`` (its hidden is a detached leaf), so only the
    (zero-initialised) ``value_head`` receives gradient — the caller passes an optimizer
    over ``value_head.parameters()`` exactly. MSE against ``returns`` over valid positions,
    ``ppo_cfg.epochs`` passes of ``ppo_cfg.minibatch_size`` windows. ``on_step`` advances the
    EMA per minibatch if given, but ``melee_train`` deliberately passes ``on_step=None`` here:
    even with only the critic moving, ``ema.update`` is not a fixed point (~ulp drift), so
    letting it run would nudge the acting EMA off byte-identical IL during warmup."""
    if not windows:
        raise ValueError("value_warmup_update called with no windows")
    n = len(windows)
    mb = max(1, ppo_cfg.minibatch_size)
    vf_losses: list[float] = []
    for _ in range(ppo_cfg.epochs):
        perm = torch.randperm(n, generator=generator).tolist()
        for start in range(0, n, mb):
            batch = collate_windows([windows[i] for i in perm[start : start + mb]], stats).to(device)
            with torch.no_grad():
                hidden = net.forward_full(batch.context)
            values = net.values(hidden)  # grad flows only to value_head; trunk hidden detached
            vf_loss = _masked_mean((batch.returns - values).pow(2), batch.valid)
            optim.zero_grad()
            vf_loss.backward()
            optim.step()
            if on_step is not None:
                on_step()
            vf_losses.append(float(vf_loss.detach()))
    return {"vf_loss": float(np.mean(vf_losses))}


@torch.no_grad()
def _windowed_kls(
    net: PolicyValueNet,
    windows: list[Window],
    stats: dict[str, FeatureStats],
    logp_ref: torch.Tensor,
    device: str,
) -> tuple[float, float]:
    """One no-grad forward over the whole window set → two approx KLs of the current
    policy: vs the pre-update recompute (``logp_ref``, rows in ``windows`` order — the
    early-stop signal: movement caused by THIS update phase) and vs the recorded
    collection log-prob (diagnostic; carries the window-context recompute floor).

    Both use the non-negative k3 estimator ``(r - 1) - log r`` with ``r = exp(logp_cur
    - logp_ref)`` (Schulman's KL approximation): unbiased for ``KL(ref || cur)`` under
    reference-policy samples and ≥ 0 pointwise, so drift never reads as negative."""
    batch = collate_windows(windows, stats).to(device)
    _, logp_cur, _ = _forward_logp(net, batch.context, batch.act_idx)
    d_upd = logp_cur - logp_ref.to(device)
    d_col = logp_cur - batch.logp_old
    kl_update = float(_masked_mean((d_upd.exp() - 1.0) - d_upd, batch.valid))
    kl_collect = float(_masked_mean((d_col.exp() - 1.0) - d_col, batch.valid))
    return kl_update, kl_collect


@torch.no_grad()
def _reference_pass(
    net: PolicyValueNet,
    windows: list[Window],
    stats: dict[str, FeatureStats],
    mb: int,
    device: str,
) -> tuple[torch.Tensor, float, float]:
    """Pre-update snapshot over ALL windows, chunked at ``mb`` windows per forward:
    per-position ``logp_ref`` [n, L] (cpu — the KL early-stop reference), plus two
    collection diagnostics measured against the recorded behavior log-prob:
    ``ratio_dev_epoch0`` (mean ``|ratio - 1|`` over valid positions) and ``ratio_ess``
    (normalized effective sample size of the importance weights, 1.0 when the recompute
    matches collection exactly)."""
    n = len(windows)
    L = windows[0].logp.shape[0]
    logp_ref = torch.zeros(n, L)
    dev_sum = 0.0
    w_sum = 0.0
    w2_sum = 0.0
    n_valid = 0
    for start in range(0, n, mb):
        batch = collate_windows(windows[start : start + mb], stats).to(device)
        _, logp, _ = _forward_logp(net, batch.context, batch.act_idx)
        logp_ref[start : start + mb] = logp.detach().cpu()
        w = (logp - batch.logp_old).exp()[batch.valid]
        dev_sum += float((w - 1.0).abs().sum())
        w_sum += float(w.sum())
        w2_sum += float(w.pow(2).sum())
        n_valid += int(batch.valid.sum())
    ratio_dev = dev_sum / max(1, n_valid)
    ratio_ess = (w_sum * w_sum) / (n_valid * w2_sum) if w2_sum > 0.0 else 0.0
    return logp_ref, ratio_dev, ratio_ess


def melee_ppo_update(
    net: PolicyValueNet,
    il_net: PolicyValueNet | None,
    optim: torch.optim.Optimizer,
    windows: list[Window],
    ppo_cfg: PPOConfig,
    kl_il_coef: float,
    stats: dict[str, FeatureStats],
    *,
    device: str = "cpu",
    generator: torch.Generator | None = None,
    on_step: Callable[[], None] | None = None,
) -> dict[str, float]:
    """Windowed PPO update over Melee rollout windows, with an approx-KL early stop
    and an optional KL-to-IL anchor.

    ``on_step`` (if given) runs after every optimizer step — the hook the caller uses to
    advance the EMA behavior weights per minibatch (mirrors ``update_with_kl_stop``'s
    per-epoch ``on_epoch_end``, one level finer).

    The per-minibatch loss is a masked transcription of tianshou ``PPO._update_with_batch``
    (matched defaults: advantage_normalization ON with ddof=1 std + eps 1e-8, value_clip
    OFF). One trunk forward per minibatch yields policy logits and values at every window
    position; only ``valid`` positions (real scored transitions) enter any loss term. The
    PPO ratio uses the RECORDED behavior log-prob (``window.logp``), so it stays a genuine
    importance weight against the collector's policy (the same off-policy discipline as
    ``BehaviorLogpPPO``). When ``il_net`` is given, ``+kl_il_coef * KL(current || IL)`` (IL
    forward under no_grad) is added — the anchor that keeps the policy near the warm-start.

    KL early-stop is DECOUPLED from the collection log-prob: the windowed recompute
    carries an irreducible context-approximation floor vs collection (mid-stream windows
    see less history than the rolling collection buffer did), so measuring drift against
    ``window.logp`` would trip ``target_kl`` from epoch 1 forever and silently cap PPO at
    one epoch. Instead, a dedicated no-grad pass BEFORE the first optimizer step snapshots
    each window's recomputed ``logp`` (an exact pre-update reference), and the between-epoch
    stop measures ``approx_kl_update`` = k3 KL of the current policy vs that snapshot —
    i.e. movement caused by THIS update phase only. The PPO ratios still use the recorded
    collection log-prob (the genuine importance weight); ``approx_kl_collect`` (current vs
    collection, same k3 form) is returned as a diagnostic alongside.

    ``ppo_cfg.minibatch_size`` counts WINDOWS per minibatch here (each window is up to
    ``L_ctx`` transitions), not individual transitions. Returns diagnostic scalars incl.
    ``ratio_dev_epoch0`` = mean ``|ratio - 1|`` over ALL valid positions in the reference
    pass (≈ 0 when the update policy still equals the collector policy), ``ratio_ess``
    (normalized ESS of the reference-pass importance weights), and ``clip_frac`` (fraction
    of valid positions outside the clip range, averaged over every minibatch run)."""
    if not windows:
        raise ValueError("melee_ppo_update called with no windows")
    n = len(windows)
    mb = max(1, ppo_cfg.minibatch_size)
    clip_losses, vf_losses, entropies, kl_ils, gnorms, clip_fracs = [], [], [], [], [], []
    approx_kl_update = 0.0
    approx_kl_collect = 0.0
    epochs_used = 0
    logp_ref, ratio_dev_epoch0, ratio_ess = _reference_pass(net, windows, stats, mb, device)
    for _ in range(ppo_cfg.epochs):
        perm = torch.randperm(n, generator=generator).tolist()
        for start in range(0, n, mb):
            mb_idx = perm[start : start + mb]
            batch = collate_windows([windows[i] for i in mb_idx], stats).to(device)
            valid = batch.valid
            values, logp, dist = _forward_logp(net, batch.context, batch.act_idx)

            adv = batch.adv
            if _ADV_NORM:
                adv_valid = adv[valid]
                if adv_valid.numel() < 2:
                    raise ValueError(
                        f"advantage normalization needs >=2 valid positions, got {adv_valid.numel()} "
                        "(ddof=1 std is NaN below that) — collect more transitions or use bigger minibatches"
                    )
                adv = (adv - adv_valid.mean()) / (adv_valid.std() + _ADV_NORM_EPS)
            ratio = (logp - batch.logp_old).exp()
            surr1 = ratio * adv
            surr2 = ratio.clamp(1.0 - ppo_cfg.clip, 1.0 + ppo_cfg.clip) * adv
            clip_loss = -_masked_mean(torch.min(surr1, surr2), valid)
            vf_loss = _masked_mean((batch.returns - values).pow(2), valid)  # value_clip OFF
            entropy = _masked_mean(dist.entropy(), valid)
            loss = clip_loss + ppo_cfg.vf_coef * vf_loss - ppo_cfg.ent_coef * entropy

            kl_il = torch.zeros((), device=loss.device)
            if il_net is not None:
                with torch.no_grad():
                    il_hidden = il_net.forward_full(batch.context)
                    il_dist = FactoredCategorical(il_net.policy_logits(il_hidden))
                kl_il = _masked_mean(dist.kl_to(il_dist), valid)
                loss = loss + kl_il_coef * kl_il

            optim.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), ppo_cfg.max_grad_norm)
            optim.step()
            if on_step is not None:
                on_step()

            with torch.no_grad():
                clip_fracs.append(float(_masked_mean(((ratio - 1.0).abs() > ppo_cfg.clip).float(), valid)))
            clip_losses.append(float(clip_loss.detach()))
            vf_losses.append(float(vf_loss.detach()))
            entropies.append(float(entropy.detach()))
            kl_ils.append(float(kl_il.detach()))
            gnorms.append(float(gnorm))
        epochs_used += 1
        approx_kl_update, approx_kl_collect = _windowed_kls(net, windows, stats, logp_ref, device)
        if approx_kl_update > ppo_cfg.target_kl:
            break
    return {
        "clip_loss": float(np.mean(clip_losses)),
        "vf_loss": float(np.mean(vf_losses)),
        "entropy": float(np.mean(entropies)),
        "approx_kl_update": approx_kl_update,
        "approx_kl_collect": approx_kl_collect,
        "kl_il": float(np.mean(kl_ils)),
        "gnorm": float(np.mean(gnorms)),
        "epochs_used": float(epochs_used),
        "ratio_dev_epoch0": ratio_dev_epoch0,
        "ratio_ess": ratio_ess,
        "clip_frac": float(np.mean(clip_fracs)),
    }
