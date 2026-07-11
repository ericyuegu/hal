# 014 — Self-play PPO for Melee

## Goal

Fine-tune the 012 imitation-learning policy with self-play PPO, AlphaStar-style.
Both ports are driven by the *same* network: the learner updates fast weights
while an EMA copy (`hal/training/ema.py`) supplies the trailing **behavior**
policy that acts in the collector, stabilizing the opponent distribution. Every
run **warm-starts** from the 012 IL checkpoint (see `MeleeRLConfig.warm_start`)
so the policy begins competent and RL only has to sharpen it. A KL-to-IL penalty
(`kl_il_coef`) keeps the policy from drifting off the human-data manifold.

The core PPO loop is validated first on Gym/Atari (tianshou + envpool), then
reused verbatim for Melee behind a shared collector interface, so the RL math is
debugged before the expensive closed-loop emulator is in the picture.

## Gate ladder

- **G1** — CartPole-v1 ≥ 475 avg return (100 eval eps) in **both** sync and
  overlap (double-buffered collector/learner) modes.
- **G2** — Pong ≥ +18 within ~10M frames; learning curve tracks a CleanRL PPO
  reference.
- **G3** — warm-started EMA self-play beats the frozen 012 IL policy head-to-head
  (≥ 50 prior-distribution matches, win-rate CI excluding 0.5) with no regression
  vs the in-game CPU baseline.
- **G4** — throughput report: GPU util, lockstep fps, overlap fraction retention
  (target ≥ 70%), and a KV-cache benchmark plus numerical-equivalence check.

## Run commands (placeholders; scripts land in later milestones)

```bash
# G1: Gym/CartPole PPO smoke
uv run experiments/014_selfplay_rl/gym_train.py --task CartPole-v1

# G2: Atari/Pong PPO
uv run experiments/014_selfplay_rl/gym_train.py --task Pong-v5

# G3: Melee self-play PPO (warm-started from 012)
uv run experiments/014_selfplay_rl/melee_train.py

# G3 eval: EMA self-play vs frozen 012 IL / vs CPU
uv run experiments/014_selfplay_rl/melee_eval.py
```

## Extension seams (designed, not built)

These are cut points chosen so the single-learner/single-EMA design generalizes
without a rewrite; none are implemented in this experiment.

- **Population play** — a `Slot → policy-handle` mapping selects which weights
  each port uses. Members carried as LoRA deltas over the shared trunk let a
  single batched forward serve a whole population (cross-member batching), so
  league play costs one forward, not N.
- **Fully-async collection** — swap the double buffer for deeper rollout queues
  and switch the advantage estimator to V-trace, correcting the off-policy lag
  that deeper queues introduce.
- **Centralized critic (MAPPO)** — a value head that sees both ports' state sits
  behind the `Critic` Protocol, so the actor path is unchanged when the critic
  goes centralized.
- **Forked learner process** — if GIL contention between collector and learner
  is *measured* (not assumed), move the learner into a forked process sharing
  weights over shared memory.
