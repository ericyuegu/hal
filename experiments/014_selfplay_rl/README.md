# 014 — Self-play PPO for Melee

## Goal

Fine-tune the 012 imitation-learning policy with **EMA mirror self-play PPO**.
Both ports are driven by the *same* network: the learner updates fast weights
while an EMA copy (`hal/training/ema.py`) supplies the trailing **behavior**
policy that acts in the collector, stabilizing the opponent distribution. (This
is a single trailing opponent, not AlphaStar-style league training — a diverse
population with PFSP matchmaking remains an extension seam, see below.) Every
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

## Run commands

```bash
# G1: Gym/CartPole PPO smoke
uv run experiments/014_selfplay_rl/gym_train.py --task CartPole-v1

# G2: Atari/Pong PPO
uv run experiments/014_selfplay_rl/gym_train.py --task Pong-v5

# G3: Melee self-play PPO (warm-started from 012)
uv run experiments/014_selfplay_rl/melee_train.py --wandb --run-name <name> --rl.n-boots 6

# G3 eval: EMA vs frozen 012 IL head-to-head, then vs lvl-9 CPU vs pinned baseline.
# NEVER run evals concurrently with a live trainer on the dev box (10+ Dolphins +
# two CUDA processes hard-froze it once); run standalone, ideally memory-capped:
#   systemd-run --user --scope -p MemoryMax=20G -- uv run ...
uv run experiments/014_selfplay_rl/melee_eval.py --ckpt runs/<run>/latest.pt --h2h-matches 50 --n-boots 4
uv run experiments/014_selfplay_rl/melee_eval.py --ckpt runs/<run>/latest.pt --vs-cpu --baseline runs/baseline_012_vs_cpu.json

# G4: KV-cache benchmark (clean GPU only; co-resident training skews it)
uv run experiments/014_selfplay_rl/bench_kv.py
```

## Post-run review fixes (2026-07-16)

External review of the G3 run surfaced four issues, all fixed after the results
below were produced (so those numbers predate the fixes and later runs are not
metric-comparable to them):

- **Burn-in windows** — the PPO recompute previously tiled streams edge-to-edge
  and scored every position, so mid-episode positions saw far less context than
  the rolling `L_ctx` window the acting policy had (a ~0.057 spurious
  `ratio_dev_epoch0` floor vs a 0.015 KL budget). Windows now overlap at
  `MeleeRLConfig.ppo_window_stride` (default 64): each `L_ctx` window scores only
  its trailing `stride` positions, burning in on the rest, and streams carry a
  context-only prefix across iteration boundaries. Learner compute scales
  ~`L_ctx/stride` (4× at defaults) — cheap, the run is collection-bound.
- **KL early-stop reference** — the epoch-0 reference logp was captured lazily
  during epoch 0 (after earlier minibatches had already moved the model), and the
  signed estimator could go negative. Now: one dedicated pre-update snapshot pass
  plus the non-negative k3 estimator `(r-1) - log r` for both `approx_kl_update`
  and `approx_kl_collect`, with `clip_frac` and `ratio_ess` logged alongside.
- **Rebuild on EMA swap** — after the per-iteration EMA→act_net copy, cached K/V
  from the old weights kept serving for up to `refresh_every-1` frames (a hybrid
  behavior state no snapshot reproduces). The stepper now force-rebuilds every
  cache on the first frame after a weight swap.
- **Rebuild GPU syncs** — the cache rebuild converted device scalars per
  (row, layer) and the all-slot no-gather decode path was never taken; both fixed.

## Results (2026-07-14, run `014_selfplay_rl_g3_seed0`, checkpoint iter 4200 / 17.2M transitions)

All four gates passed. (Pre-fix numbers — see above.)

- **G1 PASS** — CartPole 500.0/500.0 avg return in sync and overlap modes.
- **G2 PASS** — Pong +20.9 at 10M frames (threshold +18), ~1713 sps, ahead of the
  CleanRL reference curve (W&B `014_pong_g2_seed0`).
- **G3 PASS** (both halves, eval JSONs in `runs/014_selfplay_rl_g3_seed0/`):
  - *Head-to-head vs frozen 012 IL* — **51/52 wins** (win rate 0.981,
    CI95 [0.942, 1.0]), mean stock diff **+2.92** [2.63, 3.19], damage
    151.0 vs 78.2 per min.
  - *vs lvl-9 CPU* (100 matches vs the 108-match pinned IL baseline) — net stock
    rate **+0.842**/min [0.706, 0.980] vs IL's **−0.728** (sign flipped, far past
    the no-regression bar); damage 184.4 dealt / 103.0 taken per min vs IL's
    122.7 / 117.7; stocks 1.45 taken / 0.61 lost per min vs 0.83 / 1.56.

### G4 throughput report (dev box: RTX 3060 12GB, 12 CPUs, 6 boots × 2 ports)

Run-average over the 4200-iteration training run (W&B history means):

| metric | value |
|---|---|
| transitions/s | 561 (~48M/day; ~700 in low-attrition stretches) |
| lockstep fps | 68.6 across 6 Dolphins (12 policy slots) |
| learner s/iter | 0.77 |
| collector queue wait s/iter | 6.7 |
| GPU util (steady state) | ~19% |

The run is **collection-bound by design**: learner work is ~10% of iteration
wall-clock, so double-buffering hides effectively all of it (overlap retention
≈ 100%, well past the ≥70% target — the binding constraint is Dolphin lockstep
throughput, not GPU contention). PPO health over the run: `epochs_used` 3.0
(the approx-KL early stop essentially never fired), `approx_kl_update` ~8e-4,
`ratio_dev_epoch0` ~0.057 (the expected window-context recompute floor),
`v_explained_var` 0.80 mean rising to 0.93 by end.

KV-cache benchmark (clean GPU, real 012 d256/L8 checkpoint, refresh_every=64;
fp32 pre-eviction equivalence max |Δlogit| 1.9e-5):

| n_slots | full ms/step | kv ms/step | speedup |
|---|---|---|---|
| 4 | 7.90 | 7.16 | 1.10× |
| 8 | 8.43 | 7.28 | 1.16× |
| 16 | 15.92 | 7.46 | **2.13×** |

At small slot counts both paths are kernel-launch-latency-bound; the FLOP
savings only become wall-clock past ~8 slots. The 50× figure sometimes quoted
for incremental decode is FLOPs-only and does not survive contact with a
12-launch-deep d256/L8 graph at batch ≤16.

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
