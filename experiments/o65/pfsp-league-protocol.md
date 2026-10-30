# O65 proxy: PFSP league and stock reward against collapse

Status: prepared; not launched.

## Treatment and control

The GAE lambda 1 run (W&B `o65-debug-84fafaef-5487e230e2`) drifted toward
smash-attack spam. Its win rate against pi_ref rose to about 70% by update 576,
where an R2 upload failure stopped it, while play degraded. Most
student windows came from live self-play, and three snapshot slots were
overwritten every 256 updates. Its reward also paid damage permanently, with an
8-second half-life, so damage in the next few seconds outweighed a stock half a
minute away. This treatment changes the opponent mixture, the reward, the
discount, the critic and the advantage scale together:

- Opponent lanes: each 64-lane shard has 16 self-play lanes and 12 lanes for
  each of four anchor slots, so about 40% of student streams are self-play.
  The previous layout had 54 self-play lanes and 4:2:2:2 anchor lanes.
- League: every 64 updates, a CPU bank of up to 32 frozen students gains the
  current student. Slot 0 stays pi_ref. Slots 1-3 hold the new entry and two
  prioritized fictitious self-play draws. Each draw mixes weight
  max(0.05, 1-s)^2 with 20% uniform probability. Eviction removes the entry the
  student beats most, sparing pi_ref and the first three banked students. New
  weights load between rollouts while the slot's games continue.
- Advantage weights: anchor-lane centered advantages are multiplied by
  clip(2(1-s), 0.25, 1.75); self-play keeps weight 1. Here s is the student's
  stock share against the slot's entry: stocks taken over stocks taken and
  lost, as a per-update EMA with alpha 0.2.
- Advantage scale: centered advantages are divided by the pooled standard
  deviation of critic-warmup advantages, which carry no KL cost, instead of
  each minibatch's spread.
- Percentile prior: u comes from Beta(15,1.2) instead of Beta(5,1.3).
- Reward: 120 per stock and 50 more on the deciding stock, with no damage
  term. A KO always pays more than keeping the opponent alive.
- Discount: gamma has a 30-second half-life, 2^(-1/1800) per frame, instead of
  0.99855 (8 seconds).
- Critic: an HL-Gauss distribution over 120 bins on [-600, 600] replaces the
  scalar residual critic. Each PPO step adds 0.1 times its cross-entropy, with
  the critic's parameters held fixed, so the trunk and decoder learn value
  features. Critic warmup lasts 128 updates instead of 32, and lambda_max is
  capped at the conditioner's maximum of 100.

[plan.md](plan.md) gives the exact rules. The resolved run record names the
advantage scale, lane mix, rotation interval and Beta prior.

The control is the GAE lambda 1 run in [gae-one-protocol.md](gae-one-protocol.md),
launched from commit `84fafaef`. It stopped at update 576, so we compare the
two runs only through update 576 and report the treatment's endpoint alone.
The treatment keeps GAE lambda 1. Because it changes several settings at once,
a result shows whether the bundle prevents collapse, not which part matters.

## Invariant inputs

- O63 bootstrap30 17M checkpoint and reference, as in `pilot-inputs.json`.
- GAE lambda 1, PPO clipping, optimizers, learning rates, windows per actor
  step and critic epochs.
- Seed 65; four groups of 1,024 matches; 256-frame rollouts; D8/R4 and D10/R2.
- Critic warmup also measures the fixed advantage scale.
- `pilot-sampler.json` and `pilot-evaluation.json`. The Beta prior is a code
  default, not a sampler file field.
- Runtime image and recorded environment from the control.

The new layout gives about 2,560 student windows per update instead of about
3,776, or roughly five actor steps instead of 7.4. Compare runs at matched
updates and report this difference.

## Endpoint and evaluation

Train from BC to update 2,000 without `--resume-learner`, with checkpoints
every 32 updates. Evaluate the final checkpoint on the original 384-game MSL
grid at D8/R4, u=0.95 and KL coefficient 1, as the control protocol specified. Report win rate
and net stocks per minute against pi_ref with uncertainty, plus training-time
anchor win rates, league slot stock-share EMAs, conditional KL, entropy and action
counts. Do not select an intermediate checkpoint by offline metrics.

Record the treatment Git SHA, resolved configuration, environment, input
hashes, W&B run URL, training command, final actor hash and evaluation results
before drawing a conclusion.
