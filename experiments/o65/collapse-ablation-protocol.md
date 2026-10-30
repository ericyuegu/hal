# O65 collapse ablation

Status: prepared; not launched.

The GAE lambda 1 run (W&B `o65-debug-84fafaef-5487e230e2`, commit `84fafaef`)
drifted into smash-attack spam by update 576, while its win rate against
pi_ref rose to about 70%. The PFSP league branch changed five settings at
once, so a healthy league run would not tell us which change mattered. Here we
run the collapsed configuration on the current code, then switch one setting
per run to a new value and look for the switch that keeps the policy healthy.

## Arms

Every arm starts from the O63 bootstrap30 17M proxy in `pilot-inputs.json`,
uses seed 65, four groups of 1,024 matches, the pilot sampler with the
Beta(5, 1.3) percentile prior of the collapsed run, and trains for 640 PPO
updates after critic warmup. The collapsed run degraded within 544 PPO
updates.

The `old` arm reproduces the collapsed run:

| Setting | `old` value | Switched value | Arm |
|---|---|---|---|
| Opponents | 54 self-play lanes, anchors 4:2:2:2; slots 1-3 take the latest snapshots every 256 updates; unweighted anchor advantages | 16 self-play lanes, anchors 12:12:12:12; PFSP league every 64 updates; anchor weights clip(2(1-s), 0.25, 1.75) | `matchup` |
| Reward | stocks plus 1 per percent dealt and -1 per percent taken; gamma 0.99855 (8 s half-life) | stocks only; gamma 2^(-1/1800) (30 s half-life) | `reward` |
| Critic | BC value head plus a residual SwiGLU over the decoder state, width 128, scalar squared error | HL-Gauss over 120 bins; normalized inputs, width 512, three residual SwiGLU blocks | `critic` |
| Advantage scale | each minibatch's spread | spread measured in critic warmup | `scale` |

All arms keep 32 critic warmup updates and no value loss in the actor step,
so the `critic` arm changes only what the critic can represent. They also
keep the lambda_max cap of 100 and the detached decoder state from the branch,
which the collapsed run lacked. The `old` arm therefore checks that the
collapse survives those changes.

## Reading the arms

A switched arm that stays healthy shows that the collapse needs the `old`
value of its setting.

- `matchup` healthy: smash spam pays only against a pool dominated by the
  student and its latest snapshots. The collapsed policy also beat pi_ref, so
  we expect this arm to collapse.
- `reward` healthy: the policy was farming per-percent damage that an 8 s
  horizon overweights. One follow-up run with only the damage term removed
  separates the damage term from the half-life.
- `critic` healthy: a critic with too little capacity or a poor output form
  biased the advantages. With GAE lambda 1, each 256-frame rollout bootstraps
  from the value of its final state with weight 0.69 to 1 at an 8 s half-life,
  so critic bias reaches every advantage.
- `scale` healthy: dividing by the minibatch spread weakened the KL term as
  the policy drifted and KL grew.
- Several arms healthy: the collapse needs all of those settings at once.
- `old` healthy: the collapse does not reproduce on the current code, and the
  other arms carry no information.

The `critic` arm calibrates lambda_max from its own warmup advantages. A more
accurate critic shrinks that spread, so we report each arm's lambda_max and
treat a difference above 20% from `old` as a confound.

## Collapse rule

We fix the rule before launch. Each update compares the student's side of the
pi_ref-slot lanes against the same lanes during critic warmup, when the actor
still equals pi_ref:

- smash ratio: the fraction of student frames in forward, up and down smash
  action states, divided by the warmup fraction
- action-state divergence: Jensen-Shannon divergence between the student's
  action-state histogram and the warmup histogram

An arm collapses when its smash ratio exceeds 2 or its divergence exceeds 0.1
nats for 64 consecutive updates. Four traced matches per arm, viewed in the
MSL viewer, check that the rule matches what we see. They do not override it.

Each arm also logs the critic's explained variance against lambda 1 targets
by lane type, and against full discounted returns on the traced matches.

## Checks inside every run

The trainer recomputes each decision's stock reward from the stock columns and
the student's side and raises an error on a mismatch.

## Launch and budget

All five arms launch together from one launcher process on Modal, one RTX PRO
6000 each, with 2.5-hour timeouts. The ablation ledger in
`runs/modal-ablation/budget.jsonl` reserves each run's timeout and caps the
program at $65; expected spend is about $38. The arms skip the held-out
evaluation.
