# O65 proxy: GAE lambda 1

Status: prepared; not launched.

## Treatment and control

The treatment changes `GAE_LAMBDA` from 0.99 to 1.0 per frame. Both actor
advantages and critic targets use this value. Gamma stays 0.99855 per frame.
Each rollout uses its observed rewards and bootstraps at its final state.
Terminal matches do not bootstrap. Timeouts bootstrap without joining the
next episode's trace.

The control is the finished W&B run
`ericyuegu/hal-o65/o65-debug-7965557b-6f4d1e097e`, from commit
`7965557b273b255e5fe07b6ec889ec36d8132782`. The treatment branch starts at
that commit. Its only training change is GAE lambda. The resolved run record
also reports gamma and GAE lambda explicitly.

Start from the original O63 bootstrap30 behavior-cloning checkpoint in
`pilot-inputs.json`. Do not pass `--resume` or `--resume-learner` for this
launch. Initialize fresh optimizer, scheduler and RNG states. All 2,000 RL
updates use GAE lambda 1, including the usual 32-update critic warmup. Warmup
calibrates the KL range using this run's advantage spread.

The historical control migrated an update-256 learner into a fresh match pool.
This run starts from BC without that migration. Its restart history and
calibrated KL range can therefore differ as well as GAE lambda. Report these
limits when comparing the two runs.

## Invariant inputs

- O63 bootstrap30 17M checkpoint and reference; checkpoint SHA-256
  `65e7e43e5cbb18a365a005922ec1cfd026ad807c045eae13b331f487b4ec82e6`.
- Reward: damage 1, stock 120, deciding-stock bonus 50. Losses have the
  opposite sign. This does not use the production reward weights.
- Seed 65; four groups of 1,024 matches; 256-frame rollouts; D8/R4 and D10/R2.
- The original pilot sampler, episode conditions, opponent slots, snapshot
  interval, PPO clipping, optimizers, learning rates and actor step size.
- BF16 decoder; the original locked dependencies and pinned runtime image
  `ghcr.io/ericyuegu/hal@sha256:c9160a7bb2dc0aa3eb6dc583b1c4753ff98d9a40f802512ca402e2b93a3a7413`.
- Use the recorded control environment:
  Python 3.14.3, Torch 2.11.0+cu130, NumPy 2.1.3 and CUDA 13.0 on an RTX PRO
  6000 Blackwell Server Edition. Record any environment difference.
- The hashes in `pilot-inputs.json`, `pilot-sampler.json` and
  `pilot-evaluation.json` remain unchanged.

## Endpoint and evaluation

Train to update 2,000, with checkpoints every 32 updates. Use one separate GCE
RTX PRO 6000. Do not use the GPU that serves netplay. Keep the VM allocated
after training. Bound the training job to 10 hours, including final evaluation;
this bound stops the job, not the VM.

Use the original 384-game MSL grid against the reference at D8/R4, u=0.95 and
KL coefficient 1. Use the same game seeds, stages and port orientations.
These seeds were already used by the historical control, so this comparison
is a paired diagnostic, not a new independent held-out result. Do not select
an intermediate checkpoint by offline metrics or extend training after seeing
the endpoint results.

Report gameplay win rate and net stocks per minute with uncertainty. Track
critic error, conditional KL, entropy, clipping and action counts during
training. The original endpoint evaluation alone does not test the reported
forward-smash behavior at smaller KL coefficients; any such evaluation must
use the same conditions for both actors and be labeled as an additional
diagnostic.

Record the treatment Git SHA, resolved configuration, source checkpoint
manifest and shard hashes, environment, GPU, input hashes, W&B run URL,
training command, final actor hash and evaluation results before drawing a
conclusion.
