# O65 self-play PPO

O65 collects MSL rollouts, fits a separate critic, then trains the actor with
joint PPO and a direct KL loss. Read [plan.md](plan.md) for the math and
[pilot-protocol.md](pilot-protocol.md) for the 17M proxy pilot.
[pfsp-league-protocol.md](pfsp-league-protocol.md) states the current
anti-collapse treatment.

Build MSL and the adapter with [the native instructions](../../hal/sim/NATIVE.md).
A local functional run can use smaller actor steps to fit the RTX 3060:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 \
PYTHONPATH="$HOME/src/melee-sim-light:$PWD" \
uv run python experiments/065_selfplay_ppo.py train \
  --checkpoint runs/proxy/final.pt --sampler experiments/o65/pilot-sampler.json \
  --data-dir "$HOME/src/melee-sim-light/data" --output runs/native-check \
  --cpus 0 1 --groups 1 --matches-per-shard 8 --updates 3 \
  --training.windows-per-step 32 --training.critic-warmup-updates 0 \
  --training.advantage-scale batch --training.checkpoint-interval 3 --verify-resume
```

This run bypasses critic warmup, so it scales advantages by each minibatch's
spread. The hardware pilot uses 512 windows per actor step and calibrates
lambda_max and the fixed advantage scale during warmup. A short local run checks execution;
use the fixed final evaluation to assess learning.

Training uses graphs for rollout, reference replay and actor backward. The
trunk and decoder compute in BF16, with FP32 probability arithmetic and critic
heads. Training uses stored behavior log-probabilities. Use `profile` to time
a warmed update and trace a separate update; `--nsys` permits an external
Nsight capture. [performance.md](performance.md) records measured server times
and the pilot projections.

Each update prints a compact, flushed progress line and appends metrics to
`metrics.jsonl`. Checkpoint lines report capture, worker and wait seconds;
snapshot lines report league rotation time. The fixed-lane pilot warms
preallocated slot policies before use and loads banked weights into them at
rotation.
The trainer records resolved
settings, environment and input hashes in `resolved.json`. It publishes
checkpoint manifests after writing their referenced files. `--resume` restores
complete training state; `--updates` sets the total update count. The optional
`--verify-resume` checks restored state and the next update against repeated
controls, then restores the starting checkpoint.

[pilot.py](pilot.py) builds MSL, trains with per-anchor progress metrics,
publishes every 32 updates, then evaluates the final checkpoint in MSL.
Inspect its Modal request locally:

```sh
uv run python -m scripts.launch_o65_modal --mode pilot --dry-run
```

The input files record simulator and checkpoint hashes, sampler counts and the
final evaluation grid. [The evaluation instructions](../../hal/eval/O65.md)
describe paired matches and outcome statistics.

## Cleanup before merging to main

The branch adds about 31,000 lines. We plan a cleanup pass before merging.
These items look cheap:

- Remove the v8 update-256 learner migration. It served the fixed-slot
  relaunch, which is finished, and real v8 checkpoints no longer load: their
  scalar residual critic differs from the HL-Gauss critic, and their learner
  trained on the damage reward. The migration spans `restore_pilot_learner`,
  `legacy_pilot_digest`, `PILOT_SOURCE`, `--resume-learner` in `pilot.py` and
  the launcher, `inherit_metrics`, and their tests.
- Pick one entry point. Inside the Modal container, `pilot.py` installs
  inputs, builds MSL, runs `065_selfplay_ppo.py train` as a subprocess,
  publishes, then runs the held-out evaluation. Folding those steps into the
  launcher or the trainer would leave a single path to read.
- Remove the GCE existing-VM launch path: `scripts/launch_o65_existing.py`,
  `scripts/launch_o65.sh`, `docker/o65-existing.sh` and
  `tests/test_launch_o65_existing.py`. Modal replaced it.
- Merge the protocols into `plan.md` and Git history. `pilot-protocol.md`
  still describes the 54-lane layout and Beta(5,1.3). `gae-one-protocol.md`
  describes a finished run. `stalls.md` and `performance.md` hold dated
  measurements.
- Decide whether to keep the `qualify` and `profile` subcommands and
  `GpuProfileCli` (about 400 lines). `GpuProfileCli` defaults to CPUs 0-15,
  so its test fails on hosts with fewer CPUs.
- Drop the remaining v8 and v9 checkpoint readers once no evaluation needs
  those actors.
- Audit the roughly 30 O65 test files for overlap, starting with the
  qualification and precision fixtures.
