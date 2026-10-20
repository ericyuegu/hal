# O65 self-play PPO

O65 collects MSL rollouts, fits a separate critic, then trains the actor with
joint PPO and a direct KL loss. Read [plan.md](plan.md) for the math and
[pilot-protocol.md](pilot-protocol.md) for the 17M proxy pilot.

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
  --training.checkpoint-interval 3 --verify-resume
```

This run bypasses critic warmup. The hardware pilot uses 512 windows per actor
step and calibrates lambda_max during warmup. A short local run checks execution;
use the fixed final evaluation to assess learning.

Training uses graphs for rollout, reference replay and actor backward. The
trunk and decoder compute in BF16, with FP32 probability arithmetic and critic
heads. Training uses stored behavior log-probabilities. Use `profile` to time
a warmed update and trace a separate update; `--nsys` permits an external
Nsight capture. [performance.md](performance.md) records measured server times
and the pilot projections.

Each update appends metrics to `metrics.jsonl`. The trainer records resolved
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
