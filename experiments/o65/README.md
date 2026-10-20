# KL-regularized self-play PPO

O65 collects MSL rollouts, fits a separate critic, then updates the actor with
joint PPO and a direct KL loss. The single-file experiment uses the O63
bootstrap30 proxy at delay 8/replan 4 and delay 10/replan 2. It checks executed
offsets against checkpoint supervision before training.

Build MSL and the adapter with [the native instructions](../../hal/sim/NATIVE.md).
Run a short deterministic check locally:

```sh
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=1 \
PYTHONPATH="$HOME/src/melee-sim-light" \
uv run python experiments/065_selfplay_ppo.py train \
  --checkpoint runs/proxy/final.pt --sampler experiments/o65/pilot-sampler.json \
  --data-dir "$HOME/src/melee-sim-light/data" --output runs/native-pilot \
  --cpus 0 1 --groups 2 --matches-per-shard 1 --updates 2 \
  --training.windows-per-step 4 --training.critic-warmup-updates 0 \
  --training.checkpoint-interval 2 --training.deterministic \
  --evaluation-every 0 --verify-resume
```

This functional check bypasses critic warmup. The defaults use Muon LR 1e-5,
four critic epochs, initial lambda maximum 100, Master
deployment identity, and data floor 100. Warmup calibrates the lambda maximum.
A short functional run cannot establish improved gameplay.

CUDA training uses graphs for dense tick storage, actor-shaped reference replay,
and actor forward/backward. The decoder and trunk compute in BF16; probability
arithmetic and critic heads use FP32. Captured casts read current actor weights.
PPO uses stored behavior log-probs. See [dense validation and measurements](performance/dense.md).

Normal training uses default Torch algorithms. `--training.deterministic`
selects deterministic algorithms and configures cuBLAS before CUDA starts.
`--verify-resume` uses the training graphs and execution mode. It checks restored
model, optimizer, scheduler, sampler, simulator and RNG states exactly, then
checks the next update within the measured spread of two control repetitions.
It restores the original boundary after the check. Training format v5 and pool snapshot v3
reject earlier persisted formats.

The sampler JSON uses `hal.o65.sampler.v1`. It reads identity/character counts,
weighted character pairs, HAL stage ids, a shared starting stock count, and
integer starting percents. [The committed sampler](pilot-sampler.json) records
its count source in [pilot-sampler-source.json](pilot-sampler-source.json).
Native reset rejects unequal starting stocks.

Each simulator lane runs a match. Each player receives a stable logical stream
id across pool rotation and resume. A block advances 128 frames. A training
window includes a player's cache history and the current block, up to 256 frames.

At each update, the loop writes timing and training metrics to `metrics.jsonl`.
It records source and input hashes, environment, and resolved settings in
`resolved.json`. Checkpoints publish an atomic manifest after every referenced
file is durable. Resume requires the same configuration, topology, source,
software, simulator, and inputs. `--updates` specifies the total target count.

Periodic evaluation publishes an actor through the checkpoint manifest and
starts a local child process. Enable it with `--evaluation-every` and
`--evaluation-protocol`. Use Xvfb when Dolphin needs a display. See
[the evaluation instructions](../../hal/eval/O65.md) for paired matches,
held-out schedules, and outcome statistics.

The [bounded pilot driver](pilot.py) installs the hash-checked inputs, builds
MSL, compares pool sizes, kills training after checkpoint publication, then
resumes. `--tiny --no-wandb` selects a local functional check. The full pilot
requires an RTX PRO 6000 and its declared CPU allocation. The
[input manifest](pilot-inputs.json) identifies the fixed source and data.
The [hardware protocol](pilot-protocol.md) defines the treatment, frequent MSL
checks, periodic Dolphin checks, learning criteria, launch, monitoring and
teardown. Run `bash scripts/launch_o65.sh --dry-run` from the final clean SHA
to inspect the one-VM request. Publish its required inputs before a real launch.
