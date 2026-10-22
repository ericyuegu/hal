# O64 PPO on one B200 node

One torchrun process owns each GPU's pool, simulator cores, CUDA graphs and
frozen anchors. Actor and critic gradients reduce separately before clipping.
Global decision counts normalize losses and advantages. Ranks pad actor steps
to the largest local count. Rollouts stay local.

Rank zero logs reduced metrics. Behavior counts and active exposure add before
computing rates. FLOPs add across ranks; phase seconds and per-GPU memory peaks
use the largest rank value. MFU divides summed FLOPs by elapsed phase seconds
and world size times the per-GPU dense BF16 peak. Ratio quantiles retain the
`_rank0` suffix. Parameter checks run at startup, update 1, every 16 updates,
after promotion and after resume.

Training format v12 includes both rank-local `behavior_state` and rank shards.
Single-GPU training also writes v12. Exact resume rejects v8 through v11 and
requires the same world size, inputs and configuration. Rank zero saves shared
weights and optimizers; other ranks save local pool, simulator and RNG state.
Every shard must be durable before rank zero publishes the manifest. Distributed
checkpoint writes and publication are synchronous.

The B200 path always loads O64 `step-0098304.pt`, SHA-256
`e6d6303fbf25993c768b1e86fecbb350433a2fb229235b556334e9eb6ae868cf`.
Inside the job, it downloads from the recorded R2 key into `o64-inputs/` on Modal
volume `hal-o65-profile`, rejects a hash mismatch, fetches and checks all 44
statistics files, and records the input identity. The source image retains the
existing pinned simulator/game delivery. Training uses all eleven production
cadences, 512 rollout frames, four groups, reward (0.1,120,120), 32 critic warmup
updates, and a 256-update endpoint. Thus the endpoint includes 224 PPO updates.
Final evaluation follows [production-protocol.md](production-protocol.md).

`--matches` is the active count **per GPU**. `--windows-per-step` is also per
GPU; both arguments are required. Matches must be multiples of 16. The launcher
uses up to 16 shards per rank, with at least 16 matches per shard. Microbatches
default to one and must divide windows per step. No capacity is qualified yet.

Measurement runs the requested number of PPO updates with critic warmup and
head starts disabled. It prints reduced phase memory (allocated and reserved),
seconds, FLOPs and MFU as JSON on rank zero and saves local evidence. It creates
no training checkpoints, publishes no checkpoint tree, uses no W&B, and consumes
no held-out games. Each sweep size starts a fresh driver and torchrun process
group, releasing CUDA graph pools between sizes. A failed size is recorded;
remaining sizes still run within the original timeout. The job returns failure
if any size fails. Evidence is copied to the results volume at job exit.

The separate ledger `runs/modal-b200/budget.jsonl` retains full timeout dollar
reservations, including failures. Set `O65_B200_MAX_USD` explicitly. These commands
launch jobs; append `--dry-run` to build locally with no Modal request, allocation
or dollar reservation. This merge ran only dry runs and CPU tests.

Two B200 GPUs, O64, two PPO updates at a small size:

```sh
uv run python -m scripts.launch_o65_modal \
  --mode ddp --name o65-o64-2gpu-check --gpus 2 \
  --matches 16 --windows-per-step 1 --ddp-updates 2 --measurement \
  --timeout-seconds 3600 \
  --b200-max-usd "${O65_B200_MAX_USD:?Set the total B200 dollar cap}"
```

Eight B200 GPUs, one allocation with successive fresh processes:

```sh
uv run python -m scripts.launch_o65_modal \
  --mode ddp --name o65-o64-capacity --gpus 8 \
  --matches 16 --windows-per-step 1 --ddp-updates 2 --measurement \
  --measurement-sizes 16/1 64/4 256/16 1024/64 2048/128 \
  --timeout-seconds 10800 \
  --b200-max-usd "${O65_B200_MAX_USD:?Set the total B200 dollar cap}"
```

The 256-update treatment, after the NCCL check and capacity measurement:

```sh
uv run python -m scripts.launch_o65_modal \
  --mode ddp --name o65-o64-outcome-256 --gpus 8 \
  --matches "${O65_MATCHES:?Set measured active matches per GPU}" \
  --windows-per-step "${O65_WINDOWS:?Set measured windows per rank actor step}" \
  --actor-microbatches "${O65_MICROBATCHES:?Set measured microbatches per actor step}" \
  --ddp-updates 256 --timeout-seconds 36000 \
  --b200-max-usd "${O65_B200_MAX_USD:?Set the total B200 dollar cap}"
```

CUDA/NCCL, CUDA graph gradient storage, Muon, rank equality and exact resume
still need hardware validation. CPU Gloo does not establish these properties.
