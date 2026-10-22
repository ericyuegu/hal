# O65 on one node with multiple GPUs

Use `torchrun` with one process per CUDA device. Each rank has its own match
pool, counter streams, simulator cores, CUDA graphs and frozen anchors. The
actor and critic reduce gradients separately, after backward and before clipping.
Their loss denominators and actor advantage moments use global decision counts.
Ranks pad actor steps to the largest local step count. Rollouts stay local.

Rank zero logs reduced metrics. Ratio quantiles have a `_rank0` suffix because
local quantiles cannot give a global quantile. Parameter checks run at startup,
after update 1, every 16 updates, after promotion and after resume.

Distributed checkpoints use training format v11. Rank zero saves the shared
learner and anchors. Other ranks save local pool, simulator and RNG state.
Every shard must be durable before rank zero publishes the manifest. Distributed
checkpoint writes and publication are synchronous. Exact resume requires the
same world size, inputs and configuration. Single-GPU checkpoints retain v10.

The launcher uses a separate dollar ledger in `runs/modal-b200/budget.jsonl`.
It retains each full timeout reservation, including failed calls. It requires
`--b200-max-usd`; no dollar cap is supplied by the code. The B200 rate comes from
[Modal pricing](https://modal.com/pricing), checked on 2026-10-22.

For a proxy check with two B200 GPUs and three PPO updates:

```sh
uv run python -m scripts.launch_o65_modal \
  --mode ddp --name o65-ddp-proxy-check \
  --gpus 2 --matches 16 --ddp-updates 3 --timeout-seconds 1800 \
  --b200-max-usd "${O65_B200_MAX_USD:?Set the B200 dollar cap}"
```

Add `--dry-run` to construct the job locally without a Modal request. The small
match count selects the O63 proxy, disables critic warmup and held-out evaluation,
and saves each update. Real NCCL validation still needs to check CUDA graph
gradient storage, Muon updates, rank equality, rank-local simulator state,
checkpoint publication and exact resume on two B200 GPUs. Gloo does not establish
these properties on CUDA.

The 8-GPU launch defaults to 256 updates. Merge the `o65-247m` changes for the O64
checkpoint, production cadence and 512-frame rollouts before production training.
