# O65 update cost and measured floor

Baseline source is `57ed1c57` on `o65-dense`. The floor stream starts from that
commit. No training optimization is admitted until hardware rates, bare-model
times, the complete update trace, and the phase floor table are recorded.

Treatment is measurement instrumentation, then active match count in the batch
sweep. Control is the unchanged dense trainer at 1,024 matches. Invariants are
the O63 bootstrap30 checkpoint, seed 65, one group, 16 simulator shards, 256
rollout frames, cadences `(8,4)` and `(10,2)`, BF16 decoding, 32 windows and 16
decisions per actor step, four critic epochs, reward, objective, GAE, simulator
and adapter. These are disposable systems measurements, not policy selection.

The preceding dense measurement is 11.399 s/update, 22,997 match-frames/s and
15,912 decisions/s (181,376 student decisions). The source and complete evidence are in [dense.md](dense.md).
The former launch-count projection is not a hardware floor.

The measurement job records the trainer's actual policy batch sizes, padded
reference window count and actor step count. The bare-model probe uses saved
end-of-update weights and recorded actor inputs. Tick tokens, queues and draws are synthetic
at the observed batch sizes. It measures trunk KV append and sampling without
queue management, critic, RNG, simulation or PPO. Actor backward uses fixed
cotangents on factor log-probabilities and full distributions. The reference
retains the actor's autograd metadata and scores four 16-slot slices per trunk
forward. Its per-chunk timing excludes trainer staging and concatenation.

Hardware GEMM shapes come from PyTorch profiler records of these model calls.
The profile records eager mathematical operations separately from compiled
timing, because the compiled decoder hides ATen matrix dimensions. Hardware
GEMM probes preserve the recorded BF16/FP32 types and operand strides.
The streaming copy uses two 256 MiB buffers and counts reads plus writes.
Empty and one-element kernels are measured at 1, 32, 256 and 1,024 nodes to
separate graph launch cost from the cost per node. Eager wall time includes
Python and Triton dispatch.

Nsight owns CUPTI for one complete warmed update. A separate clean update
provides times and rates. NVTX marks rollout, reference, preparation, critic,
actor, each tick and Muon. CUDA graph nodes are traced individually. Traced
times are never used as clean throughput. Raw reports, allocator snapshots,
model inputs, configuration and hashes stay under `runs/floor/` locally and
`runs/modal-floor/` for downloaded cloud results.

The floor stream has a separate fixed budget ledger at
`runs/modal-floor/budget.jsonl`: at most 7,200 reserved GPU-seconds and USD 20,
including the launcher's time margin and build allowance. Each call stops its
container. Only `o65-floor` may be pushed or launched by this stream.

## Results

Measurement is in progress. No floor or speedup is claimed yet.
