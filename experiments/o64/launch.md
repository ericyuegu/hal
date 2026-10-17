# O64 production launch

The stable production phase was submitted from clean pushed commit
`7955ec17e945ff0269f715056cdf5fe5c6bf53a7` on branch
`codex/o64-production-20261017`. The code and hardware qualification are in
[validation.md](validation.md) and [performance/report.md](performance/report.md).

- [W&B run](https://forge.coreweave.com/wandb/ericyuegu/hal/runs/z2s5pt0z): `z2s5pt0z`.
- [Modal App](https://modal.com/apps/ap-oZUCPd5fmLIZxrNmE7vq72): `ap-oZUCPd5fmLIZxrNmE7vq72`.
- FunctionCall: `fc-01M55JQ83T7QC37HDBZXDRB5T1`.
- Initial attempt: `ta-01M55JQ8H2FC15TH1177GW72AR`.
- Durable retry state: `hal-modal-state/a45c8bd5526d45b98da6d02f79841336.json`.
- Resolved configuration: [resolved_config.json](resolved_config.json).
- Full launch identity and argv: [production_launch.json](production_launch.json).

The run uses bootstrap30, the full 44-source corpus, seed 0, batch 512, BF16,
Muon, and the 247,944,158-parameter architecture. Offsets and loss weights are
specified in [treatment.md](treatment.md). B200 training uses 32 requested CPU
cores, a 48-core limit, 128/384 GiB memory request/limit and 2,048 GiB SSD. Gameplay
uses L40S, u95, 96 matchups, and timing (4, 2, 2). Checkpoints occur every 2,048 updates;
automatic gameplay evaluations occur every 8,192 updates. The stable phase stops
at 98,304. The separate 32,768-update decay phase awaits its immutable stable
checkpoint and recorded lineage.

## Startup observation

At the recorded observation, the actual validation cache contained 2,048 samples
and took 69.2 seconds to fill. All 24 training workers had started, the model was
constructed, and background materialization was progressing at roughly
730–800 MiB/s. The main thread was actively tracing the backward graph inside
PyTorch AOTAutograd. GPU utilization was zero at that sampled point. No real
optimizer update or durable checkpoint had been observed yet. The 28.3% MFU is
from the matched warmed synthetic benchmark, not this startup interval.

The standalone image lacked py-spy. A temporary pinned py-spy 0.4.2 tool failed
to identify this CPython 3.14.3 interpreter. One stack-only diagnostic was then
requested through Python 3.14's supported
[sys.remote_exec interface](https://docs.python.org/3.14/library/sys.html#sys.remote_exec).
It wrote a faulthandler stack dump without changing model, optimizer, data or
RNG state. The retained bounded tail is
[production_startup_stack_tail.txt](production_startup_stack_tail.txt).
Per-thread CPU counters are in
[production_startup_thread_cpu.json](production_startup_thread_cpu.json).
The 64 raw-shard materialization threads account for much of the process CPU;
this is startup work, separate from steady GPU throughput.

The W&B configuration matches the recorded arm, architecture, batch, seed,
stable boundary and 44 sources. Its environment snapshot records CPython 3.14.3,
one NVIDIA B200, and the training argv. See
[production_wandb_snapshot.json](production_wandb_snapshot.json).

The first local submission stopped before creating an App because the clean
checkout's virtualenv lacked loguru. Reusing the validated workspace environment
through `UV_PROJECT_ENVIRONMENT` resolved it. The successful launch identity
above is the only production training launch.
