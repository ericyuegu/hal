# O60/O61 validation-boundary resume

The first proxy attempts used Git `3e118310cd22745efe387536d2f95942a1eb88bb`.
Both completed update 4,096 and uploaded exact-resume checkpoints. They then
failed in offline validation because its first eager FlexAttention graph ran
inside the training loop's `fail_on_recompile` compiler stance.

The source transition permits the validation-only graphs to compile with the
default stance. The strict stance remains active for every training update.
Model structure, loss, optimizer, schedule, data selection, batch geometry,
random streams, and checkpoint state are unchanged. The transition also exposes
the existing checked resume-lineage contract through the O60 and O61 CLIs.

Verification:

- Focused O60/O61 and two-process Gloo tests: 17 passed.
- Nested compiler-stance check: a new static shape compiled under the local
  default stance while the surrounding stance remained `fail_on_recompile`.
- Full repository validation is recorded in the commit that contains this file.

Each relaunch supplies a source-transition record containing its own checkpoint
SHA-256, this file's SHA-256, the old Git SHA, and the new Git SHA.

## G4 checkpoint control plane

The first validation-fixed relaunches used Git
`2ebabff9cdef9c60952ede0f195037cbe5b586bd`. Both reproduced finite training
through update 6,144, then hung before writing that boundary checkpoint. Both
GPUs were idle, each rank's main thread consumed one CPU core mostly in system
time, and the update-4,096 checkpoint hashes were unchanged. This rules out host
memory and checkpoint serialization and isolates the NCCL object collective.

Tensor collectives remain on NCCL. Python checkpoint and control objects now use
a separate Gloo process group. Automatic GCE gameplay evaluation runs in an
isolated subprocess on rank zero's GPU when
`HAL_LOCAL_CLOSED_LOOP_EVAL=1`; it evaluates the uploaded immutable milestone
and waits for completion before training continues. The existing Modal broker
path is unchanged.

The proxy runs now restart from update zero on one GPU each. Their global and
local batch is 512, with two 256-sample microbatches, 131,072 replay slots, and
24 loader workers. This keeps the optimizer batch, aggregate replay capacity,
and 232-update minimum replay gap unchanged. Checkpoint format version 2 rejects
the earlier two-rank checkpoints because their world size and partition identity
cannot satisfy exact resume.
