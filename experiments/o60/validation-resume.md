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
