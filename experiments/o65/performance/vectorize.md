# O65 batched compute

## Merge qualification

The merged experiment at `499b05eb` matches fresh captures from the quickfix
experiment at `a5fb7706` exactly at 16 and 64 native matches. Each capture covers
two deterministic updates, with resets and timeouts (`max_frame=0`), seed 65,
the recorded O63 checkpoint, four shards, and both development cadences.
The capture files identify the complete configuration, environment and inputs.
No merge defect was found.

The experiment source copies were checked against their Git objects before the
fixture was replaced. The new fixture is the **quickfix control**, not a capture
from the merged implementation. Every compared field matches: native frame and
controller bytes, decision tensors, behavior/old/reference log-probabilities,
values, advantages, batch hashes and both post-update parameter hashes.

Raw evidence is in `runs/vectorize/{control,merge}-parity-{16,64}/capture.json`.
The maintained resume verifier also passes on the merge at 16 matches:

- Next batch: `a6a2b43b7e1cf0bd3735a20102f8d1cba561af9babb8faf821c9d3cd6d048b23`.
- Optimizers and scheduler: `f869d946b2013b3a6a4bccfdfa0a40c8c70acad17930da96e70501fcd134dc38`.
- Maximum parameter error: zero.

Every GPU command holds `/home/ericgu/data/scratch/o65-tmp/gpu.lock`, uses a
systemd scope with `MemoryMax=10G`, and sets `OMP_NUM_THREADS=2`. Local jobs
use only 16 or 64 matches. Logs and pytest basetemps stay under `runs/vectorize`
and `$TMPDIR`; jobs run serially.
