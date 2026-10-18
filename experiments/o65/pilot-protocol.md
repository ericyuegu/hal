# RTX PRO 6000 debug pilot

This protocol qualifies the 17M O63 bootstrap30 loop. It does not qualify O64
or production delays. Run [the driver](pilot.py) through
[the launch script](../../scripts/launch_o65.sh). Do not extend a failed run
or change its settings without recording a new treatment.

## Treatment, control and fixed inputs

Treatment: joint PPO with the exact conditional KL reward and direct KL loss,
one actor epoch, fresh Muon state, LR 1e-5, zero weight decay, clipping 0.2,
importance cap 2, gradient norm cap 1, 32 windows per step, and 16 decisions
per window. The critic uses its own optimizer, LR 3e-4 and four epochs.
The actor LR warms up over 32 optimizer steps. Discount is 0.99855 per frame;
GAE lambda is 0.99 per frame. Keep the current reward, masks and detach points.
No reward, precision or optimizer ablation is part of this launch.

Control: the frozen, hash-pinned base O63 model, pi_ref, under the same
u*=0.95, identities, stage, character pair, cadence, port orientations and
evaluation schedule. Training still uses its configured frozen reference and
snapshot anchors. Evaluation always uses the base O63 anchor.

Fixed inputs are [pilot-inputs.json](pilot-inputs.json),
[pilot-sampler.json](pilot-sampler.json), its
[count provenance](pilot-sampler-source.json), [pilot-evaluation.json](pilot-evaluation.json),
the committed HAL source, uv.lock, and the pinned container digest in the launch
script. The delivery schema is `hal.o65.delivery.v2`:

- MSL fork: `https://github.com/ericyuegu/melee-sim-light.git`,
  commit `2dbca5e3214d88859ac80333cc19caa328d4be4a` on `o65-hal-parity`.
  Verify the detached HEAD, a clean tracked tree and SHA-256 of `git archive`:
  `3da991920685a9702cee22ae8b78184bb5b356acb0e1c90a9ead85d4298fc4d3`.
- Base O63 checkpoint: the R2 key and hash in the plan and delivery manifest;
  SHA-256 `65e7e43e5cbb18a365a005922ec1cfd026ad807c045eae13b331f487b4ec82e6`.
- Verify the input archive's size and hash before extraction. Verify statistics,
  sampler, count provenance and the game manifest, then every raw game file.
  The strict checkpoint loader checks feature statistics, recorded training
  data identity and supervised offsets. No training corpus is downloaded.
- Only D=8,R=4 and D=10,R=2 are admitted. Executed offsets are 9..12 and
  11..12. Reject every unsupported offset; do not substitute a cadence.
- Training seed: 65. Selection seeds: 17 and 29 on FD Fox/Falco. Held-out
  seeds: 10017, 10029, 10041, 10053, 10067, 10079, 10093, 10111 on BF
  Fox/Falco. Both port orientations run for each seed and cadence. Identity
  0 is fixed for both evaluation players. The evaluated student lambda is 1;
  this debug run does not select lambda or a checkpoint from a sweep.

The launcher's clean HEAD is the exact HAL SHA sent to the VM. The orchestrator
must publish that SHA on `ericyuegu/hal` before launch. The VM checks the SHA
after checkout. The handoff report records the final expected SHA. The launcher
does not push. It never uses the upstream MSL remote.

## Pool and throughput qualification

Use one g4-standard-48: one 96 GB RTX PRO 6000, 48 vCPUs and 180 GB RAM,
as listed in [Google's GPU table](https://docs.cloud.google.com/compute/docs/gpus).
Use 16 native shard threads, two Torch CPU threads, and four groups. Begin
with three separate, three-update qualifications at 256, 512 and 1,024 active
matches (16, 32 and 64 lanes per shard), one group each. Exclude the first
update from steady throughput. These are disposable timing runs with no
warmup; they do not select optimizer settings or learning checkpoints.

The planned training pool has 1,024 active and 4,096 resident matches.
The [local report](performance/report.md) measured 1,592 match-frames/s at
256 and 1,839 at 512, with 8.04 and 11.12 GiB peak reserved, on a 12 GB RTX
3060 with 12 shards. Native-only scaling measured 27,296, 131,713 and 166,045
match-frames/s with 1, 6 and 12 shard threads. This suggests GPU work limits
the complete path before simulator stepping does. It does not measure the VM.

Assumption: the larger GPU can use 1,024 active matches efficiently, and 16
server-vCPU shard threads can feed it. CPU models, SMT placement, graph sizes,
deterministic kernels and resets can invalidate that assumption. The driver
records its CPU list; check `lscpu -e` and `nvidia-smi topo -m` on the VM to
interpret placement. Available CPU ids are not proof of physical core layout.

Stop if the 1,024-match rate is over 10% below the best candidate, any measured
rate is invalid, or any candidate reserves 80 GiB or more. Inspect
throughput.json; do not silently choose another pool. Time checkpoints,
evaluation and startup separately because update throughput excludes them.

## Warmup, training and evaluation

1. Build MSL and the HAL adapter from verified source. Record compiler and
   library hashes. Fetch the hash-checked Melee and Dolphin fixtures.
2. Measure the base model against itself in both simulators with the selection
   protocol. This establishes each simulator's baseline; it cannot establish
   simulator correctness for distinct policies. The native replay and
   controller parity gates remain required evidence.
3. Train through 512 updates with deterministic algorithms. Each update collects
   256 frames per active match. Critic warmup requires at least 172 updates,
   at least 11,008 frames in every group, and a fresh-feature explained-variance
   plateau (eight observations, tolerance 0.01). Only critic heads train then.
   Calibrate lambda_max to three times the largest measured per-cadence
   advantage spread, subject to the implementation's recorded bounds.
   Lambda_min stays 0.01. Store the resolved maximum before PPO begins.
   A failure to leave warmup is a failed debug stage, not permission to bypass it.
4. Publish a checkpoint every 32 updates. At every boundary, stop the trainer
   and run MSL selection evaluation against pi_ref. At 128, 256, 384 and 512,
   also run Dolphin against that same reference. Evaluation uses the fixed
   lambda=1 and both cadences. Selection matches have a 1,924-frame budget
   (30 seconds of active play plus the countdown and final observation);
   they diagnose progress and may not produce scored wins. The earlier prep
   used 7,324 frames. Local eager evaluation measures about 17 total simulator
   frames per wall second on 132-frame matches, so that budget would consume
   most of a VM day. This shorter diagnostic budget is a provisional cost
   choice. The full-match held-out learning criterion remains unchanged.
   These jobs use the same GPU serially, so evaluation cannot exhaust training
   memory or lag behind an unbounded child queue. Reopen the same checkpoint
   and W&B run before the next training segment.
5. At update 256, SIGKILL the trainer process group after atomic checkpoint
   publication while it still has work left. Evaluate that durable snapshot,
   then resume it to 288. At 288, run the existing exact-resume verifier on the
   full pool: the next batch and optimizer update run uninterrupted and from
   restore. Require identical batch hashes, optimizer/scheduler hashes and
   zero model parameter error. Restore the 288 boundary after the check.
   Continue through 512 with the same configuration and run id.
6. Evaluate only the final checkpoint once on the held-out grid in both
   simulators, with a 28,924-frame budget. Keep the held-out claim and every
   partial result, including on failure. No held-out retry or selection is
   allowed under the same protocol.

The driver writes resolved configuration, source/input hashes, per-update
metrics, compiler/build logs, throughput.json, kill.json, resume-check.json,
checkpoint manifests and paired match reports. Trainer and driver evaluation
metrics resume the same online W&B run in project `hal-o65`, with global_step
as data and one slash per metric key. Missing credentials or evaluation
failures fail the job. Disk results remain the source of complete intervals
and match outcomes.

## Success criteria

The systems stage passes only if builds and input checks pass, all scheduled
matches publish, warmup calibrates and exits, the actor takes PPO steps,
W&B records both training and evaluations, and the forced kill plus exact
resume passes on the actual RTX PRO 6000. The hard cap expiring is a failed
completion. Preserve its partial evidence.

The learning stage passes only if the final held-out MSL comparison has
net_stock_lcb > 0 **and** the two-sided 95% win-rate interval's lower bound is
above 0.5, in each cadence at the fixed lambda=1. Report mean net stocks/min
with its 95% interval, raw match net stocks, terminal win rate with its interval,
completed pairs, budget stops and timeouts. Treat paired seed/start units as
samples. Missing bounds or incomplete win pairs mean inconclusive evidence;
do not score a frame-budget stock lead as a win. The small held-out set can
fail to resolve a real effect. Do not extend it after seeing the result.

Dolphin is a periodic sanity check and a separate gameplay/latency benchmark.
Report the same paired intervals and event statistics there. A clear reversal
or a new failure pattern blocks a gameplay claim, even if MSL passes. A null
Dolphin interval does not establish agreement. Diagnose distinct-policy MSL
versus Dolphin event statistics before any production stage. Self-play win
rate and offline loss cannot establish improvement over pi_ref.

## Launch, monitoring and teardown

The pilot uses the existing `hal-netplay-g4-1015-us-west4-a` instance in
`us-west4-a`, project `centering-star-502613-k3`. The orchestrator starts it
only after the user confirms netplay cleanup. The launcher rejects active
netplay services or containers. It leaves their files in place.

Before starting, publish the final HAL commit, check local gcloud
authentication, and verify input availability in R2. The existing VM service
account has Secret Accessor on the five credential secrets and Artifact
Registry Reader on `hal-netplay`. The launcher checks secret metadata
locally. The job reads secret values through the VM service account.
The job uses the image digest in `scripts/launch_o65_existing.py`, replaces
the image's HAL source with the requested commit, verifies HEAD, then runs
`uv sync --locked`. Delivery still checks the MSL fork commit and source
archive, checkpoint, input archive, statistics and raw game hashes.

The read-only R2 size check passed for the input archive and checkpoint on
2026-10-18. If the archive needs publication again, the orchestrator can
verify and upload it from this worktree with:

```sh
uv run python - <<'PYTHON'
from pathlib import Path
from hal import r2
from scripts.o65_delivery import read_manifest, verify_blob
manifest = read_manifest(Path("experiments/o65/pilot-inputs.json"))
archive = Path("runs/o65-prep-final-payloads/inputs.tar")
verify_blob(archive, manifest.inputs)
r2.client().upload_file(str(archive), r2.bucket(), manifest.inputs.key)
PYTHON
```

After committing, render locally for inspection:

```sh
bash scripts/launch_o65.sh --dry-run --render runs/o65-existing-job.sh
```

The dry run writes the job script and prints the SCP and SSH commands.
It makes no cloud requests. For the approved start, set a unique run name
and use the same name for monitoring, stopping and result retrieval:

```sh
NAME="hal-o65-proxy-$(git rev-parse --short=8 HEAD)"
bash scripts/launch_o65.sh --run-name "$NAME"
```

The launcher checks R2 input sizes and HAL publication before copying the
rendered script with `gcloud compute scp`. Over SSH, it installs the script
under `/var/lib/hal-o65/$NAME/`, starts `$NAME.service` with `systemd-run`,
then returns. Systemd supervises the detached job after SSH disconnects.
The transient unit has a 85,500-second runtime limit and a 900-second stop
limit, so training ends at 23h45m and cleanup finishes within the remaining
15 minutes. The limit starts with this job, irrespective of VM uptime.
At the limit, systemd kills the job's host processes, then invokes
`ExecStopPost` to remove its training container and upload results.
Systemd also invokes this hook after normal exit, failed startup and a
manual stop. See [the systemd service reference](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html#ExecStopPost=).

The post-stop script allows 810 seconds for container removal and upload,
then 60 seconds for cleanup, with a five-second kill allowance on each.
The upload container has its own 720-second limit. It writes a hash-indexed
result tar, then publishes its manifest under
`o65/pilot-results/$NAME/`. The tar contains logs, resolved settings,
evaluation reports and retained checkpoints. Input delivery and scratch
files stay out of the upload. After the attempt, the job removes its
containers, secret files, delivery directory and scratch. It preserves
local result files, exit status and the job script for inspection.
An upload failure remains visible in `output/upload-exit-code` and the
journal. A network or storage failure can prevent remote publication;
local results remain available for recovery.

The manual stop follows the same upload and cleanup path:

```sh
bash scripts/launch_o65.sh --action stop --run-name "$NAME"
```

This command waits for systemd's stop operation, including result upload.
Stopping uses the original run name even if the local checkout has changed.
The VM stays allocated after all exit paths. The launcher does not change
instance metadata or scheduling. It does not install host packages or
change Docker configuration.

Monitor the job and retrieve results:

```sh
gcloud compute ssh hal-netplay-g4-1015-us-west4-a --project centering-star-502613-k3 --zone us-west4-a --command "sudo systemctl status $NAME.service --no-pager"
gcloud compute ssh hal-netplay-g4-1015-us-west4-a --project centering-star-502613-k3 --zone us-west4-a --command "sudo journalctl -u $NAME.service -n 100 --no-pager"
gcloud compute ssh hal-netplay-g4-1015-us-west4-a --project centering-star-502613-k3 --zone us-west4-a --command "sudo tail -n 5 /var/lib/hal-o65/$NAME/output/pilot/train/metrics.jsonl"
gcloud compute ssh hal-netplay-g4-1015-us-west4-a --project centering-star-502613-k3 --zone us-west4-a --command "sudo grep -m 1 https://wandb.ai/ /var/lib/hal-o65/$NAME/output/pilot/train-00000032.log"
gcloud compute ssh hal-netplay-g4-1015-us-west4-a --project centering-star-502613-k3 --zone us-west4-a --command 'df -h /; nvidia-smi'
uv run python scripts/o65_results.py fetch --prefix "o65/pilot-results/$NAME" --root "runs/$NAME-results"
```

The W&B run id is `o65-debug-<HAL SHA prefix>-<output-path hash>`, in project
`hal-o65`. Inside the training container, the output path includes the run
name so separate jobs on the same HAL commit get separate W&B runs.
The fetch command verifies the tar and every extracted file.
The result directory remains on the VM after completion. Keep it until the
retrieved evidence has been checked. Teardown consists of stopping the job
if needed and allowing its cleanup to finish. The VM remains running.

### Disk check

Read-only inspection on 2026-10-18 found 48,176,799,744 bytes free
(44.87 GiB) on the 100 GB boot disk. Docker already has the requested image
digest, 9,056,653,301 bytes (8.43 GiB), and lists the NVIDIA container
runtime. The driver is 580.178.04 with CUDA 13.0. The netplay runner,
health service and overlay container were still active during inspection.
The launcher requires a cached image and at least 40 GiB free before
training. Docker runs use `--pull=never`; a missing image fails the job.

For the 4,096-match pool, allow about 2 GiB per complete resume checkpoint,
including frame history and compressed simulator state. This estimate
extrapolates the local four-match checkpoint: 207.54 MiB of training state,
66.28 MiB for its actor and 0.22 MiB of compressed simulator shards.
It is a planning allowance, not a full-pool measurement. The compact
retention mode keeps the latest resume checkpoint, update 256 and update
288, plus all actor snapshots. After each completed selection evaluation,
the driver moves older manifests to `retired-checkpoints/` and removes
unreferenced training and simulator files. Historical manifests record
hashes; they cannot resume the retired state. Training settings, evaluation
boundaries and resume checks stay the same. Other callers retain all
checkpoints by default.

Budget 8 GiB for retained results and another 8 GiB for their upload tar.
Allow 4 GiB for delivery and emulator fixtures, including both ISO copies
made by the fixture fetcher, and 12 GiB for the checkout, environment
updates, compiler and caches. This totals about 32 GiB beyond the cached
image, leaving about 13 GiB at the inspected free-space level. Monitor disk
use during qualification because full-pool serialization and compiler
caches remain estimates. Retaining all 16 complete checkpoints would put
the upload copy at risk of filling this disk.

The start reads GitHub publication refs and local secret metadata, copies
the script, creates a transient unit, reads five secret values on the VM,
clones HAL and the MSL fork, resolves the locked environment, builds MSL
and the adapter, fetches verified inputs and emulator fixtures, runs the
fixed pilot, sends W&B metrics, uploads R2 results, then removes job scratch.
The job installs its compiler inside its disposable container.

Budget assumption: about 5,500 match-frames/s gives 48 seconds per
1,024-match PPO update. Allow 12 to 24 hours for preparation, qualification,
warmup, training, evaluations, resume checks and upload. This estimate
has not been measured on this VM. The existing instance continues to incur
charges after the job ends. A cap expiry preserves partial evidence and
fails completion. Long held-out matches can still exhaust the cap.
