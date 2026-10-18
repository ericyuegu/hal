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

Use one g4-standard-48: one 96 GiB RTX PRO 6000, 48 vCPUs and 180 GiB RAM.
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

No launch has been performed for this preparation. Before allocation, publish
the final HAL commit, renew local gcloud authentication, and grant the VM
service account Secret Accessor for the five default secret ids and Artifact
Registry Reader for the pinned image. The pinned environment image is an
existing HAL image; boot replaces its baked source with the checked HAL SHA
and runs `uv sync --locked`. Its entrypoint is overridden with Bash.

The input archive is currently local and **absent from R2**. The orchestrator
must publish it before launch. From this prepared worktree, this explicit
command verifies it before upload (it is not run as part of local preparation):

```sh
uv run python - <<'PY'
from pathlib import Path
from hal import r2
from scripts.o65_delivery import read_manifest, verify_blob
manifest = read_manifest(Path("experiments/o65/pilot-inputs.json"))
archive = Path("runs/o65-prep-final-payloads/inputs.tar")
verify_blob(archive, manifest.inputs)
r2.client().upload_file(str(archive), r2.bucket(), manifest.inputs.key)
PY
```

Run the local render, then the single launch command from the final clean SHA:

```sh
bash scripts/launch_o65.sh --dry-run
bash scripts/launch_o65.sh
```

The instance is `hal-o65-proxy-<first eight HAL SHA digits>`, in project
`centering-star-502613-k3`, zone `us-central1-b`. It is one on-demand
g4-standard-48 with one GPU and a 500 GB Hyperdisk Balanced boot disk.
The GCE max-run-duration is 24 hours. A VM-local systemd timer also forces
shutdown after 24 hours from VM boot. At 23h45m from boot, a second timer
kills training and starts cleanup, reserving 15 minutes for an upload bounded
to 12 minutes plus a 30-second kill allowance.

On normal exit or failure, the VM uploads a hash-indexed result tar and then
its manifest to `o65/pilot-results/<instance>/`. It uploads logs, configuration,
evaluations and checkpoints, excluding delivery inputs and scratch files.
If bootstrap cannot pull the image, credentials fail, R2 is unavailable or
upload exceeds its bound, remote publication can fail. The failure is logged;
the hard cap still stops billing for compute and the boot disk is retained
for recovery. No software can guarantee upload during a network outage.

Monitor and fetch results (set NAME to the reported instance name):

```sh
gcloud compute instances get-serial-port-output "$NAME" --project centering-star-502613-k3 --zone us-central1-b
gcloud compute ssh "$NAME" --project centering-star-502613-k3 --zone us-central1-b --command 'sudo tail -n 100 /var/log/hal-training.log'
uv run python scripts/o65_results.py fetch --prefix "o65/pilot-results/$NAME" --root "runs/$NAME-results"
```

The training log identifies the W&B run; its id is
`o65-debug-<HAL SHA prefix>-<output-path hash>`. Open the `hal-o65` project in
the configured W&B account. The fetch command verifies both the tar and each
extracted file. If upload failed, start the stopped VM only for a separately
bounded recovery; keep the disk until evidence is recovered.

Delete the VM and its boot disk after results are verified:

```sh
gcloud compute instances delete "$NAME" --project centering-star-502613-k3 --zone us-central1-b --delete-disks=boot --quiet
```

Outward actions of a real launch: check R2 input availability, read GitHub
publication refs, gcloud account and Secret Manager metadata locally; create exactly one VM and its boot disk
with an external IP and the selected service account; read five secret values
on the VM; configure Docker authentication, pull the pinned Artifact Registry
image, and install Docker/toolchain packages if missing; clone HAL and the MSL
fork; resolve the locked Python environment from package/Git sources; download
the R2 input archive, checkpoint and Melee fixture plus the public Dolphin
fixture; send training and evaluation metrics to W&B; upload results to R2;
stop the VM. It does not create evaluation VMs, publish images, push Git,
change IAM or delete the retained disk. Teardown is the separate command above.

Budget assumption: roughly three times the local 512-match complete-path rate
gives about 5,500 match-frames/s, or 48 seconds per 1,024-match PPO update.
Allow 12–24 VM-hours including warmup, process starts, qualification, evaluation,
resume checks and uploads. This is an estimate; eager evaluation can dominate.
At the listed US-central on-demand rate of $4.49993/hour, compute costs about
$54–108, with a 24-hour compute ceiling near $108. Add disk, IP, network and
storage charges; disk charges continue after shutdown until deletion.
The rate was checked on 2026-10-18 in
[Google's accelerator price table](https://cloud.google.com/products/compute/pricing/accelerator-optimized?hl=en).
The VM qualification replaces the throughput assumption before any extension.
Long matches can still exhaust the cap, especially in held-out evaluation;
that produces incomplete evidence, not a successful learning decision.
