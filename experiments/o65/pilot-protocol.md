# RTX PRO 6000 proxy pilot

We test learning from the 17M O63 bootstrap30 checkpoint on a Modal RTX PRO
6000. [pilot.py](pilot.py) runs training and final evaluation;
[scripts/launch_o65_modal.py](../../scripts/launch_o65_modal.py) constructs the
job. Use [plan.md](plan.md) for the objective and frame conventions.

## Treatment and control

We run 2,000 updates, including critic warmup, on four groups of 1,024 matches.
Each group uses 16 shards with 64 lanes each. Each shard fixes 54 lanes to
self-play, four to pi_ref and two to each of three snapshot slots. Thus each
group has 160 anchor lanes (15.625%). Anchor ports are random per episode.
Only student players train. Opponent roles stay fixed through resets.

| Setting | Value |
| --- | --- |
| Actor | Joint PPO plus exact conditional KL reward and direct KL gradient |
| Actor optimizer | Fresh Muon state, LR 1e-5, zero weight decay |
| Actor steps | 512 windows, 16 decisions/window, one epoch, 32-step LR warmup |
| Clipping | PPO epsilon 0.2, gradient norm cap 1 |
| Critic | Separate optimizer, LR 3e-4, four epochs |
| Discounts | gamma 0.99855/frame, GAE lambda 0.99/frame |
| Rollout | 256 frames per active match per update |
| Seed and cadence | Seed 65, (D=8,R=4) and (D=10,R=2) |

An update contributes about 3,776 student windows and eight actor steps.
Padding in the final step contributes to neither decision selection,
advantage normalization nor gradients. We reject unsupervised executed offsets.

We independently sample each player's u from Beta(5,1.3), clipped to
[0.1,0.999]. The reference uses the same u on each scored student path.
We draw identities conditional on character using checkpoint pretraining counts,
tempering 1.0, floor 1, deployment rate 0 and 10% masking. Master receives its
ordinary count weight. The proxy uses 112,188 ranked training rows. We count
all 332 unordered character pairs with at least one training replay, weight by
sqrt(count), then choose orientation uniformly. We start both players at four
stocks and zero damage on a uniformly drawn legal stage. Stadium stays frozen.

For control, we use the frozen O63 policy with the same training conditions.
Final evaluation fixes u=0.95, masked identity and lambda=1 on both sides.
The larger actor step, broader episode sampling and sampled u are scientific
treatments. Cleanup preserves these settings, reward, precision and masks.

## Inputs

The committed files record the fixed inputs:

- [pilot-inputs.json](pilot-inputs.json): delivery v2, checkpoint, MSL source,
  game data and feature statistics.
- [pilot-sampler.json](pilot-sampler.json) and
  [pilot-sampler-source.json](pilot-sampler-source.json): training counts and
  sampling weights.
- [pilot-evaluation.json](pilot-evaluation.json): final grid and seeds.

We use MSL fork commit `2dbca5e3214d88859ac80333cc19caa328d4be4a` and the
O63 checkpoint with SHA-256
`65e7e43e5cbb18a365a005922ec1cfd026ad807c045eae13b331f487b4ec82e6`.
The delivery checks downloaded files against the manifest before training.
It includes extracted game data and statistics, rather than the replay corpus.

To regenerate counts from checkpoint pretraining sources, supply a manifest
per source under `<source-name>/manifest.jsonl`:

```sh
uv run python -m scripts.build_o65_sampler \
  --checkpoint runs/proxy/final.pt \
  --manifests-root runs/bigstep/training-sources
```

The builder reads cfg.source_names and the professional identity sidecar,
records training-row counts and input hashes, then writes the sampler and
final grid. Update sampler hashes in the delivery manifest after regeneration.

## Training and final evaluation

We verify delivery, build MSL and the HAL adapter, then record the source,
environment and build hashes. Simulator parity tests remain required.

Before head starts, we compile and capture all fixed tick graphs for the
student, pi_ref and three permanent snapshot objects, on every group and
buffer. Both ports stay in each policy's batch, masked when it does not own
them. Snapshot slots start from pi_ref. At each promotion we truncate all live
episodes in the oldest slot's lanes, then copy student weights into that
object. Boundary values bootstrap the student; no truncation counts as a win.
The lanes reset before their next step. We need no retired slots.

We also capture the fixed actor, reference-chunk and value graphs at startup.
Each update logs post-startup compiler starts and capture attempts to W&B and
stdout; a new capture fails with its key. The local check is three PPO updates
at 16 matches with snapshot_interval=1, including resets. The relaunch must
measure the complete pilot's graph memory; the local check cannot qualify a
96 GiB fit.

Before warmup, each match draws an independent integer offset uniformly from
0 through 1,200 frames. Both players use pi_ref. Initial starts wait until
frame 1,280 minus the offset; odd starts take one neutral frame before joining
the two-frame loop. The pool then advances to frame 1,280. Games that end reset
on normal block boundaries. The head start trains neither actor nor critic.
Warmup retains observations and committed actions. We log duration, match FPS
and episode-age quantiles to W&B, and write every offset and resulting age to
head-start.json. Partial episodes do not enter full-episode critic diagnostics.
The server target is under two minutes at about 45k match-frames/s; local tests
do not qualify PRO 6000 timing.

Critic warmup lasts exactly 32 updates, independent of explained variance and
group clocks. At the end of update 32, we calibrate lambda_max to three times the
largest per-cadence advantage spread within the conditioner bounds. We retain
lambda_min=0.01 and record the calibrated maximum before PPO. Failure to leave
warmup fails this stage. The first PPO update is update 33.

One trainer process runs all updates. It saves complete checkpoints every 32
updates and at update 2,000. A worker publishes immutable local snapshots to
the Modal Volume and R2. The queue holds at most two snapshots, including the
active publication. A worker failure or full queue fails the run. The trainer
drains publication before final evaluation.
Training progress comes from existing anchor matches. After a real preemption,
we resume from the latest published complete checkpoint. The first warmup and PPO
updates record allocated and reserved bytes by phase. We preserve partial
memory records in phase-memory.json after an OOM. The trainer uses
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. A three-update, 16-match
RTX 3060 check passed graph capture and replay with this setting. It does not
qualify the full pilot memory fit. Each 512-window actor step and the
reference pass run as four 128-window chunks. Every captured CUDA graph keeps
a private pool at its peak, and at 512 windows the reference graph alone
exceeded the card. The chunks share the step's advantage statistics and
denominator, so their summed gradient equals the whole step's gradient, and
the optimizer still steps once per 512 windows.

We evaluate the final checkpoint once against pi_ref in MSL. The frozen grid
draws 32 distinct matchups without replacement from training weights using
seed 65095. We cross these draws with six stages and both port orientations,
using (D=8,R=4) for 384 matches. We use game/policy seed 10017. Each game starts
at four stocks and zero damage, then continues to a stockout or the eight-minute
timeout. The 28,924-frame budget includes the countdown and timeout frame.
We exclude timeouts and frame-budget stops from wins.
We consume the held-out set before play and preserve partial results on
failure. We do not retry or extend final evaluation after consumption.

## Progress metrics

We use W&B project hal-o65, with global_step as data and metric step. Disk
metrics retain the same results. Each update records anchor 0 under
`gameplay/pi_ref_*` and each snapshot under `gameplay/anchor<ID>_*`:

| Metric | Calculation |
| --- | --- |
| net_stocks_per_min | Student stocks taken minus lost, divided by stepped frames/3600 |
| net_damage_per_min | Positive damage dealt minus taken over the same exposure |
| count and win_rate | Completed stockouts in this update; omit the rate at count zero |
| count_32_updates and win_rate_32_updates | Pooled wins/counts across the last 32 updates; omit the rate at count zero |

We include unfinished matches and countdown frames in exposure, exclude reset
waits and truncation wins, preserve each episode's anchor id through reset
processing. Training format v9 retains rolling counts and records each slot's weight
update and the fresh-pool update origin. Per-slot metrics record the weight
update used for the rollout. A promotion clears that slot's rolling counts. It also preserves partial
episodes from head starts and accounts for the initial 1,280-frame pool advance.

We also record pre-clip gradient norms, cap fractions, conditional KL, entropy,
reward components, episode-age/u coverage, parameter movement, action counts,
phase timing and memory. The gradient-noise estimate uses actual trained
counts and pre-clip gradients, while weights change between samples.
Completed episodes contribute cumulative KL cost and initial-value Monte
Carlo error. Truncations contribute neither Monte Carlo comparison nor wins.

Final evaluation records match length, blast-zone death-animation onsets and
stocks lost at zero damage since respawn. The last measure is a self-destruct
proxy because native observations contain no KO-credit field.

## Launch and storage

Inspect the job locally without a Modal request:

```sh
uv run python -m scripts.launch_o65_modal --mode pilot \
  --name o65-bigstep-pilot --dry-run --output runs/modal-pilot
```

The fixed-slot relaunch starts a new output directory and W&B run from the
stopped pilot's update-256 learner. It validates commit `3d4ad115`, the original
configuration, environment, artifact hashes, and actor/reference agreement.
It restores model, both optimizers, scheduler, lambda_max, RNGs, counters and
rolling win history. It copies snapshot 1 into permanent slot 1, and imports
disk/W&B metrics through update 256. The new pool uses calibrated lambda_max
and the usual pi_ref head start. Slot 1's partial head-start episodes truncate
before its saved weights take effect. This is a learner migration with a
fresh simulator pool, not exact continuation of the old experiment.

Inspect this launch without a Modal request:

```sh
uv run python -m scripts.launch_o65_modal --mode pilot \
  --name o65-fixed-resume --dry-run --output runs/fixed/modal \
  --resume-learner o65-pilot-3d4ad115/pilot/train/checkpoints/update-00000256.json
```

After publishing the reviewed o65-bigstep HEAD, the operator can launch with:

```sh
uv run python -m scripts.launch_o65_modal --mode pilot \
  --name o65-fixed-resume --timeout-seconds 36000 \
  --resume-learner o65-pilot-3d4ad115/pilot/train/checkpoints/update-00000256.json
```

The Modal Secret hal requires the four AWS/R2 variables and WANDB_API_KEY.
We request an RTX PRO 6000, 17 CPU cores and 64 GiB RAM. We use 16 available
CPU ids for shards. Modal does not expose physical-core or NUMA topology.

We store compilation results under hal-o65-profile:/compile-cache, separated
by source, Torch, Triton, CUDA and GPU identity. Each process captures its own
CUDA graphs. The publication worker copies snapshots into Modal Volume
hal-o65-profile:/<run-name>/pilot, commits them, then copies changed files to
R2 o65/pilot-results/<run-name>/state/. We publish the file hash index last.
Only this worker replaces the training resume tree. Final process cleanup
stores local diagnostics under last-attempt, including unpublished state.

We retain the latest complete resume state and every actor snapshot, move
retired manifests into history, then remove unreferenced training/simulator
files. A corrupt latest checkpoint fails. We publish the held-out consumption
marker before final play so preemption preserves that decision.

Before publishing run state, we record source SHA and input hashes in
pilot-identity.json. We require the same recorded source and inputs on resume,
preserve W&B identity across process restarts, move uncommitted metrics into
separate evidence before trimming the active log. Resume restores simulator
state, queues, episode conditions, metrics, optimizers, scheduler and RNGs.

## Runtime estimate and limit

[performance.md](performance.md) records the 11.4 s/update server measurement
with 32-window actor steps. The async and 512-window projections give 7.74 to
8.60 s/update. Charging all 2,000 updates at PPO cost gives an estimated 4.30
to 4.78 GPU-hours, or $18.69 to $20.75 compute before startup, publication and
final evaluation. We make no allowance for cheaper critic warmup in that sum.

We allow 6 to 8 GPU-hours and $29.07 to $37.76 compute for the full pilot,
including a $3 build reserve, compilation, publication and final play.
The final grid's server runtime is unmeasured. These estimates use the rates
recorded on 2026-10-20: GPU $0.000842/s, CPU $0.0000131/core/s and RAM
$0.00000222/GiB/s. Storage and excess egress are additional charges.

We enforce a 36,000-second timeout, plus a 360-second startup/teardown reserve,
under a separate $50/36,360-second ledger. The resulting maximum compute
allowance is $46.88 including the $3 build reserve. The deadline survives
preemption. Expiry fails completion and preserves published state.

## Success criteria

For systems success, we require calibrated warmup exit, real PPO steps,
W&B anchor metrics, complete checkpoint publication and final MSL evaluation.
Missing required fixtures or timeout expiry fail validation.

For learning success, (D=8,R=4) at lambda=1 must have net_stock_lcb>0 and
a two-sided 95% win-rate interval with lower bound above 0.5. We use paired
seed/start units. Missing bounds or incomplete win pairs leave the result
inconclusive. Training metrics do not select the checkpoint or establish
learning success. MSL-specific gains remain a risk; production and Dolphin
validation require separate runs. We do not extend this run after viewing results.

The v8 learner migration can be removed after the fixed-slot run has a validated
v9 checkpoint. Reading old v8 actors for evaluation remains supported.
