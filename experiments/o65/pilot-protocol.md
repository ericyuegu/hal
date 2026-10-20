# RTX PRO 6000 debug pilot

This protocol tests learning in the 17M O63 bootstrap30 loop on one Modal RTX
PRO 6000. It does not qualify O64 or production delays. No job has been launched
by this stream. The entry point is
[scripts/launch_o65_modal.py](../../scripts/launch_o65_modal.py), mode `pilot`.

## Scientific treatment and control

Treatment: joint PPO with exact conditional KL in the reward and the direct KL
loss, one actor epoch, fresh Muon state, actor LR 1e-5, zero weight decay,
PPO clipping 0.2, gradient norm cap 1, 512 windows per actor step, and
16 decisions per window. The actor LR warms up over 32 optimizer steps.
The critic uses LR 3e-4 and four epochs. Discount is 0.99855 per frame;
GAE lambda is 0.99 per frame. Reward, precision, executed factor masks and
detach points remain the specified treatment.

Run 2,000 total updates, including critic warmup. The pool has four groups of
1,024 matches: 4,096 resident matches, 16 native shards per group, 64 lanes per
shard. Each group reserves 154 matches for anchors, with alternating anchor
ports. Only student players train. An update has about 3,788 student windows
and eight actor steps. Each window enters one step. Padding in the last step
is excluded from decision selection, advantage normalization, loss and gradients.

Episode sampling:

- Count identities by character over the checkpoint's own pretraining sources,
  `cfg.source_names`. The proxy uses the 112,188 training rows in
  ranked-anonymized-1-policy-world-v8. Use tempering 1.0, floor 1, deployment
  rate 0 and 10% identity masking. Master receives ordinary count weight.
  The builder supports professional sources through the identity sidecar.
- Both players independently draw u from Beta(5.0, 1.3), clipped to [0.1, 0.999].
  Student and pi_ref use the same u on each scored conditional action path.
  Anchors use the same draw rule. The residual critic receives both players' u.
- Start at four stocks and 0% for both players. Draw uniformly from the six
  stages in `hal.policy.INCLUDED_STAGES`. Pokemon Stadium is frozen.
- Count every unordered matchup seen at least once in the ranked-anonymized-1
  training rows. Weight each pair by `count**0.5`, then choose orientation
  uniformly. The regenerated table has **332 pairs**, replacing the 259-pair
  natural prior with its ten-replay cutoff. Draw identities conditional on
  character. Validation and test rows do not contribute counts.

Control: the hash-pinned O63 base policy, pi_ref. Training anchor matches use
the same sampling and conditioning as the student. The single final evaluation
uses u=0.95, masked identity, lambda 1 and each admitted cadence on both sides.
The percentile, identity, stage, matchup and actor-batch changes are scientific
treatments. They are recorded separately from the removal of evaluation overhead.

## Fixed inputs and provenance

Use [pilot-inputs.json](pilot-inputs.json), [pilot-sampler.json](pilot-sampler.json),
[count provenance](pilot-sampler-source.json), and
[pilot-evaluation.json](pilot-evaluation.json). Delivery schema is
`hal.o65.delivery.v2`. MSL is the `ericyuegu/melee-sim-light` fork at
`2dbca5e3214d88859ac80333cc19caa328d4be4a`, source archive SHA-256
`3da991920685a9702cee22ae8b78184bb5b356acb0e1c90a9ead85d4298fc4d3`.
The O63 checkpoint SHA-256 is
`65e7e43e5cbb18a365a005922ec1cfd026ad807c045eae13b331f487b4ec82e6`.
The manifest pins the input archive, statistics, game manifest and sampler hashes.
Verify every raw game file. Do not download the training corpus to the job.

Only D=8,R=4 and D=10,R=2 are admitted: executed offsets 9..12 and 11..12.
The loader rejects unsupported offsets. Training seed is 65.

Regenerate the identity counts, matchup weights and final evaluation grid from
source replay manifests (one `<source-name>/manifest.jsonl` per source):

```sh
uv run python -m scripts.build_o65_sampler \
  --checkpoint runs/proxy/final.pt \
  --manifests-root runs/bigstep/training-sources
```

The builder records source, checkpoint and sidecar hashes, training-row counts,
matchup counts and checkpoint training-index hashes. Update the delivery
manifest's sampler hashes after regeneration.

## Warmup, training and final evaluation

1. Verify source and inputs, then build MSL and the HAL adapter. Record source
   SHA, environment and build hashes. Existing simulator parity gates remain
   required; this driver does not run a separate baseline evaluation.
2. Train through 2,000 updates, collecting 256 frames per active match per update.
   Warmup requires at least 172 updates, 11,008 frames in every group, and eight
   fresh-feature explained-variance observations within tolerance 0.01. Only
   critic heads train. Calibrate lambda_max to three times the largest per-cadence
   advantage spread within the conditioner bounds. Keep lambda_min=0.01.
   Record the calibrated maximum before PPO. Failure to leave warmup fails
   this stage.
3. Publish complete checkpoints every 32 updates and at update 2,000. Log learning
   from the anchor matches already in each training rollout. There are no
   training evaluations, separate evaluation jobs, selection grid, forced kill
   at 256 or resume verification at 288. Resume from the latest complete
   checkpoint after a real preemption.
4. Evaluate the final checkpoint once in MSL against pi_ref, at u=0.95. The
   fixed grid draws 32 distinct matchups without replacement from the training
   weights, using Python's seeded sampler with seed 65095. Cross these pairs
   with all six stages, both port orientations and both cadences: **768 matches**.
   Game/policy seed is 10017. Use 28,924 frames per match. Timeouts and frame
   budgets do not count as wins. The grid is frozen before training; it selects
   neither a checkpoint nor lambda. No Dolphin evaluation is in this protocol.
   Preserve partial results and the held-out claim on failure. Do not retry or
   extend held-out play after consuming it.

The first warmup and PPO updates record peak allocated and reserved GPU bytes
for each phase. `phase-memory.json` survives a CUDA OOM. The 512-window step
must fit directly; there is no automatic batch reduction or accumulation.

## Learning monitor and diagnostics

W&B project is `hal-o65`. `global_step` is data and the metric step; metric keys
have one slash. Disk metrics retain the same evidence.

Every update logs each anchor separately, using `gameplay/pi_ref_*` for anchor
0 and `gameplay/anchor<ID>_*` for snapshots:

- `net_stocks_per_min`: student stocks taken minus lost, divided by observed
  stepped match frames / 3600. Include unfinished matches and countdown frames;
  exclude reset waits.
- `net_damage_per_min`: student damage dealt minus taken over the same exposure.
  Positive percent changes measure damage; percent resets do not subtract it.
- `count` and `win_rate`: completed stockout matches in this update. Exclude
  truncations. Log count zero when no match finishes; omit its undefined rate.
- `count_32_updates` and `win_rate_32_updates`: pooled completed-match counts
  and wins over the last 32 updates, including updates with no completions.
  This is a count-weighted rate, not the mean of per-update rates. Omit the rate
  when that window has no completed match against the anchor.

These metrics use the existing rollout records and results. Each student
interval retains its opponent anchor id across resets. Rolling counts are
checkpointed in training format v7. Training anchor results are progress
signals under changing policies and episode conditions; final gameplay evidence
comes from the fixed final MSL grid.

Keep the existing gradient norms, cap-binding fractions, conditional KL, entropy,
task reward components, episode-age/u bins, parameter update/reference distances,
action distributions, gradient-noise estimate, phase timings and memory metrics.
The online gradient-noise estimate uses pre-clip gradients and actual trained
counts; weights move between samples. Completed episodes report cumulative KL
cost and initial-value Monte Carlo error. Truncations do not enter that error.

The final evaluation records match length, blast-zone death-animation onsets,
and the zero-damage self-destruct proxy. That proxy counts stocks lost at 0%
with no damage since respawn; the native observation has no KO-credit field.

## Modal launch and durability

Inspect the complete job without a Modal request:

```sh
uv run python -m scripts.launch_o65_modal --mode pilot \
  --name o65-bigstep-pilot --dry-run --output runs/modal-pilot
```

After publishing the clean `o65-bigstep` HEAD, the launch command is:

```sh
uv run python -m scripts.launch_o65_modal --mode pilot \
  --name o65-bigstep-pilot --timeout-seconds 36000
```

The `hal` Modal Secret needs the four AWS/R2 variables and `WANDB_API_KEY`.
Use one RTX PRO 6000, 17 physical CPU cores and 64 GiB host memory. Sixteen
available CPU IDs drive the shards. No pool sweep runs by default.

Compile artifacts use `hal-o65-profile:/compile-cache`, with strict source,
Torch, Triton, CUDA and GPU identity checks. Compatible artifacts can be reused
at publication boundaries and after preemption. Each process recaptures graphs.

At each publication boundary, copy atomic local training state to Modal Volume
`hal-o65-profile:/<run-name>/pilot`, commit it, then back up changed files to
R2 `o65/pilot-results/<run-name>/state/`, publishing the hash index last.
Keep the latest complete resume state and every actor snapshot. Retired
manifests retain hashes; updates 256 and 288 have no special retention.
Preemption restores the latest complete checkpoint under the original request.
A corrupt latest checkpoint fails instead of falling back. Publish a held-out
consumption marker before final play. That marker and the evaluator's claim
prevent a retry after interruption.

The pilot identity binds source SHA and input hashes before any other state is
published. Preserve W&B identity across publication segments and preemption.
Retain metrics beyond the durable checkpoint separately, then trim the active
log. Exact resume includes simulator state, queues, u, metric accumulators,
optimizers, scheduler and RNGs; the pilot no longer spends runtime verifying it.

## Estimates and hard limit

The previous 1,024-match measurement was 11.4 s/update with 32-window actor
steps. The async projection (rollout 6.704 s to 4.246..4.500 s), plus an unmeasured
0.6..1.2 s reduction in repeated step overhead, gives **7.74..8.60 s/update,
estimated**. Charging all 2,000 updates at PPO cost gives **4.30..4.78 GPU-hours**
and **$18.69..$20.75 compute**, before startup, final evaluation and publication.
This estimate gives no credit for cheaper critic warmup.

Allow **6..8 GPU-hours and $29.07..$37.76 compute for the full pilot, estimated**,
including a $3 build allowance. The additional allowance covers compilation,
32-update publication/process restarts and the one 768-match final MSL grid.
Final-grid runtime on the RTX PRO 6000 is unmeasured. These are projections,
not new throughput measurements. Rates checked 2026-10-20: GPU $0.000842/s,
CPU $0.0000131/core/s and RAM $0.00000222/GiB/s
([Modal pricing](https://modal.com/pricing)). Storage and excess egress are extra.

Hard timeout: **36,000 seconds (10 hours)**, plus 360 seconds reserved for
startup and teardown. The absolute deadline survives preemption. Maximum
reserved compute including the $3 build allowance is **$46.88**, under a
separate **$50 / 36,360-second** pilot ledger. Timeout fails completion and
preserves the last published state.

## Success criteria

Systems success requires verified inputs, calibrated warmup exit, real PPO
steps, W&B learning-monitor evidence, checkpoint publication and the single
final MSL evaluation. Missing required fixtures and hard-cap expiry are failures.

Learning success requires final MSL net_stock_lcb>0 and the two-sided 95%
win-rate interval lower bound above 0.5 in each cadence, at lambda 1. Paired
seed/start units are the samples. Missing bounds, incomplete win pairs and
frame-budget stock leads are inconclusive. Training self-play win rate and
offline loss do not establish improvement. Simulator-specific gains remain a
risk; this pilot makes no Dolphin or production gameplay claim. Do not extend
the run after inspecting results.
