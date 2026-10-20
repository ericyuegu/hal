# RTX PRO 6000 debug pilot

This protocol qualifies the 17M O63 bootstrap30 loop on one Modal RTX PRO 6000.
It does not qualify O64 or production delays. No job has been launched by the
bigstep stream. The maintained entry point is
[scripts/launch_o65_modal.py](../../scripts/launch_o65_modal.py), mode `pilot`.

## Scientific treatment and control

Treatment: joint PPO with exact conditional KL in the reward and the direct KL
loss, one actor epoch, fresh Muon state, actor LR 1e-5, zero weight decay,
PPO clipping 0.2, gradient norm cap 1, **512 windows per actor step**, and
16 decisions per window. The previous pilot used 32 windows per step; that
was a deviation from the plan. The actor LR still warms up over exactly 32
optimizer steps. The critic still uses LR 3e-4 and four epochs. Discount is
0.99855 per frame and GAE lambda is 0.99 per frame. Reward, precision, executed
factor masks and detach points remain the specified treatment.

The user chose **2,000 total updates**, replacing 512. Critic warmup is included
in that total. The pool has four groups of 1,024 matches: 4,096 resident matches,
16 native shards per group, 64 lanes per shard. Each group reserves 154 matches
for anchors, with alternating anchor ports. Only student players train. With
roughly 3,788 student windows, an update takes about eight actor steps. Each
window enters one step. The last step is padded and its padding is excluded
from decision selection, advantage normalization, loss and gradient reductions.

The following are explicit scientific changes from the previous pilot:

- Identities use the checkpoint's own pretraining sources, `cfg.source_names`.
  The proxy uses ranked-anonymized-1-policy-world-v8 alone. Count player
  appearances by character in its 112,188 training rows. Use tempering 1.0,
  data floor 1, deployment rate 0, and mask 10% of players, matching
  `identity_dropout=0.1`. Master remains in ordinary count-weighted draws.
  Professional identities are not sampled for this proxy. The builder also
  supports an O64 checkpoint's professional sources through its identity sidecar.
  No gate rejects an otherwise valid table merely for changing the mix.
- Both players independently draw episode percentile u from Beta(5.0, 1.3),
  clipped to [0.1, 0.999], replacing fixed u=0.95. Student and pi_ref use the
  same u on every scored state and conditional action path. Anchors draw from
  the same distribution. The residual critic receives both players' u values.
  Episode metadata and checkpoints preserve u.
- Every episode starts at 0% for both players. The former pilot sampled
  0, 30, 60 and 90%. Warmup and asynchronous episode ends supply phase diversity.
  Every update records episode-age bins in frames.
- Draw stages uniformly from the six stages in `hal.policy.INCLUDED_STAGES`.
  Pokemon Stadium is frozen at the MSL boundary.
- Draw unordered character pairs from `hal.eval.matchups.MATCHUP_PRIOR`: the
  259 ranked train matchups with at least ten replays. Choose orientation
  uniformly, then draw each identity conditional on its character. This replaces
  the former FD Fox/Falco-only training sampler.

Control: the hash-pinned base O63 policy, with matched episode sampling and
conditioning. All MSL and Dolphin **evaluations keep u=0.95**, masked identity,
student lambda 1 and both cadences, to compare checkpoints directly.

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
The loader rejects unsupported offsets. Training seed is 65. Selection uses
seeds 17 and 29, FD Fox/Falco; held-out uses 10017, 10029, 10041, 10053,
10067, 10079, 10093 and 10111, BF Fox/Falco. Each seed runs both port
orientations and both cadences. The held-out set is used once.

Regenerate the identity table with the checkpoint's source replay manifests
(one `<source-name>/manifest.jsonl` per source):

```sh
uv run python -m scripts.build_o65_sampler \
  --checkpoint runs/proxy/final.pt \
  --manifests-root runs/bigstep/training-sources
```

The builder records manifest, checkpoint and sidecar hashes, training-row
counts and the checkpoint's training-index hashes. It excludes validation and
test rows. The delivery manifest's sampler hashes must then be updated.

## Warmup, training and evaluation

1. Verify source and inputs, build MSL and the HAL adapter, and fetch verified
   Melee and Dolphin fixtures. Record source SHA, environment and build hashes.
2. Run the fixed base-versus-base control in MSL and Dolphin. This is a baseline,
   not proof of simulator correctness for distinct policies. Replay and controller
   parity gates remain required.
3. Train through 2,000 total updates. Every update collects 256 frames per active
   match. Warmup requires at least 172 updates, 11,008 frames in every group,
   and eight fresh-feature explained-variance observations within tolerance 0.01.
   Only critic heads train. Calibrate lambda_max to three times the largest
   per-cadence advantage spread within the declared conditioner bounds. Keep
   lambda_min=0.01. Record the calibrated maximum before PPO starts. Failure
   to leave warmup fails this stage.
4. Publish and evaluate every 32 updates, plus the final update 2,000. Run MSL
   against the base and the three most recent 256-update snapshot anchors.
   Also run Dolphin every 128 updates against those same anchors. Evaluation
   uses 1,924 frames (30 active seconds plus countdown and final observation).
   Budget stops are not wins. Training and evaluation use the GPU serially.
5. At update 256, SIGKILL the trainer process group after atomic checkpoint
   publication while more work remains. Resume the same W&B run to 288.
   At 288, require state-exact restoration and the noise-qualified next update:
   repeat the same next update twice as controls, then once from resume.
   Complete-model and optimizer-family RMS error may reach twice control spread
   plus eight dtype rounding units at the control scale (minimum 1e-3).
   Integer state and scheduler progress remain exact. Batch hashes and maxima
   remain diagnostic. Finite checks are unchanged. Restore the 288 boundary
   after verification and continue. Interrupted verification can rerun from
   the preserved 288 checkpoint without advancing production training.
6. Evaluate only the final checkpoint on the held-out grid, in both simulators,
   with 28,924 frames. Preserve partial results and the held-out claim on failure.
   No held-out retry or extension is allowed under this protocol.

The first warmup update and first PPO update record peak allocated and reserved
GPU bytes for rollout, preparation, reference, critic and actor as applicable.
`phase-memory.json` is written as phases finish, including on CUDA OOM. An OOM
fails with the phase name. The unmeasured 512-window step must fit directly:
there is no accumulation, recomputation or automatic batch-size reduction.

## Diagnostics

W&B project is `hal-o65`. `global_step` is data, defined as the metric step;
keys have one slash. Disk metrics and evaluation reports retain full evidence.
Log pre-clip actor/critic gradient norms and cap-binding fractions, per-factor
reference KL, KL per executed frame by cadence, lambda and u bin, entropy by u
and factor, task reward components, episode age, and per-episode return and win
rate by u. Parameter-group diagnostics include reference distance and update
norm divided by the group's pre-update weight norm. Action distributions on
the same sampled contexts report student/reference button press rates, trigger
means and stick-magnitude bins (edges 0.01, 0.25, 0.5, 0.75, 1.01).

The gradient-noise estimate uses pre-clip gradients across actor steps, with
trained-decision weighting and finite-sample signal/noise correction, then
converts to windows using the observed decisions per window. Its unit is
windows. It reports signal and validity; a nonpositive estimated signal does
not produce an infinite noise scale. Weights move between these samples, so
this is an online diagnostic, not a controlled critical-batch measurement.

Completed episodes report cumulative student KL cost per match and the initial
value against the full discounted return, with and without KL. Costs follow the
plan's live-rollout-tail convention. Truncations do not enter Monte Carlo error
or win-rate metrics. These accumulators are part of exact resume state.

Each evaluation cell identifies its anchor and reports match length, deaths
by blast-zone death-animation onset (bottom 0, left 1, right 2, top 3..10),
and an explicit zero-damage self-destruct proxy. The proxy counts a stock lost
at 0% with no damage since respawn; it is not exact KO attribution. The native
observation has no last-hit or KO-credit field.

## Modal launch and durability

Inspect the complete job without contacting Modal:

```sh
uv run python -m scripts.launch_o65_modal --mode pilot \
  --name o65-bigstep-pilot --dry-run --output runs/modal-pilot
```

After publishing the clean `o65-bigstep` HEAD, the exact launch command is:

```sh
uv run python -m scripts.launch_o65_modal --mode pilot \
  --name o65-bigstep-pilot --timeout-seconds 85500
```

The `hal` Modal Secret must contain the four AWS/R2 variables and
`WANDB_API_KEY`. The job has one RTX PRO 6000, 17 physical CPU cores and
64 GiB host memory. Sixteen distinct available CPU IDs drive the shards;
Modal does not expose physical-core topology. No pool sweep runs by default.

Compile artifacts use the async stream's persisted directory,
`hal-o65-profile:/compile-cache`. Source, Torch, Triton, CUDA and GPU identity
checks remain strict. These scientific/source changes require new compatible
artifacts; old source entries are not loaded unsafely. Later 32-update segments
and preemption resumes reuse the compatible persisted artifacts. CUDA graphs
are recaptured in each process.

The trainer writes atomic checkpoints on local POSIX storage. At each stopped
training/evaluation boundary, the driver copies complete state to Modal Volume
`hal-o65-profile:/<run-name>/pilot`, commits it, and backs up changed files to
R2 `o65/pilot-results/<run-name>/state/`, publishing the hash index last.
The latest resume state, updates 256 and 288, and all actor snapshots are kept.
Retired manifests retain their hashes. Preemption reuses the same request and
restores the latest complete checkpoint from the Volume. A corrupt latest
checkpoint fails instead of falling back. Selection evaluation can resume by
repeating a fixed-seed interrupted cell; its partial evidence is retained.
A held-out consumption marker is committed before starting held-out play.
The evaluator's hidden claim is also preserved. Either marker prevents a retry
after interruption. A failed resume-verification report also blocks continuation.

The driver preserves W&B identity across segments and preemption. Metrics
past the durable checkpoint are retained separately, then trimmed from the
active disk log. The two-control resume verifier includes simulator state,
queues, episode u, metric accumulators, optimizers, scheduler and every RNG.

## Estimates and hard limit

The old 1,024-match measurement was 11.4 s/update with 32-window actor steps.
Using the async projection (rollout 6.704 s to 4.246..4.500 s), and an unmeasured
0.6..1.2 s reduction in repeated step/optimizer overhead, gives **7.74..8.60
s/update, estimated**. Model FLOPs are not divided by 16 when batch size grows.
Charging all 2,000 updates at PPO cost gives **4.30..4.78 GPU-hours** and
**$18.69..$20.75 compute**, before startup, evaluations, publication and process
restarts. Critic warmup should cost less; this arithmetic gives it no credit.

Allow **8..12 GPU-hours and $38..$55 compute** for the full protocol, estimated.
The larger evaluation grid, compilation, new shapes, memory pressure and remote
publication can invalidate that allowance. No new timing study was run locally.
Rates checked 2026-10-20: GPU $0.000842/s, physical CPU $0.0000131/core/s,
RAM $0.00000222/GiB/s ([Modal pricing](https://modal.com/pricing)). These figures
exclude storage and egress beyond included allowances.

Hard timeout: **85,500 seconds (23h45m)**, with 360 seconds reserved for startup
and teardown. The absolute request deadline survives preemption; Modal's
per-attempt timeout does not extend it. Maximum reserved compute, including
$3 build allowance, is **$106.62**, under a separate $110/85,860-second pilot
ledger. Timeout expiry fails completion and preserves the last published state.
See [Modal timeout rules](https://modal.com/docs/guide/timeouts) and
[preemption rules](https://modal.com/docs/guide/preemption).

## Success criteria

Systems success requires verified inputs, all scheduled evaluations, calibrated
warmup exit, real PPO steps, W&B evidence, forced kill, and the state-exact and
noise-qualified resume checks on the actual RTX PRO 6000. Required missing
fixtures and hard-cap expiry are failures.

Learning success requires the final held-out MSL result to have net_stock_lcb>0
and the two-sided 95% win-rate interval lower bound above 0.5, in each cadence,
at fixed lambda 1. Paired seed/start units are the samples. Missing bounds,
incomplete win pairs and frame-budget stock leads are inconclusive. Dolphin
is a separate gameplay check: a clear reversal or new failure pattern blocks
a gameplay claim. Neither self-play win rate nor offline loss establishes
improvement. Do not extend the run after inspecting its results.
