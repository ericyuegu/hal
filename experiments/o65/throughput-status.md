# O65 throughput status and decisions

This file records the state of the O65 247M self-play PPO trainer: what we measured, what the user
approved, what we tried and dropped, how the production run is configured, and what to watch while
it runs. Numbers come from one B200 devbox with 1024 active matches, 512-frame rollouts and the O64
checkpoint `step-0098304.pt`, unless a line says otherwise.

## Production run

Launched on 2026-10-24 from `o65-rollout-speed` at `ca58a905` as Modal run `o65-o64-outcome-256-r2`:

    uv run python -m scripts.launch_o65_modal --mode ddp --name o65-o64-outcome-256-r2 --gpus 8 \
      --matches 1024 --windows-per-step 512 --actor-microbatches 8 --ddp-updates 256 \
      --damage-weight 0.1 --game-win-weight 120 --timeout-seconds 12600 --b200-max-usd 455

The production identity records the Git SHA, so a resume has to launch from `ca58a905`; later
commits, including documentation, change the SHA and the resume check refuses them.
The local launcher process blocks on the Modal call; its log is
`~/data/scratch/o65-launch-outcome-256-r2.log`; the trainer writes its own log to `/opt/o65-work/o65-o64-outcome-256-r2/run.log` inside the container, and only `modal container exec` can read it. W&B project `hal-o65`, run id equal to the run name.

Settings, checked against the qualifying bench's `resolved.json` by an Opus review:

- 8 B200 ranks under torchrun, 1024 active matches per rank, four rotating match groups per rank
  (4096 matches held per rank, 32,768 in total), 512-frame rollouts, 256 updates.
- Actor: Muon, lr 1e-5 with 32 lr-warmup steps, momentum 0.95, clip 0.2, every valid decision per
  window, 512 windows per optimizer step per rank, eight microbatches of 64 windows.
- Critic: AdamW lr 3e-4, four passes, value scale 150, 32 critic-only warmup updates.
- Rewards 0.1 damage, 120 stock, 120 game win. Gamma 0.99855 per frame, GAE lambda 0.99,
  max frame 28,800, production cadence grid, BF16 decoder, seed 65.
- Head start 1200 frames, rounded to 1536 so every group clock sits on a 128-frame boundary.
- Checkpoint every 32 updates through a background writer, snapshot every 64, three snapshot slots.

Launch-time changes made after the bench:

- `scripts/o65_production.py` hardcoded `--groups 1`; it now passes 4.
- Modal allows at most 64 CPUs per function, and the launcher had asked for 136 (17 per rank). Each
  rank now runs eight simulator shards of 128 matches, and the rank controllers share those CPUs.
  The bench used sixteen shards of 64 matches, so rollout tick time on Modal is unmeasured.
- The DDP job mounts the devbox compile cache (`hal-o65-devbox-v2` at `/cache`) instead of a fresh
  per-run archive. Triton kernels should hit. Inductor graph keys include `cuda:N`, so ranks 1 to 7
  may recompile graphs; expect startup between the 16-minute warm and 35-minute cold figures.
- The resume identity ignores virtual CPU IDs, so a replacement container can resume.
- `production-256.json` records 1024 active matches, 512 windows per step and eight microbatches.

Cost: about $60 per hour for 8 B200s with 64 cores and 512 GiB. Expected time is about 2.3 hours
(startup, head start, 32 warmup updates at about 14 s, 224 updates at 27 s, eight checkpoints, the
final held-out evaluation), so about $140 to $170. The 12,600 s timeout reserves $220.81. The B200
ledger in `runs/modal-b200/budget.jsonl` is cumulative: $165.58 of earlier reservations plus this
run, under the $390 cap. A failed first attempt reserved $220.81 without starting; that line was
removed from the ledger.

### What to watch

- Startup: all eight ranks finish capture and the first rollout tick time is near 28 ms.
- Update 32: the first DDP checkpoint, and the end of critic warmup. Lambda calibration raises an
  error if three advantage standard deviations exceed `lambda_max = 100`; earlier stds were 11 to
  18. Watch `critic/advantage_std_*`.
- Memory: the bench peaked at 96.0 GiB allocated and 163.8 GiB reserved of 178.3 GiB. NCCL and DDP
  buffers take some of the remaining 15 GiB.
- Support validation: a reverted memory change once failed the check that student and reference
  give nonzero probability to the same actions. The cause is unknown and could exist in the kept
  code. A failure stops the run loudly; resume from the last checkpoint.
- Policy movement: `policy/old_kl_step{step}` per optimizer step and the clip fraction. KL per
  update far above 0.01 to 0.02 calls for a target-KL stop or a lower lr.
- Gameplay against the reference and snapshots: `gameplay/pi_ref_win_rate_32_updates` and the
  anchor equivalents.

## Where an update spends its time

| Phase | Control (`a0ae12df`) | Merged four-group bench (`ff2eae4d`) |
|---|---:|---:|
| Rollout | 14.6 s, 43.6 ms per tick | 8.2 s, 28.7 ms per tick, 14% MFU |
| Preparation | 6.3 s | 4.4 s, 3.7 s of it reference, 22% MFU |
| Actor | 10.9 s, 16 decisions per window, 118 steps | 13.4 s, every decision, 15 steps, 24% MFU |
| Critic | 1.0 s | 0.95 s |
| Update | 32.8 s | 26.8 to 27.4 s |

That is 19.4k match-frames per second per GPU. At 8 GPUs, about 155k match-frames per second and
about 110k trained decisions per second. Switching groups costs nothing visible.

Rollout gains came from symmetric A/B buffer prefetch, one stacked graph for the four anchor models,
direct trunk decode attention, capture priority for anchors, and BF16 weight caching. GPU idle per
tick fell from 10.2 to 4.6 ms. Preparation and actor gains came from grouping windows by (delay,
replan) so no graph runs empty slots or padded queue positions, and from compiling the BF16 trunk.

Memory at the bench: 96.0 GiB allocated, reserved 139, 155, 155, 164 GiB across the warm updates.
Graph pools hold 35.8 GiB that the eager critic cannot reuse. The first four-group attempt ran out
of memory because each group kept its own KV cache and graphs; the groups now share them. The second
ran out in preparation; shared serial graph pools, chunked KL and earlier release of reference
activations fixed it. The local suite also exposed a Torch 2.11.0 reference-count bug in the
pool-filtered memory snapshot binding; the trainer works around it (`e7099609`).

## How the trainer uses data

- Each update collects 512 frames from one group of 1024 matches per rank while the other three
  groups stay frozen. Consecutive updates use disjoint matches; a match returns every fourth update,
  about 104 s or 60 optimizer steps later. Over 256 updates each match plays about 32k frames, one
  to three games.
- An update yields 8192 windows per rank: 1024 matches, two players, four 128-frame windows. About
  7550 are student windows. The actor shuffles them into 15 disjoint optimizer steps and uses each
  window once. About one window in five shares a match with another window in its step. The critic
  makes four passes.
- Each window replays 128 frames of context plus 128 frames of decisions, so the trunk computes each
  block twice. That reproduces the rollout's KV cache, which rebuilds from the previous 128 frames at
  every 128-frame boundary.
- No rollout activations are kept. Training recomputes everything from stored frames, actions,
  behavior log-probs and distributions.
- The reference policy scores every decision during preparation; the KL enters the reward.

## Critic

The critic is about 6M FP32 parameters on detached policy features. A base term applies O64's value
head (RMSNorm, SwiGLU 1024 to 512 to 1) to the trunk state. A residual term starts at zero and reads
the decoder state before the first sampled factor, both players' return conditioning, and the
opponent's next 24 queued frames through a SwiGLU queue encoder. Only the critic sees the opponent
queue. Explained variance was 0.60 to 0.63 from the first update. Ideas for a later run: an
opponent-identity input, an HL-Gauss classification loss, and the zero-sum structure of the two
players' values.

## Gradient noise

On one rollout with fixed weights:

- FP32 gradients of two disjoint 512-window batches have cosine -0.011.
- One batch against the whole update has cosine 0.255.
- BF16 against FP32 on the same batch: 0.66 to 0.73, eager or compiled.

So BF16 error is small next to minibatch noise. Each 512-window gradient is mostly noise; signal is
roughly 3 percent of its squared norm, a noise scale near 15k windows. DDP averages 4096 windows per
step. The bench used critic warmup 0, so production advantages should be less noisy. Muon takes
fixed-size steps, so noise matters more than under SGD; fewer, larger steps per update are an
option for a later run.

## Approved by the user

- Keep the block-restart trunk KV cache and the overlapping 256-frame windows.
- Keep 64-window graph chunks and 512-window optimizer batches through microbatches.
- Train every valid decision in each window.
- Group reference and actor windows by (delay, replan).
- Compile the BF16 trunk for the reference pass and the actor without the old 1.5x precision gate.
  That gate failed only on KL errors near 1e-5 nats where KL was near zero.
- Stack the four anchor models into one graph; use direct trunk decode attention.
- Keep 1024 active matches per GPU with four groups; run 256 updates.
- B200 sandboxes either profile or shut down. Batch changes into one bench. Keep compile caches on
  `hal-o65-devbox-v2`. Sandboxes stop after 10 idle minutes.
- Production may request more host RAM if a measurement shows the need. The measured peak was
  about 26 to 29 GiB per rank.

## Measured and dropped

- Overlapping rollout with training under a one-update-stale behavior policy: 0.946 of serial time
  against a 0.8 gate. The GPU was already 91 percent busy.
- Stale decoder K/V: refreshing every second decision gave KL 0.0056 and 4 percent of PPO ratios
  outside [0.8, 1.25]; never refreshing gave 24 percent. The decoder graph would shrink 7 to 20
  percent, because sequential sampling dominates decoder time.
- FP8 trunk K/V: KL 0.001, 0.6 percent of ratios outside the clip, little time saved.
- Larger batches: the student graph costs a fixed 2.7 ms plus 11.5 us per match, so more matches
  save 10 to 15 percent per sample and enlarge each update.
- Fusing head, softmax and sampling: under 1 ms per student graph.
- Expandable allocator segments: no memory gain with the captured graphs.
- Boundary cache release: lowered reserved memory to 140 GiB but failed support validation; reverted.
- A 2048-match bench and the 110 ms per-step overhead at 64-window batches: skipped to save credits.

## Open questions

- Mean game length: no game finished in six bench updates.
- The support-validation failure in the reverted change: cause unknown. The trainer now reports the
  factor, chunk, category and both log probabilities on failure.
- The required roundtrip integration tests need a dev MDS fixture that is missing locally; the two
  session-cleanup tests pass.

## Branches and reports

- `o65-rollout-speed` in `~/src/hal-o65-247m`: integration branch and production source. Reports in
  `experiments/o65/`: `rotating-groups-report.md`, `memory-headroom-report.md` and their JSON.
- `o65-tick` in `~/src/hal-o65-tick`: rollout work. Report `~/data/scratch/o65-tick3-report.md`.
- `o65-train` in `~/src/hal-o65-train`: training work. Report
  `experiments/o65/training-throughput-round2-measurement.md` on that branch.
- `o65-probe` in `~/src/hal-o65-probe`: stale and FP8 K/V probe,
  `~/data/scratch/o65-probe-measurements.md`. Not merged.
- `o65-overlap` in `~/src/hal-o65-overlap`: overlap probe, `experiments/o65/o65-overlap-report.md`
  on that branch. Not merged.
- Agent briefs and logs: `/tmp/claude-1000/-home-ericgu-src-hal/c49ac9e4-5488-472e-b5ff-cbdf532e937b/scratchpad/codex/`.

## Working notes

- The first production attempt (`o65-o64-outcome-256`) was stopped after about an hour, roughly $65,
  recorded in the budget ledger. Its rank 0 needed 33 minutes of startup; ranks 1 to 7 were still
  compiling after 48 minutes. Inductor keys graphs by device index, so each rank compiled for its own
  `cuda:N` and missed the cache, and every lookup went to the Modal network volume. Ranks now reorder
  `CUDA_VISIBLE_DEVICES` so each compiles for `cuda:0` (`ca58a905`).
- Next improvement: keep compiler cache lookups on local disk. Pack the cache into one archive on the
  volume and unpack it to local disk at container start, as `hal.inference.compile_cache` does.

- Codex `exec` cannot take a message mid-task. Stop it while no sandbox job runs and continue with
  `codex exec resume <session id> -` plus the message on stdin.
- Wait on agents by process ID. A `pgrep -f` loop matches its own command line and never exits, and
  `pkill -f` with a pattern from the same shell command kills that shell.
- The devbox `bench` path hardcodes one group unless `--groups` is passed.

## Standing constraints

- Push only to the user's fork (`origin`), never to the owner's remote.
- Never stop or delete the GCE RTX PRO 6000 VMs. Leave `hal-netplay-g4-1015-us-west4-a` alone.
  Keep VMs allocated after training.
- Ask before large spends.
- Clean `/tmp`, `~/data/scratch/o65-*`, scratchpad files and stream worktrees when the work ends.
