# O65 throughput status and decisions

This file records the measured state of the O65 247M self-play PPO trainer, the choices the user
approved, the options we measured and dropped, and the steps left before the 8xB200 run. Numbers
come from one B200 devbox with 1024 active matches, 512-frame rollouts and the O64 checkpoint
`step-0098304.pt`, unless a line says otherwise.

## Where an update spends its time

| Phase | Control (`a0ae12df`) | Latest branch result |
|---|---:|---:|
| Rollout | 14.6 s, 43.6 ms per tick | 8.155 s, 28.730 ms per ordinary tick, merged four-group run |
| Preparation | 6.3 s | 4.438 s, including 3.663 s reference |
| Actor | 10.9 s, 16 decisions per window, 118 steps | 13.388 s, every decision, 15 steps |
| Critic | 1.0 s | 0.949 s |
| Update | 32.8 s | 26.980 s measured; memory headroom unqualified |

The merged branch completed six four-group updates (groups 0, 1, 2, 3, 0, 1). This is the
production 512-window/eight-microbatch/all-decision workload; the original control used sampled
actor decisions, so the table is not an isolated throughput comparison. The previous matched
four-group memory control failed before an update. See
[memory-headroom-report.md](memory-headroom-report.md) and its measurement JSON.

The measured run reserved 163.801 GiB at peak, above the requested ceiling of about 160 GiB.
The lower 110–120 GiB target is unmet. A cache-release follow-up failed support validation in
the fourth group's preparation and was reverted. Capacity fields remain unset. The single
B200 sandbox flushed `/cache` and is down. No 8-GPU throughput claim or launch follows from this run.

## How the trainer uses data

- Production rotates four match groups per GPU (`groups: 4` in `production-256.json`). Each update
  collects 512 frames from one group of 1024 matches while the other three stay frozen. A GPU holds
  4096 matches; 8 GPUs hold 32,768. A match returns every fourth update.
- An update yields 8192 windows per GPU: 1024 matches, two players, four 128-frame windows. About
  7550 are student windows. The actor shuffles them into 15 disjoint optimizer steps of 512 and
  uses each window once. The critic makes four passes.
- Each window replays 128 frames of context plus 128 frames of decisions, so the trunk computes
  each 128-frame block twice. Training does this to reproduce the rollout's KV cache, which rebuilds
  from the previous 128 frames at every 128-frame boundary.
- No rollout activations are kept. Training recomputes everything from the stored frames, actions,
  behavior log-probs and distributions.
- The reference policy scores every decision during preparation, and the KL enters the reward.
- The actor optimizer is Muon (momentum 0.95, lr 1e-5) with state kept across updates.
  The trainer logs `policy/old_kl_step{step}` for each optimizer step.

## Approved by the user

- Keep the block-restart trunk KV cache and the overlapping 256-frame windows.
- Keep 64-window graph chunks. Keep 512-window optimizer batches through microbatches.
- Train every valid decision in each window (`decisions_per_window: "all"`).
- Group reference and actor windows by (delay, replan) so no graph runs empty slots or padded queue
  positions.
- Compile the trunk in BF16 for the reference pass and the actor, without the old 1.5x precision
  gate. That gate failed only on KL errors near 1e-5 nats, measured where KL was near zero.
- Stack the four anchor models into one graph. Use direct trunk decode attention.
- B200 sandboxes either profile or shut down. Batch changes into one bench. Keep compile caches on
  the Modal Volume `hal-o65-devbox-v2`. Sandboxes stop after 10 idle minutes.
- Production may request more than 64 GiB host RAM per GPU if a measurement shows the need. The
  latest sampled host peak was 28.98 GiB in the full profile (23.14 GiB in the gradient probe),
  so this evidence does not require more than 64 GiB.

## Measured and dropped

- Overlapping rollout with training under a one-update-stale behavior policy: the pair took 0.946
  of serial time against a 0.8 gate. The GPU was already 91 percent busy.
- Stale decoder K/V: refreshing every second decision gave KL 0.0056 and 4 percent of PPO ratios
  outside [0.8, 1.25]; never refreshing gave 24 percent. The decoder graph would shrink only 7 to 20
  percent.
- FP8 trunk K/V: KL 0.001, 0.6 percent of ratios outside the clip, little time saved.
- Larger batches: the student graph takes a fixed 2.7 ms plus 11.5 us per match, so more matches
  per GPU save 10 to 15 percent per sample and enlarge each PPO update.
- Fusing head, softmax and sampling: under 1 ms per student graph.
- A 2048-match bench and the per-optimizer-step overhead (about 110 ms per step at 64-window
  batches) are deferred to save credits; production takes 15 steps per update.

## Open questions

- Fixed-weight gradient comparison: FP32 disjoint batches have cosine -0.011. Eager BF16 vs
  FP32 is 0.727/0.657; compiled BF16 is 0.698/0.676. FP32 batch 0 vs the full update is 0.255.
  These two minibatches vary more than the precision comparisons; this does not establish
  convergence or unbiased precision error. The full report records the inputs and scope.
- No finished games occurred in six profile updates, so mean game length remains unavailable.
  Games stop at 28,800 frames. A longer gameplay measurement is needed.
- The policy may move too far per update under Muon's fixed step size. Watch
  `policy/old_kl_step{step}` and the clip fraction in the first updates. If KL per update is far
  above 0.01 to 0.02, add a target-KL stop or lower the lr.

## Before the 8xB200 run

0. Shared serial graph pools, chunked KL and reference activation release removed the first
   preparation OOM: six full updates completed at 1024 active matches. The graph pools reserve
   35.766 GiB; the full run peaks at 96.013 GiB allocated and 163.801 GiB reserved. The latter
   misses the headroom criterion. The rejected boundary-cache-release repeat completed three
   updates, then failed support validation; cause remains unresolved and diagnostics were added.
1. Diagnose that support failure from actual offending probabilities, then reduce or reuse eager
   workspace and repeat all six updates below about 160 GiB reserved. Do not qualify the rejected
   repeat from its lower memory peaks. Reports: `memory-headroom-report.md`,
   `memory-headroom-measurement.json`, `memory-headroom-validation.json`.
2. Fill `active_matches_per_gpu`, `windows_per_step` and `actor_microbatches` in
   `production-256.json`.
3. Make the production launcher start from the warm compile cache so 8 GPUs do not spend about
   30 minutes compiling.
4. Complete the required roundtrip integration tests once the correct dev MDS fixture is present.
   The six fixture-dependent cases remain blocked; `.env` is present in this worktree.
5. Ask the user to approve the launch with the measured cost. Launch the 256-update run directly
   and watch the first updates for DDP errors, memory and policy KL, instead of a separate
   multi-GPU check. Evaluate head-to-head against O64 every 32 updates and extend toward about
   1000 updates only while win rate still rises.

## Launch decision

The user approved launching the 8xB200 job for 256 updates once the four-group bench shows an
update near 26 s and memory with headroom. The run uses `production-256.json` with its schedule
constants (checkpoint every 32 updates, snapshot every 64, critic warmup 32), after checking each
one. Keep 1024 active matches per GPU (8192 active, 32,768 held across four groups). Checkpoints are
written by a background thread (`CheckpointWorker` in `065_selfplay_ppo.py`); the final one is
synchronous.

Budget: Modal bills about $7.56 per B200-hour with 17 cores and 64 GiB, about $60 per hour for 8.
256 updates at 26 s take 1.9 hours, plus startup; the cap is $200.

## Branches and reports

- `o65-rollout-speed` in `~/src/hal-o65-247m`: integration branch.
- `o65-tick` in `~/src/hal-o65-tick`: rollout work. Report `~/data/scratch/o65-tick3-report.md`.
- `o65-train` in `~/src/hal-o65-train`: training work. Report
  `experiments/o65/training-throughput-round2-measurement.md` on that branch.
- `o65-probe` in `~/src/hal-o65-probe`: stale and FP8 K/V probe. Results
  `~/data/scratch/o65-probe-measurements.md`. Not merged.
- `o65-overlap` in `~/src/hal-o65-overlap`: rollout and training overlap probe. Report
  `experiments/o65/o65-overlap-report.md` on that branch. Not merged.

## Standing constraints

- Push only to the user's fork (`origin`), never to the owner's remote.
- Never stop or delete the GCE RTX PRO 6000 VMs. Leave `hal-netplay-g4-1015-us-west4-a` alone.
  Keep VMs allocated after training.
- Ask before large spends. The 8xB200 run needs explicit approval.
- Clean `/tmp`, `~/data/scratch/o65-*`, scratchpad files and stream worktrees when the work ends.
