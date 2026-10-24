# O65 throughput status and decisions

This file records the measured state of the O65 247M self-play PPO trainer, the choices the user
approved, the options we measured and dropped, and the steps left before the 8xB200 run. Numbers
come from one B200 devbox with 1024 active matches, 512-frame rollouts and the O64 checkpoint
`step-0098304.pt`, unless a line says otherwise.

## Where an update spends its time

| Phase | Control (`a0ae12df`) | Latest branch result |
|---|---:|---:|
| Rollout | 14.6 s, 43.6 ms per tick | 7.9 s, 27.7 ms per tick (`o65-tick` `364eaeb3`) |
| Preparation | 6.3 s | 4.1 s (`o65-train` `a3d080e3`) |
| Actor | 10.9 s, 16 decisions per window, 118 steps | 12.9 s, every decision, 15 steps |
| Critic | 1.0 s | about 1.0 s |
| Update | 32.8 s | about 26 s estimated, not yet measured merged |

The merged branch `o65-rollout-speed` (merge commits `e00a990c`, `c5a17bd3`) is being benchmarked
with four match groups. Until that report lands, the 26 s figure is an estimate.

At 26 s per update, 8 GPUs give about 161k match-frames per second and about 114k trained
decisions per second. 256 updates take about 1.9 hours plus startup.

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
  last measured host peak was 18 to 32 GiB, so 64 GiB suffices.

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

- BF16 actor gradients have cosine 0.66 to 0.83 against FP32, eager or compiled. The merged bench
  compares this with the cosine between FP32 gradients of disjoint 512-window batches. If minibatch
  noise is much larger, BF16 error does not matter.
- Mean game length is unmeasured. Games stop at 28,800 frames. Each match plays about 32k frames
  over 256 updates, which is one to three games.
- The policy may move too far per update under Muon's fixed step size. Watch
  `policy/old_kl_step{step}` and the clip fraction in the first updates. If KL per update is far
  above 0.01 to 0.02, add a target-KL stop or lower the lr.

## Before the 8xB200 run

0. The first four-group bench ran out of GPU memory in startup capture (176 GiB allocated, 93.7 GiB
   in graph pools). The policy keys its trunk KV cache and tick and restart graphs by group, so all
   four groups hold their own copies. The integration agent is making the groups share them.
1. Read the merged four-group bench: update time, memory per phase, host RAM, group switch cost,
   game length, gradient-noise comparison.
2. Fill `active_matches_per_gpu`, `windows_per_step` and `actor_microbatches` in
   `production-256.json`.
3. Make the production launcher start from the warm compile cache so 8 GPUs do not spend about
   30 minutes compiling.
4. Run the required integration tests. Agent worktrees lacked `.env`, so the tests could not find
   the ISO; `hal-o65-247m` now has a copy.
5. Ask the user to approve the launch with the measured cost. Launch the 256-update run directly
   and watch the first updates for DDP errors, memory and policy KL, instead of a separate
   multi-GPU check. Evaluate head-to-head against O64 every 32 updates and extend toward about
   1000 updates only while win rate still rises.

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
