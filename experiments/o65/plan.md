# O65: self-play PPO in melee-sim-light

Status: plan. Nothing here is implemented or measured. Numbers marked *est.*
come from FLOP counts and assumed utilization; the pilot replaces them with
measurements. The worked example uses one B200 with $D = 12$ and $R = 4$.

O65 fine-tunes the O64 production percentile checkpoint with KL-regularized
self-play PPO. Rollouts run in melee-sim-light (MSL) through HAL's native
adapter, and finalists are confirmed in Dolphin.

## Decisions

| Topic | Decision |
| --- | --- |
| Starting policy | The O64 production checkpoint. A frozen copy serves as $\pi_{\text{ref}}$. |
| Objective | Maximize $\mathbb{E}[\text{return}] - \lambda\,\mathrm{KL}(\pi \,\Vert\, \pi_{\text{ref}})$. The KL enters the reward, computed exactly from each factor's full logits along the sampled path. |
| $\lambda$ | Sampled per player per episode, log-uniform over $[\lambda_{\min}, \lambda_{\max}]$ with $\lambda_{\min} = 0.01$. Critic warmup sets $\lambda_{\max}$ from the measured advantage scale, provisionally 10 to 100. $\log\lambda$ enters a zero-initialized column of the conditioner. |
| $u^*$ | 0.95 in the student and the reference. The percentile head goes unused. |
| Policy loss | GSPO with a length-normalized ratio per decision and a truncated importance weight for the rollout-to-$\pi_{\text{old}}$ gap. One epoch ($E = 1$). |
| Critic | `value_head` plus a zero-initialized residual head on the detached decoder state after the queue. Inputs include $\log\lambda$, plus the opponent's $\log\lambda$ and $D$ as privileged inputs. Several epochs on cached features. |
| Delay and replanning | $D$ sampled per player per episode from $[10, 21]$. $R$ sampled from $\{4, 5, 6\}$ subject to $D + R \le 24$, with $R = 2$ at $D = 21$. |
| Offsets | RL uses offsets 1 to 24. Offsets 27 and 30 go unused. |
| Context | Block restart: 128-frame blocks with contexts of 129 to 256 frames. Deployment uses the same scheme. |
| Rollout per update | 256 frames (2 blocks). GAE bootstraps at the end. |
| Discount | $\gamma = 0.99855$ per frame. $\lambda_{\text{gae}}$ starts at 0.99 per frame. |
| Identities | (identity, character) pairs drawn with weight $\text{count}^{0.5}$ above a data floor. The masked identity at about 10%. The deployment identity (masked or Master) oversampled to 20 to 25%. |
| Starting states | Fresh matches, plus a fraction started from sampled human replay states with off-stage, ledge and recovery states oversampled. |
| Episode ends | A match end is terminal. A time-out is a truncation. No timer feature. |
| Resets in windows | Varlen flash attention masks the episode boundary, so window shapes stay static. |
| Evaluation | An in-loop group of matches against $\pi_{\text{ref}}$ with frozen weights. Finalists confirmed in Dolphin. |
| Hardware | An 8×B200 node for production. The pilot runs on RTX PRO 6000 Blackwell. |

## Objective and update rules

$$
\max_\theta\; \mathbb{E}_{\pi_\theta}\Big[\sum_t \gamma^t r_t\Big] - \lambda\,\mathrm{KL}\big(\pi_\theta \,\Vert\, \pi_{\text{ref}}\big),
\qquad \pi_\lambda \propto \pi_{\text{ref}} \exp(Q/\lambda),
\qquad \log\lambda \sim \mathcal{U}\big[\log\lambda_{\min},\, \log\lambda_{\max}\big]
$$

A smaller $\lambda$ gives a higher expected return, and $\lambda \to \infty$
recovers $\pi_{\text{ref}}$. Every $\lambda$ starts at $\pi_{\text{ref}}$
because the new conditioner column is zero.

$\lambda$ carries units of reward per nat. At the optimum, each action gets
reweighted by $\exp(A/\lambda)$. At $\lambda = 1$, an action worth 3 damage
more than its alternatives gains a factor of about 20 over $\pi_{\text{ref}}$.
So the range has to extend to values comparable to the per-decision advantage
scale before any part of it stays close to $\pi_{\text{ref}}$. We want that
close end for two reasons:

1. It steadies early training.
2. It gives a conservative deployment option if small $\lambda$ exploits
   $\pi_{\text{ref}}$ rather than improving play.

Critic warmup measures the advantage scale, and we set $\lambda_{\max}$ from it.

### Decisions and factors

A player decides every $R$ frames. At decision $j$, made at frame $t_j$:

- the decoder replays the $D$ committed actions at offsets 1 to $D$;
- it samples offsets $D+1$ to $D+R$, five controller factors per offset, in
  decode order.

$\mathcal{E}_j$ denotes the $5R$ executed factors of decision $j$.

### Reward and KL

$$
\begin{aligned}
\mathrm{KL}_j &= \sum_{i \in \mathcal{E}_j} \mathrm{KL}\big(\pi_\theta(\cdot \mid c_i)\,\Vert\,\pi_{\text{ref}}(\cdot \mid c_i)\big) \\
r_j &= \sum_{f=1}^{R} \gamma^{f-1}\, r_{t_j+f} \;-\; \lambda\, \mathrm{KL}_j
\end{aligned}
$$

$r_f$ is the per-frame reward from `hal/training/returns.py`: damage dealt
minus damage taken, 120 per stock difference, and 50 more on the deciding
stock. The context $c_i$ holds $h_j$, the forced prefix, the earlier sampled
factors and $u^*$. Both models apply the same button mask. $\mathrm{KL}_j$ and
the frame rewards both accrue per frame, so $\lambda$ keeps its meaning across
$R$.

### Advantages

$$
\begin{aligned}
\gamma_R &= \gamma^R, \qquad \lambda_R = \lambda_{\text{gae}}^R \\
\delta_j &= r_j + \gamma_R\,(1 - d_j)\,V_{j+1} - V_j \\
\hat A_j &= \delta_j + \gamma_R\,\lambda_R\,(1 - d_j)\,\hat A_{j+1} \\
V^{\text{target}}_j &= \hat A_j + V_j
\end{aligned}
$$

We lift the per-frame $\gamma$ and $\lambda_{\text{gae}}$ to the decision grid.
$d_j$ marks a match end; a time-out bootstraps. GAE runs over each stream's 256
rollout frames and bootstraps with $V$ at the end.

$\lambda_{\text{gae}}$ starts at 0.99 per frame, which credits about 100
frames. At 0.95 per frame, $\lambda_R = 0.81$ at $R = 4$, credit reaches back
only about 20 frames, and the advantage leans almost entirely on a new,
partly privileged critic. We lower it only if advantage variance hurts.

### Policy loss (GSPO)

$$
\begin{aligned}
s_j(\theta) &= \exp\Big(\frac{1}{n_j} \sum_{i \in \mathcal{E}_j} \big[\log \pi_\theta(a_i \mid c_i) - \log \pi_{\text{old}}(a_i \mid c_i)\big]\Big), \qquad n_j = 5R \\
w_j &= \min\Big(\exp \sum_{i \in \mathcal{E}_j} \big[\log \pi_{\text{old}}(a_i \mid c_i) - \log \mu(a_i \mid c_i)\big],\; C\Big) \\
L_\pi &= -\frac{1}{J} \sum_{j} \mathrm{sg}(w_j)\, \min\big(s_j\,\hat A_j,\; \operatorname{clip}(s_j,\, 1-\epsilon,\, 1+\epsilon)\,\hat A_j\big)
\end{aligned}
$$

- $\pi_{\text{old}}$ log-probs come from the student's no-grad pass, which uses
  the training step's kernels.
- $\mu$ is the rollout policy. It has the same weights, but the rollout decodes
  from the KV cache with other kernels and batch shapes, so its log-probs differ
  slightly from $\pi_{\text{old}}$. The actions come from $\mu$, so the
  truncated weight $w_j$ corrects for the gap. $\mathrm{sg}$ stops the
  gradient. An uncorrected gap of this kind acts as off-policy data and has
  destabilized RL on language models.
- We normalize $\hat A$ per minibatch, and per $D$ group if their scales
  differ.
- The length-normalized ratio stays close to 1, so $\epsilon$ is much smaller
  than PPO's 0.2. The pilot tunes it to a clip fraction of roughly 10 to 20%.
- The policy loss trains 16 decisions per window, sampled at uniform positions.
  GAE uses every decision.
- With one epoch, each window enters exactly one gradient step.

### Critic loss

Squared error to $V^{\text{target}}$, scaled like O64's value loss. The critic
trains for $K_v$ epochs on features cached from the no-grad pass, with no trunk
or decoder recompute.

### Optimizer

- BC's optimizer family, with the learning rate swept on the pilot. RL
  fine-tuning needs a much smaller rate, and each step carries about 8 times
  BC's tokens.
- Weight decay 0. Decay pulls weights toward zero, away from $\pi_{\text{ref}}$,
  and so works against the KL term.
- Muon's orthogonalized updates scale differently from Adam's, so the sweep
  covers that.
- Gradient clipping and a short warmup.

## Model changes

- **Conditioner.** `Linear(1, 128)` becomes `Linear(2, 128)`. The new column
  starts at zero and reads $\log\lambda$, rescaled to $[-1, 1]$ over the final
  sampling range. $u^*$ stays at 0.95 and enters as $(2u - 1) \cdot 120$.
- **Critic.**
  $V = \text{value\_head}(\text{trunk}) + \text{head}_{\text{res}}(z_{D+1},\ \log\lambda,\ \log\lambda_{\text{opp}},\ D_{\text{opp}})$,
  with $z_{D+1}$ the decoder state at offset $D+1$, after it has consumed the
  queue. Both feature inputs are detached. The output layer of
  $\text{head}_{\text{res}}$ starts at zero, so $V$ starts exactly at
  `value_head`. The opponent's queue state can join later.
- **No $D$ input to the policy.** The decoder reads offset embeddings, so it
  already knows where sampling starts.
- **Percentile head.** It stays in the checkpoint and receives no loss.
- **Inference.** A per-stream trunk KV cache with per-stream lengths, an
  incremental cache of the decoder's history projection, and decode variants
  compiled per $(D, R)$.

## Context, caches and windows

BC supervised only contexts of 129 to 256 frames (`direct_loss_start = 128`).
Each block lasts 128 frames, and a decision inside a block reads from 128 frames
before the block's start up to itself.

1. At each block start, we encode the previous 128 frames from scratch under the
   current weights. Positions restart at 0.
2. During the block, we append each frame to the cache. The window grows from
   129 to 256 frames and drops nothing, so a single causal pass over the window
   reproduces every cached key and value.
3. At the block end, we discard the cache.

128 is the longest block this checkpoint supports: a longer block would put
decisions at contexts below 129 frames, which BC never trained. The update
boundary falls on a block boundary, and the weight change invalidates the cache
there anyway.

A reset inside a window starts a new varlen segment. The training pass masks
the same boundary.

For netplay, a second cache starts at each block's start, receives every frame
and swaps in at the boundary. This removes the 128-token prefill spike and keeps
the cost at 2 trunk tokens per frame.

## Rollout system

Each GPU runs one process. `hal/sim/native_adapter.c` already projects
`MslObservation` into HAL's 71 schema columns, quantizes actions to wire inputs
as the pinned libmelee does, and computes the reward in canonical float32. It
writes one 440-byte `HalFrame` per match-frame. O65 builds on that adapter.

### Pool and groups

Each GPU holds 4k matches in $G = 4$ groups of 1k. Each group is a separate
native batch with separate buffers. One group steps per update while the others
wait. To suspend a group we stop stepping it; to resume it we prefill from its
frame history. Neither step copies simulator state.

### Data path

1. The adapter writes each step's `HalFrame` records into a page-locked buffer.
2. We copy them into a GPU ring of 384 frames per match: 128 frames of context
   plus the 256-frame rollout. At about 160k match-frames/s per GPU during
   rollout, that comes to about 70 MB/s host-to-device *est.*
3. The GPU builds HAL's `Context` from the columns, takes rewards from the
   adapter's reward field and runs the policy.
4. The sampled indices go back as 14-float action vectors, and the adapter
   quantizes them into `MslInput`.

The columns carry both players, so each stream's ego view follows HAL's port
convention without a slot swap.

### Stepping loop

The active group splits into halves A and B. While the GPU decides for one half,
the simulator steps the other half $R$ frames from its queue. A decision made at
frame $t$ first executes at frame $t + D + 1$, after $t + R$, so the overlap is
valid. `ctypes.CDLL` releases the GIL during the native call, so simulator
threads for separate batches run in parallel.

Per half at $D = 12, R = 4$ *est.*:

- GPU: $1\text{k streams} \times 4 \text{ frames} \times 0.94 \text{ GFLOP} \approx 3.8$ TFLOP, about 11 ms at 15% decode utilization.
- Simulator: 500 matches × 4 frames on about 4 cores, about 10 ms at an assumed
  50k frames/s per core.

### Episodes and resets

A finished match resets in place. A fresh reset samples:

- a matchup from the training prior;
- a stage from the supported set, with Stadium frozen;
- each player's (identity, character) pair;
- $\lambda$, $D$ and $R$ for each player;
- a stock count and starting percents, to mix match phases.

A fraction of resets start from a human replay state instead:

1. We sample a frame from an admitted replay, weighting off-stage, ledge and
   recovery states up. The failure budget puts 91% of deaths at the bottom blast
   zone, and fresh matches arrive at those states rarely.
2. MSL replays the recorded inputs and RNG from game start to that frame. At
   about 120k frames/s per core, a full match takes about 0.1 core-seconds.
3. The 128 replay frames before the start become the first block's context. The
   replay's next $D$ actions fill each player's committed queue.
4. Stage, characters and identities follow the replay, with the usual masking
   rate. We sample $\lambda$, $D$ and $R$ as for a fresh match.

Frames before `GAME_START_FRAME` (−123) stay out of the context, as in
`hal/data/extract.py`. The simulator runs the countdown with neutral inputs.

A reset mid-block restarts that stream's cache and positions.
`flash_attn_with_kvcache` handles per-stream lengths through `cache_seqlens`.

### Decode grouping

We group streams by $(D, R)$ for compiled decode. `window_policy` already keys
compiled decoders on bucket, horizon and committed frames. The two players of a
match may fall in different groups.

### In-loop evaluation group

Every $N$ updates, a fixed group of matches plays the current weights, frozen at
the start of the evaluation, against $\pi_{\text{ref}}$. It covers a grid of
$\lambda$ and $D$ with fixed seeds. It runs inference only and contributes no
training data.

## One update

1. Prefill block 1: for each stream in the active group, encode the previous 128
   frames.
2. Roll out 128 frames. Each decision stores its sampled indices, $\log\mu$ and
   $V$.
3. Prefill block 2 from block 1's frames, then roll out 128 more.
4. Run the no-grad pass over all windows. The student and the reference both run
   every decision, which yields the exact per-decision KL, $\pi_{\text{old}}$
   log-probs and the critic features. We cache the critic features.
5. Compute rewards and GAE over each stream's 256 frames.
6. Run the critic epochs on the cached features.
7. Run one GSPO epoch. Each step takes 512 windows per GPU with DDP across the
   node, and we spread a match's windows across steps.
8. Checkpoint every $N$ updates.

Estimated timeline per B200 at $D = 12, R = 4$, 1k active matches (2k streams)
*est.*:

| Phase | Time |
| --- | ---: |
| Rollout, two blocks with prefills | 1.6 s |
| No-grad pass, student and reference | 1.6 s |
| GAE and critic epochs | 0.1 s |
| GSPO, 8 steps | 2.2 s |
| Update total | 5.5 s |

Assumed utilization of the 2.25 PFLOP/s dense BF16 peak: 35% for prefill and the
no-grad pass, 15% for decode, 30% for training.

## Throughput and sizing

A training pass costs about 6 FLOPs per parameter per token, and a forward pass
about 2. The trunk has 151M parameters and the decoder 81M. With
$k = (D + R)/R$ decoder tokens per frame:

| Pass | Tokens | GFLOP |
| --- | --- | ---: |
| Rollout, per stream-frame | 1 trunk append + 1 amortized prefill + $k$ decoder | $0.60 + 0.16k$ |
| No-grad pass, per window, both models | 256 trunk + $(128/R)(D+R)$ decoder, each | $2 \times (77 + 0.16 \cdot \text{decoder tokens})$ |
| GSPO step, per window | 256 trunk + $16\,(D+R)$ decoder | $232 + 0.49 \cdot 16\,(D+R)$ |

Worked example per B200 *est.*:

| Quantity | $D = 12, R = 4$ | $D = 21, R = 2$ |
| --- | ---: | ---: |
| Rollout per update | 1.6 s | 3.4 s |
| No-grad pass | 1.6 s | 3.2 s |
| GSPO epoch (8 steps) | 2.2 s | 2.4 s |
| Update total | 5.5 s | 9.2 s |
| Match-frames/s per GPU | 46k | 28k |
| Match-frames/s per node | 370k | 220k |

A mix of $D$ falls between the two columns. At large $D$ the no-grad pass
dominates, because it runs the decoder through $D + R$ tokens for every decision
of both models.

### Calibration against BC

[The O64 throughput report](../o64/performance/report.md) measured a BC update
of 512 windows on one B200 at 0.534 s for an estimated 340.8 TFLOP, or 28.3% MFU,
with 175.9 GiB peak reserved. That agrees with the 30% training utilization
assumed here. An RL GSPO window at $D = 12, R = 4$ with 16 decisions costs about
0.56 of a BC window, since it runs 256 decoder tokens instead of 832. Decode and
prefill utilization remain unmeasured, so the pilot measures them first.

### CPU and host

- Simulator demand comes in bursts. It runs only during rollout, at about 160k
  match-frames/s per GPU, which needs about 4 cores per GPU at an assumed 50k
  frames/s per server core, or about 32 per node *est.* Bind them to each GPU's
  socket.
- The pool needs about 1.5 to 2 MB of host RAM per match for the simulator
  arena and its config, or about 64 GB per node at 32k matches.
- Replay starts add fast-forward work, about 0.1 core-seconds per start.
- Feature building runs on the GPU, which removes the Python per-frame path from
  the loop.

## Memory

Per B200 at 1k active matches *est.*:

| Phase | Fixed | KV cache | Other | Total |
| --- | ---: | ---: | ---: | ---: |
| Rollout | 14 GB | 27 GB | 2 GB | 43 GB |
| No-grad pass | 14 GB | 27 GB | 8 GB | 49 GB |
| GSPO step | 14 GB | 27 GB | 82 GB | 123 GB |

- **Fixed:** student weights, gradients and optimizer state, BF16 reference
  weights and workspace.
- **KV cache:** 13.6 MB per stream for 2k streams. It holds the trunk's keys and
  values over 256 frames in 12 layers, plus the decoder's history projection.
- **GSPO activations:** about 160 MB per window for 512 windows without
  recomputation. The trunk accounts for about 107 MB and the decoder's
  $16 \times 16$ tokens for about 53 MB, at roughly $34d$ bytes per token per
  layer.
- **Allocator overhead.** The same rule predicts about 157 GB for the O64 BC
  step, which measured 175.9 GiB (189 GB) peak reserved. Applying that 20% to
  the GSPO step gives about 148 GB of a B200's 192 GB.
- **The reference stores no activations.** No gradient flows through
  $\pi_{\text{ref}}$, and it runs only in the no-grad pass.
- **The KV cache can stay resident** at 1k active matches, which allows static
  allocation and keeps CUDA graphs available for decode. Beyond about 4k active
  matches per GPU, we would release the cache between phases, through the
  caching allocator with `expandable_segments` or through CUDA virtual memory
  release as in torch_memory_saver.
- **A larger step** of about 650 windows per GPU fits without accumulation, if
  the gradient noise scale calls for it.

## Decorrelation

| Quantity | Value |
| --- | ---: |
| Windows per gradient step (node) | 4,096 (512 × 8 GPUs) |
| Distinct matches per step | 4,096 |
| Steps per update | 8 |
| Windows per match visit | 4 (2 players × 2 blocks), each in a different step |
| Updates between visits to a match | 4 ($G$) |
| Steps between visits | at least 24 |
| Window repeats | none ($E = 1$) |

Steps per update equal $A_{\text{local}}/128$, where $A_{\text{local}}$ counts
active matches per GPU. At 1k active matches, each update runs 8 steps before
new data arrives, which keeps the drift from $\pi_{\text{old}}$ small. The pilot
sweeps $A_{\text{local}}$ from 1k down to 512 and compares the KL to
$\pi_{\text{old}}$ across steps against decode utilization. The gradient noise
scale measured on the pilot decides whether 4,096 windows per step lies above or
below the critical batch size.

## Critic warmup

- **Rollouts.** $\pi_{\text{ref}}$ plays under PPO's sampling: $u^* = 0.95$ and
  sampled identities, $D$, $R$, $\lambda$ and starting states. At init the
  student equals $\pi_{\text{ref}}$ for every $\lambda$, so these rollouts are
  on-policy for all of them.
- **Training.** Only the critic heads train: `value_head` and
  $\text{head}_{\text{res}}$ with its privileged inputs, all on detached
  features. Warmup skips the no-grad pass, since the KL is zero.
- **$\lambda$ dependence waits for PPO.** During warmup $\lambda$ changes no
  behavior, so the critic cannot yet learn how the return depends on either
  player's $\lambda$. Those weights stay near zero until PPO separates the
  policies, which is the correct value until then. $D$ does change behavior, so
  the critic learns $D$ and queue effects during warmup.
- **$\lambda$ calibration.** Warmup measures the spread of per-decision
  advantages under $\pi_{\text{ref}}$, by $D$ and $R$. We set $\lambda_{\max}$
  to a few times that spread before PPO starts and record it in the resolved
  configuration.
- **Phase mixing.** Every fresh match starts at frame −123 at the same time.
  Warmup continues until each match has advanced about one match length, roughly
  $43 \text{ visits} \times G = 172$ updates, or about 5 minutes on the node at
  rollout-only cost *est.* Stock counts, starting percents and replay starts mix
  phases further.
- **Ending.** Warmup ends once the critic's explained variance plateaus and the
  phases have mixed.

## Exact resume

We checkpoint at update boundaries. Every update starts with a prefill, so the
KV cache stays out of the checkpoint. A checkpoint holds:

1. the model, optimizer, scheduler and the critic cache's step counter;
2. every pool match's state from the adapter's save path, compressed, at a size
   not yet measured;
3. each match's last 128 `HalFrame` records and both players' committed queues;
4. episode metadata: $\lambda$, $D$, $R$, identities, characters, start type,
   episode counters and seed keys;
5. the pool schedule position and the evaluation schedule position.

Sampling, resets and replay-start draws use counter-based RNG keyed by (run
seed, stream, episode, frame), which leaves no generator state to save. Tests
compare the next batch and the next optimizer update after a resume with an
uninterrupted run.

## Evaluation and run protocol

### Controls before RL

- $\pi_{\text{ref}}$ at $u^* = 0.95$ at each $D$ and $R$ in Dolphin, with block
  restart against the current sliding context.
- $\pi_{\text{ref}}$ head-to-head in MSL against the same matches in Dolphin, to
  check gameplay agreement within confidence intervals.

### First run

- **Treatment:** PPO from $\pi_{\text{ref}}$ with the decisions above.
- **Control:** $\pi_{\text{ref}}$ under the same $u^*$, $D$ and $R$ sets,
  identity sampling, matchup prior, stage set and evaluation protocol.
- **Invariant inputs:** checkpoint, $u^*$, identity sampling, matchup prior,
  stage set, evaluation seeds, simulator build and data manifest.
- **Selection:** `net_stock_lcb` and head-to-head win rate against
  $\pi_{\text{ref}}$ in MSL at matched $D$, over a $\lambda$ grid. Finalists are
  confirmed in Dolphin. Offline metrics serve only for diagnosis.
- **Records:** Git SHA, resolved configuration, seed, environment, simulator and
  adapter hashes, data manifests and checkpoint hashes.

### Pilot on RTX PRO 6000 Blackwell

1. Run the 17M proxy architecture end to end on one GPU to debug the loop,
   logging and resume.
2. Run the 248M checkpoint with a small active set to measure per-phase
   utilization, the $\mu$-to-$\pi_{\text{old}}$ gap and the gradient noise
   scale.
3. Test DDP on two GPUs. NCCL rejects two ranks on one device, so a gloo CPU
   test covers the gradient sync logic as a fallback.

## Metrics

| Group | Metrics |
| --- | --- |
| Policy | KL per frame to $\pi_{\text{ref}}$ by $D$ group and $\lambda$ bin, KL to $\pi_{\text{old}}$ across the steps of an update, clip fraction, ratio quantiles, entropy per controller group |
| Critic | explained variance overall and by position in the block, value loss, advantage mean and std by $D$ group |
| Numerics | $\log(\mu/\pi_{\text{old}})$ per decision, its quantiles by $D$ group, fraction of truncated weights |
| Gameplay | net stocks, damage, win rate and match length by $\lambda$ bin and $D$, from training matches and the evaluation group |
| Systems | time per phase, decode and training utilization, simulator frames/s per core, host-to-device bytes, peak HBM per phase, compiled variant count |
| Data | identity, character, stage and matchup counts, time-outs, resets per update, replay-start fraction by state tag |

W&B logging records `global_step` as data with `define_metric` and keeps keys
one level deep.

## Risks

- **Simulator exploits.** MSL validation covers admitted human replays with
  recorded RNG. RL policies will visit states outside that set, so finalists
  are confirmed in Dolphin.
- **Non-transitive cycling.** $\lambda$ and $D$ sampling diversify opponents.
  Head-to-head against fixed anchors ($\pi_{\text{ref}}$ and earlier
  checkpoints) detects cycling. Opponent snapshots come next if it appears.
- **KL growth at small $\lambda$ and large $D$.** $\pi_{\text{ref}}$ is weakest
  at large $D$, so RL moves furthest there. The advantage scale may differ by
  $D$, and one $\lambda$ range may not fit every delay. Warmup measures the
  scale per $D$.
- **Replay-start bias.** A replay start puts human history in the context. That
  matches BC's training distribution, but neither policy produced that history,
  so the first block after a replay start is slightly off-policy for both
  players.
- **Camping.** Deferred. If it appears, an asymmetric reward change addresses
  it; a timer feature would make the observation non-Markovian.
- **Rare characters.** The matchup prior keeps them rare, and they also have the
  weakest $\pi_{\text{ref}}$. Their KL and gameplay need separate monitoring.
- **Privileged critic.** It keeps the ego's history as input, which keeps its
  baseline unbiased in a partially observed game (Baisero and Amato, 2022).
- **Deterministic order.** Exact resume depends on a deterministic data order,
  not on deterministic kernels.

## TODOs

### melee-sim-light

- [ ] Expose the replay-start path publicly: the initial seed plus per-frame RNG
      injection from recorded replays. HAL's parity test and RL replay starts
      both use it.
- [ ] Benchmark on the B200 node's server CPU. The published baseline uses a
      desktop V-cache core.
- [ ] Measure thread scaling across separate batches stepped in parallel.
- [ ] Measure save size, add compression, and test save and restore across
      batches for determinism.
- [ ] Confirm that Stadium always runs frozen, and document the character-id
      space.

### HAL: simulator boundary (`hal/sim`, torch-free)

- [ ] Integration test of `native_adapter.c` against the real MSL build. Current
      tests use a stub.
- [ ] Parity test: replay through MSL and the adapter's 71 columns against
      `hal/data/extract.py`, bitwise on admitted replays.
- [ ] Parity test: the adapter's reward against `hal/training/returns.py` on the
      same replays.
- [ ] Parity test: the adapter's quantized wire inputs against the Dolphin input
      path for the same action stream.
- [ ] Adapter output into caller-provided page-locked buffers. Optionally, a
      multi-frame step that takes $R$ frames of inputs per call.
- [ ] Group batches with independent stepping, plus a thread pool bound to the
      GPU's socket.

### HAL: model and inference

- [ ] Widen the conditioner to `Linear(2, 128)` with a zero column. A parity
      test confirms identical outputs to O64 at init for every $\lambda$.
- [ ] Critic residual head with privileged inputs and a zero-initialized output.
      A parity test confirms $V$ equals `value_head` at init.
- [ ] GPU `Context` builder from adapter columns, with a parity test against the
      current Python feature path.
- [ ] Block-restart inference: a per-stream trunk KV cache with
      `cache_seqlens` and an incremental decoder history projection. A parity
      test compares log-probs with the full-window pass.
- [ ] Block restart in Dolphin evaluation and netplay, with the second-cache
      handoff for netplay.
- [ ] Decode variants per $(D, R)$.

### HAL: RL loop (joins model and simulation)

- [ ] Rollout engine: pool rotation, half alternation, resets with episode
      sampling, and in-loop evaluation groups.
- [ ] Episode sampler: matchup prior, tempered (identity, character) table with
      a data floor, masked and deployment-identity rates, $\lambda$, $D$ and
      $R$, stock counts and starting percents.
- [ ] Replay-start sampler: an index of admitted replay frames tagged off-stage,
      ledge and recovery, with oversampling weights and a fast-forward worker
      pool.
- [ ] No-grad pass: exact KL with the button mask, $\pi_{\text{old}}$ log-probs
      and cached critic features.
- [ ] GAE at decision granularity with terminal and truncation flags.
- [ ] GSPO loss with decision subsampling, minibatch placement across steps and
      the truncated importance weight.
- [ ] $\lambda$ calibration from warmup advantages, written into the resolved
      configuration.
- [ ] Critic epochs on cached features.
- [ ] Varlen masking for resets inside training windows.
- [ ] DDP with equal step counts across ranks.
- [ ] Counter-based RNG for sampling, resets and replay starts.
- [ ] Checkpoint and exact resume, including the pool, with tests on the next
      batch and the next optimizer update.
- [ ] The metrics above.

### Experiment

- [ ] `experiments/065_selfplay_ppo.py` holds the RL configuration and loop and
      imports only `hal`. Code moves into `hal` once its interface is stable.
- [ ] Record the treatment, control and invariant inputs before the first run.
- [ ] Run the pre-RL controls.
- [ ] Pilot stages 1 to 3 on RTX PRO 6000 Blackwell.

### Validation before handoff

- [ ] `ruff format --check`, `ruff check`, `ty check` on the maintained targets,
      and `pytest -m "not integration"`.
- [ ] The integration suite with `HAL_REQUIRE_INTEGRATION=1`. This work changes
      wire format handling, controller input, session stepping and offline-live
      parity.

## Open decisions

- $\lambda_{\max}$, from warmup advantages
- Truncation cap $C$
- Whether $\mu$ can replace $\pi_{\text{old}}$. If the measured gap is tiny, the
  no-grad pass runs only $\pi_{\text{ref}}$ and saves about 0.8 s of the 5.5 s
  update. Exact KL then needs the student's per-factor distributions stored at
  rollout, a few GB per update.
- Replay-start fraction and state oversampling weights
- $\lambda_{\text{gae}}$, if 0.99 per frame proves too noisy
- GSPO $\epsilon$ and the learning rate
- Windows per step, from the measured gradient noise scale
- $A_{\text{local}}$ and $G$, from the drift-versus-utilization sweep
- Critic epochs $K_v$
- When to add the opponent's queue state to the critic
- Opponent snapshots, if cycling appears
- The deployment identity: masked or Master
