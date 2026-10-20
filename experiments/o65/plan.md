# O65: self-play PPO in melee-sim-light

Status: the single-GPU proxy loop, native replay parity, paired evaluation,
and complete-state resume are implemented and tested locally. RTX 3060 measurements
cover complete updates through 512 active matches and the BF16 decoder ablation.
The RTX PRO 6000 learning pilot, O64 production run, two-GPU NCCL path, and
deployment latency measurements remain open. See [measured results](performance/report.md)
and [validation](validation.md). Numbers marked *est.* below remain estimates;
the worked production example uses a B200 with $D = 16$ and $R = 4$.

O65 fine-tunes the O64 production percentile checkpoint with KL-regularized
self-play PPO. Rollouts run in melee-sim-light (MSL) through HAL's native
adapter. The proxy pilot monitors training anchor matches and evaluates only
the final checkpoint in MSL; it has no Dolphin evaluation.

## Decisions

| Topic | Decision |
| --- | --- |
| Starting policy | The O64 production checkpoint. A frozen copy serves as $\pi_{\text{ref}}$. |
| Objective | Maximize $\mathbb{E}[\text{return}] - \lambda\,\mathrm{KL}(\pi \,\Vert\, \pi_{\text{ref}})$. The per-factor KL enters the reward, and a differentiable KL loss supplies its direct gradient. |
| $\lambda$ | Sampled per player per episode, log-uniform over $[\lambda_{\min}, \lambda_{\max}]$ with $\lambda_{\min} = 0.01$. Critic warmup sets $\lambda_{\max}$, provisionally 10 to 100. |
| $u$ | Independently per player and episode: Beta(5.0, 1.3), clipped to [0.1, 0.999]. The student and reference use the same u. Anchors use the same draw rule. Evaluation fixes u=0.95. The percentile head stays frozen. |
| Policy loss | PPO on the joint ratio of each decision's executed factors, with standard clipping. One epoch ($E = 1$). |
| Policy conditioning | $\log\lambda$, $D$ and $R$ enter new zero-initialized conditioner columns. |
| Critic | `value_head` plus a zero-initialized residual head on the detached decoder state after the queue. The residual head also reads $u$, $\log\lambda$, $D$, and privileged opponent inputs: $u$, $\log\lambda$, $D$ and the committed queue. The critic heads train with their own optimizer. |
| Delay and replanning | $D \in \{16, \dots, 21\}$ and $R \in \{2, 4\}$, sampled per player per episode subject to $D + R \le 24$. $D = 21$ therefore always uses $R = 2$, which leaves 11 configurations. |
| Offsets | RL uses offsets 1 to 24. Offsets 27 and 30 go unused. |
| Context | Block restart: 128-frame blocks with contexts of up to 256 frames. Deployment uses the same scheme. |
| Rollout per update | 256 frames (2 blocks) for each match in the active group. Both values of $R$ divide 256, so every rollout ends on a decision boundary. |
| Discount | $\gamma = 0.99855$ per frame. $\lambda_{\text{gae}} = 0.99$ per frame to start. |
| Episode ends | A match end is terminal. A time-out is a truncation, and the observation carries no timer. |
| Opponents | About 15% of matches pair the student with a frozen anchor: $\pi_{\text{ref}}$ or a recent snapshot. Only the student side trains. |
| Identities | Match the checkpoint pretraining sources. Count identities per character over training replay rows, using the identity sidecar for professional sources. Weight by count (tempering 1.0), floor 1, mask 10%, deployment rate 0. The proxy uses ranked ids only; O64 includes trained professional ids. |
| Evaluation | No evaluation during the proxy pilot. Monitor existing training anchor matches each update; evaluate only the final checkpoint in MSL against pi_ref at u=0.95 on a frozen grid drawn from training matchup weights across six stages and both cadences. No selection grid or Dolphin evaluation. |
| Hardware | An 8×B200 node for production. The pilot runs on RTX PRO 6000 Blackwell. |

## Objective and update rules

$$
\max_\theta\; \mathbb{E}_{\pi_\theta}\Big[\sum_t \gamma^t r_t\Big] - \lambda\,\mathrm{KL}\big(\pi_\theta \,\Vert\, \pi_{\text{ref}}\big),
\qquad \log\lambda \sim \mathcal{U}\big[\log\lambda_{\min},\, \log\lambda_{\max}\big]
$$

The new conditioner columns start at zero, so every $\lambda$ starts at
$\pi_{\text{ref}}$. For a fixed opponent and exact optimization, the optimum is
$\pi_\lambda \propto \pi_{\text{ref}} \exp(Q_\lambda/\lambda)$, with
$Q_\lambda$ the regularized action value, and a smaller $\lambda$ gives a higher
return. Our policy shares weights across $\lambda$ and plays opponents that keep
changing, so we measure the return against KL for each $\lambda$ bin rather
than assume the ordering holds.

### Scale of $\lambda$

$\lambda$ carries units of reward per nat. Near the optimum, the policy
reweights each action by about $\exp(A/\lambda)$. At $\lambda = 1$, an action
worth 3 damage more than its alternatives gains a factor of about 20 over
$\pi_{\text{ref}}$. For the range to include policies close to
$\pi_{\text{ref}}$, $\lambda_{\max}$ has to be comparable to the per-decision
advantage scale. We keep that close end for two reasons:

1. It steadies early training.
2. It leaves a conservative deployment option, in case small $\lambda$ learns
   to exploit $\pi_{\text{ref}}$ more than it improves play.

The KL cost grows quickly at large $\lambda$. At $\lambda = 10$, a KL of 0.1 nat
per frame costs 60 reward units per second, half a stock per second under this
reward.

We calibrate in two steps:

1. Critic warmup measures the spread of per-decision advantages under
   $\pi_{\text{ref}}$, by $D$ and $R$. We set $\lambda_{\max}$ to a few times
   that spread.
2. Early PPO measures return and KL by $\lambda$ bin. Advantage spread includes
   opponent randomness and critic error, so this second measurement checks the
   first.

### Decisions and factors

The decoder samples four categorical factors per frame, in the order C-stick,
main stick, triggers, buttons. Both triggers share a joint categorical. A player
decides every $R$ frames. At decision $j$, made at frame $t_j$:

1. the decoder replays the $D$ committed actions at offsets 1 to $D$;
2. it samples offsets $D+1$ to $D+R$, which are $4R$ factors.

$\mathcal{E}_j$ denotes the factors of decision $j$ executed before the episode
ends: all $4R$ of them, except in the last decisions of a match.

### Transitions and credit

We treat the committed queue as part of the state. Decision $j$ receives the
rewards of the $\Delta_j$ frames after it, with $\Delta_j = R$ except at a match
end. Those rewards mostly follow from earlier decisions. Decision $j$ changes the
queue, and later returns credit it for that. We leave rewards unshifted by $D$.

Each player decides on frames where its group's clock is a multiple of its $R$.
Each update rolls out 256 frames, a multiple of both values of $R$, so no decision
straddles the end of a rollout. Within a rollout:

- a match end cuts the last interval short, and we discount it by
  $\gamma^{\Delta_j}$;
- decisions made in the last $D$ frames of a match never execute, so we mask
  them out of the policy and KL losses; GAE still uses their values;
- decisions in the last $D$ frames of a rollout, in a match that is still
  running, execute in the next rollout, and we train on all $R$ of their
  frames. The match occasionally ends before they execute. Their advantage then
  cannot depend on the unexecuted action, so the gradient stays unbiased and
  only gains noise;
- a reset discards the queue, so the new episode begins with neutral actions.

Dense tick/player tensors hold the observation and decision frame, sampled
indices, committed prefix, behavior log-probs, critic features and values,
actual duration $\Delta_j$, policy version, episode and segment ids, and the
decided, terminal and boundary masks. Each player keeps its slot across resets.

### Reward and KL

$$
\begin{aligned}
\mathrm{KL}_j &= \sum_{i \in \mathcal{E}_j} \mathrm{KL}\big(\pi_\theta(\cdot \mid c_i)\,\Vert\,\pi_{\text{ref}}(\cdot \mid c_i)\big) \\
r_j &= \sum_{f=1}^{\Delta_j} \gamma^{f-1}\, r_{t_j+f} \;-\; \lambda\, \mathrm{KL}_j
\end{aligned}
$$

$r_f$ is the per-frame reward from `hal/training/returns.py`: damage dealt minus
damage taken, 120 per stock difference, and 50 more on the deciding stock. The
context $c_i$ holds the history $h_j$, the forced prefix, the earlier sampled
factors and the same episode $u$ in both policies. Both models apply the same button mask. $\mathrm{KL}_j$ sums
exact per-factor KLs along the sampled path, which makes it an unbiased estimate
of the KL of the whole chunk. The rollout stores the student distributions under
$\pi_{\text{old}}$; the reference pass scores the same actions. $\mathrm{KL}_j$ and the frame rewards both accrue per frame,
so $\lambda$ keeps its meaning across $R$.

### KL gradient

With $\mathrm{KL}_j$ treated as a constant reward, the policy gradient credits
earlier actions for the states they lead to. That covers only part of the
gradient of the objective. Each conditional KL also depends on $\theta$ directly,
and with a constant reward that term averages to zero. We add it back as a
differentiable loss on the trained decisions:

$$
L_{\text{KL}} = \frac{1}{J} \sum_{j} \frac{\lambda_j}{\sigma_A} \sum_{i \in \mathcal{E}_j} \mathrm{KL}\big(\pi_\theta(\cdot \mid c_i)\,\Vert\,\pi_{\text{ref}}(\cdot \mid c_i)\big)
$$

- $\lambda_j$ belongs to the deciding player.
- $\sigma_A$ is the advantage standard deviation used to normalize $\hat A$.
  Dividing by it keeps the ratio between the return term and the KL term the
  same as in the objective.
- The no-grad pass stores $\pi_{\text{ref}}$'s full distributions for the
  trained decisions: 355 values per frame across the four factors, about 45 KB
  per window at $R = 4$.
- The reward term plus $L_{\text{KL}}$ gives the gradient of the regularized
  objective.

As a check, the pilot also runs the simpler estimator: the sampled log-ratio
$\sum_{i \in \mathcal{E}_j} [\log\mu(a_i \mid c_i) - \log\pi_{\text{ref}}(a_i \mid c_i)]$
as the reward penalty, without $L_{\text{KL}}$. Its gradient has the same
expectation with higher variance. A unit test on a small categorical chain
compares both estimators with the gradient of the enumerated objective.

### Advantages

$$
\begin{aligned}
\delta_j &= r_j + \gamma^{\Delta_j}\, b_j\, V'_{j+1} - V_j \\
\hat A_j &= \delta_j + (\gamma\lambda_{\text{gae}})^{\Delta_j}\, c_j\, \hat A_{j+1} \\
V^{\text{target}}_j &= \hat A_j + V_j
\end{aligned}
$$

We keep two masks per decision:

- $b_j$ allows a bootstrap. It is 0 after a match end and 1 everywhere else.
- $c_j$ continues the trace. It is 0 at any episode boundary and at the end of
  the rollout.

$V'_{j+1}$ is the value of the state after decision $j$'s interval:

- the next decision's value, inside an episode;
- the value of the last state before a time-out reset, computed during rollout
  before the reset;
- the value at the rollout's final decision boundary, computed during rollout.

When the match's group next becomes active, its first decision re-encodes the
context under the updated weights. The end-of-rollout value uses a 256-frame
context, and that next decision will receive 129 frames. The pilot measures the gap on a subset by recomputing that value
with the re-encoded context.

$\lambda_{\text{gae}}$ starts at 0.99 per frame. Lifted to the decision grid,
0.95 per frame would give 0.81 at $R = 4$ and credit only about 20 frames, which
leaves the advantage almost entirely to a new critic.

### Policy loss

$$
\begin{aligned}
\rho_j(\theta) &= \exp \sum_{i \in \mathcal{E}_j} \big[\log \pi_\theta(a_i \mid c_i) - \log \pi_{\text{old}}(a_i \mid c_i)\big] \\
L_\pi &= -\frac{1}{J} \sum_{j} \min\big(\rho_j\,\hat A_j,\; \operatorname{clip}(\rho_j,\, 1-\epsilon,\, 1+\epsilon)\,\hat A_j\big)
\end{aligned}
$$

- $\rho_j$ is the joint ratio of the chunk, 8 or 16 factors.
- $\epsilon$ starts at 0.2, and the pilot tunes it to a clip fraction of roughly
  10 to 20%.
- $\pi_{\text{old}} = \mu$: the rollout's stored behavior log-probabilities
  define PPO's old policy. There is no student no-grad scoring pass and no
  truncated importance correction. Rollout stores the student distributions,
  critic features, and values; only the reference replays sampled actions.
- Numerical qualification compares each candidate with an all-FP32 reference
  on identical weights and inputs, with TF32 disabled. For factor log-probs,
  KL, values, and actor/critic gradients, candidate error must be at most
  1.5 times the current default error plus a recorded small absolute floor.
  Statistical parity with commit `aacdfee0` replaces old-trainer bit parity.
- We normalize $\hat A$ per minibatch, using the global mean and standard
  deviation across ranks.
- The policy loss trains 16 decisions per window, sampled uniformly from the
  unmasked decisions. GAE uses every decision.
- Each window enters exactly one gradient step.
- The total loss is $L_\pi + L_{\text{KL}}$, averaged over the global count of
  trained decisions.

### Critic loss

The critic regresses on $V^{\text{target}}$ with squared error, scaled like
O64's value loss. It trains for $K_v$ epochs on features cached during rollout, with no trunk or decoder recompute. Its parameters (`value_head` and
$\text{head}_{\text{res}}$) live in a separate optimizer. HAL's Muon optimizer
fills a missing gradient with zeros and keeps stepping momentum, so critic-only
epochs through the shared optimizer would move the actor.

Cached features go stale once the actor updates. We report the critic's
explained variance on the next rollout's fresh features, before any fitting on
them. The proxy pilot compares initial values against full discounted training
episode returns, excluding truncations.

### Optimizer

- The actor uses BC's optimizer family with fresh state, and the learning rate
  comes from a sweep on the pilot. RL fine-tuning needs a much smaller rate, and
  each step carries about 8 times BC's tokens.
- Muon's orthogonalized updates scale differently from Adam's, so the sweep
  covers that.
- The actor's parameter set excludes the critic heads and the percentile head.
  We freeze the percentile head before wrapping the model in DDP.
- Weight decay 0. Decay pulls weights toward zero, away from $\pi_{\text{ref}}$,
  and so works against the KL term.
- Gradient clipping and a short warmup.

## Reward horizon

| Quantity | Value |
| --- | ---: |
| Reward discount half-life | 478 frames, 7.96 s |
| Weight of a reward 30 s later | 0.073 |
| Weight of a reward 60 s later | 0.0054 |
| GAE trace half-life, $\gamma\lambda_{\text{gae}}$ | about 60 frames |

The critic carries credit beyond the trace. We expect training to improve
punishes, damage avoidance, recovery and edgeguards first, since those pay off
within seconds. Larger $D$ should favor anticipation and safer positioning.

The damage term in `hal/training/returns.py` is part of the task definition. It
can favor safe damage over a risky kill, or discourage a trade worth taking. A
comparison against an outcome-focused reward is an optional later arm.

## Model changes

- **Conditioner.** The `ReturnConditioner` embedding grows from `Linear(1, E)` to
  `Linear(4, E)`. The new columns read $\log\lambda$, $D$ and $R$, each rescaled
  to $[-1, 1]$ over its range, and start at zero. The embedding then matches
  O64's exactly at init, and so does every per-layer projection. Episode $u$ enters
  through BC's existing percentile input, $2u-1$, replacing fixed 0.95.
- **Critic.**
  $V = \text{value\_head}(\text{trunk}) + \text{head}_{\text{res}}(z_{D+1},\ u,\ \log\lambda,\ D,\ u_{\text{opp}},\ \log\lambda_{\text{opp}},\ D_{\text{opp}},\ q_{\text{opp}})$.
  - $z_{D+1}$ is the decoder state at the input to offset $D+1$. It has consumed
    the queue and precedes the first sampled factor, so the baseline excludes
    the ego's sampled action.
  - $q_{\text{opp}}$ encodes the opponent's committed queue. Each queued frame's
    action goes through the codec embeddings plus an offset embedding, and a
    masked mean feeds a small MLP.
  - $q_{\text{opp}}$ spans every committed opponent action, including the rest
    of its current chunk, so its length tells the critic where the opponent
    stands in its cadence. The head also reads $R$ and $R_{\text{opp}}$.
  - Both feature inputs are detached. The output layer starts at zero, so $V$
    starts at `value_head`.
- **Percentile head.** It stays in the checkpoint, frozen.
- **Episode boundaries.** A per-frame segment id drives trunk attention, decoder
  history attention and feature construction, as the next section describes.
- **Inference.** A per-stream trunk KV cache with per-stream lengths, an
  incremental cache of the decoder's history projection, and two compiled decode
  variants, for 2 and 4 sampled offsets, padded across $D$.

## Context, caches and episode boundaries

BC trained with `ctx_pad = 0` and supervised window positions from 128 onward
(`direct_loss_start = 128`), so most of its supervised decisions had 129 to 256
frames of history.

### Block restart

Each stream's cache runs in 128-frame blocks:

1. At a block start, we re-encode up to 128 previous frames of the stream's
   current episode under the current weights, with positions restarting at 0.
2. During the block, we append each frame. The window grows by up to 128 frames
   and drops nothing, so a single causal pass over the window reproduces every
   cached key and value.
3. At the block end, we discard the cache.

A reset restarts the stream's cache at the new episode's first frame. Each
decision then sees between 129 and 256 frames once its episode is 128 frames
old, and the episode's whole history before that. An episode's first decision
comes at the first tick of its clock, with whatever history exists.

The training window for a stream and block covers that stream's cache span
during the block. It may also hold the tail of the previous episode as a
separate segment.

128 is the longest block this checkpoint supports: a longer block would put
mid-episode decisions at contexts below 129 frames. An update always falls on a
block boundary, and the weight change invalidates the cache there anyway.

For netplay, a second cache starts at each block's start, receives every frame
and swaps in at the boundary. That removes the 128-token prefill spike at a cost
of 2 trunk tokens per frame.

### Episode boundaries

A per-frame segment id marks episode boundaries inside a window. We apply it at
every place the previous match could leak in:

- **Trunk attention:** varlen segments through `cu_seqlens`. The current varlen
  path in `hal/models/attention.py` handles an ignored prefix plus a valid
  suffix, so it needs general segments.
- **Decoder history attention:** `hal/models/action_sequence.py` masks history by
  a per-row `ctx_pad`. We replace that with a per-query segment start.
- **Feature construction:** previous-action features at an episode's first frame
  take BC's first-frame values.
- **Value bootstraps:** the $b_j$ and $c_j$ masks above.

An acceptance test overwrites every byte of the previous episode in a window and
checks that all outputs after the reset stay bitwise identical.

## Rollout system

Each GPU runs a single process. `hal/sim/native_adapter.c` already projects
`MslObservation` into HAL's 71 schema columns. It quantizes actions to controller
inputs as the libmelee version HAL depends on does, and it computes the reward in
canonical float32. It writes a 440-byte `HalFrame` per match-frame. O65 builds on
that adapter.

### Pool, groups and cadence

Each GPU holds 4k matches in $G = 4$ groups of 1k. One group steps per update
while the others wait. To suspend a group we stop stepping it.
To resume it we prefill from its frame history, so neither step copies simulator
state.

Each group keeps a decision clock. A stream with $R = 2$ decides on every even
frame of the clock, and a stream with $R = 4$ on every fourth frame. Decisions
therefore happen on two kinds of frames:

- every fourth frame, all streams decide, and the decoder samples up to 4
  offsets, stopping the $R = 2$ streams after 2;
- on the remaining even frames, only the $R = 2$ streams decide, sampling 2
  offsets.

Both decode batches pad across $D$ with per-stream forced lengths.

### Simulator shards

A native batch steps on a single thread. Each half of the active group therefore
splits into shards of about 125 matches, one native batch and one thread per
shard. Each shard has its own page-locked input and output buffers. A CUDA event
on each host-to-device copy gates the shard's next write to that buffer.

A finished match resets in place on its shard's thread. If the pilot finds
resets slow enough to stall a shard, a worker pool can prepare fresh matches in
standby lanes for the shard to swap in.

### Data path

1. Each shard writes its step's `HalFrame` records into its page-locked buffer.
2. We copy them into a GPU ring of 384 frames per match: 128 frames of context
   plus the 256-frame rollout. At about 135k match-frames/s per GPU during
   rollout, that comes to about 60 MB/s host-to-device *est.*
3. The GPU builds HAL's `Context` from the columns, takes rewards from the
   adapter's reward field and runs the policy.
4. The sampled indices go back as 14-float action vectors, and the adapter
   quantizes them into `MslInput`.

The columns carry both players, so each stream's ego view follows HAL's port
convention without a slot swap.

### Stepping loop

The active group splits into halves A and B. While the GPU decides for one half,
the shards of the other half step 2 frames from its queue. A decision made at
frame $t$ first executes at frame $t + D + 1$, well after $t + 2$, so the overlap
is valid. `ctypes.CDLL` releases the GIL during the native call, so shard threads
run in parallel.

Per half and per 2 frames at $D = 16, R = 4$ *est.*:

- GPU: $1\text{k streams} \times 2 \text{ frames} \times 1.1 \text{ GFLOP} \approx 2.2$ TFLOP, about 6.6 ms at 15% decode utilization.
- Simulator: 4 shards of 125 matches, 2 frames each, about 5 ms at an assumed
  50k frames/s per core.

### Episodes and resets

A finished match waits for the next block boundary and then resets in place. Every
episode therefore begins at a block start, as it does in deployment, and its blocks
fall on the same episode frames in training and evaluation. The wait averages 64
frames, under 1% of a match. A fresh reset samples:

- an unordered matchup weighted by sqrt(training replay count), including all 332 observed ranked-anonymized-1 pairs (minimum one replay), then uniform orientation;
- a stage uniformly from the six legal stages, with Stadium frozen;
- each player's identity, conditional on the sampled character;
- $u$, $\lambda$, $D$ and $R$ independently for each player;
- a stock count shared by both players and 0% starting damage for both;
- whether one player is a frozen anchor.

### Frozen anchors

About 15% of matches pair a student player with a frozen anchor: $\pi_{\text{ref}}$
at the sampled episode $u$, or a snapshot from a small pool refreshed every few hundred
updates. Anchors decode with their own weights in a separate batch. Only the
student side contributes training windows, about 8% fewer windows per
match-frame overall.

### Pilot learning monitor

Every update measures the anchor matches already in its rollout: student stocks
taken minus lost per active minute, damage dealt minus taken per active minute,
and completed stockout win rate and count, separately by anchor id with pi_ref
as its own key. A pooled win rate and count cover the last 32 updates. Exclude
truncations from wins and reset waits from exposure. Preserve opponent identity
across resets and the rolling counts across preemption. There is no dedicated
evaluation group during training.

## One update

1. Prefill block 1: for each stream in the active group, encode the previous 128
   frames.
2. Roll out block 1. Each decision stores its record. At time-out resets, we
   compute the value of the last state before the reset.
3. Prefill block 2, roll out its 128 frames, and compute the end-of-rollout
   bootstrap value.
4. Replay the reference over all windows. Combine its distributions with the
   stored student distributions to compute $\mathrm{KL}_j$. PPO uses stored
   behavior log-probs, values, and critic features directly.
5. Compute rewards and GAE over each stream's 256 frames.
6. Run the critic epochs on the cached features.
7. Run one PPO epoch. Each step takes 512 windows per GPU with DDP across the
   node, and we spread a match's windows across steps.
8. Checkpoint every $N$ updates.

Estimated timeline per B200 at $D = 16, R = 4$ with 1k active matches (2k
streams) *est.*:

| Phase | Time |
| --- | ---: |
| Rollout, two blocks with prefills | 1.9 s |
| No-grad pass, student and reference | 1.9 s |
| GAE and critic epochs | 0.1 s |
| PPO, 8 steps | 2.4 s |
| Update total | 6.3 s |

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
| PPO step, per window | 256 trunk + $16\,(D+R)$ decoder | $232 + 0.49 \cdot 16\,(D+R)$ |

Worked example per B200 *est.*:

| Quantity | $D = 16, R = 4$ | $D = 21, R = 2$ |
| --- | ---: | ---: |
| Rollout per update | 1.9 s | 3.4 s |
| No-grad pass | 1.9 s | 3.2 s |
| PPO epoch (8 steps) | 2.4 s | 2.4 s |
| Update total | 6.3 s | 9.2 s |
| Match-frames/s per GPU | 41k | 28k |
| Match-frames/s per node | 325k | 220k |

The sampled mix of configurations falls between the two columns. At large $D$ the no-grad pass dominates, because it runs the decoder
through $D + R$ tokens for every decision of both models.

### Calibration against BC

[The O64 throughput report](../o64/performance/report.md) measured a BC update of
512 windows on a single B200 at 0.534 s for an estimated 340.8 TFLOP, or 28.3%
MFU, with 175.9 GiB peak reserved. That agrees with the 30% training utilization
assumed here. An RL PPO window at $D = 16, R = 4$ with 16 decisions costs about
0.61 of a BC window, since it runs 320 decoder tokens instead of 832. Decode and
prefill utilization remain unmeasured, and the pilot measures them first.

### CPU and host

- Simulator demand comes in bursts. It runs only during rollout, at about 135k
  match-frames/s per GPU, or 3 to 4 shard threads per GPU at an assumed 50k
  frames/s per server core *est.* That makes about 32 cores per node, bound to
  each GPU's socket.
- The pool needs about 1.5 to 2 MB of host RAM per match for the simulator
  arena and its config, or about 64 GB per node at 32k matches.
- Feature building runs on the GPU, which removes the Python per-frame path from
  the loop.

## Memory

Per B200 at 1k active matches *est.*:

| Phase | Fixed | KV cache | Other | Total |
| --- | ---: | ---: | ---: | ---: |
| Rollout | 14 GB | 27 GB | 2 GB | 43 GB |
| No-grad pass | 14 GB | 27 GB | 8 GB | 49 GB |
| PPO step | 14 GB | 27 GB | 89 GB | 130 GB |

- **Fixed:** student weights, gradients and optimizer state, BF16 weights for
  $\pi_{\text{ref}}$ and the anchor snapshots, and workspace.
- **KV cache:** 13.6 MB per stream for 2k streams. It holds the trunk's keys and
  values over 256 frames in 12 layers, plus the decoder's history projection.
- **PPO activations:** about 173 MB per window for 512 windows without
  recomputation. The trunk accounts for about 107 MB and the decoder's
  $16 \times 20$ tokens for about 66 MB, at roughly $34d$ bytes per token per
  layer.
- **Allocator overhead.** The same rule predicts about 157 GB for the O64 BC
  step, which measured 175.9 GiB (189 GB) peak reserved. Applying that 20% to
  the PPO step gives about 156 GB of a B200's 192 GB.
- **Reference and anchors:** no activations. No gradient flows through them.
- **Static allocation.** The KV cache can stay resident at 1k active matches,
  which keeps CUDA graphs available for decode. Beyond about 4k active matches
  per GPU, we would release the cache between phases through the caching
  allocator with `expandable_segments`, or through CUDA virtual memory release
  as in torch_memory_saver.
- **A larger step** of about 600 windows per GPU fits without accumulation, if
  the gradient noise scale calls for it.

## Decorrelation

| Quantity | Value |
| --- | ---: |
| Windows per gradient step (node) | 4,096 (512 × 8 GPUs) |
| Distinct matches per step | about 4,096 |
| Steps per update | about 8 |
| Windows per match per rollout | up to 4 (2 players × 2 blocks), each in a different step |
| Updates between a match's rollouts | 4 ($G$) |
| Steps between a match's rollouts | at least 24 |
| Window repeats | none ($E = 1$) |

Steps per update come to about $A_{\text{local}}/128$, where $A_{\text{local}}$
counts active matches per GPU. At 1k active matches, each update runs about 8
steps before new data arrives, which keeps the drift from $\pi_{\text{old}}$
small. The pilot sweeps $A_{\text{local}}$ from 1k down to 512 and compares the
KL to $\pi_{\text{old}}$ across steps with decode utilization. The gradient
noise scale measured on the pilot decides whether 4,096 windows per step lies
above or below the critical batch size.

## Critic warmup

- **Rollouts.** $\pi_{\text{ref}}$ plays under PPO's sampling: independent clipped Beta episode $u$,
  identities, $D$, $R$, $\lambda$, stock counts and starting percents. At init the
  student equals $\pi_{\text{ref}}$ for every $\lambda$, so these rollouts are
  on-policy for all of them.
- **Training.** Only the critic heads train, on detached features and with
  their own optimizer. Warmup skips the KL terms, which are zero.
- **$\lambda$ dependence waits for PPO.** During warmup $\lambda$ changes no
  behavior, so the critic cannot yet learn how the return depends on either
  player's $\lambda$. Those weights stay near zero until PPO separates the
  policies, which is the correct value until then. $D$, $R$ and the queues do
  change behavior, and the critic learns their effects during warmup.
- **$\lambda$ calibration.** Warmup measures the spread of per-decision
  advantages by $D$ and $R$, and we set $\lambda_{\max}$ from it before PPO
  starts. The resolved configuration records the value.
- **Phase mixing.** Fresh matches all start at frame −123 together. Warmup
  continues until each match has advanced about one match length. A match
  takes about 43 rollouts of 256 frames and rolls out every $G$-th update, so
  that comes to $43 \times G = 172$ updates, or about 5 minutes on the node at
  rollout-only cost *est.* Starting damage is always zero. Desynchronized episode ends supply phase diversity;
  log episode-age distributions each update.
- **Ending.** Warmup ends once the critic's explained variance on fresh rollouts
  plateaus and the phases have mixed.

## Checkpoints and resume

We checkpoint at update boundaries. Every rollout ends on a decision boundary and
every update starts with a prefill, so no transition is pending and the KV cache
stays out of the checkpoint. Each rank writes a shard holding:

1. the model, both optimizers and the scheduler (rank 0 only);
2. every pool match's simulator state from the adapter's save path, compressed;
3. each match's last 129 `HalFrame` records (the current observation and up to
   128 prior frames), both players' committed queues and the group's decision clock;
4. episode metadata: $u$, $\lambda$, $D$, $R$, identities, characters, anchor
   assignments, episode counters and stable logical stream ids;
5. the pool schedule position, anchor pool, episode metric accumulators and the
   last 32 updates of per-anchor wins and completed-match counts.

After a barrier confirms every shard is durable, rank 0 writes a manifest with
the shard hashes and publishes it by atomic rename. A resume reads only
checkpoints with a manifest.

`hal_native_restore` re-projects the whole batch for every lane it restores, so
restoring a full batch lane by lane costs quadratic work. The adapter needs a
batch restore that projects once.

Training format v7 stores dense pool format v4, the seed-derived device
counter key and rolling per-anchor win counts. Earlier training and pool formats are rejected. Snapshot tensors
own their storage. Restore clears all execution caches before the next prefill.
The verifier uses the same tick, reference and actor graph paths and execution
mode as training. It checks exact restoration of model, both optimizers,
scheduler, pool and simulator state, sampler cursor, counter key, progress and
every RNG state. It repeats the next update twice as controls, rewinding the
same boundary between them, then compares a third resumed update. For the
model and each optimizer state family, RMS error may reach twice the control spread plus eight dtype rounding units at the
control scale (minimum scale 1e-3). Maximum and per-tensor errors remain diagnostic because
reduction noise can move between tensors. Integer update state and scheduler
progress must match exactly. Report v5 records the errors and tolerances.

### Random numbers

Every draw comes from a counter-based generator keyed by (run seed, domain,
logical stream id, episode, frame, offset, factor). The domains separate action
sampling, reset choices, decision subsampling, minibatch
placement, critic shuffling and evaluation. Logical stream ids survive
rebatching and restore, so the generator has no state to save.

### Resume scope

Resume restores the complete training state. On the same hardware, software
and world size:

- model, optimizer, scheduler, simulator, sampler cursor and every RNG state
  restore exactly;
- the next optimizer update matches within tolerance measured from two control
  repetitions of that update. Atomic adds and other nondeterministic reductions
  are permitted in training and verification.

Next-batch hashes are recorded as diagnostics. They need not match: numerical
noise can flip a sampled category and then change later observations, windows
and decision subsampling. No NaN or infinity is accepted in an executed
log-probability, KL, advantage, loss or gradient. Non-finite training values
raise an error; they are never masked or clamped away.

Bitwise identical execution is out of scope.

## Distributed training

- Each GPU runs a DDP rank over the actor. The critic heads sync gradients
  through a separate all-reduce after each critic epoch.
- We freeze the percentile head before wrapping, so DDP skips it.
- Losses divide by global counts of trained decisions and critic targets, which
  we all-reduce before each step.
- Ranks run equal step counts. We log each rank's rollout time to find
  stragglers before the synchronized phases.
- The pilot tests the real CUDA and NCCL path on two GPUs. A Gloo CPU test
  checks the sync logic only.

## Deployment

In deployment, $D$ counts every frame from the observed frame to the first
frame a new action can execute: input delay, transport and the compute
allowance together. The forced prefix holds every action already committed for
those frames. The netplay runner computes $D$ this way from its schedule and
uses a policy trained at that $D$.

Deployment needs its own measurements: batch-1 latency with the two-cache
handoff, and the rate of missed deadlines at each $D$ and $R$.

## Evaluation and run protocol

### Simulator validation before RL

- Bitwise parity of the adapter's columns and rewards against
  `hal/data/extract.py` and `hal/training/returns.py` on admitted replays.
- Free-running play from a fresh match, checked for determinism across save and
  restore.
- Event statistics in MSL and Dolphin for the same pairs of distinct policies:
  deaths, ledge grabs, techs, match length and stock differential. A symmetric
  $\pi_{\text{ref}}$ ditto can stay near 50% in a wrong simulator, so the
  comparison uses distinct opponents.
- $\pi_{\text{ref}}$ at $u^* = 0.95$ at each $D$ and $R$ in Dolphin, with block
  restart against the current sliding context.

### During training

- The proxy pilot runs no evaluations or selection grid. It publishes complete
  checkpoints every 32 updates and at the final update. There is no forced kill
  at 256 or scheduled resume verification at 288; real preemptions still resume.
- Progress signals come from existing training anchor matches, by anchor id.
  Current-versus-current win rate stays near 50% by construction.
- Evaluate only the final checkpoint once in MSL against pi_ref at u=0.95,
  lambda 1, both cadences, and masked identity. Freeze 32 distinct weighted
  matchup draws (seed 65095), cross all six stages and both orientations, and
  use held-out seed 10017: 768 matches. No Dolphin evaluation is in this pilot.
- Confidence intervals treat each paired seed/start as a sample. Training
  anchor results are diagnostic and do not select a checkpoint or lambda.

### First run

- **Treatment:** PPO from $\pi_{\text{ref}}$ with the decisions above.
- **Control:** $\pi_{\text{ref}}$ under the same episode $u$ distribution, $(D, R)$ configurations,
  identity sampling, matchup prior, stage set and evaluation protocol.
- **Invariant inputs:** checkpoint, episode $u$ distribution, identity sampling, matchup prior,
  stage set, evaluation seeds, simulator build and data manifest.
- **Final report:** `net_stock_lcb` and head-to-head win rate against pi_ref in
  MSL at each cadence, at fixed lambda 1. The pilot selects neither a checkpoint
  nor lambda. Offline metrics serve only for diagnosis.
- **Records:** Git SHA, resolved configuration, seed, environment, simulator and
  adapter hashes, data manifests and checkpoint hashes.

## Pilot stages

Each stage needs its evidence before the next begins.

| Stage | Evidence |
| --- | --- |
| Math checks | KL gradients against an enumerated objective, joint ratios, delayed-action credit on a toy delay task, terminal and truncation masks |
| State and inference | No attention across episodes, cached against full-window log-probs, queue and reset parity |
| Debug loop | The O63 proxy under the development configuration, end to end on one RTX PRO 6000: loop, logging, checkpoint and resume, and a learning signal against $\pi_{\text{ref}}$ |
| Minimal learning | The 248M checkpoint on one matchup and stage, fixed $D$, $R$ and $\lambda$, joint PPO, separate critic optimizer |
| Gameplay | Proxy pilot: final MSL improvement against pi_ref; production gameplay validation remains open |
| Scaling | All 11 $(D, R)$ configurations mixed in each group, real two-GPU NCCL training, full-pool checkpoint and restore, complete-path throughput |
| Expanded run | $\lambda$ range, identities and anchors, each added with its own comparison |

### Development configuration

We build and debug the loop on the O63 bootstrap30 proxy before the O64
checkpoint enters. We load `final.pt` from the R2 key
`runs/261017-081127_063_percentile_conditioning_o63v1-bootstrap30-d256-L12-h4-c256-t256x6-ff1024-dual-nonlinear-head-rc1e128-drop0.2-muon-wsd-b150-wmax10-g0.99855__o63-ranked1-bootstrap30/final.pt`
after 16,384 updates and run it on a single RTX PRO 6000.

The proxy's decoder trained only on offsets 1 to 12, 16, 20, 24 and 28, so we
keep every executed offset among those the proxy saw.

- $D = 8$ with $R = 4$ executes offsets 9 to 12.
- $D = 10$ with $R = 2$ executes offsets 11 and 12, and it exercises mixed
  cadence within a group.

The configuration loader reads the checkpoint's supervised offsets and rejects
any $(D, R)$ whose executed offsets $D + 1, \dots, D + R$ fall outside them.
The O64 checkpoint passes the production range under the same check.

The proxy trained for delay 2 and replan 2, so delay 8 sits somewhat out of its
distribution. The proxy runs show whether the loop learns. They say nothing
about production gameplay.

## Bigstep pilot scientific changes

The first real proxy pilot uses 512 windows per actor step (previous pilot 32),
16 decisions per window, one epoch, actor LR 1e-5 and 32 optimizer warmup steps.
Critic settings are unchanged. The user chose 2,000 total updates (previous 512),
including critic warmup, on four groups of 1,024 matches. See
[pilot-protocol.md](pilot-protocol.md) for control inputs, launch, costs and gates.

Episode identities now follow the checkpoint's own training replay counts,
without deployment/Master oversampling; the proxy has ranked training only.
`build_o65_sampler.py` uses `cfg.source_names` and the identity sidecar. No mix
identity check rejects an intentionally changed valid table. Percentile changes
from fixed 0.95 to independent clipped Beta(5,1.3) per player, in both student
and reference; the residual critic receives both players' percentiles. Evaluation
keeps 0.95. Starts change from mixed 0/30/60/90% to zero. Training stages change
from FD only to all six legal stages uniformly, and matchups change from Fox/Falco
to all 332 observed ranked train pairs weighted by sqrt(count), with uniform orientation. These are
scientific treatments, not systems refactors. Episode-age and u-bin diagnostics
check the resulting coverage. Training/evaluation artifact formats are versioned
for the added episode state and critic inputs.

## Metrics

| Group | Metrics |
| --- | --- |
| Policy | KL per frame to $\pi_{\text{ref}}$ by $D$, $R$ and $\lambda$ bin, KL to $\pi_{\text{old}}$ across the steps of an update, clip fraction, ratio quantiles, entropy per controller factor, $L_{\text{KL}}$ |
| Return and KL | return against KL by $\lambda$ bin, cumulative KL cost per match |
| Critic | explained variance on fresh features before fitting, by $R$ and by position in the block; value against Monte Carlo returns; value loss; advantage mean and standard deviation by $D$ and $R$; end-of-rollout bootstrap gap |
| Numerics | joint log-ratio against stored behavior across actor steps; common-input errors against the all-FP32 reference |
| Gameplay | Each update: net stocks/minute and net damage/minute over active training exposure, completed-match win rate/count and pooled last-32-update win rate/count per anchor id (pi_ref separate). Final checkpoint: MSL paired net stocks, win rate and match length against pi_ref at u=0.95, lambda 1 and both cadences; no training or Dolphin evaluation. |
| Systems | time per phase, decode and training utilization, decode batch sizes, simulator frames/s per shard, reset wait time, host-to-device bytes, peak HBM per phase, per-rank rollout time |
| Data | identity, character, stage and matchup counts, anchor share, time-outs, resets per update |

W&B logging records `global_step` as data with `define_metric` and keeps keys
one level deep.

## Risks

- **Simulator exploits.** MSL validation covers admitted human replays. RL
  policies will visit states outside that set. The throughput-first proxy pilot
  omits Dolphin evaluation, so it cannot establish simulator transfer.
- **Non-transitive cycling.** Shared weights move together across $\lambda$,
  $D$ and identities, so that diversity alone cannot preserve an old
  counterstrategy. Per-anchor training metrics can expose cycling, but their
  changing sampling conditions limit that evidence.
- **KL growth at small $\lambda$ and large $D$.** $\pi_{\text{ref}}$ is weakest
  at large $D$, so RL moves furthest there. The advantage scale may differ by
  $D$, so warmup measures it per $D$.
- **Reward shaping.** The damage term biases play toward safe damage. The
  outcome-focused arm tests this if gameplay suggests it.
- **Identity style.** At low $\lambda$, every identity faces the same
  competitive pressure, and recognizable human styles may fade.
- **Camping.** Deferred. If it appears, an asymmetric reward change addresses
  it. A timer feature would make the observation non-Markovian.
- **Time-outs.** A tournament clock expiring decides a real game. We treat it as
  a truncation, which leaves timed endings outside the task.
- **Rare characters.** The matchup prior keeps them rare, and they also have the
  weakest $\pi_{\text{ref}}$. Their KL and gameplay need separate monitoring.
- **Privileged critic.** It keeps the ego's history as input, which keeps its
  baseline unbiased in a partially observed game (Baisero and Amato, 2022).

## TODOs

### melee-sim-light

- [x] Expose the recorded-replay path publicly: the initial seed and per-frame
      RNG injection from recorded replays. HAL's parity tests use it.
- [ ] Benchmark on the B200 node's server CPU. The published baseline uses a
      desktop V-cache core.
- [ ] Measure thread scaling across separate batches stepped in parallel.
- [x] Measure save size, add compression, and test save and restore across
      batches for determinism.
- [x] Confirm that Stadium always runs frozen, and document the character-id
      space.

### HAL: simulator boundary (`hal/sim`, torch-free)

- [x] Integration tests of `native_adapter.c` against real MSL, plus stub tests
      for invalid inputs and cleanup.
- [x] Parity test: replay through MSL and the adapter's 71 columns against
      `hal/data/extract.py`, bitwise on admitted replays.
- [x] Parity test: the adapter's reward against `hal/training/returns.py` on the
      same replays.
- [x] Parity test: the adapter's quantized controller inputs against the Dolphin
      input path for the same action stream.
- [x] Batch restore that projects once, replacing per-lane restores of a full
      batch.
- [x] Adapter output into caller-provided page-locked buffers. The loop steps
      each frame to retain intermediate rewards; multi-frame calls remain optional.
- [x] Shards: one native batch and one thread per shard, with separate buffers,
      bound to the GPU's socket.

### HAL: model and inference

- [x] Widen the conditioner embedding to four inputs with zero columns. A parity
      test confirms identical outputs to O64 at init for every $\lambda$, $D$
      and $R$.
- [x] Critic residual head with the opponent-queue encoder and a
      zero-initialized output. A parity test confirms $V$ equals `value_head` at
      init.
- [x] Per-frame segment ids in trunk varlen attention, decoder history attention
      and feature construction, with the byte-overwrite test.
- [x] GPU `Context` builder from adapter columns, with a parity test against the
      current Python feature path.
- [x] Block-restart inference: a per-stream trunk KV
      cache with `cache_seqlens` and an incremental decoder history projection.
      A parity test compares log-probs with the full-window pass.
- [x] Two compiled decode variants, for 2 and 4 sampled offsets, padded across $D$.
- [x] Block restart in Dolphin evaluation and netplay, with the second-cache
      handoff.
- [ ] Netplay: subtract transport and compute allowance from $D$, and measure
      batch-1 latency and missed deadlines.

### HAL: RL loop (joins model and simulation)

- [x] Rollout engine: pool rotation, group decision clocks for mixed $R$, half
      alternation over shards, resets, frozen anchors and evaluation groups.
- [x] Episode sampler: matchup prior, tempered (identity, character) table with
      a data floor, masked and deployment-identity rates, $\lambda$, $D$, $R$,
      stock counts, starting percents and anchor assignment.
- [x] Decision records with all fields listed above.
- [x] No-grad pass: $\mathrm{KL}_j$ with the button mask, $\pi_{\text{old}}$
      log-probs, $\pi_{\text{ref}}$ distributions for trained decisions, cached
      critic features.
- [x] GAE with $\Delta_j$, bootstrap masks and trace masks, plus time-out and
      end-of-rollout bootstrap values.
- [x] Joint-ratio PPO on stored behavior log-probs, $L_{\text{KL}}$, decision
      subsampling and masking of unexecuted decisions.
- [x] Separate critic optimizer and critic epochs on cached features.
- [x] $\lambda$ calibration from warmup advantages.
- [ ] DDP with frozen parameters excluded, a separate critic all-reduce, and
      global loss normalization.
- [x] Counter-based RNG with separate domains and logical stream ids.
- [x] Sharded checkpoints with an atomic manifest, and resume tests on the next
      batch and the next optimizer update.
- [x] Anchor snapshot pool.
- [ ] Complete the production metric set. The proxy logs phase timing, PPO,
      KL, critic, identity, and simulator diagnostics.

### Evaluation

- [x] Periodic evaluator implemented; disabled for the lean proxy pilot.
- [x] Final-only held-out pilot grid drawn from the training weights; no selection set.
- [ ] MSL against Dolphin event statistics for distinct policy pairs.

### Experiment

- [x] `experiments/065_selfplay_ppo.py` holds the RL configuration and loop and
      imports only `hal`. Code moves into `hal` once its interface is stable.
- [x] Record the treatment, control and invariant inputs before the first run.
- [ ] Run the simulator validation before RL.
- [ ] Run the pilot stages in order.

### Validation before handoff

- [ ] `ruff format --check`, `ruff check`, `ty check` on the maintained targets,
      and `pytest -m "not integration"`.
- [ ] The integration suite with `HAL_REQUIRE_INTEGRATION=1`, because this work
      changes controller input, session stepping and offline-live parity.

## Open decisions

- $\lambda_{\max}$, from warmup advantages and early return-KL curves
- Anchor share and snapshot refresh interval
- $\lambda_{\text{gae}}$, if 0.99 per frame proves too noisy
- PPO $\epsilon$ and the learning rate
- Windows per step, from the measured gradient noise scale
- $A_{\text{local}}$, from the drift-versus-utilization sweep
- The share of $R = 4$ among configurations with $D \le 20$
- Critic epochs $K_v$
- The deployment identity: masked or Master
- The outcome-focused reward arm
