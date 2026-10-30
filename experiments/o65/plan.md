# O65: self-play PPO in melee-sim-light

We fine-tune HAL's Melee policy with KL-regularized self-play PPO, using
melee-sim-light (MSL) for rollouts. We first test the 17M O63 bootstrap30 proxy
on an RTX PRO 6000. The single-GPU loop, native replay parity, paired evaluation
and complete-state resume have local tests. We still need pilot learning
results, O64 production training, two-GPU NCCL validation and deployment latency
measurements. [performance.md](performance.md) records measured runtime.

## Treatment, control and inputs

For treatment, we train joint PPO with an exact conditional KL reward penalty
and its direct gradient. We keep a frozen starting policy as pi_ref. For control,
we evaluate pi_ref under the same episode conditions and final evaluation grid.
We fix the checkpoint, representation, simulator build, game data, reward,
sampling distributions and evaluation seeds before training. We record their
hashes with the Git SHA, resolved settings, seed and software environment.

The proxy pilot follows [pilot-protocol.md](pilot-protocol.md). We run 2,000
updates on four groups of 1,024 matches, with 512 windows per actor step,
16 sampled decisions per window and one actor epoch. Training includes critic
warmup. We monitor existing anchor matches each update, then evaluate only the
final checkpoint in MSL. We select neither a checkpoint nor lambda from the
training metrics. Production gameplay validation also needs Dolphin.

### Development configuration

We load the O63 bootstrap30 `final.pt` after 16,384 updates from:

```text
runs/261017-081127_063_percentile_conditioning_o63v1-bootstrap30-d256-L12-h4-c256-t256x6-ff1024-dual-nonlinear-head-rc1e128-drop0.2-muon-wsd-b150-wmax10-g0.99855__o63-ranked1-bootstrap30/final.pt
```

The proxy supervised offsets 1 through 12, 16, 20, 24 and 28. We use (D=8,R=4)
to execute offsets 9 through 12, and (D=10,R=2) to execute offsets 11 and 12.
Before rollout, we reject each configuration with an unsupervised executed
offset in D+1 through D+R. The proxy trained at delay 2/replan 2, so these runs
check the learning loop under different delays. They do not establish
production gameplay performance.

For O64 production, we sample D from 16 through 21 and R from {2,4}, subject
to D+R<=24. This gives 11 configurations; D=21 permits only R=2. RL uses
offsets 1 through 24, leaving offsets 27 and 30 unused. Production targets an
8xB200 node. We retain the same supervised-offset check when loading O64.

## Episodes and sampling

At each reset, we draw an unordered matchup with weight sqrt(training replay
count), then choose its orientation uniformly. The proxy table includes all
332 observed ranked training pairs, with at least one replay per pair.
We draw a stage uniformly from the six legal stages, with Stadium frozen,
then start both players at four stocks and zero damage.

For each player, we independently draw:

- An identity conditional on character, weighted by its pretraining replay
  count with tempering 1.0 and floor 1. We mask 10% of draws and use deployment
  rate 0. The proxy uses ranked training ids. For O64, we also count professional
  ids through the identity sidecar. Validation and test rows contribute no counts.
- Percentile u from Beta(15,1.2), clipped to [0.1,0.999]. The median is about
  0.94 and 5% of draws fall below 0.8, so RL mostly trains the high-skill
  conditions used at deployment. The student and reference receive the same u
  when scoring a conditional action path.
- log(lambda) uniformly from log(0.01) through log(lambda_max), plus an admitted
  (D,R) pair. Critic warmup calibrates lambda_max within the conditioner bounds.

Each shard has fixed opponent roles. Of its 64 lanes, 16 use student self-play
and 12 use each of four anchor slots: pi_ref in slot 0 and league slots 1-3.
Self-play lanes give the student two training rows; anchor lanes give one, so
about 40% of student windows come from self-play. We spread the anchor lanes
across the shard. The anchor port is random at each episode reset; matchups,
stages and player conditions keep the distributions above. Only student
players contribute training windows. There is no per-episode anchor share or
anchor-policy draw. A 16-lane development shard uses three lanes per anchor
slot and four self-play lanes; a shard with fewer than four lanes uses only
self-play.

We create pi_ref and all three league slot policy objects once. A CPU league
bank holds pi_ref as entry 0 plus frozen student copies, up to 32 entries.
Every 64 updates we bank the student and give it a stock share of 0.5. If the
bank is full, we evict the entry with the highest student stock share,
excluding the first four entries and the new one. The new entry always loads
into a league slot. Two more slot entries come from prioritized fictitious
self-play. We draw entries from counter domain "league" with probability

$$
p_k=0.8\,\frac{\max(0.05,1-s_k)^2}{\sum_m\max(0.05,1-s_m)^2}+\frac{0.2}{n},
$$

where s_k is the student's stock share against entry k and n counts the
candidates. The uniform part returns entries whose shares went stale while they
sat outside the slots. The two draws exclude the new entry and are without
replacement unless only one other entry exists; then both draws return it. A
slot whose entry is a target keeps it. If one entry fills two slots, the slot
with older weights is reassigned first. Remaining targets fill the other slots
in slot order.

The stock share is the stocks the student takes divided by the stocks it takes
and loses. After each update, we sum both counts over the anchor lanes of each
slot, credit them to the slot's entry, and move that entry's EMA toward the
update's share with alpha 0.2. An update without stock changes leaves the EMA
alone. Stocks fall throughout a game, so the share needs no completed match:
a typical match lasts about 160 updates, and a slot changes entries every 64.

We load new weights into a slot between rollouts and leave its episodes
running, as self-play lanes do after every student update. Each rollout starts
on a block boundary and re-encodes its history with the current weights, so no
cached feature from the old entry survives. Actions the old entry committed
still execute. Win counts from completed matches stay diagnostics; one match
can span several entries.

Before critic warmup, the pilot draws an independent integer head start from
0 through 1,200 frames per match using a separate counter domain. Both players
use pi_ref. We delay initial starts so each match has played its offset when
the pool reaches frame 1,280, a rollout boundary. Finished games reset normally.
Head starts produce no training samples or optimizer steps. Warmup retains the
observations and committed queues. We record elapsed time and the resulting
episode-age distribution. Episodes already in progress are excluded from
full-episode critic diagnostics until their next reset.

## Decisions and delayed execution

At observation frame t_j, the player replays its D committed actions at offsets
1 through D, then samples R actions at offsets D+1 through D+R. For each frame,
the decoder samples C-stick, main stick, joint triggers, then buttons. We apply
the same button mask in the student and reference. The first new action
executes at t_j+D+1.

Let E_j contain the decision's executed factors. A terminal match can cut this
set short. We mask decisions with no executed factors from policy and KL losses,
while retaining their values for GAE. Decisions in the final D frames of a live
rollout train on all R frames, because their actions execute in the next
rollout. An intervening match end adds gradient noise without action-dependent
credit. We keep rewards on their observed frames instead of shifting them by D.
A reset discards the queue and starts with neutral actions.

A group's clock determines decision times: R=2 on even frames, R=4 every fourth
frame. We collect 256 frames per active match, ending on a decision boundary.
Dense tick/player records retain observations, sampled actions, queues,
behavior distributions and log-probabilities, critic features and values,
interval duration, policy version, episode ids and masks.

## Objective and losses

We maximize discounted task reward minus lambda times KL to pi_ref:

$$
\max_\theta\;\mathbb E_{\pi_\theta}\left[\sum_t\gamma^t r_t\right]
-\lambda\,\mathrm{KL}(\pi_\theta\Vert\pi_{\mathrm{ref}}),
\qquad \gamma=0.99855.
$$

For decision j, with interval duration Delta_j, we calculate:

$$
\mathrm{KL}_j=\sum_{i\in E_j}
\mathrm{KL}(\pi_\theta(\cdot\mid c_i)\Vert\pi_{\mathrm{ref}}(\cdot\mid c_i)),
\qquad
r_j=\sum_{f=1}^{\Delta_j}\gamma^{f-1}r_{t_j+f}-\lambda_j\mathrm{KL}_j.
$$

Each c_i includes history, the committed prefix, earlier sampled factors and
the same episode u. We score the reference along the sampled student path.
Summed conditional KL estimates the chunk KL. We use HAL's canonical float32
reward: damage dealt minus taken, 120 per stock difference, plus 50 on the
deciding stock. Both reward and KL accrue per executed frame, so lambda keeps
its units of reward per nat across cadences.

### Advantages

We use lambda_gae=1 per frame:

$$
\delta_j=r_j+\gamma^{\Delta_j}b_jV'_{j+1}-V_j,
\qquad
\hat A_j=\delta_j+(\gamma\lambda_{\mathrm{gae}})^{\Delta_j}c_j\hat A_{j+1},
\qquad V_j^{\mathrm{target}}=\hat A_j+V_j.
$$

We set bootstrap mask b_j to zero after a match end. We set trace mask c_j to
zero at an episode boundary or rollout end. At a timeout, we bootstrap from the
last state before reset. At a live rollout end, we compute the final boundary
value. A timeout truncates training; the observation contains no timer.

At the next rollout, we re-encode under updated weights. The preceding rollout's
bootstrap uses 256 context frames, while the next first decision uses 129.
We diagnose the resulting value gap by recomputing a subset. We retain each
actual interval duration when discounting a terminal interval.

For a timeout between decision ticks, we currently evaluate with the configured
D-frame prefix. The committed queue can be longer until the next decision.
We still need to choose a continuation value for that state. The earlier dense
validation also records this open scientific choice at commit `2ff3e77a`.

### Actor

We use stored behavior log-probabilities as pi_old, with no separate student
scoring pass or truncated importance correction:

$$
\rho_j=\exp\sum_{i\in E_j}[\log\pi_\theta(a_i\mid c_i)-\log\pi_{\mathrm{old}}(a_i\mid c_i)],
$$
$$
L_\pi=-\frac1J\sum_j\min(\rho_j\tilde A_j,
\operatorname{clip}(\rho_j,1-\epsilon,1+\epsilon)\tilde A_j).
$$

We subtract per factor before summing the joint log-ratio. We center
advantages per minibatch with the global mean. We divide by a fixed sigma_A:
the standard deviation of critic-warmup advantages, pooled over timing
configurations. Warmup advantages carry no KL cost, so sigma_A does not grow
with KL and weaken the direct KL term below. The batch standard deviation is
logged for diagnosis. The `advantage_scale="batch"` setting restores
per-minibatch normalization and is the only option when warmup is disabled.

Before scaling, we multiply each anchor-lane centered advantage by
clip((2(1-s))^p, 0.25, 1.75), with p=1 and s the student's stock-share EMA
against that slot's entry. Self-play advantages keep weight 1. Opponents the
student already beats contribute less gradient.
Clipping starts at epsilon=0.2. We uniformly select up to 16 eligible decisions
per window, use each window in one step, spread a match's windows across steps.
Padding contributes to neither normalization nor loss.

The detached KL reward credits earlier actions for later states. We also add
the direct conditional KL gradient:

$$
L_{\mathrm{KL}}=\frac1J\sum_j\frac{\lambda_j}{\sigma_A}
\sum_{i\in E_j}\mathrm{KL}(\pi_\theta(\cdot\mid c_i)\Vert\pi_{\mathrm{ref}}(\cdot\mid c_i)).
$$

We train L_pi+L_KL, averaging over trained decisions. We detach reference
probabilities, behavior probabilities, advantage statistics and targets. Only
current student probabilities receive actor gradients. Unit tests compare
this estimator and a sampled log-ratio reward estimator against an enumerated
categorical-chain objective. A later pilot comparison can measure estimator
variance without changing the objective.

The actor uses fresh Muon state, LR 1e-5, zero weight decay, gradient norm cap 1
and a 32-step optimizer warmup. It excludes critic heads and the frozen
percentile head. We reject non-finite executed log-probabilities, KL,
advantages, losses or gradients before an optimizer step.

### Critic and conditioning

We widen the existing conditioner from u alone to u, log(lambda), D and R.
The new columns start at zero; we rescale the episode settings to [-1,1].
The frozen percentile head stays in the checkpoint.

We predict value from the existing value head plus a zero-initialized residual
head. The residual reads detached decoder state immediately before the first
ego draw, ego conditions, opponent conditions and the opponent's committed
queue. Both conditions include u, log(lambda), D and R. We embed each queued
action, zero the padding, append a validity bit and flatten the queue in
firing order into an MLP, so each action keeps its offset. The queue includes
the remaining actions of an opponent's current chunk.

We detach trunk and decoder features so critic fitting cannot move the actor.
The critic uses its own optimizer, LR 3e-4 and four epochs on cached rollout
features. HAL's Muon substitutes zero for missing gradients and continues
momentum updates, so a shared optimizer would move the actor during critic-only
fitting. We measure explained variance on fresh features before fitting and
compare initial values with completed-episode Monte Carlo returns. Truncations
are excluded from that comparison.

During warmup, only critic heads train. We skip reference replay and zero KL
terms while the student still equals pi_ref. Warmup lasts exactly 32 updates,
independent of explained variance and group clocks. We then set lambda_max to
three times the largest per-cadence advantage spread within the conditioner bounds.
We record that value before the first PPO update.

## Context and simulator execution

BC supervises decisions with 129 through 256 context frames. We use 128-frame
blocks: re-encode up to 128 preceding frames at a block start, append each new
frame, then discard the cache at block end. A young episode uses all available
history. A finished match waits for the next block boundary before reset, so
training and deployment begin episode blocks at the same frame.

A per-frame segment id prevents cross-episode attention in the trunk and
decoder history, and resets previous-action features. Tests overwrite every
byte of the preceding episode and compare post-reset outputs. For deployment,
a second cache receives frames throughout the block and replaces the active
cache at the boundary, distributing prefill work across frames.

Each GPU runs a process with a rotating pool of four groups. Only one group
advances per update. We divide it into halves, alternate policy work with
simulator stepping, and divide each half into native shards. Each shard uses a
thread and a native batch. Its page-locked buffers transfer the adapter's
71 columns and canonical reward to a GPU history ring. The GPU constructs
features and delayed action queues. Native workers perform no Torch operations.
CUDA events protect buffer reuse; a failed input copy releases pending workers.

## Update and resume

For each update, we prefill and collect two blocks, compute boundary values,
replay the reference, calculate KL and GAE, fit the critic, then run one PPO
epoch. After updating, we can publish a checkpoint at the decision boundary.
We omit caches because the next rollout starts with a prefill.

Training format v10 retains pool snapshot v4. It records league metadata
(entry ids, updates and weight hashes), stock-share EMAs, the entry in each slot,
the next entry id and the update at which a fresh pool started. Each league
entry's weights are written once as a content-addressed `league_N` shard and
reused by later checkpoints. Resume checks every shard hash and that each slot's
loaded weights match its entry. Exact resume rejects v9 and v8; evaluation can
still read their actors. It records partial episodes from head starts
and validates group progress from the initial pool clock. A checkpoint records model,
optimizers, scheduler, compressed simulator states, the last 129 frame records,
committed queues, group clocks, episode conditions and ids, pool position,
snapshots, metric accumulators and the last 32 anchor win counts. Each rank
writes durable files before publishing an atomic manifest with hashes.
We reject earlier formats and mismatched resume settings.

One trainer process runs the whole pilot. It saves every 32 updates and at the
final update. At each checkpoint boundary, we capture independent CPU tensors,
raw simulator states, RNG states and publication metadata. One background worker serializes,
compresses, hashes, writes and publishes the checkpoint to the Modal Volume and
R2. Only one checkpoint can be in progress. If it is still running at the next
checkpoint boundary, we wait and log the wait. A worker failure fails training
before the next update. Resume uses the latest published checkpoint.

Tick batches retain both ports of each lane. The student has one row per
player; each anchor has both ports of its own lanes. Ownership and activity
masks control queue and behavior writes. Resets, waits, randomized ports and
cadence changes cannot change a batch shape. For every group, buffer and policy,
startup captures (append=1,R=4), (append=2,R=2) and (append=2,R=4), before head
starts. Startup also captures fixed player-count value batches, diagnostic
value batches, one padded reference chunk and the actor microbatch. Reference
chunks reuse one graph regardless of the number of live windows.

After startup we count Torch compiler starts and CUDA capture attempts. Every
update logs both counts to disk, W&B and stdout. A new capture raises an error
that names its graph key. Promotion keeps policy parameters, graph storage and
batch layouts in place. The next block prefill rebuilds history with the new
weights; no retired snapshot objects are needed.

The fixed-slot relaunch has a separate learner migration mode, not exact
resume. Its sole source is the v8 update-256 checkpoint from commit
`3d4ad115881857d0c6c1ba039e41225bd7df3fe8`. It validates source, input hashes,
environment, configuration, all shard hashes and actor/reference agreement.
It restores student weights, both optimizers, scheduler, calibrated lambda_max,
RNGs, actor-step and update counters, warmup advantage moments, and rolling
anchor wins. The saved snapshot becomes league entry 1 with weight update 256
and loads into slot 1; slots 2 and 3 retain pi_ref. The new
pool samples with restored lambda_max, captures graphs and runs the usual
pi_ref head start. Slot 1's head-start episodes truncate before its saved
weights take effect. Simulator state, queues and partial-episode accumulators
are discarded. This deliberate pool reset and fixed opponent distribution are
a new treatment. Disk and new W&B history retain metrics through update 256.

We derive independent counter draws from run seed, domain, logical stream id,
episode, frame, offset and factor. We separate action draws, reset choices,
decision subsampling, window placement and evaluation. We also restore Python,
NumPy, Torch CPU and all CUDA RNG states.

On the same hardware, software and world size, we require exact state
restoration. We compare the next update against two control repetitions after
rewinding the same checkpoint. For each model or optimizer state family,
RMS error may be at most twice control spread plus eight dtype rounding units
at control scale, with minimum scale 1e-3. Integer state and scheduler progress
must match. Maximum errors and next-batch hashes are diagnostic: reduction
noise can change a category draw and later observations. We preserve these
limits when checking resume; bitwise execution is outside scope.

For production DDP, we must exclude frozen parameters, synchronize critic
gradients separately, normalize by global counts and keep equal step counts
across ranks. CPU Gloo tests do not qualify CUDA/NCCL execution.

## Evaluation and risks

The proxy pilot logs each training anchor separately: net stocks and damage
per active minute, completed stockout wins/counts, pooled last-32-update
wins/counts. Reset waits contribute no exposure; truncations contribute no wins.
We preserve these counts through resume. Snapshot metrics use stable slot ids
and record the update of the weights used for that rollout. Loading new
weights into a slot clears its rolling win counts before it next plays. We log
league size, each slot's entry, stock-share EMA and advantage weight, the fixed
and batch advantage scales, and the mean opponent weight. We log KL by cadence and lambda bin,
entropy, clipping, ratio quantiles, gradient norms, parameter movement,
episode-age/u coverage, timing and memory. W&B uses global_step as data and
metric step, with one slash per metric key.

We evaluate the final checkpoint once against pi_ref at u=0.95, lambda=1 and
masked identity. A frozen grid uses 32 weighted training matchup draws, six
stages and both orientations at (D=8,R=4), for 384 MSL matches. Each match
starts at four stocks and zero damage, then continues to a stockout or the
eight-minute timeout. We calculate confidence intervals over paired seed/start
samples. We preserve partial results after failure and consume the held-out
set before play.

- MSL parity on human replays does not establish transfer after RL visits new
  states. Production validation needs event statistics in MSL and Dolphin for
  distinct policies, plus deployment latency and missed-deadline measurements.
- Shared weights can cycle between strategies. Anchor metrics help diagnose
  cycling, though changing episode conditions limit direct comparisons.
- Small lambda and large D can increase KL. We monitor advantage scale and KL
  by cadence and lambda bin. Rare characters need separate checks.
- Damage reward can favor safe damage over a risky stock. An outcome-focused
  reward comparison remains a separate scientific treatment.
- Competitive pressure can reduce identity style. Camping and timed-game
  behavior remain research limits; timeouts currently truncate the task.
- Privileged opponent inputs enter only the critic. We retain ego history
  and exclude the ego's sampled action to keep its baseline independent of
  that draw.

Before a production run, we still need server CPU scaling, two-GPU NCCL,
production metric coverage and gameplay validation. Later tuning may compare
lambda_max, anchor share/refresh, lambda_gae, PPO clipping, learning rate,
windows per step, active pool size, cadence proportions and critic epochs.
Each comparison requires a stated treatment, control and fixed inputs.
