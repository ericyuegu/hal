# Historical O50 gameplay intervention plan

## Status

The bounded runner and validation gates are implemented and tested. The
preselected local `short32` direction-only pilot completed successfully for all
three conditions. A fourth, post-hoc control then completed the early/slump
by historical/portable comparison; its results are recorded below. A
research-only policy composes the public O50 runtime to
preserve the four-frame sampling semantics from the two frozen training
commits. Stable HAL code is unchanged.

No Modal gameplay launch was made because uploading the HAL source snapshot
required additional approval. The registry image `ghcr.io/ericyuegu/hal:cuda13`
contains the runtime dependencies but not the `hal` package; the earlier read-only export shell
confirmed `ModuleNotFoundError: hal` without a source mount. The wrapper has a
`--confirm-source-upload` gate and was not invoked.

## Runner

`historical_gameplay.py` prepares these public interfaces:

- `checkpoint_interventions.intervention_specs` and `apply_intervention`;
- `O50Model`, `O50Policy`, and `export_o50_policy`;
- `PolicyBatchAdapter` and `RuntimeConfig`;
- `default_session_cfg` and `sweep_vs_cpu_prior_with_rows`;
- `vs_cpu_metrics` and `matched_vs_cpu_deltas`.

It validates the model-export byte count and SHA-256, original checkpoint
SHA-256 and R2 ETag, W&B run ID, update boundary, experiment ID, data protocol,
MDS schema, raw config, ordered source list, source-selection hash, and all 44
source-manifest hashes. It resolves only the hosted per-source statistics. It
does not load replay shards.

The portable policy requires a rank or connect-code string, but the historical
evaluation supplied player ID 0. `MaskedIdentityO50Model` overrides the public
`context_tokens` method and replaces only `ego_player_id` with an all-zero
tensor. The adapter receives `PLATINUM` to satisfy its public input contract;
that resolved ID never reaches a context token. No global or private HAL member
is patched.

`historical_gameplay_modal.py` defines one H100 job with 32 CPUs, 64 GiB RAM,
10 GiB ephemeral disk, a one-hour cap, and no Modal retry policy. It mounts the
existing `hal-mechinterp-20260911` checkpoint Volume. It returns JSON to the
caller and writes nothing to W&B or R2. Any eventual launch must separately
confirm source upload and persistent Volume writes. Each condition writes to a
separate replay subtree; the runner rejects a non-empty run root, then records
every replay path, byte count, and SHA-256. A local-CUDA run can instead use a
scoped disk-backed research scratch directory.

CUDA execution is an explicit second protocol axis. `modal_h100` rejects any
non-H100 device; `local_cuda` accepts the selected CUDA device without changing
the gameplay protocol. Compiled and eager inference are explicit choices. The
result records the selected mode, actual device name and index, compute
capability, device memory, CUDA and Torch versions, peak allocated memory, and
broker inference latency percentiles. There is no CPU or eager fallback.

## Protocols

All protocols fix delay 2, replan interval 2, context length 256, evaluation
seed 0, CPU level 9, the prefix-stable CPU-compatible training matchup prior,
and zero session-start retries.

| Name | Label | Boots | Frame budget | Purpose |
| --- | --- | ---: | ---: | --- |
| `historical96` | `o50-historical-prior-96boot-f7200-v1` | 96 | 7,200 | Exact historical gameplay protocol; disabled in the bounded Modal entrypoint |
| `pilot32` | `o50-intervention-pilot-32boot-f7200-v1` | 32 | 7,200 | Same per-boot budget with one schedule prefix |
| `short32` | `o50-intervention-short-32boot-f1800-v1` | 32 | 1,800 | Integration and direction-only pilot |

Each run evaluates three conditions in separate sessions: Muon update 24,576,
Muon update 57,344, and one non-identity intervention. The runner rejects an
intervention that duplicates either baseline.

The boot character schedule and policy sampler seeds are held fixed. Dolphin
restart stages and game randomness are not verified common random numbers.
Comparisons therefore use the public character-blocked, boot-clustered
bootstrap and are labeled as blocked comparisons, not paired game-RNG effects.

Every condition must report the requested boot count and zero crashed boots.
The runner writes the incomplete condition record before it raises on a failure.
There is no automatic retry, replacement boot, or silent pooling across protocol
labels.

The prepared direction-only pilot is early update 24,576 with frozen
latent-prefix semantics, slump update 57,344 with the same semantics, and the
same slump weights with `portable_committed_prefix`. The deployment-semantic
treatment was selected from source-protocol inspection, not from offline or
gameplay performance. Its concrete execution axis is `local_cuda`, eager,
`short32`, with replay output under
`/home/ericgu/src/hal/runs/mechinterp-20260911-scratch/gameplay-short32-portable/`.

### Frozen delay and decoded offsets

Both frozen sources sample all four autoregressive offsets every two frames.
`DelayedTruncationPolicy` then appends only `plans[2:4]` to its execution queue.
Thus sampled offsets +1 and +2 are latent ancestors and are discarded; sampled
offsets +3 and +4 are conditioned on that latent prefix and later execute.
The predictor explicitly rejects a committed prefix.

The relevant source blocks are byte-identical between the named runs:

- Muon commit `cdf84f904d322d2a8c342fe3ea75d926e10346ba`, experiment blob
  `2c0337f8c1cbc01a289c89403ecdf259a3d388b0`, lines 2112–2161 and
  2180–2234;
- AdamW commit `d67c68cb36c830bb4100ba8ef1b4cefc452539de`, experiment blob
  `3cc2ccee7e4c28af76f38612a08023e70a544fa0`, lines 2616–2665 and
  2684–2738.

Current public `O50Policy` instead takes the two pending transport actions,
forces them at offsets +1 and +2, and samples only offsets +3 and +4. It then
submits that tail to `ActionTransport`. This is a sensible committed-prefix
runtime, but it is not the frozen protocol.

The current-tree evidence is `hal/inference/o50.py` lines 543–552 and 774–802,
`hal/inference/o50_model.py` lines 640–681, `hal/eval/policy.py` lines 50–89,
and `hal/inference/transport.py` lines 20–36, all at base commit
`52da5b2de1d402b86404fc4d24e382409649e525`.

The research policy prepares an inner public `O50Policy` at delay 0/replan 2 so
its slot RNG samples all four offsets. A decoder facade rotates sampled offsets
+3/+4 to the front; the inner policy emits those two actions. The outer public
`PolicyBatchAdapter` retains delay 2 and therefore executes them after the same
two-frame queue delay as the frozen controller. Applied-action history, stream
resets, batching, and sampling seeds still flow through public HAL contracts.
A fixed-seed test compares its selected action indices with an independent
formula: sample all four unforced offsets, then take `[:, 2:4]`. A separate
four-frame test crosses two replans. It verifies the +3/+4 queue order, context
growth, applied-action history, and unforced latent-prefix decoder arguments.
These tests establish the controller contract; they do not claim bitwise parity
with the frozen compiler and GPU stack.

`portable_committed_prefix` is a separate selectable treatment. It uses the
same slump weights with current public O50 semantics. Weight interventions from
`checkpoint_interventions.py` retain the historical latent-prefix policy. These
labels must not be pooled.

A proper frozen deployed-action likelihood for +3/+4 marginalizes the latent
sampled +1/+2 prefix. Per-horizon teacher NLL for +3/+4 is still useful, but it
conditions on the expert prefix and is not that marginal. A scorer that mixes a
sampled prefix with the original recorded suffix is also not automatically a
proper likelihood.

### Documented next control

A later control can remove within-plan temporal dependence while preserving
each executed action's conditional marginal at a fixed pre-replan context.
Draw two independent full four-frame plans, execute +3 from plan A, and execute
+4 from plan B. One trunk result can be reused and the two decoder rows can be
batched. A second domain-separated `SlotGroupRng` must use the same stream,
reset-generation, and action-group keys. Public `O50Policy` does not expose its
due-state or RNG injection, so a research wrapper would have to track exact
replans and discontinuities. That is tractable in eager mode but must not use a
mutable side channel under `torch.compile`. The control preserves local
conditional marginals, not the distribution of later visited states. It is not
implemented in today's runner.

## Completed direction-only pilot

The immutable raw result is
`/home/ericgu/src/hal/runs/mechinterp-20260911-scratch/gameplay-short32-portable/result.json`
(SHA-256 `03b07b51a6e0910a9c6b6b620f511f500c75466856bdd235d058b618ff738edf`).
All three conditions completed 32 of 32 scheduled boots with zero crashes,
failed workers, timed-out workers, or retries. The 96 finalized replay files
occupy 63,928,191 bytes; the raw JSON records every path, byte count, and
SHA-256.

| Condition | Net stocks/min | Net damage/min | Wall time |
| --- | ---: | ---: | ---: |
| Early 24,576, historical | -0.1342 | +49.71 | 466.35 s |
| Slump 57,344, historical | -0.7379 | +12.86 | 467.80 s |
| Slump 57,344, portable committed prefix | 0.0000 | +32.75 | 477.37 s |

The separate joint-net analysis is
`artifacts/forensics/gameplay_short32_net_analysis.json` (SHA-256
`79544f256f61103f960c7b8ab0bf1cf638d643d8b064a3bb834aeaef832a27c1`).
It uses one seeded table of 10,000 boot-resample indices for the correlated
taken/lost arithmetic. All 32 scheduled boots are matched and rated; both
conditions in each comparison have 14.9067 active minutes, and there are no
zero-active or excluded boots. Equal-boot and pooled-ratio points are therefore
the same, but both estimands remain explicit in the artifact.

Portable committed-prefix minus historical latent-prefix on the identical
slump weights is +0.7379 net stocks/min, directly bootstrapped 95% CI [+0.0671,
+1.4758]. Net damage is +19.89/min, CI [-22.57, +62.77]. Slump minus early is
-0.6038 net stocks/min, CI [-1.2746, +0.0671], and -36.85 net damage/min, CI
[-79.69, +7.35].

The strongest warranted conclusion is that committed-prefix deployment
semantics materially improved short-budget stock performance for the slump
weights in this blocked 32-boot pilot. The damage evidence is inconclusive.
Separate Dolphin sessions do not share game RNG, and this 1,800-frame protocol
neither certifies historical parity nor establishes the cause of the original
weight-training regression.

### Completed post-hoc factorial control

After observing the positive slump portable-minus-historical stock result, one
bounded post-hoc control was selected: evaluate update 24,576 under portable
committed-prefix semantics with the same `short32` schedule, seed 0, eager local
CUDA path, fixtures, player-ID mask, and zero-retry rule. This adds no weight
change and completes the checkpoint (early/slump) by decoder semantics
(historical/portable) 2×2.

The control asked whether the portable benefit is a general decoder-semantics
main effect or is larger at the slump checkpoint. The raw result is
`/home/ericgu/src/hal/runs/mechinterp-20260911-scratch/gameplay-short32-early-portable/result.json`
(SHA-256 `a4937ba0f5de193f22fc91bb053e16475eddbed83b592c32e8f682f13292d6fd`).
It completed all 32 scheduled boots with no crash, retry, or zero-active boot.
The 32 replay files occupy 21,298,685 bytes and are individually hashed in the
raw result. The run took 453.91 seconds.

Early portable achieved 1.0063 stocks taken/min, 95% boot CI [0.5367, 1.4758],
and 0.2683 stocks lost/min, CI [0.0671, 0.5367]. It dealt 212.86 damage/min, CI
[192.50, 233.29], and took 129.66 damage/min, CI [116.05, 142.38]. Its net
points were +0.7379 stocks/min and +83.20 damage/min.

The joint-boot factorial analysis is
`artifacts/forensics/gameplay_short32_factorial_analysis.json` (SHA-256
`ffa78e65932d34ad4ff535a66ae820b7a52a338f9d1fa03a009faedda037ea85`).
It uses the contrast `(slump portable - slump historical) - (early portable -
early historical)`, one seeded table of 10,000 resamples for all four cells and
outcomes, and all 32 scheduled boots. Every cell has exactly 14.9067 active
minutes; no boot is excluded or zero-active. The component interactions are:

| Outcome | Interaction/min | Nominal 95% CI |
| --- | ---: | ---: |
| Stocks taken | -0.2013 | [-0.8721, +0.4696] |
| Stocks lost | -0.0671 | [-0.7379, +0.6038] |
| Damage dealt | +0.37 | [-39.93, +40.70] |
| Damage taken | +13.97 | [-10.94, +39.39] |
| Net stocks | -0.1342 | [-1.2075, +0.8721] |
| Net damage | -13.60 | [-69.56, +39.55] |

The point estimate does not support a slump-specific controller benefit. The
portable stock effect was +0.8721/min at the early checkpoint and +0.7379/min
at the slump checkpoint. The interaction interval is wide and nominal because
the fourth cell was selected after seeing the slump effect. The strongest
warranted result is therefore a general short-protocol benefit from portable
committed-prefix deployment on both checkpoints, not evidence that it repairs
the weight regression. No further gameplay was run after this control.

The exact three-condition runner source is preserved at
`artifacts/forensics/source_snapshots/historical_gameplay.raw_run.9ea508ce.py.txt`
(SHA-256 `9ea508ce8d9e5a1a107227d5bc49221f7f2300e41d10c76da0344bbcb8abf7bc`).

## Assets and runtime estimate

The required existing assets are identified and hash-pinned in `hal.fixtures`:

- ISO: R2 `fixtures/ssbm.ciso`, 1,449,165,376 bytes, SHA-256
  `b7de482eb955c8a96b6746dfa043b69ae7bf6c7c2a09ac382b9da126faa7055c`;
- emulator: ExiAI 0.2.2 GitHub release, 64,035,320-byte source image, SHA-256
  `afca6e00868e11403e1ddb7dae11174f2cac19b09964932835e5e8c5f5439f7f`;
- models: the verified 24,576 and 57,344 exports already on Volume
  `hal-mechinterp-20260911`.

The local fixture resolver verified the existing paths without a download:

- ISO: `/home/ericgu/src/hal/data/emulator/ssbm.ciso`;
- emulator root: `/home/ericgu/src/hal/data/emulator/exiai`;
- emulator entrypoint:
  `/home/ericgu/src/hal/data/emulator/exiai/squashfs-root/AppRun`.

The result payload records these paths and the pinned hosted hashes.

The named production run logs provide the best no-launch estimate. A 96-boot,
7,200-frame evaluation at max parallelism 32 took 980.3 seconds for Muon and
955.5 seconds for AdamW on the historical evaluator, with decode p95 about
27.5 ms. Three 96-boot conditions therefore need about 48 minutes before
allowing for new asset download or compiler-cache misses. A 32-boot full-budget
H100 pilot should take roughly 15–24 minutes for three conditions. The earlier
H100 extrapolation for the 1,800-frame pilot was 6–15 minutes. It does not apply
to the slower local RTX 3060 path measured below.

A bounded eager RTX 3060 measurement used the verified update-24,576 export at
full context length 256. Batch 8 measured 95.89 ms p50, 104.70 ms p95, and
0.93 GB peak allocated memory. Batch 32 measured 398.63 ms p50, 410.84 ms p95,
and 1.95 GB. Filling the public 256-step context took 10.51 and 38.87 seconds,
respectively. The provenance JSON is
`artifacts/forensics/local_cuda_full_context.json` (SHA-256
`42800ab94e6c3cfdd2b1a40db2db31a10382888941c514df1f25b171282b907b`).

At batch 32, 900 replans over the 1,800-frame budget imply about 6.0 minutes of
steady full-context model time per condition, or 18.0 minutes for three. Context
ramp, fixture startup, Dolphin, and result hashing raise the planning range to
22–30 minutes. The completed conditions used 1,411.52 wall-seconds in total,
about 23.5 minutes, plus small orchestration overhead.
