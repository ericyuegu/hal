# 037 decoder factorization matrix

Status: implementation and local validation are complete. The paid launch gate is in progress. No production
result is official until all four seed-0 runs finish at step 16,384 and pass the W&B/R2 artifact audit.

## Question

This experiment separates two kinds of action conditioning in the d256 026/036 actor:

| Cell | Future-action conditioning | Learned controller-group conditioning |
|---|---|---|
| D0 | independent | independent |
| D1 | independent | autoregressive |
| D2 | selected-offset autoregressive | independent |
| D3 | selected-offset autoregressive | autoregressive |

The report calls the left group column **no learned group conditioning**. It does not claim strict
probabilistic independence because every cell keeps the same trigger-to-button legality mask.

## Correct lineage

- 016/M1.3 is an older D0-like model. It is not this experiment's D0 control.
- 019/M3.2b is an older D1-like model. It is not this experiment's D1 control.
- There is no earlier production D2 run.
- 024/M2.1 and 026/M2.2 are both D3-like models.
- 036/A3 is D3 plus MC-AWR and a detached value head.

The audited blog experiment table is the official historical record. This report does not relabel 024 as D2.

## Fixed model and training recipe

Every production cell uses the following configuration:

- causal d256, 8-layer, 4-head observation trunk;
- 128-frame context and the base observation bundle;
- 128-wide, 2-layer, 4-head temporal decoder with FF width 256;
- group-head hidden width 256;
- action and offset embedding widths 16;
- target chunk length 20 and offsets `(1,2,3,4,5,6,9,12,16,20)`;
- group order `c_stick, main_stick, triggers, buttons`;
- the structured learned/semantic controller codec from 036;
- `temporal_state_film=False`;
- batch 512, 16,384 updates, 8,388,608 examples, and seed 0;
- Muon LR 0.02, AdamW LR 0.00085, weight decay 0.01, and 500 warm-up updates;
- BF16 training;
- ranked-anonymized v7 data, replay reservoir sampling, and data order from 036.

`validate_production_config` rejects a matrix launch if one of these fixed values changes. The four `--cell`
commands also set and check the exact W&B run names.

## Information-flow controls

Future independent keeps the full temporal decoder. Every selected offset receives the observation-trunk
state, the observed controller action, and its own offset embedding. The observed action is repeated at all
offsets. Temporal attention uses a diagonal mask, so one offset cannot read another offset. Stepwise inference
still builds the same caches and runs the same modules, but attention selects only the current token and every
offset again receives the observed action. It never receives an earlier sampled future frame.

Future selected-AR is the 036 chain. Offset 1 receives the observed controller action. Every later selected
offset receives the ground-truth action at the previous selected offset in training and the sampled action at
the previous selected offset in inference. Attention is causal. The sparse tail is a chain over selected
offsets, so offset 9 follows offset 6.

Group independent keeps every output head and every group FiLM layer. The FiLM layers run on fixed zero inputs,
so learned logits do not read earlier same-frame group IDs or embeddings. Sampling still follows the fixed group
order because the legality mask needs the sampled trigger ID before it masks button logits.

Group autoregressive is the exact 036 teacher-forced FiLM chain. Later learned logits receive ground-truth
earlier groups in training and sampled earlier groups in inference.

The legality mask is fixed in every cell. A digital L or R shoulder click is valid only when the corresponding
trigger group is at its full value. This changes support in the same way in all four cells.

## Objective

The initial matrix fixes `actor_weighting="uniform"`. The actor uses plain behavioral cloning:

- primary loss: mean joint frame NLL over offsets 1, 2, 3, and 4;
- auxiliary loss: mean joint frame NLL over offsets 5, 6, 9, 12, 16, and 20;
- auxiliary weight: 1.0.

Each joint frame NLL is the sum of the four group NLLs. The primary and auxiliary means are normalized by their
own offset counts, exactly as in 026 and 036.

Every cell keeps the 036 value head. It fits the same Monte-Carlo returns, but its input is always detached from
the policy trunk. The value loss therefore updates only the value head. `actor_weighting="mc_awr"` is supported
for a later D3 BC-versus-AWR run, but it is rejected by the production matrix validator.

No Q learning, IQL, rank weighting, or world-model loss is present.

## Parameter and compute control

All four cells have the same state-dict keys and optimizer partition:

| Count | Value |
|---|---:|
| Total parameters | 7,147,504 |
| Policy parameters | 7,081,711 |
| Value parameters | 65,793 |
| Trainable parameters | 7,147,504 |

The W&B `parameters/receiving_grad` count is recorded after the first backward pass. This is separate from the
trainable count because the fixed base observation bundle leaves some optional v6-only parameters unused.

The audited `6 * N * T` estimate is 46.047 PF per run. It uses all trainable parameters and
`8,388,608 * 128` processed trunk positions. It is the same estimate used in the blog audit.

The decoder estimator reports 5,363,968 nominal MACs per valid trunk position for all four cells at all ten
training offsets. This count treats the fixed-shape SDPA call as dense, so mask choice does not change the
nominal count. Counting only mask-visible attention pairs gives 5,317,888 MACs for D0/D1 and 5,340,928 for
D2/D3. The 23,040-MAC difference is 0.43% of the nominal decoder count; it describes information-visible pairs,
not a claimed kernel saving.

The batch-one inference estimate, including the full 128-frame trunk, is the same across cells:

| Horizon | FLOPs per replan | Amortized FLOPs per executed frame | Temporal decoder calls |
|---|---:|---:|---:|
| H1 | 1.703 GF | 1.703 GF | 1 |
| H2 | 1.704 GF | 0.852 GF | 2 |
| H4 | 1.706 GF | 0.426 GF | 4 |
| H6 | 1.708 GF | 0.285 GF | 6 |

These are multiply-add estimates at two FLOPs per MAC. The final checkpoint benchmark records measured p50 and
p95 replan latency on the training L40S, with compile mode, BF16 precision, batch one, hardware, decoder calls,
and both per-replan and amortized values. Independent-arm latency is labeled implementation latency because the
first implementation preserves the matched sequential work and is not an optimized parallel lower bound.

## Validation and metrics

Focused tests cover:

- exact same-seed D3 actor parameters and outputs against 036;
- compatible 036 actor loading;
- parameter and optimizer equality across cells;
- detached value gradients;
- both future-conditioning paths and the ban on later-action visibility;
- both group-conditioning paths and sampled-ancestor inference;
- fixed legality, group-loss summation, and offset normalization;
- checkpoint flag/output round trips;
- random streams keyed by evaluator slot, match generation, absolute action frame, and group;
- evaluation-only horizon overrides;
- small end-to-end training for D0, D1, D2, and D3.

Validation logs joint, primary, auxiliary, per-offset, and per-group NLL; group accuracy; change-event F1;
exact-frame and dense-prefix accuracy; rollout-conditioned NLL; teacher-forced NLL; and their gap.

The local closed-loop smoke used an untrained d64/L1 test checkpoint, H1, one boot, and a 600-frame budget. It
completed one match with zero crashes, exercised 599 compiled replans, and wrote a protocol-bearing
`metrics.json`, `match_rows.json`, and replay. This checks integration only. Its model size, boot count, and frame
budget do not match production and its game metrics are not architecture evidence.

The final local gate passed:

- 23 focused 037 tests;
- 104 tests in the complete relevant 026/036/037, return-target, and Modal-launcher suite;
- Ruff, Python compilation, and `git diff --check`;
- the focused type-error gate, with 33 configured warnings and no errors.

The repository-wide type checker still reports old hard errors in historical experiments 029 and 032. Those
files are outside this change. A repository-wide pytest attempt also exhausted the shared `/tmp` quota while a
different worktree was running Dolphin tests. After removing only this attempt's temporary files and isolating
the relevant suite on CPU, all 104 relevant tests passed.

## Evaluation protocol

Only the final step-16,384 checkpoint is used for architecture comparison. Each checkpoint is evaluated without
retraining at H1, H2, H4, and H6. H4 is the main result; the other three are the execution-horizon sweep. H8 is
invalid because offsets 7 and 8 were not trained.

Every horizon uses 96 boots, the frozen 026 matchup list and seed, 7,200 frames per boot, level-9 CPU, instant
legal-stage restarts, temperature 1, the same action hygiene, and the current one-sided `net_stock_lcb` and
`net_dmg_lcb` fields. Each `metrics.json` contains its full protocol, checkpoint SHA-256, and a canonical protocol
SHA-256. W&B uses `eval_h1/*`, `eval_h2/*`, `eval_h4/*`, and `eval_h6/*`; H4 is also mirrored to `eval/*`.

## Paid launch gate

Each job requests one 48 GiB L40S, 16 physical CPU cores with a 16-core limit, 64 GiB system memory, and
512 GiB ephemeral SSD. The training config uses exactly 16 data-loader workers. R2 holds checkpoints and replay
evidence; the Modal state volume holds only the small retry record. The attempt timeout is eight hours, while
the expected wall time is 2.25 to 3 hours.

The planning estimate uses the measured 036 training step time, 0.3313 seconds, plus final validation, four
96-boot evaluations, compilation, upload, and latency measurement. It is not a measured 037 runtime. At the
2026-08-20 [Modal prices](https://modal.com/pricing), the request costs about $3.22 per running hour: $1.951 for
the L40S, $0.755 for 16 CPU cores, and $0.511 for 64 GiB memory. The 512 GiB disk request implies 25.6 GiB for
billing, below the explicit memory request, so it adds no further requested-memory charge. Expected cost is
$7.24 to $9.65 per run, or $28.96 to $38.61 for four runs. Preemption, retries, or measured resource use above
the request can increase that amount.

The exact no-rent commands are:

```bash
uv run scripts/launch_modal.py --dry-run --gpu L40S --cpu 16 --cpu-limit 16 --memory-gib 64 --disk-gib 512 --timeout-hours 8 --app-name hal-037-d0 -- uv run experiments/037_factorization_matrix.py --cell D0
uv run scripts/launch_modal.py --dry-run --gpu L40S --cpu 16 --cpu-limit 16 --memory-gib 64 --disk-gib 512 --timeout-hours 8 --app-name hal-037-d1 -- uv run experiments/037_factorization_matrix.py --cell D1
uv run scripts/launch_modal.py --dry-run --gpu L40S --cpu 16 --cpu-limit 16 --memory-gib 64 --disk-gib 512 --timeout-hours 8 --app-name hal-037-d2 -- uv run experiments/037_factorization_matrix.py --cell D2
uv run scripts/launch_modal.py --dry-run --gpu L40S --cpu 16 --cpu-limit 16 --memory-gib 64 --disk-gib 512 --timeout-hours 8 --app-name hal-037-d3 -- uv run experiments/037_factorization_matrix.py --cell D3
```

After all four no-rent audits pass on the clean, pushed implementation commit, the exact paid commands are:

```bash
uv run scripts/launch_modal.py --gpu L40S --cpu 16 --cpu-limit 16 --memory-gib 64 --disk-gib 512 --timeout-hours 8 --app-name hal-037-d0 -- uv run experiments/037_factorization_matrix.py --cell D0
uv run scripts/launch_modal.py --gpu L40S --cpu 16 --cpu-limit 16 --memory-gib 64 --disk-gib 512 --timeout-hours 8 --app-name hal-037-d1 -- uv run experiments/037_factorization_matrix.py --cell D1
uv run scripts/launch_modal.py --gpu L40S --cpu 16 --cpu-limit 16 --memory-gib 64 --disk-gib 512 --timeout-hours 8 --app-name hal-037-d2 -- uv run experiments/037_factorization_matrix.py --cell D2
uv run scripts/launch_modal.py --gpu L40S --cpu 16 --cpu-limit 16 --memory-gib 64 --disk-gib 512 --timeout-hours 8 --app-name hal-037-d3 -- uv run experiments/037_factorization_matrix.py --cell D3
```

## Result audit checklist

For each run, verify the exact W&B name and ID, full config, final step, 8,388,608 examples, parameter counts,
throughput, step time, validation metrics, and all four horizon result namespaces. Then verify the exact R2 run
directory, `final.pt` byte size and SHA-256, and every uploaded `metrics.json`. Each metrics artifact must name
the same checkpoint hash, horizon, factorization flags, matchup digest, seed, boot count, frame budget, restart
rule, precision, compile mode, and hardware bucket used by W&B.

Resolve every W&B/R2 mismatch before adding rows to the official audit. Add one row per actual training run and
one row per actual horizon result. Do not combine descendants. State any parameter, FLOP, data, update, latency,
or protocol mismatch directly. Do not interpret the architecture until all four cells pass this audit.
