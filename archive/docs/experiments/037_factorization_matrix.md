# 037 decoder factorization matrix

Status: complete. All four seed-0 runs finished at step 16,384. Their final checkpoints, W&B records, R2
artifacts, repaired evaluations, and shared-L40S latency results passed the final audit on 2026-08-20.

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

- 25 focused 037 tests;
- 106 tests in the complete relevant 026/036/037, return-target, and Modal-launcher suite;
- Ruff, Python compilation, and `git diff --check`;
- the focused type-error gate, with 33 configured warnings and no errors.

The repository-wide type checker still reports old hard errors in historical experiments 029 and 032. Those
files are outside this change. A repository-wide pytest attempt also exhausted the shared `/tmp` quota while a
different worktree was running Dolphin tests. After removing only this attempt's temporary files and isolating
the relevant suite on CPU, all 106 relevant tests passed.

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

## Production run audit

The four jobs started from implementation commit `f0f4ef29a019df0d85037537c21d6947e0e5a8b3`. Every job trained
from step zero on one L40S. Modal reported no retry or preemption for any training job. The final audit checked
the complete W&B config against the config stored in `final.pt`, loaded every checkpoint, and read all eight
uploaded `metrics.json` files per run.

| Cell | W&B run | Exact R2 prefix | `final.pt` SHA-256 | Bytes | R2 objects |
|---|---|---|---|---:|---:|
| D0 | [98r9smrj](https://wandb.ai/ericyuegu/hal/runs/98r9smrj) | `runs/037-D0-future-independent-group-independent-bc-seed0/` | `50da28a2f45409789a579b1cd4218e9e7d698f95117adfba61614e9fa6f12a23` | 60,685,475 | 459 |
| D1 | [a117chkw](https://wandb.ai/ericyuegu/hal/runs/a117chkw) | `runs/037-D1-future-independent-group-ar-bc-seed0/` | `665a21857debb092cd80cdd3bf60257c9a732df7608fa2d513c34d91ac2bf88e` | 60,685,475 | 426 |
| D2 | [50q39o9j](https://wandb.ai/ericyuegu/hal/runs/50q39o9j) | `runs/037-D2-future-ar-group-independent-bc-seed0/` | `eede17dbaba78f1bdcb0f26171183ab25657e8d4c031c5325477a52989d3f7dc` | 60,685,475 | 435 |
| D3 | [5wfk2esf](https://wandb.ai/ericyuegu/hal/runs/5wfk2esf) | `runs/037-D3-future-ar-group-ar-bc-seed0/` | `0d62a529df3a08a20c170cca325cecf5c3de489e12f15b02496e66d93c01bc5c` | 60,685,475 | 434 |

Every checkpoint contains step 16,384 and a stored H4 configuration. Immutable W&B training histories end at
zero-based `global_step=16,383` with 8,388,608 samples. The final audit restored the conventional summary value
`global_step=16,384` and the same sample count on all four original runs. This matters because the evaluation
repair had accidentally copied stale mutable summary counters: D0 14,336, D1 14,615, D2 14,893, and D3 13,777.
It did not change training history or checkpoints.

All four cells report the same 7,147,504 total and trainable parameters, 7,081,711 policy parameters, 65,793
value parameters, and 7,147,242 parameters receiving gradients. Each processed 8,388,608 examples in 16,384
updates and has the same 46.047 PF `6NT` estimate.

The table below summarizes measured training steps after the first 1,000 updates. Step time and throughput come
from immutable per-step W&B history, not the mutable W&B runtime field.

| Cell | Mean step (s) | p50 step (s) | p95 step (s) | Mean samples/s | Mean samples/wall-s |
|---|---:|---:|---:|---:|---:|
| D0 | 0.3972 | 0.3638 | 0.6061 | 1,375.7 | 1,341.3 |
| D1 | 0.3867 | 0.3592 | 0.6163 | 1,398.7 | 1,365.7 |
| D2 | 0.3901 | 0.3688 | 0.5860 | 1,386.1 | 1,361.5 |
| D3 | 0.4231 | 0.3989 | 0.6045 | 1,254.5 | 1,234.8 |

## Final validation

| Metric | D0 | D1 | D2 | D3 |
|---|---:|---:|---:|---:|
| Joint NLL | 5.424088 | 5.250740 | 3.212632 | 3.188311 |
| Primary NLL | 1.893409 | 1.858838 | 1.181881 | 1.182139 |
| Auxiliary NLL | 3.530679 | 3.391902 | 2.030751 | 2.006172 |
| Group accuracy | 0.854619 | 0.858688 | 0.916706 | 0.917177 |
| Change-event F1 | 0.300930 | 0.324461 | 0.206841 | 0.212560 |
| Exact-frame accuracy | 0.640240 | 0.638982 | 0.599273 | 0.598853 |
| Dense-prefix H1 accuracy | 0.790268 | 0.796980 | 0.796980 | 0.795302 |
| Dense-prefix H2 accuracy | 0.648490 | 0.652685 | 0.661074 | 0.660235 |
| Dense-prefix H4 accuracy | 0.470638 | 0.473993 | 0.500839 | 0.499161 |
| Dense-prefix H6 accuracy | 0.355705 | 0.362416 | 0.395973 | 0.397651 |
| Teacher-forced NLL | 2.875771 | 2.778676 | 1.691203 | 1.676559 |
| Exact rollout-conditioned NLL | +Inf | +Inf | +Inf | +Inf |
| Exact teacher/rollout gap | +Inf | +Inf | +Inf | +Inf |

The exact rollout-conditioned NLL can be infinite for a clear support reason. A sampled trigger can make the
recorded target button illegal under the fixed production mask. The actor then assigns that recorded target
zero probability. This occurred in every cell and is not a numerical training failure. The audit therefore
also measured a separately labeled finite diagnostic before applying the fixed trigger/button support mask:

| Cell | Finite teacher NLL | Finite rollout NLL | Finite gap | Target button support rate |
|---|---:|---:|---:|---:|
| D0 | 7.336620 | 7.336662 | 0.000041 | 0.963926 |
| D1 | 5.950180 | 6.107087 | 0.156908 | 0.962500 |
| D2 | 5.233901 | 7.486019 | 2.252117 | 0.960822 |
| D3 | 4.120091 | 6.589160 | 2.469070 | 0.959396 |

D0's finite gap is effectively zero, as expected when neither learned factorization uses sampled ancestors.
The larger D2 and D3 gaps show exposure from the selected-offset future chain. This finite diagnostic does not
replace the exact masked NLL; both are retained.

The complete offset and group decomposition is:

| Offset | Cell | C-stick | Main stick | Triggers | Buttons | Joint |
|---:|---|---:|---:|---:|---:|---:|
| 1 | D0 | 0.061952 | 0.784511 | 0.111158 | 0.231816 | 1.189438 |
| 1 | D1 | 0.061773 | 0.782538 | 0.110096 | 0.228714 | 1.183121 |
| 1 | D2 | 0.062232 | 0.781652 | 0.109937 | 0.230304 | 1.184124 |
| 1 | D3 | 0.062915 | 0.783326 | 0.109595 | 0.228633 | 1.184469 |
| 2 | D0 | 0.097192 | 1.150741 | 0.166136 | 0.341901 | 1.755970 |
| 2 | D1 | 0.097055 | 1.146067 | 0.160549 | 0.329788 | 1.733460 |
| 2 | D2 | 0.064436 | 0.777377 | 0.110038 | 0.229378 | 1.181230 |
| 2 | D3 | 0.065038 | 0.779650 | 0.109519 | 0.227282 | 1.181488 |
| 3 | D0 | 0.121544 | 1.410723 | 0.211196 | 0.415337 | 2.158800 |
| 3 | D1 | 0.121587 | 1.402238 | 0.199008 | 0.392305 | 2.115138 |
| 3 | D2 | 0.064284 | 0.776563 | 0.110003 | 0.229468 | 1.180318 |
| 3 | D3 | 0.064783 | 0.778997 | 0.109315 | 0.227462 | 1.180557 |
| 4 | D0 | 0.139464 | 1.607252 | 0.252352 | 0.470362 | 2.469430 |
| 4 | D1 | 0.139474 | 1.595471 | 0.232033 | 0.436656 | 2.403634 |
| 4 | D2 | 0.064335 | 0.776882 | 0.110486 | 0.230149 | 1.181851 |
| 4 | D3 | 0.064529 | 0.779642 | 0.109816 | 0.228055 | 1.182042 |
| 5 | D0 | 0.155318 | 1.763565 | 0.290323 | 0.515071 | 2.724277 |
| 5 | D1 | 0.154768 | 1.749231 | 0.261594 | 0.471586 | 2.637179 |
| 5 | D2 | 0.064704 | 0.778421 | 0.111127 | 0.230734 | 1.184986 |
| 5 | D3 | 0.064892 | 0.780958 | 0.110507 | 0.228785 | 1.185142 |
| 6 | D0 | 0.168838 | 1.891099 | 0.324959 | 0.552509 | 2.937404 |
| 6 | D1 | 0.168109 | 1.874653 | 0.289022 | 0.500209 | 2.831993 |
| 6 | D2 | 0.064816 | 0.780875 | 0.112016 | 0.232121 | 1.189828 |
| 6 | D3 | 0.065290 | 0.783567 | 0.111173 | 0.230168 | 1.190198 |
| 9 | D0 | 0.199792 | 2.159706 | 0.407683 | 0.636444 | 3.403625 |
| 9 | D1 | 0.199225 | 2.140570 | 0.355160 | 0.567940 | 3.262896 |
| 9 | D2 | 0.126373 | 1.430646 | 0.214533 | 0.423456 | 2.195007 |
| 9 | D3 | 0.127428 | 1.431712 | 0.205056 | 0.405075 | 2.169271 |
| 12 | D0 | 0.224436 | 2.338153 | 0.462802 | 0.700842 | 3.726233 |
| 12 | D1 | 0.222973 | 2.318981 | 0.403278 | 0.621651 | 3.566884 |
| 12 | D2 | 0.130006 | 1.460717 | 0.219815 | 0.437527 | 2.248065 |
| 12 | D3 | 0.131458 | 1.462376 | 0.210120 | 0.417507 | 2.221462 |
| 16 | D0 | 0.250933 | 2.516811 | 0.521352 | 0.766732 | 4.055828 |
| 16 | D1 | 0.249251 | 2.497628 | 0.456918 | 0.681404 | 3.885201 |
| 16 | D2 | 0.153130 | 1.695624 | 0.268598 | 0.506979 | 2.624331 |
| 16 | D3 | 0.154332 | 1.696592 | 0.251710 | 0.477479 | 2.580114 |
| 20 | D0 | 0.272442 | 2.669510 | 0.574201 | 0.820551 | 4.336705 |
| 20 | D1 | 0.271315 | 2.651656 | 0.508705 | 0.735584 | 4.167259 |
| 20 | D2 | 0.161196 | 1.762419 | 0.285345 | 0.533329 | 2.742289 |
| 20 | D3 | 0.162108 | 1.759518 | 0.266341 | 0.502878 | 2.690845 |

The four group rows sum to the joint NLL at every offset. The reported primary and auxiliary values then use
the same fixed offset means in every cell.

## Evaluation failure and repair

The first built-in evaluation request was wrong: it asked for 32 concurrent Dolphin processes on a 16-CPU
job. All 96 workers at every cell and horizon failed during startup, before one policy decode or match. These
files are preserved at `replays/final_h{H}/metrics.json`. They are failure evidence, not policy results.

| Cell | H | Scheduled boots | Startup crashes | Matches | Decode calls | Wall seconds | Protocol SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| D0 | 1 | 96 | 96 | 0 | 0 | 572.492 | `e055b1c41a34f9e8781dbe1342ce9ecf790f94cfd71a9b74728fd929653f6a46` |
| D0 | 2 | 96 | 96 | 0 | 0 | 578.436 | `4736f8d1cd4354f789483db0100b38db8f71a4491ebb39d56e74eaf5d2140d12` |
| D0 | 4 | 96 | 96 | 0 | 0 | 575.471 | `b1f1dfc934575358601c2f98db7c612fa5ad35d0a2700bcf3c1acb7ab1712aa9` |
| D0 | 6 | 96 | 96 | 0 | 0 | 582.997 | `da904f4619a4bb0cfa249021374260e8fcdcc9090779ca33c34d53c5843cc09b` |
| D1 | 1 | 96 | 96 | 0 | 0 | 573.455 | `af441781da6d44bde8639d71baf1c2b218017cf89b199605c93877ed2565bfd4` |
| D1 | 2 | 96 | 96 | 0 | 0 | 574.896 | `716f3ed7612e015b814734e20aa8a322c00511a582da53ab0e5b40e754f3d512` |
| D1 | 4 | 96 | 96 | 0 | 0 | 583.244 | `390d651c85d3c61ea95b67775ff397938eed5dde0d6a5e36a98c567e8840a898` |
| D1 | 6 | 96 | 96 | 0 | 0 | 574.524 | `d5d0919306c4e0ab9506b52ef2fea0d2f4a7ec0112e065a4bb9ba225bc4f96e9` |
| D2 | 1 | 96 | 96 | 0 | 0 | 570.974 | `3e1a689aed12654d509cb5b98cdba679cf5d8af232e4785e78cb2457d921cf85` |
| D2 | 2 | 96 | 96 | 0 | 0 | 572.072 | `67bcc647e241cdbdb936578b08b9ce72b85fc7e084461db5d4c5e2de24eeaca5` |
| D2 | 4 | 96 | 96 | 0 | 0 | 570.931 | `f9b97a572c628e363a4303713eb50497024f7fd9530a637c2698b0e1319c0342` |
| D2 | 6 | 96 | 96 | 0 | 0 | 570.466 | `ac28fcada7e8e7360b7818b24094577d3a43b82321c009dc2630458b3fef6d02` |
| D3 | 1 | 96 | 96 | 0 | 0 | 590.710 | `f55b82237adc333f0e0df7d1d238542fd5935f438afb3dbdc875c6edc2446bb5` |
| D3 | 2 | 96 | 96 | 0 | 0 | 591.337 | `0908883b2c5d3e63376c0c70d6ea4ee19c7a8035139d247c4ba3d76167cccf16` |
| D3 | 4 | 96 | 96 | 0 | 0 | 574.110 | `b6d26872457bac0b685cc9508c6e1b57651e5032f075e9ddb92f4e67e140e611` |
| D3 | 6 | 96 | 96 | 0 | 0 | 576.925 | `ed77a0c055f445c767e67230ee3871c296852c795bf4106cab998c40da7ef092` |

The repair added a permanent 037 guard: concurrent Dolphin boots cannot exceed the usable CPUs or configured
workers. Production has 16 CPUs and 16 workers, so every repaired evaluation used `max_parallel=16`. Focused
tests also cover a non-power-of-two case: ten CPUs are capped to eight concurrent boots. The repair reused each
unchanged final checkpoint, wrote distinct `replays/final_h{H}_p16-repair/` artifacts, and logged the selected
metrics back to the original W&B run. No model was retrained.

## Selected closed-loop results

These are the audited 16-worker results. The intervals on the four component rates are the protocol's reported
confidence intervals. `net_stock_lcb` and `net_dmg_lcb` are the standard one-sided LCB fields.

| Cell | H | Boots | Matches | Crashes | Stock/min | Stock LCB | Damage/min | Damage LCB | Stocks taken/min (CI) | Stocks lost/min (CI) | Damage dealt/min (CI) | Damage taken/min (CI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| D0 | 1 | 96 | 101 | 0 | 0.0955 | -0.0453 | 35.5039 | 26.3547 | 1.0343 [0.9497, 1.1250] | 0.9389 [0.8587, 1.0188] | 152.1737 [145.9098, 158.6941] | 116.6697 [112.0585, 121.0767] |
| D0 | 2 | 96 | 98 | 0 | -0.1590 | -0.2924 | 21.0396 | 12.0242 | 0.8057 [0.7369, 0.8746] | 0.9647 [0.8746, 1.0547] | 137.3373 [131.3506, 143.0554] | 116.2977 [111.4467, 121.2255] |
| D0 | 4 | 96 | 113 | 0 | -0.7017 | -0.8457 | 2.7712 | -5.8843 | 0.6592 [0.5954, 0.7283] | 1.3609 [1.2578, 1.4681] | 123.0214 [117.3499, 128.4344] | 120.2502 [115.5067, 125.0479] |
| D0 | 6 | 96 | 121 | 0 | -0.9051 | -1.0581 | -19.0119 | -26.6320 | 0.5643 [0.5006, 0.6329] | 1.4694 [1.3510, 1.5835] | 108.0756 [103.4405, 112.9327] | 127.0875 [122.7218, 131.3882] |
| D1 | 1 | 96 | 101 | 0 | 0.1326 | -0.0013 | 40.7768 | 31.4097 | 1.0292 [0.9547, 1.1092] | 0.8965 [0.8166, 0.9813] | 154.7443 [147.9501, 161.5387] | 113.9675 [109.5876, 118.3207] |
| D1 | 2 | 96 | 97 | 0 | 0.0477 | -0.0813 | 36.1129 | 27.3800 | 0.9542 [0.8854, 1.0285] | 0.9065 [0.8218, 0.9860] | 149.9202 [144.5665, 155.7042] | 113.8074 [108.8448, 118.5173] |
| D1 | 4 | 96 | 101 | 0 | -0.3129 | -0.4381 | 13.0490 | 4.8455 | 0.7531 [0.6896, 0.8218] | 1.0661 [0.9858, 1.1519] | 130.4172 [125.0772, 135.3636] | 117.3682 [112.5561, 121.8184] |
| D1 | 6 | 96 | 104 | 0 | -0.4670 | -0.6078 | -1.4068 | -10.1160 | 0.6421 [0.5735, 0.7111] | 1.1091 [1.0075, 1.2055] | 120.4671 [114.4229, 126.6534] | 121.8739 [117.5452, 126.0684] |
| D2 | 1 | 96 | 102 | 0 | -0.1857 | -0.3176 | 29.4782 | 20.5230 | 0.8807 [0.8171, 0.9443] | 1.0664 [0.9757, 1.1629] | 149.0674 [143.4735, 154.6736] | 119.5891 [114.5815, 124.7217] |
| D2 | 2 | 96 | 104 | 0 | -0.0637 | -0.2066 | 28.0807 | 18.6906 | 0.9182 [0.8437, 0.9980] | 0.9819 [0.8918, 1.0782] | 143.1784 [136.5186, 149.6309] | 115.0977 [110.5939, 119.8580] |
| D2 | 4 | 96 | 102 | 0 | -0.1963 | -0.3306 | 18.0595 | 10.3874 | 0.9019 [0.8278, 0.9763] | 1.0982 [1.0126, 1.1840] | 137.8198 [133.0124, 142.7748] | 119.7603 [115.5329, 124.0528] |
| D2 | 6 | 96 | 103 | 0 | -0.2335 | -0.3837 | 15.6934 | 7.1205 | 0.8225 [0.7430, 0.8974] | 1.0560 [0.9545, 1.1580] | 135.8605 [130.3889, 141.4938] | 120.1671 [115.4510, 124.7752] |
| D3 | 1 | 96 | 99 | 0 | 0.0530 | -0.0740 | 36.3244 | 28.1024 | 1.0339 [0.9648, 1.1029] | 0.9809 [0.8960, 1.0606] | 153.4248 [147.8112, 159.1077] | 117.1004 [112.9681, 121.2647] |
| D3 | 2 | 96 | 105 | 0 | 0.0000 | -0.1428 | 31.5488 | 22.4929 | 0.9394 [0.8593, 1.0189] | 0.9394 [0.8492, 1.0300] | 148.2183 [141.5273, 155.1705] | 116.6694 [112.7005, 120.6374] |
| D3 | 4 | 96 | 106 | 0 | -0.0956 | -0.2504 | 24.7452 | 15.6568 | 0.9558 [0.8703, 1.0416] | 1.0513 [0.9550, 1.1527] | 145.0240 [139.1730, 151.1630] | 120.2789 [115.4542, 125.1216] |
| D3 | 6 | 96 | 100 | 0 | -0.0212 | -0.1644 | 25.8506 | 17.7811 | 0.9758 [0.8855, 1.0621] | 0.9971 [0.9172, 1.0819] | 143.3836 [137.4768, 149.0403] | 117.5331 [113.9293, 121.5953] |

The selected protocol digests are:

| Cell | H1 | H2 | H4 | H6 |
|---|---|---|---|---|
| D0 | `f0e74b70eb41d715543966a050108fe5c2764e2902ed2356ccd76113d7d909c2` | `d7d4272c385afec91fb40612dc2dd008868431138cf46146e80ece6f0b22ce2b` | `10cf9e03d93669eb6a8404ec044985dcecb93823aa9b06fa6eab07d6604044b3` | `a39d42df4ac4f7dfa492749adc08425ac9b0921fd79f6afb17f685c0b18e78f7` |
| D1 | `04f78125a174f6977cd74cc06e7bae19ed73d0dcad267ae58fb91ccaf9d78040` | `54849c4f9eca407d8b82a4c73beb0668a05ec5fe1f009ef917c495ab23bcffc1` | `732552552b0b6e5eeb96f26d5a65706d445097e2d68b55648c4d606a362a2e36` | `6e19f76a1cad4e03ccf8ac1390d553673713d782081081150b9928412d9766c8` |
| D2 | `721dfd69f9a2e21b9a2fd357581bb1f5d4eddceab53a1fb31cf1e4ae2c8f6483` | `5e94c9143dcb2be953120cf368f81d3e31b12cafe423a995dc5e56bd4ba5df68` | `4d9cc2c5f73d2dc5b17c1b42a468d82eb89622a414b0a35907218d294ce3ec60` | `6b5aebc6e707b5e16117f5afec7074d7427c0194bff933ffd6a9647e2b05e20d` |
| D3 | `cebfe3ae9df00d4f32de261e400f20ae3536c9cf6a9097c21d6ec3c1bd37ff5e` | `f382f135b48c792e3ed756e559ec2d579b7968695f7806a5acdf785d5b49d065` | `2f2aa01632022d3abe55a10a8410b53b784e8401d10cc1525c561958c08597b1` | `2e33ca91b0f62a5ac2754f141a40a468da3261ebb3a9625c79a40b37ae938500` |

## H4 architecture result

H4 is the planned main comparison:

| Cell | Stock/min | Stock LCB | Damage/min | Damage LCB |
|---|---:|---:|---:|---:|
| D0 | -0.7017 | -0.8457 | 2.7712 | -5.8843 |
| D1 | -0.3129 | -0.4381 | 13.0490 | 4.8455 |
| D2 | -0.1963 | -0.3306 | 18.0595 | 10.3874 |
| D3 | -0.0956 | -0.2504 | 24.7452 | 15.6568 |

Both conditioning choices improve the H4 point estimates and standard LCBs in either background. Future
conditioning has the larger measured effect. Learned group conditioning improves D0 to D1 and D2 to D3. D3
is best on all four H4 columns. The gains overlap: adding either factor first helps more than adding it after
the other factor. This is a one-seed matrix, and the protocol does not provide paired confidence intervals for
factor main effects or the interaction, so this statement describes the measured matrix rather than claiming
a separately significant interaction.

The horizon sweep changes the ranking. D1 is best at H1 and H2, while D3 is best at H4 and H6 by both standard
LCBs. The future-independent cells degrade more sharply as more planned frames are executed. No H8 result was
run because offsets 7 and 8 do not exist in the trained offset set.

## Shared-runtime latency

All rows below were measured sequentially in one audit process on the same NVIDIA L40S
`GPU-231a0dc4-7fa8-f7fa-f4e8-1ed3dcfd0b66`, driver 580.95.05, PyTorch 2.11.0+cu130, CUDA 13.0, BF16,
default compile mode, and inference batch one. Each row uses three warm-ups and 100 measured replans.

| Cell | H | p50 replan (ms) | p95 replan (ms) | Decoder calls | FLOPs/replan | FLOPs/executed frame | ms/executed frame | Label |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| D0 | 1 | 5.132 | 5.309 | 1 | 1.703G | 1.703G | 5.132 | implementation latency |
| D0 | 2 | 6.572 | 6.676 | 2 | 1.704G | 0.852G | 3.286 | implementation latency |
| D0 | 4 | 9.515 | 9.611 | 4 | 1.706G | 0.426G | 2.379 | implementation latency |
| D0 | 6 | 12.345 | 13.636 | 6 | 1.708G | 0.285G | 2.057 | implementation latency |
| D1 | 1 | 5.345 | 5.422 | 1 | 1.703G | 1.703G | 5.345 | implementation latency |
| D1 | 2 | 7.070 | 7.663 | 2 | 1.704G | 0.852G | 3.535 | implementation latency |
| D1 | 4 | 10.031 | 10.753 | 4 | 1.706G | 0.426G | 2.508 | implementation latency |
| D1 | 6 | 13.156 | 14.134 | 6 | 1.708G | 0.285G | 2.193 | implementation latency |
| D2 | 1 | 5.212 | 5.457 | 1 | 1.703G | 1.703G | 5.212 | implementation latency |
| D2 | 2 | 6.839 | 7.870 | 2 | 1.704G | 0.852G | 3.420 | implementation latency |
| D2 | 4 | 9.439 | 9.651 | 4 | 1.706G | 0.426G | 2.360 | implementation latency |
| D2 | 6 | 12.666 | 13.700 | 6 | 1.708G | 0.285G | 2.111 | implementation latency |
| D3 | 1 | 5.362 | 5.562 | 1 | 1.703G | 1.703G | 5.362 | measured architecture latency |
| D3 | 2 | 6.939 | 7.445 | 2 | 1.704G | 0.852G | 3.470 | measured architecture latency |
| D3 | 4 | 10.340 | 10.531 | 4 | 1.706G | 0.426G | 2.585 | measured architecture latency |
| D3 | 6 | 13.416 | 14.253 | 6 | 1.708G | 0.285G | 2.236 | measured architecture latency |

D0, D1, and D2 keep matched sequential modules and work. Their measurements are implementation latency, not
an optimized independent-decoder lower bound.

## Cost and final evidence

The four training jobs, including their failed built-in evaluations, cost $34.4599 on the 2026-08-20 Modal
billing report. The four 16-worker repair jobs cost $9.1218. Three audit attempts cost $1.0098: the first found
the zero-based W&B history convention, the second found the BF16 diagnostic-context bug, and the third passed.
Both failed audit attempts stopped before uploading the report or modifying a W&B run. Total experiment-037
Modal cost was $44.5915.

The final report is `runs/037_factorization_matrix_audit/shared_l40s_audit.json`, 145,389 bytes, SHA-256
`e54d9275134f17d8c42572668b169c40d25b5c9058ee607f5067a2f5ab396810`. It contains every checkpoint hash,
all 32 metric-artifact hashes and parsed payloads, the exact pre-audit W&B counters, finite learned-logit
diagnostics, hardware identity, and shared latency measurements. The same report hash and R2 key are stored on
all four original W&B runs.

After the production fixes, the final local gate passed 25 focused 037 tests, Ruff, the focused type check,
Python compilation, and the complete relevant test suite. Historical source files 016, 019, 024, 026, and 036
were not changed.
