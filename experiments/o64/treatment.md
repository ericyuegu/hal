# O64: production percentile conditioning

Status: production stable phase launched from `7955ec17`. Its recorded startup
is compiling and materializing the training cache. See [launch.md](launch.md).
Implementation, local validation, corpus audit and full-batch B200 qualification
passed. Gameplay and steady full-corpus training results are pending.

Train the O59 production architecture with the O63 bootstrap30 treatment.
This scales the strongest observed proxy setting, bootstrap30/u95, and changes
the supervised action offsets and loss weights as requested. It is not a
controlled test of model size alone. O63 and its original primary result are
preserved in commit `87c03ce4` and its recorded source history.

## Model and objective

The trunk has width 1024, 12 layers, and 16 attention heads. The temporal decoder
has width 1024, six layers, 16 attention heads, and MLP width 4096. Context length
is 256. The action heads retain their O59 production widths and action codec.
The training chunk is 30 frames, with 26 supervised offsets: 1 through 24,
then 27 and 30. The model has **247,944,158 parameters**, including the detached
percentile head. The offset embedding adds 32 parameters relative to O63's
production configuration.

Policy loss coefficients sum to one:

| Offset | Loss coefficient |
| --- | ---: |
| 1 | 0.33 |
| 2 through 24 | `0.664 / (offset * sum(1/k for k in range(2, 25)))` |
| 27 | 0.004 |
| 30 | 0.002 |

The first six offsets receive about 67.6835% of policy loss weight. The remaining
offsets receive about 32.3165%. AWR still applies only to offsets 1 through 6;
the far-offset objective remains unweighted by AWR. This replaces O63's 85%/15%
allocation. The next-frame share is exactly the requested 33%.

The bootstrap target remains
`returns[t] + gamma**30 * (V(s[t+30]) - returns[t+30])`.
Both states must be eligible and inside the context. The final 30 of 128 suffix
positions have no label, independent of the action offsets. The head has 65
symlog bins over [-1000, 1000], HL-Gauss sigma 0.75 bin widths, and loss weight
1.0. It remains trainable; calibration drift is reported without automatic
freezing. The labeler seed remains `seed ^ 0x063AB`.

## Fixed training inputs

- Full 44-source policy-world-v8 corpus, with the original selection, manifest,
  statistics, identity, and schema hash checks. No source subset.
- Seed 0, batch 512, BF16, Muon/AdamW and the O59 learning-rate schedule.
- Gamma 0.99855, stock reward 120, final-stock reward 50, damage weight 1.
- AWR beta 150 and cap 10; AWR and conditioning start at update 4,097.
- Conditioning dropout 20%; identity dropout 10%.
- Eight distinct windows per replay generation, 131,072 replay slots, phase
  blocks of 25 batches, and at least 200 batches between replay reuse.
- Context 256 plus target chunk 30 requires at least **293 actual frames** to
  select eight distinct windows. The corpus audit checks actual frame counts,
  split row identities, index counts, and manifest hashes without MDS downloads.
- The stable run stops explicitly at update **98,304**. A separate decay
  continuation uses an immutable parent checkpoint and recorded lineage.
- Checkpoints every 2,048 updates; offline validation every 4,096. Validation
  uses the same 2,048-sample cohort in batches of 128. NLL is diagnostic and
  does not select the policy. Validation decodes only the final context prefix
  that it scores; full-prefix reference tests cover the unchanged NLL. It
  retains full-suffix value/percentile computation and labeler draws. Earlier
  full-suffix decoder implementations exceeded memory with resident training
  CUDA graphs at both batch 128 and batch 32. Training is unchanged.

## Gameplay and hardware

Use u95 for automatic 96-matchup gameplay evaluations every 8,192 updates and
at completion. The final checkpoint also gets an unconditioned evaluation in
its own evidence directory and W&B namespace. Keep prediction/delay/replan
timing at (4, 2, 2), seed 0, and the existing dashboard cluster-bootstrap
`net_stock_lcb` metric.

Use Modal B200 for training, as in O59 production; use L40S for closed-loop
evaluation. The B200 synthetic preflight passed with batch 512, all 26 offsets,
optimizer-state allocation, compilation, CUDA-graph capture and replay, and
updates across the conditioning/AWR boundary. The final check also completed
all 2,048 validation samples at batch 128 in 7.62 seconds and then passed eight
more training updates. Peak allocated/reserved memory was 166.60/175.86 GiB.
Matched warmed training throughput was about 960 windows/s, with estimated
BF16 MFU 28.3%. See [performance/report.md](performance/report.md). Do not change the effective
batch, model, or objective to conceal a failed memory check.

The launch must use a clean pushed commit. Record that SHA, the resolved config,
data identity, environment, Modal App and FunctionCall IDs, W&B run, and artifact
hashes alongside the training evidence.

The full metadata audit passed for 1,295,370 training replays and 13,159
validation replays. Their minimum actual frame counts are 1,668 and 2,566,
respectively. All 44 pinned train indexes and all split row identities matched.
No replay is shorter than 293 frames. The audit streamed JSONL manifests and
indexes only, without MDS shards. See [corpus_length_audit.json](corpus_length_audit.json).

## Launch command

```sh
uv run scripts/launch_modal.py \
  --gpu B200 --closed-loop-gpu L40S \
  --cpu 32 --cpu-limit 48 \
  --memory-gib 128 --memory-limit-gib 384 --disk-gib 2048 \
  --timeout-hours 24 \
  -- uv run experiments/064_production_percentile.py train \
  --cfg.arm bootstrap30 \
  --cfg.train-compile-mode reduce-overhead \
  --stop-after-update 98304 \
  --comment bootstrap30-u95-dense30
```
