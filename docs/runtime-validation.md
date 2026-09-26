# Runtime validation, 2026-09-26

## Contract

Both local evaluation and nonblocking netplay use the prediction API and action
scheduler. Production requests contain new observations only. The old model
`step()` interface and action queues have been removed.

The measured workload is checkpoint `vywk3cih` (O59), input delay 2, thinking
allowance 1, replan interval 4, horizon 8, temperature 1, return 20, identity
`IBDW#0`, and seed 1001 unless noted. Offsets +1..+3 are fixed; +4..+7 are the
next action chunk and +8 is its reserve. Measurements use one RTX 3060,
compiled BF16 inference, CUDA graphs, and one Torch CPU thread. Timing processes
call `gc.collect(); gc.freeze()` after preparation. This GC setting is confined
to benchmark programs and is recorded in their output.

Streaming KV's retained-history semantics and BF16 rounding differences are
explicitly accepted. Cropped-window inference is a diagnostic control, not a
bitwise correctness requirement. See [the numerical contract](kv-cache.md).
CPU reference tests compare cached inference to full-sequence causal sliding
attention, including partial updates, ring wrap, reset, and history attention.
Parallel-prefix tests compare temporal states, K/V, and RNG draw counts against
serial decoding. Checkpoint-specific numerical diagnostics are in that document.

## Latency and throughput

| Workload | Median inference | p95 | p99 | Throughput |
| --- | ---: | ---: | ---: | ---: |
| Original KV, full history requests, update 2 | 21.28 ms | 22.36 ms | 23.93 ms | Model only |
| Original KV, incremental requests, update 2 | 12.02 ms | 12.60 ms | 13.16 ms | Model only |
| Final KV, incremental requests, update 4 | 8.27 ms | 8.75 ms | 9.25 ms | Model only |
| Local match before prefix/staging optimization | 10.83 ms | 11.49 ms | 13.38 ms | 243.16 FPS |
| Final direct local match, versus level-9 CPU | 8.29 ms | 8.69 ms | 9.43 ms | 296.74 FPS |
| Production vector evaluator, versus level-9 CPU | 8.44 ms | 8.78 ms | 9.52 ms | 248–252 gameplay FPS |
| One netplay model, recorded-input opponent | 10.55 ms | 11.38 ms | 13.15 ms | 59.94 steady FPS |
| Two netplay models on the same 3060, peer A | 14.25 ms | 18.67 ms | 19.38 ms | 59.93 steady FPS |
| Two netplay models on the same 3060, peer B | 17.50 ms | 18.93 ms | 20.15 ms | 59.92 steady FPS |

Model-only baseline uses identical replay observations with neutral applied and
fixed actions, 2,400 frames, excluding the first 400. Full versus incremental
requests produce identical baseline action hashes. Final model-only measurement
uses the same replay with its recorded applied actions, neutral fixed actions,
2,400 frames, excluding the first 300. Thus the final row is a measured workload,
not an exact matched-input action-parity experiment. The final component profile
below repeats the baseline neutral-action inputs.

Local matches completed 3,940 and 6,316 frames respectively. Their game states
differ, so the FPS change is not a controlled gameplay-strength comparison.
Local inference timings include startup: worst calls were 50.78 and 47.50 ms.
The production vector evaluator also completed a separate 3,748-frame match
using `PolicyBatchAdapter` and process workers. Steady progress intervals were
248–252 FPS (initial interval 237); total wall throughput, including Dolphin
startup, was 168.52 FPS. Prediction p50/p95/p99 was 8.44/8.78/9.52 ms, with a
48.52 ms cold maximum. Artifacts are in `runs/runtime-final/vector-eval/`.

The earlier approximately 6 ms result used a shorter four-step decoder and a
two-frame replan interval; it is not the workload above.

The two-model netplay game completed 5,646 frames on both peers with seeds
1001/1002. Complete-match FPS was 58.61/58.60, including a 2.14-second terminal
observation interval. Steady FPS excludes the first 300 intervals and the final
terminal interval. Eight complete 600-frame windows were 59.815–59.954 FPS for
A and 59.811–59.954 for B. This is not a rock-solid 59.9 FPS guarantee.

The two models compete for the same GPU. This run fails the 12 ms p95 inference
target and the one-frame deadline: A/B recorded 640/928 missed frame deadlines,
189/274 prefix mismatches, two exhausted chunks each, and five neutral fallback
frames each. These counters include countdown/startup. Transport corrections
were 2/0. Average emulator FPS must not be used as evidence that plans arrived
on time.

The one-model netplay run used `runs/runtime-final/replay_peer.py` as the other
peer. It replays recorded controller actions without model inference, while both
Dolphins render on the 3060. The match completed 2,703 frames. Complete-match FPS
was 57.25, including startup and the terminal 2.14-second interval; steady FPS was
59.939. Its four complete 600-frame windows were 59.945, 59.945, 59.926, and
59.949 FPS. The model recorded four missed frame deadlines, zero prefix
mismatches, zero exhausted chunks, two neutral fallback frames, and zero
transport corrections. Worst request latency was 18.08 ms. This meets the
12 ms p95 target for one served model, but does not prove zero jitter or two-model
capacity on one 3060. Artifacts are in `netplay-single/` and `recorded-peer/`.

## Component profile

The same 800 replay observations and neutral actions were profiled before and
after; the first 400 frames were excluded. Each row reports median / p95 ms per
request. CUDA events and host spans use different clocks. Host spans can wait
for earlier GPU work; their values must not be added to the GPU table.

| GPU component | Before | After |
| --- | ---: | ---: |
| Observation upload and quantization | 1.336 / 1.505 | 0.375 / 0.396 |
| Trunk and projected history | 3.379 / 3.396 | 1.696 / 1.701 |
| Plan metadata | 0.038 / 0.046 | 0.003 / 0.004 |
| Fixed action prefix upload/quantization | 0.390 / 0.434 | 0.326 / 0.347 |
| Sampling draws upload | 0.185 / 0.208 | 0.186 / 0.200 |
| Runtime constants | 0.058 / 0.071 | 0.053 / 0.063 |
| Action decoder | 6.212 / 6.233 | 5.009 / 5.018 |
| Output conversion/copy | 0.068 / 0.083 | 0.064 / 0.070 |

| Exclusive host span | Before | After |
| --- | ---: | ---: |
| Observation ingestion | 0.316 / 0.369 | 0.255 / 0.272 |
| KV staging | 2.684 / 2.838 | 0.356 / 0.376 |
| KV graph dispatch | 0.167 / 0.186 | 0.064 / 0.079 |
| Other cache advancement | 0.204 / 0.234 | 0.100 / 0.110 |
| Plan context construction | 1.555 / 1.560 | 0.024 / 0.029 |
| Fixed prefix, including wait for trunk | 0.372 / 0.415 | 1.852 / 1.874 |
| RNG generation/upload | 0.170 / 0.190 | 0.171 / 0.183 |
| Runtime constants | 0.045 / 0.055 | 0.041 / 0.050 |
| Decoder dispatch | 0.049 / 0.065 | 0.055 / 0.066 |
| Output copy, including wait for decoder | 6.185 / 6.195 | 4.973 / 4.980 |
| Other planning/action conversion | 0.342 / 0.385 | 0.321 / 0.343 |
| Request validation and bookkeeping | 0.259 / 0.298 | 0.128 / 0.139 |

Final instrumented complete-call median/p95 is 8.37/8.45 ms. The isolated profile
source and raw spans are in `runs/runtime-final/profile-{baseline,final}.json`
and corresponding scripts/source copies. Profiling did not modify maintained
model code.

## Reproduce

```sh
PYTHONPATH=. uv run python experiments/benchmark_kv_cache.py \
  runs/netplay/o59-vywk3cih.hal \
  runs/netplay/selfplay-o59-20260924-retry1/replays-a/Game_20260924T152607.slp \
  runs/runtime-final/new-model-measurement \
  --frames 2400 --prediction-horizon 8 --fixed-prefix 3 \
  --replan-interval 4 --update-frames 4
```

For each real netplay peer, run `experiments/benchmark_kv_netplay.py` with its
own account, port, seed, and output directory, plus `--history-mode kv_cache
--update-frames 4`. Its default timing is 2/1/4/8. Results include raw inference
and frame intervals, schedule counters, environment, Git SHA, and source,
checkpoint, emulator, ISO, and replay hashes.

All artifacts from this work are under `runs/runtime-final/`. No cloud resources
were created. The prior L4 deletion and empty-project verification remain
recorded in `runs/netplay/gce-l4-debug/3060-match/README.md`. A fresh Compute
Engine listing failed because gcloud requires account reauthentication.

## Final checks

Commands ran with workspace scratch directories for `TMPDIR`, `UV_CACHE_DIR`,
and `TORCHINDUCTOR_CACHE_DIR`; Dolphin tests required host socket/config access.

- `uv run ruff format --check .`: failed on five pre-existing generated files
  under `outputs/o59-rank1-analysis-scripts/`; 379 files passed. Those unrelated
  files were left unchanged.
- `uv run ruff check .`: failed with 66 pre-existing diagnostics under `outputs/`.
- `uv run ruff format --check hal tests experiments/benchmark_kv_cache.py experiments/benchmark_kv_netplay.py experiments/eval_kv_cache.py`:
  passed, 276 files.
- `uv run ruff check hal tests experiments/benchmark_kv_cache.py experiments/benchmark_kv_netplay.py experiments/eval_kv_cache.py`:
  passed.
- `uv run ty check --python-version 3.14 --error-on-warning hal experiments/051_muon_parameterization.py scripts/cache_modal_fixtures.py scripts/launch_gce.py scripts/launch_modal.py scripts/launch_vast.py scripts/replay_policy_fault.py`:
  passed, zero diagnostics.
- `uv run pytest -q --ignore=tests/experiments -m "not integration" -o faulthandler_timeout=90`:
  1,314 passed, 23 deselected, no skips; 29 dependency warnings, 134.86 seconds.
- `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration`:
  seven passed, two deselected, no skips; seven multiprocessing warnings,
  53.26 seconds.
- `bash -n deploy/netplay/run-host.sh` and `git diff --cached --check`: passed.

Earlier attempts encountered sandbox socket/config restrictions and the host's
`/tmp` quota. Workspace scratch directories and host permissions resolved those
environment failures. Initial targeted checks also exposed stale legacy-API tests
and two import-order errors; those were fixed before the final checks above.
