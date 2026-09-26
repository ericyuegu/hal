# KV cache inference

The history decoder now defaults to bounded KV cache inference when loaded through `load_policy`, `hal-play`, or the netplay runner. The older O50 backend keeps its cropped-window default. Use `--history-mode window` for the history decoder's control path. `--compiled` enables the measured CUDA graph path.

The KV cache changes the model's history semantics. Each layer attends to the last 256 frames, but retained keys and values keep information from older frames. This retained-history behavior is explicitly accepted for production inference because of the measured speedup. It is not exact cropped-window parity. BF16 differences from changed matrix shapes and RoPE coordinates are also accepted. Cache correctness is checked against an uncached full-sequence reference with the same causal sliding attention window; cropped-window comparisons remain numerical diagnostics.

The [DeepSeek-V4.1 report, section 3.2.2](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf) makes a related tradeoff: its bounded replay reconstructs approximate sliding-window states to reduce serving costs. HAL uses a different mechanism, retaining streaming states rather than reconstructing them. The shared decision is to accept measured numerical differences for performance; the report is not evidence of HAL correctness.

## Implementation

- Each trunk layer owns a GPU KV ring. The decoder's projected history uses another ring. Attention reads physical slots directly, with absolute RoPE positions and a causal window mask.
- Updates process one, two, or four new frames together. Each ring has `256 + update_frames - 1` slots so every query retains its full causal window. The default remains two; `--kv-update-frames 4` processes a four-frame replan interval in one trunk call.
- KV append uses an in-place custom operator. This keeps compilation from materializing a complete cache update. The bounded feature staging buffer also avoids a full GPU feature history.
- CUDA graph replay copies only new inputs. Cache storage stays fixed across frames and game resets. Reset invalidates position metadata rather than clearing all keys and values.
- CUDA linear weights are converted to BF16 once. Other parameters retain their original types. This removes repeated autocast conversions of the same linear weights.
- The temporal action cache starts fresh for each plan. Known fixed-prefix actions are processed in parallel before the sampled tail. This preserves the causal computation but can introduce BF16 rounding differences and change individual sampled actions. Temperature and return settings remain live decoder inputs.

Both local and netplay runtimes use the same prediction API. The target live schedule uses input delay 2, thinking allowance 1, replan interval 4, and prediction horizon 8: offsets +1 through +3 are fixed, +4 through +7 are used, and +8 is a reserve action.

## RTX 3060 measurements

Batch size 1, checkpoint `vywk3cih`, 1,600 recorded frames, first 300 excluded from steady-state timings. Each measurement includes the full policy call and a CUDA synchronization. No compilation was permitted after preparation. All modes use the same observations, policy seed 1001, IBDW identity, temperature 1, desired return 20, transport 2, and replan 2.

| Mode | Replan p95 | Intervening-frame p95 | Peak allocated GPU memory |
| --- | ---: | ---: | ---: |
| Window, FP32 storage with BF16 autocast | 20.13 ms | 0.95 ms | 964 MiB |
| Window, BF16 linear storage | 12.29 ms | 0.86 ms | 488 MiB |
| KV cache, eager | 37.82 ms | 0.22 ms | 500 MiB |
| KV cache, compiled without CUDA graphs | 11.51 ms | 0.17 ms | 500 MiB |
| KV cache, one frame per update, CUDA graphs | 6.37 ms | 2.54 ms | 504 MiB |
| KV cache, two frames per update, CUDA graphs | **6.49 ms** | **0.17 ms** | **518 MiB** |

Two-frame updates reduce sustained work. Their first live call after preparation was 7.21 ms. The GPU profile attributes about 1.65 ms of kernel time to the incremental trunk and history projection, and 3.10 ms to the four-step decoder. These kernel sums are separate from full-call wall latency.

The 13 KV rings hold 13,684,736 bytes. Each two-frame update writes 106,496 bytes of new KV entries (104 KiB). A 20-frame profile contains only 17,130 bytes of device-to-device memcpy traffic in total, with 130 bounded KV append kernels. The memcpy total excludes kernel writes; it is not a total memory-traffic estimate.

Results, source hashes, artifact hashes, profiles, and traces are under `runs/netplay/streaming-o59/qualified-*`. The original two-policy result of 37.6 FPS is a historical baseline. Current netplay infrastructure also has changes made before this experiment, including the Vulkan backend, so use the new matched netplay control for attribution.

## Reproduce

```sh
PYTHONPATH=. uv run python experiments/benchmark_kv_cache.py \
  runs/netplay/o59-vywk3cih.hal \
  runs/netplay/selfplay-o59-20260924-retry1/replays-a/Game_20260924T152607.slp \
  runs/netplay/streaming-o59/new-measurement
```

Use `--history-mode window`, `--bf16-window-linears`, `--update-frames 1`, `--no-cuda-graphs`, or `--no-compiled` for the corresponding controls. The output directory must be new.

`experiments/benchmark_kv_netplay.py` runs one peer through the nonblocking netplay runtime with an explicit prediction shape. Start two peers with distinct accounts and Slippi ports, each targeting the other's connect code. It records completed replays, timing, final stocks, resolved configuration, and source/artifact hashes. Policy seeds do not fix Slippi's game RNG; each replay records the actual game state.

## Numerical checks

An eager BF16-autocast diagnostic uses identical recorded observations and teacher-forced actions at 13 positions through frame index 2,048. Changing only linear weight storage gives exactly identical hidden states and logits at every sampled position.

For the KV cache versus the BF16 window control, mean conditional KL divergence is 0.000049 nats before eviction and 0.000232 nats after eviction. All 208 conditional argmax decisions match. Maximum finite logit differences are 1.0 and 1.5 respectively; these raw differences include low-probability categories. The latest hidden state's relative L2 difference reaches 3.7%.

Before eviction, BF16 matrix shapes and RoPE coordinates already introduce rounding differences. After eviction, retained KV states also change the effective history. These diagnostics do not establish gameplay equivalence or identical stochastic actions. The inputs, per-group results, and diagnostic source are saved as `runs/netplay/streaming-o59/numerics.{json,py}`.

A separate check of parallel fixed-prefix decoding used the same O59 checkpoint at 13 replay positions through frame 2,047, with neutral observed/fixed actions and unchanged streaming KV. Mean conditional KL against serial prefix decoding was 0.00000615 nats (maximum 0.0000961). All 260 argmax choices matched; 5 of 260 stochastic group choices differed with the same uniforms, including downstream autoregressive effects. Float32 tests verify prefix states, per-layer K/V, and RNG draw counts. This is an accepted BF16 execution-order difference, not a guarantee of identical sampled actions. The diagnostic is saved as `runs/runtime-final/{numerics_final.py,prefix-numerics.json}`.

## Matched netplay throughput

Two compiled policies and two Dolphins shared the 3060, with the same accounts, Fox matchup, IBDW identity, transport 2, replan 2, temperature 1, return 20, and policy seeds 1001/1002.

| Mode on both peers | Completed frames | Full-match FPS | Policy p95, peers | Transport corrections |
| --- | ---: | ---: | ---: | ---: |
| Window | 3,948 | 37.8 | 29.7 / 29.4 ms | 0 / 0 |
| KV cache | 5,884 | 58.3 | 8.3 / 10.2 ms | 0 / 0 |

The KV cache game advanced 3,000 frames between progress reports at frames 2,400 and 5,400 in 50.1 seconds (59.9 FPS). Its full-match mean includes a slow final game-end step and occasional Dolphin frames above the 16.7 ms budget. Different game lengths reflect different action trajectories; FPS is the comparison here. The first KV cache game's result writer encountered a NaN in the live trajectory's terminal stock field after both peers had completed and logged the game. The validated replays supplied final stocks; the recovered metadata identifies the reporting failure.

Match artifacts are under `runs/netplay/streaming-o59/{streaming-selfplay,window-selfplay}`.

The netplay runner accepts `--history-mode auto`, `window`, or `kv_cache`. The local launcher forwards `HAL_NETPLAY_HISTORY_MODE`; its default is `auto`, which selects the backend's default.

## Gameplay qualification

`experiments/eval_kv_cache.py` ran the checkpoint's final 96-matchup, level-9 CPU protocol on a Modal L40S. Each matchup boot received 7,200 emulator frames and could contain more than one game after instant restart. The return target was the checkpoint's exact p90 value, 19.9760597229004. The original [W&B evaluation](https://wandb.ai/ericyuegu/hal/runs/vywk3cih?nw=nwuserericyuegu) at history step 1322 scored 1.204895 net stocks per active minute with cropped-window inference. KV cache scored 1.225828, a difference of +0.020933 NSM, within the 0.2 NSM tolerance. All 96 boots completed, with 151 games, 691,200 emulator frames, and no crashes. Full-job emulator throughput was 134.45 FPS; the broker's inference latency p95 was 6.06 ms.

The KV run used one active session because this cache implementation has batch size 1. Wall throughput therefore is not a matched comparison with the original parallel evaluation. The matchup schedule, frame budget, transport delay, replan interval, player masking, and p90 target match the original protocol. Source commit `68a69eb054a9108524b7be731d880feda889f40b`, configuration, replay rows, metrics, and 157 artifact hashes are under `r2://hal/runs/kv-cache-eval-vywk3cih-68a69eb0/eval96-p90/`.
