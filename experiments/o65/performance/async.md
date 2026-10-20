# O65 rollout host transfers and persistent compilation

Treatment: consolidate controller and observation transfers on the main thread.
Control: `o65-floor` (`376eb42b`), with shard timing instrumentation from
`2e129429`. The unchanged inputs are proxy checkpoint
`65e7e43e5cbb18a365a005922ec1cfd026ad807c045eae13b331f487b4ec82e6`, seed 65,
one group, eight shards, 256 rollout frames, `(D,R)=(8,4)/(10,2)`, BF16 decoding,
FP32 controller values, 32 windows and 16 decisions per actor step, four critic
epochs, simulator, adapter, reward and objective. These measurements do not
select a learning configuration.

The machine has an RTX 3060 12 GB and a Ryzen 5 5600X: six physical cores and
twelve logical CPUs. The requested eight shards use logical CPUs 0 through 7. They share two
physical cores with sibling threads; this is not an eight-core host.

## Host path

Each half retains its CUDA stream. It owns one pinned FP32 input tensor and one
pinned byte tensor for observations. The latter holds each 440-byte native
record followed by 112 bytes of FP32 controller readback. Shards own fixed,
disjoint slices. The main thread copies all controller inputs for a half once,
records an event, and completes a host future after that copy finishes. Workers
wait only on that host future. A failed copy completes pending futures with an
error, so workers cannot read incomplete inputs or block cleanup. They step and pack observations without Torch or
CUDA. The main thread then makes one observation copy for the half and records
an event that protects the next host write.

For an ordinary two-frame tick with eight active shards, this changes controller
D2H copies from eight to two and observation H2D copies from 32 to two. The
payload bytes and values stay the same. Reset transfers remain on the main
thread. A worker stops each ended lane immediately and lets other lanes finish
the tick. Tests compare this with the former stop-and-resubmit path through
staggered ends and resets, on CPU and CUDA.

The native worker records simulator time, packing time, input readiness wait,
and residual host work for each tick. Simulator time is the complete native
step, including adapter quantization, projection and reward. Packing time covers
host staging and controller readback; it does not separate C physics from C
projection. Main-thread records separate input-event
wait, observation-buffer reuse wait, worker wait and copy submission. Waiting
thread-seconds overlap and must not be added to elapsed rollout time.

## Simulator floor

The first Torch-free probe measured 64 matches for 256 frames from a fresh
match on CPU 0: 0.581442 s wall, 0.573955 s native stepping, 28,178 match-frames/s.
It includes the pregame interval. The maintained benchmark warms 256 frames
before measuring the next 256, matching the timed trainer update's frame range
133 to 389. Its settings and all artifact hashes are recorded with the results.
Fixed inward half-stick inputs replace policy inference in this probe. This is
a workload-specific simulator floor, not a gameplay or universal CPU rate.

Run it without Torch or a GPU:

```sh
PYTHONPATH=.:$MSL_SOURCE MSL_CORE_LIBRARY=$MSL_SOURCE/build/melee_core/python/libmelee_core.so \
  uv run python -m scripts.benchmark_o65_sim \
  --data-dir "$MSL_SOURCE/data" --cpu 0 --output runs/async/simulator-floor.json
```

## Compilation identity and lifetime

`--compile-cache DIR` selects a directory by a SHA-256 identity over the full
Torch and Triton versions, CUDA version, GPU name and capability, and the source
hash. The experiment selects the source files: `hal/**/*.py`, the trainer,
`pyproject.toml` and `uv.lock`. A different identity selects a separate directory.
A wrong identity file, invalid artifact manifest, changed artifact hash or
rejected Torch artifact raises before training. No incompatible cache is used.

Both `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` point inside that directory.
The process loads and saves Torch's public cache artifacts and records their
counts by kind. See the [PyTorch cache documentation](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html).
CUDA graph addresses remain local to each process; every process captures its
graphs again. Reported capture time excludes the two warmup calls. Preparation
time includes compilation, allocation and those warmup calls.

The Modal trainer command selects `/o65-results/compile-cache` on Volume
`hal-o65-profile`. The existing final Volume commit also persists this cache.
No Modal job was launched for this change. The next Modal run must qualify reuse
on the PRO 6000; local RTX 3060 artifacts have a different identity.

## Shape families and padding

There are three policy batches: student and anchor rows in the first half,
and student rows in the second. Each needs three tick variants:
`(append=1,R=4)`, `(append=2,R=4)` and `(append=2,R=2)`. Block boundaries always
fall on an R=4 tick, so there is no `(append=1,R=2)` variant. Capture reports list
every batch, R, append width, preparation time and capture time.

Padding the three batches to their shared maximum would reduce these shape
families from nine to three and increase total policy rows from `2*M` to `3*M`,
or 50%. Separate mutable caches still need separate CUDA graph captures. Policy
guards and frontend tracing can also remain distinct, so removing six shape
families is an upper bound on compile savings, not a measured padding result.

For context, the earlier PRO 6000 bare probes in [floor.md](floor.md) give a
serialized tick sum of 3.139410 s at rows 154, 870 and 1,024. Repeating the
1,024-row measurements for all three batches projects 4.097472 s, before
prefills: about 31% more GPU tick time. Including the measured prefills projects
4.468962 s versus 3.380142 s. These sums exclude transfer scheduling and buffer
overlap. Padding has not been implemented.

## Measurement records

Full Nsight traces attribute eager launches, graph launches, kernels and GPU
interval unions. `scripts/analyze_o65_async.py` also reports each complete tick,
with GPU work assigned by its issuing API. Clean phase wall and main-thread CPU
times come from a separate update in the same process. Traced wall times are
not used for training throughput. An initial concurrent validation run is
retained as diagnostic evidence; clean comparisons run after validation stops.


## RTX 3060 results

Each row is one clean update after one warmup update. GPU busy and launch
counts come from the separate complete traced update. Preparation excludes
reference; actor includes Muon and auxiliary Adam.

| Matches | Path | Seconds/update | Match-frames/s | Student decisions/s |
| ---: | --- | ---: | ---: | ---: |
| 16 | old | 3.355962 | 1,221 | 915 |
| 16 | staged | 3.144906 | 1,302 | 977 |
| 64 | old | 5.158561 | 3,176 | 2,246 |
| 64 | staged | 4.983099 | 3,288 | 2,325 |

The staged path reduces complete update time by 6.3% at 16 matches and 3.4%
at 64. Rollout falls from 2.4965 to 2.2854 s and from 3.4716 to 3.3089 s.
These are single-update systems measurements, not confidence intervals.

| Matches | Path | Phase | Wall s | Main CPU s | Eager launches | Graph launches | Kernels | GPU busy s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | old | rollout | 2.496524 | 1.263085 | 66,361 | 384 | 588,217 | 1.640589 |
| 16 | old | reference | 0.282873 | 0.282857 | 148 | 1 | 9,990 | 0.282658 |
| 16 | old | preparation | 0.075654 | 0.075647 | 2,413 | 2 | 4,723 | 0.057255 |
| 16 | old | critic | 0.017240 | 0.013768 | 629 | 0 | 629 | 0.012692 |
| 16 | old | actor | 0.480102 | 0.479943 | 4,308 | 4 | 19,204 | 0.460866 |
| 16 | staged | rollout | 2.285447 | 2.049885 | 57,881 | 384 | 579,737 | 1.544993 |
| 16 | staged | reference | 0.281886 | 0.281867 | 148 | 1 | 9,990 | 0.282727 |
| 16 | staged | preparation | 0.074981 | 0.074980 | 2,413 | 2 | 4,723 | 0.056523 |
| 16 | staged | critic | 0.016758 | 0.013753 | 629 | 0 | 629 | 0.012510 |
| 16 | staged | actor | 0.482025 | 0.481911 | 4,308 | 4 | 19,204 | 0.460207 |
| 64 | old | rollout | 3.471577 | 1.414759 | 66,361 | 384 | 595,979 | 2.439728 |
| 64 | old | reference | 0.503836 | 0.503573 | 148 | 1 | 17,854 | 0.503765 |
| 64 | old | preparation | 0.155705 | 0.155672 | 2,449 | 2 | 4,759 | 0.137840 |
| 64 | old | critic | 0.046020 | 0.042386 | 625 | 0 | 625 | 0.041431 |
| 64 | old | actor | 0.977027 | 0.976771 | 8,616 | 8 | 38,408 | 0.921245 |
| 64 | staged | rollout | 3.308930 | 3.023727 | 57,881 | 384 | 587,499 | 2.418840 |
| 64 | staged | reference | 0.505917 | 0.505872 | 148 | 1 | 17,854 | 0.506743 |
| 64 | staged | preparation | 0.154831 | 0.154821 | 2,449 | 2 | 4,759 | 0.137927 |
| 64 | staged | critic | 0.044894 | 0.041285 | 625 | 0 | 625 | 0.041591 |
| 64 | staged | actor | 0.964223 | 0.964139 | 8,616 | 8 | 38,408 | 0.922881 |

Rollout CUDA API calls move from 17 host threads to one. Controller D2H copies
fall from 1,024 to 256 per rollout; total rollout D2H copies fall from 1,033 to
265. Observation H2D copies fall from 4,096 to 256; total H2D copies fall from
4,226 to 386. The additional metadata copies remain. Total bytes are identical
between the two paths at each match count. Main-thread CPU time rises because
this thread now owns all CUDA dispatch and copy-event waits. The native
workers perform no CUDA calls.

| Matches | Path | Shard total wall s | Native step s | Host packing s | Input wait s | Other s |
| ---: | --- | --- | --- | --- | --- | --- |
| 16 | old | 1.0243–1.5256 | 0.0596–0.0692 | 0.2390–0.2678 | 0.6126–1.1235 | 0.0774–0.0901 |
| 16 | staged | 1.0957–1.2808 | 0.1274–0.1512 | 0.0170–0.0209 | 0.9320–1.1209 | 0.0018–0.0021 |
| 64 | old | 1.7629–2.3337 | 0.1364–0.1587 | 0.1482–0.1768 | 1.3430–1.9475 | 0.0588–0.0974 |
| 64 | staged | 1.9224–2.1871 | 0.1748–0.2180 | 0.0201–0.0244 | 1.7011–1.9601 | 0.0020–0.0024 |

These are per-shard ranges over one clean rollout, not sums on the critical
path. Native step wall includes adapter work and any scheduler or GIL delay
before returning to Python. The old packing field includes its transfer
submission; the new field contains only host staging and readback.

| Matches | Path | Tick | Mean GPU busy ms | Mean shard native ms | Mean shard packing ms | Mean shard input wait ms |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 16 | old | boundary | 33.3058 | 0.4715 | 1.6149 | 3.7646 |
| 16 | old | R4 | 13.8844 | 0.4655 | 1.5813 | 5.0363 |
| 16 | old | R2 | 11.1393 | 0.4639 | 1.6785 | 2.6934 |
| 16 | staged | boundary | 33.1366 | 0.7356 | 0.1639 | 14.9195 |
| 16 | staged | R4 | 13.0941 | 0.6914 | 0.1534 | 6.8272 |
| 16 | staged | R2 | 10.4165 | 0.6515 | 0.1540 | 4.7037 |
| 64 | old | boundary | 79.1707 | 1.1346 | 1.2298 | 8.3369 |
| 64 | old | R4 | 20.2316 | 1.3361 | 2.3485 | 12.6982 |
| 64 | old | R2 | 16.0414 | 1.3843 | 2.1832 | 9.3313 |
| 64 | staged | boundary | 79.1979 | 1.1210 | 0.1779 | 20.9380 |
| 64 | staged | R4 | 20.1113 | 1.6064 | 0.1752 | 14.4889 |
| 64 | staged | R2 | 15.8307 | 1.6169 | 0.1765 | 10.8423 |

Every tick’s GPU union, launch counts, kernel count and shard split are retained in [async-measurements.json](async-measurements.json).
Staged ticks also retain main-thread copy/worker waits. Raw paths, sizes and hashes are in [async-artifacts.json](async-artifacts.json).
The legacy trainer is `2e129429` plus the diagnostic
[NSight export patch](async-baseline.patch), with the same current HAL libraries.
The staged runtime is `974a9ca0`; later test and report changes do not change
its compilation identity. Raw reports remain under `runs/async/` locally.

## Cold start and cache reuse

| Matches | Cache state | First-update warmup s | CUDA graph capture s |
| ---: | --- | ---: | ---: |
| 16 | empty | 1264.879000 | 0.773354 |
| 16 | persisted | 132.772201 | 0.713171 |
| 64 | 16-match artifacts; new shapes | 1375.349529 | 1.174401 |
| 64 | persisted 64-match shapes | 140.940837 | 0.982367 |

The fresh 16-match cache saves 1,132.1 s on the next process, a 9.5× reduction
in first-update warmup. Portable load takes 3.22 s. It loads 12 Inductor and
11 AOT-autograd artifacts, plus autotuning records. Frontend tracing, fixed
shape preparation, graph capture and the first update still run. The 64-match
new-shape run must compile its new variants despite the valid 16-match cache;
this is a shape miss, not an identity mismatch.

The six non-maximum tick shapes account for 721.2 s of cold preparation.
Removing those shape families could avoid at most this preparation cost;
frontend guards, warmup work and separate captures remain. This is an upper
bound, not a measured padding speedup. The work tradeoff is the 50% increase
in policy rows described above.

## PRO 6000 projection

At 1,024 matches on 16 physical cores, the warmed 64-match one-core probe
projects 0.534 s for simulation plus adapter dispatch. The prior PRO 6000
trace measured 3.006885 s of rollout GPU busy. The new local traced residual
(`rollout wall - max(native/packing critical shard, GPU busy)`) is 1.239 s
at 16 matches and 1.493 s at 64. Thus:

```text
projected rollout = max(0.534, 3.006885) + (1.239 to 1.493)
                  = 4.246 to 4.500 seconds
```

**This is a projection. GPU work is the larger component, so it projects a
GPU bound, not a simulator bound.** It assumes linear simulator scaling, the
desktop probe’s per-core rate on the server, unchanged PRO GPU work from the
old trace, and unchanged local host residuals. It does not establish a server
CPU rate or a complete hardware floor. A PRO 6000 run remains necessary.

## Correctness and limits

CPU and CUDA parity cover four shards sharing both staging buffers, staggered
ends and resets over two rollouts. They compare every dense rollout tensor,
frame-store tensor and pool snapshot. A separate CUDA copy test preserves every
FP32 controller bit, including negative zero. Native workers run under guards
that reject CUDA APIs and CUDA tensor operations. Failed copy and interruption
tests release pending workers without exposing incomplete inputs; a native
snapshot test confirms that a failed input transfer does not step its batch.

The real proxy/native resume report passes all seven model and optimizer state
families at the existing v5 tolerances. All ten state hashes restore exactly
in each of three rewinds. Model RMS error is `2.69885e-8` against `1.10639e-6`
tolerance; maximum parameter error `5.67734e-6` remains diagnostic. Neither
resume tolerances nor the recorded FP32-reference precision limits were changed.
The model, objective, optimizer and decoder precision are unchanged. No non-finite
training value was accepted.

The required v7 fixture is absent. Its six skips are validation failures under
AGENTS.md; they are not waived. Two production hardware tests are disabled on
this host. The optional optimizer CUDA graph is deferred while the step 1–3
handoff remains blocked by that fixture. No Modal job or cloud machine was launched.
