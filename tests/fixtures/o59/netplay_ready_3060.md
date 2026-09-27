# 059 netplay prepared-profile latency on RTX 3060

The candidate was the capability-v2 059 bundle with checkpoint SHA-256
`52b5233ed506f59f514f7e90a6a6111206152db7413f451dc35be5c30d1e671b`.
The bundle SHA-256 was
`0ff1daf80caa36a94a713c4ccba9223db8d7ba7c1379b5865bbc40b8a8c2f3ec`.
The treatment used the prepared CUDA BF16 cached policy, one admitted stream,
the local process transport, a 0.5 ms maximum coalescing wait, a horizon of
eight, and four-frame updates. Each profile had 20 warmup calls and 200
measured calls. The timing includes request delivery, engine scheduling,
inference, and response delivery. It does not include Dolphin stepping or
controller submission.

| Physical delay | Fixed prefix | p50 | p95 | p99 | Maximum |
|---:|---:|---:|---:|---:|---:|
| 2 frames | 3 actions | 8.598 ms | 8.979 ms | 9.648 ms | 13.208 ms |
| 3 frames | 4 actions | 7.686 ms | 8.016 ms | 8.918 ms | 9.573 ms |

Both profiles met the preparation check's one-frame p99 limit. The delay-2
profile also met its 12 ms p95 limit. Preparation of the shared model and both
profiles took 85.338 seconds; the candidate process peak allocated GPU memory
was 946.994 MiB. The preparation time is below the 120-second recovery target,
but this run did not inject a process failure or test recovery.

The GPU was **co-resident idle**, not isolated. Before capture, NVIDIA reported
an existing compute process (PID 2982847, 1198 MiB) and 0% GPU and memory
utilization at the sampled instant. It remained present after capture. The
existing process was not stopped. The post-capture compute-process list also
included the capture process itself (PID 2134601, 950 MiB).

Raw samples, all source SHA-256 values before and after capture, runtime
versions, bundle identity, and GPU residency are in
[`netplay-ready-3060-final.json`](../../../runs/refactor-059/netplay-ready-3060-final.json)
(SHA-256 `5431bb3197dd864bf1ba7d99f656778cfdc5b7d1763d34394405ffcb315ef9e2`).
The source hashes matched before and after. The execution log is
[`netplay-ready-3060-final.log`](../../../runs/refactor-059/netplay-ready-3060-final.log).
The reproducible capture source is [`capture_netplay_ready.py`](capture_netplay_ready.py).

The command was:

```bash
UV_CACHE_DIR=/tmp/hal-uv-cache \
TMPDIR=/home/ericgu/src/hal/runs/refactor-059/tmp \
TORCHINDUCTOR_CACHE_DIR=/home/ericgu/src/hal/runs/refactor-059/inductor-capture \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/ericgu/src/hal/.venv/bin/python \
tests/fixtures/o59/capture_netplay_ready.py \
runs/refactor-059/o59-capability-v2.hal \
runs/refactor-059/netplay-ready-3060-final.json --capacity 1
```

This is a short prepared-profile readiness measurement. It does not close the
three-trial 2,400-frame 3060 service gate, the 30-minute match/rematch soak, the
fault-injection gates, or the RTX 6000 Ada capacity gate.
