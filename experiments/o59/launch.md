# O59 v3 production run

Train the 268,857,469-parameter policy from seed zero through update 98,304.
This is the stable parent run. A separate decay continuation can branch from
its immutable checkpoint for 32,768 updates; this launch does not start that
continuation.

## Treatment and reference

The model has a 16-layer, width-1024 trunk and a four-layer, width-1024 temporal
decoder. The decoder MLP has width 4096. Both the decoder and trunk-skip action
heads use RMSNorm, a width-1024 SiLU hidden layer, and a vocabulary projection.

Version 3 fixes button-head normalization parity: epsilon is 1e-5 in training,
diagnostics, and inference. Other heads retain 1e-6. Training arithmetic is
unchanged from v2. Versions 1 and 2 are rejected at checkpoint load because
their model behavior differs.

There is no concurrent control run. The v1 benchmark is a historical execution
reference; its 1,403.7 samples/s and 29.6% MFU do not describe this model. Measure
v3 throughput and memory after compilation. Use gameplay evaluation to assess
the trained policy; offline validation is diagnostic.

Fixed inputs are the complete 44-source policy-world-v8 corpus, its pinned MDS
manifests and schemas, production mixture statistics, identity artifacts,
seed zero, batch 512, and the existing Muon/Adam parameterization. Sample 32
policy prefixes per window and retain all 128 suffix prefixes for the critic.
Use BF16 and max-autotune. Keep production diagnostics and eager Muon.

## Launch

```sh
uv run scripts/launch_modal.py \
  --gpu B200 --closed-loop-gpu L40S \
  --cpu 32 --cpu-limit 48 \
  --memory-gib 128 --memory-limit-gib 384 --disk-gib 2048 \
  --timeout-hours 24 \
  -- uv run experiments/059_muon_history_decoder.py train \
  --cfg.train-compile-mode max-autotune \
  --stop-after-update 98304 \
  --comment v3-b512
```

Pass the stop explicitly so automatic retries retain the same boundary. A fresh
production run otherwise stops at 98,304 by default, while a resumed run without
this flag can continue to the configured maximum of 131,072.

The launcher requires a clean, pushed commit and verifies the Modal profile and
the `hal` Secret. It records the Git SHA, App ID, FunctionCall ID, and launch ID.
The checkpoint records the resolved model/training configuration, optimizer and
scheduler, loader cursor, all RNG state, dataset identities, statistics hashes,
identity hashes, and environment. Training artifacts go to the run's R2 prefix;
benchmark artifacts remain in their separate Modal Volume.

## Boundaries and monitoring

- Warmup: 4,096 updates. AWR begins at update 4,097.
- Durable checkpoint: every 2,048 updates.
- Offline validation: every 4,096 updates.
- Gameplay: 96 matchups every 8,192 updates and at completion, on spawned L40S
  workers. These evaluations use additional GPU allocations.
- Stop: 98,304 updates, with `final.pt` and the numbered boundary checkpoint.
- Recovery: launcher retries infrastructure failures and resumes a compatible
  R2 checkpoint. A terminal training error does not trigger a fresh training run.

Establish v3's own warmed throughput, MFU, loader-wait, and memory baselines.
Watch loss and gradients, GPU memory, cgroup current/peak/limit, checkpoint age,
and attempt identity. Exclude compilation, validation, checkpoint work, and
profiler windows from throughput comparisons. The larger model's full-size B200
memory fit remains a startup check; small local CUDA tests do not establish it.

Completion requires the FunctionCall result, launcher state, W&B run, and R2
artifacts to agree. Preserve the last healthy checkpoint if a failure occurs.

## Local validation before launch

All commands passed on 2026-09-20:

```sh
uv run ruff format --check .
uv run ruff check .
uv run ty check --python-version 3.14 --error-on-warning \
  hal experiments/051_muon_parameterization.py \
  scripts/cache_modal_fixtures.py scripts/launch_gce.py \
  scripts/launch_modal.py scripts/launch_vast.py scripts/replay_policy_fault.py \
  experiments/059_muon_history_decoder.py
uv run pytest -q tests/experiments/test_059_muon_history_decoder.py \
  tests/test_muon.py tests/test_benchmark_o59_modal.py
uv run pytest -q --ignore=tests/experiments -m 'not integration' \
  --basetemp=results/o59-launch-preflight/tmp/unit
HAL_REQUIRE_INTEGRATION=1 uv run pytest -q \
  tests/test_roundtrip.py tests/test_session_cleanup.py -m integration \
  --basetemp=results/o59-launch-preflight/tmp/integration
```

Focused: 74 passed, 15 warnings. Broad: 1,166 passed, 18 integration tests
deselected, 29 warnings. Required Dolphin integration: 7 passed, 2 non-integration
tests deselected, 7 warnings. No tests skipped. CUDA resume checks ran locally.

The new near-zero button-input regression failed before the fix, with maximum
absolute logit difference 0.165, and passed afterward. The focused checks also
verify exact head output/gradient parity and reject v1/v2 checkpoint identities.
One sandboxed uv probe could not write its cache; rerunning with the existing
uv permission succeeded.
