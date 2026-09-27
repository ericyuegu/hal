# Conditioned evaluation: throughput repair

## Scope

Evaluate unchanged O52 and O56 checkpoints against level-9 CPUs with actual
pending controller actions forced through the decoder. This is not direct H2H.
Keep `d=2`, `r=2`, `H=4`, sampling, identity conditioning, and matchup schedules
unchanged. Use RTX PRO 6000 GPUs and 32 process workers.

## Cause and fix

Switching evaluation to `O50Policy` bypassed the optimized context builder in
`RecedingHorizon`. Restoring process workers in `958da9eb` did not fix that.
The O52 process run settled at 6.0–6.2 emulator frames/s per worker. A short
py-spy recording showed the parent spending most of its sampled time checking
types across stored history and rebuilding columns from Python dictionaries.

Commit `b7a7710c` moves the existing ring implementation into
`hal/training/context_history.py`. Both `RecedingHorizon` and `O50Policy` now
use it. Observations are preprocessed once; contexts use contiguous ring slices
and packed device transfers. The decoder, pending queue, RNG, and transport
behavior are unchanged. There is no second ring implementation.

The portable policy still validates numeric types, now once per incoming
observation rather than once per stored observation at each replan. It rejects
type changes until reset. It emits all float mask fields to keep the compiled
input shape fixed; the historical caller retains conditional mask emission.

A local CPU microbenchmark compared the old and new `_context` methods on
32 streams with 256 frames each. Median times over ten warmed calls were
165.44 ms and 0.407 ms. This excludes ingestion, decoding, and Dolphin;
it is not an end-to-end speed claim.

## Verification

- `uv run ruff format --check .`: 333 files passed.
- `uv run ruff check .`: passed.
- `uv run ty check --python-version 3.14 --error-on-warning hal experiments/051_muon_parameterization.py scripts/cache_modal_fixtures.py scripts/launch_gce.py scripts/launch_modal.py scripts/launch_vast.py scripts/replay_policy_fault.py`: passed.
- `uv run pytest -q --ignore=tests/experiments -m "not integration" -rs`: 1,142 passed, 18 deselected, no skips.
- `uv run pytest -q tests/test_closed_loop_rings.py tests/test_o50_policy.py tests/test_o50_parity.py tests/test_policy_adapter.py tests/experiments/test_040_scaled_awr_bc.py -m "not integration"`: 110 passed, one CUDA-only skip, one deselected.
- `uv run pytest -q tests/experiments/test_002_flow_matching_rtc.py tests/experiments/test_050_scaled_temporal_awr.py tests/experiments/test_052_adamw_temporal_awr.py tests/experiments/test_056_decoder_capacity_reallocation.py -m "not integration"`: 84 passed.
- `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py tests/test_policy_adapter.py -m integration`: eight passed, nine deselected.

Checks used `UV_CACHE_DIR=/tmp/hal-uv-cache` because the default cache was
read-only in the sandbox. The first integration attempt failed all eight tests
because the sandbox prevented Dolphin startup. The unrestricted rerun passed.
The first full suite stalled under sandbox IPC restrictions and was terminated;
the unrestricted rerun passed. An initial focused run exposed an obsolete
`experiment.ACTION_CHANNELS` import in the O56 parity test after the earlier
cleanup. It now imports the stable action schema, and the rerun passed.

Input parity is bit-exact through cold padding, wraparound, one-stream reset,
both ports, changed batch order, nontrivial statistics, NaNs, and integer item
sentinels. Existing decoder and actual-controller integration tests also pass.

## Operations

Remote branch: `debug-conditioned-process-eval`. The slash form could not be
created because the remote already has a branch named `debug`.

Stopped slow process runs:

- O52: `ap-MBj0eLBgX3ugCbbqJGWSNT`, call `fc-01M2V81F4K0DPA2H78F2DYKKZB`.
- O56: `ap-mYWpC081t3N3KDsgBcmQfI`, call `fc-01M2V81F8V7RK46V2SZV6SRXG7`.

The stopped runs are incomplete and must not supply aggregate gameplay results.
No checkpoint was modified.

Smoke evaluation:

- App: `ap-EFH4IUdumbBHETXL6w6YX0`.
- Call: `fc-01M2V9RXBBP66JYE1BFCQ8BHQ1`.
- Launch: `5ae05cba4efd4dd28ebf806df593e9aa`.
- Git: `b7a7710c5913b79d114e503a17748cfccf634a7a`.
- O52, 32 boots, 7,200 frames, 32 workers; no W&B metric update.
- Output directory: `eval_context_smoke_b7a7710c` under the O52 run.
- Completed successfully: 32/32 boots, zero crashes, FunctionCall result 0.
- Decoder p50/p95: 4.51/4.88 ms. Gameplay-loop wall time including worker
  startup and the profile interval: 329.75 s.

Warmed smoke intervals were 38.1, 38.3, 37.7, and 37.6 FPS with 32/32 workers.
This is about 6x the stopped run's 6.0–6.2 FPS, but below the cited historical
44 FPS interval. Exclude the 31.9 FPS interval ending at 15:27:33 PDT: it includes
the follow-up eight-second py-spy recording.

The follow-up recording had 688 samples: 266 in input validation, 111 in
ingestion, 82 in context construction/device transfer, 29 in `_decode`, and
200 elsewhere (including IPC and worker polling). These are stack sample
counts, not GPU kernel timings. They support retaining validation/IPC as the
next profile targets, not another history implementation.

Full 96-boot reruns were submitted after the smoke passed. Both use Git
`b7a7710c`, RTX PRO 6000, 32 workers, and explicit expected checkpoint hashes.
Their output directory is `eval_conditioned_rings_v1_step_0016384_s4` directly
under each run, with shared W&B metrics in `eval_conditioned`.

- O52 app: `ap-tdMhKBrANrkzAWy25V54p3`.
  Call: `fc-01M2VAA98HG6HV404G1DG0PM2A`.
  Launch: `aa154e68011e457ea6d67c6d51acefe3`.
- O56 app: `ap-GIPZM1TCmw7TpBGLbDExDd`.
  Call: `fc-01M2VAA7VG825TRE8FFH2A4WZY`.
  Launch: `1f850df0c0234e50b60900fb9c5e8539`.

Historical evidence is under `runs/<run>/checkpoints/eval96-step-0016384/`
in R2, not directly under the run root. Both historical artifacts use match-row
schema 6. Checkpoint SHA-256 values:

- O52: `16c702fe3964a59c2f26d88207ef90137d93c07bf67a5d4213f4fde5d25b8631`.
- O56: `c8a63b3d0413cf5e90df727b1f2ed9d49f586b75ec0629e4b7b6752b38192ecb`.

Both used 96 boots, 7,200 frames, 32 parallel workers, seed 0, masked player
identity, delay 2, replan interval 2, and horizon 4. Their schedule SHA-256 is
`a2202b353e3e769f2ab25e673226ef29fb6f949f4391c2b9f3003afdc7ce3c15`.
The old compile mode was `default`; the portable path uses `reduce-overhead`.
CPU allocation metadata also differs across the historical launches. Do not
treat the old artifacts as a perfectly isolated one-variable performance test.

## Remaining limits

The full reruns are still in progress at this handoff. Do not use the 32-boot
smoke score as the 96-boot conditioning result. Poll the two FunctionCalls,
verify successful exit and 96 completed boots, then inspect the persisted
protocols before comparing gameplay or wavedash rates.

The process driver's cumulative-FPS log has a separate variable-shadowing bug:
the acknowledgement loop overwrites `started` with a nanosecond timestamp.
Use interval FPS, not the displayed `-0.0 cumulative fps`. This logging defect
does not cause the throughput drop and was left outside this context change.

The adapter still exchanges observations and actions every frame. The historical
chunk path used fewer IPC round trips. This is intentional for actual applied
action history and single-owner transport; change it only with timing parity
tests and a profile that shows a material cost.

Validation occurs at both the adapter and public policy boundaries. It remains
unchanged here. Do not remove those checks without establishing which boundary
owns each invariant.

Historical truncation and corrected conditioning consume random draws
differently. Equal seeds do not imply equal trajectories. Compare paired boots
with uncertainty, not individual frame trajectories. An air dodge is not itself
a failed wavedash: report the existing jump-initiated-air-dodge heuristic and its
landing window explicitly. Historical evidence must pass protocol and artifact
checks before any before/after comparison.
