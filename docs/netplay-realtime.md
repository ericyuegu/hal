# Real-time netplay

The live runner and `hal-play` use nonblocking Dolphin at speed 1.0. Offline
and historical synchronous evaluation retain their original execution settings.
The training configuration and checkpoint formats are unchanged.

The timing contract is `B = C + 1`, `D = B + T`, `R = B`, and `H >= D + R`.
An action at offset +1 produces the next state. With `C=1, T=2, H=8`, offsets
+1 through +4 are forced, new actions start at +5, and two actions remain in
reserve. The worker uses absolute game frames, retains actual observations,
and submits only for the latest observation after draining buffered frames.

One request may be outstanding per stream. Early results wait for their
handoff. Late results skip expired submission deadlines and retain a reachable
suffix, with conditioning mismatches recorded. An exhausted plan uses neutral;
slow inference does not terminate a game. Confirmed inference loss drains the
available plan, flushes neutral, and records a service failure and bot forfeiture.
The runner removes its availability record and exits; its supervisor must restart
it, which runs calibration again before workers can claim reservations.

Calibration measures each feasible contiguous horizon/prefix shape through the
same request transport and batching path used in play. Each configured transport
delay receives 20 warmup and 200 measured calls at the configured slot count.
Compilation is excluded and forbidden during measured calls and live serving.
The selected shape is qualified again. The adjacent `.calibration.json` records
samples, schedule, resolved sampling seed, model bundle hash, Git SHA, runtime
configuration, hardware, and software versions. History and policy random streams
are reset before serving. No feasible schedule means no available server.

Slot health schema 2 and runner health schema 3 include the schedule, missed
deadlines, conditioning mismatches, exhausted chunks, neutral fallback frames,
and transport corrections. Existing frontend service status displays sustained
degradation; it does not terminate a match. Deadline counts include skipped
new-plan offsets and frame opportunities lost while draining observations.

The pinned libmelee source archive and its patch are in `vendor/`. It adds
`Console.step(flush_controllers=False)`, including suppression of game-start
and rollback writes. Defaults remain backward compatible. The same patch was
applied to the separate libmelee checkout without changing its unrelated edits.

Figures 1 and 14 and the article are maintained in the separate `ericyuegu2`
checkout. Both figures use `src/figures/timing-math.mjs`; Figure 14 no longer
extracts its active markup or renderer from the archived bundle.

## Qualification status

The following qualification record predates the local GPU and account setup.
It does not describe the current host state. The later O59 measurements and
completed netplay games are recorded in `docs/kv-cache.md`.

At the time of this initial record, the netplay test environment lacked
`HAL_NETPLAY_POLICY`, `HAL_NETPLAY_USER_JSON_1`, `HAL_NETPLAY_USER_JSON_2`,
`HAL_NETPLAY_CONNECT_CODE_1`, and `HAL_NETPLAY_CONNECT_CODE_2`.

Treatment: real-time chunk scheduling with calibrated inference/delivery budget.
Control: the original synchronous, blocking, uncapped execution path.
Invariant inputs for live qualification: checkpoint, seed, player identities,
match setup, Slippi transport delay, emulator build, and hardware/software stack.
No before/after live FPS or frame-latency claim came from that initial record.
The injected-delay test writes per-account frame intervals, request latency,
deadline misses, controller placement checks, and observed progress during
outstanding inference to its pytest temporary directory.

## Validation commands and results

- `uv run ruff format --check .`: passed (350 files).
- `uv run ruff check .`: passed.
- `uv run ty check --python-version 3.14 --error-on-warning hal experiments/051_muon_parameterization.py scripts/cache_modal_fixtures.py scripts/launch_gce.py scripts/launch_modal.py scripts/launch_vast.py scripts/replay_policy_fault.py`: passed, zero diagnostics.
- `uv run pytest -q --ignore=tests/experiments -m "not integration"`: **1,194 passed, 22 deselected**, no skips. 29 dependency/multiprocessing warnings.
- `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration`: **7 passed, 2 deselected**, no skips. Seven multiprocessing/fork deprecation warnings.
- `HAL_REQUIRE_NETPLAY_INTEGRATION=1 HAL_REQUIRE_NETPLAY_POLICY_INTEGRATION=1 HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1 uv run pytest -q tests/test_netplay_integration.py tests/test_netplay_policy_integration.py tests/test_netplay_hardware.py -m integration`: **10 failed, 2 fixture errors**, no skips. Missing required fixtures are failures.

The netplay failures are:

- Two original impulse-placement cases, delays 2 and 3: missing account fixture 1.
- Two new real-time inference/delivery overrun cases, delays 2 and 3: missing account fixture 1.
- Four original compiled-policy hardware cases, slot counts 1/2 and delays 2/3: missing policy bundle.
- Two new calibrated chunk hardware cases, slot counts 1 and 2: missing policy bundle.
- Two full-game chunk-policy cases, delays 2 and 3: fixture setup errors from the missing policy bundle.

`nvidia-smi --query-gpu=name --format=csv,noheader` failed because it could not communicate with the NVIDIA driver.

Blog checks:

- `pnpm check`: passed, zero errors/warnings; 12 existing hints in other figure scripts and the sync script.
- `node --test tests/*.test.mjs`: 8 passed, no skips.
- Sites `build-site.mjs` / `pnpm run build`: passed.
- `pnpm run build:private`: passed; includes the revised article and figures.
- Chromium checks at 390px and 900px: both figures rendered without page errors or page overflow; early, late, exhausted, and unavailable states passed. Screenshots inspected.

Resolved intermediate failures: missing optional server packages during a type
check; typed health payload parsing; the O59 decoder's old four-head runtime
guard; a countdown callback mock; one Modal dependency-layer mock; one missing
`zip(strict=...)`; and formatting. Initial blog commands could not find pnpm
(and one explicit PATH omitted Node); installing the declared pnpm version under
`/tmp` and using the existing Node installation resolved those command failures.
