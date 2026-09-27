# Real-time netplay

The current module layout, frame contract, and measurement limits are documented
in [Inference runtime](inference.md). Local evaluation and nonblocking
netplay share prediction and scheduling contracts. Checkpoint formats are unchanged.

The following sections retain the historical qualification record. Their module
names, schema versions, and validation counts describe that earlier revision.

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
