# 059 runtime controls

The official dense control came from Git source `d7454f9d1a136f7c745d2d4478af22e19cd65faf` in `/tmp/hal-059-control-d7454f9`. The archived [capture helper](../../../archive/scripts/capture_official_chunks_control.py) uses the baseline 059 test fixture, Torch seed 19, decode seed 3, physical delay 0, fixed prefix 2, replan interval 2, horizon 4, and 26 four-action chunks beginning at observation frame 10. Its file SHA-256 is `dbd3fbbef3082ab563cd7562adcacf2756b07033aa90aae6aafc065c9c47306f`.

Run the control from that checkout with the pinned project environment:

```bash
cd /tmp/hal-059-control-d7454f9
PYTHONPATH=. /home/ericgu/src/hal/.venv/bin/python /home/ericgu/src/hal/archive/scripts/capture_official_chunks_control.py > /home/ericgu/src/hal/runs/refactor-059/official-26-chunk-control.json
```

The control and candidate raw records are `runs/refactor-059/official-26-chunk-{control,candidate}.json` (file SHA-256 `392329141bb7c24c35b760ffdecf14c27ff1a7a0590c2a2012bd581301a899b4` and `cd5de0cdddf0d8364827e20cf1b21cc6c7c0fba706dfccbf4a43c3673d1a0a9c`). Their 26 × 4 × 14 float32 action arrays are exactly equal. Both arrays hash to `201e83608dfaf194244d22b7036962feb98b178a8e7254332ce8f94d52255699`. The maintained regression is `tests/experiments/test_059_muon_action_sequence.py::test_official_process_chunks_match_pre_refactor_control`.

The local replay and session integration command is:

```bash
HAL_REQUIRE_INTEGRATION=1 .venv/bin/pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration
```

The restricted sandbox failed before gameplay because libmelee could not create an ENet host (`MemoryError`); its first-failure log is `runs/refactor-059/runtime-integration-sandbox-first-failure.log`. The same command with local port access passed 7 tests, deselected 2, and emitted 6 Python fork warnings in 53.57 seconds. Its log is `runs/refactor-059/runtime-integration-escalated.log`. No test fixture was skipped as missing.

Focused lifecycle checks cover bounded request timeouts and delivery-thread cleanup, prepared-profile admission, stream release/rematch routing, failed capture rejection, slot failure forfeit without another claim, partial pipe-allocation cleanup, and process-driver reaping. The corresponding tests are `tests/test_inference_client.py`, `tests/test_inference_engine.py`, `tests/test_window_policy_preparation.py`, `tests/test_netplay_runner.py`, and `tests/test_process_vec.py`.

The first compiled, separate-process RTX 3060 netplay smoke is recorded in [3060-functional-smoke-20260927-1](../../../runs/refactor-059/3060-functional-smoke-20260927-1/manifest.json), with [raw result](../../../runs/refactor-059/3060-functional-smoke-20260927-1/run-result.json) and `runs/refactor-059/3060-functional-smoke-20260927-1.log`. Both delay-2 and delay-3 prediction shapes prepared, and service readiness took 94.25 seconds. The match did not start: both Dolphin processes lacked an X display and failed GTK initialization. The harness was stopped with SIGINT; shutdown left zero live qualification descendants. This run is a failed functional smoke, not a latency or gameplay result. A separate existing live service owns one of the two local accounts, so no second smoke was started with that account.

The later runtime regression command was `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/pytest -q tests/test_netplay_driver.py tests/test_netplay_runner.py tests/test_qualify_netplay_059.py`: 47 passed. It covers the two-second no-usable-plan watchdog before any accepted plan, match-local forfeit with a healthy stream release, escalation on failed release, sidecar worker identity, and descendant cleanup. The harness now fails before preparation without `DISPLAY`, uses separate Slippi ports by default, and records hashes for every maintained `hal/**/*.py` source plus itself.

The [idle-service failure manifest](../../../runs/refactor-059/3060-idle-faults-20260927-2/manifest.json), [raw report](../../../runs/refactor-059/3060-idle-faults-20260927-2/report.json), and `runs/refactor-059/3060-idle-faults-20260927-2.log` record the opt-in GPU fault test. It used account B, one empty isolated queue, port 51461, compiled inference, and no Dolphin session, Slippi login, or reservation. The runner reached ready in 32.148 seconds. After `SIGSTOP` to its verified GPU child, that child terminated in 1.957 seconds and a replacement reached prepared readiness in 31.081 seconds. After `SIGKILL` to the replacement, the runner exited unavailable in 1.157 seconds. There were zero reservations, no live descendants, and matching source hashes before and after. The pre-existing live service retained its original PID and ready 1/1 status. The first fixture attempt, `3060-idle-faults-20260927-1`, failed before preparation or signal injection because health-status reads wrap an absent startup file in `ValueError`; the fixture now treats only that file-absence cause as pending, and its regression test rejects malformed status content.

The hardware command used `HAL_REQUIRE_IDLE_NETPLAY_FAULTS=1`, `HAL_NETPLAY_POLICY` pointing to the capability-v2 bundle, `HAL_NETPLAY_IDLE_ACCOUNT` pointing to account B, and `HAL_NETPLAY_IDLE_OUTPUT` pointing to the immutable run directory with `pytest -q tests/test_idle_runner_faults_hardware.py -m integration`. One hardware test passed and two non-integration tests were deselected in 67.33 seconds. This checks an idle engine and supervisor; it does not replace live-match fault injection or the long-running netplay soak.
