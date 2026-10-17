# Live overlay and idle self-play implementation

October 16, 2026. Implementation is in the working tree. **Not deployed.**
Production still runs the prior matched runner and overlay releases. Admissions
were not paused, and no compute instance was started or stopped.

## Behavior

The overlay retains the approved white/teal layout and fonts. It uses exact
website sprites, actual submitted conditioning for difficulty and imitation,
physical controller readback, full controller geometry, and a horizontal model
value display. Custom and Unconditioned settings are explicit. The value uses
the existing six-frame EMA; only the ±120 bar is clipped.

Telemetry uses a versioned, nonblocking local Unix datagram socket. Samples
carry game, inference generation, frame, sequence, monotonic time, physical
inputs, submitted conditioning, and the existing returned value. Gameplay
callbacks perform no rasterization or file writes. Inputs expire after 250 ms;
values expire after one second. Game changes clear both. Pillow composes cached
controller layers in the overlay process. One OBS image source alternates two
PNG buffers in `/dev/shm`. Sprites, controller SVG/PNG layers, fonts, and licenses
are included in the wheel, image source, and independent overlay bundle.

Production startup enables `--idle-self-play`; its launch scripts now require
2–16 slots. The demo uses two dedicated connections to the same qualified model
engine, one Dolphin, return 120, temperature 1, and physical delay 2. Rotation
uses a recorded seed; records include model sampling seeds and artifact hashes.
It creates no human reservations and uploads no demo replays.

Queue push events cancel the demo. Local claims stay closed until Dolphin and
both inference streams release. A 600 ms watchdog kills a hung worker, checks
all Dolphin process groups, invalidates both streams, and disables demos until
restart. Human workers wait on queue events. OBS is retained during handoffs.

## Verification

All commands use `UV_CACHE_DIR=/tmp/hal-uv-cache` because the default cache is
read-only in the sandbox. Socket and emulator checks used approved local socket
access. Unit overlay tests use `HAL_OVERLAY_BUFFER_DIR=/tmp`. Emulator checks
run under Xvfb with software OpenGL; they do not start a training/evaluation VM.

| Command | Result |
| --- | --- |
| `uv run ruff format --check .` | Passed: 338 files. |
| `uv run ruff check .` | Passed. |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts` | Passed: zero diagnostics. |
| `uv run pytest -q -m "not integration"` | 1,886 passed, eight skipped, 30 deselected; 26 existing warnings; 163.68 seconds. |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration` | Eight passed, six deselected, zero skips; 58.02 seconds. |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q -rP tests/test_idle_self_play_integration.py -m integration` | Five passed, zero skips; 34.17 seconds. |
| Focused control/runner/telemetry checks | Final groups: 105 passed and 99 passed. |
| OBS, alternating buffer, and overlay asset checks | 33 passed; two existing WebSocket warnings. |
| `uv build --wheel --out-dir /tmp/hal-overlay-wheel-final` | Passed. Every packaged asset matches the source. An isolated wheel extraction imports and constructs the renderer. |
| `bash -n deploy/netplay/gce-up.sh deploy/netplay/gce-startup.sh deploy/netplay/gce-overlay-install.sh` | Passed. |
| `git diff --check` | Passed. |

The eight unit skips are:

- Six tests in `tests/test_policy_world_schema.py`: the optional local v7 subset
  is absent.
- `tests/test_netplay_hardware.py::test_declared_profiles_meet_complete_path_budget`
  and `tests/test_netplay_hardware.py::test_prepared_profiles_do_not_compile_or_capture_during_play`:
  opt-in production GPU qualification was not enabled locally.

No required emulator fixture was missing. The new emulator tests cover startup,
menu and active-play cancellation, two inference connections with live rendering,
and a deliberately hung worker with Dolphin in a separate process group. Unit
checks cover claim barriers, global human activity, control/lease loss, repeated
handoffs, stream invalidation failure, stale telemetry, conditioning accuracy,
physical buttons, smoothing, clipping, asset identity, and overlay restarts.

Local emulator throughput with the controller/value renderer active was
**59.895 FPS**. The synthetic neutral-policy transport measured p50 **0.806 ms**
and p99 **2.256 ms** across the two inference connections. These are test-policy
measurements, **not 059 GPU model qualification**. Active-play cancellation and
normal resource release measured about **0.37 seconds** in the local smoke test;
the real hung-worker test also enforced the one-second cleanup limit.

## Failures found and resolved

The first `uv run` could not write the default cache. The first socket tests were
blocked by the sandbox; they passed with local socket access. Initial overlay
assertions still expected the removed difficulty behavior and old stream schema.
Those expectations now cover the new version and labels.

Focused tests exposed equal-timestamp telemetry ordering and a timeout bound
captured as a default argument. Explicit sample sequences and a runtime timeout
lookup fixed both. An initial cancellation test omitted `dolphin_path`; it was
corrected. A scratch emulator script needed its multiprocessing main guard.
The initial isolated-wheel check selected a Python without Pillow; the corrected
check uses the project interpreter with the extracted wheel first on its path.

The initial menu cancellation integration check took 1.003 seconds. The pinned
libmelee handshake blocked in `recv()` without checking cancellation. A bounded,
version-checked wrapper now polls cancellation. A full receiver pipe that ignored
SIGTERM also exceeded one second; cancellable sessions now bound both receiver
shutdown grace periods. The existing default shutdown behavior is retained.

The first required replay integration run failed at frame 129. The unchanged
HEAD session module reproduced the failure at frame 185. The fresh-recording
parity test used nonblocking input, so host scheduling could change when an input
landed. Both sides of that fixture now use blocking input. The focused parity
check and the full required integration suite pass. No simulation rule or policy
representation was changed to make the comparison pass.

An early full unit run used old capacity assertions while the production launch
minimum changed to two slots: six assertions failed. The final full run passes.
All intermediate and final logs are retained in the evidence directory below.

## Production acceptance still required

The current production baseline was read without running another model:
8-stream shared qualification p50 **10.225 ms**, p95 **15.064 ms**, p99
**15.883 ms**. Both delays use the same measurement, with limits of 33.333 ms
and 16.667 ms. This is the prior release's qualification, not this implementation.

Local `gcloud` credentials require reauthentication. Direct SSH to the recorded
VM succeeds with its existing GCE host key, but the VM service account lacks
`compute.instances.get` and cannot manage startup metadata. A persistent matched
release therefore still needs operator `gcloud auth login`.

After access is restored: commit/package the matched source, pause admissions,
wait for zero queued/active humans, deploy both releases and startup metadata,
and restart in the authorized maintenance window. Requalify both delay budgets,
measure the 059 path with the display active, verify actual human handoff and OBS
continuity, and reopen admissions. Restore the prior matched release if acceptance
fails. Do not start any training/evaluation VM. Unrelated website changes remain
outside this implementation.

Evidence: ignored `runs/netplay/release-20261016-live-overlay/`, including logs,
production baseline, a layout preview, and a reproducible source archive.
