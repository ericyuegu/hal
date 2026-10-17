# Live overlay and idle self-play implementation

October 16, 2026 (rollout on October 17 UTC). **Candidate rolled back.**
The matched runner and overlay were deployed during the authorized maintenance
window. A demo frame timeout failed acceptance. Production now runs the prior
matched releases, with eight healthy slots and its admission flag restored.
The existing daily queue budget still blocks new reservations. No training or
evaluation VM was started. The corrections below have not been activated.

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

## Initial candidate verification

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

## Production attempt and rollback

Candidate source: `b87285191bd12b4a351b5a2232ec02e7ed5b2b81`.
Image digest: `sha256:8c44bd052fd8db3c729342f051806965721d8e6f5fd617544cf3718d9874384a`.
The image derives from the prior deployed digest and preserves its inference
Python dependencies. It includes the new package data and installed controller
assets. The independent overlay bundle uses the same source revision.

After Google Cloud authentication was restored, admissions were paused. The
queue had zero queued and active reservations. The image, overlay bundle, host
units, and persistent VM metadata were switched together. The original units,
metadata, image, and overlay link were saved for rollback.

The new 059 engine qualified all eight streams with OBS and rendering active:
200 complete-path samples, p50 **13.717 ms**, p95 **15.184 ms**, p99 **15.687 ms**.
Both delays share this measurement. Delay 2 allows 33.333 ms; delay 3 allows
16.667 ms. The preceding runtime startup record measured p50 10.225 ms, p95
15.064 ms, and p99 15.883 ms. There was no active-game FPS baseline for that
release; no measured FPS improvement is claimed. Self-play reached **59.980 FPS**, frame-interval p95 **18.007 ms**,
and policy round-trip p95 **7.253 ms**. One Python model process and OBS were
present on the GPU. The actual OBS scene showed the website sprite, Medium,
port 1, physical controller, and live model value. OBS sent 60 FPS with no
network or encoding drops. Its startup render skips stopped increasing.

The first demo stopped after about one minute with a one-second frame-read
timeout. The supervisor cleaned up Dolphin, invalidated both streams, and
disabled further demos. All eight human workers stayed healthy. No demo was
uploaded or represented as a human reservation.

Two attempts to create the short operator acceptance reservation returned HTTP
503. The existing queue guard reported **35,208 / 35,000** daily command-write
credits. Neither request created a reservation. The limit was not raised or
bypassed. Creating and claiming a reservation costs 48 credits; durable phase
reports and cleanup would bring this short test to roughly 100–200 credits.
The guard is private queue policy in `web/netplay-api/src/budget.ts`, with a
separate recurring-work reserve. It resets by UTC day. Self-play, inference,
controller telemetry, and composition are local; they do not write game frames
to Cloudflare. The public queue still owns human reservations and stream leases.

The prior matched runtime `ffed25c3c944d55ed37bb5564d8956418171f193` and independent
overlay `b5b8dd08a910d2ae136633d486197fa564259695` were restored, including startup
metadata. Both delay budgets requalified at p99 **15.637 ms**. Eight slots were
ready, OBS was sending 60 FPS with no network drops or reconnect, and one model
process remained loaded. The admission flag was restored to open. The unchanged
daily credit guard still rejects new reservations until its UTC reset.

The independent overlay restart check preserved the loaded model and human
readiness. That production check ran after the demo stopped; it does not prove
an active-demo handoff. The live public handoff and repeated production match
transitions remain unverified. Do not activate the corrected candidate before
acceptance can finish.

Three older replay upload receipts still fail with `worker did not play this
game`. They predate this release and remain intact. The old runner also briefly
lost queue control before deployment and recovered all eight slots. Neither
condition was changed as part of the overlay work.

## Corrections after acceptance failure

The cancellation path used short `Console.step()` polls. Pinned libmelee flushes
controllers on each call, so a partial frame could cause extra input submissions.
Cancellation now polls inside a version-checked receiver dispatch wrapper.
`Console.step()` keeps its normal frame wait and game-start neutral flush.
Cancellation remains checked at intervals of at most five milliseconds.

An attempted fix that disabled flushing on retries failed the real emulator
startup checks: it also disabled libmelee's game-start neutral flush. That
approach was removed. A new full-match test exposed a one-second startup frame
budget; the demo now uses the normal five-second budget while retaining the
short cancellation checks and independent 600 ms watchdog.

The startup audit also found that a spawned worker can still share the
supervisor's process group before `setsid()`. The watchdog now kills individual
demo descendants in that group. It never signals the supervisor group. A real
spawn regression covers cancellation before initialization.

The expanded local integration checks combine queue cancellation, real Dolphin,
two real inference connections, and live rendering. Human claims stay closed
until Dolphin and both streams are gone. These checks use no Cloudflare calls
and a synthetic test policy; they do not replace production 059 qualification
or the public admission check.

## Corrected candidate verification

The four required format, lint, type, and unit commands were rerun. The final
unit run passed **1,889 tests**, with eight unchanged skips, 32 deselected, and
26 dependency warnings, in 162.13 seconds. Skip reasons are listed above.

The corrected emulator suite passed **seven tests**, zero skips, in 160.92
seconds. This includes local queue cancellation with real Dolphin and two
inference connections, 90 seconds of sustained play, a full match-to-menu
transition, cancellation during startup/menus/play, and hung-worker cleanup.
The real spawn watchdog regression passed separately in 2.00 seconds.

The sustained test measured **59.429 FPS**, synthetic-policy inference p50
**0.899 ms**, and p99 **4.550 ms** with live controller/value rendering. These
are local test-policy results, not a corrected production 059 qualification.
The short check measured 59.912 FPS. Both inference routes were empty and
Dolphin was gone before the local human claim gate opened.

The required replay/session integration command passed **eight tests**, six
deselected, zero skips, in 58.66 seconds. Seven existing dependency warnings
concern fork in a multithreaded process. Formatting (338 files), lint, and the
zero-diagnostic type target passed. The corrected wheel passes asset hash
comparison and isolated installed renderer/session imports. Exact artifact
hashes are kept in the evidence directory.

An initial focused rerun lacked socket permission (13 failures); it passed with
local socket access. The first corrected wheel build lacked network access to
build requirements; the approved retry and actual isolated-wheel check passed.
No required fixture was missing.

Evidence: ignored `runs/netplay/release-20261016-live-overlay/`, including every
validation log, original and restored VM metadata, source archives and hashes,
image build/push logs, budget results, demo failure record, and an actual OBS
scene screenshot. Unrelated website changes remain outside the release.
