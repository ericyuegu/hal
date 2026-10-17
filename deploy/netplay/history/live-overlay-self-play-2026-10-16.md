# Live overlay and idle self-play implementation

October 16, 2026 (rollout on October 17 UTC). **Activated.**
Production runs matching runtime and independent overlay
`37e7ec0d2b2f3188197b219babd585435997a5ef`, with eight healthy
slots and admissions open. Earlier candidates were rolled back while the cleanup
failures below were corrected. Final acceptance passed. The owner authorized a
bounded, date-specific queue-budget allowance. No training or evaluation VM was
started. The existing public human queue and stream lease remain in Cloudflare;
self-play creates no public queue jobs. A later Sheik selection failure was
corrected and requalified as recorded at the end of this file.

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

## Final cleanup corrections

Runtime `106cc6613a1e57f437c7689fe22eaafe0fc538f8` exits its dedicated demo worker
only after explicit Dolphin/inference cleanup and its final record write. Python
interpreter finalizers had otherwise kept a cleaned-up worker alive beyond the
600 ms watchdog. A spawned-worker regression installs a three-second exit
handler and checks cancellation during startup, menus, and play. Unexpected
exceptions still produce a nonzero exit.

The production process monitor then found six defunct AppImage descendants,
not six live games. Docker's Python PID 1 did not reap adopted children. Launch
source `bd17c48143b562506e54827aa3d4deacc5c7ab30` enables Docker init in production
and Compose. A CPU-only probe on the actual host reproduced the zombie without
init and confirmed reaping with init. The runner was restarted with init and
both delay budgets were requalified. Its systemd dependency also stopped OBS;
the overlay service was restarted. Stream continuity below refers to human
handoffs after maintenance, not this service restart.

The final image digest is
`sha256:8bdd72300be77ea7af14e709904f381aeed1be1d756ac7e6f8644bd78231bfcb`.
Installed image imports/assets and the independent bundle preflight passed.
The overlay uses private stream-state version 2 with this runtime. Updating the
final overlay did not restart the runner, model, or active demo.

## Final presentation

Overlay `63378a6a4d5558cafa5f9b87d86dd3ddb8187990` removes only the pinned
Dolphin window's 304-pixel pillarboxes, preserving the entire native Melee image
and proportional scaling. The live screenshot confirms the game starts at row
43 with the HAL/SSBM header and ends after row 1068 with the 20xx.xyz footer,
including antialiased text edges. It retains the website sprites and controller
readback, and labels the bar **State value**. The idle text says **playing right
now**, counting fresh local human game health and excluding waiting connections
and self-play. It makes no additional public queue calls.

The requested README count addendum was removed. Unrelated website and research
changes were excluded from the overlay bundle and commits.

## Final validation

| Command or check | Result |
| --- | --- |
| `uv run ruff format --check .` | Passed: 340 files. |
| `uv run ruff check .` | Passed. |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts` | Passed: zero diagnostics. |
| `uv run pytest -q -m "not integration"` | 1,953 passed, eight unchanged skips, 35 deselected, 26 existing warnings; 175.60 seconds. Includes concurrent unrelated research tests without releasing those changes. |
| `HAL_REQUIRE_INTEGRATION=1 xvfb-run -a uv run pytest -q -rP tests/test_idle_self_play_integration.py tests/test_roundtrip.py tests/test_session_cleanup.py -m integration` | 18 passed, six deselected, zero skips, 12 warnings; 242.11 seconds. |
| Final capture/value regressions | 38 passed; the final one-pixel footer correction then passed 28 focused tests. |
| Worker tests and TypeScript | All 112 tests passed; type checking passed. |
| Installed bundle activation | Imports, assets, stream schema, fresh heartbeat, and active output passed. |

The full unit command was interrupted during user steering before the completed
run above. An attempted detached launch also exited before executing tests; it
is not counted as validation. A temporary screenshot helper first used a missing
`python` executable; rerunning with `python3` produced the actual after image.
The final monitor analysis excluded playing samples whose FPS window had not
filled, after an initial aggregate rejected null values. This did not alter any
recorded measurements. Earlier failed checks and the eight skip reasons remain
listed above. No required emulator fixture was missing.

The new spawned-worker test initially supplied an integer state value, which
the inference boundary rejected; a float fixed the fixture. Ruff initially
rejected an empty test-cleanup exception handler and, later, an unused import;
both final checks pass. The first public canary helper read the obsolete flat
phase field. It canceled its reservation safely; the corrected helper reads
`observed.phase` without changing the public API.

## Final production acceptance

With Docker init enabled, both declared delays qualified at eight-stream load
with 200 samples: p50 **10.443 ms**, p95 **15.379 ms**, p99 **16.000 ms**.
Delay 2 allows 33.333 ms; delay 3 allows 16.667 ms. Both use prefix 4, replan 4,
and horizon 8. The prior release's p99 was 15.637 ms; no speedup is claimed.

Two earlier public canaries reached the human connection menu in 8.347 and
6.051 seconds and confirmed demo release within 0.396 and 0.374 seconds from
POST start. A control-disconnect check released the demo in **114 ms**, recovered
control, and resumed self-play without disabling demos. The command that closed
the sockets also printed an RTNETLINK error; the runner's actual control-close
and release timestamps establish the result.

A further canary with Docker init reached the menu in 6.708 seconds. After the
final overlay update, the last canary reached `waiting_for_player` in
**6.428 seconds**. Its old Dolphin was absent within **336 ms** of POST start,
and the supervisor confirmed full demo release within **406 ms**. It then
claimed the human reservation. The test canceled at the menu without connecting
a peer or playing a human match, and self-play resumed as game 10.

The final 80-second monitor recorded at most one Dolphin, no retained zombie,
and one unchanged OBS PID. All 79 OBS output samples were active, with monotonic
output duration, zero output skips, zero congestion, and no reconnect. The same
single GPU model process remained loaded. A prior 360-second monitor with init
also observed at most one Dolphin through repeated handoffs and match changes.
A transient defunct child was reaped before the next Dolphin started.

With the final controller and state-value display active, 58 measured playing
samples ranged from **59.854 to 59.972 FPS**, median **59.941 FPS**. The final
sample had frame-interval p95 **18.005 ms** and complete client round-trip p95
**7.044 ms**. These are two-stream self-play measurements. Eight simultaneous
human games and a connected human rematch were not tested. Reboot and managed
replacement remain unverified. Training and evaluation VMs remain stopped.

Admissions were reopened after these checks. Public capacity reported ready,
eight healthy slots, zero active and zero queued human reservations. The admin
flag was `paused=false`; recorded command spending was **35,920 / 37,000**.

## Temporary budget allowance

The owner authorized clearing the internal daily block for acceptance. API source
`2fb3b2e7` introduced a date-bound 1,000-credit allowance; source `aa50ad03` raised
that bounded allowance to 2,000 credits to cover the cleanup retest. Active API
version is `b150c6c2-dc9e-42d1-a0ad-b304c9a71b92`, with
`ACCEPTANCE_WRITE_BUDGET_DAY=2026-10-17`. Its limit is 37,000 command credits on
that UTC day and automatically returns to the normal 35,000 at **October 18,
00:00 UTC** (October 17, 5 pm Pacific). Existing spending is retained. The
200-credit admission reserve, database-size guard, protocol 4, schema 7,
`global-v7` identity, and public APIs remain unchanged. No provider plan changed.

Account analytics at the final check reported **13,644 Durable Object requests**,
**13,504 SQLite row writes**, **226,862 row reads**, and **6,054 Worker requests**,
with **zero Worker errors** for October 17 UTC. Command credits are conservative
internal accounting, not those provider totals. The measured request/write
counts are each well below their separate 100,000/day free limits. The public
queue remains in Cloudflare; local demo games do not enter it.

Evidence remains under ignored `runs/netplay/release-20261016-live-overlay/`.
Rollback images, independent overlay releases, and original host unit backups
were retained. No credentials were committed.

## Sheik selection failure and correction (October 17)

At 08:01:47 UTC, production demo game 14 exited because port 1 was Zelda
instead of requested Sheik. The recorded seed was 0; the selection was
Sheik/Ness on Fountain of Dreams. The character identity check rejected it,
all demo resources were released, and the supervisor disabled further demos.
Eight human slots stayed healthy, with no queued or active reservation. This
case was missing from the earlier acceptance coverage.

The stage retry in `Session._drive_menus` released A on alternate frames,
overriding libmelee's hold required for local Sheik selection. Runtime source
`37e7ec0d2b2f3188197b219babd585435997a5ef` keeps A held for Sheik while preserving
normal retries for other characters. The character check and failure policy
remain in place. New unit tests and real-Dolphin tests cover Sheik on either
port, two Sheiks, and two Zeldas. Before the fix, two of four cases failed in
both groups with the same port-1 mismatch; after it, all four emulator cases
and all 26 Session unit tests passed.

Formatting (340 files), lint, and zero-diagnostic typing passed. The full unit
suite passed **1,958 tests**, eight unchanged skips, 39 deselected, and 26 warnings
in 215.37 seconds. The combined emulator run passed all 14 self-play tests and
seven other tests, but its instant-restart parity fixture diverged at frame 185.
That fixture still used nonblocking input, unlike the already corrected fresh
recording test. Both record and replay now use blocking input; no game logic or
comparison tolerance changed. The final required replay/cleanup command passed
**eight tests**, six deselected, zero skips, seven warnings, in 58.01 seconds.
The isolated prior Session also passed that timing-sensitive test on one rerun;
this is not evidence of a Sheik-related physics change.

The first isolated prior-code check lacked its relocated fixture paths; linking
the same existing fixtures corrected that setup. An initial unit attempt lacked
build-network access after concurrent packaging edits; the approved run passed.
A temporary image staging tag was corrected and checked against the image's
source label before publication. No mismatched image was published or activated.

The matching runner and overlay image is
`sha256:9a2c7c0b1b68703f65dbabebb1ef55e6f3f2e7df83e15d402179aa5d2a0411f9`.
Installed-image and mounted-bundle preflights passed. The release retains the
same 059 model artifacts, dependencies, eight slots, Docker init, overlay layout,
and private/public protocols. Startup metadata was updated. Admissions were
paused with no human reservations, then the matched services were restarted.
The prior units and overlay link were retained under
`/var/lib/hal-release-backups/sheik-20261017/`.

The corrected release qualified both delays at capacity eight with 200 samples:
p50 **10.690 ms**, p95 **15.304 ms**, p99 **15.924 ms** (the qualification's
nearest-rank p99). Both budgets passed, including delay 3's 16.667 ms limit.
Cold-start fixture preparation and compiled inference preparation completed;
self-play started again at 08:19 UTC.

The final public canary released all demo resources within **403 ms** from
POST start and reached the human connection menu in **3.431 seconds**. It was
canceled without a peer, and self-play resumed as game 2. The 80-second monitor
observed at most one Dolphin, no retained zombie, and one unchanged OBS PID.
All 79 OBS samples had active output, zero output skips, zero congestion, and
no reconnect. The same one GPU model process remained loaded across the handoff.
There were 61 measured self-play samples: **59.940 median FPS**, range
59.916–60.119. The final sample had frame-interval p95 **18.021 ms** and complete
client round-trip p95 **7.247 ms**. No connected human match was played.

Admissions reopened with eight healthy slots, zero active or queued human
reservations, and **36,088 / 37,000** internal credits used. The temporary budget
expiry is unchanged. Training and evaluation VMs remain stopped. Unrelated user
research and website edits remain uncommitted and were excluded from deployment.
The screenshot is `production-sheik-fixed.png` in the ignored evidence directory.

Existing pending replay sidecars still produce ownership-rejection warnings on
retry. They were retained for separate investigation; no ownership check was
weakened and no pending replay was deleted or relabeled as a demo/human game.
