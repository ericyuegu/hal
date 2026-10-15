# Netplay Edge Queue — Plan 5: Twitch Streaming

**Goal:** Lease one Twitch stream to one live runner, prefer that runner's slot 0 for the next reservation, capture its dedicated display and audio with NVENC, and keep an informative idle card visible between games without ever exposing a player's connect code.

**Architecture:** The Durable Object owns a single stream row and returns a typed grant in session status responses. The runner asks for streaming by default. It starts one Xvfb per slot and one PulseAudio daemon for the streamed box, gives slot 0 a 1280×720 display and audio, and keeps other slots at 640×480 with audio disabled. A thread supervises ffmpeg while the lease exists. The stream slot writes a small local state file with only HAL settings and game count; the supervisor renders that state or an idle card into ffmpeg's reloadable overlay file.

**Tech Stack:** Cloudflare Durable Objects with SQLite, TypeScript 5.9 and vitest, Python 3.14, Xvfb, PulseAudio, ffmpeg with `h264_nvenc`, Dolphin/Slippi 3.6.4, pytest.

**Spec:** `docs/superpowers/specs/2026-09-27-netplay-edge-queue-design.md`

The requested `superpowers:writing-plans` and `superpowers:subagent-driven-development` skills are not installed in this session. The owner directed implementation without subagents. This plan follows the established plan format and is executed directly.

## Global Constraints

- Work only in `/home/ericgu/src/hal-edge-queue` on `netplay-edge-queue`.
- Do not deploy a Worker, change DNS, push an image, create cloud resources, push Git, or start a real Twitch stream.
- A manual Twitch check is an owner action and must use `?bandwidthtest=true`.
- `TWITCH_STREAM_KEY` exists only as a Worker secret and in the current lease holder's memory and ffmpeg command. Do not log it or persist it.
- Every runner route stays idempotent. The Python client retries every request.
- The Durable Object schema changes from version 1 to version 2. There is no migration. Old local state must be wiped.
- The stream lease belongs to one live, non-draining session and always names slot 0. Drain, explicit end, and 30 seconds of silence free it.
- While a live stream slot is idle, claims from all other slots return `204`. A held lease is still returned before this preference check.
- FIFO order does not change. The added delay is at most the runner's 250 ms claim interval.
- The overlay is built only from policy character, imitation, desired return, game number, site name, and public queue depth. Its API accepts no player connect code.
- Detailed measurements stay local. R2 receives only replay `.slp` files and their small metadata JSON.
- Commit messages have no attribution trailer. Do not push.

## File Structure

```text
web/netplay-api/src/env.ts, sessions.ts, queue.ts       stream lease and claim preference
web/netplay-api/test/sessions.test.ts, retries.test.ts  lease lifecycle, idempotency, preference
web/netplay-api/vitest.config.ts, .dev.vars.example     test binding and secret name
hal/netplay_service/queue_client.py                     StreamGrant and queue depth
hal/sim/session.py, hal/sim/netplay.py                  Dolphin display and audio boundary
hal/netplay_service/stream.py                           Xvfb, PulseAudio, overlay, ffmpeg supervision
hal/netplay_service/runner.py                           process lifecycle and slot 0 stream state
tests/test_netplay_queue_client.py                      strict grant parsing
tests/test_session.py, test_netplay_session.py          exact Dolphin config writes
tests/test_netplay_stream.py                            display, audio, overlay, ffmpeg, backoff
tests/test_netplay_runner.py                            grant gain/loss and slot configuration
deploy/netplay/Dockerfile, compose.yaml                 runtime packages and NVENC capability
deploy/netplay/README.md                                manual stream and performance checklist
```

---

### Task 1: Lease the stream in the Durable Object

**Files:** `web/netplay-api/src/env.ts`, `sessions.ts`, `queue.ts`, `vitest.config.ts`, `.dev.vars.example`, `test/sessions.test.ts`, `test/retries.test.ts`, `test/schema.test.ts`.

**Status response:** `{"draining": false, "stream": null}` or `{"draining": false, "stream": {"slot": 0, "key": "<secret>"}}`.

- [ ] **Step 1: Test first-status grant, repeated report, competing sessions, opt-out, drain, explicit end, silence, and admin status.** Assert only the holder receives the key and the key does not appear in events or admin output.
- [ ] **Step 2: Test claim preference.** Queue one job. Assert a non-stream slot receives `204` while the live stream slot is idle; slot 0 receives the oldest job; other slots resume once slot 0 holds a job. Assert a repeated claim still returns a held lease.
- [ ] **Step 3: Run `npm test -- sessions.test.ts retries.test.ts schema.test.ts` in `web/netplay-api`.** Expected: new tests fail.
- [ ] **Step 4: Add a one-row `stream` table to `SESSION_SCHEMA`, add it to the schema guard, and bump `STORE_SCHEMA_VERSION` to `2`.** Initialize row 1 with a null holder. Do not add a migration.
- [ ] **Step 5: Grant on the next status report from a live, non-draining session with `wants_stream = 1`.** Free the row in `drain`, `end`, and therefore `endSilent`. Return only the Boolean holder result from `SessionStore`; `Queue` adds `env.TWITCH_STREAM_KEY` to the response and refuses an empty secret.
- [ ] **Step 6: Add `deferToStreamSlot(session, slot)`.** It returns true only for a non-holder slot while the holder reported within five seconds, is not draining, and slot 0 holds no job. Call it after the idempotent held-lease check and before `claimNext`.
- [ ] **Step 7: Run `npm test` and `npm run typecheck` in `web/netplay-api`.** Expected: pass.

### Task 2: Parse stream grants in Python

**Files:** `hal/netplay_service/queue_client.py`, `tests/test_netplay_queue_client.py`.

**Interfaces:**

```python
@dataclass(frozen=True, slots=True)
class StreamGrant:
    slot: int
    key: str

@dataclass(frozen=True, slots=True)
class SessionState:
    draining: bool
    stream: StreamGrant | None
```

- [ ] **Step 1: Test null, valid, extra-field, wrong-slot, empty-key, and wrong-type grants.** Test `RunnerClient.queue_depth()` against the public capacity response.
- [ ] **Step 2: Run `uv run pytest -q tests/test_netplay_queue_client.py -k 'stream or queue_depth'`.** Expected: fail.
- [ ] **Step 3: Parse the exact grant shape.** Require slot 0 and a non-empty key. `queue_depth()` accepts the full capacity object but validates `queued` as a non-negative integer.
- [ ] **Step 4: Run the focused test again.** Expected: pass.

### Task 3: Configure Dolphin at its boundary

**Files:** `hal/sim/session.py`, `hal/sim/netplay.py`, `tests/test_session.py`, `tests/test_netplay_session.py`.

**Interface:** `set_dolphin_stream_output(console, enabled: bool)`. Enabled writes `EFBScale = 2`, a 1280×720 window at `(0, 0)`, and PulseAudio. Disabled writes the existing native scale and no-audio backend.

- [ ] **Step 1: Test exact `GFX.ini` and `Dolphin.ini` values for stream and headless sessions while preserving unrelated keys.**
- [ ] **Step 2: Run `uv run pytest -q tests/test_session.py tests/test_netplay_session.py -k 'stream or audio or resolution'`.** Expected: fail.
- [ ] **Step 3: Implement the direct config writes beside `set_dolphin_internal_resolution`.** Add a removal comment that names pinned libmelee 0.47.0 and says to remove the workaround when it exposes all window geometry fields. Pass `stream_output` through `NetplaySession`.
- [ ] **Step 4: Run `uv run pytest -q tests/test_session.py tests/test_netplay_session.py`.** Expected: pass.

### Task 4: Supervise displays, audio, overlay, and ffmpeg

**Files:** `hal/netplay_service/stream.py`, `tests/test_netplay_stream.py`.

**ffmpeg contract:** X11 at 60 fps; `hal_stream.monitor`; reloadable `drawtext`; `h264_nvenc`; CBR 6 Mb/s; 2 second GOP; AAC 160 kb/s; FLV to `rtmp://live.twitch.tv/app/<key>` with optional `?bandwidthtest=true`.

- [ ] **Step 1: Test Xvfb commands and cleanup.** Slot 0 is 1280×720×24 when stream-capable; every other slot is 640×480×24. Display numbers are unique and TCP is disabled.
- [ ] **Step 2: Test one PulseAudio command and environment.** It loads a `hal_stream` null sink and makes its monitor available through a private Unix socket.
- [ ] **Step 3: Test overlay state and rendering.** A game line uses only character, imitation, desired return, game count, and `20xx.xyz`. An idle line contains `Play HAL at 20xx.xyz` and queue depth. Construct a job with `CRYO#610` and assert that string never appears in the state JSON or overlay.
- [ ] **Step 4: Test the exact ffmpeg command and supervisor lifecycle.** Gain starts ffmpeg; loss terminates it; failures restart after 1, 2, 4, 8, 16, then 30 seconds; shutdown reaps the child. Assert no stream key is logged.
- [ ] **Step 5: Run `uv run pytest -q tests/test_netplay_stream.py`.** Expected: fail.
- [ ] **Step 6: Implement direct context managers and a single supervisor thread.** Use atomic local files. Do not add a general process framework.
- [ ] **Step 7: Run the focused test again.** Expected: pass.

### Task 5: Connect streaming to the runner

**Files:** `hal/netplay_service/runner.py`, `tests/test_netplay_runner.py`.

- [ ] **Step 1: Test CLI defaults.** The runner asks for a stream by default; `--no-stream` opts out; `--display-base` creates one display per slot; `--twitch-bandwidth-test` changes only the RTMP query.
- [ ] **Step 2: Test slot configuration.** Each child receives its own `DISPLAY`. Slot 0 uses stream output and `PULSE_SINK=hal_stream`; other slots disable audio. Slot restart keeps the same display.
- [ ] **Step 3: Test a reporter state sequence of null, grant, same grant, null.** Assert one ffmpeg start, no duplicate on the repeated grant, and prompt stop on lease loss and shutdown.
- [ ] **Step 4: Write stream-slot state before each game and idle state after each reservation.** The serialized fields must not include `player_code`, `connect_code`, or bot account data.
- [ ] **Step 5: Run `uv run pytest -q tests/test_netplay_runner.py -k stream`.** Expected: fail.
- [ ] **Step 6: Start display and PulseAudio resources in `run`, outside inference-generation recovery.** Pass the display and stream flag through `SlotConfig`. Poll `SessionReporter.state()` in the existing supervisor loop and send grant changes to `StreamSupervisor`.
- [ ] **Step 7: Make `main` pass `wants_stream=not args.no_stream`.** Require a reporter and session client whenever streaming is enabled. Keep `--no-stream` available for the performance control and recovery.
- [ ] **Step 8: Run `uv run pytest -q tests/test_netplay_runner.py tests/test_netplay_stream.py`.** Expected: pass.

### Task 6: Package and prepare manual measurement

**Files:** `docker/entrypoint.sh`, `deploy/netplay/Dockerfile`, `compose.yaml`, `.env.example`, `README.md`, `tests/test_netplay_deploy.py`.

- [ ] **Step 1: Test that the image installs ffmpeg, PulseAudio, Xvfb, `xsetroot`, and a font; Compose exposes NVIDIA video capability; the entry point does not start an extra Xvfb.**
- [ ] **Step 2: Add the runtime packages and update local and host examples.** Add `TWITCH_STREAM_KEY` only to Worker secret documentation. Do not put it in the runner dotenv.
- [ ] **Step 3: Add the owner manual stream checklist.** It must use `--twitch-bandwidth-test`, check idle/game transitions and audio, force one ffmpeg restart, verify lease handoff, and inspect the picture for connect codes.
- [ ] **Step 4: Add control/treatment tables for the RTX 3060 and first G4.** For streaming off/on and stream slot/headless slot, record game FPS, frame-interval p95, Dolphin-step p95, and policy-round-trip p95. Mark unrun cells pending. Explain that the owner must fill them from the same Git SHA, policy, peer, stage, delay, and slot count.
- [ ] **Step 5: Run `uv run pytest -q tests/test_netplay_deploy.py`.** Expected: pass.

### Task 7: Handoff checks and Plan 5 status

**Files:** `docs/superpowers/specs/2026-09-27-netplay-edge-queue-design.md`, `deploy/netplay/README.md`.

- [ ] **Step 1: Run `uv run ruff format --check .`.** Expected: pass.
- [ ] **Step 2: Run `uv run ruff check .`.** Expected: pass.
- [ ] **Step 3: Run `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts`.** Expected: zero diagnostics.
- [ ] **Step 4: Run `uv run pytest -q -m "not integration"`.** Expected: pass.
- [ ] **Step 5: Run `npm test` and `npm run typecheck` in `web/netplay-api`.** Expected: pass.
- [ ] **Step 6: Run `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_netplay_queue_integration.py -m integration`.** Expected: pass.
- [ ] **Step 7: Run `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration`.** Expected: pass with all fixtures present.
- [ ] **Step 8: Record every command, result, failure, and skip in `deploy/netplay/README.md`. Update the Plan 5 row in the spec to `done; manual stream check owner-gated`. Run `git diff --check`, review the final diff, and commit with `git commit -m "Add leased Twitch streaming"`. Do not push.**
