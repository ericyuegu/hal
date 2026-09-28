# Netplay edge queue

Date: 2026-09-27
Status: design approved in conversation; awaiting written-spec review.

## Goal

Move the netplay queue and public API off the GPU box and onto Cloudflare, so
that GPU boxes are interchangeable, stateless workers. The site stays up when a
box crashes or is replaced. Bringing up a replacement box takes one command and
no per-box configuration.

A follow-on spec, the page redesign, builds on the API defined here. Its
agreed decisions are recorded at the end so that nothing is lost between the
two.

## Plans

| Plan | Scope | Status |
| --- | --- | --- |
| 1 | API Worker and Durable Object (`web/netplay-api`), golden transcripts | done |
| 2 | Runner and admin clients, retry-safe runner routes, shared queue contract | done |
| 3 | Runner cutover to `RemoteQueue`; delete the Python service; page and deploy scripts | not written |
| 4 | Host bring-up: image, `gce-up.sh`, G4 verification | not written |
| 5 | Twitch streaming: stream lease, claim preference, displays, ffmpeg | not written |

Plans 1, 2, and 3 run in order. Plans 4 and 5 each need Plan 3.

## Non-goals

- No change to the queue's behavior visible to players. The port keeps every
  state, transition, deadline, and error code (see "State machine port").
- No change to gameplay, inference, scheduling, or the policy bundle format.
- No Modal support. Modal is deferred until the local and Google Cloud paths
  work.
- No bot check (Turnstile). Rate limits and a queue cap are the only abuse
  controls on player routes.
- No backward compatibility. Nothing has been deployed, so old databases,
  status files, and environment files are discarded, not migrated.

## Decisions

| Decision | Choice |
| --- | --- |
| Where the queue lives | One SQLite-backed Durable Object behind an API Worker |
| Policies across boxes | One active policy at a time; runners with another bundle are refused |
| Hostname | Page at `20xx.xyz`; API at `20xx.xyz/v1/*`; optional 301 from `hal.ericyuegu.com` |
| Runner transport | HTTPS for state transitions; one WebSocket per active job for pushed settings |
| Runner and admin auth | Cloudflare Access service tokens plus a Worker-checked bearer token |
| Player abuse controls | Per-IP rate limit on job creation, queue-length cap, no Turnstile |
| Box configuration | Three secrets plus `--slots N`; everything else is fetched and hash-checked |
| Bot Slippi accounts | Leased to sessions by the queue |
| Hosts | This machine (RTX 3060) and Google Cloud G4 (1x RTX PRO 6000 Blackwell); Modal later |
| Twitch stream | One stream at a time, leased by the queue to one session, which streams its slot 0 |
| Stream idle behavior | New reservations go to the streamed slot first; an idle card shows between games |
| Stream consent | Every game on the streamed slot is broadcast; the page says so; the overlay never shows a connect code |
| Stream audio | Game audio on the streamed slot only; other slots stay silent and headless |

## Architecture

```
browser ─▶ 20xx.xyz/*     page Worker (web/netplay, vinext, static client)
        ─▶ 20xx.xyz/v1/*  API Worker (web/netplay-api, TypeScript)
                            └─▶ Durable Object "queue" (single instance, SQLite)
                                  jobs · games · sessions · accounts · stream · policy · events

GPU box: hal-netplay-runner ── outbound HTTPS + WebSocket ──▶ 20xx.xyz/v1/runner/*
         downloads policy, ISO, emulator, account JSON from private R2
         uploads replays, measurements, audit records to private R2
         stream holder only: ffmpeg ── RTMP ──▶ twitch.tv
```

The page and API share an origin, so the API sends no CORS headers and the page
needs no build-time API URL. GPU boxes accept no inbound connections.

## API Worker and Durable Object

### Layout

`web/netplay-api/` is a new Worker project:

- `src/index.ts`: routing, auth checks, rate limiting; forwards to the Durable Object.
- `src/queue.ts`: the Durable Object class and its SQLite schema.
- `src/state.ts`: the job state machine as plain functions over a storage handle.
- `wrangler.jsonc`: route `20xx.xyz/v1/*`, the Durable Object binding and its
  migration, a rate-limit binding, and secret names.
- `test/`: vitest tests in Cloudflare's local Workers runtime, and `test/transcripts/`.

### Storage

All queue state lives in one Durable Object instance, addressed by a fixed name.
Requests to it run one at a time, and each request runs in one storage
transaction.

Tables (schema version 1 of the new store; no migrations):

- `jobs`, `games`: the same columns and meanings as the current
  `hal/netplay_service/queue.py` v3 schema.
- `sessions`: `id`, `host` (free text from the runner), `bundle_sha256`,
  `git_sha`, `slots`, `wants_stream`, `started_at`, `last_seen_at`, `draining`,
  latest `RunnerStatus` payload, `ended_at`.
- `stream`: exactly one row: `session_id` (null when free), `slot` (always 0),
  `granted_at`. See "Streaming".
- `accounts`: `connect_code`, `r2_key`, `sha256`, `session_id` (null when free),
  `slot`, `leased_at`.
- `policy`: exactly one row. It holds the published policy config (see
  "Publishing a policy").
- `events`: `at`, `kind`, `job_id`, `session_id`, `detail` (JSON). Rows older
  than 30 days are deleted by the alarm.

### Alarm

One alarm is always set for the earliest pending deadline:

- connect deadline (becomes `no_show`)
- rematch deadline (completes the reservation)
- job lease expiry (retry or fail, as today)
- session silence (see "Sessions")
- event retention

When it fires, it processes everything due and sets the next alarm. This
replaces the API process's `reap_expired` loop.

## Routes

### Player routes (public)

Unchanged paths and JSON shapes:

- `GET /v1/options`: built from the published policy config.
- `GET /v1/capacity`: aggregated from live sessions' `RunnerStatus` payloads,
  using the same rules as `aggregate_runner_status` today. Status `unavailable`
  when no session is live.
- `POST /v1/jobs`, `GET /v1/jobs/{id}`, `PATCH /v1/jobs/{id}/policy`,
  `DELETE /v1/jobs/{id}`, `POST /v1/jobs/{id}/rematch`: bearer job token as today.
  The server stores only a SHA-256 digest of each token.

New errors:

- `429` with `Retry-After` when one IP creates jobs too fast (Workers rate-limit
  binding; initial limit 5 creations per minute).
- `503` when the queue holds its cap of waiting jobs (initial cap 20) or when no
  policy has been published.

### Runner routes (`/v1/runner/*`)

Every request must pass Cloudflare Access with a service token, and must carry
`Authorization: Bearer <runner token>`. The Worker compares the token's SHA-256
against the set in the `RUNNER_TOKEN_SHA256` secret in constant time. Each box
has its own token.

| Route | Purpose | Replaces |
| --- | --- | --- |
| `POST /v1/runner/sessions` | Start a session: `{session_id, protocol_version, host, bundle_sha256, git_sha, slots, stream}`, with a runner-chosen ID. Returns the session ID, the policy config, and one leased account per slot. A repeat with identical fields returns the same session. `409` if the runner protocol version differs from the Worker's, if the bundle is not the active policy, if too few accounts are free, or if the ID exists with other fields. | runner startup and per-box account config |
| `POST /v1/runner/sessions/{sid}/status` | Report the `RunnerStatus` payload, about every 2 s. Doubles as the session heartbeat. The response is `{draining, stream}`: `stream` is `null` unless this session holds the stream lease, and then it carries the stream key. | status files read by the API |
| `POST /v1/runner/sessions/{sid}/claim` | `{slot}` → a job or `204`. Returns the slot's `leased` job if it holds one, so a repeat cannot take a second job. Refused while the session is draining, or while the slot holds a job past `leased`. Applies the stream-slot preference (see "Streaming"). | `claim_next` |
| `POST /v1/runner/sessions/{sid}/drain` | Stop claiming; keep current sets. | local stop flag |
| `DELETE /v1/runner/sessions/{sid}` | End the session: fail its remaining leases with the current generation-abort codes and free its accounts. A repeat returns the first result. | `fail_worker_generation` |
| `POST /v1/runner/jobs/{id}/heartbeat` | Extend the job lease. | `heartbeat` |
| `POST /v1/runner/jobs/{id}/connecting` | `{connect_code}` | `mark_connecting` |
| `POST /v1/runner/jobs/{id}/playing` | | `mark_playing` |
| `POST /v1/runner/jobs/{id}/no-show` | | `mark_no_show` |
| `POST /v1/runner/jobs/{id}/no-contest` | | `mark_no_contest` |
| `POST /v1/runner/jobs/{id}/finish-game` | `{game_number, actual_stage, result}` | `finish_game` |
| `POST /v1/runner/jobs/{id}/fail` | `{error_code, retryable}` | `fail` |
| `POST /v1/runner/jobs/{id}/forfeit` | | `forfeit_service_failure` |
| `POST /v1/runner/jobs/{id}/replay` | `{game_number, key, sha256, size, etag}`. Accepted only from the worker that finished that game, even after its session ended. | `record_replay` |
| `GET /v1/runner/jobs/{id}` | Current job for the owning worker. | `get_worker_job` |
| `WS /v1/runner/jobs/{id}/live` | Server pushes `settings {revision, desired_return, temperature}` on connect and on every change, and `released` when the job is canceled, expires, or is reassigned. | `_LivePolicySettings` polling SQLite |

Each job route names the session and slot (`X-HAL-Session`, `X-HAL-Slot`
headers), and the Durable Object derives the worker ID from them.

### Admin routes (`/v1/admin/*`)

Protected by a Cloudflare Access application tied to the owner's login, plus an
admin bearer token.

- `PUT /v1/admin/policy`: publish a policy config.
- `PUT /v1/admin/accounts`: replace the bot account list (connect code, R2 key,
  SHA-256). Accounts held by live sessions cannot be removed.
- `POST /v1/admin/pause`, `POST /v1/admin/resume`: stop or restart accepting new
  jobs. Waiting jobs keep their place.
- `GET /v1/admin/status`: sessions, slots, accounts, active jobs, queue depth.
- `GET /v1/admin/events?job=&session=&since=`: the event timeline.

## State machine port

The TypeScript store is a port of `queue.py`, not a redesign. It keeps:

- statuses and allowed transitions
- one active job per player connect code
- FIFO order with a retried job placed at the front; at most two attempts
- rematch: character, imitation, and stage may change; delay may not
- `cancel_after_game`
- `IDLE_TIMEOUT_SECONDS` (600) for both connect and rematch deadlines
- `service_failure_bot_forfeit`, `service_generation_aborted`, `lease_expired`
  and every other error code
- idempotent replay recording

Deliberate changes, each with its own test:

1. **Sessions replace runner generations.** A session that sends no status for
   30 s is ended as if `DELETE /sessions/{sid}` had been called.
2. **Longer lease grace while playing.** A job lease lasts 20 s in `leased`,
   `connecting`, `rematch_wait`, and `rematch_ready`, and 60 s in `playing`, so
   a short network outage does not fail a game that Dolphin is still running
   locally.
3. **Idempotent runner routes.** A retried transition that has already
   been applied returns `200` with the current job. `finish-game` is keyed by
   `game_number`, so a retry cannot record a game twice. A transition from the
   wrong state still returns `409`. Session start, claim, and session end are
   also safe to repeat (see "Runner routes"), because the client retries every
   request.
4. **Policy and account checks.** Sessions must match the active bundle, and
   each slot holds exactly one leased account.
5. **Rate limit, queue cap, and pause** (`429`, `503`).

## Runner changes

### Queue client

`hal/netplay_service/queue_client.py` adds `RemoteQueue`. It has the same
method names and arguments that `runner.py` calls on `QueueStore` today, so the
runner's reservation logic changes only where the store is constructed. It
uses `httpx`, which moves from the `dev` group to the runtime dependencies, and
`websockets` for the settings socket. Requests use bounded retries: exponential backoff from 0.25 s to 4 s for
connection errors and `5xx`, and no retry on `4xx`. A `409` from a transition
raises the same `InvalidTransitionError` the runner already handles. That class
and the `RunnerQueue` protocol live in `hal/netplay_service/queue_contract.py`.
`RemoteQueue` is its production implementation.

`_LivePolicySettings` keeps its interface. It reads settings from the job's
WebSocket instead of polling SQLite, reconnects with backoff, and sets
`released` on a `released` message or when `GET /v1/runner/jobs/{id}` reports a
terminal state after a reconnect.

### Startup

`hal-netplay-runner --slots N` needs only:

- `HAL_NETPLAY_API_URL` (for example `https://20xx.xyz`)
- `HAL_NETPLAY_RUNNER_TOKEN`, `CF_ACCESS_CLIENT_ID`, `CF_ACCESS_CLIENT_SECRET`
- R2 credentials (`AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)

Startup order:

1. Fetch the `ISO` and `NETPLAY_EMULATOR` fixtures through `hal/fixtures.py`.
   The ISO comes from `fixtures/ssbm.ciso` in R2. The emulator is the verified
   upstream Slippi Online 3.6.4 AppImage from its pinned GitHub release.
2. Read the active policy and start a session. The response carries the policy config and the leased
   accounts.
3. Start the status reporter immediately. It keeps the session alive during
   downloads and qualification.
4. Download the policy bundle and each leased account's `user.json` from R2,
   and verify their SHA-256.
5. Load the bundle and run the existing `check_realtime_budget` qualification.
   If it fails, end the session and exit non-zero with the measured timings.
6. Start the slot processes, which begin claiming.

Downloads are cached under `~/.cache/hal-netplay/<sha256>` and are reused only
after their hash is verified again.

### Shutdown

- First SIGINT or SIGTERM: call `drain`, stop claiming, let active sets finish,
  then end the session. A `--drain-timeout` (default 15 min) bounds the wait.
- Second signal, or drain timeout: forfeit active games with the existing
  codes, end the session, and exit.
- Crash: the Durable Object ends the session after 30 s of silence.

### Local artifacts

Replays upload to R2 as today, and each upload is then reported with
`POST /jobs/{id}/replay`. Measurement files and engine audit records are also
uploaded to R2 next to the replays, so a lost box does not lose them. The
Prometheus metrics endpoint moves into the runner on a local port.

## Streaming

Streaming is implemented in Plan 5. Until then the Worker grants no stream
lease, runners start sessions with `stream: false`, and every status response
carries `"stream": null`.

One slot streams to Twitch at all times. Dolphin always runs on the same box as
the GPU and model.

### Stream lease

- The Durable Object holds one stream lease. Sessions ask for it with
  `stream: true` at start; every box asks by default (`--no-stream` opts out).
- When the lease is free, the next status response to a live, non-draining
  session that asked for it grants it. The lease is freed when the holder's
  session ends by drain, `DELETE`, or 30 s of silence. A replacement box
  therefore takes over the stream without configuration.
- The grant carries the Twitch stream key, which is held only as the Worker
  secret `TWITCH_STREAM_KEY`. Boxes keep three secrets.
- The holder streams its slot 0. `/v1/admin/status` shows the holder.

### Claim preference

- A claim from any slot other than the stream slot returns `204` while the
  stream slot is live (its session reported status within 5 s and is not
  draining) and has no job. The stream slot's next claim, at most one claim
  interval later, takes the job.
- When no session holds the stream, or the stream slot has a job, claims work as
  before.
- FIFO order is unchanged. The added wait is at most one claim interval.

### Displays and audio

- The runner no longer runs under one `xvfb-run` wrapper. It starts one Xvfb per
  slot and passes that slot's `DISPLAY` to its Dolphin.
- Headless slots: 640×480 display, audio disabled, native EFB scale. This is
  today's behavior.
- Stream slot: 1280×720×24 display, EFB scale 2, render window sized and
  placed to fill the display, audio backend PulseAudio routed to the null sink
  `hal_stream` with `PULSE_SINK`.
- The runner starts one PulseAudio daemon per box (`--exit-idle-time=-1`) that
  provides the null sink.
- Window geometry and audio backend are written to Dolphin's config with the
  same boundary workaround as `set_dolphin_internal_resolution` in
  `hal/sim/session.py`, including a removal note naming the libmelee version.

### Stream process

`hal/netplay_service/stream.py` supervises ffmpeg while the session holds the
lease:

- Inputs: `x11grab` of the stream display at 60 fps, and `pulse` from
  `hal_stream.monitor`.
- Overlay: `drawtext` from a text file with `reload=1`. The runner writes one
  line, for example "HAL · Master-rank Falco · difficulty 25 · Game 2 of 5 ·
  play at 20xx.xyz". The line is built only from HAL's settings and the game
  count. It never contains a connect code.
- Encoding: `h264_nvenc`, CBR 6 Mb/s, keyframe every 2 s, AAC 160 kb/s.
- Output: FLV to `rtmp://live.twitch.tv/app/<key>`.
- Restart with backoff from 1 s to 30 s when ffmpeg exits. Stop on lease loss
  and on shutdown.

Between games, the stream display's root background is an idle card ("Play HAL
at 20xx.xyz" and the queue depth), so the stream never goes black when Dolphin
exits. The runner refreshes the card when the queue depth changes.

### Performance

Before and after on the RTX 3060 and on the first G4, with streaming on and off,
for the stream slot and one headless slot: game FPS, frame-interval p95, Dolphin
step p95, and policy round-trip p95. `check_realtime_budget` still gates
startup. Results go in `deploy/netplay/README.md`.

## Publishing a policy

`hal-netplay-admin publish-policy runs/policies/X.halpolicy`:

1. Read and validate the bundle manifest.
2. Upload the bundle to `s3://hal/netplay/policies/<sha256>.halpolicy` if absent.
3. Build the policy config: `schema_version`, `bundle_sha256`, `bundle_r2_key`,
   `vocabulary_sha256`, characters, stages, imitations (the current list until
   the page redesign adds the roster), delays, `desired_return_range`, defaults,
   `max_games`, `no_show_seconds`, `rematch_seconds`, and `masked_identity`.
4. `PUT /v1/admin/policy`.

After publishing, `/v1/options` serves the new config, and sessions running
another bundle get `409` on their next claim, drain, and exit with a message
that names both hashes.

Other admin commands:

- `hal-netplay-admin accounts upload <user.json>...`: upload account files to R2
  and publish the account list.
- `hal-netplay-admin status`
- `hal-netplay-admin events [--job ID | --session ID] [--since 1h]`
- `hal-netplay-admin pause | resume`

## Hosts

One image, `hal-netplay-runner:<git-sha>`, built from `deploy/netplay/Dockerfile`
and pushed to a registry, runs everywhere. It contains code, dependencies, and
system libraries, but no ISO, bundle, or account files. For streaming it adds
ffmpeg with NVENC, PulseAudio, and a tool to set the X root background.
Containers need `NVIDIA_DRIVER_CAPABILITIES` to include `video` for NVENC.

| Host | Command |
| --- | --- |
| This machine (RTX 3060) | `uv run hal-netplay-runner --slots 1`, or `docker run` with the image |
| Google Cloud | `deploy/netplay/gce-up.sh <name> [--zone Z] [--machine-type T]` |

`gce-up.sh` creates a G4 VM with one RTX PRO 6000 Blackwell GPU
(`g4-standard-48` by default) from a GPU-ready image, installs nothing by hand,
reads secrets from Secret Manager, and starts the pinned image with restart on
failure. `gce-down.sh <name>` sends SIGTERM, waits for the drain, and deletes
the VM. An optional single-VM managed instance group with autohealing replaces a
dead VM without intervention.

To verify on the first G4 bring-up, and record in `deploy/netplay/README.md`:

- The driver and the pinned `torch==2.11.0` build run on Blackwell (sm_120).
- Dolphin renders with Vulkan under the chosen image. Decide whether the NVIDIA
  RTX Virtual Workstation image and license are required, or whether the
  standard data-center driver is enough.
- Slippi direct connect succeeds from the VM's external IP (UDP hole punching).
- The realtime budget passes at the configured slot count.
- NVENC streaming runs next to the model without breaking the realtime budget.

For player latency, the zone matters more than the GPU. Choose the zone nearest
the expected players.

## Crash recovery

1. A box dies. Within 30 s its session ends, its leases fail with the existing
   codes, and its accounts are freed. Waiting players keep their place.
2. `hal-netplay-admin status` shows zero capacity and the ended session.
3. Run the bring-up command on any host. Capacity returns after the image
   starts and qualifies.

## Logging

- **Runner:** JSON lines on stdout via loguru. Every line carries `session`,
  `slot`, `job`, `game`, and `event`. Logged events: each download with key and
  hash, session start, account lease, claim, connect, game start, game end with
  result, each applied settings revision, replay upload, drain, and exit. Errors
  log the exception and the job's last known state. Stream events: lease
  granted and lost, ffmpeg start and exit with its code, restart count, and
  bitrate and dropped-frame figures sampled from ffmpeg's progress output.
- **Durable Object:** the `events` table records every job transition, session
  start and end, lease expiry, account lease and release, stream lease grant and
  release, policy publish, and
  every refused runner or admin call with its reason. Retained for 30 days.
- **Worker:** Cloudflare Workers Logs and `wrangler tail`.

## Security

- Runner and admin routes: Cloudflare Access service tokens (runners) or owner
  login (admin), plus Worker-checked bearer tokens stored as SHA-256 digests.
  Tokens are per box and revocable by removing one digest.
- Player routes: random 256-bit job tokens stored as digests; connect codes are
  never returned to other players; per-IP creation rate limit; queue cap.
- No CORS headers. Auth uses headers, not cookies.
- GPU boxes accept no inbound connections.
- ISO, bundles, and account files live only in the private R2 bucket.
- The Twitch stream key lives only as a Worker secret and is sent only to the
  current stream holder.
- The stream overlay is built from HAL's settings and the game count; it never
  includes a player's connect code.

## Testing

1. **Golden transcripts from the current service.** Before deleting the Python
   service, scripted scenarios drive `api.py` and `queue.py` through FastAPI's
   test client with a fake clock: one scenario per behavior in
   `tests/test_netplay_queue.py` and `tests/test_netplay_api.py`, plus rematch,
   cancel-after-game, and every failure path. Each scenario saves requests and
   responses as JSON, with IDs and tokens normalized, under
   `web/netplay-api/test/transcripts/`.
2. **Replay the transcripts** against the Durable Object in vitest. Responses
   must match exactly.
3. **New tests** for each deliberate change: session silence, lease grace,
   idempotent transitions, policy and account checks, rate limit, cap, pause,
   the alarm schedule, and the settings WebSocket. Streaming: the stream lease
   has at most one holder; it is freed on drain, `DELETE`, and silence; the key
   goes only to the holder; a non-stream claim gets `204` while the stream slot
   is live and idle, and gets the job otherwise.
4. **Python tests** for `RemoteQueue` (retry policy, error mapping) and for
   runner startup (hash verification, qualification failure, drain on first
   signal, forfeit on second). Streaming: the ffmpeg command; the overlay text
   never contains a connect code; restart backoff; per-slot `DISPLAY` and
   `PULSE_SINK`; the stream slot's Dolphin config writes.
5. **Integration** (`-m integration`): `RemoteQueue` against a local
   `wrangler dev`, driving one slot with a fake game through claim, connect,
   play, finish, and release, including a retried call and a session that stops
   reporting.
6. **Manual stream check** on this machine: one stream slot and one headless
   slot against local `wrangler dev`, pushing to Twitch with
   `?bandwidthtest=true` appended to the key so nothing goes live. Confirm video,
   audio, overlay, the idle card between games, and recovery after killing
   ffmpeg. Record the performance figures from "Streaming".
7. The AGENTS handoff checks: ruff format and check, ty, pytest, and the listed
   integration tests.

## Removed

- `hal/netplay_service/api.py`, `hal/netplay_service/queue.py`, and their tests
  (their behavior is preserved in the transcripts).
- The `hal-netplay-api` entry point.
- The `api`, `tunnel`, and `state` Compose services; `cloudflared` and
  `xvfb-run` in `run-host.sh` (the runner starts its own displays).
- `deploy/netplay/deploy-frontend.sh`, replaced by `deploy/netplay/deploy-web.sh`.
- `NEXT_PUBLIC_HAL_API_URL` in the page.
- Environment keys `HAL_NETPLAY_ALLOWED_ORIGINS`, `HAL_NETPLAY_ALLOWED_HOSTS`,
  `HAL_NETPLAY_PUBLIC_API_URL`, `CLOUDFLARE_TUNNEL_TOKEN`, `HAL_NETPLAY_POLICY`,
  `HAL_ISO_PATH`, `HAL_NETPLAY_EMULATOR_PATH`, `HAL_NETPLAY_USER_JSON_*`.

## Local development

`deploy/netplay/run-local.sh` starts `wrangler dev` for the API (local Durable
Object storage, Access skipped, a dev runner token), the page dev server with
`/v1` proxied to it, and a runner pointed at `http://localhost:8787`. Local
files can stand in for R2 through a `--local-assets DIR` runner flag, which is
still hash-checked.

## Risks

- **Port fidelity.** About 1,200 lines of Python move to TypeScript. The golden
  transcripts are the guard; any behavior they do not cover is at risk.
- **G4 unknowns.** Blackwell driver and torch support, Vulkan under the chosen
  image, and UDP hole punching from Google Cloud are unverified.
- **Cloudflare dependency.** The queue is unavailable if Cloudflare is. The page
  already depends on it.
- **Stream load.** Rendering at EFB scale 2, `x11grab`, and NVENC share the box
  with inference. The stream slot's latency must be measured, not assumed.

## Page redesign (follow-on spec), recorded decisions

These were agreed in the same conversation and are implemented in the page
redesign spec against the API above.

- **Page:** the one-sentence design ("I want to play Mang0's Falco at difficulty
  25."), Loud Notebook style (paper, ink, 2px outlines, orange accent, Schibsted
  Grotesk and IBM Plex Mono). Mockup: `.superpowers/brainstorm/*/content/sentence-v6.html`.
- **Grammar:** ranks read as adjectives ("Master-rank Falco"); players and
  "anyone" take a possessive.
- **Defaults:** Master rank; `desired_return` 20.
- **Pickers:** searchable, immediately typable, tabbable, arrow-key navigation,
  full character names, nickname search (for example `puff`, `icies`, `cf`,
  `hbox`, `plup`).
- **Hotkeys:** `P`/`1`/`/` player, `C`/`2` character, `D`/`3` difficulty,
  `[`/`]` difficulty ±5, `K` connect code, Enter to play (Ctrl/Cmd+Enter from a
  text field), Esc to close, `?` for the shortcut sheet.
- **Difficulty:** shown as 0–100, a linear rescale of `desired_return` in
  [−20, 140]. The API keeps raw units. Labels "chill" and "locked-in" with the
  numbers shown. Live: changes apply mid-game without an Apply button.
- **Style (imitation):** changes apply from the next game only; no mid-game KV
  cache rebuild. The job exposes `active_imitation`.
- **Advanced:** frame delay only. Temperature and replan interval are not
  exposed; replan interval stays fixed at 4 frames.
- **Roster:** the 38 professional players in `PROFESSIONAL_PLAYER_SLUGS` plus
  iBDW, Leffen, Pipsqueak, and Hungrybox, and the three ranks. Each maps to the
  connect code with the most training frames, generated from the identity
  sidecar and checked against the bundle's vocabulary.
- **Records:** games record the imitation used; the measurement record becomes
  schema 4 with a `policy_settings` timeline; browser prefs move to a new key
  storing raw `desired_return`.
- **How-to:** a "How do I play?" popover linking to https://slippi.gg/netplay.
- **Streaming notice:** one line near Play: "Games may be streamed live on
  twitch.tv/<channel>." with a link.
- **Open:** character-select portraits extracted from the ISO or stock icons
  first; whether to add vitest to the page project (this spec already adds it
  to the API Worker).
