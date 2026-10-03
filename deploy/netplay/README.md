# HAL netplay operations

[Current deployment status](status.md) records the exact deployed versions,
image, host, asset hashes and pending activation steps. **G4 activation is on
hold.** This runbook describes the integrated protocol-4 system.

## Architecture

```text
Browser ── static page + one status WebSocket ── Cloudflare Queue
                                                     │
                                     one host WebSocket + HTTP commands
                                                     │
                                     G4 supervisor / local queue relay
                                                     │
                                  shared inference + up to eight Dolphins
                                                     │
                                       Slippi peer connections (UDP)

G4 ── replay + small result JSON ── R2
Stream slot Dolphin ── OBS / NVENC ── Twitch
```

The Worker uses one SQLite Durable Object for the shared Slippi account, pairing
and stream leases, sessions and queue. Protocol 4 uses schema 7 / `global-v7`.
Schema changes use a fresh object; there are no migrations. Keep old history.

The browser receives pushed capacity and its private job. The G4 supervisor
sends health and fresh slot progress every ten seconds. Slot reports and idle
checks stay on its authenticated loopback relay. Claims follow work notifications;
settings, game results, completion and replay records use idempotent commands.
The frame loop does not wait for Cloudflare. Host silence expires after 30 seconds.

HAL#9000 is the shared account. Initial pairing is serialized because parallel
searches paired HAL with itself. Connected games run concurrently. The initial
connection deadline is 60 seconds. Reservations have no game cap; after 15 minutes
with someone waiting, the current game finishes and the reservation yields.

The timing profile is delay 2, inference allowance 2, prefix 4, replan 4, horizon 8.
Inference batches ready streams; idle slots do not run continuous model inference.
Dolphin starts when a reservation is assigned. Ranked uses a separate process;
see [ranked play](ranked.md).

## Code map

| Files | Responsibility |
| --- | --- |
| `web/netplay-api/src/{queue,store,sessions,live,budget}.ts` | Durable state, leases, push and quota budgets. |
| `web/netplay/app/page.tsx`, `lib/netplay-api.ts` | Player controls and browser socket. |
| `hal/netplay_service/{runner,reservation,control,queue_client}.py` | GPU/slot supervision, continuous sessions and shared control. |
| `hal/inference/`, `hal/eval/netplay.py`, `hal/sim/netplay.py` | Model batching, scheduling and emulator boundary. |
| `hal/netplay_service/{stream,obs,value_meter,stream_monitor}.py` | Displays, audio, capture, overlay and diagnostics. |
| `hal/netplay_service/replays.py`, `hal/eval/ranked_replays.py` | Verified uploads and ranked upload receipts. |
| `hal/fixtures.py`, `hal/netplay_service/assets.py` | Static fixtures and verified policy/account downloads. |

## Assets and secrets

Images contain code and locked dependencies, not ISO, emulator, policy or accounts.
The runner downloads and verifies assets. The emulator is the official pinned
Slippi Online 3.6.4 release, not the old local patched 3.5.1 file. R2 replay uploads
contain SLP and small JSON files, not video. There is no replay-expiry command.

GCE reads `hal-netplay-runner-env` from Secret Manager into a root-only `/run`
file. Cloudflare Access and bearer tokens protect runner/admin routes. The Worker
provides the Twitch key only to the stream-lease holder. Do not put secrets in
images, Git, URLs or command arguments.

Use the existing encrypted operator store without scratch imports:

```sh
uv run python scripts/netplay_admin.py \
  --credentials /home/ericgu/src/hal/runs/netplay/credentials status
```

Other arguments go to `hal-netplay-admin`, including `pause`, `resume`,
`publish-policy BUNDLE`, `accounts upload USER_JSON`, and `events`.
The helper decrypts only into memory and the child environment. It supports the
existing host-bound systemd-creds files; no credential migration is needed.
Mutating admin commands require the relevant deployment authorization.

Old Phillip campaign and deployment helpers are preserved under
`archive/netplay-2026-09-operator-tools/`. They target an obsolete protocol and
are not supported launch tools. The campaign remains paused. `runs/netplay/`
contains private assets and experimental evidence; do not treat it as source.

## Local development

```sh
uv sync --locked
cp deploy/netplay/.env.example deploy/netplay/.env
# Fill the local asset mirror, credentials and full Git SHA in the private env file.
deploy/netplay/run-local.sh
```

The launcher starts the local Worker on 8787, page on 3000 and runner, with a
process-group cleanup trap. It explicitly disables Worker maintenance
locally. Policy/account objects use their normal R2 keys in the local mirror.
A fresh schema needs policy and accounts published with the local admin endpoint.
Use a new state directory for schema 7; retain old local state only as evidence.
The launcher defaults to no stream. Local streaming requires bandwidth-test mode.

## Host and Compose

`run-host.sh ENV_FILE` runs directly on a prepared GPU host. `compose.yaml` builds
and runs the same image on a prepared host. `compose-ada.yaml` is an optional
two-slot override, not the production G4 configuration.

Both paths need NVIDIA drivers, Vulkan, Xvfb and xsetroot. Streaming also needs
NVIDIA Xorg, Openbox, OBS, PulseAudio and Xauthority. For Compose set
`HAL_NETPLAY_STREAM_DISPLAY=:90`, `HAL_NETPLAY_X11_SOCKET=/tmp/.X11-unix/X90`,
`HAL_NETPLAY_XAUTHORITY` and `HAL_NETPLAY_VULKAN_ICD`. Only X90 is shared read-only;
private Xvfb sockets remain writable. Missing host bind files fail explicitly.

`gce-startup.sh` supplies the matching NVIDIA libraries, Xorg :90, Vulkan manifest
and preflight check, Secret Manager access, and systemd runner/health containers.
Slot 0 uses NVIDIA OpenGL; other slots use NVIDIA Vulkan on private Xvfb displays.
One PulseAudio daemon serves the box. The first signal drains; a second signal
or the 15-minute drain deadline forfeits. A crash is covered by session silence.

## Build and deploy

Use a clean committed checkout and a full SHA:

```sh
sha=$(git rev-parse HEAD)
docker build --file deploy/netplay/Dockerfile --build-arg HAL_GIT_SHA="$sha" \
  --tag "hal-netplay-runner:$sha" .
```

Registry pushes and billable resources require authorization. `gce-up.sh --help`
describes creation of one G4 or an optional size-one MIG. `gce-down.sh` **deletes**
the VM and disk after drain; it is not a suspend command. No MIG is deployed.
Use [status.md](status.md) to update the existing host instead of creating another.

Worker source defaults to maintenance **on**. The deployment script requires an
explicit choice so a normal deploy cannot silently open service:

```sh
deploy/netplay/deploy-api.sh --maintenance on
deploy/netplay/deploy-web.sh
```

These commands deploy publicly; they are not validation commands. Turning
maintenance off is a separate `deploy-api.sh --maintenance off` operation after
policy restoration and readiness checks. Paused admissions and Worker maintenance
are distinct: maintenance blocks all API/DO access, including admin calls.

## Twitch stream

OBS captures only Dolphin, including menus, at 1080p60 with NVENC. The queue
selects one stream slot; other emulators are not captured. Direct play shows its
settings and game number. Ranked displays the smoothed value head on the right.
Neither overlay may contain a connect code. Stream failure must not stop gameplay.

## Verification checklist

Before handoff, run Ruff format/check, ty, non-integration pytest, Worker tests and
typecheck, and frontend lint/typecheck/build. CI covers those paths. The manual
integration workflow covers roundtrip, cleanup and the local Worker/runner wire.

Before an authorized live activation:

- [ ] Confirm exact image/source, protocol, schema, policy/account hashes and one GPU.
- [ ] Confirm NVIDIA OpenGL/Vulkan inside the image, private sockets and no llvmpipe.
- [ ] Qualify eight inference streams against 33.333 ms; record p50/p95/p99.
- [ ] Verify claims, serialized pairing, cancellation cleanup, reconnect and stale-slot expiry.
- [ ] Verify replay upload and receipt after game six; preserve pending sidecars on failure.
- [ ] Test first-signal drain, timeout/second-signal abort and host silence.
- [ ] Measure game FPS/frame latency, OBS render/encode drops and Worker CPU/SQL costs.
- [ ] For the manual stream check use `?bandwidthtest=true` unless live streaming is
      explicitly authorized. Check 1080p60, audio, menus and no connect codes.
- [ ] Resume admissions only after readiness. Record deployment IDs and results in status.

Local cost tests are not production billing telemetry. The supported envelope,
limits and overload policy are in [free-tier design](free-tier-design.md).
No free-tier guarantee covers unlimited traffic, other Workers on the account,
or indefinite R2 replay retention. Reboot/MIG recovery and eight live games still
need their own verification.

## Historical evidence

[Deployment history](history.md), [ranked history](ranked-history.md),
[eight-slot qualification](direct8.md), [shared-control rollout](free-tier-rollout.md),
[replay teardown](g4-teardown.md), and [Phillip campaign](x-pilot.md) retain exact
commands, failures, skips and measurements. These records are not current setup
instructions. The last measured eight-stream inference p99 was 21.148 ms;
the separate neutral-input renderer check measured 59.94 FPS. Neither establishes
performance of the integrated release or eight simultaneous model games.
