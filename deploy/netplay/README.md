# HAL netplay deployment

The queue runs in `web/netplay-api`. A GPU host runs only `hal-netplay-runner`.
The runner downloads and verifies its static fixtures, starts a remote session,
keeps that session alive while it downloads policy and account assets, qualifies
both delay profiles, and then starts its slots.

## Current status — 2026-10-01 UTC

Direct play is back online at [20xx.xyz](https://20xx.xyz). The replacement
G4 runs **one validated slot**, using `HAL#647`. The expanded player list is
published. Ranked and the Phillip campaign remain paused.

The code accepts up to sixteen slots, but sixteen are **not live**. Only one
bot account is available, and the sixteen-stream prediction check failed:
p99 was 33.812 ms against a 16.667 ms limit. Additional accounts, inference
performance work, a suitable Cloudflare plan, and a full concurrent gameplay
check are required before opening more slots. See the
[capacity and deployment report](direct16.md) for evidence and commands.

- Site: [20xx.xyz](https://20xx.xyz). Worker `hal-netplay-web` serves the page;
  `hal-netplay-api` handles `/v1/*`. The queue cost fix remains deployed.
- Queue: one SQLite-backed Durable Object, class `Queue`, instance `global`,
  storage schema 2. It owns reservations, sessions, account leases, policy
  settings, the stream lease, and events.
- GPU: `hal-netplay-g4`, project `centering-star-502613-k3`, zone
  `us-west1-a`, external address `34.83.210.75`. It is a standalone
  `g4-standard-48` with exactly one RTX PRO 6000 Blackwell GPU and a 100 GB
  boot disk. No managed instance group is deployed.
- Runner: systemd starts `hal-netplay-runner` and `hal-netplay-health`
  containers. The complete runner image uses commit `8831869a`. The host
  startup script includes fixes through `6ec246ff`.
- Video: [hal_20xx on Twitch](https://www.twitch.tv/hal_20xx). NVIDIA Xorg
  `:90` renders Dolphin. OBS captures only its render window at 1080p60
  and uses NVENC. Menus remain visible while a reservation owns Dolphin.
- Live check: 1,800 frames in 30.019 seconds, or 59.96 FPS, against the
  owner's local peer. Both one-slot timing profiles passed. The short test
  ended by deliberate disconnect and freed the slot.

The former VM and disk were deleted before this replacement. All 105 Ranked
replays and its one unfinished direct-play replay remain verified in R2.
The [teardown report](g4-teardown.md) records their locations and local backup.

```text
Browser -> page Worker + API Worker -> Durable Object
                                ^
                                | HTTPS transitions + WebSocket settings
                                v
                      G4 runner supervisor
                         |            |
                  GPU inference <-> Dolphin <-> Slippi peer (UDP)
                                      |
                                      v
                                OBS -> Twitch

Runner <- verified fixtures / policy / account assets
Runner -> R2 replay + metadata -> local matchup recordings
```

The runner starts one shared GPU inference process and one Dolphin process
per slot. Frame observations and actions stay on the GPU host. Cloudflare
handles queue state, settings, and health reports. The GPU host has no public
HAL web API. Slippi uses its own peer connection.

### Image and recovery limits

The current registry image is:

```text
us-west1-docker.pkg.dev/centering-star-502613-k3/hal-netplay/hal-netplay-runner:8831869a7b315fc2755f895b399c4f84a968a84a
```

Its manifest digest is
`sha256:b3f341cd60e15df2c899d6b2bf6bce50afe0297d3b8d03762d40252e59d9a501`.
It contains the maintained runtime, OBS, and ranked scripts. No source
override mounts are required. ISO, policy, accounts, and credentials are
fetched at runtime.

Fresh-host bring-up was tested. The startup script installs missing Docker
and matching NVIDIA GLX/video libraries before creating the services.
Reboot recovery and managed instance group replacement have not been tested.

### Assets, secrets, and operator state

The image contains code and dependencies. The runner verifies downloaded
assets by SHA-256. R2 stores the private ISO, policy bundle, account JSON, and
replays. The official netplay emulator comes from its pinned GitHub release.
Replay uploads contain only `.slp` files and small JSON metadata. The bucket
has no automatic replay expiration rule. Video and detailed frame measurements are not uploaded to R2.

The VM service account reads `hal-netplay-runner-env` from Secret Manager.
Its dotenv file and Xauthority cookie are root-only files under `/run`.
Cloudflare Access and bearer tokens protect runner and admin operations.
The API Worker holds the Twitch key and releases it only to the stream lease
holder. Keys are not built into images or committed to Git.

The paused Phillip campaign uses operator scripts on the local RTX 3060
machine. Its encrypted credentials and verified recordings remain under
`runs/netplay/x-pilot-master120/`. These tools are separate from the public
service. See [the run protocol](x-pilot.md).

## Static fixtures

`hal/fixtures.py` owns both static runtime files.

- `ISO` is the private `fixtures/ssbm.ciso` R2 object.
- `NETPLAY_EMULATOR` is the official Slippi Online 3.6.4 AppImage. The verified
  release file is 111,679,992 bytes with SHA-256
  `e0f984e5bbecb98e3a746da1f173a475b06c3a1ba6b73e2e31bbe85a5f5a5e8a`.

The older file at `~/data/dolphin/slippi/Slippi_Online-x86_64.AppImage` is a
local 3.5.1 build and is not used.

## Local development

Install the Python and Node dependencies. Copy `.env.example` to `.env`, set
`HAL_GIT_SHA`, and provide a directory for the policy and account objects named
by their Worker R2 keys:

```sh
cp deploy/netplay/.env.example deploy/netplay/.env
deploy/netplay/run-local.sh
```

The launcher starts the Worker at `127.0.0.1:8787`, waits for it, starts the
page at `127.0.0.1:3000`, and then starts the runner. The page uses same-origin
`/v1` requests; Vite proxies them to the Worker. Ctrl-C stops all three process
groups. Local assets still pass the normal SHA-256 checks.

Local development opts out of streaming. Set `HAL_NETPLAY_LOCAL_STREAM=1`,
`HAL_TWITCH_BANDWIDTH_TEST=1`, and `HAL_NETPLAY_LOCAL_TWITCH_KEY` only for the
owner-run bandwidth test.

## Host run

Set the runner URL and token, Cloudflare Access service token, R2 credentials,
full Git SHA, and slot count in `.env`. Then run:

```sh
deploy/netplay/run-host.sh
```

The first signal drains the remote session. A second signal or the 15 minute
deadline aborts active games. A hard crash is covered by the Worker's 30 second
session silence limit.

A direct host run needs Xvfb and `xsetroot`. An opted-in streaming host also
needs the pinned OBS package, Openbox, `glxinfo`, PulseAudio, and an authenticated
NVIDIA Xorg display. The image supplies the user-space tools. The GCE startup
script supplies the host display. Launchers reject a missing command.

## Runner image

Build the image from a clean checkout at the commit that it will run:

```sh
sha=$(git rev-parse HEAD)
docker build --build-arg HAL_GIT_SHA="$sha" \
  --tag "hal-netplay-runner:$sha" \
  --file deploy/netplay/Dockerfile .
```

The Dockerfile copies code and locked dependencies only. It does not copy the
ISO, emulator, policy bundle, account JSON, or secrets. The runner downloads
and verifies those files at startup.

Tag the image for the chosen registry with the same full SHA, then push it.
Pushing is an owner action. For Artifact Registry the commands are:

```sh
registry=REGION-docker.pkg.dev/PROJECT/REPOSITORY
docker tag "hal-netplay-runner:$sha" "$registry/hal-netplay-runner:$sha"
docker push "$registry/hal-netplay-runner:$sha"
```

## Google Cloud G4 host

The boot service reads one dotenv secret from Secret Manager. It must contain:

```text
HAL_NETPLAY_API_URL
HAL_NETPLAY_RUNNER_TOKEN
CF_ACCESS_CLIENT_ID
CF_ACCESS_CLIENT_SECRET
AWS_ENDPOINT_URL
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_BUCKET
```

Create the secret and its value outside these scripts. The attached service
account needs Secret Manager Secret Accessor on that secret and registry read
access. The VM uses the `cloud-platform` OAuth scope. Secret values do not go
in VM metadata.

Create a standalone host after the image and secret exist:

```sh
deploy/netplay/gce-up.sh NAME \
  --project PROJECT \
  --zone ZONE \
  --image "REGION-docker.pkg.dev/PROJECT/REPOSITORY/hal-netplay-runner:$sha" \
  --secret RUNNER_DOTENV_SECRET \
  --service-account RUNNER_SERVICE_ACCOUNT \
  --slots 1
```

The default is `g4-standard-48`, which has one RTX PRO 6000 Blackwell GPU. The
default boot image is the GPU-ready Deep Learning VM family
`common-cu129-ubuntu-2404-nvidia-580`. Use `--virtual-workstation` only if the
first Vulkan check shows that the standard data-center driver cannot create the
required display and rendering context.

Stop the runner, wait for its 15 minute drain deadline, and delete the VM:

```sh
deploy/netplay/gce-down.sh NAME --project PROJECT --zone ZONE
```

`--force` permits deletion after a failed drain. It can forfeit an active game.

For automatic replacement of a dead VM, add `--managed` to both commands. The
up command creates a size-one zonal managed instance group, a health check on
port 9101, and a firewall rule limited to Google Cloud health-check sources.
Its 40 minute initial delay covers fixture download and qualification.

### First G4 verification checklist

Running this checklist creates billable resources.

- [ ] Record the image URI, full Git SHA, zone, machine type, boot image family,
  slot count, and start/end times.
- [ ] Confirm the attached GPU and driver: `nvidia-smi`.
- [ ] Confirm the pinned PyTorch build sees Blackwell:
  `docker exec hal-netplay-runner python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_capability())'`.
  Expected: PyTorch 2.11.0 and capability `(12, 0)`.
- [ ] Confirm Vulkan in the runner container:
  `docker exec hal-netplay-runner vulkaninfo --summary`. Record whether the
  standard driver works. If it fails for the required rendering path, repeat
  with a host created using `--virtual-workstation` and record that decision.
- [ ] Complete a Slippi direct-connect game from the VM external IP. Record the
  peer region and whether UDP hole punching succeeded without an inbound rule.
- [ ] Save the qualification budget file and the runner status. Record game
  FPS, frame-interval p95, Dolphin-step p95, policy-round-trip p95, model
  inference p95, and batch-wait p95 at the configured slot count.
- [ ] After Plan 5, run NVENC streaming beside model inference and repeat the
  realtime measurements with streaming on and off.
- [ ] Stop the host with `gce-down.sh`. Confirm the session drained, capacity
  fell to zero, the account leases were freed, and the VM or managed group was
  deleted.

Live G4 result on 2026-09-29:

- VM `hal-netplay-g4` ran in `us-west1-a` on `g4-standard-48` with one NVIDIA
  RTX PRO 6000 Blackwell Server Edition. The driver was 580.173.02.
- The runner image and Git SHA were
  `us-west1-docker.pkg.dev/centering-star-502613-k3/hal-netplay/hal-netplay-runner:728d96018e242332854a6a77ea5d2ff6eb17012c`.
  It ran one slot. PyTorch 2.11.0+cu130 reported CUDA capability `(12, 0)`.
- The standard data-center driver could not create a Vulkan instance in the
  container. The runner used OpenGL and completed the direct-connect game.
- A peer in Los Angeles connected without an inbound firewall rule. The game
  ran for 5,647 frames on Yoshi's Story. The runner recorded 58.7 FPS, 18.2 ms
  frame p95, 18.0 ms Dolphin p95, and 7.1 ms policy p95.
- The runner uploaded the 2,209,026 byte replay, released the job, and returned
  the public queue to zero active and zero queued jobs with one healthy slot.
- The VM remains running for continued owner testing.

## Twitch stream

Set `TWITCH_STREAM_KEY` as a secret on the API Worker. It does not belong in a
runner dotenv file, VM metadata, image, or repository:

```sh
cd web/netplay-api
npx wrangler secret put TWITCH_STREAM_KEY
```

Writing the secret and deploying the Worker are owner actions. The Worker gives
the key only to the live session that holds its one stream lease. That runner
streams slot 0. `--no-stream` opts a runner out.

The stream slot uses a dedicated NVIDIA Xorg display. Set
`HAL_NETPLAY_STREAM_DISPLAY=:90` and pass its private Xauthority cookie.
The GCE startup script installs graphics libraries that match the loaded
NVIDIA 580 server driver, starts Xorg, and refreshes NVIDIA container metadata.
A monitor is not required. The runner rejects software OpenGL rendering.
Other slots retain isolated 640×480 Xvfb displays.

OBS Studio 30.2.3 (Ubuntu package `30.2.3+dfsg-3~bpo24.04.1`) captures only
Slippi's `Dolphin` window through Xcomposite. Openbox manages that display.
OBS encodes 1920×1080 at 60 fps using texture NVENC, 6 Mb/s CBR, P5,
two-second keyframes, two B frames, and no lookahead. Audio comes only from
`hal_stream.monitor` at AAC 160 kb/s. The overlay uses an OBS text source.
It contains HAL's settings and game count; it has no connect-code field.
Dolphin stays visible whenever its render window exists, including menus and
connection. The desktop and launcher are excluded. When Dolphin exits between
reservations, the background is empty until the next window opens.

The stream lease owns the OBS process. Its profile and credentials live in a
private temporary directory and are removed on shutdown. Do not publish the
OBS control port (4455). OBS reconnects RTMP; the runner restarts OBS with
backoff after process or control failures. Inspect `obs.log` and
`runner-status.overlay.obs.json` next to the runner status file.

### Owner manual stream check

This check is externally visible. Use `HAL_TWITCH_BANDWIDTH_TEST=1` unless the
owner explicitly approves a public stream. Confirm the OBS stream key ends in
`?bandwidthtest=true` for a bandwidth test before continuing.

- [ ] Record the full Git SHA, policy SHA, hardware, slot count, delay, stage,
  peer and region, driver, and OBS package version.
- [ ] Start the runner with `HAL_TWITCH_BANDWIDTH_TEST=1`. Confirm admin status
  shows one stream holder and that no second session receives the lease.
- [ ] During connection and menus, confirm the Dolphin window remains visible
  with no waiting card. Confirm a waiting job goes to slot 0 before another
  idle slot.
- [ ] Play a game on slot 0. Confirm 1920×1080 video, game audio, the HAL setting
  line, and no connect code in the overlay. Dolphin menus may show their own
  codes under the approved continuous-capture behavior.
- [ ] Run `pkill -TERM obs` inside the runner container. Confirm the
  supervisor restarts it with bounded backoff and the runner keeps its game.
- [ ] Drain the holder. Confirm OBS stops, the lease becomes free, and the
  next opted-in live session takes it on a status report.
- [ ] Stop every runner. Confirm runner-owned Xvfb, Openbox, PulseAudio, Dolphin, and OBS processes
  are gone.

The owner approved the public stream. The current OBS rollout passed the
1080p60 game capture, menu capture, overlay privacy, and OBS restart checks.
Stream lease transfer and full host shutdown checks remain pending. The
verification record below preserves the earlier 720p ffmpeg measurements.

### Performance measurement

Use the same Git SHA, policy, peer, stage, delay, slot count, and game length
for every control and treatment. Capture values during active play:

```sh
uv run python deploy/netplay/capture-stream-metrics.py \
  --label rtx3060-stream-off --hardware RTX-3060 --streaming off \
  --status-path runs/netplay/runner-status.json \
  --output runs/netplay/measurements/rtx3060-stream-off.json
```

Repeat with streaming on in bandwidth-test mode. Do the same on the first G4.
The script refuses a slot without active-game values and records the Git SHA
with all four required metrics. These runs need a live peer and the manual
stream check, so their cells remain pending for the owner.

| Hardware | Streaming | Slot role | Game FPS | Frame p95 ms | Dolphin p95 ms | Policy p95 ms |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| RTX 3060 | off | slot 0 control | pending | pending | pending | pending |
| RTX 3060 | off | headless control | pending | pending | pending | pending |
| RTX 3060 | on | stream slot 0 | pending | pending | pending | pending |
| RTX 3060 | on | headless treatment | pending | pending | pending | pending |
| G4 RTX PRO 6000 | off | slot 0 control | pending | pending | pending | pending |
| G4 RTX PRO 6000 | off | headless control | pending | pending | pending | pending |
| G4 RTX PRO 6000 | on | stream slot 0 | 58.7 | 18.2 | 18.0 | 7.1 |
| G4 RTX PRO 6000 | on | headless treatment | pending | pending | pending | pending |

The G4 treatment is one complete live game with NVENC enabled. A matched
streaming-off control is still required before drawing a performance
conclusion.

## Web deployment

`deploy-web.sh` builds and deploys the page. Running it is an owner action
because it is externally visible:

```sh
deploy/netplay/deploy-web.sh
```

## Verification record

The entries below are milestone records. Their source SHA and date determine
which runtime they describe; the current deployment is summarized above.

The [x_pilot run protocol](x-pilot.md) records the requested `gm-v2` matchup
schedule, missing characters, authentication requirements, and replay checks.

Plan 3 focused checks:

- `uv run pytest -q tests/test_netplay_runner.py`: 55 passed.
- `uv run pytest -q tests/test_qualify_netplay_059.py`: 62 passed.
- `uv run ruff format --check .`: passed, 263 files formatted.
- `uv run ruff check .`: passed.
- `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts`: passed.
- `uv run pytest -q -m "not integration"`: 1,417 passed, 8 skipped, 21 deselected.
- `npm test` in `web/netplay-api`: 101 passed. `npm run typecheck`: passed.
- `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_netplay_queue_integration.py -m integration`: 3 passed.
- `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration`: 7 passed, 6 deselected.
- `npm run build` in `web/netplay`: passed after `npm ci`. The first attempt failed because `vinext` was not installed.

The first emulator-suite attempt failed because this worktree lacked its ignored
fixtures. The second found the ISO and emulator through explicit environment
paths but lacked the MDS and archive. The exact required command passed after
the worktree linked to the existing read-only fixtures in the main checkout.

Detailed match measurements and engine audits stay local for qualification.
R2 receives only each replay and its small metadata JSON. This storage decision
was approved after Plan 3 implementation.

Plan 4 checks:

- `uv run pytest -q tests/test_netplay_gce.py tests/test_netplay_deploy.py`: 15 passed.
- `bash -n deploy/netplay/gce-startup.sh deploy/netplay/gce-up.sh deploy/netplay/gce-down.sh`: passed as part of the focused test.
- `uv run ruff format --check .`: passed, 265 files formatted.
- `uv run ruff check .`: passed.
- `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts`: passed.
- `uv run pytest -q -m "not integration"`: 1,425 passed, 8 skipped, 21 deselected. It emitted 24 upstream Python 3.14, Torch, and multiprocessing warnings.
- `npm test` in `web/netplay-api`: 101 passed. Workerd logged one closed WebSocket while the suite shut down. `npm run typecheck`: passed.
- `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_netplay_queue_integration.py -m integration`: 3 passed.
- Docker build and registry push: skipped. The static Dockerfile test passed. Registry push is an owner action.
- Live G4 creation and verification: skipped because it costs money and is an owner action.

Plan 5 checks:

- `uv run pytest -q tests/test_netplay_runner.py`: 58 passed.
- `uv run pytest -q tests/test_netplay_stream.py`: 6 passed.
- `uv run pytest -q tests/test_session.py tests/test_netplay_session.py`: 47 passed.
- `uv run pytest -q tests/test_netplay_deploy.py`: 9 passed.
- `uv run pytest -q tests/test_netplay_queue_client.py -k 'stream or queue_depth'`: 10 passed, 34 deselected.
- `uv run ruff format --check .`: passed, 268 files formatted.
- `uv run ruff check .`: passed.
- `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts`: passed.
- `uv run pytest -q -m "not integration"`: 1,448 passed, 8 skipped, 21 deselected. It emitted 24 upstream Python 3.14, Torch, and multiprocessing warnings.
- `npm test` in `web/netplay-api`: 105 passed. Workerd logged one closed WebSocket while the suite shut down. `npm run typecheck`: passed.
- `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_netplay_queue_integration.py -m integration`: 3 passed.
- `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration`: 7 passed, 6 deselected. It emitted 6 upstream multiprocessing warnings.
- `bash -n` on the netplay launchers, GCE scripts, and Docker entry point: passed.
- `uv run python deploy/netplay/capture-stream-metrics.py --help`: passed.
- Host runtime probe: Xvfb and `xsetroot` are installed. ffmpeg and PulseAudio are absent on this checkout's host. Use the runner image or install them before a direct streaming run.
- Docker build and image push: skipped. The static package and capability tests passed. Image push is an owner action.
- Worker deployment, `TWITCH_STREAM_KEY` write, live Twitch bandwidth test, RTX 3060 treatment measurement, and G4 measurement: skipped because they are owner actions. The scripts, local capture helper, checklist, and pending result table are above.


## GPU rendering and OBS verification, 2026-09-29

Source milestone: `a978dbf3` (`Stream Dolphin through OBS on the GPU`).
The owner approved this change and the public stream. The VM still has one
RTX PRO 6000 Blackwell GPU. The page is <https://20xx.xyz>; the stream is
<https://www.twitch.tv/hal_20xx>.

The original stream used Xvfb, Mesa llvmpipe, and ffmpeg. Dolphin now uses
NVIDIA GLX on Xorg `:90`. OBS uses NVIDIA EGL, native Xcomposite window
capture, and texture NVENC. Both renderer checks name the RTX PRO 6000.
The container must contain `10_nvidia.json` as well as the injected NVIDIA
libraries. Missing EGL registration made OBS select llvmpipe even after
Dolphin's GLX path was fixed. The image now supplies that registration, and
the OBS controller explicitly selects it.

The live game window and OBS output are both 1920×1080. A program screenshot
confirmed game video, black aspect-ratio bars, and the HAL overlay. It showed
no desktop, launcher, or connect code. At this milestone, OBS had three
inputs: the Dolphin window, the overlay, and `hal_stream.monitor`. It hid the window
outside a fresh playing state; the later continuous-capture change removes
that restriction. The first-run wizard is disabled in the OBS
profile. Openbox forces the exact Dolphin render window to fullscreen.

### Measurements

The policy bundle remained
`0ff1daf80caa36a94a713c4ccba9223db8d7ba7c1379b5865bbc40b8a8c2f3ec`.
The checkpoint remained
`52b5233ed506f59f514f7e90a6a6111206152db7413f451dc35be5c30d1e671b`.
Both samples used the same G4, one slot, compiled inference, delay 2, Fox,
and the iBDW imitation. The peer ran on the local RTX 3060. The driver was
580.173.02. OBS was Ubuntu package `30.2.3+dfsg-3~bpo24.04.1`.

| Pipeline | Game FPS | Frame p95 ms | Dolphin p95 ms | Policy p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Xvfb / llvmpipe / ffmpeg | 59.932 | 18.153 | 18.029 | 7.282 |
| NVIDIA Xorg / OBS | 59.934 | 17.999 | 17.926 | 7.905 |

These are rolling live samples from different games, not a controlled stage
and seed comparison. The simulation already ran near 60 fps. This change
fixed the rendering and capture path. It does not establish an inference
speed improvement. The full two-slot, stream-on/off matrix above remains
pending. Viewer latency was not measured.

During a 126.65-second OBS interval spanning live play and game transitions:

- Output: 60.00 fps, 6.183 Mb/s including audio and transport overhead.
- Render misses: 0. Encoder skips: 0. Network drops: 0.
- Final average OBS frame render time: 0.248 ms.
- Two render misses occurred during OBS startup, before the interval.
- The earlier 15-second ffmpeg file probe encoded 897 frames at 59.61 fps,
  with two drops. That file probe did not measure Twitch delivery or distinct
  game frames and is not directly comparable to OBS's counters.

The raw report is `/var/lib/hal-netplay/obs-verification.json` on the VM.
No measurement files or screenshots were uploaded to R2.

### Live deployment and cleanup

The existing registry image remains the base. The VM has a local derived
image, `hal-netplay-runner:obs-local`, with the pinned OBS packages and EGL
registration. It was not pushed to a registry. The current container received
the same EGL registration directly before OBS restarted. Source files are
mounted from `/var/lib/hal-netplay/hotfix/obs-v1`; their source milestone is
`a978dbf3`. The base image label still identifies `728d9601`.

`90-obs.conf` selects the local image, source mounts, authenticated X socket,
and display `:90`. The VM startup metadata now contains the updated
`gce-startup.sh`, so the host can recreate its display and Xauthority cookie
after reboot. A reboot was not performed. The explicit EGL environment guard
in the final source applies on the next full runner start; the running OBS
was independently verified to use NVIDIA EGL.

The OBS probe container was removed. Its temporary profile and the stale
profile from the first rollout were removed. The active profile remains
private. The previous source files and base image remain available for
rollback. No second VM, GPU, registry push, R2 asset upload, Worker deployment,
or Git push occurred during this change.

The old runner did not exit after an idle drain. A second TERM completed each
restart after the active game ended. This existing drain behavior still needs
a separate fix. Local matches resumed after the required integration checks
released port 51441.

### Commands and results

Commands below ran from this worktree unless another location is given.
Repeated probes are grouped by purpose. No credentials are included.

| Command | Result |
| --- | --- |
| `uv run pytest -q tests/test_netplay_obs.py tests/test_netplay_stream.py tests/test_netplay_runner.py tests/test_netplay_deploy.py` | 80 passed before the final EGL regression was added. |
| `uv run pytest -q tests/test_netplay_obs.py tests/test_netplay_stream.py tests/test_netplay_deploy.py` | Final focused check: 23 passed. |
| `uv run ruff format --check .` | Passed; 270 files formatted. |
| `uv run ruff check .` | Passed. |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts` | Passed. |
| `uv run pytest -q -rs -m 'not integration'` | Final run: 1,459 passed, 8 skipped, 21 deselected; 24 upstream warnings. Earlier runs had 1,458 passes before the EGL test. |
| `npm test` in `web/netplay-api` | 105 passed. Workerd logged a closed WebSocket during shutdown. |
| `npm run typecheck` in `web/netplay-api` | Passed. |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_netplay_queue_integration.py -m integration` | 3 passed. |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration` | Final run: 7 passed, 6 deselected; 6 upstream multiprocessing warnings. |
| `bash -n deploy/netplay/gce-startup.sh deploy/netplay/run-host.sh deploy/netplay/run-local.sh` | Passed. |
| `git diff --check` and `git diff --cached --check` | Passed. |
| `git commit -m 'Stream Dolphin through OBS on the GPU'` | Created `a978dbf3`; format, lint, and type pre-commit hooks passed. |
| `gcloud compute ssh hal-netplay-g4 --project centering-star-502613-k3 --zone us-west1-a --command=...` | Installed matching graphics packages; configured Xorg and Xauthority; refreshed CDI; built the local OBS image; staged source; drained and restarted the runner; inspected logs, renderer maps, screenshots, and counters; removed the probe container and stale profiles. |
| `gcloud compute scp ... --project centering-star-502613-k3 --zone us-west1-a` | Copied source to the VM and retrieved the program image and measurement report. |
| Remote `apt-get -s install ...` followed by `apt-get install ...` | Installed Xorg, NVIDIA GL 580.173.02, xauth, and diagnostic tools. `libnvidia-common-580-server` also had to be pinned to the same version. |
| Remote `systemctl restart nvidia-cdi-refresh.service` | Refreshed NVIDIA container metadata after the graphics install. |
| Remote `glxinfo -B` and `eglinfo -B` | Final host and container checks selected NVIDIA. |
| Remote `docker build -t hal-netplay-runner:obs-local /var/lib/hal-netplay/obs-image` | Passed. Final local image: `e255ef5d8a8c`; no registry push. |
| Remote `systemctl restart --no-block hal-netplay-runner` and `docker kill --signal TERM hal-netplay-runner` | Drained each current game, then used the second signal to finish shutdown. |
| Remote `docker exec hal-netplay-runner pkill -TERM -x obs` | Verified OBS restart without stopping the active match; the replacement used NVIDIA EGL and texture NVENC. |
| Remote OBS `GetSourceScreenshot`, `GetStats`, and `GetStreamStatus` | Confirmed the game-only image, 1080p60 output, and zero drops during the recorded interval. |
| `gcloud compute instances add-metadata hal-netplay-g4 --project centering-star-502613-k3 --zone us-west1-a --metadata-from-file=startup-script=deploy/netplay/gce-startup.sh` | Updated startup metadata on the existing VM. |
| `env TMPDIR=/home/ericgu/src/hal-edge-queue/runs/netplay/tmp uv run python /tmp/hal_queue_cody_fox_forever.py` | Continuous Cody Fox matches resumed and remain running. |

Failures found and resolved:

- The first graphics install failed because the common NVIDIA package selected
  a newer driver dependency. Pinning it to 580.173.02 resolved the conflict.
- Xorg first failed with `UseDisplayDevice=None`; G4's virtual display does not
  support that option. Removing it allowed NVIDIA rendering.
- OBS probes first rejected the Ubuntu build suffix, then reached the control
  socket before OBS was ready. Exact package/API version checks and the
  documented readiness response resolved both cases.
- A focused test expected the label `Cody`; the established label is `iBDW`.
  The test was corrected. Ruff also found import ordering and a nested context
  statement; both were corrected.
- OBS's first-run wizard blocked output. `FirstRun=true` prevents the wizard.
  The window capture API returns `AppRun.wrapped`, not the second WM_CLASS
  string shown by `xwininfo`. The selector now uses the observed API value.
- OBS selected Mesa because its EGL registration file was missing. The image
  now includes NVIDIA's registration. The final log has no CUDA/OpenGL
  interoperability error.
- The first required Dolphin test run stalled and was interrupted after
  266 seconds. A visible-output retry timed out at 180 seconds and reported
  port 51441 in use. After the local match finished, the loop was paused and
  the exact required command passed in 55.89 seconds. Test-owned blocked
  processes were reaped. The timeout run reported leaked shared-memory cleanup.
- `uvx py-spy dump --pid ...` lacked ptrace permission. The sudo retry required
  interactive authentication. `unshare --user --map-root-user --net ...` was
  denied. Pausing the match loop avoided both requirements.
- Early source lookups used an obsolete `.cpp` path and an unquoted URL query;
  corrected requests fetched OBS's `.c` source. Exploratory reads of
  `compose.yml` and `live_settings.py` failed; the maintained files were then
  located. An early `xwininfo` call targeted an image without that tool; the
  host tool and container namespace supplied the diagnostic instead.
- A diagnostic `python -c` from `/opt/hal` found the base source tree before
  the installed patched module. Running it with `docker exec -w /tmp` selected
  the installed code. Diagnostic direct websocket connections emitted a
  deprecation warning; their sockets were closed.

Skips and limits:

- Two hardware tests require `HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1`.
- Six schema tests require the unavailable local v7 subset. That fixture is
  still absent; these skips are not evidence that those schema tests passed.
- The stream-off and second-slot performance controls remain unmeasured.
- Twitch viewer latency and reboot recovery were not tested.
- No push or new asset upload was needed. The live stream remains public under
  the owner's existing approval.

## x_pilot run — 2026-09-29

The local Cody Fox loop has stopped. G4 now plays x_pilot's `PHAI#591` bot
with `MASTER` player conditioning, raw desired return 120, delay 2, and
temperature 1. The supported schedule has 91 of the 96 CPU evaluation
pairings. `gm-v2` lacks Ganondorf, Dr. Mario, and Ness for five rows.

The first two verified games were HAL Fox against `gm-v2-falco` and
`gm-v2-marth`; HAL lost both. The first complete game averaged 58.7 FPS,
with frame interval p95 18.0 ms and policy round-trip p95 7.4 ms. A sample
during the second game measured 59.95 game FPS and 60.00 OBS FPS. OBS had
zero network drops and no new encoder skips in that second sample.
These opponent and network conditions differ from the earlier local run.

The effective source is `e942072e`. It includes the shared return range fix
from `1dd743b2` and bounded Slippi receiver cleanup. The cleanup fix was
installed after both recordings were saved. It prevents a full receiver
pipe from blocking the next reservation. The source remains mounted under
`/var/lib/hal-netplay/hotfix/obs-v1`; no registry image was pushed.

Slippi recordings and verified results are retained locally under
`runs/netplay/x-pilot-master120/`. Operator credentials now use encrypted,
host-bound storage under `runs/netplay/credentials/`. The G4 continues to
use its existing Secret Manager secret. See [the run record](x-pilot.md)
for controls, credential handling, all commands, test results, and limits.

## Waiting card — 2026-09-29

Superseded by the continuous emulator capture change below.

Source `f5976b7c` replaces the empty background between games with a navy
1080p card: `NEXT MATCH`, `Getting the next game ready`, and the site link.
The existing overlay still shows queue depth. OBS builds the card in a
separate scene and places it below Dolphin. The opaque 1920×1080 Dolphin
window covers it during gameplay; hiding that window reveals the card.
Connection menus and connect codes remain hidden.

The card uses native OBS color and text sources. It adds no image assets,
downloads, browser process, or R2 objects. The scene was first updated live.
Screenshots confirmed the card and unchanged game capture. After game nine
was saved, the runner was restarted to load the new startup configuration.
The source bind mount is `/var/lib/hal-netplay/hotfix/obs-v1/obs.py`.

Before the change, OBS reported 60.00 FPS, zero encoder skips, zero network
drops, and three render skips over roughly 70,900 frames. After adding the
card, OBS still reported 60.00 FPS; all three counters were unchanged over
roughly 83,600 frames. Average render time changed from 0.246 ms during play
to 0.138 ms while idle. These different states are not a controlled speed
comparison. The earlier game sample was 59.93 FPS.

### Commands and checks

Commands ran in this worktree; npm commands ran in `web/netplay-api`.
Repeated read-only probes are grouped below.

| Command or operation | Result |
| --- | --- |
| `cat AGENTS.md`; `git status --short`; `git branch --show-current`; `sed`, `cat`, and `rg` over OBS, its supervisor, tests, and the streaming spec | Confirmed the branch, clean starting state, capture privacy boundary, and missing background. |
| Official obs-websocket 5.5.2 protocol reads | Confirmed scene creation and ordering requests. Two raw OBS color-source fetches failed with cache misses; the live `GetInputKindList` confirmed `color_source_v3`. |
| `uv run pytest -q tests/test_netplay_obs.py tests/test_netplay_stream.py` | 16 passed. Includes card placement, startup composition, capture transitions, and absence of connect codes. |
| `uv run ruff format hal/netplay_service/obs.py tests/test_netplay_obs.py` | One file formatted; one unchanged. |
| `uv run ruff format --check .` | 270 files passed. |
| `uv run ruff check .` | Passed. |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts` | Passed with zero diagnostics. |
| `uv run pytest -q -m 'not integration'` | 1,493 passed, 8 skipped, 21 deselected, 24 warnings in 141.80 s. |
| `npm test` | 105 passed in 10 files. workerd printed its WebSocketPipe disconnect diagnostic; no tests failed. |
| `npm run typecheck` | Passed. |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_netplay_queue_integration.py -m integration` | 3 passed in 41.72 s. |
| Dolphin round-trip and cleanup integration suite | Not rerun: this change only adds OBS scene sources. Session stepping, controller input, replay extraction, and Dolphin configuration are unchanged. The previous required run passed seven tests. |
| `gcloud compute ssh hal-netplay-g4 --project centering-star-502613-k3 --zone us-west1-a --command ...` | Read source kinds, scene geometry, frame counters, and slot status. The first automatic approval review timed out; one allowed retry succeeded. No secrets were printed. |
| `gcloud compute scp hal/netplay_service/obs.py hal-netplay-g4:/tmp/hal_obs_waiting.py --project centering-star-502613-k3 --zone us-west1-a` | Copied the source for live configuration and the persistent mount. SHA-256: `a24756ccf7a5dcd5084756c68845692b492c22b3f074c31ab3f7e9193b6e92f6`. |
| Remote `docker cp`, authenticated OBS `CreateScene`, `CreateInput`, `CreateSceneItem`, `SetSceneItemIndex`, and `SetSceneItemEnabled` | Built the card off air, then enabled it below the active capture. The ongoing game continued. |
| OBS `GetSourceScreenshot`, `GetSceneItemList`, `GetStats`, and `GetStreamStatus`; `gcloud compute scp ... /tmp/` | Saved and inspected the card and game screenshots. The actual idle-program check also confirmed Dolphin was disabled. |
| `touch runs/netplay/x-pilot-master120/stop-after-game`; `tail` of the event log | The scheduler finished and verified game nine, then stopped before row ten. |
| Remote source hash check, `systemctl daemon-reload`, runner stop/start | Installed the committed OBS module and effective source label `f5976b7c` at the completed-game boundary. |
| `git diff --check`; `git diff`; focused `git add`; `git commit -m 'Show a waiting card between matches'` | Passed; committed `f5976b7c`. Format, lint, and type commit hooks passed. No attribution trailers. |

The eight skips are the same two opt-in production GPU checks and six tests
with an absent optional local v7 subset. Warnings concern Python 3.14
TorchScript, uncompiled flex attention, and fork from a threaded process.
No new cloud resource, Worker deployment, registry push, R2 asset upload,
or Git push was performed.

## Continuous emulator capture — 2026-09-29

The owner requested that Dolphin remain visible in menus and during
connection, with no waiting card. Source `3d2bdc40` implements that request.
OBS now selects the exact Dolphin render window whenever it exists. Window
visibility no longer depends on the game's state or heartbeat. The launcher
and desktop remain outside the capture. When Dolphin exits between
reservations, there is no emulator window to capture until the next launch.

The waiting-card scene and its code were removed. The overlay still uses
HAL's labels and queue depth without adding a connect code. Dolphin's own
menus are visible as requested. The spec records this approved change.

The rollout used the existing `obs.py` and `stream.py` source mounts. The
scheduler saved row 47 and paused before row 49; row 48 is unsupported.
All 46 completed recordings were retained before the runner restart.

The live check captured Dolphin's name-entry menu while the slot was
`connecting`, then confirmed normal gameplay. The resumed game measured
59.94 FPS, frame interval p95 18.00 ms, and policy round-trip p95 7.58 ms.
OBS measured 60.00 FPS with zero encoder skips and zero network drops over
roughly 12,500 frames. Its four render skips include startup.
Before the restart, idle OBS measured 60.00 FPS with zero encoder skips,
five render skips, and 39 network drops over roughly 435,600 frames.
Counters reset on restart; these samples are not a controlled comparison.

### Commands and checks

| Command or operation | Result |
| --- | --- |
| `cat AGENTS.md`; `git status --short`; targeted `sed`, `cat`, and `rg` over OBS, its supervisor, tests, spec, and README | Confirmed the existing capture gate, card, callers, and clean worktree. |
| `uv run pytest -q tests/test_netplay_obs.py tests/test_netplay_stream.py` | 16 passed. Covers capture with game and waiting labels, window loss, no card at startup, and overlay privacy. |
| `uv run ruff format hal/netplay_service/obs.py hal/netplay_service/stream.py tests/test_netplay_obs.py tests/test_netplay_stream.py` | Four files already formatted. |
| `uv run ruff format --check .` | 270 files passed. |
| `uv run ruff check .` | Passed. |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts` | Passed with zero diagnostics. |
| `uv run pytest -q -m 'not integration'` | 1,493 passed, 8 skipped, 21 deselected, 24 warnings in 141.67 s. |
| `npm test` in `web/netplay-api` | 105 passed in 10 files. workerd printed its WebSocketPipe disconnect diagnostic; no tests failed. |
| `npm run typecheck` in `web/netplay-api` | Passed. |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_netplay_queue_integration.py -m integration` | 3 passed in 41.64 s. |
| Dolphin round-trip and cleanup integration suite | Not rerun: this change affects OBS capture selection and overlay dispatch only. Session stepping, controller input, replay extraction, and Dolphin configuration are unchanged. |
| `touch runs/netplay/x-pilot-master120/stop-after-game`; `tail` of its event log | Stopped at the completed-game boundary after row 47. |
| `gcloud compute ssh hal-netplay-g4 --project centering-star-502613-k3 --zone us-west1-a --command ...` with authenticated OBS requests | Removed the live waiting-card scene and selected Dolphin if present. At that moment the prior game had closed Dolphin. Saved baseline counters without printing credentials. |
| `gcloud compute scp hal/netplay_service/obs.py hal/netplay_service/stream.py hal-netplay-g4:/tmp/ --project centering-star-502613-k3 --zone us-west1-a` | Staged the source modules. Remote SHA-256 checks matched the committed local files. |
| Remote `systemctl daemon-reload`, runner stop/start, and readiness polling | Loaded the persistent source mounts with effective source `3d2bdc40` after the recording was saved. |
| Python run-manifest update and runtime download; `systemd-run --user --unit=hal-xpilot-games ... hal_xpilot_run.py --first-job .../private/first-game.json`; `journalctl --user -u hal-xpilot-games -n 5 --no-pager` | Recorded the source change and new sampling seeds. Resumed at row 49 with `MASTER`/120 unchanged. |
| Authenticated OBS menu-to-game probe; `gcloud compute scp hal-netplay-g4:/var/lib/hal-netplay/continuous-menu.png /tmp/ --project centering-star-502613-k3 --zone us-west1-a`; local image inspection | Verified the actual name-entry menu during connection, no waiting-card scene, and normal gameplay. Saved performance counters in `continuous-capture-after.json` on G4. |
| `git diff --check`; `git diff --stat`; `sha256sum` of both source files; focused `git add`; `git commit -m 'Keep Dolphin menus on stream'` | Passed. Created `3d2bdc40`; format, lint, and type commit hooks passed. No attribution trailers. |

No command failed. The eight skips remain two opt-in production GPU checks
and six tests with an absent optional local v7 subset. Warnings concern
Python 3.14 TorchScript, uncompiled flex attention, and threaded fork.
No cloud resource, Worker deployment, registry push, R2 asset upload, or Git
push was added.

## Deployment cleanup — 2026-09-29

This cleanup updates the current deployment summary and the spec's Plans table.
It removes stale instructions for hiding menus and showing a waiting card.
Runtime source remains `3d2bdc40`. No runner restart or deployment was needed.

Three duplicate operator scripts were removed from `/tmp` after SHA-256
comparison with their retained `control/` copies and a process-argument check.
The active services use the retained copies. Recordings, credentials, runtime
snapshots, screenshots, and remote rollback files were preserved.

The matchup service had stopped after row 49 because x_pilot did not
acknowledge the next play command. The canceled attempt had zero completed
games. Its failure was recorded before resuming the same row. Row 50 then
reached `playing`; both local operator services were active. At that point,
47 completed recordings were verified. Progress continues in the local log.

A read-only G4 sample after recovery reported 59.94 game FPS, frame interval
p95 18.13 ms, and policy round-trip p95 6.80 ms. OBS reported 60.00 FPS, zero
encoder skips, zero network drops, and four render misses since startup over
roughly 36,700 frames. This was a health check, not a performance treatment.

### Commands, failures, and skips

Commands ran in this worktree. Repeated inspections are grouped by purpose.

| Command or operation | Result |
| --- | --- |
| `cat AGENTS.md`; `git status --short`; `git branch --show-current`; `git log -12 --oneline` | Confirmed the required worktree, `netplay-edge-queue`, and clean starting state. |
| Targeted `cat`, `sed`, `rg`, and Python reads of the spec, README, run protocol, Worker routing/schema, Vite/generated Wrangler config, runner, OBS, fixtures, replay uploader, Compose, and GCE startup | Checked the architecture against source and found stale deployment text. |
| `cat web/netplay/wrangler.jsonc` | Failed: that file does not exist. The page uses `vite.config.ts` and generated `dist/server/wrangler.json`; both were read successfully. |
| `gcloud compute instances describe hal-netplay-g4 --project centering-star-502613-k3 --zone us-west1-a --format='json(name,status,machineType,guestAccelerators,networkInterfaces[].accessConfigs[].natIP)'` | Running `g4-standard-48`, exactly one RTX PRO 6000, external IP `34.177.115.232`. |
| `gcloud compute ssh hal-netplay-g4 --project centering-star-502613-k3 --zone us-west1-a --command=...` with filtered Python `docker inspect`, `systemctl is-active`, `dpkg-query -W`, `obs --version`, and status-file reads | Runner active. Verified local image, eight source mounts, source SHA, OBS package/binary versions, and live metrics. No credentials were printed. |
| `uv run python` with public HTTP GETs of `https://20xx.xyz/` and `/v1/capacity` | Both HTTP 200; one healthy slot. |
| `systemctl --user is-active`, `systemctl --user show`, `journalctl --user -u hal-xpilot-games.service`, and local event/chat reads | Found the scheduler's exit-code failure after the unacknowledged play command. The chat listener remained active. |
| `uv run python` with an authenticated GET of the failed job and an append to `events.jsonl` | Confirmed `canceled`, zero games; retained the failed attempt before recovery. |
| `systemctl --user restart hal-xpilot-games.service`; subsequent service and event checks | Resumed row 50; both services active and the game reached `playing`. |
| Python SHA-256, `/proc` argument, and service `ExecStart` checks; removal of `/tmp/hal_credentials.py`, `/tmp/hal_xpilot_chat.py`, `/tmp/hal_xpilot_run.py` | All three were unused duplicate copies. The active `control/` files were preserved. |
| Python updates to README, spec, and run protocol | Corrected deployment facts without changing runtime code or configuration. |
| `git diff --check`; `git diff --stat`; review of the full documentation diff | Passed; changes are limited to three Markdown files. |
| Python local Markdown link and schema/capture checks | Passed: four local links, schema 2, instance name `global`, and current capture instructions. |

The runtime gates are recorded under Continuous emulator capture above:
1,493 Python tests and 105 Worker tests passed; all format, lint, type, and
queue integration checks passed. Eight Python tests were skipped: two opt-in
GPU qualification checks and six tests with the absent optional local v7
subset. These remain skips, not passes. No tests were rerun for this
Markdown-only cleanup. The diff and local documentation links were checked.
Reboot, managed replacement, stream lease transfer, matched stream-on/off
performance controls, and viewer latency remain unverified. No image push,
new cloud resource, R2 asset upload, Worker deployment, or Git push occurred.


## Ranked value meter — 2026-09-30

The value head now accompanies every inference response. Ranked displays
its six-game-frame EMA in a separate OBS overlay process. Restarting that
process reloads the meter without restarting the player or OBS.

Matched G4 inference p50/p95/p99 changed from **5.051/5.085/5.095 ms** to
**4.678/4.743/5.049 ms**. Each run used 300 measured samples and the same
bundle, inputs, seed, conditioning, and timing. Action sequences matched
exactly; neither run compiled or captured CUDA graphs while serving.

See [Ranked deployment](ranked.md#value-meter--2026-09-30) for the 3060
comparison, precision check, runtime files, and overlay reload procedure.


The first live game with the meter averaged **58.914 emulator FPS**, with
**18.053 ms frame p95** and **6.658 ms inference p95**. OBS stayed at 60 FPS
with no encoder/network drops. An overlay restart during the game preserved
the Ranked, Dolphin, and OBS process IDs and added no dropped stream frames.
The completed replay uploaded, and the helper started the next game.
