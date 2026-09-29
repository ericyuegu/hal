# HAL netplay deployment

The queue runs in `web/netplay-api`. A GPU host runs only `hal-netplay-runner`.
The runner downloads and verifies its static fixtures, starts a remote session,
keeps that session alive while it downloads policy and account assets, qualifies
both delay profiles, and then starts its slots.

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
Connecting, idle, missing, and stale game windows are hidden.

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
- [ ] While slot 0 is idle, confirm the video shows `Play HAL at 20xx.xyz` and
  the public queue depth. Confirm a waiting job goes to slot 0 before another
  idle slot.
- [ ] Play a game on slot 0. Confirm 1920×1080 video, game audio, the HAL setting
  line, and no player or bot connect code anywhere in the picture.
- [ ] Run `pkill -TERM obs` inside the runner container. Confirm the
  supervisor restarts it with bounded backoff and the runner keeps its game.
- [ ] Drain the holder. Confirm OBS stops, the lease becomes free, and the
  next opted-in live session takes it on a status report.
- [ ] Stop every runner. Confirm runner-owned Xvfb, Openbox, PulseAudio, Dolphin, and OBS processes
  are gone.

The owner approved a public stream for the 2026-09-29 G4 test. Twitch showed
the live 1280x720 game. The active-game display contained no connect code.
ffmpeg 6.1.1 used NVENC while the policy ran. The idle-card, restart, and
lease-transfer checks remain pending.

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
no desktop, launcher, or connect code. OBS has exactly three inputs: the
Dolphin window, the overlay, and `hal_stream.monitor`. The window is hidden
outside a fresh playing state. The first-run wizard is disabled in the OBS
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
