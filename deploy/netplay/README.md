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
needs ffmpeg with `h264_nvenc` and PulseAudio. The runner image installs all
four. The launchers reject a missing command instead of changing behavior.

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

The runner starts one Xvfb for every slot. Stream slot 0 uses a 1280×720
display and PulseAudio. Other slots use 640×480 displays and no audio. ffmpeg
captures slot 0 at 60 fps, uses NVENC at 6 Mb/s CBR and AAC at 160 kb/s, and
reloads its overlay from a local text file. The overlay state contains HAL's
character, imitation, desired return, and game count only. It has no field for
a player or bot connect code.

### Owner manual stream check

This check is externally visible. Use `HAL_TWITCH_BANDWIDTH_TEST=1` unless the
owner explicitly approves a public stream. Confirm the ffmpeg target ends in
`?bandwidthtest=true` for a bandwidth test before continuing.

- [ ] Record the full Git SHA, policy SHA, hardware, slot count, delay, stage,
  peer and region, driver, and ffmpeg version.
- [ ] Start the runner with `HAL_TWITCH_BANDWIDTH_TEST=1`. Confirm admin status
  shows one stream holder and that no second session receives the lease.
- [ ] While slot 0 is idle, confirm the video shows `Play HAL at 20xx.xyz` and
  the public queue depth. Confirm a waiting job goes to slot 0 before another
  idle slot.
- [ ] Play a game on slot 0. Confirm 1280×720 video, game audio, the HAL setting
  line, and no player or bot connect code anywhere in the picture.
- [ ] Run `pkill -TERM ffmpeg` inside the runner container. Confirm the
  supervisor restarts it with bounded backoff and the runner keeps its game.
- [ ] Drain the holder. Confirm ffmpeg stops, the lease becomes free, and the
  next opted-in live session takes it on a status report.
- [ ] Stop every runner. Confirm Xvfb, PulseAudio, Dolphin, and ffmpeg processes
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
