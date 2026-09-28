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

## Host run

Set the runner URL and token, Cloudflare Access service token, R2 credentials,
full Git SHA, and slot count in `.env`. Then run:

```sh
deploy/netplay/run-host.sh
```

The first signal drains the remote session. A second signal or the 15 minute
deadline aborts active games. A hard crash is covered by the Worker's 30 second
session silence limit.

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

Running this checklist creates billable resources. It is prepared for the
owner and was not run during Plan 4.

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

Plan 4 live G4 result: **not run; owner action required**.

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
