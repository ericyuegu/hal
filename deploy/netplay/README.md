# HAL netplay deployment

The API and runner use the optional Python extra:

```text
uv sync --extra netplay-server
```

The runner and `hal-play` use eager PyTorch by default. The production Compose
command passes `--compiled`; deploy it only after the policy and GPU pass the
batch-one no-recompile qualification. Compose intentionally runs one slot. Do
not raise its capacity until two concurrent live games pass the batch-two
qualification and the frame-latency limits below.

The runner reports rolling game FPS and p95 frame interval, Dolphin step,
policy round trip, model inference, and batching wait through `/v1/capacity`
and `/metrics`. After 120 playable frames, a slot is degraded below 59 FPS,
above a 20 ms frame-interval p95, or above its delay-specific policy deadline.

Eight continuous seconds of bad frame cadence closes Dolphin and retries the
reservation once. A two-second frame-stream stall retries immediately. Slow
inference removes the slot from healthy capacity but does not restart Dolphin.
An inference-engine or worker-process failure exits the runner, and Compose's
`unless-stopped` policy restarts it. The API rejects new reservations only when
no healthy slot remains.

## Requirements

The host needs Docker Engine, Docker Compose, the NVIDIA Container Toolkit, and
an NVIDIA GPU that passed `tests/test_netplay_hardware.py`. Confirm that a
container can use the GPU before deployment.

Qualify the compiled policy on the deployment GPU:

```text
HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1 \
HAL_NETPLAY_POLICY=/absolute/path/to/policy.halpolicy \
uv run pytest -q tests/test_netplay_hardware.py -m integration
```

Use the tested Slippi 3.6.4 AppImage and the same Melee CISO used by local
qualification. Do not replace either file without running the compatibility and
timing tests. The runner checks the Dolphin executable hash before each launch.

## Run one slot

From the repository root, copy the environment template to a protected path
outside the repository:

```text
cp deploy/netplay/.env.example /secure/path/hal-netplay.env
```

Set all values in that file. `HAL_GIT_SHA` must be the full commit that you
will deploy. Set the policy bundle, one account JSON file, game assets, R2
credentials, Sites origin, API hostname, and Cloudflare tunnel token.

Configure the Cloudflare tunnel hostname to send traffic to `http://api:8080`.
The API has no host port and the runner receives no inbound Internet traffic.

Validate the resolved Compose configuration:

```text
docker compose --env-file /secure/path/hal-netplay.env \
  -f deploy/netplay/compose.yaml config --quiet
```

Build and start the API, one-slot runner, and tunnel:

```text
docker compose --env-file /secure/path/hal-netplay.env \
  -f deploy/netplay/compose.yaml up --build -d
```

Check container state, API readiness, capacity, and runner logs:

```text
docker compose --env-file /secure/path/hal-netplay.env \
  -f deploy/netplay/compose.yaml ps
docker compose --env-file /secure/path/hal-netplay.env \
  -f deploy/netplay/compose.yaml exec api \
  curl --fail http://localhost:8080/health/ready
docker compose --env-file /secure/path/hal-netplay.env \
  -f deploy/netplay/compose.yaml exec api \
  curl --fail http://localhost:8080/v1/capacity
docker compose --env-file /secure/path/hal-netplay.env \
  -f deploy/netplay/compose.yaml logs --tail=100 runner
```

The capacity response must report one healthy slot before users submit jobs.
Use `docker compose logs -f runner api` with the same environment file and
Compose file options for live logs.

## Run without Docker

Install the Python and frontend dependencies once:

```text
uv sync --extra netplay-server
cd web/netplay
npm ci
cd ../..
```

The direct host deployment uses the same completed environment file as the
Compose deployment. Run each command from the repository root in a separate
terminal. Install `xvfb-run` and `cloudflared` on the host before you start.

Start the API:

```text
set -a
source /secure/path/hal-netplay.env
set +a
HAL_NETPLAY_DATABASE="$PWD/runs/netplay/queue.sqlite3" \
HAL_NETPLAY_CAPACITY=1 \
HAL_NETPLAY_ALLOWED_ORIGINS=https://play.example.com,http://localhost:3000,http://127.0.0.1:3000 \
HAL_NETPLAY_ALLOWED_HOSTS=api.example.com,localhost,127.0.0.1 \
HAL_NETPLAY_RUNNER_STATUS="$PWD/runs/netplay/runner-status.json" \
uv run hal-netplay-api --host 127.0.0.1 --port 8080
```

Start the one-slot runner. `xvfb-run` gives Slippi the display that its Vulkan
renderer requires:

```text
set -a
source /secure/path/hal-netplay.env
set +a
xvfb-run -a uv run hal-netplay-runner "$HAL_NETPLAY_POLICY" \
  --compiled \
  --database "$PWD/runs/netplay/queue.sqlite3" \
  --user-jsons "$HAL_NETPLAY_USER_JSON_A" \
  --slippi-ports 51441 \
  --iso-path "$HAL_ISO_PATH" \
  --dolphin-path "$HAL_NETPLAY_EMULATOR_PATH" \
  --replay-dir "$PWD/runs/netplay/replays" \
  --status-path "$PWD/runs/netplay/runner-status.json" \
  --git-sha "$HAL_GIT_SHA"
```

Start the configured Cloudflare tunnel to expose the API:

```text
set -a
source /secure/path/hal-netplay.env
set +a
cloudflared tunnel --no-autoupdate run \
  --token "$CLOUDFLARE_TUNNEL_TOKEN"
```

Check the direct API and runner:

```text
curl --fail http://127.0.0.1:8080/health/ready
curl --fail http://127.0.0.1:8080/v1/capacity
```

For local frontend development, run:

```text
cd web/netplay
NEXT_PUBLIC_HAL_API_URL=http://localhost:8080 \
npm run dev -- --host 127.0.0.1 --port 3000
```

These processes stop when their terminals close. Use systemd or another process
supervisor for automatic restart on a production host.

## Frontend

The frontend is an independent static Sites project in `web/netplay`. Its npm
dependencies are not HAL dependencies. Build it with the public API origin:

```text
cd web/netplay
npm ci
NEXT_PUBLIC_HAL_API_URL=https://api.example.com npm run build
```

Deploy the built Worker and static assets after `npx wrangler login`:

```text
npx wrangler deploy --config dist/server/wrangler.json
```

Open the deployed frontend hostname to submit a job. Its exact origin must be
present in `HAL_NETPLAY_ALLOWED_ORIGINS` when the API starts.

## Operations

Install the 30-day R2 lifecycle rule:

```text
docker compose --env-file /secure/path/hal-netplay.env \
  -f deploy/netplay/compose.yaml exec runner \
  hal-netplay-admin install-replay-lifecycle
```

Restart only the runner after a transient GPU or Dolphin fault:

```text
docker compose --env-file /secure/path/hal-netplay.env \
  -f deploy/netplay/compose.yaml restart runner
```

Stop the service without deleting its named state volume:

```text
docker compose --env-file /secure/path/hal-netplay.env \
  -f deploy/netplay/compose.yaml down
```

SQLite state and replay upload spools live in the `state` volume. Back up this
volume. Do not run two runner containers against the same Slippi account.

Production replays are private at:

```text
r2://hal/netplay/v1/replays/YYYY/MM/DD/<PLAYER-CODE>/<reservation-id>/game-NN.slp
```

Each replay has a matching JSON object. A failed upload leaves the replay and an
`.upload.json` sidecar in the state volume. An idle slot retries these files.
Soak tests use `soak_replay_directory()` instead and never upload their replays.
