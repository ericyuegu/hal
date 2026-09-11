# HAL netplay deployment

Production runs one compiled policy slot. Use the tested Slippi 3.6.4 AppImage,
the qualified Melee CISO, one Slippi account, and an NVIDIA GPU. The runner
checks the Dolphin hash before each game.

## Run locally

Install `uv`, `npm`, and `xvfb-run`. From the repository root:

```text
cp deploy/netplay/.env.example deploy/netplay/.env
deploy/netplay/run-local.sh
```

Set the local paths and R2 values in `.env`, then open
`http://127.0.0.1:3000`. The command starts the API, one-slot runner, and
frontend. Press Ctrl-C to stop all three. Cloudflare values can stay empty.

## Publish with Cloudflare

Set the public origins, API URL, and tunnel token in `.env`. Install
`cloudflared`, then run these commands in separate terminals:

```text
deploy/netplay/run-host.sh
deploy/netplay/deploy-frontend.sh
```

The scripts install their locked dependencies. If Wrangler requests
authentication, run `npx wrangler login` once.

Configure the Cloudflare tunnel to send the API hostname to
`http://127.0.0.1:8080`. The frontend origin must match
`HAL_NETPLAY_ALLOWED_ORIGINS` exactly. Both scripts accept a different
environment-file path as their only argument. An empty tunnel token disables
Cloudflare.

## Docker alternative

Install Docker Compose and the NVIDIA Container Toolkit. Configure the tunnel
to send the API hostname to `http://api:8080`, then run:

```text
cd deploy/netplay
docker compose up --build -d
./deploy-frontend.sh
```

Compose uses `.env`, keeps state in its `state` volume, and restarts failed
services unless you stop them. Do not run two runners with the same Slippi
account.

## Verify and operate

The capacity response must show one healthy slot before users join:

```text
curl --fail http://127.0.0.1:8080/health/ready
curl --fail http://127.0.0.1:8080/v1/capacity
```

For Docker, use `docker compose logs -f api runner`, `docker compose restart
runner`, and `docker compose down` from `deploy/netplay`.

The runner reports FPS and p95 frame, Dolphin, policy, model, and batching
times. It retries one stalled or persistently slow game after closing Dolphin.
Slow inference pauses new reservations. A policy-worker failure exits the
runner so the process supervisor can restart it.

Install the 30-day R2 replay lifecycle rule once:

```text
uv run --env-file deploy/netplay/.env \
  hal-netplay-admin install-replay-lifecycle
```

## GPU qualification

Before production, qualify the policy and GPU. Delay 2 policy p95 must be below
33.3 ms, delay 3 must be below 16.7 ms, and no compilation can occur after
connect:

```text
HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1 \
HAL_NETPLAY_POLICY=/absolute/path/to/policy.halpolicy \
uv run pytest -q tests/test_netplay_hardware.py -m integration
```
