# HAL netplay deployment

The frontend is an independent static Sites project in `web/netplay`. Its npm
dependencies are not HAL dependencies. Build it with the public API origin:

```text
cd web/netplay
NEXT_PUBLIC_HAL_API_URL=https://api.example.com npm run build
```

The API and runner use the optional Python extra:

```text
uv sync --extra netplay-server
```

The runner uses eager PyTorch by default. Pass `--compiled` only after the
deployed policy and GPU have passed the no-recompile latency qualification.

## Host setup

Use the tested Slippi 3.6.4 AppImage and the same Melee CISO used by local
qualification. Do not replace either file without running the compatibility and
timing tests. The runner checks the Dolphin executable hash before each launch.

Copy `deploy/netplay/.env.example` to an untracked file outside the repository.
Set the exact deployed Git SHA, policy bundle, two account JSON files, game
assets, R2 credentials, Sites origin, API hostname, and Cloudflare tunnel token.
Each slot's account must have a distinct connect code.

Configure the Cloudflare tunnel hostname to send traffic to `http://api:8080`.
The API has no host port and the runner receives no inbound Internet traffic.

Start the service:

```text
docker compose --env-file /secure/path/hal-netplay.env \
  -f deploy/netplay/compose.yaml up --build -d
```

Install the 30-day R2 lifecycle rule:

```text
docker compose --env-file /secure/path/hal-netplay.env \
  -f deploy/netplay/compose.yaml exec runner \
  hal-netplay-admin install-replay-lifecycle
```

SQLite state and replay upload spools live in the `state` volume. Back up this
volume. Do not run two runner containers against the same two Slippi slots.

Production replays are private at:

```text
r2://hal/netplay/v1/replays/YYYY/MM/DD/<PLAYER-CODE>/<reservation-id>/game-NN.slp
```

Each replay has a matching JSON object. A failed upload leaves the replay and an
`.upload.json` sidecar in the state volume. An idle slot retries these files.
Soak tests use `soak_replay_directory()` instead and never upload their replays.
