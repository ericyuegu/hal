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
