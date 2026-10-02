# Shared queue control rollout — October 2, 2026

## Release state

Implementation and local validation are complete. Deployment is in progress.
The G4 has not been restarted. Google Cloud needs a fresh `gcloud auth login`.
The production capacity endpoint still returned HTTP 500 / Cloudflare 1101
before deployment. The reported request quota resets October 3 at 00:00 UTC.

The rollout uses runner protocol 4, storage schema 7 and `global-v7`.
Old production objects and all R2 replays are preserved. No plan upgrade,
new VM, new account, credential upload or storage deletion is part of this change.

## Architecture

- Static HTML and JavaScript load from Cloudflare's asset service.
- Each browser opens one hibernating WebSocket for capacity and its own job.
  Its token is sent in the subscription message, not the URL. Public messages
  contain no connect code or stream key. Keepalives have no SQL writes.
- The G4 supervisor opens one authenticated WebSocket. It sends health and fresh
  slot progress every ten seconds. The object updates one session row.
- Eight slot processes use an authenticated loopback relay. Idle checks and
  periodic observations stay on G4. Work notifications trigger claims. Changed
  observations, game results, completion and replay records use idempotent HTTP
  commands. Reports and completion include the original attempt number.
- Settings and cancellations are pushed to the affected player and host. After
  reconnect, the host receives current assignments and replaces stale state.
- Inference, controller input, Dolphin, OBS and Twitch traffic stay outside
  Cloudflare. The stream's queue-depth lookup uses the local relay.

Host silence remains 30 seconds. Nonplaying leases remain 20 seconds; playing
leases remain 60 seconds. Capacity freshness is now 20 seconds. A healthy host
cannot renew a frozen slot: the relay requires fresh slot health and observations,
and the object requires advancing progress for the same job attempt.

The frame loop does not wait on the relay or Cloudflare. Reports run in the
existing background thread. No inference schedule, precision or model changed.

## Limits and overload

The object accepts at most 100 browser sockets. A reconnect replaces a host's
previous socket. Messages are limited to 16 KiB. Larger snapshots use numbered
pages, with a 256 KiB assembly limit. Unacknowledged output is bounded; slow
consumers must reconnect. Browsers back off after failed connections.

A persistent UTC-day write budget gates new reservations and settings. Existing
games receive a wind-down instruction before the completion reserve is needed.
A 100 MiB database guard stops admissions and lets games finish. Admin status
includes database bytes and the write budget. Neither guard deletes history.

`MAINTENANCE=on` returns a small Worker response without touching the object.
It still consumes a Worker request. Logs and traces sample one percent of events.
These controls do not guarantee free service against unlimited traffic or other
applications consuming the same account's allowance.

R2 transfers remain direct between G4 and R2. Replay retention is unchanged.
Keeping every replay indefinitely cannot guarantee free R2 storage.

## Measured local load

`web/netplay-api/test/day-costs.test.ts` runs one UTC day with:

- eight assigned slots and twenty queued players;
- 100 browser sockets and ten-second host health messages;
- 10,000 settings command attempts;
- 10,000 completed jobs and 10,000 completed sessions already stored;
- 5,000 old events due for retention cleanup;
- a conservative alarm every twenty seconds.

| Counter | Result |
| --- | ---: |
| SQL rows read | 564,421 |
| Recurring writes, including conservative alarm charges | 17,304 |
| Settings and accounting writes | 17,392 |
| Retention writes | 5,000 |
| Total writes | 39,696 |
| Settings accepted | 4,348 |
| Settings refused after the budget reserve was reached | 5,652 |

The test counts cursor-reported SQL work and adds two storage writes per alarm.
Browser presence receipt times and acknowledgements are simulated through the
runtime interfaces. This is a control-plane test, not 24 hours of gameplay or
production billing telemetry. Focused tests cover game results, replay retries,
lease expiry, reconnection, cancellation and stale attempts separately.

The design's request scenario remains 25,000 dynamic Worker requests and 44,832
base DO requests per day. Reserve a further 5,000 DO requests for subscription
and acknowledgement messages: 49,832 total. Outgoing messages are not billed
requests; incoming socket messages are charged at 20:1. Actual traffic must stay
within the stated connection and command envelope. See `free-tier-design.md`.

## Validation commands

All commands ran in `/home/ericgu/src/hal-edge-queue` or its stated subdirectory.
Main and other worktrees were not modified.

| Command | Result |
| --- | --- |
| `uv run ruff format --check .` | Passed; 293 files. |
| `uv run ruff check .` | Passed. |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts` | Passed. |
| `uv run pytest -q -m 'not integration'` | 1,648 passed; eight skipped; 23 deselected. |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py tests/test_netplay_queue_integration.py -m integration` | 11 passed; six deselected. No missing required fixture. |
| `npm test` in `web/netplay-api` | 99 passed in the final full Worker run. |
| `npm run typecheck` in `web/netplay-api` | Passed. |
| `npx oxfmt app/page.tsx lib/netplay-api.ts` in `web/netplay` | Passed. |
| `npm run lint`, `npx tsc --noEmit`, `npm run build` in `web/netplay` | Passed. Build prerendered two routes; root is static. |
| `uv run pytest -q -rs -m 'not integration' tests/test_netplay_hardware.py tests/test_policy_schema.py tests/test_policy_world_schema.py` | 23 passed; eight skips identified below. |
| `git diff --check` | Passed. |

Focused commands also passed:

- `npm test -- test/live.test.ts`: authentication, private push, keepalive,
  frozen progress, stale attempts, paging, slow consumers and overload.
- `npm test -- test/day-costs.test.ts`: full-day accounting.
- `npm test -- test/costs.test.ts test/day-costs.test.ts --update`: reviewed
  accounting and freshness snapshots after the implementation changed.
- `uv run pytest -q tests/test_netplay_runner.py tests/test_netplay_queue_client.py`:
  111 passed; the final control/runner/client check passed 113 tests.
- `uv run pytest -q tests/test_netplay_control.py`: two passed.
- `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_netplay_queue_integration.py -m integration`:
  four passed. Focused reruns also passed with `-k shared_control` and
  `-k 'not reporter_keeps'`.
- Ruff formatting and focused lint/type commands ran while editing. Final
  repository checks supersede their intermediate results.

### Failures fixed during implementation

- The first write probe used unavailable `python`; subsequent scripts used
  `python3` or `uv run`.
- One `npm run typecheck` ran at the repository root, which has no package.json.
  One edit script used frontend paths from the API directory. Both were rerun
  from the correct directory.
- Automatic approval review timed out once while writing the socket tests.
  Retrying the local file write separately succeeded.
- The old five-second capacity test and SQL snapshots failed after the new
  deadline and lease lookup. Their expected values now match the new contract.
- A socket-close test exposed reserved close code 1005. The handler now uses
  a valid reply code. A settings test used 120 outside its fixture's 0–40 range;
  that test now uses 30.
- New type checks found a missing SQL cursor type argument, an unvalidated slot
  conversion, a JSON mapping access, and a test RPC return typed as never. Each
  was fixed. Test lambdas failed Ruff and were changed to named functions.
- Frontend lint found unused polling imports. They were removed.
- The first daily test read 1,926,721 rows. Alarms were broadcasting unchanged
  jobs. Shared positions, targeted settings push and constant-cost counts fixed
  it. A later mixed test read 1,216,810 rows and crossed two UTC billing days;
  the final test covers one billing day and passes the read target.
- The freshness-alarm regression initially expected 30 seconds. The new
  capacity deadline is 20 seconds; earlier alarms are still retained.

### Skips and limits

- Two tests in `tests/test_netplay_hardware.py` require
  `HAL_REQUIRE_NETPLAY_HARDWARE_QUALIFICATION=1` on the production GPU.
- Six tests using `tests/test_policy_world_schema.py:37` require the optional
  local v7 subset, which is absent.
- Existing test warnings concern Python 3.14 TorchScript, unfused FlexAttention,
  OBS WebSocket context-manager use, and multiprocessing fork from threads.
- G4 startup qualification, live matching, production Worker CPU profiling,
  production reconnect/fan-out timing, and Twitch checks remain blocked until
  Google authentication and the Cloudflare quota permit the relaunch.
- This change has no before/after gameplay FPS result yet. The local control-plane
  measurements above do not substitute for a G4 gameplay measurement.

## Operations checked before deployment

- Read AGENTS.md, the approved design, affected code, tests and skills. `rg`,
  `cat`, `sed`, Git status/diff/branch checks and official Cloudflare documentation
  supplied the implementation evidence. Initial lookups for `src/types.ts`,
  `wrangler.toml`, `transcripts.test.ts` and `deploy-api.sh` found no such files;
  the maintained files were discovered before editing.
- `curl --silent --show-error --max-time 15 --output /tmp/hal-queue-live-status.txt --write-out '%{http_code}\\n' https://20xx.xyz/v1/capacity`
  returned 500 / error 1101 before deployment.
- `npx wrangler whoami` succeeded for the existing account and OAuth login.
- A read-only Cloudflare REST inventory found one SQLite namespace for
  `hal-netplay-api_Queue` and two objects with stored data. No data was deleted.
  Pagination confirmed no further objects. The account has four Workers: the
  two netplay Workers and two others sharing the quota. This endpoint does not
  report database bytes, active alarms or daily usage.
- `gcloud compute instances describe hal-netplay-g4 --project centering-star-502613-k3 --zone us-west1-a --format='value(status,machineType)' --quiet`
  failed because Google requires reauthentication. No VM start was attempted.

## Remaining release steps

1. Commit the tested source and build the runner image from that exact commit.
2. Deploy the API in maintenance and publish the static frontend.
3. After Google login and the quota reset, publish the unchanged production policy
   and HAL#9000 account references into `global-v7`. Keep admissions paused.
4. Push the image, update startup metadata, then start the existing one-GPU G4
   with eight slots. Run its startup qualification and verify shared control.
5. Check live Worker CPU, socket recovery and gameplay. Resume admissions only
   after those checks pass.

Deployment IDs and image results will be appended after those steps run.
