# Queue read quota fix

Historical record. Status and commands below apply to the recorded release.
See [current deployment status](status.md) and [the runbook](README.md).

The owner paused the evaluation after Cloudflare reported the daily
5,000,000 SQL rows-read limit. The public capacity endpoint returned HTTP 500.
The local scheduler and G4 runner were stopped. The owner then approved the
fix deployment and G4 shutdown. The guest accepted shutdown on September 30;
SSH became unavailable. Google API status verification needs renewed login.
The 48 verified recordings and NSM summary remain local.

## Cause

Each idle slot claimed work, then waited 250 ms. Every successful request
scheduled an alarm, including reads and empty claims. The deadline query
scanned historical jobs twice. Capacity scanned them again to count active
jobs. Session queries also scanned ended sessions. The effect grew with
completed history, even when there was no waiting player.

The code had a partial index containing active jobs. SQLite did not use it
for these narrower status predicates: it does not infer that an arbitrary
subset of an `IN` list implies the index predicate. The fix includes the
complete predicate in these queries. `EXPLAIN QUERY PLAN` confirmed index use.

## Changes

- Deadline, expiry, and active-count queries use the existing active-job index.
- Live session queries look up the IDs held by account leases. Each live
  session owns at least one account; start and end update both atomically.
  Primary-key lookups avoid scanning ended sessions. Multiple slots do not
  cause a session to be counted or ended twice.
- Read routes and empty claims do not read or write the alarm.
- Mutations preserve an earlier scheduled alarm. A new earlier deadline moves
  it forward; a firing alarm processes due work and schedules its successor.
  This avoids a billed alarm write on every heartbeat.
- Events expire at their exact retention deadline. The previous strict `<`
  predicate could retain an event while scheduling its already-due alarm.
- An idle slot waits five seconds before its next claim. The stop event can
  interrupt that wait. Draining is checked before the next claim.

The four SQL schema definitions and `STORE_SCHEMA_VERSION = 2` are unchanged.
No migration, data reset, or new index is required. Protocols, retry behavior,
lease grace, gameplay stepping, and policy settings are unchanged.

## Measurements

Control: source `5636d5e9`. Treatment: this fix. Both probes use the same
workerd runtime, one live session, no queued jobs, and the same history sizes.
They consume each SQL cursor, then sum `rowsRead`. Test-clock settings add a
few reads absent in production. Alarm API writes are checked separately.

| Completed jobs | Empty claim, before → after | Status report, before → after | Capacity, before → after |
| ---: | ---: | ---: | ---: |
| 0 | 13 → 8 | 9 → 11 | 10 → 7 |
| 50 | 111 → 8 | 107 → 11 | 157 → 7 |
| 100 | 211 → 8 | 207 → 11 | 307 → 7 |

These are local measurements, not a production usage audit. The maintained
regression adds 10,000 completed jobs and 10,000 ended sessions. Polls,
heartbeats, and alarm processing have exactly the same measured costs as
with no history, at fixed active work. Each reads fewer than 64 SQL rows.
Read routes and empty claims perform zero SQL writes and zero alarm operations.

The same regression verifies that an earlier alarm survives heartbeats and
object reconstruction, a new earlier lease deadline takes precedence, and
expiry still occurs. It also covers multi-slot session expiry and exact event
retention. Runner tests cover stop, drain, and job pickup after an idle wait.

This change targets recurring traffic. Admin history retrieval still reads
the requested history, and new-job sequence allocation still uses a maximum
over stored jobs. These are not per-poll operations. Free-tier suitability
also depends on total requests, writes, active slots, and viewers.

Raw results and logs are in the ignored `runs/netplay/quota-incident/` directory:
`read-counts.json`, `read-counts-after.json`, the probe source and snapshots,
and the `fix-*.log` files. No emulator throughput run was performed: the
runner is paused, and this change does not alter frame stepping or inference.

## Deployment — September 30, 2026

The owner approved deployment and asked to take the GPU worker down.

- Source: `a425fc2946cda915a66bde53b9e834fa38e20999`.
- API Worker: `hal-netplay-api`, route `20xx.xyz/v1/*`.
- Active version: `ed23da11-c3ae-4208-aa79-586ba38feb99`, deployed at
  `2026-09-30T14:50:49.618Z`. Wrangler confirmed 100% traffic on this version.
- Remote bindings and route matched local configuration before deployment.
  The existing Durable Object and secrets remain in place. Schema stays 2.
- G4 runner source was staged at `2026-09-30T14:51:39.499698+00:00` in
  `/var/lib/hal-netplay/hotfix/obs-v1/runner.py`. Its SHA-256 is
  `821ea8074e731c9dba9836f2d73214adff62c052d188bfc22526e241964d9f30`.
  The prior file matched committed source `3d2bdc40`. Both the file and the
  systemd override have rollback copies. The override now records the full
  new source SHA. `systemctl daemon-reload` succeeded; the runner was not started.
- The run manifest points to `runtime-staged-a425fc29.json`. It marks this
  source as pending, not as code that produced any completed game.
- `sudo shutdown -h now` on `hal-netplay-g4` returned zero. SSH first remained
  available during shutdown, then reset the connection and became unavailable.
  The only attached disk is persistent Hyperdisk Balanced. No disk, image,
  replay, or VM was deleted.

The Google login expired before the stop. `gcloud compute instances describe`
failed with a reauthentication error. Direct SSH used the already trusted host
key and the existing key pair. The guest accepted the documented shutdown
command. Final `TERMINATED` status in the Google API remains **unverified**.
After `gcloud auth login eric@20xx.xyz`, verify it with:

```sh
gcloud compute instances describe hal-netplay-g4 \
  --project centering-star-502613-k3 --zone us-west1-a \
  --format='value(status)'
```

The public capacity check still returned HTTP 500 after deployment. Deployment
does not reset the exhausted quota. The email gives October 1, 2026, 00:00 UTC
(September 30, 5 PM Pacific) as the reset. Live API health remains unverified.

Keep the evaluation, runner, and stream paused. Resume only when the owner
asks. Before resuming, renew Google authentication, check quota recovery, and
verify the staged source and one healthy runner slot. Record the new runtime
snapshot and source boundary before the next game. Preserve prior results
and failed attempts. The retained boot configuration starts the runner when
the VM starts; a VM start is therefore also a service resume.

The frontend was not redeployed. No paid plan upgrade, registry push, database
reset, R2 upload, game request, stream restart, or Git push was performed.

## Commands and results

Repeated source inspections and probes are grouped by purpose.

| Command or operation | Result |
| --- | --- |
| `cat AGENTS.md`; `git status --short`; `git branch --show-current`; targeted `rg`, `sed`, and `cat` | Confirmed the worktree, clean starting state, query paths, polling intervals, callers, and test seams. |
| Read Workers and Python style skills, plus official Cloudflare alarm documentation | Confirmed alarm coalescing and alarm-handler `getAlarm` behavior. |
| Python in-memory SQLite `EXPLAIN QUERY PLAN` probe | All three revised active-job queries used `idx_jobs_one_active_player`. |
| `npm test -- test/costs.test.ts test/sessions.test.ts test/routes.test.ts test/retries.test.ts` | 45 passed. |
| `uv run pytest -q tests/test_netplay_runner.py` | 61 passed in 4.03 s. |
| `uv run ruff format hal/netplay_service/runner.py tests/test_netplay_runner.py` | One file formatted; one unchanged. |
| `uv run ruff format --check .` | Passed; 270 files formatted. |
| `uv run ruff check .` | Passed. |
| `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts` | Passed with zero diagnostics. |
| `uv run pytest -q -m 'not integration'` | 1,496 passed, 8 skipped, 21 deselected, 24 warnings in 141.40 s. |
| `npm test` in `web/netplay-api` | Final run: 110 passed in 11 files, 7.76 s. Workerd printed its known WebSocketPipe shutdown diagnostic; no test failed. |
| `npm run typecheck` in `web/netplay-api` | Final run passed. The first run found a missing generic argument on the test's `SqlStorageCursor`; it was corrected. |
| `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_netplay_queue_integration.py -m integration` | 3 passed in 41.89 s. |
| `npm --prefix web/netplay-api test -- test/rows-read-probe.test.ts --update` | Same-input measurement probe passed; three snapshots recorded. The temporary test was then archived outside the maintained test tree. |
| Python snapshot parsing and comparison with the incident baseline | Nine matching measurements saved; results appear above. |
| Python comparison of all SQL schema definitions with `git show HEAD:...` | All four definitions were unchanged. |
| `git diff --check` and diff review | Passed. |

An exploratory read of `web/netplay-api/test/events.test.ts` failed because
that file does not exist. Existing event checks are in `sessions.test.ts`;
the new retention regression is in `costs.test.ts`.

The eight Python skips are two opt-in production GPU qualification checks
and six schema tests with the absent optional local v7 subset. They are not
passes. Warnings concern Python 3.14 TorchScript, uncompiled flex attention,
and fork from a threaded process.

The Dolphin round-trip and cleanup integration suite was not rerun. This fix
affects queue SQL, alarms, and the idle wait before a claim. It does not touch
session stepping, controller input, replay extraction, or offline/live parity.

## Deployment commands and results

All local work used `/home/ericgu/src/hal-edge-queue` on `netplay-edge-queue`.
Worker commands used `web/netplay-api`. SSH used the existing key at
`~/.ssh/google_compute_engine`, `BatchMode=yes`, and strict host-key checks.
The verified host alias was `compute.4269454647821964449` in
`~/.ssh/google_compute_known_hosts`; the address was `34.177.115.232`.

| Command or operation | Result |
| --- | --- |
| `cat AGENTS.md`; Git status, branch, log, and targeted source/document reads | Clean starting tree; source `a425fc29`; correct worktree and branch. |
| Read Wrangler and commit-message skills; official Wrangler permissions/commands and Google stop documentation | Confirmed deployment checks and guest shutdown procedure. The initial Wrangler `/commands/deploy/` documentation URL was unavailable; the command index linked to `/commands/workers/`. |
| `cat package.json`; `cat wrangler.jsonc`; `npx wrangler deploy --help`; `npx wrangler whoami` | Local Wrangler 4.124.0; existing account and OAuth scopes support deployment. |
| `gcloud compute instances describe hal-netplay-g4 --project centering-star-502613-k3 --zone us-west1-a --format='json(name,status,machineType,guestAccelerators,disks.deviceName,disks.autoDelete)'` | Failed: Google reauthentication required. No resource changed. |
| `gcloud auth list --format='table(account,status)'` | Existing active account is `eric@20xx.xyz`; no second account available. |
| Direct `ssh ... ericgu@34.177.115.232 'hostname; id -un'` with strict checking | First attempt failed because the IP address had no known-host entry. |
| Python `ssh-keyscan -T 10` comparison against existing Google known hosts; SSH with verified `HostKeyAlias` | Exact key match; host metadata confirmed `hal-netplay-g4`. No new key was trusted. |
| Read-only SSH: `hostname`, `id`, instance metadata, `lsblk`, `systemctl show`, and `docker ps` | One 100 GB persistent disk; runner failed/inactive; runner container absent. Only the health container remained before shutdown. |
| Python Cloudflare settings/route GETs using the existing OAuth token in memory | Bindings and route matched. Printed names/types only; no secret values. |
| `npm run deploy -- --dry-run --outdir ../../runs/netplay/quota-incident/deploy-dry-run` | Passed; 73.34 KiB bundle, 17.30 KiB gzip. |
| `npm run deploy -- --message 'Reduce queue database reads (a425fc2946cd)' --strict` | Passed; active version recorded above. |
| `git show` plus Python SHA-256 checks; read-only SSH source/override/disk inspection | Installed source matched `3d2bdc40`; new file matched `a425fc29`; persistent disk confirmed. |
| Python over SSH: assert stopped state and old hashes, back up files, atomically replace runner and override, `systemctl daemon-reload`, verify new hashes | Passed. Saved staged-source records on host and locally; runner stayed stopped. |
| `curl -sS --max-time 20 -w '\nHTTP %{http_code}\n' https://20xx.xyz/v1/capacity` | Request completed; HTTP 500 with Cloudflare error 1101. Live health did not pass. |
| SSH `sudo shutdown -h now` | Returned zero. Shutdown requested through the guest OS. |
| Subsequent SSH `hostname` and shutdown-state checks | First probe succeeded during shutdown; next probe failed with connection reset; final probe timed out after 10 seconds. These are reachability checks, not Google API status confirmation. |
| `npx wrangler deployments list --help`; `npx wrangler deployments list` | Confirmed version `ed23da11-c3ae-4208-aa79-586ba38feb99` at 100%. |
| `systemctl --user show hal-xpilot-games.service -p ActiveState -p SubState`; `test -f runs/netplay/x-pilot-master120/stop-after-game` | Local scheduler remains failed/inactive; pause sentinel exists. |
| `git diff --check`; local Markdown link check; staged diff review | Passed. |
| `git add deploy/netplay/README.md deploy/netplay/queue-costs.md`; `git commit -m "Record queue deployment and G4 shutdown"`; final Git status | Deployment record committed on `netplay-edge-queue`; final status checked after commit. |

No code changed after the passing test gates above. Broad tests and emulator
integration were not repeated for this deployment and documentation update.
The runner startup, stream check, and new games were skipped to preserve the
requested pause. Google control-plane status verification was blocked by the
expired login.

## References

- [Cloudflare alarm semantics](https://developers.cloudflare.com/durable-objects/api/alarms/)
- [Cloudflare row and alarm pricing](https://developers.cloudflare.com/durable-objects/platform/pricing/)
- [SQL cursor row counters](https://developers.cloudflare.com/durable-objects/api/sqlite-storage-api/)
- [Wrangler Worker commands](https://developers.cloudflare.com/workers/wrangler/commands/workers/)
- [Google guest shutdown](https://docs.cloud.google.com/compute/docs/instances/stop-start-instance#stop_an_instance_from_the_guest_os)
