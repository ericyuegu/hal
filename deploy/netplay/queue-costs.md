# Queue read quota fix

The owner paused the evaluation after Cloudflare reported the daily
5,000,000 SQL rows-read limit. The public capacity endpoint returned HTTP 500.
The local scheduler and G4 runner were stopped. The G4 VM remains running
and billable. The 48 verified recordings and NSM summary remain local.

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

## Deployment

The fix is local and tested. Cloudflare and the stopped G4 container still
have the previous code. Keep the evaluation paused until the owner resumes it.

The existing deployment rules require owner approval for `wrangler deploy`
and externally visible changes. After approval:

1. Deploy `web/netplay-api` with `npm run deploy`. The schema guard remains 2.
2. Stage the committed `hal/netplay_service/runner.py` in the G4 source mount
   at `/var/lib/hal-netplay/hotfix/obs-v1/runner.py`. Verify its SHA-256 and
   record the new effective source SHA in the service configuration and run
   manifest before the next start.
3. Keep the runner stopped while the daily quota remains exhausted. Deployment
   does not reset the quota. The email gives October 1, 2026, 00:00 UTC
   (September 30, 5 PM Pacific) as the reset.
4. After quota recovery and approval to resume, start the runner, verify one
   healthy slot and the stream, then resume the supported matchup schedule.
   Keep all prior results and failed attempts.

No Worker deployment, G4 source copy, runner restart, paid plan upgrade,
registry push, database reset, or Git push was performed for this fix.

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

## References

- [Cloudflare alarm semantics](https://developers.cloudflare.com/durable-objects/api/alarms/)
- [Cloudflare row and alarm pricing](https://developers.cloudflare.com/durable-objects/platform/pricing/)
- [SQL cursor row counters](https://developers.cloudflare.com/durable-objects/api/sqlite-storage-api/)
