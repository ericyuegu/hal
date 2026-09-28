# Netplay Edge Queue — Plan 3: Runner Cutover

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task by task. Each task has a fresh Opus implementer and an Opus reviewer before the next task.

**Goal:** Run netplay slots against the Cloudflare queue, remove the Python API and SQLite queue, and make local development use the Worker.

**Architecture:** The supervisor starts one remote session and keeps its heartbeat alive through downloads and qualification. Spawned slots use `RemoteQueue` with the same session ID and fixed slot numbers. The supervisor drains on one signal, aborts on the next or at the timeout, and ends the session. Static ISO and emulator assets come from `hal.fixtures`; the verified asset cache still handles policy bundles and account JSON.

**Tech Stack:** Python 3.14, uv, httpx 0.28.1, websockets 17.1, boto3, pytest; TypeScript Worker, vitest, wrangler; shell, vinext.

**Spec:** `docs/superpowers/specs/2026-09-27-netplay-edge-queue-design.md`

**Status:** Self-reviewed for direct execution. The owner waived subagent review and asked for implementation in this session.

Plan 2's “Decisions carried into Plan 3” section is binding. The owner replaced its asset-pin decision with the fixture decision in the task request. Plan 5 owns the stream lease and runner display changes. Plan 3 sends `stream: false`.

## Global Constraints

- Edit only `/home/ericgu/src/hal-edge-queue` on `netplay-edge-queue`. Preserve unrelated edits.
- The Worker retries every runner request. Every new or changed runner route remains idempotent. Bump `STORE_SCHEMA_VERSION` only for a schema change and wipe old local DO state; there are no migrations.
- `hal.fixtures.ISO` is `fixtures/ssbm.ciso`, SHA-256 `b7de482eb955c8a96b6746dfa043b69ae7bf6c7c2a09ac382b9da126faa7055c`. Do not create or read `deploy/netplay/assets.json`.
- `NETPLAY_EMULATOR` is the unmodified upstream Slippi Online 3.6.4 AppImage at `https://github.com/project-slippi/Ishiiruka/releases/download/v3.6.4/Slippi_Online-x86_64.AppImage`, SHA-256 `e0f984e5bbecb98e3a746da1f173a475b06c3a1ba6b73e2e31bbe85a5f5a5e8a`, size `111679992`. GitHub's release metadata and the local tested file agree. Do not substitute the older local 3.5.1 build or the exi-ai build.
- Reject mismatched policy, account, replay, status, and fixture hashes or schemas. Never silently fall back to a local file or stale queue state.
- The first signal drains. The second signal or 15 minute drain deadline forfeits active work and ends the session. An unclean crash is covered by 30 seconds of session silence.
- `QueueUnavailableError` from a job heartbeat does not interrupt a live Dolphin game. A 409 from an expired lease is a lost lease. A claim 409 stops that slot and logs the detail. A 410 on any call ends the runner nonzero.
- A replay sidecar remains on disk if the Worker rejects it. A replay recorded after its original session ended uses the original `worker_id` and a `RemoteQueue` bound to its session ID.
- No deploy, DNS change, VM, registry push, R2 upload, Twitch broadcast, or git push. No AI attribution in commits.
- Every behavior change gets a focused regression test. Every refactor gets parity coverage. Complete every task's focused tests before review.

## Review Focus

1. A session is created but the first status report or qualification fails: the same session ID is ended, and no account remains leased. Test in Task 2.
2. The second SIGTERM arrives during a live game: the runner forfeits it and ends the session without waiting for the first drain deadline. Test in Task 4.
3. A lost response makes a `claim` retry return the same job; a slot never starts two reservations. Test in Task 3 and the Worker integration test.
4. A replay from an ended session is accepted under its original worker; a 409 preserves the sidecar. Test in Task 5.
5. A status reporter receives 410 during startup or play: claiming stops and the process exits nonzero. Test in Tasks 2 and 4.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `hal/fixtures.py` | Add verified `NETPLAY_EMULATOR`; keep `ISO` as the single ISO pin. |
| `hal/netplay_service/assets.py` | Keep `PinnedAsset`, `AssetCache`, `R2Source`, `LocalSource`, policy/account key functions, and `ensure_uploaded`; delete manifest and pinned ISO/emulator code. |
| `hal/netplay_service/admin.py` | Delete `assets pin`; keep policy and account publication. |
| `hal/netplay_service/runner.py` | Session lifecycle, qualification, remote slot processes, errors, drain, sidecars, artifacts, local metrics. |
| `hal/netplay_service/queue_client.py` | Remote queue, session reporter, settings socket; change only if a focused test shows a contract gap. |
| `tests/test_netplay_runner.py`, `tests/test_netplay_assets.py`, `tests/test_netplay_admin.py` | Runner and asset regression tests using a `RunnerQueue` fake; remove SQLite fixture assumptions. |
| `tests/test_netplay_queue_integration.py` | Real `wrangler dev` path through session, claim, transition, replay, and silence. |
| `scripts/qualify_netplay_059.py`, `tests/test_qualify_netplay_059.py`, `tests/fixtures/o59/idle_runner_faults.py` | Port qualification harnesses to `RemoteQueue` against `local_worker`. |
| `hal/netplay_service/api.py`, `hal/netplay_service/queue.py`, `scripts/record_netplay_transcripts.py`, old API/queue/transcript tests | Delete after the Worker and runner tests cover their behavior. |
| `deploy/netplay/run-host.sh`, `run-local.sh`, `compose.yaml`, `compose-ada.yaml`, `deploy-web.sh`, `README.md` | Runner-only host and local Worker launcher; no Python API or tunnel. |
| `web/netplay/lib/netplay-api.ts`, `web/netplay/vite.config.ts` | Same-origin `/v1` in production and local `/v1` proxy. |
| `pyproject.toml`, `.env.example`, spec Startup section and Plans table | Remove Python API entry point and stale environment settings; document fixture startup. |

### Task 1: Pin static fixtures and remove the asset manifest

**Files:** `hal/fixtures.py`, `hal/netplay_service/assets.py`, `hal/netplay_service/admin.py`, `tests/test_netplay_assets.py`, `tests/test_netplay_admin.py`.

**Interfaces:** `ensure(ISO) -> Path`, `ensure(NETPLAY_EMULATOR) -> Path`; `AssetCache.get(PinnedAsset) -> Path` remains for bundle and account files.

- [ ] **Step 1: Verify the emulator pin.** Run `sha256sum data/emulator/slippi-3.6.4/Slippi_Online-x86_64.AppImage` where the tested local file exists and compare it with GitHub's `v3.6.4` release metadata. Expected: SHA-256 `e0f984e5bbecb98e3a746da1f173a475b06c3a1ba6b73e2e31bbe85a5f5a5e8a`, size `111679992`.
- [ ] **Step 2: Write fixture tests.** Assert `NETPLAY_EMULATOR` has exactly one source, the expected hash and byte size, `dest == Path("data/emulator/slippi-3.6.4/Slippi_Online-x86_64.AppImage")`, and `ensure` rejects a corrupt cached file. Keep the existing ISO hash assertion. Run: `uv run pytest -q tests/test_netplay_assets.py tests/test_fixtures.py` (create `tests/test_fixtures.py`). Expected: the new pin test fails before implementation.
- [ ] **Step 3: Add `NETPLAY_EMULATOR: Final[Fixture] = Fixture(name="Slippi_Online-x86_64.AppImage", sha256="e0f984e5bbecb98e3a746da1f173a475b06c3a1ba6b73e2e31bbe85a5f5a5e8a", size_bytes=111_679_992, dest=Path("data/emulator/slippi-3.6.4/Slippi_Online-x86_64.AppImage"), url="https://github.com/project-slippi/Ishiiruka/releases/download/v3.6.4/Slippi_Online-x86_64.AppImage")`. Add it to `ALL` and `BY_NAME`. Extend `ensure` to mark this AppImage executable after verifying it. Do not change `DOLPHIN_EXIAI`.
- [ ] **Step 4: Delete `ASSET_MANIFEST_VERSION`, `AssetManifest`, `_pinned`, and `pinned_asset_key` from `assets.py`, and delete `pin_assets` and the `assets` parser branch from `admin.py`. Keep `PinnedAsset`, `AssetCache`, `R2Source`, `LocalSource`, `policy_bundle_key`, `account_key`, `sha256_file`, and `ensure_uploaded`. Replace tests for the removed surface with tests that still check cache corruption, size mismatch, and hash mismatch. Run: `uv run pytest -q tests/test_netplay_assets.py tests/test_netplay_admin.py`. Expected: pass.
- [ ] **Step 5: Commit.** Run: `git add hal/fixtures.py hal/netplay_service/assets.py hal/netplay_service/admin.py tests/test_netplay_assets.py tests/test_netplay_admin.py && git commit -m "Pin netplay emulator fixture"`.

### Task 2: Start and keep alive a remote session through qualification

**Files:** `hal/netplay_service/runner.py`, `tests/test_netplay_runner.py`, `tests/test_netplay_queue_integration.py`.

**Interfaces:** `RunnerClient.active_policy() -> PolicyConfig`, `start_session(session_id=..., host=..., bundle_sha256=..., git_sha=..., slots=..., wants_stream=False) -> StartedSession`, `SessionReporter(client, session_id, status)`, `RunnerClient.end_session(session_id) -> int`. The `status` callback returns a schema-valid `RunnerStatus` with `SlotState.STARTING` until slots become ready.

- [ ] **Step 1: Add a focused startup test with a fake `RunnerClient`.** Record call order as `ensure(ISO)`, `ensure(NETPLAY_EMULATOR)`, `active_policy`, `start_session`, `report_status`, bundle/account downloads, qualification, slot starts. Make qualification fail and assert `end_session` is still called with the original ID. Simulate a lost start response with `QueueUnavailableError`; retry only with the same ID. Assert `wants_stream is False`.
- [ ] **Step 2: Run: `uv run pytest -q tests/test_netplay_runner.py -k 'remote_start or session_start'`.** Expected: the new test fails before implementation.
- [ ] **Step 3: Replace the CLI's positional policy, database, user JSON and asset path inputs with `--slots N`, `--local-assets DIR`, `--drain-timeout` (default 900 seconds), runtime tuning flags, and `HAL_NETPLAY_API_URL`/runner token/Access/R2 environment inputs. Use `runner_endpoint(os.environ)` before network work. Validate `1 <= slots <= 8`, a positive finite drain timeout, and unique Slippi ports. Set the host from `socket.gethostname()` and Git SHA from the image/environment; refuse a missing SHA.
- [ ] **Step 4: Fetch `ISO` and `NETPLAY_EMULATOR` with `hal.fixtures.ensure` (or a fixture-backed local source for `--local-assets`, still hash checked). Fetch active policy, generate one `new_session_id()`, start the session, and immediately enter `SessionReporter` before fetching the bundle and account JSON. Use `AssetCache.get(PinnedAsset(policy.bundle_r2_key, policy.bundle_sha256))` and each `AccountGrant`'s key/hash. Validate the fetched bundle with `read_action_sequence_artifact` and compare its vocabulary SHA-256 to the Worker policy. Compare each fetched account's connect code to its grant. Then run the existing `check_realtime_budget` and start slots.
- [ ] **Step 5: Use `try/finally` around the session, including the first synchronous reporter call. If `start_session` yields a policy bundle mismatch (`QueueProtocolError`), call `end_session` with that same ID and exit nonzero. If reporter `state()` raises `SessionEndedError`, stop claiming, forfeit active work, and exit nonzero. Log each 409 start detail, which distinguishes policy, accounts, protocol, or ID collision. Run: `uv run pytest -q tests/test_netplay_runner.py -k 'remote_start or session_start'`. Expected: pass.
- [ ] **Step 6: Commit.** `git add hal/netplay_service/runner.py tests/test_netplay_runner.py tests/test_netplay_queue_integration.py && git commit -m "Start remote runner sessions"`.

### Task 3: Replace slot SQLite access and live-settings polling

**Files:** `hal/netplay_service/runner.py`, `tests/test_netplay_runner.py`, `tests/test_netplay_queue_integration.py`.

**Interfaces:** Child processes receive the serializable `QueueEndpoint` and the session ID. Each child constructs `RemoteQueue(endpoint, session_id)` after spawn and closes it before exit. Worker IDs are always `slot_worker_id(session_id, slot)`; no generation or restart suffix.

- [ ] **Step 1: Add a `RunnerQueue` fake for unit tests.** Implement only the protocol methods from `queue_contract.py` with a lock and explicit `Job` transitions used by runner tests. Drive claim, connecting, playing, finish, fail, forfeit, get, replay; assert the fake's calls and resulting state. Use real `RemoteQueue` plus `local_worker` in integration tests to verify HTTP shapes and retry idempotency. Remove `QueueStore` imports from runner tests.
- [ ] **Step 2: Run: `uv run pytest -q tests/test_netplay_runner.py -k 'claim or live_policy'`.** Expected: the new remote behavior tests fail.
- [ ] **Step 3: Replace each `QueueStore(...)` in `runner.py` with a `RemoteQueue` owned inside the process that uses it. Keep one fixed worker ID per session slot. Distinguish a claim 409 (`InvalidTransitionError`) from an outage (`QueueUnavailableError`): log the 409 detail and stop that slot; feed an outage through the slot's existing recovery path. Propagate `SessionEndedError` to the supervisor; do not bury it in the reservation's broad recovery handler. On a 409 after a playing outage, record lost lease and stop that reservation rather than retrying a game transition.
- [ ] **Step 4: Replace `_LivePolicySettings._poll` with `RemoteQueue.connect_live(job_id, worker_id)`. Parse exact `settings` and `released` messages. On socket loss, reconnect with bounded backoff and call `get_worker_job` once after reconnect; set `released` on `InvalidTransitionError` or terminal state. Keep the latest immutable `(desired_return, temperature)` tuple. Unit test a socket disconnect, new revision, and release. Run: `uv run pytest -q tests/test_netplay_runner.py -k 'claim or live_policy'`. Expected: pass.
- [ ] **Step 5: Extend `tests/test_netplay_queue_integration.py` with one claim and a lost-response replay of that claim against local Worker. Assert the same job ID and no second lease. Run: `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_netplay_queue_integration.py -m integration`. Expected: pass.
- [ ] **Step 6: Commit.** `git add hal/netplay_service/runner.py tests/test_netplay_runner.py tests/test_netplay_queue_integration.py && git commit -m "Use remote queue in slots"`.

### Task 4: Drain and abort sessions without interrupting live sets

**Files:** `hal/netplay_service/runner.py`, `tests/test_netplay_runner.py`.

**Interfaces:** `SessionReporter.state().draining` is the claim gate. The supervisor owns `RunnerClient.drain` and `end_session`. Child processes receive separate `stop_claiming` and `abort` events; a drain does not set `abort`.

- [ ] **Step 1: Write tests for first signal, second signal, timeout, remote drain, 410, and slot crash. Use fake clock and fake child processes.** First signal must call `drain` once and wait for active sets to finish. Second signal and timeout must forfeit live games, end the session and return nonzero. `state().draining` prevents new claims. A slot crash must not silently restart with a new worker ID or steal a held lease.
- [ ] **Step 2: Run: `uv run pytest -q tests/test_netplay_runner.py -k 'drain or signal or ended_session'`.** Expected: new tests fail.
- [ ] **Step 3: Replace `_ShutdownFlag.requested` with a signal count or two events. On the first signal, call `RunnerClient.drain`, set `stop_claiming`, wait until slots report idle or exit, and keep `SessionReporter` running. On second signal or monotonic deadline, set `abort`, use `forfeit_service_failure` for known live jobs, stop child processes, and call `end_session` once. In all exit paths, close the reporter, child processes, HTTP client, and partial pipes. Retain the 30 second silence fallback for a hard crash.
- [ ] **Step 4: For a crashed slot, fail its known active job through its original fixed worker ID before replacing the child. If it crashed between remote claim and local job-ID receipt, wait for the Worker's lease expiry before admitting the replacement; do not create a second worker identity. Assert that the other slots keep their active games. Run: `uv run pytest -q tests/test_netplay_runner.py -k 'drain or signal or ended_session or slot_crash'`. Expected: pass.
- [ ] **Step 5: Commit.** `git add hal/netplay_service/runner.py tests/test_netplay_runner.py && git commit -m "Drain remote sessions safely"`.

### Task 5: Preserve delayed replays and remote measurements

**Files:** `hal/netplay_service/runner.py`, `hal/netplay_service/replays.py`, `tests/test_netplay_runner.py`, `tests/test_netplay_replays.py`.

**Interfaces:** Sidecar schema 2 contains `worker_id = "<session>/slot-<n>"`. A leftover sidecar is recorded through `RemoteQueue(endpoint, original_session)`; only successful `record_replay` removes it.

- [ ] **Step 1: Add tests for an ended-session sidecar, a replay 409, and R2 upload failure. Assert that 409 and upload failure leave the replay and sidecar. Assert a successful record removes both. Add tests that measurement and engine audit files use deterministic R2 keys adjacent to replay keys and are hash checked.
- [ ] **Step 2: Run: `uv run pytest -q tests/test_netplay_runner.py tests/test_netplay_replays.py -k 'sidecar or audit or measurement'`.** Expected: new tests fail.
- [ ] **Step 3: In `_complete_pending_upload`, parse the sidecar worker ID into session and slot, construct `RemoteQueue` for that original session, call `record_replay`, then remove the sidecar only after a successful acknowledgement. Log 409 with its detail and the sidecar path. Upload measurement JSON and engine audit JSON to private R2 with SHA-256 metadata and a key beside the replay's `netplay/...` prefix. Use `ensure_uploaded` and retain failed local files for retry. Make upload errors visible in logs without stopping Dolphin.
- [ ] **Step 4: Start the runner's local Prometheus endpoint on a configurable local port. Test that it binds only loopback and exposes a session heartbeat and slot-health values. Run: `uv run pytest -q tests/test_netplay_runner.py tests/test_netplay_replays.py -k 'sidecar or audit or measurement or metrics'`. Expected: pass.
- [ ] **Step 5: Commit.** `git add hal/netplay_service/runner.py hal/netplay_service/replays.py tests/test_netplay_runner.py tests/test_netplay_replays.py && git commit -m "Preserve remote runner artifacts"`.

### Task 6: Port qualification harnesses and delete the Python service

**Files:** `scripts/qualify_netplay_059.py`, `tests/test_qualify_netplay_059.py`, `tests/fixtures/o59/idle_runner_faults.py`, `hal/netplay_service/api.py`, `hal/netplay_service/queue.py`, `scripts/record_netplay_transcripts.py`, `tests/test_netplay_api.py`, `tests/test_netplay_queue.py`, `tests/test_netplay_transcripts.py`, `tests/test_netplay_contract.py`, `pyproject.toml`.

**Interfaces:** Qualification uses `RemoteQueue` and `local_worker`; its public CLI behavior and measured outputs remain the same. Golden transcript JSON remains as evidence under `web/netplay-api/test/transcripts/`.

- [ ] **Step 1: Port the qualification tests to the local Worker harness. Create a local policy and account list, start a session, and drive the same claim/failure scenarios. Run: `uv run pytest -q tests/test_qualify_netplay_059.py`. Expected: parity tests pass against the Worker before deletion.
- [ ] **Step 2: Replace `QueueStore` in `scripts/qualify_netplay_059.py` and `tests/fixtures/o59/idle_runner_faults.py` with `RemoteQueue` and `RunnerClient` against `local_worker`. Keep each sidecar's original worker ID. Run: `uv run pytest -q tests/test_qualify_netplay_059.py`. Expected: pass.
- [ ] **Step 3: Delete `api.py`, `queue.py`, `scripts/record_netplay_transcripts.py` and their direct tests. Update `tests/test_netplay_contract.py` to check the remote protocol. Delete `hal-netplay-api` from `[project.scripts]` and remove dependencies used only by the API. Run: `rg -n 'netplay_service\.(api|queue)|QueueStore|hal-netplay-api|record_netplay_transcripts' hal scripts tests pyproject.toml`. Expected: no active source references.
- [ ] **Step 4: Run: `uv run pytest -q tests/test_qualify_netplay_059.py tests/test_netplay_contract.py` and `npm test --prefix web/netplay-api`. Expected: pass, with the Worker golden transcripts unchanged.
- [ ] **Step 5: Commit.** `git add -A hal/netplay_service scripts tests pyproject.toml uv.lock && git commit -m "Remove Python netplay service"`.

### Task 7: Replace deployment and page wiring

**Files:** `deploy/netplay/run-host.sh`, `deploy/netplay/run-local.sh`, `deploy/netplay/compose.yaml`, `deploy/netplay/compose-ada.yaml`, `deploy/netplay/deploy-frontend.sh`, `deploy/netplay/deploy-web.sh`, `deploy/netplay/README.md`, `web/netplay/lib/netplay-api.ts`, `web/netplay/vite.config.ts`, `.env.example`, `tests/test_netplay_deploy.py`, spec Startup section.

**Interfaces:** `run-local.sh [environment-file]` launches `wrangler dev` with local DO state and dev tokens, the page with `/v1` proxy, and one runner with `HAL_NETPLAY_API_URL=http://127.0.0.1:8787` and optional `--local-assets DIR`. `deploy-web.sh` builds and deploys the page only; running it is owner-gated.

- [ ] **Step 1: Write shell tests with stub `npm`, `uv`, and signals. Assert local startup order (Worker ready, page, runner), cleanup of all three child process groups, and no Python API, tunnel or `xvfb-run` process. Test page requests use relative `/v1` and local proxy reaches Worker. Run: `uv run pytest -q tests/test_netplay_deploy.py`. Expected: new tests fail.
- [ ] **Step 2: Make `run-host.sh` start only `hal-netplay-runner --slots N`, passing the three runner auth secrets and R2 credentials. Rewrite `run-local.sh` to call `local_worker`/`wrangler dev` for the API, start vinext with Vite proxy `/v1 -> http://127.0.0.1:8787`, then run the runner against that URL. Keep deterministic signal cleanup. Add `--local-assets DIR` with hash checks.
- [ ] **Step 3: Remove `api`, `tunnel`, and `state` Compose services. Keep a runner-only service with `NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility` (Plan 5 adds `video`). Delete `deploy-frontend.sh`; add `deploy-web.sh` with `npm ci`, `npm run build`, and the existing `npx wrangler deploy --config dist/server/wrangler.json`. Do not execute deploy. Use relative `/v1` in the page and remove `NEXT_PUBLIC_HAL_API_URL` from build and environment.
- [ ] **Step 4: Rewrite the spec's Startup section to say `ensure(ISO)` and `ensure(NETPLAY_EMULATOR)`, then session start, immediate status reporter, bundle/account downloads, qualification, slots. Update README and `.env.example`; remove obsolete keys listed in the spec's Removed section. Run: `uv run pytest -q tests/test_netplay_deploy.py`, `npm run build --prefix web/netplay`, and `rg -n 'NEXT_PUBLIC_HAL_API_URL|cloudflared|xvfb-run|hal-netplay-api|assets.json' deploy/netplay web/netplay hal/netplay_service pyproject.toml`. Expected: tests and build pass, and grep finds no active references.
- [ ] **Step 5: Commit.** `git add -A deploy/netplay web/netplay .env.example docs/superpowers/specs/2026-09-27-netplay-edge-queue-design.md tests/test_netplay_deploy.py && git commit -m "Switch netplay deployment to Worker"`.

### Task 8: Handoff checks and Plan 3 status

**Files:** `docs/superpowers/specs/2026-09-27-netplay-edge-queue-design.md`, `deploy/netplay/README.md`.

- [ ] **Step 1: Run `uv run ruff format --check .`. Expected: pass.**
- [ ] **Step 2: Run `uv run ruff check .`. Expected: pass.**
- [ ] **Step 3: Run `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts`. Expected: zero diagnostics.**
- [ ] **Step 4: Run `uv run pytest -q -m "not integration"`. Expected: pass.**
- [ ] **Step 5: Run `npm test` and `npm run typecheck` from `web/netplay-api`. Expected: pass.**
- [ ] **Step 6: Run `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_netplay_queue_integration.py -m integration`. Expected: pass, with required fixtures present.**
- [ ] **Step 7: Run `HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration`. Expected: pass, with required fixtures present.**
- [ ] **Step 8: Record each command, result, failure, and skip in `deploy/netplay/README.md`. Update only the spec Plans row for Plan 3 to `done`; keep Plans 4 and 5 unchanged. Run `git diff --check`, review the final diff and commit with `git commit -m "Complete netplay runner cutover"`. Do not push.**
