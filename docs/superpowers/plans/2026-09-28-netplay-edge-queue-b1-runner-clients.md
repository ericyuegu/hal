# Netplay Edge Queue — Plan B1: Runner and Admin Clients

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Python everything it needs to talk to the Plan A queue Worker: a retrying runner client, a live-settings socket, a session heartbeat, an admin client and CLI, pinned and hash-checked assets, and a local `wrangler dev` harness. Make every runner route safe to retry, and give the runner one typed queue interface that both the SQLite store and the remote client implement.

**Architecture:** The Worker makes session start, claim, and session end idempotent, and it checks that a replay comes from the worker that played the game. `hal/netplay_service/queue_contract.py` holds the shared queue errors and the `RunnerQueue` protocol. `QueueStore` and `RemoteQueue` both implement that protocol, and the runner is typed against it. `hal/netplay_service/queue_client.py` wraps one `httpx.Client` per process with bounded retries. It maps Worker status codes to exceptions. `RemoteQueue` implements `RunnerQueue`, `RunnerClient` covers the session routes, `SessionReporter` sends the session heartbeat, and `AdminClient` covers the admin routes. `hal/netplay_service/assets.py` pins the ISO and emulator in `deploy/netplay/assets.json` and keeps verified copies under `~/.cache/hal-netplay/<sha256>/`. `hal-netplay-admin` publishes policies, accounts, and asset pins. `hal/netplay_service/local_worker.py` runs the Worker under `wrangler dev` for the integration test.

**Tech Stack:** Python 3.14 (uv), httpx 0.28.1, websockets 17.1 (sync client), boto3 (R2), loguru, pytest; TypeScript Worker from Plan A (vitest, wrangler 4.124).

**Spec:** `docs/superpowers/specs/2026-09-27-netplay-edge-queue-design.md`

The program has four plans. Plan A built the Worker. Plan B moves the runner onto it, in two parts. Plan C adds host bring-up. Plan D adds Twitch streaming. B1 is this plan. It changes the Worker's runner routes, adds the Python clients and tools, and makes `QueueStore` and `runner.py` conform to the shared `RunnerQueue` protocol. The runner still constructs `QueueStore` and runs on SQLite. B2 is not written yet. It will move the runner onto `RemoteQueue`, port the qualification harnesses, delete the Python service, and update the page and deploy scripts. The section "Decisions carried into Plan B2" at the end lists what B2 must implement. Plan A must be complete before B1 starts, and B1 must be complete before B2 starts.

**Streaming is Plan D.** B1 implements no stream lease. The session API is forward compatible. `start_session` sends `stream` from its `wants_stream` argument. Every status response has the shape `{"draining": bool, "stream": null}`. The client parses it into a `SessionState` value and refuses a non-null `stream`, because no B1 runner asks for the lease. Plan D defines the grant, adds it to `SessionState`, and fills `stream`. The routes and the call sites stay as they are.

## Global Constraints

- **Idempotency is the retry precondition.** The client retries every request, so every runner and admin route must be safe to repeat after a lost response:
  - A session start carries a client-chosen `session_id`. A repeat with identical fields returns the same session and accounts.
  - A claim returns the slot's existing `leased` job before it takes another job.
  - A repeated session end returns the result of the first end.
  - Transitions are idempotent from Plan A. Heartbeat, status, drain, the admin `PUT`s, pause, and resume are naturally idempotent.
  - Any new route must meet this rule, or the client must be changed not to retry it.
- Retries: exponential backoff 0.25 s, 0.5 s, 1 s, 2 s, 4 s (six attempts) on connection errors and `5xx`; no retry on `4xx`.
- Status mapping: `409` → `InvalidTransitionError`; `410` → `SessionEndedError` (a subclass of `InvalidTransitionError`); other `4xx` → `QueueRejectedError`; retries exhausted → `QueueUnavailableError`; a malformed body → `QueueProtocolError`. `QueueError`, `InvalidTransitionError`, and `SessionEndedError` live in `queue_contract.py`, and `queue.py` raises the same classes.
- Every runner and admin request sends `Authorization: Bearer <token>`, and it sends the Cloudflare Access headers when they are configured. An `https://` URL requires both Access values; an `http://` URL is accepted only for `127.0.0.1` or `localhost` (local `wrangler dev`) and may omit them.
- Runners read `CF_ACCESS_CLIENT_ID` and `CF_ACCESS_CLIENT_SECRET`. The admin CLI reads `HAL_NETPLAY_ADMIN_ACCESS_CLIENT_ID` and `HAL_NETPLAY_ADMIN_ACCESS_CLIENT_SECRET`, so that one `.env` can hold both a local runner and the admin tool without either borrowing the other's token.
- Job routes send `X-HAL-Session` and `X-HAL-Slot`. Worker IDs are `"<session_id>/slot-<n>"`, the same string as the Worker's `workerId`. Session IDs are 16 URL-safe characters from `secrets.token_urlsafe(12)`. The Worker accepts `^[A-Za-z0-9_-]{16,64}$`.
- Environment: `HAL_NETPLAY_API_URL`, `HAL_NETPLAY_RUNNER_TOKEN`, `HAL_NETPLAY_ADMIN_TOKEN`, `CF_ACCESS_CLIENT_ID`, `CF_ACCESS_CLIENT_SECRET`, `HAL_NETPLAY_ADMIN_ACCESS_CLIENT_ID`, `HAL_NETPLAY_ADMIN_ACCESS_CLIENT_SECRET`, `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_BUCKET`.
- R2 keys: bundles `netplay/policies/<sha256>.halpolicy`; accounts `netplay/accounts/<sha256>.json`; ISO and emulator `netplay/assets/<sha256>/<file name>`.
- Cache: `~/.cache/hal-netplay/<sha256>/<file name>`; a cached file is used only after its SHA-256 is verified again.
- An uploaded object is trusted only when its size and its `sha256` metadata both match the local file.
- Policy config: schema version 1 with exactly the 13 fields that `web/netplay-api/src/policy.ts` accepts. Where the spec lists `max_games`, `no_show_seconds`, and `rematch_seconds`, the Worker serves them from constants and rejects them in the config; the Worker code wins.
- Follow `AGENTS.md`: typed Python, frozen slotted dataclasses for values, no `**kwargs` at boundaries, specific exceptions, tests in `tests/`, `uv run`.
- Edit `web/netplay-api/` only in Tasks 1–3.
- Every task leaves `ruff format --check`, `ruff check`, `ty`, the non-integration pytest suite, and the Worker's vitest suite passing.
- Commit messages are short and apt, with no attribution trailer.

## Review Focus

1. **A response is lost after the Worker applied the request, and the client retries.** This applies to session start, claim, session end, and a transition. Expected: the retry returns the first result. There is no second session, no second account lease, no second job on the slot, and no second game record. This is pinned in Task 2 (Worker) and Task 9 (integration, through the real Worker).
2. **A runner records a replay for a game that another worker played.** Expected: `409`. The worker that played the game can still record it after its session has ended. Pinned in Task 3.
3. **The runner's queue calls type-check against both implementations.** Expected: `ty` passes with `runner.py` typed on `RunnerQueue`, and both `QueueStore` and `RemoteQueue` are assignable to it. Pinned in Tasks 4 and 5.
4. **A box downloads bundles and qualifies for longer than 30 s after it starts a session.** Expected: `SessionReporter` keeps the session alive, and the Worker never ends it for silence. Pinned in Task 9.
5. **A misconfigured box: `HAL_NETPLAY_API_URL` with a trailing slash, an `https://` URL without Access credentials, or a missing token.** Expected: a `ValueError` that names the variable, raised before any request. Pinned in Task 5.
6. **The Worker returns a job body with an extra field or an unknown status.** Expected: `QueueProtocolError`, never `KeyError` or a partly built `Job`. Pinned in Task 5.
7. **A cached ISO is truncated on disk, or an R2 object carries matching hash metadata but the wrong size.** Expected: the cache fetches the ISO again, and `ensure_uploaded` refuses the object. Pinned in Task 7.

---

## File Structure

```
web/netplay-api/src/queue.ts, http.ts           + GET /v1/runner/policy (Task 1); idempotent start/claim/end (Task 2); replay ownership (Task 3)
web/netplay-api/src/sessions.ts, store.ts       idempotent start/end, claim returns the slot's lease, games.worker (Tasks 2–3)
web/netplay-api/test/runner-policy.test.ts      route test (Task 1)
web/netplay-api/test/retries.test.ts            repeated start, claim, end; replay ownership (Tasks 2–3)
pyproject.toml, uv.lock                         httpx, websockets, prometheus-client become runtime deps
hal/netplay_service/queue_contract.py           QueueError, InvalidTransitionError, SessionEndedError, RunnerQueue
hal/netplay_service/queue.py                    (modified) raises contract errors; implements RunnerQueue
hal/netplay_service/runner.py                   (modified) typed on RunnerQueue; replay sidecar v2 carries worker_id
hal/netplay_service/domain.py                   PolicyConfig; Job loses SQLite-only fields; account_connect_code
hal/netplay_service/queue_client.py             endpoints, client errors, parse_job, RemoteQueue, RunnerClient, SessionReporter, AdminClient
hal/netplay_service/assets.py                   PinnedAsset, AssetManifest, AssetCache, R2Source, LocalSource, ensure_uploaded
hal/netplay_service/admin.py                    publish-policy, accounts upload, assets pin, status, events, pause, resume
hal/netplay_service/local_worker.py             wrangler dev harness with dev tokens
deploy/netplay/assets.json                      ISO and emulator pins (written by `hal-netplay-admin assets pin`)
deploy/netplay/README.md                        admin Access service token
tests/test_netplay_contract.py                  both stores satisfy RunnerQueue; QueueStore conformance changes
tests/test_netplay_domain.py                    PolicyConfig and account file tests
tests/test_netplay_queue_client.py              retry policy, error mapping, request shapes, reporter
tests/test_netplay_assets.py                    manifest, cache, uploads
tests/test_netplay_admin.py                     admin commands
tests/test_netplay_queue_integration.py         RemoteQueue against wrangler dev (-m integration)
```

---

### Task 0: Green baseline

The branch does not pass the handoff checks before B1 starts:

- `ruff check` reports I001 (unsorted imports) in `experiments/059_muon_action_sequence.py` and `scripts/eval_kv_cache.py`. Both came from `main`.
- `tests/test_netplay_deploy.py::test_local_launcher_waits_for_host_cleanup` times out in any checkout without `web/netplay/node_modules`. In that case `run-local.sh` runs `npm ci` in the foreground. The test's `npm` stub loops forever, and bash defers the SIGINT trap until the foreground child exits. The test passes on the main checkout only because that checkout has `node_modules`.

**Files:**
- Modify: `experiments/059_muon_action_sequence.py`, `scripts/eval_kv_cache.py` (import order only)
- Modify: `tests/test_netplay_deploy.py` (the `npm` stub in `test_local_launcher_waits_for_host_cleanup`)

- [ ] **Step 1: Reproduce both failures**

Run: `uv run ruff check --output-format concise . ; uv run pytest -q tests/test_netplay_deploy.py::test_local_launcher_waits_for_host_cleanup`
Expected: two I001 findings, and `subprocess.TimeoutExpired ... timed out after 7 seconds`.

- [ ] **Step 2: Sort the imports**

Run: `uv run ruff check --fix --select I001 experiments/059_muon_action_sequence.py scripts/eval_kv_cache.py`
Expected: `git diff` shows only reordered import lines.

- [ ] **Step 3: Make the launcher test independent of `node_modules`**

In `test_local_launcher_waits_for_host_cleanup`, replace the `npm` stub with one that finishes `npm ci` at once and blocks only for `npm run dev`:

```python
    _write_executable(
        command_dir / "npm",
        'if [[ ${1:-} == ci ]]; then exit 0; fi\ntrap \'exit 0\' TERM\nwhile true; do sleep 1; done\n',
    )
```

- [ ] **Step 4: Verify**

Run: `uv run ruff check . && uv run pytest -q tests/test_netplay_deploy.py`
Expected: ruff passes, and every deploy test passes in this worktree, which has no `web/netplay/node_modules`.

- [ ] **Step 5: Commit**

```bash
git add experiments/059_muon_action_sequence.py scripts/eval_kv_cache.py tests/test_netplay_deploy.py
git commit -m "Sort imports and make the launcher test hermetic"
```

---

### Task 1: Serve the active policy to runners

A runner must name its bundle's SHA-256 when it starts a session, but it learns which bundle is active only from the queue. Plan A has no runner route that returns the active policy (`/v1/options` omits the hash and bundle key). This task adds `GET /v1/runner/policy`.

**Files:**
- Modify: `web/netplay-api/src/queue.ts` (add `activePolicy` beside `startSession`)
- Modify: `web/netplay-api/src/http.ts` (route it inside the runner block)
- Test: `web/netplay-api/test/runner-policy.test.ts`

**Interfaces:**
- Consumes: Plan A `Queue.run`, `Queue.requirePolicy`, test helpers `call`, `publish`, `resetQueue`, `POLICY`.
- Produces: `GET /v1/runner/policy` → `200` with the published `PolicyConfig`, `503 {"detail": "no policy has been published"}` when none, `401` without a runner token.

- [ ] **Step 1: Write the failing test**

`web/netplay-api/test/runner-policy.test.ts`:

```ts
import { beforeEach, describe, expect, it } from "vitest";
import { POLICY, call, publish, resetQueue } from "./helpers";

beforeEach(async () => {
  await resetQueue();
});

describe("runner policy", () => {
  it("serves the active policy config to runners only", async () => {
    expect(await call("GET", "/v1/runner/policy", { runner: true })).toMatchObject({
      status: 503,
      body: { detail: "no policy has been published" },
    });
    await publish();
    const result = await call("GET", "/v1/runner/policy", { runner: true });
    expect(result.status).toBe(200);
    expect(result.body).toEqual(POLICY);
    expect((await call("GET", "/v1/runner/policy")).status).toBe(401);
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd web/netplay-api && npx vitest run test/runner-policy.test.ts`
Expected: FAIL — the first call returns `404` (`not found`), not `503`.

- [ ] **Step 3: Add the Durable Object method**

In `web/netplay-api/src/queue.ts`, directly above `async startSession(raw: unknown)`, add:

```ts
  async activePolicy(): Promise<ApiResult> {
    return this.run(() => this.requirePolicy());
  }
```

- [ ] **Step 4: Route it**

In `web/netplay-api/src/http.ts`, inside `if (path.startsWith("/v1/runner/")) {`, directly after the `POST /v1/runner/sessions` branch, add:

```ts
      if (method === "GET" && path === "/v1/runner/policy") return respond(await queue.activePolicy());
```

- [ ] **Step 5: Run the Worker suite**

Run: `cd web/netplay-api && npx vitest run && npm run typecheck`
Expected: all tests pass, including `runner policy`; typecheck prints nothing.

- [ ] **Step 6: Commit**

```bash
git add web/netplay-api/src/queue.ts web/netplay-api/src/http.ts web/netplay-api/test/runner-policy.test.ts
git commit -m "Serve the active policy to runners"
```

---

### Task 2: Make session start, claim, and session end safe to retry

Plan A made job transitions idempotent, but three runner routes are not:

- `startSession` generates a new ID and leases accounts on every call. If a response is lost, the retry leaves an orphan session that holds accounts until the 30 s silence alarm.
- `claimNext` always leases the next queued job. If a response is lost, the retry gives the same slot a second job, and the first job then expires as `lease_expired`.
- `endSession` returns `410` when repeated. If a response is lost, the retry raises `SessionEndedError` although the end succeeded.

The client retries every route (Global Constraints), so the Worker must make these three safe to repeat. The same task changes the status response to the forward-compatible shape `{draining, stream}`.

**Files:**
- Modify: `web/netplay-api/src/sessions.ts` (`start` accepts a repeat; `end` records its failure count; `report` returns `stream`; schema adds `failed_jobs`)
- Modify: `web/netplay-api/src/store.ts` (`claimNext` returns the worker's existing `leased` job)
- Modify: `web/netplay-api/src/queue.ts` (`startSession` reads `session_id`; `endSession` returns the recorded result for an ended session)
- Modify: `web/netplay-api/test/helpers.ts`, `test/routes.test.ts`, `test/transcripts.test.ts` (start bodies carry `session_id`)
- Test: `web/netplay-api/test/retries.test.ts`

**Interfaces:**
- `POST /v1/runner/sessions` body: `{session_id, host, bundle_sha256, git_sha, slots, stream}`; all required. `session_id` must match `^[A-Za-z0-9_-]{16,64}$` (`422` otherwise).
  - A repeat with the same `session_id` and identical fields returns `201` with the same body: the same accounts and the current policy.
  - A repeat with different fields returns `409 {"detail": "session <id> exists with different settings"}`.
  - A repeat after the session has ended returns `410 {"detail": "session has ended"}`.
- `POST /v1/runner/sessions/{sid}/claim`:
  - If the slot's worker holds a job in `leased`, the claim returns that job with `200` and does not increment `attempt`.
  - If the worker holds a job in any other in-service status, the claim returns `409 {"detail": "slot already holds job <id>"}`.
  - Otherwise the claim behaves as before.
- `DELETE /v1/runner/sessions/{sid}` on an ended session: `200 {"failed": <count from the first end>}`. An unknown session still gets `404`. A session that the silence alarm ended reports the alarm's count.
- `POST /v1/runner/sessions/{sid}/status` → `200 {"draining": bool, "stream": null}`. Plan D fills `stream`.

- [ ] **Step 1: Write the failing tests**

`web/netplay-api/test/retries.test.ts`:

```ts
import { beforeEach, describe, expect, it } from "vitest";
import { CREATE, POLICY, call, publish, report, resetQueue, seedAccounts } from "./helpers";

const SESSION = "retry-session-0001";
const START = { session_id: SESSION, host: "box", bundle_sha256: POLICY.bundle_sha256, git_sha: "g", slots: 2, stream: false };

beforeEach(async () => {
  await resetQueue();
  await publish();
  await seedAccounts(3);
});

async function start(body: Record<string, unknown> = START) {
  return call("POST", "/v1/runner/sessions", { runner: true, body });
}

describe("retried runner calls", () => {
  it("returns the same session and accounts when a start is repeated", async () => {
    const first = await start();
    expect(first.status).toBe(201);
    const second = await start();
    expect(second).toMatchObject({ status: 201, body: first.body });
    const status = await call("GET", "/v1/admin/status", { admin: true });
    expect(status.body.sessions).toHaveLength(1);
    expect(status.body.accounts.filter((row: { session_id: string | null }) => row.session_id !== null)).toHaveLength(2);
  });

  it("refuses a reused session id with other settings, after an end, or in a bad format", async () => {
    await start();
    expect(await start({ ...START, slots: 1 })).toMatchObject({
      status: 409,
      body: { detail: `session ${SESSION} exists with different settings` },
    });
    await call("DELETE", `/v1/runner/sessions/${SESSION}`, { runner: true });
    expect(await start()).toMatchObject({ status: 410, body: { detail: "session has ended" } });
    expect((await start({ ...START, session_id: "short" })).status).toBe(422);
  });

  it("returns the slot's leased job when a claim is repeated", async () => {
    await start();
    await report(SESSION, 2);
    await call("POST", "/v1/jobs", { body: CREATE });
    await call("POST", "/v1/jobs", { body: { ...CREATE, player_code: "OTHER#1" } });
    const first = await call("POST", `/v1/runner/sessions/${SESSION}/claim`, { runner: true, body: { slot: 0 } });
    const second = await call("POST", `/v1/runner/sessions/${SESSION}/claim`, { runner: true, body: { slot: 0 } });
    expect(first.status).toBe(200);
    expect(second.body).toEqual(first.body);
    expect(second.body.attempt).toBe(1);
    const other = await call("POST", `/v1/runner/sessions/${SESSION}/claim`, { runner: true, body: { slot: 1 } });
    expect(other.body.player_code).toBe("OTHER#1");
  });

  it("refuses a claim from a slot whose job is past leased", async () => {
    await start();
    await report(SESSION, 2);
    await call("POST", "/v1/jobs", { body: CREATE });
    const job = (await call("POST", `/v1/runner/sessions/${SESSION}/claim`, { runner: true, body: { slot: 0 } })).body;
    const runner = { session: SESSION, slot: 0 };
    await call("POST", `/v1/runner/jobs/${job.id}/connecting`, { runner, body: { connect_code: "BOT0#1" } });
    expect(await call("POST", `/v1/runner/sessions/${SESSION}/claim`, { runner: true, body: { slot: 0 } })).toMatchObject({
      status: 409,
      body: { detail: `slot already holds job ${job.id}` },
    });
  });

  it("returns the first result when an end is repeated", async () => {
    await start();
    await report(SESSION, 2);
    await call("POST", "/v1/jobs", { body: CREATE });
    await call("POST", `/v1/runner/sessions/${SESSION}/claim`, { runner: true, body: { slot: 0 } });
    const first = await call("DELETE", `/v1/runner/sessions/${SESSION}`, { runner: true });
    expect(first).toMatchObject({ status: 200, body: { failed: 1 } });
    expect(await call("DELETE", `/v1/runner/sessions/${SESSION}`, { runner: true })).toMatchObject({
      status: 200,
      body: { failed: 1 },
    });
    expect((await call("DELETE", "/v1/runner/sessions/unknown-session-01", { runner: true })).status).toBe(404);
  });

  it("reports drain and an empty stream grant in status responses", async () => {
    await start();
    expect(await report(SESSION, 2)).toMatchObject({ status: 200, body: { draining: false, stream: null } });
  });
});
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd web/netplay-api && npx vitest run test/retries.test.ts`
Expected: FAIL. The start body is refused with `422 unexpected field session_id`.

- [ ] **Step 3: Accept a repeated start and record end results in `sessions.ts`**

Add `failed_jobs INTEGER` after `end_reason TEXT` in the `sessions` table of `SESSION_SCHEMA`.

At the top of `start`, before the policy check, add the repeat path:

```ts
    const existing = this.rows("SELECT * FROM sessions WHERE id = ?", id)[0];
    if (existing !== undefined) {
      if (existing.ended_at !== null) throw new HttpError(410, "session has ended");
      const same =
        existing.host === input.host &&
        existing.bundle_sha256 === input.bundle_sha256 &&
        existing.git_sha === input.git_sha &&
        existing.slots === input.slots &&
        existing.wants_stream === (input.stream ? 1 : 0);
      if (!same) throw new HttpError(409, `session ${id} exists with different settings`);
      const accounts = this.rows(
        "SELECT slot, connect_code, r2_key, sha256 FROM accounts WHERE session_id = ? ORDER BY slot",
        id,
      ).map((row) => ({
        slot: row.slot as number,
        connect_code: row.connect_code as string,
        r2_key: row.r2_key as string,
        sha256: row.sha256 as string,
      }));
      return { session_id: id, accounts };
    }
```

Change `report` to return the forward-compatible shape. Plan D replaces `null` with the grant:

```ts
  report(id: string, raw: unknown): { draining: boolean; stream: null } {
    ...
    return { draining: row.draining === 1, stream: null };
  }
```

In `end`, record the count in the same `UPDATE` that sets `ended_at`:

```ts
    this.sql.exec(
      "UPDATE sessions SET ended_at = ?, end_reason = ?, failed_jobs = ? WHERE id = ?",
      this.now(),
      reason,
      failed.length,
      id,
    );
```

Add a method that answers a repeated end:

```ts
  // A runner repeats DELETE when it loses the first response; it gets the first result.
  endedResult(id: string): number | null {
    const row = this.rows("SELECT ended_at, failed_jobs FROM sessions WHERE id = ?", id)[0];
    if (row === undefined) throw new HttpError(404, "session not found");
    return row.ended_at === null ? null : (row.failed_jobs as number);
  }
```

- [ ] **Step 4: Return the slot's lease from `claimNext` in `store.ts`**

At the top of `claimNext(worker)`:

```ts
    // A repeated claim after a lost response must not give the slot a second job.
    const held = this.first(`SELECT id, status FROM jobs WHERE lease_owner = ? AND status IN (${IN_SERVICE})`, worker);
    if (held !== null) {
      if (held.status === "leased") return this.response(this.reload(held.id as string));
      throw new HttpError(409, `slot already holds job ${held.id}`);
    }
```

- [ ] **Step 5: Read `session_id` and repeat ends in `queue.ts`**

In `startSession`, drop `const id = randomToken(12);` and read the ID from the body:

```ts
  async startSession(raw: unknown): Promise<ApiResult> {
    return this.run(() => {
      const names = ["session_id", "host", "bundle_sha256", "git_sha", "slots", "stream"];
      const value = fields(raw, names, names);
      const id = str(value.session_id, "session_id");
      if (!SESSION_ID.test(id)) throw new HttpError(422, "session_id must be 16 to 64 URL-safe characters");
      ...
```

Here `const SESSION_ID = /^[A-Za-z0-9_-]{16,64}$/;` is a module constant beside `QUEUE_SCHEMA`. The rest of the method is unchanged. `this.sessions.start` handles the repeat. Log `session_started` and `account_leased` only when the session is new. Check `this.sessions.summary()` or compare before and after. The simplest form is to read `existing` first:

```ts
      return this.tx(() => {
        const repeated = this.sessions.exists(id);
        const started = this.sessions.start(id, input, this.policy());
        if (!repeated) {
          this.events.log("session_started", { session: id, host: input.host, slots: input.slots, git_sha: input.git_sha });
          for (const grant of started.accounts) {
            this.events.log("account_leased", { session: id, slot: grant.slot, connect_code: grant.connect_code });
          }
        }
        return { ...started, policy: this.policy() };
      });
```

Add `exists(id: string): boolean` to `SessionStore`, returning whether a row with that ID exists. Remove `randomToken` from the `queue.ts` import list if nothing else uses it.

In `endSession`, answer a repeat before ending:

```ts
  async endSession(sessionId: string): Promise<ApiResult> {
    return this.run(() => {
      const previous = this.sessions.endedResult(sessionId);
      if (previous !== null) return { failed: previous };
      ...unchanged...
```

- [ ] **Step 6: Give every existing start body a session ID**

- In `test/helpers.ts` `startSession`, add `session_id: crypto.randomUUID()` to the body. A UUID has 36 characters from `[0-9a-f-]`.
- In `test/routes.test.ts`, lines 89–92 and 185–187, add `session_id: crypto.randomUUID()` to both bodies.
- In `test/transcripts.test.ts`, lines 124–134, add `session_id: crypto.randomUUID()` to both bodies.
- Also give a session ID to any other start body that `grep -n '"/v1/runner/sessions"' test/*.ts` finds.
- `test/sessions.test.ts` calls `SessionStore.start` directly with IDs like `"s1"`. It needs no change, because the ID format is checked only at the route.

- [ ] **Step 7: Run the Worker suite**

Run: `cd web/netplay-api && npx vitest run && npm run typecheck`
Expected: all tests pass, including `retried runner calls` and the unchanged golden transcripts. Typecheck prints nothing.

- [ ] **Step 8: Commit**

```bash
git add web/netplay-api/src web/netplay-api/test
git commit -m "Make session start, claim, and end safe to retry"
```

---

### Task 3: Accept a replay only from the worker that played the game

`recordReplay` checks neither the session nor the slot. Any authenticated runner can attach replay metadata to another job's game if it knows the job ID. The check cannot use `lease_owner`, because a completed job's lease is already released. It also cannot use `last_worker`, because a retried job's later worker would then own an earlier game. Instead, each game row records the worker that finished it. A replay is accepted only from that worker. The replay is accepted even after that worker's session has ended, so an upload that was deferred across a runner restart can still be recorded.

**Files:**
- Modify: `web/netplay-api/src/store.ts` (`games.worker`; `finishGame` stores it; `recordReplay` takes and checks `worker`)
- Modify: `web/netplay-api/src/sessions.ts` (`recordingWorker`)
- Modify: `web/netplay-api/src/queue.ts` (the `replay` action uses `recordingWorker`)
- Test: `web/netplay-api/test/retries.test.ts` (append), and any `store.test.ts` call to `recordReplay`

**Interfaces:**
- `games` gains `worker TEXT NOT NULL` (the worker that finished the game).
- `JobStore.recordReplay(id, worker, gameNumber, key, sha256, size, etag)`: `409 {"detail": "worker did not play this game"}` when `games.worker !== worker`. The other behavior is unchanged: it is idempotent for identical metadata, and it returns `409` for different metadata or an absent game.
- `SessionStore.recordingWorker(id, slot): string`: it returns `404` for an unknown session and `422` for a slot out of range. Unlike `jobWorker`, it accepts an ended session.

- [ ] **Step 1: Write the failing tests**

Append to `web/netplay-api/test/retries.test.ts`:

```ts
describe("replay ownership", () => {
  async function playOneGame(): Promise<string> {
    await start();
    await report(SESSION, 2);
    await call("POST", "/v1/jobs", { body: { ...CREATE, player_code: "OWNER#1" } });
    const job = (await call("POST", `/v1/runner/sessions/${SESSION}/claim`, { runner: true, body: { slot: 0 } })).body;
    const runner = { session: SESSION, slot: 0 };
    await call("POST", `/v1/runner/jobs/${job.id}/connecting`, { runner, body: { connect_code: "BOT0#1" } });
    await call("POST", `/v1/runner/jobs/${job.id}/playing`, { runner });
    await call("POST", `/v1/runner/jobs/${job.id}/finish-game`, {
      runner,
      body: { game_number: 1, actual_stage: "BATTLEFIELD", result: "win" },
    });
    return job.id as string;
  }

  const REPLAY = { game_number: 1, key: "replays/a.slp", sha256: "a".repeat(64), size: 10, etag: "e" };

  it("refuses a replay from another slot or another session", async () => {
    const id = await playOneGame();
    const other = { ...START, session_id: "other-session-0001", slots: 1 };
    expect((await call("POST", "/v1/runner/sessions", { runner: true, body: other })).status).toBe(201);
    for (const runner of [
      { session: SESSION, slot: 1 },
      { session: "other-session-0001", slot: 0 },
    ]) {
      expect(await call("POST", `/v1/runner/jobs/${id}/replay`, { runner, body: REPLAY })).toMatchObject({
        status: 409,
        body: { detail: "worker did not play this game" },
      });
    }
  });

  it("accepts the playing worker's replay after its session ends", async () => {
    const id = await playOneGame();
    await call("DELETE", `/v1/runner/sessions/${SESSION}`, { runner: true });
    const runner = { session: SESSION, slot: 0 };
    expect((await call("POST", `/v1/runner/jobs/${id}/replay`, { runner, body: REPLAY })).status).toBe(200);
    expect((await call("POST", `/v1/runner/jobs/${id}/replay`, { runner, body: REPLAY })).status).toBe(200);
    expect(
      (await call("POST", `/v1/runner/jobs/${id}/replay`, { runner: { session: "never-existed-0001", slot: 0 }, body: REPLAY }))
        .status,
    ).toBe(404);
  });
});
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd web/netplay-api && npx vitest run test/retries.test.ts`
Expected: FAIL. The cross-slot replay returns `200`, and the replay after the end returns `410`.

- [ ] **Step 3: Record and check the game's worker**

In `store.ts`, add `worker TEXT NOT NULL,` after `result TEXT NOT NULL,` in the `games` table. In `finishGame`, insert it:

```ts
    this.exec(
      "INSERT INTO games(job_id, game_number, actual_stage, result, worker, created_at) VALUES (?, ?, ?, ?, ?, ?)",
      id,
      gameNumber,
      stage,
      result,
      worker,
      now,
    );
```

Change `recordReplay` to take `worker` second and check it:

```ts
  recordReplay(id: string, worker: string, gameNumber: number, key: string, sha256: string, size: number, etag: string): JobResponse {
    const game = this.first(
      "SELECT worker, replay_key, replay_sha256, replay_size, replay_etag FROM games WHERE job_id = ? AND game_number = ?",
      id,
      gameNumber,
    );
    if (game === null) throw new HttpError(409, "game is absent");
    if (game.worker !== worker) throw new HttpError(409, "worker did not play this game");
    ...unchanged...
```

Update each `recordReplay` call in `test/store.test.ts` to pass the worker that finished the game.

- [ ] **Step 4: Resolve the recording worker without requiring a live session**

In `sessions.ts`:

```ts
  // A deferred replay upload can outlive its session, so an ended session may still record.
  recordingWorker(id: string, slot: number): string {
    const row = this.rows("SELECT * FROM sessions WHERE id = ?", id)[0];
    if (row === undefined) throw new HttpError(404, "session not found");
    return this.slotWorker(row, slot);
  }
```

In `queue.ts` `runnerJob`, resolve the worker per action:

```ts
        const worker =
          action === "replay" ? this.sessions.recordingWorker(sessionId, slot) : this.sessions.jobWorker(sessionId, slot);
```

Pass `worker` as the second argument in the `replay` case.

- [ ] **Step 5: Run the Worker suite**

Run: `cd web/netplay-api && npx vitest run && npm run typecheck`
Expected: all pass. The transcripts are unchanged, because their replays come from the playing worker.

- [ ] **Step 6: Commit**

```bash
git add web/netplay-api/src web/netplay-api/test
git commit -m "Accept replays only from the worker that played the game"
```

---

### Task 4: Shared queue contract

The runner catches `hal.netplay_service.queue.InvalidTransitionError`. A remote client that defines its own class with the same name would pass through those handlers. The two stores also differ in their signatures: `finish_game` has no `game_number` on `QueueStore`, and `record_replay` has no worker. This task puts the errors that both stores raise in a neutral module. It adds the `RunnerQueue` protocol, which covers exactly the calls the runner makes, and makes `QueueStore` and `runner.py` conform to it. A protocol is justified because there are two real implementations and `QueueStore` stays useful as the runner's test seam.

**Files:**
- Create: `hal/netplay_service/queue_contract.py`
- Modify: `hal/netplay_service/queue.py` (import the contract errors; `finish_game(..., game_number)`; `record_replay(job_id, worker_id, game_number, ...)`; lease default 20 s)
- Modify: `hal/netplay_service/runner.py` (type every store parameter as `RunnerQueue`; drop the explicit lease arguments; pass `game_number` and `worker_id`; replay sidecar v2 carries `worker_id`)
- Modify: `hal/netplay_service/api.py`, `scripts/record_netplay_transcripts.py`, `tests/test_netplay_queue.py` (import errors from the contract)
- Test: `tests/test_netplay_contract.py`, `tests/test_netplay_runner.py` (sidecar)

**Interfaces:**
- `queue_contract.py`: `class QueueError(RuntimeError)`, `class InvalidTransitionError(QueueError)`, `class SessionEndedError(InvalidTransitionError)`, and `class RunnerQueue(Protocol)` with:
  - `claim_next(worker_id: str) -> Job | None`
  - `heartbeat(job_id: str, worker_id: str) -> None`
  - `mark_connecting(job_id: str, worker_id: str, connect_code: str) -> None`
  - `mark_playing(job_id: str, worker_id: str) -> None`
  - `mark_no_show(job_id: str, worker_id: str) -> None`
  - `mark_no_contest(job_id: str, worker_id: str) -> None`
  - `finish_game(job_id: str, worker_id: str, *, game_number: int, actual_stage: str, result: str) -> JobStatus`
  - `fail(job_id: str, worker_id: str, error_code: str, *, retryable: bool) -> JobStatus`
  - `forfeit_service_failure(job_id: str, worker_id: str) -> None`
  - `record_replay(job_id: str, worker_id: str, game_number: int, *, key: str, sha256: str, size: int, etag: str) -> None`
  - `get_worker_job(job_id: str, worker_id: str) -> Job`
- `queue.py` keeps `AuthenticationError` and `ActiveJobError` (player-route errors, subclasses of the contract's `QueueError`). It stops defining `QueueError` and `InvalidTransitionError`.
- `QueueStore.finish_game` raises `InvalidTransitionError("game_number N does not follow game M")` when `game_number != game_count + 1`. This is the Worker's rule and message.
- `QueueStore.record_replay` accepts `worker_id` and does not check it. The SQLite store serves one runner process whose slots are the only workers, so a replay from another worker's session cannot occur. A comment states this invariant. The store is deleted in B2.
- `QueueStore.claim_next` and `heartbeat` default to `lease_seconds=20.0`. That is the runner's current explicit value and the Worker's `LEASE_SECONDS`. The runner stops passing it. `mark_connecting` keeps its `IDLE_TIMEOUT_SECONDS` default, and the runner stops passing that too.
- Replay sidecar schema v2 adds `worker_id`. The reader rejects v1 with `pending replay metadata has the wrong schema`, as it rejects any other schema. Per the spec there is no backward compatibility, and a v1 sidecar can exist only on a box running the pre-B1 runner.

- [ ] **Step 1: Write the failing tests**

`tests/test_netplay_contract.py`:

```python
from pathlib import Path

import pytest

from hal.netplay_service.domain import MatchChoices
from hal.netplay_service.queue import QueueStore
from hal.netplay_service.queue_contract import InvalidTransitionError
from hal.netplay_service.queue_contract import RunnerQueue


def _playing(tmp_path: Path) -> tuple[QueueStore, str]:
    store = QueueStore(tmp_path / "queue.sqlite3")
    created = store.create_job("CRYO#610", MatchChoices("FOX", "IBDW#0", 2))
    store.claim_next("slot-0")
    store.mark_connecting(created.job.id, "slot-0", "HAL#1")
    store.mark_playing(created.job.id, "slot-0")
    return store, created.job.id


def test_queue_store_is_a_runner_queue(tmp_path: Path) -> None:
    store: RunnerQueue = QueueStore(tmp_path / "queue.sqlite3")
    assert store.claim_next("slot-0") is None


def test_finish_game_rejects_a_game_number_out_of_order(tmp_path: Path) -> None:
    store, job_id = _playing(tmp_path)
    with pytest.raises(InvalidTransitionError, match="game_number 2 does not follow game 0"):
        store.finish_game(job_id, "slot-0", game_number=2, actual_stage="BATTLEFIELD", result="win")
    store.finish_game(job_id, "slot-0", game_number=1, actual_stage="BATTLEFIELD", result="win")
    store.record_replay(job_id, "slot-0", 1, key="replays/a.slp", sha256="a" * 64, size=1, etag="e")


def test_claim_lease_defaults_to_twenty_seconds(tmp_path: Path) -> None:
    now = [1000.0]
    store = QueueStore(tmp_path / "queue.sqlite3", now=lambda: now[0])
    created = store.create_job("CRYO#610", MatchChoices("FOX", "IBDW#0", 2))
    store.claim_next("slot-0")
    now[0] += 19.0
    assert store.reap_expired() == 0
    now[0] += 2.0
    assert store.reap_expired() == 1
    assert created.job.id
```

In `tests/test_netplay_runner.py`, add a round-trip test for the sidecar. Model it on the existing pending-upload tests: `grep -n "_write_pending_upload\|_read_pending_upload" tests/test_netplay_runner.py`. The test writes metadata with `worker_id="sess/slot-1"`, reads it back, and asserts that `worker_id` survives. It then rewrites the payload with `"schema_version": 1` and without `worker_id`, and asserts that `RuntimeError` matches `wrong schema`.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_netplay_contract.py tests/test_netplay_runner.py -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'hal.netplay_service.queue_contract'`.

- [ ] **Step 3: Write `queue_contract.py`**

```python
"""Errors and the queue interface shared by the SQLite store and the remote queue client."""

from typing import Protocol

from hal.netplay_service.domain import Job
from hal.netplay_service.domain import JobStatus


class QueueError(RuntimeError):
    pass


class InvalidTransitionError(QueueError):
    """The worker does not own the job in a state that allows the operation."""


class SessionEndedError(InvalidTransitionError):
    """The runner's session has ended, so every lease it held is gone."""


class RunnerQueue(Protocol):
    """The queue operations a runner slot performs; lease durations belong to the queue."""

    def claim_next(self, worker_id: str) -> Job | None: ...

    def heartbeat(self, job_id: str, worker_id: str) -> None: ...

    def mark_connecting(self, job_id: str, worker_id: str, connect_code: str) -> None: ...

    def mark_playing(self, job_id: str, worker_id: str) -> None: ...

    def mark_no_show(self, job_id: str, worker_id: str) -> None: ...

    def mark_no_contest(self, job_id: str, worker_id: str) -> None: ...

    def finish_game(
        self, job_id: str, worker_id: str, *, game_number: int, actual_stage: str, result: str
    ) -> JobStatus: ...

    def fail(self, job_id: str, worker_id: str, error_code: str, *, retryable: bool) -> JobStatus: ...

    def forfeit_service_failure(self, job_id: str, worker_id: str) -> None: ...

    def record_replay(
        self, job_id: str, worker_id: str, game_number: int, *, key: str, sha256: str, size: int, etag: str
    ) -> None: ...

    def get_worker_job(self, job_id: str, worker_id: str) -> Job: ...
```

- [ ] **Step 4: Conform `QueueStore`**

In `queue.py`:
- Delete the `QueueError` and `InvalidTransitionError` classes. Import both from `hal.netplay_service.queue_contract`.
- Change the `lease_seconds` default of `claim_next` and `heartbeat` to `20.0`.
- In `finish_game`, add the keyword `game_number: int`. After `_owned_job`, raise `InvalidTransitionError(f"game_number {game_number} does not follow game {row['game_count']}")` when `game_number != row["game_count"] + 1`. Use the argument in place of the computed value.
- In `record_replay`, add `worker_id: str` as the second positional parameter, and add the invariant comment described in Interfaces.

Update importers to take `InvalidTransitionError` from `queue_contract`: `api.py`, `runner.py`, `scripts/record_netplay_transcripts.py`, and `tests/test_netplay_queue.py`. Update every `finish_game` and `record_replay` call in `tests/`, `scripts/`, and `tests/fixtures/` to the new signatures. Find them with `grep -rn "finish_game(\|record_replay(" hal tests scripts`. For each `finish_game` call, pass `game_number=<job>.game_count + 1`, where the job is the one that call site has in hand.

- [ ] **Step 5: Type the runner on `RunnerQueue`**

In `runner.py`:
- Import `RunnerQueue` and `InvalidTransitionError` from `queue_contract`.
- Annotate every parameter now typed `QueueStore` as `RunnerQueue`: `_complete_pending_upload`, `_drain_pending_uploads`, `_retry_pending_uploads`, `_heartbeat`, `_LivePolicySettings.__init__`, `_Match.__init__` (line 861), and the functions at lines 995 and 1211. Keep `store = QueueStore(config.database)` at line 1183. It is the only construction.
- Drop `lease_seconds=20.0` at lines 812 and 1200, and drop `timeout_seconds=IDLE_TIMEOUT_SECONDS` at line 1026.
- At line 1092, pass `game_number=job.game_count + 1`. That is the number the log line and the replay metadata already use.
- In `_write_pending_upload(path, metadata, worker_id)`, write `payload["worker_id"] = worker_id` and `schema_version` 2. `_read_pending_upload` expects the v2 field set and returns `(replay, metadata, worker_id)`. `_complete_pending_upload` passes that `worker_id` to `store.record_replay`. The caller at line 1128 passes `config.worker_id`.

- [ ] **Step 6: Run the affected tests and the type check**

Run:

```bash
uv run pytest tests/test_netplay_contract.py tests/test_netplay_queue.py tests/test_netplay_api.py \
  tests/test_netplay_runner.py tests/test_netplay_transcripts.py tests/test_qualify_netplay_059.py -q
uv run ty check --python-version 3.14 --error-on-warning hal scripts
```

Expected: all pass. The golden transcripts are unchanged, because the recorder passes its lease durations explicitly and plays games in order. `ty` reports no diagnostics, which confirms that `QueueStore` satisfies `RunnerQueue` at every runner call site.

- [ ] **Step 7: Commit**

```bash
git add hal/netplay_service scripts tests
git commit -m "Share queue errors and a runner queue protocol"
```

---

### Task 5: Queue client transport and job transitions

**Files:**
- Modify: `pyproject.toml` (dependencies), `uv.lock`
- Modify: `hal/netplay_service/domain.py` (add `PolicyConfig`; drop `lease_owner`, `lease_expires_at`, `created_at`, `updated_at` from `Job`)
- Modify: `hal/netplay_service/queue.py` `QueueStore._job` (stops passing the dropped fields)
- Modify: `tests/test_netplay_queue.py:254` and `:329` (assert through the store, not the dropped fields)
- Modify: `tests/test_netplay_runner.py:49-69` (`_job` helper drops the four fields)
- Create: `hal/netplay_service/queue_client.py`
- Test: `tests/test_netplay_domain.py`, `tests/test_netplay_queue_client.py`, `tests/test_netplay_contract.py` (append)

The Worker's job body has no lease owner, lease expiry, or timestamps, and the runner never reads them, so `Job` drops them rather than invent values.

**Interfaces:**
- Produces (`domain.py`): `POLICY_CONFIG_VERSION = 1`; `@dataclass(frozen=True, slots=True) class PolicyConfig` with fields `bundle_sha256: str`, `bundle_r2_key: str`, `vocabulary_sha256: str`, `characters: tuple[Choice, ...]`, `imitations: tuple[Choice, ...]`, `stages: tuple[Choice, ...]`, `online_delays: tuple[int, ...]`, `desired_return_range: tuple[float, float]`, `default_desired_return: float`, `temperature_range: tuple[float, float]`, `default_temperature: float`, `masked_identity: bool`; methods `to_payload() -> dict[str, object]`, `from_payload(payload: object) -> PolicyConfig`.
- Produces (`queue_client.py`):
  - `RETRY_DELAYS_SECONDS`.
  - Client errors `QueueRejectedError(status: int, detail: str)`, `QueueUnavailableError`, and `QueueProtocolError`. Each subclasses the contract's `QueueError`. A `409` raises the contract's `InvalidTransitionError`, and a `410` raises its `SessionEndedError`.
  - `QueueEndpoint(url, token, access_client_id=None, access_client_secret=None)` with `headers() -> dict[str, str]`.
  - `runner_endpoint(environment: Mapping[str, str]) -> QueueEndpoint`, which reads `HAL_NETPLAY_API_URL`, `HAL_NETPLAY_RUNNER_TOKEN`, `CF_ACCESS_CLIENT_ID`, and `CF_ACCESS_CLIENT_SECRET`.
  - `admin_endpoint(environment: Mapping[str, str]) -> QueueEndpoint`, which reads `HAL_NETPLAY_API_URL`, `HAL_NETPLAY_ADMIN_TOKEN`, `HAL_NETPLAY_ADMIN_ACCESS_CLIENT_ID`, and `HAL_NETPLAY_ADMIN_ACCESS_CLIENT_SECRET`.
  - `slot_worker_id(session_id: str, slot: int) -> str` and `parse_job(payload: object) -> Job`.
  - `class RemoteQueue(endpoint, session_id, *, client: httpx.Client | None = None, sleep: Callable[[float], None] = time.sleep)`. It implements `RunnerQueue` (Task 4) with the same parameter names, and it adds `close()`.
- Lease durations are owned by the server, so `RemoteQueue` has no lease arguments.

- [ ] **Step 1: Move the network dependencies to runtime**

In `pyproject.toml`, change `dependencies` to add three entries in alphabetical position:

```toml
dependencies = [
    "beartype>=0.22.9",
    "boto3",
    "fsspec",
    "httpx",
    "jaxtyping>=0.3.9",
    "loguru",
    "melee",
    "modal>=1.5.3",
    "mosaicml-streaming==0.13.0",
    "numpy",
    "peppi-py",
    "prometheus-client",
    "py7zr==1.1.0",
    "pyarrow>=22",
    "s3fs>=2025",
    "scipy>=1.17.1",
    "torch==2.11.0",
    "tqdm",
    "tyro",
    "vastai>=1.0.13",
    "wandb",
    "websockets",
]
```

Change the extra to keep only what `api.py` needs until Plan B2 deletes it:

```toml
[project.optional-dependencies]
netplay-server = [
    "fastapi",
    "uvicorn[standard]",
]
```

Remove `"httpx",` from `[dependency-groups] dev`.

Run: `uv lock && uv sync --extra netplay-server`
Expected: `uv lock` resolves without changing the locked versions of `httpx` (0.28.1), `websockets` (17.1), or `prometheus-client` (0.26.0); `git diff uv.lock` shows only the moved requirement markers.

- [ ] **Step 2: Write the failing domain tests**

`tests/test_netplay_domain.py`:

```python
import pytest

from hal.netplay_service.domain import CHARACTERS
from hal.netplay_service.domain import IMITATIONS
from hal.netplay_service.domain import STAGES
from hal.netplay_service.domain import Choice
from hal.netplay_service.domain import PolicyConfig


def _config() -> PolicyConfig:
    return PolicyConfig(
        bundle_sha256="a" * 64,
        bundle_r2_key=f"netplay/policies/{'a' * 64}.halpolicy",
        vocabulary_sha256="b" * 64,
        characters=CHARACTERS,
        imitations=IMITATIONS,
        stages=STAGES,
        online_delays=(2, 3),
        desired_return_range=(0.0, 40.0),
        default_desired_return=20.0,
        temperature_range=(0.8, 1.1),
        default_temperature=1.0,
        masked_identity=False,
    )


def test_policy_config_round_trips_the_worker_payload() -> None:
    payload = _config().to_payload()
    assert payload["schema_version"] == 1
    assert payload["characters"][0] == {"value": "FOX", "label": "Fox"}  # type: ignore[index]
    assert payload["online_delays"] == [2, 3]
    assert PolicyConfig.from_payload(payload) == _config()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"extra": 1}, "fields changed"),
        ({"schema_version": True}, "schema_version"),
        ({"schema_version": 2}, "schema_version"),
        ({"bundle_sha256": "A" * 64}, "bundle_sha256"),
        ({"online_delays": [1]}, "online_delays"),
        ({"default_desired_return": 41.0}, "desired_return"),
        ({"characters": [{"value": "FOX"}]}, "value and label"),
        ({"masked_identity": 0}, "masked_identity"),
    ],
)
def test_policy_config_rejects_drift(change: dict[str, object], message: str) -> None:
    payload = {**_config().to_payload(), **change}
    with pytest.raises(ValueError, match=message):
        PolicyConfig.from_payload(payload)


def test_policy_config_rejects_duplicate_choices() -> None:
    with pytest.raises(ValueError, match="unique"):
        PolicyConfig(**{**_fields(), "stages": (Choice("BATTLEFIELD", "Battlefield"),) * 2})


def _fields() -> dict[str, object]:
    config = _config()
    return {name: getattr(config, name) for name in PolicyConfig.__dataclass_fields__}
```

- [ ] **Step 3: Write the failing client tests**

`tests/test_netplay_queue_client.py`:

```python
import json

import httpx
import pytest

from hal.netplay_service.domain import JobStatus
from hal.netplay_service.queue_client import RETRY_DELAYS_SECONDS
from hal.netplay_service.queue_client import QueueEndpoint
from hal.netplay_service.queue_client import QueueProtocolError
from hal.netplay_service.queue_client import QueueRejectedError
from hal.netplay_service.queue_client import QueueUnavailableError
from hal.netplay_service.queue_client import RemoteQueue
from hal.netplay_service.queue_client import admin_endpoint
from hal.netplay_service.queue_client import parse_job
from hal.netplay_service.queue_client import runner_endpoint
from hal.netplay_service.queue_client import slot_worker_id
from hal.netplay_service.queue_contract import InvalidTransitionError
from hal.netplay_service.queue_contract import SessionEndedError

ENDPOINT = QueueEndpoint("https://20xx.xyz", "runner-token", "cf-id", "cf-secret")
WORKER = slot_worker_id("sess", 1)


def _job_body(**changes: object) -> dict[str, object]:
    body: dict[str, object] = {
        "id": "job-1",
        "player_code": "CRYO#610",
        "character": "FOX",
        "imitation": "IBDW#0",
        "online_delay": 2,
        "desired_return": 20,
        "temperature": 1,
        "policy_revision": 0,
        "requested_stage": None,
        "status": "leased",
        "queue_position": None,
        "attempt": 1,
        "game_count": 0,
        "connect_code": None,
        "actual_stage": None,
        "last_result": None,
        "error_code": None,
        "connect_deadline": None,
        "rematch_deadline": None,
        "cancel_after_game": False,
    }
    body.update(changes)
    return body


class _Script:
    """Answer requests in order with scripted responses or transport errors."""

    def __init__(self, *outcomes: httpx.Response | httpx.TransportError) -> None:
        self.requests: list[httpx.Request] = []
        self._outcomes = list(outcomes)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, httpx.TransportError):
            raise outcome
        return outcome


def _queue(script: _Script, sleeps: list[float]) -> RemoteQueue:
    client = httpx.Client(base_url=ENDPOINT.url, transport=httpx.MockTransport(script))
    return RemoteQueue(ENDPOINT, "sess", client=client, sleep=sleeps.append)


def test_claim_sends_auth_access_and_slot_and_maps_204_to_none() -> None:
    script = _Script(httpx.Response(204))
    assert _queue(script, []).claim_next(WORKER) is None
    request = script.requests[0]
    assert (request.method, request.url.path) == ("POST", "/v1/runner/sessions/sess/claim")
    assert json.loads(request.content) == {"slot": 1}
    assert request.headers["Authorization"] == "Bearer runner-token"
    assert request.headers["CF-Access-Client-Id"] == "cf-id"
    assert request.headers["CF-Access-Client-Secret"] == "cf-secret"


def test_job_routes_name_the_session_and_slot() -> None:
    script = _Script(httpx.Response(200, json=_job_body(status="rematch_wait", game_count=1)))
    status = _queue(script, []).finish_game("job-1", WORKER, game_number=1, actual_stage="BATTLEFIELD", result="win")
    assert status is JobStatus.REMATCH_WAIT
    request = script.requests[0]
    assert request.url.path == "/v1/runner/jobs/job-1/finish-game"
    assert (request.headers["X-HAL-Session"], request.headers["X-HAL-Slot"]) == ("sess", "1")
    assert json.loads(request.content) == {"game_number": 1, "actual_stage": "BATTLEFIELD", "result": "win"}


def test_connection_errors_and_5xx_retry_with_backoff_then_succeed() -> None:
    sleeps: list[float] = []
    script = _Script(
        httpx.ConnectError("refused"),
        httpx.Response(502),
        httpx.Response(200, json=_job_body(status="connecting", connect_code="HAL#1")),
    )
    _queue(script, sleeps).mark_connecting("job-1", WORKER, "HAL#1")
    assert sleeps == [0.25, 0.5]
    assert len(script.requests) == 3


def test_retries_are_bounded() -> None:
    sleeps: list[float] = []
    script = _Script(*(httpx.Response(503) for _ in range(len(RETRY_DELAYS_SECONDS) + 1)))
    with pytest.raises(QueueUnavailableError, match="after 6 attempts: HTTP 503"):
        _queue(script, sleeps).heartbeat("job-1", WORKER)
    assert sleeps == list(RETRY_DELAYS_SECONDS)


@pytest.mark.parametrize(
    ("status", "error"),
    [(409, InvalidTransitionError), (410, SessionEndedError), (422, QueueRejectedError), (401, QueueRejectedError)],
)
def test_4xx_is_not_retried_and_maps_to_errors(status: int, error: type[Exception]) -> None:
    sleeps: list[float] = []
    script = _Script(httpx.Response(status, json={"detail": "worker does not own this job"}))
    with pytest.raises(error, match="worker does not own this job"):
        _queue(script, sleeps).mark_playing("job-1", WORKER)
    assert sleeps == []
    assert len(script.requests) == 1


def test_session_end_is_a_lost_lease() -> None:
    assert issubclass(SessionEndedError, InvalidTransitionError)


def test_worker_from_another_session_is_rejected_before_a_request() -> None:
    script = _Script()
    with pytest.raises(ValueError, match="does not belong to session sess"):
        _queue(script, []).mark_no_show("job-1", slot_worker_id("other", 0))
    assert script.requests == []


def test_parse_job_rejects_drift() -> None:
    job = parse_job(_job_body(desired_return=None))
    assert job.choices.desired_return is None and job.status is JobStatus.LEASED
    with pytest.raises(QueueProtocolError, match="fields changed"):
        parse_job({**_job_body(), "lease_owner": "x"})
    with pytest.raises(QueueProtocolError, match="invalid values"):
        parse_job(_job_body(status="paused"))
    with pytest.raises(QueueProtocolError, match="invalid values"):
        parse_job(_job_body(attempt=True))


def test_endpoint_rejects_misconfiguration_and_hides_secrets() -> None:
    with pytest.raises(ValueError, match="trailing slash"):
        QueueEndpoint("https://20xx.xyz/", "t", "id", "secret")
    with pytest.raises(ValueError, match="both"):
        QueueEndpoint("https://20xx.xyz", "t", "id")
    with pytest.raises(ValueError, match="HAL_NETPLAY_RUNNER_TOKEN"):
        runner_endpoint({"HAL_NETPLAY_API_URL": "http://127.0.0.1:8787"})
    with pytest.raises(ValueError, match="CF_ACCESS_CLIENT_ID"):
        runner_endpoint({"HAL_NETPLAY_API_URL": "https://20xx.xyz", "HAL_NETPLAY_RUNNER_TOKEN": "t"})
    with pytest.raises(ValueError, match="HAL_NETPLAY_ADMIN_ACCESS_CLIENT_ID"):
        admin_endpoint({"HAL_NETPLAY_API_URL": "https://20xx.xyz", "HAL_NETPLAY_ADMIN_TOKEN": "t"})
    local = runner_endpoint(
        {"HAL_NETPLAY_API_URL": "http://127.0.0.1:8787", "HAL_NETPLAY_RUNNER_TOKEN": "dev", "CF_ACCESS_CLIENT_ID": ""}
    )
    assert local.headers() == {"Authorization": "Bearer dev"}
    admin = admin_endpoint(
        {
            "HAL_NETPLAY_API_URL": "https://20xx.xyz",
            "HAL_NETPLAY_ADMIN_TOKEN": "a",
            "HAL_NETPLAY_ADMIN_ACCESS_CLIENT_ID": "admin-id",
            "HAL_NETPLAY_ADMIN_ACCESS_CLIENT_SECRET": "admin-secret",
            "CF_ACCESS_CLIENT_ID": "runner-id",
            "CF_ACCESS_CLIENT_SECRET": "runner-secret",
        }
    )
    assert admin.headers()["CF-Access-Client-Id"] == "admin-id"
    assert "cf-secret" not in repr(ENDPOINT) and "runner-token" not in repr(ENDPOINT)
```

Append to `tests/test_netplay_contract.py`. This test pins conformance for `ty`. It fails type checking, not at run time, if `RemoteQueue` drifts from the protocol:

```python
from hal.netplay_service.queue_client import QueueEndpoint
from hal.netplay_service.queue_client import RemoteQueue


def test_remote_queue_is_a_runner_queue() -> None:
    remote = RemoteQueue(QueueEndpoint("http://127.0.0.1:8787", "t"), "sess")
    queue: RunnerQueue = remote
    assert queue is remote
    remote.close()
```

- [ ] **Step 4: Run them to verify they fail**

Run: `uv run pytest tests/test_netplay_domain.py tests/test_netplay_queue_client.py tests/test_netplay_contract.py -q`
Expected: FAIL at collection — `ImportError: cannot import name 'PolicyConfig'` and `ModuleNotFoundError: No module named 'hal.netplay_service.queue_client'`.

- [ ] **Step 5: Add `PolicyConfig` and trim `Job` in `domain.py`**

Add `from pathlib import Path` beside the other imports (it is used in Task 8; add it now so the import block changes once). Replace the `Job` dataclass with:

```python
@dataclass(frozen=True, slots=True)
class Job:
    id: str
    player_code: str
    choices: MatchChoices
    status: JobStatus
    queue_position: int | None
    attempt: int
    game_count: int
    connect_code: str | None
    actual_stage: str | None
    last_result: str | None
    error_code: str | None
    connect_deadline: float | None
    rematch_deadline: float | None
    cancel_after_game: bool
    policy_revision: int = 0
```

Append to `domain.py`:

```python
POLICY_CONFIG_VERSION: Final[int] = 1
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
_POLICY_FIELDS: Final[frozenset[str]] = frozenset(
    (
        "schema_version",
        "bundle_sha256",
        "bundle_r2_key",
        "vocabulary_sha256",
        "characters",
        "imitations",
        "stages",
        "online_delays",
        "desired_return_range",
        "default_desired_return",
        "temperature_range",
        "default_temperature",
        "masked_identity",
    )
)


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """The active policy as published to the queue Worker, which checks the same rules."""

    bundle_sha256: str
    bundle_r2_key: str
    vocabulary_sha256: str
    characters: tuple[Choice, ...]
    imitations: tuple[Choice, ...]
    stages: tuple[Choice, ...]
    online_delays: tuple[int, ...]
    desired_return_range: tuple[float, float]
    default_desired_return: float
    temperature_range: tuple[float, float]
    default_temperature: float
    masked_identity: bool

    def __post_init__(self) -> None:
        for name, value in (("bundle_sha256", self.bundle_sha256), ("vocabulary_sha256", self.vocabulary_sha256)):
            if _SHA256.fullmatch(value) is None:
                raise ValueError(f"policy config {name} must be lowercase SHA-256 hex")
        if not self.bundle_r2_key:
            raise ValueError("policy config bundle_r2_key must be non-empty")
        for name, choices in (("characters", self.characters), ("imitations", self.imitations), ("stages", self.stages)):
            values = [choice.value for choice in choices]
            if not values or len(set(values)) != len(values) or not all(c.value and c.label for c in choices):
                raise ValueError(f"policy config {name} must be non-empty, labeled, and unique")
        if not self.online_delays or any(delay not in (2, 3) for delay in self.online_delays):
            raise ValueError("policy config online_delays must be a non-empty subset of (2, 3)")
        for name, (low, high), default in (
            ("desired_return", self.desired_return_range, self.default_desired_return),
            ("temperature", self.temperature_range, self.default_temperature),
        ):
            if not (math.isfinite(low) and math.isfinite(high) and low < high and low <= default <= high):
                raise ValueError(f"policy config {name} range and default are invalid")

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLICY_CONFIG_VERSION,
            "bundle_sha256": self.bundle_sha256,
            "bundle_r2_key": self.bundle_r2_key,
            "vocabulary_sha256": self.vocabulary_sha256,
            "characters": [{"value": choice.value, "label": choice.label} for choice in self.characters],
            "imitations": [{"value": choice.value, "label": choice.label} for choice in self.imitations],
            "stages": [{"value": choice.value, "label": choice.label} for choice in self.stages],
            "online_delays": list(self.online_delays),
            "desired_return_range": list(self.desired_return_range),
            "default_desired_return": self.default_desired_return,
            "temperature_range": list(self.temperature_range),
            "default_temperature": self.default_temperature,
            "masked_identity": self.masked_identity,
        }

    @classmethod
    def from_payload(cls, payload: object) -> PolicyConfig:
        if not isinstance(payload, dict) or set(payload) != _POLICY_FIELDS:
            raise ValueError("policy config fields changed")
        if _integer(payload["schema_version"], "schema_version") != POLICY_CONFIG_VERSION:
            raise ValueError(f"policy config schema_version must be {POLICY_CONFIG_VERSION}")
        return cls(
            bundle_sha256=_text(payload["bundle_sha256"], "bundle_sha256"),
            bundle_r2_key=_text(payload["bundle_r2_key"], "bundle_r2_key"),
            vocabulary_sha256=_text(payload["vocabulary_sha256"], "vocabulary_sha256"),
            characters=_choices(payload["characters"], "characters"),
            imitations=_choices(payload["imitations"], "imitations"),
            stages=_choices(payload["stages"], "stages"),
            online_delays=tuple(_integer(delay, "online_delays") for delay in _list(payload["online_delays"], "online_delays")),
            desired_return_range=_range(payload["desired_return_range"], "desired_return_range"),
            default_desired_return=_number(payload["default_desired_return"], "default_desired_return"),
            temperature_range=_range(payload["temperature_range"], "temperature_range"),
            default_temperature=_number(payload["default_temperature"], "default_temperature"),
            masked_identity=_boolean(payload["masked_identity"], "masked_identity"),
        )


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _range(value: object, name: str) -> tuple[float, float]:
    items = _list(value, name)
    if len(items) != 2:
        raise ValueError(f"{name} must have two numbers")
    return _number(items[0], name), _number(items[1], name)


def _choices(value: object, name: str) -> tuple[Choice, ...]:
    choices: list[Choice] = []
    for item in _list(value, name):
        if not isinstance(item, dict) or set(item) != {"value", "label"}:
            raise ValueError(f"{name} entries need exactly value and label")
        choices.append(Choice(_text(item["value"], name), _text(item["label"], name)))
    return tuple(choices)
```

- [ ] **Step 6: Stop setting the dropped fields in `queue.py`**

In `hal/netplay_service/queue.py` `QueueStore._job` (lines 201–243), delete these four keyword arguments from the `Job(...)` call:

```python
            lease_owner=row["lease_owner"],
            lease_expires_at=row["lease_expires_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
```

In `tests/test_netplay_queue.py`, replace line 254 (`assert canceled.lease_owner is None`) with:

```python
    with pytest.raises(InvalidTransitionError, match="worker does not own this job"):
        store.get_worker_job(job.id, "slot-0")
```

and replace line 329 (`assert other.status is JobStatus.LEASED and other.lease_owner == owners[3]`) with:

```python
    assert other.status is JobStatus.LEASED
    assert store.get_worker_job(other.id, owners[3]).status is JobStatus.LEASED
```

If `InvalidTransitionError` or `pytest` is not yet imported in that file, add `from hal.netplay_service.queue_contract import InvalidTransitionError` and `import pytest` in the sorted import block.

In `tests/test_netplay_runner.py` `_job` (lines 49–69), delete the four lines `lease_owner="slot-0",`, `lease_expires_at=None,`, `created_at=0,`, `updated_at=0,`.

- [ ] **Step 7: Write `queue_client.py` (transport, errors, job transitions)**

`hal/netplay_service/queue_client.py`:

```python
"""Clients for the netplay queue Worker's runner and admin routes."""

import re
import time
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Final

import httpx
from loguru import logger

from hal.netplay_service.domain import Job
from hal.netplay_service.domain import JobStatus
from hal.netplay_service.domain import MatchChoices
from hal.netplay_service.domain import validate_player_code
from hal.netplay_service.domain import validate_stage
from hal.netplay_service.queue_contract import InvalidTransitionError
from hal.netplay_service.queue_contract import QueueError
from hal.netplay_service.queue_contract import SessionEndedError

# Every runner and admin route is idempotent at the Worker (a repeat after a lost
# response returns the first result), so any request may be retried. A 4xx is final.
RETRY_DELAYS_SECONDS: Final[tuple[float, ...]] = (0.25, 0.5, 1.0, 2.0, 4.0)
_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(10.0, connect=5.0)
_WORKER_ID: Final[re.Pattern[str]] = re.compile(r"(?P<session>[A-Za-z0-9_-]+)/slot-(?P<slot>[0-9]+)")
_JOB_FIELDS: Final[frozenset[str]] = frozenset(
    (
        "id",
        "player_code",
        "character",
        "imitation",
        "online_delay",
        "desired_return",
        "temperature",
        "policy_revision",
        "requested_stage",
        "status",
        "queue_position",
        "attempt",
        "game_count",
        "connect_code",
        "actual_stage",
        "last_result",
        "error_code",
        "connect_deadline",
        "rematch_deadline",
        "cancel_after_game",
    )
)


class QueueRejectedError(QueueError):
    """The queue refused a request with a 4xx other than 409 or 410."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"queue refused the request ({status}): {detail}")
        self.status = status
        self.detail = detail


class QueueUnavailableError(QueueError):
    """Connection errors or 5xx outlasted every retry."""


class QueueProtocolError(QueueError):
    """The queue returned a body this client does not understand."""


@dataclass(frozen=True, slots=True)
class QueueEndpoint:
    """Connection settings; each process builds its own client from them."""

    url: str
    token: str = field(repr=False)
    access_client_id: str | None = None
    access_client_secret: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.url.startswith(("https://", "http://")) or self.url.endswith("/"):
            raise ValueError("HAL_NETPLAY_API_URL must be an http(s) origin without a trailing slash")
        if not self.token:
            raise ValueError("the queue bearer token must be non-empty")
        if (self.access_client_id is None) != (self.access_client_secret is None):
            raise ValueError("set both CF_ACCESS_CLIENT_ID and CF_ACCESS_CLIENT_SECRET, or neither")
        if self.url.startswith("https://") and self.access_client_id is None:
            raise ValueError("a public queue URL needs Cloudflare Access service-token credentials")

    def headers(self) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.token}"}
        if self.access_client_id is not None and self.access_client_secret is not None:
            headers["CF-Access-Client-Id"] = self.access_client_id
            headers["CF-Access-Client-Secret"] = self.access_client_secret
        return headers


def _endpoint(environment: Mapping[str, str], token: str, access_id: str, access_secret: str) -> QueueEndpoint:
    missing = [name for name in ("HAL_NETPLAY_API_URL", token) if not environment.get(name)]
    url = environment.get("HAL_NETPLAY_API_URL", "")
    if url.startswith("https://"):
        missing += [name for name in (access_id, access_secret) if not environment.get(name)]
    if missing:
        raise ValueError(f"set {', '.join(missing)}")
    return QueueEndpoint(
        url,
        environment[token],
        environment.get(access_id) or None,
        environment.get(access_secret) or None,
    )


def runner_endpoint(environment: Mapping[str, str]) -> QueueEndpoint:
    return _endpoint(environment, "HAL_NETPLAY_RUNNER_TOKEN", "CF_ACCESS_CLIENT_ID", "CF_ACCESS_CLIENT_SECRET")


def admin_endpoint(environment: Mapping[str, str]) -> QueueEndpoint:
    """The admin tool has its own Access service token, so one .env can also hold a runner's."""
    return _endpoint(
        environment,
        "HAL_NETPLAY_ADMIN_TOKEN",
        "HAL_NETPLAY_ADMIN_ACCESS_CLIENT_ID",
        "HAL_NETPLAY_ADMIN_ACCESS_CLIENT_SECRET",
    )


def slot_worker_id(session_id: str, slot: int) -> str:
    """Match the Worker's `workerId(session, slot)`."""
    return f"{session_id}/slot-{slot}"


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text or response.reason_phrase
    if isinstance(body, dict) and isinstance(body.get("detail"), str):
        return body["detail"]
    return response.text


def _checked(response: httpx.Response) -> httpx.Response:
    if response.status_code < 400:
        return response
    detail = _detail(response)
    if response.status_code == 409:
        raise InvalidTransitionError(detail)
    if response.status_code == 410:
        raise SessionEndedError(detail)
    raise QueueRejectedError(response.status_code, detail)


class _Api:
    """One HTTP client with bounded retries and status-code mapping."""

    def __init__(self, endpoint: QueueEndpoint, client: httpx.Client | None, sleep: Callable[[float], None]) -> None:
        self.endpoint = endpoint
        self._client = httpx.Client(base_url=endpoint.url, timeout=_TIMEOUT) if client is None else client
        self._sleep = sleep

    def close(self) -> None:
        self._client.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        body: object = None,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        merged = {**self.endpoint.headers(), **(headers or {})}
        delays: tuple[float | None, ...] = (*RETRY_DELAYS_SECONDS, None)
        for attempt, delay in enumerate(delays, start=1):
            try:
                response = self._client.request(method, path, json=body, headers=merged, params=params)
            except httpx.TransportError as error:
                failure = f"{type(error).__name__}: {error}"
            else:
                if response.status_code < 500:
                    return _checked(response)
                failure = f"HTTP {response.status_code}"
            if delay is None:
                raise QueueUnavailableError(f"{method} {path} failed after {attempt} attempts: {failure}")
            logger.bind(event="queue_retry").warning(
                "queue request {} {} failed: {}; retrying in {}s", method, path, failure, delay
            )
            self._sleep(delay)
        raise AssertionError("the retry loop always returns or raises")


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected a string, got {value!r}")
    return value


def _optional_text(value: object) -> str | None:
    return None if value is None else _text(value)


def _integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"expected an integer, got {value!r}")
    return value


def _optional_integer(value: object) -> int | None:
    return None if value is None else _integer(value)


def _number(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"expected a number, got {value!r}")
    return float(value)


def _optional_number(value: object) -> float | None:
    return None if value is None else _number(value)


def _boolean(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"expected a boolean, got {value!r}")
    return value


def parse_job(payload: object) -> Job:
    """Read one job body; the player and runner routes return the same shape."""
    if not isinstance(payload, dict) or set(payload) != _JOB_FIELDS:
        raise QueueProtocolError("job response fields changed")
    try:
        return Job(
            id=_text(payload["id"]),
            player_code=validate_player_code(_text(payload["player_code"])),
            choices=MatchChoices(
                character=_text(payload["character"]),
                imitation=_text(payload["imitation"]),
                online_delay=_integer(payload["online_delay"]),
                requested_stage=_optional_text(payload["requested_stage"]),
                desired_return=_optional_number(payload["desired_return"]),
                temperature=_number(payload["temperature"]),
            ),
            status=JobStatus(_text(payload["status"])),
            queue_position=_optional_integer(payload["queue_position"]),
            attempt=_integer(payload["attempt"]),
            game_count=_integer(payload["game_count"]),
            connect_code=_optional_text(payload["connect_code"]),
            actual_stage=_optional_text(payload["actual_stage"]),
            last_result=_optional_text(payload["last_result"]),
            error_code=_optional_text(payload["error_code"]),
            connect_deadline=_optional_number(payload["connect_deadline"]),
            rematch_deadline=_optional_number(payload["rematch_deadline"]),
            cancel_after_game=_boolean(payload["cancel_after_game"]),
            policy_revision=_integer(payload["policy_revision"]),
        )
    except (TypeError, ValueError) as error:
        raise QueueProtocolError(f"job response contains invalid values: {error}") from error


def _json(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError as error:
        raise QueueProtocolError(f"queue response is not JSON: {response.text[:200]}") from error


class RemoteQueue:
    """The `RunnerQueue` for one session, over the Worker's runner routes."""

    def __init__(
        self,
        endpoint: QueueEndpoint,
        session_id: str,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.endpoint = endpoint
        self.session_id = session_id
        self._api = _Api(endpoint, client, sleep)

    def close(self) -> None:
        self._api.close()

    def _slot(self, worker_id: str) -> int:
        match = _WORKER_ID.fullmatch(worker_id)
        if match is None or match["session"] != self.session_id:
            raise ValueError(f"worker {worker_id!r} does not belong to session {self.session_id}")
        return int(match["slot"])

    def _slot_headers(self, worker_id: str) -> dict[str, str]:
        return {"X-HAL-Session": self.session_id, "X-HAL-Slot": str(self._slot(worker_id))}

    def _transition(self, job_id: str, worker_id: str, action: str, body: object = None) -> Job:
        response = self._api.request(
            "POST", f"/v1/runner/jobs/{job_id}/{action}", body=body, headers=self._slot_headers(worker_id)
        )
        return parse_job(_json(response))

    def claim_next(self, worker_id: str) -> Job | None:
        response = self._api.request(
            "POST", f"/v1/runner/sessions/{self.session_id}/claim", body={"slot": self._slot(worker_id)}
        )
        return None if response.status_code == 204 else parse_job(_json(response))

    def heartbeat(self, job_id: str, worker_id: str) -> None:
        self._transition(job_id, worker_id, "heartbeat")

    def mark_connecting(self, job_id: str, worker_id: str, connect_code: str) -> None:
        self._transition(job_id, worker_id, "connecting", {"connect_code": validate_player_code(connect_code)})

    def mark_playing(self, job_id: str, worker_id: str) -> None:
        self._transition(job_id, worker_id, "playing")

    def mark_no_show(self, job_id: str, worker_id: str) -> None:
        self._transition(job_id, worker_id, "no-show")

    def mark_no_contest(self, job_id: str, worker_id: str) -> None:
        self._transition(job_id, worker_id, "no-contest")

    def finish_game(self, job_id: str, worker_id: str, *, game_number: int, actual_stage: str, result: str) -> JobStatus:
        body = {"game_number": game_number, "actual_stage": validate_stage(actual_stage), "result": result}
        return self._transition(job_id, worker_id, "finish-game", body).status

    def fail(self, job_id: str, worker_id: str, error_code: str, *, retryable: bool) -> JobStatus:
        return self._transition(job_id, worker_id, "fail", {"error_code": error_code, "retryable": retryable}).status

    def forfeit_service_failure(self, job_id: str, worker_id: str) -> None:
        self._transition(job_id, worker_id, "forfeit")

    def record_replay(
        self, job_id: str, worker_id: str, game_number: int, *, key: str, sha256: str, size: int, etag: str
    ) -> None:
        body = {"game_number": game_number, "key": key, "sha256": sha256, "size": size, "etag": etag}
        self._transition(job_id, worker_id, "replay", body)

    def get_worker_job(self, job_id: str, worker_id: str) -> Job:
        response = self._api.request("GET", f"/v1/runner/jobs/{job_id}", headers=self._slot_headers(worker_id))
        return parse_job(_json(response))
```

- [ ] **Step 8: Run the tests**

Run: `uv run pytest tests/test_netplay_domain.py tests/test_netplay_queue_client.py tests/test_netplay_contract.py tests/test_netplay_queue.py tests/test_netplay_runner.py tests/test_netplay_transcripts.py -q && uv run ty check --python-version 3.14 --error-on-warning hal tests/test_netplay_contract.py`
Expected: all pass. (`test_netplay_transcripts.py` re-records from `queue.py`; the dropped `Job` fields never reached a response, so the transcripts do not change.)

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock hal/netplay_service/domain.py hal/netplay_service/queue.py \
  hal/netplay_service/queue_client.py tests/test_netplay_domain.py tests/test_netplay_queue_client.py \
  tests/test_netplay_queue.py tests/test_netplay_runner.py tests/test_netplay_contract.py
git commit -m "Add the netplay queue client"
```

---

### Task 6: Session, heartbeat, admin, and live-settings clients

**Files:**
- Modify: `hal/netplay_service/queue_client.py` (append)
- Test: `tests/test_netplay_queue_client.py` (append)

**Interfaces:**
- Consumes: Task 5 `_Api`, `_json`, `parse_job`, `QueueProtocolError`, `QueueUnavailableError`; Task 4 `QueueError`, `SessionEndedError`; `PolicyConfig`; `hal.netplay_service.health.RunnerStatus.to_payload()`.
- Produces (value types): `@dataclass(frozen=True, slots=True) class Account(connect_code: str, r2_key: str, sha256: str)`; `AccountGrant(slot: int, connect_code: str, r2_key: str, sha256: str)`; `StartedSession(session_id: str, policy: PolicyConfig, accounts: tuple[AccountGrant, ...])`; `SessionState(draining: bool)`.
- Produces `new_session_id() -> str`, which returns `secrets.token_urlsafe(12)` (16 characters). The caller creates the ID once and passes it to `start_session`, so a repeat at the caller's level is also a safe retry.
- Produces `class RunnerClient(endpoint, *, client=None, sleep=time.sleep)` with:
  - `close()` and `active_policy() -> PolicyConfig`.
  - `start_session(*, session_id: str, host: str, bundle_sha256: str, git_sha: str, slots: int, wants_stream: bool) -> StartedSession`.
  - `report_status(session_id: str, status: RunnerStatus) -> SessionState`.
  - `drain(session_id) -> None` and `end_session(session_id) -> int`, which returns the failed lease count.
- `report_status` requires the response fields to be exactly `draining` and `stream`. In B1, `stream` must be `null`, because no runner asks for the stream and the Worker grants none. A non-null `stream` raises `QueueProtocolError("this client cannot stream; the Worker granted the stream lease")`. Plan D adds a grant field to `SessionState` and parses it. The call sites keep the structured result.
- Produces `class SessionReporter(client: RunnerClient, session_id: str, status: Callable[[], RunnerStatus], *, interval_seconds: float = 2.0)`, a context manager:
  - It reports once in `__enter__`, so a bad status or an ended session fails at once. It then reports from a daemon thread every `interval_seconds`.
  - `state() -> SessionState` returns the latest response. After a `QueueError` other than `QueueUnavailableError`, `state()` raises that error, and the thread stops.
  - `QueueUnavailableError` is logged and retried on the next tick, because the Worker allows 30 s of silence.
  - `__exit__` stops and joins the thread.
  - The runner enters it right after `start_session`, before downloads and qualification, which can take longer than the 30 s silence limit.
- Produces `class AdminClient(endpoint, *, client=None, sleep=time.sleep)` with `close()`, `put_policy(config: PolicyConfig) -> None`, `put_accounts(accounts: Sequence[Account]) -> None`, `set_paused(paused: bool) -> None`, `status() -> dict[str, object]`, and `events(*, job: str | None, session: str | None, since: float | None) -> list[dict[str, object]]`.
- Produces `RemoteQueue.connect_live(job_id: str, worker_id: str, *, open_timeout: float = 2.0) -> ClientConnection`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_netplay_queue_client.py`:

```python
import threading
import time
from collections.abc import Callable

from websockets.sync.server import ServerConnection
from websockets.sync.server import serve

from hal.netplay_service.domain import CHARACTERS
from hal.netplay_service.domain import IMITATIONS
from hal.netplay_service.domain import STAGES
from hal.netplay_service.domain import PolicyConfig
from hal.netplay_service.health import RunnerStatus
from hal.netplay_service.health import SlotState
from hal.netplay_service.health import SlotStatus
from hal.netplay_service.health import aggregate_runner_status
from hal.netplay_service.queue_client import Account
from hal.netplay_service.queue_client import AdminClient
from hal.netplay_service.queue_client import RunnerClient
from hal.netplay_service.queue_client import SessionReporter
from hal.netplay_service.queue_client import SessionState

POLICY = PolicyConfig(
    bundle_sha256="a" * 64,
    bundle_r2_key=f"netplay/policies/{'a' * 64}.halpolicy",
    vocabulary_sha256="b" * 64,
    characters=CHARACTERS,
    imitations=IMITATIONS,
    stages=STAGES,
    online_delays=(2, 3),
    desired_return_range=(0.0, 40.0),
    default_desired_return=20.0,
    temperature_range=(0.8, 1.1),
    default_temperature=1.0,
    masked_identity=False,
)


def _runner(script: _Script) -> RunnerClient:
    return RunnerClient(ENDPOINT, client=httpx.Client(base_url=ENDPOINT.url, transport=httpx.MockTransport(script)))


def _admin(script: _Script) -> AdminClient:
    return AdminClient(ENDPOINT, client=httpx.Client(base_url=ENDPOINT.url, transport=httpx.MockTransport(script)))


def test_start_session_reads_policy_and_one_account_per_slot() -> None:
    grant = {"slot": 0, "connect_code": "BOT0#1", "r2_key": "netplay/accounts/x.json", "sha256": "c" * 64}
    script = _Script(
        httpx.Response(201, json={"session_id": "sess", "accounts": [grant], "policy": POLICY.to_payload()})
    )
    started = _runner(script).start_session(
        session_id="session-00000001", host="box", bundle_sha256="a" * 64, git_sha="d" * 40, slots=1, wants_stream=False
    )
    assert started.session_id == "sess" and started.policy == POLICY
    assert started.accounts[0].connect_code == "BOT0#1"
    assert json.loads(script.requests[0].content) == {
        "session_id": "session-00000001",
        "host": "box",
        "bundle_sha256": "a" * 64,
        "git_sha": "d" * 40,
        "slots": 1,
        "stream": False,
    }


def test_start_session_rejects_a_missing_or_misnumbered_account() -> None:
    body = {"session_id": "sess", "accounts": [], "policy": POLICY.to_payload()}
    with pytest.raises(QueueProtocolError, match="one account per slot"):
        _runner(_Script(httpx.Response(201, json=body))).start_session(
            session_id="session-00000001", host="box", bundle_sha256="a" * 64, git_sha="d" * 40, slots=1, wants_stream=False
        )


def _runner_status() -> RunnerStatus:
    slot = SlotStatus(0, SlotState.IDLE, None, None, None, None, None, 0, time.time())
    return aggregate_runner_status("a" * 64, (slot,), time.time(), model_inference_p95_ms=None, batch_wait_p95_ms=None)


def test_status_report_returns_the_session_state() -> None:
    script = _Script(httpx.Response(200, json={"draining": True, "stream": None}))
    assert _runner(script).report_status("sess", _runner_status()) == SessionState(draining=True)
    assert json.loads(script.requests[0].content)["schema_version"] == 5
    assert script.requests[0].url.path == "/v1/runner/sessions/sess/status"


def test_status_report_refuses_a_stream_grant() -> None:
    script = _Script(httpx.Response(200, json={"draining": False, "stream": {"slot": 0, "key": "live_x"}}))
    with pytest.raises(QueueProtocolError, match="cannot stream"):
        _runner(script).report_status("sess", _runner_status())


def _wait_for(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + 5
    while not condition():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_reporter_keeps_reporting_through_outages_and_stops_on_session_end() -> None:
    script = _Script(
        httpx.Response(200, json={"draining": False, "stream": None}),
        *(httpx.Response(503) for _ in range(6)),
        httpx.Response(200, json={"draining": True, "stream": None}),
        httpx.Response(410, json={"detail": "session has ended"}),
    )
    client = RunnerClient(
        ENDPOINT,
        client=httpx.Client(base_url=ENDPOINT.url, transport=httpx.MockTransport(script)),
        sleep=lambda _: None,
    )
    with SessionReporter(client, "sess", _runner_status, interval_seconds=0.01) as reporter:
        assert reporter.state() == SessionState(draining=False)

        def ended() -> bool:
            try:
                reporter.state()
            except SessionEndedError:
                return True
            return False

        # One report retries through six 503s, the next sees the drain, the last sees the end.
        _wait_for(ended)
    assert len(script.requests) == 9


def test_active_policy_drain_and_end() -> None:
    script = _Script(
        httpx.Response(200, json=POLICY.to_payload()),
        httpx.Response(200, json={"draining": True}),
        httpx.Response(200, json={"failed": 2}),
    )
    client = _runner(script)
    assert client.active_policy() == POLICY
    client.drain("sess")
    assert client.end_session("sess") == 2
    assert [(r.method, r.url.path) for r in script.requests] == [
        ("GET", "/v1/runner/policy"),
        ("POST", "/v1/runner/sessions/sess/drain"),
        ("DELETE", "/v1/runner/sessions/sess"),
    ]


def test_admin_routes() -> None:
    script = _Script(
        httpx.Response(200, json=POLICY.to_payload()),
        httpx.Response(200, json=[]),
        httpx.Response(200, json={"paused": True}),
        httpx.Response(200, json={"events": [{"kind": "job_created"}]}),
    )
    admin = _admin(script)
    admin.put_policy(POLICY)
    admin.put_accounts([Account("BOT0#1", "netplay/accounts/x.json", "c" * 64)])
    admin.set_paused(True)
    assert admin.events(job="j1", session=None, since=12.5) == [{"kind": "job_created"}]
    assert json.loads(script.requests[1].content) == [
        {"connect_code": "BOT0#1", "r2_key": "netplay/accounts/x.json", "sha256": "c" * 64}
    ]
    assert script.requests[2].url.path == "/v1/admin/pause"
    assert dict(script.requests[3].url.params) == {"job": "j1", "since": "12.5"}


def test_live_socket_sends_auth_and_slot_headers() -> None:
    seen: dict[str, str | None] = {}

    def handler(connection: ServerConnection) -> None:
        seen["path"] = connection.request.path if connection.request is not None else None
        headers = connection.request.headers if connection.request is not None else {}
        seen["auth"] = headers.get("Authorization")
        seen["slot"] = headers.get("X-HAL-Slot")
        connection.send('{"type": "released"}')

    with serve(handler, "127.0.0.1", 0) as server:
        port = server.socket.getsockname()[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = QueueEndpoint(f"http://127.0.0.1:{port}", "runner-token")
        queue = RemoteQueue(endpoint, "sess")
        with queue.connect_live("job-1", slot_worker_id("sess", 1)) as socket:
            assert json.loads(socket.recv(timeout=2)) == {"type": "released"}
        server.shutdown()
    assert seen == {"path": "/v1/runner/jobs/job-1/live", "auth": "Bearer runner-token", "slot": "1"}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_netplay_queue_client.py -q`
Expected: FAIL at collection — `ImportError: cannot import name 'Account'`.

- [ ] **Step 3: Implement the clients**

Add to the imports of `queue_client.py`:

```python
import secrets
import threading
from collections.abc import Sequence

from websockets.sync.client import ClientConnection
from websockets.sync.client import connect

from hal.netplay_service.domain import PolicyConfig
from hal.netplay_service.health import RunnerStatus
from hal.netplay_service.queue_contract import SessionEndedError
```

`SessionEndedError` is already imported from Task 5. Keep a single import.

Add this method to `RemoteQueue`:

```python
    def connect_live(self, job_id: str, worker_id: str, *, open_timeout: float = 2.0) -> ClientConnection:
        """Open the job's settings socket; `https` becomes `wss` and `http` becomes `ws`."""
        url = "ws" + self.endpoint.url.removeprefix("http") + f"/v1/runner/jobs/{job_id}/live"
        headers = {**self.endpoint.headers(), **self._slot_headers(worker_id)}
        return connect(url, additional_headers=headers, open_timeout=open_timeout)
```

Append:

```python
@dataclass(frozen=True, slots=True)
class Account:
    connect_code: str
    r2_key: str
    sha256: str


@dataclass(frozen=True, slots=True)
class AccountGrant:
    slot: int
    connect_code: str
    r2_key: str
    sha256: str


@dataclass(frozen=True, slots=True)
class StartedSession:
    session_id: str
    policy: PolicyConfig
    accounts: tuple[AccountGrant, ...]


@dataclass(frozen=True, slots=True)
class SessionState:
    draining: bool


def new_session_id() -> str:
    return secrets.token_urlsafe(12)


def _object(payload: object, fields: frozenset[str], name: str) -> dict[str, object]:
    if not isinstance(payload, dict) or set(payload) != fields:
        raise QueueProtocolError(f"{name} response fields changed")
    return payload


def _grant(payload: object) -> AccountGrant:
    grant = _object(payload, frozenset(("slot", "connect_code", "r2_key", "sha256")), "account grant")
    try:
        return AccountGrant(
            slot=_integer(grant["slot"]),
            connect_code=validate_player_code(_text(grant["connect_code"])),
            r2_key=_text(grant["r2_key"]),
            sha256=_text(grant["sha256"]),
        )
    except (TypeError, ValueError) as error:
        raise QueueProtocolError(f"account grant contains invalid values: {error}") from error


class RunnerClient:
    """Session routes: active policy, start, status heartbeat, drain, and end."""

    def __init__(
        self,
        endpoint: QueueEndpoint,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api = _Api(endpoint, client, sleep)

    def close(self) -> None:
        self._api.close()

    def active_policy(self) -> PolicyConfig:
        payload = _json(self._api.request("GET", "/v1/runner/policy"))
        try:
            return PolicyConfig.from_payload(payload)
        except ValueError as error:
            raise QueueProtocolError(f"active policy is invalid: {error}") from error

    def start_session(
        self, *, session_id: str, host: str, bundle_sha256: str, git_sha: str, slots: int, wants_stream: bool
    ) -> StartedSession:
        body = {
            "session_id": session_id,
            "host": host,
            "bundle_sha256": bundle_sha256,
            "git_sha": git_sha,
            "slots": slots,
            "stream": wants_stream,
        }
        payload = _object(
            _json(self._api.request("POST", "/v1/runner/sessions", body=body)),
            frozenset(("session_id", "accounts", "policy")),
            "session start",
        )
        accounts = payload["accounts"]
        if not isinstance(accounts, list):
            raise QueueProtocolError("session start accounts must be a list")
        grants = tuple(_grant(item) for item in accounts)
        if tuple(grant.slot for grant in grants) != tuple(range(slots)):
            raise QueueProtocolError(f"session start must lease one account per slot: {grants}")
        if payload["session_id"] != session_id:
            raise QueueProtocolError(f"session start returned {payload['session_id']!r}, not {session_id!r}")
        try:
            policy = PolicyConfig.from_payload(payload["policy"])
        except ValueError as error:
            raise QueueProtocolError(f"session start contains invalid values: {error}") from error
        return StartedSession(session_id, policy, grants)

    def report_status(self, session_id: str, status: RunnerStatus) -> SessionState:
        payload = _object(
            _json(self._api.request("POST", f"/v1/runner/sessions/{session_id}/status", body=status.to_payload())),
            frozenset(("draining", "stream")),
            "status",
        )
        if payload["stream"] is not None:
            raise QueueProtocolError("this client cannot stream; the Worker granted the stream lease")
        try:
            return SessionState(draining=_boolean(payload["draining"]))
        except TypeError as error:
            raise QueueProtocolError(f"status response is invalid: {error}") from error

    def drain(self, session_id: str) -> None:
        self._api.request("POST", f"/v1/runner/sessions/{session_id}/drain")

    def end_session(self, session_id: str) -> int:
        payload = _object(
            _json(self._api.request("DELETE", f"/v1/runner/sessions/{session_id}")), frozenset(("failed",)), "end"
        )
        try:
            return _integer(payload["failed"])
        except TypeError as error:
            raise QueueProtocolError(f"end response is invalid: {error}") from error


class SessionReporter:
    """Report runner status in the background; the report is also the session heartbeat.

    Enter it right after the session starts. The Worker ends a session after 30 s
    without a report, and downloads and qualification can take longer than that.
    """

    def __init__(
        self,
        client: RunnerClient,
        session_id: str,
        status: Callable[[], RunnerStatus],
        *,
        interval_seconds: float = 2.0,
    ) -> None:
        self._client = client
        self._session_id = session_id
        self._status = status
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._state: SessionState | None = None
        self._error: QueueError | None = None
        self._thread = threading.Thread(target=self._run, name=f"session-{session_id}", daemon=True)

    def __enter__(self) -> SessionReporter:
        # The first report is synchronous and strict, so startup fails fast on any queue error.
        self._state = self._client.report_status(self._session_id, self._status())
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def state(self) -> SessionState:
        with self._lock:
            if self._error is not None:
                raise self._error
            assert self._state is not None, "enter the reporter before reading its state"
            return self._state

    def _report(self) -> None:
        try:
            state = self._client.report_status(self._session_id, self._status())
        except QueueUnavailableError as error:
            # The Worker tolerates 30 s of silence; the next tick tries again.
            logger.bind(event="session_status").warning("session status report failed: {}", error)
            return
        except QueueError as error:
            with self._lock:
                self._error = error
            self._stop.set()
            return
        with self._lock:
            self._state = state

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._report()


class AdminClient:
    """Admin routes: policy, accounts, pause, status, and events."""

    def __init__(
        self,
        endpoint: QueueEndpoint,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api = _Api(endpoint, client, sleep)

    def close(self) -> None:
        self._api.close()

    def put_policy(self, config: PolicyConfig) -> None:
        self._api.request("PUT", "/v1/admin/policy", body=config.to_payload())

    def put_accounts(self, accounts: Sequence[Account]) -> None:
        body = [{"connect_code": a.connect_code, "r2_key": a.r2_key, "sha256": a.sha256} for a in accounts]
        self._api.request("PUT", "/v1/admin/accounts", body=body)

    def set_paused(self, paused: bool) -> None:
        self._api.request("POST", "/v1/admin/pause" if paused else "/v1/admin/resume")

    def status(self) -> dict[str, object]:
        payload = _json(self._api.request("GET", "/v1/admin/status"))
        if not isinstance(payload, dict):
            raise QueueProtocolError("admin status must be an object")
        return payload

    def events(self, *, job: str | None, session: str | None, since: float | None) -> list[dict[str, object]]:
        params = {
            name: value
            for name, value in (("job", job), ("session", session), ("since", None if since is None else str(since)))
            if value is not None
        }
        payload = _object(
            _json(self._api.request("GET", "/v1/admin/events", params=params)), frozenset(("events",)), "events"
        )
        events = payload["events"]
        if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
            raise QueueProtocolError("admin events must be a list of objects")
        return events
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_netplay_queue_client.py -q`
Expected: all pass.

- [ ] **Step 5: Type-check the new module**

Run: `uv run ty check --python-version 3.14 --error-on-warning hal/netplay_service/queue_client.py hal/netplay_service/domain.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add hal/netplay_service/queue_client.py tests/test_netplay_queue_client.py
git commit -m "Add session, heartbeat, admin, and live-settings clients"
```

---

### Task 7: Pinned assets and the verified cache

**Files:**
- Create: `hal/netplay_service/assets.py`
- Test: `tests/test_netplay_assets.py`

**Interfaces:**
- Produces: `ASSET_MANIFEST_VERSION = 1`; `class AssetError(RuntimeError)`; `sha256_file(path: Path) -> str`; `policy_bundle_key(sha256: str) -> str`; `account_key(sha256: str) -> str`; `pinned_asset_key(sha256: str, name: str) -> str`; `@dataclass(frozen=True, slots=True) class PinnedAsset(key: str, sha256: str, executable: bool = False)` with property `name`; `AssetManifest(iso: PinnedAsset, emulator: PinnedAsset)` with `read(path) -> AssetManifest`, `to_payload() -> dict[str, object]`, `write(path) -> None`; `class AssetSource(Protocol)` with `fetch(key: str, destination: Path) -> None`; `R2Source(client: Any, bucket: str)`; `LocalSource(root: Path)`; `AssetCache(root: Path, source: AssetSource)` with `get(asset: PinnedAsset) -> Path`; `ensure_uploaded(remote: Any, bucket: str, path: Path, key: str, sha256: str) -> bool`.
- `AssetSource` is a protocol because it has two implementations (R2 and a local mirror for `--local-assets`).

- [ ] **Step 1: Write the failing tests**

`tests/test_netplay_assets.py`:

```python
import hashlib
import json
from pathlib import Path
from typing import BinaryIO

import pytest
from botocore.exceptions import ClientError

from hal.netplay_service.assets import AssetCache
from hal.netplay_service.assets import AssetError
from hal.netplay_service.assets import AssetManifest
from hal.netplay_service.assets import LocalSource
from hal.netplay_service.assets import PinnedAsset
from hal.netplay_service.assets import ensure_uploaded
from hal.netplay_service.assets import pinned_asset_key


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mirror(tmp_path: Path, key: str, data: bytes) -> LocalSource:
    path = tmp_path / "mirror" / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return LocalSource(tmp_path / "mirror")


class _Bucket:
    """An in-memory stand-in for the two S3 calls `ensure_uploaded` makes."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.puts = 0

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        data, metadata = self.objects[Key]
        return {"ContentLength": len(data), "Metadata": metadata}

    def put_object(
        self, *, Bucket: str, Key: str, Body: BinaryIO, ContentType: str, Metadata: dict[str, str]
    ) -> dict[str, str]:
        self.puts += 1
        self.objects[Key] = (Body.read(), dict(Metadata))
        return {"ETag": '"etag"'}


def test_manifest_round_trips_and_rejects_drift(tmp_path: Path) -> None:
    manifest = AssetManifest(
        PinnedAsset(pinned_asset_key("a" * 64, "ssbm.ciso"), "a" * 64),
        PinnedAsset(pinned_asset_key("b" * 64, "Slippi_Online-x86_64.AppImage"), "b" * 64, executable=True),
    )
    path = tmp_path / "assets.json"
    manifest.write(path)
    assert AssetManifest.read(path) == manifest
    payload = json.loads(path.read_text())
    payload["extra"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fields changed"):
        AssetManifest.read(path)


@pytest.mark.parametrize("key", ["", "/abs", "a/../b"])
def test_pinned_asset_rejects_unsafe_keys(key: str) -> None:
    with pytest.raises(ValueError, match="key"):
        PinnedAsset(key, "a" * 64)


def test_cache_fetches_verifies_and_marks_executables(tmp_path: Path) -> None:
    data = b"emulator"
    asset = PinnedAsset(pinned_asset_key(_digest(data), "Slippi.AppImage"), _digest(data), executable=True)
    cache = AssetCache(tmp_path / "cache", _mirror(tmp_path, asset.key, data))
    path = cache.get(asset)
    assert path == tmp_path / "cache" / asset.sha256 / "Slippi.AppImage"
    assert path.read_bytes() == data
    assert path.stat().st_mode & 0o111


def test_cache_rejects_a_wrong_hash_and_leaves_nothing(tmp_path: Path) -> None:
    asset = PinnedAsset("netplay/assets/x/ssbm.ciso", "c" * 64)
    cache = AssetCache(tmp_path / "cache", _mirror(tmp_path, asset.key, b"not the iso"))
    with pytest.raises(AssetError, match="expected " + "c" * 64):
        cache.get(asset)
    assert list((tmp_path / "cache" / asset.sha256).iterdir()) == []


def test_cache_replaces_a_corrupted_cached_file(tmp_path: Path) -> None:
    data = b"iso bytes"
    asset = PinnedAsset(pinned_asset_key(_digest(data), "ssbm.ciso"), _digest(data))
    cache = AssetCache(tmp_path / "cache", _mirror(tmp_path, asset.key, data))
    cached = cache.get(asset)
    cached.write_bytes(b"iso")
    assert cache.get(asset).read_bytes() == data


def test_local_source_names_a_missing_key(tmp_path: Path) -> None:
    cache = AssetCache(tmp_path / "cache", LocalSource(tmp_path / "empty"))
    with pytest.raises(AssetError, match="local asset is missing"):
        cache.get(PinnedAsset("netplay/accounts/a.json", "a" * 64))


def test_ensure_uploaded_is_idempotent_and_refuses_a_different_object(tmp_path: Path) -> None:
    path = tmp_path / "bundle.halpolicy"
    path.write_bytes(b"bundle")
    bucket = _Bucket()
    assert ensure_uploaded(bucket, "hal", path, "netplay/policies/x.halpolicy", _digest(b"bundle"))
    assert not ensure_uploaded(bucket, "hal", path, "netplay/policies/x.halpolicy", _digest(b"bundle"))
    assert bucket.puts == 1
    with pytest.raises(AssetError, match="different hash"):
        ensure_uploaded(bucket, "hal", path, "netplay/policies/x.halpolicy", "f" * 64)


def test_ensure_uploaded_refuses_a_truncated_object_with_matching_metadata(tmp_path: Path) -> None:
    path = tmp_path / "bundle.halpolicy"
    path.write_bytes(b"bundle")
    bucket = _Bucket()
    bucket.objects["netplay/policies/x.halpolicy"] = (b"bun", {"sha256": _digest(b"bundle")})
    with pytest.raises(AssetError, match="3 bytes; expected 6"):
        ensure_uploaded(bucket, "hal", path, "netplay/policies/x.halpolicy", _digest(b"bundle"))
    assert bucket.puts == 0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_netplay_assets.py -q`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'hal.netplay_service.assets'`.

- [ ] **Step 3: Write `assets.py`**

`hal/netplay_service/assets.py`:

```python
"""Pinned runner assets: a manifest, a hash-checked cache, and content-addressed uploads."""

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Final
from typing import Protocol

from botocore.exceptions import ClientError
from loguru import logger

ASSET_MANIFEST_VERSION: Final[int] = 1
_SHA256: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")


class AssetError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def policy_bundle_key(sha256: str) -> str:
    return f"netplay/policies/{sha256}.halpolicy"


def account_key(sha256: str) -> str:
    return f"netplay/accounts/{sha256}.json"


def pinned_asset_key(sha256: str, name: str) -> str:
    return f"netplay/assets/{sha256}/{name}"


@dataclass(frozen=True, slots=True)
class PinnedAsset:
    key: str
    sha256: str
    executable: bool = False

    def __post_init__(self) -> None:
        parts = self.key.split("/")
        if not self.key or self.key.startswith("/") or any(part in ("", ".", "..") for part in parts):
            raise ValueError(f"asset key must be a relative R2 key: {self.key!r}")
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError(f"asset {self.key} SHA-256 must be lowercase hex")

    @property
    def name(self) -> str:
        return self.key.rsplit("/", 1)[-1]


def _pinned(payload: object, name: str, *, executable: bool) -> PinnedAsset:
    if not isinstance(payload, dict) or set(payload) != {"key", "sha256"}:
        raise ValueError(f"asset manifest {name} needs exactly key and sha256")
    key, digest = payload["key"], payload["sha256"]
    if not isinstance(key, str) or not isinstance(digest, str):
        raise ValueError(f"asset manifest {name} values must be strings")
    return PinnedAsset(key, digest, executable)


@dataclass(frozen=True, slots=True)
class AssetManifest:
    """The ISO and emulator every runner uses, pinned by R2 key and SHA-256."""

    iso: PinnedAsset
    emulator: PinnedAsset

    @classmethod
    def read(cls, path: Path) -> AssetManifest:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "iso", "emulator"}:
            raise ValueError(f"asset manifest fields changed: {path}")
        if payload["schema_version"] != ASSET_MANIFEST_VERSION or isinstance(payload["schema_version"], bool):
            raise ValueError(f"asset manifest schema_version must be {ASSET_MANIFEST_VERSION}: {path}")
        return cls(
            _pinned(payload["iso"], "iso", executable=False),
            _pinned(payload["emulator"], "emulator", executable=True),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": ASSET_MANIFEST_VERSION,
            "iso": {"key": self.iso.key, "sha256": self.iso.sha256},
            "emulator": {"key": self.emulator.key, "sha256": self.emulator.sha256},
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_payload(), indent=2, sort_keys=True) + "\n")


class AssetSource(Protocol):
    def fetch(self, key: str, destination: Path) -> None: ...


class R2Source:
    def __init__(self, client: Any, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def fetch(self, key: str, destination: Path) -> None:
        try:
            self._client.download_file(self._bucket, key, str(destination))
        except ClientError as error:
            raise AssetError(f"cannot download s3://{self._bucket}/{key}: {error}") from error


class LocalSource:
    """Serve R2 keys from a local mirror of the bucket layout, for development."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def fetch(self, key: str, destination: Path) -> None:
        source = self._root / key
        if not source.is_file():
            raise AssetError(f"local asset is missing: {source}")
        shutil.copyfile(source, destination)


class AssetCache:
    """Keep verified files under <root>/<sha256>/<name>; verify again on every use."""

    def __init__(self, root: Path, source: AssetSource) -> None:
        self._root = root
        self._source = source

    def get(self, asset: PinnedAsset) -> Path:
        directory = self._root / asset.sha256
        path = directory / asset.name
        mode = 0o755 if asset.executable else 0o644
        log = logger.bind(event="asset")
        if path.is_file():
            if sha256_file(path) == asset.sha256:
                path.chmod(mode)
                log.info("asset {} sha256={} verified in cache", asset.key, asset.sha256)
                return path
            log.warning("cached asset {} failed verification; fetching it again", path)
            path.unlink()
        directory.mkdir(parents=True, exist_ok=True)
        partial = directory / f".{asset.name}.{os.getpid()}.partial"
        try:
            self._source.fetch(asset.key, partial)
            actual = sha256_file(partial)
            if actual != asset.sha256:
                raise AssetError(f"{asset.key} has SHA-256 {actual}; expected {asset.sha256}")
            partial.chmod(mode)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
        log.info("asset {} sha256={} downloaded", asset.key, asset.sha256)
        return path


def ensure_uploaded(remote: Any, bucket: str, path: Path, key: str, sha256: str) -> bool:
    """Upload a content-addressed object once; an existing object must carry the same hash."""
    try:
        head = remote.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
            raise
    else:
        if head.get("Metadata", {}).get("sha256") != sha256:
            raise AssetError(f"s3://{bucket}/{key} exists with a different hash")
        # Matching metadata on a truncated object would fail every runner download.
        if head.get("ContentLength") != path.stat().st_size:
            raise AssetError(
                f"s3://{bucket}/{key} has {head.get('ContentLength')} bytes; expected {path.stat().st_size}"
            )
        return False
    with path.open("rb") as body:
        remote.put_object(
            Bucket=bucket, Key=key, Body=body, ContentType="application/octet-stream", Metadata={"sha256": sha256}
        )
    head = remote.head_object(Bucket=bucket, Key=key)
    if head.get("ContentLength") != path.stat().st_size or head.get("Metadata", {}).get("sha256") != sha256:
        raise AssetError(f"s3://{bucket}/{key} differs from {path} after upload")
    return True
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_netplay_assets.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add hal/netplay_service/assets.py tests/test_netplay_assets.py
git commit -m "Add pinned netplay assets and a verified cache"
```

---

### Task 8: Admin commands

**Files:**
- Modify: `hal/netplay_service/domain.py` (add `account_connect_code`)
- Replace: `hal/netplay_service/admin.py`
- Create: `deploy/netplay/assets.json` (written by the new `assets pin` command in Step 6)
- Modify: `deploy/netplay/README.md` (admin Access service token)
- Test: `tests/test_netplay_admin.py`, `tests/test_netplay_domain.py` (append)

**Interfaces:**
- Consumes: `AdminClient`, `Account`, `admin_endpoint` (Tasks 5–6); `AssetManifest`, `PinnedAsset`, `ensure_uploaded`, key functions, `sha256_file` (Task 7); `read_action_sequence_artifact` (`hal/inference/action_sequence_artifact.py`).
- Produces (`domain.py`): `account_connect_code(path: Path) -> str`.
- Produces (`admin.py`): `policy_config(bundle_sha256: str, vocabulary_sha256: str, capability_version: int, supported_delays: tuple[int, ...]) -> PolicyConfig`; `policy_config_for(bundle: Path) -> PolicyConfig`; `publish_policy(bundle: Path, config: PolicyConfig, admin: AdminClient, remote: Any, bucket: str) -> bool`; `upload_accounts(paths: Sequence[Path], admin: AdminClient, remote: Any, bucket: str) -> tuple[Account, ...]`; `pin_assets(iso: Path, emulator: Path, manifest: Path, remote: Any, bucket: str) -> AssetManifest`; `parse_since(value: str, now: float) -> float`; `main(argv: Sequence[str] | None = None) -> None`.
- Admin auth: `HAL_NETPLAY_ADMIN_TOKEN` plus `HAL_NETPLAY_ADMIN_ACCESS_CLIENT_ID` and `HAL_NETPLAY_ADMIN_ACCESS_CLIENT_SECRET` (see `admin_endpoint`). The spec protects admin routes with the owner's Access login, which a CLI cannot complete. The admin Access application therefore gets a second policy: Service Auth for one admin service token. The owner's browser login still works. Step 6 documents this, so the CLI works against production at the end of B1.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_netplay_domain.py`:

```python
import json
from pathlib import Path

from hal.netplay_service.domain import account_connect_code


def test_account_connect_code_is_read_without_fallback(tmp_path: Path) -> None:
    account = tmp_path / "user.json"
    account.write_text(json.dumps({"connectCode": "HAL#1", "playKey": "secret"}))
    assert account_connect_code(account) == "HAL#1"
    account.write_text(json.dumps({"connectCode": "hal#1"}))
    with pytest.raises(ValueError, match="exact uppercase"):
        account_connect_code(account)
    account.write_text("[]")
    with pytest.raises(ValueError, match="must contain an object"):
        account_connect_code(account)
```

`tests/test_netplay_admin.py`:

```python
import hashlib
import json
from pathlib import Path
from typing import BinaryIO
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

from hal.netplay_service.admin import parse_since
from hal.netplay_service.admin import pin_assets
from hal.netplay_service.admin import policy_config
from hal.netplay_service.admin import publish_policy
from hal.netplay_service.admin import upload_accounts
from hal.netplay_service.assets import AssetManifest
from hal.netplay_service.queue_client import Account
from hal.netplay_service.queue_client import AdminClient


class _Bucket:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        data, metadata = self.objects[Key]
        return {"ContentLength": len(data), "Metadata": metadata}

    def put_object(
        self, *, Bucket: str, Key: str, Body: BinaryIO, ContentType: str, Metadata: dict[str, str]
    ) -> dict[str, str]:
        self.objects[Key] = (Body.read(), dict(Metadata))
        return {"ETag": '"etag"'}


def _account(path: Path, code: str) -> Path:
    path.write_text(json.dumps({"connectCode": code, "playKey": "k"}))
    return path


def test_policy_config_requires_both_netplay_delays() -> None:
    config = policy_config("a" * 64, "b" * 64, 2, (0, 2, 3))
    assert config.bundle_r2_key == f"netplay/policies/{'a' * 64}.halpolicy"
    assert config.online_delays == (2, 3)
    assert not config.masked_identity
    assert config.imitations[0].value == "MASKED"
    with pytest.raises(ValueError, match="capability-v2"):
        policy_config("a" * 64, "b" * 64, 1, (2,))


def test_publish_policy_uploads_then_publishes(tmp_path: Path) -> None:
    bundle = tmp_path / "policy.halpolicy"
    bundle.write_bytes(b"bundle")
    digest = hashlib.sha256(b"bundle").hexdigest()
    config = policy_config(digest, "b" * 64, 2, (0, 2, 3))
    bucket, admin = _Bucket(), Mock(spec=AdminClient)
    assert publish_policy(bundle, config, admin, bucket, "hal")
    assert bucket.objects[config.bundle_r2_key][1] == {"sha256": digest}
    admin.put_policy.assert_called_once_with(config)


def test_accounts_upload_is_content_addressed(tmp_path: Path) -> None:
    first = _account(tmp_path / "a.json", "BOT0#1")
    second = _account(tmp_path / "b.json", "BOT1#1")
    bucket, admin = _Bucket(), Mock(spec=AdminClient)
    accounts = upload_accounts((first, second), admin, bucket, "hal")
    digest = hashlib.sha256(first.read_bytes()).hexdigest()
    assert accounts[0] == Account("BOT0#1", f"netplay/accounts/{digest}.json", digest)
    admin.put_accounts.assert_called_once_with(accounts)


def test_accounts_upload_refuses_duplicate_codes_before_any_upload(tmp_path: Path) -> None:
    first = _account(tmp_path / "a.json", "BOT0#1")
    second = _account(tmp_path / "b.json", "BOT0#1")
    bucket, admin = _Bucket(), Mock(spec=AdminClient)
    with pytest.raises(ValueError, match="BOT0#1 appears twice"):
        upload_accounts((first, second), admin, bucket, "hal")
    assert bucket.objects == {}
    admin.put_accounts.assert_not_called()


def test_pin_assets_uploads_and_writes_the_manifest(tmp_path: Path) -> None:
    iso = tmp_path / "ssbm.ciso"
    iso.write_bytes(b"iso")
    emulator = tmp_path / "Slippi_Online-x86_64.AppImage"
    emulator.write_bytes(b"emulator")
    manifest_path = tmp_path / "assets.json"
    manifest = pin_assets(iso, emulator, manifest_path, _Bucket(), "hal")
    assert AssetManifest.read(manifest_path) == manifest
    assert manifest.emulator.executable
    assert manifest.iso.key == f"netplay/assets/{hashlib.sha256(b'iso').hexdigest()}/ssbm.ciso"


@pytest.mark.parametrize(("value", "expected"), [("90s", 910.0), ("15m", 100.0), ("1h", -2600.0), ("2d", -171800.0)])
def test_parse_since(value: str, expected: float) -> None:
    assert parse_since(value, 1000.0) == expected


def test_parse_since_rejects_other_forms() -> None:
    with pytest.raises(ValueError, match="like 30m"):
        parse_since("yesterday", 0.0)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_netplay_admin.py tests/test_netplay_domain.py -q`
Expected: FAIL — `ImportError: cannot import name 'account_connect_code'` and `cannot import name 'parse_since'`.

- [ ] **Step 3: Add `account_connect_code` to `domain.py`**

Add `import json` to the imports, and append:

```python
def account_connect_code(path: Path) -> str:
    """Read the connect code from a Slippi user.json without falling back to defaults."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Slippi account JSON {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Slippi account JSON {path} must contain an object")
    value = payload.get("connectCode")
    if not isinstance(value, str):
        raise ValueError(f"Slippi account JSON {path} has no connectCode")
    return validate_player_code(value)
```

- [ ] **Step 4: Replace `admin.py`**

`hal/netplay_service/admin.py`:

```python
"""Administrative commands for the netplay service."""

import argparse
import json
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Final

from hal import r2
from hal.inference.action_sequence_artifact import read_action_sequence_artifact
from hal.netplay_service.assets import AssetManifest
from hal.netplay_service.assets import PinnedAsset
from hal.netplay_service.assets import account_key
from hal.netplay_service.assets import ensure_uploaded
from hal.netplay_service.assets import pinned_asset_key
from hal.netplay_service.assets import policy_bundle_key
from hal.netplay_service.assets import sha256_file
from hal.netplay_service.domain import CHARACTERS
from hal.netplay_service.domain import IMITATIONS
from hal.netplay_service.domain import STAGES
from hal.netplay_service.domain import PolicyConfig
from hal.netplay_service.domain import account_connect_code
from hal.netplay_service.queue_client import Account
from hal.netplay_service.queue_client import AdminClient
from hal.netplay_service.queue_client import admin_endpoint
from hal.netplay_service.replays import ensure_replay_lifecycle

_SINCE: Final[re.Pattern[str]] = re.compile(r"([0-9]+)([smhd])")
_UNIT_SECONDS: Final[dict[str, int]] = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def policy_config(
    bundle_sha256: str, vocabulary_sha256: str, capability_version: int, supported_delays: tuple[int, ...]
) -> PolicyConfig:
    delays = tuple(delay for delay in (2, 3) if delay in supported_delays)
    if capability_version < 2 or delays != (2, 3):
        raise ValueError("netplay needs a capability-v2 bundle that supports delays 2 and 3")
    # The roster stays the current imitation list until the page redesign adds its own.
    return PolicyConfig(
        bundle_sha256=bundle_sha256,
        bundle_r2_key=policy_bundle_key(bundle_sha256),
        vocabulary_sha256=vocabulary_sha256,
        characters=CHARACTERS,
        imitations=IMITATIONS,
        stages=STAGES,
        online_delays=delays,
        desired_return_range=(0.0, 40.0),
        default_desired_return=20.0,
        temperature_range=(0.8, 1.1),
        default_temperature=1.0,
        masked_identity=False,
    )


def policy_config_for(bundle: Path) -> PolicyConfig:
    """Validate every bundle member and derive the published config from it."""
    artifact = read_action_sequence_artifact(bundle)
    return policy_config(
        sha256_file(bundle),
        artifact.vocabulary.sha256,
        artifact.capability_version,
        artifact.spec.supported_transport_delays,
    )


def publish_policy(bundle: Path, config: PolicyConfig, admin: AdminClient, remote: Any, bucket: str) -> bool:
    uploaded = ensure_uploaded(remote, bucket, bundle, config.bundle_r2_key, config.bundle_sha256)
    admin.put_policy(config)
    return uploaded


def upload_accounts(paths: Sequence[Path], admin: AdminClient, remote: Any, bucket: str) -> tuple[Account, ...]:
    codes = [account_connect_code(path) for path in paths]
    for code in codes:
        if codes.count(code) > 1:
            raise ValueError(f"connect code {code} appears twice")
    accounts: list[Account] = []
    for path, code in zip(paths, codes, strict=True):
        digest = sha256_file(path)
        key = account_key(digest)
        ensure_uploaded(remote, bucket, path, key, digest)
        accounts.append(Account(code, key, digest))
    admin.put_accounts(accounts)
    return tuple(accounts)


def pin_assets(iso: Path, emulator: Path, manifest: Path, remote: Any, bucket: str) -> AssetManifest:
    iso_sha256 = sha256_file(iso)
    emulator_sha256 = sha256_file(emulator)
    pinned = AssetManifest(
        PinnedAsset(pinned_asset_key(iso_sha256, iso.name), iso_sha256),
        PinnedAsset(pinned_asset_key(emulator_sha256, emulator.name), emulator_sha256, executable=True),
    )
    ensure_uploaded(remote, bucket, iso, pinned.iso.key, iso_sha256)
    ensure_uploaded(remote, bucket, emulator, pinned.emulator.key, emulator_sha256)
    pinned.write(manifest)
    return pinned


def parse_since(value: str, now: float) -> float:
    match = _SINCE.fullmatch(value)
    if match is None:
        raise ValueError("--since must look like 30m, 1h, or 2d")
    return now - int(match[1]) * _UNIT_SECONDS[match[2]]


def _admin_client() -> AdminClient:
    return AdminClient(admin_endpoint(os.environ))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="hal-netplay-admin")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("install-replay-lifecycle", help="install the private replay 30-day lifecycle rule")
    publish = commands.add_parser("publish-policy", help="upload a bundle and make it the active policy")
    publish.add_argument("bundle", type=Path)
    accounts = commands.add_parser("accounts", help="manage bot Slippi accounts")
    account_commands = accounts.add_subparsers(dest="account_command", required=True)
    upload = account_commands.add_parser("upload", help="upload user.json files and replace the account list")
    upload.add_argument("paths", type=Path, nargs="+")
    assets = commands.add_parser("assets", help="manage the pinned ISO and emulator")
    asset_commands = assets.add_subparsers(dest="asset_command", required=True)
    pin = asset_commands.add_parser("pin", help="upload the ISO and emulator and write the pin file")
    pin.add_argument("--iso", type=Path, required=True)
    pin.add_argument("--emulator", type=Path, required=True)
    pin.add_argument("--manifest", type=Path, default=Path("deploy/netplay/assets.json"))
    commands.add_parser("status", help="print sessions, accounts, capacity, and the active policy")
    events = commands.add_parser("events", help="print the event timeline as JSON lines")
    scope = events.add_mutually_exclusive_group()
    scope.add_argument("--job")
    scope.add_argument("--session")
    events.add_argument("--since", help="for example 30m, 1h, or 2d")
    commands.add_parser("pause", help="stop accepting new reservations")
    commands.add_parser("resume", help="accept new reservations again")
    args = parser.parse_args(argv)

    if args.command == "install-replay-lifecycle":
        ensure_replay_lifecycle()
        return
    if args.command == "assets":
        remote = r2.client()
        try:
            pinned = pin_assets(args.iso, args.emulator, args.manifest, remote, r2.bucket())
        finally:
            remote.close()
        print(json.dumps(pinned.to_payload(), indent=2, sort_keys=True))
        return
    admin = _admin_client()
    try:
        if args.command == "publish-policy":
            config = policy_config_for(args.bundle)
            remote = r2.client()
            try:
                publish_policy(args.bundle, config, admin, remote, r2.bucket())
            finally:
                remote.close()
            print(json.dumps(config.to_payload(), indent=2, sort_keys=True))
        elif args.command == "accounts":
            remote = r2.client()
            try:
                uploaded = upload_accounts(args.paths, admin, remote, r2.bucket())
            finally:
                remote.close()
            for account in uploaded:
                print(f"{account.connect_code} {account.r2_key}")
        elif args.command == "status":
            print(json.dumps(admin.status(), indent=2, sort_keys=True))
        elif args.command == "events":
            since = None if args.since is None else parse_since(args.since, time.time())
            for event in admin.events(job=args.job, session=args.session, since=since):
                print(json.dumps(event, sort_keys=True))
        elif args.command in ("pause", "resume"):
            admin.set_paused(args.command == "pause")
        else:
            raise AssertionError(args.command)
    finally:
        admin.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_netplay_admin.py tests/test_netplay_domain.py tests/test_netplay_replays.py -q`
Expected: all pass.

- [ ] **Step 6: Document the admin service token**

In `deploy/netplay/README.md`, under "One-time Cloudflare setup", replace item 2 with:

```markdown
2. Create two Cloudflare Access applications:
   - `20xx.xyz/v1/runner/*` with a Service Auth policy and one service token per GPU box.
   - `20xx.xyz/v1/admin/*` with two policies: the owner's login, for the browser, and Service Auth for
     one `hal-netplay-admin` service token, for the CLI. Put that token in `.env` as
     `HAL_NETPLAY_ADMIN_ACCESS_CLIENT_ID` and `HAL_NETPLAY_ADMIN_ACCESS_CLIENT_SECRET`, beside
     `HAL_NETPLAY_ADMIN_TOKEN`. Runners never receive it.
```

- [ ] **Step 7: Pin the ISO and emulator**

This uploads the two files to the private bucket and writes the pin file. It needs R2 credentials and `AWS_BUCKET=hal` in the environment, and the production Slippi AppImage that `hal.paths.NETPLAY_EMULATOR_PATH` names (`~/data/emulator/slippi-3.6.4/Slippi_Online-x86_64.AppImage` by default; set `HAL_NETPLAY_EMULATOR_PATH` if it lives elsewhere).

Run:

```bash
set -a; source .env; set +a
uv run hal-netplay-admin assets pin \
  --iso ~/data/emulator/ssbm.ciso \
  --emulator "${HAL_NETPLAY_EMULATOR_PATH:-$HOME/data/emulator/slippi-3.6.4/Slippi_Online-x86_64.AppImage}"
```

Expected: it prints the manifest and writes `deploy/netplay/assets.json`. The ISO entry is
`"key": "netplay/assets/b7de482eb955c8a96b6746dfa043b69ae7bf6c7c2a09ac382b9da126faa7055c/ssbm.ciso"` with the same `sha256` (this is the hash of the ISO on the development machine). The emulator entry has the AppImage's hash. If the emulator file is missing, stop and ask for its path; do not pin a different build.

- [ ] **Step 8: Commit**

```bash
git add hal/netplay_service/domain.py hal/netplay_service/admin.py deploy/netplay/assets.json \
  deploy/netplay/README.md tests/test_netplay_admin.py tests/test_netplay_domain.py
git commit -m "Add netplay admin commands and asset pins"
```

---

### Task 9: Local Worker harness and the queue integration test

**Files:**
- Create: `hal/netplay_service/local_worker.py`
- Test: `tests/test_netplay_queue_integration.py`

The test drives the real Worker through the boundaries that unit tests cannot reach:
- a lost response for session start, claim, a transition, and session end;
- the settings WebSocket: initial settings, a pushed policy change, and `released`;
- a session that stays silent past 30 s while another session keeps reporting from `SessionReporter` with a `starting` status, as a box does while it downloads and qualifies.

**Interfaces:**
- Consumes: Plan A Worker project `web/netplay-api` (with `npm ci` done); Tasks 5–6 clients.
- Produces: `DEV_RUNNER_TOKEN = "dev-runner-token"`, `DEV_ADMIN_TOKEN = "dev-admin-token"`, `WORKER_PROJECT: Path`, `class LocalWorkerError(RuntimeError)`, `free_port() -> int`, `@contextmanager local_worker(state_dir: Path, *, port: int, startup_timeout_seconds: float = 90.0) -> Iterator[str]` (yields the base URL).
- Consumers: this task's integration test, and the qualification ports in Plan B2.

- [ ] **Step 1: Write the failing integration test**

`tests/test_netplay_queue_integration.py`:

```python
"""RemoteQueue, RunnerClient, and SessionReporter against the real queue Worker under `wrangler dev`."""

import json
import os
import time
from collections.abc import Callable
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from hal.netplay_service.domain import CHARACTERS
from hal.netplay_service.domain import IMITATIONS
from hal.netplay_service.domain import STAGES
from hal.netplay_service.domain import JobStatus
from hal.netplay_service.domain import PolicyConfig
from hal.netplay_service.health import RunnerStatus
from hal.netplay_service.health import SlotState
from hal.netplay_service.health import SlotStatus
from hal.netplay_service.health import aggregate_runner_status
from hal.netplay_service.local_worker import DEV_ADMIN_TOKEN
from hal.netplay_service.local_worker import DEV_RUNNER_TOKEN
from hal.netplay_service.local_worker import WORKER_PROJECT
from hal.netplay_service.local_worker import free_port
from hal.netplay_service.local_worker import local_worker
from hal.netplay_service.queue_client import Account
from hal.netplay_service.queue_client import AdminClient
from hal.netplay_service.queue_client import QueueEndpoint
from hal.netplay_service.queue_client import RemoteQueue
from hal.netplay_service.queue_client import RunnerClient
from hal.netplay_service.queue_client import SessionReporter
from hal.netplay_service.queue_client import StartedSession
from hal.netplay_service.queue_client import new_session_id
from hal.netplay_service.queue_client import slot_worker_id
from hal.netplay_service.queue_contract import SessionEndedError

pytestmark = pytest.mark.integration

POLICY = PolicyConfig(
    bundle_sha256="0" * 64,
    bundle_r2_key="netplay/policies/integration.halpolicy",
    vocabulary_sha256="1" * 64,
    characters=CHARACTERS,
    imitations=IMITATIONS,
    stages=STAGES,
    online_delays=(2, 3),
    desired_return_range=(0.0, 40.0),
    default_desired_return=20.0,
    temperature_range=(0.8, 1.1),
    default_temperature=1.0,
    masked_identity=False,
)
ACCOUNTS = [Account(f"BOT{index}#1", f"netplay/accounts/{index}.json", str(index) * 64) for index in range(4)]


@pytest.fixture(scope="module")
def worker_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    if not (WORKER_PROJECT / "node_modules" / ".bin" / "wrangler").is_file():
        message = f"run `npm ci` in {WORKER_PROJECT}"
        if os.environ.get("HAL_REQUIRE_INTEGRATION") == "1":
            pytest.fail(message)
        pytest.skip(message)
    with local_worker(tmp_path_factory.mktemp("worker"), port=free_port()) as url:
        yield url


@pytest.fixture
def admin(worker_url: str) -> Iterator[AdminClient]:
    client = AdminClient(QueueEndpoint(worker_url, DEV_ADMIN_TOKEN))
    client.put_policy(POLICY)
    client.put_accounts(ACCOUNTS)
    yield client
    client.close()


def _status(state: SlotState = SlotState.IDLE) -> RunnerStatus:
    now = time.time()
    slot = SlotStatus(0, state, None, None, None, None, None, 0, now)
    return aggregate_runner_status(POLICY.bundle_sha256, (slot,), now, model_inference_p95_ms=None, batch_wait_p95_ms=None)


class _LoseFirstResponse(httpx.BaseTransport):
    """Deliver each matching request, and lose the response to the first one."""

    def __init__(self, matches: Callable[[httpx.Request], bool]) -> None:
        self._inner = httpx.HTTPTransport()
        self._matches = matches
        self.lost = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._inner.handle_request(request)
        if self.lost == 0 and self._matches(request):
            self.lost += 1
            response.read()
            response.close()
            raise httpx.ReadError("response lost by the test", request=request)
        return response

    def close(self) -> None:
        self._inner.close()


def _lossy(url: str, method: str, suffix: str) -> tuple[httpx.Client, _LoseFirstResponse]:
    transport = _LoseFirstResponse(lambda r: r.method == method and r.url.path.endswith(suffix))
    return httpx.Client(base_url=url, transport=transport), transport


def _start(sessions: RunnerClient) -> StartedSession:
    return sessions.start_session(
        session_id=new_session_id(),
        host="integration",
        bundle_sha256=POLICY.bundle_sha256,
        git_sha="a" * 40,
        slots=1,
        wants_stream=False,
    )


def _create(url: str, player_code: str) -> tuple[str, str]:
    created = httpx.post(
        f"{url}/v1/jobs",
        json={"player_code": player_code, "character": "FOX", "imitation": "IBDW#0", "online_delay": 2},
        timeout=10,
    )
    assert created.status_code == 201, created.text
    return created.json()["id"], created.json()["token"]


def _player_job(url: str, job_id: str, token: str) -> dict[str, object]:
    response = httpx.get(f"{url}/v1/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"}, timeout=10)
    return response.json()


def _live_sessions(admin: AdminClient) -> list[dict[str, object]]:
    sessions = admin.status()["sessions"]
    assert isinstance(sessions, list)
    return [row for row in sessions if row["ended_at"] is None]


def test_lost_responses_never_duplicate_sessions_jobs_or_games(worker_url: str, admin: AdminClient) -> None:
    endpoint = QueueEndpoint(worker_url, DEV_RUNNER_TOKEN)
    sleeps: list[float] = []

    start_client, start_loss = _lossy(worker_url, "POST", "/v1/runner/sessions")
    started = _start(RunnerClient(endpoint, client=start_client, sleep=sleeps.append))
    assert start_loss.lost == 1
    assert [row["id"] for row in _live_sessions(admin)] == [started.session_id]
    sessions = RunnerClient(endpoint)
    sessions.report_status(started.session_id, _status())

    first_id, first_token = _create(worker_url, "CRYO#610")
    second_id, second_token = _create(worker_url, "OTHER#1")
    worker = slot_worker_id(started.session_id, 0)
    claim_client, claim_loss = _lossy(worker_url, "POST", "/claim")
    claimed = RemoteQueue(endpoint, started.session_id, client=claim_client, sleep=sleeps.append).claim_next(worker)
    assert claim_loss.lost == 1
    assert claimed is not None and (claimed.id, claimed.attempt) == (first_id, 1)
    assert _player_job(worker_url, second_id, second_token)["status"] == "queued"

    queue = RemoteQueue(endpoint, started.session_id)
    queue.mark_connecting(first_id, worker, started.accounts[0].connect_code)
    playing_client, playing_loss = _lossy(worker_url, "POST", "/playing")
    RemoteQueue(endpoint, started.session_id, client=playing_client, sleep=sleeps.append).mark_playing(first_id, worker)
    assert playing_loss.lost == 1
    finish = {"game_number": 1, "actual_stage": "BATTLEFIELD", "result": "win"}
    assert queue.finish_game(first_id, worker, **finish) is JobStatus.REMATCH_WAIT
    assert queue.finish_game(first_id, worker, **finish) is JobStatus.REMATCH_WAIT
    assert queue.get_worker_job(first_id, worker).game_count == 1

    end_client, end_loss = _lossy(worker_url, "DELETE", f"/v1/runner/sessions/{started.session_id}")
    assert RunnerClient(endpoint, client=end_client, sleep=sleeps.append).end_session(started.session_id) == 1
    assert end_loss.lost == 1
    assert sleeps == [0.25, 0.25, 0.25, 0.25]
    first = _player_job(worker_url, first_id, first_token)
    assert (first["status"], first["error_code"]) == ("failed", "service_generation_aborted")
    httpx.delete(f"{worker_url}/v1/jobs/{second_id}", headers={"Authorization": f"Bearer {second_token}"}, timeout=10)


def test_live_socket_pushes_settings_and_release(worker_url: str, admin: AdminClient) -> None:
    endpoint = QueueEndpoint(worker_url, DEV_RUNNER_TOKEN)
    sessions = RunnerClient(endpoint)
    started = _start(sessions)
    sessions.report_status(started.session_id, _status())
    job_id, token = _create(worker_url, "LIVE#1")
    worker = slot_worker_id(started.session_id, 0)
    queue = RemoteQueue(endpoint, started.session_id)
    assert queue.claim_next(worker) is not None
    with queue.connect_live(job_id, worker) as socket:
        assert json.loads(socket.recv(timeout=5)) == {
            "type": "settings",
            "revision": 0,
            "desired_return": 20,
            "temperature": 1,
        }
        httpx.patch(
            f"{worker_url}/v1/jobs/{job_id}/policy",
            headers={"Authorization": f"Bearer {token}"},
            json={"desired_return": 35},
            timeout=10,
        ).raise_for_status()
        assert json.loads(socket.recv(timeout=5))["revision"] == 1
        httpx.delete(f"{worker_url}/v1/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"}, timeout=10)
        assert json.loads(socket.recv(timeout=5)) == {"type": "released"}
    sessions.end_session(started.session_id)


def test_reporter_keeps_a_starting_session_alive_while_a_silent_one_ends(worker_url: str, admin: AdminClient) -> None:
    endpoint = QueueEndpoint(worker_url, DEV_RUNNER_TOKEN)
    sessions = RunnerClient(endpoint)
    starting = _start(sessions)
    silent = _start(sessions)
    sessions.report_status(silent.session_id, _status())
    with SessionReporter(sessions, starting.session_id, lambda: _status(SlotState.STARTING)) as reporter:
        deadline = time.monotonic() + 60
        while silent.session_id in {row["id"] for row in _live_sessions(admin)}:
            assert time.monotonic() < deadline, "the Worker did not end a session that stopped reporting"
            time.sleep(1)
        assert starting.session_id in {row["id"] for row in _live_sessions(admin)}
        assert reporter.state().draining is False
    with pytest.raises(SessionEndedError):
        sessions.report_status(silent.session_id, _status())
    assert sessions.end_session(starting.session_id) == 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `(cd web/netplay-api && npm ci) && HAL_REQUIRE_INTEGRATION=1 uv run pytest tests/test_netplay_queue_integration.py -m integration -q`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'hal.netplay_service.local_worker'`.

- [ ] **Step 3: Write `local_worker.py`**

`hal/netplay_service/local_worker.py`:

```python
"""Run the queue Worker under `wrangler dev` with local storage and development tokens."""

import hashlib
import os
import signal
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import httpx

DEV_RUNNER_TOKEN: Final[str] = "dev-runner-token"
DEV_ADMIN_TOKEN: Final[str] = "dev-admin-token"
WORKER_PROJECT: Final[Path] = Path(__file__).resolve().parents[2] / "web" / "netplay-api"


class LocalWorkerError(RuntimeError):
    pass


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _tail(path: Path) -> str:
    return path.read_text(errors="replace")[-4000:]


def _wait_ready(process: subprocess.Popen[bytes], url: str, log_path: Path, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise LocalWorkerError(f"wrangler dev exited with {code}:\n{_tail(log_path)}")
        try:
            if httpx.get(f"{url}/v1/capacity", timeout=1.0).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.25)
    raise LocalWorkerError(f"wrangler dev did not answer within {timeout_seconds:g} s:\n{_tail(log_path)}")


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


@contextmanager
def local_worker(state_dir: Path, *, port: int, startup_timeout_seconds: float = 90.0) -> Iterator[str]:
    """Yield the base URL of a local Worker whose storage lives under `state_dir`."""
    wrangler = WORKER_PROJECT / "node_modules" / ".bin" / "wrangler"
    if not wrangler.is_file():
        raise LocalWorkerError(f"{wrangler} is missing; run `npm ci` in {WORKER_PROJECT}")
    state_dir.mkdir(parents=True, exist_ok=True)
    log_path = state_dir / "wrangler.log"
    command = [
        str(wrangler),
        "dev",
        "--ip",
        "127.0.0.1",
        "--port",
        str(port),
        "--persist-to",
        str(state_dir / "storage"),
        "--show-interactive-dev-session=false",
        "--var",
        f"RUNNER_TOKEN_SHA256:{_digest(DEV_RUNNER_TOKEN)}",
        "--var",
        f"ADMIN_TOKEN_SHA256:{_digest(DEV_ADMIN_TOKEN)}",
    ]
    url = f"http://127.0.0.1:{port}"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=WORKER_PROJECT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_ready(process, url, log_path, startup_timeout_seconds)
            yield url
        finally:
            _stop(process)
```

- [ ] **Step 4: Run the integration test**

Run: `HAL_REQUIRE_INTEGRATION=1 uv run pytest tests/test_netplay_queue_integration.py -m integration -q`
Expected: `3 passed` in roughly 40–70 s. The silent session ends 30 s after its last report. If `wrangler dev` ignores the `--var` digests because a developer `.dev.vars` file in `web/netplay-api` sets the same names, the admin call fails with `401 admin token is invalid`; move `.dev.vars` aside and rerun, then report the precedence you observed.

- [ ] **Step 5: Commit**

```bash
git add hal/netplay_service/local_worker.py tests/test_netplay_queue_integration.py
git commit -m "Test the queue client against wrangler dev"
```

---

### Task 10: Handoff checks

**Files:** none new.

- [ ] **Step 1: Run the Worker suite**

Run: `cd web/netplay-api && npm test && npm run typecheck`
Expected: all vitest tests pass; typecheck prints nothing.

- [ ] **Step 2: Run the AGENTS handoff checks**

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check --python-version 3.14 --error-on-warning \
  hal \
  experiments/059_muon_action_sequence.py \
  scripts
uv run pytest -q -m "not integration"
HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_netplay_queue_integration.py -m integration
```

Expected: every command passes. Plan B1 does not touch replay extraction, wire format, controller input, session stepping, or offline/live parity. Task 4 changes only the runner's queue types, its lease arguments, and its replay sidecar, so the roundtrip and session-cleanup integration suite is not required here. Plan B2 runs it. Report each command, failure, and skip.

---

## Decisions carried into Plan B2

Plan B2 is not written yet. It must implement these decisions, which B1 depends on:

1. **Session start.** The runner creates the session ID once with `new_session_id()`. It passes the same ID to any repeated `start_session` call, including one after `QueueUnavailableError`. It passes `wants_stream=False` until Plan D. If `start_session` raises `QueueProtocolError` because the Worker returned a different policy bundle, the runner calls `end_session` for that session ID immediately, and then exits.
2. **Heartbeat from the start.** The runner enters `SessionReporter` immediately after `start_session`, before it downloads the bundle and accounts and before `check_realtime_budget`. Until the slots are ready, the status reports slot state `starting`. When `state()` raises `SessionEndedError`, the runner stops claiming, forfeits active games, and exits non-zero.
3. **Drain.** `SessionReporter.state().draining` replaces the local stop flag. First signal: `drain`, finish active sets, then `end_session`.
4. **Claim errors.** Any `409` on a claim stops that slot from claiming, and the slot logs the detail. The three causes are all terminal for the slot: the session is draining, the bundle is no longer active, or the slot holds a job past `leased`. Ending the session fails any held job. `SessionEndedError` (`410`) on any call ends the runner as in item 2.
5. **Client errors.** `QueueUnavailableError` during a game does not stop Dolphin. Lease grace is 60 s while playing, so the heartbeat thread logs the error and keeps trying. On any other call, it fails the slot through the existing recovery path. The retries of one call can take about 68 s in the worst case, which is longer than both leases (20 s, and 60 s while playing). If an outage lasts longer than the lease, the next transition returns `409`. The slot treats that `409` as a lost lease, not as a recovery.
6. **Replay sidecars.** Sidecar schema v2 already carries `worker_id` (`<session>/slot-<n>`). A restarted runner uploads a leftover sidecar and records it with that `worker_id`. The runner builds a `RemoteQueue` for the sidecar's session ID, which the Worker accepts after that session ended (Task 3). A `409` for the replay is logged, and the sidecar is kept for inspection. The runner never deletes a sidecar whose replay was refused.
7. **Live settings.** `_LivePolicySettings` reads `connect_live`. It reconnects with backoff and sets `released` on a `released` message, or when `get_worker_job` raises `InvalidTransitionError` after a reconnect.
8. **Deletion.** B2 deletes `QueueStore`. The runner's unit tests then need a `RunnerQueue` fake. The simplest is an in-memory port of the store's transitions, or `RemoteQueue` against `local_worker`. B2 chooses between them.
9. **Asset pins.** `deploy/netplay/assets.json` does not exist yet. Running `hal-netplay-admin assets pin` (Task 8 Step 7) is a B2 prerequisite. The owner does it.

**Note on session start errors.** A `409` from `start_session` has four causes: the bundle is not the active policy, too few bot accounts are free, the runner protocol version is not the Worker's, or the session ID exists with different settings. Only the detail text tells them apart, so B2 logs the detail.
