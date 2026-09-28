# Netplay Edge Queue — Plan A: API Worker and Durable Object

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `web/netplay-api/`, a Cloudflare Worker plus one SQLite-backed Durable Object that owns the netplay queue, runner sessions, bot-account leases, the active policy, and an event log, and prove it matches the current Python queue with golden transcripts.

**Architecture:** The Worker authenticates, rate-limits, and routes requests; each route calls one RPC method on a single Durable Object instance named `global`, which returns an `ApiResult` (`status`, `body`, `headers`). The Durable Object keeps all state in SQLite, runs each mutation in `transactionSync`, and keeps one alarm set for the earliest deadline. Golden transcripts recorded from today's FastAPI service replay against the Worker to prove parity.

**Tech Stack:** TypeScript 5.9, Cloudflare Workers (Durable Objects with SQLite storage, WebSocket hibernation, rate-limit binding), wrangler 4.x, vitest 4.1 with `@cloudflare/vitest-pool-workers` 0.22. Python 3.14 (uv) for the transcript recorder.

**Spec:** `docs/superpowers/specs/2026-09-27-netplay-edge-queue-design.md`

This is Plan A of four. Plan B moves the runner onto this API and deletes the Python service. Plan C adds host bring-up (image, `gce-up.sh`). Plan D adds Twitch streaming (stream lease, claim preference, displays, ffmpeg). Plan A adds no stream lease; Plan D extends the tables and claim logic built here.

## Global Constraints

- Paths: page at `20xx.xyz`; API route `20xx.xyz/v1/*`.
- Durable Object: one instance, addressed by name `global`.
- Lease durations: 20 s in `leased`, `connecting`, `rematch_wait`, `rematch_ready`; 60 s in `playing`.
- Session silence: a session with no status for 30 s is ended. A session is live for capacity if it reported within 5 s.
- `IDLE_TIMEOUT_SECONDS` = 600 for both connect and rematch deadlines. At most 5 games per reservation. At most 2 attempts.
- Rate limit: 5 job creations per minute per IP (`429` with `Retry-After: 60`). Queue cap: 20 waiting jobs (`503`).
- Event retention: 30 days.
- Request bodies over 16 KiB: `413`. Every response carries `Cache-Control: no-store`, `Content-Security-Policy: default-src 'none'; frame-ancestors 'none'`, `Referrer-Policy: no-referrer`, `X-Content-Type-Options: nosniff`. No CORS headers.
- Tokens: job tokens are 32 random bytes, base64url; job IDs are 18 random bytes, base64url. Only SHA-256 hex digests of job, runner, and admin tokens are stored. Runner and admin secrets are comma-separated SHA-256 hex lists (`RUNNER_TOKEN_SHA256`, `ADMIN_TOKEN_SHA256`).
- No backward compatibility: no migrations; the Durable Object schema is created fresh.
- Error bodies are `{"detail": "<message>"}`. Error messages that exist in `hal/netplay_service/queue.py` and `api.py` are copied verbatim (the transcripts pin them).
- Commit messages are short and apt, with no attribution trailer.
- Follow `AGENTS.md`: comments state a reason, constraint, or invariant; no narration.

## Review Focus

1. **A runner retries a transition whose first attempt was applied but whose response was lost.** Expected: `200` with the current job, not `409`, for `connecting`, `playing`, `no-show`, `no-contest`, `finish-game` (same `game_number`), `fail` (same `error_code`), and `forfeit`. Pinned in Task 5.
2. **A runner's clock is wrong.** Expected: session liveness uses the Durable Object's receipt time, never the payload's `updated_at`. Pinned in Task 6.
3. **A policy is republished while an old-bundle session has a game in progress.** Expected: that session's claims get `409`, but its in-progress transitions (`finish-game`, `fail`) still succeed. Pinned in Task 7.
4. **The account list is replaced while a session holds an account.** Expected: `409` naming the leased account; nothing changes. Pinned in Task 6.
5. **Malformed or oversized bodies on any route.** Expected: `400`, `413`, or `422`, never `500`. Pinned in Task 7.

---

## File Structure

```
web/netplay-api/
  package.json            scripts: test, typecheck, deploy, dev
  tsconfig.json
  wrangler.jsonc          route, Durable Object binding + migration, rate limit
  vitest.config.ts        cloudflareTest plugin, test-only bindings
  .dev.vars.example       local dev secrets (digests of dev tokens)
  src/
    env.ts                Env bindings type
    domain.ts             constants, statuses, HttpError, validators, worker IDs
    policy.ts             PolicyConfig parsing, choice checks, /v1/options body
    requests.ts           request-body parsing for every route
    store.ts              JobStore: jobs + games tables and the state machine
    sessions.ts           SessionStore: sessions + accounts, capacity
    events.ts             EventLog: events table
    queue.ts              Queue Durable Object: RPC surface, alarm, WebSocket
    http.ts               Worker routing, auth, rate limit, headers
    index.ts              exports the Worker and the Queue class
  test/
    env.d.ts              types for cloudflare:test
    helpers.ts            request helpers, fixtures
    domain.test.ts
    store.test.ts
    transitions.test.ts
    sessions.test.ts
    routes.test.ts
    live.test.ts
    transcripts.test.ts
    transcripts/          *.json recorded from the Python service, policy.json
scripts/record_netplay_transcripts.py   recorder (deleted in Plan B)
tests/test_netplay_transcripts.py       drift check (deleted in Plan B)
```

---

### Task 1: Scaffold the Worker project

**Files:**
- Create: `web/netplay-api/package.json`, `tsconfig.json`, `wrangler.jsonc`, `vitest.config.ts`, `.dev.vars.example`, `.gitignore`
- Create: `web/netplay-api/src/env.ts`, `src/index.ts`, `src/http.ts` (stub), `src/queue.ts` (stub)
- Create: `web/netplay-api/test/env.d.ts`, `test/smoke.test.ts`

**Interfaces:**
- Produces: `Env` (in `src/env.ts`); default Worker export; exported class `Queue`.

- [ ] **Step 1: Create an isolated branch**

Run from the repo root:

```bash
git worktree add ../hal-edge-queue -b netplay-edge-queue HEAD
cd ../hal-edge-queue
```

All later paths are relative to this worktree.

- [ ] **Step 2: Write the project files**

`web/netplay-api/package.json`:

```json
{
  "name": "hal-netplay-api",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "engines": { "node": ">=22.13.0" },
  "scripts": {
    "dev": "wrangler dev",
    "deploy": "wrangler deploy",
    "test": "vitest run",
    "typecheck": "tsc --noEmit"
  },
  "devDependencies": {
    "@cloudflare/vitest-pool-workers": "0.22.0",
    "@cloudflare/workers-types": "4.20260515.1",
    "typescript": "5.9.3",
    "vitest": "4.1.11",
    "wrangler": "4.92.0"
  }
}
```

`web/netplay-api/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "es2023",
    "lib": ["es2023"],
    "module": "es2022",
    "moduleResolution": "bundler",
    "types": ["@cloudflare/workers-types"],
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "resolveJsonModule": true,
    "noEmit": true,
    "skipLibCheck": true
  },
  "include": [
    "src",
    "test",
    "node_modules/@cloudflare/vitest-pool-workers/types/cloudflare-test.d.ts"
  ]
}
```

`web/netplay-api/wrangler.jsonc`:

```jsonc
{
  "name": "hal-netplay-api",
  "main": "src/index.ts",
  "compatibility_date": "2026-09-01",
  "routes": [{ "pattern": "20xx.xyz/v1/*", "zone_name": "20xx.xyz" }],
  "durable_objects": {
    "bindings": [{ "name": "QUEUE", "class_name": "Queue" }]
  },
  "migrations": [{ "tag": "v1", "new_sqlite_classes": ["Queue"] }],
  "ratelimits": [
    {
      "name": "JOB_RATE_LIMIT",
      "namespace_id": "1001",
      "simple": { "limit": 5, "period": 60 }
    }
  ],
  "observability": { "enabled": true }
}
```

`web/netplay-api/vitest.config.ts`:

```ts
import { cloudflareTest } from "@cloudflare/vitest-pool-workers";
import { defineConfig } from "vitest/config";

// Digests of "runner-test-token", "runner-other-token", and "admin-test-token".
const RUNNER_DIGESTS =
  "ba2f9b108067689cfe677aa6917d364a6dc00f98e1c0701dfdd62e7d7de257d6," +
  "165860d05bc9beff640e7acb9c414d85e8df436292b9c8e821e872288e48643e";
const ADMIN_DIGEST = "1d4f144f52846450e02414b4f60277722e181fe96d30a2392aef2a7838a6aeae";

export default defineConfig({
  plugins: [
    cloudflareTest({
      wrangler: { configPath: "./wrangler.jsonc" },
      miniflare: {
        bindings: {
          HAL_TEST_CLOCK: "1",
          RUNNER_TOKEN_SHA256: RUNNER_DIGESTS,
          ADMIN_TOKEN_SHA256: ADMIN_DIGEST,
        },
      },
    }),
  ],
});
```

`web/netplay-api/.dev.vars.example`:

```
# Copy to .dev.vars for `wrangler dev`. Values are SHA-256 hex digests of the dev tokens.
# echo -n dev-runner-token | sha256sum
RUNNER_TOKEN_SHA256=<digest of dev-runner-token>
ADMIN_TOKEN_SHA256=<digest of dev-admin-token>
```

`web/netplay-api/.gitignore`:

```
node_modules/
.wrangler/
.dev.vars
```

- [ ] **Step 3: Write the source stubs**

`web/netplay-api/src/env.ts`:

```ts
import type { Queue } from "./queue";

export interface Env {
  QUEUE: DurableObjectNamespace<Queue>;
  JOB_RATE_LIMIT: RateLimit;
  RUNNER_TOKEN_SHA256: string;
  ADMIN_TOKEN_SHA256: string;
  // Set only by vitest.config.ts; enables the controllable clock and reset.
  HAL_TEST_CLOCK?: string;
}
```

`web/netplay-api/src/queue.ts` (replaced in Task 7):

```ts
import { DurableObject } from "cloudflare:workers";
import type { Env } from "./env";

export class Queue extends DurableObject<Env> {}
```

`web/netplay-api/src/http.ts` (replaced in Task 7):

```ts
import type { Env } from "./env";

export async function handle(_request: Request, _env: Env): Promise<Response> {
  return Response.json({ detail: "not found" }, { status: 404 });
}
```

`web/netplay-api/src/index.ts`:

```ts
import type { Env } from "./env";
import { handle } from "./http";

export { Queue } from "./queue";

export default {
  fetch(request, env) {
    return handle(request, env);
  },
} satisfies ExportedHandler<Env>;
```

`web/netplay-api/test/env.d.ts`:

```ts
declare namespace Cloudflare {
  interface Env extends import("../src/env").Env {}
}
```

- [ ] **Step 4: Write the smoke test**

`web/netplay-api/test/smoke.test.ts`:

```ts
import { SELF } from "cloudflare:test";
import { expect, it } from "vitest";

it("answers unknown routes with a JSON 404", async () => {
  const response = await SELF.fetch("https://20xx.xyz/v1/nope");
  expect(response.status).toBe(404);
  expect(await response.json()).toEqual({ detail: "not found" });
});
```

- [ ] **Step 5: Install and run**

```bash
cd web/netplay-api && npm install && npm test && npm run typecheck
```

Expected: 1 test passes; `tsc` reports no errors. If `tsc` cannot find the `cloudflare:test` module, confirm the file `node_modules/@cloudflare/vitest-pool-workers/types/cloudflare-test.d.ts` exists and fix the `include` path to match the installed package; do not change package versions.

- [ ] **Step 6: Commit**

```bash
git add web/netplay-api
git commit -m "Scaffold netplay API worker"
```

---

### Task 2: Record golden transcripts from the Python service

**Files:**
- Create: `scripts/record_netplay_transcripts.py`
- Create: `tests/test_netplay_transcripts.py`
- Create: `web/netplay-api/test/transcripts/*.json` (generated)

**Interfaces:**
- Produces: transcript files. Format:

```json
{
  "name": "create_poll_cancel",
  "steps": [
    {"kind": "player", "method": "POST", "path": "/v1/jobs", "token": null,
     "body": {"player_code": "CRYO#610", "...": "..."},
     "response": {"status": 201, "body": {"id": "$job1", "token": "$token1", "...": "..."}}},
    {"kind": "worker", "worker": "a:0", "op": "claim", "job": null, "args": {},
     "compare": "full", "response": {"status": 200, "body": {"id": "$job1", "...": "..."}}},
    {"kind": "advance", "seconds": 600}
  ]
}
```

  - `worker` is `"<session alias>:<slot>"`, or a bare session alias for `end-session`.
  - `op` is one of `claim`, `heartbeat`, `connecting`, `playing`, `no-show`, `no-contest`, `finish-game`, `fail`, `forfeit`, `replay`, `get`, `end-session`.
  - `compare` is `full`, `status` (status code only), or `status_field` (status code and `body.status`). Error responses (status ≥ 400) are always compared in full.
  - `$jobN` / `$tokenN` alias values in order of first appearance in a response's top-level `id` / `token`.
  - `advance` moves the clock and then runs expiry.
  - `policy.json` holds the policy config matching today's defaults.

- [ ] **Step 1: Write the drift test**

`tests/test_netplay_transcripts.py`:

```python
import importlib.util
import json
from pathlib import Path

from hal.paths import REPO_DIR

_SPEC = importlib.util.spec_from_file_location(
    "record_netplay_transcripts", Path(REPO_DIR) / "scripts" / "record_netplay_transcripts.py"
)
assert _SPEC is not None and _SPEC.loader is not None
recorder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(recorder)

COMMITTED = Path(REPO_DIR) / "web" / "netplay-api" / "test" / "transcripts"


def test_committed_transcripts_match_the_python_service(tmp_path: Path) -> None:
    recorder.record_all(tmp_path)
    produced = {path.name: json.loads(path.read_text()) for path in tmp_path.glob("*.json")}
    committed = {path.name: json.loads(path.read_text()) for path in COMMITTED.glob("*.json")}
    assert produced == committed


def test_recording_is_deterministic(tmp_path: Path) -> None:
    recorder.record_all(tmp_path / "first")
    recorder.record_all(tmp_path / "second")
    for path in sorted((tmp_path / "first").glob("*.json")):
        assert path.read_text() == (tmp_path / "second" / path.name).read_text()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_netplay_transcripts.py -q`
Expected: FAIL (`scripts/record_netplay_transcripts.py` does not exist).

- [ ] **Step 3: Write the recorder**

`scripts/record_netplay_transcripts.py`:

```python
"""Record the Python netplay service's behavior as JSON transcripts.

The Worker port in web/netplay-api replays these transcripts and must produce
the same responses. This script and its transcripts are the parity contract
for the port; delete them with the Python service.
"""

import argparse
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from hal.netplay_service.api import ApiConfig
from hal.netplay_service.api import create_app
from hal.netplay_service.domain import CHARACTERS
from hal.netplay_service.domain import IMITATIONS
from hal.netplay_service.domain import STAGES
from hal.netplay_service.domain import Job
from hal.netplay_service.queue import InvalidTransitionError
from hal.netplay_service.queue import QueueStore
from hal.paths import REPO_DIR

START = 1_000_000.0
LEASE_SECONDS = 20.0
BOT_CODE = "HALBOT#1"
DEFAULT_OUTPUT = Path(REPO_DIR) / "web" / "netplay-api" / "test" / "transcripts"
_COMPARE = {
    "claim": "full",
    "get": "full",
    "end-session": "full",
    "finish-game": "status_field",
    "fail": "status_field",
}


def _job_body(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "player_code": job.player_code,
        "character": job.choices.character,
        "imitation": job.choices.imitation,
        "online_delay": job.choices.online_delay,
        "desired_return": job.choices.desired_return,
        "temperature": job.choices.temperature,
        "policy_revision": job.policy_revision,
        "requested_stage": job.choices.requested_stage,
        "status": job.status.value,
        "queue_position": job.queue_position,
        "attempt": job.attempt,
        "game_count": job.game_count,
        "connect_code": job.connect_code,
        "actual_stage": job.actual_stage,
        "last_result": job.last_result,
        "error_code": job.error_code,
        "connect_deadline": job.connect_deadline,
        "rematch_deadline": job.rematch_deadline,
        "cancel_after_game": job.cancel_after_game,
    }


class Recorder:
    def __init__(self, root: Path, name: str) -> None:
        self.name = name
        self.clock = START
        self.steps: list[dict[str, Any]] = []
        self.aliases: dict[str, str] = {}
        database = root / f"{name}.sqlite3"
        self.store = QueueStore(database, now=lambda: self.clock)
        # No `with` block: the app's background reaper must not run, so expiry
        # happens only at explicit `advance` steps.
        self.client = TestClient(create_app(ApiConfig(database, allowed_hosts=("testserver",)), self.store))

    def _alias(self, value: str, prefix: str) -> str:
        if value not in self.aliases:
            count = sum(alias.startswith(f"${prefix}") for alias in self.aliases.values()) + 1
            self.aliases[value] = f"${prefix}{count}"
        return self.aliases[value]

    def _normalize(self, body: Any) -> Any:
        if isinstance(body, dict):
            if isinstance(body.get("id"), str):
                self._alias(body["id"], "job")
            if isinstance(body.get("token"), str):
                self._alias(body["token"], "token")
            return {key: self._normalize(value) for key, value in body.items()}
        if isinstance(body, list):
            return [self._normalize(value) for value in body]
        if isinstance(body, str):
            return self.aliases.get(body, body)
        return body

    def _real(self, text: str) -> str:
        for real, alias in sorted(self.aliases.items(), key=lambda item: -len(item[1])):
            text = text.replace(alias, real)
        return text

    def player(self, method: str, path: str, *, token: str | None = None, body: Any = None) -> Any:
        headers = {} if token is None else {"Authorization": f"Bearer {self._real(token)}"}
        response = self.client.request(method, self._real(path), headers=headers, json=body)
        payload = self._normalize(response.json() if response.content else None)
        self.steps.append(
            {
                "kind": "player",
                "method": method,
                "path": path,
                "token": token,
                "body": body,
                "response": {"status": response.status_code, "body": payload},
            }
        )
        return payload

    def worker(self, worker: str, op: str, job: str | None = None, **args: Any) -> Any:
        try:
            status, body = self._call(worker, op, None if job is None else self._real(job), args)
        except InvalidTransitionError as error:
            status, body = 409, {"detail": str(error)}
        except ValueError as error:
            status, body = 422, {"detail": str(error)}
        payload = self._normalize(body)
        self.steps.append(
            {
                "kind": "worker",
                "worker": worker,
                "op": op,
                "job": job,
                "args": args,
                "compare": _COMPARE.get(op, "status"),
                "response": {"status": status, "body": payload},
            }
        )
        return payload

    def _call(self, worker: str, op: str, job: str | None, args: dict[str, Any]) -> tuple[int, Any]:
        owner = f"worker-{worker}"
        if op == "claim":
            claimed = self.store.claim_next(owner, lease_seconds=LEASE_SECONDS)
            return (204, None) if claimed is None else (200, _job_body(claimed))
        if op == "end-session":
            owners = [f"worker-{worker}:{slot}" for slot in range(args["slots"])]
            return 200, {"failed": self.store.fail_worker_generation(owners)}
        assert job is not None
        if op == "heartbeat":
            self.store.heartbeat(job, owner, lease_seconds=LEASE_SECONDS)
        elif op == "connecting":
            self.store.mark_connecting(job, owner, args["connect_code"])
        elif op == "playing":
            self.store.mark_playing(job, owner)
        elif op == "no-show":
            self.store.mark_no_show(job, owner)
        elif op == "no-contest":
            self.store.mark_no_contest(job, owner)
        elif op == "forfeit":
            self.store.forfeit_service_failure(job, owner)
        elif op == "finish-game":
            status = self.store.finish_game(job, owner, actual_stage=args["actual_stage"], result=args["result"])
            return 200, {"status": status.value}
        elif op == "fail":
            status = self.store.fail(job, owner, args["error_code"], retryable=args["retryable"])
            return 200, {"status": status.value}
        elif op == "replay":
            self.store.record_replay(
                job,
                args["game_number"],
                key=args["key"],
                sha256=args["sha256"],
                size=args["size"],
                etag=args["etag"],
            )
        elif op == "get":
            return 200, _job_body(self.store.get_worker_job(job, owner))
        else:
            raise AssertionError(op)
        return 200, None

    def advance(self, seconds: float) -> None:
        self.clock += seconds
        self.store.reap_expired()
        self.steps.append({"kind": "advance", "seconds": seconds})

    def create(self, player_code: str = "CRYO#610", **overrides: Any) -> Any:
        body = {"player_code": player_code, "character": "FOX", "imitation": "IBDW#0", "online_delay": 2}
        body.update(overrides)
        return self.player("POST", "/v1/jobs", body=body)

    def start_game(self, worker: str, job: str) -> None:
        self.worker(worker, "claim")
        self.worker(worker, "connecting", job, connect_code=BOT_CODE)
        self.worker(worker, "playing", job)


def create_poll_cancel(r: Recorder) -> None:
    job = r.create()
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])
    r.player("DELETE", f"/v1/jobs/{job['id']}", token=job["token"])
    r.player("DELETE", f"/v1/jobs/{job['id']}", token=job["token"])


def policy_revision(r: Recorder) -> None:
    job = r.create(desired_return=None, temperature=0.9)
    path = f"/v1/jobs/{job['id']}/policy"
    r.player("PATCH", path, token=job["token"], body={"desired_return": 30})
    r.player("PATCH", path, token=job["token"], body={"temperature": 1.05})
    r.player("PATCH", path, token=job["token"], body={"desired_return": None})
    r.player("PATCH", path, token=job["token"], body={})
    r.player("PATCH", path, token=job["token"], body={"temperature": None})
    r.player("DELETE", f"/v1/jobs/{job['id']}", token=job["token"])
    r.player("PATCH", path, token=job["token"], body={"desired_return": 10})


def credentials_hidden(r: Recorder) -> None:
    job = r.create()
    r.player("GET", f"/v1/jobs/{job['id']}", token="wrong-token")
    r.player("GET", "/v1/jobs/unknown-job-id", token=job["token"])
    r.player("GET", f"/v1/jobs/{job['id']}")
    r.player("DELETE", f"/v1/jobs/{job['id']}", token="wrong-token")


def one_active_per_player(r: Recorder) -> None:
    job = r.create()
    r.create()
    r.player("DELETE", f"/v1/jobs/{job['id']}", token=job["token"])
    r.create()


def fifo_and_retry_front(r: Recorder) -> None:
    first = r.create("AAAA#1")
    second = r.create("BBBB#2")
    third = r.create("CCCC#3")
    r.worker("a:0", "claim")
    r.player("GET", f"/v1/jobs/{second['id']}", token=second["token"])
    r.player("GET", f"/v1/jobs/{third['id']}", token=third["token"])
    r.worker("a:0", "fail", first["id"], error_code="dolphin_crash", retryable=True)
    r.player("GET", f"/v1/jobs/{first['id']}", token=first["token"])
    r.player("GET", f"/v1/jobs/{second['id']}", token=second["token"])
    r.worker("a:1", "claim")
    r.worker("a:1", "fail", first["id"], error_code="dolphin_crash", retryable=True)
    r.player("GET", f"/v1/jobs/{first['id']}", token=first["token"])
    r.worker("a:1", "claim")
    r.worker("a:0", "fail", second["id"], error_code="bad_state", retryable=False)
    r.player("GET", f"/v1/jobs/{third['id']}", token=third["token"])


def validation_errors(r: Recorder) -> None:
    r.create("CR")
    r.create("cryo#610")
    r.create(character="WALUIGI")
    r.create(imitation="NOBODY#0")
    r.create(online_delay=4)
    r.create(imitation="MASKED")
    r.create(desired_return=50)
    r.create(temperature=2.0)
    r.create(extra=True)
    r.player("POST", "/v1/jobs", body={"player_code": "CRYO#610"})


def connect_deadline_no_show(r: Recorder) -> None:
    job = r.create()
    r.worker("a:0", "claim")
    r.worker("a:0", "connecting", job["id"], connect_code=BOT_CODE)
    r.advance(599)
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])
    r.advance(1)
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])
    r.worker("a:0", "heartbeat", job["id"])
    r.create()


def worker_no_show(r: Recorder) -> None:
    job = r.create()
    r.worker("a:0", "claim")
    r.worker("a:0", "connecting", job["id"], connect_code=BOT_CODE)
    r.worker("a:0", "no-show", job["id"])
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])


def full_set(r: Recorder) -> None:
    job = r.create()
    r.start_game("a:0", job["id"])
    stages = ["BATTLEFIELD", "FINAL_DESTINATION", "DREAMLAND", "POKEMON_STADIUM", "YOSHIS_STORY"]
    for number, stage in enumerate(stages, start=1):
        r.worker("a:0", "heartbeat", job["id"])
        r.worker("a:0", "finish-game", job["id"], game_number=number, actual_stage=stage, result="loss")
        r.worker(
            "a:0",
            "replay",
            job["id"],
            game_number=number,
            key=f"replays/{number}.slp",
            sha256="ab" * 32,
            size=1000 + number,
            etag=f"etag-{number}",
        )
        r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])
        if number < len(stages):
            r.player(
                "POST",
                f"/v1/jobs/{job['id']}/rematch",
                token=job["token"],
                body={"character": "FALCO", "imitation": "ZAIN#0", "stage": "BATTLEFIELD"},
            )
            r.worker("a:0", "playing", job["id"])
    r.worker("a:0", "get", job["id"])


def rematch_rules(r: Recorder) -> None:
    job = r.create(online_delay=3)
    rematch = f"/v1/jobs/{job['id']}/rematch"
    choice = {"character": "MARTH", "imitation": "MANG#0", "stage": "DREAMLAND"}
    r.player("POST", rematch, token=job["token"], body=choice)
    r.start_game("a:0", job["id"])
    r.worker("a:0", "finish-game", job["id"], game_number=1, actual_stage="BATTLEFIELD", result="win")
    r.player("POST", rematch, token=job["token"], body={**choice, "stage": "HYRULE"})
    r.player("POST", rematch, token=job["token"], body=choice)
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])
    r.worker("a:0", "playing", job["id"])
    r.worker("a:0", "finish-game", job["id"], game_number=2, actual_stage="DREAMLAND", result="tie")
    r.advance(600)
    r.player("POST", rematch, token=job["token"], body=choice)
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])


def rematch_timeout(r: Recorder) -> None:
    job = r.create()
    r.start_game("a:0", job["id"])
    r.worker("a:0", "finish-game", job["id"], game_number=1, actual_stage="BATTLEFIELD", result="win")
    r.advance(599)
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])
    r.advance(1)
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])


def cancel_during_play(r: Recorder) -> None:
    job = r.create()
    r.start_game("a:0", job["id"])
    r.player("DELETE", f"/v1/jobs/{job['id']}", token=job["token"])
    r.worker("a:0", "finish-game", job["id"], game_number=1, actual_stage="BATTLEFIELD", result="win")
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])
    queued = r.create("QUEUE#9")
    r.worker("a:0", "claim")
    r.player("DELETE", f"/v1/jobs/{queued['id']}", token=queued["token"])
    r.worker("a:0", "heartbeat", queued["id"])


def no_contest(r: Recorder) -> None:
    job = r.create()
    r.start_game("a:0", job["id"])
    r.worker("a:0", "no-contest", job["id"])
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])


def replay_recording(r: Recorder) -> None:
    job = r.create()
    r.start_game("a:0", job["id"])
    r.worker("a:0", "replay", job["id"], game_number=1, key="k", sha256="cd" * 32, size=5, etag="e")
    r.worker("a:0", "finish-game", job["id"], game_number=1, actual_stage="BATTLEFIELD", result="win")
    replay = {"game_number": 1, "key": "k", "sha256": "cd" * 32, "size": 5, "etag": "e"}
    r.worker("a:0", "replay", job["id"], **replay)
    r.worker("a:0", "replay", job["id"], **replay)
    r.worker("a:0", "replay", job["id"], **{**replay, "etag": "other"})


def service_forfeit(r: Recorder) -> None:
    job = r.create()
    r.start_game("a:0", job["id"])
    r.worker("a:0", "forfeit", job["id"])
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])
    leased = r.create("LEASE#2")
    r.worker("a:0", "claim")
    r.worker("a:0", "forfeit", leased["id"])


def end_session(r: Recorder) -> None:
    playing = r.create("PLAY#1")
    leased = r.create("LEASE#2")
    other = r.create("OTHER#3")
    done = r.create("DONE#4")
    r.start_game("a:0", playing["id"])
    r.worker("a:1", "claim")
    r.start_game("b:0", other["id"])
    r.worker("b:0", "finish-game", other["id"], game_number=1, actual_stage="BATTLEFIELD", result="win")
    r.player("DELETE", f"/v1/jobs/{done['id']}", token=done["token"])
    r.worker("a", "end-session", slots=2)
    for job in (playing, leased, other):
        r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])


def lease_expiry(r: Recorder) -> None:
    job = r.create()
    r.worker("a:0", "claim")
    r.advance(19)
    r.worker("a:0", "heartbeat", job["id"])
    r.advance(20)
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])
    r.worker("a:1", "claim")
    r.advance(20)
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])
    r.worker("a:1", "heartbeat", job["id"])


def ownership_and_state(r: Recorder) -> None:
    job = r.create()
    r.worker("a:1", "connecting", job["id"], connect_code=BOT_CODE)
    r.worker("a:0", "claim")
    r.worker("a:1", "connecting", job["id"], connect_code=BOT_CODE)
    r.worker("a:0", "playing", job["id"])
    r.worker("a:0", "connecting", job["id"], connect_code="bad code")
    r.worker("a:0", "finish-game", job["id"], game_number=1, actual_stage="BATTLEFIELD", result="win")
    r.worker("a:0", "finish-game", job["id"], game_number=1, actual_stage="HYRULE", result="win")
    r.worker("a:0", "no-contest", job["id"])
    r.worker("a:0", "fail", job["id"], error_code="", retryable=True)
    r.worker("a:1", "get", job["id"])
    r.worker("a:0", "get", job["id"])


SCENARIOS: dict[str, Callable[[Recorder], None]] = {
    function.__name__: function
    for function in (
        create_poll_cancel,
        policy_revision,
        credentials_hidden,
        one_active_per_player,
        fifo_and_retry_front,
        validation_errors,
        connect_deadline_no_show,
        worker_no_show,
        full_set,
        rematch_rules,
        rematch_timeout,
        cancel_during_play,
        no_contest,
        replay_recording,
        service_forfeit,
        end_session,
        lease_expiry,
        ownership_and_state,
    )
}


def policy_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "bundle_sha256": "0" * 64,
        "bundle_r2_key": "netplay/policies/transcripts.halpolicy",
        "vocabulary_sha256": "1" * 64,
        "characters": [{"value": choice.value, "label": choice.label} for choice in CHARACTERS],
        "imitations": [{"value": choice.value, "label": choice.label} for choice in IMITATIONS],
        "stages": [{"value": choice.value, "label": choice.label} for choice in STAGES],
        "online_delays": [2, 3],
        "desired_return_range": [0.0, 40.0],
        "default_desired_return": 20.0,
        "temperature_range": [0.8, 1.1],
        "default_temperature": 1.0,
        "masked_identity": False,
    }


def record_all(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        for name, scenario in SCENARIOS.items():
            recorder = Recorder(Path(scratch), name)
            scenario(recorder)
            transcript = {"name": name, "steps": recorder.steps}
            (output / f"{name}.json").write_text(json.dumps(transcript, indent=2, sort_keys=True) + "\n")
    (output / "policy.json").write_text(json.dumps(policy_config(), indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    record_all(args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Generate the transcripts and read them**

Run: `uv run scripts/record_netplay_transcripts.py`
Expected: 18 scenario files plus `policy.json` in `web/netplay-api/test/transcripts/`.

Read `full_set.json`, `end_session.json`, and `validation_errors.json`. Confirm that:
- every scenario creates its jobs with `201`;
- `end_session.json`'s `end-session` step reports `{"failed": 2}`;
- `validation_errors.json` records `CR` with a list `detail` and `cryo#610` with the string detail `player_code must be an exact uppercase Slippi connect code such as CRYO#610`.

If a scenario produced an unexpected `500` or exception, fix the scenario (not the service) and regenerate.

- [ ] **Step 5: Run the drift tests**

Run: `uv run pytest tests/test_netplay_transcripts.py -q`
Expected: 2 passed.

- [ ] **Step 6: Run the Python checks for the new files**

```bash
uv run ruff format scripts/record_netplay_transcripts.py tests/test_netplay_transcripts.py
uv run ruff check scripts/record_netplay_transcripts.py tests/test_netplay_transcripts.py
uv run ty check --python-version 3.14 --error-on-warning scripts
```

Expected: no diagnostics.

- [ ] **Step 7: Commit**

```bash
git add scripts/record_netplay_transcripts.py tests/test_netplay_transcripts.py web/netplay-api/test/transcripts
git commit -m "Record netplay queue golden transcripts"
```

---

### Task 3: Domain constants and policy config

**Files:**
- Create: `web/netplay-api/src/domain.ts`, `src/policy.ts`, `src/requests.ts`
- Test: `web/netplay-api/test/domain.test.ts`

**Interfaces:**
- Produces (`domain.ts`): constants `IDLE_TIMEOUT_SECONDS`, `MAX_GAMES`, `MAX_ATTEMPTS`, `LEASE_SECONDS`, `PLAYING_LEASE_SECONDS`, `SESSION_SILENCE_SECONDS`, `SESSION_LIVE_SECONDS`, `EVENT_RETENTION_SECONDS`, `QUEUE_CAP`, `MAX_BODY_BYTES`; type `JobStatus`; `TERMINAL_STATUSES`; class `HttpError(status, detail, headers?)`; `pyRepr(value)`; `validatePlayerCode(value)`; `workerId(sessionId, slot)`; `sha256Hex(text): Promise<string>`; `randomToken(bytes): string`; `sameDigest(a, b): boolean`.
- Produces (`policy.ts`): `Choice`, `PolicyConfig`, `parsePolicyConfig(raw)`, `checkChoice(list, value, name)`, `optionsBody(policy)`.
- Produces (`requests.ts`): `fields(raw, allowed, required)`, `str`, `num`, `int`, `bool`, `nullableNum` helpers; `parseCreate(raw, policy)`, `parsePolicyUpdate(raw, policy)`, `parseRematch(raw, policy)`.

- [ ] **Step 1: Write the failing tests**

`web/netplay-api/test/domain.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { HttpError, pyRepr, sameDigest, sha256Hex, validatePlayerCode, workerId } from "../src/domain";
import { checkChoice, optionsBody, parsePolicyConfig } from "../src/policy";
import { parseCreate, parsePolicyUpdate, parseRematch } from "../src/requests";
import policyJson from "./transcripts/policy.json";

const policy = parsePolicyConfig(policyJson);

function error(fn: () => unknown): HttpError {
  try {
    fn();
  } catch (caught) {
    if (caught instanceof HttpError) return caught;
    throw caught;
  }
  throw new Error("expected an HttpError");
}

describe("domain", () => {
  it("validates connect codes exactly like the Python service", () => {
    expect(validatePlayerCode("CRYO#610")).toBe("CRYO#610");
    expect(error(() => validatePlayerCode("cryo#610"))).toMatchObject({
      status: 422,
      detail: "player_code must be an exact uppercase Slippi connect code such as CRYO#610",
    });
    expect(error(() => validatePlayerCode("ABCDEFGHI#1")).status).toBe(422);
  });

  it("quotes values like Python repr", () => {
    expect(pyRepr("FOX")).toBe("'FOX'");
    expect(pyRepr("it's")).toBe(`"it's"`);
  });

  it("derives worker IDs from session and slot", () => {
    expect(workerId("s1", 0)).toBe("s1/slot-0");
  });

  it("hashes and compares digests", async () => {
    expect(await sha256Hex("runner-test-token")).toBe(
      "ba2f9b108067689cfe677aa6917d364a6dc00f98e1c0701dfdd62e7d7de257d6",
    );
    expect(sameDigest("ab", "ab")).toBe(true);
    expect(sameDigest("ab", "ac")).toBe(false);
    expect(sameDigest("ab", "abc")).toBe(false);
  });
});

describe("policy", () => {
  it("rejects unknown or missing fields", () => {
    expect(error(() => parsePolicyConfig({ ...policyJson, extra: 1 })).status).toBe(422);
    const { bundle_sha256: _, ...missing } = policyJson;
    expect(error(() => parsePolicyConfig(missing)).status).toBe(422);
    expect(error(() => parsePolicyConfig({ ...policyJson, bundle_sha256: "XYZ" })).status).toBe(422);
    expect(error(() => parsePolicyConfig({ ...policyJson, desired_return_range: [40, 0] })).status).toBe(422);
  });

  it("names unsupported choices like the Python service", () => {
    expect(error(() => checkChoice(policy.characters, "WALUIGI", "character")).detail).toBe(
      "unsupported character 'WALUIGI'",
    );
  });

  it("hides MASKED unless the policy supports it", () => {
    expect(optionsBody(policy).imitations.some((choice) => choice.value === "MASKED")).toBe(false);
    const masked = { ...policy, masked_identity: true };
    expect(optionsBody(masked).imitations.some((choice) => choice.value === "MASKED")).toBe(true);
    expect(optionsBody(policy)).toMatchObject({ max_games: 5, no_show_seconds: 600, rematch_seconds: 600 });
  });
});

describe("requests", () => {
  const base = { player_code: "CRYO#610", character: "FOX", imitation: "IBDW#0", online_delay: 2 };

  it("applies create defaults", () => {
    expect(parseCreate(base, policy)).toEqual({ ...base, desired_return: 20, temperature: 1 });
    expect(parseCreate({ ...base, desired_return: null }, policy).desired_return).toBeNull();
  });

  it("rejects malformed create bodies with 422", () => {
    for (const body of [
      null,
      [],
      { ...base, extra: 1 },
      { player_code: "CRYO#610" },
      { ...base, player_code: "CR" },
      { ...base, online_delay: 2.5 },
      { ...base, desired_return: 41 },
      { ...base, temperature: 0.5 },
      { ...base, character: 7 },
    ]) {
      expect(error(() => parseCreate(body, policy)).status).toBe(422);
    }
  });

  it("requires at least one policy field", () => {
    expect(error(() => parsePolicyUpdate({}, policy)).detail).toBe("provide desired_return or temperature");
    expect(parsePolicyUpdate({ temperature: null }, policy)).toEqual({ temperature: null });
  });

  it("requires every rematch field", () => {
    expect(error(() => parseRematch({ character: "FOX", imitation: "IBDW#0" }, policy)).status).toBe(422);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd web/netplay-api && npx vitest run test/domain.test.ts`
Expected: FAIL (modules not found).

- [ ] **Step 3: Write `src/domain.ts`**

```ts
export const IDLE_TIMEOUT_SECONDS = 600;
export const MAX_GAMES = 5;
export const MAX_ATTEMPTS = 2;
export const LEASE_SECONDS = 20;
// Dolphin keeps running locally through a short network outage, so a playing
// lease tolerates more silence than the other states.
export const PLAYING_LEASE_SECONDS = 60;
export const SESSION_SILENCE_SECONDS = 30;
export const SESSION_LIVE_SECONDS = 5;
export const EVENT_RETENTION_SECONDS = 30 * 24 * 60 * 60;
export const QUEUE_CAP = 20;
export const MAX_BODY_BYTES = 16 * 1024;

export type JobStatus =
  | "queued"
  | "leased"
  | "connecting"
  | "playing"
  | "rematch_wait"
  | "rematch_ready"
  | "complete"
  | "failed"
  | "canceled"
  | "no_show";

export const TERMINAL_STATUSES: ReadonlySet<string> = new Set(["complete", "failed", "canceled", "no_show"]);

export class HttpError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
    readonly headers: Record<string, string> = {},
  ) {
    super(detail);
  }
}

const PLAYER_CODE = /^[A-Z0-9]{1,8}#[0-9]{1,4}$/;

// Error messages quote values the way Python's repr() does; the golden
// transcripts pin those messages.
export function pyRepr(value: string): string {
  if (value.includes("'") && !value.includes('"')) return `"${value}"`;
  return `'${value.replaceAll("\\", "\\\\").replaceAll("'", "\\'")}'`;
}

export function validatePlayerCode(value: string): string {
  if (!PLAYER_CODE.test(value)) {
    throw new HttpError(422, "player_code must be an exact uppercase Slippi connect code such as CRYO#610");
  }
  return value;
}

export function workerId(sessionId: string, slot: number): string {
  return `${sessionId}/slot-${slot}`;
}

export async function sha256Hex(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

export function randomToken(bytes: number): string {
  const data = crypto.getRandomValues(new Uint8Array(bytes));
  return btoa(String.fromCharCode(...data)).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}

export function sameDigest(a: string, b: string): boolean {
  const left = new TextEncoder().encode(a);
  const right = new TextEncoder().encode(b);
  return left.byteLength === right.byteLength && crypto.subtle.timingSafeEqual(left, right);
}
```

- [ ] **Step 4: Write `src/policy.ts`**

```ts
import { HttpError, IDLE_TIMEOUT_SECONDS, MAX_GAMES, pyRepr } from "./domain";

export interface Choice {
  value: string;
  label: string;
}

export interface PolicyConfig {
  schema_version: 1;
  bundle_sha256: string;
  bundle_r2_key: string;
  vocabulary_sha256: string;
  characters: Choice[];
  imitations: Choice[];
  stages: Choice[];
  online_delays: number[];
  desired_return_range: [number, number];
  default_desired_return: number;
  temperature_range: [number, number];
  default_temperature: number;
  masked_identity: boolean;
}

const FIELDS = [
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
] as const;
const SHA256 = /^[0-9a-f]{64}$/;

function invalid(detail: string): never {
  throw new HttpError(422, `policy config: ${detail}`);
}

function choices(value: unknown, name: string): Choice[] {
  if (!Array.isArray(value) || value.length === 0) invalid(`${name} must be a non-empty list`);
  const seen = new Set<string>();
  for (const item of value) {
    if (typeof item !== "object" || item === null || Array.isArray(item)) invalid(`${name} entries must be objects`);
    const entry = item as Record<string, unknown>;
    if (Object.keys(entry).sort().join() !== "label,value") invalid(`${name} entries need exactly value and label`);
    if (typeof entry.value !== "string" || !entry.value || typeof entry.label !== "string" || !entry.label) {
      invalid(`${name} values and labels must be non-empty strings`);
    }
    if (seen.has(entry.value)) invalid(`${name} repeats ${entry.value}`);
    seen.add(entry.value);
  }
  return value as Choice[];
}

function range(value: unknown, name: string): [number, number] {
  if (
    !Array.isArray(value) ||
    value.length !== 2 ||
    !value.every((item) => typeof item === "number" && Number.isFinite(item)) ||
    value[0] >= value[1]
  ) {
    invalid(`${name} must be two increasing finite numbers`);
  }
  return value as [number, number];
}

function within(value: unknown, [low, high]: [number, number], name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < low || value > high) {
    invalid(`${name} must lie in its range`);
  }
  return value;
}

export function parsePolicyConfig(raw: unknown): PolicyConfig {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) invalid("must be an object");
  const value = raw as Record<string, unknown>;
  const keys = Object.keys(value).sort();
  if (keys.join() !== [...FIELDS].sort().join()) invalid(`fields must be exactly ${FIELDS.join(", ")}`);
  if (value.schema_version !== 1) invalid("schema_version must be 1");
  for (const name of ["bundle_sha256", "vocabulary_sha256"] as const) {
    if (typeof value[name] !== "string" || !SHA256.test(value[name] as string)) invalid(`${name} must be SHA-256 hex`);
  }
  if (typeof value.bundle_r2_key !== "string" || !value.bundle_r2_key) invalid("bundle_r2_key must be non-empty");
  choices(value.characters, "characters");
  choices(value.imitations, "imitations");
  choices(value.stages, "stages");
  const delays = value.online_delays;
  if (!Array.isArray(delays) || delays.length === 0 || !delays.every((delay) => delay === 2 || delay === 3)) {
    invalid("online_delays must be a non-empty subset of [2, 3]");
  }
  within(value.default_desired_return, range(value.desired_return_range, "desired_return_range"), "default_desired_return");
  within(value.default_temperature, range(value.temperature_range, "temperature_range"), "default_temperature");
  if (typeof value.masked_identity !== "boolean") invalid("masked_identity must be a boolean");
  return value as unknown as PolicyConfig;
}

export function checkChoice(list: Choice[], value: string, name: string): string {
  if (!list.some((choice) => choice.value === value)) throw new HttpError(422, `unsupported ${name} ${pyRepr(value)}`);
  return value;
}

export function optionsBody(policy: PolicyConfig) {
  return {
    characters: policy.characters,
    imitations: policy.masked_identity
      ? policy.imitations
      : policy.imitations.filter((choice) => choice.value !== "MASKED"),
    stages: policy.stages,
    online_delays: policy.online_delays,
    desired_return_range: policy.desired_return_range,
    default_desired_return: policy.default_desired_return,
    temperature_range: policy.temperature_range,
    default_temperature: policy.default_temperature,
    max_games: MAX_GAMES,
    no_show_seconds: IDLE_TIMEOUT_SECONDS,
    rematch_seconds: IDLE_TIMEOUT_SECONDS,
  };
}
```

- [ ] **Step 5: Write `src/requests.ts`**

```ts
import { HttpError } from "./domain";
import type { PolicyConfig } from "./policy";

export type Fields = Record<string, unknown>;

// Shape errors mirror FastAPI's request validation: status 422.
export function fields(raw: unknown, allowed: readonly string[], required: readonly string[]): Fields {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new HttpError(422, "request body must be a JSON object");
  }
  const value = raw as Fields;
  for (const key of Object.keys(value)) {
    if (!allowed.includes(key)) throw new HttpError(422, `unexpected field ${key}`);
  }
  for (const key of required) {
    if (!(key in value)) throw new HttpError(422, `missing field ${key}`);
  }
  return value;
}

export function str(value: unknown, name: string): string {
  if (typeof value !== "string") throw new HttpError(422, `${name} must be a string`);
  return value;
}

export function num(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new HttpError(422, `${name} must be a number`);
  return value;
}

export function int(value: unknown, name: string): number {
  if (!Number.isInteger(value)) throw new HttpError(422, `${name} must be an integer`);
  return value as number;
}

export function bool(value: unknown, name: string): boolean {
  if (typeof value !== "boolean") throw new HttpError(422, `${name} must be a boolean`);
  return value;
}

export function nullableNum(value: unknown, name: string): number | null {
  return value === null ? null : num(value, name);
}

function inRange(value: number, [low, high]: [number, number], name: string): number {
  if (value < low || value > high) throw new HttpError(422, `${name} must be in [${low}, ${high}]`);
  return value;
}

export interface CreateRequest {
  player_code: string;
  character: string;
  imitation: string;
  online_delay: number;
  desired_return: number | null;
  temperature: number;
}

export function parseCreate(raw: unknown, policy: PolicyConfig): CreateRequest {
  const value = fields(
    raw,
    ["player_code", "character", "imitation", "online_delay", "desired_return", "temperature"],
    ["player_code", "character", "imitation", "online_delay"],
  );
  const code = str(value.player_code, "player_code");
  if (code.length < 3 || code.length > 13) throw new HttpError(422, "player_code must have 3 to 13 characters");
  const desired = "desired_return" in value ? nullableNum(value.desired_return, "desired_return") : 20;
  return {
    player_code: code,
    character: str(value.character, "character"),
    imitation: str(value.imitation, "imitation"),
    online_delay: int(value.online_delay, "online_delay"),
    desired_return: desired === null ? null : inRange(desired, policy.desired_return_range, "desired_return"),
    temperature:
      "temperature" in value
        ? inRange(num(value.temperature, "temperature"), policy.temperature_range, "temperature")
        : 1,
  };
}

export interface PolicyUpdate {
  desired_return?: number | null;
  temperature?: number | null;
}

export function parsePolicyUpdate(raw: unknown, policy: PolicyConfig): PolicyUpdate {
  const value = fields(raw, ["desired_return", "temperature"], []);
  if (Object.keys(value).length === 0) throw new HttpError(422, "provide desired_return or temperature");
  const update: PolicyUpdate = {};
  if ("desired_return" in value) {
    const desired = nullableNum(value.desired_return, "desired_return");
    update.desired_return = desired === null ? null : inRange(desired, policy.desired_return_range, "desired_return");
  }
  if ("temperature" in value) {
    const temperature = nullableNum(value.temperature, "temperature");
    update.temperature =
      temperature === null ? null : inRange(temperature, policy.temperature_range, "temperature");
  }
  return update;
}

export interface RematchRequest {
  character: string;
  imitation: string;
  stage: string;
}

export function parseRematch(raw: unknown, _policy: PolicyConfig): RematchRequest {
  const value = fields(raw, ["character", "imitation", "stage"], ["character", "imitation", "stage"]);
  return {
    character: str(value.character, "character"),
    imitation: str(value.imitation, "imitation"),
    stage: str(value.stage, "stage"),
  };
}
```

- [ ] **Step 6: Run the tests**

Run: `npx vitest run test/domain.test.ts && npm run typecheck`
Expected: all pass; no type errors.

- [ ] **Step 7: Commit**

```bash
git add web/netplay-api/src/domain.ts web/netplay-api/src/policy.ts web/netplay-api/src/requests.ts web/netplay-api/test/domain.test.ts
git commit -m "Add netplay API domain, policy, and request parsing"
```

---

### Task 4: Job store — player operations

**Files:**
- Create: `web/netplay-api/src/store.ts`
- Create: `web/netplay-api/test/helpers.ts` (store part)
- Test: `web/netplay-api/test/store.test.ts`

**Interfaces:**
- Consumes: `HttpError`, `TERMINAL_STATUSES`, `JobStatus` from `domain.ts`.
- Produces: `JOB_SCHEMA: string`; `type Row = Record<string, SqlStorageValue>`; `interface MatchChoices`; `interface JobResponse`; `class JobStore(sql: SqlStorage, now: () => number)` with `row(id)`, `response(row)`, `createJob(id, tokenDigest, playerCode, choices)`, `authenticated(id, digest)`, `getJob(id, digest)`, `updatePolicy(id, digest, desiredReturn, temperature)`, `cancel(id, digest)`, `requestRematch(id, digest, character, imitation, stage)`, `queueDepth()`, `activeCount()`. Task 5 adds the worker methods to the same class.
- Test helper produces: `withStore(fn: (store: JobStore, clock: Clock) => void)`, where `Clock` has `now` and `advance(seconds)`.

- [ ] **Step 1: Write the test helper**

`web/netplay-api/test/helpers.ts`:

```ts
import { env, runInDurableObject } from "cloudflare:test";
import { JOB_SCHEMA, JobStore } from "../src/store";

export const START = 1_000_000;

export class Clock {
  now = START;
  advance(seconds: number): void {
    this.now += seconds;
  }
}

export function queueStub() {
  return env.QUEUE.get(env.QUEUE.idFromName("global"));
}

// Runs store-level tests inside a Durable Object so they use real SQLite storage.
export async function withStore(fn: (store: JobStore, clock: Clock, sql: SqlStorage) => void): Promise<void> {
  const stub = env.QUEUE.get(env.QUEUE.idFromName(crypto.randomUUID()));
  await runInDurableObject(stub, (_instance, state) => {
    state.storage.sql.exec(JOB_SCHEMA);
    const clock = new Clock();
    fn(new JobStore(state.storage.sql, () => clock.now), clock, state.storage.sql);
  });
}

export const CHOICES = {
  character: "FOX",
  imitation: "IBDW#0",
  online_delay: 2,
  desired_return: 20,
  temperature: 1,
};
```

- [ ] **Step 2: Write the failing tests**

`web/netplay-api/test/store.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { HttpError } from "../src/domain";
import { CHOICES, withStore } from "./helpers";

function status(fn: () => unknown): number {
  try {
    fn();
  } catch (error) {
    if (error instanceof HttpError) return error.status;
    throw error;
  }
  return 200;
}

describe("JobStore player operations", () => {
  it("creates queued jobs with FIFO positions", () =>
    withStore((store) => {
      const first = store.createJob("j1", "d1", "AAAA#1", CHOICES);
      const second = store.createJob("j2", "d2", "BBBB#2", CHOICES);
      expect(first).toMatchObject({ id: "j1", status: "queued", queue_position: 1, attempt: 0, cancel_after_game: false });
      expect(second.queue_position).toBe(2);
      expect(store.queueDepth()).toBe(2);
      expect(store.activeCount()).toBe(0);
    }));

  it("allows one active job per player", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "CRYO#610", CHOICES);
      let caught: unknown;
      try {
        store.createJob("j2", "d2", "CRYO#610", CHOICES);
      } catch (error) {
        caught = error;
      }
      expect(caught).toMatchObject({ status: 409, detail: "player CRYO#610 already has an active reservation" });
      store.cancel("j1", "d1");
      expect(store.createJob("j3", "d3", "CRYO#610", CHOICES).status).toBe("queued");
    }));

  it("hides jobs behind their token digest", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "CRYO#610", CHOICES);
      expect(status(() => store.getJob("j1", "wrong"))).toBe(404);
      expect(status(() => store.getJob("missing", "d1"))).toBe(404);
      expect(store.getJob("j1", "d1").id).toBe("j1");
    }));

  it("advances the policy revision and refuses finished jobs", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "CRYO#610", CHOICES);
      const updated = store.updatePolicy("j1", "d1", null, 0.9);
      expect(updated).toMatchObject({ desired_return: null, temperature: 0.9, policy_revision: 1 });
      store.cancel("j1", "d1");
      expect(status(() => store.updatePolicy("j1", "d1", 10, 1))).toBe(409);
    }));

  it("cancels queued jobs and returns terminal jobs unchanged", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "CRYO#610", CHOICES);
      expect(store.cancel("j1", "d1").status).toBe("canceled");
      expect(store.cancel("j1", "d1").status).toBe("canceled");
    }));

  it("refuses a rematch unless the job waits for one", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "CRYO#610", CHOICES);
      let caught: unknown;
      try {
        store.requestRematch("j1", "d1", "FALCO", "ZAIN#0", "BATTLEFIELD");
      } catch (error) {
        caught = error;
      }
      expect(caught).toMatchObject({ status: 409, detail: "job is not waiting for a rematch" });
    }));
});
```

- [ ] **Step 3: Run to verify failure**

Run: `npx vitest run test/store.test.ts`
Expected: FAIL (`../src/store` not found).

- [ ] **Step 4: Write `src/store.ts` (player operations)**

```ts
import { HttpError, type JobStatus, TERMINAL_STATUSES, sameDigest } from "./domain";

export type Row = Record<string, SqlStorageValue>;

export interface MatchChoices {
  character: string;
  imitation: string;
  online_delay: number;
  desired_return: number | null;
  temperature: number;
}

export interface JobResponse {
  id: string;
  player_code: string;
  character: string;
  imitation: string;
  online_delay: number;
  desired_return: number | null;
  temperature: number;
  policy_revision: number;
  requested_stage: string | null;
  status: JobStatus;
  queue_position: number | null;
  attempt: number;
  game_count: number;
  connect_code: string | null;
  actual_stage: string | null;
  last_result: string | null;
  error_code: string | null;
  connect_deadline: number | null;
  rematch_deadline: number | null;
  cancel_after_game: boolean;
}

const ACTIVE = "'queued','leased','connecting','playing','rematch_wait','rematch_ready'";
const IN_SERVICE = "'leased','connecting','playing','rematch_wait','rematch_ready'";
const TERMINAL = "'complete','failed','canceled','no_show'";

// Same columns and meanings as hal/netplay_service/queue.py schema v3, plus
// last_worker, which lets a retried runner call recognize its own applied change.
export const JOB_SCHEMA = `
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  token_digest TEXT NOT NULL,
  player_code TEXT NOT NULL,
  character TEXT NOT NULL,
  imitation TEXT NOT NULL,
  online_delay INTEGER NOT NULL CHECK (online_delay IN (2, 3)),
  desired_return REAL,
  temperature REAL NOT NULL,
  policy_revision INTEGER NOT NULL DEFAULT 0,
  requested_stage TEXT,
  status TEXT NOT NULL,
  queue_seq INTEGER NOT NULL,
  retry_front INTEGER NOT NULL DEFAULT 0 CHECK (retry_front IN (0, 1)),
  attempt INTEGER NOT NULL DEFAULT 0,
  game_count INTEGER NOT NULL DEFAULT 0,
  connect_code TEXT,
  actual_stage TEXT,
  last_result TEXT,
  error_code TEXT,
  connect_deadline REAL,
  rematch_deadline REAL,
  cancel_after_game INTEGER NOT NULL DEFAULT 0 CHECK (cancel_after_game IN (0, 1)),
  lease_owner TEXT,
  lease_expires_at REAL,
  last_worker TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_one_active_player ON jobs(player_code) WHERE status IN (${ACTIVE});
CREATE INDEX IF NOT EXISTS idx_jobs_queue ON jobs(retry_front DESC, queue_seq) WHERE status = 'queued';
CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs(lease_expires_at) WHERE lease_owner IS NOT NULL;
CREATE TABLE IF NOT EXISTS games (
  job_id TEXT NOT NULL REFERENCES jobs(id),
  game_number INTEGER NOT NULL,
  actual_stage TEXT NOT NULL,
  result TEXT NOT NULL,
  replay_key TEXT,
  replay_sha256 TEXT,
  replay_size INTEGER,
  replay_etag TEXT,
  created_at REAL NOT NULL,
  PRIMARY KEY (job_id, game_number)
);
`;

export class JobStore {
  constructor(
    private readonly sql: SqlStorage,
    private readonly now: () => number,
  ) {}

  protected first(query: string, ...params: SqlStorageValue[]): Row | null {
    const rows = this.sql.exec<Row>(query, ...params).toArray();
    return rows[0] ?? null;
  }

  protected exec(query: string, ...params: SqlStorageValue[]): number {
    return this.sql.exec(query, ...params).rowsWritten;
  }

  protected time(): number {
    return this.now();
  }

  row(id: string): Row | null {
    return this.first("SELECT * FROM jobs WHERE id = ?", id);
  }

  protected reload(id: string): Row {
    const row = this.row(id);
    if (row === null) throw new Error(`job ${id} vanished inside its transaction`);
    return row;
  }

  response(row: Row): JobResponse {
    const status = row.status as JobStatus;
    let position: number | null = null;
    if (status === "queued") {
      const ahead = this.first(
        `SELECT COUNT(*) AS n FROM jobs WHERE status = 'queued'
           AND (retry_front > ? OR (retry_front = ? AND queue_seq < ?))`,
        row.retry_front,
        row.retry_front,
        row.queue_seq,
      );
      position = 1 + Number(ahead?.n ?? 0);
    }
    return {
      id: row.id as string,
      player_code: row.player_code as string,
      character: row.character as string,
      imitation: row.imitation as string,
      online_delay: row.online_delay as number,
      desired_return: row.desired_return as number | null,
      temperature: row.temperature as number,
      policy_revision: row.policy_revision as number,
      requested_stage: row.requested_stage as string | null,
      status,
      queue_position: position,
      attempt: row.attempt as number,
      game_count: row.game_count as number,
      connect_code: row.connect_code as string | null,
      actual_stage: row.actual_stage as string | null,
      last_result: row.last_result as string | null,
      error_code: row.error_code as string | null,
      connect_deadline: row.connect_deadline as number | null,
      rematch_deadline: row.rematch_deadline as number | null,
      cancel_after_game: row.cancel_after_game === 1,
    };
  }

  createJob(id: string, tokenDigest: string, playerCode: string, choices: MatchChoices): JobResponse {
    const now = this.time();
    const seq = Number(this.first("SELECT COALESCE(MAX(queue_seq), 0) + 1 AS seq FROM jobs")?.seq);
    try {
      this.exec(
        `INSERT INTO jobs(id, token_digest, player_code, character, imitation, online_delay,
           desired_return, temperature, requested_stage, status, queue_seq, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 'queued', ?, ?, ?)`,
        id,
        tokenDigest,
        playerCode,
        choices.character,
        choices.imitation,
        choices.online_delay,
        choices.desired_return,
        choices.temperature,
        seq,
        now,
        now,
      );
    } catch (error) {
      if (error instanceof Error && error.message.includes("jobs.player_code")) {
        throw new HttpError(409, `player ${playerCode} already has an active reservation`);
      }
      throw error;
    }
    return this.response(this.reload(id));
  }

  authenticated(id: string, digest: string): Row {
    const row = this.row(id);
    if (row === null || !sameDigest(row.token_digest as string, digest)) throw new HttpError(404, "job not found");
    return row;
  }

  getJob(id: string, digest: string): JobResponse {
    return this.response(this.authenticated(id, digest));
  }

  updatePolicy(id: string, digest: string, desiredReturn: number | null, temperature: number): JobResponse {
    const row = this.authenticated(id, digest);
    if (TERMINAL_STATUSES.has(row.status as string)) throw new HttpError(409, "cannot update a finished reservation");
    this.exec(
      `UPDATE jobs SET desired_return = ?, temperature = ?, policy_revision = policy_revision + 1,
         updated_at = ? WHERE id = ?`,
      desiredReturn,
      temperature,
      this.time(),
      id,
    );
    return this.response(this.reload(id));
  }

  cancel(id: string, digest: string): JobResponse {
    const row = this.authenticated(id, digest);
    const status = row.status as string;
    if (TERMINAL_STATUSES.has(status)) return this.response(row);
    if (status === "playing") {
      this.exec("UPDATE jobs SET cancel_after_game = 1, updated_at = ? WHERE id = ?", this.time(), id);
    } else {
      this.exec(
        `UPDATE jobs SET status = 'canceled', lease_owner = NULL, lease_expires_at = NULL,
           connect_deadline = NULL, rematch_deadline = NULL, updated_at = ? WHERE id = ?`,
        this.time(),
        id,
      );
    }
    return this.response(this.reload(id));
  }

  // Throws inside the caller's transaction on expiry, so the completion is rolled
  // back exactly as in the Python service; the alarm completes the job.
  requestRematch(id: string, digest: string, character: string, imitation: string, stage: string): JobResponse {
    const now = this.time();
    const row = this.authenticated(id, digest);
    if (row.status !== "rematch_wait") throw new HttpError(409, "job is not waiting for a rematch");
    const deadline = row.rematch_deadline as number | null;
    if (deadline === null || deadline <= now) throw new HttpError(409, "rematch window expired");
    this.exec(
      `UPDATE jobs SET status = 'rematch_ready', character = ?, imitation = ?, requested_stage = ?,
         rematch_deadline = NULL, updated_at = ? WHERE id = ?`,
      character,
      imitation,
      stage,
      now,
      id,
    );
    return this.response(this.reload(id));
  }

  queueDepth(): number {
    return Number(this.first("SELECT COUNT(*) AS n FROM jobs WHERE status = 'queued'")?.n);
  }

  activeCount(): number {
    return Number(this.first(`SELECT COUNT(*) AS n FROM jobs WHERE status IN (${IN_SERVICE})`)?.n);
  }
}
```

- [ ] **Step 5: Run the tests**

Run: `npx vitest run test/store.test.ts && npm run typecheck`
Expected: 6 passed; no type errors.

- [ ] **Step 6: Commit**

```bash
git add web/netplay-api/src/store.ts web/netplay-api/test/helpers.ts web/netplay-api/test/store.test.ts
git commit -m "Add job store player operations"
```

---

### Task 5: Job store — runner transitions and expiry

**Files:**
- Modify: `web/netplay-api/src/store.ts` (add methods to `JobStore`)
- Test: `web/netplay-api/test/transitions.test.ts`

**Interfaces:**
- Consumes: Task 4's `JobStore`, `Row`, `TERMINAL`.
- Produces on `JobStore`: `claimNext(worker): JobResponse | null`; `heartbeat(id, worker): JobResponse`; `markConnecting(id, worker, connectCode): JobResponse`; `markPlaying(id, worker)`; `markNoShow(id, worker)`; `markNoContest(id, worker)`; `finishGame(id, worker, gameNumber, stage, result)`; `fail(id, worker, errorCode, retryable)`; `forfeit(id, worker)`; `recordReplay(id, gameNumber, key, sha256, size, etag): JobResponse`; `failWorkers(workers: string[]): string[]`; `reapExpired(): string[]`; `nextDeadline(): number | null`; `workerJob(id, worker): JobResponse`. Every transition returns the job after the change.

- [ ] **Step 1: Write the failing tests**

`web/netplay-api/test/transitions.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { HttpError } from "../src/domain";
import type { JobStore } from "../src/store";
import { CHOICES, withStore } from "./helpers";

function refused(fn: () => unknown): { status: number; detail: string } {
  try {
    fn();
  } catch (error) {
    if (error instanceof HttpError) return { status: error.status, detail: error.detail };
    throw error;
  }
  throw new Error("expected an HttpError");
}

function playing(store: JobStore, worker = "w0"): void {
  store.createJob("j1", "d1", "CRYO#610", CHOICES);
  store.claimNext(worker);
  store.markConnecting("j1", worker, "HALBOT#1");
  store.markPlaying("j1", worker);
}

describe("runner transitions", () => {
  it("claims in FIFO order with a 20 s lease", () =>
    withStore((store, clock) => {
      store.createJob("j1", "d1", "AAAA#1", CHOICES);
      store.createJob("j2", "d2", "BBBB#2", CHOICES);
      expect(store.claimNext("w0")).toMatchObject({ id: "j1", status: "leased", attempt: 1 });
      expect(store.row("j1")).toMatchObject({ lease_owner: "w0", lease_expires_at: clock.now + 20 });
      expect(store.claimNext("w1")?.id).toBe("j2");
      expect(store.claimNext("w2")).toBeNull();
    }));

  it("gives a playing lease 60 s and other states 20 s", () =>
    withStore((store, clock) => {
      playing(store);
      expect(store.row("j1")?.lease_expires_at).toBe(clock.now + 60);
      clock.advance(10);
      store.heartbeat("j1", "w0");
      expect(store.row("j1")?.lease_expires_at).toBe(clock.now + 60);
      store.finishGame("j1", "w0", 1, "BATTLEFIELD", "win");
      store.heartbeat("j1", "w0");
      expect(store.row("j1")?.lease_expires_at).toBe(clock.now + 20);
    }));

  it("does not expire a playing lease after 30 s of silence", () =>
    withStore((store, clock) => {
      playing(store);
      clock.advance(30);
      expect(store.reapExpired()).toEqual([]);
      expect(store.row("j1")?.status).toBe("playing");
      clock.advance(30);
      expect(store.reapExpired()).toEqual(["j1"]);
    }));

  it("accepts a retried transition that was already applied", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "CRYO#610", CHOICES);
      store.claimNext("w0");
      store.markConnecting("j1", "w0", "HALBOT#1");
      expect(store.markConnecting("j1", "w0", "HALBOT#1").status).toBe("connecting");
      store.markPlaying("j1", "w0");
      expect(store.markPlaying("j1", "w0").status).toBe("playing");
      store.finishGame("j1", "w0", 1, "BATTLEFIELD", "win");
      expect(store.finishGame("j1", "w0", 1, "BATTLEFIELD", "win")).toMatchObject({ status: "rematch_wait", game_count: 1 });
      expect(refused(() => store.finishGame("j1", "w0", 1, "DREAMLAND", "win")).detail).toBe(
        "game already recorded with a different result",
      );
      expect(Number(store.row("j1")?.game_count)).toBe(1);
    }));

  it("accepts retried no-show, no-contest, fail, and forfeit", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "AAAA#1", CHOICES);
      store.claimNext("w0");
      store.markConnecting("j1", "w0", "HALBOT#1");
      store.markNoShow("j1", "w0");
      expect(store.markNoShow("j1", "w0").status).toBe("no_show");

      store.createJob("j2", "d2", "BBBB#2", CHOICES);
      store.claimNext("w0");
      store.markConnecting("j2", "w0", "HALBOT#1");
      store.markPlaying("j2", "w0");
      store.markNoContest("j2", "w0");
      expect(store.markNoContest("j2", "w0").status).toBe("canceled");

      store.createJob("j3", "d3", "CCCC#3", CHOICES);
      store.claimNext("w0");
      store.fail("j3", "w0", "dolphin_crash", true);
      expect(store.fail("j3", "w0", "dolphin_crash", true).status).toBe("queued");

      store.createJob("j4", "d4", "DDDD#4", CHOICES);
      store.claimNext("w1");
      store.claimNext("w1");
      store.markConnecting("j3", "w1", "HALBOT#1");
      store.forfeit("j3", "w1");
      expect(store.forfeit("j3", "w1").error_code).toBe("service_failure_bot_forfeit");
    }));

  it("refuses transitions from other workers and wrong states", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "CRYO#610", CHOICES);
      expect(refused(() => store.markConnecting("j1", "w0", "HALBOT#1"))).toEqual({
        status: 409,
        detail: "worker does not own this job",
      });
      store.claimNext("w0");
      expect(refused(() => store.markPlaying("j1", "w0"))).toEqual({
        status: 409,
        detail: "job status 'leased' is not valid for this operation",
      });
      expect(refused(() => store.heartbeat("j1", "w1"))).toEqual({
        status: 409,
        detail: "worker does not own an active job lease",
      });
      expect(refused(() => store.finishGame("j1", "w0", 1, "BATTLEFIELD", "")).status).toBe(422);
      expect(refused(() => store.fail("j1", "w0", "", true)).status).toBe(422);
    }));

  it("requires game numbers in order", () =>
    withStore((store) => {
      playing(store);
      expect(refused(() => store.finishGame("j1", "w0", 2, "BATTLEFIELD", "win")).detail).toBe(
        "game_number 2 does not follow game 0",
      );
    }));

  it("completes after five games or after a cancel during play", () =>
    withStore((store) => {
      playing(store);
      for (let number = 1; number <= 5; number += 1) {
        const job = store.finishGame("j1", "w0", number, "BATTLEFIELD", "loss");
        if (number < 5) {
          store.requestRematch("j1", "d1", "FOX", "IBDW#0", "BATTLEFIELD");
          store.markPlaying("j1", "w0");
        } else {
          expect(job.status).toBe("complete");
        }
      }
    }));

  it("retries a failed job once at the front of the queue", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "AAAA#1", CHOICES);
      store.createJob("j2", "d2", "BBBB#2", CHOICES);
      store.claimNext("w0");
      expect(store.fail("j1", "w0", "dolphin_crash", true)).toMatchObject({ status: "queued", queue_position: 1 });
      store.claimNext("w0");
      expect(store.fail("j1", "w0", "dolphin_crash", true).status).toBe("failed");
    }));

  it("fails only the named workers' leases", () =>
    withStore((store) => {
      playing(store, "a/slot-0");
      store.createJob("j2", "d2", "BBBB#2", CHOICES);
      store.claimNext("b/slot-0");
      expect(store.failWorkers(["a/slot-0", "a/slot-1"])).toEqual(["j1"]);
      expect(store.row("j1")).toMatchObject({ status: "failed", error_code: "service_failure_bot_forfeit", last_result: "win" });
      expect(store.row("j2")?.status).toBe("leased");
    }));

  it("records a replay once per game", () =>
    withStore((store) => {
      playing(store);
      expect(refused(() => store.recordReplay("j1", 1, "k", "a".repeat(64), 5, "e")).detail).toBe("game is absent");
      store.finishGame("j1", "w0", 1, "BATTLEFIELD", "win");
      store.recordReplay("j1", 1, "k", "a".repeat(64), 5, "e");
      store.recordReplay("j1", 1, "k", "a".repeat(64), 5, "e");
      expect(refused(() => store.recordReplay("j1", 1, "k", "a".repeat(64), 5, "other")).detail).toBe(
        "game already has a different replay",
      );
    }));

  it("reports the earliest deadline", () =>
    withStore((store, clock) => {
      expect(store.nextDeadline()).toBeNull();
      store.createJob("j1", "d1", "CRYO#610", CHOICES);
      store.claimNext("w0");
      expect(store.nextDeadline()).toBe(clock.now + 20);
      store.markConnecting("j1", "w0", "HALBOT#1");
      store.heartbeat("j1", "w0");
      expect(store.nextDeadline()).toBe(clock.now + 20);
    }));
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run test/transitions.test.ts`
Expected: FAIL (`claimNext` is not a function).

- [ ] **Step 3: Add the methods to `JobStore` in `src/store.ts`**

Add these imports to the existing import line: `IDLE_TIMEOUT_SECONDS`, `LEASE_SECONDS`, `MAX_ATTEMPTS`, `MAX_GAMES`, `PLAYING_LEASE_SECONDS`. Then add inside `class JobStore`, after `activeCount()`:

```ts
  private owned(id: string, worker: string, expected: readonly JobStatus[]): Row {
    const row = this.row(id);
    if (row === null || row.lease_owner !== worker) throw new HttpError(409, "worker does not own this job");
    if (!expected.includes(row.status as JobStatus)) {
      throw new HttpError(409, `job status '${row.status}' is not valid for this operation`);
    }
    return row;
  }

  // A runner retries a call whose response it lost. When this worker already
  // applied the change, the retry returns the current job instead of a 409.
  private applied(id: string, worker: string, done: (row: Row) => boolean): JobResponse | null {
    const row = this.row(id);
    return row !== null && row.last_worker === worker && done(row) ? this.response(row) : null;
  }

  private leaseSeconds(status: string): number {
    return status === "playing" ? PLAYING_LEASE_SECONDS : LEASE_SECONDS;
  }

  workerJob(id: string, worker: string): JobResponse {
    const row = this.row(id);
    if (row === null || row.lease_owner !== worker) throw new HttpError(409, "worker does not own this job");
    return this.response(row);
  }

  claimNext(worker: string): JobResponse | null {
    const now = this.time();
    const row = this.first("SELECT id FROM jobs WHERE status = 'queued' ORDER BY retry_front DESC, queue_seq ASC LIMIT 1");
    if (row === null) return null;
    this.exec(
      `UPDATE jobs SET status = 'leased', retry_front = 0, attempt = attempt + 1, lease_owner = ?,
         last_worker = ?, lease_expires_at = ?, updated_at = ? WHERE id = ?`,
      worker,
      worker,
      now + LEASE_SECONDS,
      now,
      row.id,
    );
    return this.response(this.reload(row.id as string));
  }

  heartbeat(id: string, worker: string): JobResponse {
    const now = this.time();
    const row = this.row(id);
    if (row === null || row.lease_owner !== worker || TERMINAL_STATUSES.has(row.status as string)) {
      throw new HttpError(409, "worker does not own an active job lease");
    }
    this.exec(
      "UPDATE jobs SET lease_expires_at = ?, updated_at = ? WHERE id = ?",
      now + this.leaseSeconds(row.status as string),
      now,
      id,
    );
    return this.response(this.reload(id));
  }

  markConnecting(id: string, worker: string, connectCode: string): JobResponse {
    const done = this.applied(id, worker, (row) => row.status === "connecting" && row.connect_code === connectCode);
    if (done) return done;
    this.owned(id, worker, ["leased"]);
    const now = this.time();
    this.exec(
      "UPDATE jobs SET status = 'connecting', connect_code = ?, connect_deadline = ?, updated_at = ? WHERE id = ?",
      connectCode,
      now + IDLE_TIMEOUT_SECONDS,
      now,
      id,
    );
    return this.response(this.reload(id));
  }

  markPlaying(id: string, worker: string): JobResponse {
    const done = this.applied(id, worker, (row) => row.status === "playing" && row.lease_owner === worker);
    if (done) return done;
    this.owned(id, worker, ["connecting", "rematch_ready"]);
    const now = this.time();
    this.exec(
      `UPDATE jobs SET status = 'playing', connect_deadline = NULL, rematch_deadline = NULL,
         lease_expires_at = ?, updated_at = ? WHERE id = ?`,
      now + PLAYING_LEASE_SECONDS,
      now,
      id,
    );
    return this.response(this.reload(id));
  }

  markNoShow(id: string, worker: string): JobResponse {
    const done = this.applied(id, worker, (row) => row.status === "no_show");
    if (done) return done;
    this.owned(id, worker, ["connecting"]);
    this.exec(
      `UPDATE jobs SET status = 'no_show', connect_deadline = NULL, lease_owner = NULL,
         lease_expires_at = NULL, updated_at = ? WHERE id = ?`,
      this.time(),
      id,
    );
    return this.response(this.reload(id));
  }

  markNoContest(id: string, worker: string): JobResponse {
    const done = this.applied(id, worker, (row) => row.status === "canceled" && row.lease_owner === null);
    if (done) return done;
    this.owned(id, worker, ["playing"]);
    this.exec(
      `UPDATE jobs SET status = 'canceled', connect_deadline = NULL, rematch_deadline = NULL,
         lease_owner = NULL, lease_expires_at = NULL, updated_at = ? WHERE id = ?`,
      this.time(),
      id,
    );
    return this.response(this.reload(id));
  }

  finishGame(id: string, worker: string, gameNumber: number, stage: string, result: string): JobResponse {
    if (!result) throw new HttpError(422, "game result must be non-empty");
    const recorded = this.first("SELECT actual_stage, result FROM games WHERE job_id = ? AND game_number = ?", id, gameNumber);
    if (recorded !== null && this.row(id)?.last_worker === worker) {
      if (recorded.actual_stage === stage && recorded.result === result) return this.response(this.reload(id));
      throw new HttpError(409, "game already recorded with a different result");
    }
    const row = this.owned(id, worker, ["playing"]);
    const played = row.game_count as number;
    if (gameNumber !== played + 1) throw new HttpError(409, `game_number ${gameNumber} does not follow game ${played}`);
    const now = this.time();
    const terminal = gameNumber >= MAX_GAMES || row.cancel_after_game === 1;
    this.exec(
      `UPDATE jobs SET status = ?, game_count = ?, actual_stage = ?, last_result = ?, rematch_deadline = ?,
         lease_owner = CASE WHEN ? THEN NULL ELSE lease_owner END,
         lease_expires_at = CASE WHEN ? THEN NULL ELSE lease_expires_at END, updated_at = ?
       WHERE id = ?`,
      terminal ? "complete" : "rematch_wait",
      gameNumber,
      stage,
      result,
      terminal ? null : now + IDLE_TIMEOUT_SECONDS,
      terminal ? 1 : 0,
      terminal ? 1 : 0,
      now,
      id,
    );
    this.exec(
      "INSERT INTO games(job_id, game_number, actual_stage, result, created_at) VALUES (?, ?, ?, ?, ?)",
      id,
      gameNumber,
      stage,
      result,
      now,
    );
    return this.response(this.reload(id));
  }

  fail(id: string, worker: string, errorCode: string, retryable: boolean): JobResponse {
    if (!errorCode) throw new HttpError(422, "error_code must be non-empty");
    const done = this.applied(
      id,
      worker,
      (row) => row.lease_owner === null && row.error_code === errorCode && (row.status === "queued" || row.status === "failed"),
    );
    if (done) return done;
    const row = this.owned(id, worker, ["leased", "connecting", "playing", "rematch_wait", "rematch_ready"]);
    const retry = retryable && (row.attempt as number) < MAX_ATTEMPTS;
    this.exec(
      `UPDATE jobs SET status = ?, retry_front = ?, error_code = ?, lease_owner = NULL, lease_expires_at = NULL,
         connect_deadline = NULL, rematch_deadline = NULL, updated_at = ? WHERE id = ?`,
      retry ? "queued" : "failed",
      retry ? 1 : 0,
      errorCode,
      this.time(),
      id,
    );
    return this.response(this.reload(id));
  }

  forfeit(id: string, worker: string): JobResponse {
    const done = this.applied(
      id,
      worker,
      (row) => row.status === "failed" && row.error_code === "service_failure_bot_forfeit",
    );
    if (done) return done;
    this.owned(id, worker, ["connecting", "playing"]);
    this.exec(
      `UPDATE jobs SET status = 'failed', last_result = 'win', error_code = 'service_failure_bot_forfeit',
         lease_owner = NULL, lease_expires_at = NULL, connect_deadline = NULL, rematch_deadline = NULL,
         updated_at = ? WHERE id = ?`,
      this.time(),
      id,
    );
    return this.response(this.reload(id));
  }

  recordReplay(id: string, gameNumber: number, key: string, sha256: string, size: number, etag: string): JobResponse {
    const game = this.first(
      "SELECT replay_key, replay_sha256, replay_size, replay_etag FROM games WHERE job_id = ? AND game_number = ?",
      id,
      gameNumber,
    );
    if (game === null) throw new HttpError(409, "game is absent");
    const same =
      game.replay_key === key && game.replay_sha256 === sha256 && game.replay_size === size && game.replay_etag === etag;
    if (!same) {
      if (game.replay_key !== null) throw new HttpError(409, "game already has a different replay");
      this.exec(
        `UPDATE games SET replay_key = ?, replay_sha256 = ?, replay_size = ?, replay_etag = ?
         WHERE job_id = ? AND game_number = ? AND replay_key IS NULL`,
        key,
        sha256,
        size,
        etag,
        id,
        gameNumber,
      );
    }
    return this.response(this.reload(id));
  }

  // Returns the IDs of jobs whose leases were closed.
  failWorkers(workers: readonly string[]): string[] {
    if (workers.length === 0) return [];
    const marks = workers.map(() => "?").join(",");
    const ids = this.sql
      .exec<Row>(
        `SELECT id FROM jobs WHERE lease_owner IN (${marks})
           AND status IN ('leased','connecting','playing','rematch_wait','rematch_ready') ORDER BY queue_seq`,
        ...workers,
      )
      .toArray()
      .map((row) => row.id as string);
    this.exec(
      `UPDATE jobs SET status = 'failed',
         last_result = CASE WHEN status IN ('connecting','playing') THEN 'win' ELSE last_result END,
         error_code = CASE WHEN status IN ('connecting','playing')
           THEN 'service_failure_bot_forfeit' ELSE 'service_generation_aborted' END,
         lease_owner = NULL, lease_expires_at = NULL, connect_deadline = NULL, rematch_deadline = NULL,
         updated_at = ?
       WHERE lease_owner IN (${marks})
         AND status IN ('leased','connecting','playing','rematch_wait','rematch_ready')`,
      this.time(),
      ...workers,
    );
    return ids;
  }

  // Returns the IDs of jobs that changed.
  reapExpired(): string[] {
    const now = this.time();
    const changed: string[] = [];
    const ids = (query: string): string[] =>
      this.sql.exec<Row>(query, now).toArray().map((row) => row.id as string);
    for (const id of ids("SELECT id FROM jobs WHERE status = 'connecting' AND connect_deadline <= ?")) {
      this.exec(
        `UPDATE jobs SET status = 'no_show', lease_owner = NULL, lease_expires_at = NULL,
           connect_deadline = NULL, updated_at = ? WHERE id = ?`,
        now,
        id,
      );
      changed.push(id);
    }
    for (const id of ids("SELECT id FROM jobs WHERE status = 'rematch_wait' AND rematch_deadline <= ?")) {
      this.exec(
        `UPDATE jobs SET status = 'complete', lease_owner = NULL, lease_expires_at = NULL,
           rematch_deadline = NULL, updated_at = ? WHERE id = ?`,
        now,
        id,
      );
      changed.push(id);
    }
    const expired = this.sql
      .exec<Row>(
        `SELECT id, attempt FROM jobs WHERE lease_owner IS NOT NULL AND lease_expires_at <= ?
           AND status NOT IN (${TERMINAL})`,
        now,
      )
      .toArray();
    for (const row of expired) {
      const retry = (row.attempt as number) < MAX_ATTEMPTS;
      this.exec(
        `UPDATE jobs SET status = ?, retry_front = ?, error_code = 'lease_expired', lease_owner = NULL,
           lease_expires_at = NULL, connect_deadline = NULL, rematch_deadline = NULL, updated_at = ? WHERE id = ?`,
        retry ? "queued" : "failed",
        retry ? 1 : 0,
        now,
        row.id,
      );
      changed.push(row.id as string);
    }
    return changed;
  }

  nextDeadline(): number | null {
    const row = this.first(
      `SELECT MIN(t) AS t FROM (
         SELECT connect_deadline AS t FROM jobs WHERE status = 'connecting' AND connect_deadline IS NOT NULL
         UNION ALL SELECT rematch_deadline FROM jobs WHERE status = 'rematch_wait' AND rematch_deadline IS NOT NULL
         UNION ALL SELECT lease_expires_at FROM jobs
           WHERE lease_owner IS NOT NULL AND status NOT IN (${TERMINAL}))`,
    );
    return (row?.t as number | null) ?? null;
  }
```

- [ ] **Step 4: Run the tests**

Run: `npx vitest run test/store.test.ts test/transitions.test.ts && npm run typecheck`
Expected: all pass. If `"does not expire a playing lease after 30 s"` fails, check that `markPlaying` sets `lease_expires_at` to `now + 60`.

- [ ] **Step 5: Commit**

```bash
git add web/netplay-api/src/store.ts web/netplay-api/test/transitions.test.ts
git commit -m "Add runner transitions and expiry to job store"
```

---

### Task 6: Sessions, accounts, capacity, and events

**Files:**
- Create: `web/netplay-api/src/sessions.ts`, `src/events.ts`
- Test: `web/netplay-api/test/sessions.test.ts`

**Interfaces:**
- Consumes: `JobStore` (`failWorkers`, `queueDepth`, `activeCount`), `PolicyConfig`, `workerId`, constants.
- Produces (`sessions.ts`): `SESSION_SCHEMA`; `interface StartRequest { host; bundle_sha256; git_sha; slots; stream }`; `interface AccountGrant { slot; connect_code; r2_key; sha256 }`; `interface Account { connect_code; r2_key; sha256 }`; `interface CapacityBody` (the 14 fields of today's `/v1/capacity`); `class SessionStore(sql, jobs, now)` with `start(id, input, policy): { session_id: string; accounts: AccountGrant[] }`, `live(id): Row`, `report(id, payload): { draining: boolean }`, `claimWorker(id, slot, policy): string`, `jobWorker(id, slot): string`, `drain(id)`, `end(id, reason): string[]`, `endSilent(): string[]`, `nextDeadline(): number | null`, `capacity(): CapacityBody`, `putAccounts(raw)`, `summary()`.
- Produces (`events.ts`): `EVENT_SCHEMA`; `class EventLog(sql, now)` with `log(kind, detail: { job?: string; session?: string; [key: string]: unknown })`, `prune()`, `query({ job?, session?, since?, limit? })`.

- [ ] **Step 1: Write the failing tests**

`web/netplay-api/test/sessions.test.ts`:

```ts
import { env, runInDurableObject } from "cloudflare:test";
import { describe, expect, it } from "vitest";
import { HttpError } from "../src/domain";
import { EVENT_SCHEMA, EventLog } from "../src/events";
import { parsePolicyConfig } from "../src/policy";
import { SESSION_SCHEMA, SessionStore } from "../src/sessions";
import { JOB_SCHEMA, JobStore } from "../src/store";
import { CHOICES, Clock } from "./helpers";
import policyJson from "./transcripts/policy.json";

const policy = parsePolicyConfig(policyJson);
const ACCOUNTS = [0, 1, 2].map((index) => ({
  connect_code: `BOT${index}#1`,
  r2_key: `netplay/accounts/bot${index}.json`,
  sha256: String(index).repeat(64),
}));

function readyStatus(slots: number, healthy = slots, updatedAt = 0) {
  return {
    schema_version: 5,
    state: healthy === slots ? "ready" : healthy === 0 ? "recovering" : "degraded",
    message: healthy === slots ? "Game servers are ready." : "One game server is recovering; other slots remain available.",
    policy_sha256: policy.bundle_sha256,
    slots,
    healthy_slots: healthy,
    target_fps: 60,
    game_fps: 59.9,
    frame_interval_p95_ms: 17,
    dolphin_step_p95_ms: 2,
    policy_round_trip_p95_ms: 9,
    model_inference_p95_ms: 5,
    batch_wait_p95_ms: 0.4,
    recoveries: 0,
    updated_at: updatedAt,
    chunk_health: [],
  };
}

interface World {
  jobs: JobStore;
  sessions: SessionStore;
  events: EventLog;
  clock: Clock;
}

async function world(fn: (w: World) => void): Promise<void> {
  const stub = env.QUEUE.get(env.QUEUE.idFromName(crypto.randomUUID()));
  await runInDurableObject(stub, (_instance, state) => {
    const sql = state.storage.sql;
    sql.exec(JOB_SCHEMA);
    sql.exec(SESSION_SCHEMA);
    sql.exec(EVENT_SCHEMA);
    const clock = new Clock();
    const now = () => clock.now;
    const jobs = new JobStore(sql, now);
    const sessions = new SessionStore(sql, jobs, now);
    sessions.putAccounts(ACCOUNTS);
    fn({ jobs, sessions, events: new EventLog(sql, now), clock });
  });
}

function refused(fn: () => unknown): { status: number; detail: string } {
  try {
    fn();
  } catch (error) {
    if (error instanceof HttpError) return { status: error.status, detail: error.detail };
    throw error;
  }
  throw new Error("expected an HttpError");
}

const START = { host: "box", bundle_sha256: policy.bundle_sha256, git_sha: "abc", slots: 2, stream: true };

describe("sessions", () => {
  it("leases one account per slot and refuses mismatched bundles", () =>
    world(({ sessions }) => {
      const started = sessions.start("s1", START, policy);
      expect(started.accounts.map((grant) => [grant.slot, grant.connect_code])).toEqual([
        [0, "BOT0#1"],
        [1, "BOT1#1"],
      ]);
      expect(refused(() => sessions.start("s2", { ...START, bundle_sha256: "f".repeat(64) }, policy)).status).toBe(409);
      expect(refused(() => sessions.start("s3", START, policy))).toEqual({
        status: 409,
        detail: "1 bot accounts are free; 2 are required",
      });
      expect(refused(() => sessions.start("s4", START, null)).status).toBe(503);
    }));

  it("keeps a leased account in the account list", () =>
    world(({ sessions }) => {
      sessions.start("s1", { ...START, slots: 1 }, policy);
      expect(refused(() => sessions.putAccounts(ACCOUNTS.slice(1)))).toEqual({
        status: 409,
        detail: "account BOT0#1 is leased by a live session",
      });
      sessions.putAccounts([ACCOUNTS[0]!]);
      expect(refused(() => sessions.start("s2", { ...START, slots: 1 }, policy)).status).toBe(409);
    }));

  it("uses receipt time, not the payload clock, for liveness", () =>
    world(({ sessions, clock }) => {
      sessions.start("s1", START, policy);
      sessions.report("s1", readyStatus(2, 2, 12345));
      clock.advance(29);
      expect(sessions.endSilent()).toEqual([]);
      sessions.report("s1", readyStatus(2, 2, 12345));
      clock.advance(29);
      expect(sessions.endSilent()).toEqual([]);
      clock.advance(1);
      sessions.endSilent();
      expect(refused(() => sessions.live("s1"))).toEqual({ status: 410, detail: "session has ended" });
    }));

  it("ends a silent session, failing its leases and freeing its accounts", () =>
    world(({ jobs, sessions, clock }) => {
      sessions.start("s1", START, policy);
      jobs.createJob("j1", "d1", "CRYO#610", CHOICES);
      jobs.claimNext(sessions.claimWorker("s1", 0, policy));
      clock.advance(30);
      expect(sessions.endSilent()).toEqual(["j1"]);
      expect(jobs.row("j1")?.error_code).toBe("service_generation_aborted");
      expect(sessions.start("s2", START, policy).accounts).toHaveLength(2);
    }));

  it("refuses claims while draining or from an old bundle", () =>
    world(({ sessions }) => {
      sessions.start("s1", START, policy);
      expect(refused(() => sessions.claimWorker("s1", 2, policy)).status).toBe(422);
      const next = { ...policy, bundle_sha256: "e".repeat(64) };
      expect(refused(() => sessions.claimWorker("s1", 0, next)).status).toBe(409);
      expect(sessions.jobWorker("s1", 0)).toBe("s1/slot-0");
      sessions.drain("s1");
      expect(refused(() => sessions.claimWorker("s1", 0, policy))).toEqual({ status: 409, detail: "session is draining" });
    }));

  it("aggregates capacity from live, non-draining sessions", () =>
    world(({ sessions, clock }) => {
      expect(sessions.capacity()).toMatchObject({
        capacity: 0,
        healthy_slots: 0,
        service_status: "unavailable",
        service_message: "Game servers are unavailable. Try again shortly.",
      });
      sessions.start("s1", START, policy);
      sessions.report("s1", readyStatus(2, 1));
      expect(sessions.capacity()).toMatchObject({
        capacity: 2,
        healthy_slots: 1,
        service_status: "degraded",
        service_message: "One game server is recovering; other slots remain available.",
      });
      sessions.putAccounts([
        ...ACCOUNTS,
        { connect_code: "BOT3#1", r2_key: "k3", sha256: "3".repeat(64) },
      ]);
      sessions.start("s2", START, policy);
      sessions.report("s2", readyStatus(2));
      expect(sessions.capacity()).toMatchObject({ capacity: 4, healthy_slots: 3, service_status: "degraded" });
      clock.advance(6);
      expect(sessions.capacity().capacity).toBe(0);
    }));

  it("rejects malformed status payloads", () =>
    world(({ sessions }) => {
      sessions.start("s1", START, policy);
      expect(refused(() => sessions.report("s1", { ...readyStatus(2), slots: 3 })).status).toBe(422);
      expect(refused(() => sessions.report("s1", { ...readyStatus(2), schema_version: 4 })).status).toBe(422);
      expect(refused(() => sessions.report("s1", "nope")).status).toBe(422);
    }));
});

describe("events", () => {
  it("filters and prunes the timeline", () =>
    world(({ events, clock }) => {
      events.log("job_created", { job: "j1" });
      events.log("session_started", { session: "s1", host: "box" });
      expect(events.query({ job: "j1" }).map((event) => event.kind)).toEqual(["job_created"]);
      expect(events.query({ session: "s1" })[0]).toMatchObject({ kind: "session_started", detail: { host: "box" } });
      clock.advance(30 * 24 * 60 * 60 + 1);
      events.prune();
      expect(events.query({})).toEqual([]);
    }));
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run test/sessions.test.ts`
Expected: FAIL (modules not found).

- [ ] **Step 3: Write `src/events.ts`**

```ts
import { EVENT_RETENTION_SECONDS } from "./domain";

export const EVENT_SCHEMA = `
CREATE TABLE IF NOT EXISTS events (
  at REAL NOT NULL,
  kind TEXT NOT NULL,
  job_id TEXT,
  session_id TEXT,
  detail TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_at ON events(at);
`;

export interface EventDetail {
  job?: string;
  session?: string;
  [key: string]: unknown;
}

export interface EventRecord {
  at: number;
  kind: string;
  job_id: string | null;
  session_id: string | null;
  detail: Record<string, unknown>;
}

export class EventLog {
  constructor(
    private readonly sql: SqlStorage,
    private readonly now: () => number,
  ) {}

  log(kind: string, { job, session, ...detail }: EventDetail): void {
    this.sql.exec(
      "INSERT INTO events(at, kind, job_id, session_id, detail) VALUES (?, ?, ?, ?, ?)",
      this.now(),
      kind,
      job ?? null,
      session ?? null,
      JSON.stringify(detail),
    );
  }

  prune(): void {
    this.sql.exec("DELETE FROM events WHERE at < ?", this.now() - EVENT_RETENTION_SECONDS);
  }

  oldest(): number | null {
    const rows = this.sql.exec<{ at: number | null }>("SELECT MIN(at) AS at FROM events").toArray();
    return rows[0]?.at ?? null;
  }

  query({ job, session, since, limit = 500 }: { job?: string; session?: string; since?: number; limit?: number }): EventRecord[] {
    const rows = this.sql
      .exec<Record<string, SqlStorageValue>>(
        `SELECT at, kind, job_id, session_id, detail FROM events
         WHERE (? IS NULL OR job_id = ?) AND (? IS NULL OR session_id = ?) AND at >= ?
         ORDER BY at, rowid LIMIT ?`,
        job ?? null,
        job ?? null,
        session ?? null,
        session ?? null,
        since ?? 0,
        Math.min(limit, 5000),
      )
      .toArray();
    return rows.map((row) => ({
      at: row.at as number,
      kind: row.kind as string,
      job_id: row.job_id as string | null,
      session_id: row.session_id as string | null,
      detail: JSON.parse(row.detail as string) as Record<string, unknown>,
    }));
  }
}
```

- [ ] **Step 4: Write `src/sessions.ts`**

```ts
import { HttpError, SESSION_LIVE_SECONDS, SESSION_SILENCE_SECONDS, validatePlayerCode, workerId } from "./domain";
import type { PolicyConfig } from "./policy";
import type { JobStore, Row } from "./store";

export const SESSION_SCHEMA = `
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  host TEXT NOT NULL,
  bundle_sha256 TEXT NOT NULL,
  git_sha TEXT NOT NULL,
  slots INTEGER NOT NULL,
  wants_stream INTEGER NOT NULL,
  started_at REAL NOT NULL,
  last_seen_at REAL NOT NULL,
  draining INTEGER NOT NULL DEFAULT 0,
  status TEXT,
  ended_at REAL,
  end_reason TEXT
);
CREATE TABLE IF NOT EXISTS accounts (
  connect_code TEXT PRIMARY KEY,
  r2_key TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  session_id TEXT,
  slot INTEGER,
  leased_at REAL
);
`;

export interface StartRequest {
  host: string;
  bundle_sha256: string;
  git_sha: string;
  slots: number;
  stream: boolean;
}

export interface Account {
  connect_code: string;
  r2_key: string;
  sha256: string;
}

export interface AccountGrant extends Account {
  slot: number;
}

export interface CapacityBody {
  capacity: number;
  healthy_slots: number;
  active: number;
  queued: number;
  service_status: string;
  service_message: string;
  target_fps: number;
  game_fps: number | null;
  frame_interval_p95_ms: number | null;
  dolphin_step_p95_ms: number | null;
  policy_round_trip_p95_ms: number | null;
  model_inference_p95_ms: number | null;
  batch_wait_p95_ms: number | null;
  recoveries: number;
}

// hal/netplay_service/health.py RunnerStatus schema version.
const RUNNER_STATUS_VERSION = 5;
const RUNNER_STATES = new Set(["ready", "degraded", "recovering", "unavailable"]);
const TIMINGS = [
  "game_fps",
  "frame_interval_p95_ms",
  "dolphin_step_p95_ms",
  "policy_round_trip_p95_ms",
  "model_inference_p95_ms",
  "batch_wait_p95_ms",
] as const;
const SHA256 = /^[0-9a-f]{64}$/;

type RunnerStatus = Record<(typeof TIMINGS)[number], number | null> & {
  state: string;
  message: string;
  slots: number;
  healthy_slots: number;
  target_fps: number;
  recoveries: number;
};

function invalidStatus(detail: string): never {
  throw new HttpError(422, `runner status: ${detail}`);
}

function parseRunnerStatus(raw: unknown, slots: number): RunnerStatus {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) invalidStatus("must be an object");
  const value = raw as Record<string, unknown>;
  if (value.schema_version !== RUNNER_STATUS_VERSION) invalidStatus(`schema_version must be ${RUNNER_STATUS_VERSION}`);
  if (typeof value.state !== "string" || !RUNNER_STATES.has(value.state)) invalidStatus("unknown state");
  if (typeof value.message !== "string" || !value.message) invalidStatus("message must be non-empty");
  if (typeof value.policy_sha256 !== "string" || !SHA256.test(value.policy_sha256)) invalidStatus("bad policy_sha256");
  if (value.slots !== slots) invalidStatus(`slots must equal the session's ${slots}`);
  const healthy = value.healthy_slots;
  if (!Number.isInteger(healthy) || (healthy as number) < 0 || (healthy as number) > slots) invalidStatus("bad healthy_slots");
  if (typeof value.target_fps !== "number" || !(value.target_fps > 0)) invalidStatus("bad target_fps");
  for (const name of TIMINGS) {
    const timing = value[name];
    if (timing !== null && (typeof timing !== "number" || !Number.isFinite(timing) || timing < 0)) invalidStatus(`bad ${name}`);
  }
  if (!Number.isInteger(value.recoveries) || (value.recoveries as number) < 0) invalidStatus("bad recoveries");
  if (typeof value.updated_at !== "number") invalidStatus("bad updated_at");
  if (!Array.isArray(value.chunk_health)) invalidStatus("chunk_health must be a list");
  return value as unknown as RunnerStatus;
}

function parseAccounts(raw: unknown): Account[] {
  if (!Array.isArray(raw)) throw new HttpError(422, "accounts must be a list");
  const seen = new Set<string>();
  return raw.map((item) => {
    if (typeof item !== "object" || item === null) throw new HttpError(422, "accounts entries must be objects");
    const entry = item as Record<string, unknown>;
    if (Object.keys(entry).sort().join() !== "connect_code,r2_key,sha256") {
      throw new HttpError(422, "accounts entries need exactly connect_code, r2_key, and sha256");
    }
    const code = validatePlayerCode(String(entry.connect_code));
    if (typeof entry.r2_key !== "string" || !entry.r2_key) throw new HttpError(422, "r2_key must be non-empty");
    if (typeof entry.sha256 !== "string" || !SHA256.test(entry.sha256)) throw new HttpError(422, "sha256 must be hex");
    if (seen.has(code)) throw new HttpError(422, `account ${code} is listed twice`);
    seen.add(code);
    return { connect_code: code, r2_key: entry.r2_key, sha256: entry.sha256 };
  });
}

export class SessionStore {
  constructor(
    private readonly sql: SqlStorage,
    private readonly jobs: JobStore,
    private readonly now: () => number,
  ) {}

  private rows(query: string, ...params: SqlStorageValue[]): Row[] {
    return this.sql.exec<Row>(query, ...params).toArray();
  }

  start(id: string, input: StartRequest, policy: PolicyConfig | null): { session_id: string; accounts: AccountGrant[] } {
    if (policy === null) throw new HttpError(503, "no policy has been published");
    if (input.bundle_sha256 !== policy.bundle_sha256) {
      throw new HttpError(409, `bundle ${input.bundle_sha256} is not the active policy ${policy.bundle_sha256}`);
    }
    const free = this.rows("SELECT * FROM accounts WHERE session_id IS NULL ORDER BY connect_code");
    if (free.length < input.slots) {
      throw new HttpError(409, `${free.length} bot accounts are free; ${input.slots} are required`);
    }
    const now = this.now();
    this.sql.exec(
      `INSERT INTO sessions(id, host, bundle_sha256, git_sha, slots, wants_stream, started_at, last_seen_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      id,
      input.host,
      input.bundle_sha256,
      input.git_sha,
      input.slots,
      input.stream ? 1 : 0,
      now,
      now,
    );
    const accounts = free.slice(0, input.slots).map((row, slot) => {
      this.sql.exec(
        "UPDATE accounts SET session_id = ?, slot = ?, leased_at = ? WHERE connect_code = ?",
        id,
        slot,
        now,
        row.connect_code,
      );
      return {
        slot,
        connect_code: row.connect_code as string,
        r2_key: row.r2_key as string,
        sha256: row.sha256 as string,
      };
    });
    return { session_id: id, accounts };
  }

  live(id: string): Row {
    const row = this.rows("SELECT * FROM sessions WHERE id = ?", id)[0];
    if (row === undefined) throw new HttpError(404, "session not found");
    if (row.ended_at !== null) throw new HttpError(410, "session has ended");
    return row;
  }

  // Liveness uses the time this report arrived, never the runner's clock.
  report(id: string, raw: unknown): { draining: boolean } {
    const row = this.live(id);
    const status = parseRunnerStatus(raw, row.slots as number);
    this.sql.exec("UPDATE sessions SET status = ?, last_seen_at = ? WHERE id = ?", JSON.stringify(status), this.now(), id);
    return { draining: row.draining === 1 };
  }

  private slotWorker(row: Row, slot: number): string {
    if (!Number.isInteger(slot) || slot < 0 || slot >= (row.slots as number)) {
      throw new HttpError(422, `slot must be in [0, ${(row.slots as number) - 1}]`);
    }
    return workerId(row.id as string, slot);
  }

  claimWorker(id: string, slot: number, policy: PolicyConfig | null): string {
    const row = this.live(id);
    const worker = this.slotWorker(row, slot);
    if (row.draining === 1) throw new HttpError(409, "session is draining");
    if (policy === null || row.bundle_sha256 !== policy.bundle_sha256) {
      throw new HttpError(409, `bundle ${row.bundle_sha256} is not the active policy ${policy?.bundle_sha256 ?? "(none)"}`);
    }
    return worker;
  }

  // In-progress transitions stay allowed after a republish or while draining.
  jobWorker(id: string, slot: number): string {
    return this.slotWorker(this.live(id), slot);
  }

  drain(id: string): void {
    this.live(id);
    this.sql.exec("UPDATE sessions SET draining = 1 WHERE id = ?", id);
  }

  end(id: string, reason: string): string[] {
    const row = this.live(id);
    const workers = Array.from({ length: row.slots as number }, (_, slot) => workerId(id, slot));
    const failed = this.jobs.failWorkers(workers);
    this.sql.exec("UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ?", this.now(), reason, id);
    this.sql.exec("UPDATE accounts SET session_id = NULL, slot = NULL, leased_at = NULL WHERE session_id = ?", id);
    return failed;
  }

  endSilent(): string[] {
    const cutoff = this.now() - SESSION_SILENCE_SECONDS;
    return this.rows("SELECT id FROM sessions WHERE ended_at IS NULL AND last_seen_at <= ?", cutoff).flatMap((row) =>
      this.end(row.id as string, "silent"),
    );
  }

  nextDeadline(): number | null {
    const row = this.rows("SELECT MIN(last_seen_at) AS t FROM sessions WHERE ended_at IS NULL")[0];
    return row?.t == null ? null : (row.t as number) + SESSION_SILENCE_SECONDS;
  }

  capacity(): CapacityBody {
    const cutoff = this.now() - SESSION_LIVE_SECONDS;
    const statuses = this.rows(
      `SELECT status FROM sessions WHERE ended_at IS NULL AND draining = 0 AND status IS NOT NULL
         AND last_seen_at >= ? ORDER BY started_at`,
      cutoff,
    ).map((row) => JSON.parse(row.status as string) as RunnerStatus);
    const base = { active: this.jobs.activeCount(), queued: this.jobs.queueDepth(), target_fps: 60 };
    if (statuses.length === 0) {
      return {
        ...base,
        capacity: 0,
        healthy_slots: 0,
        service_status: "unavailable",
        service_message: "Game servers are unavailable. Try again shortly.",
        game_fps: null,
        frame_interval_p95_ms: null,
        dolphin_step_p95_ms: null,
        policy_round_trip_p95_ms: null,
        model_inference_p95_ms: null,
        batch_wait_p95_ms: null,
        recoveries: 0,
      };
    }
    const capacity = statuses.reduce((sum, status) => sum + status.slots, 0);
    const healthy = statuses.reduce((sum, status) => sum + status.healthy_slots, 0);
    let state: string;
    let message: string;
    if (statuses.length === 1) {
      state = statuses[0]!.state;
      message = statuses[0]!.message;
    } else if (statuses.every((status) => status.state === "ready")) {
      state = "ready";
      message = "Game servers are ready.";
    } else if (healthy === 0) {
      state = "recovering";
      message = "Game servers are recovering.";
    } else {
      state = "degraded";
      message = "Some game servers are degraded; others remain available.";
    }
    const values = (name: (typeof TIMINGS)[number]) =>
      statuses.map((status) => status[name]).filter((value): value is number => value !== null);
    const lowest = (name: (typeof TIMINGS)[number]) => (values(name).length ? Math.min(...values(name)) : null);
    const highest = (name: (typeof TIMINGS)[number]) => (values(name).length ? Math.max(...values(name)) : null);
    return {
      ...base,
      capacity,
      healthy_slots: healthy,
      service_status: state,
      service_message: message,
      game_fps: lowest("game_fps"),
      frame_interval_p95_ms: highest("frame_interval_p95_ms"),
      dolphin_step_p95_ms: highest("dolphin_step_p95_ms"),
      policy_round_trip_p95_ms: highest("policy_round_trip_p95_ms"),
      model_inference_p95_ms: highest("model_inference_p95_ms"),
      batch_wait_p95_ms: highest("batch_wait_p95_ms"),
      recoveries: statuses.reduce((sum, status) => sum + status.recoveries, 0),
    };
  }

  putAccounts(raw: unknown): void {
    const accounts = parseAccounts(raw);
    const codes = new Set(accounts.map((account) => account.connect_code));
    for (const row of this.rows("SELECT connect_code FROM accounts WHERE session_id IS NOT NULL")) {
      if (!codes.has(row.connect_code as string)) {
        throw new HttpError(409, `account ${row.connect_code} is leased by a live session`);
      }
    }
    const keep = accounts.map(() => "?").join(",");
    this.sql.exec(`DELETE FROM accounts WHERE connect_code NOT IN (${keep || "''"})`, ...accounts.map((a) => a.connect_code));
    for (const account of accounts) {
      this.sql.exec(
        `INSERT INTO accounts(connect_code, r2_key, sha256) VALUES (?, ?, ?)
         ON CONFLICT(connect_code) DO UPDATE SET r2_key = excluded.r2_key, sha256 = excluded.sha256`,
        account.connect_code,
        account.r2_key,
        account.sha256,
      );
    }
  }

  summary() {
    return {
      sessions: this.rows(
        `SELECT id, host, bundle_sha256, git_sha, slots, wants_stream, started_at, last_seen_at, draining,
           ended_at, end_reason FROM sessions WHERE ended_at IS NULL OR ended_at >= ? ORDER BY started_at`,
        this.now() - 24 * 60 * 60,
      ),
      accounts: this.rows("SELECT connect_code, session_id, slot, leased_at FROM accounts ORDER BY connect_code"),
    };
  }
}
```

Note: the `validatePlayerCode` import is used by `parseAccounts`; `parseRunnerStatus` accepts `RUNNER_STATES` exactly as `hal/netplay_service/health.py` `RunnerState`.

- [ ] **Step 5: Run the tests**

Run: `npx vitest run test/sessions.test.ts && npm run typecheck`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add web/netplay-api/src/sessions.ts web/netplay-api/src/events.ts web/netplay-api/test/sessions.test.ts
git commit -m "Add runner sessions, account leases, capacity, and events"
```

---

### Task 7: The Queue Durable Object and Worker routes

**Files:**
- Replace: `web/netplay-api/src/queue.ts`, `src/http.ts`
- Modify: `web/netplay-api/test/helpers.ts` (add route helpers)
- Delete: `web/netplay-api/test/smoke.test.ts` (covered by routes tests)
- Test: `web/netplay-api/test/routes.test.ts`

**Interfaces:**
- Consumes: everything from Tasks 3–6.
- Produces: `interface ApiResult { status: number; body?: unknown; headers?: Record<string, string> }`; `class Queue` RPC methods: `options()`, `capacity()`, `createJob(raw)`, `getJob(id, token)`, `updatePolicy(id, token, raw)`, `cancelJob(id, token)`, `rematch(id, token, raw)`, `startSession(raw)`, `reportStatus(sid, raw)`, `claim(sid, raw)`, `drain(sid)`, `endSession(sid)`, `runnerJob(sid, slot, jobId, action, raw)`, `workerJob(sid, slot, jobId)`, `putPolicy(raw)`, `putAccounts(raw)`, `setPaused(paused)`, `adminStatus()`, `adminEvents(query)`, `logRefusal(kind, detail)`, `setTestClock(t)`, `resetForTest()`; `fetch(request)` for the live WebSocket (Task 8); `alarm()`.
- Test helpers produce: `RUNNER_TOKEN`, `ADMIN_TOKEN`, `call(method, path, options)`, `publish(policy?)`, `seedAccounts(n)`, `startSession(slots?)`, `report(session, slots, healthy?)`, `runAlarm()`, `setClock(t)`, `resetQueue()`.

- [ ] **Step 1: Add route helpers to `test/helpers.ts`**

Change the first import line of `test/helpers.ts` to
`import { env, runDurableObjectAlarm, runInDurableObject, SELF } from "cloudflare:test";`,
add `import policyJson from "./transcripts/policy.json";` below it, and append:

```ts
export const RUNNER_TOKEN = "runner-test-token";
export const OTHER_RUNNER_TOKEN = "runner-other-token";
export const ADMIN_TOKEN = "admin-test-token";
export const POLICY = policyJson;

export interface Call {
  status: number;
  body: any;
  headers: Headers;
}

export interface CallOptions {
  body?: unknown;
  rawBody?: string;
  token?: string;
  runner?: { session: string; slot: number } | true;
  runnerToken?: string;
  admin?: boolean;
  ip?: string;
}

export async function call(method: string, path: string, options: CallOptions = {}): Promise<Call> {
  const headers = new Headers({ "CF-Connecting-IP": options.ip ?? `198.51.100.${Math.floor(Math.random() * 250)}` });
  if (options.token !== undefined) headers.set("Authorization", `Bearer ${options.token}`);
  if (options.runner) {
    headers.set("Authorization", `Bearer ${options.runnerToken ?? RUNNER_TOKEN}`);
    if (options.runner !== true) {
      headers.set("X-HAL-Session", options.runner.session);
      headers.set("X-HAL-Slot", String(options.runner.slot));
    }
  }
  if (options.admin) headers.set("Authorization", `Bearer ${ADMIN_TOKEN}`);
  let body = options.rawBody;
  if (options.body !== undefined) {
    body = JSON.stringify(options.body);
    headers.set("Content-Type", "application/json");
  }
  const response = await SELF.fetch(new Request(`https://20xx.xyz${path}`, { method, headers, body }));
  const text = await response.text();
  return { status: response.status, body: text ? JSON.parse(text) : null, headers: response.headers };
}

export async function resetQueue(): Promise<void> {
  await queueStub().resetForTest();
  await queueStub().setTestClock(START);
}

export async function setClock(t: number): Promise<void> {
  await queueStub().setTestClock(t);
}

export async function runAlarm(): Promise<boolean> {
  return runDurableObjectAlarm(queueStub());
}

export async function publish(policy: unknown = POLICY): Promise<void> {
  const result = await call("PUT", "/v1/admin/policy", { admin: true, body: policy });
  if (result.status !== 200) throw new Error(`publish failed: ${JSON.stringify(result.body)}`);
}

export async function seedAccounts(count: number): Promise<void> {
  const accounts = Array.from({ length: count }, (_, index) => ({
    connect_code: `BOT${index}#1`,
    r2_key: `netplay/accounts/bot${index}.json`,
    sha256: (index % 10).toString().repeat(64),
  }));
  const result = await call("PUT", "/v1/admin/accounts", { admin: true, body: accounts });
  if (result.status !== 200) throw new Error(`accounts failed: ${JSON.stringify(result.body)}`);
}

export function runnerStatus(slots: number, healthy = slots) {
  return {
    schema_version: 5,
    state: healthy === slots ? "ready" : healthy === 0 ? "recovering" : "degraded",
    message: healthy === slots ? "Game servers are ready." : "One game server is recovering; other slots remain available.",
    policy_sha256: POLICY.bundle_sha256,
    slots,
    healthy_slots: healthy,
    target_fps: 60,
    game_fps: null,
    frame_interval_p95_ms: null,
    dolphin_step_p95_ms: null,
    policy_round_trip_p95_ms: null,
    model_inference_p95_ms: null,
    batch_wait_p95_ms: null,
    recoveries: 0,
    updated_at: 0,
    chunk_health: [],
  };
}

export async function report(session: string, slots: number, healthy = slots): Promise<Call> {
  return call("POST", `/v1/runner/sessions/${session}/status`, { runner: true, body: runnerStatus(slots, healthy) });
}

export async function startSession(slots = 2): Promise<string> {
  const result = await call("POST", "/v1/runner/sessions", {
    runner: true,
    body: { host: "test-box", bundle_sha256: POLICY.bundle_sha256, git_sha: "abc123", slots, stream: false },
  });
  if (result.status !== 201) throw new Error(`session failed: ${JSON.stringify(result.body)}`);
  await report(result.body.session_id, slots);
  return result.body.session_id as string;
}

export const CREATE = { player_code: "CRYO#610", character: "FOX", imitation: "IBDW#0", online_delay: 2 };
```

- [ ] **Step 2: Write the failing route tests**

`web/netplay-api/test/routes.test.ts`:

```ts
import { beforeEach, describe, expect, it } from "vitest";
import {
  ADMIN_TOKEN,
  CREATE,
  OTHER_RUNNER_TOKEN,
  POLICY,
  call,
  publish,
  report,
  resetQueue,
  runAlarm,
  seedAccounts,
  setClock,
  startSession,
  START,
} from "./helpers";

beforeEach(async () => {
  await resetQueue();
});

async function ready(slots = 2): Promise<string> {
  await publish();
  await seedAccounts(4);
  return startSession(slots);
}

describe("security and shape", () => {
  it("sets security headers and no CORS headers", async () => {
    const result = await call("GET", "/v1/options");
    expect(result.headers.get("Cache-Control")).toBe("no-store");
    expect(result.headers.get("Content-Security-Policy")).toBe("default-src 'none'; frame-ancestors 'none'");
    expect(result.headers.get("Referrer-Policy")).toBe("no-referrer");
    expect(result.headers.get("X-Content-Type-Options")).toBe("nosniff");
    expect(result.headers.get("Access-Control-Allow-Origin")).toBeNull();
  });

  it("answers malformed bodies with 4xx, never 500", async () => {
    const session = await ready();
    const big = "x".repeat(16 * 1024 + 1);
    const cases: [string, string, { rawBody?: string; body?: unknown }, object][] = [
      ["POST", "/v1/jobs", { rawBody: "{not json" }, {}],
      ["POST", "/v1/jobs", { rawBody: big }, {}],
      ["POST", "/v1/jobs", { body: [1, 2] }, {}],
      ["PATCH", "/v1/jobs/x/policy", { rawBody: "[" }, { token: "t" }],
      ["POST", `/v1/runner/sessions/${session}/claim`, { body: { slot: "zero" } }, { runner: true }],
      ["POST", `/v1/runner/sessions/${session}/status`, { body: null }, { runner: true }],
      ["POST", "/v1/runner/sessions", { rawBody: "{" }, { runner: true }],
      ["PUT", "/v1/admin/policy", { body: { schema_version: 1 } }, { admin: true }],
      ["PUT", "/v1/admin/accounts", { body: "nope" }, { admin: true }],
    ];
    for (const [method, path, body, auth] of cases) {
      const result = await call(method, path, { ...body, ...auth });
      expect(result.status, `${method} ${path}`).toBeGreaterThanOrEqual(400);
      expect(result.status, `${method} ${path}`).toBeLessThan(500);
    }
    expect((await call("POST", "/v1/jobs", { rawBody: big })).status).toBe(413);
  });

  it("requires runner and admin tokens", async () => {
    expect((await call("POST", "/v1/runner/sessions", { body: {} })).status).toBe(401);
    expect((await call("POST", "/v1/runner/sessions", { token: ADMIN_TOKEN, body: {} })).status).toBe(401);
    expect((await call("PUT", "/v1/admin/policy", { runner: true, body: POLICY })).status).toBe(401);
    expect((await call("GET", "/v1/admin/status", { admin: true })).status).toBe(200);
    const refused = await call("GET", "/v1/admin/events", { admin: true });
    expect(refused.body.events.map((event: { kind: string }) => event.kind)).toContain("refused");
  });

  it("accepts any listed runner token", async () => {
    await publish();
    await seedAccounts(2);
    const result = await call("POST", "/v1/runner/sessions", {
      runner: true,
      runnerToken: OTHER_RUNNER_TOKEN,
      body: { host: "b", bundle_sha256: POLICY.bundle_sha256, git_sha: "g", slots: 1, stream: false },
    });
    expect(result.status).toBe(201);
  });
});

describe("player routes", () => {
  it("serves options from the published policy", async () => {
    expect((await call("GET", "/v1/options")).status).toBe(503);
    await publish();
    const result = await call("GET", "/v1/options");
    expect(result.status).toBe(200);
    expect(result.body.imitations.some((choice: { value: string }) => choice.value === "MASKED")).toBe(false);
    expect(result.body.max_games).toBe(5);
  });

  it("refuses jobs without a live session, while paused, and at the cap", async () => {
    await publish();
    expect((await call("POST", "/v1/jobs", { body: CREATE })).body).toEqual({
      detail: "Game servers are unavailable. Try again shortly.",
    });
    await seedAccounts(2);
    await startSession();
    await call("POST", "/v1/admin/pause", { admin: true });
    expect((await call("POST", "/v1/jobs", { body: CREATE })).status).toBe(503);
    await call("POST", "/v1/admin/resume", { admin: true });
    for (let index = 0; index < 20; index += 1) {
      const created = await call("POST", "/v1/jobs", { body: { ...CREATE, player_code: `P${index}#1` } });
      expect(created.status).toBe(201);
    }
    const full = await call("POST", "/v1/jobs", { body: { ...CREATE, player_code: "LAST#1" } });
    expect(full).toMatchObject({ status: 503, body: { detail: "The queue is full. Try again in a few minutes." } });
  });

  it("rate-limits job creation per address", async () => {
    await ready();
    const statuses: number[] = [];
    for (let index = 0; index < 6; index += 1) {
      const result = await call("POST", "/v1/jobs", { ip: "203.0.113.9", body: { ...CREATE, player_code: `R${index}#1` } });
      statuses.push(result.status);
      if (result.status === 429) expect(result.headers.get("Retry-After")).toBe("60");
    }
    expect(statuses.slice(0, 5)).toEqual([201, 201, 201, 201, 201]);
    expect(statuses[5]).toBe(429);
  });

  it("reports capacity from sessions", async () => {
    const session = await ready();
    await call("POST", "/v1/jobs", { body: CREATE });
    expect((await call("GET", "/v1/capacity")).body).toMatchObject({
      capacity: 2,
      healthy_slots: 2,
      queued: 1,
      active: 0,
      service_status: "ready",
    });
    await report(session, 2, 1);
    expect((await call("GET", "/v1/capacity")).body.service_status).toBe("degraded");
  });
});

describe("runner routes", () => {
  it("runs a reservation through claim, connect, play, and finish", async () => {
    const session = await ready();
    const job = (await call("POST", "/v1/jobs", { body: CREATE })).body;
    const runner = { session, slot: 0 };
    const claimed = await call("POST", `/v1/runner/sessions/${session}/claim`, { runner: true, body: { slot: 0 } });
    expect(claimed).toMatchObject({ status: 200, body: { id: job.id, status: "leased" } });
    expect((await call("POST", `/v1/runner/sessions/${session}/claim`, { runner: true, body: { slot: 1 } })).status).toBe(204);
    const path = `/v1/runner/jobs/${job.id}`;
    expect((await call("POST", `${path}/connecting`, { runner, body: { connect_code: "HALBOT#1" } })).body.status).toBe(
      "connecting",
    );
    expect((await call("POST", `${path}/playing`, { runner })).body.status).toBe("playing");
    const finished = await call("POST", `${path}/finish-game`, {
      runner,
      body: { game_number: 1, actual_stage: "BATTLEFIELD", result: "win" },
    });
    expect(finished.body).toMatchObject({ status: "rematch_wait", game_count: 1 });
    expect((await call("GET", path, { runner })).body.status).toBe("rematch_wait");
    const events = await call("GET", `/v1/admin/events?job=${job.id}`, { admin: true });
    expect(events.body.events.map((event: { kind: string }) => event.kind)).toEqual([
      "job_created",
      "job_claimed",
      "job_connecting",
      "job_playing",
      "game_finished",
    ]);
  });

  it("refuses a session whose bundle is not the active policy", async () => {
    await publish();
    await seedAccounts(2);
    const result = await call("POST", "/v1/runner/sessions", {
      runner: true,
      body: { host: "b", bundle_sha256: "f".repeat(64), git_sha: "g", slots: 1, stream: false },
    });
    expect(result.status).toBe(409);
  });

  it("lets an old-bundle session finish its game but not claim", async () => {
    const session = await ready(1);
    const job = (await call("POST", "/v1/jobs", { body: CREATE })).body;
    const runner = { session, slot: 0 };
    await call("POST", `/v1/runner/sessions/${session}/claim`, { runner: true, body: { slot: 0 } });
    await call("POST", `/v1/runner/jobs/${job.id}/connecting`, { runner, body: { connect_code: "HALBOT#1" } });
    await call("POST", `/v1/runner/jobs/${job.id}/playing`, { runner });
    await publish({ ...POLICY, bundle_sha256: "e".repeat(64) });
    const finished = await call("POST", `/v1/runner/jobs/${job.id}/finish-game`, {
      runner,
      body: { game_number: 1, actual_stage: "BATTLEFIELD", result: "win" },
    });
    expect(finished.status).toBe(200);
    const fail = await call("POST", `/v1/runner/jobs/${job.id}/fail`, {
      runner,
      body: { error_code: "policy_changed", retryable: false },
    });
    expect(fail.status).toBe(200);
    expect((await call("POST", `/v1/runner/sessions/${session}/claim`, { runner: true, body: { slot: 0 } })).status).toBe(409);
  });

  it("ends a silent session from the alarm", async () => {
    const session = await ready(1);
    const job = (await call("POST", "/v1/jobs", { body: CREATE })).body;
    await call("POST", `/v1/runner/sessions/${session}/claim`, { runner: true, body: { slot: 0 } });
    await setClock(START + 30);
    expect(await runAlarm()).toBe(true);
    expect((await call("GET", `/v1/jobs/${job.id}`, { token: job.token })).body.status).toBe("failed");
    expect((await call("POST", `/v1/runner/sessions/${session}/status`, { runner: true, body: {} })).status).toBe(410);
  });

  it("drains and ends a session on request", async () => {
    const session = await ready(1);
    expect((await call("POST", `/v1/runner/sessions/${session}/drain`, { runner: true })).status).toBe(200);
    expect((await call("POST", `/v1/runner/sessions/${session}/claim`, { runner: true, body: { slot: 0 } })).status).toBe(409);
    expect((await call("GET", "/v1/capacity")).body.capacity).toBe(0);
    expect((await call("DELETE", `/v1/runner/sessions/${session}`, { runner: true })).body).toEqual({ failed: 0 });
  });

  it("requires session and slot headers on job routes", async () => {
    const session = await ready(1);
    expect((await call("POST", "/v1/runner/jobs/x/playing", { runner: true })).status).toBe(400);
    expect((await call("POST", "/v1/runner/jobs/x/playing", { runner: { session, slot: 5 } })).status).toBe(422);
    expect((await call("POST", "/v1/runner/jobs/x/playing", { runner: { session: "nope", slot: 0 } })).status).toBe(404);
  });
});
```

- [ ] **Step 3: Run to verify failure**

Run: `npx vitest run test/routes.test.ts`
Expected: FAIL (`resetForTest` is not a function).

- [ ] **Step 4: Write `src/queue.ts`**

```ts
import { DurableObject } from "cloudflare:workers";
import {
  HttpError,
  QUEUE_CAP,
  TERMINAL_STATUSES,
  randomToken,
  sha256Hex,
  validatePlayerCode,
} from "./domain";
import type { Env } from "./env";
import { EVENT_SCHEMA, EventLog } from "./events";
import { type PolicyConfig, checkChoice, optionsBody, parsePolicyConfig } from "./policy";
import { bool, fields, int, parseCreate, parsePolicyUpdate, parseRematch, str } from "./requests";
import { SESSION_SCHEMA, SessionStore, type StartRequest } from "./sessions";
import { JOB_SCHEMA, JobStore } from "./store";

export interface ApiResult {
  status: number;
  body?: unknown;
  headers?: Record<string, string>;
}

const QUEUE_SCHEMA = `
CREATE TABLE IF NOT EXISTS policy (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  config TEXT NOT NULL,
  published_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
`;

export type RunnerAction =
  | "heartbeat"
  | "connecting"
  | "playing"
  | "no-show"
  | "no-contest"
  | "finish-game"
  | "fail"
  | "forfeit"
  | "replay";

export class Queue extends DurableObject<Env> {
  private readonly jobs: JobStore;
  private readonly sessions: SessionStore;
  private readonly events: EventLog;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    const sql = ctx.storage.sql;
    sql.exec(JOB_SCHEMA);
    sql.exec(SESSION_SCHEMA);
    sql.exec(EVENT_SCHEMA);
    sql.exec(QUEUE_SCHEMA);
    const now = () => this.now();
    this.jobs = new JobStore(sql, now);
    this.sessions = new SessionStore(sql, this.jobs, now);
    this.events = new EventLog(sql, now);
  }

  private setting(key: string): string | null {
    const rows = this.ctx.storage.sql.exec<{ value: string }>("SELECT value FROM settings WHERE key = ?", key).toArray();
    return rows[0]?.value ?? null;
  }

  private setSetting(key: string, value: string): void {
    this.ctx.storage.sql.exec(
      "INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
      key,
      value,
    );
  }

  private now(): number {
    if (this.env.HAL_TEST_CLOCK === "1") {
      const clock = this.setting("test_clock");
      if (clock !== null) return Number(clock);
    }
    return Date.now() / 1000;
  }

  private policy(): PolicyConfig | null {
    const rows = this.ctx.storage.sql.exec<{ config: string }>("SELECT config FROM policy WHERE id = 1").toArray();
    return rows[0] === undefined ? null : (JSON.parse(rows[0].config) as PolicyConfig);
  }

  private requirePolicy(): PolicyConfig {
    const policy = this.policy();
    if (policy === null) throw new HttpError(503, "no policy has been published");
    return policy;
  }

  private tx<T>(fn: () => T): T {
    return this.ctx.storage.transactionSync(fn);
  }

  private async run(fn: () => unknown, okStatus = 200): Promise<ApiResult> {
    try {
      const body = await fn();
      await this.scheduleAlarm();
      return body === null || body === undefined ? { status: 204 } : { status: okStatus, body };
    } catch (error) {
      if (error instanceof HttpError) {
        return { status: error.status, body: { detail: error.detail }, headers: error.headers };
      }
      throw error;
    }
  }

  private async scheduleAlarm(): Promise<void> {
    const oldest = this.events.oldest();
    const candidates = [
      this.jobs.nextDeadline(),
      this.sessions.nextDeadline(),
      oldest === null ? null : oldest + 30 * 24 * 60 * 60,
    ].filter((value): value is number => value !== null);
    if (candidates.length === 0) {
      await this.ctx.storage.deleteAlarm();
      return;
    }
    // Under the test clock every deadline is a fake time in the past. Park the
    // alarm a day ahead so only runDurableObjectAlarm fires it, never a race.
    const at = this.env.HAL_TEST_CLOCK === "1" ? Date.now() + 86_400_000 : Math.ceil(Math.min(...candidates) * 1000);
    await this.ctx.storage.setAlarm(at);
  }

  async alarm(): Promise<void> {
    const changed = this.tx(() => {
      const ended = this.sessions.endSilent();
      const expired = this.jobs.reapExpired();
      this.events.prune();
      for (const id of expired) this.events.log("job_expired", { job: id, status: this.jobs.row(id)?.status });
      return [...ended, ...expired];
    });
    this.release(changed);
    await this.scheduleAlarm();
  }

  // Player routes

  async options(): Promise<ApiResult> {
    return this.run(() => optionsBody(this.requirePolicy()));
  }

  async capacity(): Promise<ApiResult> {
    return this.run(() => this.sessions.capacity());
  }

  async createJob(raw: unknown): Promise<ApiResult> {
    const id = randomToken(18);
    const token = randomToken(32);
    const digest = await sha256Hex(token);
    return this.run(() => {
      const policy = this.requirePolicy();
      const request = parseCreate(raw, policy);
      // Check order matches the Python API so the golden transcripts agree.
      if (!policy.online_delays.includes(request.online_delay)) {
        throw new HttpError(422, "online delay is unsupported by this policy");
      }
      if (request.imitation === "MASKED" && !policy.masked_identity) {
        throw new HttpError(422, "masked identity is unsupported by this policy");
      }
      if (this.setting("paused") === "1") throw new HttpError(503, "The queue is paused. Try again shortly.");
      if (this.jobs.queueDepth() >= QUEUE_CAP) throw new HttpError(503, "The queue is full. Try again in a few minutes.");
      const capacity = this.sessions.capacity();
      if (capacity.service_status === "unavailable") {
        throw new HttpError(503, "Game servers are unavailable. Try again shortly.");
      }
      if (capacity.healthy_slots === 0) throw new HttpError(503, capacity.service_message);
      checkChoice(policy.characters, request.character, "character");
      checkChoice(policy.imitations, request.imitation, "imitation");
      validatePlayerCode(request.player_code);
      const job = this.tx(() => {
        const created = this.jobs.createJob(id, digest, request.player_code, request);
        this.events.log("job_created", { job: id, character: request.character, imitation: request.imitation });
        return created;
      });
      return { ...job, token };
    }, 201);
  }

  async getJob(id: string, token: string): Promise<ApiResult> {
    const digest = await sha256Hex(token);
    return this.run(() => this.jobs.getJob(id, digest));
  }

  async updatePolicy(id: string, token: string, raw: unknown): Promise<ApiResult> {
    const digest = await sha256Hex(token);
    return this.run(() => {
      const update = parsePolicyUpdate(raw, this.requirePolicy());
      const job = this.tx(() => {
        const current = this.jobs.getJob(id, digest);
        const desired = "desired_return" in update ? update.desired_return ?? null : current.desired_return;
        const temperature = "temperature" in update ? update.temperature : current.temperature;
        if (temperature === null || temperature === undefined) throw new HttpError(422, "temperature cannot be null");
        const updated = this.jobs.updatePolicy(id, digest, desired, temperature);
        this.events.log("policy_updated", { job: id, revision: updated.policy_revision });
        return updated;
      });
      this.broadcastSettings(id);
      return job;
    });
  }

  async cancelJob(id: string, token: string): Promise<ApiResult> {
    const digest = await sha256Hex(token);
    return this.run(() => {
      const job = this.tx(() => {
        const canceled = this.jobs.cancel(id, digest);
        this.events.log("job_cancel_requested", { job: id, status: canceled.status });
        return canceled;
      });
      this.release([id]);
      return job;
    });
  }

  async rematch(id: string, token: string, raw: unknown): Promise<ApiResult> {
    const digest = await sha256Hex(token);
    return this.run(() => {
      const policy = this.requirePolicy();
      const request = parseRematch(raw, policy);
      checkChoice(policy.characters, request.character, "character");
      checkChoice(policy.imitations, request.imitation, "imitation");
      checkChoice(policy.stages, request.stage, "stage");
      return this.tx(() => {
        const job = this.jobs.requestRematch(id, digest, request.character, request.imitation, request.stage);
        this.events.log("rematch_ready", { job: id, character: request.character, stage: request.stage });
        return job;
      });
    });
  }

  // Runner routes

  async startSession(raw: unknown): Promise<ApiResult> {
    const id = randomToken(12);
    return this.run(() => {
      const value = fields(raw, ["host", "bundle_sha256", "git_sha", "slots", "stream"], ["host", "bundle_sha256", "git_sha", "slots", "stream"]);
      const input: StartRequest = {
        host: str(value.host, "host"),
        bundle_sha256: str(value.bundle_sha256, "bundle_sha256"),
        git_sha: str(value.git_sha, "git_sha"),
        slots: int(value.slots, "slots"),
        stream: bool(value.stream, "stream"),
      };
      if (input.slots < 1 || input.slots > 8) throw new HttpError(422, "slots must be in [1, 8]");
      return this.tx(() => {
        const started = this.sessions.start(id, input, this.policy());
        this.events.log("session_started", { session: id, host: input.host, slots: input.slots, git_sha: input.git_sha });
        for (const grant of started.accounts) {
          this.events.log("account_leased", { session: id, slot: grant.slot, connect_code: grant.connect_code });
        }
        return { ...started, policy: this.policy() };
      });
    }, 201);
  }

  async reportStatus(sessionId: string, raw: unknown): Promise<ApiResult> {
    return this.run(() => this.tx(() => this.sessions.report(sessionId, raw)));
  }

  async claim(sessionId: string, raw: unknown): Promise<ApiResult> {
    return this.run(() => {
      const slot = int(fields(raw, ["slot"], ["slot"]).slot, "slot");
      return this.tx(() => {
        const worker = this.sessions.claimWorker(sessionId, slot, this.policy());
        const job = this.jobs.claimNext(worker);
        if (job !== null) this.events.log("job_claimed", { job: job.id, session: sessionId, slot });
        return job;
      });
    });
  }

  async drain(sessionId: string): Promise<ApiResult> {
    return this.run(() =>
      this.tx(() => {
        this.sessions.drain(sessionId);
        this.events.log("session_draining", { session: sessionId });
        return { draining: true };
      }),
    );
  }

  async endSession(sessionId: string): Promise<ApiResult> {
    return this.run(() => {
      const failed = this.tx(() => {
        const ids = this.sessions.end(sessionId, "ended");
        this.events.log("session_ended", { session: sessionId, failed: ids.length });
        return ids;
      });
      this.release(failed);
      return { failed: failed.length };
    });
  }

  async workerJob(sessionId: string, slot: number, jobId: string): Promise<ApiResult> {
    return this.run(() => this.jobs.workerJob(jobId, this.sessions.jobWorker(sessionId, slot)));
  }

  async runnerJob(sessionId: string, slot: number, jobId: string, action: RunnerAction, raw: unknown): Promise<ApiResult> {
    return this.run(() => {
      const policy = this.requirePolicy();
      const job = this.tx(() => {
        const worker = this.sessions.jobWorker(sessionId, slot);
        const log = (kind: string, detail: Record<string, unknown> = {}) =>
          this.events.log(kind, { job: jobId, session: sessionId, slot, ...detail });
        switch (action) {
          case "heartbeat":
            return this.jobs.heartbeat(jobId, worker);
          case "connecting": {
            const code = validatePlayerCode(str(fields(raw, ["connect_code"], ["connect_code"]).connect_code, "connect_code"));
            const result = this.jobs.markConnecting(jobId, worker, code);
            log("job_connecting");
            return result;
          }
          case "playing": {
            const result = this.jobs.markPlaying(jobId, worker);
            log("job_playing");
            return result;
          }
          case "no-show": {
            const result = this.jobs.markNoShow(jobId, worker);
            log("job_no_show");
            return result;
          }
          case "no-contest": {
            const result = this.jobs.markNoContest(jobId, worker);
            log("job_no_contest");
            return result;
          }
          case "finish-game": {
            const value = fields(raw, ["game_number", "actual_stage", "result"], ["game_number", "actual_stage", "result"]);
            const stage = checkChoice(policy.stages, str(value.actual_stage, "actual_stage"), "stage");
            const number = int(value.game_number, "game_number");
            const result = this.jobs.finishGame(jobId, worker, number, stage, str(value.result, "result"));
            log("game_finished", { game_number: number, stage, result: value.result });
            return result;
          }
          case "fail": {
            const value = fields(raw, ["error_code", "retryable"], ["error_code", "retryable"]);
            const code = str(value.error_code, "error_code");
            const result = this.jobs.fail(jobId, worker, code, bool(value.retryable, "retryable"));
            log("job_failed", { error_code: code, status: result.status });
            return result;
          }
          case "forfeit": {
            const result = this.jobs.forfeit(jobId, worker);
            log("job_forfeited");
            return result;
          }
          case "replay": {
            const value = fields(
              raw,
              ["game_number", "key", "sha256", "size", "etag"],
              ["game_number", "key", "sha256", "size", "etag"],
            );
            const number = int(value.game_number, "game_number");
            const result = this.jobs.recordReplay(
              jobId,
              number,
              str(value.key, "key"),
              str(value.sha256, "sha256"),
              int(value.size, "size"),
              str(value.etag, "etag"),
            );
            log("replay_recorded", { game_number: number, key: value.key });
            return result;
          }
        }
      });
      if (TERMINAL_STATUSES.has(job.status) || job.status === "queued") this.release([jobId]);
      return job;
    });
  }

  // Admin routes

  async putPolicy(raw: unknown): Promise<ApiResult> {
    return this.run(() => {
      const policy = parsePolicyConfig(raw);
      this.tx(() => {
        this.ctx.storage.sql.exec(
          `INSERT INTO policy(id, config, published_at) VALUES (1, ?, ?)
           ON CONFLICT(id) DO UPDATE SET config = excluded.config, published_at = excluded.published_at`,
          JSON.stringify(policy),
          this.now(),
        );
        this.events.log("policy_published", { bundle_sha256: policy.bundle_sha256 });
      });
      return policy;
    });
  }

  async putAccounts(raw: unknown): Promise<ApiResult> {
    return this.run(() =>
      this.tx(() => {
        this.sessions.putAccounts(raw);
        this.events.log("accounts_published", { count: Array.isArray(raw) ? raw.length : 0 });
        return this.sessions.summary().accounts;
      }),
    );
  }

  async setPaused(paused: boolean): Promise<ApiResult> {
    return this.run(() =>
      this.tx(() => {
        this.setSetting("paused", paused ? "1" : "0");
        this.events.log(paused ? "queue_paused" : "queue_resumed", {});
        return { paused };
      }),
    );
  }

  async adminStatus(): Promise<ApiResult> {
    return this.run(() => ({
      paused: this.setting("paused") === "1",
      policy: this.policy(),
      capacity: this.sessions.capacity(),
      ...this.sessions.summary(),
    }));
  }

  async adminEvents(query: { job?: string; session?: string; since?: number; limit?: number }): Promise<ApiResult> {
    return this.run(() => ({ events: this.events.query(query) }));
  }

  async logRefusal(detail: Record<string, unknown>): Promise<void> {
    this.tx(() => this.events.log("refused", detail));
    await this.scheduleAlarm();
  }

  // Test seams, enabled only by the HAL_TEST_CLOCK binding in vitest.config.ts.

  async setTestClock(seconds: number): Promise<void> {
    if (this.env.HAL_TEST_CLOCK !== "1") throw new Error("test clock is disabled");
    this.setSetting("test_clock", String(seconds));
  }

  async resetForTest(): Promise<void> {
    if (this.env.HAL_TEST_CLOCK !== "1") throw new Error("test reset is disabled");
    this.tx(() => {
      for (const table of ["games", "jobs", "sessions", "accounts", "events", "policy", "settings"]) {
        this.ctx.storage.sql.exec(`DELETE FROM ${table}`);
      }
    });
    await this.ctx.storage.deleteAlarm();
  }

  // Live settings (Task 8 fills these in).

  private broadcastSettings(_jobId: string): void {}

  private release(_jobIds: readonly string[]): void {}
}
```

- [ ] **Step 5: Write `src/http.ts`**

```ts
import { HttpError, MAX_BODY_BYTES, sameDigest, sha256Hex } from "./domain";
import type { Env } from "./env";
import type { ApiResult, RunnerAction } from "./queue";

const SECURITY_HEADERS: Record<string, string> = {
  "Cache-Control": "no-store",
  "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
};
const RUNNER_ACTIONS = new Set<RunnerAction>([
  "heartbeat",
  "connecting",
  "playing",
  "no-show",
  "no-contest",
  "finish-game",
  "fail",
  "forfeit",
  "replay",
]);

function respond(result: ApiResult): Response {
  const headers = new Headers({ ...SECURITY_HEADERS, ...result.headers });
  if (result.body === undefined) return new Response(null, { status: result.status, headers });
  headers.set("Content-Type", "application/json");
  return new Response(JSON.stringify(result.body), { status: result.status, headers });
}

function failure(status: number, detail: string, headers: Record<string, string> = {}): Response {
  return respond({ status, body: { detail }, headers });
}

async function readBody(request: Request): Promise<unknown> {
  const length = request.headers.get("Content-Length");
  if (length !== null) {
    const parsed = Number(length);
    if (!Number.isInteger(parsed) || parsed < 0) throw new HttpError(400, "invalid Content-Length");
    if (parsed > MAX_BODY_BYTES) throw new HttpError(413, "request body is too large");
  }
  const text = await request.text();
  if (new TextEncoder().encode(text).byteLength > MAX_BODY_BYTES) throw new HttpError(413, "request body is too large");
  if (text === "") return undefined;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new HttpError(422, "request body must be JSON");
  }
}

function bearer(request: Request): string | null {
  const header = request.headers.get("Authorization");
  if (header === null) return null;
  const [scheme, value] = header.split(" ", 2);
  return scheme?.toLowerCase() === "bearer" && value ? value : null;
}

async function tokenMatches(token: string | null, digests: string): Promise<boolean> {
  if (token === null) return false;
  const digest = await sha256Hex(token);
  // Check every entry so the time taken does not reveal which one matched.
  return digests
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean)
    .reduce((found, entry) => sameDigest(entry, digest) || found, false);
}

function runnerSlot(request: Request): { session: string; slot: number } {
  const session = request.headers.get("X-HAL-Session");
  const slot = request.headers.get("X-HAL-Slot");
  if (!session || slot === null || !/^\d+$/.test(slot)) {
    throw new HttpError(400, "X-HAL-Session and X-HAL-Slot headers are required");
  }
  return { session, slot: Number(slot) };
}

export async function handle(request: Request, env: Env): Promise<Response> {
  const queue = env.QUEUE.get(env.QUEUE.idFromName("global"));
  const url = new URL(request.url);
  const path = url.pathname;
  const method = request.method;
  try {
    // Player routes
    if (method === "GET" && path === "/v1/options") return respond(await queue.options());
    if (method === "GET" && path === "/v1/capacity") return respond(await queue.capacity());
    if (method === "POST" && path === "/v1/jobs") {
      const address = request.headers.get("CF-Connecting-IP") ?? "unknown";
      const { success } = await env.JOB_RATE_LIMIT.limit({ key: address });
      if (!success) {
        return failure(429, "Too many reservations from this address. Try again in a minute.", { "Retry-After": "60" });
      }
      return respond(await queue.createJob(await readBody(request)));
    }
    const job = path.match(/^\/v1\/jobs\/([^/]+)(\/policy|\/rematch)?$/);
    if (job) {
      const [, id, suffix] = job as [string, string, string | undefined];
      // FastAPI resolves the bearer dependency before it reads the body.
      const token = bearer(request);
      if (method === "PATCH" && suffix === "/policy") {
        if (token === null) return failure(401, "job token is required");
        return respond(await queue.updatePolicy(id, token, await readBody(request)));
      }
      if (method === "POST" && suffix === "/rematch") {
        if (token === null) return failure(401, "job token is required");
        return respond(await queue.rematch(id, token, await readBody(request)));
      }
      if (suffix === undefined && (method === "GET" || method === "DELETE")) {
        if (token === null) return failure(401, "job token is required");
        return respond(method === "GET" ? await queue.getJob(id, token) : await queue.cancelJob(id, token));
      }
    }

    // Runner routes
    if (path.startsWith("/v1/runner/")) {
      if (!(await tokenMatches(bearer(request), env.RUNNER_TOKEN_SHA256))) {
        await queue.logRefusal({ scope: "runner", method, path });
        return failure(401, "runner token is invalid");
      }
      if (method === "POST" && path === "/v1/runner/sessions") {
        return respond(await queue.startSession(await readBody(request)));
      }
      const session = path.match(/^\/v1\/runner\/sessions\/([^/]+)(\/status|\/claim|\/drain)?$/);
      if (session) {
        const [, id, suffix] = session as [string, string, string | undefined];
        if (method === "POST" && suffix === "/status") return respond(await queue.reportStatus(id, await readBody(request)));
        if (method === "POST" && suffix === "/claim") return respond(await queue.claim(id, await readBody(request)));
        if (method === "POST" && suffix === "/drain") return respond(await queue.drain(id));
        if (method === "DELETE" && suffix === undefined) return respond(await queue.endSession(id));
      }
      const runnerJob = path.match(/^\/v1\/runner\/jobs\/([^/]+)(?:\/([a-z-]+))?$/);
      if (runnerJob) {
        const [, id, action] = runnerJob as [string, string, string | undefined];
        if (action === "live") return queue.fetch(request);
        const { session: sessionId, slot } = runnerSlot(request);
        if (method === "GET" && action === undefined) return respond(await queue.workerJob(sessionId, slot, id));
        if (method === "POST" && action !== undefined && RUNNER_ACTIONS.has(action as RunnerAction)) {
          return respond(await queue.runnerJob(sessionId, slot, id, action as RunnerAction, await readBody(request)));
        }
      }
    }

    // Admin routes
    if (path.startsWith("/v1/admin/")) {
      if (!(await tokenMatches(bearer(request), env.ADMIN_TOKEN_SHA256))) {
        await queue.logRefusal({ scope: "admin", method, path });
        return failure(401, "admin token is invalid");
      }
      if (method === "PUT" && path === "/v1/admin/policy") return respond(await queue.putPolicy(await readBody(request)));
      if (method === "PUT" && path === "/v1/admin/accounts") return respond(await queue.putAccounts(await readBody(request)));
      if (method === "POST" && path === "/v1/admin/pause") return respond(await queue.setPaused(true));
      if (method === "POST" && path === "/v1/admin/resume") return respond(await queue.setPaused(false));
      if (method === "GET" && path === "/v1/admin/status") return respond(await queue.adminStatus());
      if (method === "GET" && path === "/v1/admin/events") {
        const since = url.searchParams.get("since");
        const limit = url.searchParams.get("limit");
        return respond(
          await queue.adminEvents({
            job: url.searchParams.get("job") ?? undefined,
            session: url.searchParams.get("session") ?? undefined,
            since: since === null ? undefined : Number(since),
            limit: limit === null ? undefined : Number(limit),
          }),
        );
      }
    }
    return failure(404, "not found");
  } catch (error) {
    if (error instanceof HttpError) return failure(error.status, error.detail, error.headers);
    throw error;
  }
}
```

- [ ] **Step 6: Delete the smoke test and run the suite**

```bash
git rm web/netplay-api/test/smoke.test.ts
npx vitest run && npm run typecheck
```

Expected: all tests pass. If the rate-limit test fails because the local runtime does not enforce the `ratelimits` binding, stop and report it; do not remove the test or the limit.

- [ ] **Step 7: Commit**

```bash
git add web/netplay-api
git commit -m "Add queue durable object and worker routes"
```

---

### Task 8: Live settings WebSocket

**Files:**
- Modify: `web/netplay-api/src/queue.ts` (`fetch`, `broadcastSettings`, `release`, WebSocket handlers)
- Test: `web/netplay-api/test/live.test.ts`

**Interfaces:**
- Consumes: Task 7's `Queue`, `JobStore.row`, `SessionStore.jobWorker`.
- Produces: `WS /v1/runner/jobs/{id}/live` (runner auth plus `X-HAL-Session` / `X-HAL-Slot`). Messages from the server: `{"type":"settings","revision":n,"desired_return":x|null,"temperature":t}` on connect and on every policy change; `{"type":"released"}` followed by close code 1000 when the job is terminal, requeued, or owned by another worker.

- [ ] **Step 1: Write the failing tests**

`web/netplay-api/test/live.test.ts`:

```ts
import { SELF } from "cloudflare:test";
import { beforeEach, describe, expect, it } from "vitest";
import { CREATE, RUNNER_TOKEN, call, publish, resetQueue, seedAccounts, startSession } from "./helpers";

beforeEach(async () => {
  await resetQueue();
});

async function open(jobId: string, session: string, slot: number): Promise<{ socket: WebSocket; messages: unknown[] }> {
  const response = await SELF.fetch(`https://20xx.xyz/v1/runner/jobs/${jobId}/live`, {
    headers: {
      Upgrade: "websocket",
      Authorization: `Bearer ${RUNNER_TOKEN}`,
      "X-HAL-Session": session,
      "X-HAL-Slot": String(slot),
    },
  });
  expect(response.status).toBe(101);
  const socket = response.webSocket!;
  const messages: unknown[] = [];
  socket.addEventListener("message", (event) => messages.push(JSON.parse(event.data as string)));
  socket.accept();
  return { socket, messages };
}

async function until(check: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 50 && !check(); attempt += 1) await new Promise((resolve) => setTimeout(resolve, 10));
  expect(check()).toBe(true);
}

async function claimed(): Promise<{ session: string; job: { id: string; token: string } }> {
  await publish();
  await seedAccounts(2);
  const session = await startSession(2);
  const job = (await call("POST", "/v1/jobs", { body: CREATE })).body;
  await call("POST", `/v1/runner/sessions/${session}/claim`, { runner: true, body: { slot: 0 } });
  return { session, job };
}

describe("live settings", () => {
  it("sends current settings, then every change", async () => {
    const { session, job } = await claimed();
    const { messages } = await open(job.id, session, 0);
    await until(() => messages.length === 1);
    expect(messages[0]).toEqual({ type: "settings", revision: 0, desired_return: 20, temperature: 1 });
    await call("PATCH", `/v1/jobs/${job.id}/policy`, { token: job.token, body: { desired_return: 35 } });
    await until(() => messages.length === 2);
    expect(messages[1]).toEqual({ type: "settings", revision: 1, desired_return: 35, temperature: 1 });
  });

  it("releases the runner when the player cancels", async () => {
    const { session, job } = await claimed();
    const { messages, socket } = await open(job.id, session, 0);
    let closed = 0;
    socket.addEventListener("close", (event) => {
      closed = event.code;
    });
    await until(() => messages.length === 1);
    await call("DELETE", `/v1/jobs/${job.id}`, { token: job.token });
    await until(() => messages.length === 2 && closed === 1000);
    expect(messages[1]).toEqual({ type: "released" });
  });

  it("refuses a socket from a worker that does not own the job", async () => {
    const { session, job } = await claimed();
    const response = await SELF.fetch(`https://20xx.xyz/v1/runner/jobs/${job.id}/live`, {
      headers: {
        Upgrade: "websocket",
        Authorization: `Bearer ${RUNNER_TOKEN}`,
        "X-HAL-Session": session,
        "X-HAL-Slot": "1",
      },
    });
    expect(response.status).toBe(409);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run test/live.test.ts`
Expected: FAIL (status 404 instead of 101).

- [ ] **Step 3: Implement the WebSocket in `src/queue.ts`**

Replace the two stub methods at the end of the class and add `fetch` and the handlers:

```ts
  private settingsMessage(jobId: string): string | null {
    const row = this.jobs.row(jobId);
    if (row === null) return null;
    return JSON.stringify({
      type: "settings",
      revision: row.policy_revision,
      desired_return: row.desired_return,
      temperature: row.temperature,
    });
  }

  async fetch(request: Request): Promise<Response> {
    const match = new URL(request.url).pathname.match(/^\/v1\/runner\/jobs\/([^/]+)\/live$/);
    if (match === null || request.headers.get("Upgrade")?.toLowerCase() !== "websocket") {
      return Response.json({ detail: "not found" }, { status: 404 });
    }
    const jobId = match[1]!;
    const slot = request.headers.get("X-HAL-Slot");
    const session = request.headers.get("X-HAL-Session");
    if (!session || slot === null || !/^\d+$/.test(slot)) {
      return Response.json({ detail: "X-HAL-Session and X-HAL-Slot headers are required" }, { status: 400 });
    }
    let worker: string;
    try {
      worker = this.sessions.jobWorker(session, Number(slot));
    } catch (error) {
      if (error instanceof HttpError) return Response.json({ detail: error.detail }, { status: error.status });
      throw error;
    }
    const row = this.jobs.row(jobId);
    if (row === null || row.lease_owner !== worker) {
      return Response.json({ detail: "worker does not own this job" }, { status: 409 });
    }
    const [client, server] = Object.values(new WebSocketPair()) as [WebSocket, WebSocket];
    this.ctx.acceptWebSocket(server, [`job:${jobId}`]);
    server.serializeAttachment({ jobId, worker });
    server.send(this.settingsMessage(jobId)!);
    return new Response(null, { status: 101, webSocket: client });
  }

  private broadcastSettings(jobId: string): void {
    const message = this.settingsMessage(jobId);
    if (message === null) return;
    for (const socket of this.ctx.getWebSockets(`job:${jobId}`)) socket.send(message);
  }

  // A socket's runner no longer owns its job: tell it, then close.
  private release(jobIds: readonly string[]): void {
    for (const jobId of new Set(jobIds)) {
      const row = this.jobs.row(jobId);
      for (const socket of this.ctx.getWebSockets(`job:${jobId}`)) {
        const { worker } = socket.deserializeAttachment() as { jobId: string; worker: string };
        if (row === null || TERMINAL_STATUSES.has(row.status as string) || row.lease_owner !== worker) {
          socket.send(JSON.stringify({ type: "released" }));
          socket.close(1000, "released");
        }
      }
    }
  }

  async webSocketMessage(_socket: WebSocket, _message: string | ArrayBuffer): Promise<void> {
    // Runners only listen on this socket.
  }

  async webSocketClose(socket: WebSocket, code: number): Promise<void> {
    socket.close(code === 1005 ? 1000 : code, "closed");
  }
```

The `release` calls that Task 7 already makes (`cancelJob`, `endSession`, `alarm`, `runnerJob`) now take effect.

- [ ] **Step 4: Run the tests**

Run: `npx vitest run && npm run typecheck`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add web/netplay-api/src/queue.ts web/netplay-api/test/live.test.ts
git commit -m "Push live policy settings to runners over WebSocket"
```

---

### Task 9: Replay the golden transcripts

**Files:**
- Test: `web/netplay-api/test/transcripts.test.ts`

**Interfaces:**
- Consumes: the transcript format from Task 2; helpers `call`, `publish`, `seedAccounts`, `report`, `resetQueue`, `setClock`, `runAlarm`, `START`, `POLICY`.

- [ ] **Step 1: Write the replayer**

`web/netplay-api/test/transcripts.test.ts`:

```ts
import { beforeEach, describe, expect, it } from "vitest";
import { POLICY, START, call, publish, report, resetQueue, runAlarm, seedAccounts, setClock } from "./helpers";

interface Step {
  kind: "player" | "worker" | "advance";
  method?: string;
  path?: string;
  token?: string | null;
  body?: unknown;
  worker?: string;
  op?: string;
  job?: string | null;
  args?: Record<string, unknown>;
  compare?: "full" | "status" | "status_field";
  seconds?: number;
  response?: { status: number; body: any };
}

interface Transcript {
  name: string;
  steps: Step[];
}

const files = import.meta.glob<Transcript>("./transcripts/*.json", { eager: true, import: "default" });
const transcripts = Object.entries(files)
  .filter(([path]) => !path.endsWith("/policy.json"))
  .map(([, transcript]) => transcript);

class Aliases {
  private readonly toReal = new Map<string, string>();
  private readonly toAlias = new Map<string, string>();
  private counts = { job: 0, token: 0 };

  learn(body: unknown): void {
    if (typeof body !== "object" || body === null || Array.isArray(body)) return;
    const record = body as Record<string, unknown>;
    for (const [key, prefix] of [
      ["id", "job"],
      ["token", "token"],
    ] as const) {
      const value = record[key];
      if (typeof value === "string" && !this.toAlias.has(value)) {
        this.counts[prefix] += 1;
        const alias = `$${prefix}${this.counts[prefix]}`;
        this.toAlias.set(value, alias);
        this.toReal.set(alias, value);
      }
    }
  }

  real(text: string): string {
    return [...this.toReal.entries()]
      .sort(([a], [b]) => b.length - a.length)
      .reduce((result, [alias, value]) => result.replaceAll(alias, value), text);
  }

  normalize(body: unknown): unknown {
    if (Array.isArray(body)) return body.map((value) => this.normalize(value));
    if (typeof body === "object" && body !== null) {
      return Object.fromEntries(Object.entries(body).map(([key, value]) => [key, this.normalize(value)]));
    }
    if (typeof body === "string") return this.toAlias.get(body) ?? body;
    return body;
  }
}

function sessionAliases(transcript: Transcript): Map<string, number> {
  const slots = new Map<string, number>();
  for (const step of transcript.steps) {
    if (step.kind !== "worker" || !step.worker) continue;
    const [session, slot] = step.worker.split(":");
    const needed = slot === undefined ? Number(step.args?.slots ?? 1) : Number(slot) + 1;
    slots.set(session!, Math.max(slots.get(session!) ?? 0, needed));
  }
  return slots;
}

function runnerRequest(op: string, session: string, slot: number, job: string, args: Record<string, unknown>) {
  const runner = { session, slot };
  switch (op) {
    case "claim":
      return call("POST", `/v1/runner/sessions/${session}/claim`, { runner: true, body: { slot } });
    case "end-session":
      return call("DELETE", `/v1/runner/sessions/${session}`, { runner: true });
    case "get":
      return call("GET", `/v1/runner/jobs/${job}`, { runner });
    case "connecting":
      return call("POST", `/v1/runner/jobs/${job}/connecting`, { runner, body: { connect_code: args.connect_code } });
    case "finish-game":
      return call("POST", `/v1/runner/jobs/${job}/finish-game`, {
        runner,
        body: { game_number: args.game_number, actual_stage: args.actual_stage, result: args.result },
      });
    case "fail":
      return call("POST", `/v1/runner/jobs/${job}/fail`, {
        runner,
        body: { error_code: args.error_code, retryable: args.retryable },
      });
    case "replay":
      return call("POST", `/v1/runner/jobs/${job}/replay`, { runner, body: args });
    default:
      return call("POST", `/v1/runner/jobs/${job}/${op}`, { runner });
  }
}

beforeEach(async () => {
  await resetQueue();
});

describe("golden transcripts from the Python service", () => {
  it("covers every recorded scenario", () => {
    expect(transcripts.length).toBe(18);
  });

  for (const transcript of transcripts) {
    it(transcript.name, async () => {
      let clock = START;
      await publish(POLICY);
      const needed = sessionAliases(transcript);
      await seedAccounts([...needed.values()].reduce((sum, slots) => sum + slots, 0) + 1);
      const sessions = new Map<string, string>();
      for (const [alias, slots] of needed) {
        const started = await call("POST", "/v1/runner/sessions", {
          runner: true,
          body: { host: alias, bundle_sha256: POLICY.bundle_sha256, git_sha: "transcript", slots, stream: false },
        });
        sessions.set(alias, started.body.session_id);
        await report(started.body.session_id, slots);
      }
      if (needed.size === 0) {
        const started = await call("POST", "/v1/runner/sessions", {
          runner: true,
          body: { host: "capacity", bundle_sha256: POLICY.bundle_sha256, git_sha: "transcript", slots: 1, stream: false },
        });
        sessions.set("capacity", started.body.session_id);
        await report(started.body.session_id, 1);
      }
      const aliases = new Aliases();

      for (const [index, step] of transcript.steps.entries()) {
        const where = `${transcript.name} step ${index}`;
        if (step.kind === "advance") {
          clock += step.seconds!;
          await setClock(clock);
          for (const [alias, id] of sessions) await report(id, needed.get(alias) ?? 1);
          await runAlarm();
          continue;
        }
        let actual;
        if (step.kind === "player") {
          const path = aliases.real(step.path!);
          const token = step.token == null ? undefined : aliases.real(step.token);
          actual = await call(step.method!, path, {
            token,
            body: step.body === null ? undefined : step.body,
            ip: `10.0.${Math.floor(index / 200)}.${index % 200}`,
          });
        } else {
          const [alias, slot] = step.worker!.split(":");
          const job = step.job == null ? "" : aliases.real(step.job);
          actual = await runnerRequest(step.op!, sessions.get(alias!)!, Number(slot ?? 0), job, step.args ?? {});
        }
        aliases.learn(actual.body);
        const expected = step.response!;
        expect(actual.status, where).toBe(expected.status);
        const pydanticList = Array.isArray(expected.body?.detail);
        if (pydanticList) continue;
        const mode = expected.status >= 400 ? "full" : step.kind === "player" ? "full" : step.compare;
        if (mode === "full") expect(aliases.normalize(actual.body), where).toEqual(expected.body);
        if (mode === "status_field") expect(actual.body.status, where).toBe(expected.body.status);
      }
    });
  }
});
```

- [ ] **Step 2: Run the replay**

Run: `npx vitest run test/transcripts.test.ts`
Expected: 19 passed (the count check plus 18 scenarios).

A failure names the scenario and step. Fix the TypeScript port to match the transcript. Do not edit a transcript by hand. If the Python behavior itself is wrong, stop and report it; a behavior change is out of scope for this plan.

- [ ] **Step 3: Run the full suite and typecheck**

Run: `npx vitest run && npm run typecheck`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add web/netplay-api/test/transcripts.test.ts web/netplay-api/src
git commit -m "Replay Python queue transcripts against the worker"
```

---

### Task 10: Local development, deploy notes, and handoff checks

**Files:**
- Modify: `deploy/netplay/README.md` (add an "API Worker" section)
- Modify: `.gitignore` (add `.superpowers/`)

**Interfaces:** none new.

- [ ] **Step 1: Document the Worker**

Add to `deploy/netplay/README.md`, before "## Hosted run":

````markdown
## API Worker

`web/netplay-api` is the netplay queue: a Cloudflare Worker with one Durable
Object. It replaces `hal-netplay-api` once Plan B moves the runner onto it.

Local development:

```sh
cd web/netplay-api
npm install
cp .dev.vars.example .dev.vars   # fill in the digests of your dev tokens
npm run dev                      # http://localhost:8787
npm test
```

One-time Cloudflare setup:

1. The `20xx.xyz` zone is on the account. `wrangler.jsonc` routes `20xx.xyz/v1/*`
   to this Worker.
2. Create two Cloudflare Access applications: `20xx.xyz/v1/runner/*` with one
   service token per GPU box, and `20xx.xyz/v1/admin/*` for the owner's login.
3. Store the token digests:

   ```sh
   echo -n "$RUNNER_TOKEN" | sha256sum   # repeat per box; join with commas
   npx wrangler secret put RUNNER_TOKEN_SHA256
   npx wrangler secret put ADMIN_TOKEN_SHA256
   ```

Deploy with `npm run deploy` from `web/netplay-api`.
````

- [ ] **Step 2: Ignore brainstorm scratch files**

Append `.superpowers/` to `.gitignore`.

- [ ] **Step 3: Run the handoff checks**

```bash
(cd web/netplay-api && npm test && npm run typecheck)
uv run ruff format --check .
uv run ruff check .
uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts
uv run pytest -q -m "not integration"
```

Expected: all pass. Report each command's result, including any skip.

This plan does not touch replay extraction, wire format, controller input, session stepping, or offline/live parity, so the integration suite is not required here; Plan B runs it.

- [ ] **Step 4: Commit**

```bash
git add deploy/netplay/README.md .gitignore
git commit -m "Document the netplay API worker"
```
