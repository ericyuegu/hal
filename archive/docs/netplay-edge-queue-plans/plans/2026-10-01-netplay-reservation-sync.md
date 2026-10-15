# Netplay Reservation Sync Implementation Plan

Transport update: the approved [protocol-4 design](../../../deploy/netplay/free-tier-design.md)
replaces the external two-second reports with a shared host WebSocket and local
relay. Continuous-session ownership and menu semantics below still apply.
See [the current runbook](../../../deploy/netplay/README.md) for operation.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the reservation's edge-triggered transition protocol with player-owned settings, runner-owned snapshots, and one end command; keep the player connected in Slippi between games; cache the public read routes.

**Architecture:** The Worker keeps three statuses (`queued`, `assigned`, `ended`), stores the player's settings with a revision and the runner's latest snapshot by sequence number, and answers every runner report with the current reservation. The runner drives Slippi's direct-mode menus through one deterministic menu driver that hovers, holds, locks in, and probes the connection, and it reports what it observes every 2 s. The page renders from the reservation's status and the runner's phase.

**Tech Stack:** Cloudflare Worker + Durable Object (TypeScript, vitest-pool-workers), Python 3.14 runner (libmelee fork, httpx, pytest), Next.js page via vinext (React 19).

**Spec:** `docs/superpowers/specs/2026-10-01-netplay-reservation-sync-design.md`

## Global Constraints

- Runner protocol version: 3 (`RUNNER_PROTOCOL_VERSION` in `web/netplay-api/src/domain.ts` and `hal/netplay_service/queue_client.py` change together).
- Store schema version: 5; Durable Object instance `global-v5`. No migrations.
- Statuses: `queued`, `assigned`, `ended`. Phases: `booting`, `waiting_for_player`, `character_select`, `in_game`, `paused`.
- End reasons: `player_canceled`, `player_left`, `player_disconnected`, `no_show`, `idle_timeout`, `yielded`, `service_failure`.
- Game results (human's side): `win`, `loss`, `no_contest`. An interrupted game is never recorded.
- Timings: report every 2 s; lease 20 s, 60 s in `in_game`/`paused`; connect 60 s; lock-in hold 5 s, cap 30 s; connection probe every 2 s after lock-in; idle 300 s at character select; pause 60 s; presence 120 s (queued only), refreshed at most every 10 s; yield after 900 s assigned when the queue is non-empty; `MAX_ATTEMPTS = 2`.
- Cache: `/v1/capacity` 3 s, `/v1/options` 10 s, 200 responses only; the browser response keeps `Cache-Control: no-store`.
- `hal/sim` must not import `torch`; the model must not import `melee`.
- Comments state a reason, constraint, or invariant; no edit narration. No `Co-Authored-By` lines in commits; short commit messages.
- Before hand-off run `uv run ruff format --check .`, `uv run ruff check .`, `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts`, `uv run pytest -q -m "not integration"`, the Worker suite (`cd web/netplay-api && npx vitest run && npx tsc --noEmit`), and the page checks (`cd web/netplay && npx tsc --noEmit -p . && npx oxlint app lib && npx oxfmt --check app lib components`).

## Review Focus

1. A report that arrives after the reservation ended (player left mid-report): the runner must stop cleanly, not crash or requeue. Pinned in Task 3 (`report` refuses an ended job) and Task 8 (`ReservationLink` marks itself released on 409).
2. A settings change racing the lock-in: the bot must lock the newest revision it has seen when the hold expires, and `locked_revision` must name that revision. Pinned in Task 6 (`test_hold_restarts_on_change_and_locks_latest`).
3. Runner retries of `end` after a lost response: a second `end` from the same worker must return the job, not 409, both after an ended reason and after a retryable requeue. Pinned in Task 3.
4. A `finished_games` list that repeats a stored game with a different result must be refused, and one that repeats it identically must be accepted. Pinned in Task 3.
5. The player pressing Leave while the runner is between games must end the reservation without waiting for the idle timeout. Pinned in Task 6 (driver aborts through `connect_abandoned`) and Task 9 (`test_leave_at_character_select_ends_as_player_canceled`).

---

## File Structure

Worker (`web/netplay-api`):
- `src/domain.ts` — constants, status/phase/reason types, protocol version.
- `src/requests.ts` — request parsing for create, settings, report, end.
- `src/store.ts` — rewritten `JobStore` for the v5 table: lifecycle, settings, snapshots, games, requeue rule, presence, yield.
- `src/queue.ts` — Durable Object routes; WebSocket code removed.
- `src/http.ts` — routing; public-read cache.
- `src/policy.ts` — `optionsBody` without set fields.
- `src/env.ts`, `wrangler.jsonc`, `vitest.config.ts` — `EDGE_CACHE` binding.
- Tests: `test/store.test.ts` (lifecycle), `test/runner-routes.test.ts` (new), `test/routes.test.ts`, `test/costs.test.ts`, `test/cache.test.ts` (new); delete `test/transcripts.test.ts`, `test/transcripts/*.json` except `policy.json`, `test/live.test.ts`, `test/transitions.test.ts`, `test/retries.test.ts`.

Runner (Python):
- `hal/netplay_service/domain.py` — `JobStatus`, `Phase`, `EndReason`, `WindDown`, `GameResult`, `Settings`, `FinishedGame`, `Observed`, new `Job`; timing constants.
- `hal/netplay_service/queue_contract.py` — `RunnerQueue` protocol v3.
- `hal/netplay_service/queue_client.py` — protocol-3 `parse_job`, `RemoteQueue.report/end`; transition methods and `connect_live` removed.
- `hal/sim/netplay.py` — `DirectMenuDriver`, `DirectSelection`, player exceptions, Frozen Stadium carry-over, `read_frames(timeout_seconds=)`, character check owned by the driver.
- `hal/eval/netplay.py` — pause tolerance (`pause_seconds`, `on_pause`, `PausedTooLong`).
- `hal/netplay_service/reservation.py` (new) — `ReservationLink` (reporter thread + desired state) and `run_reservation` (the per-reservation loop).
- `hal/netplay_service/runner.py` — slot worker calls `run_reservation`; `_heartbeat`, `_LivePolicySettings`, `_ReservationLive`, `_setup`, `_run_reservation` removed; `_handle_reservation` maps exceptions to end reasons.
- `scripts/qualify_netplay_059.py` — new player routes.
- Tests: `tests/test_netplay_queue_client.py`, `tests/test_netplay_contract.py`, `tests/test_netplay_session.py`, `tests/test_netplay_direct_driver.py` (new), `tests/test_netplay_reservation.py` (new), `tests/test_netplay_runner.py`, `tests/test_netplay_realtime.py`, `tests/test_qualify_netplay_059.py`, `tests/test_netplay_direct_integration.py` (new, integration).

Page (`web/netplay`):
- `lib/netplay-api.ts` — v3 job view and routes.
- `app/page.tsx` — reservation view by phase; settings editing; finished games; Lock in now; Leave.
- `components/sentence.tsx` — stage blank on the main sentence; random stage option.

---

### Task 0: Commit the pending presence and copy work

The working tree holds tested, uncommitted work from earlier in the session (presence release, "HAL plays like" copy, Enter-to-play). Commit it so each later task diffs cleanly.

**Files:** already modified: `web/netplay-api/src/{domain,queue,store}.ts`, `web/netplay-api/test/{costs,routes,transcripts,transitions}.test.ts`, `web/netplay/app/page.tsx`, `web/netplay/components/sentence.tsx`, `docs/superpowers/specs/2026-09-27-netplay-edge-queue-design.md`.

- [ ] **Step 1: Verify**

Run: `cd web/netplay-api && npx vitest run && npx tsc --noEmit`
Expected: `124 passed`, no type errors.

Run: `cd web/netplay && npx tsc --noEmit -p . && npx oxlint app lib && npx oxfmt --check app/page.tsx components/sentence.tsx`
Expected: exit 0.

- [ ] **Step 2: Commit in two commits**

```bash
git add web/netplay-api/src web/netplay-api/test
git commit -m "Release reservations whose page is gone"
git add web/netplay/app/page.tsx web/netplay/components/sentence.tsx docs/superpowers/specs/2026-09-27-netplay-edge-queue-design.md
git commit -m "Say HAL plays like a player; Enter plays"
```

---

### Task 1: Cache the public read routes in the Worker

**Files:**
- Modify: `web/netplay-api/src/env.ts`, `web/netplay-api/src/http.ts`, `web/netplay-api/wrangler.jsonc`, `web/netplay-api/vitest.config.ts`
- Create: `web/netplay-api/test/cache.test.ts`

**Interfaces:**
- Produces: `servePublic(cache: Cache, request: Request, seconds: number, produce: () => Promise<Response>): Promise<Response>` exported from `src/http.ts`; `Env.EDGE_CACHE: "on" | "off"`.

- [ ] **Step 1: Write the failing test**

```ts
// web/netplay-api/test/cache.test.ts
import { describe, expect, it } from "vitest";
import { servePublic } from "../src/http";

function request(path: string): Request {
  return new Request(`https://cache-test.example${path}?${crypto.randomUUID()}`);
}

describe("public read cache", () => {
  it("serves a cached 200 without producing it again", async () => {
    const req = request("/v1/capacity");
    let produced = 0;
    const produce = async () => {
      produced += 1;
      return Response.json({ n: produced }, { headers: { "Cache-Control": "no-store" } });
    };
    const first = await servePublic(caches.default, req, 3, produce);
    const second = await servePublic(caches.default, req, 3, produce);
    expect(await first.json()).toEqual({ n: 1 });
    expect(await second.json()).toEqual({ n: 1 });
    expect(produced).toBe(1);
    expect(second.headers.get("Cache-Control")).toBe("no-store");
  });

  it("never caches an error", async () => {
    const req = request("/v1/options");
    let produced = 0;
    const produce = async () => {
      produced += 1;
      return Response.json({ detail: "no policy" }, { status: 503 });
    };
    await servePublic(caches.default, req, 10, produce);
    await servePublic(caches.default, req, 10, produce);
    expect(produced).toBe(2);
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd web/netplay-api && npx vitest run test/cache.test.ts`
Expected: FAIL — `servePublic` is not exported.

- [ ] **Step 3: Implement**

`src/env.ts`: add the binding.

```ts
  // "on" in deployment, "off" in tests whose assertions read fresh capacity.
  EDGE_CACHE: "on" | "off";
```

`src/http.ts`: add above `handle`:

```ts
const PUBLIC_CACHE_SECONDS: Readonly<Record<string, number>> = { "/v1/options": 10, "/v1/capacity": 3 };

/** Serve a public read from the colo cache so viewers do not each reach the Durable Object. */
export async function servePublic(
  cache: Cache,
  request: Request,
  seconds: number,
  produce: () => Promise<Response>,
): Promise<Response> {
  const key = new Request(request.url, { method: "GET" });
  const hit = await cache.match(key);
  if (hit !== undefined) {
    const copy = new Response(hit.body, hit);
    copy.headers.set("Cache-Control", "no-store");
    return copy;
  }
  const response = await produce();
  if (response.status === 200) {
    const stored = new Response(response.clone().body, response);
    stored.headers.set("Cache-Control", `public, max-age=${seconds}`);
    await cache.put(key, stored);
  }
  return response;
}
```

In `handle`, before the routing `try`, reject an invalid binding, and route the two reads through the cache:

```ts
  if (env.EDGE_CACHE !== "on" && env.EDGE_CACHE !== "off") throw new Error("EDGE_CACHE must be on or off");
```

```ts
    const seconds = method === "GET" ? PUBLIC_CACHE_SECONDS[path] : undefined;
    if (seconds !== undefined) {
      const produce = async () => respond(path === "/v1/options" ? await queue.options() : await queue.capacity());
      return env.EDGE_CACHE === "on" ? servePublic(caches.default, request, seconds, produce) : produce();
    }
```

Delete the two old `GET /v1/options` and `GET /v1/capacity` lines.

`wrangler.jsonc`: add `"vars": { "EDGE_CACHE": "on" },`. `vitest.config.ts`: add `EDGE_CACHE: "off",` to `bindings`.

- [ ] **Step 4: Run the suite**

Run: `cd web/netplay-api && npx vitest run && npx tsc --noEmit`
Expected: all pass, including `cache.test.ts`.

- [ ] **Step 5: Commit**

```bash
git add web/netplay-api
git commit -m "Cache options and capacity at the edge"
```

---

### Task 2: Worker domain and request parsing for protocol 3

**Files:**
- Modify: `web/netplay-api/src/domain.ts`, `web/netplay-api/src/requests.ts`, `web/netplay-api/src/policy.ts`
- Test: `web/netplay-api/test/domain.test.ts`

**Interfaces:**
- Produces (domain.ts): `RUNNER_PROTOCOL_VERSION = 3`; `type JobStatus = "queued" | "assigned" | "ended"`; `PHASES`, `type Phase`; `END_REASONS`, `type EndReason`; `RUNNER_END_REASONS`; `GAME_RESULTS`, `type GameResult`; `type WindDown = "player" | "yield"`; `IN_GAME_PHASES: ReadonlySet<Phase>`; `YIELD_AFTER_SECONDS = 900`; existing `LEASE_SECONDS`, `PLAYING_LEASE_SECONDS`, `MAX_ATTEMPTS`, `PLAYER_PRESENCE_SECONDS`, `PRESENCE_WRITE_SECONDS`. Removes `MAX_GAMES`, `CONNECT_TIMEOUT_SECONDS`, `IDLE_TIMEOUT_SECONDS`, `TERMINAL_STATUSES`.
- Produces (requests.ts): `parseCreate(raw, policy): CreateRequest` (adds `stage: string | null`); `parseSettingsUpdate(raw, policy): SettingsUpdate`; `parseReport(raw): ObservedReport`; `parseEnd(raw): { reason: EndReason; retryable: boolean }`. Removes `parsePolicyUpdate`, `parseRematch`.

- [ ] **Step 1: Write the failing tests** (append to `test/domain.test.ts`)

```ts
import { parseEnd, parseReport, parseSettingsUpdate } from "../src/requests";

describe("protocol 3 requests", () => {
  const policy = parsePolicyConfig(POLICY);

  it("parses a full report", () => {
    const report = parseReport({
      seq: 4, phase: "character_select", phase_seconds_left: 3.5, bot_code: "HAL#9000",
      seen_revision: 2, locked_revision: 1,
      finished_games: [{ number: 1, stage: "BATTLEFIELD", result: "no_contest" }],
    });
    expect(report.finished_games[0]).toEqual({ number: 1, stage: "BATTLEFIELD", result: "no_contest" });
  });

  it("refuses an unknown phase, result, or a locked revision ahead of the seen one", () => {
    const base = { seq: 1, phase: "in_game", phase_seconds_left: null, bot_code: null, seen_revision: 1, locked_revision: 1, finished_games: [] };
    expect(() => parseReport({ ...base, phase: "rematch_wait" })).toThrow("phase");
    expect(() => parseReport({ ...base, finished_games: [{ number: 1, stage: "BATTLEFIELD", result: "tie" }] })).toThrow("result");
    expect(() => parseReport({ ...base, locked_revision: 2 })).toThrow("locked_revision");
  });

  it("accepts only runner end reasons", () => {
    expect(parseEnd({ reason: "no_show", retryable: false })).toEqual({ reason: "no_show", retryable: false });
    expect(() => parseEnd({ reason: "player_left", retryable: false })).toThrow("reason");
  });

  it("parses partial settings and allows null stage and desired_return", () => {
    expect(parseSettingsUpdate({ stage: null, desired_return: null }, policy)).toEqual({ stage: null, desired_return: null });
    expect(() => parseSettingsUpdate({}, policy)).toThrow("provide");
    expect(() => parseSettingsUpdate({ revision: 3 }, policy)).toThrow("unexpected field");
  });
});
```

Add `parsePolicyConfig` and `POLICY` imports if missing (`../src/policy`, `./helpers`).

- [ ] **Step 2: Run to verify failure**

Run: `cd web/netplay-api && npx vitest run test/domain.test.ts`
Expected: FAIL — `parseReport` not exported.

- [ ] **Step 3: Implement**

`src/domain.ts` — replace the timing/status block (from `CONNECT_TIMEOUT_SECONDS` through `TERMINAL_STATUSES`) with:

```ts
export const MAX_ATTEMPTS = 2;
export const LEASE_SECONDS = 20;
// Dolphin keeps running locally through a short network outage, so a game in
// progress tolerates more silence than the other phases.
export const PLAYING_LEASE_SECONDS = 60;
export const SESSION_SILENCE_SECONDS = 30;
export const SESSION_LIVE_SECONDS = 5;
// A player's page polls its job every second. Background tabs can throttle
// timers to once a minute, so a page counts as gone only after two missed minutes.
export const PLAYER_PRESENCE_SECONDS = 120;
// Polls refresh presence at most this often, to bound Durable Object writes.
export const PRESENCE_WRITE_SECONDS = 10;
// A reservation yields its slot after this long once someone is waiting.
export const YIELD_AFTER_SECONDS = 15 * 60;
export const EVENT_RETENTION_SECONDS = 30 * 24 * 60 * 60;
export const QUEUE_CAP = 20;
export const MAX_BODY_BYTES = 16 * 1024;
// Any change to a runner route's request or response shape bumps this with
// RUNNER_PROTOCOL_VERSION in hal/netplay_service/queue_client.py.
export const RUNNER_PROTOCOL_VERSION = 3;

export type JobStatus = "queued" | "assigned" | "ended";
export const PHASES = ["booting", "waiting_for_player", "character_select", "in_game", "paused"] as const;
export type Phase = (typeof PHASES)[number];
export const IN_GAME_PHASES: ReadonlySet<Phase> = new Set<Phase>(["in_game", "paused"]);
export const END_REASONS = [
  "player_canceled", "player_left", "player_disconnected", "no_show", "idle_timeout", "yielded", "service_failure",
] as const;
export type EndReason = (typeof END_REASONS)[number];
export const RUNNER_END_REASONS: ReadonlySet<EndReason> = new Set<EndReason>([
  "player_canceled", "player_disconnected", "no_show", "idle_timeout", "yielded", "service_failure",
]);
export const GAME_RESULTS = ["win", "loss", "no_contest"] as const;
export type GameResult = (typeof GAME_RESULTS)[number];
export type WindDown = "player" | "yield";
```

Keep `HttpError`, `PLAYER_CODE`, `pyRepr`, `validatePlayerCode`, `workerId`, `sha256Hex`, `randomToken`, `sameDigest` unchanged.

`src/requests.ts` — keep `fields`, `str`, `num`, `int`, `bool`, `nullableNum`, `inRange`. Replace `CreateRequest`/`parseCreate` and everything after with:

```ts
import { END_REASONS, GAME_RESULTS, HttpError, PHASES, RUNNER_END_REASONS, type EndReason, type GameResult, type Phase } from "./domain";
import { checkChoice, type PolicyConfig } from "./policy";

export interface CreateRequest {
  player_code: string;
  character: string;
  imitation: string;
  stage: string | null;
  online_delay: number;
  desired_return: number | null;
  temperature: number;
}

export function parseCreate(raw: unknown, policy: PolicyConfig): CreateRequest {
  const value = fields(
    raw,
    ["player_code", "character", "imitation", "stage", "online_delay", "desired_return", "temperature"],
    ["player_code", "character", "imitation", "online_delay"],
  );
  const code = str(value.player_code, "player_code");
  if (code.length < 3 || code.length > 13) throw new HttpError(422, "player_code must have 3 to 13 characters");
  const desired =
    "desired_return" in value ? nullableNum(value.desired_return, "desired_return") : policy.default_desired_return;
  const stage = "stage" in value && value.stage !== null ? str(value.stage, "stage") : null;
  return {
    player_code: code,
    character: str(value.character, "character"),
    imitation: str(value.imitation, "imitation"),
    stage,
    online_delay: int(value.online_delay, "online_delay"),
    desired_return: desired === null ? null : inRange(desired, policy.desired_return_range, "desired_return"),
    temperature:
      "temperature" in value
        ? inRange(num(value.temperature, "temperature"), policy.temperature_range, "temperature")
        : policy.default_temperature,
  };
}

export interface SettingsUpdate {
  character?: string;
  imitation?: string;
  stage?: string | null;
  desired_return?: number | null;
  temperature?: number;
}

export function parseSettingsUpdate(raw: unknown, policy: PolicyConfig): SettingsUpdate {
  const value = fields(raw, ["character", "imitation", "stage", "desired_return", "temperature"], []);
  if (Object.keys(value).length === 0) throw new HttpError(422, "provide at least one setting");
  const update: SettingsUpdate = {};
  if ("character" in value) update.character = checkChoice(policy.characters, str(value.character, "character"), "character");
  if ("imitation" in value) update.imitation = checkChoice(policy.imitations, str(value.imitation, "imitation"), "imitation");
  if ("stage" in value) update.stage = value.stage === null ? null : checkChoice(policy.stages, str(value.stage, "stage"), "stage");
  if ("desired_return" in value) {
    const desired = nullableNum(value.desired_return, "desired_return");
    update.desired_return = desired === null ? null : inRange(desired, policy.desired_return_range, "desired_return");
  }
  if ("temperature" in value) update.temperature = inRange(num(value.temperature, "temperature"), policy.temperature_range, "temperature");
  return update;
}

export interface FinishedGame {
  number: number;
  stage: string;
  result: GameResult;
}

export interface ObservedReport {
  seq: number;
  phase: Phase;
  phase_seconds_left: number | null;
  bot_code: string | null;
  seen_revision: number;
  locked_revision: number | null;
  finished_games: FinishedGame[];
}

function oneOf<T extends string>(value: unknown, allowed: readonly T[], name: string): T {
  if (typeof value !== "string" || !(allowed as readonly string[]).includes(value)) {
    throw new HttpError(422, `${name} must be one of ${allowed.join(", ")}`);
  }
  return value as T;
}

export function parseReport(raw: unknown): ObservedReport {
  const names = ["seq", "phase", "phase_seconds_left", "bot_code", "seen_revision", "locked_revision", "finished_games"];
  const value = fields(raw, names, names);
  const seq = int(value.seq, "seq");
  const seen = int(value.seen_revision, "seen_revision");
  const locked = value.locked_revision === null ? null : int(value.locked_revision, "locked_revision");
  if (seq < 1 || seen < 1) throw new HttpError(422, "seq and seen_revision must be positive");
  if (locked !== null && (locked < 1 || locked > seen)) throw new HttpError(422, "locked_revision must be in [1, seen_revision]");
  const left = value.phase_seconds_left === null ? null : num(value.phase_seconds_left, "phase_seconds_left");
  if (left !== null && left < 0) throw new HttpError(422, "phase_seconds_left must be non-negative");
  if (!Array.isArray(value.finished_games)) throw new HttpError(422, "finished_games must be a list");
  const games = value.finished_games.map((item, index) => {
    const game = fields(item, ["number", "stage", "result"], ["number", "stage", "result"]);
    const number = int(game.number, "game number");
    if (number !== index + 1) throw new HttpError(422, "finished_games must be numbered 1, 2, 3, …");
    return { number, stage: str(game.stage, "game stage"), result: oneOf(game.result, GAME_RESULTS, "game result") };
  });
  return {
    seq,
    phase: oneOf(value.phase, PHASES, "phase"),
    phase_seconds_left: left,
    bot_code: value.bot_code === null ? null : str(value.bot_code, "bot_code"),
    seen_revision: seen,
    locked_revision: locked,
    finished_games: games,
  };
}

export function parseEnd(raw: unknown): { reason: EndReason; retryable: boolean } {
  const value = fields(raw, ["reason", "retryable"], ["reason", "retryable"]);
  const reason = oneOf(value.reason, END_REASONS, "reason");
  if (!RUNNER_END_REASONS.has(reason)) throw new HttpError(422, `reason ${reason} is not a runner reason`);
  return { reason, retryable: bool(value.retryable, "retryable") };
}
```

`src/policy.ts` — `optionsBody` drops `max_games`, `no_show_seconds`, `rematch_seconds`; remove their imports.

- [ ] **Step 4: Run**

Run: `cd web/netplay-api && npx vitest run test/domain.test.ts`
Expected: the new tests pass. (Other files fail to compile until Task 3; that is expected and fixed there.)

- [ ] **Step 5: Commit**

```bash
git add web/netplay-api/src/domain.ts web/netplay-api/src/requests.ts web/netplay-api/src/policy.ts web/netplay-api/test/domain.test.ts
git commit -m "Define protocol 3 request shapes"
```

---

### Task 3: Rewrite the job store for the v5 lifecycle

**Files:**
- Rewrite: `web/netplay-api/src/store.ts`
- Rewrite test: `web/netplay-api/test/store.test.ts`; delete `test/transitions.test.ts`
- Modify: `web/netplay-api/test/helpers.ts` (`CHOICES` → `NEW_JOB`)

**Interfaces:**
- Consumes: Task 2 types.
- Produces:
  - `interface Settings { revision; character; imitation; stage: string | null; desired_return: number | null; temperature }`
  - `interface JobView { id; player_code; online_delay; status: JobStatus; end_reason: EndReason | null; queue_position: number | null; attempt; settings: Settings; observed: { seq; phase: Phase; bot_code: string | null; seen_revision; locked_revision: number | null } | null; phase_deadline: number | null; games: FinishedGame[]; wind_down: WindDown | null; lock_requests: number }`
  - `class JobStore` with: `createJob(id, digest, request: CreateRequest): JobView`, `getJob(id, digest): JobView`, `updateSettings(id, digest, update: SettingsUpdate): JobView`, `requestLock(id, digest): JobView`, `leave(id, digest): JobView`, `claimNext(worker): JobView | null`, `heldLease(worker): JobView | null`, `hasLease(worker): boolean`, `report(id, worker, report: ObservedReport): { view: JobView; phaseChanged: boolean; newGames: FinishedGame[] }`, `end(id, worker, reason, retryable): JobView`, `workerJob(id, worker): JobView`, `recordReplay(...)`, `failWorkers(workers): string[]`, `reapExpired(): string[]`, `applyYield(): string[]`, `nextDeadline(): number | null`, `queueDepth(): number`, `activeCount(): number`, `row(id): Row | null`, `view(row): JobView`.

- [ ] **Step 1: Write the failing store tests**

Replace `test/store.test.ts` with lifecycle tests. `helpers.ts`: replace `CHOICES` with

```ts
export const NEW_JOB = {
  player_code: "CRYO#610", character: "FOX", imitation: "IBDW#0", stage: null,
  online_delay: 2, desired_return: 20, temperature: 1,
};
```

and keep `withStore`. Tests:

```ts
import { describe, expect, it } from "vitest";
import { HttpError } from "../src/domain";
import type { ObservedReport } from "../src/requests";
import { NEW_JOB, withStore } from "./helpers";

function refused(fn: () => unknown): { status: number; detail: string } {
  try { fn(); } catch (error) { if (error instanceof HttpError) return { status: error.status, detail: error.detail }; throw error; }
  throw new Error("expected an HttpError");
}

function snapshot(seq: number, extra: Partial<ObservedReport> = {}): ObservedReport {
  return { seq, phase: "character_select", phase_seconds_left: 5, bot_code: "HAL#9000", seen_revision: 1, locked_revision: null, finished_games: [], ...extra };
}

describe("job lifecycle", () => {
  it("creates settings at revision 1 and bumps it on each change", () =>
    withStore((store) => {
      store.createJob("j1", "d1", NEW_JOB);
      expect(store.getJob("j1", "d1").settings).toMatchObject({ revision: 1, character: "FOX", stage: null });
      const next = store.updateSettings("j1", "d1", { character: "FALCO", stage: "POKEMON_STADIUM" });
      expect(next.settings).toMatchObject({ revision: 2, character: "FALCO", stage: "POKEMON_STADIUM" });
    }));

  it("claims FIFO with a fresh snapshot and a 20 s lease", () =>
    withStore((store, clock) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.createJob("j2", "d2", { ...NEW_JOB, player_code: "AAAA#1" });
      expect(store.claimNext("w0")).toMatchObject({ id: "j1", status: "assigned", attempt: 1, observed: null });
      expect(store.row("j1")).toMatchObject({ lease_owner: "w0", lease_expires_at: clock.now + 20, assigned_at: clock.now });
    }));

  it("keeps the newest snapshot by seq and lengthens the lease in game", () =>
    withStore((store, clock) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.claimNext("w0");
      store.report("j1", "w0", snapshot(2, { phase: "in_game", phase_seconds_left: null }));
      expect(store.row("j1")?.lease_expires_at).toBe(clock.now + 60);
      const stale = store.report("j1", "w0", snapshot(1));
      expect(stale.view.observed?.phase).toBe("in_game");
      expect(stale.phaseChanged).toBe(false);
    }));

  it("converts the phase's seconds left to an absolute deadline", () =>
    withStore((store, clock) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.claimNext("w0");
      expect(store.report("j1", "w0", snapshot(1, { phase_seconds_left: 4.5 })).view.phase_deadline).toBe(clock.now + 4.5);
    }));

  it("records each game once and refuses a changed result", () =>
    withStore((store) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.claimNext("w0");
      const game = { number: 1, stage: "BATTLEFIELD", result: "win" as const };
      expect(store.report("j1", "w0", snapshot(1, { finished_games: [game] })).newGames).toEqual([game]);
      expect(store.report("j1", "w0", snapshot(2, { finished_games: [game] })).newGames).toEqual([]);
      expect(refused(() => store.report("j1", "w0", snapshot(3, { finished_games: [{ ...game, result: "loss" }] }))).status).toBe(409);
      expect(store.getJob("j1", "d1").games).toEqual([game]);
    }));

  it("refuses a seen revision that does not exist yet", () =>
    withStore((store) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.claimNext("w0");
      expect(refused(() => store.report("j1", "w0", snapshot(1, { seen_revision: 2 }))).status).toBe(422);
    }));

  it("refuses reports and ends from another worker or after the end", () =>
    withStore((store) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.claimNext("w0");
      expect(refused(() => store.report("j1", "w1", snapshot(1))).status).toBe(409);
      store.end("j1", "w0", "no_show", false);
      expect(refused(() => store.report("j1", "w0", snapshot(2))).status).toBe(409);
      expect(refused(() => store.end("j1", "w1", "no_show", false)).status).toBe(409);
    }));

  it("returns the current job when the same worker repeats an end", () =>
    withStore((store) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.claimNext("w0");
      store.end("j1", "w0", "player_disconnected", false);
      expect(store.end("j1", "w0", "player_disconnected", false)).toMatchObject({ status: "ended", end_reason: "player_disconnected" });
      store.createJob("j2", "d2", { ...NEW_JOB, player_code: "AAAA#1" });
      store.claimNext("w0");
      expect(store.end("j2", "w0", "service_failure", true).status).toBe("queued");
      expect(store.end("j2", "w0", "service_failure", true).status).toBe("queued");
    }));

  it("requeues a retryable failure at the front once, then ends it", () =>
    withStore((store) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.createJob("j2", "d2", { ...NEW_JOB, player_code: "AAAA#1" });
      store.claimNext("w0");
      expect(store.end("j1", "w0", "service_failure", true)).toMatchObject({ status: "queued", queue_position: 1 });
      expect(store.claimNext("w0")?.id).toBe("j1");
      expect(store.end("j1", "w0", "service_failure", true)).toMatchObject({ status: "ended", end_reason: "service_failure" });
    }));

  it("does not requeue a failed reservation the player asked to stop", () =>
    withStore((store) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.claimNext("w0");
      store.leave("j1", "d1");
      expect(store.end("j1", "w0", "service_failure", true)).toMatchObject({ status: "ended", end_reason: "player_canceled" });
    }));

  it("ends a queued job on leave and winds down an assigned one", () =>
    withStore((store) => {
      store.createJob("j1", "d1", NEW_JOB);
      expect(store.leave("j1", "d1")).toMatchObject({ status: "ended", end_reason: "player_canceled" });
      store.createJob("j2", "d2", NEW_JOB);
      store.claimNext("w0");
      expect(store.leave("j2", "d2")).toMatchObject({ status: "assigned", wind_down: "player" });
    }));

  it("counts lock requests while assigned", () =>
    withStore((store) => {
      store.createJob("j1", "d1", NEW_JOB);
      expect(refused(() => store.requestLock("j1", "d1")).status).toBe(409);
      store.claimNext("w0");
      expect(store.requestLock("j1", "d1").lock_requests).toBe(1);
    }));

  it("requeues an expired lease once, then ends it", () =>
    withStore((store, clock) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.claimNext("w0");
      clock.advance(21);
      store.getJob("j1", "d1");
      expect(store.reapExpired()).toEqual(["j1"]);
      expect(store.row("j1")).toMatchObject({ status: "queued", retry_front: 1, lease_owner: null, phase: null });
      store.claimNext("w0");
      clock.advance(21);
      store.getJob("j1", "d1");
      store.reapExpired();
      expect(store.row("j1")).toMatchObject({ status: "ended", end_reason: "service_failure" });
    }));

  it("releases a queued job whose page stopped polling, but never an assigned one", () =>
    withStore((store, clock) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.createJob("j2", "d2", { ...NEW_JOB, player_code: "AAAA#1" });
      store.claimNext("w0");
      for (let t = 0; t < 120; t += 10) {
        clock.advance(10);
        store.report("j1", "w0", snapshot(t + 1));
      }
      expect(store.reapExpired()).toEqual(["j2"]);
      expect(store.row("j2")).toMatchObject({ status: "ended", end_reason: "player_left" });
      expect(store.row("j1")?.status).toBe("assigned");
    }));

  it("records a poll at most every 10 s", () =>
    withStore((store, clock) => {
      store.createJob("j1", "d1", NEW_JOB);
      const created = clock.now;
      clock.advance(9);
      store.getJob("j1", "d1");
      expect(store.row("j1")?.player_seen_at).toBe(created);
      clock.advance(1);
      store.getJob("j1", "d1");
      expect(store.row("j1")?.player_seen_at).toBe(clock.now);
    }));

  it("winds down a long reservation only when someone is waiting", () =>
    withStore((store, clock) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.claimNext("w0");
      clock.advance(900);
      expect(store.applyYield()).toEqual([]);
      store.createJob("j2", "d2", { ...NEW_JOB, player_code: "AAAA#1" });
      expect(store.applyYield()).toEqual(["j1"]);
      expect(store.row("j1")?.wind_down).toBe("yield");
      expect(store.applyYield()).toEqual([]);
    }));

  it("fails held jobs of an ended session by the same requeue rule", () =>
    withStore((store) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.claimNext("s/slot-0");
      expect(store.failWorkers(["s/slot-0"])).toEqual(["j1"]);
      expect(store.row("j1")?.status).toBe("queued");
    }));

  it("reports the earliest deadline", () =>
    withStore((store, clock) => {
      expect(store.nextDeadline()).toBeNull();
      store.createJob("j1", "d1", NEW_JOB);
      expect(store.nextDeadline()).toBe(clock.now + 120);
      store.claimNext("w0");
      expect(store.nextDeadline()).toBe(clock.now + 20);
    }));
});
```

Delete `test/transitions.test.ts`.

- [ ] **Step 2: Run to verify failure**

Run: `cd web/netplay-api && npx vitest run test/store.test.ts`
Expected: FAIL (old store API).

- [ ] **Step 3: Rewrite `src/store.ts`**

```ts
import {
  HttpError,
  IN_GAME_PHASES,
  LEASE_SECONDS,
  MAX_ATTEMPTS,
  PLAYER_PRESENCE_SECONDS,
  PLAYING_LEASE_SECONDS,
  PRESENCE_WRITE_SECONDS,
  YIELD_AFTER_SECONDS,
  sameDigest,
  type EndReason,
  type GameResult,
  type JobStatus,
  type Phase,
  type WindDown,
} from "./domain";
import type { CreateRequest, FinishedGame, ObservedReport, SettingsUpdate } from "./requests";

export type Row = Record<string, SqlStorageValue>;

export interface Settings {
  revision: number;
  character: string;
  imitation: string;
  stage: string | null;
  desired_return: number | null;
  temperature: number;
}

export interface JobView {
  id: string;
  player_code: string;
  online_delay: number;
  status: JobStatus;
  end_reason: EndReason | null;
  queue_position: number | null;
  attempt: number;
  settings: Settings;
  observed: { seq: number; phase: Phase; bot_code: string | null; seen_revision: number; locked_revision: number | null } | null;
  phase_deadline: number | null;
  games: FinishedGame[];
  wind_down: WindDown | null;
  lock_requests: number;
}

export const JOB_SCHEMA = `
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  token_digest TEXT NOT NULL,
  player_code TEXT NOT NULL,
  online_delay INTEGER NOT NULL CHECK (online_delay IN (2, 3)),
  status TEXT NOT NULL CHECK (status IN ('queued', 'assigned', 'ended')),
  end_reason TEXT,
  queue_seq INTEGER NOT NULL,
  retry_front INTEGER NOT NULL DEFAULT 0 CHECK (retry_front IN (0, 1)),
  attempt INTEGER NOT NULL DEFAULT 0,
  settings_revision INTEGER NOT NULL,
  character TEXT NOT NULL,
  imitation TEXT NOT NULL,
  stage TEXT,
  desired_return REAL,
  temperature REAL NOT NULL,
  observed_seq INTEGER NOT NULL DEFAULT 0,
  phase TEXT,
  bot_code TEXT,
  seen_revision INTEGER,
  locked_revision INTEGER,
  phase_deadline REAL,
  wind_down TEXT CHECK (wind_down IN ('player', 'yield')),
  lock_requests INTEGER NOT NULL DEFAULT 0,
  lease_owner TEXT,
  lease_expires_at REAL,
  last_worker TEXT,
  assigned_at REAL,
  player_seen_at REAL NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_one_active_player ON jobs(player_code) WHERE status IN ('queued', 'assigned');
CREATE INDEX IF NOT EXISTS idx_jobs_queue ON jobs(retry_front DESC, queue_seq) WHERE status = 'queued';
CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs(lease_expires_at) WHERE lease_owner IS NOT NULL;
CREATE TABLE IF NOT EXISTS games (
  job_id TEXT NOT NULL REFERENCES jobs(id),
  game_number INTEGER NOT NULL,
  stage TEXT NOT NULL,
  result TEXT NOT NULL,
  worker TEXT NOT NULL,
  replay_key TEXT,
  replay_sha256 TEXT,
  replay_size INTEGER,
  replay_etag TEXT,
  created_at REAL NOT NULL,
  PRIMARY KEY (job_id, game_number)
);
`;

// Clears everything the runner owned, for a requeue or a fresh claim.
const CLEAR_OBSERVED = `observed_seq = 0, phase = NULL, bot_code = NULL, seen_revision = NULL,
  locked_revision = NULL, phase_deadline = NULL`;

export class JobStore {
  constructor(
    private readonly sql: SqlStorage,
    private readonly now: () => number,
  ) {}

  protected first(query: string, ...params: SqlStorageValue[]): Row | null {
    return this.sql.exec<Row>(query, ...params).toArray()[0] ?? null;
  }

  protected exec(query: string, ...params: SqlStorageValue[]): void {
    this.sql.exec(query, ...params);
  }

  protected time(): number {
    return this.now();
  }

  row(id: string): Row | null {
    return this.first("SELECT * FROM jobs WHERE id = ?", id);
  }

  private reload(id: string): Row {
    const row = this.row(id);
    if (row === null) throw new Error(`job ${id} vanished inside its transaction`);
    return row;
  }

  view(row: Row): JobView {
    const status = row.status as JobStatus;
    let position: number | null = null;
    if (status === "queued") {
      const ahead = this.first(
        `SELECT COUNT(*) AS n FROM jobs WHERE status = 'queued'
           AND (retry_front > ? OR (retry_front = ? AND queue_seq < ?))`,
        row.retry_front, row.retry_front, row.queue_seq,
      );
      position = 1 + Number(ahead?.n ?? 0);
    }
    const games = this.sql
      .exec<Row>("SELECT game_number, stage, result FROM games WHERE job_id = ? ORDER BY game_number", row.id)
      .toArray()
      .map((game) => ({ number: game.game_number as number, stage: game.stage as string, result: game.result as GameResult }));
    return {
      id: row.id as string,
      player_code: row.player_code as string,
      online_delay: row.online_delay as number,
      status,
      end_reason: row.end_reason as EndReason | null,
      queue_position: position,
      attempt: row.attempt as number,
      settings: {
        revision: row.settings_revision as number,
        character: row.character as string,
        imitation: row.imitation as string,
        stage: row.stage as string | null,
        desired_return: row.desired_return as number | null,
        temperature: row.temperature as number,
      },
      observed:
        row.phase === null
          ? null
          : {
              seq: row.observed_seq as number,
              phase: row.phase as Phase,
              bot_code: row.bot_code as string | null,
              seen_revision: row.seen_revision as number,
              locked_revision: row.locked_revision as number | null,
            },
      phase_deadline: row.phase_deadline as number | null,
      games,
      wind_down: row.wind_down as WindDown | null,
      lock_requests: row.lock_requests as number,
    };
  }

  createJob(id: string, tokenDigest: string, request: CreateRequest): JobView {
    const now = this.time();
    const seq = Number(this.first("SELECT COALESCE(MAX(queue_seq), 0) + 1 AS seq FROM jobs")?.seq);
    try {
      this.exec(
        `INSERT INTO jobs(id, token_digest, player_code, online_delay, status, queue_seq, settings_revision,
           character, imitation, stage, desired_return, temperature, player_seen_at, created_at, updated_at)
         VALUES (?, ?, ?, ?, 'queued', ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)`,
        id, tokenDigest, request.player_code, request.online_delay, seq, request.character, request.imitation,
        request.stage, request.desired_return, request.temperature, now, now, now,
      );
    } catch (error) {
      if (error instanceof Error && error.message.includes("jobs.player_code")) {
        throw new HttpError(409, `player ${request.player_code} already has an active reservation`);
      }
      throw error;
    }
    return this.view(this.reload(id));
  }

  authenticated(id: string, digest: string): Row {
    const row = this.row(id);
    if (row === null || !sameDigest(row.token_digest as string, digest)) throw new HttpError(404, "job not found");
    return row;
  }

  getJob(id: string, digest: string): JobView {
    const row = this.authenticated(id, digest);
    const now = this.time();
    if (now - (row.player_seen_at as number) < PRESENCE_WRITE_SECONDS) return this.view(row);
    this.exec("UPDATE jobs SET player_seen_at = ? WHERE id = ?", now, id);
    return this.view(this.reload(id));
  }

  updateSettings(id: string, digest: string, update: SettingsUpdate): JobView {
    const row = this.authenticated(id, digest);
    if (row.status === "ended") throw new HttpError(409, "cannot change a finished reservation");
    this.exec(
      `UPDATE jobs SET character = ?, imitation = ?, stage = ?, desired_return = ?, temperature = ?,
         settings_revision = settings_revision + 1, updated_at = ? WHERE id = ?`,
      update.character ?? (row.character as string),
      update.imitation ?? (row.imitation as string),
      "stage" in update ? (update.stage ?? null) : (row.stage as string | null),
      "desired_return" in update ? (update.desired_return ?? null) : (row.desired_return as number | null),
      update.temperature ?? (row.temperature as number),
      this.time(),
      id,
    );
    return this.view(this.reload(id));
  }

  requestLock(id: string, digest: string): JobView {
    const row = this.authenticated(id, digest);
    if (row.status !== "assigned") throw new HttpError(409, "only an assigned reservation can lock in");
    this.exec("UPDATE jobs SET lock_requests = lock_requests + 1, updated_at = ? WHERE id = ?", this.time(), id);
    return this.view(this.reload(id));
  }

  leave(id: string, digest: string): JobView {
    const row = this.authenticated(id, digest);
    if (row.status === "queued") this.finish(id, "player_canceled");
    else if (row.status === "assigned" && row.wind_down !== "player") {
      this.exec("UPDATE jobs SET wind_down = 'player', updated_at = ? WHERE id = ?", this.time(), id);
    }
    return this.view(this.reload(id));
  }

  queueDepth(): number {
    return Number(this.first("SELECT COUNT(*) AS n FROM jobs WHERE status = 'queued'")?.n);
  }

  activeCount(): number {
    return Number(this.first("SELECT COUNT(*) AS n FROM jobs WHERE status = 'assigned'")?.n);
  }

  private owned(id: string, worker: string): Row {
    const row = this.row(id);
    if (row === null || row.lease_owner !== worker || row.status !== "assigned") {
      throw new HttpError(409, "worker does not own this job");
    }
    return row;
  }

  workerJob(id: string, worker: string): JobView {
    return this.view(this.owned(id, worker));
  }

  // A repeated claim after a lost response returns the job the slot already holds,
  // but only before the runner reported progress; afterwards the slot is busy.
  heldLease(worker: string): JobView | null {
    const held = this.first("SELECT * FROM jobs WHERE lease_owner = ? AND status = 'assigned'", worker);
    if (held === null) return null;
    if (held.phase === null) return this.view(held);
    throw new HttpError(409, `slot already holds job ${held.id}`);
  }

  hasLease(worker: string): boolean {
    return this.first("SELECT 1 FROM jobs WHERE lease_owner = ? AND status = 'assigned'", worker) !== null;
  }

  claimNext(worker: string): JobView | null {
    const now = this.time();
    const row = this.first("SELECT id FROM jobs WHERE status = 'queued' ORDER BY retry_front DESC, queue_seq ASC LIMIT 1");
    if (row === null) return null;
    this.exec(
      `UPDATE jobs SET status = 'assigned', retry_front = 0, attempt = attempt + 1, lease_owner = ?, last_worker = ?,
         lease_expires_at = ?, assigned_at = ?, ${CLEAR_OBSERVED}, updated_at = ? WHERE id = ?`,
      worker, worker, now + LEASE_SECONDS, now, now, row.id as string,
    );
    return this.view(this.reload(row.id as string));
  }

  report(id: string, worker: string, report: ObservedReport): { view: JobView; phaseChanged: boolean; newGames: FinishedGame[] } {
    const row = this.owned(id, worker);
    if (report.seen_revision > (row.settings_revision as number)) {
      throw new HttpError(422, "seen_revision is ahead of the reservation's settings");
    }
    const now = this.time();
    const fresh = report.seq > (row.observed_seq as number);
    const phase = fresh ? report.phase : (row.phase as Phase | null);
    if (fresh) {
      this.exec(
        `UPDATE jobs SET observed_seq = ?, phase = ?, bot_code = ?, seen_revision = ?, locked_revision = ?,
           phase_deadline = ? WHERE id = ?`,
        report.seq, report.phase, report.bot_code, report.seen_revision, report.locked_revision,
        report.phase_seconds_left === null ? null : now + report.phase_seconds_left, id,
      );
    }
    const newGames = report.finished_games.filter((game) => this.recordGame(id, worker, game, now));
    const lease = phase !== null && IN_GAME_PHASES.has(phase) ? PLAYING_LEASE_SECONDS : LEASE_SECONDS;
    this.exec("UPDATE jobs SET lease_expires_at = ?, updated_at = ? WHERE id = ?", now + lease, now, id);
    return { view: this.view(this.reload(id)), phaseChanged: fresh && report.phase !== row.phase, newGames };
  }

  // Returns true when the game is new.
  private recordGame(id: string, worker: string, game: FinishedGame, now: number): boolean {
    const stored = this.first("SELECT stage, result FROM games WHERE job_id = ? AND game_number = ?", id, game.number);
    if (stored !== null) {
      if (stored.stage === game.stage && stored.result === game.result) return false;
      throw new HttpError(409, `game ${game.number} is already recorded with a different result`);
    }
    this.exec(
      "INSERT INTO games(job_id, game_number, stage, result, worker, created_at) VALUES (?, ?, ?, ?, ?, ?)",
      id, game.number, game.stage, game.result, worker, now,
    );
    return true;
  }

  end(id: string, worker: string, reason: EndReason, retryable: boolean): JobView {
    const row = this.row(id);
    if (row !== null && row.last_worker === worker && row.lease_owner === null) {
      // A retry after a lost response: the first call already ended or requeued the job.
      if (row.status === "ended" || (row.status === "queued" && row.retry_front === 1)) return this.view(row);
    }
    this.owned(id, worker);
    if (reason === "service_failure" && retryable) this.requeueOrEnd(id);
    else this.finish(id, reason);
    return this.view(this.reload(id));
  }

  private finish(id: string, reason: EndReason): void {
    this.exec(
      `UPDATE jobs SET status = 'ended', end_reason = ?, lease_owner = NULL, lease_expires_at = NULL,
         phase_deadline = NULL, updated_at = ? WHERE id = ?`,
      reason, this.time(), id,
    );
  }

  // One rule for every runner failure: requeue at the front once, unless the player asked to stop.
  private requeueOrEnd(id: string): void {
    const row = this.reload(id);
    if (row.wind_down === "player") return this.finish(id, "player_canceled");
    if (row.wind_down === "yield") return this.finish(id, "yielded");
    if ((row.attempt as number) >= MAX_ATTEMPTS) return this.finish(id, "service_failure");
    this.exec(
      `UPDATE jobs SET status = 'queued', retry_front = 1, lease_owner = NULL, lease_expires_at = NULL,
         assigned_at = NULL, ${CLEAR_OBSERVED}, updated_at = ? WHERE id = ?`,
      this.time(), id,
    );
  }

  recordReplay(id: string, worker: string, gameNumber: number, key: string, sha256: string, size: number, etag: string): JobView {
    const game = this.first(
      "SELECT worker, replay_key, replay_sha256, replay_size, replay_etag FROM games WHERE job_id = ? AND game_number = ?",
      id, gameNumber,
    );
    if (game === null) throw new HttpError(409, "game is absent");
    if (game.worker !== worker) throw new HttpError(409, "worker did not play this game");
    const same = game.replay_key === key && game.replay_sha256 === sha256 && game.replay_size === size && game.replay_etag === etag;
    if (!same) {
      if (game.replay_key !== null) throw new HttpError(409, "game already has a different replay");
      this.exec(
        `UPDATE games SET replay_key = ?, replay_sha256 = ?, replay_size = ?, replay_etag = ?
         WHERE job_id = ? AND game_number = ? AND replay_key IS NULL`,
        key, sha256, size, etag, id, gameNumber,
      );
    }
    return this.view(this.reload(id));
  }

  // Returns the IDs of jobs whose leases were closed.
  failWorkers(workers: readonly string[]): string[] {
    if (workers.length === 0) return [];
    const marks = workers.map(() => "?").join(",");
    const ids = this.sql
      .exec<Row>(`SELECT id FROM jobs WHERE status = 'assigned' AND lease_owner IN (${marks}) ORDER BY queue_seq`, ...workers)
      .toArray()
      .map((row) => row.id as string);
    for (const id of ids) this.requeueOrEnd(id);
    return ids;
  }

  // Returns the IDs of jobs that changed.
  reapExpired(): string[] {
    const now = this.time();
    const gone = this.sql
      .exec<Row>("SELECT id FROM jobs WHERE status = 'queued' AND player_seen_at <= ?", now - PLAYER_PRESENCE_SECONDS)
      .toArray()
      .map((row) => row.id as string);
    for (const id of gone) this.finish(id, "player_left");
    const expired = this.sql
      .exec<Row>("SELECT id FROM jobs WHERE status = 'assigned' AND lease_expires_at <= ?", now)
      .toArray()
      .map((row) => row.id as string);
    for (const id of expired) this.requeueOrEnd(id);
    return [...gone, ...expired];
  }

  // Returns the IDs of jobs newly asked to yield.
  applyYield(): string[] {
    if (this.queueDepth() === 0) return [];
    const ids = this.sql
      .exec<Row>(
        "SELECT id FROM jobs WHERE status = 'assigned' AND wind_down IS NULL AND assigned_at <= ?",
        this.time() - YIELD_AFTER_SECONDS,
      )
      .toArray()
      .map((row) => row.id as string);
    for (const id of ids) this.exec("UPDATE jobs SET wind_down = 'yield', updated_at = ? WHERE id = ?", this.time(), id);
    return ids;
  }

  nextDeadline(): number | null {
    const row = this.first(
      `SELECT MIN(t) AS t FROM (
         SELECT lease_expires_at AS t FROM jobs WHERE status = 'assigned' AND lease_expires_at IS NOT NULL
         UNION ALL SELECT player_seen_at + ${PLAYER_PRESENCE_SECONDS} FROM jobs WHERE status = 'queued'
         UNION ALL SELECT assigned_at + ${YIELD_AFTER_SECONDS} FROM jobs
           WHERE status = 'assigned' AND wind_down IS NULL AND assigned_at IS NOT NULL)`,
    );
    return (row?.t as number | null) ?? null;
  }
}
```

- [ ] **Step 4: Run**

Run: `cd web/netplay-api && npx vitest run test/store.test.ts`
Expected: all lifecycle tests pass.

- [ ] **Step 5: Commit**

```bash
git add web/netplay-api/src/store.ts web/netplay-api/test/store.test.ts web/netplay-api/test/helpers.ts
git rm web/netplay-api/test/transitions.test.ts
git commit -m "Rewrite the job store around settings and snapshots"
```

---

### Task 4: Wire the Durable Object and routes to protocol 3

**Files:**
- Modify: `web/netplay-api/src/queue.ts`, `web/netplay-api/src/http.ts`, `web/netplay-api/src/sessions.ts`
- Create: `web/netplay-api/test/runner-routes.test.ts`
- Modify: `web/netplay-api/test/routes.test.ts`, `test/costs.test.ts`, `test/sessions.test.ts`, `test/pairing.test.ts`, `test/runner-policy.test.ts`, `test/schema.test.ts`
- Delete: `test/transcripts.test.ts`, `test/transcripts/*.json` except `policy.json`, `test/live.test.ts`, `test/retries.test.ts`

**Interfaces:**
- Consumes: Task 3 `JobStore`, Task 2 parsers.
- Produces (Queue DO): `createJob(raw)`, `getJob(id, token)`, `updateSettings(id, token, raw)`, `requestLock(id, token)`, `leaveJob(id, token)`, `report(session, slot, id, raw)`, `endJob(session, slot, id, raw)`, `recordReplay(session, slot, id, raw)`, `workerJob(session, slot, id)`; `STORE_SCHEMA_VERSION = 5`.
- Routes: `PATCH /v1/jobs/:id/settings`, `POST /v1/jobs/:id/lock`, `DELETE /v1/jobs/:id`, `POST /v1/runner/jobs/:id/report`, `POST /v1/runner/jobs/:id/end`, `POST /v1/runner/jobs/:id/replay`, `GET /v1/runner/jobs/:id`.

- [ ] **Step 1: Write the failing route tests** (`test/runner-routes.test.ts`)

```ts
import { beforeEach, describe, expect, it } from "vitest";
import { CREATE, START, call, publish, report, resetQueue, runAlarm, seedAccounts, setClock, startSession } from "./helpers";

beforeEach(async () => {
  await resetQueue();
});

async function assigned() {
  await publish();
  await seedAccounts(2);
  const session = await startSession(1);
  const job = (await call("POST", "/v1/jobs", { body: CREATE })).body;
  const runner = { session, slot: 0 };
  const claimed = await call("POST", `/v1/runner/sessions/${session}/claim`, { runner: true, body: { slot: 0 } });
  expect(claimed.body).toMatchObject({ id: job.id, status: "assigned", settings: { revision: 1 } });
  return { session, job, runner };
}

function observed(seq: number, extra: Record<string, unknown> = {}) {
  return { seq, phase: "character_select", phase_seconds_left: 5, bot_code: "BOT0#1", seen_revision: 1, locked_revision: null, finished_games: [], ...extra };
}

describe("protocol 3 routes", () => {
  it("rejects a protocol 2 runner", async () => {
    await publish();
    await seedAccounts(1);
    const started = await call("POST", "/v1/runner/sessions", {
      runner: true,
      body: { protocol_version: 2, session_id: "s".repeat(22), host: "h", bundle_sha256: "0".repeat(64), git_sha: "g", slots: 1, stream: false },
    });
    expect(started.status).toBe(409);
  });

  it("delivers settings changes in the report response", async () => {
    const { job, runner } = await assigned();
    await call("PATCH", `/v1/jobs/${job.id}/settings`, { token: job.token, body: { character: "FALCO" } });
    const response = await call("POST", `/v1/runner/jobs/${job.id}/report`, { runner, body: observed(1) });
    expect(response.body.settings).toMatchObject({ revision: 2, character: "FALCO" });
  });

  it("shows the page the runner's phase, deadline, and games", async () => {
    const { job, runner } = await assigned();
    await call("POST", `/v1/runner/jobs/${job.id}/report`, {
      runner,
      body: observed(1, { finished_games: [{ number: 1, stage: "POKEMON_STADIUM", result: "loss" }] }),
    });
    const view = (await call("GET", `/v1/jobs/${job.id}`, { token: job.token })).body;
    expect(view).toMatchObject({
      status: "assigned",
      observed: { phase: "character_select", bot_code: "BOT0#1" },
      phase_deadline: START + 5,
      games: [{ number: 1, stage: "POKEMON_STADIUM", result: "loss" }],
    });
  });

  it("winds down on leave and ends when the runner says so", async () => {
    const { job, runner } = await assigned();
    expect((await call("DELETE", `/v1/jobs/${job.id}`, { token: job.token })).body.wind_down).toBe("player");
    const reported = await call("POST", `/v1/runner/jobs/${job.id}/report`, { runner, body: observed(1) });
    expect(reported.body.wind_down).toBe("player");
    const ended = await call("POST", `/v1/runner/jobs/${job.id}/end`, { runner, body: { reason: "player_canceled", retryable: false } });
    expect(ended.body).toMatchObject({ status: "ended", end_reason: "player_canceled" });
    expect((await call("POST", `/v1/runner/jobs/${job.id}/report`, { runner, body: observed(2) })).status).toBe(409);
  });

  it("counts lock requests and settings only from the job's token", async () => {
    const { job } = await assigned();
    expect((await call("POST", `/v1/jobs/${job.id}/lock`, { token: job.token })).body.lock_requests).toBe(1);
    expect((await call("POST", `/v1/jobs/${job.id}/lock`, { token: "wrong" })).status).toBe(404);
  });

  it("requeues a reservation whose runner went silent", async () => {
    const { session, job } = await assigned();
    await setClock(START + 21);
    await report(session, 1);
    expect(await runAlarm()).toBe(true);
    expect((await call("GET", `/v1/jobs/${job.id}`, { token: job.token })).body).toMatchObject({ status: "queued", queue_position: 1 });
  });

  it("removes the transition routes", async () => {
    const { job, runner } = await assigned();
    for (const action of ["connecting", "playing", "finish-game", "no-show", "fail", "forfeit", "heartbeat"]) {
      expect((await call("POST", `/v1/runner/jobs/${job.id}/${action}`, { runner, body: {} })).status).toBe(404);
    }
    expect((await call("POST", `/v1/jobs/${job.id}/rematch`, { token: job.token, body: {} })).status).toBe(404);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd web/netplay-api && npx vitest run test/runner-routes.test.ts`
Expected: FAIL.

- [ ] **Step 3: Implement**

`src/queue.ts`:
- `STORE_SCHEMA_VERSION = 5`.
- Remove `RunnerAction`, `LiveAttachment`, `fetch`, `openSockets`, `broadcastSettings`, `release`, `settingsMessage`, `webSocketMessage`, `webSocketClose`, and every `this.release(...)` call.
- `alarm()` body:

```ts
    this.tx(() => {
      const streamHolder = this.sessions.streamHolder();
      const ended = this.sessions.endSilent();
      for (const id of ended.sessions) this.events.log("session_ended", { session: id, reason: "silent" });
      for (const id of ended.jobs) this.events.log("job_released", { job: id, status: this.jobs.row(id)?.status, reason: "session_silent" });
      if (streamHolder !== null && ended.sessions.includes(streamHolder)) {
        this.events.log("stream_lease_released", { session: streamHolder, reason: "session_silent" });
      }
      for (const id of this.jobs.reapExpired()) {
        const row = this.jobs.row(id);
        this.events.log("job_released", { job: id, status: row?.status, reason: row?.end_reason ?? "lease_expired" });
      }
      for (const id of this.jobs.applyYield()) this.events.log("wind_down", { job: id, reason: "yield" });
      this.events.prune();
    });
    await this.scheduleAlarm();
```

- Player methods:

```ts
  async createJob(raw: unknown): Promise<ApiResult> {
    const id = randomToken(18);
    const token = randomToken(32);
    const digest = await sha256Hex(token);
    return this.run(() => {
      const policy = this.requirePolicy();
      const request = parseCreate(raw, policy);
      if (!policy.online_delays.includes(request.online_delay)) throw new HttpError(422, "online delay is unsupported by this policy");
      if (request.imitation === "MASKED" && !policy.masked_identity) throw new HttpError(422, "masked identity is unsupported by this policy");
      if (this.setting("paused") === "1") throw new HttpError(503, "The queue is paused. Try again shortly.");
      if (this.jobs.queueDepth() >= QUEUE_CAP) throw new HttpError(503, "The queue is full. Try again in a few minutes.");
      const capacity = this.sessions.capacity();
      if (capacity.service_status === "unavailable") throw new HttpError(503, "Game servers are unavailable. Try again shortly.");
      if (capacity.healthy_slots === 0) throw new HttpError(503, capacity.service_message);
      checkChoice(policy.characters, request.character, "character");
      checkChoice(policy.imitations, request.imitation, "imitation");
      if (request.stage !== null) checkChoice(policy.stages, request.stage, "stage");
      validatePlayerCode(request.player_code);
      const job = this.tx(() => {
        const created = this.jobs.createJob(id, digest, request);
        this.events.log("job_created", { job: id, character: request.character, imitation: request.imitation });
        return created;
      });
      return { ...job, token };
    }, { status: 201 });
  }

  async getJob(id: string, token: string): Promise<ApiResult> {
    const digest = await sha256Hex(token);
    return this.run(() => this.tx(() => this.jobs.getJob(id, digest)), { alarm: false });
  }

  async updateSettings(id: string, token: string, raw: unknown): Promise<ApiResult> {
    const digest = await sha256Hex(token);
    return this.run(() => {
      const update = parseSettingsUpdate(raw, this.requirePolicy());
      return this.tx(() => {
        const job = this.jobs.updateSettings(id, digest, update);
        this.events.log("settings_updated", { job: id, revision: job.settings.revision });
        return job;
      });
    }, { alarm: false });
  }

  async requestLock(id: string, token: string): Promise<ApiResult> {
    const digest = await sha256Hex(token);
    return this.run(() => this.tx(() => {
      const job = this.jobs.requestLock(id, digest);
      this.events.log("lock_requested", { job: id, count: job.lock_requests });
      return job;
    }), { alarm: false });
  }

  async leaveJob(id: string, token: string): Promise<ApiResult> {
    const digest = await sha256Hex(token);
    return this.run(() => this.tx(() => {
      const job = this.jobs.leave(id, digest);
      this.events.log(job.status === "ended" ? "job_ended" : "wind_down", { job: id, reason: job.end_reason ?? "player" });
      return job;
    }));
  }
```

`getJob` writes presence, so it runs inside `tx`; it keeps `alarm: false` because presence only moves a deadline later.

- Runner methods (replace `runnerJob`):

```ts
  async report(sessionId: string, slot: number, jobId: string, raw: unknown): Promise<ApiResult> {
    return this.run(() => {
      const observed = parseReport(raw);
      const policy = this.requirePolicy();
      for (const game of observed.finished_games) checkChoice(policy.stages, game.stage, "stage");
      return this.tx(() => {
        const worker = this.sessions.jobWorker(sessionId, slot);
        const { view, phaseChanged, newGames } = this.jobs.report(jobId, worker, observed);
        if (phaseChanged) this.events.log("phase_changed", { job: jobId, session: sessionId, slot, phase: observed.phase });
        for (const game of newGames) this.events.log("game_finished", { job: jobId, session: sessionId, slot, ...game });
        const yielded = this.jobs.applyYield();
        for (const id of yielded) this.events.log("wind_down", { job: id, reason: "yield" });
        // applyYield may have just set this job's wind_down; answer with the current row.
        return yielded.includes(jobId) ? this.jobs.view(this.jobs.row(jobId)!) : view;
      });
    });
  }

  async endJob(sessionId: string, slot: number, jobId: string, raw: unknown): Promise<ApiResult> {
    return this.run(() => {
      const { reason, retryable } = parseEnd(raw);
      return this.tx(() => {
        const worker = this.sessions.jobWorker(sessionId, slot);
        const job = this.jobs.end(jobId, worker, reason, retryable);
        this.events.log("job_ended", { job: jobId, session: sessionId, slot, reason, status: job.status });
        return job;
      });
    });
  }

  async recordReplay(sessionId: string, slot: number, jobId: string, raw: unknown): Promise<ApiResult> {
    return this.run(() => this.tx(() => {
      const value = fields(raw, ["game_number", "key", "sha256", "size", "etag"], ["game_number", "key", "sha256", "size", "etag"]);
      const worker = this.sessions.recordingWorker(sessionId, slot);
      const number = int(value.game_number, "game_number");
      const job = this.jobs.recordReplay(jobId, worker, number, str(value.key, "key"), str(value.sha256, "sha256"), int(value.size, "size"), str(value.etag, "etag"));
      this.events.log("replay_recorded", { job: jobId, session: sessionId, slot, game_number: number, key: value.key });
      return job;
    }), { alarm: false });
  }

  async workerJob(sessionId: string, slot: number, jobId: string): Promise<ApiResult> {
    return this.run(() => this.jobs.workerJob(jobId, this.sessions.jobWorker(sessionId, slot)), { alarm: false });
  }
```

`endSession` logs `job_released` instead of `job_failed`.

`src/http.ts` player block becomes:

```ts
    const job = path.match(/^\/v1\/jobs\/([^/]+)(\/settings|\/lock)?$/);
    if (job) {
      const [, id, suffix] = job as [string, string, string | undefined];
      const token = bearer(request);
      if (token === null) return failure(401, "job token is required");
      if (method === "PATCH" && suffix === "/settings") return respond(await queue.updateSettings(id, token, await readBody(request)));
      if (method === "POST" && suffix === "/lock") return respond(await queue.requestLock(id, token));
      if (method === "GET" && suffix === undefined) return respond(await queue.getJob(id, token));
      if (method === "DELETE" && suffix === undefined) return respond(await queue.leaveJob(id, token));
    }
```

Runner job block becomes:

```ts
      const runnerJob = path.match(/^\/v1\/runner\/jobs\/([^/]+)(?:\/(report|end|replay))?$/);
      if (runnerJob) {
        const [, id, action] = runnerJob as [string, string, string | undefined];
        const { session: sessionId, slot } = runnerSlot(request);
        if (method === "GET" && action === undefined) return respond(await queue.workerJob(sessionId, slot, id));
        if (method === "POST" && action === "report") return respond(await queue.report(sessionId, slot, id, await readBody(request)));
        if (method === "POST" && action === "end") return respond(await queue.endJob(sessionId, slot, id, await readBody(request)));
        if (method === "POST" && action === "replay") return respond(await queue.recordReplay(sessionId, slot, id, await readBody(request)));
      }
```

Remove `RUNNER_ACTIONS` and the `RunnerAction` import.

Test updates:
- Delete `transcripts.test.ts`, `live.test.ts`, `retries.test.ts`, and every `test/transcripts/*.json` except `policy.json`. The removed scenarios are covered by `store.test.ts` and `runner-routes.test.ts`.
- `routes.test.ts`: replace uses of `/policy`, `/rematch`, `connecting`, `playing`, `finish-game`, `fail`, `heartbeat` with `report`/`end`/`/settings`. The presence test added earlier becomes: a queued job whose page stops polling is `ended` with `end_reason: "player_left"` after `runAlarm()` at `START + 121`; the "frees the slot when the player's page leaves a rematch" test is deleted (presence no longer applies to assigned jobs).
- `costs.test.ts`: the world's live job becomes `jobs.claimNext(worker)` then `jobs.report("live-job", worker, { seq: 1, phase: "in_game", phase_seconds_left: null, bot_code: "BOT0#1", seen_revision: 1, locked_revision: 1, finished_games: [] })`; the historical insert uses the v5 columns (`status = 'ended'`, `end_reason = 'player_canceled'`, `settings_revision = 1`, `character`, `imitation`, `temperature`, `player_seen_at`). Replace "heartbeat" measurements with `report`. Re-record snapshots with `npx vitest run test/costs.test.ts -u` and check that no read count grows with history.
- `sessions.test.ts`, `pairing.test.ts`, `runner-policy.test.ts`, `schema.test.ts`: replace `protocol_version: 2` with `3`, `CHOICES` with `NEW_JOB`, transition calls with `report`/`end`, and `"failed"` expectations for session end with the requeue rule (`"queued"` on attempt 1).

- [ ] **Step 4: Run the whole Worker suite**

Run: `cd web/netplay-api && npx vitest run && npx tsc --noEmit`
Expected: all pass, no type errors.

- [ ] **Step 5: Commit**

```bash
git add -A web/netplay-api
git commit -m "Serve protocol 3 from the queue Worker"
```

---

### Task 5: Python domain and protocol-3 queue client

**Files:**
- Modify: `hal/netplay_service/domain.py`, `hal/netplay_service/queue_contract.py`, `hal/netplay_service/queue_client.py`
- Test: `tests/test_netplay_queue_client.py`, `tests/test_netplay_contract.py`

**Interfaces:**
- Produces (domain.py):

```python
CONNECT_TIMEOUT_SECONDS: Final[int] = 60
IDLE_TIMEOUT_SECONDS: Final[int] = 300
PAUSE_TIMEOUT_SECONDS: Final[int] = 60
LOCK_HOLD_SECONDS: Final[float] = 5.0
LOCK_HOLD_CAP_SECONDS: Final[float] = 30.0
CONNECTION_PROBE_SECONDS: Final[float] = 2.0
REPORT_INTERVAL_SECONDS: Final[float] = 2.0

class JobStatus(StrEnum): QUEUED = "queued"; ASSIGNED = "assigned"; ENDED = "ended"
class Phase(StrEnum): BOOTING = "booting"; WAITING_FOR_PLAYER = "waiting_for_player"; CHARACTER_SELECT = "character_select"; IN_GAME = "in_game"; PAUSED = "paused"
class EndReason(StrEnum): PLAYER_CANCELED = "player_canceled"; PLAYER_LEFT = "player_left"; PLAYER_DISCONNECTED = "player_disconnected"; NO_SHOW = "no_show"; IDLE_TIMEOUT = "idle_timeout"; YIELDED = "yielded"; SERVICE_FAILURE = "service_failure"
class WindDown(StrEnum): PLAYER = "player"; YIELD = "yield"
class GameResult(StrEnum): WIN = "win"; LOSS = "loss"; NO_CONTEST = "no_contest"

@dataclass(frozen=True, slots=True)
class Settings: revision: int; character: str; imitation: str; stage: str | None; desired_return: float | None; temperature: float
@dataclass(frozen=True, slots=True)
class FinishedGame: number: int; stage: str; result: GameResult
@dataclass(frozen=True, slots=True)
class Observed:
    seq: int; phase: Phase; phase_seconds_left: float | None; bot_code: str | None
    seen_revision: int; locked_revision: int | None; finished_games: tuple[FinishedGame, ...]
    def to_payload(self) -> dict[str, object]
@dataclass(frozen=True, slots=True)
class Job:
    id: str; player_code: str; online_delay: int; status: JobStatus; end_reason: EndReason | None
    queue_position: int | None; attempt: int; settings: Settings; phase: Phase | None
    phase_deadline: float | None; games: tuple[FinishedGame, ...]; wind_down: WindDown | None; lock_requests: int
```

`MatchChoices` stays (used by player clients to create jobs) and gains `stage: str | None = None` in place of `requested_stage`. `TERMINAL_STATUSES` is removed.

- Produces (queue_client.py): `RUNNER_PROTOCOL_VERSION = 3`; `parse_job(payload) -> Job`; `RemoteQueue.claim_next(worker_id) -> Job | None`, `.report(job_id, worker_id, observed: Observed) -> Job`, `.end(job_id, worker_id, reason: EndReason, *, retryable: bool) -> Job`, `.record_replay(...)`, `.get_worker_job(...)`, `.finish_pairing(...)`. Removed: `heartbeat`, `mark_*`, `finish_game`, `fail`, `forfeit_service_failure`, `connect_live`.
- Produces (queue_contract.py): `RunnerQueue` protocol with exactly the `RemoteQueue` methods above.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_netplay_queue_client.py`, replacing tests of removed methods)

```python
_JOB_V3 = {
    "id": "job",
    "player_code": "CRYO#610",
    "online_delay": 2,
    "status": "assigned",
    "end_reason": None,
    "queue_position": None,
    "attempt": 1,
    "settings": {"revision": 2, "character": "FALCO", "imitation": "MANG#0", "stage": None, "desired_return": 20.0, "temperature": 1.0},
    "observed": {"seq": 3, "phase": "in_game", "bot_code": "HAL#9000", "seen_revision": 2, "locked_revision": 2},
    "phase_deadline": None,
    "games": [{"number": 1, "stage": "BATTLEFIELD", "result": "win"}],
    "wind_down": None,
    "lock_requests": 0,
}


def test_parse_job_reads_protocol_3() -> None:
    job = parse_job(_JOB_V3)
    assert job.settings == Settings(2, "FALCO", "MANG#0", None, 20.0, 1.0)
    assert job.phase is Phase.IN_GAME
    assert job.games == (FinishedGame(1, "BATTLEFIELD", GameResult.WIN),)


def test_parse_job_refuses_changed_fields() -> None:
    with pytest.raises(QueueProtocolError, match="fields changed"):
        parse_job({**_JOB_V3, "game_count": 1})


def test_report_posts_the_snapshot_with_slot_headers() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_JOB_V3)

    queue = RemoteQueue(_ENDPOINT, "session", client=httpx.Client(base_url=_ENDPOINT.url, transport=httpx.MockTransport(handler)))
    observed = Observed(4, Phase.CHARACTER_SELECT, 3.0, "HAL#9000", 2, 1, (FinishedGame(1, "BATTLEFIELD", GameResult.WIN),))
    job = queue.report("job", "session/slot-0", observed)
    assert job.id == "job"
    assert seen[0].url.path == "/v1/runner/jobs/job/report"
    assert seen[0].headers["X-HAL-Slot"] == "0"
    assert json.loads(seen[0].content) == observed.to_payload()


def test_end_maps_409_to_invalid_transition() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": "worker does not own this job"})

    queue = RemoteQueue(_ENDPOINT, "session", client=httpx.Client(base_url=_ENDPOINT.url, transport=httpx.MockTransport(handler)))
    with pytest.raises(InvalidTransitionError):
        queue.end("job", "session/slot-0", EndReason.NO_SHOW, retryable=False)
```

Use the file's existing `_ENDPOINT` and imports; add `json`, `Settings`, `Phase`, `FinishedGame`, `GameResult`, `Observed`, `EndReason`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_netplay_queue_client.py`
Expected: FAIL on import of the new names.

- [ ] **Step 3: Implement**

`domain.py`: replace `JobStatus`, `TERMINAL_STATUSES`, the timeout constants, `MatchChoices.requested_stage`, and `Job` with the definitions in **Interfaces**, plus validation in `__post_init__`:

```python
@dataclass(frozen=True, slots=True)
class Settings:
    revision: int
    character: str
    imitation: str
    stage: str | None
    desired_return: float | None
    temperature: float

    def __post_init__(self) -> None:
        if self.revision < 1:
            raise ValueError("settings revision must be positive")
        validate_character(self.character)
        validate_imitation(self.imitation)
        if self.stage is not None:
            validate_stage(self.stage)
        validate_desired_return(self.desired_return)
        validate_temperature(self.temperature)


@dataclass(frozen=True, slots=True)
class FinishedGame:
    number: int
    stage: str
    result: GameResult

    def __post_init__(self) -> None:
        if self.number < 1:
            raise ValueError("game number must be positive")
        validate_stage(self.stage)


@dataclass(frozen=True, slots=True)
class Observed:
    """The runner's complete view of its reservation; the Worker keeps the newest by seq."""

    seq: int
    phase: Phase
    phase_seconds_left: float | None
    bot_code: str | None
    seen_revision: int
    locked_revision: int | None
    finished_games: tuple[FinishedGame, ...]

    def to_payload(self) -> dict[str, object]:
        return {
            "seq": self.seq,
            "phase": self.phase.value,
            "phase_seconds_left": self.phase_seconds_left,
            "bot_code": self.bot_code,
            "seen_revision": self.seen_revision,
            "locked_revision": self.locked_revision,
            "finished_games": [
                {"number": game.number, "stage": game.stage, "result": game.result.value} for game in self.finished_games
            ],
        }
```

`queue_client.py`: set `RUNNER_PROTOCOL_VERSION: Final = 3`; replace `_JOB_FIELDS` with `{"id", "player_code", "online_delay", "status", "end_reason", "queue_position", "attempt", "settings", "observed", "phase_deadline", "games", "wind_down", "lock_requests"}`; rewrite `parse_job`:

```python
_SETTINGS_FIELDS: Final = frozenset(("revision", "character", "imitation", "stage", "desired_return", "temperature"))
_OBSERVED_FIELDS: Final = frozenset(("seq", "phase", "bot_code", "seen_revision", "locked_revision"))
_GAME_FIELDS: Final = frozenset(("number", "stage", "result"))


def parse_job(payload: object) -> Job:
    """Read one job body; the player and runner routes return the same shape."""
    if not isinstance(payload, dict) or set(payload) != _JOB_FIELDS:
        raise QueueProtocolError("job response fields changed")
    fields = cast(dict[str, object], payload)
    try:
        settings = _object(fields["settings"], _SETTINGS_FIELDS, "settings")
        observed = None if fields["observed"] is None else _object(fields["observed"], _OBSERVED_FIELDS, "observed")
        games = fields["games"]
        if not isinstance(games, list):
            raise TypeError("games must be a list")
        return Job(
            id=_text(fields["id"]),
            player_code=validate_player_code(_text(fields["player_code"])),
            online_delay=_integer(fields["online_delay"]),
            status=JobStatus(_text(fields["status"])),
            end_reason=None if fields["end_reason"] is None else EndReason(_text(fields["end_reason"])),
            queue_position=_optional_integer(fields["queue_position"]),
            attempt=_integer(fields["attempt"]),
            settings=Settings(
                revision=_integer(settings["revision"]),
                character=_text(settings["character"]),
                imitation=_text(settings["imitation"]),
                stage=_optional_text(settings["stage"]),
                desired_return=_optional_number(settings["desired_return"]),
                temperature=_number(settings["temperature"]),
            ),
            phase=None if observed is None else Phase(_text(observed["phase"])),
            phase_deadline=_optional_number(fields["phase_deadline"]),
            games=tuple(
                FinishedGame(_integer(game["number"]), _text(game["stage"]), GameResult(_text(game["result"])))
                for game in (_object(item, _GAME_FIELDS, "game") for item in games)
            ),
            wind_down=None if fields["wind_down"] is None else WindDown(_text(fields["wind_down"])),
            lock_requests=_integer(fields["lock_requests"]),
        )
    except (TypeError, ValueError) as error:
        raise QueueProtocolError(f"job response contains invalid values: {error}") from error
```

Move `_object` above `parse_job`. Replace the transition methods of `RemoteQueue` with:

```python
    def report(self, job_id: str, worker_id: str, observed: Observed) -> Job:
        response = self._api.request(
            "POST", f"/v1/runner/jobs/{job_id}/report", body=observed.to_payload(), headers=self._slot_headers(worker_id)
        )
        return parse_job(_json(response))

    def end(self, job_id: str, worker_id: str, reason: EndReason, *, retryable: bool) -> Job:
        response = self._api.request(
            "POST",
            f"/v1/runner/jobs/{job_id}/end",
            body={"reason": reason.value, "retryable": retryable},
            headers=self._slot_headers(worker_id),
        )
        return parse_job(_json(response))
```

`record_replay` posts to `/v1/runner/jobs/{job_id}/replay` with the same body and ignores the parsed job. Delete `connect_live` and the `websockets` imports. Update `queue_contract.RunnerQueue` to declare `claim_next`, `report`, `end`, `record_replay`, `get_worker_job`, `finish_pairing`.

- [ ] **Step 4: Run**

Run: `uv run pytest -q tests/test_netplay_queue_client.py tests/test_netplay_contract.py`
Expected: pass. Fix `tests/test_netplay_contract.py` assertions about removed methods to the new protocol.

- [ ] **Step 5: Commit**

```bash
git add hal/netplay_service/domain.py hal/netplay_service/queue_contract.py hal/netplay_service/queue_client.py tests/test_netplay_queue_client.py tests/test_netplay_contract.py
git commit -m "Speak protocol 3 from the runner client"
```

---

### Task 6: Direct-mode menu driver and Frozen Stadium carry-over

**Files:**
- Modify: `hal/sim/netplay.py`
- Create: `tests/test_netplay_direct_driver.py`
- Modify: `tests/test_netplay_session.py`

**Interfaces:**
- Produces (hal/sim/netplay.py):

```python
class PlayerDisconnected(Exception): ...
class PlayerIdle(Exception): ...
class PlayerNoShow(Exception): ...

@dataclass(frozen=True, slots=True)
class DirectSelection:
    revision: int
    character: melee.Character
    costume: int
    stage: melee.Stage  # RANDOM_STAGE when the player chose random
    identity: str       # player identity the policy is conditioned on; part of the lock-in

class DirectMenuDriver:
    """MenuDriver for Slippi direct mode across consecutive games."""
    def __init__(self, *, opponent_code: str, selection: Callable[[], DirectSelection],
                 lock_requests: Callable[[], int], connect_timeout_seconds: float, idle_timeout_seconds: float,
                 hold_seconds: float, hold_cap_seconds: float, probe_interval_seconds: float,
                 on_change: Callable[[], None] = lambda: None,
                 clock: Callable[[], float] = time.monotonic) -> None
    # on_change runs on the menu thread whenever phase, deadline, or locked changes.
    phase: Literal["waiting_for_player", "character_select"]
    deadline: float | None        # monotonic seconds
    locked: DirectSelection | None
    def begin_game(self) -> None  # call before each start_match/start_rematch
    def __call__(self, state: melee.GameState | None, controller: melee.Controller) -> bool
```

- `NetplaySession.read_frames(timeout_seconds: float | None = None)`; `start_rematch` copies `frozen_stadium_selected` from the old helper into the new one; `_discover_ports` skips the character equality check when `menu_driver` is set (the driver owns the selection, and the runner checks `driver.locked`).

- [ ] **Step 1: Write the failing driver tests** (`tests/test_netplay_direct_driver.py`)

```python
from dataclasses import dataclass, field

import melee
import pytest

from hal.sim.netplay import DirectMenuDriver, DirectSelection, PlayerDisconnected, PlayerIdle, PlayerNoShow


@dataclass
class _Cursor:
    x: float = 0.0
    y: float = 0.0


@dataclass
class _Player:
    character: melee.Character = melee.Character.FOX
    costume: int = 0
    coin_down: bool = False
    cursor: _Cursor = field(default_factory=_Cursor)
    cpu_level: int = 0
    is_holding_cpu_slider: bool = False
    controller_status: melee.ControllerStatus = melee.ControllerStatus.CONTROLLER_HUMAN


@dataclass
class _State:
    menu_state: melee.Menu
    submenu: melee.SubMenu = melee.SubMenu.ONLINE_CSS
    frame: int = 0
    menu_selection: int = 0
    ready_to_start: int = 0
    players: dict[int, _Player] = field(default_factory=lambda: {1: _Player(), 2: _Player()})


class _Controller:
    port = 1

    def __init__(self) -> None:
        self.pressed: list[melee.Button] = []
        self.prev = melee.ControllerState()

    def press_button(self, button: melee.Button) -> None:
        self.pressed.append(button)
        self.prev.button[button] = True

    def release_button(self, button: melee.Button) -> None:
        self.prev.button[button] = False

    def release_all(self) -> None:
        for button in list(self.prev.button):
            self.prev.button[button] = False

    def tilt_analog(self, *_args: object) -> None:
        pass


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _driver(selection: list[DirectSelection], locks: list[int], clock: _Clock) -> DirectMenuDriver:
    return DirectMenuDriver(
        opponent_code="CRYO#610",
        selection=lambda: selection[-1],
        lock_requests=lambda: locks[-1],
        connect_timeout_seconds=60,
        idle_timeout_seconds=300,
        hold_seconds=5,
        hold_cap_seconds=30,
        probe_interval_seconds=2,
        clock=clock,
    )


FOX = DirectSelection(1, melee.Character.FOX, 0, melee.Stage.RANDOM_STAGE, "IBDW#0")
FALCO = DirectSelection(2, melee.Character.FALCO, 0, melee.Stage.POKEMON_STADIUM, "IBDW#0")


def _starts(controller: _Controller) -> int:
    return controller.pressed.count(melee.Button.BUTTON_START)


def _between_games(driver: DirectMenuDriver) -> None:
    driver.connected = True
    driver.begin_game()


def test_hold_restarts_on_change_and_locks_latest() -> None:
    clock, selection = _Clock(), [FOX]
    driver = _driver(selection, [0], clock)
    _between_games(driver)
    controller = _Controller()
    css = _State(melee.Menu.SLIPPI_ONLINE_CSS)
    driver(css, controller)
    assert driver.phase == "character_select" and driver.deadline == pytest.approx(105.0)
    clock.now = 103.0
    selection.append(FALCO)
    css.players[1].character = melee.Character.FALCO
    driver(css, controller)
    assert driver.deadline == pytest.approx(108.0)
    clock.now = 107.9
    driver(css, controller)
    assert driver.locked is None and _starts(controller) == 0
    clock.now = 108.1
    css.frame = 1
    driver(css, controller)
    assert driver.locked == FALCO and _starts(controller) == 1


def test_hold_is_capped() -> None:
    clock, selection = _Clock(), [FOX]
    driver = _driver(selection, [0], clock)
    _between_games(driver)
    css = _State(melee.Menu.SLIPPI_ONLINE_CSS, frame=1)
    controller = _Controller()
    for step in range(40):
        clock.now = 100.0 + step
        selection.append(DirectSelection(step + 2, melee.Character.FOX, 0, melee.Stage.RANDOM_STAGE, f"P{step}#1"))
        driver(css, controller)
        if driver.locked is not None:
            break
    assert driver.locked is not None and clock.now <= 131.0


def test_lock_request_locks_now() -> None:
    clock, locks = _Clock(), [0]
    driver = _driver([FOX], locks, clock)
    _between_games(driver)
    css = _State(melee.Menu.SLIPPI_ONLINE_CSS, frame=1)
    controller = _Controller()
    driver(css, controller)
    locks.append(1)
    driver(css, controller)
    assert driver.locked == FOX


def test_probe_presses_start_every_interval_after_lock() -> None:
    clock = _Clock()
    driver = _driver([FOX], [1], clock)
    _between_games(driver)
    css = _State(melee.Menu.SLIPPI_ONLINE_CSS, frame=1)
    controller = _Controller()
    driver(css, controller)
    assert _starts(controller) == 1
    clock.now = 101.0
    driver(css, controller)
    assert _starts(controller) == 1
    clock.now = 102.1
    driver(css, controller)
    assert _starts(controller) == 2


def test_code_entry_after_connection_means_disconnected() -> None:
    driver = _driver([FOX], [0], _Clock())
    _between_games(driver)
    with pytest.raises(PlayerDisconnected):
        driver(_State(melee.Menu.SLIPPI_ONLINE_CSS, submenu=melee.SubMenu.NAME_ENTRY_SUBMENU), _Controller())


def test_idle_timeout_at_character_select() -> None:
    clock = _Clock()
    driver = _driver([FOX], [0], clock)
    _between_games(driver)
    css = _State(melee.Menu.SLIPPI_ONLINE_CSS)
    driver(css, _Controller())
    clock.now = 400.1
    with pytest.raises(PlayerIdle):
        driver(css, _Controller())


def test_no_show_while_waiting_for_the_player() -> None:
    clock = _Clock()
    driver = _driver([FOX], [0], clock)
    driver.begin_game()
    driver(_State(melee.Menu.SLIPPI_ONLINE_CSS, submenu=melee.SubMenu.NAME_ENTRY_SUBMENU), _Controller())
    assert driver.phase == "waiting_for_player"
    clock.now = 160.1
    with pytest.raises(PlayerNoShow):
        driver(_State(melee.Menu.SLIPPI_ONLINE_CSS), _Controller())


def test_frozen_stadium_toggle_is_pressed_once_per_dolphin_session() -> None:
    clock, selection = _Clock(), [DirectSelection(1, melee.Character.FOX, 0, melee.Stage.POKEMON_STADIUM, "IBDW#0")]
    driver = _driver(selection, [0], clock)
    for _ in range(2):
        _between_games(driver)
        controller = _Controller()
        stage_select = _State(melee.Menu.STAGE_SELECT, frame=30)
        stage_select.players[1].cursor = _Cursor(15, 3.5)
        for frame in range(30, 120):
            stage_select.frame = frame
            driver(stage_select, controller)
        toggles = controller.pressed.count(melee.Button.BUTTON_Z)
        assert toggles == (1 if _ == 0 else 0)
```

Session test (append to `tests/test_netplay_session.py`):

```python
def test_rematch_keeps_the_frozen_stadium_toggle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _session(tmp_path)
    session._console = Mock()
    session._controller = Mock()
    helper = melee.MenuHelper()
    helper.frozen_stadium_selected = True
    session._menu_helper = helper
    session._last_frame_id = None
    monkeypatch.setattr(session, "_navigate_to_live", lambda *_args, **_kwargs: {"id": 0})
    session.start_rematch(NetplaySetup(melee.Character.FOX, "CRYO#610"))
    assert session._menu_helper is not helper
    assert session._menu_helper.frozen_stadium_selected is True
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_netplay_direct_driver.py tests/test_netplay_session.py -k "driver or frozen or hold or probe or idle or no_show or lock or disconnect"`
Expected: FAIL — names not defined; frozen test fails.

- [ ] **Step 3: Implement in `hal/sim/netplay.py`**

```python
class PlayerDisconnected(Exception):
    """The remote player left the direct-mode session."""


class PlayerIdle(Exception):
    """The remote player stayed at character select past the idle timeout."""


class PlayerNoShow(Exception):
    """The remote player did not connect within the connect timeout."""


@dataclass(frozen=True, slots=True)
class DirectSelection:
    revision: int
    character: melee.Character
    costume: int
    stage: melee.Stage
    identity: str


class DirectMenuDriver:
    """Drive Slippi 3.6.4 direct mode for one Dolphin session.

    Slippi has no un-ready, and the bot cannot see the remote lock-in, so the bot
    hovers its character and locks in last. Start while connected and locked is a
    no-op; once the remote player has gone, Start opens code entry, which is the
    only observable sign of a disconnect.
    """

    def __init__(
        self,
        *,
        opponent_code: str,
        selection: Callable[[], DirectSelection],
        lock_requests: Callable[[], int],
        connect_timeout_seconds: float,
        idle_timeout_seconds: float,
        hold_seconds: float,
        hold_cap_seconds: float,
        probe_interval_seconds: float,
        on_change: Callable[[], None] = lambda: None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._on_change = on_change
        self._opponent_code = opponent_code
        self._selection = selection
        self._lock_requests = lock_requests
        self._connect_timeout = connect_timeout_seconds
        self._idle_timeout = idle_timeout_seconds
        self._hold = hold_seconds
        self._hold_cap = hold_cap_seconds
        self._probe_interval = probe_interval_seconds
        self._clock = clock
        # Slippi keeps the Frozen Stadium toggle for the life of the Dolphin process.
        self._frozen_stadium = False
        self.connected = False
        self.phase: Literal["waiting_for_player", "character_select"] = "waiting_for_player"
        self.deadline: float | None = None
        self.locked: DirectSelection | None = None
        self._helper = melee.MenuHelper()
        self._arrived: float | None = None
        self._hold_until = 0.0
        self._hovered: DirectSelection | None = None
        self._lock_seen = 0
        self._next_probe = 0.0
        self._connect_started: float | None = None

    def begin_game(self) -> None:
        self._helper = melee.MenuHelper()
        self._helper.frozen_stadium_selected = self._frozen_stadium
        self.locked = None
        self._arrived = None
        self._hovered = None
        self._lock_seen = self._lock_requests()
        self.phase = "character_select" if self.connected else "waiting_for_player"
        self.deadline = None

    def __call__(self, state: melee.GameState | None, controller: melee.Controller) -> bool:
        if state is None:
            controller.release_all()
            return True
        now = self._clock()
        menu, submenu = state.menu_state, getattr(state, "submenu", None)
        if menu in (melee.Menu.MAIN_MENU, melee.Menu.PRESS_START):
            if self.connected:
                raise PlayerDisconnected("Slippi returned to the main menu")
            melee.MenuHelper.choose_direct_online(state, controller)
            return True
        if menu == melee.Menu.POSTGAME_SCORES:
            self._helper.skip_postgame(controller)
            return True
        if menu == melee.Menu.STAGE_SELECT:
            selection = self.locked or self._selection()
            self._helper.choose_stage(
                stage=selection.stage,
                gamestate=state,
                controller=controller,
                character=selection.character,
                frozen_stadium=True,
                autostart=True,
            )
            self._frozen_stadium = self._helper.frozen_stadium_selected
            return True
        if menu != melee.Menu.SLIPPI_ONLINE_CSS:
            controller.release_all()
            return True
        if submenu == melee.SubMenu.NAME_ENTRY_SUBMENU:
            if self.connected:
                raise PlayerDisconnected("Start opened code entry")
            if self._connect_started is None:
                self._connect_started = now
                self.deadline = now + self._connect_timeout
            self._helper.enter_direct_code(gamestate=state, controller=controller, connect_code=self._opponent_code)
            return True
        if not self.connected:
            return self._connect(state, controller, now)
        return self._between_games(state, controller, now)

    def _connect(self, state: melee.GameState, controller: melee.Controller, now: float) -> bool:
        # Game 1: pick the newest selection, then Start opens code entry; the match
        # starts when both players have entered each other's codes.
        if self._connect_started is not None and now >= self._connect_started + self._connect_timeout:
            raise PlayerNoShow(f"no connection within {self._connect_timeout:.0f}s")
        selection = self._selection()
        self.locked = selection
        self._helper.choose_character(
            character=selection.character, gamestate=state, controller=controller, costume=selection.costume, start=True
        )
        return True

    def _between_games(self, state: melee.GameState, controller: melee.Controller, now: float) -> bool:
        if self._arrived is None:
            self._arrived = now
            self.phase = "character_select"
        if now >= self._arrived + self._idle_timeout:
            raise PlayerIdle(f"no game started within {self._idle_timeout:.0f}s")
        if self.locked is not None:
            self.deadline = self._arrived + self._idle_timeout
            if now >= self._next_probe:
                self._next_probe = now + self._probe_interval
                controller.release_all()
                controller.press_button(melee.Button.BUTTON_START)
            else:
                controller.release_button(melee.Button.BUTTON_START)
            return True
        selection = self._selection()
        if self._hovered is None or (selection.character, selection.costume, selection.stage, selection.identity) != (
            self._hovered.character, self._hovered.costume, self._hovered.stage, self._hovered.identity
        ):
            self._hold_until = min(now + self._hold, self._arrived + self._hold_cap)
        self._hovered = selection
        self.deadline = self._hold_until
        requested = self._lock_requests() > self._lock_seen
        hovering = state.players[1].character == selection.character
        if hovering and (requested or now >= self._hold_until):
            self.locked = selection
            self._next_probe = now + self._probe_interval
            controller.release_all()
            controller.press_button(melee.Button.BUTTON_START)
            return True
        self._helper.choose_character(
            character=selection.character, gamestate=state, controller=controller, costume=selection.costume, start=False
        )
        return True
```

Every assignment to `phase`, `deadline`, or `locked` in these methods goes through one private helper that calls `self._on_change()` when the triple `(phase, deadline, locked)` differs from before; the snippets show plain assignments for readability. Add a test: `on_change` fires once when the hold deadline moves and once at lock-in.

Set `self.connected = True` from the runner after the first game reaches frame zero (`driver.connected = True` in Task 9), not inside the driver, because only a live game proves the connection.

`NetplaySession` changes:
- `read_frames(self, timeout_seconds: float | None = None)`: use `timeout_seconds if timeout_seconds is not None else self.step_timeout_seconds` for the deadline.
- `start_rematch`: replace `self._menu_helper = melee.MenuHelper()` with

```python
        # MenuHelper latches its choices, so a rematch needs a fresh one, but Slippi
        # keeps the Frozen Stadium toggle for the whole Dolphin session.
        frozen = self._menu_helper.frozen_stadium_selected
        self._menu_helper = melee.MenuHelper()
        self._menu_helper.frozen_stadium_selected = frozen
```

- `_discover_ports`: wrap the final `if ego.character != setup.character` check in `if self.menu_driver is None:` and add a comment: `# A custom driver owns the selection and its caller checks the locked character.`

- [ ] **Step 4: Run**

Run: `uv run pytest -q tests/test_netplay_direct_driver.py tests/test_netplay_session.py`
Expected: pass. If `choose_character`'s Slippi path needs more fake state, extend `_State`/`_Player` in the test, not the driver.

- [ ] **Step 5: Commit**

```bash
git add hal/sim/netplay.py tests/test_netplay_direct_driver.py tests/test_netplay_session.py
git commit -m "Drive direct mode with a hold, lock, and connection probe"
```

---

### Task 7: Tolerate pauses in a live match

**Files:**
- Modify: `hal/eval/netplay.py`
- Test: `tests/test_netplay_realtime.py`

**Interfaces:**
- Produces: `class PausedTooLong(RuntimeError)`; `run_netplay_match(..., pause_seconds: float | None = None, on_pause: Callable[[bool], None] | None = None)`. With `pause_seconds` set, a `FrameTimeout` from `read_frames(timeout_seconds=_PAUSE_DETECT_SECONDS)` reports `on_pause(True)` and keeps waiting until frames resume (`on_pause(False)`) or the stall reaches `pause_seconds` (raise `PausedTooLong`). Without it, behavior is unchanged.

- [ ] **Step 1: Write the failing test** (append to `tests/test_netplay_realtime.py`, reusing its fake session helpers)

```python
def test_pause_is_reported_and_resumes(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeRealtimeSession(stalls_after_frame=3, stall_reads=2)
    events: list[bool] = []
    result = _run(session, pause_seconds=60, on_pause=events.append)
    assert events == [True, False]
    assert result.trajectory is not None


def test_pause_past_the_limit_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = iter([0.0, 0.0, 30.0, 61.0, 61.0])
    monkeypatch.setattr(netplay.time, "monotonic", lambda: next(clock, 61.0))
    session = _FakeRealtimeSession(stalls_after_frame=3, stall_reads=100)
    with pytest.raises(netplay.PausedTooLong):
        _run(session, pause_seconds=60, on_pause=lambda _paused: None)
```

`_FakeRealtimeSession(stalls_after_frame, stall_reads)` raises `FrameTimeout` from `read_frames` `stall_reads` times after emitting `stalls_after_frame` frames, then resumes. Add it next to the file's existing fake session, built the same way. `_run(session, **kwargs)` wraps `run_netplay_match` with the file's existing fake client, runtime, and timing.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_netplay_realtime.py -k pause`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `hal/eval/netplay.py`:

```python
# A paused netplay game emits no frames; this much silence counts as a pause.
_PAUSE_DETECT_SECONDS: Final[float] = 2.0


class PausedTooLong(RuntimeError):
    """The game stayed paused past the allowed pause."""
```

`_NetplayLifecycle.__init__` takes `pause_seconds: float | None` and `on_pause: Callable[[bool], None] | None`. Replace the read in `run`:

```python
        stalled_since: float | None = None
        while len(captured) < max_frames:
            try:
                if stalled_since is None:
                    self.session.submit(self.choose(int(current["id"])))
                step_started = time.perf_counter()
                frames, in_game = self.session.read_frames(
                    timeout_seconds=None if self.pause_seconds is None else _PAUSE_DETECT_SECONDS
                )
            except FrameTimeout as error:
                if self.pause_seconds is None:
                    with suppress(OSError, EOFError):
                        self.session.submit(NEUTRAL_CONTROLLER_ACTION)
                    raise DolphinConnectionLost("Dolphin connection lost during netplay") from error
                now = time.monotonic()
                if stalled_since is None:
                    stalled_since = now - _PAUSE_DETECT_SECONDS
                    if self.on_pause is not None:
                        self.on_pause(True)
                if now - stalled_since >= self.pause_seconds:
                    raise PausedTooLong(f"no frames for {now - stalled_since:.0f}s") from error
                continue
            except (OSError, EOFError) as error:
                with suppress(OSError, EOFError):
                    self.session.submit(NEUTRAL_CONTROLLER_ACTION)
                raise DolphinConnectionLost("Dolphin connection lost during netplay") from error
            if stalled_since is not None:
                stalled_since = None
                if self.on_pause is not None:
                    self.on_pause(False)
```

The rest of the loop body is unchanged. Thread `pause_seconds` and `on_pause` through `run_netplay_match` into the lifecycle.

- [ ] **Step 4: Run**

Run: `uv run pytest -q tests/test_netplay_realtime.py tests/test_netplay_driver.py`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add hal/eval/netplay.py tests/test_netplay_realtime.py
git commit -m "Treat a frame stall in game as a bounded pause"
```

---

### Task 8: ReservationLink — report and receive in the background

**Files:**
- Create: `hal/netplay_service/reservation.py` (link only in this task)
- Create: `tests/test_netplay_reservation.py`

**Interfaces:**
- Consumes: Task 5 `RunnerQueue.report/end`, domain types.
- Produces:

```python
class ReservationLink:
    def __init__(self, queue: RunnerQueue, job: Job, worker_id: str, *,
                 interval_seconds: float = REPORT_INTERVAL_SECONDS, clock: Callable[[], float] = time.monotonic) -> None
    def __enter__(self) -> ReservationLink   # first report synchronous
    def __exit__(self, *_: object) -> None
    def set_phase(self, phase: Phase, deadline: float | None) -> None   # deadline in this clock's seconds
    def set_bot_code(self, code: str) -> None
    def lock(self, revision: int | None) -> None   # None between games until the bot locks in
    def add_game(self, game: FinishedGame) -> None
    def settings(self) -> Settings
    def settings_at(self, revision: int) -> Settings
    def wind_down(self) -> WindDown | None
    def lock_requests(self) -> int
    def released(self) -> bool         # the Worker ended or took the job
    def should_abort(self) -> bool     # released() or wind_down() is not None
    def end(self, reason: EndReason, *, retryable: bool = False) -> None  # flushes a final report first
```

- [ ] **Step 1: Write the failing tests**

```python
import threading

import pytest

from hal.netplay_service.domain import (
    EndReason, FinishedGame, GameResult, Job, JobStatus, Observed, Phase, Settings, WindDown,
)
from hal.netplay_service.queue_contract import InvalidTransitionError
from hal.netplay_service.reservation import ReservationLink


def _job(**changes: object) -> Job:
    values: dict[str, object] = dict(
        id="job", player_code="CRYO#610", online_delay=2, status=JobStatus.ASSIGNED, end_reason=None,
        queue_position=None, attempt=1, settings=Settings(1, "FOX", "IBDW#0", None, 20.0, 1.0), phase=None,
        phase_deadline=None, games=(), wind_down=None, lock_requests=0,
    )
    values.update(changes)
    return Job(**values)  # type: ignore[arg-type]


class _Queue:
    def __init__(self, responses: list[Job | Exception]) -> None:
        self.responses = responses
        self.reports: list[Observed] = []
        self.ends: list[tuple[EndReason, bool]] = []
        self.lock = threading.Lock()

    def report(self, _job_id: str, _worker_id: str, observed: Observed) -> Job:
        with self.lock:
            self.reports.append(observed)
            response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(response, Exception):
            raise response
        return response

    def end(self, _job_id: str, _worker_id: str, reason: EndReason, *, retryable: bool) -> Job:
        self.ends.append((reason, retryable))
        return _job(status=JobStatus.ENDED, end_reason=reason)


def test_reports_carry_increasing_seq_and_full_games() -> None:
    queue = _Queue([_job()])
    with ReservationLink(queue, _job(), "s/slot-0", interval_seconds=0.01) as link:
        link.set_phase(Phase.IN_GAME, None)
        link.add_game(FinishedGame(1, "BATTLEFIELD", GameResult.WIN))
        link.end(EndReason.PLAYER_DISCONNECTED)
    seqs = [report.seq for report in queue.reports]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    assert queue.reports[-1].finished_games == (FinishedGame(1, "BATTLEFIELD", GameResult.WIN),)
    assert queue.ends == [(EndReason.PLAYER_DISCONNECTED, False)]


def test_settings_and_wind_down_follow_the_response() -> None:
    newer = _job(settings=Settings(2, "FALCO", "MANG#0", "POKEMON_STADIUM", 30.0, 1.0), wind_down=WindDown.PLAYER)
    queue = _Queue([newer])
    with ReservationLink(queue, _job(), "s/slot-0", interval_seconds=0.01) as link:
        assert link.settings().character == "FALCO"
        assert link.settings_at(1).character == "FOX"
        assert link.wind_down() is WindDown.PLAYER
        assert link.should_abort()
        assert queue.reports[-1].seen_revision == 2


def test_a_refused_report_marks_the_link_released() -> None:
    queue = _Queue([InvalidTransitionError("worker does not own this job")])
    with ReservationLink(queue, _job(), "s/slot-0", interval_seconds=0.01) as link:
        assert link.released()
        link.end(EndReason.NO_SHOW)
    assert queue.ends == []


def test_an_ended_job_marks_the_link_released() -> None:
    queue = _Queue([_job(status=JobStatus.ENDED, end_reason=EndReason.PLAYER_LEFT)])
    with ReservationLink(queue, _job(), "s/slot-0", interval_seconds=0.01) as link:
        assert link.released()
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_netplay_reservation.py`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement** (`hal/netplay_service/reservation.py`)

```python
"""One reservation on one runner slot: the reporting link and the Slippi loop."""

import threading
import time
from collections.abc import Callable

from loguru import logger

from hal.netplay_service.domain import (
    REPORT_INTERVAL_SECONDS, EndReason, FinishedGame, Job, JobStatus, Observed, Phase, Settings, WindDown,
)
from hal.netplay_service.queue_client import QueueUnavailableError
from hal.netplay_service.queue_contract import InvalidTransitionError, RunnerQueue


class ReservationLink:
    """Report the runner's snapshot every interval and keep the Worker's latest answer.

    The main loop only reads desired state and writes observations; it never waits
    on the network. A refused report means the Worker no longer gives this slot the
    job, so the link stops and the loop aborts at its next check.
    """

    def __init__(
        self,
        queue: RunnerQueue,
        job: Job,
        worker_id: str,
        *,
        interval_seconds: float = REPORT_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._queue = queue
        self._job_id = job.id
        self._worker_id = worker_id
        self._interval = interval_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._seq = 0
        self._job = job
        self._history: dict[int, Settings] = {job.settings.revision: job.settings}
        self._phase = Phase.BOOTING
        self._deadline: float | None = None
        self._bot_code: str | None = None
        self._locked: int | None = None
        self._games: list[FinishedGame] = []
        self._released = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"reservation-{job.id}", daemon=True)

    def __enter__(self) -> ReservationLink:
        self._send()
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def _snapshot(self) -> Observed:
        with self._lock:
            self._seq += 1
            left = None if self._deadline is None else max(0.0, self._deadline - self._clock())
            return Observed(
                seq=self._seq,
                phase=self._phase,
                phase_seconds_left=left,
                bot_code=self._bot_code,
                seen_revision=self._job.settings.revision,
                locked_revision=self._locked,
                finished_games=tuple(self._games),
            )

    def _send(self) -> None:
        if self._released:
            return
        try:
            job = self._queue.report(self._job_id, self._worker_id, self._snapshot())
        except QueueUnavailableError as error:
            # The lease tolerates a few lost reports; the next tick tries again.
            logger.bind(job=self._job_id, event="report").warning("reservation report failed: {}", error)
            return
        except InvalidTransitionError:
            with self._lock:
                self._released = True
            return
        with self._lock:
            self._job = job
            self._history[job.settings.revision] = job.settings
            if job.status is not JobStatus.ASSIGNED:
                self._released = True

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._send()
            if self._released:
                return

    def set_phase(self, phase: Phase, deadline: float | None) -> None:
        with self._lock:
            self._phase, self._deadline = phase, deadline

    def set_bot_code(self, code: str) -> None:
        with self._lock:
            self._bot_code = code

    def lock(self, revision: int | None) -> None:
        with self._lock:
            self._locked = revision

    def add_game(self, game: FinishedGame) -> None:
        with self._lock:
            self._games.append(game)

    def settings(self) -> Settings:
        with self._lock:
            return self._job.settings

    def settings_at(self, revision: int) -> Settings:
        with self._lock:
            return self._history[revision]

    def wind_down(self) -> WindDown | None:
        with self._lock:
            return self._job.wind_down

    def lock_requests(self) -> int:
        with self._lock:
            return self._job.lock_requests

    def released(self) -> bool:
        with self._lock:
            return self._released

    def should_abort(self) -> bool:
        with self._lock:
            return self._released or self._job.wind_down is not None

    def end(self, reason: EndReason, *, retryable: bool = False) -> None:
        """Flush the final snapshot so the last game is recorded, then end the job."""
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        self._send()
        if self.released():
            return
        try:
            self._queue.end(self._job_id, self._worker_id, reason, retryable=retryable)
        except InvalidTransitionError:
            pass
        with self._lock:
            self._released = True
```

- [ ] **Step 4: Run**

Run: `uv run pytest -q tests/test_netplay_reservation.py`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add hal/netplay_service/reservation.py tests/test_netplay_reservation.py
git commit -m "Report reservations in the background"
```

---

### Task 9: Run a reservation through direct mode

**Files:**
- Modify: `hal/netplay_service/reservation.py` (add `run_reservation`)
- Modify: `hal/netplay_service/runner.py`
- Modify: `tests/test_netplay_runner.py`, extend `tests/test_netplay_reservation.py`

**Interfaces:**
- Consumes: Tasks 5–8.
- Produces: `run_reservation(config: SlotConfig, store: RemoteQueue, policy: InferenceClient, runtime: RuntimeConfig, job: Job, stop: StopEvent, health: SlotHealth, timing: FrameTiming) -> None`, where `SlotHealth` is a `Protocol` with `connecting(delay)`, `playing()`, `idle()`, `configure_schedule(timing)`, and the observer methods `run_netplay_match` already uses. `SlotConfig` stays in `runner.py`; `reservation.py` imports it under `TYPE_CHECKING` only, so `runner.py` can import `run_reservation` without a cycle.

Behavior (the loop):

1. Enter `ReservationLink`; `link.set_phase(Phase.BOOTING, None)`; `link.set_bot_code(config.bot_connect_code)`.
2. Build `DirectMenuDriver(opponent_code=job.player_code, selection=_selection(link), lock_requests=link.lock_requests, connect_timeout_seconds=CONNECT_TIMEOUT_SECONDS, idle_timeout_seconds=IDLE_TIMEOUT_SECONDS, hold_seconds=LOCK_HOLD_SECONDS, hold_cap_seconds=LOCK_HOLD_CAP_SECONDS, probe_interval_seconds=CONNECTION_PROBE_SECONDS, on_change=mirror)`. `_selection(link)` maps `link.settings()` to `DirectSelection(revision, melee.Character[character], 0, melee.Stage[stage] if stage else melee.Stage.RANDOM_STAGE, imitation)`. `mirror()` copies the driver into the link: `link.set_phase(Phase(driver.phase), driver.deadline)` and `link.lock(None if driver.locked is None else driver.locked.revision)`. The driver and the link use the same monotonic clock.
3. Open `NetplaySession(..., connect_timeout_seconds=CONNECT_TIMEOUT_SECONDS + IDLE_TIMEOUT_SECONDS, connect_abandoned=link.should_abort, realtime=True, menu_driver=driver, step_timeout_seconds=FRAME_STALL_SECONDS)`.
4. Loop while not `stop.is_set()`:
   - `driver.begin_game()`, then `mirror()`. `locked_revision` is therefore `None` at the start of every character-select visit and names the locked revision once the bot locks in.
   - `health.connecting(delay)` on the first game.
   - `run_netplay_match(session, NetplaySetup(locked_char_placeholder…))`: pass `setup=NetplaySetup(melee.Character[link.settings().character], job.player_code, local_code=config.bot_connect_code)`, `rematch=games_played > 0`, `player_identity=None if settings.imitation == "MASKED" else settings.imitation` evaluated from `link.settings_at(driver.locked.revision)` inside `on_live`, `policy_settings=lambda: (link.settings().desired_return, link.settings().temperature)`, `pause_seconds=PAUSE_TIMEOUT_SECONDS`, `on_pause=lambda paused: link.set_phase(Phase.PAUSED if paused else Phase.IN_GAME, None)`, `on_live=_on_live`.
   - `_on_live`: `driver.connected = True`; `assert driver.locked is not None`; check the ego character equals `driver.locked.character` (raise `RuntimeError` otherwise); `link.lock(driver.locked.revision)`; `link.set_phase(Phase.IN_GAME, None)`; `health.playing()`.
   - Player identity must be fixed before the match starts, so `run_netplay_match` takes `player_identity` as a callable in this task: change its parameter to `player_identity: Callable[[], str | None] | str | None` and resolve it once at frame zero. Keep a `str | None` for ranked callers.
   - After the match: `result = _game_result(replay_end, play_result)`, a module function in `reservation.py`: `EndMethod.NO_CONTEST` → `GameResult.NO_CONTEST`; a completed game → `win`/`loss` from `_human_result` (a stock tie is recorded as `no_contest`); any other end method raises `RuntimeError` as today. `link.add_game(FinishedGame(n, stage, result))`; write the measurement and pending upload as today.
   - Read `is_frozen_ps` from the replay's game start when the stage is Pokémon Stadium and log a warning if it is False and HAL picked the stage.
   - If `link.wind_down()` is set → `link.end(EndReason.PLAYER_CANCELED if wind_down is WindDown.PLAYER else EndReason.YIELDED)`; return.
   - `health.idle()`; continue to the next game (the driver handles character select).
5. Exceptions (in `runner._handle_reservation`, which wraps `run_reservation`; the link must be reachable, so `run_reservation` catches and ends inside the `with ReservationLink` block):

| Exception | End |
|---|---|
| `PlayerNoShow` | `NO_SHOW` |
| `PlayerDisconnected` | `PLAYER_DISCONNECTED` |
| `PlayerIdle`, `PausedTooLong` | `IDLE_TIMEOUT` |
| `ConnectAbandoned` | if `link.released()`: nothing; else `PLAYER_CANCELED` or `YIELDED` from `link.wind_down()` |
| `DolphinConnectionLost`, `NoUsableActionPlan`, `FrameTimeout`, `_RecoverableRuntimeError`, other `Exception` | `SERVICE_FAILURE`, `retryable=True` |
| `KeyError`, `ValueError` | `SERVICE_FAILURE`, `retryable=False` |
| `InferenceUnavailable` | `SERVICE_FAILURE`, `retryable=True`, then re-raise |
| `QueueUnavailableError`, `SessionEndedError` | re-raise (slot stops; the Worker's lease rule handles the job) |

`runner.py` deletions: `_heartbeat`, `_LivePolicySettings`, `_ReservationLive`, `_setup`, `_run_reservation`, `CONNECT_TIMEOUT_SECONDS`/`TERMINAL_STATUSES` imports, the `websockets` import. `_slot_worker` calls `_handle_reservation(config, store, policy, runtime, job, stop, health, timing)`, which calls `run_reservation` and then `store.finish_pairing(job.id, config.worker_id, job.attempt)` in `finally`, as today.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_netplay_reservation.py`)

```python
from unittest.mock import Mock

import melee

from hal.eval.replays import ReplayEnd
from hal.netplay_service import reservation
from hal.sim.netplay import ConnectAbandoned, PlayerDisconnected


class _Session:
    def __init__(self, *_args: object, **kwargs: object) -> None:
        self.kwargs = kwargs

    def __enter__(self) -> "_Session":
        return self

    def __exit__(self, *_args: object) -> None:
        pass


def _play(results: list[object]):
    def play(*_args: object, **kwargs: object) -> object:
        outcome = results.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        on_live = kwargs["on_live"]
        assert callable(on_live)
        on_live()
        return outcome
    return play


def test_disconnect_after_a_game_ends_as_player_disconnected(tmp_path, monkeypatch) -> None:
    queue = _Queue([_job()])
    monkeypatch.setattr(reservation, "NetplaySession", _Session)
    monkeypatch.setattr(reservation, "run_netplay_match", _play([_fake_result(), PlayerDisconnected("left")]))
    monkeypatch.setattr(reservation, "read_new_replay_end", lambda *_args: ReplayEnd(tmp_path / "g.slp", EndMethod.GAME))
    monkeypatch.setattr(reservation, "_game_result", lambda *_args: GameResult.LOSS)
    reservation.run_reservation(_slot_config(tmp_path), queue, Mock(), _RUNTIME, _job(), _never(), Mock(), _TIMING)
    assert queue.ends == [(EndReason.PLAYER_DISCONNECTED, False)]
    assert queue.reports[-1].finished_games[0].result is GameResult.LOSS


def test_leave_at_character_select_ends_as_player_canceled(tmp_path, monkeypatch) -> None:
    queue = _Queue([_job(wind_down=WindDown.PLAYER)])
    monkeypatch.setattr(reservation, "NetplaySession", _Session)
    monkeypatch.setattr(reservation, "run_netplay_match", _play([ConnectAbandoned("wind down")]))
    reservation.run_reservation(_slot_config(tmp_path), queue, Mock(), _RUNTIME, _job(), _never(), Mock(), _TIMING)
    assert queue.ends == [(EndReason.PLAYER_CANCELED, False)]


def test_no_contest_is_recorded_and_the_session_continues(tmp_path, monkeypatch) -> None:
    queue = _Queue([_job()])
    monkeypatch.setattr(reservation, "NetplaySession", _Session)
    monkeypatch.setattr(reservation, "run_netplay_match", _play([_fake_result(), PlayerDisconnected("left")]))
    monkeypatch.setattr(reservation, "read_new_replay_end", lambda *_args: ReplayEnd(tmp_path / "g.slp", EndMethod.NO_CONTEST))
    reservation.run_reservation(_slot_config(tmp_path), queue, Mock(), _RUNTIME, _job(), _never(), Mock(), _TIMING)
    assert queue.reports[-1].finished_games == (FinishedGame(1, "BATTLEFIELD", GameResult.NO_CONTEST),)


def test_service_failure_is_retryable_and_records_no_result(tmp_path, monkeypatch) -> None:
    queue = _Queue([_job()])
    monkeypatch.setattr(reservation, "NetplaySession", _Session)
    monkeypatch.setattr(reservation, "run_netplay_match", _play([DolphinConnectionLost("gone")]))
    reservation.run_reservation(_slot_config(tmp_path), queue, Mock(), _RUNTIME, _job(), _never(), Mock(), _TIMING)
    assert queue.ends == [(EndReason.SERVICE_FAILURE, True)]
    assert queue.reports[-1].finished_games == ()
```

Helpers in the test module: `_fake_result()` returns a `PlayResult` with `stage=melee.Stage.BATTLEFIELD.value` and the fields `_write_match_measurement` needs (copy the construction from `tests/test_netplay_runner.py::test_match_measurement_preserves_source_frame_and_schedule_counters`); `_slot_config(tmp_path)` copies `tests/test_netplay_runner.py::_slot_config`; `_RUNTIME = RuntimeConfig(1, (2, 3))`; `_TIMING = FrameTiming(2, 2, 4, 2, 8)`; `_never()` returns a `Mock` whose `is_set` returns `False`.

Delete from `tests/test_netplay_runner.py` the tests of removed code: `test_live_policy_settings_*`, `test_heartbeat_forfeits_active_job_when_runner_aborts`, `test_abandoned_connect_frees_slot_without_no_show`, `test_no_contest_ends_reservation_without_a_result`, `test_first_game_is_random_and_rematch_uses_requested_stage`, `test_inference_failure_stops_slot_after_forfeiting_current_reservation`, `test_no_usable_plan_forfeits_match_without_stopping_healthy_slot`, and the `_LiveConnection`/`_LiveQueue` fakes. Port `test_recoverable_failure_*`, `test_stream_slot_writes_only_public_game_state_and_returns_to_idle`, `test_local_qualification_retains_replay_without_publishing`, and `test_result_is_reported_from_the_human_side` to patch `reservation.run_netplay_match`/`reservation.NetplaySession` and assert `end(...)` calls instead of `fail`/`forfeit`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_netplay_reservation.py tests/test_netplay_runner.py`
Expected: FAIL.

- [ ] **Step 3: Implement** `run_reservation` per the behavior above; edit `runner.py` per the deletions. Keep `_write_match_measurement`, `_write_match_failure`, `_write_pending_upload`, `_complete_pending_upload`, `_human_result`, `_stage_name` in `runner.py` if `reservation.py` can import them without a cycle; otherwise move them to `reservation.py` unchanged and re-import them in `runner.py` where the CLI or uploads still use them.

- [ ] **Step 4: Run**

Run: `uv run pytest -q tests/test_netplay_reservation.py tests/test_netplay_runner.py tests/test_netplay_realtime.py`
Expected: pass.

Run: `uv run ty check --python-version 3.14 --error-on-warning hal experiments/059_muon_action_sequence.py scripts`
Expected: zero diagnostics.

- [ ] **Step 5: Commit**

```bash
git add hal/netplay_service/reservation.py hal/netplay_service/runner.py hal/eval/netplay.py tests/test_netplay_reservation.py tests/test_netplay_runner.py
git commit -m "Keep the player connected between games"
```

---

### Task 10: Page — render by phase, edit settings, lock in, leave

**Files:**
- Modify: `web/netplay/lib/netplay-api.ts`, `web/netplay/app/page.tsx`, `web/netplay/components/sentence.tsx`, `web/netplay/app/globals.css`

**Interfaces:**
- Produces (`lib/netplay-api.ts`):

```ts
export type Phase = 'booting' | 'waiting_for_player' | 'character_select' | 'in_game' | 'paused';
export type EndReason = 'player_canceled' | 'player_left' | 'player_disconnected' | 'no_show' | 'idle_timeout' | 'yielded' | 'service_failure';
export type Settings = { revision: number; character: string; imitation: string; stage: string | null; desired_return: number | null; temperature: number };
export type FinishedGame = { number: number; stage: string; result: 'win' | 'loss' | 'no_contest' };
export type Job = {
  id: string; player_code: string; online_delay: number; status: 'queued' | 'assigned' | 'ended'; end_reason: EndReason | null;
  queue_position: number | null; attempt: number; settings: Settings;
  observed: { seq: number; phase: Phase; bot_code: string | null; seen_revision: number; locked_revision: number | null } | null;
  phase_deadline: number | null; games: FinishedGame[]; wind_down: 'player' | 'yield' | null; lock_requests: number;
};
export type SettingsUpdate = Partial<Omit<Settings, 'revision'>>;
export function updateSettings(id: string, token: string, values: SettingsUpdate): Promise<Job>;  // PATCH /settings
export function requestLock(id: string, token: string): Promise<Job>;                            // POST /lock
export function leaveJob(id: string, token: string): Promise<Job>;                               // DELETE
```

`Options` drops `max_games`, `no_show_seconds`, `rematch_seconds`; `CreateJob` gains `stage?: string | null`. Remove `cancelJob`, `updatePolicy`, `requestRematch`, `Rematch`, `PolicySettings`.

- [ ] **Step 1: Update the API module** with the types and three functions above (same `request`/`authorized` helpers).

- [ ] **Step 2: Rewrite the reservation view in `app/page.tsx`**

- Replace `terminal` with `job.status === 'ended'`.
- `useSettingsEditor(job, saved, update, setError)`: keeps a local draft per field, sends `updateSettings` 400 ms after the last edit, and returns `{ value(field), set(field, value), pending }`. It replaces `useLiveDifficulty`.
- `Reservation` renders:
  - `<Sentence lead={ended ? 'HAL played like' : 'HAL plays like'} …>` with editable `imitation`, `character`, `stage`, and difficulty while not ended. The stage blank uses `options.stages` plus a "Random" choice mapped to `null`.
  - A status line from `settingsStatus(job)`: "Saving…" while a PATCH is in flight. For character, identity, or stage: during `character_select` before lock-in, "Applies to this game"; otherwise "Applies next game" until `observed.locked_revision` reaches the revision that last changed one of them, then "In effect". The page tracks that revision from its own PATCH responses. Difficulty: "In effect" once `observed?.seen_revision >= settings.revision`.
  - `StatusBody` by state: `queued` → position; `observed === null || phase === 'booting'` → "Starting HAL"; `waiting_for_player` → `CopyCode` with `observed.bot_code`, connect steps, `Countdown(deadline=phase_deadline, total=60)`; `character_select` with `observed.locked_revision === null` → "Pick your character in Slippi", `Countdown(total=30, suffix='until HAL locks in')`, **Lock in now** (`requestLock`); `character_select` with a locked revision → "HAL is locked in. Lock in on Slippi to start."; `in_game` → `Game ${games.length + 1}`; `paused` → "Game paused", `Countdown(total=60)`; `ended` → `endText(job.end_reason)`.
  - `GameList`: one row per finished game (number, stage label, "You won" / "HAL won" / "No contest").
  - Footer button: `queued` → "Leave queue"; `assigned` and phase `in_game`/`paused` → "Stop after this game" (disabled once `wind_down === 'player'`, then "Ending after this game"); other assigned phases → "Leave"; all call `leaveJob`. Ended → "Change settings" / "Queue again →" as today.
- `endText(reason)`:

```ts
const END_TEXT: Record<EndReason, string> = {
  player_canceled: 'You ended the session.',
  player_left: 'This page was closed for two minutes, so your spot went to the next player.',
  player_disconnected: 'You left the Slippi session.',
  no_show: 'HAL waited but no connection arrived. Queue again when you are ready.',
  idle_timeout: 'No game started for a while, so HAL freed the slot.',
  yielded: 'Others were waiting, so HAL moved on after your game. Thanks for playing.',
  service_failure: 'HAL had a problem it could not recover from. Queue again whenever you like.',
};
```

- `statusTitle`, `tabTitle`, `useStatusAlerts`: key on `status` and `observed?.phase` (`waiting_for_player` → "● Connect now", notification "Direct-connect to {bot_code} in Slippi."; `character_select` → "Pick your character").
- Delete `RematchForm`, `Progress`'s rematch statuses (steps: Queue = queued/booting, Connect = waiting_for_player, Play = character_select/in_game/paused, Done = ended), `cancelLabel`, `resultText`'s `tie`.
- The join form passes `stage: null` in `CreateJob`.

- [ ] **Step 3: Sentence stage blank** — `components/sentence.tsx`: `stage?: Slot<string | null>`; render "on {stage label or 'a random stage'}"; the stage picker adds `{ value: '', name: 'Random', alias: '' }` and maps `''` ↔ `null`.

- [ ] **Step 4: Run the page checks and click through**

Run: `cd web/netplay && npx tsc --noEmit -p . && npx oxlint app lib && npx oxfmt --check app lib components`
Expected: exit 0.

Manual: with the local Worker (protocol 3) and the debug console updated to post `report`/`end`, join, step through `booting → waiting_for_player → character_select → in_game → character_select`, change the character during character select, press Lock in now, leave mid-game, and see each `end_reason` line.

- [ ] **Step 5: Commit**

```bash
git add web/netplay
git commit -m "Show the session by phase and edit HAL between games"
```

---

### Task 11: Qualification script on the new routes

**Files:**
- Modify: `scripts/qualify_netplay_059.py`, `tests/test_qualify_netplay_059.py`

**Interfaces:**
- Consumes: Task 5 `parse_job`, `MatchChoices(stage=)`.

- [ ] **Step 1: Update the test** in `tests/test_qualify_netplay_059.py` that exercises `_PlayerQueue` and `_validate_live_reservation`: `_validate_live_reservation(job)` accepts `status == assigned` with `phase in (in_game, paused)` and rejects `ended`; `_PlayerQueue.leave(job_id, token)` sends `DELETE /v1/jobs/{id}`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_qualify_netplay_059.py`
Expected: FAIL.

- [ ] **Step 3: Implement**
- `_PlayerQueue.create_job` posts `stage: None`; delete `request_rematch`; add `leave`.
- `_validate_live_reservation`: require `JobStatus.ASSIGNED` and `Phase.IN_GAME`/`Phase.PAUSED` while the peer is in a game.
- Loop: after each peer game, wait until `len(job.games)` reaches `game_number`; assess the measurement; if the qualification is complete, call `store.leave(...)` and wait for `JobStatus.ENDED` with `EndReason.PLAYER_CANCELED`; otherwise `first = peer.start_rematch(setup)` with the peer's `NetplaySetup(stage=melee.Stage.BATTLEFIELD)` (the peer picks only when it lost).

- [ ] **Step 4: Run**

Run: `uv run pytest -q tests/test_qualify_netplay_059.py`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/qualify_netplay_059.py tests/test_qualify_netplay_059.py
git commit -m "Qualify against the continuous session"
```

---

### Task 12: Two-Dolphin direct-mode integration test

**Files:**
- Create: `tests/test_netplay_direct_integration.py`

The probe showed the behaviors this test pins. It runs two real Slippi Dolphins and needs two distinct accounts: `HAL_NETPLAY_BOT_ACCOUNT` and `HAL_NETPLAY_PEER_ACCOUNT` (paths to `user.json`). Without them it skips, and with `HAL_REQUIRE_INTEGRATION=1` it fails.

- [ ] **Step 1: Write the test**

```python
"""Slippi 3.6.4 direct mode across consecutive games, with two local Dolphins."""

import os
import threading
import time
from pathlib import Path

import melee
import peppi_py
import pytest

from hal.controller import NEUTRAL_CONTROLLER_ACTION, ControllerAction
from hal.netplay_service.domain import account_connect_code
from hal.paths import ISO_PATH, NETPLAY_EMULATOR_PATH
from hal.sim.netplay import DirectMenuDriver, DirectSelection, NetplaySession, NetplaySetup, PlayerDisconnected

pytestmark = pytest.mark.integration

_WALK_LEFT = ControllerAction(main_x=-1.0, main_y=0.0, c_x=0.0, c_y=0.0, trigger_l=0.0, trigger_r=0.0, buttons=0)


def _accounts() -> tuple[Path, Path]:
    bot, peer = os.environ.get("HAL_NETPLAY_BOT_ACCOUNT"), os.environ.get("HAL_NETPLAY_PEER_ACCOUNT")
    if not bot or not peer:
        if os.environ.get("HAL_REQUIRE_INTEGRATION") == "1":
            pytest.fail("HAL_NETPLAY_BOT_ACCOUNT and HAL_NETPLAY_PEER_ACCOUNT are required")
        pytest.skip("two Slippi accounts are required")
    return Path(bot), Path(peer)


def _session(account: Path, port: int, replay_dir: Path, driver: object | None = None) -> NetplaySession:
    return NetplaySession(
        ISO_PATH, dolphin_path=NETPLAY_EMULATOR_PATH, user_json_path=account, online_delay=2,
        replay_dir=replay_dir, slippi_port=port, realtime=True, graphics_backend="OGL",
        connect_timeout_seconds=600, step_timeout_seconds=60, menu_driver=driver,
    )


def _play_out(session: NetplaySession, action: ControllerAction) -> None:
    while True:
        session.submit(action)
        _frames, in_game = session.read_frames()
        if not in_game:
            return


def test_direct_mode_session(tmp_path: Path) -> None:
    bot_account, peer_account = _accounts()
    bot_code, peer_code = account_connect_code(bot_account), account_connect_code(peer_account)
    selection = [DirectSelection(1, melee.Character.FOX, 0, melee.Stage.POKEMON_STADIUM)]
    locks = [0]
    driver = DirectMenuDriver(
        opponent_code=peer_code, selection=lambda: selection[-1], lock_requests=lambda: locks[-1],
        connect_timeout_seconds=120, idle_timeout_seconds=300, hold_seconds=5, hold_cap_seconds=30,
        probe_interval_seconds=2,
    )
    errors: list[BaseException] = []
    peer_done = threading.Event()

    def peer() -> None:
        try:
            with _session(peer_account, 51472, tmp_path / "peer") as session:
                setup = NetplaySetup(melee.Character.MARTH, bot_code, stage=melee.Stage.BATTLEFIELD)
                session.start_match(setup)
                _play_out(session, NEUTRAL_CONTROLLER_ACTION)       # game 1: bot walks off
                session.start_rematch(setup)
                _play_out(session, NEUTRAL_CONTROLLER_ACTION)       # game 2: bot walks off again
                session.start_rematch(setup)
                _play_out(session, NEUTRAL_CONTROLLER_ACTION)       # game 3: ends with L+R+A+Start
                peer_done.wait(120)
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=peer, daemon=True)
    thread.start()
    with _session(bot_account, 51471, tmp_path / "bot", driver) as bot:
        for game in range(3):
            driver.begin_game()
            if game == 1:
                selection.append(DirectSelection(2, melee.Character.FALCO, 0, melee.Stage.POKEMON_STADIUM))
            (bot.start_match if game == 0 else bot.start_rematch)(NetplaySetup(melee.Character.FOX, peer_code, local_code=bot_code))
            driver.connected = True
            assert driver.locked is not None
            assert driver.locked.revision == (1 if game == 0 else 2)
            _play_out(bot, _WALK_LEFT)
        driver.begin_game()
        peer_done.set()
        with pytest.raises(PlayerDisconnected):
            bot.start_rematch(NetplaySetup(melee.Character.FALCO, peer_code, local_code=bot_code))
    thread.join(60)
    assert not errors, errors
    replays = sorted((tmp_path / "bot").rglob("*.slp"))
    starts = [peppi_py.read_slippi(str(path), skip_frames=True).start for path in replays]
    stadium = [start for start in starts if start.stage == melee.Stage.POKEMON_STADIUM.value]
    assert len(stadium) >= 2 and all(start.is_frozen_ps for start in stadium)
```

The bot walks off and loses games 1 and 2, so the bot picks the stage for games 2 and 3 (Pokémon Stadium twice), which exercises the Frozen Stadium carry-over. The peer leaves its session after game 3, which the bot's connection probe must report.

- [ ] **Step 2: Run without accounts** — `uv run pytest -q tests/test_netplay_direct_integration.py -m integration` → skipped; with `HAL_REQUIRE_INTEGRATION=1` → fails on missing accounts.

- [ ] **Step 3: Run with accounts** (stop any netplay host that uses either account first):

```bash
HAL_REQUIRE_INTEGRATION=1 HAL_NETPLAY_BOT_ACCOUNT=$HOME/data/slippi_user_a.json \
HAL_NETPLAY_PEER_ACCOUNT=$HOME/data/slippi_user_b.json \
xvfb-run -a uv run pytest -q tests/test_netplay_direct_integration.py -m integration
```

Expected: PASS. If a step fails, capture a screenshot of each Xvfb display before changing the driver.

- [ ] **Step 4: Commit**

```bash
git add tests/test_netplay_direct_integration.py
git commit -m "Test direct mode across consecutive games"
```

---

### Task 13: Full verification and docs

- [ ] **Step 1:** Update `deploy/netplay/README.md` and `deploy/netplay/direct8.md` where they describe sets, rematches, or the 5-game cap; state the continuous session, the yield rule, and the end reasons in one short section.

- [ ] **Step 2:** Run every check from **Global Constraints**, plus:

```bash
HAL_REQUIRE_INTEGRATION=1 uv run pytest -q tests/test_roundtrip.py tests/test_session_cleanup.py -m integration
```

Report each command, failure, and skip.

- [ ] **Step 3: Commit**

```bash
git add deploy/netplay
git commit -m "Document the continuous netplay session"
```
