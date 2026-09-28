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
    await publish();
    const count = async () => (await call("GET", "/v1/admin/events", { admin: true })).body.events.length as number;
    const before = await count();
    expect((await call("POST", "/v1/runner/sessions", { body: {} })).status).toBe(401);
    expect((await call("POST", "/v1/runner/sessions", { token: ADMIN_TOKEN, body: {} })).status).toBe(401);
    expect((await call("PUT", "/v1/admin/policy", { runner: true, body: POLICY })).status).toBe(401);
    expect((await call("GET", "/v1/admin/status", { admin: true })).status).toBe(200);
    expect(await count()).toBe(before);
  });

  it("requires a runner token before the live socket reaches the queue", async () => {
    expect((await call("GET", "/v1/runner/jobs/x/live")).status).toBe(401);
  });

  it("validates event query parameters", async () => {
    for (const query of ["since=abc", "since=", "since=Infinity", "limit=0", "limit=-1", "limit=5001", "limit=1.5", "limit=x"]) {
      expect((await call("GET", `/v1/admin/events?${query}`, { admin: true })).status, query).toBe(422);
    }
    await publish();
    await publish();
    const result = await call("GET", "/v1/admin/events?since=0&limit=1", { admin: true });
    expect(result.status).toBe(200);
    expect(result.body.events).toHaveLength(1);
  });

  it("accepts any listed runner token", async () => {
    await publish();
    await seedAccounts(2);
    const result = await call("POST", "/v1/runner/sessions", {
      runner: true,
      runnerToken: OTHER_RUNNER_TOKEN,
      body: { protocol_version: 1, session_id: crypto.randomUUID(), host: "b", bundle_sha256: POLICY.bundle_sha256, git_sha: "g", slots: 1, stream: false },
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
      body: { protocol_version: 1, session_id: crypto.randomUUID(), host: "b", bundle_sha256: "f".repeat(64), git_sha: "g", slots: 1, stream: false },
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
    const ended = (await call("GET", `/v1/admin/events?session=${session}`, { admin: true })).body.events;
    expect(ended).toContainEqual(expect.objectContaining({ kind: "session_ended", detail: { reason: "silent" } }));
    const failed = (await call("GET", `/v1/admin/events?job=${job.id}`, { admin: true })).body.events;
    expect(failed).toContainEqual(expect.objectContaining({ kind: "job_failed", detail: { reason: "session_silent" } }));
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
