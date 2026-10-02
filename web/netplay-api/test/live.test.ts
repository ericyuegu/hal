import { SELF, runInDurableObject } from "cloudflare:test";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { call, CREATE, publish, queueStub, resetQueue, RUNNER_TOKEN, runnerStatus, seedAccounts, setClock, START, startSession, NEW_JOB, POLICY } from "./helpers";
import { WriteBudget, MAX_DATABASE_BYTES } from "../src/budget";
import { LiveConnections } from "../src/live";
import { SessionStore } from "../src/sessions";
import { JobStore } from "../src/store";
import { parsePolicyConfig } from "../src/policy";

async function socket(session?: string) {
  const headers = new Headers({ Upgrade: "websocket" });
  if (session) {
    headers.set("Authorization", `Bearer ${RUNNER_TOKEN}`);
    headers.set("X-HAL-Session", session);
  }
  const response = await SELF.fetch(`https://20xx.xyz/v1/${session ? "runner/" : ""}live`, { headers });
  expect(response.status).toBe(101);
  const ws = response.webSocket!;
  const messages: Record<string, any>[] = [];
  ws.addEventListener("message", (event) => {
    if (event.data !== "pong") messages.push(JSON.parse(event.data as string));
  });
  ws.accept();
  return { ws, messages };
}

beforeEach(async () => { await resetQueue(); await publish(); await seedAccounts(1); });

describe("live control", () => {
  it("authenticates hosts before upgrading", async () => {
    const response = await SELF.fetch("https://20xx.xyz/v1/runner/live", { headers: { Upgrade: "websocket" } });
    expect(response.status).toBe(401);
  });

  it("sends a public snapshot with no connect code or stream key", async () => {
    await startSession();
    const { ws, messages } = await socket();
    ws.send(JSON.stringify({ type: "subscribe", job_id: null }));
    await vi.waitFor(() => expect(messages.some((m) => m.type === "capacity")).toBe(true));
    expect(JSON.stringify(messages)).not.toContain("BOT0#1");
    expect(JSON.stringify(messages)).not.toContain("key");
    ws.close();
  });

  it("requires a reservation token and pushes settings immediately", async () => {
    await startSession();
    const job = (await call("POST", "/v1/jobs", { body: CREATE })).body;
    const { ws, messages } = await socket();
    ws.send(JSON.stringify({ type: "subscribe", job_id: job.id, token: "wrong" }));
    await vi.waitFor(() => expect(messages.some((m) => m.type === "error")).toBe(true));
    expect(messages.some((m) => m.type === "job")).toBe(false);
    ws.send(JSON.stringify({ type: "subscribe", job_id: job.id, token: job.token }));
    await vi.waitFor(() => expect(messages.some((m) => m.type === "job")).toBe(true));
    await call("PATCH", `/v1/jobs/${job.id}/settings`, { token: job.token, body: { desired_return: 30 } });
    await vi.waitFor(() => expect(messages.some((m) => m.job?.settings.desired_return === 30)).toBe(true));
    ws.close();
  });

  it("writes one row for the host heartbeat", async () => {
    const session = await startSession(8);
    const { ws } = await socket(session);
    await runInDurableObject(queueStub(), async (queue, state) => {
      const server = state.getWebSockets(`host:${session}`)[0]!;
      const cursors: SqlStorageCursor<Record<string, SqlStorageValue>>[] = [];
      const original = state.storage.sql.exec.bind(state.storage.sql);
      const spy = vi.spyOn(state.storage.sql, "exec").mockImplementation((query, ...args) => {
        const cursor = original(query, ...args); cursors.push(cursor); return cursor;
      });
      try {
        await queue.webSocketMessage(server, JSON.stringify({ type: "health", status: runnerStatus(8), progress: [] }));
        expect(cursors.reduce((n, c) => n + c.rowsWritten, 0)).toBe(1);
      } finally { spy.mockRestore(); }
    });
    ws.close();
  });

  it("does not renew frozen progress or a different attempt", async () => {
    await runInDurableObject(queueStub(), (_queue, state) => {
      const sql = state.storage.sql;
      let now = START;
      const jobs = new JobStore(sql, () => now);
      const sessions = new SessionStore(sql, jobs, () => now);
      sessions.start("progress-session", { host: "test", bundle_sha256: POLICY.bundle_sha256, git_sha: "test", slots: 8, stream: false }, parsePolicyConfig(POLICY));
      jobs.createJob("progress-job", "digest", NEW_JOB);
      jobs.claimNext("progress-session/slot-0");
      const progress = [{ slot: 0, job_id: "progress-job", attempt: 1, seq: 1 }];
      sessions.report("progress-session", runnerStatus(8), progress);
      now += 15;
      sessions.report("progress-session", runnerStatus(8), progress);
      expect(sessions.leaseExpiry(jobs.row("progress-job")!, sessions.runtimeLeases())).toBe(START + 20);
      progress[0]!.seq = 2;
      sessions.report("progress-session", runnerStatus(8), progress);
      expect(sessions.leaseExpiry(jobs.row("progress-job")!, sessions.runtimeLeases())).toBe(START + 35);
      sql.exec("UPDATE jobs SET attempt = 2 WHERE id = 'progress-job'");
      expect(sessions.leaseExpiry(jobs.row("progress-job")!, sessions.runtimeLeases())).toBe(0);
    });
  });

  it("job GET does not write browser presence", async () => {
    await startSession();
    const job = (await call("POST", "/v1/jobs", { body: CREATE })).body;
    await setClock(START + 15);
    await call("GET", `/v1/jobs/${job.id}`, { token: job.token });
    await runInDurableObject(queueStub(), (_queue, state) => {
      expect(state.storage.sql.exec<{ player_seen_at: number }>("SELECT player_seen_at FROM jobs WHERE id = ?", job.id).one().player_seen_at).toBe(START);
    });
  });
});


it("pages large private snapshots below the message size limit", async () => {
  await startSession();
  const job = (await call("POST", "/v1/jobs", { body: CREATE })).body;
  await runInDurableObject(queueStub(), (_queue, state) => {
    state.storage.sql.exec(`WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 400)
      INSERT INTO games(job_id, game_number, stage, result, worker, created_at)
      SELECT ?, x, 'BATTLEFIELD', 'win', 'test', 0 FROM n`, job.id);
  });
  const { ws, messages } = await socket();
  ws.send(JSON.stringify({ type: "subscribe", job_id: job.id, token: job.token }));
  await vi.waitFor(() => {
    const pages = messages.filter((m) => m.type === "page");
    expect(pages.length).toBeGreaterThan(1);
    expect(pages.length).toBe(pages[0]!.count);
    for (const page of pages) expect(new TextEncoder().encode(JSON.stringify(page)).byteLength).toBeLessThan(16384);
    expect(JSON.parse(pages.map((m) => m.text).join("")).job.games).toHaveLength(400);
  });
  ws.close(1000);
});

it("closes slow consumers and rejects a different attempt", async () => {
  const session = await startSession();
  const job = (await call("POST", "/v1/jobs", { body: CREATE })).body;
  await call("POST", `/v1/runner/sessions/${session}/claim`, { runner: true, body: { slot: 0 } });
  const response = await call("POST", `/v1/runner/jobs/${job.id}/report`, { runner: { session, slot: 0, attempt: 2 }, body: {
    seq: 1, phase: "in_game", phase_seconds_left: null, bot_code: "HAL#9000", seen_revision: 1,
    locked_revision: 1, finished_games: [],
  }});
  expect(response.status).toBe(409);
  const { ws } = await socket();
  await runInDurableObject(queueStub(), (_queue, state) => {
    const connections = new LiveConnections(state, () => START);
    const server = state.getWebSockets("browser")[0]!;
    for (let n = 0; n < 65; n++) connections.send(server, { type: "test", n });
    expect(server.readyState).not.toBe(WebSocket.OPEN);
  });
  ws.close(1000);
});


it("pushes wind-down before refusing further settings changes", async () => {
  const session = await startSession();
  const job = (await call("POST", "/v1/jobs", { body: CREATE })).body;
  await call("POST", `/v1/runner/sessions/${session}/claim`, { runner: true, body: { slot: 0 } });
  const { ws, messages } = await socket();
  ws.send(JSON.stringify({ type: "subscribe", job_id: job.id, token: job.token }));
  await vi.waitFor(() => expect(messages.some((m) => m.type === "job")).toBe(true));
  await runInDurableObject(queueStub(), (_queue, state) => {
    state.storage.sql.exec("UPDATE settings SET value = ? WHERE key = 'write_budget'", JSON.stringify({ day: Math.floor(START / 86400), spent: 34792 }));
  });
  const changed = await call("PATCH", `/v1/jobs/${job.id}/settings`, { token: job.token, body: { desired_return: 30 } });
  expect(changed.body.wind_down).toBe("yield");
  await vi.waitFor(() => expect(messages.some((m) => m.job?.wind_down === "yield")).toBe(true));
  const refused = await call("PATCH", `/v1/jobs/${job.id}/settings`, { token: job.token, body: { desired_return: 20 } });
  expect(refused.status).toBe(503);
  expect((await call("POST", `/v1/runner/jobs/${job.id}/end`, { runner: { session, slot: 0 }, body: { reason: "yielded", retryable: false } })).status).toBe(200);
  ws.close(1000);
});


it("stops admission before database storage reaches the free-tier limit", async () => {
  await runInDurableObject(queueStub(), (_queue, state) => {
    const budget = new WriteBudget({ exec: state.storage.sql.exec.bind(state.storage.sql), databaseSize: MAX_DATABASE_BYTES }, () => START);
    expect(() => budget.requireCapacity()).toThrow("history needs maintenance");
    expect(budget.charge(1)).toBe(true);
  });
});

it("answers browser keepalives through the runtime auto-response", async () => {
  const { ws } = await socket();
  const pong = new Promise<void>((resolve) => {
    ws.addEventListener("message", (event) => { if (event.data === "pong") resolve(); });
  });
  ws.send("ping");
  await pong;
  ws.close(1000);
});
