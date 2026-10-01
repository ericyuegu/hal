import { env, runInDurableObject } from "cloudflare:test";
import { describe, expect, it, vi } from "vitest";
import { EVENT_RETENTION_SECONDS, sha256Hex, workerId } from "../src/domain";
import { EventLog } from "../src/events";
import { parsePolicyConfig } from "../src/policy";
import { Queue, type ApiResult } from "../src/queue";
import { SessionStore } from "../src/sessions";
import { JobStore, type Row } from "../src/store";
import { NEW_JOB, POLICY, runnerStatus } from "./helpers";

const SESSION = "cost-test-session";
const TOKEN = "cost-test-job-token";
const NOW = 1_000_000;

interface World {
  queue: Queue;
  state: DurableObjectState;
  jobs: JobStore;
  sessions: SessionStore;
}

async function world(history: number, use: (world: World) => Promise<void>, slots = 2): Promise<void> {
  const stub = env.QUEUE.get(env.QUEUE.idFromName(crypto.randomUUID()));
  await runInDurableObject(stub, async (queue, state) => {
    await queue.setTestClock(NOW);
    const sql = state.storage.sql;
    const policy = parsePolicyConfig(POLICY);
    sql.exec("INSERT INTO policy(id, config, published_at) VALUES (1, ?, ?)", JSON.stringify(policy), NOW);
    if (history > 0) {
      sql.exec(
        `WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < ?)
         INSERT INTO jobs(id, token_digest, player_code, online_delay, status, end_reason, queue_seq,
                          settings_revision, character, imitation, temperature, player_seen_at, created_at, updated_at)
         SELECT 'old-job-' || x, 'digest', 'OLD' || x || '#1', 2, 'ended', 'player_canceled', x,
                1, 'FOX', 'IBDW#0', 1, ?, ?, ? FROM n`, history, NOW - 100, NOW - 100, NOW - 100,
      );
      sql.exec(
        `WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < ?)
         INSERT INTO sessions(id, host, bundle_sha256, git_sha, slots, wants_stream,
                              started_at, last_seen_at, ended_at)
         SELECT 'old-session-' || x, 'old-host', ?, 'old-source', 1, 0, ?, ?, ? FROM n`,
        history, policy.bundle_sha256, NOW - 100, NOW - 100, NOW - 50,
      );
    }
    const jobs = new JobStore(sql, () => NOW);
    const sessions = new SessionStore(sql, jobs, () => NOW);
    sessions.putAccounts(Array.from({ length: slots }, (_, i) => ({ connect_code: `BOT${i}#1`, r2_key: `accounts/${i}`, sha256: "a".repeat(64) })));
    sessions.start(SESSION, { host: "probe", bundle_sha256: policy.bundle_sha256, git_sha: "probe", slots, stream: false }, policy);
    jobs.createJob("live-job", await sha256Hex(TOKEN), { ...NEW_JOB, player_code: "PLAYER#1" });
    const worker = workerId(SESSION, 0);
    jobs.claimNext(worker);
    jobs.report("live-job", worker, {
      seq: 1,
      phase: "in_game",
      phase_seconds_left: null,
      bot_code: "BOT0#1",
      seen_revision: 1,
      locked_revision: 1,
      finished_games: [],
    });
    await queue.reportStatus(SESSION, runnerStatus(slots));
    await use({ queue, state, jobs, sessions });
  });
}

async function measure(state: DurableObjectState, operation: () => Promise<ApiResult | void>) {
  const cursors: SqlStorageCursor<Row>[] = [];
  const sql = state.storage.sql;
  const original = sql.exec.bind(sql);
  const exec = vi.spyOn(sql, "exec").mockImplementation((query, ...args) => {
    const cursor = original(query, ...args);
    cursors.push(cursor);
    return cursor;
  });
  const setAlarm = vi.spyOn(state.storage, "setAlarm");
  const deleteAlarm = vi.spyOn(state.storage, "deleteAlarm");
  const getAlarm = vi.spyOn(state.storage, "getAlarm");
  try {
    const response = await operation();
    if (response !== undefined) expect([200, 204]).toContain(response.status);
    return {
      rowsRead: cursors.reduce((n, cursor) => n + cursor.rowsRead, 0),
      rowsWritten: cursors.reduce((n, cursor) => n + cursor.rowsWritten, 0),
      alarmReads: getAlarm.mock.calls.length,
      alarmWrites: setAlarm.mock.calls.length + deleteAlarm.mock.calls.length,
    };
  } finally {
    exec.mockRestore();
    setAlarm.mockRestore();
    deleteAlarm.mockRestore();
    getAlarm.mockRestore();
  }
}

describe("recurring queue costs", () => {
  it("measures sixteen-slot poll costs independently of historical rows", async () => {
    const totals: Record<string, Awaited<ReturnType<typeof measure>>>[] = [];
    for (const history of [0, 10_000]) {
      await world(history, async ({ queue, state }) => {
        const idleClaim = await measure(state, () => queue.claim(SESSION, { slot: 15 }));
        const report = await measure(state, () => queue.reportStatus(SESSION, runnerStatus(16)));
        const capacity = await measure(state, () => queue.capacity());
        expect(idleClaim.rowsWritten).toBe(0);
        expect(idleClaim.alarmWrites).toBe(0);
        totals.push({ idleClaim, report, capacity });
      }, 16);
    }
    expect(totals[1]).toEqual(totals[0]);
    expect(totals[0]).toMatchInlineSnapshot(`
      {
        "capacity": {
          "alarmReads": 0,
          "alarmWrites": 0,
          "rowsRead": 22,
          "rowsWritten": 0,
        },
        "idleClaim": {
          "alarmReads": 0,
          "alarmWrites": 0,
          "rowsRead": 8,
          "rowsWritten": 0,
        },
        "report": {
          "alarmReads": 1,
          "alarmWrites": 0,
          "rowsRead": 26,
          "rowsWritten": 1,
        },
      }
    `);
  });

  it("keeps poll, snapshot, and alarm reads bounded with 10,000 completed jobs and sessions", async () => {
    const totals: Record<string, Awaited<ReturnType<typeof measure>>>[] = [];
    for (const history of [0, 10_000]) {
      await world(history, async ({ queue, state }) => {
        const calls = {
          idleClaim: () => queue.claim(SESSION, { slot: 1 }),
          report: () => queue.reportStatus(SESSION, runnerStatus(2)),
          capacity: () => queue.capacity(),
          playerJob: () => queue.getJob("live-job", TOKEN),
          workerJob: () => queue.workerJob(SESSION, 0, "live-job"),
          snapshot: () =>
            queue.report(SESSION, 0, "live-job", {
              seq: 2,
              phase: "in_game",
              phase_seconds_left: null,
              bot_code: "BOT0#1",
              seen_revision: 1,
              locked_revision: 1,
              finished_games: [],
            }),
          alarm: () => queue.alarm(),
        };
        const costs: Record<string, Awaited<ReturnType<typeof measure>>> = {};
        for (const [name, call] of Object.entries(calls)) {
          costs[name] = await measure(state, call);
          expect(costs[name]!.rowsRead, `${history} old rows: ${name}`).toBeLessThan(64);
        }
        totals.push(costs);
      });
    }
    expect(totals[1]).toEqual(totals[0]);
  });

  it("does no SQL or alarm writes for reads and empty claims", () => world(50, async ({ queue, state }) => {
    const cost = await measure(state, async () => {
      await queue.options();
      await queue.capacity();
      await queue.activePolicy();
      await queue.getJob("live-job", TOKEN);
      await queue.workerJob(SESSION, 0, "live-job");
      await queue.adminStatus();
      await queue.adminEvents({});
      expect(await queue.claim(SESSION, { slot: 1 })).toMatchObject({ status: 204 });
    });
    expect(cost.rowsWritten).toBe(0);
    expect(cost.alarmReads).toBe(0);
    expect(cost.alarmWrites).toBe(0);
  }));

  it("counts a multi-slot session once and expires it through its account leases", () => world(100, async ({ state, jobs }) => {
    const sessions = new SessionStore(state.storage.sql, jobs, () => NOW + 30);
    expect(sessions.nextDeadline()).toBe(NOW + 30);
    expect(sessions.endSilent()).toEqual({ sessions: [SESSION], jobs: ["live-job"] });
    expect(sessions.nextDeadline()).toBeNull();
    expect(sessions.endSilent()).toEqual({ sessions: [], jobs: [] });
    expect(sessions.summary().accounts.every(account => account.session_id === null)).toBe(true);
  }));
});

describe("alarm writes", () => {
  it("keeps an earlier alarm across reports and reconstruction, then schedules a new earlier deadline", async () => {
    const stub = env.QUEUE.get(env.QUEUE.idFromName(crypto.randomUUID()));
    await runInDurableObject(stub, async (_instance, state) => {
      const clock = vi.spyOn(Date, "now").mockReturnValue(Date.now() + 86_400_000);
      const queue = new Queue(state, { ...env, HAL_TEST_CLOCK: "0" });
      const now = Date.now() / 1000;
      const jobs = new JobStore(state.storage.sql, () => Date.now() / 1000);
      const sessions = new SessionStore(state.storage.sql, jobs, () => Date.now() / 1000);
      const policy = parsePolicyConfig(POLICY);
      try {
        await queue.putPolicy(POLICY);
        const retention = await state.storage.getAlarm();
        expect(retention).toBe((now + EVENT_RETENTION_SECONDS) * 1000);
        sessions.putAccounts([{ connect_code: "BOT#1", r2_key: "accounts/bot", sha256: "a".repeat(64) }]);
        sessions.start(SESSION, { host: "probe", bundle_sha256: policy.bundle_sha256, git_sha: "probe", slots: 1, stream: false }, policy);
        await queue.reportStatus(SESSION, runnerStatus(1));
        const firstAlarm = await state.storage.getAlarm();
        expect(firstAlarm).toBe((now + 30) * 1000);
        clock.mockReturnValue((now + 2) * 1000);
        const reopened = new Queue(state, { ...env, HAL_TEST_CLOCK: "0" });
        const cost = await measure(state, () => reopened.reportStatus(SESSION, runnerStatus(1)));
        expect(cost.alarmWrites).toBe(0);
        expect(await state.storage.getAlarm()).toBe(firstAlarm);
        jobs.createJob("early-job", "digest", { ...NEW_JOB, player_code: "PLAYER#2" });
        expect(await reopened.claim(SESSION, { slot: 0 })).toMatchObject({ status: 200 });
        expect(await state.storage.getAlarm()).toBe((now + 22) * 1000);
        clock.mockReturnValue((now + 22) * 1000);
        await state.storage.deleteAlarm();
        await reopened.alarm();
        expect(jobs.row("early-job")!.status).toBe("queued");
        expect(await state.storage.getAlarm()).toBe((now + 32) * 1000);
      } finally {
        clock.mockRestore();
      }
    });
  });

  it("prunes events at the exact retention deadline so the alarm can clear", async () => {
    const stub = env.QUEUE.get(env.QUEUE.idFromName(crypto.randomUUID()));
    await runInDurableObject(stub, async (queue, state) => {
      new EventLog(state.storage.sql, () => NOW).log("old_event", {});
      await queue.setTestClock(NOW + EVENT_RETENTION_SECONDS);
      await state.storage.setAlarm(Date.now() + 86_400_000);
      await queue.alarm();
      expect(new EventLog(state.storage.sql, () => NOW).oldest()).toBeNull();
      expect(await state.storage.getAlarm()).toBeNull();
    });
  });
});
