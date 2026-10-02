import { env, runInDurableObject } from "cloudflare:test";
import { expect, it, vi } from "vitest";
import { JobStore, type Row } from "../src/store";
import { sha256Hex } from "../src/domain";
import { NEW_JOB, POLICY, runnerStatus } from "./helpers";

it("keeps eight slots' full-day recurring work under the free-tier reserve", async () => {
  const stub = env.QUEUE.get(env.QUEUE.idFromName(crypto.randomUUID()));
  await runInDurableObject(stub, async (queue, state) => {
    const start = 10 * 86_400;
    let now = start;
    await queue.setTestClock(now);
    await queue.putPolicy(POLICY);
    await queue.putAccounts([{ connect_code: "HAL#9000", r2_key: "account", sha256: "a".repeat(64) }]);
    const session = "day-cost-session";
    await queue.startSession({ protocol_version: 4, session_id: session, host: "day", bundle_sha256: POLICY.bundle_sha256, git_sha: "day", slots: 8, stream: false });
    const digest = await sha256Hex("test-player-token");
    state.storage.sql.exec(`WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 10000)
      INSERT INTO jobs(id, token_digest, player_code, online_delay, status, queue_seq, settings_revision,
        character, imitation, temperature, player_seen_at, created_at, updated_at)
      SELECT 'old-' || x, 'digest', 'OLD' || x || '#1', 2, 'ended', x, 1, 'FOX', 'IBDW#0', 1, 0, 0, 0 FROM n`);
    state.storage.sql.exec(`WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 10000)
      INSERT INTO sessions(id, host, bundle_sha256, git_sha, slots, wants_stream, started_at, last_seen_at, ended_at)
      SELECT 'old-' || x, 'old', ?, 'old', 1, 0, 0, 0, 1 FROM n`, POLICY.bundle_sha256);
    state.storage.sql.exec(`WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 5000)
      INSERT INTO events(at, kind, detail) SELECT -2000000, 'old', '{}' FROM n`);
    const jobs = new JobStore(state.storage.sql, () => now);
    for (let slot = 0; slot < 8; slot++) {
      jobs.createJob(`job-${slot}`, "digest", { ...NEW_JOB, player_code: `P${slot}#1` });
      jobs.claimNext(`${session}/slot-${slot}`);
      jobs.report(`job-${slot}`, `${session}/slot-${slot}`, {
        seq: 1, phase: "in_game", phase_seconds_left: null, bot_code: "HAL#9000",
        seen_revision: 1, locked_revision: 1, finished_games: [],
      });
    }
    for (let n = 0; n < 20; n++) jobs.createJob(`queued-${n}`, digest, { ...NEW_JOB, player_code: `Q${n}#1` });
    const response = await queue.fetch(new Request("https://test/v1/runner/live", { headers: { Upgrade: "websocket", "X-HAL-Session": session } }));
    const client = response.webSocket!;
    client.accept();
    const host = state.getWebSockets(`host:${session}`)[0]!;
    // Presence receipt times come from Cloudflare, not a heartbeat SQL update.
    for (let n = 0; n < 100; n++) {
      const pair = new WebSocketPair();
      state.acceptWebSocket(pair[1], ["browser"]);
      pair[1].serializeAttachment({ role: "browser", job: n < 20 ? `queued-${n}` : null, seen: start, sent: 0, acked: 0, pending: [] });
      pair[0].accept();
    }
    const timestamp = vi.spyOn(state, "getWebSocketAutoResponseTimestamp").mockImplementation(() => new Date(now * 1000));
    const original = state.storage.sql.exec.bind(state.storage.sql);
    let cursors: SqlStorageCursor<Row>[] = [];
    let retentionCursors: SqlStorageCursor<Row>[] = [];
    const sql = vi.spyOn(state.storage.sql, "exec").mockImplementation((query, ...args) => {
      const cursor = original(query, ...args);
      if (!query.includes("test_clock") && !(args.includes("test_clock"))) cursors.push(cursor);
      if (query.startsWith("DELETE FROM events")) retentionCursors.push(cursor);
      return cursor;
    });
    let reads = 0;
    let writes = 0;
    let alarms = 0;
    let commandWrites = 0;
    let commands = 0;
    let admitted = 0;
    let refused = 0;
    let retentionWrites = 0;
    try {
      for (let tick = 1; tick <= 8640; tick++) {
        now = start + (tick - 1) * 10;
        await queue.setTestClock(now);
        // Model browser acknowledgements, which only change socket attachments.
        for (const ws of state.getWebSockets()) {
          const peer = ws.deserializeAttachment();
          ws.serializeAttachment({ ...peer, acked: peer.sent, pending: [] });
        }
        await queue.webSocketMessage(host, JSON.stringify({ type: "health", status: { ...runnerStatus(8), game_fps: 60 + tick % 5 / 1000 }, progress:
          Array.from({ length: 8 }, (_, slot) => ({ slot, job_id: `job-${slot}`, attempt: 1, seq: tick + 1 })) }));
        if (tick % 2 === 0) {
          await state.storage.deleteAlarm();
          await queue.alarm();
          alarms++;
        }
        reads += cursors.reduce((sum, cursor) => sum + cursor.rowsRead, 0);
        writes += cursors.reduce((sum, cursor) => sum + cursor.rowsWritten, 0);
        retentionWrites += retentionCursors.reduce((sum, cursor) => sum + cursor.rowsWritten, 0);
        cursors = [];
        retentionCursors = [];
        // Ten thousand command attempts; excess settings traffic must leave the
        // write reserve available for games and cleanup.
        for (let j = 0; j < (tick <= 1360 ? 2 : 1); j++) {
          const result = await queue.updateSettings(`queued-${commands % 20}`, "test-player-token", { desired_return: commands % 40 });
          expect([200, 503]).toContain(result.status);
          if (result.status === 200) admitted++; else refused++;
          commands++;
        }
        reads += cursors.reduce((sum, cursor) => sum + cursor.rowsRead, 0);
        commandWrites += cursors.reduce((sum, cursor) => sum + cursor.rowsWritten, 0);
        cursors = [];
      }
      // Charge both alarm removal and replacement conservatively.
      const recurringWrites = writes - retentionWrites + 2 * alarms;
      const totalWrites = recurringWrites + retentionWrites + commandWrites;
      console.log(JSON.stringify({ day: "eight slots, twenty waiting", reads, writes, alarms, recurringWrites }));
      expect({ reads, recurringWrites, commandWrites, retentionWrites, totalWrites, commands, admitted, refused }).toMatchInlineSnapshot(`
        {
          "admitted": 4348,
          "commandWrites": 17392,
          "commands": 10000,
          "reads": 564421,
          "recurringWrites": 17304,
          "refused": 5652,
          "retentionWrites": 5000,
          "totalWrites": 39696,
        }
      `);
      expect(totalWrites).toBeLessThan(70_000);
      expect(commands).toBe(10_000);
      expect(refused).toBeGreaterThan(0);
      expect(recurringWrites).toBeLessThanOrEqual(20_000);
      expect(reads).toBeLessThan(1_000_000);
      expect(jobs.activeCount()).toBe(8);
      expect(jobs.queueDepth()).toBe(20);
    } finally {
      sql.mockRestore(); timestamp.mockRestore();
      for (const ws of state.getWebSockets()) ws.close(1000);
      client.close(1000);
    }
  });
}, 120_000);
