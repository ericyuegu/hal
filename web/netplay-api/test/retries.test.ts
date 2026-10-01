import { beforeEach, describe, expect, it } from "vitest";
import { CREATE, POLICY, START as T0, call, publish, report, resetQueue, runAlarm, seedAccounts, setClock } from "./helpers";

const SESSION = "retry-session-0001";
const START = { protocol_version: 2, session_id: SESSION, host: "box", bundle_sha256: POLICY.bundle_sha256, git_sha: "g", slots: 2, stream: false };

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
    expect(status.body.accounts.filter((row: { session_id: string | null }) => row.session_id !== null)).toHaveLength(1);
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

  it("refuses another runner protocol before creating or repeating a session", async () => {
    const refused = { status: 409, body: { detail: "runner protocol 1 is not the Worker's 2" } };
    expect(await start({ ...START, protocol_version: 1 })).toMatchObject(refused);
    expect((await call("GET", "/v1/admin/status", { admin: true })).body.sessions).toHaveLength(0);
    await start();
    expect(await start({ ...START, protocol_version: 1 })).toMatchObject(refused);
    const { protocol_version: _, ...unversioned } = START;
    expect((await start(unversioned)).status).toBe(422);
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
    const events = await call("GET", `/v1/admin/events?job=${first.body.id}`, { admin: true });
    expect(events.body.events.filter((event: { kind: string }) => event.kind === "job_claimed")).toHaveLength(1);
    const other = await call("POST", `/v1/runner/sessions/${SESSION}/claim`, { runner: true, body: { slot: 1 } });
    expect(other.status).toBe(204);
  });

  it("returns the slot's leased job when a claim is repeated after a drain", async () => {
    await start();
    await report(SESSION, 2);
    await call("POST", "/v1/jobs", { body: CREATE });
    const first = await call("POST", `/v1/runner/sessions/${SESSION}/claim`, { runner: true, body: { slot: 0 } });
    await call("POST", `/v1/runner/sessions/${SESSION}/drain`, { runner: true });
    const second = await call("POST", `/v1/runner/sessions/${SESSION}/claim`, { runner: true, body: { slot: 0 } });
    expect(second).toMatchObject({ status: 200, body: first.body });
    expect((await call("POST", `/v1/runner/sessions/${SESSION}/claim`, { runner: true, body: { slot: 1 } })).status).toBe(409);
  });

  it("returns the slot's leased job when a claim is repeated after a policy republish", async () => {
    await start();
    await report(SESSION, 2);
    await call("POST", "/v1/jobs", { body: CREATE });
    const first = await call("POST", `/v1/runner/sessions/${SESSION}/claim`, { runner: true, body: { slot: 0 } });
    await publish({ ...POLICY, bundle_sha256: "e".repeat(64) });
    const second = await call("POST", `/v1/runner/sessions/${SESSION}/claim`, { runner: true, body: { slot: 0 } });
    expect(second).toMatchObject({ status: 200, body: first.body });
    expect((await call("POST", `/v1/runner/sessions/${SESSION}/claim`, { runner: true, body: { slot: 1 } })).status).toBe(409);
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

  it("reports the silence alarm's count when a silent session is ended", async () => {
    await start();
    await report(SESSION, 2);
    await call("POST", "/v1/jobs", { body: CREATE });
    await call("POST", `/v1/runner/sessions/${SESSION}/claim`, { runner: true, body: { slot: 0 } });
    await setClock(T0 + 30);
    expect(await runAlarm()).toBe(true);
    expect(await call("DELETE", `/v1/runner/sessions/${SESSION}`, { runner: true })).toMatchObject({
      status: 200,
      body: { failed: 1 },
    });
  });

  it("reports drain and an empty stream grant in status responses", async () => {
    await start();
    expect(await report(SESSION, 2)).toMatchObject({ status: 200, body: { draining: false, stream: null } });
  });

  it("grants one stream without exposing its key through admin state", async () => {
    const streamed = { ...START, stream: true };
    await start(streamed);
    expect(await report(SESSION, 2)).toMatchObject({
      status: 200,
      body: { draining: false, stream: { slot: 0, key: "live_test_stream_key" } },
    });
    expect((await report(SESSION, 2)).body.stream).toEqual({ slot: 0, key: "live_test_stream_key" });
    const status = await call("GET", "/v1/admin/status", { admin: true });
    expect(status.body.stream).toMatchObject({ session_id: SESSION, slot: 0 });
    expect(JSON.stringify(status.body)).not.toContain("live_test_stream_key");
    const events = await call("GET", "/v1/admin/events", { admin: true });
    expect(JSON.stringify(events.body)).not.toContain("live_test_stream_key");
    expect(events.body.events.filter((event: { kind: string }) => event.kind === "stream_lease_granted")).toHaveLength(1);
    await call("POST", `/v1/runner/sessions/${SESSION}/drain`, { runner: true });
    expect((await call("GET", "/v1/admin/status", { admin: true })).body.stream).toBeNull();
    const released = await call("GET", "/v1/admin/events", { admin: true });
    expect(released.body.events.filter((event: { kind: string }) => event.kind === "stream_lease_released")).toHaveLength(1);
  });

  it("prefers an idle live stream slot without changing FIFO or retry behavior", async () => {
    await start({ ...START, stream: true });
    await report(SESSION, 2);
    await call("POST", "/v1/jobs", { body: CREATE });
    expect(
      await call("POST", `/v1/runner/sessions/${SESSION}/claim`, { runner: true, body: { slot: 1 } }),
    ).toMatchObject({ status: 204, body: null });
    const first = await call("POST", `/v1/runner/sessions/${SESSION}/claim`, { runner: true, body: { slot: 0 } });
    expect(first).toMatchObject({ status: 200, body: { player_code: CREATE.player_code } });
    expect(
      await call("POST", `/v1/runner/sessions/${SESSION}/claim`, { runner: true, body: { slot: 0 } }),
    ).toMatchObject({ status: 200, body: first.body });

    await call("POST", `/v1/runner/jobs/${first.body.id}/connecting`, {
      runner: { session: SESSION, slot: 0 },
      body: { connect_code: "BOT0#1" },
    });

    await call("POST", "/v1/jobs", { body: { ...CREATE, player_code: "OTHER#1" } });
    expect((await call("POST", `/v1/runner/sessions/${SESSION}/claim`, {
      runner: true, body: { slot: 1 },
    })).status).toBe(204);
    await call("POST", `/v1/runner/jobs/${first.body.id}/playing`, {
      runner: { session: SESSION, slot: 0 },
    });
    expect(
      await call("POST", `/v1/runner/sessions/${SESSION}/claim`, { runner: true, body: { slot: 1 } }),
    ).toMatchObject({ status: 200, body: { player_code: "OTHER#1" } });
  });
});

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
