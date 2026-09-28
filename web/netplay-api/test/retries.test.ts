import { beforeEach, describe, expect, it } from "vitest";
import { CREATE, POLICY, START as T0, call, publish, report, resetQueue, runAlarm, seedAccounts, setClock } from "./helpers";

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
    const events = await call("GET", `/v1/admin/events?job=${first.body.id}`, { admin: true });
    expect(events.body.events.filter((event: { kind: string }) => event.kind === "job_claimed")).toHaveLength(1);
    const other = await call("POST", `/v1/runner/sessions/${SESSION}/claim`, { runner: true, body: { slot: 1 } });
    expect(other.body.player_code).toBe("OTHER#1");
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
});
