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
  const claimed = await call("POST", `/v1/runner/sessions/${session}/claim`, {
    runner: true,
    body: { slot: 0 },
  });
  expect(claimed.body).toMatchObject({ id: job.id, status: "assigned", settings: { revision: 1 } });
  return { session, job, runner };
}

function observed(seq: number, extra: Record<string, unknown> = {}) {
  return {
    seq,
    phase: "character_select",
    phase_seconds_left: 5,
    bot_code: "BOT0#1",
    seen_revision: 1,
    locked_revision: null,
    finished_games: [],
    ...extra,
  };
}

describe("protocol 3 routes", () => {
  it("rejects a protocol 2 runner", async () => {
    await publish();
    await seedAccounts(1);
    const started = await call("POST", "/v1/runner/sessions", {
      runner: true,
      body: {
        protocol_version: 2,
        session_id: "s".repeat(22),
        host: "h",
        bundle_sha256: "0".repeat(64),
        git_sha: "g",
        slots: 1,
        stream: false,
      },
    });
    expect(started.status).toBe(409);
  });

  it("delivers settings changes in the report response", async () => {
    const { job, runner } = await assigned();
    await call("PATCH", `/v1/jobs/${job.id}/settings`, { token: job.token, body: { character: "FALCO" } });
    const response = await call("POST", `/v1/runner/jobs/${job.id}/report`, {
      runner,
      body: observed(1),
    });
    expect(response.body.settings).toMatchObject({ revision: 2, character: "FALCO" });
  });

  it("shows the page the runner's phase, deadline, and games", async () => {
    const { job, runner } = await assigned();
    await call("POST", `/v1/runner/jobs/${job.id}/report`, {
      runner,
      body: observed(1, {
        finished_games: [{ number: 1, stage: "POKEMON_STADIUM", result: "loss" }],
      }),
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
    const reported = await call("POST", `/v1/runner/jobs/${job.id}/report`, {
      runner,
      body: observed(1),
    });
    expect(reported.body.wind_down).toBe("player");
    const ended = await call("POST", `/v1/runner/jobs/${job.id}/end`, {
      runner,
      body: { reason: "player_canceled", retryable: false },
    });
    expect(ended.body).toMatchObject({ status: "ended", end_reason: "player_canceled" });
    expect(
      (await call("POST", `/v1/runner/jobs/${job.id}/report`, { runner, body: observed(2) })).status,
    ).toBe(409);
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
    expect((await call("GET", `/v1/jobs/${job.id}`, { token: job.token })).body).toMatchObject({
      status: "queued",
      queue_position: 1,
    });
  });

  it("removes the transition routes", async () => {
    const { job, runner } = await assigned();
    for (const action of ["connecting", "playing", "finish-game", "no-show", "fail", "forfeit", "heartbeat"]) {
      expect((await call("POST", `/v1/runner/jobs/${job.id}/${action}`, { runner, body: {} })).status).toBe(404);
    }
    expect((await call("POST", `/v1/jobs/${job.id}/rematch`, { token: job.token, body: {} })).status).toBe(404);
  });
});
