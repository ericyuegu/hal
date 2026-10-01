import { beforeEach, describe, expect, it } from "vitest";
import { CREATE, call, publish, resetQueue, seedAccounts, startSession } from "./helpers";

beforeEach(async () => {
  await resetQueue();
  await publish();
  await seedAccounts(1);
});

async function create(code: string) {
  const result = await call("POST", "/v1/jobs", { body: { ...CREATE, player_code: code } });
  expect(result.status).toBe(201);
  return result.body;
}
function claim(session: string, slot: number) {
  return call("POST", `/v1/runner/sessions/${session}/claim`, { runner: true, body: { slot } });
}
function cleanup(session: string, slot: number, job: { id: string; attempt: number }) {
  return call("POST", `/v1/runner/sessions/${session}/pairing-finished`, {
    runner: true, body: { slot, job_id: job.id, attempt: job.attempt },
  });
}
async function play(session: string, slot: number, id: string) {
  const runner = { session, slot };
  expect(
    (
      await call("POST", `/v1/runner/jobs/${id}/report`, {
        runner,
        body: {
          seq: 1,
          phase: "in_game",
          phase_seconds_left: null,
          bot_code: "BOT0#1",
          seen_revision: 1,
          locked_revision: 1,
          finished_games: [],
        },
      })
    ).status,
  ).toBe(200);
}

describe("shared account pairing", () => {
  it("allows eight games but only one new pairing at a time", async () => {
    const session = await startSession(8);
    const first = await create("FIRST#1");
    const second = await create("SECOND#1");
    const held = await claim(session, 0);
    expect(held.body.id).toBe(first.id);
    expect((await claim(session, 1)).status).toBe(204);
    expect((await claim(session, 0)).body).toEqual(held.body);
    await play(session, 0, first.id);
    // A live game releases the pairing gate, so the session's other slots can claim.
    expect((await claim(session, 1)).body.id).toBe(second.id);
    // A late cleanup from the first game cannot clear the second pairing.
    expect((await cleanup(session, 0, held.body)).status).toBe(200);
    expect((await call("GET", `/v1/runner/sessions/${session}/pairing`, { runner: true })).body.pairing.job_id).toBe(second.id);
    expect((await call("GET", "/v1/capacity")).body.active).toBe(2);
  });

  it("retains a canceled pairing until its Dolphin has closed", async () => {
    const session = await startSession(8);
    const first = await create("CANCEL#1");
    const second = await create("NEXT#1");
    const held = await claim(session, 0);
    expect((await call("DELETE", `/v1/jobs/${first.id}`, { token: first.token })).status).toBe(200);
    expect((await claim(session, 1)).status).toBe(204);
    expect(
      (
        await call("POST", `/v1/runner/jobs/${first.id}/end`, {
          runner: { session, slot: 0 },
          body: { reason: "player_canceled", retryable: false },
        })
      ).status,
    ).toBe(200);
    expect((await cleanup(session, 0, held.body)).status).toBe(200);
    expect((await cleanup(session, 0, held.body)).status).toBe(200);
    expect((await claim(session, 1)).body.id).toBe(second.id);
  });

  it("does not let a delayed cleanup clear another attempt of the same job", async () => {
    const session = await startSession(8);
    const job = await create("RETRY#1");
    const first = await claim(session, 0);
    expect(
      (
        await call("POST", `/v1/runner/jobs/${job.id}/end`, {
          runner: { session, slot: 0 },
          body: { reason: "service_failure", retryable: true },
        })
      ).status,
    ).toBe(200);
    await cleanup(session, 0, first.body);
    const second = await claim(session, 0);
    expect(second.body.attempt).toBe(first.body.attempt + 1);
    await cleanup(session, 0, first.body);
    expect((await call("GET", `/v1/runner/sessions/${session}/pairing`, { runner: true })).body.pairing.attempt).toBe(second.body.attempt);
    expect((await claim(session, 1)).status).toBe(204);
  });

  it("releases the account and pending pairing when its session ends", async () => {
    const session = await startSession(8);
    await create("END#1");
    const held = await claim(session, 0);
    expect((await call("DELETE", `/v1/runner/sessions/${session}`, { runner: true })).status).toBe(200);
    expect((await cleanup(session, 0, held.body)).status).toBe(200);
    const next = await startSession(8);
    expect((await call("GET", `/v1/runner/sessions/${next}/pairing`, { runner: true })).body.pairing).toBeNull();
  });
});
