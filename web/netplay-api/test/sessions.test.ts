import { env, runInDurableObject } from "cloudflare:test";
import { describe, expect, it } from "vitest";
import { HttpError } from "../src/domain";
import { EVENT_SCHEMA, EventLog } from "../src/events";
import { parsePolicyConfig } from "../src/policy";
import { SESSION_SCHEMA, SessionStore } from "../src/sessions";
import { JOB_SCHEMA, JobStore } from "../src/store";
import { Clock, NEW_JOB } from "./helpers";
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
  it("leases one account per session and refuses mismatched bundles", () =>
    world(({ sessions }) => {
      const started = sessions.start("s1", START, policy);
      expect(started.accounts.map((grant) => [grant.slot, grant.connect_code])).toEqual([
        [0, "BOT0#1"],
        [1, "BOT0#1"],
      ]);
      expect(refused(() => sessions.start("s2", { ...START, bundle_sha256: "f".repeat(64) }, policy)).status).toBe(409);
      sessions.start("s2", START, policy);
      sessions.start("s3", START, policy);
      expect(refused(() => sessions.start("s5", START, policy))).toEqual({
        status: 409, detail: "no bot accounts are free",
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
      expect(sessions.endSilent()).toEqual({ sessions: [], jobs: [] });
      sessions.report("s1", readyStatus(2, 2, 12345));
      clock.advance(29);
      expect(sessions.endSilent()).toEqual({ sessions: [], jobs: [] });
      clock.advance(1);
      sessions.endSilent();
      expect(refused(() => sessions.live("s1"))).toEqual({ status: 410, detail: "session has ended" });
    }));

  it("leases one stream until its holder drains, ends, or goes silent", () =>
    world(({ sessions, clock }) => {
      sessions.start("s1", START, policy);
      expect(sessions.report("s1", readyStatus(2))).toEqual({
        draining: false,
        streamHolder: true,
        streamGranted: true,
      });
      expect(sessions.report("s1", readyStatus(2))).toEqual({
        draining: false,
        streamHolder: true,
        streamGranted: false,
      });
      expect(sessions.summary().stream).toMatchObject({ session_id: "s1", slot: 0 });

      sessions.drain("s1");
      expect(sessions.summary().stream).toBeNull();
      sessions.start("s2", { ...START, slots: 1 }, policy);
      expect(sessions.report("s2", readyStatus(1)).streamHolder).toBe(true);
      sessions.end("s2", "ended");
      expect(sessions.summary().stream).toBeNull();

      sessions.start("s3", { ...START, slots: 1 }, policy);
      sessions.report("s3", readyStatus(1));
      clock.advance(30);
      sessions.endSilent();
      expect(sessions.summary().stream).toBeNull();
    }));

  it("does not grant a stream to an opted-out or competing session", () =>
    world(({ sessions }) => {
      sessions.start("s1", { ...START, stream: false, slots: 1 }, policy);
      expect(sessions.report("s1", readyStatus(1)).streamHolder).toBe(false);
      sessions.start("s2", { ...START, slots: 1 }, policy);
      sessions.start("s3", { ...START, slots: 1 }, policy);
      expect(sessions.report("s2", readyStatus(1)).streamHolder).toBe(true);
      expect(sessions.report("s3", readyStatus(1)).streamHolder).toBe(false);
    }));

  it("ends a silent session, failing its leases and freeing its accounts", () =>
    world(({ jobs, sessions, clock }) => {
      sessions.start("s1", START, policy);
      jobs.createJob("j1", "d1", NEW_JOB);
      jobs.claimNext(sessions.claimWorker("s1", 0, policy));
      clock.advance(30);
      expect(sessions.endSilent()).toEqual({ sessions: ["s1"], jobs: ["j1"] });
      expect(jobs.row("j1")?.status).toBe("queued");
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
      clock.advance(21);
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
