import { QUEUE_INSTANCE } from "../src/queue";
import { env, runDurableObjectAlarm, runInDurableObject, SELF } from "cloudflare:test";
import policyJson from "./transcripts/policy.json";
import { JOB_SCHEMA, JobStore } from "../src/store";

export const START = 1_000_000;

export class Clock {
  now = START;
  advance(seconds: number): void {
    this.now += seconds;
  }
}

export function queueStub() {
  return env.QUEUE.get(env.QUEUE.idFromName(QUEUE_INSTANCE));
}

// Runs store-level tests inside a Durable Object so they use real SQLite storage.
export async function withStore(fn: (store: JobStore, clock: Clock, sql: SqlStorage) => void): Promise<void> {
  const stub = env.QUEUE.get(env.QUEUE.idFromName(crypto.randomUUID()));
  await runInDurableObject(stub, (_instance, state) => {
    state.storage.sql.exec(JOB_SCHEMA);
    const clock = new Clock();
    fn(new JobStore(state.storage.sql, () => clock.now), clock, state.storage.sql);
  });
}

export const NEW_JOB = {
  player_code: "CRYO#610",
  character: "FOX",
  imitation: "IBDW#0",
  stage: null,
  online_delay: 2,
  desired_return: 20,
  temperature: 1,
};

export const RUNNER_TOKEN = "runner-test-token";
export const OTHER_RUNNER_TOKEN = "runner-other-token";
export const ADMIN_TOKEN = "admin-test-token";
export const POLICY = policyJson;

export interface Call {
  status: number;
  body: any;
  headers: Headers;
}

export interface CallOptions {
  body?: unknown;
  rawBody?: string;
  token?: string;
  runner?: { session: string; slot: number; attempt?: number } | true;
  runnerToken?: string;
  admin?: boolean;
  ip?: string;
}

export async function call(method: string, path: string, options: CallOptions = {}): Promise<Call> {
  const headers = new Headers({ "CF-Connecting-IP": options.ip ?? `198.51.100.${Math.floor(Math.random() * 250)}` });
  if (options.token !== undefined) headers.set("Authorization", `Bearer ${options.token}`);
  if (options.runner) {
    headers.set("Authorization", `Bearer ${options.runnerToken ?? RUNNER_TOKEN}`);
    if (options.runner !== true) {
      headers.set("X-HAL-Session", options.runner.session);
      headers.set("X-HAL-Slot", String(options.runner.slot));
      headers.set("X-HAL-Attempt", String(options.runner.attempt ?? 1));
    }
  }
  if (options.admin) headers.set("Authorization", `Bearer ${ADMIN_TOKEN}`);
  let body = options.rawBody;
  if (options.body !== undefined) {
    body = JSON.stringify(options.body);
    headers.set("Content-Type", "application/json");
  }
  const response = await SELF.fetch(new Request(`https://20xx.xyz${path}`, { method, headers, body }));
  const text = await response.text();
  return { status: response.status, body: text ? JSON.parse(text) : null, headers: response.headers };
}

export async function resetQueue(): Promise<void> {
  await queueStub().resetForTest();
  await queueStub().setTestClock(START);
}

export async function setClock(t: number): Promise<void> {
  await queueStub().setTestClock(t);
}

export async function runAlarm(): Promise<boolean> {
  return runDurableObjectAlarm(queueStub());
}

export async function publish(policy: unknown = POLICY): Promise<void> {
  const result = await call("PUT", "/v1/admin/policy", { admin: true, body: policy });
  if (result.status !== 200) throw new Error(`publish failed: ${JSON.stringify(result.body)}`);
}

export async function seedAccounts(count: number): Promise<void> {
  const accounts = Array.from({ length: count }, (_, index) => ({
    connect_code: `BOT${index}#1`,
    r2_key: `netplay/accounts/bot${index}.json`,
    sha256: (index % 10).toString().repeat(64),
  }));
  const result = await call("PUT", "/v1/admin/accounts", { admin: true, body: accounts });
  if (result.status !== 200) throw new Error(`accounts failed: ${JSON.stringify(result.body)}`);
}

export function runnerStatus(slots: number, healthy = slots) {
  return {
    schema_version: 5,
    state: healthy === slots ? "ready" : healthy === 0 ? "recovering" : "degraded",
    message: healthy === slots ? "Game servers are ready." : "One game server is recovering; other slots remain available.",
    policy_sha256: POLICY.bundle_sha256,
    slots,
    healthy_slots: healthy,
    target_fps: 60,
    game_fps: null,
    frame_interval_p95_ms: null,
    dolphin_step_p95_ms: null,
    policy_round_trip_p95_ms: null,
    model_inference_p95_ms: null,
    batch_wait_p95_ms: null,
    recoveries: 0,
    updated_at: 0,
    chunk_health: [],
  };
}

export async function report(session: string, slots: number, healthy = slots): Promise<Call> {
  return call("POST", `/v1/runner/sessions/${session}/status`, { runner: true, body: runnerStatus(slots, healthy) });
}

export async function startSession(slots = 2): Promise<string> {
  const result = await call("POST", "/v1/runner/sessions", {
    runner: true,
    body: { protocol_version: 4, session_id: crypto.randomUUID(), host: "test-box", bundle_sha256: POLICY.bundle_sha256, git_sha: "abc123", slots, stream: false },
  });
  if (result.status !== 201) throw new Error(`session failed: ${JSON.stringify(result.body)}`);
  await report(result.body.session_id, slots);
  return result.body.session_id as string;
}

export const CREATE = { player_code: "CRYO#610", character: "FOX", imitation: "IBDW#0", online_delay: 2 };
