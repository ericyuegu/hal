import { DurableObject } from "cloudflare:workers";
import {
  HttpError,
  QUEUE_CAP,
  TERMINAL_STATUSES,
  randomToken,
  sha256Hex,
  validatePlayerCode,
} from "./domain";
import type { Env } from "./env";
import { EVENT_SCHEMA, EventLog } from "./events";
import { type PolicyConfig, checkChoice, optionsBody, parsePolicyConfig } from "./policy";
import { bool, fields, int, parseCreate, parsePolicyUpdate, parseRematch, str } from "./requests";
import { SESSION_SCHEMA, SessionStore, type StartRequest } from "./sessions";
import { JOB_SCHEMA, JobStore } from "./store";

export interface ApiResult {
  status: number;
  body?: unknown;
  headers?: Record<string, string>;
}

const QUEUE_SCHEMA = `
CREATE TABLE IF NOT EXISTS policy (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  config TEXT NOT NULL,
  published_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
`;

export type RunnerAction =
  | "heartbeat"
  | "connecting"
  | "playing"
  | "no-show"
  | "no-contest"
  | "finish-game"
  | "fail"
  | "forfeit"
  | "replay";

interface LiveAttachment {
  jobId: string;
  worker: string;
  released: boolean;
}

export class Queue extends DurableObject<Env> {
  private readonly jobs: JobStore;
  private readonly sessions: SessionStore;
  private readonly events: EventLog;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    const sql = ctx.storage.sql;
    sql.exec(JOB_SCHEMA);
    sql.exec(SESSION_SCHEMA);
    sql.exec(EVENT_SCHEMA);
    sql.exec(QUEUE_SCHEMA);
    const now = () => this.now();
    this.jobs = new JobStore(sql, now);
    this.sessions = new SessionStore(sql, this.jobs, now);
    this.events = new EventLog(sql, now);
  }

  private setting(key: string): string | null {
    const rows = this.ctx.storage.sql.exec<{ value: string }>("SELECT value FROM settings WHERE key = ?", key).toArray();
    return rows[0]?.value ?? null;
  }

  private setSetting(key: string, value: string): void {
    this.ctx.storage.sql.exec(
      "INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
      key,
      value,
    );
  }

  private now(): number {
    if (this.env.HAL_TEST_CLOCK === "1") {
      const clock = this.setting("test_clock");
      if (clock !== null) return Number(clock);
    }
    return Date.now() / 1000;
  }

  private policy(): PolicyConfig | null {
    const rows = this.ctx.storage.sql.exec<{ config: string }>("SELECT config FROM policy WHERE id = 1").toArray();
    return rows[0] === undefined ? null : (JSON.parse(rows[0].config) as PolicyConfig);
  }

  private requirePolicy(): PolicyConfig {
    const policy = this.policy();
    if (policy === null) throw new HttpError(503, "no policy has been published");
    return policy;
  }

  private tx<T>(fn: () => T): T {
    return this.ctx.storage.transactionSync(fn);
  }

  private async run(fn: () => unknown, okStatus = 200): Promise<ApiResult> {
    try {
      const body = await fn();
      await this.scheduleAlarm();
      return body === null || body === undefined ? { status: 204 } : { status: okStatus, body };
    } catch (error) {
      if (error instanceof HttpError) {
        return { status: error.status, body: { detail: error.detail }, headers: error.headers };
      }
      throw error;
    }
  }

  private async scheduleAlarm(): Promise<void> {
    const oldest = this.events.oldest();
    const candidates = [
      this.jobs.nextDeadline(),
      this.sessions.nextDeadline(),
      oldest === null ? null : oldest + 30 * 24 * 60 * 60,
    ].filter((value): value is number => value !== null);
    if (candidates.length === 0) {
      await this.ctx.storage.deleteAlarm();
      return;
    }
    // Under the test clock every deadline is a fake time in the past. Park the
    // alarm a day ahead so only runDurableObjectAlarm fires it, never a race.
    const at = this.env.HAL_TEST_CLOCK === "1" ? Date.now() + 86_400_000 : Math.ceil(Math.min(...candidates) * 1000);
    await this.ctx.storage.setAlarm(at);
  }

  async alarm(): Promise<void> {
    const changed = this.tx(() => {
      const ended = this.sessions.endSilent();
      for (const id of ended.sessions) this.events.log("session_ended", { session: id, reason: "silent" });
      for (const id of ended.jobs) this.events.log("job_failed", { job: id, reason: "session_silent" });
      const expired = this.jobs.reapExpired();
      this.events.prune();
      for (const id of expired) this.events.log("job_expired", { job: id, status: this.jobs.row(id)?.status });
      return [...ended.jobs, ...expired];
    });
    this.release(changed);
    await this.scheduleAlarm();
  }

  // Player routes

  async options(): Promise<ApiResult> {
    return this.run(() => optionsBody(this.requirePolicy()));
  }

  async capacity(): Promise<ApiResult> {
    return this.run(() => this.sessions.capacity());
  }

  async createJob(raw: unknown): Promise<ApiResult> {
    const id = randomToken(18);
    const token = randomToken(32);
    const digest = await sha256Hex(token);
    return this.run(() => {
      const policy = this.requirePolicy();
      const request = parseCreate(raw, policy);
      // Check order matches the Python API so the golden transcripts agree.
      if (!policy.online_delays.includes(request.online_delay)) {
        throw new HttpError(422, "online delay is unsupported by this policy");
      }
      if (request.imitation === "MASKED" && !policy.masked_identity) {
        throw new HttpError(422, "masked identity is unsupported by this policy");
      }
      if (this.setting("paused") === "1") throw new HttpError(503, "The queue is paused. Try again shortly.");
      if (this.jobs.queueDepth() >= QUEUE_CAP) throw new HttpError(503, "The queue is full. Try again in a few minutes.");
      const capacity = this.sessions.capacity();
      if (capacity.service_status === "unavailable") {
        throw new HttpError(503, "Game servers are unavailable. Try again shortly.");
      }
      if (capacity.healthy_slots === 0) throw new HttpError(503, capacity.service_message);
      checkChoice(policy.characters, request.character, "character");
      checkChoice(policy.imitations, request.imitation, "imitation");
      validatePlayerCode(request.player_code);
      const job = this.tx(() => {
        const created = this.jobs.createJob(id, digest, request.player_code, request);
        this.events.log("job_created", { job: id, character: request.character, imitation: request.imitation });
        return created;
      });
      return { ...job, token };
    }, 201);
  }

  async getJob(id: string, token: string): Promise<ApiResult> {
    const digest = await sha256Hex(token);
    return this.run(() => this.jobs.getJob(id, digest));
  }

  async updatePolicy(id: string, token: string, raw: unknown): Promise<ApiResult> {
    const digest = await sha256Hex(token);
    return this.run(() => {
      const update = parsePolicyUpdate(raw, this.requirePolicy());
      const job = this.tx(() => {
        const current = this.jobs.getJob(id, digest);
        const desired = "desired_return" in update ? update.desired_return ?? null : current.desired_return;
        const temperature = "temperature" in update ? update.temperature : current.temperature;
        if (temperature === null || temperature === undefined) throw new HttpError(422, "temperature cannot be null");
        const updated = this.jobs.updatePolicy(id, digest, desired, temperature);
        this.events.log("policy_updated", { job: id, revision: updated.policy_revision });
        return updated;
      });
      this.broadcastSettings(id);
      return job;
    });
  }

  async cancelJob(id: string, token: string): Promise<ApiResult> {
    const digest = await sha256Hex(token);
    return this.run(() => {
      const job = this.tx(() => {
        const canceled = this.jobs.cancel(id, digest);
        this.events.log("job_cancel_requested", { job: id, status: canceled.status });
        return canceled;
      });
      this.release([id]);
      return job;
    });
  }

  async rematch(id: string, token: string, raw: unknown): Promise<ApiResult> {
    const digest = await sha256Hex(token);
    return this.run(() => {
      const policy = this.requirePolicy();
      const request = parseRematch(raw, policy);
      checkChoice(policy.characters, request.character, "character");
      checkChoice(policy.imitations, request.imitation, "imitation");
      checkChoice(policy.stages, request.stage, "stage");
      return this.tx(() => {
        const job = this.jobs.requestRematch(id, digest, request.character, request.imitation, request.stage);
        this.events.log("rematch_ready", { job: id, character: request.character, stage: request.stage });
        return job;
      });
    });
  }

  // Runner routes

  async activePolicy(): Promise<ApiResult> {
    return this.run(() => this.requirePolicy());
  }

  async startSession(raw: unknown): Promise<ApiResult> {
    const id = randomToken(12);
    return this.run(() => {
      const value = fields(raw, ["host", "bundle_sha256", "git_sha", "slots", "stream"], ["host", "bundle_sha256", "git_sha", "slots", "stream"]);
      const input: StartRequest = {
        host: str(value.host, "host"),
        bundle_sha256: str(value.bundle_sha256, "bundle_sha256"),
        git_sha: str(value.git_sha, "git_sha"),
        slots: int(value.slots, "slots"),
        stream: bool(value.stream, "stream"),
      };
      if (input.slots < 1 || input.slots > 8) throw new HttpError(422, "slots must be in [1, 8]");
      return this.tx(() => {
        const started = this.sessions.start(id, input, this.policy());
        this.events.log("session_started", { session: id, host: input.host, slots: input.slots, git_sha: input.git_sha });
        for (const grant of started.accounts) {
          this.events.log("account_leased", { session: id, slot: grant.slot, connect_code: grant.connect_code });
        }
        return { ...started, policy: this.policy() };
      });
    }, 201);
  }

  async reportStatus(sessionId: string, raw: unknown): Promise<ApiResult> {
    return this.run(() => this.tx(() => this.sessions.report(sessionId, raw)));
  }

  async claim(sessionId: string, raw: unknown): Promise<ApiResult> {
    return this.run(() => {
      const slot = int(fields(raw, ["slot"], ["slot"]).slot, "slot");
      return this.tx(() => {
        const worker = this.sessions.claimWorker(sessionId, slot, this.policy());
        const job = this.jobs.claimNext(worker);
        if (job !== null) this.events.log("job_claimed", { job: job.id, session: sessionId, slot });
        return job;
      });
    });
  }

  async drain(sessionId: string): Promise<ApiResult> {
    return this.run(() =>
      this.tx(() => {
        this.sessions.drain(sessionId);
        this.events.log("session_draining", { session: sessionId });
        return { draining: true };
      }),
    );
  }

  async endSession(sessionId: string): Promise<ApiResult> {
    return this.run(() => {
      const failed = this.tx(() => {
        const ids = this.sessions.end(sessionId, "ended");
        this.events.log("session_ended", { session: sessionId, failed: ids.length });
        return ids;
      });
      this.release(failed);
      return { failed: failed.length };
    });
  }

  async workerJob(sessionId: string, slot: number, jobId: string): Promise<ApiResult> {
    return this.run(() => this.jobs.workerJob(jobId, this.sessions.jobWorker(sessionId, slot)));
  }

  async runnerJob(sessionId: string, slot: number, jobId: string, action: RunnerAction, raw: unknown): Promise<ApiResult> {
    return this.run(() => {
      const policy = this.requirePolicy();
      const job = this.tx(() => {
        const worker = this.sessions.jobWorker(sessionId, slot);
        const log = (kind: string, detail: Record<string, unknown> = {}) =>
          this.events.log(kind, { job: jobId, session: sessionId, slot, ...detail });
        switch (action) {
          case "heartbeat":
            return this.jobs.heartbeat(jobId, worker);
          case "connecting": {
            const code = validatePlayerCode(str(fields(raw, ["connect_code"], ["connect_code"]).connect_code, "connect_code"));
            const result = this.jobs.markConnecting(jobId, worker, code);
            log("job_connecting");
            return result;
          }
          case "playing": {
            const result = this.jobs.markPlaying(jobId, worker);
            log("job_playing");
            return result;
          }
          case "no-show": {
            const result = this.jobs.markNoShow(jobId, worker);
            log("job_no_show");
            return result;
          }
          case "no-contest": {
            const result = this.jobs.markNoContest(jobId, worker);
            log("job_no_contest");
            return result;
          }
          case "finish-game": {
            const value = fields(raw, ["game_number", "actual_stage", "result"], ["game_number", "actual_stage", "result"]);
            const stage = checkChoice(policy.stages, str(value.actual_stage, "actual_stage"), "stage");
            const number = int(value.game_number, "game_number");
            const result = this.jobs.finishGame(jobId, worker, number, stage, str(value.result, "result"));
            log("game_finished", { game_number: number, stage, result: value.result });
            return result;
          }
          case "fail": {
            const value = fields(raw, ["error_code", "retryable"], ["error_code", "retryable"]);
            const code = str(value.error_code, "error_code");
            const result = this.jobs.fail(jobId, worker, code, bool(value.retryable, "retryable"));
            log("job_failed", { error_code: code, status: result.status });
            return result;
          }
          case "forfeit": {
            const result = this.jobs.forfeit(jobId, worker);
            log("job_forfeited");
            return result;
          }
          case "replay": {
            const value = fields(
              raw,
              ["game_number", "key", "sha256", "size", "etag"],
              ["game_number", "key", "sha256", "size", "etag"],
            );
            const number = int(value.game_number, "game_number");
            const result = this.jobs.recordReplay(
              jobId,
              number,
              str(value.key, "key"),
              str(value.sha256, "sha256"),
              int(value.size, "size"),
              str(value.etag, "etag"),
            );
            log("replay_recorded", { game_number: number, key: value.key });
            return result;
          }
        }
      });
      if (TERMINAL_STATUSES.has(job.status) || job.status === "queued") this.release([jobId]);
      return job;
    });
  }

  // Admin routes

  async putPolicy(raw: unknown): Promise<ApiResult> {
    return this.run(() => {
      const policy = parsePolicyConfig(raw);
      this.tx(() => {
        this.ctx.storage.sql.exec(
          `INSERT INTO policy(id, config, published_at) VALUES (1, ?, ?)
           ON CONFLICT(id) DO UPDATE SET config = excluded.config, published_at = excluded.published_at`,
          JSON.stringify(policy),
          this.now(),
        );
        this.events.log("policy_published", { bundle_sha256: policy.bundle_sha256 });
      });
      return policy;
    });
  }

  async putAccounts(raw: unknown): Promise<ApiResult> {
    return this.run(() =>
      this.tx(() => {
        this.sessions.putAccounts(raw);
        this.events.log("accounts_published", { count: Array.isArray(raw) ? raw.length : 0 });
        return this.sessions.summary().accounts;
      }),
    );
  }

  async setPaused(paused: boolean): Promise<ApiResult> {
    return this.run(() =>
      this.tx(() => {
        this.setSetting("paused", paused ? "1" : "0");
        this.events.log(paused ? "queue_paused" : "queue_resumed", {});
        return { paused };
      }),
    );
  }

  async adminStatus(): Promise<ApiResult> {
    return this.run(() => ({
      paused: this.setting("paused") === "1",
      policy: this.policy(),
      capacity: this.sessions.capacity(),
      ...this.sessions.summary(),
    }));
  }

  async adminEvents(query: { job?: string; session?: string; since?: number; limit?: number }): Promise<ApiResult> {
    return this.run(() => ({ events: this.events.query(query) }));
  }

  // Test seams, enabled only by the HAL_TEST_CLOCK binding in vitest.config.ts.

  async setTestClock(seconds: number): Promise<void> {
    if (this.env.HAL_TEST_CLOCK !== "1") throw new Error("test clock is disabled");
    this.setSetting("test_clock", String(seconds));
  }

  async resetForTest(): Promise<void> {
    if (this.env.HAL_TEST_CLOCK !== "1") throw new Error("test reset is disabled");
    this.tx(() => {
      for (const table of ["games", "jobs", "sessions", "accounts", "events", "policy", "settings"]) {
        this.ctx.storage.sql.exec(`DELETE FROM ${table}`);
      }
    });
    await this.ctx.storage.deleteAlarm();
  }

  // Live settings

  private settingsMessage(jobId: string): string | null {
    const row = this.jobs.row(jobId);
    if (row === null) return null;
    return JSON.stringify({
      type: "settings",
      revision: row.policy_revision,
      desired_return: row.desired_return,
      temperature: row.temperature,
    });
  }

  async fetch(request: Request): Promise<Response> {
    const match = new URL(request.url).pathname.match(/^\/v1\/runner\/jobs\/([^/]+)\/live$/);
    if (match === null || request.headers.get("Upgrade")?.toLowerCase() !== "websocket") {
      return Response.json({ detail: "not found" }, { status: 404 });
    }
    const jobId = match[1]!;
    const slot = request.headers.get("X-HAL-Slot");
    const session = request.headers.get("X-HAL-Session");
    if (!session || slot === null || !/^\d+$/.test(slot)) {
      return Response.json({ detail: "X-HAL-Session and X-HAL-Slot headers are required" }, { status: 400 });
    }
    let worker: string;
    try {
      worker = this.sessions.jobWorker(session, Number(slot));
    } catch (error) {
      if (error instanceof HttpError) return Response.json({ detail: error.detail }, { status: error.status });
      throw error;
    }
    const row = this.jobs.row(jobId);
    if (row === null || row.lease_owner !== worker) {
      return Response.json({ detail: "worker does not own this job" }, { status: 409 });
    }
    const [client, server] = Object.values(new WebSocketPair()) as [WebSocket, WebSocket];
    this.ctx.acceptWebSocket(server, [`job:${jobId}`]);
    server.serializeAttachment({ jobId, worker, released: false } satisfies LiveAttachment);
    server.send(this.settingsMessage(jobId)!);
    return new Response(null, { status: 101, webSocket: client });
  }

  // getWebSockets still lists a socket the server closed until the client
  // acknowledges the close, and send() on it throws.
  private openSockets(jobId: string): { socket: WebSocket; worker: string }[] {
    return this.ctx.getWebSockets(`job:${jobId}`).flatMap((socket) => {
      const attachment = socket.deserializeAttachment() as LiveAttachment;
      if (attachment.released || socket.readyState !== WebSocket.READY_STATE_OPEN) return [];
      return [{ socket, worker: attachment.worker }];
    });
  }

  private broadcastSettings(jobId: string): void {
    const message = this.settingsMessage(jobId);
    if (message === null) return;
    for (const { socket } of this.openSockets(jobId)) socket.send(message);
  }

  // A socket's runner no longer owns its job: tell it, then close.
  private release(jobIds: readonly string[]): void {
    for (const jobId of new Set(jobIds)) {
      const row = this.jobs.row(jobId);
      for (const { socket, worker } of this.openSockets(jobId)) {
        if (row === null || TERMINAL_STATUSES.has(row.status as string) || row.lease_owner !== worker) {
          socket.send(JSON.stringify({ type: "released" }));
          socket.close(1000, "released");
          socket.serializeAttachment({ jobId, worker, released: true } satisfies LiveAttachment);
        }
      }
    }
  }

  async webSocketMessage(_socket: WebSocket, _message: string | ArrayBuffer): Promise<void> {
    // Runners only listen on this socket.
  }

  async webSocketClose(socket: WebSocket, code: number): Promise<void> {
    // 1005, 1006, and 1015 are reserved: a peer reports them but must never send them.
    socket.close(code === 1005 || code === 1006 || code === 1015 ? 1000 : code, "closed");
  }
}
