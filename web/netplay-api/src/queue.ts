import { DurableObject } from "cloudflare:workers";
import {
  HttpError,
  MAX_BODY_BYTES,
  IN_GAME_PHASES,
  QUEUE_CAP,
  RUNNER_PROTOCOL_VERSION,
  randomToken,
  sha256Hex,
  validatePlayerCode,
} from "./domain";
import { WriteBudget } from "./budget";
import { LiveConnections } from "./live";
import type { SlotProgress } from "./sessions";
import type { Env } from "./env";
import { EVENT_SCHEMA, EventLog } from "./events";
import { type PolicyConfig, checkChoice, optionsBody, parsePolicyConfig } from "./policy";
import { bool, fields, int, parseCreate, parseEnd, parseReport, parseSettingsUpdate, str } from "./requests";
import { SESSION_SCHEMA, SessionStore, type StartRequest } from "./sessions";
import { JOB_SCHEMA, JobStore, type JobView } from "./store";

export interface ApiResult {
  status: number;
  body?: unknown;
  headers?: Record<string, string>;
}

const SESSION_ID = /^[A-Za-z0-9_-]{16,64}$/;

// Every table is created with CREATE TABLE IF NOT EXISTS, which keeps an older
// table unchanged. Bump this with any table change and use a fresh instance.
// A mismatched store refuses every request; there are no migrations.
export const STORE_SCHEMA_VERSION = 7;
export const QUEUE_INSTANCE = `global-v${STORE_SCHEMA_VERSION}`;
const STORE_TABLES = ["games", "jobs", "pairings", "sessions", "accounts", "stream", "events", "policy", "settings", "counts"];

const QUEUE_SCHEMA = `
CREATE TABLE IF NOT EXISTS policy (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  config TEXT NOT NULL,
  published_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
`;

export class Queue extends DurableObject<Env> {
  private readonly jobs: JobStore;
  private readonly sessions: SessionStore;
  private readonly events: EventLog;
  private readonly schemaError: string | null;

  private readonly budget: WriteBudget;
  private readonly connections: LiveConnections;
  private runtime: Map<string, SlotProgress[]> | null = null;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    const sql = ctx.storage.sql;
    this.schemaError = ctx.storage.transactionSync(() => {
      const names = STORE_TABLES.map(() => "?").join(", ");
      const existing = sql
        .exec<{ name: string }>(`SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (${names})`, ...STORE_TABLES)
        .toArray()
        .map((row) => row.name);
      if (existing.length > 0) {
        const found = existing.includes("settings") ? this.setting("schema_version") : null;
        return found === String(STORE_SCHEMA_VERSION)
          ? null
          : `queue storage schema version ${found ?? "(missing)"} is not the Worker's ${STORE_SCHEMA_VERSION}`;
      }
      sql.exec(JOB_SCHEMA);
      sql.exec(SESSION_SCHEMA);
      sql.exec(EVENT_SCHEMA);
      sql.exec(QUEUE_SCHEMA);
      this.setSetting("schema_version", String(STORE_SCHEMA_VERSION));
      return null;
    });
    const now = () => this.now();
    this.budget = new WriteBudget(sql, now);
    this.connections = new LiveConnections(ctx, now);
    this.jobs = new JobStore(sql, now, (id, seen) => this.connections.presence(id, seen), (row) => {
      this.runtime ??= this.sessions.runtimeLeases();
      return this.sessions.leaseExpiry(row, this.runtime);
    });
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

  private requireSchema(): void {
    if (this.schemaError !== null) throw new HttpError(503, this.schemaError);
  }

  private async run(
    fn: () => unknown,
    { status = 200, alarm = true, push = alarm, cost = 0, admission = false }:
      { status?: number; alarm?: boolean; push?: boolean | "job"; cost?: number | (() => number); admission?: boolean } = {},
  ): Promise<ApiResult> {
    try {
      this.requireSchema();
      this.runtime = null;
      if (admission) this.budget.requireCapacity();
      let body = await fn();
      let windDown = false;
      const credits = typeof cost === "function" ? cost() : cost;
      if (body !== null && body !== undefined && credits > 0 && this.budget.charge(credits)) {
        windDown = this.jobs.windDownForBudget();
        if (windDown && push === "job") body = this.jobs.view(this.jobs.row((body as JobView).id)!);
      }
      // Empty claims do not change deadlines. Read routes also opt out.
      if (alarm && body !== null && body !== undefined) await this.scheduleAlarm();
      if (windDown) this.broadcast();
      else if (push && body !== null && body !== undefined) {
        if (push === "job") this.broadcastJob(body as JobView);
        else this.broadcast();
      }
      return body === null || body === undefined ? { status: 204 } : { status, body };
    } catch (error) {
      if (error instanceof HttpError) {
        return { status: error.status, body: { detail: error.detail }, headers: error.headers };
      }
      throw error;
    }
  }

  private async scheduleAlarm(): Promise<void> {
    const existing = await this.ctx.storage.getAlarm();
    const oldest = this.events.oldest();
    const candidates = [
      this.jobs.nextDeadline(),
      this.sessions.nextDeadline(),
      oldest === null ? null : oldest + 30 * 24 * 60 * 60,
    ].filter((value): value is number => value !== null);
    if (candidates.length === 0) {
      if (existing !== null) await this.ctx.storage.deleteAlarm();
      return;
    }
    // Under the test clock every deadline is a fake time in the past. Park the
    // alarm a day ahead so only runDurableObjectAlarm fires it, never a race.
    const at = this.env.HAL_TEST_CLOCK === "1" ? Date.now() + 86_400_000 : Math.ceil(Math.min(...candidates) * 1000);
    // An earlier wakeup is safe and avoids a billed write on every heartbeat.
    if (existing === null || at < existing) await this.ctx.storage.setAlarm(at);
  }

  async alarm(): Promise<void> {
    if (this.schemaError !== null) throw new Error(this.schemaError);
    this.runtime = null;
    let changed = false;
    this.tx(() => {
      const streamHolder = this.sessions.streamHolder();
      const ended = this.sessions.endSilent();
      changed = ended.sessions.length > 0;
      for (const id of ended.sessions) this.events.log("session_ended", { session: id, reason: "silent" });
      for (const id of ended.jobs) {
        this.events.log("job_released", {
          job: id,
          status: this.jobs.row(id)?.status,
          reason: "session_silent",
        });
      }
      if (streamHolder !== null && ended.sessions.includes(streamHolder)) {
        this.events.log("stream_lease_released", { session: streamHolder, reason: "session_silent" });
      }
      for (const id of this.jobs.reapExpired()) {
        changed = true;
        const row = this.jobs.row(id);
        this.events.log("job_released", {
          job: id,
          status: row?.status,
          reason: row?.end_reason ?? "lease_expired",
        });
      }
      for (const id of this.jobs.applyYield()) {
        changed = true;
        this.events.log("wind_down", { job: id, reason: "yield" });
      }
      this.events.prune();
    });
    await this.scheduleAlarm();
    this.broadcast(changed);
  }

  async fetch(request: Request): Promise<Response> {
    try {
      this.requireSchema();
      if (request.headers.get("Upgrade")?.toLowerCase() !== "websocket") throw new HttpError(426, "WebSocket upgrade required");
      const url = new URL(request.url);
      const session = url.pathname === "/v1/runner/live" ? request.headers.get("X-HAL-Session") : null;
      if (url.pathname === "/v1/runner/live" && session === null) throw new HttpError(401, "session required");
      if (session !== null) this.sessions.live(session);
      return this.connections.open(session);
    } catch (error) {
      if (error instanceof HttpError) return Response.json({ detail: error.detail }, { status: error.status });
      throw error;
    }
  }

  async webSocketMessage(ws: WebSocket, message: string | ArrayBuffer): Promise<void> {
    try {
      this.requireSchema();
      if (typeof message !== "string" || new TextEncoder().encode(message).byteLength > MAX_BODY_BYTES) {
        ws.close(1009, "message too large");
        return;
      }
      let value: Record<string, unknown>;
      try { value = JSON.parse(message) as Record<string, unknown>; }
      catch { throw new HttpError(422, "invalid JSON"); }
      if (value === null || typeof value !== "object") throw new HttpError(422, "message must be an object");
      const peer = this.connections.peer(ws);
      if (value.ack !== undefined) this.connections.acknowledge(ws, int(value.ack, "ack"));
      if (value.type === "ack") return;
      if (peer.role === "browser" && value.type === "subscribe") {
        const id = value.job_id === null ? null : str(value.job_id, "job_id");
        if (id !== null) this.jobs.authenticated(id, await sha256Hex(str(value.token, "token")));
        if (peer.job !== null) this.jobs.savePresence(peer.job, this.connections.presence(peer.job, this.connections.lastSeen(ws)));
        const updated = this.connections.peer(ws);
        ws.serializeAttachment({ ...updated, job: id, seen: this.now() });
        this.snapshot(ws);
        return;
      }
      if (peer.role === "host" && value.type === "health") {
        if (!this.env.TWITCH_STREAM_KEY && this.sessions.wantsStream(peer.session)) throw new HttpError(503, "TWITCH_STREAM_KEY is not configured");
        this.runtime = null;
        const state = this.tx(() => this.sessions.report(peer.session, value.status, value.progress));
        if (state.streamGranted) this.events.log("stream_lease_granted", { session: peer.session, slot: 0 });
        this.connections.send(ws, { type: "health", draining: state.draining,
          stream: state.streamHolder ? { slot: 0, key: this.env.TWITCH_STREAM_KEY } : null });
        this.broadcast(false);
        return;
      }
      if (peer.role === "host" && value.type === "subscribe") { this.snapshot(ws); return; }
      throw new HttpError(422, "unknown live message");
    } catch (error) {
      if (!(error instanceof HttpError)) throw error;
      this.connections.send(ws, { type: "error", status: error.status, detail: error.detail });
    }
  }

  private snapshot(ws: WebSocket, capacity = this.sessions.capacity(), jobs = new Map<string, unknown>(), positions = this.jobs.queuePositions()): void {
    const peer = this.connections.peer(ws);
    this.connections.send(ws, { type: "capacity", capacity });
    if (peer.role === "browser" && peer.job !== null) {
      if (!jobs.has(peer.job)) {
        const row = this.jobs.row(peer.job);
        jobs.set(peer.job, row === null ? null : this.jobs.view(row, positions.get(peer.job)));
      }
      this.connections.send(ws, { type: "job", job: jobs.get(peer.job) });
    }
    if (peer.role === "host") {
      try {
        const session = this.sessions.live(peer.session);
        this.connections.send(ws, { type: "state", draining: session.draining === 1,
          stream: this.sessions.streamHolder() === peer.session ? { slot: 0, key: this.env.TWITCH_STREAM_KEY } : null });
      } catch (error) {
        if (!(error instanceof HttpError)) throw error;
        this.connections.send(ws, { type: "error", status: error.status, detail: error.detail });
        ws.close(1008, "session ended");
        return;
      }
      this.connections.send(ws, { type: "work", pairing: this.sessions.pairing(peer.session),
        assignments: this.jobs.assigned().filter((row) => (row.lease_owner as string).startsWith(`${peer.session}/slot-`))
          .map((row) => ({ id: row.id, slot: Number((row.lease_owner as string).split("/slot-")[1]), attempt: row.attempt })) });
      for (const row of this.jobs.assigned()) {
        if (!(row.lease_owner as string).startsWith(`${peer.session}/slot-`)) continue;
        this.connections.send(ws, { type: "job", job: this.jobs.view(row) });
      }
    }
  }

  private broadcastJob(job: JobView): void {
    const owner = job.status === "assigned" ? this.jobs.row(job.id)?.lease_owner : null;
    for (const ws of this.connections.sockets()) {
      const peer = this.connections.peer(ws);
      if ((peer.role === "browser" && peer.job === job.id)
        || (peer.role === "host" && typeof owner === "string" && owner.startsWith(`${peer.session}/slot-`))) {
        this.connections.send(ws, { type: "job", job });
      }
    }
  }

  private broadcast(jobs = true): void {
    const sockets = this.connections.sockets();
    if (sockets.length === 0) return;
    const capacity = this.sessions.capacity();
    const views = new Map<string, unknown>();
    const positions = jobs ? this.jobs.queuePositions() : new Map<string, number>();
    for (const ws of sockets) {
      if (jobs) this.snapshot(ws, capacity, views, positions);
      else this.connections.send(ws, { type: "capacity", capacity });
    }
  }

  async webSocketClose(ws: WebSocket, code: number): Promise<void> {
    const peer = this.connections.peer(ws);
    if (peer.role === "browser" && peer.job !== null) {
      this.jobs.savePresence(peer.job, this.connections.presence(peer.job, this.connections.lastSeen(ws)));
    }
    ws.close(code === 1005 || code === 1006 ? 1000 : code, "closed");
  }

  async webSocketError(ws: WebSocket): Promise<void> { await this.webSocketClose(ws, 1011); }

  // Player routes

  async options(): Promise<ApiResult> {
    return this.run(() => optionsBody(this.requirePolicy()), { alarm: false });
  }

  async capacity(): Promise<ApiResult> {
    return this.run(() => this.sessions.capacity(), { alarm: false });
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
      if (request.stage !== null) checkChoice(policy.stages, request.stage, "stage");
      validatePlayerCode(request.player_code);
      const job = this.tx(() => {
        const created = this.jobs.createJob(id, digest, request);
        this.events.log("job_created", { job: id, character: request.character, imitation: request.imitation });
        return created;
      });
      return { ...job, token };
    }, { status: 201, cost: 24, admission: true });
  }

  async getJob(id: string, token: string): Promise<ApiResult> {
    const digest = await sha256Hex(token);
    return this.run(() => this.tx(() => this.jobs.getJob(id, digest)), { alarm: false });
  }

  async updateSettings(id: string, token: string, raw: unknown): Promise<ApiResult> {
    const digest = await sha256Hex(token);
    return this.run(() => {
      const update = parseSettingsUpdate(raw, this.requirePolicy());
      return this.tx(() => {
        const job = this.jobs.updateSettings(id, digest, update);
        this.events.log("settings_updated", { job: id, revision: job.settings.revision });
        return job;
      });
    }, { alarm: false, push: "job", cost: 8, admission: true });
  }

  async requestLock(id: string, token: string): Promise<ApiResult> {
    const digest = await sha256Hex(token);
    return this.run(() => this.tx(() => {
      const job = this.jobs.requestLock(id, digest);
      this.events.log("lock_requested", { job: id, count: job.lock_requests });
      return job;
    }), { alarm: false, push: "job", cost: 8, admission: true });
  }

  async leaveJob(id: string, token: string): Promise<ApiResult> {
    const digest = await sha256Hex(token);
    return this.run(() => this.tx(() => {
      const job = this.jobs.leave(id, digest);
      this.events.log(job.status === "ended" ? "job_ended" : "wind_down", {
        job: id,
        reason: job.end_reason ?? "player",
      });
      return job;
    }));
  }

  // Runner routes

  async activePolicy(): Promise<ApiResult> {
    return this.run(() => this.requirePolicy(), { alarm: false });
  }

  async startSession(raw: unknown): Promise<ApiResult> {
    return this.run(() => {
      const names = ["protocol_version", "session_id", "host", "bundle_sha256", "git_sha", "slots", "stream"];
      const value = fields(raw, names, names);
      // Checked before the repeat-start path, so no session ever runs a different protocol.
      const protocol = int(value.protocol_version, "protocol_version");
      if (protocol !== RUNNER_PROTOCOL_VERSION) {
        throw new HttpError(409, `runner protocol ${protocol} is not the Worker's ${RUNNER_PROTOCOL_VERSION}`);
      }
      const id = str(value.session_id, "session_id");
      if (!SESSION_ID.test(id)) throw new HttpError(422, "session_id must be 16 to 64 URL-safe characters");
      const input: StartRequest = {
        host: str(value.host, "host"),
        bundle_sha256: str(value.bundle_sha256, "bundle_sha256"),
        git_sha: str(value.git_sha, "git_sha"),
        slots: int(value.slots, "slots"),
        stream: bool(value.stream, "stream"),
      };
      if (input.slots < 1 || input.slots > 16) throw new HttpError(422, "slots must be in [1, 16]");
      return this.tx(() => {
        const repeated = this.sessions.exists(id);
        const started = this.sessions.start(id, input, this.policy());
        if (!repeated) {
          this.events.log("session_started", { session: id, host: input.host, slots: input.slots, git_sha: input.git_sha });
          for (const grant of started.accounts.slice(0, 1)) {
            this.events.log("account_leased", { session: id, slot: grant.slot, connect_code: grant.connect_code });
          }
        }
        return { ...started, policy: this.policy() };
      });
    }, { status: 201, cost: 16 });
  }

  async reportStatus(sessionId: string, raw: unknown): Promise<ApiResult> {
    return this.run(() => {
      if (!this.env.TWITCH_STREAM_KEY && this.sessions.wantsStream(sessionId)) {
        throw new HttpError(503, "TWITCH_STREAM_KEY is not configured");
      }
      const state = this.tx(() => {
        const reported = this.sessions.report(sessionId, raw);
        if (reported.streamGranted) this.events.log("stream_lease_granted", { session: sessionId, slot: 0 });
        return reported;
      });
      return {
        draining: state.draining,
        stream: state.streamHolder ? { slot: 0, key: this.env.TWITCH_STREAM_KEY } : null,
      };
    });
  }

  async claim(sessionId: string, raw: unknown): Promise<ApiResult> {
    return this.run(() => {
      const slot = int(fields(raw, ["slot"], ["slot"]).slot, "slot");
      return this.tx(() => {
        // A held lease is returned even after a drain or republish, or it would expire as lease_expired.
        const held = this.jobs.heldLease(this.sessions.jobWorker(sessionId, slot));
        if (held !== null) return held;
        const worker = this.sessions.claimWorker(sessionId, slot, this.policy());
        if (this.sessions.pairing(sessionId) !== null) return null;
        if (this.sessions.deferToStreamSlot(sessionId, slot)) return null;
        const job = this.jobs.claimNext(worker);
        if (job !== null) {
          this.sessions.startPairing(sessionId, slot, job.id, job.attempt);
          this.events.log("job_claimed", { job: job.id, session: sessionId, slot });
        }
        return job;
      });
    }, { cost: 24 });
  }

  async pairing(sessionId: string): Promise<ApiResult> {
    return this.run(() => {
      this.sessions.live(sessionId);
      return { pairing: this.sessions.pairing(sessionId) };
    }, { alarm: false });
  }

  async finishPairing(sessionId: string, raw: unknown): Promise<ApiResult> {
    return this.run(() => this.tx(() => {
      const value = fields(raw, ["slot", "job_id", "attempt"], ["slot", "job_id", "attempt"]);
      const slot = int(value.slot, "slot");
      this.sessions.recordingWorker(sessionId, slot);
      const attempt = int(value.attempt, "attempt");
      if (attempt < 1) throw new HttpError(422, "attempt must be positive");
      this.sessions.finishPairing(sessionId, slot, str(value.job_id, "job_id"), attempt);
      return {};
    }), { alarm: false, push: true, cost: 8 });
  }

  async drain(sessionId: string): Promise<ApiResult> {
    return this.run(() =>
      this.tx(() => {
        const released = this.sessions.streamHolder() === sessionId;
        const alreadyDraining = this.sessions.live(sessionId).draining === 1;
        this.sessions.drain(sessionId);
        if (!alreadyDraining) this.events.log("session_draining", { session: sessionId });
        if (released) this.events.log("stream_lease_released", { session: sessionId, reason: "draining" });
        return { draining: true };
      }),
    );
  }

  async endSession(sessionId: string): Promise<ApiResult> {
    return this.run(() => {
      const previous = this.sessions.endedResult(sessionId);
      if (previous !== null) return { failed: previous };
      const failed = this.tx(() => {
        const released = this.sessions.streamHolder() === sessionId;
        const ids = this.sessions.end(sessionId, "ended");
        this.events.log("session_ended", { session: sessionId, failed: ids.length });
        for (const id of ids) {
          this.events.log("job_released", { job: id, status: this.jobs.row(id)?.status, reason: "session_ended" });
        }
        if (released) this.events.log("stream_lease_released", { session: sessionId, reason: "ended" });
        return ids;
      });
      return { failed: failed.length };
    });
  }

  async report(sessionId: string, slot: number, jobId: string, raw: unknown, attempt: number): Promise<ApiResult> {
    return this.run(() => {
      const observed = parseReport(raw);
      const policy = this.requirePolicy();
      for (const game of observed.finished_games) checkChoice(policy.stages, game.stage, "stage");
      return this.tx(() => {
        const worker = this.sessions.jobWorker(sessionId, slot);
        if (this.jobs.row(jobId)?.attempt !== attempt) throw new HttpError(409, "job attempt has changed");
        const { view, phaseChanged, newGames } = this.jobs.report(jobId, worker, observed);
        // A live game no longer needs the session's one pairing, so other slots may claim.
        if (view.observed !== null && IN_GAME_PHASES.has(view.observed.phase)) this.sessions.finishPairing(sessionId, slot, jobId, view.attempt);
        if (phaseChanged) {
          this.events.log("phase_changed", { job: jobId, session: sessionId, slot, phase: observed.phase });
        }
        for (const game of newGames) {
          this.events.log("game_finished", { job: jobId, session: sessionId, slot, ...game });
        }
        const yielded = this.jobs.applyYield();
        for (const id of yielded) this.events.log("wind_down", { job: id, reason: "yield" });
        // applyYield may have just set this job's wind_down; answer with the current row.
        return yielded.includes(jobId) ? this.jobs.view(this.jobs.row(jobId)!) : view;
      });
    }, { cost: () => 24 + 8 * parseReport(raw).finished_games.length });
  }

  async endJob(sessionId: string, slot: number, jobId: string, raw: unknown, attempt: number): Promise<ApiResult> {
    return this.run(() => {
      const { reason, retryable } = parseEnd(raw);
      return this.tx(() => {
        const worker = this.sessions.jobWorker(sessionId, slot);
        const before = this.jobs.row(jobId);
        if (before?.attempt !== attempt) throw new HttpError(409, "job attempt has changed");
        const job = this.jobs.end(jobId, worker, reason, retryable);
        if (before?.status !== job.status) this.events.log("job_ended", { job: jobId, session: sessionId, slot, reason, status: job.status });
        return job;
      });
    }, { cost: 24 });
  }

  async recordReplay(sessionId: string, slot: number, jobId: string, raw: unknown): Promise<ApiResult> {
    return this.run(() => this.tx(() => {
      const value = fields(
        raw,
        ["game_number", "key", "sha256", "size", "etag"],
        ["game_number", "key", "sha256", "size", "etag"],
      );
      const worker = this.sessions.recordingWorker(sessionId, slot);
      const number = int(value.game_number, "game_number");
      const recorded = this.jobs.hasReplay(jobId, number);
      const job = this.jobs.recordReplay(
        jobId,
        worker,
        number,
        str(value.key, "key"),
        str(value.sha256, "sha256"),
        int(value.size, "size"),
        str(value.etag, "etag"),
      );
      if (!recorded) this.events.log("replay_recorded", { job: jobId, session: sessionId, slot, game_number: number, key: value.key });
      return job;
    }), { alarm: false, cost: 8 });
  }

  async workerJob(sessionId: string, slot: number, jobId: string): Promise<ApiResult> {
    return this.run(() => this.jobs.workerJob(jobId, this.sessions.jobWorker(sessionId, slot)), { alarm: false });
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
    return this.run(
      () => ({
        write_budget: this.budget.summary(),
        database_bytes: this.ctx.storage.sql.databaseSize,
        paused: this.setting("paused") === "1",
        policy: this.policy(),
        capacity: this.sessions.capacity(),
        ...this.sessions.summary(),
      }),
      { alarm: false },
    );
  }

  async adminEvents(query: { job?: string; session?: string; since?: number; limit?: number }): Promise<ApiResult> {
    return this.run(() => ({ events: this.events.query(query) }), { alarm: false });
  }

  // Test seams, enabled only by the HAL_TEST_CLOCK binding in vitest.config.ts.

  async setTestClock(seconds: number): Promise<void> {
    if (this.env.HAL_TEST_CLOCK !== "1") throw new Error("test clock is disabled");
    this.setSetting("test_clock", String(seconds));
  }

  async resetForTest(): Promise<void> {
    if (this.env.HAL_TEST_CLOCK !== "1") throw new Error("test reset is disabled");
    this.tx(() => {
      for (const table of STORE_TABLES) this.ctx.storage.sql.exec(`DELETE FROM ${table}`);
      this.setSetting("schema_version", String(STORE_SCHEMA_VERSION));
      this.sessions.resetStream();
      this.ctx.storage.sql.exec("INSERT INTO counts(id, queued, active) VALUES (1, 0, 0)");
    });
    await this.ctx.storage.deleteAlarm();
  }

}
