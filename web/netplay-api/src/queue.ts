import { DurableObject } from "cloudflare:workers";
import {
  HttpError,
  IN_GAME_PHASES,
  QUEUE_CAP,
  RUNNER_PROTOCOL_VERSION,
  randomToken,
  sha256Hex,
  validatePlayerCode,
} from "./domain";
import type { Env } from "./env";
import { EVENT_SCHEMA, EventLog } from "./events";
import { type PolicyConfig, checkChoice, optionsBody, parsePolicyConfig } from "./policy";
import { bool, fields, int, parseCreate, parseEnd, parseReport, parseSettingsUpdate, str } from "./requests";
import { SESSION_SCHEMA, SessionStore, type StartRequest } from "./sessions";
import { JOB_SCHEMA, JobStore } from "./store";

export interface ApiResult {
  status: number;
  body?: unknown;
  headers?: Record<string, string>;
}

const SESSION_ID = /^[A-Za-z0-9_-]{16,64}$/;

// Every table is created with CREATE TABLE IF NOT EXISTS, which keeps an older
// table unchanged. Bump this with any table change and use a fresh instance.
// A mismatched store refuses every request; there are no migrations.
export const STORE_SCHEMA_VERSION = 5;
export const QUEUE_INSTANCE = `global-v${STORE_SCHEMA_VERSION}`;
const STORE_TABLES = ["games", "jobs", "pairings", "sessions", "accounts", "stream", "events", "policy", "settings"];

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

  private requireSchema(): void {
    if (this.schemaError !== null) throw new HttpError(503, this.schemaError);
  }

  private async run(
    fn: () => unknown,
    { status = 200, alarm = true }: { status?: number; alarm?: boolean } = {},
  ): Promise<ApiResult> {
    try {
      this.requireSchema();
      const body = await fn();
      // Empty claims do not change deadlines. Read routes also opt out.
      if (alarm && body !== null && body !== undefined) await this.scheduleAlarm();
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
    this.tx(() => {
      const streamHolder = this.sessions.streamHolder();
      const ended = this.sessions.endSilent();
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
        const row = this.jobs.row(id);
        this.events.log("job_released", {
          job: id,
          status: row?.status,
          reason: row?.end_reason ?? "lease_expired",
        });
      }
      for (const id of this.jobs.applyYield()) this.events.log("wind_down", { job: id, reason: "yield" });
      this.events.prune();
    });
    await this.scheduleAlarm();
  }

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
    }, { status: 201 });
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
    }, { alarm: false });
  }

  async requestLock(id: string, token: string): Promise<ApiResult> {
    const digest = await sha256Hex(token);
    return this.run(() => this.tx(() => {
      const job = this.jobs.requestLock(id, digest);
      this.events.log("lock_requested", { job: id, count: job.lock_requests });
      return job;
    }), { alarm: false });
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
    }, { status: 201 });
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
    });
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
    }), { alarm: false });
  }

  async drain(sessionId: string): Promise<ApiResult> {
    return this.run(() =>
      this.tx(() => {
        const released = this.sessions.streamHolder() === sessionId;
        this.sessions.drain(sessionId);
        this.events.log("session_draining", { session: sessionId });
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

  async report(sessionId: string, slot: number, jobId: string, raw: unknown): Promise<ApiResult> {
    return this.run(() => {
      const observed = parseReport(raw);
      const policy = this.requirePolicy();
      for (const game of observed.finished_games) checkChoice(policy.stages, game.stage, "stage");
      return this.tx(() => {
        const worker = this.sessions.jobWorker(sessionId, slot);
        const { view, phaseChanged, newGames } = this.jobs.report(jobId, worker, observed);
        // A live game no longer needs the session's one pairing, so other slots may claim.
        if (IN_GAME_PHASES.has(observed.phase)) this.sessions.finishPairing(sessionId, slot, jobId, view.attempt);
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
    });
  }

  async endJob(sessionId: string, slot: number, jobId: string, raw: unknown): Promise<ApiResult> {
    return this.run(() => {
      const { reason, retryable } = parseEnd(raw);
      return this.tx(() => {
        const worker = this.sessions.jobWorker(sessionId, slot);
        const job = this.jobs.end(jobId, worker, reason, retryable);
        this.events.log("job_ended", { job: jobId, session: sessionId, slot, reason, status: job.status });
        return job;
      });
    });
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
      const job = this.jobs.recordReplay(
        jobId,
        worker,
        number,
        str(value.key, "key"),
        str(value.sha256, "sha256"),
        int(value.size, "size"),
        str(value.etag, "etag"),
      );
      this.events.log("replay_recorded", { job: jobId, session: sessionId, slot, game_number: number, key: value.key });
      return job;
    }), { alarm: false });
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
    });
    await this.ctx.storage.deleteAlarm();
  }

}
