import { HttpError, SESSION_LIVE_SECONDS, SESSION_SILENCE_SECONDS, validatePlayerCode, workerId } from "./domain";
import type { PolicyConfig } from "./policy";
import type { JobStore, Row } from "./store";

export const SESSION_SCHEMA = `
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  host TEXT NOT NULL,
  bundle_sha256 TEXT NOT NULL,
  git_sha TEXT NOT NULL,
  slots INTEGER NOT NULL,
  wants_stream INTEGER NOT NULL,
  started_at REAL NOT NULL,
  last_seen_at REAL NOT NULL,
  draining INTEGER NOT NULL DEFAULT 0,
  status TEXT,
  ended_at REAL,
  end_reason TEXT,
  failed_jobs INTEGER
);
CREATE TABLE IF NOT EXISTS accounts (
  connect_code TEXT PRIMARY KEY,
  r2_key TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  session_id TEXT,
  slot INTEGER,
  leased_at REAL
);
CREATE TABLE IF NOT EXISTS stream (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  session_id TEXT,
  slot INTEGER NOT NULL DEFAULT 0 CHECK (slot = 0),
  granted_at REAL
);
INSERT OR IGNORE INTO stream(id, session_id, slot, granted_at) VALUES (1, NULL, 0, NULL);
`;

// Every live session holds at least one account; start/end change both in one transaction.
// Use those IDs for primary-key lookups instead of scanning ended sessions.
const LEASED_SESSION_IDS = "SELECT session_id FROM accounts WHERE session_id IS NOT NULL";

export interface StartRequest {
  host: string;
  bundle_sha256: string;
  git_sha: string;
  slots: number;
  stream: boolean;
}

export interface Account {
  connect_code: string;
  r2_key: string;
  sha256: string;
}

export interface AccountGrant extends Account {
  slot: number;
}

export interface CapacityBody {
  capacity: number;
  healthy_slots: number;
  active: number;
  queued: number;
  service_status: string;
  service_message: string;
  target_fps: number;
  game_fps: number | null;
  frame_interval_p95_ms: number | null;
  dolphin_step_p95_ms: number | null;
  policy_round_trip_p95_ms: number | null;
  model_inference_p95_ms: number | null;
  batch_wait_p95_ms: number | null;
  recoveries: number;
}

// hal/netplay_service/health.py RunnerStatus schema version.
const RUNNER_STATUS_VERSION = 5;
const RUNNER_STATES = new Set(["ready", "degraded", "recovering", "unavailable"]);
const TIMINGS = [
  "game_fps",
  "frame_interval_p95_ms",
  "dolphin_step_p95_ms",
  "policy_round_trip_p95_ms",
  "model_inference_p95_ms",
  "batch_wait_p95_ms",
] as const;
const SHA256 = /^[0-9a-f]{64}$/;

type RunnerStatus = Record<(typeof TIMINGS)[number], number | null> & {
  state: string;
  message: string;
  slots: number;
  healthy_slots: number;
  target_fps: number;
  recoveries: number;
};

function invalidStatus(detail: string): never {
  throw new HttpError(422, `runner status: ${detail}`);
}

function parseRunnerStatus(raw: unknown, slots: number): RunnerStatus {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) invalidStatus("must be an object");
  const value = raw as Record<string, unknown>;
  if (value.schema_version !== RUNNER_STATUS_VERSION) invalidStatus(`schema_version must be ${RUNNER_STATUS_VERSION}`);
  if (typeof value.state !== "string" || !RUNNER_STATES.has(value.state)) invalidStatus("unknown state");
  if (typeof value.message !== "string" || !value.message) invalidStatus("message must be non-empty");
  if (typeof value.policy_sha256 !== "string" || !SHA256.test(value.policy_sha256)) invalidStatus("bad policy_sha256");
  if (value.slots !== slots) invalidStatus(`slots must equal the session's ${slots}`);
  const healthy = value.healthy_slots;
  if (!Number.isInteger(healthy) || (healthy as number) < 0 || (healthy as number) > slots) invalidStatus("bad healthy_slots");
  if (typeof value.target_fps !== "number" || !(value.target_fps > 0)) invalidStatus("bad target_fps");
  for (const name of TIMINGS) {
    const timing = value[name];
    if (timing !== null && (typeof timing !== "number" || !Number.isFinite(timing) || timing < 0)) invalidStatus(`bad ${name}`);
  }
  if (!Number.isInteger(value.recoveries) || (value.recoveries as number) < 0) invalidStatus("bad recoveries");
  if (typeof value.updated_at !== "number") invalidStatus("bad updated_at");
  if (!Array.isArray(value.chunk_health)) invalidStatus("chunk_health must be a list");
  return value as unknown as RunnerStatus;
}

function parseAccounts(raw: unknown): Account[] {
  if (!Array.isArray(raw)) throw new HttpError(422, "accounts must be a list");
  const seen = new Set<string>();
  return raw.map((item) => {
    if (typeof item !== "object" || item === null) throw new HttpError(422, "accounts entries must be objects");
    const entry = item as Record<string, unknown>;
    if (Object.keys(entry).sort().join() !== "connect_code,r2_key,sha256") {
      throw new HttpError(422, "accounts entries need exactly connect_code, r2_key, and sha256");
    }
    const code = validatePlayerCode(String(entry.connect_code));
    if (typeof entry.r2_key !== "string" || !entry.r2_key) throw new HttpError(422, "r2_key must be non-empty");
    if (typeof entry.sha256 !== "string" || !SHA256.test(entry.sha256)) throw new HttpError(422, "sha256 must be hex");
    if (seen.has(code)) throw new HttpError(422, `account ${code} is listed twice`);
    seen.add(code);
    return { connect_code: code, r2_key: entry.r2_key, sha256: entry.sha256 };
  });
}

export class SessionStore {
  constructor(
    private readonly sql: SqlStorage,
    private readonly jobs: JobStore,
    private readonly now: () => number,
  ) {}

  private rows(query: string, ...params: SqlStorageValue[]): Row[] {
    return this.sql.exec<Row>(query, ...params).toArray();
  }

  exists(id: string): boolean {
    return this.rows("SELECT 1 FROM sessions WHERE id = ?", id).length > 0;
  }

  start(id: string, input: StartRequest, policy: PolicyConfig | null): { session_id: string; accounts: AccountGrant[] } {
    // A runner repeats a start when it loses the first response; it gets the same accounts.
    const existing = this.rows("SELECT * FROM sessions WHERE id = ?", id)[0];
    if (existing !== undefined) {
      if (existing.ended_at !== null) throw new HttpError(410, "session has ended");
      const same =
        existing.host === input.host &&
        existing.bundle_sha256 === input.bundle_sha256 &&
        existing.git_sha === input.git_sha &&
        existing.slots === input.slots &&
        existing.wants_stream === (input.stream ? 1 : 0);
      if (!same) throw new HttpError(409, `session ${id} exists with different settings`);
      const accounts = this.rows(
        "SELECT slot, connect_code, r2_key, sha256 FROM accounts WHERE session_id = ? ORDER BY slot",
        id,
      ).map((row) => ({
        slot: row.slot as number,
        connect_code: row.connect_code as string,
        r2_key: row.r2_key as string,
        sha256: row.sha256 as string,
      }));
      return { session_id: id, accounts };
    }
    if (policy === null) throw new HttpError(503, "no policy has been published");
    if (input.bundle_sha256 !== policy.bundle_sha256) {
      throw new HttpError(409, `bundle ${input.bundle_sha256} is not the active policy ${policy.bundle_sha256}`);
    }
    const free = this.rows("SELECT * FROM accounts WHERE session_id IS NULL ORDER BY connect_code");
    if (free.length < input.slots) {
      throw new HttpError(409, `${free.length} bot accounts are free; ${input.slots} are required`);
    }
    const now = this.now();
    this.sql.exec(
      `INSERT INTO sessions(id, host, bundle_sha256, git_sha, slots, wants_stream, started_at, last_seen_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      id,
      input.host,
      input.bundle_sha256,
      input.git_sha,
      input.slots,
      input.stream ? 1 : 0,
      now,
      now,
    );
    const accounts = free.slice(0, input.slots).map((row, slot) => {
      this.sql.exec(
        "UPDATE accounts SET session_id = ?, slot = ?, leased_at = ? WHERE connect_code = ?",
        id,
        slot,
        now,
        row.connect_code,
      );
      return {
        slot,
        connect_code: row.connect_code as string,
        r2_key: row.r2_key as string,
        sha256: row.sha256 as string,
      };
    });
    return { session_id: id, accounts };
  }

  live(id: string): Row {
    const row = this.rows("SELECT * FROM sessions WHERE id = ?", id)[0];
    if (row === undefined) throw new HttpError(404, "session not found");
    if (row.ended_at !== null) throw new HttpError(410, "session has ended");
    return row;
  }

  wantsStream(id: string): boolean {
    return this.live(id).wants_stream === 1;
  }

  // Liveness uses the time this report arrived, never the runner's clock.
  report(id: string, raw: unknown): { draining: boolean; streamHolder: boolean; streamGranted: boolean } {
    const row = this.live(id);
    const status = parseRunnerStatus(raw, row.slots as number);
    const now = this.now();
    this.sql.exec("UPDATE sessions SET status = ?, last_seen_at = ? WHERE id = ?", JSON.stringify(status), now, id);
    let holder = this.rows("SELECT session_id FROM stream WHERE id = 1")[0]?.session_id;
    let streamGranted = false;
    if (holder === null && row.wants_stream === 1 && row.draining === 0) {
      this.sql.exec("UPDATE stream SET session_id = ?, slot = 0, granted_at = ? WHERE id = 1", id, now);
      holder = id;
      streamGranted = true;
    }
    return { draining: row.draining === 1, streamHolder: holder === id, streamGranted };
  }

  private slotWorker(row: Row, slot: number): string {
    if (!Number.isInteger(slot) || slot < 0 || slot >= (row.slots as number)) {
      throw new HttpError(422, `slot must be in [0, ${(row.slots as number) - 1}]`);
    }
    return workerId(row.id as string, slot);
  }

  claimWorker(id: string, slot: number, policy: PolicyConfig | null): string {
    const row = this.live(id);
    const worker = this.slotWorker(row, slot);
    if (row.draining === 1) throw new HttpError(409, "session is draining");
    if (policy === null || row.bundle_sha256 !== policy.bundle_sha256) {
      throw new HttpError(409, `bundle ${row.bundle_sha256} is not the active policy ${policy?.bundle_sha256 ?? "(none)"}`);
    }
    return worker;
  }

  deferToStreamSlot(id: string, slot: number): boolean {
    const holder = this.rows(
      `SELECT sessions.id, sessions.last_seen_at FROM stream
       JOIN sessions ON sessions.id = stream.session_id
       WHERE stream.id = 1 AND sessions.ended_at IS NULL AND sessions.draining = 0
         AND sessions.last_seen_at >= ?`,
      this.now() - SESSION_LIVE_SECONDS,
    )[0];
    if (holder === undefined || (holder.id === id && slot === 0)) return false;
    return !this.jobs.hasLease(workerId(holder.id as string, 0));
  }

  streamHolder(): string | null {
    return (this.rows("SELECT session_id FROM stream WHERE id = 1")[0]?.session_id as string | null) ?? null;
  }

  // In-progress transitions stay allowed after a republish or while draining.
  jobWorker(id: string, slot: number): string {
    return this.slotWorker(this.live(id), slot);
  }

  // A deferred replay upload can outlive its session, so an ended session may still record.
  recordingWorker(id: string, slot: number): string {
    const row = this.rows("SELECT * FROM sessions WHERE id = ?", id)[0];
    if (row === undefined) throw new HttpError(404, "session not found");
    return this.slotWorker(row, slot);
  }

  drain(id: string): void {
    this.live(id);
    this.sql.exec("UPDATE sessions SET draining = 1 WHERE id = ?", id);
    this.releaseStream(id);
  }

  end(id: string, reason: string): string[] {
    const row = this.live(id);
    const workers = Array.from({ length: row.slots as number }, (_, slot) => workerId(id, slot));
    const failed = this.jobs.failWorkers(workers);
    this.sql.exec(
      "UPDATE sessions SET ended_at = ?, end_reason = ?, failed_jobs = ? WHERE id = ?",
      this.now(),
      reason,
      failed.length,
      id,
    );
    this.sql.exec("UPDATE accounts SET session_id = NULL, slot = NULL, leased_at = NULL WHERE session_id = ?", id);
    this.releaseStream(id);
    return failed;
  }

  private releaseStream(id: string): void {
    this.sql.exec(
      "UPDATE stream SET session_id = NULL, granted_at = NULL WHERE id = 1 AND session_id = ?",
      id,
    );
  }

  resetStream(): void {
    this.sql.exec("INSERT INTO stream(id, session_id, slot, granted_at) VALUES (1, NULL, 0, NULL)");
  }

  // A runner repeats DELETE when it loses the first response; it gets the first result.
  endedResult(id: string): number | null {
    const row = this.rows("SELECT ended_at, failed_jobs FROM sessions WHERE id = ?", id)[0];
    if (row === undefined) throw new HttpError(404, "session not found");
    return row.ended_at === null ? null : (row.failed_jobs as number);
  }

  endSilent(): { sessions: string[]; jobs: string[] } {
    const cutoff = this.now() - SESSION_SILENCE_SECONDS;
    const sessions = this.rows(
      `SELECT id FROM sessions WHERE id IN (${LEASED_SESSION_IDS})
         AND ended_at IS NULL AND last_seen_at <= ?`,
      cutoff,
    ).map((row) => row.id as string);
    return { sessions, jobs: sessions.flatMap((id) => this.end(id, "silent")) };
  }

  nextDeadline(): number | null {
    const row = this.rows(
      `SELECT MIN(last_seen_at) AS t FROM sessions WHERE id IN (${LEASED_SESSION_IDS}) AND ended_at IS NULL`,
    )[0];
    return row?.t == null ? null : (row.t as number) + SESSION_SILENCE_SECONDS;
  }

  capacity(): CapacityBody {
    const cutoff = this.now() - SESSION_LIVE_SECONDS;
    const statuses = this.rows(
      `SELECT status FROM sessions WHERE id IN (${LEASED_SESSION_IDS}) AND ended_at IS NULL
         AND draining = 0 AND status IS NOT NULL AND last_seen_at >= ? ORDER BY started_at`,
      cutoff,
    ).map((row) => JSON.parse(row.status as string) as RunnerStatus);
    const base = { active: this.jobs.activeCount(), queued: this.jobs.queueDepth(), target_fps: 60 };
    if (statuses.length === 0) {
      return {
        ...base,
        capacity: 0,
        healthy_slots: 0,
        service_status: "unavailable",
        service_message: "Game servers are unavailable. Try again shortly.",
        game_fps: null,
        frame_interval_p95_ms: null,
        dolphin_step_p95_ms: null,
        policy_round_trip_p95_ms: null,
        model_inference_p95_ms: null,
        batch_wait_p95_ms: null,
        recoveries: 0,
      };
    }
    const capacity = statuses.reduce((sum, status) => sum + status.slots, 0);
    const healthy = statuses.reduce((sum, status) => sum + status.healthy_slots, 0);
    let state: string;
    let message: string;
    if (statuses.length === 1) {
      state = statuses[0]!.state;
      message = statuses[0]!.message;
    } else if (statuses.every((status) => status.state === "ready")) {
      state = "ready";
      message = "Game servers are ready.";
    } else if (healthy === 0) {
      state = "recovering";
      message = "Game servers are recovering.";
    } else {
      state = "degraded";
      message = "Some game servers are degraded; others remain available.";
    }
    const values = (name: (typeof TIMINGS)[number]) =>
      statuses.map((status) => status[name]).filter((value): value is number => value !== null);
    const lowest = (name: (typeof TIMINGS)[number]) => (values(name).length ? Math.min(...values(name)) : null);
    const highest = (name: (typeof TIMINGS)[number]) => (values(name).length ? Math.max(...values(name)) : null);
    return {
      ...base,
      capacity,
      healthy_slots: healthy,
      service_status: state,
      service_message: message,
      game_fps: lowest("game_fps"),
      frame_interval_p95_ms: highest("frame_interval_p95_ms"),
      dolphin_step_p95_ms: highest("dolphin_step_p95_ms"),
      policy_round_trip_p95_ms: highest("policy_round_trip_p95_ms"),
      model_inference_p95_ms: highest("model_inference_p95_ms"),
      batch_wait_p95_ms: highest("batch_wait_p95_ms"),
      recoveries: statuses.reduce((sum, status) => sum + status.recoveries, 0),
    };
  }

  putAccounts(raw: unknown): void {
    const accounts = parseAccounts(raw);
    const codes = new Set(accounts.map((account) => account.connect_code));
    for (const row of this.rows("SELECT connect_code FROM accounts WHERE session_id IS NOT NULL")) {
      if (!codes.has(row.connect_code as string)) {
        throw new HttpError(409, `account ${row.connect_code} is leased by a live session`);
      }
    }
    const keep = accounts.map(() => "?").join(",");
    this.sql.exec(`DELETE FROM accounts WHERE connect_code NOT IN (${keep || "''"})`, ...accounts.map((a) => a.connect_code));
    for (const account of accounts) {
      this.sql.exec(
        `INSERT INTO accounts(connect_code, r2_key, sha256) VALUES (?, ?, ?)
         ON CONFLICT(connect_code) DO UPDATE SET r2_key = excluded.r2_key, sha256 = excluded.sha256`,
        account.connect_code,
        account.r2_key,
        account.sha256,
      );
    }
  }

  summary() {
    const stream = this.rows("SELECT session_id, slot, granted_at FROM stream WHERE id = 1")[0];
    return {
      sessions: this.rows(
        `SELECT id, host, bundle_sha256, git_sha, slots, wants_stream, started_at, last_seen_at, draining,
           ended_at, end_reason FROM sessions WHERE ended_at IS NULL OR ended_at >= ? ORDER BY started_at`,
        this.now() - 24 * 60 * 60,
      ),
      accounts: this.rows("SELECT connect_code, session_id, slot, leased_at FROM accounts ORDER BY connect_code"),
      stream:
        stream?.session_id == null
          ? null
          : { session_id: stream.session_id, slot: stream.slot, granted_at: stream.granted_at },
    };
  }
}
