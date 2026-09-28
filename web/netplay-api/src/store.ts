import { HttpError, type JobStatus, TERMINAL_STATUSES, sameDigest } from "./domain";

export type Row = Record<string, SqlStorageValue>;

export interface MatchChoices {
  character: string;
  imitation: string;
  online_delay: number;
  desired_return: number | null;
  temperature: number;
}

export interface JobResponse {
  id: string;
  player_code: string;
  character: string;
  imitation: string;
  online_delay: number;
  desired_return: number | null;
  temperature: number;
  policy_revision: number;
  requested_stage: string | null;
  status: JobStatus;
  queue_position: number | null;
  attempt: number;
  game_count: number;
  connect_code: string | null;
  actual_stage: string | null;
  last_result: string | null;
  error_code: string | null;
  connect_deadline: number | null;
  rematch_deadline: number | null;
  cancel_after_game: boolean;
}

const ACTIVE = "'queued','leased','connecting','playing','rematch_wait','rematch_ready'";
const IN_SERVICE = "'leased','connecting','playing','rematch_wait','rematch_ready'";
const TERMINAL = "'complete','failed','canceled','no_show'";

// Same columns and meanings as hal/netplay_service/queue.py schema v3, plus
// last_worker, which lets a retried runner call recognize its own applied change.
export const JOB_SCHEMA = `
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  token_digest TEXT NOT NULL,
  player_code TEXT NOT NULL,
  character TEXT NOT NULL,
  imitation TEXT NOT NULL,
  online_delay INTEGER NOT NULL CHECK (online_delay IN (2, 3)),
  desired_return REAL,
  temperature REAL NOT NULL,
  policy_revision INTEGER NOT NULL DEFAULT 0,
  requested_stage TEXT,
  status TEXT NOT NULL,
  queue_seq INTEGER NOT NULL,
  retry_front INTEGER NOT NULL DEFAULT 0 CHECK (retry_front IN (0, 1)),
  attempt INTEGER NOT NULL DEFAULT 0,
  game_count INTEGER NOT NULL DEFAULT 0,
  connect_code TEXT,
  actual_stage TEXT,
  last_result TEXT,
  error_code TEXT,
  connect_deadline REAL,
  rematch_deadline REAL,
  cancel_after_game INTEGER NOT NULL DEFAULT 0 CHECK (cancel_after_game IN (0, 1)),
  lease_owner TEXT,
  lease_expires_at REAL,
  last_worker TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_one_active_player ON jobs(player_code) WHERE status IN (${ACTIVE});
CREATE INDEX IF NOT EXISTS idx_jobs_queue ON jobs(retry_front DESC, queue_seq) WHERE status = 'queued';
CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs(lease_expires_at) WHERE lease_owner IS NOT NULL;
CREATE TABLE IF NOT EXISTS games (
  job_id TEXT NOT NULL REFERENCES jobs(id),
  game_number INTEGER NOT NULL,
  actual_stage TEXT NOT NULL,
  result TEXT NOT NULL,
  replay_key TEXT,
  replay_sha256 TEXT,
  replay_size INTEGER,
  replay_etag TEXT,
  created_at REAL NOT NULL,
  PRIMARY KEY (job_id, game_number)
);
`;

export class JobStore {
  constructor(
    private readonly sql: SqlStorage,
    private readonly now: () => number,
  ) {}

  protected first(query: string, ...params: SqlStorageValue[]): Row | null {
    const rows = this.sql.exec<Row>(query, ...params).toArray();
    return rows[0] ?? null;
  }

  protected exec(query: string, ...params: SqlStorageValue[]): number {
    return this.sql.exec(query, ...params).rowsWritten;
  }

  protected time(): number {
    return this.now();
  }

  row(id: string): Row | null {
    return this.first("SELECT * FROM jobs WHERE id = ?", id);
  }

  protected reload(id: string): Row {
    const row = this.row(id);
    if (row === null) throw new Error(`job ${id} vanished inside its transaction`);
    return row;
  }

  response(row: Row): JobResponse {
    const status = row.status as JobStatus;
    let position: number | null = null;
    if (status === "queued") {
      const ahead = this.first(
        `SELECT COUNT(*) AS n FROM jobs WHERE status = 'queued'
           AND (retry_front > ? OR (retry_front = ? AND queue_seq < ?))`,
        row.retry_front as number,
        row.retry_front as number,
        row.queue_seq as number,
      );
      position = 1 + Number(ahead?.n ?? 0);
    }
    return {
      id: row.id as string,
      player_code: row.player_code as string,
      character: row.character as string,
      imitation: row.imitation as string,
      online_delay: row.online_delay as number,
      desired_return: row.desired_return as number | null,
      temperature: row.temperature as number,
      policy_revision: row.policy_revision as number,
      requested_stage: row.requested_stage as string | null,
      status,
      queue_position: position,
      attempt: row.attempt as number,
      game_count: row.game_count as number,
      connect_code: row.connect_code as string | null,
      actual_stage: row.actual_stage as string | null,
      last_result: row.last_result as string | null,
      error_code: row.error_code as string | null,
      connect_deadline: row.connect_deadline as number | null,
      rematch_deadline: row.rematch_deadline as number | null,
      cancel_after_game: row.cancel_after_game === 1,
    };
  }

  createJob(id: string, tokenDigest: string, playerCode: string, choices: MatchChoices): JobResponse {
    const now = this.time();
    const seq = Number(this.first("SELECT COALESCE(MAX(queue_seq), 0) + 1 AS seq FROM jobs")?.seq);
    try {
      this.exec(
        `INSERT INTO jobs(id, token_digest, player_code, character, imitation, online_delay,
           desired_return, temperature, requested_stage, status, queue_seq, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 'queued', ?, ?, ?)`,
        id,
        tokenDigest,
        playerCode,
        choices.character,
        choices.imitation,
        choices.online_delay,
        choices.desired_return,
        choices.temperature,
        seq,
        now,
        now,
      );
    } catch (error) {
      if (error instanceof Error && error.message.includes("jobs.player_code")) {
        throw new HttpError(409, `player ${playerCode} already has an active reservation`);
      }
      throw error;
    }
    return this.response(this.reload(id));
  }

  authenticated(id: string, digest: string): Row {
    const row = this.row(id);
    if (row === null || !sameDigest(row.token_digest as string, digest)) throw new HttpError(404, "job not found");
    return row;
  }

  getJob(id: string, digest: string): JobResponse {
    return this.response(this.authenticated(id, digest));
  }

  updatePolicy(id: string, digest: string, desiredReturn: number | null, temperature: number): JobResponse {
    const row = this.authenticated(id, digest);
    if (TERMINAL_STATUSES.has(row.status as string)) throw new HttpError(409, "cannot update a finished reservation");
    this.exec(
      `UPDATE jobs SET desired_return = ?, temperature = ?, policy_revision = policy_revision + 1,
         updated_at = ? WHERE id = ?`,
      desiredReturn,
      temperature,
      this.time(),
      id,
    );
    return this.response(this.reload(id));
  }

  cancel(id: string, digest: string): JobResponse {
    const row = this.authenticated(id, digest);
    const status = row.status as string;
    if (TERMINAL_STATUSES.has(status)) return this.response(row);
    if (status === "playing") {
      this.exec("UPDATE jobs SET cancel_after_game = 1, updated_at = ? WHERE id = ?", this.time(), id);
    } else {
      this.exec(
        `UPDATE jobs SET status = 'canceled', lease_owner = NULL, lease_expires_at = NULL,
           connect_deadline = NULL, rematch_deadline = NULL, updated_at = ? WHERE id = ?`,
        this.time(),
        id,
      );
    }
    return this.response(this.reload(id));
  }

  // Throws inside the caller's transaction on expiry, so the completion is rolled
  // back exactly as in the Python service; the alarm completes the job.
  requestRematch(id: string, digest: string, character: string, imitation: string, stage: string): JobResponse {
    const now = this.time();
    const row = this.authenticated(id, digest);
    if (row.status !== "rematch_wait") throw new HttpError(409, "job is not waiting for a rematch");
    const deadline = row.rematch_deadline as number | null;
    if (deadline === null || deadline <= now) throw new HttpError(409, "rematch window expired");
    this.exec(
      `UPDATE jobs SET status = 'rematch_ready', character = ?, imitation = ?, requested_stage = ?,
         rematch_deadline = NULL, updated_at = ? WHERE id = ?`,
      character,
      imitation,
      stage,
      now,
      id,
    );
    return this.response(this.reload(id));
  }

  queueDepth(): number {
    return Number(this.first("SELECT COUNT(*) AS n FROM jobs WHERE status = 'queued'")?.n);
  }

  activeCount(): number {
    return Number(this.first(`SELECT COUNT(*) AS n FROM jobs WHERE status IN (${IN_SERVICE})`)?.n);
  }
}
