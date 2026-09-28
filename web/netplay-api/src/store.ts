import {
  HttpError,
  IDLE_TIMEOUT_SECONDS,
  type JobStatus,
  LEASE_SECONDS,
  MAX_ATTEMPTS,
  MAX_GAMES,
  PLAYING_LEASE_SECONDS,
  TERMINAL_STATUSES,
  sameDigest,
} from "./domain";

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
  worker TEXT NOT NULL,
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

  private owned(id: string, worker: string, expected: readonly JobStatus[]): Row {
    const row = this.row(id);
    if (row === null || row.lease_owner !== worker) throw new HttpError(409, "worker does not own this job");
    if (!expected.includes(row.status as JobStatus)) {
      throw new HttpError(409, `job status '${row.status}' is not valid for this operation`);
    }
    return row;
  }

  // A runner retries a call whose response it lost. When this worker already
  // applied the change, the retry returns the current job instead of a 409.
  private applied(id: string, worker: string, done: (row: Row) => boolean): JobResponse | null {
    const row = this.row(id);
    return row !== null && row.last_worker === worker && done(row) ? this.response(row) : null;
  }

  private leaseSeconds(status: string): number {
    return status === "playing" ? PLAYING_LEASE_SECONDS : LEASE_SECONDS;
  }

  workerJob(id: string, worker: string): JobResponse {
    const row = this.row(id);
    if (row === null || row.lease_owner !== worker) throw new HttpError(409, "worker does not own this job");
    return this.response(row);
  }

  // A repeated claim after a lost response must not give the slot a second job.
  // Callers check this before claimNext, which leases without regard to held jobs.
  heldLease(worker: string): JobResponse | null {
    const held = this.first(`SELECT id, status FROM jobs WHERE lease_owner = ? AND status IN (${IN_SERVICE})`, worker);
    if (held === null) return null;
    if (held.status === "leased") return this.response(this.reload(held.id as string));
    throw new HttpError(409, `slot already holds job ${held.id}`);
  }

  hasLease(worker: string): boolean {
    return this.first(`SELECT 1 FROM jobs WHERE lease_owner = ? AND status IN (${IN_SERVICE})`, worker) !== null;
  }

  claimNext(worker: string): JobResponse | null {
    const now = this.time();
    const row = this.first("SELECT id FROM jobs WHERE status = 'queued' ORDER BY retry_front DESC, queue_seq ASC LIMIT 1");
    if (row === null) return null;
    this.exec(
      `UPDATE jobs SET status = 'leased', retry_front = 0, attempt = attempt + 1, lease_owner = ?,
         last_worker = ?, lease_expires_at = ?, updated_at = ? WHERE id = ?`,
      worker,
      worker,
      now + LEASE_SECONDS,
      now,
      row.id as string,
    );
    return this.response(this.reload(row.id as string));
  }

  heartbeat(id: string, worker: string): JobResponse {
    const now = this.time();
    const row = this.row(id);
    if (row === null || row.lease_owner !== worker || TERMINAL_STATUSES.has(row.status as string)) {
      throw new HttpError(409, "worker does not own an active job lease");
    }
    this.exec(
      "UPDATE jobs SET lease_expires_at = ?, updated_at = ? WHERE id = ?",
      now + this.leaseSeconds(row.status as string),
      now,
      id,
    );
    return this.response(this.reload(id));
  }

  markConnecting(id: string, worker: string, connectCode: string): JobResponse {
    const done = this.applied(id, worker, (row) => row.status === "connecting" && row.connect_code === connectCode);
    if (done) return done;
    this.owned(id, worker, ["leased"]);
    const now = this.time();
    this.exec(
      "UPDATE jobs SET status = 'connecting', connect_code = ?, connect_deadline = ?, updated_at = ? WHERE id = ?",
      connectCode,
      now + IDLE_TIMEOUT_SECONDS,
      now,
      id,
    );
    return this.response(this.reload(id));
  }

  markPlaying(id: string, worker: string): JobResponse {
    const done = this.applied(id, worker, (row) => row.status === "playing" && row.lease_owner === worker);
    if (done) return done;
    this.owned(id, worker, ["connecting", "rematch_ready"]);
    const now = this.time();
    this.exec(
      `UPDATE jobs SET status = 'playing', connect_deadline = NULL, rematch_deadline = NULL,
         lease_expires_at = ?, updated_at = ? WHERE id = ?`,
      now + PLAYING_LEASE_SECONDS,
      now,
      id,
    );
    return this.response(this.reload(id));
  }

  markNoShow(id: string, worker: string): JobResponse {
    const done = this.applied(id, worker, (row) => row.status === "no_show");
    if (done) return done;
    this.owned(id, worker, ["connecting"]);
    this.exec(
      `UPDATE jobs SET status = 'no_show', connect_deadline = NULL, lease_owner = NULL,
         lease_expires_at = NULL, updated_at = ? WHERE id = ?`,
      this.time(),
      id,
    );
    return this.response(this.reload(id));
  }

  markNoContest(id: string, worker: string): JobResponse {
    const done = this.applied(id, worker, (row) => row.status === "canceled" && row.lease_owner === null);
    if (done) return done;
    this.owned(id, worker, ["playing"]);
    this.exec(
      `UPDATE jobs SET status = 'canceled', connect_deadline = NULL, rematch_deadline = NULL,
         lease_owner = NULL, lease_expires_at = NULL, updated_at = ? WHERE id = ?`,
      this.time(),
      id,
    );
    return this.response(this.reload(id));
  }

  finishGame(id: string, worker: string, gameNumber: number, stage: string, result: string): JobResponse {
    if (!result) throw new HttpError(422, "game result must be non-empty");
    const recorded = this.first("SELECT actual_stage, result FROM games WHERE job_id = ? AND game_number = ?", id, gameNumber);
    if (recorded !== null && this.row(id)?.last_worker === worker) {
      if (recorded.actual_stage === stage && recorded.result === result) return this.response(this.reload(id));
      throw new HttpError(409, "game already recorded with a different result");
    }
    const row = this.owned(id, worker, ["playing"]);
    const played = row.game_count as number;
    if (gameNumber !== played + 1) throw new HttpError(409, `game_number ${gameNumber} does not follow game ${played}`);
    const now = this.time();
    const terminal = gameNumber >= MAX_GAMES || row.cancel_after_game === 1;
    this.exec(
      `UPDATE jobs SET status = ?, game_count = ?, actual_stage = ?, last_result = ?, rematch_deadline = ?,
         lease_owner = CASE WHEN ? THEN NULL ELSE lease_owner END,
         lease_expires_at = CASE WHEN ? THEN NULL ELSE lease_expires_at END, updated_at = ?
       WHERE id = ?`,
      terminal ? "complete" : "rematch_wait",
      gameNumber,
      stage,
      result,
      terminal ? null : now + IDLE_TIMEOUT_SECONDS,
      terminal ? 1 : 0,
      terminal ? 1 : 0,
      now,
      id,
    );
    this.exec(
      "INSERT INTO games(job_id, game_number, actual_stage, result, worker, created_at) VALUES (?, ?, ?, ?, ?, ?)",
      id,
      gameNumber,
      stage,
      result,
      worker,
      now,
    );
    return this.response(this.reload(id));
  }

  fail(id: string, worker: string, errorCode: string, retryable: boolean): JobResponse {
    if (!errorCode) throw new HttpError(422, "error_code must be non-empty");
    const done = this.applied(
      id,
      worker,
      (row) => row.lease_owner === null && row.error_code === errorCode && (row.status === "queued" || row.status === "failed"),
    );
    if (done) return done;
    const row = this.owned(id, worker, ["leased", "connecting", "playing", "rematch_wait", "rematch_ready"]);
    const retry = retryable && (row.attempt as number) < MAX_ATTEMPTS;
    this.exec(
      `UPDATE jobs SET status = ?, retry_front = ?, error_code = ?, lease_owner = NULL, lease_expires_at = NULL,
         connect_deadline = NULL, rematch_deadline = NULL, updated_at = ? WHERE id = ?`,
      retry ? "queued" : "failed",
      retry ? 1 : 0,
      errorCode,
      this.time(),
      id,
    );
    return this.response(this.reload(id));
  }

  forfeit(id: string, worker: string): JobResponse {
    const done = this.applied(
      id,
      worker,
      (row) => row.status === "failed" && row.error_code === "service_failure_bot_forfeit",
    );
    if (done) return done;
    this.owned(id, worker, ["connecting", "playing"]);
    this.exec(
      `UPDATE jobs SET status = 'failed', last_result = 'win', error_code = 'service_failure_bot_forfeit',
         lease_owner = NULL, lease_expires_at = NULL, connect_deadline = NULL, rematch_deadline = NULL,
         updated_at = ? WHERE id = ?`,
      this.time(),
      id,
    );
    return this.response(this.reload(id));
  }

  recordReplay(id: string, worker: string, gameNumber: number, key: string, sha256: string, size: number, etag: string): JobResponse {
    const game = this.first(
      "SELECT worker, replay_key, replay_sha256, replay_size, replay_etag FROM games WHERE job_id = ? AND game_number = ?",
      id,
      gameNumber,
    );
    if (game === null) throw new HttpError(409, "game is absent");
    if (game.worker !== worker) throw new HttpError(409, "worker did not play this game");
    const same =
      game.replay_key === key && game.replay_sha256 === sha256 && game.replay_size === size && game.replay_etag === etag;
    if (!same) {
      if (game.replay_key !== null) throw new HttpError(409, "game already has a different replay");
      this.exec(
        `UPDATE games SET replay_key = ?, replay_sha256 = ?, replay_size = ?, replay_etag = ?
         WHERE job_id = ? AND game_number = ? AND replay_key IS NULL`,
        key,
        sha256,
        size,
        etag,
        id,
        gameNumber,
      );
    }
    return this.response(this.reload(id));
  }

  // Returns the IDs of jobs whose leases were closed.
  failWorkers(workers: readonly string[]): string[] {
    if (workers.length === 0) return [];
    const marks = workers.map(() => "?").join(",");
    const ids = this.sql
      .exec<Row>(
        `SELECT id FROM jobs WHERE lease_owner IN (${marks})
           AND status IN ('leased','connecting','playing','rematch_wait','rematch_ready') ORDER BY queue_seq`,
        ...workers,
      )
      .toArray()
      .map((row) => row.id as string);
    this.exec(
      `UPDATE jobs SET status = 'failed',
         last_result = CASE WHEN status IN ('connecting','playing') THEN 'win' ELSE last_result END,
         error_code = CASE WHEN status IN ('connecting','playing')
           THEN 'service_failure_bot_forfeit' ELSE 'service_generation_aborted' END,
         lease_owner = NULL, lease_expires_at = NULL, connect_deadline = NULL, rematch_deadline = NULL,
         updated_at = ?
       WHERE lease_owner IN (${marks})
         AND status IN ('leased','connecting','playing','rematch_wait','rematch_ready')`,
      this.time(),
      ...workers,
    );
    return ids;
  }

  // Returns the IDs of jobs that changed.
  reapExpired(): string[] {
    const now = this.time();
    const changed: string[] = [];
    const ids = (query: string): string[] =>
      this.sql.exec<Row>(query, now).toArray().map((row) => row.id as string);
    for (const id of ids("SELECT id FROM jobs WHERE status = 'connecting' AND connect_deadline <= ?")) {
      this.exec(
        `UPDATE jobs SET status = 'no_show', lease_owner = NULL, lease_expires_at = NULL,
           connect_deadline = NULL, updated_at = ? WHERE id = ?`,
        now,
        id,
      );
      changed.push(id);
    }
    for (const id of ids("SELECT id FROM jobs WHERE status = 'rematch_wait' AND rematch_deadline <= ?")) {
      this.exec(
        `UPDATE jobs SET status = 'complete', lease_owner = NULL, lease_expires_at = NULL,
           rematch_deadline = NULL, updated_at = ? WHERE id = ?`,
        now,
        id,
      );
      changed.push(id);
    }
    const expired = this.sql
      .exec<Row>(
        `SELECT id, attempt FROM jobs WHERE lease_owner IS NOT NULL AND lease_expires_at <= ?
           AND status NOT IN (${TERMINAL})`,
        now,
      )
      .toArray();
    for (const row of expired) {
      const retry = (row.attempt as number) < MAX_ATTEMPTS;
      this.exec(
        `UPDATE jobs SET status = ?, retry_front = ?, error_code = 'lease_expired', lease_owner = NULL,
           lease_expires_at = NULL, connect_deadline = NULL, rematch_deadline = NULL, updated_at = ? WHERE id = ?`,
        retry ? "queued" : "failed",
        retry ? 1 : 0,
        now,
        row.id as string,
      );
      changed.push(row.id as string);
    }
    return changed;
  }

  nextDeadline(): number | null {
    const row = this.first(
      `SELECT MIN(t) AS t FROM (
         SELECT connect_deadline AS t FROM jobs WHERE status = 'connecting' AND connect_deadline IS NOT NULL
         UNION ALL SELECT rematch_deadline FROM jobs WHERE status = 'rematch_wait' AND rematch_deadline IS NOT NULL
         UNION ALL SELECT lease_expires_at FROM jobs
           WHERE lease_owner IS NOT NULL AND status NOT IN (${TERMINAL}))`,
    );
    return (row?.t as number | null) ?? null;
  }
}
