import {
  HttpError,
  IN_GAME_PHASES,
  LEASE_SECONDS,
  MAX_ATTEMPTS,
  PLAYER_PRESENCE_SECONDS,
  PLAYING_LEASE_SECONDS,
  PRESENCE_WRITE_SECONDS,
  YIELD_AFTER_SECONDS,
  sameDigest,
  type EndReason,
  type GameResult,
  type JobStatus,
  type Phase,
  type WindDown,
} from "./domain";
import type { CreateRequest, FinishedGame, ObservedReport, SettingsUpdate } from "./requests";

export type Row = Record<string, SqlStorageValue>;

export interface Settings {
  revision: number;
  character: string;
  imitation: string;
  stage: string | null;
  desired_return: number | null;
  temperature: number;
}

export interface JobView {
  id: string;
  player_code: string;
  online_delay: number;
  status: JobStatus;
  end_reason: EndReason | null;
  queue_position: number | null;
  attempt: number;
  settings: Settings;
  observed: {
    seq: number;
    phase: Phase;
    bot_code: string | null;
    seen_revision: number;
    locked_revision: number | null;
  } | null;
  phase_deadline: number | null;
  games: FinishedGame[];
  wind_down: WindDown | null;
  lock_requests: number;
}

export const JOB_SCHEMA = `
CREATE TABLE IF NOT EXISTS counts (id INTEGER PRIMARY KEY CHECK (id = 1), queued INTEGER NOT NULL, active INTEGER NOT NULL);
INSERT OR IGNORE INTO counts(id, queued, active) VALUES (1, 0, 0);

CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  token_digest TEXT NOT NULL,
  player_code TEXT NOT NULL,
  online_delay INTEGER NOT NULL CHECK (online_delay IN (2, 3)),
  status TEXT NOT NULL CHECK (status IN ('queued', 'assigned', 'ended')),
  end_reason TEXT,
  queue_seq INTEGER NOT NULL,
  retry_front INTEGER NOT NULL DEFAULT 0 CHECK (retry_front IN (0, 1)),
  attempt INTEGER NOT NULL DEFAULT 0,
  settings_revision INTEGER NOT NULL,
  character TEXT NOT NULL,
  imitation TEXT NOT NULL,
  stage TEXT,
  desired_return REAL,
  temperature REAL NOT NULL,
  observed_seq INTEGER NOT NULL DEFAULT 0,
  phase TEXT,
  bot_code TEXT,
  seen_revision INTEGER,
  locked_revision INTEGER,
  phase_deadline REAL,
  wind_down TEXT CHECK (wind_down IN ('player', 'yield')),
  lock_requests INTEGER NOT NULL DEFAULT 0,
  lease_owner TEXT,
  lease_expires_at REAL,
  last_worker TEXT,
  assigned_at REAL,
  player_seen_at REAL NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE TRIGGER IF NOT EXISTS jobs_count_insert AFTER INSERT ON jobs WHEN NEW.status IN ('queued', 'assigned') BEGIN
  UPDATE counts SET queued = queued + (NEW.status = 'queued'), active = active + (NEW.status = 'assigned') WHERE id = 1;
END;
CREATE TRIGGER IF NOT EXISTS jobs_count_update AFTER UPDATE OF status ON jobs WHEN OLD.status != NEW.status BEGIN
  UPDATE counts SET queued = queued + (NEW.status = 'queued') - (OLD.status = 'queued'),
    active = active + (NEW.status = 'assigned') - (OLD.status = 'assigned') WHERE id = 1;
END;
CREATE TRIGGER IF NOT EXISTS jobs_count_delete AFTER DELETE ON jobs WHEN OLD.status IN ('queued', 'assigned') BEGIN
  UPDATE counts SET queued = queued - (OLD.status = 'queued'), active = active - (OLD.status = 'assigned') WHERE id = 1;
END;
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_one_active_player ON jobs(player_code) WHERE status IN ('queued', 'assigned');
CREATE INDEX IF NOT EXISTS idx_jobs_queue ON jobs(retry_front DESC, queue_seq) WHERE status = 'queued';
CREATE INDEX IF NOT EXISTS idx_jobs_assigned ON jobs(lease_expires_at, assigned_at) WHERE status = 'assigned';
CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs(lease_expires_at) WHERE lease_owner IS NOT NULL;
CREATE TABLE IF NOT EXISTS games (
  job_id TEXT NOT NULL REFERENCES jobs(id),
  game_number INTEGER NOT NULL,
  stage TEXT NOT NULL,
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

// Clears everything the runner owned, for a requeue or a fresh claim.
const CLEAR_OBSERVED = `observed_seq = 0, phase = NULL, bot_code = NULL, seen_revision = NULL,
  locked_revision = NULL, phase_deadline = NULL`;

export class JobStore {
  constructor(
    private readonly sql: SqlStorage,
    private readonly now: () => number,
    private readonly presence?: (id: string, persisted: number) => number,
    private readonly lease?: (row: Row) => number,
  ) {}

  protected first(query: string, ...params: SqlStorageValue[]): Row | null {
    return this.sql.exec<Row>(query, ...params).toArray()[0] ?? null;
  }

  protected exec(query: string, ...params: SqlStorageValue[]): void {
    this.sql.exec(query, ...params);
  }

  protected time(): number {
    return this.now();
  }

  row(id: string): Row | null {
    return this.first("SELECT * FROM jobs WHERE id = ?", id);
  }

  private reload(id: string): Row {
    const row = this.row(id);
    if (row === null) throw new Error(`job ${id} vanished inside its transaction`);
    return row;
  }

  view(row: Row, queuePosition?: number): JobView {
    const status = row.status as JobStatus;
    let position: number | null = queuePosition ?? null;
    if (status === "queued" && queuePosition === undefined) {
      const ahead = this.first(
        `SELECT COUNT(*) AS n FROM jobs WHERE status = 'queued'
           AND (retry_front > ? OR (retry_front = ? AND queue_seq < ?))`,
        row.retry_front as SqlStorageValue,
        row.retry_front as SqlStorageValue,
        row.queue_seq as SqlStorageValue,
      );
      position = 1 + Number(ahead?.n ?? 0);
    }
    const games = this.sql
      .exec<Row>("SELECT game_number, stage, result FROM games WHERE job_id = ? ORDER BY game_number", row.id)
      .toArray()
      .map((game) => ({
        number: game.game_number as number,
        stage: game.stage as string,
        result: game.result as GameResult,
      }));
    return {
      id: row.id as string,
      player_code: row.player_code as string,
      online_delay: row.online_delay as number,
      status,
      end_reason: row.end_reason as EndReason | null,
      queue_position: position,
      attempt: row.attempt as number,
      settings: {
        revision: row.settings_revision as number,
        character: row.character as string,
        imitation: row.imitation as string,
        stage: row.stage as string | null,
        desired_return: row.desired_return as number | null,
        temperature: row.temperature as number,
      },
      observed:
        row.phase === null
          ? null
          : {
              seq: row.observed_seq as number,
              phase: row.phase as Phase,
              bot_code: row.bot_code as string | null,
              seen_revision: row.seen_revision as number,
              locked_revision: row.locked_revision as number | null,
            },
      phase_deadline: row.phase_deadline as number | null,
      games,
      wind_down: row.wind_down as WindDown | null,
      lock_requests: row.lock_requests as number,
    };
  }

  createJob(id: string, tokenDigest: string, request: CreateRequest): JobView {
    const now = this.time();
    const seq = Number(this.first("SELECT COALESCE(MAX(rowid), 0) + 1 AS seq FROM jobs")?.seq);
    try {
      this.exec(
        `INSERT INTO jobs(id, token_digest, player_code, online_delay, status, queue_seq, settings_revision,
           character, imitation, stage, desired_return, temperature, player_seen_at, created_at, updated_at)
         VALUES (?, ?, ?, ?, 'queued', ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)`,
        id,
        tokenDigest,
        request.player_code,
        request.online_delay,
        seq,
        request.character,
        request.imitation,
        request.stage,
        request.desired_return,
        request.temperature,
        now,
        now,
        now,
      );
    } catch (error) {
      if (error instanceof Error && error.message.includes("jobs.player_code")) {
        throw new HttpError(409, `player ${request.player_code} already has an active reservation`);
      }
      throw error;
    }
    return this.view(this.reload(id));
  }

  authenticated(id: string, digest: string): Row {
    const row = this.row(id);
    if (row === null || !sameDigest(row.token_digest as string, digest)) throw new HttpError(404, "job not found");
    return row;
  }

  getJob(id: string, digest: string): JobView {
    const row = this.authenticated(id, digest);
    const now = this.time();
    if (this.presence !== undefined) return this.view(row);
    if (now - (row.player_seen_at as number) < PRESENCE_WRITE_SECONDS) return this.view(row);
    this.exec("UPDATE jobs SET player_seen_at = ? WHERE id = ?", now, id);
    return this.view(this.reload(id));
  }

  updateSettings(id: string, digest: string, update: SettingsUpdate): JobView {
    const row = this.authenticated(id, digest);
    if (row.status === "ended") throw new HttpError(409, "cannot change a finished reservation");
    this.exec(
      `UPDATE jobs SET character = ?, imitation = ?, stage = ?, desired_return = ?, temperature = ?,
         settings_revision = settings_revision + 1, updated_at = ? WHERE id = ?`,
      update.character ?? (row.character as string),
      update.imitation ?? (row.imitation as string),
      "stage" in update ? (update.stage ?? null) : (row.stage as string | null),
      "desired_return" in update ? (update.desired_return ?? null) : (row.desired_return as number | null),
      update.temperature ?? (row.temperature as number),
      this.time(),
      id,
    );
    return this.view(this.reload(id));
  }

  requestLock(id: string, digest: string): JobView {
    const row = this.authenticated(id, digest);
    if (row.status !== "assigned") throw new HttpError(409, "only an assigned reservation can lock in");
    this.exec("UPDATE jobs SET lock_requests = lock_requests + 1, updated_at = ? WHERE id = ?", this.time(), id);
    return this.view(this.reload(id));
  }

  leave(id: string, digest: string): JobView {
    const row = this.authenticated(id, digest);
    if (row.status === "queued") this.finish(id, "player_canceled");
    else if (row.status === "assigned" && row.wind_down !== "player") {
      this.exec("UPDATE jobs SET wind_down = 'player', updated_at = ? WHERE id = ?", this.time(), id);
    }
    return this.view(this.reload(id));
  }

  queuePositions(): Map<string, number> {
    return new Map(this.sql.exec<{ id: string }>("SELECT id FROM jobs WHERE status = 'queued' ORDER BY retry_front DESC, queue_seq")
      .toArray().map((row, index) => [row.id, index + 1]));
  }

  queueDepth(): number {
    return Number(this.first("SELECT queued AS n FROM counts WHERE id = 1")?.n);
  }

  activeCount(): number {
    return Number(this.first("SELECT active AS n FROM counts WHERE id = 1")?.n);
  }

  private owned(id: string, worker: string): Row {
    const row = this.row(id);
    if (row === null || row.lease_owner !== worker || row.status !== "assigned") {
      throw new HttpError(409, "worker does not own this job");
    }
    return row;
  }

  workerJob(id: string, worker: string): JobView {
    return this.view(this.owned(id, worker));
  }

  // A repeated claim after a lost response returns the job the slot already holds,
  // but only before the runner reported progress; afterwards the slot is busy.
  heldLease(worker: string): JobView | null {
    const held = this.first("SELECT * FROM jobs WHERE lease_owner = ? AND status = 'assigned'", worker);
    if (held === null) return null;
    if (held.phase === null) return this.view(held);
    throw new HttpError(409, `slot already holds job ${held.id}`);
  }

  hasLease(worker: string): boolean {
    return this.first("SELECT 1 FROM jobs WHERE lease_owner = ? AND status = 'assigned'", worker) !== null;
  }

  claimNext(worker: string): JobView | null {
    const now = this.time();
    const row = this.first(
      "SELECT id FROM jobs WHERE status = 'queued' ORDER BY retry_front DESC, queue_seq ASC LIMIT 1",
    );
    if (row === null) return null;
    this.exec(
      `UPDATE jobs SET status = 'assigned', retry_front = 0, attempt = attempt + 1, lease_owner = ?, last_worker = ?,
         lease_expires_at = ?, assigned_at = ?, ${CLEAR_OBSERVED}, updated_at = ? WHERE id = ?`,
      worker,
      worker,
      now + LEASE_SECONDS,
      now,
      now,
      row.id as string,
    );
    return this.view(this.reload(row.id as string));
  }

  report(
    id: string,
    worker: string,
    report: ObservedReport,
  ): { view: JobView; phaseChanged: boolean; newGames: FinishedGame[] } {
    const row = this.owned(id, worker);
    if (report.seen_revision > (row.settings_revision as number)) {
      throw new HttpError(422, "seen_revision is ahead of the reservation's settings");
    }
    const now = this.time();
    const fresh = report.seq > (row.observed_seq as number);
    const phase = fresh ? report.phase : (row.phase as Phase | null);
    if (fresh) {
      this.exec(
        `UPDATE jobs SET observed_seq = ?, phase = ?, bot_code = ?, seen_revision = ?, locked_revision = ?,
           phase_deadline = ? WHERE id = ?`,
        report.seq,
        report.phase,
        report.bot_code,
        report.seen_revision,
        report.locked_revision,
        report.phase_seconds_left === null ? null : now + report.phase_seconds_left,
        id,
      );
    }
    const newGames = report.finished_games.filter((game) => this.recordGame(id, worker, game, now));
    const lease = phase !== null && IN_GAME_PHASES.has(phase) ? PLAYING_LEASE_SECONDS : LEASE_SECONDS;
    if (fresh) this.exec("UPDATE jobs SET lease_expires_at = ?, updated_at = ? WHERE id = ?", now + lease, now, id);
    return { view: this.view(this.reload(id)), phaseChanged: fresh && report.phase !== row.phase, newGames };
  }

  // Returns true when the game is new.
  private recordGame(id: string, worker: string, game: FinishedGame, now: number): boolean {
    const stored = this.first("SELECT stage, result FROM games WHERE job_id = ? AND game_number = ?", id, game.number);
    if (stored !== null) {
      if (stored.stage === game.stage && stored.result === game.result) return false;
      throw new HttpError(409, `game ${game.number} is already recorded with a different result`);
    }
    const last = this.first("SELECT MAX(game_number) AS n FROM games WHERE job_id = ?", id)?.n ?? 0;
    if (game.number !== Number(last) + 1) throw new HttpError(422, "finished_games must be numbered 1, 2, 3, …");
    this.exec(
      "INSERT INTO games(job_id, game_number, stage, result, worker, created_at) VALUES (?, ?, ?, ?, ?, ?)",
      id,
      game.number,
      game.stage,
      game.result,
      worker,
      now,
    );
    return true;
  }

  end(id: string, worker: string, reason: EndReason, retryable: boolean): JobView {
    const row = this.row(id);
    if (row !== null && row.last_worker === worker && row.lease_owner === null) {
      // A retry after a lost response: the first call already ended or requeued the job.
      if (row.status === "ended" || (row.status === "queued" && row.retry_front === 1)) return this.view(row);
    }
    this.owned(id, worker);
    if (reason === "service_failure" && retryable) this.requeueOrEnd(id);
    else this.finish(id, reason);
    return this.view(this.reload(id));
  }

  private finish(id: string, reason: EndReason): void {
    this.exec(
      `UPDATE jobs SET status = 'ended', end_reason = ?, lease_owner = NULL, lease_expires_at = NULL,
         phase_deadline = NULL, updated_at = ? WHERE id = ?`,
      reason,
      this.time(),
      id,
    );
  }

  // One rule for every runner failure: requeue at the front once, unless the player asked to stop.
  private requeueOrEnd(id: string): void {
    const row = this.reload(id);
    if (row.wind_down === "player") return this.finish(id, "player_canceled");
    if (row.wind_down === "yield") return this.finish(id, "yielded");
    if ((row.attempt as number) >= MAX_ATTEMPTS) return this.finish(id, "service_failure");
    this.exec(
      `UPDATE jobs SET status = 'queued', retry_front = 1, lease_owner = NULL, lease_expires_at = NULL,
         assigned_at = NULL, ${CLEAR_OBSERVED}, updated_at = ? WHERE id = ?`,
      this.time(),
      id,
    );
  }

  windDownForBudget(): boolean {
    return this.sql.exec("UPDATE jobs SET wind_down = 'yield' WHERE status = 'assigned' AND wind_down IS NULL").rowsWritten > 0;
  }

  hasReplay(id: string, number: number): boolean {
    return this.first("SELECT 1 FROM games WHERE job_id = ? AND game_number = ? AND replay_key IS NOT NULL", id, number) !== null;
  }

  recordReplay(
    id: string,
    worker: string,
    gameNumber: number,
    key: string,
    sha256: string,
    size: number,
    etag: string,
  ): JobView {
    const game = this.first(
      "SELECT worker, replay_key, replay_sha256, replay_size, replay_etag FROM games WHERE job_id = ? AND game_number = ?",
      id,
      gameNumber,
    );
    if (game === null) throw new HttpError(409, "game is absent");
    if (game.worker !== worker) throw new HttpError(409, "worker did not play this game");
    const same =
      game.replay_key === key &&
      game.replay_sha256 === sha256 &&
      game.replay_size === size &&
      game.replay_etag === etag;
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
    return this.view(this.reload(id));
  }

  // Returns the IDs of jobs whose leases were closed.
  failWorkers(workers: readonly string[]): string[] {
    if (workers.length === 0) return [];
    const marks = workers.map(() => "?").join(",");
    const ids = this.sql
      .exec<Row>(
        `SELECT id FROM jobs WHERE status = 'assigned' AND lease_owner IN (${marks}) ORDER BY queue_seq`,
        ...workers,
      )
      .toArray()
      .map((row) => row.id as string);
    for (const id of ids) this.requeueOrEnd(id);
    return ids;
  }

  // Returns the IDs of jobs that changed.
  reapExpired(): string[] {
    const now = this.time();
    const gone = this.sql
      .exec<Row>("SELECT id, player_seen_at FROM jobs WHERE status = 'queued' AND player_seen_at <= ?", now - PLAYER_PRESENCE_SECONDS)
      .toArray()
      .filter((row) => this.playerSeen(row) <= now - PLAYER_PRESENCE_SECONDS)
      .map((row) => row.id as string);
    for (const id of gone) this.finish(id, "player_left");
    const expired = this.sql
      .exec<Row>("SELECT * FROM jobs WHERE status = 'assigned' AND lease_expires_at <= ?", now)
      .toArray()
      .filter((row) => this.leaseExpiry(row) <= now)
      .map((row) => row.id as string);
    for (const id of expired) this.requeueOrEnd(id);
    return [...gone, ...expired];
  }

  // Returns the IDs of jobs newly asked to yield.
  applyYield(): string[] {
    if (this.queueDepth() === 0) return [];
    const ids = this.sql
      .exec<Row>(
        "SELECT id FROM jobs WHERE status = 'assigned' AND wind_down IS NULL AND assigned_at <= ?",
        this.time() - YIELD_AFTER_SECONDS,
      )
      .toArray()
      .map((row) => row.id as string);
    for (const id of ids) {
      this.exec("UPDATE jobs SET wind_down = 'yield', updated_at = ? WHERE id = ?", this.time(), id);
    }
    return ids;
  }

  private playerSeen(row: Row): number {
    return this.presence?.(row.id as string, row.player_seen_at as number) ?? row.player_seen_at as number;
  }

  private leaseExpiry(row: Row): number {
    return Math.max(row.lease_expires_at as number, this.lease?.(row) ?? 0);
  }

  savePresence(id: string, seen: number): void {
    this.exec("UPDATE jobs SET player_seen_at = MAX(player_seen_at, ?) WHERE id = ? AND status = 'queued'", seen, id);
  }

  assigned(): Row[] {
    return this.sql.exec<Row>("SELECT * FROM jobs WHERE status = 'assigned'").toArray();
  }

  nextDeadline(): number | null {
    const queued = this.sql.exec<Row>("SELECT id, player_seen_at FROM jobs WHERE status = 'queued'").toArray();
    const deadlines = queued.map((row) => this.playerSeen(row) + PLAYER_PRESENCE_SECONDS);
    for (const row of this.assigned()) {
      deadlines.push(this.leaseExpiry(row));
      // Yield is due only when there is someone to take the slot.
      if (queued.length > 0 && row.wind_down === null) deadlines.push((row.assigned_at as number) + YIELD_AFTER_SECONDS);
    }
    return deadlines.length === 0 ? null : Math.min(...deadlines);
  }
}
