import { EVENT_RETENTION_SECONDS } from "./domain";

export const EVENT_SCHEMA = `
CREATE TABLE IF NOT EXISTS events (
  at REAL NOT NULL,
  kind TEXT NOT NULL,
  job_id TEXT,
  session_id TEXT,
  detail TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_at ON events(at);
`;

export interface EventDetail {
  job?: string;
  session?: string;
  [key: string]: unknown;
}

export interface EventRecord {
  at: number;
  kind: string;
  job_id: string | null;
  session_id: string | null;
  detail: Record<string, unknown>;
}

export class EventLog {
  constructor(
    private readonly sql: SqlStorage,
    private readonly now: () => number,
  ) {}

  log(kind: string, { job, session, ...detail }: EventDetail): void {
    this.sql.exec(
      "INSERT INTO events(at, kind, job_id, session_id, detail) VALUES (?, ?, ?, ?, ?)",
      this.now(),
      kind,
      job ?? null,
      session ?? null,
      JSON.stringify(detail),
    );
  }

  prune(): void {
    this.sql.exec("DELETE FROM events WHERE at < ?", this.now() - EVENT_RETENTION_SECONDS);
  }

  oldest(): number | null {
    const rows = this.sql.exec<{ at: number | null }>("SELECT MIN(at) AS at FROM events").toArray();
    return rows[0]?.at ?? null;
  }

  query({ job, session, since, limit = 500 }: { job?: string; session?: string; since?: number; limit?: number }): EventRecord[] {
    const rows = this.sql
      .exec<Record<string, SqlStorageValue>>(
        `SELECT at, kind, job_id, session_id, detail FROM events
         WHERE (? IS NULL OR job_id = ?) AND (? IS NULL OR session_id = ?) AND at >= ?
         ORDER BY at, rowid LIMIT ?`,
        job ?? null,
        job ?? null,
        session ?? null,
        session ?? null,
        since ?? 0,
        Math.min(limit, 5000),
      )
      .toArray();
    return rows.map((row) => ({
      at: row.at as number,
      kind: row.kind as string,
      job_id: row.job_id as string | null,
      session_id: row.session_id as string | null,
      detail: JSON.parse(row.detail as string) as Record<string, unknown>,
    }));
  }
}
