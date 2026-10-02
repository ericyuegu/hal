import { HttpError } from "./domain";

// These are conservative write credits, including accounting, indexes and
// retention deletes. Admission also reserves the cost of finishing a set.
// Recurring health/alarm work has a separate 20,000-write daily reserve.
export const COMMAND_WRITE_BUDGET = 35_000;
export const RESERVATION_RESERVE = 200;
export const MAX_DATABASE_BYTES = 100 * 1024 * 1024;

export class WriteBudget {
  constructor(private readonly sql: Pick<SqlStorage, "exec" | "databaseSize">, private readonly now: () => number) {}

  private current(): { day: number; spent: number } {
    const day = Math.floor(this.now() / 86_400);
    const row = this.sql.exec<{ value: string }>("SELECT value FROM settings WHERE key = 'write_budget'").toArray()[0];
    if (row === undefined) return { day, spent: 0 };
    const saved = JSON.parse(row.value) as { day: number; spent: number };
    return saved.day === day ? saved : { day, spent: 0 };
  }

  requireCapacity(): void {
    if (this.sql.databaseSize >= MAX_DATABASE_BYTES) {
      throw new HttpError(503, "The queue history needs maintenance. Existing games can finish.");
    }
    if (this.current().spent + RESERVATION_RESERVE >= COMMAND_WRITE_BUDGET) {
      throw new HttpError(503, "The queue has reached today's control budget. Existing games can finish.", { "Retry-After": "300" });
    }
  }

  charge(credits: number): boolean {
    const current = this.current();
    current.spent += credits;
    this.sql.exec("INSERT INTO settings(key, value) VALUES ('write_budget', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", JSON.stringify(current));
    return current.spent + RESERVATION_RESERVE >= COMMAND_WRITE_BUDGET || this.sql.databaseSize >= MAX_DATABASE_BYTES;
  }

  summary(): { day: number; spent: number; limit: number } {
    return { ...this.current(), limit: COMMAND_WRITE_BUDGET };
  }
}
