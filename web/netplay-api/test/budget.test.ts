import { describe, expect, it } from "vitest";
import { COMMAND_WRITE_BUDGET, WriteBudget } from "../src/budget";
import { withStore } from "./helpers";

const DAY = Date.parse("2026-10-17T00:00:00Z") / 86_400_000;

function spent(sql: SqlStorage, day: number, credits: number): void {
  sql.exec("INSERT OR REPLACE INTO settings(key, value) VALUES ('write_budget', ?)", JSON.stringify({ day, spent: credits }));
}

describe("temporary acceptance write allowance", () => {
  it("retains accounting and enforces the extra 1,000-credit cap", async () => {
    await withStore((_store, clock, sql) => {
      clock.now = DAY * 86_400;
      spent(sql, DAY, 35_224);
      const budget = new WriteBudget(sql, () => clock.now, "2026-10-17");
      expect(budget.summary()).toEqual({ day: DAY, spent: 35_224, limit: 36_000 });
      expect(() => budget.requireCapacity()).not.toThrow();
      expect(budget.charge(36)).toBe(false);
      expect(budget.summary().spent).toBe(35_260);
      expect(budget.charge(540)).toBe(true);
      expect(() => budget.requireCapacity()).toThrow("control budget");
    });
  });

  it("expires at UTC midnight without carrying extra credits forward", async () => {
    await withStore((_store, clock, sql) => {
      clock.now = (DAY + 1) * 86_400 - 1;
      const budget = new WriteBudget(sql, () => clock.now, "2026-10-17");
      expect(budget.summary().limit).toBe(36_000);
      clock.now += 1;
      expect(budget.summary()).toEqual({ day: DAY + 1, spent: 0, limit: COMMAND_WRITE_BUDGET });
      spent(sql, DAY + 1, 35_000);
      expect(() => budget.requireCapacity()).toThrow("control budget");
    });
  });

  it.each([undefined, "2026-10-16", "2026-10-18"])("keeps the normal guard for %s", async (allowance) => {
    await withStore((_store, clock, sql) => {
      clock.now = DAY * 86_400;
      spent(sql, DAY, 35_224);
      const budget = new WriteBudget(sql, () => clock.now, allowance);
      expect(budget.summary().limit).toBe(COMMAND_WRITE_BUDGET);
      expect(() => budget.requireCapacity()).toThrow("control budget");
    });
  });

  it.each(["", "2026-02-30", "2026-10-17T00:00:00Z", "not-a-date"])("rejects invalid configuration %s", async (allowance) => {
    await withStore((_store, clock, sql) => {
      expect(() => new WriteBudget(sql, () => clock.now, allowance)).toThrow("ACCEPTANCE_WRITE_BUDGET_DAY");
    });
  });
});
