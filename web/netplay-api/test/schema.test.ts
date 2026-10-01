import { env, runInDurableObject } from "cloudflare:test";
import { describe, expect, it } from "vitest";
import { Queue, STORE_SCHEMA_VERSION } from "../src/queue";

// Rewrites a fresh store, then builds a second Queue over it the way a redeployed Worker would.
async function reopen(change: (sql: SqlStorage) => void, use: (queue: Queue) => Promise<void>): Promise<void> {
  const stub = env.QUEUE.get(env.QUEUE.idFromName(crypto.randomUUID()));
  await runInDurableObject(stub, async (_instance, state) => {
    change(state.storage.sql);
    await use(new Queue(state, env));
  });
}

describe("store schema version", () => {
  it("records the version in a fresh store and serves requests", async () => {
    const stub = env.QUEUE.get(env.QUEUE.idFromName(crypto.randomUUID()));
    const stored = await runInDurableObject(stub, (_instance, state) =>
      state.storage.sql.exec<{ value: string }>("SELECT value FROM settings WHERE key = 'schema_version'").one().value,
    );
    expect(stored).toBe(String(STORE_SCHEMA_VERSION));
    await reopen(
      () => {},
      async (queue) => expect(await queue.capacity()).toMatchObject({ status: 200 }),
    );
  });

  it.each([
    ["a different version", (sql: SqlStorage) => sql.exec("UPDATE settings SET value = '0' WHERE key = 'schema_version'"), "0"],
    ["no version", (sql: SqlStorage) => sql.exec("DELETE FROM settings WHERE key = 'schema_version'"), "(missing)"],
    ["no settings table", (sql: SqlStorage) => sql.exec("DROP TABLE settings"), "(missing)"],
  ])("refuses every request from a store with %s", async (_name, change, found) => {
    const detail = `queue storage schema version ${found} is not the Worker's ${STORE_SCHEMA_VERSION}`;
    await reopen(change, async (queue) => {
      expect(await queue.capacity()).toMatchObject({ status: 503, body: { detail } });
      expect(await queue.startSession({})).toMatchObject({ status: 503, body: { detail } });
      expect(await queue.workerJob("session", 0, "job")).toMatchObject({ status: 503, body: { detail } });
      await expect(queue.alarm()).rejects.toThrow(detail);
    });
  });
});
