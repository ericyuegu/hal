import { env, runInDurableObject } from "cloudflare:test";
import { JOB_SCHEMA, JobStore } from "../src/store";

export const START = 1_000_000;

export class Clock {
  now = START;
  advance(seconds: number): void {
    this.now += seconds;
  }
}

export function queueStub() {
  return env.QUEUE.get(env.QUEUE.idFromName("global"));
}

// Runs store-level tests inside a Durable Object so they use real SQLite storage.
export async function withStore(fn: (store: JobStore, clock: Clock, sql: SqlStorage) => void): Promise<void> {
  const stub = env.QUEUE.get(env.QUEUE.idFromName(crypto.randomUUID()));
  await runInDurableObject(stub, (_instance, state) => {
    state.storage.sql.exec(JOB_SCHEMA);
    const clock = new Clock();
    fn(new JobStore(state.storage.sql, () => clock.now), clock, state.storage.sql);
  });
}

export const CHOICES = {
  character: "FOX",
  imitation: "IBDW#0",
  online_delay: 2,
  desired_return: 20,
  temperature: 1,
};
