/// <reference types="vite/client" />
import { beforeEach, describe, expect, it } from "vitest";
import { POLICY, START, call, publish, report, resetQueue, runAlarm, seedAccounts, setClock } from "./helpers";

interface Step {
  kind: "player" | "worker" | "advance";
  method?: string;
  path?: string;
  token?: string | null;
  body?: unknown;
  worker?: string;
  op?: string;
  job?: string | null;
  args?: Record<string, unknown>;
  compare?: "full" | "status" | "status_field";
  seconds?: number;
  response?: { status: number; body: any };
}

interface Transcript {
  name: string;
  steps: Step[];
}

const files = import.meta.glob<Transcript>("./transcripts/*.json", { eager: true, import: "default" });
const transcripts = Object.entries(files)
  .filter(([path]) => !path.endsWith("/policy.json"))
  .map(([, transcript]) => transcript);

class Aliases {
  private readonly toReal = new Map<string, string>();
  private readonly toAlias = new Map<string, string>();
  private counts = { job: 0, token: 0 };

  learn(body: unknown): void {
    if (typeof body !== "object" || body === null || Array.isArray(body)) return;
    const record = body as Record<string, unknown>;
    for (const [key, prefix] of [
      ["id", "job"],
      ["token", "token"],
    ] as const) {
      const value = record[key];
      if (typeof value === "string" && !this.toAlias.has(value)) {
        this.counts[prefix] += 1;
        const alias = `$${prefix}${this.counts[prefix]}`;
        this.toAlias.set(value, alias);
        this.toReal.set(alias, value);
      }
    }
  }

  real(text: string): string {
    return [...this.toReal.entries()]
      .sort(([a], [b]) => b.length - a.length)
      .reduce((result, [alias, value]) => result.replaceAll(alias, value), text);
  }

  normalize(body: unknown): unknown {
    if (Array.isArray(body)) return body.map((value) => this.normalize(value));
    if (typeof body === "object" && body !== null) {
      return Object.fromEntries(Object.entries(body).map(([key, value]) => [key, this.normalize(value)]));
    }
    if (typeof body === "string") return this.toAlias.get(body) ?? body;
    return body;
  }
}

function sessionAliases(transcript: Transcript): Map<string, number> {
  const slots = new Map<string, number>();
  for (const step of transcript.steps) {
    if (step.kind !== "worker" || !step.worker) continue;
    const [session, slot] = step.worker.split(":");
    const needed = slot === undefined ? Number(step.args?.slots ?? 1) : Number(slot) + 1;
    slots.set(session!, Math.max(slots.get(session!) ?? 0, needed));
  }
  return slots;
}

function runnerRequest(op: string, session: string, slot: number, job: string, args: Record<string, unknown>) {
  const runner = { session, slot };
  switch (op) {
    case "claim":
      return call("POST", `/v1/runner/sessions/${session}/claim`, { runner: true, body: { slot } });
    case "end-session":
      return call("DELETE", `/v1/runner/sessions/${session}`, { runner: true });
    case "get":
      return call("GET", `/v1/runner/jobs/${job}`, { runner });
    case "connecting":
      return call("POST", `/v1/runner/jobs/${job}/connecting`, { runner, body: { connect_code: args.connect_code } });
    case "finish-game":
      return call("POST", `/v1/runner/jobs/${job}/finish-game`, {
        runner,
        body: { game_number: args.game_number, actual_stage: args.actual_stage, result: args.result },
      });
    case "fail":
      return call("POST", `/v1/runner/jobs/${job}/fail`, {
        runner,
        body: { error_code: args.error_code, retryable: args.retryable },
      });
    case "replay":
      return call("POST", `/v1/runner/jobs/${job}/replay`, { runner, body: args });
    default:
      return call("POST", `/v1/runner/jobs/${job}/${op}`, { runner });
  }
}

beforeEach(async () => {
  await resetQueue();
});

describe("golden transcripts from the Python service", () => {
  it("covers every recorded scenario", () => {
    expect(transcripts.length).toBe(18);
  });

  for (const [ordinal, transcript] of transcripts.entries()) {
    it(transcript.name, async () => {
      let clock = START;
      await publish(POLICY);
      const needed = sessionAliases(transcript);
      await seedAccounts([...needed.values()].reduce((sum, slots) => sum + slots, 0) + 1);
      const sessions = new Map<string, string>();
      for (const [alias, slots] of needed) {
        const started = await call("POST", "/v1/runner/sessions", {
          runner: true,
          body: { session_id: crypto.randomUUID(), host: alias, bundle_sha256: POLICY.bundle_sha256, git_sha: "transcript", slots, stream: false },
        });
        sessions.set(alias, started.body.session_id);
        await report(started.body.session_id, slots);
      }
      if (needed.size === 0) {
        const started = await call("POST", "/v1/runner/sessions", {
          runner: true,
          body: { session_id: crypto.randomUUID(), host: "capacity", bundle_sha256: POLICY.bundle_sha256, git_sha: "transcript", slots: 1, stream: false },
        });
        sessions.set("capacity", started.body.session_id);
        await report(started.body.session_id, 1);
      }
      const aliases = new Aliases();

      for (const [index, step] of transcript.steps.entries()) {
        const where = `${transcript.name} step ${index}`;
        if (step.kind === "advance") {
          clock += step.seconds!;
          await setClock(clock);
          for (const [alias, id] of sessions) await report(id, needed.get(alias) ?? 1);
          await runAlarm();
          continue;
        }
        let actual;
        if (step.kind === "player") {
          const path = aliases.real(step.path!);
          const token = step.token == null ? undefined : aliases.real(step.token);
          actual = await call(step.method!, path, {
            token,
            body: step.body === null ? undefined : step.body,
            // The rate limiter outlives resetQueue, so every transcript needs its own addresses.
            ip: `10.${ordinal}.${Math.floor(index / 200)}.${index % 200}`,
          });
        } else {
          const [alias, slot] = step.worker!.split(":");
          const job = step.job == null ? "" : aliases.real(step.job);
          actual = await runnerRequest(step.op!, sessions.get(alias!)!, Number(slot ?? 0), job, step.args ?? {});
        }
        aliases.learn(actual.body);
        const expected = step.response!;
        expect(actual.status, where).toBe(expected.status);
        const pydanticList = Array.isArray(expected.body?.detail);
        if (pydanticList) continue;
        const mode = expected.status >= 400 ? "full" : step.kind === "player" ? "full" : step.compare;
        if (mode === "full") expect(aliases.normalize(actual.body), where).toEqual(expected.body);
        if (mode === "status_field") expect(actual.body.status, where).toBe(expected.body.status);
      }
    });
  }
});
