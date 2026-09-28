import { HttpError, MAX_BODY_BYTES, sameDigest, sha256Hex } from "./domain";
import type { Env } from "./env";
import type { ApiResult, RunnerAction } from "./queue";

const SECURITY_HEADERS: Record<string, string> = {
  "Cache-Control": "no-store",
  "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
};
const RUNNER_ACTIONS = new Set<RunnerAction>([
  "heartbeat",
  "connecting",
  "playing",
  "no-show",
  "no-contest",
  "finish-game",
  "fail",
  "forfeit",
  "replay",
]);

function respond(result: ApiResult): Response {
  const headers = new Headers({ ...SECURITY_HEADERS, ...result.headers });
  if (result.body === undefined) return new Response(null, { status: result.status, headers });
  headers.set("Content-Type", "application/json");
  return new Response(JSON.stringify(result.body), { status: result.status, headers });
}

function failure(status: number, detail: string, headers: Record<string, string> = {}): Response {
  return respond({ status, body: { detail }, headers });
}

async function readBody(request: Request): Promise<unknown> {
  const length = request.headers.get("Content-Length");
  if (length !== null) {
    const parsed = Number(length);
    if (!Number.isInteger(parsed) || parsed < 0) throw new HttpError(400, "invalid Content-Length");
    if (parsed > MAX_BODY_BYTES) throw new HttpError(413, "request body is too large");
  }
  const text = await request.text();
  if (new TextEncoder().encode(text).byteLength > MAX_BODY_BYTES) throw new HttpError(413, "request body is too large");
  if (text === "") return undefined;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new HttpError(422, "request body must be JSON");
  }
}

function bearer(request: Request): string | null {
  const header = request.headers.get("Authorization");
  if (header === null) return null;
  const [scheme, value] = header.split(" ", 2);
  return scheme?.toLowerCase() === "bearer" && value ? value : null;
}

async function tokenMatches(token: string | null, digests: string): Promise<boolean> {
  if (token === null) return false;
  const digest = await sha256Hex(token);
  // Check every entry so the time taken does not reveal which one matched.
  return digests
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean)
    .reduce((found, entry) => sameDigest(entry, digest) || found, false);
}

function runnerSlot(request: Request): { session: string; slot: number } {
  const session = request.headers.get("X-HAL-Session");
  const slot = request.headers.get("X-HAL-Slot");
  if (!session || slot === null || !/^\d+$/.test(slot)) {
    throw new HttpError(400, "X-HAL-Session and X-HAL-Slot headers are required");
  }
  return { session, slot: Number(slot) };
}

// EventLog.query trusts its inputs, and SQLite treats a negative LIMIT as no limit.
function eventsSince(raw: string | null): number | undefined {
  if (raw === null) return undefined;
  const since = raw.trim() === "" ? Number.NaN : Number(raw);
  if (!Number.isFinite(since)) throw new HttpError(422, "since must be a finite number");
  return since;
}

function eventsLimit(raw: string | null): number | undefined {
  if (raw === null) return undefined;
  const limit = /^\d+$/.test(raw) ? Number(raw) : Number.NaN;
  if (!Number.isInteger(limit) || limit < 1 || limit > 5000) throw new HttpError(422, "limit must be an integer in [1, 5000]");
  return limit;
}

export async function handle(request: Request, env: Env): Promise<Response> {
  const queue = env.QUEUE.get(env.QUEUE.idFromName("global"));
  const url = new URL(request.url);
  const path = url.pathname;
  const method = request.method;
  try {
    // Player routes
    if (method === "GET" && path === "/v1/options") return respond(await queue.options());
    if (method === "GET" && path === "/v1/capacity") return respond(await queue.capacity());
    if (method === "POST" && path === "/v1/jobs") {
      const address = request.headers.get("CF-Connecting-IP") ?? "unknown";
      const { success } = await env.JOB_RATE_LIMIT.limit({ key: address });
      if (!success) {
        return failure(429, "Too many reservations from this address. Try again in a minute.", { "Retry-After": "60" });
      }
      return respond(await queue.createJob(await readBody(request)));
    }
    const job = path.match(/^\/v1\/jobs\/([^/]+)(\/policy|\/rematch)?$/);
    if (job) {
      const [, id, suffix] = job as [string, string, string | undefined];
      // FastAPI resolves the bearer dependency before it reads the body.
      const token = bearer(request);
      if (method === "PATCH" && suffix === "/policy") {
        if (token === null) return failure(401, "job token is required");
        return respond(await queue.updatePolicy(id, token, await readBody(request)));
      }
      if (method === "POST" && suffix === "/rematch") {
        if (token === null) return failure(401, "job token is required");
        return respond(await queue.rematch(id, token, await readBody(request)));
      }
      if (suffix === undefined && (method === "GET" || method === "DELETE")) {
        if (token === null) return failure(401, "job token is required");
        return respond(method === "GET" ? await queue.getJob(id, token) : await queue.cancelJob(id, token));
      }
    }

    // Runner routes
    if (path.startsWith("/v1/runner/")) {
      if (!(await tokenMatches(bearer(request), env.RUNNER_TOKEN_SHA256))) {
        await queue.logRefusal({ scope: "runner", method, path });
        return failure(401, "runner token is invalid");
      }
      if (method === "POST" && path === "/v1/runner/sessions") {
        return respond(await queue.startSession(await readBody(request)));
      }
      const session = path.match(/^\/v1\/runner\/sessions\/([^/]+)(\/status|\/claim|\/drain)?$/);
      if (session) {
        const [, id, suffix] = session as [string, string, string | undefined];
        if (method === "POST" && suffix === "/status") return respond(await queue.reportStatus(id, await readBody(request)));
        if (method === "POST" && suffix === "/claim") return respond(await queue.claim(id, await readBody(request)));
        if (method === "POST" && suffix === "/drain") return respond(await queue.drain(id));
        if (method === "DELETE" && suffix === undefined) return respond(await queue.endSession(id));
      }
      const runnerJob = path.match(/^\/v1\/runner\/jobs\/([^/]+)(?:\/([a-z-]+))?$/);
      if (runnerJob) {
        const [, id, action] = runnerJob as [string, string, string | undefined];
        if (action === "live") return queue.fetch(request);
        const { session: sessionId, slot } = runnerSlot(request);
        if (method === "GET" && action === undefined) return respond(await queue.workerJob(sessionId, slot, id));
        if (method === "POST" && action !== undefined && RUNNER_ACTIONS.has(action as RunnerAction)) {
          return respond(await queue.runnerJob(sessionId, slot, id, action as RunnerAction, await readBody(request)));
        }
      }
    }

    // Admin routes
    if (path.startsWith("/v1/admin/")) {
      if (!(await tokenMatches(bearer(request), env.ADMIN_TOKEN_SHA256))) {
        await queue.logRefusal({ scope: "admin", method, path });
        return failure(401, "admin token is invalid");
      }
      if (method === "PUT" && path === "/v1/admin/policy") return respond(await queue.putPolicy(await readBody(request)));
      if (method === "PUT" && path === "/v1/admin/accounts") return respond(await queue.putAccounts(await readBody(request)));
      if (method === "POST" && path === "/v1/admin/pause") return respond(await queue.setPaused(true));
      if (method === "POST" && path === "/v1/admin/resume") return respond(await queue.setPaused(false));
      if (method === "GET" && path === "/v1/admin/status") return respond(await queue.adminStatus());
      if (method === "GET" && path === "/v1/admin/events") {
        return respond(
          await queue.adminEvents({
            job: url.searchParams.get("job") ?? undefined,
            session: url.searchParams.get("session") ?? undefined,
            since: eventsSince(url.searchParams.get("since")),
            limit: eventsLimit(url.searchParams.get("limit")),
          }),
        );
      }
    }
    return failure(404, "not found");
  } catch (error) {
    if (error instanceof HttpError) return failure(error.status, error.detail, error.headers);
    throw error;
  }
}
