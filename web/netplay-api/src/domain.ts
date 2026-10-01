export const CONNECT_TIMEOUT_SECONDS = 60;
export const IDLE_TIMEOUT_SECONDS = 600;
export const MAX_GAMES = 5;
export const MAX_ATTEMPTS = 2;
export const LEASE_SECONDS = 20;
// Dolphin keeps running locally through a short network outage, so a playing
// lease tolerates more silence than the other states.
export const PLAYING_LEASE_SECONDS = 60;
export const SESSION_SILENCE_SECONDS = 30;
// A player's page polls its job every second. Background tabs can throttle
// timers to once a minute, so a page counts as gone only after two missed minutes.
export const PLAYER_PRESENCE_SECONDS = 120;
// Polls refresh presence at most this often, to bound Durable Object writes.
export const PRESENCE_WRITE_SECONDS = 10;
export const SESSION_LIVE_SECONDS = 5;
export const EVENT_RETENTION_SECONDS = 30 * 24 * 60 * 60;
export const QUEUE_CAP = 20;
export const MAX_BODY_BYTES = 16 * 1024;
// The runner parses job bodies strictly, so any change to a runner route's
// request or response shape bumps this with RUNNER_PROTOCOL_VERSION in
// hal/netplay_service/queue_client.py.
export const RUNNER_PROTOCOL_VERSION = 2;

export type JobStatus =
  | "queued"
  | "leased"
  | "connecting"
  | "playing"
  | "rematch_wait"
  | "rematch_ready"
  | "complete"
  | "failed"
  | "canceled"
  | "no_show";

export const TERMINAL_STATUSES: ReadonlySet<string> = new Set(["complete", "failed", "canceled", "no_show"]);

export class HttpError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
    readonly headers: Record<string, string> = {},
  ) {
    super(detail);
  }
}

const PLAYER_CODE = /^[A-Z0-9]{1,8}#[0-9]{1,4}$/;

// Error messages quote values the way Python's repr() does; the golden
// transcripts pin those messages.
export function pyRepr(value: string): string {
  if (value.includes("'") && !value.includes('"')) return `"${value}"`;
  return `'${value.replaceAll("\\", "\\\\").replaceAll("'", "\\'")}'`;
}

export function validatePlayerCode(value: string): string {
  if (!PLAYER_CODE.test(value)) {
    throw new HttpError(422, "player_code must be an exact uppercase Slippi connect code such as CRYO#610");
  }
  return value;
}

export function workerId(sessionId: string, slot: number): string {
  return `${sessionId}/slot-${slot}`;
}

export async function sha256Hex(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

export function randomToken(bytes: number): string {
  const data = crypto.getRandomValues(new Uint8Array(bytes));
  return btoa(String.fromCharCode(...data)).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
}

export function sameDigest(a: string, b: string): boolean {
  const left = new TextEncoder().encode(a);
  const right = new TextEncoder().encode(b);
  return left.byteLength === right.byteLength && crypto.subtle.timingSafeEqual(left, right);
}
