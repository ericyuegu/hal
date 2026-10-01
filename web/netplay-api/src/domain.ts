export const MAX_ATTEMPTS = 2;
export const LEASE_SECONDS = 20;
// Dolphin keeps running locally through a short network outage, so a game in
// progress tolerates more silence than the other phases.
export const PLAYING_LEASE_SECONDS = 60;
export const SESSION_SILENCE_SECONDS = 30;
// A player's page polls its job every second. Background tabs can throttle
// timers to once a minute, so a page counts as gone only after two missed minutes.
export const PLAYER_PRESENCE_SECONDS = 120;
// Polls refresh presence at most this often, to bound Durable Object writes.
export const PRESENCE_WRITE_SECONDS = 10;
// A reservation yields its slot after this long once someone is waiting.
export const YIELD_AFTER_SECONDS = 15 * 60;
export const SESSION_LIVE_SECONDS = 5;
export const EVENT_RETENTION_SECONDS = 30 * 24 * 60 * 60;
export const QUEUE_CAP = 20;
export const MAX_BODY_BYTES = 16 * 1024;
// Any change to a runner route's request or response shape bumps this with
// RUNNER_PROTOCOL_VERSION in hal/netplay_service/queue_client.py.
export const RUNNER_PROTOCOL_VERSION = 3;

export type JobStatus = "queued" | "assigned" | "ended";
export const PHASES = ["booting", "waiting_for_player", "character_select", "in_game", "paused"] as const;
export type Phase = (typeof PHASES)[number];
export const IN_GAME_PHASES: ReadonlySet<Phase> = new Set<Phase>(["in_game", "paused"]);
export const END_REASONS = [
  "player_canceled",
  "player_left",
  "player_disconnected",
  "no_show",
  "idle_timeout",
  "yielded",
  "service_failure",
] as const;
export type EndReason = (typeof END_REASONS)[number];
export const RUNNER_END_REASONS: ReadonlySet<EndReason> = new Set<EndReason>([
  "player_canceled",
  "player_disconnected",
  "no_show",
  "idle_timeout",
  "yielded",
  "service_failure",
]);
export const GAME_RESULTS = ["win", "loss", "no_contest"] as const;
export type GameResult = (typeof GAME_RESULTS)[number];
export type WindDown = "player" | "yield";

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
