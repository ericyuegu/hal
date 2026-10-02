import {
  END_REASONS,
  GAME_RESULTS,
  HttpError,
  PHASES,
  RUNNER_END_REASONS,
  type EndReason,
  type GameResult,
  type Phase,
} from "./domain";
import { checkChoice, type PolicyConfig } from "./policy";

export type Fields = Record<string, unknown>;

// Shape errors mirror FastAPI's request validation: status 422.
export function fields(raw: unknown, allowed: readonly string[], required: readonly string[]): Fields {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new HttpError(422, "request body must be a JSON object");
  }
  const value = raw as Fields;
  for (const key of Object.keys(value)) {
    if (!allowed.includes(key)) throw new HttpError(422, `unexpected field ${key}`);
  }
  for (const key of required) {
    if (!(key in value)) throw new HttpError(422, `missing field ${key}`);
  }
  return value;
}

export function str(value: unknown, name: string): string {
  if (typeof value !== "string") throw new HttpError(422, `${name} must be a string`);
  return value;
}

export function num(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new HttpError(422, `${name} must be a number`);
  return value;
}

export function int(value: unknown, name: string): number {
  if (!Number.isInteger(value)) throw new HttpError(422, `${name} must be an integer`);
  return value as number;
}

export function bool(value: unknown, name: string): boolean {
  if (typeof value !== "boolean") throw new HttpError(422, `${name} must be a boolean`);
  return value;
}

export function nullableNum(value: unknown, name: string): number | null {
  return value === null ? null : num(value, name);
}

function inRange(value: number, [low, high]: [number, number], name: string): number {
  if (value < low || value > high) throw new HttpError(422, `${name} must be in [${low}, ${high}]`);
  return value;
}

export interface CreateRequest {
  player_code: string;
  character: string;
  imitation: string;
  stage: string | null;
  online_delay: number;
  desired_return: number | null;
  temperature: number;
}

export function parseCreate(raw: unknown, policy: PolicyConfig): CreateRequest {
  const value = fields(
    raw,
    ["player_code", "character", "imitation", "stage", "online_delay", "desired_return", "temperature"],
    ["player_code", "character", "imitation", "online_delay"],
  );
  const code = str(value.player_code, "player_code");
  if (code.length < 3 || code.length > 13) throw new HttpError(422, "player_code must have 3 to 13 characters");
  const desired =
    "desired_return" in value ? nullableNum(value.desired_return, "desired_return") : policy.default_desired_return;
  const stage = "stage" in value && value.stage !== null ? str(value.stage, "stage") : null;
  return {
    player_code: code,
    character: str(value.character, "character"),
    imitation: str(value.imitation, "imitation"),
    stage,
    online_delay: int(value.online_delay, "online_delay"),
    desired_return: desired === null ? null : inRange(desired, policy.desired_return_range, "desired_return"),
    temperature:
      "temperature" in value
        ? inRange(num(value.temperature, "temperature"), policy.temperature_range, "temperature")
        : policy.default_temperature,
  };
}

export interface SettingsUpdate {
  character?: string;
  imitation?: string;
  stage?: string | null;
  desired_return?: number | null;
  temperature?: number;
}

export function parseSettingsUpdate(raw: unknown, policy: PolicyConfig): SettingsUpdate {
  const value = fields(raw, ["character", "imitation", "stage", "desired_return", "temperature"], []);
  if (Object.keys(value).length === 0) throw new HttpError(422, "provide at least one setting");
  const update: SettingsUpdate = {};
  if ("character" in value) {
    update.character = checkChoice(policy.characters, str(value.character, "character"), "character");
  }
  if ("imitation" in value) {
    update.imitation = checkChoice(policy.imitations, str(value.imitation, "imitation"), "imitation");
  }
  if ("stage" in value) {
    update.stage = value.stage === null ? null : checkChoice(policy.stages, str(value.stage, "stage"), "stage");
  }
  if ("desired_return" in value) {
    const desired = nullableNum(value.desired_return, "desired_return");
    update.desired_return = desired === null ? null : inRange(desired, policy.desired_return_range, "desired_return");
  }
  if ("temperature" in value) {
    update.temperature = inRange(num(value.temperature, "temperature"), policy.temperature_range, "temperature");
  }
  return update;
}

export interface FinishedGame {
  number: number;
  stage: string;
  result: GameResult;
}

export interface ObservedReport {
  seq: number;
  phase: Phase;
  phase_seconds_left: number | null;
  bot_code: string | null;
  seen_revision: number;
  locked_revision: number | null;
  finished_games: FinishedGame[];
}

function oneOf<T extends string>(value: unknown, allowed: readonly T[], name: string): T {
  if (typeof value !== "string" || !(allowed as readonly string[]).includes(value)) {
    throw new HttpError(422, `${name} must be one of ${allowed.join(", ")}`);
  }
  return value as T;
}

export function parseReport(raw: unknown): ObservedReport {
  const names = [
    "seq",
    "phase",
    "phase_seconds_left",
    "bot_code",
    "seen_revision",
    "locked_revision",
    "finished_games",
  ];
  const value = fields(raw, names, names);
  const seq = int(value.seq, "seq");
  const seen = int(value.seen_revision, "seen_revision");
  const locked = value.locked_revision === null ? null : int(value.locked_revision, "locked_revision");
  if (seq < 1 || seen < 1) throw new HttpError(422, "seq and seen_revision must be positive");
  if (locked !== null && (locked < 1 || locked > seen)) {
    throw new HttpError(422, "locked_revision must be in [1, seen_revision]");
  }
  const left = value.phase_seconds_left === null ? null : num(value.phase_seconds_left, "phase_seconds_left");
  if (left !== null && left < 0) throw new HttpError(422, "phase_seconds_left must be non-negative");
  if (!Array.isArray(value.finished_games)) throw new HttpError(422, "finished_games must be a list");
  const games = value.finished_games.map((item, index) => {
    const game = fields(item, ["number", "stage", "result"], ["number", "stage", "result"]);
    const number = int(game.number, "game number");
    if (number < 1 || (index > 0 && number !== (value.finished_games as { number: number }[])[index - 1]!.number + 1)) {
      throw new HttpError(422, "finished_games must have consecutive positive numbers");
    }
    return {
      number,
      stage: str(game.stage, "game stage"),
      result: oneOf(game.result, GAME_RESULTS, "game result"),
    };
  });
  return {
    seq,
    phase: oneOf(value.phase, PHASES, "phase"),
    phase_seconds_left: left,
    bot_code: value.bot_code === null ? null : str(value.bot_code, "bot_code"),
    seen_revision: seen,
    locked_revision: locked,
    finished_games: games,
  };
}

export function parseEnd(raw: unknown): { reason: EndReason; retryable: boolean } {
  const value = fields(raw, ["reason", "retryable"], ["reason", "retryable"]);
  const reason = oneOf(value.reason, END_REASONS, "reason");
  if (!RUNNER_END_REASONS.has(reason)) throw new HttpError(422, `reason ${reason} is not a runner reason`);
  return { reason, retryable: bool(value.retryable, "retryable") };
}
